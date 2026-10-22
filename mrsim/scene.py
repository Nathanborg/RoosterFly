"""End-to-end frame synthesis: grow -> project -> rasterise -> composite.

One cached root system yields many frames (different tube pose, axial position,
azimuth), which is where nearly all the speed comes from: growth is expensive,
framing is cheap.  That same machinery is what makes v2 revisit sequences a small
addition rather than a rewrite.

The emitted ground truth deliberately includes THREE lengths:

``length_fragments_3d_mm``
    sum over fragments of the true 3D arc length,
``length_fragments_2d_mm``
    sum over fragments of the projected length,
``length_union_2d_mm``
    length of the UNION in the image plane -- the annotator-equivalent.

The first two double-count every crossing; only the third is what a human tracing
the frame would produce.  The discrepancy is large and depth-correlated in dense
frames, so shipping only the analytic sum (as is conventional) quietly poisons any
downstream area/length identity.  ``overlap_fraction`` is emitted alongside so
consumers can stratify or gate on it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import optics as O
from . import raster as R
from . import soil as S
from . import traits as T
from .grow import RootSystem, subdivide
from .params import PlantParams, get_species
from .tube import (
    MM_PER_CM,
    CameraParams,
    TubeFrame,
    TubeParams,
    VisibilityModel,
    frame_window,
    project,
)


# --------------------------------------------------------------------------
# root appearance
# --------------------------------------------------------------------------
_LUMA = np.array([0.299, 0.587, 0.114])
# render_frame's match_local_photometry(strength=0.18) moves the root mean
# halfway x 0.18 toward the soil: a root keeps 1 - 0.09 of its contrast
_PHOTOMETRY_STRENGTH = 0.18
_PHOTOMETRY_KEEP = 1.0 - 0.5 * _PHOTOMETRY_STRENGTH
MASK_MODES = ("calibrated", "geometric")


# --------------------------------------------------------------------------
# v3 per-root colour palette
# --------------------------------------------------------------------------
# Real roots range from white through cream, yellow and tan to reddish and
# dark brown -- by species, by age, by soil staining -- and different roots in
# ONE frame differ.  v2 drew one young->old ramp per FRAME, so every root in a
# frame had the same hue and only age moved it.  The palette below is drawn
# per ROOT (fixed for the root across every frame of its system).  Each entry
# is (tip, old): the pale, unsuberized tip colour and the browned colour the
# root reaches at maturation; age moves a segment from one to the other, so a
# root is pale at its tip and browner toward its base.
ROOT_PALETTE = (
    ("white",         (0.94, 0.93, 0.89), (0.84, 0.80, 0.70)),
    ("cream",         (0.92, 0.88, 0.76), (0.80, 0.72, 0.56)),
    ("yellowish",     (0.90, 0.84, 0.58), (0.78, 0.68, 0.40)),
    ("tan",           (0.82, 0.70, 0.52), (0.66, 0.52, 0.36)),
    ("light_brown",   (0.72, 0.58, 0.42), (0.52, 0.38, 0.25)),
    ("reddish_brown", (0.62, 0.42, 0.32), (0.44, 0.25, 0.17)),
    ("dark_brown",    (0.46, 0.34, 0.24), (0.27, 0.18, 0.11)),
)
PALETTE_NAMES = tuple(p[0] for p in ROOT_PALETTE)
DEFAULT_ROOT_PALETTE = ("white:0.20,cream:0.20,yellowish:0.10,tan:0.20,light_brown:0.15,"
                        "reddish_brown:0.08,dark_brown:0.07")
# luma/chroma decomposition used by the palette floor: c = luma * 1 + chroma,
# with luma(chroma) = 0 because the luma weights sum to 1.  Shifting luma
# alone therefore keeps the chroma VECTOR -- hue and saturation -- exactly.
_RGB_LO, _RGB_HI = 0.02, 1.0


def parse_palette(spec) -> Dict[str, float]:
    """``'white:0.2,tan:0.3'`` (or a dict) -> normalised ``{name: weight}``.

    ``''``/None -> ``{}`` (palette off).  Unknown names raise KeyError, so a
    typo cannot silently drop a colour from the mix; weights must be >= 0
    with a positive sum.
    """
    if not spec:
        return {}
    if isinstance(spec, dict):
        items = list(spec.items())
    else:
        items = []
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            name, _, w = part.partition(":")
            items.append((name.strip(), float(w) if w.strip() else 1.0))
    out: Dict[str, float] = {}
    for name, w in items:
        if name not in PALETTE_NAMES:
            raise KeyError("unknown root colour %r (have %s)" % (name, ", ".join(PALETTE_NAMES)))
        if float(w) < 0:
            raise ValueError("palette weight for %r is negative" % name)
        out[name] = out.get(name, 0.0) + float(w)
    tot = sum(out.values())
    if tot <= 0:
        raise ValueError("palette weights sum to zero: %r" % (spec,))
    return {k: v / tot for k, v in out.items()}


@dataclass
class RootPalette:
    """Per-root colours of one system (index = ``Root.index``).

    ``kind``        palette entry per root (index into PALETTE_NAMES)
    ``tip`` / ``old``  (n, 3) colours at age 0 and at maturation
    ``may_darken``  roots allowed to be DARKER than the soil when they cannot
                    be brighter without bleaching (share ~ dark_root_frac)
    """

    kind: np.ndarray
    tip: np.ndarray
    old: np.ndarray
    may_darken: np.ndarray

    def __len__(self) -> int:
        return int(self.kind.shape[0])


def sample_root_palette(n_roots: int, weights, rng, dark_root_frac: float = 0.15,
                        luma_sd: float = 0.07, browning=(0.5, 1.0)) -> RootPalette:
    """Draw every root's colour from ``weights`` (``parse_palette`` output).

    Per root: a palette entry, a lightness jitter ``exp(N(0, luma_sd))`` on
    both ends (two 'tan' roots are not identical), a browning strength in
    ``browning`` (how far toward the entry's old colour the root gets), and
    the ``may_darken`` flag with probability ``dark_root_frac``.  Always
    exactly ``3n`` uniforms and ``n`` normals, whatever the weights, so
    changing the mix never shifts another root's draws.
    """
    w = parse_palette(weights)
    if not w:
        raise ValueError("sample_root_palette needs a non-empty palette")
    n = int(n_roots)
    u = rng.random((n, 3)) if n else np.zeros((0, 3))
    z = rng.standard_normal(n) if n else np.zeros(0)
    names = list(PALETTE_NAMES)
    p = np.array([w.get(nm, 0.0) for nm in names], np.float64)
    cdf = np.cumsum(p) / p.sum()
    kind = np.minimum(np.searchsorted(cdf, u[:, 0], side="right"), len(names) - 1)
    tips = np.array([e[1] for e in ROOT_PALETTE], np.float64)[kind]
    olds = np.array([e[2] for e in ROOT_PALETTE], np.float64)[kind]
    b = browning[0] + (browning[1] - browning[0]) * u[:, 1]
    olds = tips + b[:, None] * (olds - tips)
    jit = np.exp(luma_sd * z)[:, None]
    return RootPalette(
        kind=kind.astype(np.int16),
        tip=np.clip(tips * jit, _RGB_LO, _RGB_HI).astype(np.float32),
        old=np.clip(olds * jit, _RGB_LO, _RGB_HI).astype(np.float32),
        may_darken=(u[:, 2] < float(dark_root_frac)))


def luma_chroma(c) -> Tuple[float, np.ndarray]:
    """``(luma, chroma vector)`` with ``c == luma + chroma``."""
    c = np.asarray(c, np.float64)
    lc = float(c @ _LUMA)
    return lc, c - lc


def luma_range(c, lo: float = _RGB_LO, hi: float = _RGB_HI) -> Tuple[float, float]:
    """Luma interval reachable by shifting luma ONLY (chroma kept in gamut)."""
    _, ch = luma_chroma(c)
    return lo - float(ch.min()), hi - float(ch.max())


def lift_luma(c, target: float, lo: float = _RGB_LO, hi: float = _RGB_HI):
    """Colour with luma ``target`` and ``c``'s hue: ``(rgb, chroma_kept)``.

    The luma is shifted with the chroma vector held fixed; only if that leaves
    the gamut is the chroma scaled down -- by the least factor that fits
    (``chroma_kept`` < 1).  Scaling RGB multiplicatively (v2) instead clips
    the red channel first on a bright target and turns a brown root cream;
    here a brown root stays brown wherever its luma range allows.
    """
    lc, ch = luma_chroma(c)
    t = float(target)
    k = 1.0
    for v in ch:
        if v > 1e-9 and t + v > hi:
            k = min(k, (hi - t) / v)
        elif v < -1e-9 and t + v < lo:
            k = min(k, (lo - t) / v)
    k = max(k, 0.0)
    out = np.clip(t + k * ch, lo, hi)
    return tuple(float(x) for x in out), float(k)


@dataclass
class RootAppearance:
    """Age -> colour/opacity.

    Note the ramp direction: a NEW, unsuberized root is the BRIGHTEST,
    highest-contrast object in a real frame -- brighter than the soil, not a faint
    translucent smudge.  Translucency belongs to very fine unsuberized laterals
    (a diameter effect), not to youth (an age effect).  Conflating the two is a
    common and very visible mistake.

    The age->colour mapping is randomised PER IMAGE because the minirhizotron
    literature explicitly discourages fixed white/tan/brown nomenclature: the same
    root images as cream, tan or dark brown depending on the instrument.
    """

    young_rgb: Tuple[float, float, float] = (0.86, 0.82, 0.74)
    old_rgb: Tuple[float, float, float] = (0.30, 0.22, 0.15)
    maturation_days: float = 22.0
    fine_diam_mm: float = 0.12       # below this, roots go translucent
    min_opacity: float = 0.35
    # --- v3 per-segment contrast floor (see segment_colour_opacity) ---------
    # target_n = 0 (default) switches it off and every colour is as before.
    # The frame context (soil luma and noise, PSF) is filled in by
    # render_frame from the frame's own background and optics.
    target_n: float = 0.0
    soil_luma: float = 0.0
    soil_sigma: float = 0.0
    psf_sigma_px: float = 0.0
    polarity: float = 1.0            # +1 roots brighter than soil, -1 darker
    # --- v3 per-root palette (None = the per-frame ramp above, as before) ---
    # With a palette, colour comes from palette.tip/old of each segment's ROOT
    # (young_rgb/old_rgb are ignored); maturation_days still sets browning by
    # age.  min_delta is the palette-mode luma floor (min_root_contrast).
    palette: Optional[RootPalette] = None
    min_delta: float = 0.0
    # --- v3 realism (palette path only; defaults = flat capsules as before) --
    # shading      cylindrical shading strength (raster.Segment.shade), roots
    #              wider than ~3 px only
    # texture      along-root luma variation: colour x exp(texture * n), n a
    #              smooth per-root noise with a ~2-6 mm wavelength
    # chroma_scale palette saturation (1 = as listed; real roots under glass
    #              are greyer than the palette names suggest)
    # vis_fade     soft visibility gate: a root fades out over the stretch where
    #              p_visible drops by vis_fade below its threshold instead of
    #              stopping dead; the faded end is also blurred by up to
    #              fade_blur_px.  Faded stretches below fade_min opacity are
    #              not drawn (and so not labelled).
    shading: float = 0.0
    texture: float = 0.0
    chroma_scale: float = 1.0
    vis_fade: float = 0.0
    fade_blur_px: float = 2.0
    fade_min: float = 0.35
    # cover_relief_px  soil crumbs on the glass hide THIN roots; a root w px wide
    #                  is hidden only by 1 / (1 + (w / cover_relief_px)^2) of the
    #                  crumb field, so thick roots are no longer chopped into beads
    # min_planarity    drop segments whose 2-D length is below this share of their
    #                  3-D length (roots pointing at the glass draw as round dots)
    # soil_film        per root, opacity x (1 - U(0, soil_film)) for roots >= 3 px:
    #                  a film of soil/water dims some roots instead of all white
    cover_relief_px: float = 0.0
    min_planarity: float = 0.0
    soil_film: float = 0.0
    # root_bank  real root strips (mrstyle.realbanks.RootBank) mapped along
    #            every root >= 3 px wide; tex_mix = share of the strip's own
    #            colour (the rest keeps the palette colour, textured)
    root_bank: Optional[object] = None
    tex_mix: float = 0.7
    # root_keep < 1: each root is drawn (and so labelled) with this probability
    # in this frame (draws from its own keep_seed stream, not the frame rng)
    root_keep: float = 1.0
    keep_seed: int = 0
    # root_hairs > 0: fine hairs drawn on a share root_hair_p of the roots 1.5-12 px wide
    # (hairs per mm of root length, each side); in the image only -- never labelled, as
    # annotators do not trace hairs
    root_hairs: float = 0.0
    root_hair_p: float = 0.4
    # tex_max_mag > 0: a root more than this many times wider than its texture strip uses only
    # the strip's cross profile (magnifying a thin root's detail ~10x drew bands on thick roots)
    tex_max_mag: float = 0.0

    @classmethod
    def sample(cls, rng) -> "RootAppearance":
        """Sample a plausible root colour ramp.

        Built from lightness plus a small warm chroma, NOT from three independent
        RGB channels.  Independent channels drift off the warm axis and produce
        green, red and magenta roots -- colours no root has, and a distribution
        the real corpus never visits, so a network trained on them learns colour
        cues that do not transfer.
        """
        def warm(value, chroma):
            v = float(np.clip(value, 0.03, 0.98))
            c = float(np.clip(chroma, 0.0, 0.5))
            return (float(np.clip(v * (1.0 + 0.45 * c), 0.02, 1.0)),
                    float(np.clip(v * (1.0 + 0.05 * c), 0.02, 1.0)),
                    float(np.clip(v * (1.0 - 0.50 * c), 0.02, 1.0)))

        # young: bright and nearly neutral -- a new unsuberized root is the
        # brightest object in the frame
        y_val = float(rng.uniform(0.70, 0.95))
        y_chr = float(rng.uniform(0.02, 0.14))
        # old: dark and warmer
        o_val = float(rng.uniform(0.12, 0.38))
        o_chr = float(rng.uniform(0.10, 0.38))
        return cls(
            young_rgb=warm(y_val, y_chr),
            old_rgb=warm(o_val, o_chr),
            maturation_days=float(rng.uniform(10.0, 45.0)),
            fine_diam_mm=float(rng.uniform(0.06, 0.20)),
            min_opacity=float(rng.uniform(0.35, 0.7)),
        )

    def with_min_contrast(self, soil_rgb, min_delta: float = 0.15,
                          polarity: str = "bright",
                          opacity_floor: Optional[float] = None) -> "RootAppearance":
        """Copy whose WHOLE age ramp is >= ``min_delta`` luma away from the soil.

        The sampled ramp runs from a bright young colour to a dark old one, so on
        most soils it crosses the soil's lightness at some age, and that stretch
        of root -- usually most of it, since most root length is old -- has no
        contrast at all.  Each end of the ramp is rescaled in lightness (hue
        ratios kept) until it sits ``min_delta`` on the ``polarity`` side of the
        soil; the ramp is linear in RGB, so every age in between then does too.

        ``polarity``: 'bright' (lighter than soil), 'dark', or 'auto' (the side
        the young colour is on).  If that side cannot be reached inside [0, 1]
        the other side is used.  ``opacity_floor`` raises ``min_opacity``, the
        translucency of fine roots (this changes alpha, hence the mask).
        Colours only otherwise: alpha and the mask are untouched, and nothing
        is drawn from any rng.
        """
        luma = np.array([0.299, 0.587, 0.114])
        ls = float(np.asarray(soil_rgb, float) @ luma)
        y = np.asarray(self.young_rgb, float)
        o = np.asarray(self.old_rgb, float)
        if polarity == "auto":
            s = 1.0 if float(y @ luma) >= ls else -1.0
        elif polarity in ("bright", "dark"):
            s = 1.0 if polarity == "bright" else -1.0
        else:
            raise ValueError("polarity must be 'bright', 'dark' or 'auto'")
        if not 0.03 <= ls + s * min_delta <= 0.97:
            s = -s
        target = ls + s * min_delta

        def push(c):
            lc = float(c @ luma)
            if s * (lc - target) >= 0.0:
                return tuple(float(v) for v in c)
            return tuple(float(v) for v in np.clip(c * (target / max(lc, 1e-3)),
                                                   0.02, 1.0))

        mo = self.min_opacity if opacity_floor is None else \
            max(self.min_opacity, float(opacity_floor))
        return replace(self, young_rgb=push(y), old_rgb=push(o), min_opacity=mo)

    def colour(self, age_days: float):
        f = float(np.clip(age_days / max(self.maturation_days, 1e-6), 0.0, 1.0))
        f = f ** 0.7
        y = np.asarray(self.young_rgb)
        o = np.asarray(self.old_rgb)
        return tuple(float(v) for v in (y * (1 - f) + o * f))

    def opacity(self, diam_mm: float) -> float:
        """Thin unsuberized roots are semi-transparent; soil shows through."""
        if diam_mm >= self.fine_diam_mm * 2.5:
            return 1.0
        f = float(np.clip(diam_mm / (self.fine_diam_mm * 2.5), 0.0, 1.0))
        return float(self.min_opacity + (1.0 - self.min_opacity) * f)

    def with_frame_context(self, soil_luma: Optional[float] = None,
                           soil_sigma: Optional[float] = None,
                           psf_sigma_px: Optional[float] = None,
                           target_n: Optional[float] = None,
                           polarity: Optional[float] = None) -> "RootAppearance":
        """Copy with the contrast-floor context set (None keeps a field)."""
        kw = {}
        for k, v in (("soil_luma", soil_luma), ("soil_sigma", soil_sigma),
                     ("psf_sigma_px", psf_sigma_px), ("target_n", target_n),
                     ("polarity", polarity)):
            if v is not None:
                kw[k] = float(v)
        return replace(self, **kw)

    def segment_colour_opacity(self, age_days: float, width_px: float,
                               opacity: float):
        """``(colour, opacity)`` of one segment, lifted to the contrast target.

        A single global luma floor (``with_min_contrast``) cannot work across
        widths: the PSF spreads a 1-px root's contrast over ~3 px, so the floor
        that makes thin roots visible makes thick ones far too contrasty (v2:
        thick contrast_n ~2.8 against a real median of ~1.3), and the floor
        that suits thick roots leaves thin ones invisible.  Here each segment
        gets the luma delta it NEEDS for normalised contrast ``target_n`` after
        blur and translucency:

            needed = target_n * soil_sigma / (opacity * erf(w / (2 sqrt(2) s)))

        ``w`` = the segment's width in px (floored at 1), ``s`` = the PSF sigma
        combined with the 1-px pixel box (sqrt(psf^2 + 1/12)): a bar of width w
        blurred by a Gaussian keeps erf(w / 2 sqrt(2) s) of its contrast at its
        centre.  ``soil_sigma`` is the frame's soil luma std after the PSF,
        sensor noise in quadrature -- the denominator of contrast_n.  The
        mean-only photometric match in render_frame removes ~9% of a root's
        contrast, which is compensated.

        A segment already at or beyond ``needed`` on the ``polarity`` side keeps
        its colour.  Otherwise its luma is pushed to ``soil_luma + polarity *
        needed`` (hue ratios kept, as ``with_min_contrast`` does; clipped to
        [0.03, 0.97]); if clipping leaves it short, opacity is raised (up to 1)
        to make up the rest.  So contrast_n is ~constant across widths for
        roots that need lifting, and the global floor can be LOWERED to bring
        thick roots down toward the real median.  ``target_n = 0`` returns
        ``(self.colour(age_days), opacity)`` exactly.  Draws nothing.
        """
        c = self.colour(age_days)
        if self.target_n <= 0.0 or self.soil_sigma <= 0.0:
            return c, opacity
        w = max(float(width_px), 1.0)
        s = math.sqrt(max(float(self.psf_sigma_px), 0.0) ** 2 + 1.0 / 12.0)
        gain = max(math.erf(w / (2.0 * math.sqrt(2.0) * s)), 1e-3)
        op = max(float(opacity), 1e-3)
        needed = self.target_n * self.soil_sigma / (_PHOTOMETRY_KEEP * op * gain)
        pol = 1.0 if self.polarity >= 0 else -1.0
        lc = float(np.asarray(c) @ _LUMA)
        if pol * (lc - self.soil_luma) >= needed:
            return c, opacity
        target = float(np.clip(self.soil_luma + pol * needed, 0.03, 0.97))
        cn = tuple(float(v) for v in np.clip(np.asarray(c) * (target / max(lc, 1e-3)),
                                             0.02, 1.0))
        got = pol * (float(np.asarray(cn) @ _LUMA) - self.soil_luma)
        if got < needed and got > 1e-6:
            opacity = float(min(1.0, op * needed / got))
        return cn, opacity

    # ---- v3 per-root palette ------------------------------------------------
    def root_colour(self, root_id: int, age_days: float):
        """Palette colour of root ``root_id`` at ``age_days``: pale tip ->
        browner base, with the same age curve as :meth:`colour`."""
        f = float(np.clip(age_days / max(self.maturation_days, 1e-6), 0.0, 1.0)) ** 0.7
        p = self.palette
        c = p.tip[root_id].astype(np.float64) * (1 - f) + p.old[root_id].astype(np.float64) * f
        return tuple(float(v) for v in c)

    def _needed_luma(self, width_px: float, opacity: float) -> float:
        """Luma delta a segment needs: the contrast_n target after blur and
        translucency (see :meth:`segment_colour_opacity`), and at least the
        global floor ``min_delta``."""
        need = float(self.min_delta)
        if self.target_n > 0.0 and self.soil_sigma > 0.0:
            w = max(float(width_px), 1.0)
            s = math.sqrt(max(float(self.psf_sigma_px), 0.0) ** 2 + 1.0 / 12.0)
            gain = max(math.erf(w / (2.0 * math.sqrt(2.0) * s)), 1e-3)
            need = max(need, self.target_n * self.soil_sigma
                       / (_PHOTOMETRY_KEEP * max(float(opacity), 1e-3) * gain))
        return need

    def root_polarities(self) -> np.ndarray:
        """Per-root polarity for THIS frame's soil (+1 brighter, -1 darker).

        Every root takes the frame polarity unless it is flagged
        ``may_darken`` AND its browned colour cannot reach even the smallest
        luma delta on that side with its chroma intact AND it can on the other
        side -- a brown root on pale soil goes darker than the soil instead of
        being bleached cream.  So the dark share is at most ``dark_root_frac``
        and only where it is needed.  No floor active -> the frame polarity
        everywhere (colours are not adjusted then).
        """
        p = self.palette
        pol0 = 1.0 if self.polarity >= 0 else -1.0
        out = np.full(len(p), pol0, np.float32)
        need = self._needed_luma(1e6, 1.0)          # thick, opaque: the minimum
        if need <= 0.0 or not p.may_darken.any():
            return out
        ch = p.old.astype(np.float64) - (p.old.astype(np.float64) @ _LUMA)[:, None]
        lmin = np.maximum(_RGB_LO - ch.min(1), 0.03)
        lmax = np.minimum(_RGB_HI - ch.max(1), 0.97)
        ls = float(self.soil_luma)
        if pol0 > 0:
            stuck, other_ok = lmax < ls + need, lmin <= ls - need
        else:
            stuck, other_ok = lmin > ls - need, lmax >= ls + need
        out[p.may_darken & stuck & other_ok] = -pol0
        return out

    def palette_colour_opacity(self, root_id: int, age_days: float, width_px: float,
                               opacity: float, pol: float):
        """``(colour, opacity)`` of one palette segment on side ``pol``.

        Like :meth:`segment_colour_opacity` but the luma is moved ALONE, only
        as far as needed (``lift_luma``): hue and chroma are kept, so a brown
        root stays brown.  If its luma range ends short of the target the luma
        stops at the range end and opacity makes up the rest; only if the
        root cannot get to the right side of the soil at all is the chroma
        reduced, by the least factor that reaches the target (a root not
        flagged ``may_darken`` on soil brighter than it).  No floor active ->
        the palette colour unchanged.  Draws nothing.
        """
        c = self.root_colour(root_id, age_days)
        need = self._needed_luma(width_px, opacity)
        if need <= 0.0:
            return c, opacity
        lc = float(np.asarray(c) @ _LUMA)
        ls = float(self.soil_luma)
        if pol * (lc - ls) >= need:
            return c, opacity
        target = float(np.clip(ls + pol * need, 0.03, 0.97))
        lo, hi = luma_range(c)
        reach = min(target, max(hi, 0.03)) if pol > 0 else max(target, min(lo, 0.97))
        if pol * (reach - ls) <= 1e-3:
            reach = target                          # wrong side even at the end: desaturate
        cn, _ = lift_luma(c, reach)
        got = pol * (float(np.asarray(cn) @ _LUMA) - ls)
        if got < need and got > 1e-6:
            opacity = float(min(1.0, max(float(opacity), 1e-3) * need / got))
        return cn, opacity


# --------------------------------------------------------------------------
# framing
# --------------------------------------------------------------------------
@dataclass
class FrameSpec:
    """Where on the tube this frame is taken."""

    axial_cm: float
    phi_center: float
    tube_angle_deg: float = 30.0


@dataclass
class FrameResult:
    rgb: np.ndarray
    alpha: np.ndarray
    mask: np.ndarray
    diameter_mm: np.ndarray
    order: np.ndarray
    instance: np.ndarray
    traits_observable: Dict[str, object]
    traits_architecture: Dict[str, object]
    meta: Dict[str, object]
    # Optional un-composited layers, populated when render_frame(keep_layers=True).
    # Style transfer uses these: the translator is shown the BACKGROUND ONLY and
    # the roots are alpha-composited back afterwards, so the mask stays exact by
    # construction and no learned model ever touches a root pixel.
    background: object = None
    root_rgb: object = None
    # v3: per-pixel diameter (mm) that is > 0 EXACTLY on ``mask`` -- the
    # max-cover segment's diameter, nearest-filled (raster.diameter_on_mask).
    # Set in geometric mask mode or with width_traits; what diam/ is written from.
    diameter_label_mm: object = None


@dataclass
class ProjectedSystem:
    """A root system projected into tube-local coordinates, ONCE.

    Tube-local ``(u, v)`` do not depend on where the frame is taken -- changing
    the frame is only a translation -- so all the expensive work (sub-millimetre
    refinement and projection of every node) is done here and reused across every
    frame cut from this system.  Doing it per frame made rendering ~50x slower
    than growth, which inverts the whole point of caching systems.

    Nodes that can never appear are dropped here too: anything deeper than a few
    multiples of the depth of field, and anything with NEGATIVE depth, i.e.
    inside the tube.  The second cull matters: ``p_visible`` clamps depth at zero,
    so without it every root that strays inside the wall reads as fully visible
    and the frame fills with roots that physically cannot be seen.
    """

    u_mm: np.ndarray            # circumferential, absolute (r_in * phi)
    v_mm: np.ndarray            # axial
    depth_mm: np.ndarray
    diam_mm: np.ndarray
    age_days: np.ndarray
    order: np.ndarray
    root_id: np.ndarray
    seg_len3d_mm: np.ndarray    # true 3D arc length of each segment
    circumference_mm: float
    n_culled_inside: int = 0
    # v3 (project_system(vis_bridge_mm > 0)): depth used ONLY for p_visible,
    # with short excursions away from the wall bridged; None = use depth_mm
    vis_depth_mm: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(self.u_mm.shape[0])


def subset_projection(proj: ProjectedSystem, keep: np.ndarray,
                      age_shift: float = 0.0) -> ProjectedSystem:
    """``proj`` restricted to the segments in ``keep`` (bool mask or indices).

    Slices EVERY per-segment array field generically, so a field added later
    (``vis_depth_mm`` was one) cannot be silently dropped -- the bug the
    hand-written copy in ``density`` would have had.  ``age_days`` is shifted
    back by ``age_shift`` days (floored at 0), which is how a younger stand is
    emulated from an older one.
    """
    import dataclasses
    n = len(proj)
    kw = {}
    for f in dataclasses.fields(proj):
        v = getattr(proj, f.name)
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
            kw[f.name] = v[keep]
        else:
            kw[f.name] = v
    kw["age_days"] = np.maximum(kw["age_days"] - age_shift, 0.0).astype(np.float32)
    return ProjectedSystem(**kw)


def _bridge_depth(depth: np.ndarray, seg3d: np.ndarray, bridge_mm: float) -> np.ndarray:
    """Grey-scale OPENING of a root's depth profile over ``bridge_mm`` of arc.

    A root weaving along the wall drifts in and out of the visible depth band;
    each short excursion away from the glass cut it into a dash, and a dashed
    root is both visibly synthetic and skeletonises too long.  An opening
    (min filter, then max filter) removes depth PEAKS narrower than the window
    -- the excursions -- while leaving every longer stretch, and every stretch
    close to the wall, exactly as it was (a plain min filter would also extend
    each visible stretch by half a window at both ends).
    """
    from scipy import ndimage as ndi
    if len(depth) < 3:
        return depth
    ds = float(np.mean(seg3d)) if seg3d.size else 0.2
    k = int(round(0.5 * float(bridge_mm) / max(ds, 1e-6)))
    if k < 1:
        return depth
    return ndi.grey_opening(np.asarray(depth, np.float64), size=2 * k + 1, mode="nearest")


def project_system(system: RootSystem, tube: TubeParams, cam: CameraParams,
                   rng, sub_mm: float = 0.2, max_depth_mm: float = 4.0,
                   frame: Optional[TubeFrame] = None,
                   vis_bridge_mm: float = 0.0) -> ProjectedSystem:
    """Refine and project every root once, in tube-local coordinates.

    Pass the SAME ``frame`` used during growth.  Rebuilding it here with a default
    origin would put the tube back at the seed and silently invalidate every
    depth.

    ``vis_bridge_mm > 0`` (v3) also stores ``vis_depth_mm``: each root's depth
    with excursions shorter than this bridged (:func:`_bridge_depth`), which
    ``_segments_for_frame`` uses for ``p_visible`` only (drawing order keeps the
    true depth), and keeps a segment if EITHER depth is within
    ``max_depth_mm``.  The soil-cover field still interrupts roots naturally.
    No random numbers are drawn for it.
    """
    if frame is None:
        frame = TubeFrame.build(tube)
    r_in_mm = tube.r_in * MM_PER_CM
    circ = 2.0 * math.pi * r_in_mm

    U, V, D, DI, AG, OR, ID, L3 = [], [], [], [], [], [], [], []
    VD = []
    bridge = float(vis_bridge_mm) > 0.0
    n_inside = 0

    band_lo = tube.r_out - 0.2                       # cm
    band_hi = tube.r_out + max_depth_mm * 0.1 + 0.2

    for root in system.roots:
        pts = root.polyline()
        if len(pts) < 2:
            continue
        rad = root.radii()

        # Cheap pre-cull on the RAW polyline: most roots in a stand never come
        # near the wall, and refining them to 0.2 mm first is where the cost was.
        _, _, rho = frame.to_local(pts)
        if rho.min() > band_hi or rho.max() < band_lo:
            continue

        # refine to <= sub_mm so the render is not visibly polygonal and sub-dx
        # tortuosity is not identically zero
        pts_r, rad_r = subdivide(pts, rad, sub_mm * 0.1, root.params.tropismS, rng)

        u, v, depth, _ = project(frame, cam, pts_r)
        diam = 2.0 * rad_r * MM_PER_CM

        ages = np.asarray(root.node_age, float)
        if len(ages) != len(pts_r):
            ages = np.interp(np.linspace(0, 1, len(pts_r)),
                             np.linspace(0, 1, len(ages)), ages)
        age_days = np.maximum(system.days - ages, 0.0)

        seg3d = np.linalg.norm(np.diff(pts_r, axis=0), axis=1) * MM_PER_CM
        d_mid = 0.5 * (depth[:-1] + depth[1:])

        inside = d_mid < 0.0
        n_inside += int(inside.sum())
        if bridge:
            vdep = _bridge_depth(depth, seg3d, vis_bridge_mm)
            vd_mid = 0.5 * (vdep[:-1] + vdep[1:])
            keep = (~inside) & (np.minimum(d_mid, vd_mid) <= max_depth_mm)
        else:
            keep = (~inside) & (d_mid <= max_depth_mm)
        if not keep.any():
            continue
        idx = np.nonzero(keep)[0]
        if bridge:
            VD.append(np.stack([vdep[idx], vdep[idx + 1]], 1))

        U.append(np.stack([u[idx], u[idx + 1]], 1))
        V.append(np.stack([v[idx], v[idx + 1]], 1))
        D.append(np.stack([depth[idx], depth[idx + 1]], 1))
        DI.append(np.stack([diam[idx], diam[idx + 1]], 1))
        AG.append(0.5 * (age_days[idx] + age_days[idx + 1]))
        OR.append(np.full(len(idx), root.order, np.int32))
        ID.append(np.full(len(idx), root.index, np.int32))
        L3.append(seg3d[idx])

    if not U:
        z2 = np.zeros((0, 2), np.float32)
        z1 = np.zeros((0,), np.float32)
        out = ProjectedSystem(z2, z2, z2, z2, z1, z1.astype(np.int32),
                              z1.astype(np.int32), z1, circ, n_inside)
        if bridge:
            out.vis_depth_mm = z2.copy()
        return out

    return ProjectedSystem(
        u_mm=np.concatenate(U).astype(np.float32),
        v_mm=np.concatenate(V).astype(np.float32),
        depth_mm=np.concatenate(D).astype(np.float32),
        diam_mm=np.concatenate(DI).astype(np.float32),
        age_days=np.concatenate(AG).astype(np.float32),
        order=np.concatenate(OR),
        root_id=np.concatenate(ID),
        seg_len3d_mm=np.concatenate(L3).astype(np.float32),
        circumference_mm=circ,
        n_culled_inside=n_inside,
        vis_depth_mm=(np.concatenate(VD).astype(np.float32) if bridge else None),
    )


def _segments_for_frame(proj: ProjectedSystem, cam: CameraParams, spec: FrameSpec,
                        tube: TubeParams, vis: VisibilityModel,
                        appear: RootAppearance, rng,
                        occl: Optional[np.ndarray] = None):
    """Window the pre-projected system into image-space capsules."""
    w, h = cam.width_px, cam.height_px
    su, sv = cam.px_per_mm_u, cam.px_per_mm_v
    r_in_mm = tube.r_in * MM_PER_CM
    u0 = r_in_mm * spec.phi_center
    v0 = spec.axial_cm * MM_PER_CM

    if len(proj) == 0:
        return [], 0.0, 0.0

    # shortest signed circumferential offset from the frame centre
    du = (proj.u_mm - u0 + 0.5 * proj.circumference_mm) % proj.circumference_mm \
        - 0.5 * proj.circumference_mm
    x = du * su + w / 2.0
    y = (proj.v_mm - v0) * sv + h / 2.0

    pad = 6.0
    inx = (np.maximum(x[:, 0], x[:, 1]) >= -pad) & (np.minimum(x[:, 0], x[:, 1]) <= w + pad)
    iny = (np.maximum(y[:, 0], y[:, 1]) >= -pad) & (np.minimum(y[:, 0], y[:, 1]) <= h + pad)
    # a segment that wrapped the seam has a huge |dx|; drop those rather than
    # drawing a spurious line straight across the frame
    nowrap = np.abs(x[:, 1] - x[:, 0]) < 0.5 * proj.circumference_mm * su
    sel = np.nonzero(inx & iny & nowrap)[0]
    if sel.size == 0:
        return [], 0.0, 0.0

    d_mid = 0.5 * (proj.depth_mm[sel, 0] + proj.depth_mm[sel, 1])
    dia_mid = 0.5 * (proj.diam_mm[sel, 0] + proj.diam_mm[sel, 1])
    if proj.vis_depth_mm is None:
        pv = vis.p_visible(d_mid, dia_mid)
    else:
        # bridged depth for DETECTION only (see project_system); the true depth
        # still orders the drawing
        pv = vis.p_visible(0.5 * (proj.vis_depth_mm[sel, 0] + proj.vis_depth_mm[sel, 1]),
                           dia_mid)

    # NOTE: the soil-cover field is NOT applied here.  Killing whole segments
    # chops roots at the sub-division granularity (~7 px) and renders them as
    # confetti; real roots are interrupted by aggregate-shaped patches, not by
    # regular dashes.  It is applied per PIXEL to the rasterised alpha instead,
    # and enters the length bookkeeping as a weight sampled along each segment.
    occ_w = None
    if occl is not None:
        xi = np.clip((0.5 * (x[sel, 0] + x[sel, 1])).astype(np.int32), 0, w - 1)
        yi = np.clip((0.5 * (y[sel, 0] + y[sel, 1])).astype(np.int32), 0, h - 1)
        occ_w = occl[yi, xi]

    # p_visible is a DETECTION PROBABILITY, not a transparency, so it gates the
    # segment rather than tinting it.  Used as alpha it composites: twenty
    # overlapping roots at p=0.15 each stack to alpha ~0.96, so a dense mat of
    # undetectable deep roots renders as an opaque sheet.  Soil is opaque --
    # stacking invisible roots cannot make them visible.
    #
    # The draw is PER ROOT, not per segment.  Depth varies smoothly along a root,
    # so neighbouring segments are almost perfectly correlated; an independent
    # draw every 0.2 mm turns each root into a dashed line of fat sausages, which
    # is both unmistakably synthetic and (because a perforated root skeletonises
    # into a network) several times too long.  One threshold per root means a root
    # shows exactly the contiguous stretch that runs close enough to the wall,
    # which is what a real frame shows.
    n_roots = int(proj.root_id.max()) + 1 if len(proj) else 1
    u_root = rng.random(n_roots)
    fade = float(appear.vis_fade) if appear.palette is not None else 0.0
    if fade > 0.0:
        # soft gate: t = 1 above the threshold, falling to 0 over `fade` below it
        t_all = np.clip((pv - u_root[proj.root_id[sel]] + fade) / fade, 0.0, 1.0)
        t_all = t_all * t_all * (3.0 - 2.0 * t_all)
        live = t_all >= float(appear.fade_min)
        t_fade = t_all[live]
    else:
        live = pv > u_root[proj.root_id[sel]]
        t_fade = None
    if appear.root_keep < 1.0:
        kept = np.random.default_rng([int(appear.keep_seed), 3]).random(n_roots) < appear.root_keep
        live = live & kept[proj.root_id[sel]]
        if t_fade is not None:
            t_fade = t_all[live]
    sel = sel[live]
    if sel.size == 0:
        return [], 0.0, 0.0
    d_mid = d_mid[live]
    dia_mid = dia_mid[live]
    wgt = np.ones(sel.shape) if occ_w is None else occ_w[live]
    if appear.palette is not None and appear.min_planarity > 0.0:
        l2 = np.hypot(x[sel, 1] - x[sel, 0], y[sel, 1] - y[sel, 0]) / su
        # thick roots (>= 20 px) must lie flatter against the glass: steep thick stretches drew
        # as stacks of translucent discs
        mp = np.where(dia_mid * su >= 20.0, max(float(appear.min_planarity), 0.6),
                      float(appear.min_planarity))
        flat = l2 >= mp * np.maximum(proj.seg_len3d_mm[sel], 1e-6)
        sel, d_mid, dia_mid, wgt = sel[flat], d_mid[flat], dia_mid[flat], wgt[flat]
        if t_fade is not None:
            t_fade = t_fade[flat]
        # a root whose drawn stretch is shorter than ~2.5x its width is a blob
        rid_s = proj.root_id[sel]
        l2px = np.hypot(x[sel, 1] - x[sel, 0], y[sel, 1] - y[sel, 0])
        tot = np.bincount(rid_s, weights=l2px, minlength=n_roots)
        wid = np.bincount(rid_s, weights=dia_mid * su * l2px, minlength=n_roots) / np.maximum(tot, 1e-6)
        long_ = tot[rid_s] >= 2.5 * np.maximum(wid[rid_s], 2.0)
        sel, d_mid, dia_mid, wgt = sel[long_], d_mid[long_], dia_mid[long_], wgt[long_]
        if t_fade is not None:
            t_fade = t_fade[long_]
        if sel.size == 0:
            return [], 0.0, 0.0

    len3d = float(np.sum(proj.seg_len3d_mm[sel] * wgt))
    seg2d = np.hypot(x[sel, 1] - x[sel, 0], y[sel, 1] - y[sel, 0]) / su
    len2d = float(np.sum(seg2d * wgt))

    if appear.palette is not None:
        # v3 palette: colour per ROOT (pale tip -> browner base), luma-only
        # floor on the root's own polarity side for this soil
        pols = appear.root_polarities()
        tex = float(appear.texture)
        if tex > 0.0:
            # smooth per-root noise along the root: two sinusoids in image space
            # with per-root phases/frequencies (wavelength ~2-6 mm)
            nr = len(appear.palette)
            tr = np.random.default_rng([int(n_roots), int(sel.size), 7])
            ph = tr.uniform(0, 2 * np.pi, (nr, 2))
            fq = tr.uniform(1.0 / 6.0, 1.0 / 2.0, (nr, 2)) * 2 * np.pi / su
            xm = 0.5 * (x[sel, 0] + x[sel, 1])
            ym = 0.5 * (y[sel, 0] + y[sel, 1])
        film_u = np.random.default_rng([len(appear.palette), 11]).random(len(appear.palette))
        bank = appear.root_bank
        if bank is not None and len(bank):
            # arclength along each root (px) at every segment start, in segment order
            rid_s = proj.root_id[sel]
            l2px = np.hypot(x[sel, 1] - x[sel, 0], y[sel, 1] - y[sel, 0])
            order = np.lexsort((np.arange(sel.size), rid_s))
            cs = np.cumsum(l2px[order])
            first = np.r_[True, rid_s[order][1:] != rid_s[order][:-1]]
            base = np.maximum.accumulate(np.where(first, cs - l2px[order], 0.0))
            s0 = np.empty(sel.size)
            s0[order] = cs - l2px[order] - base
            # one strip per root, of similar width (root mean width over its segments)
            wsum = np.bincount(rid_s, weights=dia_mid * su * l2px, minlength=n_roots)
            lsum = np.bincount(rid_s, weights=l2px, minlength=n_roots)
            rw = wsum / np.maximum(lsum, 1e-6)
            tu = np.random.default_rng([len(appear.palette), 13]).random(len(appear.palette))
            tex_of = {}
        segs = []
        for k, (i, d, dm) in enumerate(zip(sel, d_mid, dia_mid)):
            rid = int(proj.root_id[i])
            col, op = appear.palette_colour_opacity(
                rid, float(proj.age_days[i]), float(dm) * su,
                appear.opacity(float(dm)), float(pols[rid]))
            blur = 0.0
            if appear.chroma_scale != 1.0 or tex > 0.0:
                c = np.asarray(col, np.float64)
                lc = float(c @ _LUMA)
                c = lc + float(appear.chroma_scale) * (c - lc)
                if tex > 0.0:
                    r_ = rid % nr
                    n_ = 0.5 * (np.sin(fq[r_, 0] * xm[k] + ph[r_, 0])
                                + np.sin(fq[r_, 1] * ym[k] + ph[r_, 1]))
                    c = c * math.exp(tex * n_)
                col = tuple(float(v) for v in np.clip(c, 0.0, 1.0))
            w_px = float(dm) * su
            if appear.soil_film > 0.0 and w_px >= 3.0:
                # a film dims a thin root; it barely hides a thick one
                k_w = min(1.0, max(0.2, 4.0 / w_px))
                op = float(op) * (1.0 - float(appear.soil_film) * k_w * film_u[rid % len(film_u)])
            if t_fade is not None:
                op = float(op) * float(t_fade[k])
                blur = float(appear.fade_blur_px) * (1.0 - float(t_fade[k]))
            if (appear.soil_film > 0.0 or t_fade is not None) and w_px >= 6.0:
                # film x fade must not leave a labelled thick root a ghost
                op = max(float(op), 0.3)
            shade = float(appear.shading) if float(dm) * su >= 3.0 else 0.0
            tex_id, s0_k, t_sc, tex_prof = -1, 0.0, 1.0, False
            if bank is not None and len(bank) and rw[rid] >= 3.0:
                if rid not in tex_of:
                    tex_of[rid] = bank.pick(rw[rid], tu[rid % len(tu)])
                tex_id = tex_of[rid]
                s0_k = float(s0[k])
                # same magnification along as across: a thick root looks like a magnified thin
                # one (stretching only across turned texture detail into rings)
                t_sc = float(bank.widths[tex_id]) / max(float(dm) * su, 1.0)
                tex_prof = appear.tex_max_mag > 0 and 1.0 / max(t_sc, 1e-6) > appear.tex_max_mag
                shade = 0.0                       # the real strip has its own profile
            elif bank is not None and len(bank):
                # thin root: a real root colour (bank strip core) at the palette luma
                if rid not in tex_of:
                    tex_of[rid] = bank.pick(max(rw[rid], 3.0), tu[rid % len(tu)])
                mc = np.asarray(bank.mean_rgb[tex_of[rid]], np.float64)
                lc_ = float(np.asarray(col) @ _LUMA)
                col = tuple(float(v) for v in np.clip(mc * lc_ / max(float(mc @ _LUMA), 1e-3), 0, 1))
            segs.append(R.Segment(
                shade=shade, blur=blur, tex_id=tex_id, s0=s0_k, tex_scale=t_sc, tex_profile=tex_prof,
                tex_mix=float(appear.tex_mix),
                x0=float(x[i, 0]), y0=float(y[i, 0]),
                x1=float(x[i, 1]), y1=float(y[i, 1]),
                r0=float(0.5 * proj.diam_mm[i, 0] * su),
                r1=float(0.5 * proj.diam_mm[i, 1] * su),
                depth=float(d), visibility=1.0, opacity=op,
                diameter_mm=float(dm), order=int(proj.order[i]),
                root_id=rid, colour=col,
            ))
        if appear.tex_max_mag > 0:
            _smooth_thick_roots(segs)
        return segs, len3d, len2d

    if appear.target_n <= 0.0:
        segs = [
            R.Segment(
                x0=float(x[i, 0]), y0=float(y[i, 0]),
                x1=float(x[i, 1]), y1=float(y[i, 1]),
                r0=float(0.5 * proj.diam_mm[i, 0] * su),
                r1=float(0.5 * proj.diam_mm[i, 1] * su),
                depth=float(d), visibility=1.0,
                opacity=appear.opacity(float(dm)),
                diameter_mm=float(dm), order=int(proj.order[i]),
                root_id=int(proj.root_id[i]),
                colour=appear.colour(float(proj.age_days[i])),
            )
            for i, d, dm in zip(sel, d_mid, dia_mid)
        ]
        return segs, len3d, len2d

    # v3: each segment lifted to the contrast target for ITS width (px)
    segs = []
    for i, d, dm in zip(sel, d_mid, dia_mid):
        col, op = appear.segment_colour_opacity(float(proj.age_days[i]), float(dm) * su,
                                                appear.opacity(float(dm)))
        segs.append(R.Segment(
            x0=float(x[i, 0]), y0=float(y[i, 0]),
            x1=float(x[i, 1]), y1=float(y[i, 1]),
            r0=float(0.5 * proj.diam_mm[i, 0] * su),
            r1=float(0.5 * proj.diam_mm[i, 1] * su),
            depth=float(d), visibility=1.0, opacity=op,
            diameter_mm=float(dm), order=int(proj.order[i]),
            root_id=int(proj.root_id[i]), colour=col,
        ))
    return segs, len3d, len2d


def render_frame(system: RootSystem, tube: TubeParams, cam: CameraParams,
                 spec: FrameSpec, rng,
                 vis: Optional[VisibilityModel] = None,
                 appear: Optional[RootAppearance] = None,
                 soil_style: Optional[S.SoilStyle] = None,
                 optics_style: Optional[O.OpticsStyle] = None,
                 background: Optional[np.ndarray] = None,
                 soil_cache=None,
                 cover: float = 0.14,
                 mask_threshold: Optional[float] = None,
                 proj: Optional[ProjectedSystem] = None,
                 frame: Optional[TubeFrame] = None,
                 tex: Optional[S.TextureCache] = None,
                 keep_layers: bool = False,
                 mask_mode: str = "calibrated",
                 mask_min_width_px: float = 1.0,
                 width_traits: bool = False) -> FrameResult:
    """Render one minirhizotron frame with full ground truth.

    Pass ``proj`` (from :func:`project_system`) when cutting many frames from one
    system -- otherwise the sub-millimetre refinement is redone every frame.

    v3 options (defaults = the v2 frame, bit for bit):

    ``mask_mode='geometric'``  label everything that is drawn: ``mask = cover >=
        0.5 & occl >= 0.5`` with radii floored at ``mask_min_width_px / 2``
        (:func:`mrsim.raster.geometric_mask`).  Independent of colour and
        opacity, so no contrast knob can move a label; no threshold bisection
        (meta ``mask_threshold`` is NaN).  ``diameter_label_mm`` is then the
        max-cover segment's diameter, > 0 exactly on the mask.
    ``appear.target_n > 0``    per-segment contrast floor
        (:meth:`RootAppearance.segment_colour_opacity`).  The background is
        rendered FIRST to measure the soil the roots must stand out from (its
        mean luma, and its luma std after the PSF with sensor noise in
        quadrature).  With a cached ``tex`` that draws no random numbers, so
        nothing else moves.
    ``width_traits``           add the v3 width traits (w2e quantiles, fine
        share, true-width length bins) to ``traits_observable``.
    """
    if mask_mode not in MASK_MODES:
        raise ValueError("mask_mode must be one of %s, not %r" % (MASK_MODES, mask_mode))
    geometric = mask_mode == "geometric"
    w, h = cam.width_px, cam.height_px
    vis = vis or VisibilityModel()
    appear = appear or RootAppearance.sample(rng)
    soil_style = soil_style or S.SoilStyle.sample(rng, cam.px_per_mm_u)
    optics_style = optics_style or O.OpticsStyle.sample(rng, cam.px_per_mm_u)

    if proj is None:
        proj = project_system(system, tube, cam, rng, frame=frame)

    # soil-cover field: roots are interrupted along their length by crumbs lying
    # on top of them at the glass, which is what makes one root read as a chain
    # of visible fragments rather than one long clean line
    if tex is not None:
        soil_cache, occl = tex.crop((h, w), rng)
    else:
        occl = S.occlusion_field(
            (h, w), rng,
            grain_px=max(6.0, soil_style.cover_grain_mm * cam.px_per_mm_u),
            cover=cover)

    v3_meta: Dict[str, object] = {}
    if appear.target_n > 0.0 or (appear.palette is not None and appear.min_delta > 0.0):
        if background is None:
            background = S.soil_background((h, w), rng, soil_style, soil_cache)
        appear, v3_meta = _with_soil_context(appear, background, optics_style)

    segs, len3d, len2d = _segments_for_frame(proj, cam, spec, tube, vis,
                                             appear, rng, occl=occl)
    out = R.rasterize(segs, w, h,
                      cover_min_radius_px=(0.5 * float(mask_min_width_px)
                                           if geometric else None),
                      textures=appear.root_bank)

    # soil crumbs and water films lying on the glass IN FRONT of the roots: a
    # per-pixel transmittance, so one root reads as a chain of visible stretches
    # with irregular, aggregate-shaped boundaries
    if appear.palette is not None and appear.cover_relief_px > 0.0:
        # crumbs hide thin roots, rarely interrupt thick ones
        wpx = np.maximum(out.diameter_mm, out.cover_diam_mm if out.cover_diam_mm is not None
                         else 0.0) * cam.px_per_mm_u
        occl = 1.0 - (1.0 - occl) / (1.0 + (wpx / float(appear.cover_relief_px)) ** 2)
        occl = occl.astype(np.float32)
    out.alpha = out.alpha * occl
    out.rgb = out.rgb * occl[..., None]

    if background is None:
        background = S.soil_background((h, w), rng, soil_style, soil_cache)

    root_rgb = np.zeros_like(background)
    lit = out.alpha > 1e-6
    root_rgb[lit] = out.rgb[lit] / out.alpha[lit][:, None]
    root_rgb = S.match_local_photometry(root_rgb, background, out.alpha, rng,
                                        strength=_PHOTOMETRY_STRENGTH)
    comp = background * (1.0 - out.alpha[..., None]) + root_rgb * out.alpha[..., None]
    if appear.root_hairs > 0.0 and segs:
        comp = _draw_root_hairs(comp, segs, appear, cam.px_per_mm_u)
    rgb = O.apply_optics(comp, rng, optics_style)

    diam_label = None
    if geometric:
        # label everything that is drawn (see raster.geometric_mask)
        mask_threshold = float("nan")
        mask = R.geometric_mask(out.cover, occl)
        diam_label = R.diameter_on_mask(out.cover_diam_mm, mask)
    else:
        # the binary mask is a calibrated approximation of a quantity we know exactly
        if mask_threshold is None:
            mask_threshold = R.calibrate_mask_threshold(out.alpha,
                                                        len2d * cam.px_per_mm_u)
        mask = R.mask_from_alpha(out.alpha, mask_threshold)
        if width_traits:
            diam_label = R.diameter_on_mask(out.diameter_mm, mask)

    area_mm2 = (w / cam.px_per_mm_u) * (h / cam.px_per_mm_v)
    if width_traits:
        obs = T.frame_traits(mask, cam.px_per_mm_u, cam.px_per_mm_v, area_mm2,
                             alpha=out.alpha, widths=True, true_diam_mm=diam_label)
    else:
        obs = T.frame_traits(mask, cam.px_per_mm_u, cam.px_per_mm_v, area_mm2,
                             alpha=out.alpha)
    obs["length_fragments_3d_mm"] = float(len3d)
    obs["length_fragments_2d_mm"] = float(len2d)
    obs["length_union_2d_mm"] = float(obs["visible_length_mm"])
    obs["overlap_fraction"] = float(out.overlap_fraction)
    obs["n_fragments"] = int(_count_fragments(mask))

    meta = {
        "camera": cam.name,
        "n_segments_inside_tube_culled": int(proj.n_culled_inside),
        "width_px": w, "height_px": h,
        "px_per_mm_u": cam.px_per_mm_u, "px_per_mm_v": cam.px_per_mm_v,
        "circumferential_scale": tube.circumferential_scale,
        "tube_id_cm": tube.id_cm, "tube_od_cm": tube.od_cm,
        "tube_angle_from_vertical_deg": spec.tube_angle_deg,
        "axial_cm": spec.axial_cm,
        "soil_depth_cm": T.soil_depth_cm(spec.axial_cm, tube.angle_from_vertical_deg),
        "phi_center_rad": spec.phi_center,
        "mask_threshold": float(mask_threshold),
        "effective_dof_mm_at_0p35": float(vis.effective_dof_mm(0.35)),
        "soil_cover": float(cover),
        "species": system.species,
        "system_age_days": float(system.days),
    }
    if geometric or width_traits or appear.target_n > 0.0 or appear.palette is not None:
        meta.update({"mask_mode": mask_mode,
                     "mask_min_width_px": float(mask_min_width_px),
                     "psf_sigma_px": float(optics_style.psf_sigma_px),
                     "contrast_target_n": float(appear.target_n)})
        meta.update(v3_meta)
    if appear.palette is not None:
        meta.update(palette_stats(segs, appear.palette, float(
            np.asarray(background, np.float32).reshape(-1, 3).mean(0) @ _LUMA)))

    return FrameResult(
        rgb=rgb, alpha=out.alpha, mask=mask,
        diameter_mm=out.diameter_mm, order=out.order, instance=out.instance,
        traits_observable=obs,
        traits_architecture=T.system_traits(system),
        meta=meta,
        background=(background.astype(np.float32) if keep_layers else None),
        root_rgb=(root_rgb.astype(np.float32) if keep_layers else None),
        diameter_label_mm=diam_label,
    )


def _smooth_thick_roots(segs, min_width_px: float = 10.0) -> None:
    """Smooth colour, opacity and blur along each THICK root (in place).

    A root is many short capsules with round caps of its own radius; on a 50-100 px root every
    small step in colour/opacity between consecutive capsules (fade, contrast lift) shows as a
    disc edge.  A Gaussian over arc length with sigma = the root's width removes the steps."""
    by_root: Dict[int, list] = {}
    for k, s in enumerate(segs):
        by_root.setdefault(s.root_id, []).append(k)
    for ks in by_root.values():
        if len(ks) < 3:
            continue
        w = np.array([segs[k].r0 + segs[k].r1 for k in ks])
        if float(np.median(w)) < min_width_px:
            continue
        s0 = np.array([segs[k].s0 for k in ks])
        sig = max(float(np.median(w)), 1.0)
        W = np.exp(-0.5 * ((s0[:, None] - s0[None, :]) / sig) ** 2)
        W /= W.sum(1, keepdims=True)
        col = W @ np.array([segs[k].colour for k in ks], np.float64)
        op = W @ np.array([segs[k].opacity for k in ks], np.float64)
        bl = W @ np.array([segs[k].blur for k in ks], np.float64)
        for j, k in enumerate(ks):
            segs[k].colour = tuple(float(v) for v in col[j])
            segs[k].opacity = float(op[j])
            segs[k].blur = float(bl[j])


def _draw_root_hairs(comp: np.ndarray, segs, appear: RootAppearance, px_per_mm: float) -> np.ndarray:
    """Fine root hairs on part of the roots: short, faint 1-px strokes leaving the root edge at
    roughly right angles, in the root's colour.  Own random stream (from the segment count), so
    the frame's main stream is untouched.  Image only: masks and labels never see them."""
    import cv2
    rng_h = np.random.default_rng([len(segs), int(1e3 * appear.root_hairs), 21])
    n_roots = max(int(max(s.root_id for s in segs)) + 1, 1)
    has = rng_h.random(n_roots) < float(appear.root_hair_p)
    h, w = comp.shape[:2]
    col = np.zeros((h, w, 3), np.uint8)       # hair colour (drawn 2 px wide to cover the AA fringe)
    acc = np.zeros((h, w), np.uint8)          # hair opacity (anti-aliased 1 px strokes)
    for s in segs:
        r = 0.5 * (s.r0 + s.r1)
        if not has[s.root_id] or r < 0.75 or r > 6.0 or s.opacity < 0.3:
            continue
        dx, dy = s.x1 - s.x0, s.y1 - s.y0
        L = math.hypot(dx, dy)
        if L < 1e-3:
            continue
        ux, uy = dx / L, dy / L
        c255 = tuple(int(255 * min(1.0, 1.08 * float(v))) for v in s.colour)
        for _ in range(int(rng_h.poisson(appear.root_hairs * L / px_per_mm * 2.0))):
            t = rng_h.random()
            side = 1.0 if rng_h.random() < 0.5 else -1.0
            ang = rng_h.normal(0.0, 0.45)                      # around the normal
            nx, ny = -uy * side, ux * side
            ca, sa = math.cos(ang), math.sin(ang)
            hx, hy = nx * ca - ny * sa, nx * sa + ny * ca
            ln = rng_h.uniform(0.15, 0.6) * px_per_mm          # 0.15-0.6 mm long
            px0 = s.x0 + dx * t + hx * r * 0.9
            py0 = s.y0 + dy * t + hy * r * 0.9
            a = int(255 * float(rng_h.uniform(0.15, 0.45)) * float(s.opacity))
            p0 = (int(round(px0 * 4)), int(round(py0 * 4)))
            p1 = (int(round((px0 + hx * ln) * 4)), int(round((py0 + hy * ln) * 4)))
            cv2.line(col, p0, p1, c255, 2, cv2.LINE_8, shift=2)
            cv2.line(acc, p0, p1, a, 1, cv2.LINE_AA, shift=2)
    af = acc.astype(np.float32)[..., None] / 255.0
    return comp * (1.0 - af) + (col.astype(np.float32) / 255.0) * af


def _with_soil_context(appear: RootAppearance, background: np.ndarray,
                       optics_style: O.OpticsStyle):
    """Fill the contrast-floor context from this frame's own soil and optics.

    ``soil_luma`` = mean background luma; ``soil_sigma`` = std of the luma of
    the PSF-blurred background, with the sensor noise added in quadrature
    (per-channel shot + read noise, independent across channels, seen
    through the luma weights) -- i.e. what contrast_n divides by.  Draws
    nothing.
    """
    bg = np.asarray(background, np.float32)
    l0 = bg @ _LUMA.astype(np.float32)
    ls = float(l0.mean())
    bl = O.apply_psf(bg, float(optics_style.psf_sigma_px))
    tex_sd = float((np.asarray(bl, np.float32) @ _LUMA.astype(np.float32)).std())
    ch_var = (float(optics_style.shot_noise) ** 2 * max(ls, 0.0)
              + float(optics_style.read_noise) ** 2)
    noise_sd = math.sqrt(ch_var) * float(np.sqrt(np.sum(_LUMA ** 2)))
    sigma = math.sqrt(tex_sd ** 2 + noise_sd ** 2)
    ap = appear.with_frame_context(soil_luma=ls, soil_sigma=sigma,
                                   psf_sigma_px=float(optics_style.psf_sigma_px))
    return ap, {"soil_luma": ls, "soil_sigma": sigma}


def palette_stats(segs, palette: RootPalette, soil_luma: float) -> Dict[str, object]:
    """What the palette actually drew in a frame (colours as drawn, before
    photometry and optics):

    ``root_palette_mix``  share of drawn 2-D length per palette entry,
                          'white:0.312,tan:0.405' (palette order, zeros omitted)
    ``dark_root_share``   share of drawn ROOTS whose median luma is below the
                          soil's mean luma (dark polarity)
    ``root_luma_med``     median segment luma
    ``root_chroma_med``   median segment chroma (max - min channel)
    """
    if not segs:
        return {"root_palette_mix": "", "dark_root_share": float("nan"),
                "root_luma_med": float("nan"), "root_chroma_med": float("nan")}
    col = np.array([s.colour for s in segs], np.float64)
    rid = np.array([s.root_id for s in segs], np.int64)
    ln = np.array([math.hypot(s.x1 - s.x0, s.y1 - s.y0) for s in segs], np.float64)
    lum = col @ _LUMA
    chroma = col.max(1) - col.min(1)
    kind = np.asarray(palette.kind, np.int64)[rid]
    tot = max(float(ln.sum()), 1e-9)
    by = np.bincount(kind, weights=ln, minlength=len(PALETTE_NAMES)) / tot
    mix = ",".join("%s:%.3f" % (n, v) for n, v in zip(PALETTE_NAMES, by) if v > 0)
    o = np.argsort(rid, kind="stable")
    _, starts = np.unique(rid[o], return_index=True)
    meds = [float(np.median(g)) for g in np.split(lum[o], starts[1:])]
    dark = float(np.mean(np.asarray(meds) < soil_luma)) if len(meds) else float("nan")
    return {"root_palette_mix": mix, "dark_root_share": dark,
            "root_luma_med": float(np.median(lum)), "root_chroma_med": float(np.median(chroma))}


def _count_fragments(mask: np.ndarray) -> int:
    """Connected components -- a real frame shows fragments, not architectures."""
    if not mask.any():
        return 0
    try:
        from scipy import ndimage as ndi
        _, n = ndi.label(mask, structure=np.ones((3, 3), np.uint8))
        return int(n)
    except Exception:
        return -1


# --------------------------------------------------------------------------
# frame sampling
# --------------------------------------------------------------------------
def grow_around_tube(plant: PlantParams, tube: TubeParams, days: float, rng,
                     lateral_offset_cm: Optional[float] = None,
                     dt: float = 1.0, max_roots: int = 20000):
    """Grow a system next to an installed tube, confined to the outside of it.

    Two things the naive call gets wrong.

    **The tube is not at the seed.**  A minirhizotron is augered in beside the
    plant, so the seed sits some centimetres away laterally and only a fraction of
    the root system ever reaches the wall.  Growing with the tube axis through the
    seed puts every root inside the tube.

    **Growth must be confined.**  Without the one-sided barrier from
    :meth:`TubeFrame.confinement`, roots pass straight through the wall and are
    then silently culled at projection time, which truncates them mid-root and
    inflates the apparent fragment count.

    ``lateral_offset_cm`` is the dominant control on how full frames are, and
    therefore on the empty-frame fraction.  Calibrate it against the real corpus
    rather than leaving it at the default.
    """
    from .grow import grow as _grow

    if lateral_offset_cm is None:
        lateral_offset_cm = tube.r_out + float(rng.uniform(1.0, 6.0))
    # place the tube axis to the side of the seed, at the surface
    origin = np.array([float(lateral_offset_cm), 0.0, 0.0])
    frame = TubeFrame.build(tube, origin=origin)
    system = _grow(plant, days=days, rng=rng, dt=dt, max_roots=max_roots,
                   confine=frame.confinement())
    return system, frame


def grow_stand(plant: PlantParams, tube: TubeParams, days: float, rng,
               n_plants: int = 4, spacing_cm: float = 20.0,
               row_offset_cm: Optional[float] = None, dt: float = 1.0,
               max_roots_per_plant: int = 8000,
               plants: Optional[Sequence[PlantParams]] = None,
               cull_margin_cm: Optional[float] = None,
               axial_window_cm: Optional[Tuple[float, float]] = None):
    """Grow a STAND of plants along an installed tube.

    A single plant beside a 1-2 m tube leaves almost the whole tube bare, so every
    frame comes out empty.  Real minirhizotrons are installed within a crop stand
    and pass through soil rooted by several plants, which is what gives a realistic
    root density along the tube -- and therefore a realistic empty-frame fraction.

    Plants are placed in a row parallel to the tube's surface projection at
    ``row_offset_cm`` from the axis.  Together with ``spacing_cm`` this is the main
    control on frame fullness; calibrate both against the real corpus rather than
    trusting the defaults.

    v3 options (defaults reproduce the old stand exactly):

    ``plants``          one preset per plant (len == n_plants), for mixed
                        stands; ``merged.species`` becomes e.g.
                        ``'cotton+switchgrass'`` (first-appearance order).
    ``cull_margin_cm``  relevance culling (:meth:`TubeFrame.relevance`): a
                        lateral whose base is farther than its reach + margin
                        from the tube -- or, with ``axial_window_cm``, from
                        that axial window -- is never grown.  With the cap
                        raised to 20000 roots per plant this is what keeps
                        growth affordable; ``merged.n_culled`` counts them.

    ``merged.root_plant`` (plant index of every root) is always set.
    """
    from .grow import RootSystem
    from .grow import grow as _grow

    if row_offset_cm is None:
        row_offset_cm = tube.r_out + 4.0
    if plants is not None and len(plants) != n_plants:
        raise ValueError("plants has %d presets for %d plants" % (len(plants), n_plants))

    frame = TubeFrame.build(tube, origin=np.zeros(3))
    confine = frame.confinement()
    # direction along the tube, projected onto the horizontal plane
    a = frame.axis.copy()
    a[2] = 0.0
    n = np.linalg.norm(a)
    along = a / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    side = np.cross(np.array([0.0, 0.0, 1.0]), along)

    merged = RootSystem(species=plant.name, days=days)
    if plants is not None:
        names: List[str] = []
        for p in plants:
            if p.name not in names:
                names.append(p.name)
        merged.species = "+".join(names)
    keep_of: Dict[int, object] = {}
    plant_of: List[int] = []
    span = (n_plants - 1) * spacing_cm
    for i in range(n_plants):
        p_i = plants[i] if plants is not None else plant
        keep = None
        if cull_margin_cm is not None:
            if id(p_i) not in keep_of:
                reach = {int(st): p_i.reach_cm(int(st)) for st in p_i.types}
                keep_of[id(p_i)] = frame.relevance(reach, float(cull_margin_cm),
                                                   axial_window_cm)
            keep = keep_of[id(p_i)]
        t = -span / 2.0 + i * spacing_cm
        sgn = 1.0 if (i % 2 == 0) else -1.0
        offset = along * t + side * (sgn * row_offset_cm)
        sub = _grow(p_i, days=days, rng=rng, dt=dt,
                    max_roots=max_roots_per_plant, confine=confine,
                    seed_offset=offset, keep_branch=keep)
        base = len(merged.roots)
        for r in sub.roots:
            r.index += base
            if r.parent is not None:
                r.parent += base
            merged.roots.append(r)
        merged.n_culled += int(sub.n_culled)
        plant_of.extend([i] * len(sub.roots))
    merged.root_plant = np.asarray(plant_of, np.int32)
    return merged, frame


def grow_and_project(plant: PlantParams, tube: TubeParams, cam: CameraParams,
                     days: float, rng, n_plants: int = 4, spacing_cm: float = 20.0,
                     row_offset_cm: Optional[float] = None,
                     max_roots_per_plant: int = 8000,
                     plants: Optional[Sequence[PlantParams]] = None,
                     cull_margin_cm: Optional[float] = None,
                     axial_range_cm: Optional[Tuple[float, float]] = None,
                     vis_bridge_mm: float = 0.0):
    """Grow a stand and project it once: ``(system, frame, proj)``.

    The ONE growth path shared by the render (``ginger._render_system``), the
    contrast pilot, the density calibration and the diameter calibration, so
    calibration and render cannot diverge.  With every v3 knob at its default
    it makes exactly the two old calls::

        grow_stand(plant, tube, days, rng, n_plants=, spacing_cm=,
                   row_offset_cm=, max_roots_per_plant=)
        project_system(system, tube, cam, rng, frame=frame)

    ``axial_range_cm`` is the range frame CENTRES are drawn from; for culling it
    is widened by half a frame height, so every pixel any frame can show is
    inside the window.
    """
    window = None
    if cull_margin_cm is not None and axial_range_cm is not None:
        half_cm = 0.5 * cam.height_px / cam.px_per_mm_v / MM_PER_CM
        window = (float(axial_range_cm[0]) - half_cm, float(axial_range_cm[1]) + half_cm)
    if plants is None and cull_margin_cm is None:
        system, frame = grow_stand(plant, tube, days, rng, n_plants=n_plants,
                                   spacing_cm=spacing_cm, row_offset_cm=row_offset_cm,
                                   max_roots_per_plant=max_roots_per_plant)
    else:
        system, frame = grow_stand(plant, tube, days, rng, n_plants=n_plants,
                                   spacing_cm=spacing_cm, row_offset_cm=row_offset_cm,
                                   max_roots_per_plant=max_roots_per_plant,
                                   plants=plants, cull_margin_cm=cull_margin_cm,
                                   axial_window_cm=window)
    if vis_bridge_mm and vis_bridge_mm > 0:
        proj = project_system(system, tube, cam, rng, frame=frame,
                              vis_bridge_mm=vis_bridge_mm)
    else:
        proj = project_system(system, tube, cam, rng, frame=frame)
    return system, frame, proj


def sample_frame_specs(n: int, cam: CameraParams, tube: TubeParams, rng,
                       below_tube_bias: float = 0.55,
                       axial_range_cm: Optional[Tuple[float, float]] = None
                       ) -> List[FrameSpec]:
    """Sample where on the tube to image.

    Azimuth is NOT uniform.  Unslurried tubes carry ~55% more root length below
    than above (126.4 vs 81.5 cm in winter wheat), so an isotropic azimuthal
    sampler is wrong: in an unrolled scan it shows up as a density gradient along
    the circumferential axis, and for a matrix camera it depends on which side of
    the tube was imaged.  ``phi = 0`` is the bottom of the tube.
    """
    if axial_range_cm is None:
        axial_range_cm = (2.0, max(4.0, tube.length_cm * 0.55))
    specs = []
    for _ in range(n):
        if rng.random() < below_tube_bias:
            phi = float(rng.normal(0.0, 0.9))            # clustered below
        else:
            phi = float(rng.uniform(-math.pi, math.pi))
        specs.append(FrameSpec(
            axial_cm=float(rng.uniform(*axial_range_cm)),
            phi_center=phi,
            tube_angle_deg=float(np.clip(rng.normal(37.5, 6.0), 0.0, 60.0)),
        ))
    return specs
