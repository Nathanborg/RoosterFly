"""Procedural soil background, aggregates, and the soil-cover occlusion field.

Two roles.

**Background texture.**  Useful for domain randomisation, but note the honest
ordering: procedural noise will never contain your tube scratches, condensation,
LED hotspots, JPEG blocking, biofilm or decayed-root ghosts, and those are
precisely the published false-positive drivers.  Mining root-free patches from a
real corpus (:func:`BackgroundBank`) is strictly more in-domain and is the
higher-leverage option wherever a real corpus exists.  Procedural soil is the
fallback and the randomisation envelope, not the target.

**The occlusion field.**  This is the part that decides whether a frame reads as
real.  Purely radial (depth-based) occlusion yields long, clean roots that fade
tidily at both ends.  Real minirhizotron roots are *speckled*: soil crumbs and
water films sit on top of them at the glass, so one root reads as a chain of
visible segments with irregular, aggregate-shaped boundaries.  :func:`occlusion_field`
generates that field, correlated with the same aggregate texture that produces the
visible soil grain, so the burial pattern and the soil look are consistent.
"""
from __future__ import annotations

import glob
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# noise primitives (pure numpy, no compiled noise dependency)
# --------------------------------------------------------------------------
def fbm_spectral(shape, rng, beta: float = 2.0, lo_cut: float = 0.0):
    """Fractional Brownian field by spectral synthesis.

    Cheaper and better controlled than summing octaves of value noise: one FFT
    pair gives an exact 1/f**beta power spectrum, which is also how the real
    corpus's soil texture is characterised in notebook 00.
    """
    h, w = shape
    white = rng.normal(size=(h, w))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fy * fy + fx * fx)
    f[0, 0] = 1.0
    amp = 1.0 / (f ** (beta / 2.0))
    amp[0, 0] = 0.0
    if lo_cut > 0:
        amp[f < lo_cut] = 0.0
    out = np.fft.ifft2(np.fft.fft2(white) * amp).real
    s = out.std()
    return (out / s if s > 1e-12 else out).astype(np.float32)


def worley(shape, rng, cell_px: float = 24.0, jitter: float = 0.9,
           order: int = 0):
    """Jittered-grid cellular noise -- the soil aggregate field.

    Returns distance to the ``order``-th nearest feature point, normalised by the
    cell size.  Vectorised over a 3x3 neighbourhood, so it is O(pixels).
    """
    h, w = shape
    ny = max(1, int(np.ceil(h / cell_px)) + 2)
    nx = max(1, int(np.ceil(w / cell_px)) + 2)
    pts = (np.stack(np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij"), -1)
           .astype(np.float32))
    pts = pts + jitter * rng.random((ny, nx, 2), dtype=np.float32)
    pts *= cell_px

    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    gy = np.clip((ys / cell_px).astype(np.int32) + 1, 1, ny - 2)
    gx = np.clip((xs / cell_px).astype(np.int32) + 1, 1, nx - 2)

    if order == 0:
        # running minimum: avoids allocating and sorting an (h, w, 9) stack,
        # which was the single most expensive operation in the whole renderer
        best = np.full((h, w), np.inf, np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                py = pts[gy + dy, gx + dx, 0]
                px = pts[gy + dy, gx + dx, 1]
                np.minimum(best, np.hypot(ys - py, xs - px), out=best)
        return (best / cell_px).astype(np.float32)

    stack = np.empty((h, w, 9), np.float32)
    k = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            py = pts[gy + dy, gx + dx, 0]
            px = pts[gy + dy, gx + dx, 1]
            stack[:, :, k] = np.hypot(ys - py, xs - px)
            k += 1
    stack.partition(order, axis=2)
    return (stack[:, :, order] / cell_px).astype(np.float32)


# --------------------------------------------------------------------------
# occlusion
# --------------------------------------------------------------------------
def occlusion_field(shape, rng, grain_px: float = 22.0, cover: float = 0.14,
                    softness: float = 0.12):
    """Stochastic soil-cover transmittance in [0, 1].

    ``cover`` is the approximate fraction of the frame lying under soil crumbs.
    Multiplying the depth-based visibility by this field is what turns
    uninterrupted rendered roots into the chain-of-fragments appearance of a real
    frame, and it is what makes fragment count per frame match a real corpus.

    The field must be close to BIMODAL -- mostly fully open, with genuinely opaque
    patches and only a narrow transition.  A gradual field merely dims the roots:
    the mask still contains the whole skeleton while the analytic length is
    down-weighted, so the two disagree by tens of percent for no physical reason.
    Soil grains sitting on the glass are opaque, so this is also the truer model.
    """
    import cv2

    a = worley(shape, rng, cell_px=grain_px, order=0)
    b = fbm_spectral(shape, rng, beta=2.2)
    fieldv = 0.72 * (a - a.mean()) / (a.std() + 1e-8) + 0.28 * b

    # Low-pass BEFORE thresholding, at a fraction of the grain size.  Without
    # this the broadband component leaves structure far finer than the grain, and
    # those specks punch HOLES inside roots rather than interrupting them.  A hole
    # is much worse than it looks: skeletonisation has to route around every one,
    # so a single straight root with a dozen pinholes skeletonises to 1.5-3x its
    # own centreline length, which corrupts the length label directly.
    sigma = max(1.0, grain_px * 0.18)
    fieldv = cv2.GaussianBlur(fieldv.astype(np.float32), (0, 0), sigma)

    thr = float(np.quantile(fieldv, float(np.clip(cover, 0.0, 0.95))))
    t = (fieldv - thr) / max(softness * (fieldv.std() + 1e-8), 1e-6)
    return np.clip(0.5 + 0.5 * np.tanh(t), 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# soil appearance
# --------------------------------------------------------------------------
@dataclass
class SoilStyle:
    """Randomisation envelope for soil appearance.

    Colour is parameterised the way soil actually varies -- a narrow warm hue band
    with low chroma, varied mainly in VALUE (lightness) and moisture -- rather than
    as free RGB.  Sampling RGB channels independently produces saturated yellows,
    reds and blue-greys that no soil has, and a network trained on them learns a
    colour distribution that the real corpus never visits.

    Defaults are broad but plausible; notebook 00 narrows them onto the measured
    corpus, and the gates in :mod:`mrsim.validate` compare against that file
    rather than against absolute bands.
    """

    value: float = 0.45            # lightness of the dry soil
    hue_warmth: float = 0.55       # 0 grey-brown -> 1 strongly red-brown
    chroma: float = 0.30           # saturation, deliberately low
    grain_px: float = 14.0
    grain_contrast: float = 0.10
    fine_contrast: float = 0.06
    beta: float = 2.3
    moisture: float = 0.5          # 0 dry/pale -> 1 wet/dark
    stones_per_cm2: float = 0.25
    organics_per_cm2: float = 0.8
    px_per_mm: float = 35.0
    # Scale of the soil crumbs that lie ON the glass in front of roots.  This is
    # NOT the visual soil grain: real aggregates (0.5-5 mm) are much larger than
    # fine roots (0.1-0.5 mm).  Tying the two together makes the occlusion field
    # finer than the roots, which perforates them into lace instead of
    # interrupting them -- and a perforated root skeletonises into a network
    # several times longer than its own centreline.
    cover_grain_mm: float = 2.0

    @classmethod
    def sample(cls, rng, px_per_mm: float = 35.0) -> "SoilStyle":
        return cls(
            value=float(rng.uniform(0.22, 0.62)),
            hue_warmth=float(rng.uniform(0.25, 0.95)),
            chroma=float(rng.uniform(0.12, 0.45)),
            grain_px=float(rng.uniform(0.15, 0.7) * px_per_mm),
            grain_contrast=float(rng.uniform(0.04, 0.16)),
            fine_contrast=float(rng.uniform(0.02, 0.10)),
            beta=float(rng.uniform(1.7, 3.0)),
            moisture=float(rng.uniform(0.0, 1.0)),
            stones_per_cm2=float(rng.uniform(0.0, 1.2)),
            organics_per_cm2=float(rng.uniform(0.0, 3.0)),
            px_per_mm=float(px_per_mm),
            cover_grain_mm=float(rng.uniform(1.6, 5.0)),
        )

    def base_rgb(self) -> np.ndarray:
        """A warm, low-chroma soil colour built from value/warmth/chroma."""
        v = self.value * (1.0 - 0.35 * self.moisture)     # wet soil is darker
        c = self.chroma * (1.0 + 0.25 * self.moisture)    # and a bit richer
        r = v * (1.0 + 0.55 * c)
        g = v * (1.0 + 0.05 * c - 0.10 * c * self.hue_warmth)
        b = v * (1.0 - 0.55 * c * (0.4 + 0.6 * self.hue_warmth))
        return np.clip(np.array([r, g, b], np.float32), 0.02, 0.98)


def soil_background(shape, rng, style: Optional[SoilStyle] = None,
                    cached_texture=None):
    """Render a procedural soil background in [0, 1] RGB.

    ``cached_texture`` lets a batch of frames share the expensive structural
    fields and re-randomise only colour and illumination.  Texture generation
    dominates per-frame cost, so this is the single most effective optimisation.
    """
    h, w = shape
    style = style or SoilStyle()
    if cached_texture is None:
        cached_texture = build_soil_texture(shape, rng, style)
    grain, fine, stones, organics = cached_texture

    lum = 1.0 + style.grain_contrast * grain + style.fine_contrast * fine
    img = lum[..., None] * style.base_rgb()[None, None, :]

    # Inclusions are SPARSE and SMALL: a few light mineral grains and dark
    # organic flecks per frame.  They are blended, not stamped as hard discs --
    # a hard-edged disc reads as a printed dot, and at any real density it buries
    # the roots, which are the only thing in the frame that matters.
    if stones is not None:
        img = img * (1.0 - stones[..., None]) + \
            np.clip(img * 1.45 + 0.12, 0, 1) * stones[..., None]
    if organics is not None:
        img = img * (1.0 - 0.55 * organics[..., None])

    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _sparse_blobs(shape, rng, count: int, r_lo: float, r_hi: float,
                  softness: float = 0.6):
    """Soft-edged sparse inclusions, as a [0, 1] coverage field."""
    import cv2
    h, w = shape
    out = np.zeros((h, w), np.float32)
    if count <= 0:
        return out
    for _ in range(int(count)):
        cx, cy = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        r = float(rng.uniform(r_lo, r_hi))
        ax = max(1, int(r * rng.uniform(0.7, 1.4)))
        ay = max(1, int(r * rng.uniform(0.7, 1.4)))
        cv2.ellipse(out, (cx, cy), (ax, ay), float(rng.uniform(0, 180)),
                    0, 360, 1.0, -1)
    if softness > 0:
        out = cv2.GaussianBlur(out, (0, 0), max(0.6, softness * (r_lo + r_hi) * 0.25))
    return np.clip(out, 0.0, 1.0)


def _domain_warp(field, rng, amount: float):
    """Displace a field by smooth noise, to break up procedural regularity."""
    import cv2
    h, w = field.shape
    if amount <= 0.5:
        return field
    dx = fbm_spectral((h, w), rng, beta=3.0) * amount
    dy = fbm_spectral((h, w), rng, beta=3.0) * amount
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return cv2.remap(field.astype(np.float32), (xs + dx).astype(np.float32),
                     (ys + dy).astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def build_soil_texture(shape, rng, style: SoilStyle):
    """The expensive structural part of the soil, suitable for caching.

    Aggregate structure is two octaves of cellular noise rather than one: a single
    octave reads as a tiled polygonal pattern, which is instantly recognisable as
    procedural.
    """
    h, w = shape
    a1 = worley(shape, rng, cell_px=max(3.0, style.grain_px), order=0)
    a2 = worley(shape, rng, cell_px=max(2.0, style.grain_px * 0.42), order=0)
    agg = 0.62 * a1 + 0.38 * a2
    # A jittered grid still reads as a lattice at low contrast.  Warping it by a
    # smooth noise field breaks the row/column regularity that gives it away.
    agg = _domain_warp(agg, rng, amount=style.grain_px * 0.45)
    grain = ((agg - agg.mean()) / (agg.std() + 1e-8)).astype(np.float32)
    fine = fbm_spectral(shape, rng, beta=style.beta)

    area_cm2 = (w / max(style.px_per_mm, 1e-6) / 10.0) * \
               (h / max(style.px_per_mm, 1e-6) / 10.0)
    mm = style.px_per_mm
    stones = _sparse_blobs(shape, rng,
                           count=int(round(style.stones_per_cm2 * area_cm2)),
                           r_lo=0.25 * mm, r_hi=1.2 * mm)
    organics = _sparse_blobs(shape, rng,
                             count=int(round(style.organics_per_cm2 * area_cm2)),
                             r_lo=0.08 * mm, r_hi=0.45 * mm)
    return grain, fine, stones, organics


# --------------------------------------------------------------------------
# texture cache
# --------------------------------------------------------------------------
@dataclass
class TextureCache:
    """Oversized soil texture and occlusion field, shared across a batch.

    Generating both per frame put cellular noise at ~55% of total render time.
    They carry no ground truth, so a batch of frames can crop independent windows
    out of one larger realisation and re-randomise only colour and illumination --
    which is where the visible variety actually comes from.  Rebuild the cache
    every ``batch`` frames so the structural texture still varies across the
    dataset.
    """

    grain: np.ndarray
    fine: np.ndarray
    stones: Optional[np.ndarray]
    organics: Optional[np.ndarray]
    occl: np.ndarray

    @classmethod
    def build(cls, frame_shape, rng, style: "SoilStyle", cover: float = 0.35,
              scale: float = 1.7) -> "TextureCache":
        h = int(frame_shape[0] * scale)
        w = int(frame_shape[1] * scale)
        grain, fine, stones, organics = build_soil_texture((h, w), rng, style)
        occl = occlusion_field(
            (h, w), rng,
            grain_px=max(6.0, style.cover_grain_mm * style.px_per_mm),
            cover=cover)
        return cls(grain, fine, stones, organics, occl)

    def _window(self, shape, rng):
        h, w = shape
        H, W = self.grain.shape
        y = int(rng.integers(0, max(1, H - h + 1)))
        x = int(rng.integers(0, max(1, W - w + 1)))
        return y, x

    def crop(self, shape, rng):
        """Return ``(soil_texture_tuple, occlusion)`` for one frame."""
        h, w = shape
        y, x = self._window(shape, rng)
        sl = (slice(y, y + h), slice(x, x + w))
        tex = (self.grain[sl], self.fine[sl],
               None if self.stones is None else self.stones[sl],
               None if self.organics is None else self.organics[sl])
        # independent window for the occlusion field, so soil grain and burial
        # pattern are correlated in scale but not pixel-locked
        y2, x2 = self._window(shape, rng)
        occl = self.occl[y2:y2 + h, x2:x2 + w]
        return tex, occl


# --------------------------------------------------------------------------
# real background bank
# --------------------------------------------------------------------------
@dataclass
class BackgroundBank:
    """Root-free patches mined from a real corpus.

    Compositing synthetic roots onto real soil is probably the single
    highest-leverage realism step available, because dataset choice dominates
    architecture choice by roughly 10x in published benchmarks.

    CIRCULARITY WARNING.  If patches are admitted by "a bootstrap segmenter fires
    below threshold", the bank is systematically enriched in that segmenter's
    FALSE NEGATIVES -- i.e. real roots relabelled as soil.  Prefer selecting whole
    frames whose *image-level trait label* is ~zero, hand-audit a sample, and
    record the provenance of every patch.
    """

    patches: List[np.ndarray] = field(default_factory=list)
    provenance: List[str] = field(default_factory=list)

    @classmethod
    def from_dir(cls, path: str, limit: Optional[int] = None) -> "BackgroundBank":
        import cv2
        files = sorted(glob.glob(os.path.join(path, "*.png"))
                       + glob.glob(os.path.join(path, "*.jpg")))
        if limit:
            files = files[:limit]
        pats, prov = [], []
        for f in files:
            im = cv2.imread(f, cv2.IMREAD_COLOR)
            if im is None:
                continue
            pats.append((im[:, :, ::-1].astype(np.float32) / 255.0))
            prov.append(f)
        return cls(pats, prov)

    def __len__(self) -> int:
        return len(self.patches)

    def sample(self, shape, rng):
        """Random crop (with reflection padding if the patch is small)."""
        if not self.patches:
            return None
        h, w = shape
        i = int(rng.integers(len(self.patches)))
        p = self.patches[i]
        ph, pw = p.shape[:2]
        if ph < h or pw < w:
            p = np.pad(p, ((0, max(0, h - ph)), (0, max(0, w - pw)), (0, 0)),
                       mode="reflect")
            ph, pw = p.shape[:2]
        y = int(rng.integers(0, ph - h + 1))
        x = int(rng.integers(0, pw - w + 1))
        out = p[y:y + h, x:x + w]
        if rng.random() < 0.5:
            out = out[:, ::-1]
        if rng.random() < 0.5:
            out = out[::-1, :]
        return np.ascontiguousarray(out)


def match_local_photometry(root_rgb, bg, alpha, rng, strength: float = 1.0):
    """Nudge root colour toward the local background statistics.

    Naive alpha compositing of a rendered root onto a real soil patch leaves a
    seam with mismatched brightness and colour temperature, and the network
    learns the seam rather than the root -- scoring beautifully on synthetic
    validation and collapsing on real data.

    Deliberately conservative: it shifts the MEAN only.  Dividing by the
    within-root standard deviation (which is ~0 for a thin uniform root) hits the
    clamp and multiplies root/soil contrast back up, undoing any width-dependent
    contrast modelling applied earlier.
    """
    lit = alpha > 0.01
    if not lit.any():
        return root_rgb
    bg_mean = bg[lit].mean(axis=0)
    rt_mean = root_rgb[lit].mean(axis=0) if root_rgb[lit].size else bg_mean
    # keep the root darker than soil by its own contrast, but re-centre the hue
    shift = strength * 0.5 * (bg_mean - rt_mean)
    out = root_rgb + shift[None, None, :]
    return np.clip(out, 0.0, 1.0)
