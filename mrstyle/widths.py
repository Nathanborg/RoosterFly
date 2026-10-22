"""Calibrate synthetic root WIDTHS against real masks, looking at masks only.

v2 frames were 78% 1-2 px roots (w2e at the skeleton p10/50/90 = 2/2/4),
while PRMI Switchgrass masks -- the only real masks at the synthetic 11.81
px/mm -- give about 4/10/17 and Ginger B roots are ~8 px wide (FWHM).  The
diameter multipliers of :class:`mrsim.diameter.DiameterSpec` move the width
distribution; this module picks them.

**Estimator.** ``w2e`` = 2 x exact EDT at the skeleton pixels
(:func:`mrsim.traits.skeleton_widths_px`, the one convention shared with CUT
and notebook 04: a 1-px line is 2, a 3-px bar 4).  One sample per skeleton
pixel, so a histogram of them is a histogram of root LENGTH.  The PRMI target
is measured with the same function (``realdata.prmi_width_table``), so the
comparison is like for like even though neither side gives absolute
diameters (the estimator is biased at crossings, ends and blobs -- on both
sides alike).

**Objective** (smaller is better), on the half-octave ``WIDTH_EDGES`` bins::

    J_w = W1(log2 w_syn, log2 w_tgt) [octaves]
          + lam * |F_syn - max(F_tgt, FINE_SHARE_MIN)|,   F = P(w2e <= THIN_W2E)

W1 between two histograms on half-octave bins is 0.5 x the summed absolute
CDF difference, so a factor 2 in every width is exactly 1 octave.  The fine
share term is a USER DECISION: PRMI masks (superpixels / corrected U-Net,
WinRHIZO boxes) under-label fine roots, and matching them alone would take
the synthetic fine share from ~75% to ~10-15% -- tiny roots would become
scarce exactly when they are the priority.  ``FINE_SHARE_MIN = 0.25`` holds at
least a quarter of synthetic root length at w2e <= 4 (true width <= ~3 px).
The synthetic F uses the exact samples (w2e <= 4 inclusive, like the
``fine_len_frac`` trait); a target known only as a histogram uses w2e < 4
(its first two bins) -- for the PRMI target that lower bound is far below
0.25 anyway, so the goal is FINE_SHARE_MIN either way.

**Grid** (:func:`default_diameter_grid`): axis_mu {1, 1.5, 2, 2.5} x lat_mu
{1, 1.5, 2, 2.5, 3, 4} x root_sd {0.2, 0.4} = 48 specs; fixed sys_sd 0.25,
along_sd 0.1, min 0.08 mm, max 2.5 mm.  Each stand is grown and projected
ONCE (:func:`mrsim.scene.grow_and_project`, the render's own path) and every
spec is rendered from it with the same frames, soil cover, per-root
visibility draws and diameter noise (common random numbers), so differences
between specs are not sampling noise.  Only geometric masks are rasterised
(no colours matter for widths).

**Selection** (:func:`bootstrap_diameter`): stands, then frames within them,
are resampled; among the specs whose J interval overlaps the best one's, the
SIMPLEST (closest to no change) is chosen -- the grid cannot tell them apart,
and a smaller change keeps the presets' own architecture.

Run this BEFORE the density calibration: thicker roots stay visible deeper
(``p_visible`` rises with diameter), so length per frame rises ~20% at the
calibrated widths.  The density step then uses the chosen spec.
"""
from __future__ import annotations

import math
import os
import time
import zlib
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mrsim.traits import THIN_W2E, WIDTH_EDGES, skeleton_widths_px
from mrstyle.realdata import PRMI_REF_SUBSET, hist_cols, width_hist as _width_hist

FINE_SHARE_MIN = 0.25                  # USER DECISION (share of length, w2e <= 4)
PRMI_W2E_QUANTILES = (4.0, 10.0, 17.0)  # measured PRMI Switchgrass p10/50/90 (fallback)
HIST_COLS = hist_cols(WIDTH_EDGES)     # h00..h10
THICK_W2E = 24.0                       # GATE_V3 reports the length share above this

# the sim design's name for the estimator; ONE implementation (mrsim.traits)
width_samples_edt2 = skeleton_widths_px


def width_hist(w) -> np.ndarray:
    """Counts of w2e samples per WIDTH_EDGES bin (h00..h10)."""
    return _width_hist(w, WIDTH_EDGES)


def _upper_edges() -> np.ndarray:
    return np.asarray(WIDTH_EDGES[1:], np.float64)


def hist_fine(hist, fine_edge: float = THIN_W2E) -> float:
    """Share of a histogram in bins lying entirely below ``fine_edge`` (w2e < 4)."""
    h = np.asarray(hist, np.float64)
    tot = h.sum()
    if tot <= 0:
        return float("nan")
    return float(h[_upper_edges() <= fine_edge + 1e-9].sum() / tot)


def hist_quantiles(hist, qs=(0.1, 0.5, 0.9)) -> List[float]:
    """Quantiles of a WIDTH_EDGES histogram, linear in log2 inside a bin (the
    open top bin is treated as one more half-octave)."""
    h = np.asarray(hist, np.float64)
    tot = h.sum()
    if tot <= 0:
        return [float("nan")] * len(qs)
    lo = np.log2(np.asarray(WIDTH_EDGES[:-1], np.float64))
    hi = lo + 0.5
    cdf = np.concatenate(([0.0], np.cumsum(h) / tot))
    out = []
    for q in qs:
        k = int(np.searchsorted(cdf, q, side="left")) - 1
        k = min(max(k, 0), len(h) - 1)
        frac = (q - cdf[k]) / max(cdf[k + 1] - cdf[k], 1e-12)
        out.append(float(2.0 ** (lo[k] + np.clip(frac, 0, 1) * (hi[k] - lo[k]))))
    return out


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------
def _target(hist, fine, source, n, extra=None) -> Dict[str, object]:
    h = np.asarray(hist, np.float64)
    q = hist_quantiles(h)
    t = {"hist": h, "fine": (None if fine is None else float(fine)), "source": source,
         "n": int(n), "p10": q[0], "p50": q[1], "p90": q[2],
         "fine_hist": hist_fine(h)}
    if extra:
        t.update(extra)
    return t


def target_from_masks(widths, subset: str = PRMI_REF_SUBSET,
                      min_images: int = 20) -> Dict[str, object]:
    """Target from the PRMI width table (``realdata.prmi_width_table`` rows,
    a DataFrame or its cached csv): the summed h00..h10 of ``subset``.

    Summing the per-image histograms pools skeleton LENGTH across images
    (a long-root image weighs more), the same weighting as the synthetic
    side.  Raises ValueError with fewer than ``min_images`` images.
    """
    import pandas as pd
    df = pd.read_csv(widths) if isinstance(widths, str) else widths
    d = df[(df["subset"].astype(str) == subset)]
    if "n_skel" in d:
        d = d[pd.to_numeric(d["n_skel"], errors="coerce").fillna(0) > 0]
    if len(d) < min_images:
        raise ValueError("only %d %s masks in the width table (need %d)"
                         % (len(d), subset, min_images))
    h = d[HIST_COLS].apply(pd.to_numeric, errors="coerce").fillna(0).sum().to_numpy()
    extra = {}
    if "w2e_p50" in d:
        extra["img_w2e_p50_median"] = float(pd.to_numeric(d["w2e_p50"], errors="coerce").median())
    if "tube_uid" in d:
        extra["n_tubes"] = int(d["tube_uid"].nunique())
    return _target(h, None, "masks:%s" % subset, len(d), extra)


def target_from_quantiles(p10: float = PRMI_W2E_QUANTILES[0],
                          p50: float = PRMI_W2E_QUANTILES[1],
                          p90: float = PRMI_W2E_QUANTILES[2],
                          n: int = 20001) -> Dict[str, object]:
    """Fallback target from three quantiles (default: the measured PRMI
    Switchgrass 4 / 10 / 17).

    A split lognormal through the three points (log-sd below and above the
    median set separately, so a skewed p10/p90 is honoured), sampled
    deterministically at ``n`` normal quantiles and floored at 2 (the w2e of
    a 1-px line, the smallest a mask can show).  ``samples`` is returned too.
    """
    from scipy.stats import norm
    if not (0 < p10 < p50 < p90):
        raise ValueError("need 0 < p10 < p50 < p90")
    z = norm.ppf((np.arange(n) + 0.5) / n)
    z90 = norm.ppf(0.9)
    s_lo = (math.log(p50) - math.log(p10)) / z90
    s_hi = (math.log(p90) - math.log(p50)) / z90
    w = np.exp(math.log(p50) + z * np.where(z < 0, s_lo, s_hi))
    w = np.maximum(w, 2.0)
    t = _target(width_hist(w), float(np.mean(w <= THIN_W2E)),
                "quantiles:%g/%g/%g" % (p10, p50, p90), n)
    t["samples"] = w
    return t


def target_from_fwhm(fwhm_px, offset: float = 1.0) -> Dict[str, object]:
    """A CHECK target from image FWHM widths (Ginger B ``root_width_px``
    samples, ~7.8 px median): ``w2e ~ fwhm + offset`` (a 1-px root has FWHM
    ~1 and w2e 2).  Annotation-free, but FWHM of a blurred root is not w2e of
    a mask -- use it to cross-check, not to fit."""
    w = np.asarray(fwhm_px, np.float64)
    w = w[np.isfinite(w)] + float(offset)
    return _target(width_hist(w), float(np.mean(w <= THIN_W2E)) if w.size else None,
                   "fwhm+%g" % offset, w.size)


# --------------------------------------------------------------------------
# objective
# --------------------------------------------------------------------------
def width_objective(syn_hist, tgt_hist, fine_min: float = FINE_SHARE_MIN,
                    lam: float = 1.0, syn_fine: Optional[float] = None,
                    tgt_fine: Optional[float] = None,
                    fine_edge: float = THIN_W2E) -> Dict[str, float]:
    """``J_w`` and its parts (see the module docstring).

    ``syn_fine`` / ``tgt_fine``: exact fine shares (w2e <= fine_edge) when
    known; else the histogram's share below ``fine_edge``.  Also returns the
    p10 / median / p90 ratios (synthetic / target).
    """
    ps = np.asarray(syn_hist, np.float64)
    pt = np.asarray(tgt_hist, np.float64)
    nan = float("nan")
    if ps.sum() <= 0 or pt.sum() <= 0:
        return dict(J=float("inf"), W1_log2=float("inf"), fine_syn=nan, fine_tgt=nan,
                    fine_goal=nan, p10_ratio=nan, med_ratio=nan, p90_ratio=nan)
    cs = np.cumsum(ps / ps.sum())[:-1]
    ct = np.cumsum(pt / pt.sum())[:-1]
    w1 = 0.5 * float(np.abs(cs - ct).sum())
    fs = hist_fine(ps, fine_edge) if syn_fine is None else float(syn_fine)
    ft = hist_fine(pt, fine_edge) if tgt_fine is None else float(tgt_fine)
    goal = max(ft, float(fine_min)) if np.isfinite(ft) else float(fine_min)
    qs, qt = hist_quantiles(ps), hist_quantiles(pt)
    return dict(J=w1 + lam * abs(fs - goal), W1_log2=w1, fine_syn=fs, fine_tgt=ft,
                fine_goal=goal, p10_ratio=qs[0] / qt[0], med_ratio=qs[1] / qt[1],
                p90_ratio=qs[2] / qt[2])


def runs_hist(runs) -> np.ndarray:
    """Summed h00..h10 of a runs table (frames pooled by length)."""
    return runs[HIST_COLS].to_numpy(np.float64).sum(axis=0)


def runs_fine(runs) -> float:
    """Exact fine share (w2e <= THIN_W2E) of a runs table."""
    n = float(runs["n_skel"].sum())
    return float(runs["n_thin"].sum()) / n if n > 0 else float("nan")


# --------------------------------------------------------------------------
# one stand -> widths under every diameter spec
# --------------------------------------------------------------------------
@dataclass
class DiamStand:
    """One stand of the diameter grid (the v2 best geometry by default:
    6 plants, 60 days, row 6 cm from the wall)."""
    species: str = "fibrous_v3"
    n_plants: int = 6
    days: float = 60.0
    spacing_cm: float = 7.0
    row_offset_cm: float = 6.0
    replicate: int = 0
    seed: int = 20261013
    tortuosity_mult: float = 1.0
    cull_margin_cm: Optional[float] = 1.0
    vis_bridge_mm: float = 0.0
    max_roots_per_plant: int = 20000
    preset: str = "cid_ci600_640"
    axial_range_cm: Tuple[float, float] = (0.0, 60.0)
    below_tube_bias: float = 0.0
    soil_cover: float = 0.14
    mask_min_width_px: float = 1.0


SPEC_COLS = ("axis_mu", "axis_sys_sd", "lat_mu", "lat_sys_sd", "root_sd", "along_sd",
             "along_corr_mm", "min_mm", "max_mm")


def default_diameter_grid(axis_mu=(1.0, 1.5, 2.0, 2.5),
                          lat_mu=(1.0, 1.5, 2.0, 2.5, 3.0, 4.0),
                          root_sd=(0.2, 0.4), sys_sd: float = 0.25, along_sd: float = 0.1,
                          min_mm: float = 0.08, max_mm: float = 2.5) -> List[dict]:
    """48 DiameterSpec dicts (4 axis x 6 lateral x 2 per-root spreads)."""
    from mrsim.diameter import DiameterSpec
    return [asdict(DiameterSpec(axis_mu=a, axis_sys_sd=sys_sd, lat_mu=l, lat_sys_sd=sys_sd,
                                root_sd=r, along_sd=along_sd, min_mm=min_mm, max_mm=max_mm))
            for a in axis_mu for l in lat_mu for r in root_sd]


def _stand_seed(stand: DiamStand) -> List[int]:
    return [int(stand.seed), int(stand.replicate),
            int(zlib.crc32(stand.species.encode("utf-8")))]


def diameter_runs(stand: DiamStand, specs: Sequence, n_frames: int = 30):
    """Grow + project ``stand`` once; per spec and frame, the geometric mask's
    widths.  One row per (spec_id, frame): species, replicate, spec fields,
    frame, L (skeleton mm), len2d, h00..h10, n_skel, n_thin, root_px_frac,
    plus the stand's grow time, n_roots and n_culled.

    Common random numbers: frame positions, soil-cover crops, appearance and
    the per-root visibility draws come from per-frame seeds fixed before the
    spec loop; the system multipliers use the same two normals for every
    spec, and the per-root and along-root noise the same stream.
    """
    import pandas as pd
    from skimage.morphology import skeletonize
    from mrsim import raster as R
    from mrsim import soil as S
    from mrsim.diameter import DiameterSpec, apply_diameter, root_z, system_multipliers
    from mrsim.params import get_species, scale_tortuosity
    from mrsim.scene import (RootAppearance, _segments_for_frame, grow_and_project,
                             sample_frame_specs)
    from mrsim.traits import skeleton_length_px
    from mrsim.tube import VisibilityModel, get_preset
    from mrstyle.density import _width_cols

    t0 = time.time()
    tube, cam = get_preset(stand.preset)
    h, w = cam.height_px, cam.width_px
    s_iso = math.sqrt(cam.px_per_mm_u * cam.px_per_mm_v)
    rng = np.random.default_rng(_stand_seed(stand))
    plant = get_species(stand.species)
    if stand.tortuosity_mult != 1.0:
        plant = scale_tortuosity(plant, stand.tortuosity_mult)
    system, frame, proj = grow_and_project(
        plant, tube, cam, stand.days, rng, n_plants=stand.n_plants,
        spacing_cm=stand.spacing_cm, row_offset_cm=tube.r_out + stand.row_offset_cm,
        max_roots_per_plant=stand.max_roots_per_plant,
        cull_margin_cm=stand.cull_margin_cm, axial_range_cm=tuple(stand.axial_range_cm),
        vis_bridge_mm=stand.vis_bridge_mm)
    grow_s = time.time() - t0
    fspecs = sample_frame_specs(n_frames, cam, tube, rng, below_tube_bias=stand.below_tube_bias,
                                axial_range_cm=tuple(stand.axial_range_cm))
    occls, appears, seeds = [], [], []
    tex = None
    for j in range(n_frames):
        if j % 25 == 0:                           # same batching as the render
            style = S.SoilStyle.sample(rng, cam.px_per_mm_u)
            tex = S.TextureCache.build((h, w), rng, style, cover=stand.soil_cover)
        occls.append(tex.crop((h, w), rng)[1])
        appears.append(RootAppearance.sample(rng))
        seeds.append(int(rng.integers(2 ** 31)))
    drng = np.random.default_rng(_stand_seed(stand) + [2])
    z_sys = drng.standard_normal(2)
    z_root = root_z(len(system.roots), drng)
    along_seed = int(drng.integers(2 ** 31))
    vis = VisibilityModel()
    rows = []
    for k, sd in enumerate(specs):
        spec = DiameterSpec.from_any(sd)
        m_axis, m_lat = system_multipliers(spec, z=z_sys)
        pj = apply_diameter(proj, spec, m_axis, m_lat, z_root,
                            rng_along=np.random.default_rng(along_seed))
        for j, fs in enumerate(fspecs):
            frng = np.random.default_rng(seeds[j])          # same per-root draws
            segs, _, len2d = _segments_for_frame(pj, cam, fs, tube, vis, appears[j], frng,
                                                 occl=occls[j])
            ras = R.rasterize(segs, w, h, cover_min_radius_px=0.5 * stand.mask_min_width_px)
            mk = R.geometric_mask(ras.cover, occls[j])
            sk = skeletonize(mk) if mk.any() else mk
            row = dict(species=stand.species, replicate=stand.replicate, spec_id=k,
                       frame=j, L=skeleton_length_px(mk, skel=sk) / s_iso,
                       len2d=float(len2d), m_axis=m_axis, m_lat=m_lat)
            for c in SPEC_COLS:
                row[c] = float(getattr(spec, c))
            row.update(_width_cols(mk, sk))
            rows.append(row)
    out = pd.DataFrame(rows)
    out["grow_s"] = grow_s
    out["total_s"] = time.time() - t0
    out["n_roots"] = len(system.roots)
    out["n_culled"] = int(system.n_culled)
    return out


def _grid_job(args):
    stand_d, specs, n_frames = args
    return diameter_runs(DiamStand(**stand_d), specs, n_frames)


def run_diameter_grid(stands: Sequence[DiamStand], specs: Sequence, n_frames: int = 30,
                      workers: Optional[int] = None, out_csv: Optional[str] = None,
                      verbose: bool = True):
    """:func:`diameter_runs` for every stand, one SPAWNED process per stand
    (``density.spawn_pool``: safe after CUDA/threads in the kernel)."""
    import pandas as pd
    from concurrent.futures import as_completed
    from mrstyle.density import spawn_pool
    specs = [asdict(s) if not isinstance(s, dict) else dict(s) for s in specs]
    workers = workers or max(1, min(len(stands), (os.cpu_count() or 2) - 1))
    t0 = time.time()
    parts = []
    with spawn_pool(workers) as ex:
        futs = [ex.submit(_grid_job, (asdict(s), specs, n_frames)) for s in stands]
        for n, f in enumerate(as_completed(futs), 1):
            p = f.result()
            parts.append(p)
            if out_csv:
                pd.concat(parts).to_csv(out_csv, index=False)
            if verbose:
                print("  diameter stand %d/%d (%s) done: grow %.0f s, total %.0f s | wall "
                      "%.1f min" % (n, len(stands), p.species.iloc[0], p.grow_s.iloc[0],
                                    p.total_s.iloc[0], (time.time() - t0) / 60), flush=True)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# scoring and selection
# --------------------------------------------------------------------------
def _species_weights(runs, weights) -> Dict[str, float]:
    present = sorted(runs.species.astype(str).unique())
    if not weights:
        return {s: 1.0 / len(present) for s in present}
    w = {str(k): float(v) for k, v in dict(weights).items() if str(k) in present and v > 0}
    tot = sum(w.values())
    if tot <= 0:
        raise ValueError("no species of the weights is in the runs")
    return {k: v / tot for k, v in w.items()}


def _mix_parts(g, wts, rng=None):
    """Mixture histogram (expected counts of one frame of the mix), fine share
    and root_px_frac of one spec's rows ``g``.  With ``rng``: stands, then
    frames within them, resampled per species."""
    import pandas as pd
    hist = np.zeros(len(HIST_COLS))
    thin = skel = px = 0.0
    for sp, wt in wts.items():
        s = g[g.species.astype(str) == sp]
        if s.empty:
            continue
        if rng is not None:
            reps = [x for _, x in s.groupby("replicate")]
            pick_r = [reps[i] for i in rng.integers(len(reps), size=len(reps))]
            s = pd.concat([x.iloc[rng.integers(len(x), size=len(x))] for x in pick_r])
        hist += wt * s[HIST_COLS].to_numpy(np.float64).mean(axis=0)
        thin += wt * float(s.n_thin.mean())
        skel += wt * float(s.n_skel.mean())
        px += wt * float(s.root_px_frac.mean())
    return hist, (thin / skel if skel > 0 else float("nan")), px


def score_diameter(runs, target, weights=None, fine_min: float = FINE_SHARE_MIN,
                   lam: float = 1.0):
    """One row per spec (best J first): the objective of the species MIXTURE.

    The mixture histogram is ``sum_s w_s * (mean counts per frame of species
    s)`` -- the expected width histogram of one frame drawn from the mix, so
    a species with more root per frame weighs more, as it will in the
    render.  Also reported: root_px_frac (PRMI 1.3-2.2%, Ginger ~2.3%; not a
    target) and the median skeleton length per frame.
    """
    import pandas as pd
    wts = _species_weights(runs, weights)
    tgt = np.asarray(target["hist"], np.float64)
    tf = target.get("fine")
    rows = []
    for k, g in runs.groupby("spec_id", sort=True):
        hist, fine, px = _mix_parts(g, wts)
        o = width_objective(hist, tgt, fine_min=fine_min, lam=lam, syn_fine=fine, tgt_fine=tf)
        q = hist_quantiles(hist)
        row = {"spec_id": int(k)}
        for c in SPEC_COLS:
            row[c] = float(g[c].iloc[0])
        row.update(o)
        row.update(w2e_p10=q[0], w2e_p50=q[1], w2e_p90=q[2], root_px_frac=px,
                   L_median=float(g.L.median()), n_frames=len(g))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["J", "spec_id"]).reset_index(drop=True)


def spec_complexity(row) -> float:
    """Distance from 'no change': |log axis_mu| + |log lat_mu| + root_sd."""
    return (abs(math.log(float(row["axis_mu"]))) + abs(math.log(float(row["lat_mu"])))
            + float(row["root_sd"]))


def bootstrap_diameter(runs, res, target, weights=None, top: int = 5, n_boot: int = 200,
                       seed: int = 0, fine_min: float = FINE_SHARE_MIN, lam: float = 1.0):
    """Bootstrap J (5-95%) of the ``top`` specs; stands, then frames resampled.

    Returns ``res.head(top)`` with J_lo, J_hi, ``distinct_from_best`` (its
    interval clears the best one's), ``complexity`` and ``chosen``: the
    simplest spec NOT distinct from the best.
    """
    rng = np.random.default_rng(seed)
    wts = _species_weights(runs, weights)
    tgt = np.asarray(target["hist"], np.float64)
    tf = target.get("fine")
    out = res.head(top).copy()
    lo, hi = [], []
    for _, r in out.iterrows():
        g = runs[runs.spec_id == r["spec_id"]]
        js = []
        for _ in range(n_boot):
            hist, fine, _ = _mix_parts(g, wts, rng=rng)
            js.append(width_objective(hist, tgt, fine_min=fine_min, lam=lam,
                                      syn_fine=fine, tgt_fine=tf)["J"])
        js = np.asarray(js)
        js = js[np.isfinite(js)]
        lo.append(float(np.quantile(js, 0.05)) if js.size else float("inf"))
        hi.append(float(np.quantile(js, 0.95)) if js.size else float("inf"))
    out["J_lo"], out["J_hi"] = lo, hi
    out["distinct_from_best"] = out["J_lo"] > out["J_hi"].iloc[0]
    out["complexity"] = [spec_complexity(r) for _, r in out.iterrows()]
    cand = out[~out.distinct_from_best]
    best_i = cand.sort_values(["complexity", "J"]).index[0]
    out["chosen"] = out.index == best_i
    return out.reset_index(drop=True)


def pick_diameter(boot) -> Dict[str, float]:
    """The chosen spec of :func:`bootstrap_diameter` as a DiameterSpec dict."""
    r = boot[boot.chosen].iloc[0]
    return {c: float(r[c]) for c in SPEC_COLS}


# --------------------------------------------------------------------------
# image-side check: FWHM of the rendered (pre-CUT) roots vs Ginger B
# --------------------------------------------------------------------------
def skeleton_samples(mask: np.ndarray, step_px: float = 15.0, max_samples: int = 200,
                     seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Centreline points ``(x, y)`` and unit normals from a mask's skeleton
    (local PCA; junctions and blobs skipped) -- realdata's sampler, the one
    the PRMI ``fwhm_f`` check uses, so both sides sample alike."""
    from mrstyle.realdata import _mask_samples
    return _mask_samples(mask, step_px=step_px, max_samples=max_samples, seed=seed)


def synthetic_fwhm(syn_dir: str, ids: Sequence[str], workers: Optional[int] = None,
                   max_samples: int = 200, verbose: bool = True):
    """Per frame: median FWHM of the rendered RGB across the mask skeleton
    (``ginger._fwhm_widths``, the estimator of Ginger B's 7.8 px).  Columns
    id, fwhm_p50, fwhm_n.  Threads (cv2/numpy release the GIL)."""
    import cv2
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    from mrstyle.ginger import _fwhm_widths

    def one(sid):
        m = cv2.imread(os.path.join(syn_dir, "mask", sid + ".png"), cv2.IMREAD_GRAYSCALE)
        im = cv2.imread(os.path.join(syn_dir, "rgb", sid + ".png"), cv2.IMREAD_COLOR)
        if m is None or im is None:
            return {"id": sid, "fwhm_p50": float("nan"), "fwhm_n": 0}
        pts, nrm = skeleton_samples(m > 127, max_samples=max_samples)
        fw = _fwhm_widths(im[:, :, ::-1], pts, nrm)
        fw = fw[np.isfinite(fw)]
        return {"id": sid, "fwhm_p50": float(np.median(fw)) if fw.size else float("nan"),
                "fwhm_n": int(fw.size)}

    with ThreadPoolExecutor(max_workers=workers or max(1, os.cpu_count() or 2)) as ex:
        rows = list(ex.map(one, list(ids)))
    df = pd.DataFrame(rows)
    if verbose and len(df):
        print("synthetic FWHM: median %.1f px over %d frames"
              % (float(df.fwhm_p50.median()), int(df.fwhm_p50.notna().sum())))
    return df
