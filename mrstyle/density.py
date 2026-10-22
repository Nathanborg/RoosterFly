"""Match the amount of root per synthetic frame to the real tiles.

Run 1's synthetic frames carried ~12x more root than the real CI-600 tiles
(median 646 vs 55 mm per 54 x 40 mm frame).  That mismatch was the other driver
of faded roots: 99% of the discriminator's 7.7 mm patches contained root on the
synthetic side versus 23% on the real side, so the adversarial gradient pushed
CUT to remove roots.  Three causes, all in the stand, none in the renderer: 8
plants 7 cm apart, a plant row 0.6 cm from the tube wall on both sides, and
frames biased toward the bottom of the tube while a CI-600 scan covers the full
circumference.

Two steps, both here:

1. **Calibrate** the stand before the full render.  Growing a stand is ~95% of
   the cost, and most levers need no new stand:

   * ``days``: growth is causal and every node carries the day it was laid, so
     the stand at day d < D is the D-day stand with every later segment removed.
   * ``n_plants``: ``grow_stand`` grows plants one after another into one list,
     and the middle k plants of an N-plant stand sit where ``grow_stand`` would
     put a k-plant stand (for (N-k)/2 odd the row is mirrored across the tube,
     which is statistically the same).
   * framing: ``below_tube_bias`` and ``axial_range_cm`` only change FrameSpecs.

   So one stand per (row offset, spacing, replicate) scores a whole grid, with
   common random numbers: frame positions, soil-cover crops and colour draws are
   identical across (n_plants, days), so differences between configs are not
   sampling noise.  Per frame only the analytic visible length is computed;
   ``render_frame`` bisects its mask threshold until the mask's skeleton length
   equals exactly this number, so it IS ``visible_length_mm`` up to bisection
   error.  :func:`validate_config` re-measures the winner with real masks.

2. **Histogram-match** the rendered set: :func:`select_matched` keeps the largest
   subset whose length histogram equals the real one.

Objective (smaller is better)::

    J = W1(log10 L_syn | L_syn >= 5 mm,  log10 L_real)  [decades]
        + lam * |P_syn(L < 5) - P_real(L < 5)|           (lam = 0.5)

0.1 decade means frames typically 26% too long or short.  Its sampling noise is
~0.07-0.11 decades at 40-120 frames, so configs closer than that cannot be told
apart; :func:`bootstrap_scores` reports the spread instead of pretending.
"""
from __future__ import annotations

import itertools
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MIN_MM = 5.0


# --------------------------------------------------------------------------
# the real target
# --------------------------------------------------------------------------
def _tiles(tiles_csv, datasets, subsets):
    import pandas as pd
    r = pd.read_csv(tiles_csv)
    r["dropped"] = r["dropped"].fillna("")
    return r[r.dataset.isin(list(datasets)) & r.subset.isin(list(subsets))]


def real_target(tiles_csv: str, datasets=("B", "C"),
                subsets=("standingcrop",)) -> np.ndarray:
    """Traced root length (mm) of the KEPT real tiles.

    Standing crop is the only subset whose traces mark ALL roots; pair traces
    mark new growth only, a lower bound.  Dataset A is excluded by default: its
    px/mm is estimated.  Tiles of scans whose trace misses the roots (most of
    dataset C) carry ``dropped='trace_mismatch'`` and fall out here.
    """
    r = _tiles(tiles_csv, datasets, subsets)
    v = r.loc[r.dropped == "", "trace_mm"].to_numpy(float)
    return v[v >= MIN_MM]


def real_lt5_fraction(tiles_csv: str, datasets=("B", "C"),
                      subsets=("standingcrop",)) -> float:
    """P(L < 5 mm) among USABLE tiles (kept + too_little_root): the near-empty
    fraction, which the kept set cannot show because it was cut at 5 mm."""
    r = _tiles(tiles_csv, datasets, subsets)
    use = r.loc[r.dropped.isin(["", "too_little_root"]), "trace_mm"].to_numpy(float)
    if use.size == 0:
        raise ValueError("no usable real tiles for %s / %s" % (datasets, subsets))
    return float(np.mean(use < MIN_MM))


def objective(syn_mm, real_mm, real_lt5: float, lam: float = 0.5) -> Dict[str, float]:
    from scipy.stats import ks_2samp, wasserstein_distance
    syn_mm = np.asarray(syn_mm, float)
    ok = syn_mm >= MIN_MM
    lt5 = float(1.0 - ok.mean()) if len(syn_mm) else 1.0
    if ok.sum() < 10:
        return dict(J=float("inf"), W1_log10=float("inf"), lt5=lt5,
                    med_ratio=float("nan"), p90_ratio=float("nan"), KS_D=float("nan"),
                    yield_ge5=float(ok.mean()) if len(syn_mm) else 0.0)
    real_mm = np.asarray(real_mm, float)
    w1 = float(wasserstein_distance(np.log10(syn_mm[ok]), np.log10(real_mm)))
    return dict(J=w1 + lam * abs(lt5 - real_lt5), W1_log10=w1, lt5=lt5,
                med_ratio=float(np.median(syn_mm[ok]) / np.median(real_mm)),
                p90_ratio=float(np.quantile(syn_mm[ok], 0.9) / np.quantile(real_mm, 0.9)),
                KS_D=float(ks_2samp(syn_mm[ok], real_mm).statistic),
                yield_ge5=float(ok.mean()))


# --------------------------------------------------------------------------
# one stand -> lengths for every (n_plants, days, framing)
# --------------------------------------------------------------------------
@dataclass
class StandSpec:
    """What needs its own growth.  n_plants_max / days_max bound the emulated grid.

    The v3 fields (trailing; defaults = the v2 stand exactly) make the
    calibration see the stands the v3 render draws: ``diameter`` (a
    :class:`mrsim.diameter.DiameterSpec` dict -- thicker roots stay visible
    deeper, so length per frame rises ~20% at the calibrated widths),
    ``tortuosity_mult``, relevance culling (``cull_margin_cm``), bridged
    visibility (``vis_bridge_mm``) and the geometric mask (``mask_mode``,
    used by ``mode='mask'``).
    """
    species: str = "generic_fibrous"
    spacing_cm: float = 7.0
    row_offset_cm: float = 0.6            # plant row distance from the tube WALL
    n_plants_max: int = 8
    days_max: float = 60.0
    replicate: int = 0
    max_roots_per_plant: int = 8000
    preset: str = "cid_ci600_640"
    seed: int = 777001
    # --- v3 ---
    diameter: Optional[dict] = None
    tortuosity_mult: float = 1.0
    cull_margin_cm: Optional[float] = None
    vis_bridge_mm: float = 0.0
    mask_mode: str = "calibrated"
    mask_min_width_px: float = 1.0

    def is_v3(self) -> bool:
        return (self.diameter is not None or self.tortuosity_mult != 1.0
                or self.cull_margin_cm is not None or self.vis_bridge_mm > 0
                or self.mask_mode != "calibrated")


@dataclass
class FramingSpec:
    below_tube_bias: float = 0.0          # CI-600 scans cover the full circumference
    axial_range_cm: Tuple[float, float] = (0.0, 60.0)
    soil_cover: float = 0.14


def _plant_blocks(system, n_plants: int) -> np.ndarray:
    """Plant index of every root.  grow_stand appends each plant's roots as one
    contiguous block starting with that plant's primary root (no parent, born
    on day 0; basal roots are born later)."""
    starts = [i for i, r in enumerate(system.roots)
              if r.parent is None and float(r.birth_day) == 0.0]
    if len(starts) != n_plants:
        raise RuntimeError("found %d plant blocks, expected %d" % (len(starts), n_plants))
    return np.searchsorted(np.asarray(starts), np.arange(len(system.roots)), side="right") - 1


def _subset_projection(proj, keep: np.ndarray, age_shift: float):
    """Kept for callers of the old name: now :func:`mrsim.scene.subset_projection`,
    which slices EVERY per-segment field (a hand-written copy here would have
    silently dropped ``vis_depth_mm`` and any field added after it)."""
    from mrsim.scene import subset_projection
    return subset_projection(proj, keep, age_shift)


def frame_area_cm2(preset: str = "cid_ci600_640") -> float:
    """Area of one synthetic frame of ``preset`` [cm^2] (22.0 for the CI-600 tile)."""
    from mrsim.tube import get_preset
    _, cam = get_preset(preset)
    return float((cam.width_px / cam.px_per_mm_u) * (cam.height_px / cam.px_per_mm_v) / 100.0)


def _width_cols(mask: np.ndarray, skel: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Per-frame width columns of a mask: h00..h10 (WIDTH_EDGES counts of
    w2e = 2 x EDT at the skeleton), n_skel, n_thin (w2e <= THIN_W2E) and
    root_px_frac -- what mrstyle.widths pools into histograms."""
    from mrsim.traits import THIN_W2E, WIDTH_EDGES, skeleton_widths_px
    from mrstyle.realdata import hist_cols, width_hist
    w = skeleton_widths_px(mask, skel=skel)
    out: Dict[str, float] = {c: int(v) for c, v in zip(hist_cols(WIDTH_EDGES),
                                                       width_hist(w, WIDTH_EDGES))}
    out["n_skel"] = int(w.size)
    out["n_thin"] = int(np.count_nonzero(w <= THIN_W2E))
    out["root_px_frac"] = float(np.mean(mask))
    return out


def _middle(n_max: int, k: int) -> List[int]:
    o = (n_max - k) // 2
    return list(range(o, o + k))


def stand_lengths(stand: StandSpec, n_plants: Sequence[int] = (2, 4, 6, 8),
                  days: Sequence[float] = (20, 30, 40, 60),
                  framings: Sequence[FramingSpec] = (FramingSpec(),),
                  n_frames: int = 50, mode: str = "analytic"):
    """Grow ONE stand, then measure ``n_frames`` per (n_plants, days, framing).

    ``mode='mask'`` rasterises and skeletonises like render_frame (5-20x slower);
    ``'analytic'`` uses the visible length the mask is calibrated to.  With
    ``stand.mask_mode='geometric'`` the mask is the v3 geometric one
    (label everything drawn) and each row also carries the width columns of
    :func:`_width_cols` (h00..h10, n_skel, n_thin, root_px_frac).

    The stand is grown through :func:`mrsim.scene.grow_and_project` -- the
    render's own path -- and, with ``stand.diameter``, its widths come from
    :func:`mrsim.diameter.apply_diameter` with a stream of its own, so the
    geometry and every frame draw are the same for every diameter spec.
    """
    import pandas as pd
    from mrsim import raster as R
    from mrsim import soil as S
    from mrsim.params import get_species, scale_tortuosity
    from mrsim.scene import (RootAppearance, _segments_for_frame, grow_and_project,
                             sample_frame_specs)
    from mrsim.traits import skeleton_length_px, soil_depth_cm
    from mrsim.tube import VisibilityModel, get_preset

    n_plants = [k for k in n_plants if k <= stand.n_plants_max
                and (stand.n_plants_max - k) % 2 == 0]
    days = [d for d in days if d <= stand.days_max]
    if not n_plants or not days:
        raise ValueError("no (n_plants, days) combination fits this stand")
    if stand.mask_mode not in ("calibrated", "geometric"):
        raise ValueError("mask_mode must be 'calibrated' or 'geometric'")
    v3 = stand.is_v3()
    geometric = stand.mask_mode == "geometric"
    t0 = time.time()
    tube, cam = get_preset(stand.preset)
    h, w = cam.height_px, cam.width_px
    s_iso = math.sqrt(cam.px_per_mm_u * cam.px_per_mm_v)
    rng = np.random.default_rng(stand.seed + 7919 * stand.replicate
                                + 104729 * int(round(10 * stand.row_offset_cm))
                                + 1299709 * int(round(10 * stand.spacing_cm)))
    plant = get_species(stand.species)
    if stand.tortuosity_mult != 1.0:
        plant = scale_tortuosity(plant, stand.tortuosity_mult)
    axial = None
    if stand.cull_margin_cm is not None:
        axial = (min(float(f.axial_range_cm[0]) for f in framings),
                 max(float(f.axial_range_cm[1]) for f in framings))
    system, frame, proj = grow_and_project(
        plant, tube, cam, stand.days_max, rng, n_plants=stand.n_plants_max,
        spacing_cm=stand.spacing_cm, row_offset_cm=tube.r_out + stand.row_offset_cm,
        max_roots_per_plant=stand.max_roots_per_plant,
        cull_margin_cm=stand.cull_margin_cm, axial_range_cm=axial,
        vis_bridge_mm=stand.vis_bridge_mm)
    t_grow = time.time() - t0
    plant_of_root = _plant_blocks(system, stand.n_plants_max)
    roots_per_plant = np.bincount(plant_of_root, minlength=stand.n_plants_max)
    if stand.diameter is not None:
        from mrsim.diameter import DiameterSpec, apply_diameter, root_z, system_multipliers
        dspec = DiameterSpec.from_any(stand.diameter)
        drng = np.random.default_rng([int(stand.seed), int(stand.replicate),
                                      int(round(10 * stand.row_offset_cm)),
                                      int(round(10 * stand.spacing_cm)), 2])
        m_axis, m_lat = system_multipliers(dspec, drng)
        proj = apply_diameter(proj, dspec, m_axis, m_lat, root_z(len(system.roots), drng),
                              rng_along=drng)
    seg_plant = plant_of_root[proj.root_id]
    seg_day = stand.days_max - proj.age_days          # day each segment was laid
    vis = VisibilityModel()

    rows = []
    for fi, fr in enumerate(framings):
        specs = sample_frame_specs(n_frames, cam, tube, rng,
                                   below_tube_bias=fr.below_tube_bias,
                                   axial_range_cm=tuple(fr.axial_range_cm))
        occls, appears, seeds = [], [], []
        tex = None
        for j in range(n_frames):
            if j % 25 == 0:                           # same batching as _render_system
                style = S.SoilStyle.sample(rng, cam.px_per_mm_u)
                tex = S.TextureCache.build((h, w), rng, style, cover=fr.soil_cover)
            occls.append(tex.crop((h, w), rng)[1])
            appears.append(RootAppearance.sample(rng))
            seeds.append(int(rng.integers(2 ** 31)))
        for k in n_plants:
            pk = np.isin(seg_plant, _middle(stand.n_plants_max, k))
            for d in days:
                keep = pk & (seg_day <= d + 1e-6)
                sub = _subset_projection(proj, keep, stand.days_max - d)
                for j, spec in enumerate(specs):
                    frng = np.random.default_rng(seeds[j])      # same per-root draws
                    segs, _, len2d = _segments_for_frame(sub, cam, spec, tube, vis,
                                                         appears[j], frng, occl=occls[j])
                    L = float(len2d)
                    extra = None
                    if mode == "mask" and geometric:
                        from skimage.morphology import skeletonize
                        ras = R.rasterize(segs, w, h,
                                          cover_min_radius_px=0.5 * stand.mask_min_width_px)
                        mk = R.geometric_mask(ras.cover, occls[j])
                        sk = skeletonize(mk) if mk.any() else mk
                        L = skeleton_length_px(mk, skel=sk) / s_iso
                        extra = _width_cols(mk, sk)
                    elif mode == "mask":
                        ras = R.rasterize(segs, w, h)
                        alpha = ras.alpha * occls[j]
                        thr = R.calibrate_mask_threshold(alpha, len2d * cam.px_per_mm_u)
                        L = skeleton_length_px(R.mask_from_alpha(alpha, thr)) / s_iso
                    row = dict(
                        species=stand.species, spacing_cm=stand.spacing_cm,
                        row_offset_cm=stand.row_offset_cm, replicate=stand.replicate,
                        n_plants=k, days=float(d), below_tube_bias=fr.below_tube_bias,
                        axial_lo=float(fr.axial_range_cm[0]),
                        axial_hi=float(fr.axial_range_cm[1]),
                        soil_cover=fr.soil_cover, max_roots_per_plant=stand.max_roots_per_plant,
                        frame=j, L=L, len2d=float(len2d),
                        soil_depth_cm=soil_depth_cm(spec.axial_cm, tube.angle_from_vertical_deg))
                    if extra is not None:
                        row.update(extra)
                    rows.append(row)
    out = pd.DataFrame(rows)
    out["t_grow_s"] = t_grow
    out["t_total_s"] = time.time() - t0
    out["roots_per_plant"] = float(roots_per_plant.mean())
    # if the cap binds, later days add no laterals and deep axes stay bare
    out["cap_hit_frac"] = float(np.mean(roots_per_plant >= stand.max_roots_per_plant))
    if v3:
        out["n_culled"] = int(system.n_culled)
        out["mask_mode"] = stand.mask_mode
        if mode == "mask":
            with np.errstate(divide="ignore", invalid="ignore"):
                out["mask_len_ratio"] = np.where(out.len2d > 0, out.L / out.len2d, np.nan)
    return out


def _job(args):
    stand_d, kw = args
    kw = dict(kw)
    if "framings" in kw:
        kw["framings"] = [FramingSpec(**f) for f in kw["framings"]]
    return stand_lengths(StandSpec(**stand_d), **kw)


def default_stands(row_offsets=(0.6, 3.0, 6.0, 10.0), spacings=(7.0,),
                   replicates: int = 2, species: str = "generic_fibrous",
                   max_roots_per_plant: int = 8000,
                   extra: Optional[dict] = None) -> List[StandSpec]:
    """8 stands: 4 row offsets x 2 replicates, each 8 plants x 60 days -- run 1's
    stand at 0.6 cm plus three sparser placements.  Emulated inside each:
    n_plants 2/4/6/8, days 20-60.  ``extra``: further StandSpec fields for
    every stand (v3: diameter, tortuosity_mult, cull_margin_cm, ...)."""
    return [StandSpec(species=species, spacing_cm=s, row_offset_cm=r, replicate=i,
                      max_roots_per_plant=max_roots_per_plant, **(extra or {}))
            for r, s, i in itertools.product(row_offsets, spacings, range(replicates))]


def spawn_pool(workers: int):
    """A ProcessPoolExecutor whose workers are SPAWNED, never forked.

    One long v3 notebook initialises torch CUDA (DINO) and the cv2 thread pool
    in the kernel, and after a Colab disconnect the user re-runs from the top:
    forking a process from that state can deadlock (Python 3.13 warns about
    it).  A spawned worker starts a clean interpreter, inherits ``sys.path``
    and imports only what the job needs.  Results do not depend on the start
    method -- every job is a pure function of its arguments.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    return ProcessPoolExecutor(max_workers=workers,
                               mp_context=multiprocessing.get_context("spawn"))


def run_stands(stands: Sequence[StandSpec], workers: Optional[int] = None,
               out_csv: Optional[str] = None, verbose: bool = True, **kw):
    """Grow every stand in its own process (growth is pure Python, so threads
    would serialise).  Workers are spawned (:func:`spawn_pool`), so this is
    safe even after TensorFlow/CUDA was initialised in the kernel."""
    import pandas as pd
    from concurrent.futures import as_completed
    if "framings" in kw:
        kw["framings"] = [asdict(f) for f in kw["framings"]]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    parts = []
    with spawn_pool(workers) as ex:
        futs = [ex.submit(_job, (asdict(s), kw)) for s in stands]
        for n, f in enumerate(as_completed(futs), 1):
            p = f.result()
            parts.append(p)
            if out_csv:
                pd.concat(parts).to_csv(out_csv, index=False)
            if verbose:
                print("  stand %d/%d done (grow %.0f s, total %.0f s, cap hit %.0f%%) "
                      "| wall %.1f min" % (n, len(stands), p.t_grow_s.iloc[0],
                                           p.t_total_s.iloc[0], 100 * p.cap_hit_frac.iloc[0],
                                           (time.time() - t0) / 60), flush=True)
    return pd.concat(parts, ignore_index=True)


CFG_KEYS = ["species", "spacing_cm", "row_offset_cm", "n_plants", "days",
            "below_tube_bias", "axial_lo", "axial_hi", "soil_cover", "max_roots_per_plant"]


def score_grid(runs, real_mm, real_lt5: float, lam: float = 0.5):
    """One row per config (pooled over replicates), best first."""
    import pandas as pd
    rows = []
    for key, g in runs.groupby(CFG_KEYS):
        o = objective(g.L.to_numpy(), real_mm, real_lt5, lam)
        rep_med = g.groupby("replicate").L.median()
        rows.append(dict(zip(CFG_KEYS, key), n_frames=len(g), n_stands=g.replicate.nunique(),
                         median_mm=float(g.L.median()), p90_mm=float(g.L.quantile(0.9)),
                         rep_median_spread=float(rep_med.max() / max(rep_med.min(), 1e-6)),
                         cap_hit_frac=float(g.cap_hit_frac.max()), **o))
    return pd.DataFrame(rows).sort_values("J").reset_index(drop=True)


def bootstrap_scores(runs, res, real_mm, real_lt5: float, top: int = 5,
                     n_boot: int = 200, lam: float = 0.5, seed: int = 0):
    """Bootstrap spread of J for the ``top`` configs (frames AND real tiles resampled).

    Returns ``res.head(top)`` with J_lo / J_hi (5-95%) and ``distinct_from_best``:
    whether the config's J interval clears the best one's.  If the top configs
    overlap, pick the simplest (fewest plants / earliest day) -- the grid cannot
    tell them apart, and a slightly dense config keeps the real tail covered for
    the final histogram match.
    """
    rng = np.random.default_rng(seed)
    real_mm = np.asarray(real_mm, float)
    out = res.head(top).copy()
    lo, hi = [], []
    for _, c in out.iterrows():
        m = np.ones(len(runs), bool)
        for k in CFG_KEYS:
            m &= (runs[k] == c[k]).to_numpy()
        reps = [grp.L.to_numpy(float) for _, grp in runs.loc[m].groupby("replicate")]
        js = []
        for _ in range(n_boot):
            # stands first, then frames within them: frames of one stand are not
            # independent, and with few stands this is what the spread really is
            pick = [reps[i] for i in rng.integers(len(reps), size=len(reps))]
            L = np.concatenate([rng.choice(x, len(x)) for x in pick])
            js.append(objective(L, rng.choice(real_mm, len(real_mm)), real_lt5, lam)["J"])
        js = np.asarray(js)
        js = js[np.isfinite(js)]
        lo.append(float(np.quantile(js, 0.05)) if js.size else float("inf"))
        hi.append(float(np.quantile(js, 0.95)) if js.size else float("inf"))
    out["J_lo"], out["J_hi"] = lo, hi
    out["distinct_from_best"] = out["J_lo"] > out["J_hi"].iloc[0]
    return out.reset_index(drop=True)


def validate_config(row, n_frames: int = 40, replicate: int = 7,
                    stand_extra: Optional[dict] = None, width_target=None,
                    fine_min: Optional[float] = None, **kw):
    """Re-measure one config with REAL masks (skeleton length), as render_frame does.

    ``stand_extra``: v3 StandSpec fields (diameter, mask_mode='geometric', ...),
    so the masks are the render's.  With a geometric mask the median
    ``mask_len_ratio`` (skeleton / analytic 2D length: thick roots merge at
    crossings and grow spurs) and ``root_px_frac`` are printed, and with
    ``width_target`` (a :mod:`mrstyle.widths` target) the width objective J_w
    of these frames -- the density result changes the lateral share, so the
    width calibration is re-checked at the chosen geometry.
    """
    st = StandSpec(species=row["species"], spacing_cm=float(row["spacing_cm"]),
                   row_offset_cm=float(row["row_offset_cm"]), n_plants_max=int(row["n_plants"]),
                   days_max=float(row["days"]), replicate=replicate,
                   max_roots_per_plant=int(row["max_roots_per_plant"]),
                   **(stand_extra or {}))
    fr = FramingSpec(below_tube_bias=float(row["below_tube_bias"]),
                     axial_range_cm=(float(row["axial_lo"]), float(row["axial_hi"])),
                     soil_cover=float(row["soil_cover"]))
    out = stand_lengths(st, n_plants=(int(row["n_plants"]),), days=(float(row["days"]),),
                        framings=(fr,), n_frames=n_frames, mode="mask", **kw)
    ok = out.len2d > 0
    if ok.any():
        print("mask vs analytic length: median ratio %.3f over %d non-empty frames"
              % (float(np.median(out.L[ok] / out.len2d[ok])), int(ok.sum())))
    if "root_px_frac" in out:
        print("  root pixels per frame: median %.2f%%" % (100 * float(out.root_px_frac.median())))
    if width_target is not None and "h00" in out:
        from mrstyle import widths as W
        o = W.width_objective(W.runs_hist(out), width_target["hist"],
                              fine_min=W.FINE_SHARE_MIN if fine_min is None else fine_min,
                              syn_fine=W.runs_fine(out), tgt_fine=width_target.get("fine"))
        out.attrs["width_objective"] = o
        print("  widths vs target: J_w %.3f (W1 %.3f oct, fine share %.2f vs goal %.2f)"
              % (o["J"], o["W1_log2"], o["fine_syn"], o["fine_goal"]))
    return out


def synth_kwargs(row) -> Dict[str, object]:
    """The chosen config as keyword arguments for ``mrstyle.ginger.SynthConfig``."""
    return dict(species=str(row["species"]), spacing_cm=float(row["spacing_cm"]),
                row_offset_cm=float(row["row_offset_cm"]), n_plants=int(row["n_plants"]),
                days=float(row["days"]), below_tube_bias=float(row["below_tube_bias"]),
                axial_range_cm=(float(row["axial_lo"]), float(row["axial_hi"])),
                soil_cover=float(row["soil_cover"]),
                max_roots_per_plant=int(row["max_roots_per_plant"]))


GEOM_KEYS = ("n_plants", "days", "spacing_cm", "row_offset_cm")


def best_per_species(runs, real_mm=None, real_lt5: float = 0.0, lam: float = 0.5,
                     top: int = 5, n_boot: int = 200, seed: int = 0,
                     real_mm_per_cm2=None, preset: str = "cid_ci600_640",
                     verbose: bool = True) -> Dict[str, Dict[str, object]]:
    """``{species: synth kwargs}``: the best stand geometry of EACH species.

    One geometry for a species mix would give the fibrous grasses and the
    sparse taproots the same plants x days, so one of them would be far off
    the real root amount.  Every species is scored against the SAME real
    target (root amount per area is mostly a property of the site, not of
    the crop), with :func:`score_grid` + :func:`bootstrap_scores` as for v2,
    and the best config is kept.  ``SynthConfig.species_geom`` takes the
    geometry keys (:data:`GEOM_KEYS`) from these.

    The target is LENGTH PER AREA: pass ``real_mm`` as mm per frame-sized tile
    (Ginger tiles are exactly a synthetic frame) or ``real_mm_per_cm2``
    (any source; converted with the area of one ``preset`` frame, which is
    what the objective's 5-mm near-empty cut is defined on).
    """
    if real_mm_per_cm2 is not None:
        real_mm = np.asarray(real_mm_per_cm2, float) * frame_area_cm2(preset)
    if real_mm is None:
        raise ValueError("pass real_mm (mm per frame) or real_mm_per_cm2")
    real_mm = np.asarray(real_mm, float)
    area = frame_area_cm2(preset)
    out: Dict[str, Dict[str, object]] = {}
    for sp, g in runs.groupby("species", sort=True):
        res = score_grid(g, real_mm, real_lt5, lam)
        boot = bootstrap_scores(g, res, real_mm, real_lt5, top=top, n_boot=n_boot,
                                lam=lam, seed=seed)
        best = boot.iloc[0]
        kw = synth_kwargs(best)
        kw.update(J=float(best["J"]), J_lo=float(best["J_lo"]), J_hi=float(best["J_hi"]),
                  median_mm=float(best["median_mm"]),
                  median_mm_per_cm2=float(best["median_mm"]) / area,
                  cap_hit_frac=float(best["cap_hit_frac"]))
        out[str(sp)] = kw
        if verbose:
            print("  %-15s %d plants x %3.0f d, row %4.1f cm | median %5.1f mm "
                  "(%.2f mm/cm2) | J %.3f [%.3f, %.3f]"
                  % (sp, kw["n_plants"], kw["days"], kw["row_offset_cm"], kw["median_mm"],
                     kw["median_mm_per_cm2"], kw["J"], kw["J_lo"], kw["J_hi"]))
    return out


def species_geom(best: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    """``best_per_species`` output -> ``SynthConfig.species_geom`` (geometry only)."""
    return {sp: {k: (int(v[k]) if k == "n_plants" else float(v[k])) for k in GEOM_KEYS}
            for sp, v in best.items()}


# --------------------------------------------------------------------------
# histogram matching of a rendered set
# --------------------------------------------------------------------------
def edges_quantile(target, n: int = 10, lo: float = MIN_MM,
                   top_factor: float = 1.05) -> np.ndarray:
    """Bins holding equal shares of the REAL tiles (deciles for n=10); the top
    edge sits just above the real maximum, so longer synthetic frames are cut."""
    t = np.asarray(target, float)
    q = np.quantile(t, np.linspace(0, 1, n + 1))
    q[0], q[-1] = lo, float(np.max(t)) * top_factor
    return np.unique(q)


def matched_counts(syn_mm, target_mm, edges):
    """``(k_b per bin, N*, limiting bin index, p_b, n_syn_b)``.

    N* = min_b n_syn_b / p_b is the largest subset whose binned histogram
    equals the target's; the bin that sets it is the one to render more of.
    """
    n_syn = np.histogram(np.asarray(syn_mm, float), edges)[0].astype(float)
    n_tgt = np.histogram(np.asarray(target_mm, float), edges)[0].astype(float)
    p = n_tgt / max(n_tgt.sum(), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cap = np.where(p > 0, n_syn / p, np.inf)
    n_star = float(np.min(cap))
    k = np.floor(n_star * p + 1e-9).astype(int)
    return k, n_star, int(np.argmin(cap)), p, n_syn.astype(int)


def select_matched(syn, target_mm, edges, n: Optional[int] = None, seed: int = 0,
                   col: str = "visible_length_mm", balance_systems: bool = True):
    """Largest subset of ``syn`` whose length histogram equals the target's.

    Within a bin, frames are taken round-robin over systems, so one dense stand
    cannot fill a bin on its own.
    """
    L = syn[col].fillna(0.0).to_numpy(float)
    k, n_star, _, p, _ = matched_counts(L, target_mm, edges)
    if n is not None:
        if n > n_star:
            raise ValueError("n=%d exceeds the achievable %d" % (n, int(n_star)))
        k = np.floor(n * p + 1e-9).astype(int)
    rng = np.random.default_rng(seed)
    b = np.digitize(L, edges) - 1
    chosen = []
    for i, kb in enumerate(k):
        if kb <= 0:
            continue
        idx = np.nonzero(b == i)[0]
        if not balance_systems or "system" not in syn.columns:
            chosen.append(rng.choice(idx, kb, replace=False))
            continue
        order = rng.permutation(len(idx))
        sub = syn.iloc[idx].iloc[order]
        rank = sub.groupby("system").cumcount().to_numpy()
        pick = np.lexsort((rng.random(len(sub)), rank))[:kb]
        chosen.append(idx[order][pick])
    sel = np.concatenate(chosen) if chosen else np.zeros(0, int)
    return syn.iloc[np.sort(sel)].copy()
