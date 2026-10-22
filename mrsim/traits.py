"""Root trait computation, in two strictly separated namespaces.

Three measurement domains must not be mixed:

* **A** the true 3D architecture of the plant,
* **B** the 2D projection intercepted by the tube wall,
* **C** what is recoverable from ONE frame -- fragments inside a ~2-6 cm window
  with a sub-millimetre depth of field.

Most published traits are *defined* in A, *estimated* in B, and simply are not
identifiable in C.  Of the nine traits commonly asked for, only four are
well-posed per-frame regression targets:

===============================  ==========================================
well-posed                       visible length; length-weighted mean
                                 diameter; length distribution by diameter
                                 class; branching density (per unit length)
marginal (emitted, flagged)      branch-point count (inflated by non-connected
                                 crossings, deflated by occlusion, and
                                 frame-size dependent); tip count (endpoints are
                                 dominated by FOV exits and burials, not apices)
NOT identifiable                 root order (needs a tree traced to a base);
                                 branching angle (needs parent/daughter identity,
                                 and it is a 3D angle on a curved wall); Fitter's
                                 indices (defined only on a complete rooted tree)
===============================  ==========================================

:func:`frame_traits` returns only the observable namespace.  Architecture traits
live in :func:`system_traits` and are reference values, never CNN targets.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np

try:
    from skimage.morphology import skeletonize
    from scipy import ndimage as ndi
    _HAVE_SKIMAGE = True
except Exception:                                    # pragma: no cover
    _HAVE_SKIMAGE = False


# --------------------------------------------------------------------------
# skeleton length
# --------------------------------------------------------------------------
# 8-neighbour offsets, split by step type
_ORTHO = ((0, 1), (1, 0))
_DIAG = ((1, 1), (1, -1))


def _neighbour_counts(skel: np.ndarray):
    """Count orthogonal and diagonal adjacencies between skeleton pixels."""
    s = skel.astype(bool)
    n_o = 0
    for dy, dx in _ORTHO:
        n_o += int(np.count_nonzero(s[max(0, dy):, max(0, dx):]
                                    & s[:s.shape[0] - dy, :s.shape[1] - dx]))
    n_d = 0
    # (1,1)
    n_d += int(np.count_nonzero(s[1:, 1:] & s[:-1, :-1]))
    # (1,-1)
    n_d += int(np.count_nonzero(s[1:, :-1] & s[:-1, 1:]))
    return n_o, n_d


def skeleton_length_px(mask: np.ndarray, method: str = "step",
                       skel: Optional[np.ndarray] = None) -> float:
    """Total skeleton length in pixels.

    Counting skeleton PIXELS underestimates diagonal runs by up to 1/sqrt(2)
    (~29%) and makes the label orientation-dependent, so a network trained on it
    learns the artifact.  Two accepted corrections:

    ``step``
        Step-weighted summation: orthogonal adjacency = 1, diagonal = sqrt(2).
        This is what skan's ``branch_distance`` and RhizoVision compute.
    ``kimura``
        Kimura et al. (1999):
        ``L = sqrt(Nd^2 + (Nd + No/2)^2) + No/2``.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if not mask.any():
        return 0.0
    if skel is None:
        skel = skeletonize(mask) if _HAVE_SKIMAGE else mask
    n_o, n_d = _neighbour_counts(skel)
    if method == "kimura":
        return float(math.sqrt(n_d ** 2 + (n_d + n_o / 2.0) ** 2) + n_o / 2.0)
    if method == "pixels":
        return float(np.count_nonzero(skel))
    return float(n_o + math.sqrt(2.0) * n_d)


# --------------------------------------------------------------------------
# skeleton topology
# --------------------------------------------------------------------------
_K8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)


def _degree(skel: np.ndarray) -> np.ndarray:
    return ndi.convolve(skel.astype(np.uint8), _K8, mode="constant") * skel


def prune_spurs(skel: np.ndarray, min_len: int = 3) -> np.ndarray:
    """Remove short skeleton spurs, which inflate BOTH tip and branch counts."""
    s = skel.copy().astype(bool)
    for _ in range(int(min_len)):
        deg = _degree(s)
        ends = s & (deg == 1)
        if not ends.any():
            break
        s &= ~ends
    return s


def count_tips_and_branches(mask: np.ndarray, min_spur: int = 3,
                            skel: Optional[np.ndarray] = None):
    """Endpoints and branch points from a mask.

    Branch points are CONNECTED COMPONENTS of the degree>=3 set, not raw pixels:
    a single junction normally yields a 2-3 pixel cluster, which silently inflates
    the count by 2-3x if counted per pixel.

    Both numbers are 'marginal' targets.  In a minirhizotron frame, endpoints are
    dominated by roots leaving the field of view and roots vanishing under soil,
    not by true apices; and two roots at different depths within the depth of
    field overlay into an X that is indistinguishable from a real branch.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if not mask.any():
        return 0, 0
    if skel is None:
        skel = skeletonize(mask)
    skel = prune_spurs(skel, min_spur)
    if not skel.any():
        return 0, 0
    deg = _degree(skel)
    n_tips = int(np.count_nonzero(skel & (deg == 1)))
    branch_px = skel & (deg >= 3)
    if not branch_px.any():
        return n_tips, 0
    _, n_branch = ndi.label(branch_px, structure=np.ones((3, 3), np.uint8))
    return n_tips, int(n_branch)


# --------------------------------------------------------------------------
# diameter
# --------------------------------------------------------------------------
def diameter_map_px(mask: np.ndarray) -> np.ndarray:
    """Local diameter as 2 x the Euclidean distance transform."""
    return 2.0 * ndi.distance_transform_edt(mask)


# --------------------------------------------------------------------------
# the ONE width convention shared by the simulator, CUT and notebook 04
# --------------------------------------------------------------------------
# Three stages once binned "width" with the same edges but three estimators
# (2*EDT, 2*DT-1 from the approximate cv2 transform, true diameter), so a table
# could shift by a whole bin between stages.  Everything that bins a MASK width
# now uses w2e = 2 x exact EDT at skeleton pixels -- the convention of
# diameter_map_px, traits.csv mean_diameter_mm and every number in the v2
# reports.  A 1-px line gives 2, a 3-px bar 4, a 7-px bar 8.  True diameter
# from a renderer's diam/ map is a different quantity (columns prefixed tw_)
# and must never be put in these bins.
#
# Half-octave histogram edges (h00..h10 in the width tables).  The values are
# the agreed contract, rounded on purpose: 2*sqrt(2) = 2.828 falls in the first
# bin, exactly as 2.83 > 2.828 says.
WIDTH_EDGES = (2.0, 2.83, 4.0, 5.66, 8.0, 11.3, 16.0, 22.6, 32.0, 45.3, 64.0,
               float("inf"))
# Report bins [lo, hi): true width about 1-2 / 3-4 / 5-7 / 8-12 / >=13 px.
REPORT_BINS_W2E = (2.0, 3.0, 5.0, 8.0, 13.0, float("inf"))
# 'Thin' everywhere (sim fine share, CUT thin audit, 04 R_thin): w2e <= 4,
# i.e. a true width of about 3 px or less (<= ~0.3 mm at 11.81 px/mm).
THIN_W2E = 4.0


def skeleton_widths_px(mask: np.ndarray, interior_only: bool = False,
                       skel: Optional[np.ndarray] = None,
                       dist: Optional[np.ndarray] = None) -> np.ndarray:
    """Width samples ``w2e = 2 x exact EDT`` at the skeleton pixels of ``mask``.

    One sample per skeleton pixel, so the distribution is length-weighted by
    construction -- what a width histogram of ROOT LENGTH needs; a per-pixel
    histogram of the mask would weight thick roots by their area instead.

    ``interior_only`` drops skeleton pixels within 3 px of an endpoint or a
    junction, where the EDT reads the root's end cap or the union of two roots
    rather than one root's width.  Off by default: masks dominated by short
    fragments (tiny roots) would otherwise lose most of their samples, which is
    exactly the population the width work is about.

    ``skel`` / ``dist`` may be passed when the caller already has them (the
    skeleton and the EDT are the two expensive steps).
    """
    m = np.asarray(mask)
    if m.dtype != bool:
        m = m > 0
    if not m.any():
        return np.zeros(0, np.float64)
    if skel is None:
        skel = skeletonize(m)
    skel = np.asarray(skel, bool)
    if dist is None:
        dist = ndi.distance_transform_edt(m)
    if interior_only and skel.any():
        deg = _degree(skel)
        special = skel & ((deg == 1) | (deg >= 3))
        if special.any():
            near = ndi.binary_dilation(special, structure=np.ones((3, 3), bool),
                                       iterations=3)
            skel = skel & ~near
    return 2.0 * np.asarray(dist, np.float64)[skel]


def length_weighted_mean_diameter(mask: np.ndarray,
                                  diam_px: Optional[np.ndarray] = None,
                                  skel: Optional[np.ndarray] = None) -> float:
    """Length-weighted mean diameter, ``sum(d*l)/sum(l)``.

    An UNWEIGHTED mean of a per-pixel diameter map is a documented error source
    (derived specific root length off by 65-93%).  Note the identity
    ``area = sum(l_p * d_p)``, so this also equals ``A/L`` -- which is exactly
    WinRHIZO's definition, and a useful cross-check.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if not mask.any():
        return float("nan")
    if skel is None:
        skel = skeletonize(mask)
    if not skel.any():
        return float("nan")
    if diam_px is None:
        diam_px = diameter_map_px(mask)
    d = diam_px[skel]
    # each skeleton pixel carries ~1 px of length; step weighting cancels in the
    # ratio to first order, so the simple mean over skeleton pixels is used here
    return float(np.mean(d))


def length_by_diameter_class(mask: np.ndarray, px_per_mm: float,
                             edges_mm: Sequence[float] = (0.0, 0.1, 0.2, 0.35,
                                                          0.5, 1.0, 2.0, 1e9),
                             skel: Optional[np.ndarray] = None,
                             diam_px: Optional[np.ndarray] = None,
                             total_len_mm: Optional[float] = None):
    """Length histogram over diameter classes -- a per-frame trait.

    Not to be confused with length-versus-DEPTH, which is a per-tube trait
    requiring frame depth metadata and an inclination correction.  The two are
    routinely conflated under the name 'root length distribution'.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    out = {"edges_mm": [float(e) for e in edges_mm],
           "length_mm": [0.0] * (len(edges_mm) - 1)}
    if not mask.any():
        return out
    if skel is None:
        skel = skeletonize(mask)
    if not skel.any():
        return out
    if diam_px is None:
        diam_px = diameter_map_px(mask)
    d_mm = diam_px / px_per_mm
    ys, xs = np.nonzero(skel)
    vals = d_mm[ys, xs]
    # total length via the step-weighted estimate, distributed over classes in
    # proportion to skeleton-pixel counts
    if total_len_mm is None:
        total_len_mm = skeleton_length_px(mask, skel=skel) / px_per_mm
    n = len(vals)
    for i in range(len(edges_mm) - 1):
        sel = (vals >= edges_mm[i]) & (vals < edges_mm[i + 1])
        out["length_mm"][i] = float(total_len_mm * (np.count_nonzero(sel) / n))
    return out


# --------------------------------------------------------------------------
# field conversions
# --------------------------------------------------------------------------
def planar_rld(length_mm: float, frame_area_mm2: float) -> float:
    """Planar root length density [mm/mm^2].

    This is what minirhizotron data should be reported as.  Converting to a
    volumetric RLD requires dividing by a depth of field, and that constant is
    contested (0.78 mm empirical vs the 2-3 mm historically assumed -- a ~3x
    difference), so the conversion belongs downstream of the model, never inside
    it.  Asking a CNN to predict volumetric RLD is asking it to predict planar
    RLD times an arbitrary constant.
    """
    if frame_area_mm2 <= 0:
        return float("nan")
    return float(length_mm / frame_area_mm2)


def volumetric_rld(length_mm: float, frame_area_mm2: float,
                   dof_mm: float = 0.78) -> float:
    """Volumetric RLD [mm/mm^3].  State ``dof_mm`` whenever you report this."""
    if frame_area_mm2 <= 0 or dof_mm <= 0:
        return float("nan")
    return float(length_mm / (frame_area_mm2 * dof_mm))


def soil_depth_cm(axial_pos_cm: float, angle_from_vertical_deg: float) -> float:
    """True soil depth from position along an inclined tube."""
    return float(axial_pos_cm * math.cos(math.radians(angle_from_vertical_deg)))


# --------------------------------------------------------------------------
# validation statistics
# --------------------------------------------------------------------------
def box_counting_dimension(mask: np.ndarray, min_box: int = 2,
                           max_box: Optional[int] = None):
    """Box-counting dimension -- a VALIDATION statistic, never a generator.

    On a ~2 cm frame this spans well under two decades of scale, so what it
    measures is a density/tortuosity texture statistic, not the architectural
    fractal dimension.  Real root systems have been shown not to be statistically
    self-similar over the available range despite R^2 > 0.99 on the log-log fit.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if not mask.any():
        return float("nan"), []
    h, w = mask.shape
    if max_box is None:
        max_box = max(min_box * 2, min(h, w) // 4)
    sizes, counts = [], []
    b = min_box
    while b <= max_box:
        hh, ww = h // b, w // b
        if hh < 1 or ww < 1:
            break
        view = mask[:hh * b, :ww * b].reshape(hh, b, ww, b)
        counts.append(int(np.count_nonzero(view.any(axis=(1, 3)))))
        sizes.append(b)
        b *= 2
    if len(sizes) < 3:
        return float("nan"), list(zip(sizes, counts))
    coef = np.polyfit(np.log(1.0 / np.asarray(sizes, float)),
                      np.log(np.asarray(counts, float)), 1)
    return float(coef[0]), list(zip(sizes, counts))


# --------------------------------------------------------------------------
# the two namespaces
# --------------------------------------------------------------------------
# True-width bins [lo, hi) in PIXELS for the tw_len_* columns (v3 traits.csv).
# True width (from the renderer's diameter, tw_px = diam_mm x px/mm) is NOT
# w2e and is never put in WIDTH_EDGES / REPORT_BINS_W2E bins.
TW_BINS_PX = (0.0, 2.0, 3.0, 5.0, 8.0, float("inf"))


def tw_len_labels(bins: Sequence[float] = TW_BINS_PX):
    """``(0, 2, 3, ..., inf)`` -> ['tw_len_0_2', 'tw_len_2_3', ..., 'tw_len_8p']."""
    def f(x):
        return ("%g" % x).replace(".", "p")
    b = [float(x) for x in bins]
    return ["tw_len_%s_%s" % (f(lo), f(hi)) if math.isfinite(hi) else "tw_len_%sp" % f(lo)
            for lo, hi in zip(b[:-1], b[1:])]


def frame_traits(mask: np.ndarray, px_per_mm_u: float, px_per_mm_v: float,
                 frame_area_mm2: float, alpha: Optional[np.ndarray] = None,
                 min_spur: int = 3, widths: bool = False,
                 true_diam_mm: Optional[np.ndarray] = None,
                 tw_bins_px: Sequence[float] = TW_BINS_PX) -> Dict[str, object]:
    """Observable traits -- the honest CNN regression targets.

    ``px_per_mm`` is passed per axis because circumferential and axial scale
    differ on a tube (``R_in/R_out``, plus ``cos(phi)`` for an orthographic
    unwrap).  The geometric mean is used for isotropic quantities, which is an
    approximation that is stated rather than hidden.

    v3 width traits (opt-in, so the default dict is unchanged): with
    ``widths=True`` or a ``true_diam_mm`` map, also

    * ``w2e_p10/p50/p90`` and ``fine_len_frac`` (share of skeleton length with
      w2e <= THIN_W2E) from :func:`skeleton_widths_px` -- the shared estimator,
      fed this function's own skeleton and EDT so nothing is computed twice;
    * ``root_px_frac``: mask pixels / frame pixels;
    * with ``true_diam_mm`` (the renderer's per-pixel diameter, > 0 on the
      mask): ``tw_len_*`` -- visible length (mm) per TRUE-width bin in px
      (:data:`TW_BINS_PX`), distributed like ``length_by_diameter_class`` -- and
      ``mean_true_diameter_mm`` (length-weighted, over skeleton pixels).
    """
    s = math.sqrt(max(px_per_mm_u, 1e-9) * max(px_per_mm_v, 1e-9))
    if mask.dtype != bool:
        mask = mask.astype(bool)

    # skeletonisation and the EDT are the two expensive steps; do each once
    skel = skeletonize(mask) if mask.any() else mask
    dmap = diameter_map_px(mask)
    length_px = skeleton_length_px(mask, skel=skel)
    length_mm = length_px / s
    n_tips, n_branch = count_tips_and_branches(mask, min_spur, skel=skel)
    mean_d_px = length_weighted_mean_diameter(mask, dmap, skel=skel)
    area_px = float(np.count_nonzero(mask))

    out = {
        # --- well-posed ---------------------------------------------------
        "visible_length_mm": float(length_mm),
        "mean_diameter_mm": float(mean_d_px / s) if mean_d_px == mean_d_px else float("nan"),
        "projected_area_mm2": float(area_px / (px_per_mm_u * px_per_mm_v)),
        "planar_rld_mm_per_mm2": planar_rld(length_mm, frame_area_mm2),
        "length_by_diameter_class": length_by_diameter_class(
            mask, s, skel=skel, diam_px=dmap, total_len_mm=length_mm),
        "branching_density_per_mm": (float(n_branch / length_mm)
                                     if length_mm > 1e-9 else float("nan")),
        # --- marginal: emitted, but do not trust them as apex/branch counts --
        "n_endpoints": int(n_tips),
        "n_apparent_branch_points": int(n_branch),
        # --- bookkeeping ---------------------------------------------------
        "px_per_mm_u": float(px_per_mm_u),
        "px_per_mm_v": float(px_per_mm_v),
        "frame_area_mm2": float(frame_area_mm2),
        "is_empty": bool(length_px <= 0),
    }
    if alpha is not None:
        out["alpha_sum"] = float(alpha.sum())
    if widths or true_diam_mm is not None:
        nan = float("nan")
        w = (skeleton_widths_px(mask, skel=skel, dist=0.5 * dmap) if mask.any()
             else np.zeros(0, np.float64))
        q = np.percentile(w, [10, 50, 90]) if w.size else [nan, nan, nan]
        out["w2e_p10"], out["w2e_p50"], out["w2e_p90"] = (float(v) for v in q)
        out["fine_len_frac"] = float(np.mean(w <= THIN_W2E)) if w.size else nan
        out["root_px_frac"] = float(area_px / max(mask.size, 1))
        if true_diam_mm is not None:
            labels = tw_len_labels(tw_bins_px)
            tw = (np.asarray(true_diam_mm, np.float64)[skel] if mask.any()
                  else np.zeros(0, np.float64))
            n = tw.size
            twp = tw * s
            for lab, lo, hi in zip(labels, tw_bins_px[:-1], tw_bins_px[1:]):
                k = int(np.count_nonzero((twp >= lo) & (twp < hi))) if n else 0
                out[lab] = float(length_mm * k / n) if n else 0.0
            pos = tw[tw > 0]
            out["mean_true_diameter_mm"] = float(pos.mean()) if pos.size else nan
    return out


def system_traits(system, px_per_mm: float = 1.0) -> Dict[str, object]:
    """Whole-architecture traits -- REFERENCE VALUES, not regression targets.

    Root order, branching angle and the topological indices live here precisely
    because they are not recoverable from a frame.  Quantifying the gap between
    this namespace and :func:`frame_traits` on the same simulated plant is the
    identifiability analysis; it needs ground truth, which is why it has not been
    done before.
    """
    roots = system.roots
    orders = [r.order for r in roots]
    lengths = [r.length for r in roots]
    total = float(sum(lengths))

    by_order: Dict[int, float] = {}
    for r in roots:
        by_order[r.order] = by_order.get(r.order, 0.0) + r.length

    angles = [float(r.params.theta) for r in roots if r.parent is not None]
    diam = [float(2.0 * r.params.a) for r in roots]

    # Fitter's topological indices, computable here because we have the tree
    children: Dict[int, int] = {}
    for r in roots:
        if r.parent is not None:
            children[r.parent] = children.get(r.parent, 0) + 1
    magnitude = sum(1 for r in roots if children.get(r.index, 0) == 0)

    return {
        "total_length_cm": total,
        "n_roots": len(roots),
        "n_tips_true": magnitude,
        "n_branch_points_true": sum(1 for r in roots if r.parent is not None),
        "max_order": int(max(orders)) if orders else 0,
        "length_by_order_cm": {int(k): float(v) for k, v in sorted(by_order.items())},
        "mean_branching_angle_rad": float(np.mean(angles)) if angles else float("nan"),
        "mean_diameter_cm": float(np.mean(diam)) if diam else float("nan"),
        "magnitude": int(magnitude),
        "_namespace": "architecture (NOT observable from a single frame)",
    }
