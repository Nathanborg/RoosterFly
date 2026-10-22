"""Smooth C2 spline root ribbons, connected-subtree topology, and seam-free tubular rasterization.

Solves three major synthetic tells in legacy minirhizotron rendering:
1. Eliminates piecewise-linear Brownian-bridge kinks via curvature-preserving C2 smoothing
   with locally adaptive curvature bounds (so radius of curvature > root radius).
2. Eliminates 'earthworm / coin-stack' capsule seams by evaluating a single continuous
   (s, t) tubular coordinate field with Phong-interpolated vertex normals per root run,
   rather than compositing dozens of overlapping 2.4-px round-ended capsules.
3. Preserves parent-child root topology when subsampling sparse frames (`connected_topology`),
   eliminating floating orphan lateral stubs, and adds biological branch-collar flaring.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

from mrsim.grow import RootSystem
from mrsim.scene import FrameSpec, ProjectedSystem
from mrsim.tube import MM_PER_CM, CameraParams, TubeParams, VisibilityModel

from .realism_params import RealismConfig


@dataclass
class RootRibbon:
    """A continuous visible root run in image space, parameterized by arc-length s."""

    root_id: int
    order: int
    parent_id: int
    pts: np.ndarray            # (M, 2) float32 [x, y] in px, C2 smoothed
    normals: np.ndarray        # (M, 2) float32 unit normals at vertices
    arc_s: np.ndarray          # (M,) float32 cumulative 2D arc length in px
    s3d_mm: np.ndarray         # (M,) float32 cumulative 3D arc length in mm
    radii_px: np.ndarray       # (M,) float32 radius in px (with collar flare & tip ogive)
    diam_mm: np.ndarray        # (M,) float32 true physical diameter in mm
    depth_mm: np.ndarray       # (M,) float32 depth behind outer tube wall in mm
    vis_weight: np.ndarray     # (M,) float32 smooth visibility/emergence profile in [0, 1]
    age_days: np.ndarray       # (M,) float32 node age in days
    is_apical_tip: bool        # True if the end of this ribbon is the actual root tip
    is_lateral_base: bool      # True if the start of this ribbon attaches to its parent


@dataclass
class RibbonFieldOut:
    """Per-pixel continuous tubular coordinate & attribute maps for all roots in a frame."""

    alpha: np.ndarray          # (H, W) float32 composite root opacity
    cover: np.ndarray          # (H, W) float32 geometric mask coverage (radius floored)
    cover_diam_mm: np.ndarray  # (H, W) float32 diameter on geometric cover
    rgb_base: np.ndarray       # (H, W, 3) float32 unshaded/textured root color in [0, 1]
    shade_map: np.ndarray      # (H, W) float32 3D cylindrical + stele shading factor
    t_coord: np.ndarray        # (H, W) float32 signed cross-axis coordinate in [-1, 1]
    s_coord_mm: np.ndarray     # (H, W) float32 arc-length coordinate in mm
    radius_px: np.ndarray      # (H, W) float32 local root radius in px
    depth_mm: np.ndarray       # (H, W) float32 depth of winning root in mm
    blur_px: np.ndarray        # (H, W) float32 local depth-defocus sigma in px
    instance: np.ndarray       # (H, W) int32 winning root_id (-1 where empty)
    order: np.ndarray          # (H, W) int32 root developmental order
    shadow_field: np.ndarray   # (H, W) float32 contact crevice shadow intensity outside root edge
    ribbons: List[RootRibbon]  # the continuous ribbons rendered in this frame
    len3d_mm: float
    len2d_mm: float


def select_connected_roots(
    system: RootSystem,
    visible_root_ids: np.ndarray,
    visible_lengths_px: np.ndarray,
    keep_frac: float,
    rng: np.random.Generator,
    connected: bool = True,
) -> np.ndarray:
    """Select a subset of roots to keep in the frame while preserving parent-lateral hierarchy.

    When `connected=True`, a lateral root is only kept if its parent (when also visible in the
    frame) is kept, and subsampling selects whole connected root subtrees first, then prunes
    distal highest-order laterals if needed to match `keep_frac`.
    """
    n_roots = max(len(system.roots), int(visible_root_ids.max() + 1) if visible_root_ids.size else 1)
    if keep_frac >= 0.999 or visible_root_ids.size == 0:
        return np.ones(n_roots, dtype=bool)

    if not connected:
        return rng.random(n_roots) < keep_frac

    vis_set = set(int(r) for r in visible_root_ids if visible_lengths_px[int(r)] > 3.0)
    if not vis_set:
        return np.zeros(n_roots, dtype=bool)

    parent_of: Dict[int, Optional[int]] = {}
    order_of: Dict[int, int] = {}
    for r in system.roots:
        parent_of[r.index] = r.parent
        order_of[r.index] = r.order

    # Find root ancestor within the visible set (or highest ancestor overall)
    def visible_root_anchor(rid: int) -> int:
        cur = rid
        visited = set()
        while cur is not None and cur not in visited:
            visited.add(cur)
            p = parent_of.get(cur, None)
            if p is not None and p in vis_set:
                cur = p
            else:
                break
        return cur

    clusters: Dict[int, List[int]] = {}
    for rid in vis_set:
        anc = visible_root_anchor(rid)
        clusters.setdefault(anc, []).append(rid)

    total_len = sum(float(visible_lengths_px[r]) for r in vis_set)
    target_len = max(12.0, keep_frac * total_len)

    anchors = list(clusters.keys())
    rng.shuffle(anchors)
    kept = np.zeros(n_roots, dtype=bool)
    accum_len = 0.0

    for anc in anchors:
        members = clusters[anc]
        # Sort members topological-first (lower order first, then longer visible run)
        members.sort(key=lambda r: (order_of.get(r, 0), -float(visible_lengths_px[r])))
        for m in members:
            p = parent_of.get(m, None)
            # If parent is visible in this frame, only keep lateral if parent was kept
            if p is not None and p in vis_set and not kept[p]:
                continue
            # Higher-order laterals have a biological drop probability when keep_frac is small
            ord_m = order_of.get(m, 0)
            if ord_m >= 2 and accum_len > 0.6 * target_len:
                if rng.random() > min(1.0, keep_frac * 1.35):
                    continue
            kept[m] = True
            accum_len += float(visible_lengths_px[m])
            if accum_len >= target_len and rng.random() < 0.7:
                break
        if accum_len >= target_len:
            break

    # Ensure at least one visible root is kept if vis_set was non-empty and keep_frac > 0
    if not kept.any() and keep_frac > 0 and anchors:
        best_anc = max(anchors, key=lambda a: visible_lengths_px[a])
        kept[best_anc] = True

    return kept


def _smooth_polyline_c2(
    pts: np.ndarray,
    radii: np.ndarray,
    smooth_px: float,
    pin_start: bool = False,
    root_id: int = 0,
    tortuosity_mult: float = 1.0,
) -> np.ndarray:
    """Curvature-regulated C2 smoothing + band-limited biological micro-tortuosity.

    1. Removes the sharp C0 corners produced by `subdivide`'s Brownian bridge at coarse nodes.
    2. Injects smooth, multi-octave biological meandering around soil peds (wavelengths ~16-75 px)
       so roots never look like rigid geometric rods.
    3. Regulates curvature so thick roots never self-intersect.
    """
    m = len(pts)
    if m < 4:
        return pts.copy()

    ds = float(np.median(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))))
    ds = max(ds, 0.5)
    med_r = float(np.median(radii))

    # Step 1: Modest C2 corner-rounding filter (removes polygon kinks without straightening bends)
    base_sigma = np.clip(0.35 * float(smooth_px) / ds, 1.1, 2.8)
    out = pts.astype(np.float64).copy()
    p0 = out[0].copy()
    p_end = out[-1].copy()

    for c in range(2):
        out[:, c] = ndi.gaussian_filter1d(out[:, c], sigma=base_sigma, mode="nearest")

    # Compute arc-length and smooth normal along the rounded curve
    seg_d = np.hypot(np.diff(out[:, 0]), np.diff(out[:, 1]))
    arc_s = np.concatenate([[0.0], np.cumsum(seg_d)])
    total_len = float(arc_s[-1])

    # Step 2: Inject smooth, band-limited biological micro-meandering around soil aggregates
    if total_len > 8.0 and tortuosity_mult > 0.05:
        nrms = _compute_smooth_normals(out.astype(np.float32)).astype(np.float64)
        rng_w = np.random.default_rng([int(root_id) * 97 + 13, 2026])
        ph1, ph2, ph3 = rng_w.uniform(0.0, 2.0 * math.pi, size=3)
        k1 = 2.0 * math.pi / rng_w.uniform(55.0, 85.0)
        k2 = 2.0 * math.pi / rng_w.uniform(24.0, 38.0)
        k3 = 2.0 * math.pi / rng_w.uniform(11.0, 17.0)

        wave = (
            0.58 * np.sin(k1 * arc_s + ph1)
            + 0.30 * np.sin(k2 * arc_s + ph2)
            + 0.12 * np.cos(k3 * arc_s + ph3)
        )
        # Amplitude scales gently with root thickness and tortuosity_mult
        amp_px = np.clip(1.35 * float(tortuosity_mult) * (0.65 + 0.25 * math.sqrt(max(med_r, 0.5))), 0.3, 4.5)
        # Taper displacement near pinned start so lateral junctions stay locked to parent
        env = np.ones(m, dtype=np.float64)
        if pin_start:
            env *= np.clip(arc_s / max(4.0 * med_r, 8.0), 0.0, 1.0) ** 1.5
        out[:, 0] += amp_px * env * wave * nrms[:, 0]
        out[:, 1] += amp_px * env * wave * nrms[:, 1]

        # Light C2 polish pass so curvature is everywhere continuous
        polish_sig = max(0.9, min(1.8, (0.65 * med_r) / ds))
        for c in range(2):
            out[:, c] = ndi.gaussian_filter1d(out[:, c], sigma=polish_sig, mode="nearest")

    # Step 3: Preserve endpoints softly
    t = np.linspace(0.0, 1.0, m)[:, None]
    if pin_start:
        w_start = np.exp(-np.arange(m) / max(base_sigma * 0.9, 1.5))[:, None]
        out = out * (1.0 - w_start) + p0[None, :] * w_start
    else:
        out += (1.0 - t) * (p0 - out[0])[None, :] + t * (p_end - out[-1])[None, :]

    return out.astype(np.float32)


def _compute_smooth_normals(pts: np.ndarray) -> np.ndarray:
    """Compute smooth unit normal vectors at every vertex of a 2D polyline."""
    m = len(pts)
    if m < 2:
        return np.zeros_like(pts)
    seg = np.diff(pts, axis=0)
    l = np.hypot(seg[:, 0], seg[:, 1])
    l = np.maximum(l, 1e-6)
    tan = seg / l[:, None]
    seg_nrm = np.stack([-tan[:, 1], tan[:, 0]], axis=1)

    v_nrm = np.empty_like(pts)
    v_nrm[0] = seg_nrm[0]
    v_nrm[-1] = seg_nrm[-1]
    if m > 2:
        v_nrm[1:-1] = 0.5 * (seg_nrm[:-1] + seg_nrm[1:])
    nlen = np.hypot(v_nrm[:, 0], v_nrm[:, 1])
    v_nrm /= np.maximum(nlen[:, None], 1e-6)
    return v_nrm.astype(np.float32)


def extract_smooth_ribbons(
    system: RootSystem,
    proj: ProjectedSystem,
    cam: CameraParams,
    spec: FrameSpec,
    tube: TubeParams,
    vis: VisibilityModel,
    cfg: RealismConfig,
    rng: np.random.Generator,
) -> Tuple[List[RootRibbon], float, float]:
    """Extract continuous, C2-smoothed, topologically connected root ribbons in frame coordinates."""
    w, h = cam.width_px, cam.height_px
    su, sv = cam.px_per_mm_u, cam.px_per_mm_v
    r_in_mm = tube.r_in * MM_PER_CM
    u0 = r_in_mm * spec.phi_center
    v0 = spec.axial_cm * MM_PER_CM

    if len(proj) == 0:
        return [], 0.0, 0.0

    du = (proj.u_mm - u0 + 0.5 * proj.circumference_mm) % proj.circumference_mm - 0.5 * proj.circumference_mm
    x = du * su + w / 2.0
    y = (proj.v_mm - v0) * sv + h / 2.0

    pad = 16.0
    inx = (np.maximum(x[:, 0], x[:, 1]) >= -pad) & (np.minimum(x[:, 0], x[:, 1]) <= w + pad)
    iny = (np.maximum(y[:, 0], y[:, 1]) >= -pad) & (np.minimum(y[:, 0], y[:, 1]) <= h + pad)
    nowrap = np.abs(x[:, 1] - x[:, 0]) < 0.5 * proj.circumference_mm * su
    sel = np.nonzero(inx & iny & nowrap)[0]
    if sel.size == 0:
        return [], 0.0, 0.0

    d_mid = 0.5 * (proj.depth_mm[sel, 0] + proj.depth_mm[sel, 1])
    dia_mid = 0.5 * (proj.diam_mm[sel, 0] + proj.diam_mm[sel, 1])
    vdep_mid = (
        d_mid
        if proj.vis_depth_mm is None
        else 0.5 * (proj.vis_depth_mm[sel, 0] + proj.vis_depth_mm[sel, 1])
    )
    pv = vis.p_visible(vdep_mid, dia_mid)

    n_roots = max(len(system.roots), int(proj.root_id.max()) + 1)
    u_root = rng.random(n_roots)
    fade = 0.32
    t_all = np.clip((pv - u_root[proj.root_id[sel]] + fade) / fade, 0.0, 1.0)
    t_all = t_all * t_all * (3.0 - 2.0 * t_all)
    live = t_all >= 0.18
    sel = sel[live]
    t_fade = t_all[live]
    if sel.size == 0:
        return [], 0.0, 0.0

    # Compute visible length per root for connected-subtree selection
    rid_s = proj.root_id[sel]
    seg2d_px = np.hypot(x[sel, 1] - x[sel, 0], y[sel, 1] - y[sel, 0])
    vis_len_by_root = np.bincount(rid_s, weights=seg2d_px, minlength=n_roots)
    vis_rids = np.unique(rid_s)

    kept_roots = select_connected_roots(
        system,
        vis_rids,
        vis_len_by_root,
        cfg.root_keep,
        rng,
        connected=cfg.connected_topology,
    )
    keep_mask = kept_roots[rid_s]
    sel = sel[keep_mask]
    t_fade = t_fade[keep_mask]
    if sel.size == 0:
        return [], 0.0, 0.0

    # Group selected segments into contiguous runs per root_id
    rid_s = proj.root_id[sel]
    parent_lookup = {r.index: (r.parent if r.parent is not None else -1) for r in system.roots}

    # Count total segments per root in proj to know if a run reaches the true apical tip
    max_idx_per_root: Dict[int, int] = {}
    for idx_all, r_all in enumerate(proj.root_id):
        max_idx_per_root[int(r_all)] = idx_all
    min_idx_per_root: Dict[int, int] = {}
    for idx_all in range(len(proj.root_id) - 1, -1, -1):
        min_idx_per_root[int(proj.root_id[idx_all])] = idx_all

    ribbons: List[RootRibbon] = []
    len3d_total = 0.0
    len2d_total = 0.0

    # Split by root_id and by contiguous segment indices (allowing small gaps <= 3 segments)
    order_sort = np.lexsort((sel, rid_s))
    sel_sorted = sel[order_sort]
    tf_sorted = t_fade[order_sort]
    rid_sorted = rid_s[order_sort]

    breaks = np.nonzero(
        (rid_sorted[1:] != rid_sorted[:-1]) | ((sel_sorted[1:] - sel_sorted[:-1]) > 4)
    )[0] + 1
    groups_sel = np.split(sel_sorted, breaks)
    groups_tf = np.split(tf_sorted, breaks)

    smooth_px = cfg.spline_smooth_mm * su

    for g_sel, g_tf in zip(groups_sel, groups_tf):
        if g_sel.size < 2:
            continue
        rid = int(proj.root_id[g_sel[0]])
        ord_val = int(proj.order[g_sel[0]])
        parent_id = parent_lookup.get(rid, -1)

        # Build vertex sequence (M = len(g_sel) + 1)
        pts = np.vstack([
            np.column_stack([x[g_sel, 0], y[g_sel, 0]]),
            [[x[g_sel[-1], 1], y[g_sel[-1], 1]]],
        ]).astype(np.float32)

        diam_v = np.concatenate([proj.diam_mm[g_sel, 0], [proj.diam_mm[g_sel[-1], 1]]]).astype(np.float32)
        depth_v = np.concatenate([proj.depth_mm[g_sel, 0], [proj.depth_mm[g_sel[-1], 1]]]).astype(np.float32)
        age_v = np.concatenate([proj.age_days[g_sel], [proj.age_days[g_sel[-1]]]]).astype(np.float32)
        vis_v = np.concatenate([g_tf, [g_tf[-1]]]).astype(np.float32)
        s3d_v = np.concatenate([[0.0], np.cumsum(proj.seg_len3d_mm[g_sel])]).astype(np.float32)

        radii_px = 0.5 * diam_v * su
        med_w_px = float(np.median(2.0 * radii_px))

        # Check planarity & minimum aspect ratio so steep end-on stubs don't render as round blobs
        raw_step = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        total_2d_px = float(raw_step.sum())
        if total_2d_px < max(3.5, 2.2 * med_w_px):
            continue
        if float(s3d_v[-1]) > 1e-4 and (total_2d_px / su) / float(s3d_v[-1]) < 0.30:
            continue

        is_lat_base = (ord_val > 0) and (int(g_sel[0]) <= min_idx_per_root.get(rid, 0) + 2)
        is_tip = int(g_sel[-1]) >= max_idx_per_root.get(rid, 0) - 2

        # Smooth 2D centerline (C2) + organic micro-meandering around soil peds
        pts_smooth = _smooth_polyline_c2(
            pts,
            radii_px,
            smooth_px=smooth_px,
            pin_start=is_lat_base,
            root_id=rid,
            tortuosity_mult=float(cfg.tortuosity_mult),
        )

        # Smooth radii along the root so width changes are gentle and organic
        if len(radii_px) >= 4:
            radii_px = ndi.gaussian_filter1d(radii_px, sigma=max(2.0, 0.8 * med_w_px), mode="nearest")
            diam_v = (2.0 * radii_px / su).astype(np.float32)

        # Compute cumulative 2D arc length in px
        seg_d = np.hypot(np.diff(pts_smooth[:, 0]), np.diff(pts_smooth[:, 1]))
        arc_s = np.concatenate([[0.0], np.cumsum(seg_d)]).astype(np.float32)

        # Apply biological branch collar flare at lateral base
        if is_lat_base and cfg.junction_collar_flare > 0.0 and len(arc_s) > 2:
            collar_len = max(2.5, 2.2 * float(radii_px[0]))
            flare = 1.0 + float(cfg.junction_collar_flare) * np.exp(-arc_s / collar_len)
            radii_px = (radii_px * flare).astype(np.float32)

        # Apply parabolic/ogive root cap taper at true apical tip
        if is_tip and len(arc_s) > 3:
            tip_zone = min(max(3.5, 2.8 * float(radii_px[-1])), 0.45 * float(arc_s[-1]))
            dist_from_tip = float(arc_s[-1]) - arc_s
            in_tip = dist_from_tip < tip_zone
            if in_tip.any():
                u_tip = np.clip(dist_from_tip[in_tip] / max(tip_zone, 1e-4), 0.0, 1.0)
                # Smooth ogive profile down to 0.38x radius at meristem apex
                ogive = 0.38 + 0.62 * np.sqrt(u_tip * (2.0 - u_tip))
                radii_px[in_tip] *= ogive.astype(np.float32)

        # Smooth visibility taper at diving ends (where root enters/leaves soil behind tube wall)
        taper_len_px = max(4.0, 1.8 * med_w_px)
        if not is_lat_base and float(arc_s[-1]) > 2.0 * taper_len_px:
            # Start of run emerges from soil
            w_in = np.clip(arc_s / taper_len_px, 0.15, 1.0)
            vis_v = np.minimum(vis_v, np.sqrt(w_in)).astype(np.float32)
        if not is_tip and float(arc_s[-1]) > 2.0 * taper_len_px:
            # End of run dives into soil
            w_out = np.clip((float(arc_s[-1]) - arc_s) / taper_len_px, 0.15, 1.0)
            vis_v = np.minimum(vis_v, np.sqrt(w_out)).astype(np.float32)

        normals = _compute_smooth_normals(pts_smooth)

        ribbons.append(
            RootRibbon(
                root_id=rid,
                order=ord_val,
                parent_id=parent_id,
                pts=pts_smooth,
                normals=normals,
                arc_s=arc_s,
                s3d_mm=s3d_v,
                radii_px=radii_px,
                diam_mm=diam_v,
                depth_mm=depth_v,
                vis_weight=vis_v,
                age_days=age_v,
                is_apical_tip=is_tip,
                is_lateral_base=is_lat_base,
            )
        )
        len3d_total += float(np.sum(proj.seg_len3d_mm[g_sel] * g_tf))
        len2d_total += float(np.sum((seg_d / su) * g_tf))

    return ribbons, len3d_total, len2d_total


def evaluate_ribbon_local_box(
    rib: RootRibbon,
    width: int,
    height: int,
    edge_roughness_px: float = 0.45,
    shadow_width_px: float = 2.8,
    shadow_asym: float = 0.25,
    depth_fade_mm: float = 1.2,
    depth_blur_max_px: float = 2.4,
    cover_min_radius_px: float = 0.5,
) -> Optional[Tuple[slice, slice, dict]]:
    """Evaluate exact continuous (s, t) tubular coordinates for a single RootRibbon on its bbox.

    Uses chunked vectorized closest-point projection onto the C2 polyline with
    Phong-interpolated vertex normals, guaranteeing C1-continuous (s, t) with zero
    internal capsule caps or bisector seams.
    """
    pts = rib.pts
    m = len(pts)
    if m < 2:
        return None

    max_r = float(rib.radii_px.max())
    pad = max_r + shadow_width_px + depth_blur_max_px + 3.5
    x0 = max(0, int(math.floor(float(pts[:, 0].min()) - pad)))
    x1 = min(width, int(math.ceil(float(pts[:, 0].max()) + pad)) + 1)
    y0 = max(0, int(math.floor(float(pts[:, 1].min()) - pad)))
    y1 = min(height, int(math.ceil(float(pts[:, 1].max()) + pad)) + 1)
    if x1 <= x0 or y1 <= y0:
        return None

    bh, bw = y1 - y0, x1 - x0
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    px = xs + 0.5
    py = ys + 0.5

    # Subsample polyline vertices if needed so segment length is ~1.5-2.5 px (fast & exact)
    total_s = float(rib.arc_s[-1])
    step_target = max(1.4, min(3.0, 0.35 * float(np.median(rib.radii_px))))
    if m > 12 and total_s / m < step_target * 0.7:
        n_sub = max(2, int(math.ceil(total_s / step_target)) + 1)
        idx = np.unique(np.rint(np.linspace(0, m - 1, n_sub)).astype(int))
        pts = rib.pts[idx]
        nrms = _compute_smooth_normals(pts)
        arc_s = rib.arc_s[idx]
        s3d = rib.s3d_mm[idx]
        rad = rib.radii_px[idx]
        diam = rib.diam_mm[idx]
        dep = rib.depth_mm[idx]
        vis_w = rib.vis_weight[idx]
        age = rib.age_days[idx]
    else:
        nrms = rib.normals
        arc_s = rib.arc_s
        s3d = rib.s3d_mm
        rad = rib.radii_px
        diam = rib.diam_mm
        dep = rib.depth_mm
        vis_w = rib.vis_weight
        age = rib.age_days

    n_seg = len(pts) - 1
    ax = pts[:-1, 0]
    ay = pts[:-1, 1]
    bx = pts[1:, 0]
    by = pts[1:, 1]
    dx = bx - ax
    dy = by - ay
    l2 = np.maximum(dx * dx + dy * dy, 1e-8)

    # Track minimum squared distance and winning segment (j*, u*) across all segments
    best_d2 = np.full((bh, bw), np.inf, dtype=np.float32)
    best_j = np.zeros((bh, bw), dtype=np.int32)
    best_u = np.zeros((bh, bw), dtype=np.float32)

    # Process segments in small batches for cache efficiency
    batch = 32
    for j0 in range(0, n_seg, batch):
        j1 = min(n_seg, j0 + batch)
        # (bh, bw, B)
        pax = px[..., None] - ax[None, None, j0:j1]
        pay = py[..., None] - ay[None, None, j0:j1]
        u = np.clip((pax * dx[None, None, j0:j1] + pay * dy[None, None, j0:j1]) / l2[None, None, j0:j1], 0.0, 1.0)
        qx = pax - u * dx[None, None, j0:j1]
        qy = pay - u * dy[None, None, j0:j1]
        d2 = qx * qx + qy * qy

        min_idx = np.argmin(d2, axis=2)
        min_d2 = np.take_along_axis(d2, min_idx[..., None], axis=2)[..., 0]
        min_u = np.take_along_axis(u, min_idx[..., None], axis=2)[..., 0]

        better = min_d2 < best_d2
        if better.any():
            best_d2 = np.where(better, min_d2, best_d2)
            best_j = np.where(better, min_idx + j0, best_j)
            best_u = np.where(better, min_u, best_u)

    dist = np.sqrt(best_d2)

    # Interpolate continuous attributes at (best_j, best_u)
    j = best_j
    u = best_u
    r_loc = (1.0 - u) * rad[j] + u * rad[j + 1]
    s_loc = (1.0 - u) * arc_s[j] + u * arc_s[j + 1]
    s3d_loc = (1.0 - u) * s3d[j] + u * s3d[j + 1]
    diam_loc = (1.0 - u) * diam[j] + u * diam[j + 1]
    dep_loc = (1.0 - u) * dep[j] + u * dep[j + 1]
    vis_loc = (1.0 - u) * vis_w[j] + u * vis_w[j + 1]
    age_loc = (1.0 - u) * age[j] + u * age[j + 1]

    # Closest point Q and Phong-interpolated normal N(s) for continuous sign
    qx = ax[j] + u * dx[j]
    qy = ay[j] + u * dy[j]
    nx = (1.0 - u) * nrms[j, 0] + u * nrms[j + 1, 0]
    ny = (1.0 - u) * nrms[j, 1] + u * nrms[j + 1, 1]
    n_norm = np.maximum(np.hypot(nx, ny), 1e-6)
    nx /= n_norm
    ny /= n_norm

    perp_dot = (px - qx) * nx + (py - qy) * ny
    sign = np.where(perp_dot >= 0.0, 1.0, -1.0).astype(np.float32)

    # Depth-into-soil effect: deeper roots narrow slightly as only their top crown touches the focal plane
    depth_pos = np.maximum(dep_loc, 0.0)
    depth_factor = np.clip(depth_pos / max(depth_fade_mm, 0.2), 0.0, 1.5)
    crown_narrow = 1.0 - 0.18 * np.clip(depth_factor, 0.0, 1.0)
    min_r_floor = max(0.18, min(0.35, float(rib.radii_px.min()) * 0.5))
    r_eff = np.maximum(r_loc * crown_narrow, min_r_floor)

    # Organic epidermal boundary micro-roughness (correlated along s, asymmetric on left/right flanks)
    if edge_roughness_px > 0.0 and total_s > 4.0:
        # Deterministic multi-frequency boundary wave per root_id + side
        k1 = 0.22 + 0.03 * (rib.root_id % 5)
        k2 = 0.61 + 0.05 * (rib.root_id % 7)
        ph1 = float((rib.root_id * 1.73) % (2 * math.pi))
        ph2 = float((rib.root_id * 2.91) % (2 * math.pi))
        wave = 0.65 * np.sin(k1 * s_loc + ph1 + 0.8 * sign) + 0.35 * np.sin(k2 * s_loc + ph2 - 1.1 * sign)
        # Scale roughness down on very thin roots so 1-px roots don't pinch off
        r_scale = np.clip((r_eff - 0.8) / 2.5, 0.15, 1.0)
        dr = float(edge_roughness_px) * r_scale * wave
        r_perturbed = np.maximum(r_eff + dr, min_r_floor)
    else:
        r_perturbed = r_eff

    # Handle non-cap ends (diving into soil or attached to parent): suppress hemispherical overhang
    # For pixels projecting onto the very first vertex (j==0, u==0) or last vertex (j==n_seg-1, u==1),
    # measure axial overhang beyond the endpoint.
    tan0_x, tan0_y = dx[0] / math.sqrt(l2[0]), dy[0] / math.sqrt(l2[0])
    tan1_x, tan1_y = dx[-1] / math.sqrt(l2[-1]), dy[-1] / math.sqrt(l2[-1])
    over_start = -((px - pts[0, 0]) * tan0_x + (py - pts[0, 1]) * tan0_y)
    over_end = (px - pts[-1, 0]) * tan1_x + (py - pts[-1, 1]) * tan1_y

    # At diving ends (not apical tip), taper overhang softly instead of a sharp semicircle cap
    end_fade = np.ones_like(dist)
    if not rib.is_apical_tip:
        m_end = (j == n_seg - 1) & (u >= 0.999) & (over_end > 0)
        if m_end.any():
            end_fade = np.where(m_end, np.exp(-1.5 * (over_end / np.maximum(r_eff, 1.0)) ** 2), end_fade)
    if not rib.is_lateral_base:
        m_start = (j == 0) & (u <= 0.001) & (over_start > 0)
        if m_start.any():
            end_fade = np.where(m_start, np.exp(-1.5 * (over_start / np.maximum(r_eff, 1.0)) ** 2), end_fade)

    # Signed distance to the perturbed root boundary
    sdf = dist - r_perturbed
    blur_loc = float(depth_blur_max_px) * np.clip(depth_factor ** 1.2, 0.0, 1.0) + 1.2 * (1.0 - vis_loc)
    aa_width = 1.0 + blur_loc
    cov_alpha = np.clip(0.5 - sdf / np.maximum(aa_width, 1e-4), 0.0, 1.0) * end_fade * vis_loc

    # Geometric mask coverage uses unperturbed radius floored at cover_min_radius_px (exact label!)
    r_geom = np.maximum(r_loc, float(cover_min_radius_px))
    sdf_geom = dist - r_geom
    # Do not extend geometric mask into faded overhang beyond diving ends
    valid_geom = (end_fade > 0.45) & (vis_loc >= 0.25)
    cov_geom = np.where(valid_geom, np.clip(0.5 - sdf_geom, 0.0, 1.0), 0.0).astype(np.float32)

    # Normalized cross-axis coordinate t in [-1, 1] inside the root
    t_coord = np.clip(sign * dist / np.maximum(r_perturbed, 0.5), -1.0, 1.0).astype(np.float32)

    # Contact crevice / air-gap shadow just outside the root boundary (sdf in [0, shadow_width_px])
    if shadow_width_px > 0.0:
        w_sh = max(float(shadow_width_px), 0.5) * np.clip(r_eff / 3.2, 0.40, 1.25)
        # Peaks slightly outside the root boundary (sdf ~ 0.65 px) and decays outward over w_sh
        sh_prof = np.exp(-0.5 * ((sdf - 0.65) / np.maximum(0.55 * w_sh, 0.45)) ** 2)
        sh_prof = np.where(sdf >= -0.2, sh_prof, 0.0)
        # Intermittent air-gap modulation along the root axis (not a solid rail line!)
        gap_wave = np.clip(
            0.45 + 0.55 * np.sin(0.085 * s_loc + float(rib.root_id * 1.9) + 0.7 * sign),
            0.0,
            1.0,
        )
        # Directional asymmetry (one side of the cylinder casts a slightly deeper crevice shadow)
        asym = np.clip(0.85 + 0.45 * float(shadow_asym) * sign, 0.25, 1.45)
        # Deeper roots (far from glass) cast weaker contact shadows on the glass
        contact_prox = np.exp(-1.4 * depth_factor) * vis_loc * end_fade
        shadow_local = (sh_prof * gap_wave * asym * contact_prox).astype(np.float32)
    else:
        shadow_local = np.zeros_like(dist)

    return (
        slice(y0, y1),
        slice(x0, x1),
        dict(
            cov_alpha=cov_alpha.astype(np.float32),
            cov_geom=cov_geom.astype(np.float32),
            t_coord=t_coord,
            s_px=s_loc.astype(np.float32),
            s3d_mm=s3d_loc.astype(np.float32),
            r_px=r_perturbed.astype(np.float32),
            diam_mm=diam_loc.astype(np.float32),
            depth_mm=dep_loc.astype(np.float32),
            blur_px=blur_loc.astype(np.float32),
            age_days=age_loc.astype(np.float32),
            shadow=shadow_local,
        ),
    )
