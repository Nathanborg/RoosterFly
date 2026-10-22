"""Anti-aliased variable-width rasterisation of root polylines.

Roots are drawn as tapered capsules (round cones) via their signed distance
field, evaluated on a per-segment bounding box.  This gives exact sub-pixel
widths and correct anti-aliasing, and yields the per-pixel diameter/order maps
and the depth ordering for free.  ``cv2.line`` with integer thickness cannot do
any of that, and quantising a 1.4 px root to 1 px is a systematic calibre bias in
exactly the regime that matters most.

Two decisions that matter more than they look:

**The mask is not a 0.5 threshold on alpha.**  A 0.6 px root at opacity 0.35
peaks at alpha ~0.21, so thresholding at 0.5 gives it ZERO mask pixels while it
is still plainly visible in the RGB and still fully counted in the length label --
the hardest and most valuable class rendered as an *unlabelled* root.  We emit the
continuous alpha as the primary label and calibrate the binary threshold so that
mask-derived length matches analytic length (:func:`calibrate_mask_threshold`).

**Attribute maps are coverage-weighted, not z-buffered.**  Writing the diameter
map only where ``cov > 0.5`` sets ``diam = 0`` for every sub-pixel root, which
then cascades into anything that consumes it.  We composite front-to-back and
weight each segment by the light it actually contributes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# signed distance to a tapered capsule
# --------------------------------------------------------------------------
def sd_round_cone(px, py, ax, ay, bx, by, r1, r2):
    """Signed distance from points to a 2D round cone (tapered capsule).

    After Inigo Quilez.  Degenerate cases (zero length, one disc containing the
    other) fall back to the enclosing-circle distance rather than producing NaN.
    """
    bax, bay = bx - ax, by - ay
    l2 = bax * bax + bay * bay
    rr = r1 - r2
    a2 = l2 - rr * rr

    if l2 < 1e-12 or a2 <= 1e-12:
        # degenerate: treat as a single disc at the larger end
        r = max(r1, r2)
        cx, cy = (ax, ay) if r1 >= r2 else (bx, by)
        return np.hypot(px - cx, py - cy) - r

    il2 = 1.0 / l2
    pax, pay = px - ax, py - ay
    y = pax * bax + pay * bay
    z = y - l2
    qx = pax * l2 - bax * y
    qy = pay * l2 - bay * y
    x2 = qx * qx + qy * qy
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2

    out = (np.sqrt(np.maximum(x2 * a2 * il2, 0.0)) + y * rr) * il2 - r1
    cap_b = np.sign(z) * a2 * z2 > k
    cap_a = np.sign(y) * a2 * y2 < k
    out = np.where(cap_b, np.sqrt(np.maximum(x2 + z2, 0.0)) * il2 - r2, out)
    out = np.where(cap_a & ~cap_b, np.sqrt(np.maximum(x2 + y2, 0.0)) * il2 - r1, out)
    return out


def coverage_from_sdf(d, aa: float = 1.0):
    """Approximate pixel coverage from a signed distance in pixels."""
    return np.clip(0.5 - d / max(aa, 1e-6), 0.0, 1.0)


# --------------------------------------------------------------------------
# segment description
# --------------------------------------------------------------------------
@dataclass
class Segment:
    """One capsule in image space.  Coordinates in pixels, depth in mm."""

    x0: float
    y0: float
    x1: float
    y1: float
    r0: float                 # radius in px
    r1: float
    depth: float              # mm behind the outer wall (for ordering)
    visibility: float = 1.0   # p(detected) from the visibility model
    opacity: float = 1.0      # material opacity (translucency of fine roots)
    diameter_mm: float = 0.0
    order: int = 0
    root_id: int = 0
    colour: Tuple[float, float, float] = (0.45, 0.35, 0.25)
    # --- v3 realism (0 = flat capsule, exactly as before) --------------------
    # shade: cylindrical shading -- colour toward the edge is multiplied down
    #   to (1 - shade) and alpha to (1 - 0.5 * shade), so a root reads as a
    #   rounded, slightly translucent cylinder, not a painted stripe
    # blur:  extra edge softness in px (out-of-focus stretches)
    shade: float = 0.0
    blur: float = 0.0
    # real root texture (rasterize(textures=...)): strip index, arclength (px)
    # at the segment start, strip rows per image px along the root, and the
    # share of the strip's own colour (rest: the segment colour x strip luma)
    tex_id: int = -1
    s0: float = 0.0
    tex_scale: float = 1.0
    tex_mix: float = 0.7
    tex_profile: bool = False     # use the strip's along-averaged cross profile only


@dataclass
class RasterOut:
    alpha: np.ndarray          # continuous coverage x visibility x opacity
    rgb: np.ndarray            # premultiplied root colour contribution
    diameter_mm: np.ndarray    # coverage-weighted
    order: np.ndarray          # coverage-weighted (rounded on read)
    instance: np.ndarray       # argmax-contribution root id, -1 where empty
    depth_mm: np.ndarray
    overlap_fraction: float    # fraction of root signal hidden behind other roots
    # --- only with rasterize(cover_min_radius_px=...) ------------------------
    # per-pixel MAX geometric coverage of every drawn capsule, radii floored at
    # cover_min_radius_px; opacity, visibility and compositing play no part
    cover: Optional[np.ndarray] = None
    # diameter of the segment that gave each pixel its max cover (0 elsewhere):
    # a crossing takes the root covering the pixel most, ties the nearer root
    cover_diam_mm: Optional[np.ndarray] = None


def _bbox(seg: Segment, w: int, h: int, pad: float = 2.0):
    r = max(seg.r0, seg.r1) + pad
    x_lo = int(np.floor(min(seg.x0, seg.x1) - r))
    x_hi = int(np.ceil(max(seg.x0, seg.x1) + r))
    y_lo = int(np.floor(min(seg.y0, seg.y1) - r))
    y_hi = int(np.ceil(max(seg.y0, seg.y1) + r))
    return max(0, x_lo), min(w, x_hi + 1), max(0, y_lo), min(h, y_hi + 1)


def _texture_colour(seg: Segment, xs, ys, textures, edge_pad: float = 1.15):
    """Per-pixel colour of a textured capsule: the real strip mapped along the
    segment (rows = arclength, columns = across), brought to the segment
    colour's luma, mixed with the segment colour modulated by the strip."""
    strip = textures.strips[seg.tex_id]
    mean_l = max(float(textures.luma[seg.tex_id]), 1e-3)
    ax, ay = seg.x1 - seg.x0, seg.y1 - seg.y0
    L2 = max(ax * ax + ay * ay, 1e-9)
    L = np.sqrt(L2)
    px, py = xs + 0.5 - seg.x0, ys + 0.5 - seg.y0
    h_raw = (px * ax + py * ay) / L2
    h = np.clip(h_raw, 0.0, 1.0)
    perp = (ax * py - ay * px) / L
    r_loc = np.maximum(seg.r0 + h * (seg.r1 - seg.r0), 0.5)
    # distance to the capsule's AXIS SEGMENT (not the infinite line): the round end caps get
    # their edge colour too, instead of the core colour out to the cap rim
    excess = np.maximum(np.maximum(h_raw - 1.0, -h_raw), 0.0) * L
    dist = np.sqrt(perp * perp + excess * excess)
    tt = np.clip(np.where(perp < 0, -dist, dist) / (r_loc * edge_pad), -1.0, 1.0)
    Ls, K = strip.shape[:2]
    # the UNclamped position along the axis: the round end caps continue the texture instead of
    # repeating one row (which drew rings across thick roots)
    row = (seg.s0 + h_raw * L) * float(seg.tex_scale)
    m = np.mod(row, 2 * Ls)
    row = np.where(m < Ls, m, 2 * Ls - 1 - m).astype(np.int32).clip(0, Ls - 1)
    col = np.rint((tt + 1.0) * 0.5 * (K - 1)).astype(np.int32)
    if seg.tex_profile and hasattr(textures, "profiles"):
        tex = textures.profiles[seg.tex_id][col]             # (h, w, 3): cross profile only
    else:
        tex = strip[row, col]                                # (h, w, 3)
    tl = tex @ np.float32([0.299, 0.587, 0.114])
    c = np.asarray(seg.colour, np.float32)
    cl = float(c @ np.float32([0.299, 0.587, 0.114]))
    mix = float(seg.tex_mix)
    out = (1.0 - mix) * c[None, None] * (tl / mean_l)[..., None] + mix * tex * (cl / mean_l)
    # brightening a strip also multiplies its chroma (camera casts turn orange):
    # pull the colour toward its own luma
    lo = out @ np.float32([0.299, 0.587, 0.114])
    out = lo[..., None] + TEX_CHROMA * (out - lo[..., None])
    return np.clip(out, 0.0, 1.0).astype(np.float32)


TEX_CHROMA = 0.5


def rasterize(segments: Sequence[Segment], width: int, height: int,
              aa: float = 1.0, cover_min_radius_px: Optional[float] = None,
              textures=None) -> RasterOut:
    """Front-to-back composite of tapered capsules.

    Segments are sorted by depth so nearer roots occlude farther ones, and each
    segment's contribution is weighted by the remaining transmittance.  That
    transmittance is also what makes the attribute maps correct without a
    z-buffer, and it gives the overlap fraction for free -- which downstream code
    needs, because the analytic sum-over-fragments length double-counts every
    crossing.

    ``cover_min_radius_px`` (v3, the geometric mask): also return
    ``RasterOut.cover``, the per-pixel max of each capsule's geometric
    coverage with both radii floored at this value, and ``cover_diam_mm``, the
    diameter of the capsule giving that max.  It ignores opacity and
    visibility on purpose: "label everything that is drawn" must not depend on
    how bright a root was drawn, or every contrast knob would move the labels.
    The floor (0.5 px = a 1-px-wide line) keeps a sub-pixel root as an
    8-connected line at any angle -- a pixel centre in each column/row lies
    within 0.5*cos(theta) <= 0.5 px of it.  Where both radii already reach the
    floor the alpha coverage is reused, so the extra cost is only for thin
    segments.  ``None`` (default) leaves the output exactly as before.

    ``textures`` (v3, real root appearance): an object with ``strips`` (list of
    (L, K, 3) float RGB) and ``luma`` (mean luma per strip), e.g.
    :class:`mrstyle.realbanks.RootBank`; segments with ``tex_id >= 0`` take
    their colour from that strip (:func:`_texture_colour`).
    """
    want_cover = cover_min_radius_px is not None
    if want_cover:
        rfl = float(cover_min_radius_px)
        cover = np.zeros((height, width), np.float32)
        cdiam = np.zeros((height, width), np.float32)
    alpha = np.zeros((height, width), np.float32)
    rgb = np.zeros((height, width, 3), np.float32)
    dmap = np.zeros((height, width), np.float32)
    omap = np.zeros((height, width), np.float32)
    wsum = np.zeros((height, width), np.float32)
    best = np.zeros((height, width), np.float32)
    inst = np.full((height, width), -1, np.int32)
    depth = np.full((height, width), np.inf, np.float32)
    trans = np.ones((height, width), np.float32)   # remaining transmittance

    total_signal = 0.0
    visible_signal = 0.0

    # Group by root.  Consecutive capsules along one root necessarily overlap
    # (they are spaced well inside their own radius), so compositing them against
    # each other is wrong twice over: a semi-transparent root beads visibly at
    # every joint, and the reported overlap fraction is dominated by a root
    # overlapping ITSELF rather than by genuine crossings.  Within a root the
    # correct combination is a per-pixel MAX -- it is one object.  Only ACROSS
    # roots is over-compositing meaningful.
    by_root: dict = {}
    for seg in segments:
        by_root.setdefault(seg.root_id, []).append(seg)

    def root_depth(items):
        return min(s.depth for s in items)

    for root_id in sorted(by_root, key=lambda k: root_depth(by_root[k])):
        items = by_root[root_id]
        boxes = [_bbox(s, width, height) for s in items]
        boxes = [b for b in boxes if b[1] > b[0] and b[3] > b[2]]
        if not boxes:
            continue
        x_lo = min(b[0] for b in boxes); x_hi = max(b[1] for b in boxes)
        y_lo = min(b[2] for b in boxes); y_hi = max(b[3] for b in boxes)
        ysl, xsl = slice(y_lo, y_hi), slice(x_lo, x_hi)

        ra = np.zeros((y_hi - y_lo, x_hi - x_lo), np.float32)
        rd = np.zeros_like(ra)
        rc = np.zeros((y_hi - y_lo, x_hi - x_lo, 3), np.float32)
        rdep = np.full_like(ra, np.inf)
        rq = np.full_like(ra, -np.inf)       # textured roots: depth inside the winning capsule
        textured = textures is not None and any(s_.tex_id >= 0 for s_ in items)
        order_val = 0.0

        for seg in items:
            sx0, sx1, sy0, sy1 = _bbox(seg, width, height)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            # evaluate each segment over its OWN bounding box, then write into the
            # root buffer; evaluating over the whole root's box is quadratic for a
            # root that spans the frame
            ys, xs = np.mgrid[sy0:sy1, sx0:sx1]
            d = sd_round_cone(xs + 0.5, ys + 0.5, seg.x0, seg.y0, seg.x1, seg.y1,
                              seg.r0, seg.r1)
            cov = coverage_from_sdf(d, aa + float(seg.blur))
            if want_cover:
                # before the empty-alpha skip: a capsule thinner than a pixel can
                # cover no pixel centre with alpha yet still deserve a label
                if seg.r0 >= rfl and seg.r1 >= rfl and not seg.blur:
                    cg = cov
                else:
                    cg = coverage_from_sdf(sd_round_cone(
                        xs + 0.5, ys + 0.5, seg.x0, seg.y0, seg.x1, seg.y1,
                        max(seg.r0, rfl), max(seg.r1, rfl)), aa)
                csub = cover[sy0:sy1, sx0:sx1]
                tk = cg > csub
                if tk.any():
                    cover[sy0:sy1, sx0:sx1] = np.where(tk, cg, csub)
                    cdiam[sy0:sy1, sx0:sx1] = np.where(
                        tk, np.float32(seg.diameter_mm), cdiam[sy0:sy1, sx0:sx1])
            if not cov.any():
                continue
            a = (cov * float(seg.visibility) * float(seg.opacity)).astype(np.float32)
            shade = None
            if seg.shade > 0.0:
                # rho: 1 on the axis, 0 at the edge (mean radius of the capsule)
                rm = max(0.5 * (seg.r0 + seg.r1), 0.5)
                rho = np.clip(-d / rm, 0.0, 1.0)
                prof = np.sqrt(np.clip(rho * (2.0 - rho), 0.0, 1.0))   # cylinder cross-section
                shade = (1.0 - float(seg.shade) * (1.0 - prof)).astype(np.float32)
                a = a * (1.0 - 0.5 * float(seg.shade) * (1.0 - prof)).astype(np.float32)
            ly, lx = slice(sy0 - y_lo, sy1 - y_lo), slice(sx0 - x_lo, sx1 - x_lo)
            sub = ra[ly, lx]
            if textured:
                # among overlapping capsules of this root, the colour comes from the one the pixel
                # lies deepest inside (q = depth / radius): no cap rims inside thick roots
                q = (-d / max(0.5 * (seg.r0 + seg.r1), 0.5)).astype(np.float32)
                qs = rq[ly, lx]
                take = (a > sub + 1e-4) | ((a >= sub - 1e-4) & (q > qs))
                rq[ly, lx] = np.where(take, q, qs)
            else:
                take = a > sub
            ra[ly, lx] = np.maximum(sub, a)
            rd[ly, lx] = np.where(take, np.float32(seg.diameter_mm), rd[ly, lx])
            rdep[ly, lx] = np.where(take, np.float32(seg.depth), rdep[ly, lx])
            if textures is not None and seg.tex_id >= 0:
                tcol = _texture_colour(seg, xs, ys, textures)
                rc[ly, lx] = np.where(take[..., None], tcol, rc[ly, lx])
            else:
                for c in range(3):
                    col = np.float32(seg.colour[c]) if shade is None else np.float32(seg.colour[c]) * shade
                    rc[ly, lx, c] = np.where(take, col, rc[ly, lx, c])
            order_val = float(seg.order)

        if not ra.any():
            continue

        t_view = trans[ysl, xsl]
        contrib = ra * t_view                # light this root actually adds

        total_signal += float(ra.sum())
        visible_signal += float(contrib.sum())

        alpha[ysl, xsl] += contrib
        for c in range(3):
            rgb[ysl, xsl, c] += contrib * rc[..., c]

        dmap[ysl, xsl] += contrib * rd
        omap[ysl, xsl] += contrib * order_val
        wsum[ysl, xsl] += contrib

        take = contrib > best[ysl, xsl]
        best[ysl, xsl] = np.where(take, contrib, best[ysl, xsl])
        inst[ysl, xsl] = np.where(take, np.int32(root_id), inst[ysl, xsl])
        depth[ysl, xsl] = np.where(take, rdep.astype(np.float32), depth[ysl, xsl])

        trans[ysl, xsl] = t_view * (1.0 - ra)

    np.clip(alpha, 0.0, 1.0, out=alpha)
    nz = wsum > 1e-8
    dmap[nz] /= wsum[nz]
    omap[nz] /= wsum[nz]
    depth[~np.isfinite(depth)] = np.nan

    overlap = 0.0
    if total_signal > 1e-9:
        overlap = 1.0 - visible_signal / total_signal

    out = RasterOut(alpha=alpha, rgb=rgb, diameter_mm=dmap, order=omap,
                    instance=inst, depth_mm=depth,
                    overlap_fraction=float(np.clip(overlap, 0.0, 1.0)))
    if want_cover:
        out.cover = cover
        out.cover_diam_mm = cdiam
    return out


# --------------------------------------------------------------------------
# the geometric mask (v3: label everything that is drawn)
# --------------------------------------------------------------------------
def geometric_mask(cover: np.ndarray, occl: Optional[np.ndarray] = None,
                   thr: float = 0.5, occl_min: float = 0.5) -> np.ndarray:
    """``cover >= thr`` where the soil cover lets the root through.

    The mask follows GEOMETRY, not opacity: a 0.6 px root at opacity 0.35
    (alpha ~0.2) is labelled as a 1-px line, a 12 px root is ~12 px wide
    whatever its colour.  The soil-cover transmittance ``occl`` (crumbs on the
    glass in front of the root) is respected -- a root hidden under a crumb is
    not visible, so it is not labelled -- and it is ~bimodal, so 0.5 is not a
    sensitive choice.  No threshold is fitted: the calibrated mode's bisection
    (12 skeletonisations per frame) is gone.
    """
    m = np.asarray(cover) >= thr
    if occl is not None:
        m &= np.asarray(occl) >= occl_min
    return m


def diameter_on_mask(diam_mm: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A per-pixel diameter map that is > 0 EXACTLY on ``mask`` (float32, mm).

    Takes ``diam_mm`` (normally ``RasterOut.cover_diam_mm``: the max-cover
    segment's diameter, so a crossing takes one root's width rather than a
    blend) on the mask, and fills any mask pixel that has none from the
    NEAREST pixel that has one (EDT indices).  Off the mask it is 0.  The
    coverage-weighted ``diameter_mm`` blends crossings and is 0 wherever a
    pixel got its label from the radius floor with zero alpha, which would put
    0-width pixels into notebook 04's true-width bins.  A mask with no
    diameter anywhere stays 0 (the caller decides what that means).
    """
    from scipy import ndimage as ndi
    m = np.asarray(mask, bool)
    d = np.where(m, np.asarray(diam_mm, np.float32), np.float32(0.0)).astype(np.float32)
    have = d > 0
    need = m & ~have
    if need.any() and have.any():
        _, (iy, ix) = ndi.distance_transform_edt(~have, return_indices=True)
        d = np.where(need, d[iy, ix], d).astype(np.float32)
    return d


# --------------------------------------------------------------------------
# mask calibration
# --------------------------------------------------------------------------
def mask_from_alpha(alpha: np.ndarray, threshold: float) -> np.ndarray:
    return alpha >= threshold


def calibrate_mask_threshold(alpha: np.ndarray, target_length_px: float,
                             lo: float = 0.02, hi: float = 0.9,
                             iters: int = 12) -> float:
    """Find the alpha threshold whose mask-derived length matches the analytic one.

    An arbitrary 0.5 is a *choice*, and a bad one: it silently deletes the
    sub-pixel class.  Fitting the threshold makes the binary mask a stated,
    reproducible approximation of a quantity we know exactly.
    """
    from .traits import skeleton_length_px   # local import: avoids a cycle

    if target_length_px <= 0 or not np.any(alpha > lo):
        return 0.5

    # Track the best threshold actually EVALUATED rather than returning the final
    # midpoint.  Mask length is a step function of the threshold: a root of
    # uniform alpha (any flat-topped root, e.g. a wide translucent one) vanishes
    # completely the moment the threshold crosses its plateau.  Plain bisection
    # walks onto the wrong side of that step and returns a threshold that erases
    # the root it was supposed to preserve.
    best_thr, best_err = 0.5, float("inf")
    # 12 bisections resolve the threshold far finer than the alpha quantisation;
    # each step costs a full skeletonisation, and this was the largest single
    # cost per frame once the textures were cached.
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        got = skeleton_length_px(mask_from_alpha(alpha, mid))
        err = abs(got - target_length_px)
        # prefer the lower threshold on a tie: keeping a faint root is the safer
        # error, since a dropped root is an unlabelled root
        if err < best_err - 1e-9 or (abs(err - best_err) <= 1e-9 and mid < best_thr):
            best_err, best_thr = err, mid
        if got > target_length_px:
            lo = mid          # too much mask -> raise the threshold
        else:
            hi = mid
    return best_thr


def mask_recall_by_width(alpha: np.ndarray, diameter_px: np.ndarray,
                         threshold: float,
                         bins: Sequence[float] = (0, 1, 2, 3, 5, 8, 1e9)):
    """Fraction of root signal surviving the threshold, stratified by width.

    This is a first-class deliverable, not a diagnostic: it is the honest answer
    to "which roots does the mask actually contain?", and it is exactly where
    published pipelines quietly lose their thin roots.
    """
    mask = mask_from_alpha(alpha, threshold)
    out = []
    for i in range(len(bins) - 1):
        sel = (diameter_px >= bins[i]) & (diameter_px < bins[i + 1]) & (alpha > 0)
        tot = float(alpha[sel].sum())
        kept = float(alpha[sel & mask].sum())
        out.append({
            "width_px_lo": float(bins[i]),
            "width_px_hi": float(bins[i + 1]),
            "signal": tot,
            "recall": (kept / tot) if tot > 1e-9 else float("nan"),
        })
    return out
