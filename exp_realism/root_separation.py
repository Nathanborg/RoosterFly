"""Individual roots from a root mask and the model's centre-line map.

Where roots touch or cross, the mask is one blob, and the skeleton of a blob
grows spurious spokes.  The model's second head predicts how central each pixel
is within its root (distance to the root's edge over its local maximum, as in
training: 1 on every root's middle line), so even inside a clump each root has
its own ridge.  :func:`separate_roots`:

1. centre lines = skeleton of the ridge (centre > ``RIDGE``) inside the mask;
2. cut at junctions into pieces (shorter than ``MIN_PIECE`` px dropped); pieces
   that continue straight through a junction (a turn under ``MAX_TURN_DEG``) are
   joined, so a root passing a crossing or a branch stays one root;
3. every mask pixel goes to its root by a watershed down the centre-line map;
   mask pieces without a centre line become roots of their own.

Length is measured along the centre lines (step-weighted: 1 per straight, sqrt 2
per diagonal step), diameter as the median of 2 x the distance to the edge of
the root's own region along its centre line (a crossing does not inflate it).  For a mask drawn by hand, where the model saw no
root, the map is made from the mask's shape (:func:`centre_map`).

On the densest window of a RootDetector Beech scan (KO_T074) this gave 236 roots
where the mask had 60 blobs, and 12 % less length than the blobs' skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

RIDGE = 0.55            # centre-line map level taken as a root's middle
MIN_PIECE = 6           # px: shorter centre-line pieces are dropped
MAX_TURN_DEG = 35.0     # pieces meeting at a junction within this of straight are one root
LINK_REACH = 12         # px around a junction used for the pieces' directions
MODEL_ZONE = 0.3        # where the model's map exceeds this (dilated 5 px) it is used, elsewhere the mask's shape

_SQRT2 = float(np.sqrt(2.0))


@dataclass
class Separation:
    labels: np.ndarray          # int32 HxW: 0 background, 1..n one per root
    centre: np.ndarray          # bool HxW: the roots' centre lines (their root is ``labels`` there)
    n: int


def geometric_centre(mask: np.ndarray) -> np.ndarray:
    """The training target from a mask alone: distance to the edge over its 7x7 local maximum."""
    m = np.ascontiguousarray(mask, dtype=np.uint8)
    if not m.any():
        return np.zeros(m.shape, np.float32)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float32)
    loc = cv2.dilate(dt, np.ones((7, 7), np.uint8))
    return np.where(m > 0, dt / np.maximum(loc, 1.0), 0.0).astype(np.float32)


def centre_map(mask: np.ndarray, model_centre: Optional[np.ndarray] = None) -> np.ndarray:
    """Centre-line map for ``mask``: the model's where it saw roots, the mask's shape elsewhere
    (a root drawn by hand), so separation works on edited annotations too."""
    m = np.asarray(mask, bool)
    geo = geometric_centre(m)
    if model_centre is None or model_centre.shape != m.shape:
        return geo
    mc = np.where(m, np.asarray(model_centre, np.float32), 0.0).astype(np.float32)
    zone = cv2.dilate((mc > MODEL_ZONE).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    return np.where(zone, mc, geo).astype(np.float32)


def _skeleton(m: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize
    return skeletonize(np.ascontiguousarray(m, dtype=bool))


def _neighbours(sk: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    return cv2.filter2D(sk.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)


def _link(sk: np.ndarray, pieces: np.ndarray, junc: np.ndarray) -> np.ndarray:
    """Join pieces that continue straight through a junction; returns piece -> root id (array)."""
    n = int(pieces.max())
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    nj, jl, js, jc = cv2.connectedComponentsWithStats(cv2.dilate(junc.astype(np.uint8), np.ones((3, 3), np.uint8)),
                                                     connectivity=8)
    H, W = sk.shape
    cos_max = np.cos(np.deg2rad(180.0 - MAX_TURN_DEG))
    k3 = np.ones((3, 3), np.uint8)
    for j in range(1, nj):
        x, y, w, h = (int(v) for v in js[j, :4])
        y0, y1 = max(0, y - LINK_REACH), min(H, y + h + LINK_REACH)
        x0, x1 = max(0, x - LINK_REACH), min(W, x + w + LINK_REACH)
        sub = pieces[y0:y1, x0:x1]
        ring = cv2.dilate((jl[y0:y1, x0:x1] == j).astype(np.uint8), k3) > 0
        touching = np.unique(sub[ring])
        touching = touching[touching > 0]
        if len(touching) < 2:
            continue
        cx, cy = jc[j][0] - x0, jc[j][1] - y0
        dirs = {}
        for t in touching:
            ys, xs = np.nonzero(sub == t)
            v = np.array([xs.mean() - cx, ys.mean() - cy])
            nv = float(np.hypot(*v))
            if nv > 1e-6:
                dirs[int(t)] = v / nv
        keys = list(dirs)
        pairs = sorted((float(np.dot(dirs[a], dirs[b])), a, b) for i, a in enumerate(keys) for b in keys[i + 1:])
        used = set()
        for c, a, b in pairs:                                # most opposite first: straight through
            if c > cos_max:
                break
            if a in used or b in used:
                continue
            used.update((a, b))
            parent[find(a)] = find(b)
    return np.array([find(i) for i in range(n + 1)])


def separate_roots(mask: np.ndarray, centre: np.ndarray) -> Separation:
    """Individual roots of ``mask`` (see the module docstring); ``centre`` from :func:`centre_map`."""
    from skimage.segmentation import watershed
    m = np.asarray(mask, bool)
    if not m.any():
        z = np.zeros(m.shape, np.int32)
        return Separation(z, np.zeros(m.shape, bool), 0)
    c = np.asarray(centre, np.float32)
    sk = _skeleton(m & (c > RIDGE))
    nb = _neighbours(sk)
    junc = sk & (nb >= 3)
    piece_px = sk & ~(cv2.dilate(junc.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    npc, pieces, st, _ = cv2.connectedComponentsWithStats(piece_px.astype(np.uint8), connectivity=8)
    small = st[:, cv2.CC_STAT_AREA] < MIN_PIECE
    small[0] = False
    pieces[small[pieces]] = 0
    markers = np.zeros(m.shape, np.int32)
    if npc > 1:
        root_of = _link(sk, pieces, junc)
        ids, inv = np.unique(root_of, return_inverse=True)   # consecutive root ids (0 stays 0)
        markers = inv[pieces].astype(np.int32)
    # mask pieces without any centre line: a root of their own
    nm, mlbl = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    has = np.zeros(nm, bool)
    has[np.unique(mlbl[markers > 0])] = True
    orphan = ~has[mlbl] & m
    if orphan.any():
        _, ol = cv2.connectedComponents(orphan.astype(np.uint8), connectivity=8)
        markers = np.where(orphan, ol + markers.max(), markers).astype(np.int32)
    labels = watershed(-c, markers=markers, mask=m).astype(np.int32)
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.reshape(m.shape).astype(np.int32)
    return Separation(labels, sk & (labels > 0), int(labels.max()))


def _root_lengths(sep: Separation) -> np.ndarray:
    """Step-weighted centre-line length of every root (px), index = root id."""
    lab = np.where(sep.centre, sep.labels, 0)
    n = sep.n + 1
    out = np.zeros(n, np.float64)
    H, W = lab.shape
    for dy, dx, w in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, _SQRT2), (1, -1, _SQRT2)):   # each link once
        if dx >= 0:
            a, b = lab[0:H - dy, 0:W - dx], lab[dy:H, dx:W]
        else:
            a, b = lab[0:H - dy, -dx:W], lab[dy:H, 0:W + dx]
        same = (a > 0) & (a == b)
        out += np.bincount(a[same], minlength=n) * w
    # isolated single pixels still count one pixel
    single = np.bincount(lab[lab > 0], minlength=n)
    return np.where(out > 0, out, np.minimum(single, 1)).astype(np.float64)


def centre_length_px(sep: Separation) -> float:
    return float(_root_lengths(sep)[1:].sum())


def root_rows(sep: Separation, px_per_mm: float) -> List[Dict[str, object]]:
    """One dict per root: id, length, mean diameter, area, centroid."""
    n = sep.n + 1
    if n <= 1:
        return []
    lab = sep.labels
    length_px = _root_lengths(sep)
    area_px = np.bincount(lab.ravel(), minlength=n).astype(np.float64)
    # each root's own region: boundaries with a neighbouring root count as its edge
    edge = np.zeros(lab.shape, bool)
    edge[:, 1:] |= (lab[:, 1:] != lab[:, :-1]) & (lab[:, :-1] > 0) & (lab[:, 1:] > 0)
    edge[1:, :] |= (lab[1:, :] != lab[:-1, :]) & (lab[:-1, :] > 0) & (lab[1:, :] > 0)
    dt = cv2.distanceTransform(((lab > 0) & ~edge).astype(np.uint8), cv2.DIST_L2, 5)
    c = sep.centre & (lab > 0)
    # median along the centre line: a crossing's wide junction does not inflate a root's diameter
    cl, cd = lab[c], 2.0 * dt[c]
    order = np.argsort(cl, kind="stable")
    cl, cd = cl[order], cd[order]
    starts = np.searchsorted(cl, np.arange(n))
    ends = np.searchsorted(cl, np.arange(n), side="right")
    dmed = np.array([float(np.median(cd[s:e])) if e > s else 0.0 for s, e in zip(starts, ends)])
    ys, xs = np.nonzero(lab)
    cxs = np.bincount(lab[ys, xs], weights=xs, minlength=n)
    cys = np.bincount(lab[ys, xs], weights=ys, minlength=n)
    rows = []
    for i in range(1, n):
        if area_px[i] <= 0:
            continue
        rows.append({
            "root_id": i,
            "length_mm": round(float(length_px[i]) / px_per_mm, 3),
            "diameter_mm": round(float(dmed[i]) / px_per_mm, 4),
            "area_mm2": round(float(area_px[i]) / px_per_mm ** 2, 4),
            "centroid_x_px": round(float(cxs[i] / area_px[i]), 1),
            "centroid_y_px": round(float(cys[i] / area_px[i]), 1),
        })
    return rows


def colour_image(rgb: np.ndarray, sep: Separation, alpha: float = 0.6) -> np.ndarray:
    """``rgb`` with every root in its own colour, outlined, and its centre line in white."""
    out = rgb.copy()
    if sep.n == 0:
        return out
    pal = np.random.default_rng(7).integers(70, 256, (sep.n + 1, 3)).astype(np.float32)
    m = sep.labels > 0
    out[m] = (out[m].astype(np.float32) * (1 - alpha) + pal[sep.labels[m]] * alpha).astype(np.uint8)
    lab = sep.labels.astype(np.float32)
    border = (cv2.morphologyEx(lab, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0) & m
    out[border] = (20, 20, 20)
    out[sep.centre] = (255, 255, 255)
    return out


__all__ = ["RIDGE", "Separation", "geometric_centre", "centre_map", "separate_roots", "centre_length_px",
           "root_rows", "colour_image"]
