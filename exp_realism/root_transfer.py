"""Move an annotation to another date of the same tube & window, at the exact spot.

Adapted from CanoPie's "Find canopy in other images" (``canopie/canopy_finder.py``:
``stackreg_align`` / ``_score_pose``), in two levels:

1. Whole frame.  Consecutive sessions shift the camera by 40-170 px, which
   StackReg alone (a local optimiser) does not find, so phase correlation on a
   small rotation grid (a 2 deg turn can hide the true peak) gives the coarse
   pose and ``pystackreg`` refines it (translation, rigid body, scaled
   rotation or affine) at half resolution.  Candidates are scored by the
   normalised cross-correlation (NCC) of the band-passed frames over their
   overlap; the best is kept, and a pair below ``MIN_MATCH`` does not line up
   (a third of the consecutive Panama dates: re-inserted tube, very different
   light) -- nothing is placed then.
2. Each root (nearby pieces travel together).  As in CanoPie, a crop around
   the root is matched in the other frame -- brought into the source grid by
   step 1, so only residual motion is left -- with a coarse
   ``matchTemplate`` inside +-``search_px`` and then ``StackReg(RIGID_BODY)``
   on the crop.  A pose is scored 0.2 x NCC over the root + 0.8 x NCC over the
   crop; the local pose is kept only where it beats the frame's.  Every root
   of a matched pair is placed (on 99 Panama date pairs, dropping roots by
   score only lowered the agreement with the next date's annotation: F1 0.40
   with all, 0.34 above 0.3); those scoring below ``MIN_ROOT_SCORE`` are
   reported for checking.
3. Snap onto the root.  Soil texture changes between dates (wet / dry), and
   soil and roots shift by 10-20 px differently across a frame, which neither
   fit sees.  Each placed root's centreline is slid (+-``SNAP_PX``, +-4 deg) to
   where the destination's root evidence along it is strongest -- the image's
   ridge strength and, when given, the model's probability map -- and moved
   there when that is clearly better (``SNAP_GAIN``) than where it landed.
   (Panama images 79 -> 80: F1 vs the user's mask 0.72 -> 0.86; over 41 date
   pairs never worse.)

The shapes are moved exactly (``cv2.warpAffine``, nearest neighbour), not
re-traced: re-tracing is :func:`temporal.carry_roots`' job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from exp_realism import temporal

MIN_MATCH = 0.25           # frame NCC (band-passed) below which two dates do not line up
MIN_OVERLAP = 0.30         # share of the frame that must overlap after the move
MIN_ROOT_SCORE = 0.30      # per-root pose score below which a placed root is flagged for checking (Panama:
                           # >= 0.3 lands on the next date's roots 36-49 % of the time, below it 8-21 %)
COARSE_ANGLES = (0.0, -1.5, 1.5, -3.0, 3.0)   # deg, rotation grid of the coarse phase correlation
MASK_WEIGHT = 0.2          # pose score = 0.2 x NCC over the root + 0.8 x NCC over its crop (CanoPie default)
MIN_PIECE_PX = 15          # groups smaller than this follow the frame's transform
MIN_FINE_PX = 150          # ... smaller than this get the template match but no StackReg fit
GROUP_GAP_PX = 8           # pieces closer than this are registered together
MAX_LOCAL_ANGLE = 8.0      # deg: a per-root rotation beyond this is a failed fit
WORK_PX = 1_000_000        # register big scans (6 MP CI-600) downscaled to about this many pixels
SNAP_PX = 24               # snap: search radius (working pixels) ...
SNAP_ANGLES = (-4.0, -2.0, 0.0, 2.0, 4.0)   # ... and turns (deg) of each root's centreline
SNAP_GAIN = 1.4            # move only when the root evidence along it rises by 40 % ...
SNAP_PENALTY = 0.3         # ... after a distance penalty of 0.3 x (shift / SNAP_PX)
SNAP_MIN = 0.12            # ... and reaches this (evidence normalised to the frame's 99.5th percentile)
MODES = ("phase", "translation", "rigid", "scaled_rotation", "affine")
MODE_LABELS = {"phase": "Shift only (phase correlation)", "translation": "Translation",
               "rigid": "Rigid body", "scaled_rotation": "Scaled rotation", "affine": "Affine"}


@dataclass
class Registration:
    M: np.ndarray               # 2x3 float32: source pixel -> destination pixel
    ncc: float                  # band-passed NCC over the overlap
    overlap: float
    method: str

    @property
    def matched(self) -> bool:
        return self.ncc >= MIN_MATCH and self.overlap >= MIN_OVERLAP

    def describe(self, shape: Tuple[int, int]) -> str:
        return describe_transform(self.M, shape) + f", match {self.ncc:.2f}"


@dataclass
class RootMatch:
    box: Tuple[int, int, int, int]   # y0, y1, x0, x1 in the destination frame
    sub: np.ndarray             # bool, the moved root inside ``box`` (crops: a 6 MP scan has 100+ roots)
    T: np.ndarray               # 2x3 source -> destination for this root
    score: float
    status: str                 # "found" | "low_score" | "outside"
    area: int
    local: bool = False         # moved by its own (per-root) fit
    snapped: bool = False       # slid onto the root in the destination (step 3)

    def paint(self, out: np.ndarray):
        y0, y1, x0, x1 = self.box
        out[y0:y1, x0:x1] |= self.sub


@dataclass
class TransferResult:
    mask: np.ndarray            # union of the roots found, destination frame
    roots: List[RootMatch] = field(default_factory=list)
    reg: Optional[Registration] = None
    message: str = ""
    placed: int = 0             # roots in ``mask``

    @property
    def n_found(self) -> int:
        return sum(r.status == "found" for r in self.roots)

    @property
    def n_total(self) -> int:
        return len(self.roots)


def describe_transform(M: np.ndarray, shape: Tuple[int, int]) -> str:
    """"shift +31,-12 px, rot 0.4 deg[, scale 1.02]" at the frame centre."""
    H, W = shape[:2]
    c = np.array([W / 2.0, H / 2.0])
    d = M[:, :2] @ c + M[:, 2] - c
    ang = math.degrees(math.atan2(float(M[1, 0]), float(M[0, 0])))
    sc = math.sqrt(abs(float(np.linalg.det(M[:, :2]))))
    return (f"shift {d[0]:+.0f},{d[1]:+.0f} px, rot {ang:+.1f}°"
            + (f", scale {sc:.3f}" if abs(sc - 1.0) >= 0.005 else ""))


# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------

def _h(M: np.ndarray) -> np.ndarray:
    return np.vstack([np.asarray(M, np.float64)[:2], [0.0, 0.0, 1.0]])


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 10:
        return -1.0
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / den) if den > 1e-9 else -1.0


def _pull(img: np.ndarray, M: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """``out(q) = img(M q)``: the other image resampled on this grid (M: this grid -> img)."""
    return cv2.warpAffine(img, np.float32(M[:2]), size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _stackreg(mode: str):
    from pystackreg import StackReg
    return StackReg({"translation": StackReg.TRANSLATION, "rigid": StackReg.RIGID_BODY,
                     "scaled_rotation": StackReg.SCALED_ROTATION, "affine": StackReg.AFFINE}[mode])


def _frame_score(M: np.ndarray, gs: np.ndarray, gd: np.ndarray) -> Tuple[float, float]:
    """(NCC, overlap) of the destination pulled onto the source through ``M`` (source -> destination)."""
    H, W = gs.shape
    valid = _pull(np.ones_like(gd), M, (W, H)) > 0.999
    ov = float(valid.mean())
    if ov < MIN_OVERLAP:
        return -1.0, ov
    return _ncc(_pull(gd, M, (W, H))[valid], gs[valid]), ov


# ------------------------------------------------------------------------------
# 1. whole frame
# ------------------------------------------------------------------------------

def register_frames(src_rgb: np.ndarray, dst_rgb: np.ndarray, mode: str = "affine",
                    gs: Optional[np.ndarray] = None, gd: Optional[np.ndarray] = None) -> Registration:
    """Map from ``src_rgb`` pixels to ``dst_rgb`` pixels: phase correlation, then StackReg ``mode``."""
    if src_rgb.shape != dst_rgb.shape:
        return Registration(np.float32([[1, 0, 0], [0, 1, 0]]), -1.0, 0.0, "different frame sizes")
    gs = temporal._bandpass(src_rgb) if gs is None else gs
    gd = temporal._bandpass(dst_rgb) if gd is None else gd
    H, W = gs.shape
    # coarse: phase correlation on a small rotation grid (a 2 deg turn alone can hide the true peak)
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    coarse = []
    for ang in COARSE_ANGLES:
        R = _h(cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, 1.0))
        (dx, dy), _ = cv2.phaseCorrelate(cv2.warpAffine(gs, np.float32(R[:2]), (W, H)), gd, win)
        M = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64) @ R
        ncc, ov = _frame_score(M, gs, gd)
        coarse.append((ncc, ov, M, "phase correlation" + (f" at {ang:+.1f}°" if ang else "")))
    coarse.sort(key=lambda c: -c[0])
    cands = list(coarse[:1])
    if mode != "phase":                                       # fine: StackReg from the best start (and a close 2nd)
        f = 0.5
        ref = cv2.resize(gd, None, fx=f, fy=f, interpolation=cv2.INTER_AREA).astype(np.float64)
        small = cv2.resize(gs, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        starts = [c for c in coarse[:2] if c[0] >= coarse[0][0] - 0.05]
        for _, _, M0, _ in starts:
            try:
                M0s = M0.copy()
                M0s[:2, 2] *= f
                mov = cv2.warpAffine(small, np.float32(M0s[:2]), small.shape[::-1]).astype(np.float64)
                tmat = _stackreg(mode).register(ref, mov)      # destination (half) -> moved source (half)
                Fi = np.linalg.inv(tmat)
                Fi[:2, 2] /= f
                F = Fi @ M0
                ncc, ov = _frame_score(F, gs, gd)
                cands.append((ncc, ov, F, f"phase correlation + StackReg {MODE_LABELS[mode].lower()}"))
            except Exception:
                pass
    ncc, ov, F, name = max(cands, key=lambda c: c[0])
    return Registration(np.float32(F[:2]), ncc, ov, name)


# ------------------------------------------------------------------------------
# 2. each root
# ------------------------------------------------------------------------------

Box = Tuple[int, int, int, int]


def _groups(mask: np.ndarray) -> List[Tuple[Box, np.ndarray]]:
    """Pieces of ``mask`` closer than GROUP_GAP_PX, as (box, bool crop) (largest first)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GROUP_GAP_PX + 1,) * 2)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(mask.astype(np.uint8), k), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in st[i, :4])
        sub = mask[y:y + h, x:x + w] & (lbl[y:y + h, x:x + w] == i)
        if sub.any():
            out.append(((y, y + h, x, x + w), sub))
    return sorted(out, key=lambda g: -int(g[1].sum()))


def _warp_piece(box: Box, sub: np.ndarray, T: np.ndarray, shape: Tuple[int, int]) -> Tuple[Box, np.ndarray]:
    """A cropped mask moved by ``T`` (source -> destination frame), cropped again in the destination."""
    H, W = shape
    y0, y1, x0, x1 = box
    c = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], np.float64) @ T[:, :2].T + T[:, 2]
    dx0, dy0 = max(0, int(np.floor(c[:, 0].min())) - 2), max(0, int(np.floor(c[:, 1].min())) - 2)
    dx1, dy1 = min(W, int(np.ceil(c[:, 0].max())) + 3), min(H, int(np.ceil(c[:, 1].max())) + 3)
    if dx1 <= dx0 or dy1 <= dy0:
        return (0, 0, 0, 0), np.zeros((0, 0), bool)
    A = _h(T)
    A = np.array([[1, 0, -dx0], [0, 1, -dy0], [0, 0, 1]], np.float64) @ A @ np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]])
    m = cv2.warpAffine(sub.astype(np.uint8), np.float32(A[:2]), (dx1 - dx0, dy1 - dy0), flags=cv2.INTER_NEAREST) > 0
    return (dy0, dy1, dx0, dx1), m


def _pose_score(Lq: np.ndarray, Dp: np.ndarray, Vp: np.ndarray, tpl: np.ndarray, rmask: np.ndarray) -> float:
    """CanoPie's pose score for a crop: ``Lq`` maps crop pixels into the pulled destination ``Dp``."""
    ch, cw = tpl.shape
    al = _pull(Dp, Lq, (cw, ch))
    ok = _pull(Vp, Lq, (cw, ch)) > 0.999
    if ok.sum() < 0.3 * ok.size:
        return -1.0
    nw = _ncc(al[ok], tpl[ok])
    m = ok & rmask
    nm = _ncc(al[m], tpl[m]) if m.sum() >= 10 else nw
    return MASK_WEIGHT * nm + (1.0 - MASK_WEIGHT) * nw


def _local_fit(box: Box, sub: np.ndarray, gs: np.ndarray, Dp: np.ndarray, Vp: np.ndarray, search_px: int,
               fine: bool = True) -> Tuple[np.ndarray, float, float, bool]:
    """Residual move of one root (``sub`` at ``box`` in the source grid): (3x3 L: source -> pulled-
    destination grid, score, score without moving, moved?).  ``fine`` adds the StackReg rigid fit."""
    H, W = gs.shape
    ys, xs = np.nonzero(sub)
    y0, y1, x0, x1 = (box[0] + int(ys.min()), box[0] + int(ys.max()) + 1,
                      box[2] + int(xs.min()), box[2] + int(xs.max()) + 1)
    pad = int(np.clip(0.5 * max(y1 - y0, x1 - x0), 30, 80))
    y0, y1, x0, x1 = max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)
    tpl = gs[y0:y1, x0:x1]
    rm = np.zeros(tpl.shape, np.uint8)                       # the root itself, on the crop
    oy, ox = box[0] - y0, box[2] - x0
    sy, sx = max(0, oy), max(0, ox)
    ey, ex = min(tpl.shape[0], oy + sub.shape[0]), min(tpl.shape[1], ox + sub.shape[1])
    rm[sy:ey, sx:ex] = sub[sy - oy:ey - oy, sx - ox:ex - ox]
    rmask = cv2.dilate(rm, np.ones((7, 7), np.uint8)) > 0
    ch, cw = tpl.shape

    def crop_map(L):                                        # crop pixel -> pulled grid, via source coords
        return L @ np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float64)

    I = np.eye(3)
    s0 = _pose_score(crop_map(I), Dp, Vp, tpl, rmask)
    best = (I, s0)
    # coarse: template match inside +-search_px (gated to real pixels)
    sy0, sy1, sx0, sx1 = max(0, y0 - search_px), min(H, y1 + search_px), max(0, x0 - search_px), min(W, x1 + search_px)
    S, R = Dp[sy0:sy1, sx0:sx1], Vp[sy0:sy1, sx0:sx1]
    if S.shape[0] < ch or S.shape[1] < cw:
        return best[0], best[1], s0, False
    res = cv2.matchTemplate(S, tpl, cv2.TM_CCOEFF_NORMED)
    frac = cv2.boxFilter((R > 0.999).astype(np.float32), -1, (cw, ch), anchor=(0, 0), normalize=True,
                         borderType=cv2.BORDER_CONSTANT)[: res.shape[0], : res.shape[1]]
    res = np.where(frac >= 0.5, np.nan_to_num(res, nan=-1.0), -1.0)
    _, mv, _, (px, py) = cv2.minMaxLoc(res)
    if mv <= -1.0:
        return best[0], best[1], s0, False
    dx, dy = (sx0 + px) - x0, (sy0 + py) - y0
    Lt = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64)
    st = _pose_score(crop_map(Lt), Dp, Vp, tpl, rmask)
    if st > best[1]:
        best = (Lt, st)
    # fine: StackReg rigid body on the crop cut at the coarse offset
    try:
        cut = Dp[y0 + dy: y1 + dy, x0 + dx: x1 + dx]
        if fine and cut.shape == tpl.shape:
            tmat = _stackreg("rigid").register(tpl.astype(np.float64) + 1.0, cut.astype(np.float64) + 1.0)
            ang = math.degrees(math.atan2(tmat[1, 0], tmat[0, 0]))
            if abs(ang) <= MAX_LOCAL_ANGLE and abs(tmat[0, 2]) <= search_px and abs(tmat[1, 2]) <= search_px:
                o = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float64)
                Lr = o @ np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]]) @ tmat @ np.linalg.inv(o)
                sr = _pose_score(crop_map(Lr), Dp, Vp, tpl, rmask)
                if sr > best[1]:
                    best = (Lr, sr)
        # the CanoPie rule: keep a local pose only where it clearly beats the frame's
    except Exception:
        pass
    moved = best[1] > s0 + 0.02 and best[0] is not I
    return (best[0] if moved else I), (best[1] if moved else s0), s0, moved


def find_roots(src_rgb: np.ndarray, src_mask: np.ndarray, dst_rgb: np.ndarray, reg: Registration,
               search_px: int = 24, min_score: float = MIN_ROOT_SCORE, local: bool = True,
               gs: Optional[np.ndarray] = None, gd: Optional[np.ndarray] = None,
               scale: float = 1.0) -> List[RootMatch]:
    """Each root of ``src_mask`` placed on ``dst_rgb`` (frame transform, refined per root when ``local``).

    With ``scale`` < 1, ``reg``, ``gs``, ``gd`` and ``search_px`` are at the working resolution (frame
    size x ``scale``); ``src_mask`` and the returned masks / transforms are at full resolution."""
    if gs is None or gd is None:
        work = (lambda im: cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)) \
            if scale < 1 else (lambda im: im)
        gs = temporal._bandpass(work(src_rgb)) if gs is None else gs
        gd = temporal._bandpass(work(dst_rgb)) if gd is None else gd
    H, W = src_mask.shape[:2]
    h, w = gs.shape
    S = np.diag([scale, scale, 1.0])
    F = _h(reg.M)
    Dp = _pull(gd, F, (w, h))                                  # destination on the source grid
    Vp = _pull(np.ones_like(gd), F, (w, h))
    reach = search_px if reg.matched else 3 * search_px       # unmatched frames: look further for each root
    out: List[RootMatch] = []
    for box, sub in _groups(src_mask.astype(bool)):
        area = int(sub.sum())
        if scale < 1:                                          # the piece on the working grid
            wb = (int(box[0] * scale), int(np.ceil(box[1] * scale)), int(box[2] * scale), int(np.ceil(box[3] * scale)))
            wsub = cv2.resize(sub.astype(np.float32), (max(1, wb[3] - wb[2]), max(1, wb[1] - wb[0])),
                              interpolation=cv2.INTER_AREA) > 0.2
        else:
            wb, wsub = box, sub
        L, score, moved = np.eye(3), None, False
        n_w = int(wsub.sum())
        if local and n_w >= MIN_PIECE_PX:                     # StackReg only where a root is big enough to steer it
            L, score, _, moved = _local_fit(wb, wsub, gs, Dp, Vp, reach, fine=n_w >= MIN_FINE_PX)
        if score is None:                                      # tiny piece: the frame's fit decides
            score = reg.ncc
        T = (np.linalg.inv(S) @ F @ L @ S)[:2].astype(np.float32)     # full-resolution source -> destination
        dbox, dsub = _warp_piece(box, sub, T, (H, W))
        inside = float(dsub.sum()) / max(area, 1)
        status = "outside" if inside < 0.5 else ("found" if score >= min_score else "low_score")
        out.append(RootMatch(dbox, dsub, T, float(score), status, area, moved))
    return out


# ------------------------------------------------------------------------------
# 3. snap each root onto the root in the destination
# ------------------------------------------------------------------------------

def root_evidence(rgb: np.ndarray, prob: Optional[np.ndarray] = None) -> np.ndarray:
    """Where roots are in a frame, ~0-1.5: light-ridge strength of grey / blue (each normalised to its
    99.5th percentile), and the model's probability (sqrt-stretched, to its 99.7th) when given."""
    from exp_realism.root_tracer import SIGMAS, _ridge_maps, image_channels
    ch = image_channels(rgb)
    ev = None
    for k in ("grey", "blue"):
        R = _ridge_maps(ch[k], SIGMAS)["light"][0]
        R = R / (np.percentile(R, 99.5) + 1e-6)
        ev = R if ev is None else np.maximum(ev, R)
    if prob is not None and prob.shape == ev.shape:
        p = np.sqrt(np.clip(prob, 0.0, 1.0))
        ev = np.maximum(ev, p / (np.percentile(p, 99.7) + 1e-6))
    return np.clip(ev, 0.0, 1.5).astype(np.float32)


def _snap(box: Box, sub: np.ndarray, E: np.ndarray, reach: int = SNAP_PX) -> Tuple[np.ndarray, float, float]:
    """Best move (3x3: a shift and a turn about the centroid) of a placed root's centreline onto ``E``:
    (move, evidence along the moved centreline, evidence where it is now)."""
    from skimage.morphology import skeletonize
    H, W = E.shape
    sk = skeletonize(sub)
    if sk.sum() < 5:
        return np.eye(3), 0.0, 0.0
    pad = reach + 6
    y0, x0 = box[0] - pad, box[2] - pad
    h, w = sub.shape[0] + 2 * pad, sub.shape[1] + 2 * pad
    canvas = np.zeros((h, w), np.float32)
    canvas[pad:pad + sub.shape[0], pad:pad + sub.shape[1]] = sk
    ys, xs = np.nonzero(canvas)
    c = (float(xs.mean()), float(ys.mean()))
    ev = np.zeros((h + 2 * reach, w + 2 * reach), np.float32)    # evidence around it, zero off the frame
    ey0, ex0 = y0 - reach, x0 - reach
    sy0, sy1, sx0, sx1 = max(0, ey0), min(H, ey0 + h + 2 * reach), max(0, ex0), min(W, ex0 + w + 2 * reach)
    if sy1 <= sy0 or sx1 <= sx0:
        return np.eye(3), 0.0, 0.0
    ev[sy0 - ey0:sy1 - ey0, sx0 - ex0:sx1 - ex0] = E[sy0:sy1, sx0:sx1]
    yy, xx = np.mgrid[-reach:reach + 1, -reach:reach + 1]
    pen = SNAP_PENALTY * np.hypot(yy, xx) / reach
    best, s0 = None, 0.0
    for a in SNAP_ANGLES:
        t = cv2.warpAffine(canvas, cv2.getRotationMatrix2D(c, a, 1.0), (w, h), flags=cv2.INTER_LINEAR)
        n = float(t.sum())
        if n < 3:
            continue
        sc = cv2.matchTemplate(ev, t, cv2.TM_CCORR) / n         # mean evidence along it, for every shift
        if a == 0.0:
            s0 = float(sc[reach, reach])
        k = np.unravel_index(int(np.argmax(sc - pen)), sc.shape)
        if best is None or sc[k] - pen[k] > best[0]:
            best = (float(sc[k] - pen[k]), float(sc[k]), k[1] - reach, k[0] - reach, a)
    if best is None:
        return np.eye(3), 0.0, s0
    _, sup, dx, dy, a = best
    M = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float64) @ _h(
        cv2.getRotationMatrix2D((c[0] + x0, c[1] + y0), a, 1.0))
    return M, sup, s0


def snap_roots(roots: List[RootMatch], dst_rgb: np.ndarray, dst_prob: Optional[np.ndarray] = None) -> int:
    """Step 3 on placed roots (in place); returns how many moved.  Big scans are matched at ``WORK_PX``."""
    H, W = dst_rgb.shape[:2]
    s = work_scale(dst_rgb.shape)
    small = (lambda im, interp=cv2.INTER_AREA: cv2.resize(im, None, fx=s, fy=s, interpolation=interp)) \
        if s < 1 else (lambda im, interp=None: im)
    prob = None if dst_prob is None or dst_prob.shape[:2] != (H, W) else small(np.asarray(dst_prob, np.float32))
    E = root_evidence(small(dst_rgb), prob)
    h, w = E.shape
    S = np.diag([s, s, 1.0])
    moved = 0
    for r in roots:
        if r.status == "outside" or not r.sub.any():
            continue
        if s < 1:
            wb = (int(r.box[0] * s), int(np.ceil(r.box[1] * s)), int(r.box[2] * s), int(np.ceil(r.box[3] * s)))
            wsub = cv2.resize(r.sub.astype(np.float32), (max(1, wb[3] - wb[2]), max(1, wb[1] - wb[0])),
                              interpolation=cv2.INTER_AREA) > 0.2
        else:
            wb, wsub = r.box, r.sub
        M, sup, s0 = _snap(wb, wsub, E)
        if sup < SNAP_MIN or sup < SNAP_GAIN * max(s0, 1e-3) or np.allclose(M, np.eye(3)):
            continue
        Mf = (np.linalg.inv(S) @ M @ S)[:2].astype(np.float32)        # full-resolution destination move
        r.box, r.sub = _warp_piece(r.box, r.sub, Mf, (H, W))
        r.T = (_h(Mf) @ _h(r.T))[:2].astype(np.float32)
        r.snapped = True
        moved += 1
    return moved


def work_scale(shape: Tuple[int, ...]) -> float:
    """Registration resolution: frames above ``WORK_PX`` are matched downscaled (as CanoPie caps its grid)."""
    return float(min(1.0, math.sqrt(WORK_PX / float(shape[0] * shape[1]))))


def transfer(src_rgb: np.ndarray, src_mask: np.ndarray, dst_rgb: np.ndarray, mode: str = "affine",
             local: bool = True, search_px: int = 24, min_score: float = MIN_ROOT_SCORE,
             force: bool = False, snap: bool = True, dst_prob: Optional[np.ndarray] = None) -> TransferResult:
    """Register the frames, place each root, snap it onto the root there, and summarise (module docstring).

    ``mask`` holds every root that lands in the frame when the frames match (or ``force``), else nothing.
    ``dst_prob``: the model's probability map of the destination, extra root evidence for the snap.
    Big scans are registered at about ``WORK_PX`` pixels; the shapes are moved at full resolution."""
    H, W = dst_rgb.shape[:2]
    empty = np.zeros((H, W), bool)
    if src_mask is None or not np.asarray(src_mask).any():
        return TransferResult(empty, [], None, "nothing to transfer: the source has no roots")
    if src_rgb.shape != dst_rgb.shape:
        return TransferResult(empty, [], None, "the two frames have different sizes")
    s = work_scale(src_rgb.shape)
    work = (lambda im: cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)) if s < 1 else (lambda im: im)
    A, B = work(src_rgb), work(dst_rgb)
    gs, gd = temporal._bandpass(A), temporal._bandpass(B)
    reg_w = register_frames(A, B, mode, gs, gd)
    roots = find_roots(src_rgb, src_mask, dst_rgb, reg_w, search_px, min_score, local, gs, gd, scale=s)
    S = np.diag([s, s, 1.0])
    reg = Registration(np.float32((np.linalg.inv(S) @ _h(reg_w.M) @ S)[:2]), reg_w.ncc, reg_w.overlap, reg_w.method)
    res = TransferResult(empty.copy(), roots, reg)
    placed = [r for r in roots if r.status != "outside"] if (reg.matched or force) else []
    n_snap = snap_roots(placed, dst_rgb, dst_prob) if (snap and placed) else 0
    for r in placed:
        r.paint(res.mask)
    res.placed = len(placed)
    where = reg.describe((H, W))
    if not placed:
        res.message = (f"the dates do not line up ({where}) - nothing placed" if not reg.matched
                       else f"no root lands inside the frame ({where})")
        return res
    low = sum(r.status == "low_score" for r in placed)
    out = sum(r.status == "outside" for r in roots)
    res.message = (f"{len(placed)} root(s) placed ({where}"
                   + (f"; {sum(r.local for r in placed)} refined locally" if local else "")
                   + (f"; {n_snap} snapped onto the root" if snap else "") + ")"
                   + (f"; {low} with a weak match - check them" if low else "")
                   + (f"; {out} fell outside the frame" if out else "")
                   + ("" if reg.matched else " - forced although the dates do not line up"))
    return res


__all__ = ["MIN_MATCH", "MIN_ROOT_SCORE", "WORK_PX", "MODES", "MODE_LABELS", "Registration", "RootMatch", "TransferResult",
           "describe_transform", "register_frames", "find_roots", "root_evidence", "snap_roots", "work_scale", "transfer"]
