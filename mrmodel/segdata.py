"""Labels, crops and augmentation for notebook 04 v2 -- numpy and OpenCV only.

Everything a training sample goes through between "decoded image + label" and
"network input", kept out of the notebook so it is tested and so the v1 cells
it replaces can be compared line by line:

* labels: :func:`mask_label` (masks; every root the simulator draws is labelled,
  no ignore band) and :func:`trace_label` (v1 cell 6 verbatim: a Ginger trace is
  a centreline without width, so a 5-px band is root and a ring around it is
  neither root nor soil);
* geometry: :func:`scale_crop_pad` crops THEN resizes -- the source window is
  ``crop / scale`` pixels, so a 736x552 PRMI image is never resized whole
  (wasted loader CPU) and a 0.5x crop is a real 768-px window, not mostly
  padding.  Padding (only when the window is larger than the image) has
  valid = 0, and labels are resized with :func:`resize_label`, which keeps a
  1-px root alive AND connected at any scale (:func:`keep_thin`);
* thin roots: :func:`width_fast` (w2e per root pixel, the ONE width convention
  of ``mrsim.traits.skeleton_widths_px``) and :func:`thin_weights`, which make
  a 1-2 px root weigh about as much per mm of centreline as a 4-px one;
* :class:`AugConfig` / :func:`augment`: v1's augmentation, unchanged and in
  the same random-draw order, plus hue and grayscale.  With ``hue=0`` and
  ``gray_p=0`` the output equals v1's for the same generator state;
* :func:`root_fade` (v1) and the data the ROOT_FADE cap check needs
  (:func:`fade_thin_visibility`, :func:`select_fade_cap`).

Every random choice takes a ``numpy.random.Generator``: the sampler hands each
sample its own seed, so a batch, its crops and its augmentation are
reproducible and a resumed run repeats the original batches (v1 drew crops
from an unseeded generator).

Python 3.8 compatible.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:                                    # pragma: no cover
    cv2 = None

from mrsim.traits import THIN_W2E, skeleton_widths_px
# the thin-root-preserving mask resize and the reference scale are shared with
# the CUT pool and the PRMI scale table: one implementation, re-exported here
from mrstyle.realdata import REF_PX_PER_MM, SCALE_CLAMP, resize_mask  # noqa: F401

# ImageNet statistics of the pretrained encoders (v1 cell 6)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# Thin-root weights, in w2e units.  The 04 design set THIN_REF 4 in the old
# 2*DT-1 convention; the same strength in w2e is about 5 (the value CUT's
# alpha_w uses as multistyle.THIN_REF_W2E).  A 1-px root (w2e 2) gets 2.5, a
# 2-px root 5/3, a 3-px root 1.25, anything from 4 px up 1.
THIN_REF_W2E = 5.0
THIN_MAX_W = 4.0

# input channels per INDEX_MODE; there is no other way to add a channel, so a
# channel can never be invented (v1 fed a zero-filled XGB map to PRMI images)
INDEX_MODES = {"off": 3, "extra-channel": 4, "index-only": 1}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(r) + 1, 2 * int(r) + 1))


def _skeletonize(m: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize
    return skeletonize(np.asarray(m, bool))


def parse_pair(spec, cast=float) -> Tuple[float, float]:
    """``'0.8,1.25'`` (or a 2-sequence) -> ``(0.8, 1.25)``, low first."""
    if isinstance(spec, str):
        parts = [p for p in spec.replace(":", ",").split(",") if p.strip()]
    else:
        parts = list(spec)
    if len(parts) != 2:
        raise ValueError("expected two numbers, got %r" % (spec,))
    a, b = cast(parts[0]), cast(parts[1])
    if not (a > 0 and b > 0):
        raise ValueError("scale bounds must be > 0, got %r" % (spec,))
    return (min(a, b), max(a, b))


def log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """A scale factor uniform in log space: 0.6x and 1/0.6x are equally likely."""
    lo, hi = float(lo), float(hi)
    if lo == hi:
        return lo
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def clamp_scale(s: float, clamp: Tuple[float, float] = SCALE_CLAMP) -> float:
    """``f x aug`` clamped to the shared [0.5, 2.0]: at 0.24x a 736x552 PRMI
    image would be 177x132 (a 384 crop mostly padding), at 4x a 96-px region
    blown up into blur."""
    return float(min(max(float(s), float(clamp[0])), float(clamp[1])))


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------
def mask_label(mask: np.ndarray, edge_ignore_px: int = 0,
               valid: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """``(y, v)`` float32 from a binary mask: y = root, v = 1 (counts in the loss).

    ``edge_ignore_px`` > 0 ignores a band of that half-width around the mask
    boundary, inside and out -- for the blocky WinRHIZO rectangles of the PRMI
    camera subsets, whose edges are not root edges.  The mask's skeleton stays
    valid, so a root thinner than the band is still learned as root rather
    than erased.  ``valid`` (e.g. RootPainter's unlabelled pixels) multiplies v.
    """
    m = np.asarray(mask) > 0
    y = m.astype(np.float32)
    v = np.ones(m.shape, np.float32) if valid is None else (np.asarray(valid) > 0).astype(np.float32)
    e = int(edge_ignore_px)
    if e > 0 and m.any():
        u8 = m.astype(np.uint8)
        k = _disk(e)
        band = (cv2.dilate(u8, k) > 0) & ~(cv2.erode(u8, k) > 0)
        band &= ~_skeletonize(m)
        v[band] = 0.0
    return y, v


def trace_label(trace: np.ndarray, radius: int = 2,
                ignore_ring: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """v1's real-trace label (cell 6 ``real_label``), unchanged.

    Trace centreline -> root within ``radius`` px (a 5-px band at radius 2),
    an ignore ring ``ignore_ring`` px beyond it (the true width is unknown),
    soil elsewhere.  Returns ``(y, v)`` float32.
    """
    sk = _skeletonize(np.asarray(trace) > 0).astype(np.uint8)
    pos = cv2.dilate(sk, _disk(radius)) > 0
    ring = (cv2.dilate(sk, _disk(radius + ignore_ring)) > 0) & ~pos
    return pos.astype(np.float32), (~ring).astype(np.float32)


def keep_thin(a: np.ndarray, hi: float = 0.5, lo: float = 0.1) -> np.ndarray:
    """Binarise an interpolated (resized / rotated) 0-1 label without cutting thin roots.

    ``a > hi`` alone breaks a 1-px root into dashes wherever the line falls
    between output pixel centres (each pixel then gets ~0.3-0.5 of it): a
    continuous root that is plainly visible in the RGB comes out as 5-50
    pieces, the gaps are trained as soil and clDice sees a broken skeleton.
    Adding the skeleton of the faint support ``a > lo`` puts a 1-px-wide
    connected centreline through every such stretch; on thick roots that
    skeleton lies inside ``a > hi`` and changes nothing.  So the result is a
    superset of the plain threshold that only adds thin-root centrelines.
    """
    a = np.asarray(a, np.float32)
    keep = a > hi
    faint = a > lo
    if faint.any():
        keep |= _skeletonize(faint)
    return keep


def resize_label(y: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    """A 0/1 training label resized with thin roots kept connected -> bool.

    Same interpolation and thresholds as ``realdata.resize_mask`` (area
    coverage > 0.25 when shrinking, bilinear > 0.5 when enlarging) -- so thick
    roots are identical -- plus :func:`keep_thin`'s skeleton of the faint
    support, which stops a 1-px root from becoming dashes.  ``resize_mask``
    itself stays untouched: CUT and the scale table share it.
    """
    m = (np.asarray(y) > 0.5).astype(np.float32)
    w, h = int(size_wh[0]), int(size_wh[1])
    if (w, h) == (m.shape[1], m.shape[0]):
        return m > 0.5
    if w * h < m.shape[0] * m.shape[1]:
        return keep_thin(cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA), hi=0.25)
    return keep_thin(cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR), hi=0.5)


def resize_valid(v: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    """A 0/1 validity map resized (area when shrinking, linear when enlarging,
    then >= 0.5) -- the same pixel-centre geometry as the image and the mask,
    unlike OpenCV's INTER_NEAREST, which is half a pixel off."""
    a = np.asarray(v, np.float32)
    w, h = int(size_wh[0]), int(size_wh[1])
    if (w, h) == (a.shape[1], a.shape[0]):
        return (a >= 0.5).astype(np.float32)
    interp = cv2.INTER_AREA if w * h < a.shape[0] * a.shape[1] else cv2.INTER_LINEAR
    return (cv2.resize(a, (w, h), interpolation=interp) >= 0.5).astype(np.float32)


# --------------------------------------------------------------------------
# crop-then-resize
# --------------------------------------------------------------------------
def _offset(n: int, win: int, rng: np.random.Generator) -> int:
    """Start of a ``win`` window over ``n`` pixels: inside the image when it
    fits, otherwise a random placement of the whole image inside the window
    (a negative start = padding before)."""
    lo, hi = min(0, n - win), max(0, n - win)
    return int(rng.integers(lo, hi + 1))


def _window(a: np.ndarray, y0: int, x0: int, win: int, pad: str) -> np.ndarray:
    """``a[y0:y0+win, x0:x0+win]``, padded where the window leaves the image."""
    H, W = a.shape[:2]
    t, l = max(0, -y0), max(0, -x0)
    b, r = max(0, y0 + win - H), max(0, x0 + win - W)
    core = a[max(0, y0):min(H, y0 + win), max(0, x0):min(W, x0 + win)]
    if not (t or l or b or r):
        return core
    pads = ((t, b), (l, r)) + ((0, 0),) * (a.ndim - 2)
    if pad == "reflect" and min(core.shape[:2]) > 1:
        return np.pad(core, pads, mode="reflect")
    if pad == "reflect":
        return np.pad(core, pads, mode="edge")
    return np.pad(core, pads, mode="constant")


def window_side(crop: int, scale: float) -> int:
    """Source pixels of a ``crop``-px network crop at ``scale`` (network px per source px)."""
    return max(1, int(round(float(crop) / float(scale))))


def scale_crop_pad(rgb: np.ndarray, y: np.ndarray, v: Optional[np.ndarray], scale: float,
                   crop: int, rng: np.random.Generator, min_valid: float = 0.5,
                   tries: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A ``crop x crop`` training sample at ``scale``: crop first, then resize.

    The source window is ``window_side(crop, scale)`` pixels.  When it fits,
    it lies inside the image; when it is larger than the image (a 0.5x crop of
    a 552-px side), the whole image is placed at random inside it and the
    rest is reflect-padded with ``v = 0`` -- the loss never sees padding.  Up
    to ``tries`` windows are drawn until at least ``min_valid`` of the window
    is valid (label ignore regions count as invalid); the best one is kept.

    The image is resized with INTER_AREA (shrinking) or INTER_CUBIC
    (enlarging), the label with :func:`resize_label` (a 1-px root survives
    0.5x and stays one connected line) and validity with :func:`resize_valid`: all three share OpenCV's
    pixel-centre mapping, so they stay aligned.  Returns ``(rgb uint8, y
    float32, v float32)``.
    """
    H, W = y.shape[:2]
    win = window_side(crop, scale)
    vv = np.ones((H, W), np.float32) if v is None else np.asarray(v, np.float32)
    best = None
    for _ in range(max(1, int(tries))):
        y0, x0 = _offset(H, win, rng), _offset(W, win, rng)
        vw = _window(vv, y0, x0, win, "constant")
        frac = float(vw.mean())
        if best is None or frac > best[0]:
            best = (frac, y0, x0, vw)
        if frac >= min_valid:
            break
    _, y0, x0, vw = best
    im = _window(np.asarray(rgb), y0, x0, win, "reflect")
    yw = _window(np.asarray(y, np.float32), y0, x0, win, "constant")
    if win == crop:
        return (np.ascontiguousarray(im), np.ascontiguousarray(yw, np.float32),
                (vw >= 0.5).astype(np.float32))
    interp = cv2.INTER_AREA if win > crop else cv2.INTER_CUBIC
    im = cv2.resize(np.ascontiguousarray(im), (crop, crop), interpolation=interp)
    yc = resize_label(yw, (crop, crop)).astype(np.float32)
    vc = resize_valid(vw, (crop, crop))
    return im, yc, vc


# --------------------------------------------------------------------------
# widths and thin-root weights
# --------------------------------------------------------------------------
def width_fast(y: np.ndarray, k: int = 7) -> np.ndarray:
    """w2e per ROOT pixel: 2 x (max of the exact EDT within a ``k x k`` box).

    w2e is ``mrsim.traits.skeleton_widths_px``'s convention (a 1-px line is 2,
    a 3-px bar 4).  The EDT peaks on the centreline, so the local maximum
    gives every pixel of a root its centreline width -- exactly for roots up
    to ``k`` px wide, the only place the thin weights differ from 1; edge
    pixels of wider roots read at least 8 (weight 1 either way).  0 off-root.
    """
    m = (np.asarray(y) > 0.5).astype(np.uint8)
    if not m.any():
        return np.zeros(m.shape, np.float32)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    mx = cv2.dilate(dt, np.ones((int(k), int(k)), np.uint8))
    return np.where(m > 0, 2.0 * mx, 0.0).astype(np.float32)


def thin_weights(y: np.ndarray, v: Optional[np.ndarray] = None, ref: float = THIN_REF_W2E,
                 max_w: float = THIN_MAX_W, width: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-pixel loss weight: ``clip(ref / w2e, 1, max_w)`` on roots, 1 on soil, x v.

    A root's pixels per mm of centreline grow with its width, so an
    unweighted loss lets a few thick roots outvote every 1-2 px root -- the
    roots recall is lowest on.  ``ref / w2e`` roughly cancels that: per mm of
    centreline a 1-px root then weighs about as much as a 4-px one.  The
    weights apply to BCE and the Tversky sums.  ``max_w = 1`` turns them off.
    """
    yy = np.asarray(y)
    m = yy > 0.5
    w = np.ones(yy.shape, np.float32)
    if max_w > 1 and m.any():
        wd = width_fast(yy) if width is None else np.asarray(width, np.float32)
        w[m] = np.clip(float(ref) / np.maximum(wd[m], 1e-6), 1.0, float(max_w))
    if v is not None:
        w = w * np.asarray(v, np.float32)
    return w.astype(np.float32)


# --------------------------------------------------------------------------
# skeleton length (v1 cell 6, the evaluation's length)
# --------------------------------------------------------------------------
def _steps(sk: np.ndarray) -> Tuple[int, int, int]:
    s = np.asarray(sk, bool)
    h = int((s[:, 1:] & s[:, :-1]).sum())
    v = int((s[1:, :] & s[:-1, :]).sum())
    d = int((s[1:, 1:] & s[:-1, :-1] & ~s[:-1, 1:] & ~s[1:, :-1]).sum()
            + (s[1:, :-1] & s[:-1, 1:] & ~s[:-1, :-1] & ~s[1:, 1:]).sum())
    return h, v, d


def skel_len_px(sk: np.ndarray) -> float:
    """v1's 8-connected centreline length: diagonals count sqrt(2) unless a
    corner pixel exists.  Kept exactly so length numbers compare with v1 (it
    differs slightly from ``traits.skeleton_length_px``'s step count)."""
    h, v, d = _steps(sk)
    return float(h + v + math.sqrt(2) * d)


def skel_len_mm(sk: np.ndarray, px_per_mm: float, px_per_mm_y: Optional[float] = None) -> float:
    """:func:`skel_len_px` in mm, anisotropic when ``px_per_mm_y`` differs
    (horizontal steps / px_x, vertical / px_y, diagonals by Pythagoras)."""
    sx = float(px_per_mm)
    sy = sx if px_per_mm_y is None else float(px_per_mm_y)
    h, v, d = _steps(sk)
    return float(h / sx + v / sy + d * math.sqrt((1.0 / sx) ** 2 + (1.0 / sy) ** 2))


# --------------------------------------------------------------------------
# augmentation (v1 cell 6 + hue + grayscale)
# --------------------------------------------------------------------------
@dataclass
class AugConfig:
    """Training augmentation strengths.  Defaults: v1's last run (rotation 5
    degrees) plus hue +-0.03 of the colour circle and grayscale with p 0.1,
    for the near-grey chicory and the many PRMI sensors."""
    brightness: float = 0.08
    contrast: float = 0.2
    saturation: float = 0.2
    gamma: float = 0.25
    flip_rot90: bool = True
    rotate_deg: float = 5.0
    blur_p: float = 0.3
    noise_p: float = 0.3
    hue: float = 0.03
    gray_p: float = 0.1
    thin_label: bool = True      # rotated label keeps 1-px roots connected (keep_thin); v1: > 0.5

    @classmethod
    def v1(cls, rotate_deg: float = 5.0) -> "AugConfig":
        """v1's augmentation exactly (the parity run), including v1's > 0.5 label rule."""
        return cls(hue=0.0, gray_p=0.0, rotate_deg=rotate_deg, thin_label=False)

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def rotate_free(img: np.ndarray, y: np.ndarray, v: np.ndarray, deg: float,
                thin_label: bool = True):
    """Rotate image + label + validity about the centre; mirrored borders (v1).

    The label is warped bilinearly like v1, then binarised with
    :func:`keep_thin` (``thin_label``) -- v1's plain ``> 0.5`` drops 13-39% of
    a 1-px root's pixels at 30-45 degrees and leaves dashes.  ``thin_label=False``
    is v1 exactly.
    """
    h, w = y.shape
    M = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), deg, 1.0)
    br = cv2.BORDER_REFLECT_101
    img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=br)
    yw = cv2.warpAffine(np.ascontiguousarray(y, np.float32), M, (w, h), flags=cv2.INTER_LINEAR,
                        borderMode=br)
    y = keep_thin(yw) if thin_label else yw > 0.5
    v = cv2.warpAffine(np.ascontiguousarray(v), M, (w, h), flags=cv2.INTER_NEAREST, borderMode=br)
    return img, y.astype(np.float32), v


def _hue_shift(x01: np.ndarray, frac: float) -> np.ndarray:
    hsv = cv2.cvtColor(np.ascontiguousarray(x01, np.float32), cv2.COLOR_RGB2HSV)
    hsv[..., 0] = np.mod(hsv[..., 0] + 360.0 * float(frac), 360.0)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment(img: np.ndarray, y: np.ndarray, v: np.ndarray, rng: np.random.Generator,
            cfg: Optional[AugConfig] = None):
    """v1's training augmentation -> ``(x float32 HxWx3 in 0-1, y, v)``.

    Same operations in the same random-draw order as v1 cell 6; hue and
    grayscale draw only when enabled, so ``AugConfig.v1()`` reproduces v1
    exactly for the same generator state.
    """
    cfg = cfg or AugConfig()
    if cfg.flip_rot90:
        if rng.random() < 0.5:
            img, y, v = img[:, ::-1], y[:, ::-1], v[:, ::-1]
        if rng.random() < 0.5:
            img, y, v = img[::-1], y[::-1], v[::-1]
        k = int(rng.integers(4))
        img, y, v = np.rot90(img, k), np.rot90(y, k), np.rot90(v, k)
    if cfg.rotate_deg > 0:
        img, y, v = rotate_free(img, y, v, rng.uniform(-cfg.rotate_deg, cfg.rotate_deg),
                                cfg.thin_label)
    x = img.astype(np.float32) / 255
    x = x * rng.uniform(1 - cfg.contrast, 1 + cfg.contrast) + rng.uniform(-cfg.brightness, cfg.brightness)
    g = x.mean(2, keepdims=True)
    x = g + (x - g) * rng.uniform(1 - cfg.saturation, 1 + cfg.saturation)
    if cfg.hue > 0:
        x = _hue_shift(np.clip(x, 0, 1), rng.uniform(-cfg.hue, cfg.hue))
    x = np.clip(x, 0, 1) ** rng.uniform(1 / (1 + cfg.gamma), 1 + cfg.gamma)
    if cfg.gray_p > 0 and rng.random() < cfg.gray_p:
        lum = x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114
        x = np.repeat(lum[..., None], 3, axis=2)
    if rng.random() < cfg.blur_p:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 1.0))
    if rng.random() < cfg.noise_p:
        x = x + rng.normal(0, rng.uniform(0.005, 0.03), x.shape)
    return (np.clip(x, 0, 1).astype(np.float32), np.ascontiguousarray(y, np.float32),
            np.ascontiguousarray(v, np.float32))


# --------------------------------------------------------------------------
# faint roots: ROOT_FADE and its cap
# --------------------------------------------------------------------------
def root_fade_k(img: np.ndarray, y: np.ndarray, k: float, scale: float = 1.0) -> np.ndarray:
    """Blend root pixels toward the LOCAL soil colour by ``k`` (v1 cell 6).

    Soft edges (blur 0.7 px x scale); the soil colour is a 6-px x scale
    Gaussian mean with the roots (dilated 5x5) excluded.  Label unchanged.
    """
    m0 = (np.asarray(y) > 0.5)
    m = cv2.GaussianBlur(m0.astype(np.float32), (0, 0), 0.7 * scale)[..., None]
    soil = 1.0 - (cv2.dilate(m0.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0).astype(np.float32)
    f = img.astype(np.float32)
    s = 6.0 * scale
    bg = cv2.GaussianBlur(f * soil[..., None], (0, 0), s) / \
        (cv2.GaussianBlur(soil, (0, 0), s)[..., None] + 1e-3)
    return np.clip(f * (1 - k * m) + bg * (k * m), 0, 255).astype(np.uint8)


def root_fade(img: np.ndarray, y: np.ndarray, rng: np.random.Generator, fade_max: float,
              p: float, scale: float = 1.0) -> np.ndarray:
    """v1's faint-root augmentation: with probability ``p``, fade by k ~ U(0, fade_max).

    Synthetic roots are ~2x more contrasty than real ones and recall is
    lowest on faint roots; the label stays 'root'.  Same draws as v1.
    """
    if fade_max <= 0 or rng.random() > p or not (np.asarray(y) > 0.5).any():
        return img
    k = rng.uniform(0, fade_max)
    return root_fade_k(img, y, k, scale)


def fade_thin_visibility(rgb: np.ndarray, mask: np.ndarray, ks: Sequence[float],
                         scale: float = 1.0, thin_w2e: float = THIN_W2E) -> Dict[str, object]:
    """Thin-root visibility of one frame after fading by each ``k``.

    Returns ``{'len_thin': thin centreline px, 'cn': [contrast_n_thin per k]}``
    measured with ``visibility.thin_regions`` -- the same thin-root measure as
    the render gate and the CUT audit, so the cap compares like with like
    against the real p10 (T_ABS).  NaN where the frame has too few thin pixels.
    """
    from mrstyle import visibility as V
    m = np.asarray(mask) > 0
    w = skeleton_widths_px(m)
    n_thin = int((w <= thin_w2e).sum()) if w.size else 0
    out = {"len_thin": n_thin, "cn": [float("nan")] * len(ks)}
    if n_thin == 0:
        return out
    regs = V.thin_regions(m, thin_w2e=thin_w2e)
    cn = []
    for k in ks:
        im = rgb if k <= 0 else root_fade_k(rgb, m.astype(np.float32), float(k), scale)
        cn.append(float(V.frame_visibility(im, regs)["contrast_n"]))
    out["cn"] = cn
    return out


def thin_share_below(cn: Sequence[float], len_thin: Sequence[float], t_abs: float) -> float:
    """Share of thin-root LENGTH in frames whose thin contrast_n < ``t_abs``
    (unmeasurable frames excluded)."""
    c = np.asarray(cn, np.float64)
    L = np.asarray(len_thin, np.float64)
    ok = np.isfinite(c) & (L > 0)
    tot = float(L[ok].sum())
    return float(L[ok & (c < t_abs)].sum() / tot) if tot > 0 else float("nan")


def select_fade_cap(ks: Sequence[float], shares: Sequence[float], share_max: float = 0.10) -> float:
    """Largest ``k`` whose below-p10 thin share -- and every smaller k's -- is
    at most ``share_max``.

    ROOT_FADE stacks on the renderer's contrast floor and on CUT's own fading,
    so an unknown share of labelled thin roots could fall below anything real.
    The cap keeps at most ~10% of thin length below the real p10, matching the
    real faint tail.  0 when even the unfaded frames exceed it (then fading
    only adds invisible labelled roots).
    """
    order = np.argsort(np.asarray(ks, np.float64))
    cap = 0.0
    for i in order:
        s = float(shares[i])
        if not (np.isfinite(s) and s <= share_max):
            break
        cap = float(ks[i])
    return cap


# --------------------------------------------------------------------------
# histogram matching and the network input
# --------------------------------------------------------------------------
def cdf_reference(images: Sequence[np.ndarray]) -> np.ndarray:
    """Pooled per-channel CDF (3 x 256) of uint8 RGB images (v1: synthetic training frames)."""
    cnt = np.zeros((3, 256))
    for im in images:
        for c in range(3):
            cnt[c] += np.bincount(np.asarray(im)[..., c].ravel(), minlength=256)
    return np.cumsum(cnt, 1) / np.maximum(cnt.sum(1, keepdims=True), 1)


def hist_match(img: np.ndarray, cdf_ref: np.ndarray) -> np.ndarray:
    """Map each channel so its CDF follows ``cdf_ref`` -- uses only this image (v1)."""
    out = np.empty_like(img)
    for c in range(3):
        cnt = np.bincount(img[..., c].ravel(), minlength=256)
        cdf = np.cumsum(cnt) / max(cnt.sum(), 1)
        lut = np.interp(cdf, cdf_ref[c], np.arange(256))
        out[..., c] = np.clip(np.round(lut), 0, 255).astype(np.uint8)[img[..., c]]
    return out


def colour_index(x01: np.ndarray) -> np.ndarray:
    """v1's index ``-1*(b1 + 2*b3 - b2)`` with b1=B, b2=G, b3=R  ->  G - B - 2R."""
    return x01[..., 1] - x01[..., 2] - 2.0 * x01[..., 0]


def in_channels(index_mode: str = "off") -> int:
    if index_mode not in INDEX_MODES:
        raise ValueError("INDEX_MODE must be one of %s" % sorted(INDEX_MODES))
    return INDEX_MODES[index_mode]


def to_input(x01: np.ndarray, index_mode: str = "off", idx_mean: float = 0.0,
             idx_std: float = 1.0) -> np.ndarray:
    """HxWx3 RGB in 0-1 -> CxHxW float32 network input (ImageNet-normalised RGB,
    plus the standardised colour index when ``index_mode`` says so).

    There is deliberately no way to append any other channel: v1's
    ``to_input`` took an XGB map and the external-check cells filled it with
    zeros for PRMI images, a channel the network had never seen empty.
    """
    x = np.asarray(x01, np.float32)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("to_input expects HxWx3 RGB, got %s" % (x.shape,))
    n = in_channels(index_mode)
    rgb = (x - MEAN) / STD
    if index_mode == "off":
        chans = rgb
    else:
        idx = ((colour_index(x) - idx_mean) / idx_std)[..., None]
        chans = np.concatenate([rgb, idx], 2) if index_mode == "extra-channel" else idx
    assert chans.shape[2] == n
    return np.ascontiguousarray(chans.transpose(2, 0, 1)).astype(np.float32)


def index_stats(images01: Sequence[np.ndarray], step: int = 4) -> Tuple[float, float]:
    """Mean and std of the colour index over (subsampled) training images."""
    v = np.concatenate([colour_index(np.asarray(x, np.float32))[::step, ::step].ravel()
                        for x in images01]) if len(images01) else np.zeros(1)
    return float(v.mean()), float(v.std() + 1e-6)


def denormalize(x_chw: np.ndarray) -> np.ndarray:
    """First three input channels back to uint8 RGB (previews)."""
    x = np.asarray(x_chw, np.float32)[:3].transpose(1, 2, 0)
    return np.clip((x * STD + MEAN) * 255, 0, 255).astype(np.uint8)


def total_scale(f: float, aug: float = 1.0, clamp: Tuple[float, float] = SCALE_CLAMP,
                net_scale: float = 1.0) -> float:
    """Network px per source px: ``clamp(f x aug) x net_scale`` (SCALE=2 runs)."""
    return clamp_scale(float(f) * float(aug), clamp) * float(net_scale)


__all__: List[str] = [
    "MEAN", "STD", "THIN_REF_W2E", "THIN_MAX_W", "INDEX_MODES", "parse_pair", "log_uniform",
    "clamp_scale", "mask_label", "trace_label", "resize_mask", "keep_thin", "resize_label",
    "resize_valid", "window_side",
    "scale_crop_pad", "width_fast", "thin_weights", "skel_len_px", "skel_len_mm", "AugConfig",
    "rotate_free", "augment", "root_fade_k", "root_fade", "fade_thin_visibility",
    "thin_share_below", "select_fade_cap", "cdf_reference", "hist_match", "colour_index",
    "in_channels", "to_input", "index_stats", "denormalize", "total_scale"]
