"""Root-following smart brush for the interactive annotator.

A stroke (or a single click) on a root the model missed becomes that root's
mask in four steps:

1. Contrast channel and tone.  Fine roots are pale on the soil in most
   minirhizotron frames and darker in a few.  Multi-scale Hessian ridge
   strength close to the stroke's centre line is compared with the soil
   around it for {grey, blue} x {light, dark}; the best-separated pair is kept
   (dark has to win clearly, or the user forces the tone).  Blue
   separates pale roots from orange/brown soil best.
2. Centreline.  A minimum-cost path through ``1 / ridge`` inside the stroke's
   corridor (the stroke dilated by the brush radius), so a sloppy stroke
   snaps onto the root's crest.
3. Follow.  From each end the crest is tracked in small steps inside a +-24 deg
   cone with heading inertia, preferring steps along the local ridge
   orientation, until the ridge (averaged over the last few steps) fades
   below ``follow_tol`` x the strength under the stroke, the heading turns
   implausibly, the image border or ``max_follow_px`` is reached, or it runs
   into an already segmented root.  A single click follows both ways along
   the ridge orientation averaged around the clicked crest.
4. Width.  At every centreline sample the intensity profile across the root
   (averaged over 5 neighbouring samples) is walked outwards on both sides to
   where it drops to soil + ``edge_frac`` x (crest - soil), the soil level
   being read in windows of growing size.  A real edge stops moving once the
   window reaches past it, while texture keeps pulling a soft edge outwards:
   the first window whose width agrees with the next one is used (the
   smallest when none settles), and the width is then kept consistent along
   the root (no sample beyond 0.6-1.5x its neighbourhood's median).  A thick
   root with a narrow bright core still gets its full diameter; a hair-thin
   root in busy soil stays thin.

Optionally (:class:`ColourCue`) the colour of roots and soil, learnt on the
image from the user's own annotation and from each stroke's crest, is a soft
extra cue: following avoids soil-coloured steps and the width profiles are
nudged towards where the colour turns to soil.

A stroke laid mostly over what is already segmented (``existing``: the
model's mask, earlier traces) re-measures that root under the stroke, inside
the brush, without following: the measured outline is added, so a prediction
narrower than the root gets its full width, and brushing the same place again
gives the same outline -- never a growing blob or the brush disc itself.
Where a wide root is still missing a part, painting over that part adds it.

The result is bounded by construction -- never wider than the edge search
around the traced centreline -- so weak evidence gives a thin line, not the
square crop box the old ridge fallback produced.

Ridge maps are computed once per image (:class:`RidgeCache`, thread-safe, can
be warmed in the background) so a stroke costs only the tracing itself.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SIGMAS = (1.0, 1.4, 2.0, 2.8, 4.0)
_GAMMA = 1.5          # scale normalisation s**gamma: peak response at s = profile std
_EPS = 1e-6
_LIGHT_PRIOR = 1.5   # auto tone: dark must out-score light 1.5x (light roots under offset strokes reach 1.2x)
_FULL_IMAGE_PX = 12_000_000  # cache whole-image maps up to this size (~32 B/px for grey + blue), else per-stroke ROI
_COLOUR_K = 61                # colour cue: local soil colour = median over this window (px)
# Colour cue, tuned on simulated strokes over 74 annotated Panama roots (F1 0.602 -> 0.606 from a stroke's
# own samples, 0.614 with the image's annotated roots; overshoot onto soil down)
_COLOUR_GAIN = ((0.0, 0.7), (0.4, 1.0), (1.0, 1.0))   # P(root | colour) -> x on-root evidence: a soft veto on soil colours
_COLOUR_STEP = 0.3            # following: step score x (1 - this + this x P(root | colour))
_COLOUR_CREST = 2.0           # a stroke's own root samples: within this of its snapped crest (px)
_COLOUR_WIDTH = 0.5           # width profiles: colour adds up to +-0.5 x this x the channel's span
_COLOUR_HALF_PX = 300_000      # colour posterior at half resolution above this box size (it is blurred anyway)
# Outline smoothness, tuned on simulated strokes over PRMI simulated frames (exact masks) and 80 annotated
# Panama frames: wiggle of the centre line -35 %, lumps along the sides -48 %, F1 -0.006
_PATH_SMOOTH = 7              # centreline: moving average over this many 1-px samples ...
_PATH_SMOOTH_W = 2.0          # ... or over this x the root's width when longer (0 = off)
_WIDTH_SIGMA = 3.0            # root edges: Gaussian smoothing of the widths along the root (samples; 0 = off)
_MASK_SIGMA = 0.0             # outline: Gaussian blur of the drawn mask, re-thresholded at 0.5 (0 = off)
_ROUTE_DEV = 1.0              # centre line under a stroke: cost of straying from the drawn line, x (dist / brush)^2
                              # (sloppy strokes, +-4 px: F1 Panama 0.536 -> 0.546, PRMI 0.564 -> 0.567)


@dataclass
class TraceParams:
    brush_radius: int = 10
    follow: bool = True
    follow_tol: float = 0.40      # stop when ridge < tol x ridge under the stroke
    max_follow_px: int = 400      # per end
    max_gap_px: int = 20          # bridge a fade this long if the crest resumes in line
    tone: str = "auto"            # "auto" | "light" | "dark"
    channel: str = ""             # force the traced channel ("grey", "blue", ...): keeps a live preview stable
    edge_frac: float = 0.30       # root edge where the profile is 30% of the way from soil to crest
    contrast_tol: float = 0.35    # following also needs >= 35% of the stroke's brightness above soil


@dataclass
class TraceResult:
    mask: np.ndarray              # HxW uint8 {0, 1}
    centerline: np.ndarray        # Nx2 float (x, y), image px
    tone: str = ""
    channel: str = ""
    followed_px: float = 0.0
    width_px: float = 0.0         # median diameter
    score: float = 0.0            # ridge under the stroke / soil texture (x contrast)
    contrast: float = 0.0         # crest brightness above soil, fraction of the ROI's dynamic range
    note: str = ""
    widths: Optional[np.ndarray] = None   # diameter along ``centerline`` (px)


# ------------------------------------------------------------------------------
# Small geometry / sampling helpers
# ------------------------------------------------------------------------------

def _resample(pts: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """Polyline resampled at ``spacing`` px (keeps a single point as is)."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return pts
    keep = np.r_[True, np.any(np.abs(np.diff(pts, axis=0)) > 1e-9, axis=1)]
    pts = pts[keep]
    if len(pts) < 2:
        return pts
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.r_[0.0, np.cumsum(seg)]
    n = max(2, int(np.ceil(cum[-1] / spacing)) + 1)
    t = np.linspace(0.0, cum[-1], n)
    return np.c_[np.interp(t, cum, pts[:, 0]), np.interp(t, cum, pts[:, 1])]


def _bilinear(img: np.ndarray, xs, ys) -> np.ndarray:
    h, w = img.shape[:2]
    xs = np.minimum(np.maximum(np.asarray(xs, np.float64), 0.0), w - 1.001)
    ys = np.minimum(np.maximum(np.asarray(ys, np.float64), 0.0), h - 1.001)
    x0, y0 = xs.astype(np.int64), ys.astype(np.int64)
    fx, fy = xs - x0, ys - y0
    v00, v01 = img[y0, x0], img[y0, x0 + 1]
    v10, v11 = img[y0 + 1, x0], img[y0 + 1, x0 + 1]
    return (v00 * (1 - fx) + v01 * fx) * (1 - fy) + (v10 * (1 - fx) + v11 * fx) * fy


def _bil1(img: np.ndarray, x: float, y: float) -> float:
    """Scalar :func:`_bilinear` (the per-step sampling of the follow loop)."""
    h, w = img.shape[:2]
    x = min(max(float(x), 0.0), w - 1.001)
    y = min(max(float(y), 0.0), h - 1.001)
    x0, y0 = int(x), int(y)
    fx, fy = x - x0, y - y0
    r0, r1 = img[y0], img[y0 + 1]
    return float((r0[x0] * (1 - fx) + r0[x0 + 1] * fx) * (1 - fy) + (r1[x0] * (1 - fx) + r1[x0 + 1] * fx) * fy)


def _disk(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _running_median(v: np.ndarray, k: int) -> np.ndarray:
    k = int(k) | 1
    if len(v) < k:
        return v
    pad = np.pad(v, k // 2, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, k)
    return np.median(win, axis=1).astype(v.dtype)


def _ang_diff(a: float, b: float) -> float:
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


# ------------------------------------------------------------------------------
# Multi-scale Hessian ridges (both tones from one eigen-decomposition)
# ------------------------------------------------------------------------------

def _ridge_maps(f: np.ndarray, sigmas: Sequence[float] = SIGMAS) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """{"light"|"dark": (strength, along-ridge angle [rad], best scale)}."""
    f = np.ascontiguousarray(f, dtype=np.float32)
    out: Dict[str, list] = {}
    for s in sigmas:
        g = cv2.GaussianBlur(f, (0, 0), s)
        a = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        c = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        b = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        k = 0.25 * (s ** _GAMMA)                        # Sobel gain 4, scale normalisation
        a *= k; b *= k; c *= k
        tmp = cv2.sqrt((a - c) ** 2 + 4.0 * b * b)
        l1 = 0.5 * (a + c + tmp)                       # larger eigenvalue
        l2 = 0.5 * (a + c - tmp)                       # smaller eigenvalue
        ang = 0.5 * cv2.phase(a - c, 2.0 * b)          # eigenvector of l1, [0, pi)
        al1, al2 = np.abs(l1), np.abs(l2)
        # light crest: across = -l2 >> 0, along = l1 ~ 0 ; dark trough: across = l1, along = l2
        light = np.maximum(-l2, 0.0) * (1.0 - np.minimum(al1 / (al2 + _EPS), 1.0))
        dark = np.maximum(l1, 0.0) * (1.0 - np.minimum(al2 / (al1 + _EPS), 1.0))
        for tone, r, th in (("light", light, ang), ("dark", dark, ang + 0.5 * np.pi)):
            if tone not in out:
                out[tone] = [r, th.copy(), np.full(r.shape, s, np.float32)]
            else:
                best, theta, scale = out[tone]
                up = r > best
                np.copyto(best, r, where=up)
                np.copyto(theta, th, where=up)
                scale[up] = s
    return {k: (v[0], v[1], v[2]) for k, v in out.items()}


def image_channels(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    rgbf = rgb.astype(np.float32)
    return {
        "grey": rgbf @ np.float32([0.299, 0.587, 0.114]),
        "blue": np.ascontiguousarray(rgbf[..., 2]),
    }


class RidgeCache:
    """Whole-image ridge maps for one image, computed lazily once (thread-safe).

    Images above ``_FULL_IMAGE_PX`` are not cached: each stroke then computes
    the maps on its own ROI -- as does a stroke made while :meth:`warm` is still
    running (a 6 MP scan takes 7-12 s), instead of waiting for it.
    """

    def __init__(self, channels: Dict[str, np.ndarray]):
        self.channels = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in channels.items()}
        self.shape = next(iter(self.channels.values())).shape[:2]
        self.enabled = self.shape[0] * self.shape[1] <= _FULL_IMAGE_PX
        self._maps: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        self._lock = threading.Lock()
        self._cancelled = False
        self.warming = False

    def ready(self) -> bool:
        return all(k in self._maps for k in self.channels)

    @classmethod
    def from_rgb(cls, rgb: np.ndarray) -> "RidgeCache":
        return cls(image_channels(rgb))

    def maps(self, name: str):
        with self._lock:
            if name not in self._maps:                  # angle and scale at half precision: 4 B/px less per tone
                self._maps[name] = {t: (r, th.astype(np.float16), sc.astype(np.float16))
                                    for t, (r, th, sc) in _ridge_maps(self.channels[name]).items()}
            return self._maps[name]

    def cancel(self):
        """Stop a running :meth:`warm` after its current channel (the image was left)."""
        self._cancelled = True

    def warm(self):
        """Compute every channel now (call from a background thread)."""
        if not self.enabled:
            return
        self.warming = True
        for name in self.channels:
            if self._cancelled:
                return
            self.maps(name)
        try:                                            # first import of skimage.graph is slow
            import skimage.graph  # noqa: F401
        except Exception:
            pass


# ------------------------------------------------------------------------------
# Colour cue: root vs soil colour learnt from the user's strokes on this image
# ------------------------------------------------------------------------------

def colour_features(rgb: np.ndarray) -> np.ndarray:
    """HxWx3 float32: CIELAB (8-bit units) minus its local median -- colour relative to the soil around."""
    lab = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB)
    return np.dstack([lab[..., c].astype(np.float32) - cv2.medianBlur(np.ascontiguousarray(lab[..., c]), _COLOUR_K)
                      for c in range(3)])


def _gauss_loglik(X: np.ndarray, wt: np.ndarray):
    """Weighted Gaussian fit of samples ``X`` (n x 3); returns its log-likelihood function."""
    wt = wt / wt.sum()
    mu = wt @ X
    D = X - mu
    C = (D * wt[:, None]).T @ D + np.eye(X.shape[1])          # +1: floor for JPEG noise / few samples
    Ci = np.linalg.inv(C).astype(np.float32)
    ld = float(np.linalg.slogdet(C)[1])
    mu = mu.astype(np.float32)

    def f(Z):                                                  # Z: n x 3 float32
        Dz = Z - mu
        return -0.5 * (((Dz @ Ci) * Dz).sum(1) + ld)
    return f


class ColourCue:
    """P(root | colour) on one image, from root and soil samples the user gave.

    One Gaussian per class in colour relative to the local soil
    (:func:`colour_features`), so a root's tint is read against its own soil.
    :meth:`set_samples` takes the user's annotation of the image (roots they
    brushed or a reloaded saved mask; soil = the band beside them and what they
    erased).  Each stroke adds its own crest and the soil beside it at trace
    time, weighted as much as all stored samples together, so even the first
    stroke on an image has a colour model.  Colour is a soft cue: brightness
    and the ridge still lead.
    """
    MAX_ROOT, MAX_SOIL = 1500, 3000
    MIN_ROOT, MIN_SOIL = 12, 40

    def __init__(self, rgb: np.ndarray):
        self._rgb = rgb
        self._feat: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self.root = np.zeros((0, 3), np.float32)
        self.soil = np.zeros((0, 3), np.float32)

    @property
    def feat(self) -> np.ndarray:
        with self._lock:
            if self._feat is None:
                self._feat = colour_features(self._rgb)
            return self._feat

    def warm(self):
        self.feat

    def _sample(self, m: Optional[np.ndarray], cap: int) -> np.ndarray:
        if m is None or not m.any():
            return np.zeros((0, 3), np.float32)
        ys, xs = np.nonzero(m)
        if len(ys) > cap:
            k = np.random.default_rng(0).choice(len(ys), cap, replace=False)
            ys, xs = ys[k], xs[k]
        return self.feat[ys, xs]

    def set_samples(self, root_mask: Optional[np.ndarray], soil_mask: Optional[np.ndarray]):
        """Stored samples from HxW bool masks (replaces the previous ones)."""
        self.root = self._sample(root_mask, self.MAX_ROOT)
        self.soil = self._sample(soil_mask, self.MAX_SOIL)

    @property
    def n_samples(self) -> Tuple[int, int]:
        return len(self.root), len(self.soil)

    def posterior(self, box: Tuple[int, int, int, int], root: Optional[np.ndarray] = None,
                  soil: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """P(root | colour) over ``box`` (y0, y1, x0, x1); ``root`` / ``soil`` (bool, box-sized) are this
        stroke's own samples.  None while either class has too few samples."""
        y0, y1, x0, x1 = box
        F = self.feat[y0:y1, x0:x1]
        fits = []
        for stored, own, cap, need in ((self.root, root, self.MAX_ROOT, self.MIN_ROOT),
                                       (self.soil, soil, self.MAX_SOIL, self.MIN_SOIL)):
            own_x = np.zeros((0, 3), np.float32)
            if own is not None and own.any():
                ys, xs = np.nonzero(own)
                if len(ys) > cap:
                    k = np.random.default_rng(1).choice(len(ys), cap, replace=False)
                    ys, xs = ys[k], xs[k]
                own_x = F[ys, xs]
            X = np.vstack([stored, own_x])
            if len(X) < need:
                return None
            if len(stored) and len(own_x):                      # this stroke counts as much as the rest
                wt = np.r_[np.full(len(stored), 0.5 / len(stored)), np.full(len(own_x), 0.5 / len(own_x))]
            else:
                wt = np.ones(len(X))
            fits.append(_gauss_loglik(X.astype(np.float64), wt))
        Fz = F[::2, ::2] if F.shape[0] * F.shape[1] > _COLOUR_HALF_PX else F
        Z = np.ascontiguousarray(Fz.reshape(-1, 3), dtype=np.float32)
        llr = np.clip(fits[0](Z) - fits[1](Z), -8.0, 8.0)
        cp = (1.0 / (1.0 + np.exp(-llr))).astype(np.float32).reshape(Fz.shape[:2])
        if Fz is not F:
            return cv2.resize(cv2.GaussianBlur(cp, (0, 0), 0.5), (F.shape[1], F.shape[0]), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(cp, (0, 0), 1.0)


def _stroke_colour(colour: ColourCue, box: Tuple[int, int, int, int], cl: np.ndarray, s_med: float,
                   busy: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Colour posterior for one stroke: its snapped crest is root, a band beside it (off ``busy``) soil."""
    y0, y1, x0, x1 = box
    h, w = y1 - y0, x1 - x0
    lm = np.zeros((h, w), np.uint8)
    ip = np.round(cl).astype(np.int32).reshape(-1, 1, 2)
    ip[..., 0] = np.clip(ip[..., 0], 0, w - 1)
    ip[..., 1] = np.clip(ip[..., 1], 0, h - 1)
    if len(ip) == 1:
        lm[ip[0, 0, 1], ip[0, 0, 0]] = 1
    else:
        cv2.polylines(lm, [ip], False, 1, thickness=1)
    d = cv2.distanceTransform((lm == 0).astype(np.uint8), cv2.DIST_L2, 3)
    inner = max(6.0, 3.0 * s_med + 2.0)                         # clear of the root's own edge
    soil = (d >= inner) & (d <= inner + 10.0)
    if busy is not None:
        soil &= ~(cv2.dilate(busy[y0:y1, x0:x1].astype(np.uint8), _disk(2)) > 0)
    return colour.posterior(box, d <= _COLOUR_CREST, soil)


def _colour_gain(cp: np.ndarray) -> np.ndarray:
    xp, fp = zip(*_COLOUR_GAIN)
    return np.interp(cp, xp, fp).astype(np.float32)


def _smooth1d(v: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing of a profile along the root (edge-padded)."""
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-x ** 2 / (2.0 * sigma ** 2))
    k /= k.sum()
    return np.convolve(np.pad(np.asarray(v, np.float64), r, mode="edge"), k, mode="valid").astype(np.float32)


def _smooth_ratio(new: np.ndarray, old: np.ndarray, k: int = 31) -> np.ndarray:
    """``new / old`` per sample as a running median over ``k`` samples (one median on short paths), 0.8-1.3x."""
    r = (new / np.maximum(old, _EPS)).astype(np.float32)
    r = _running_median(r, k) if len(r) >= k else np.full_like(r, np.median(r))
    return np.clip(r, 0.8, 1.3)


# ------------------------------------------------------------------------------
# Centreline: snapped path through the stroke corridor, then ridge following
# ------------------------------------------------------------------------------

def _snap(score: np.ndarray, pt: np.ndarray, r: int) -> np.ndarray:
    h, w = score.shape
    x, y = int(round(pt[0])), int(round(pt[1]))
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    win = score[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    win = np.where((xx - x) ** 2 + (yy - y) ** 2 <= r * r, win, -1.0)
    k = int(np.argmax(win))
    return np.array([x0 + k % win.shape[1], y0 + k // win.shape[1]], np.float64)


def _route(cost: np.ndarray, a: np.ndarray, b: np.ndarray,
           box: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """Minimum-cost 8-connected path a -> b (x, y); straight line if skimage is missing.

    ``box`` (y0, y1, x0, x1) limits the search to where the path may go (the stroke corridor): the
    route's set-up cost grows with the array, which dominated strokes on big frames."""
    oy, ox = (box[0], box[2]) if box is not None else (0, 0)
    sub = cost[box[0]:box[1], box[2]:box[3]] if box is not None else cost
    try:
        from skimage.graph import route_through_array
        path, _ = route_through_array(
            sub, (int(a[1]) - oy, int(a[0]) - ox), (int(b[1]) - oy, int(b[0]) - ox), fully_connected=True,
            geometric=True
        )
        return np.array([[c + ox, r + oy] for r, c in path], np.float64)
    except Exception:
        return _resample(np.vstack([a, b]), 1.0)


def _crest_direction(E: np.ndarray, pt: np.ndarray, half_len: float) -> float:
    """Orientation [rad] of the straight segment through ``pt`` with the most ridge along it."""
    angs = np.deg2rad(np.arange(0, 180, 5))
    t = np.r_[-np.arange(half_len, 1.5, -1.0), np.arange(2.0, half_len + 0.5, 1.0)]   # skip the centre
    best, best_a = -1.0, 0.0
    for a in angs:
        xs, ys = pt[0] + t * np.cos(a), pt[1] + t * np.sin(a)
        v = _bilinear(E, xs, ys)
        sc = float(np.mean(np.minimum(v, 1.2))) + 0.5 * float(min(v[t < 0].mean(), v[t > 0].mean()))
        if sc > best:
            best, best_a = sc, float(a)
    return best_a


def _local_orientation(R: np.ndarray, theta: np.ndarray, pt: np.ndarray, r: int) -> float:
    """Ridge orientation around ``pt``: R-weighted mean of doubled angles."""
    h, w = R.shape
    x, y = int(round(pt[0])), int(round(pt[1]))
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    wgt = R[y0:y1, x0:x1].astype(np.float64)
    th = theta[y0:y1, x0:x1].astype(np.float64)
    c, s = float((wgt * np.cos(2 * th)).sum()), float((wgt * np.sin(2 * th)).sum())
    if abs(c) + abs(s) < _EPS:
        return float(theta[y, x])
    return 0.5 * float(np.arctan2(s, c))


def _follow(
    E: np.ndarray,
    on_root: np.ndarray,
    look: np.ndarray,
    theta: np.ndarray,
    p0: np.ndarray,
    d0: np.ndarray,
    step: float,
    max_len: float,
    gap_px: float,
    max_gap_px: float,
    stop_mask: Optional[np.ndarray],
    occupied: np.ndarray,
    col: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], str]:
    """Track a crest from ``p0`` heading ``d0`` until it fades, turns away or meets a root.

    ``E`` (ridge strength, capped so a stronger neighbour does not pull harder than
    an equal one), ``look`` (brightness above soil relative to the stroked root:
    1 = same) and ``col`` (P(root | colour), optional) steer; ``on_root`` >= 1 where
    the ridge and the brightness are still root-like decides when it has faded.
    Returns the tracked points and why tracking stopped.
    """
    h, w = E.shape
    d = d0 / (np.hypot(*d0) + _EPS)
    p = p0.astype(np.float64)
    pts: List[np.ndarray] = []
    evid: List[float] = []
    heads: List[float] = []
    turns = np.deg2rad([-24, -16, -8, 0, 8, 16, 24])
    offs = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    win = max(3, int(np.ceil(gap_px / step)))         # steps averaged for the fade test
    hist = max(6, int(np.ceil(20.0 / step)))          # ~20 px heading memory
    travelled, jumps, reason = 0.0, 0, "max length"

    def inside(q):
        return 0 <= q[0] < w - 1 and 0 <= q[1] < h - 1

    def recentre(q, nd):
        perp = np.array([-nd[1], nd[0]])
        cand = q[None, :] + offs[:, None] * perp[None, :]
        cand[:, 0] = np.clip(cand[:, 0], 0, w - 1.001)
        cand[:, 1] = np.clip(cand[:, 1], 0, h - 1.001)
        ev = _bilinear(E, cand[:, 0], cand[:, 1])
        k = int(np.argmax(ev))
        return cand[k], _bil1(on_root, cand[k, 0], cand[k, 1])

    def aligned(q, nd, tol_deg):
        th = float(theta[int(round(q[1])), int(round(q[0]))])
        return abs(np.cos(th - np.arctan2(nd[1], nd[0]))) >= np.cos(np.deg2rad(tol_deg))

    def resume_after_gap(p, d):
        """First point within ``max_gap_px`` straight ahead where an aligned crest resumes."""
        for t in np.deg2rad([0, -6, 6, -12, 12]):
            c, s = np.cos(t), np.sin(t)
            nd = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
            for L in np.arange(2 * step, max_gap_px + step, step):
                q = p + L * nd
                if not inside(q + 2 * step * nd):
                    break
                ok = True
                for k in range(3):                            # crest must hold for 3 steps
                    qq, e = recentre(q + k * step * nd, nd)
                    if e < 1.0 or not aligned(qq, nd, 25):
                        ok = False
                        break
                if ok:
                    return recentre(q, nd)[0]
        return None

    while travelled < max_len:
        best = None
        for t in turns:
            c, s = np.cos(t), np.sin(t)
            nd = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
            q = p + step * nd
            if not inside(q):
                continue
            mid = p + 0.5 * step * nd
            e = 0.5 * (_bil1(E, q[0], q[1]) + _bil1(E, mid[0], mid[1]))
            th = float(theta[int(round(q[1])), int(round(q[0]))])
            align = abs(np.cos(th - np.arctan2(nd[1], nd[0])))
            lk = _bil1(look, q[0], q[1])
            same = np.exp(-np.log(max(lk, 0.05)) ** 2 / (2 * 0.6 ** 2))     # 1 when it looks like the root
            score = min(e, 1.2) * (0.4 + 0.6 * align) * (1.0 - 0.3 * abs(t) / turns[-1]) * (0.3 + 0.7 * same)
            if col is not None:
                score *= 1.0 - _COLOUR_STEP + _COLOUR_STEP * _bil1(col, q[0], q[1])
            if best is None or score > best[0]:
                best = (score, q, nd)
        if best is None:
            reason = "image border"
            break
        q, e_q = recentre(best[1], best[2])
        qi = (int(round(q[1])), int(round(q[0])))
        if travelled > 3 * step and occupied[qi]:
            reason = "looped back"
            break
        pts.append(q)
        evid.append(e_q)
        travelled += step
        if len(evid) >= win and float(np.mean(np.minimum(evid[-win:], 2.0))) < 1.0:
            # faded: back up to the last on-crest step, then try to bridge a short gap
            on = [i for i, e in enumerate(evid) if e >= 1.0]
            keep = on[-1] + 1 if on else 0
            del pts[keep:], evid[keep:]
            p_last = pts[-1] if pts else p0.astype(np.float64)
            nxt = resume_after_gap(p_last, d) if (jumps < 3 and max_gap_px > 0) else None
            if nxt is None:
                reason = "faded"
                break
            n_fill = max(1, int(np.hypot(*(nxt - p_last)) // step))
            for k in range(1, n_fill + 1):                    # straight bridge over the gap
                pts.append(p_last + (nxt - p_last) * k / n_fill)
                evid.append(1.0)
            travelled += float(np.hypot(*(nxt - p_last)))
            jumps += 1
            p = nxt
            continue
        if stop_mask is not None and stop_mask[qi]:
            reason = "joined root"
            break
        if len(pts) > 4:
            back = pts[-5]
            occupied[int(round(back[1])), int(round(back[0]))] = 1
        mv = q - p
        mv /= np.hypot(*mv) + _EPS
        d = 0.75 * d + 0.25 * mv
        d /= np.hypot(*d) + _EPS
        heads.append(float(np.arctan2(d[1], d[0])))
        if len(heads) > hist and _ang_diff(heads[-1], heads[-1 - hist]) > np.deg2rad(55):
            reason = "bent away"
            break
        p = q
    if reason != "joined root":
        # drop the unsupported tail: everything after the last step that was on the crest
        on = [i for i, e in enumerate(evid) if e >= 1.0]
        pts = pts[: on[-1] + 1] if on else []
    return pts, reason


# ------------------------------------------------------------------------------
# Width: root edges found on intensity profiles across the centreline
# ------------------------------------------------------------------------------

def _smooth_path(cl: np.ndarray, spacing: float = 1.0, k: Optional[int] = None) -> np.ndarray:
    """Centreline without the 8-connected pixel jitter (moving average over ``k`` samples), resampled; ends kept."""
    cl = np.asarray(cl, np.float64).reshape(-1, 2)
    if len(cl) >= 5:
        k = min(int(k or _PATH_SMOOTH) | 1, (len(cl) // 2) * 2 - 1)
        pad = np.pad(cl, ((k // 2, k // 2), (0, 0)), mode="edge")
        ker = np.ones(k) / k
        sm = np.stack([np.convolve(pad[:, j], ker, mode="valid") for j in range(2)], 1)
        sm[0], sm[-1] = cl[0], cl[-1]
        cl = sm
    return _resample(cl, spacing)


def _normals(path: np.ndarray, theta0: Optional[float] = None) -> np.ndarray:
    if len(path) == 1:
        t0 = 0.0 if theta0 is None else float(theta0)
        return np.array([[-np.sin(t0), np.cos(t0)]])
    t = np.gradient(path, axis=0)
    if len(t) >= 5:
        ker = np.ones(5) / 5
        t = np.stack([np.convolve(np.pad(t[:, j], 2, mode="edge"), ker, mode="valid") for j in range(2)], 1)
    t /= np.hypot(t[:, 0], t[:, 1])[:, None] + _EPS
    return np.stack([-t[:, 1], t[:, 0]], 1)


def _fill_nan(v: np.ndarray, default: float) -> np.ndarray:
    good = np.isfinite(v)
    if good.all():
        return v
    if not good.any():
        return np.full_like(v, default)
    idx = np.arange(len(v))
    return np.interp(idx, idx[good], v[good])


_LEVELS = (6.0, 10.0, 16.0, 24.0, 34.0, 48.0, 60.0)     # soil windows tried, px beyond the start


def _mask_extents(mask: np.ndarray, path: np.ndarray, nrm: np.ndarray, reach: float) -> np.ndarray:
    """Per sample and side (n x 2), how far ``mask`` reaches from the centreline (0 where outside)."""
    s = np.arange(0.0, reach + 0.25, 0.5)
    out = np.zeros((len(path), 2))
    mf = mask.astype(np.float32)
    for side, sign in enumerate((1.0, -1.0)):
        xs = path[:, 0, None] + sign * s[None, :] * nrm[:, 0, None]
        ys = path[:, 1, None] + sign * s[None, :] * nrm[:, 1, None]
        inside = _bilinear(mf, xs.ravel(), ys.ravel()).reshape(len(path), len(s)) > 0.5
        e = np.where(inside.all(1), s[-1], s[np.argmax(~inside, axis=1)])
        out[:, side] = np.where(inside[:, 0], e, 0.0)
    return out


def _profile_extents(fs: np.ndarray, path: np.ndarray, nrm: np.ndarray, search_r: float,
                     edge_frac: float, min_contrast: float, span: float,
                     min_width: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Distance from the centreline to the root edge on each side, per path sample.

    The edge is where the profile across the root (averaged over 5 neighbouring
    samples) first drops below soil +
    f x (crest - soil), f = ``edge_frac`` for crisp, contrasted roots rising to
    0.5 for faint, soft ones.  The soil is the low end (20th percentile) of a
    window beyond the start; windows of growing size are tried, and the one
    used is the first whose root width (median along the path) agrees with the
    next window's, else the smallest -- or, given ``min_width`` (the model's
    width where the stroke lies over its prediction), the first window whose
    width reaches it (the largest if none does).
    Sides without contrast mirror the other side, gaps are interpolated, and
    widths are kept within 0.6-1.5x the median of their 31-sample neighbourhood.
    """
    import warnings

    n = len(path)
    levels = [L for L in _LEVELS if L < search_r] + [float(search_r)]
    s = np.arange(0.0, search_r + 0.25, 0.5)
    m = len(s)
    c0 = np.arange(0.0, 1.75, 0.5)                                         # crest: within 1.5 px of the line
    crest = np.full(n, -np.inf)
    profs = []
    for side, sign in enumerate((1.0, -1.0)):
        cx = path[:, 0, None] + sign * c0[None, :] * nrm[:, 0, None]
        cy = path[:, 1, None] + sign * c0[None, :] * nrm[:, 1, None]
        crest = np.maximum(crest, _bilinear(fs, cx.ravel(), cy.ravel()).reshape(n, -1).max(1))
        xs = path[:, 0, None] + sign * s[None, :] * nrm[:, 0, None]
        ys = path[:, 1, None] + sign * s[None, :] * nrm[:, 1, None]
        P = _bilinear(fs, xs.ravel(), ys.ravel()).reshape(n, m).astype(np.float32)
        if n >= 5:
            P = cv2.blur(P, (1, 5), borderType=cv2.BORDER_REPLICATE)
        profs.append(P)
    crest = crest.astype(np.float32)
    if n >= 5:
        crest = cv2.blur(crest.reshape(-1, 1), (1, 5), borderType=cv2.BORDER_REPLICATE).ravel()

    rows = np.arange(n)
    edges = np.full((2, n, len(levels)), np.nan)
    for side, P in enumerate(profs):
        for li, L in enumerate(levels):
            sub = P[:, : min(m, int(round(L / 0.5)) + 1)]
            soil = np.percentile(sub, 20, axis=1)
            con = crest - soil
            frac = np.interp(con / span, [0.12, 0.35], [max(0.5, edge_frac), edge_frac])
            thr = soil + frac * con
            below = sub[:, 1:] < thr[:, None]
            j = np.argmax(below, axis=1) + 1
            v0, v1 = sub[rows, j - 1], sub[rows, j]
            pos = s[j - 1] + 0.5 * np.clip((v0 - thr) / (v0 - v1 + _EPS), 0.0, 1.0)
            ok = below.any(1) & (con >= min_contrast)
            edges[side, ok, li] = pos[ok]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        W = np.nanmedian(edges[0] + edges[1], axis=0)
    valid = np.isfinite(W)
    settled = [k for k in range(len(levels) - 1)
               if valid[k] and valid[k + 1] and W[k + 1] - W[k] <= max(1.0, 0.15 * W[k])]
    if min_width is not None and valid.any():
        reach = np.where(valid & (W >= min_width))[0]
        k = int(reach[0]) if len(reach) else int(np.where(valid)[0][-1])
    elif settled:
        k = settled[0]
    else:
        k = int(np.argmax(valid)) if valid.any() else 0
    a, b = edges[0, :, k], edges[1, :, k]
    a = np.where(np.isfinite(a), a, b)
    b = np.where(np.isfinite(b), b, a)
    a, b = _fill_nan(a, 1.0), _fill_nan(b, 1.0)
    w = a + b                                                              # consistent along the root
    wm = _running_median(w.astype(np.float32), 31)
    scale = np.clip(w, 0.6 * wm, 1.5 * wm) / np.maximum(w, _EPS)
    a = np.clip(_running_median((a * scale).astype(np.float32), 7), 0.75, search_r)
    b = np.clip(_running_median((b * scale).astype(np.float32), 7), 0.75, search_r)
    return a, b


def _strip_mask(shape: Tuple[int, int], path: np.ndarray, nrm: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Root outline: quads between consecutive cross-sections, round ends, the centre line."""
    m = np.zeros(shape, np.uint8)
    left = path + a[:, None] * nrm
    right = path - b[:, None] * nrm
    sh = 16.0                                                               # 4-bit sub-pixel drawing
    for i in range(len(path) - 1):
        quad = np.array([left[i], left[i + 1], right[i + 1], right[i]])
        cv2.fillPoly(m, [np.round(quad * sh).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    for i in {0, len(path) - 1}:
        c = path[i] + 0.5 * (a[i] - b[i]) * nrm[i]
        cv2.circle(m, (int(round(c[0] * sh)), int(round(c[1] * sh))), int(round(0.5 * (a[i] + b[i]) * sh)),
                   1, -1, shift=4)
    if len(path) > 1:
        cv2.polylines(m, [np.round(path * sh).astype(np.int32).reshape(-1, 1, 2)], False, 1, 1, shift=4)
    return m


def _path_contrast(f: np.ndarray, path: np.ndarray, k_bg: int) -> Tuple[float, float]:
    """(brightness of the path above the local median soil, as a fraction of the span; span)."""
    lo, hi = _span(f)
    span = max(hi - lo, _EPS)
    # the soil level is only needed on the path: filter the box the median window reaches from it
    h, w = f.shape
    m = k_bg // 2 + 3
    bx0, by0 = max(0, int(path[:, 0].min()) - m), max(0, int(path[:, 1].min()) - m)
    bx1, by1 = min(w, int(np.ceil(path[:, 0].max())) + m + 1), min(h, int(np.ceil(path[:, 1].max())) + m + 1)
    fb = f[by0:by1, bx0:bx1]
    u8 = np.clip((fb - lo) * (255.0 / span), 0, 255).astype(np.uint8)
    bg = cv2.medianBlur(u8, k_bg).astype(np.float32) * (span / 255.0) + lo
    con = float(np.median(_bilinear(cv2.dilate(fb - bg, np.ones((3, 3), np.uint8)), path[:, 0] - bx0, path[:, 1] - by0)))
    return con / span, span


def _span(f: np.ndarray) -> Tuple[float, float]:
    """1st / 99th percentile of a map (every 2nd pixel of a large one: same values, 4x faster)."""
    sub = f[::2, ::2] if f.size > 400_000 else f
    lo, hi = np.percentile(sub, [1, 99])
    return float(lo), float(hi)


def _width_channel(fs, span, ch_name, tone, chans, width_channels, box, path, k_bg) -> Tuple[np.ndarray, float]:
    """Channel to measure the width on: grey when it shows the root (JPEG stores colour, so blue,
    at half resolution), else the most contrasted image channel; the traced map only without any."""
    y0, y1, x0, x1 = box
    cands: Dict[str, np.ndarray] = {}
    for k in ("grey", "blue"):
        if k in chans:
            cands[k] = chans[k]
        elif width_channels is not None and k in width_channels:
            cands[k] = np.ascontiguousarray(width_channels[k][y0:y1, x0:x1], dtype=np.float32)
    if not cands:
        return fs, span
    best: Dict[str, Tuple[float, np.ndarray, float]] = {}
    for k, v in cands.items():
        signs = (1.0 if tone == "light" else -1.0,) if k in chans else (1.0, -1.0)   # image tone unknown on prob
        for sg in signs:
            g = cv2.GaussianBlur(sg * v, (0, 0), 0.8)
            rel, sp = _path_contrast(g, path, k_bg)
            if k not in best or rel > best[k][0]:
                best[k] = (rel, g, sp)
    top = max(rel for rel, _, _ in best.values())
    if top <= 0:
        return fs, span
    pick = "grey" if "grey" in best and best["grey"][0] >= 0.6 * top else max(best, key=lambda k: best[k][0])
    return best[pick][1], best[pick][2]


# ------------------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------------------

def trace_root(
    rgb: np.ndarray,
    stroke_pts: Sequence[Tuple[float, float]],
    params: TraceParams,
    stop_mask: Optional[np.ndarray] = None,
    cache: Optional[RidgeCache] = None,
    footprint: Optional[np.ndarray] = None,
    existing: Optional[np.ndarray] = None,
    prior: Optional[np.ndarray] = None,
    width_channels: Optional[Dict[str, np.ndarray]] = None,
    preview: bool = False,
    colour: Optional[ColourCue] = None,
) -> TraceResult:
    """Trace the root under a stroke.  ``rgb`` uint8 HxWx3, points (x, y) in image px.

    ``stop_mask`` (HxW bool/uint8): already segmented roots; following stops on
    reaching them.  ``cache`` supplies precomputed ridge maps and the channels
    to trace on (e.g. a cache built on the model probability map for a
    probability-only brush); without it the image channels are used.

    ``existing`` (HxW): what is segmented already.  A stroke laid mostly over it
    re-measures the root there, bounded by ``footprint`` (the stroke's brush
    area, HxW), and does not follow.  ``prior`` (HxW, the model's mask): where
    the stroke lies over it, the root is measured to the first edge at least
    as wide as the prediction (a fixed mask, so repeating a stroke gives the
    same outline).  ``preview`` is the cheap version for drawing the
    snapped stroke live while the mouse is down: small ROI, no following, no
    trimming.

    ``width_channels`` (the image's grey/blue, HxW) measure the width when the
    cache traces another map (the model probability): the model shows where a
    root runs, the image how wide it is.

    ``colour`` (:class:`ColourCue` of this image) adds P(root | colour) as a
    soft cue: following prefers root-coloured steps and fades sooner on
    soil-coloured ones (the ridge/brightness evidence x 0.6-1.2), and the width
    profiles are nudged towards where the colour turns from root to soil.
    """
    H, W = rgb.shape[:2]
    pts = np.asarray(stroke_pts, np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return TraceResult(np.zeros((H, W), np.uint8), pts)
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
    r = max(2, int(params.brush_radius))
    follow = bool(params.follow) and not preview

    # ROI: stroke bbox + room for the edge search and the follow
    margin = 4 * r + 24 + (0 if preview else 24) + (int(params.max_follow_px) if follow else 0)
    x0 = int(max(0, np.floor(pts[:, 0].min()) - margin))
    x1 = int(min(W, np.ceil(pts[:, 0].max()) + margin + 1))
    y0 = int(max(0, np.floor(pts[:, 1].min()) - margin))
    y1 = int(min(H, np.ceil(pts[:, 1].max()) + margin + 1))
    off = np.array([x0, y0], np.float64)
    sp = _resample(pts - off, 2.0)
    h, w = y1 - y0, x1 - x0

    if cache is None:
        cache = RidgeCache.from_rgb(rgb)
    if cache.enabled and (cache.ready() or not cache.warming):   # still warming: this stroke's ROI, no wait
        chans = {k: v[y0:y1, x0:x1] for k, v in cache.channels.items()}
        maps = {k: {t: tuple(m[y0:y1, x0:x1] for m in v) for t, v in cache.maps(k).items()} for k in chans}
    else:
        chans = {k: np.ascontiguousarray(v[y0:y1, x0:x1]) for k, v in cache.channels.items()}
        maps = {k: _ridge_maps(v) for k, v in chans.items()}
    tones = ("light", "dark") if params.tone == "auto" else (params.tone,)

    # Stroke corridor, closeness to the stroke's centre line, soil ring around it
    line = np.zeros((h, w), np.uint8)
    ip = np.round(sp).astype(np.int32).reshape(-1, 1, 2)
    if len(sp) == 1:
        line[ip[0, 0, 1], ip[0, 0, 0]] = 1
    else:
        cv2.polylines(line, [ip], False, 1, thickness=1)
    d_line = cv2.distanceTransform((line == 0).astype(np.uint8), cv2.DIST_L2, 3)
    corridor = d_line <= r
    near = np.exp(-(d_line ** 2) / (2.0 * float(r) ** 2)).astype(np.float32)
    ring = (d_line > r) & (d_line <= 4 * r)
    sx = np.clip(np.round(sp[:, 0]).astype(np.int64), 0, w - 1)
    sy = np.clip(np.round(sp[:, 1]).astype(np.int64), 0, h - 1)

    # 1. channel x tone with the best stroke-vs-soil ridge contrast (scored around the stroke only)
    by0, bx0 = max(0, int(sy.min()) - 4 * r - 1), max(0, int(sx.min()) - 4 * r - 1)
    win = (slice(by0, int(sy.max()) + 4 * r + 2), slice(bx0, int(sx.max()) + 4 * r + 2))
    ring_w, near_w = ring[win], near[win]
    best = None
    names = [params.channel] if params.channel in chans else list(chans)
    for name in names:
        for tone in tones:
            R, th, sc = maps[name][tone]
            Rw = R[win]
            under = float(np.median(cv2.dilate(Rw * near_w, _disk(r))[sy - by0, sx - bx0]))
            soil = float(np.percentile(Rw[ring_w], 90)) if ring_w.any() else float(np.percentile(Rw, 90))
            score = under / (soil + _EPS)
            rank = score * (_LIGHT_PRIOR if (tone == "light" and params.tone == "auto") else 1.0)
            if best is None or rank > best[0]:
                best = (rank, score, name, tone, R, th, sc, soil)
    _, score, ch_name, tone, R, theta, scale, soil = best
    f_root = chans[ch_name] if tone == "light" else -chans[ch_name]   # root = high

    # 2. centreline through the corridor
    ref0 = float(np.median(cv2.dilate(R[win] * near[win], _disk(r))[sy - by0, sx - bx0]))
    if ref0 <= _EPS:
        ref0 = float(R.max()) + _EPS
    E0 = np.clip(R / ref0, 0.0, 1.5)
    is_click = float(np.hypot(*(sp - sp[0]).T).max()) < max(3.0, 0.5 * r)
    if is_click:                                              # a click is usually placed on the root itself
        near = np.exp(-(d_line ** 2) / (2.0 * max(2.0, 0.5 * r) ** 2)).astype(np.float32)
    pull = np.where(corridor, E0 * (0.5 + 0.5 * near), -1.0)       # crest, preferably near the stroke
    # where texture is busy and no crest stands out, the drawn (stabilised) line is the best guide: straying
    # from it costs on its own, so the centre line does not snake from bump to bump inside the brush
    dev = 0.0 if is_click else _ROUTE_DEV * (d_line / float(r)) ** 2
    cost = np.where(corridor, 1.0 / (0.05 + E0 * (0.5 + 0.5 * near)) + dev, 1e4).astype(np.float64)
    a = _snap(pull, sp[0], r)
    if is_click:
        cl = a[None, :]
    else:
        b = _snap(pull, sp[-1], r)
        way = [a]
        n_way = int(np.hypot(*(sp[-1] - sp[0])) // 60)       # waypoints keep long curved strokes on track
        for k in range(1, n_way + 1):
            idx = int(round(k * (len(sp) - 1) / (n_way + 1)))
            way.append(_snap(pull, sp[idx], max(2, r // 2)))
        way.append(b)
        cy, cx = np.nonzero(corridor)
        cbox = (int(cy.min()), int(cy.max()) + 1, int(cx.min()), int(cx.max()) + 1)
        segs = [_route(cost, way[i], way[i + 1], cbox) for i in range(len(way) - 1)]
        cl = np.vstack([segs[0]] + [s_[1:] for s_ in segs[1:]])

    # Strength on the snapped crest becomes the unit of evidence
    cs = _bilinear(R, cl[:, 0], cl[:, 1])
    ref = float(np.median(cs)) if len(cs) else ref0
    if ref <= 0.05 * ref0:
        ref = ref0
    E = np.clip(R / ref, 0.0, 2.0)
    # In busy soil the crest must also stand out from the texture around the stroke
    stop_level = max(float(params.follow_tol), min(0.85 * soil / ref, 0.75))

    s_med = float(np.median(_bilinear(scale, cl[:, 0], cl[:, 1])))
    # the brush says how big roots are here; >= 24 px so a root's edges can be seen to settle
    search_r = float(np.clip(max(2.0 * r, 24.0, 4.0 * s_med + 4.0), 8.0, 60.0))

    # Brightness of the crest above the local soil, relative to that under the stroke
    fs = cv2.GaussianBlur(f_root, (0, 0), 0.8)
    lo, hi = _span(fs)
    span = max(hi - lo, _EPS)
    f_u8 = np.clip((fs - lo) * (255.0 / span), 0, 255).astype(np.uint8)
    k_bg = int(np.clip(2 * search_r + 1, 15, 255)) | 1
    soil_lvl = cv2.medianBlur(f_u8, k_bg).astype(np.float32) * (span / 255.0) + lo
    above = fs - soil_lvl
    min_con = 0.03 * span
    ref_con = float(np.median(_bilinear(cv2.dilate(above, np.ones((3, 3), np.uint8)), cl[:, 0], cl[:, 1])))
    if ref_con > min_con:
        C = above / ref_con
    else:                                                     # no photometric contrast to hold on to
        C = np.ones_like(above)
    # root-like: ridge kept, not much fainter -- and not several times brighter (another structure)
    on_root = np.minimum(np.minimum(E / stop_level, C / float(params.contrast_tol)),
                         3.5 / np.maximum(C, 1e-3)).astype(np.float32)
    cp = None
    if colour is not None:                                    # ... and, softly, root-coloured
        cp = _stroke_colour(colour, (y0, y1, x0, x1), cl, s_med, existing if existing is not None else stop_mask)
        if cp is not None:
            on_root *= _colour_gain(cp)

    # Mostly over what is segmented already: widen it rather than trace a new root
    ex = existing[y0:y1, x0:x1].astype(bool) if existing is not None else None
    widen = False
    if ex is not None and ex.any():
        cx_ = np.clip(np.round(cl[:, 0]).astype(int), 0, w - 1)
        cy_ = np.clip(np.round(cl[:, 1]).astype(int), 0, h - 1)
        widen = float(ex[cy_, cx_].mean()) >= 0.6
    follow = follow and not widen

    # Trim stroke ends that ran past the root onto soil
    if len(cl) > 4 and not widen and not preview:
        ce = _bilinear(on_root, cl[:, 0], cl[:, 1])
        good = np.where(ce >= 1.0)[0]
        if len(good) >= 2:
            cl = cl[good[0]: good[-1] + 1]

    # 3. follow the crest past both ends
    followed, reasons = 0.0, []
    theta0 = _crest_direction(E, cl[0], max(8.0, 4.0 * s_med)) if len(cl) == 1 else None
    if follow:
        sm = stop_mask[y0:y1, x0:x1].astype(bool) if stop_mask is not None else None
        occupied = np.zeros((h, w), np.uint8)
        dense = _resample(cl, 1.0)
        occupied[np.clip(np.round(dense[:, 1]).astype(int), 0, h - 1),
                 np.clip(np.round(dense[:, 0]).astype(int), 0, w - 1)] = 1
        step = float(np.clip(1.0 + s_med, 1.5, 3.0))
        gap = max(6.0, 3.0 * (2.0 * s_med + 1.0))
        if len(cl) == 1:
            u = np.array([np.cos(theta0), np.sin(theta0)])
            heads = [(cl[0], u), (cl[0], -u)]
        else:
            k = min(len(cl) - 1, 8)
            heads = [(cl[-1], cl[-1] - cl[-1 - k]), (cl[0], cl[0] - cl[k])]
        ext = []
        for p0, d0 in heads:
            if np.hypot(*d0) < _EPS:
                ext.append([])
                continue
            e, why = _follow(E, on_root, C, theta, p0, d0, step, float(params.max_follow_px), gap,
                             float(params.max_gap_px), sm, occupied, cp)
            ext.append(e)
            reasons.append(why)
            followed += float(np.hypot(*np.diff(np.vstack([p0] + e), axis=0).T).sum()) if e else 0.0
        fwd, bwd = ext
        parts = []
        if bwd:
            parts.append(np.array(bwd[::-1]))
        parts.append(cl)
        if fwd:
            parts.append(np.array(fwd))
        cl = np.vstack(parts)

    # 4. width from the profiles across the root
    edge_frac = float(params.edge_frac)
    path = _smooth_path(cl, 1.0)
    if len(path) == 1:                        # a click without following: a short piece along the root, not a disc
        u = np.array([np.cos(theta0), np.sin(theta0)])
        half = max(3.0, 0.6 * r)
        path = _resample(np.vstack([path[0] - half * u, path[0] + half * u]), 1.0)
        path[:, 0] = np.clip(path[:, 0], 0, w - 1)
        path[:, 1] = np.clip(path[:, 1], 0, h - 1)
    nrm = _normals(path, theta0)
    fw, span_w = _width_channel(fs, span, ch_name, tone, chans, width_channels, (y0, y1, x0, x1), path, k_bg)

    def measure(path, nrm):
        """Root edges on both sides of ``path`` (and the prediction's width as a floor when widening)."""
        min_width = None
        if prior is not None and widen:
            pe = _mask_extents(prior[y0:y1, x0:x1], path, nrm, search_r).sum(1)
            if (pe > 0).mean() >= 0.5:
                min_width = 0.9 * float(np.median(pe[pe > 0]))
        a, b = _profile_extents(fw, path, nrm, search_r, edge_frac, 0.03 * span_w, span_w, min_width=min_width)
        if cp is not None:
            # the edges where root and soil also differ in colour; applied as a ratio smoothed along the
            # root, so colour noise in busy soil changes the width gently instead of adding lumps
            ac, bc = _profile_extents(fw + (_COLOUR_WIDTH * span_w) * (cp - 0.5), path, nrm, search_r, edge_frac,
                                      0.03 * span_w, span_w, min_width=min_width)
            a = np.clip(a * _smooth_ratio(ac, a), 0.75, search_r)
            b = np.clip(b * _smooth_ratio(bc, b), 0.75, search_r)
        return a, b

    a, b = measure(path, nrm)
    if _PATH_SMOOTH_W > 0 and len(path) > 15:
        # a wide root's crest wanders inside it (texture ridges): smooth its centre line over a stretch
        # proportional to its width, then measure again -- no hooks, loops or wobbly sides
        k = int(np.clip(round(_PATH_SMOOTH_W * float(np.median(a + b))), 7, 41)) | 1
        if k > int(_PATH_SMOOTH):
            path = _smooth_path(cl, 1.0, k)
            nrm = _normals(path, theta0)
            a, b = measure(path, nrm)
    if _WIDTH_SIGMA > 0 and len(a) >= 5:                     # edges that run smoothly along the root
        a, b = _smooth1d(a, _WIDTH_SIGMA), _smooth1d(b, _WIDTH_SIGMA)
    sub = _strip_mask((h, w), path, nrm, a, b)
    if _MASK_SIGMA > 0:                                      # no pixel stair-steps on the outline
        sm = cv2.GaussianBlur(sub.astype(np.float32), (0, 0), _MASK_SIGMA) > 0.5
        line = np.zeros_like(sub)
        cv2.polylines(line, [np.round(path * 16).astype(np.int32).reshape(-1, 1, 2)], False, 1, 1, shift=4)
        sub = (sm | (line > 0)).astype(np.uint8)              # a hair-thin root keeps its centre line
    if widen and footprint is not None:                      # re-measuring a segment stays inside the brush
        fp = cv2.dilate((footprint[y0:y1, x0:x1] > 0).astype(np.uint8), _disk(2))
        sub = (sub.astype(bool) & (fp > 0)).astype(np.uint8)
    out = np.zeros((H, W), np.uint8)
    out[y0:y1, x0:x1] = sub
    width = float(np.median(a + b))
    note = (f"{tone} root on {ch_name} (contrast x{score:.1f}), width ~{width:.1f} px"
            + (f", followed +{followed:.0f} px (ends: {' / '.join(reasons)})" if follow else "")
            + (", re-measured over the existing segment" if widen else "")
            + (", colour cue" if cp is not None else ""))
    return TraceResult(out, path + off, tone, ch_name, followed, width, score, ref_con / span, note, a + b)


__all__ = ["SIGMAS", "TraceParams", "TraceResult", "RidgeCache", "ColourCue", "colour_features", "image_channels",
           "trace_root"]
