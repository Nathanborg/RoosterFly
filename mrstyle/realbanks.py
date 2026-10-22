"""Real-image banks for the simulator: real soil backgrounds and real root textures.

Two banks, both cut from PRMI (Xu et al. 2022, Dryad 10.5061/dryad.2v6wwpzp4) TRAIN
images only, resampled to the synthetic scale (``realdata.scale_table`` factor f):

* soil bank  ``<dir>/soil/<id>.png`` (640x480) + ``soil.csv``: crops of ROOT-FREE
  images (``has_root`` False with a mask, i.e. annotated empty).  Root-free rather
  than inpainted: PRMI masks under-trace fine roots, so an inpainted frame keeps
  unlabelled real roots -- label noise in exactly the class we care about.
* root bank  ``<dir>/roots/<id>.png`` + ``roots.csv``: straightened real root
  strips.  Each is a skeleton branch of a PRMI mask, sampled every pixel along the
  branch and at ``K`` points across it from edge to edge (``t`` in [-1, 1] of the
  local half-width), so the image is (length, K, 3).  The simulator maps a strip
  along a synthetic root (``mrsim.raster`` texture lookup): the geometry and so
  the labels stay synthetic; only the appearance is real.

Loaded by :func:`load_soil_bank` / :func:`load_root_bank` (cached per process).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

STRIP_K = 16
FRAME_WH = (640, 480)


# --------------------------------------------------------------------------
# soil bank
# --------------------------------------------------------------------------
def trim_borders(rgb: np.ndarray, margin_frac: float = 0.04, dark: float = 0.55) -> np.ndarray:
    """Drop the camera's black bars and vignetted edges: edge rows/columns whose
    mean luma is below ``dark`` x the median row/column luma, then a fixed
    ``margin_frac`` on every side."""
    y = rgb.astype(np.float32) @ np.float32([0.299, 0.587, 0.114])
    rows, cols = y.mean(1), y.mean(0)

    def span(v):
        ok = v >= dark * np.median(v)
        idx = np.nonzero(ok)[0]
        return (int(idx[0]), int(idx[-1]) + 1) if idx.size else (0, len(v))
    r0, r1 = span(rows)
    c0, c1 = span(cols)
    my = int(margin_frac * (r1 - r0))
    mx = int(margin_frac * (c1 - c0))
    return rgb[r0 + my:r1 - my, c0 + mx:c1 - mx]


def _cover_crop(rgb: np.ndarray, wh=FRAME_WH, rng=None, max_up: float = 1.5):
    """A ``wh`` crop; an image smaller than ``wh`` is enlarged to cover it
    (at most ``max_up``, else None)."""
    w, h = wh
    H, W = rgb.shape[:2]
    s = max(w / W, h / H, 1.0)
    if s > max_up:
        return None
    if s > 1.0:
        rgb = cv2.resize(rgb, (int(np.ceil(W * s)), int(np.ceil(H * s))),
                         interpolation=cv2.INTER_CUBIC)
        H, W = rgb.shape[:2]
    rng = rng or np.random.default_rng(0)
    y = int(rng.integers(0, H - h + 1))
    x = int(rng.integers(0, W - w + 1))
    return rgb[y:y + h, x:x + w]


def build_soil_bank(z, index, scale: Dict[str, float], out_dir: str, n_per_subset: int = 40,
                    seed: int = 0, verbose: bool = True):
    """Write ``n_per_subset`` root-free soil frames per PRMI subset (train split,
    balanced over tubes).  Returns the bank table."""
    import pandas as pd
    from mrstyle import realdata as RD
    os.makedirs(os.path.join(out_dir, "soil"), exist_ok=True)
    rng = np.random.default_rng(seed)
    free = index[(index.split == "train") & index.has_mask.astype(bool)
                 & ~index.has_root.astype(bool) & index.has_image.astype(bool)]
    rows = []
    for sub, g in free.groupby("subset"):
        pick = RD.balanced_sample(g, "tube_uid", n_per_subset, seed=seed)
        f = float(scale.get(sub, 1.0))
        k = 0
        for _, r in pick.iterrows():
            rgb, _ = RD.prmi_read(z, r, mask=False)
            rgb = trim_borders(rgb)
            rgb, _ = RD.resample_rgb_mask(rgb, None, f)
            crop = _cover_crop(rgb, rng=rng, max_up=2.2)
            if crop is None or crop.std() < 2.0:
                continue
            sid = "soil_%s_%03d" % (RD.safe_name(sub)[:24], k)
            cv2.imwrite(os.path.join(out_dir, "soil", sid + ".png"), crop[:, :, ::-1])
            rows.append(dict(id=sid, source="prmi", subset=sub, prmi_id=r.prmi_id,
                             tube_uid=r.tube_uid, f=f,
                             luma=float(crop.reshape(-1, 3).mean(0) @ [0.299, 0.587, 0.114])))
            k += 1
        if verbose:
            print("soil bank %-36s %3d frames (f=%.2f)" % (sub, k, f))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out_dir, "soil.csv"), index=False)
    return tab


# --------------------------------------------------------------------------
# root bank
# --------------------------------------------------------------------------
_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeleton_branches(skel: np.ndarray, min_len: int = 30) -> List[np.ndarray]:
    """Branches of a skeleton between ends/junctions as ordered (N, 2) [y, x]
    arrays (junction pixels removed, so branches do not touch)."""
    sk = skel.astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    deg = cv2.filter2D(sk, -1, k, borderType=cv2.BORDER_CONSTANT) - sk
    sk = sk * (deg <= 2)
    n, lab = cv2.connectedComponents(sk, connectivity=8)
    out = []
    for c in range(1, n):
        ys, xs = np.nonzero(lab == c)
        if ys.size < min_len:
            continue
        pts = set(zip(ys.tolist(), xs.tolist()))
        # start from an end (one neighbour), else anywhere (a loop)
        start = None
        for p in pts:
            if sum((p[0] + dy, p[1] + dx) in pts for dy, dx in _NB) == 1:
                start = p
                break
        start = start or next(iter(pts))
        path, seen, cur = [start], {start}, start
        while True:
            nxt = [(cur[0] + dy, cur[1] + dx) for dy, dx in _NB
                   if (cur[0] + dy, cur[1] + dx) in pts and (cur[0] + dy, cur[1] + dx) not in seen]
            if not nxt:
                break
            # prefer 4-neighbours (keeps the walk on the line)
            nxt.sort(key=lambda q: abs(q[0] - cur[0]) + abs(q[1] - cur[1]))
            cur = nxt[0]
            seen.add(cur)
            path.append(cur)
        if len(path) >= min_len:
            out.append(np.asarray(path, np.float64))
    return out


def _smooth_resample(path: np.ndarray, win: int = 9) -> np.ndarray:
    """Moving-average smoothing, then resampling every 1 px of arclength."""
    if len(path) > win:
        k = np.ones(win) / win
        pad = win // 2
        p = np.vstack([np.convolve(np.pad(path[:, i], pad, mode="edge"), k, "valid")
                       for i in range(2)]).T
    else:
        p = path
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 2:
        return p
    t = np.arange(0.0, s[-1], 1.0)
    return np.vstack([np.interp(t, s, p[:, i]) for i in range(2)]).T


def extract_strips(rgb: np.ndarray, mask: np.ndarray, min_w: float = 3.0,
                   min_len_w: float = 5.0, min_len: int = 40, k: int = STRIP_K,
                   edge_pad: float = 1.15, max_strips: int = 6, rng=None):
    """Straightened strips of the roots in one image: list of (strip uint8 (L, k, 3),
    width_px).  ``edge_pad`` > 1 samples a little past the mask edge so the soft
    real edge comes along."""
    from skimage.morphology import skeletonize
    m = (mask > 0).astype(np.uint8)
    if m.sum() < 200:
        return []
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    edt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    skel = skeletonize(m > 0)
    out = []
    br = skeleton_branches(skel, min_len=min_len)
    rng = rng or np.random.default_rng(0)
    rng.shuffle(br)
    img = rgb.astype(np.float32)
    for b in br:
        w = 2.0 * float(np.median(edt[b[:, 0].astype(int), b[:, 1].astype(int)]))
        if w < min_w or len(b) < max(min_len, min_len_w * w):
            continue
        p = _smooth_resample(b)
        # trim the ends (junction / tip distortion)
        cut = int(np.ceil(w))
        if len(p) <= 2 * cut + min_len:
            continue
        p = p[cut:-cut]
        tan = np.gradient(p, axis=0)
        tan /= np.maximum(np.hypot(tan[:, 0], tan[:, 1]), 1e-6)[:, None]
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)      # (dy, dx) rotated 90 degrees
        t = np.linspace(-1.0, 1.0, k) * (0.5 * w * edge_pad)
        yy = (p[:, 0:1] + nrm[:, 0:1] * t[None]).astype(np.float32)
        xx = (p[:, 1:2] + nrm[:, 1:2] * t[None]).astype(np.float32)
        strip = cv2.remap(img, xx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        inside = cv2.remap(m.astype(np.float32), xx[:, k // 4:-k // 4], yy[:, k // 4:-k // 4],
                           cv2.INTER_NEAREST)
        if inside.mean() < 0.9:                         # strip left the root (crossing, bend)
            continue
        out.append((np.clip(strip, 0, 255).astype(np.uint8), w))
        if len(out) >= max_strips:
            break
    return out


def build_root_bank(z, index, scale: Dict[str, float], out_dir: str, n_images_per_subset: int = 60,
                    seed: int = 0, verbose: bool = True):
    """Straightened real-root strips from PRMI train images with roots."""
    import pandas as pd
    from mrstyle import realdata as RD
    os.makedirs(os.path.join(out_dir, "roots"), exist_ok=True)
    rng = np.random.default_rng(seed)
    rooty = index[(index.split == "train") & index.has_mask.astype(bool)
                  & index.has_root.astype(bool) & index.has_image.astype(bool)]
    rows = []
    for sub, g in rooty.groupby("subset"):
        pick = RD.balanced_sample(g, "tube_uid", n_images_per_subset, seed=seed)
        f = float(scale.get(sub, 1.0))
        n0 = len(rows)
        for _, r in pick.iterrows():
            rgb, m = RD.prmi_read(z, r)
            if m is None or not m.any():
                continue
            rgb, m = RD.resample_rgb_mask(rgb, m, f)
            for j, (strip, w) in enumerate(extract_strips(rgb, m, rng=rng)):
                rid = "root_%s_%05d_%d" % (RD.safe_name(sub)[:24], len(rows), j)
                cv2.imwrite(os.path.join(out_dir, "roots", rid + ".png"), strip[:, :, ::-1])
                lum = strip.reshape(-1, 3).astype(np.float64) @ [0.299, 0.587, 0.114]
                rows.append(dict(id=rid, subset=sub, prmi_id=r.prmi_id, tube_uid=r.tube_uid,
                                 width_px=round(w, 2), length_px=strip.shape[0],
                                 luma=float(lum.mean() / 255.0)))
        if verbose:
            print("root bank %-36s %4d strips (f=%.2f)" % (sub, len(rows) - n0, f))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(out_dir, "roots.csv"), index=False)
    return tab


# --------------------------------------------------------------------------
# loading (cached per process)
# --------------------------------------------------------------------------
_CACHE: Dict[Tuple[str, str], object] = {}


def load_soil_bank(bank_dir: str) -> List[str]:
    """Paths of the soil frames (sorted, so a seed picks the same frame anywhere)."""
    key = ("soil", bank_dir)
    if key not in _CACHE:
        d = os.path.join(bank_dir, "soil")
        _CACHE[key] = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png"))
        if not _CACHE[key]:
            raise FileNotFoundError("empty soil bank: " + d)
    return _CACHE[key]


def soil_background(bank_dir: str, rng, wh=FRAME_WH, jitter: float = 0.06, hue: float = 0.0,
                    sat: float = 0.0) -> np.ndarray:
    """A random bank soil frame as float32 RGB in [0, 1], randomly flipped, with a
    brightness/colour jitter (log-sd ``jitter`` per channel and overall), and optionally a hue
    rotation of U(-hue, hue) of the colour circle and a saturation factor exp(N(0, sat)).
    ``rng`` is the caller's own stream (hue/sat draw only when > 0)."""
    paths = load_soil_bank(bank_dir)
    p = paths[int(rng.integers(len(paths)))]
    im = cv2.imread(p, cv2.IMREAD_COLOR)[:, :, ::-1]
    if im.shape[1] != wh[0] or im.shape[0] != wh[1]:
        im = cv2.resize(im, wh, interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        im = im[:, ::-1]
    if rng.random() < 0.5:
        im = im[::-1]
    g = np.exp(rng.normal(0.0, jitter, 3)) * np.exp(rng.normal(0.0, jitter))
    out = np.clip(im.astype(np.float32) / 255.0 * g[None, None].astype(np.float32), 0.0, 1.0)
    if hue > 0 or sat > 0:
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)
        if hue > 0:
            hsv[..., 0] = np.mod(hsv[..., 0] + 360.0 * float(rng.uniform(-hue, hue)), 360.0)
        if sat > 0:
            hsv[..., 1] = np.clip(hsv[..., 1] * float(np.exp(rng.normal(0.0, sat))), 0.0, 1.0)
        out = np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0)
    return out


class RootBank:
    """Root strips as float32 RGB in [0, 1] (each (L, K, 3)), their widths (px)
    and mean luma (0..1)."""

    def __init__(self, strips: List[np.ndarray], widths: np.ndarray, luma: np.ndarray):
        self.strips, self.widths, self.luma = strips, np.asarray(widths, np.float64), \
            np.asarray(luma, np.float64)
        self.logw = np.log(np.maximum(self.widths, 1.0))
        # mean colour of each strip's core (middle half across): the colour of
        # thin roots, which are too narrow to texture
        k = strips[0].shape[1] if strips else 0
        self.mean_rgb = np.array([s[:, k // 4:k - k // 4].reshape(-1, 3).mean(0)
                                  for s in strips]) if strips else np.zeros((0, 3))
        # each strip averaged ALONG the root: its cross profile (bright core, darker edges, colour)
        # -- the texture of roots far wider than the strip (magnifying detail draws bands)
        self.profiles = [s.mean(0).astype(np.float32) for s in strips]

    def __len__(self) -> int:
        return len(self.strips)

    def pick(self, width_px: float, u: float, n_near: int = 8) -> int:
        """A strip of similar width: one of the ``n_near`` nearest in log width."""
        d = np.abs(self.logw - np.log(max(float(width_px), 1.0)))
        near = np.argsort(d)[:max(1, min(n_near, len(d)))]
        return int(near[int(u * len(near)) % len(near)])


def load_root_bank(bank_dir: str) -> RootBank:
    key = ("roots", bank_dir)
    if key not in _CACHE:
        import pandas as pd
        tab = pd.read_csv(os.path.join(bank_dir, "roots.csv"))
        strips = [cv2.imread(os.path.join(bank_dir, "roots", i + ".png"),
                             cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
                  for i in tab.id]
        _CACHE[key] = RootBank(strips, tab.width_px.values, tab.luma.values)
    return _CACHE[key]
