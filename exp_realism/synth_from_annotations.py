"""Synthetic training frames that look like the user's annotated roots, on the user's own soil.

The annotator's saved masks say where the roots of a site are and what they look
like.  This turns them into extra training data with the v4 realism simulator
(``exp_realism.engine.RealismSimulator.render_v4_frame``):

* **Soil**: each annotated frame with its annotated roots inpainted away.  Those
  pixels are *ignored* in training (zero weight), so imperfect inpainting -- or a
  root the annotator missed -- teaches nothing; only simulated roots are labelled.
* **Roots like the user's** (:func:`measure_annotations` then :meth:`SynthGenerator.fit`):
  - width: the simulator's diameter scale is chosen by test renders so the widths at
    skeleton pixels (length-weighted, ``2 x EDT``) match the annotated ones;
  - colour: every annotated root's colour relative to the soil ring around it becomes
    a palette entry (``RealismConfig.root_color_deltas``); a gain fitted by test
    renders makes up for translucency, burial and blur softening the drawn colour;
  - amount: each frame draws a target root length from the annotated frames' lengths;
    the simulator keeps connected root subtrees (``root_keep``) of a camera frame until
    the drawn centre-line length is within 25 % of it -- so a frame also costs only as
    much compositing as the user's frames have root.

Only frames given to it are used, so the fine-tune passes its training windows only:
held-out windows never leak into training through their soil.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SIM_WH = (640, 480)                  # the simulator's camera (cid_ci600_640, 11.81 px/mm)
_LUMA = np.array([0.299, 0.587, 0.114], np.float32)
_DISK = {r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)) for r in (1, 2, 3, 5)}
DIAM_GRID = (0.5, 0.7, 1.0, 1.4, 1.9, 2.5, 3.3)
DIAM_MIN_FRACS = (0.0, 0.5, 0.8)        # x the annotated p10 width: floor for the thinnest roots
GAIN_GRID = (1.0, 1.4, 1.9, 2.6)
N_STANDS = 3


# ==============================================================================
# Measuring the annotations
# ==============================================================================

def component_color_deltas(rgb_u8: np.ndarray, mask: np.ndarray, min_area: int = 6) -> List[List[float]]:
    """Per root (connected component): median root colour minus median of the soil ring
    2-5 px around it (other roots excluded), RGB in [-1, 1]."""
    m = mask.astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return []
    rgb = rgb_u8.astype(np.float32) / 255.0
    near_roots = cv2.dilate(m, _DISK[2]) > 0
    H, W = m.shape
    out = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area < min_area:
            continue
        x0, y0, x1, y1 = max(0, x - 8), max(0, y - 8), min(W, x + w + 8), min(H, y + h + 8)
        comp = (lbl[y0:y1, x0:x1] == i).astype(np.uint8)
        core = cv2.erode(comp, np.ones((3, 3), np.uint8))
        core = core if core.sum() >= 3 else comp
        ring = (cv2.dilate(comp, _DISK[5]) > 0) & ~near_roots[y0:y1, x0:x1]
        if ring.sum() < 10:
            continue
        sub = rgb[y0:y1, x0:x1]
        d = np.median(sub[core > 0], axis=0) - np.median(sub[ring], axis=0)
        out.append([round(float(v), 4) for v in d])
    return out


def measure_annotations(frames: Sequence[Tuple[np.ndarray, np.ndarray]], px_per_mm: float = 11.81) -> Dict:
    """Root statistics of annotated frames [(rgb uint8, mask bool)]: length per frame, widths,
    piece lengths and colour relative to the soil (all in px; mm for reading)."""
    from skimage.morphology import skeletonize
    from mrsim.traits import skeleton_length_px, skeleton_widths_px
    from mrstyle.realbanks import skeleton_branches

    lengths, widths, pieces, deltas, empty = [], [], [], [], 0
    for rgb, mask in frames:
        m = mask.astype(bool)
        if m.sum() < 6:
            empty += 1
            continue
        sk = skeletonize(m)
        lengths.append(float(skeleton_length_px(m, skel=sk)))
        widths.append(skeleton_widths_px(m, skel=sk))
        pieces.extend(len(b) for b in skeleton_branches(sk, min_len=5))
        deltas.extend(component_color_deltas(rgb, m))
    w = np.concatenate(widths) if widths else np.zeros(0)
    q = (10, 25, 50, 75, 90)

    def quant(a):
        return {f"p{p}": round(float(np.percentile(a, p)), 2) for p in q} if len(a) else {}

    dl = np.array([np.dot(d, _LUMA) for d in deltas]) if deltas else np.zeros(0)
    return {
        "n_frames": len(frames),
        "n_with_roots": len(lengths),
        "empty_frac": round(empty / max(1, len(frames)), 3),
        "px_per_mm": px_per_mm,
        "length_px": [round(v, 1) for v in lengths if v >= 10.0],      # specks have no usable length
        "length_mm_q": quant(np.array(lengths) / px_per_mm),
        "width_px_q": quant(w),
        "width_mm_q": quant(w / px_per_mm),
        "piece_len_mm_q": quant(np.array(pieces) / px_per_mm),
        "color_deltas": deltas,
        "color_dluma_q": quant(dl),
        "n_roots": len(deltas),
    }


def inpaint_roots(rgb_u8: np.ndarray, mask: np.ndarray, grow_px: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """(frame with the annotated roots inpainted away, ignore map = those pixels grown by ``grow_px``)."""
    m = cv2.dilate(mask.astype(np.uint8), _DISK[min(grow_px, 5)] if grow_px in _DISK else np.ones((2 * grow_px + 1,) * 2, np.uint8))
    if not m.any():
        return rgb_u8.copy(), np.zeros(mask.shape, bool)
    return cv2.inpaint(rgb_u8, m * 255, 5, cv2.INPAINT_TELEA), m > 0


def default_base_config(project_root: str, image_folder: str = "") -> Dict:
    """Starting simulator settings: a tuned config saved for this folder, else the dataset whose
    name appears in the folder path (e.g. datasets/panama for '.../Panama/...'), else defaults."""
    cand = [os.path.join(image_folder, "_annotations", "_sim", "realism_config.json")] if image_folder else []
    ds = os.path.join(project_root, "datasets")
    low = image_folder.replace("\\", "/").lower()
    if os.path.isdir(ds):
        for d in sorted(os.listdir(ds)):
            if d.lower() and f"/{d.lower()}/" in f"{low}/":
                cand.append(os.path.join(ds, d, "realism_config.json"))
    for p in cand:
        if os.path.isfile(p):
            try:
                with open(p, "r") as fh:
                    return json.load(fh)
            except (OSError, ValueError):
                pass
    return {}


# ==============================================================================
# Generator
# ==============================================================================

class SynthGenerator:
    """Renders simulated roots matched to the annotations onto the annotated frames' own soil."""

    def __init__(self, project_root: str, frames: Sequence[Tuple[str, np.ndarray, np.ndarray]],
                 base_cfg: Optional[Dict] = None, stats: Optional[Dict] = None, seed: int = 2026):
        from exp_realism.engine import RealismSimulator
        from exp_realism.realism_params import RealismConfig

        self.sim = RealismSimulator(repo_root=project_root, lazy_root_bank=True)
        self.seed = int(seed)
        self.stats = stats or measure_annotations([(rgb, m) for _, rgb, m in frames])
        self.backgrounds = []
        for stem, rgb, m in frames:
            soil, ignore = inpaint_roots(rgb, m)
            shape = soil.shape[:2]
            if (shape[1], shape[0]) != SIM_WH:
                soil = cv2.resize(soil, SIM_WH, interpolation=cv2.INTER_AREA)
                ignore = cv2.resize(ignore.astype(np.uint8), SIM_WH, interpolation=cv2.INTER_NEAREST) > 0
            self.backgrounds.append((stem, soil.astype(np.float32) / 255.0, ignore, shape))
        cfg = RealismConfig.from_dict(base_cfg or {})
        if not cfg.root_color_deltas and self.stats.get("color_deltas"):
            rng = np.random.default_rng(self.seed)
            d = list(self.stats["color_deltas"])
            if len(d) > 300:
                d = [d[i] for i in rng.choice(len(d), 300, replace=False)]
            cfg = cfg.copy(root_color_deltas=d)
        self.cfg = cfg.copy(seed=self.seed)
        self.fit_report: Dict = {}

    # -- one simulated frame ------------------------------------------------------------------
    def _background(self, i_bg: int, flip: int):
        stem, soil, ignore, shape = self.backgrounds[i_bg % len(self.backgrounds)]
        if flip & 1:
            soil, ignore = soil[:, ::-1], ignore[:, ::-1]
        if flip & 2:
            soil, ignore = soil[::-1], ignore[::-1]
        return np.ascontiguousarray(soil), np.ascontiguousarray(ignore)

    def _ribbons(self, cfg, stand: int, frame: int, keep: float):
        from mrsim.tube import VisibilityModel
        from exp_realism.spline_roots import extract_smooth_ribbons
        system, _, proj, tube, cam, specs = self.sim._get_or_grow_stand(cfg, stand_idx=stand)
        spec = specs[frame % len(specs)]
        rng = np.random.default_rng([int(cfg.seed), int(stand), int(frame), 44])
        ribbons, _, _ = extract_smooth_ribbons(system, proj, cam, spec, tube, VisibilityModel(),
                                               cfg.copy(root_keep=float(keep)), rng)
        return ribbons

    @staticmethod
    def _arc(ribbons) -> float:
        return float(sum(r.arc_s[-1] for r in ribbons)) if ribbons else 0.0

    def _ribbons_for_length(self, cfg, stand: int, frame: int, target_px: float):
        """Connected root subtrees of one camera frame adding up to ~``target_px`` of centre line."""
        target_px = max(float(target_px), 20.0)
        full = self._ribbons(cfg, stand, frame, 1.0)
        total = self._arc(full)
        if total < 0.6 * target_px:
            return None                                       # too little root in this view
        keep = float(np.clip(target_px / max(total, 1.0), 0.002, 1.0))
        rib = full
        for _ in range(4):
            rib = self._ribbons(cfg, stand, frame, keep)
            got = self._arc(rib)
            if got > 0 and abs(np.log(got / target_px)) < np.log(1.25):
                break
            keep = float(np.clip(keep * target_px / max(got, 1.0), 0.002, 1.0))
        return rib

    def _compose(self, cfg, ribbons, soil, seed):
        from exp_realism.soil_interaction import composite_realism_frame
        bank = self.sim.ensure_root_bank() if cfg.use_real_root_bank else None
        return composite_realism_frame(ribbons, soil, cfg, root_bank=bank, rng=np.random.default_rng(seed))

    def _frame(self, cfg, i_bg: int, seed: int, target_px: float):
        """(composite dict, ignore map) of a frame with ~``target_px`` of simulated root."""
        rng = np.random.default_rng([self.seed, int(seed), 7])
        flip = int(rng.integers(4))
        ribbons = None
        for _ in range(6):                                    # a view with enough root for the target
            stand, frame = int(rng.integers(N_STANDS)), int(rng.integers(40))
            ribbons = self._ribbons_for_length(cfg, stand, frame, target_px)
            if ribbons:
                break
        soil, ignore = self._background(i_bg, flip)
        return self._compose(cfg, ribbons or [], soil, [self.seed, int(seed), 55]), ignore

    def _probes(self, n: int) -> List[Tuple[int, int]]:
        rng = np.random.default_rng(self.seed + 1)
        return [(int(rng.integers(len(self.backgrounds))), 1000 + i) for i in range(n)]

    # -- fit to the annotations -----------------------------------------------------------------
    def fit(self, progress: Optional[Callable[[str, int, int], None]] = None, n_probe: int = 6):
        """Diameter scale + floor and colour gain from test renders; returns the fitted config.

        Probe frames carry the annotated median root length, so widths and colours are compared
        on frames like the user's."""
        from skimage.morphology import skeletonize
        from mrsim.traits import skeleton_widths_px

        st = self.stats
        real_w = st.get("width_px_q") or {}
        real_len = np.array(st.get("length_px") or [])
        if not real_w or not len(real_len):
            return self.cfg
        target = float(np.median(real_len))
        probes = self._probes(n_probe)
        total = (len(DIAM_GRID) + len(DIAM_MIN_FRACS) + len(GAIN_GRID)) * n_probe
        done = 0
        t0 = time.time()

        def tick(what):
            nonlocal done
            done += 1
            if progress:
                progress(what, done, total)

        def widths(cfg):
            ws = []
            for i_bg, seed in probes:
                out, _ = self._frame(cfg, i_bg, seed, target)
                m = out["mask"]
                if m.sum() > 10:
                    ws.append(skeleton_widths_px(m, skel=skeletonize(m)))
                tick("diameter")
            return np.concatenate(ws) if ws else None

        def wloss(w, pairs):
            return sum(abs(np.log(max(np.percentile(w, p), 0.5) / max(real_w[f"p{p}"], 0.5))) * wt for p, wt in pairs)

        # 1. diameter scale: length-weighted widths at skeleton pixels, p50 (and p25/p75)
        best = None
        for scale in DIAM_GRID:
            w = widths(self.cfg.copy(diam_scale=float(scale)))
            if w is not None:
                loss = wloss(w, ((25, 0.5), (50, 1.0), (75, 0.5)))
                if best is None or loss < best[0]:
                    best = (loss, float(scale), w)
        if best is None:
            return self.cfg
        cfg = self.cfg.copy(diam_scale=best[1])
        # 2. diameter floor: the thin end (p10/p25) -- annotators paint thin roots a few px wide
        best_min = (wloss(best[2], ((10, 1.0), (25, 0.5))), float(cfg.diam_min_mm), best[2])
        p10_mm = real_w["p10"] / float(st.get("px_per_mm", 11.81))
        for frac in DIAM_MIN_FRACS[1:]:
            c = cfg.copy(diam_min_mm=float(max(cfg.diam_min_mm, frac * p10_mm)))
            w = widths(c)
            if w is not None:
                loss = wloss(w, ((10, 1.0), (25, 0.5)))
                if loss < best_min[0]:
                    best_min = (loss, float(c.diam_min_mm), w)
        cfg = cfg.copy(diam_min_mm=best_min[1])
        w_fit = best_min[2]

        # 3. colour gain: drawn root-vs-soil luma difference like the annotated one
        real_dl = (st.get("color_dluma_q") or {}).get("p50")
        best_g = None
        if cfg.root_color_deltas and real_dl is not None:
            for g in GAIN_GRID:
                c = cfg.copy(root_color_gain=float(g))
                dls = []
                for i_bg, seed in probes:
                    out, _ = self._frame(c, i_bg, seed, target)
                    rgb_u8 = np.clip(out["rgb"] * 255.0, 0, 255).astype(np.uint8)
                    dls.extend(float(np.dot(d, _LUMA)) for d in component_color_deltas(rgb_u8, out["mask"]))
                    tick("colour")
                if dls:
                    loss = abs(float(np.median(dls)) - real_dl)
                    if best_g is None or loss < best_g[0]:
                        best_g = (loss, g, round(float(np.median(dls)), 4))
            if best_g is not None:
                cfg = cfg.copy(root_color_gain=float(best_g[1]))
        self.cfg = cfg
        self.fit_report = {
            "diam_scale": best[1], "diam_min_mm": round(best_min[1], 3),
            "width_px_synthetic": {f"p{p}": round(float(np.percentile(w_fit, p)), 2) for p in (10, 25, 50, 75, 90)},
            "width_px_real": real_w,
            "length_px_real_median": round(target, 1),
            "color_gain": float(cfg.root_color_gain),
            "color_dluma_real_p50": real_dl, "color_dluma_synthetic_p50": best_g[2] if best_g else None,
            "seconds": round(time.time() - t0, 1),
        }
        return cfg

    # -- synthetic training frames ------------------------------------------------------------------
    def render(self, i_bg: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """(rgb uint8, root mask, valid mask, info) of one frame on background ``i_bg``; its root
        length is drawn from the annotated frames' lengths."""
        from skimage.morphology import skeletonize
        from mrsim.traits import skeleton_length_px

        rng = np.random.default_rng([self.seed, int(seed)])
        lengths = self.stats.get("length_px") or [400.0]
        target = float(rng.choice(lengths))
        out, ignore = self._frame(self.cfg, i_bg, seed, target)
        mask = out["mask"].astype(bool)
        got = skeleton_length_px(mask, skel=skeletonize(mask)) if mask.any() else 0.0
        rgb = np.clip(out["rgb"] * 255.0, 0, 255).astype(np.uint8)
        valid = ~ignore | mask                      # annotated real roots: ignored unless a simulated root covers them
        shape = self.backgrounds[i_bg % len(self.backgrounds)][3]
        if (shape[1], shape[0]) != SIM_WH:
            rgb = cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            valid = cv2.resize(valid.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        return rgb, mask, valid, {"background": self.backgrounds[i_bg % len(self.backgrounds)][0],
                                  "length_px": round(float(got), 1), "target_px": round(target, 1)}


def montage(items: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]], cols: int = 3, width: int = 320) -> np.ndarray:
    """Preview grid (RGB): each synthetic frame with its root outline (red) and ignored pixels (blue)."""
    tiles = []
    for rgb, mask, valid in items:
        v = rgb.copy()
        cv2.drawContours(v, cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0],
                         -1, (255, 40, 40), 1)
        v[~valid] = (0.5 * v[~valid] + 0.5 * np.array([60, 120, 255])).astype(np.uint8)
        h = int(round(v.shape[0] * width / v.shape[1]))
        tiles.append(cv2.resize(v, (width, h), interpolation=cv2.INTER_AREA))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


__all__ = ["measure_annotations", "component_color_deltas", "inpaint_roots", "default_base_config",
           "SynthGenerator", "montage"]
