"""Same tube & window across dates: the previous annotation guides the next one.

Minirhizotron frames are named ``..._T61_L004_2018.12.11_103144_3_mct.JPG``:
tube 61, window (location) 4, date, time, session -- or, from CI-600 scanners
(RootDetector's samples), ``CD_T017_L002_02.06.19_141800_020_CA.tiff`` with the
date as DD.MM.YY.  What precedes ``_T`` (site code) is part of the window.
Folder order already groups a window's dates, so annotating date after date
with "Save & Next" walks each window forward in time.

For a frame without its own annotation, :func:`find_previous_annotation`
finds the newest earlier date of the same window that has one, and
:func:`align` estimates the camera shift between the two dates (phase
correlation of band-passed grey images; consecutive sessions here move by
~40 px).  When the match is confident, :func:`carry_roots` re-traces every
previous root on the new image: each root's skeleton becomes a stroke for the
root tracer, so it snaps onto where the root is now, takes its current width,
and follows it further if it has grown.  Roots with no ridge left are
reported as not found instead of being carried.
"""

from __future__ import annotations

import os
import re
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from exp_realism.root_tracer import RidgeCache, TraceParams, trace_root

_FRAME_RE = re.compile(
    r"_T(?P<tube>\d+)_L(?P<loc>\d+)_(?P<date>\d{4}\.\d{2}\.\d{2}|\d{2}\.\d{2}\.\d{2})_(?P<time>\d{6})_(?P<session>\d+)",
    re.I,
)
MIN_ALIGN_CONF = 0.10       # phase-correlation peak; good pairs here score 0.25-0.45, mismatches < 0.03
MIN_ROOT_SCORE = 1.3        # ridge-vs-soil contrast below which a carried root counts as gone


@dataclass(frozen=True)
class FrameKey:
    tube: str
    loc: str
    date: str                   # YYYY.MM.DD
    time: str
    session: int
    site: str = ""              # what precedes "_T<tube>" ("rf-", "CD", ...)

    @property
    def window(self) -> Tuple[str, str, str]:
        return self.site, self.tube, self.loc

    @property
    def when(self) -> Tuple[str, str]:
        return self.date, self.time


def parse_frame(path: str) -> Optional[FrameKey]:
    name = os.path.basename(path)
    m = _FRAME_RE.search(name)
    if not m:
        return None
    date = m["date"]
    if len(date) == 8:                                      # DD.MM.YY (CI-600 scans)
        dd, mm, yy = date.split(".")
        date = f"20{yy}.{mm}.{dd}"
    return FrameKey(m["tube"], m["loc"], date, m["time"], int(m["session"]), name[: m.start()])


def earlier_frames(path: str, paths: Sequence[str]) -> List[str]:
    """Frames of the same tube & window taken before ``path``, newest first."""
    key = parse_frame(path)
    if key is None:
        return []
    out = []
    for p in paths:
        k = parse_frame(p)
        if k is not None and k.window == key.window and k.when < key.when:
            out.append((k.when, p))
    return [p for _, p in sorted(out, reverse=True)]


def window_frames(path: str, paths: Sequence[str]) -> List[str]:
    """Every frame of the same tube & window as ``path`` (itself included), oldest first."""
    key = parse_frame(path)
    if key is None:
        return [path]
    out = [(k.when, p) for p in paths for k in [parse_frame(p)] if k is not None and k.window == key.window]
    return [p for _, p in sorted(out)]


def find_previous_annotation(
    path: str, paths: Sequence[str], get_mask: Callable[[str], Optional[np.ndarray]]
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Newest earlier frame of the window for which ``get_mask`` returns an annotation."""
    for p in earlier_frames(path, paths):
        m = get_mask(p)
        if m is not None and m.any():
            return p, m
    return None, None


def _bandpass(rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), 1.5) - cv2.GaussianBlur(g, (0, 0), 8.0)


def align(prev_rgb: np.ndarray, cur_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
    """2x3 matrix mapping previous-date pixels onto the current frame, and its confidence."""
    if prev_rgb.shape != cur_rgb.shape:
        return np.float32([[1, 0, 0], [0, 1, 0]]), 0.0
    a, b = _bandpass(prev_rgb), _bandpass(cur_rgb)
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), conf = cv2.phaseCorrelate(a, b, win)
    return np.float32([[1, 0, dx], [0, 1, dy]]), float(conf)


def warp_mask(mask: np.ndarray, M: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    return cv2.warpAffine(mask.astype(np.uint8), M, (shape[1], shape[0]), flags=cv2.INTER_NEAREST) > 0


def _skeleton_paths(comp: np.ndarray, min_len: int = 8) -> List[np.ndarray]:
    """A root's skeleton as ordered (x, y) paths: the longest first, then its branches."""
    from skimage.morphology import skeletonize
    sk = skeletonize(comp)
    pix = set(map(tuple, np.argwhere(sk)))
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def far(start, pool):
        prev, q, last = {start: None}, deque([start]), start
        while q:
            last = q.popleft()
            for dy, dx in nbrs:
                n = (last[0] + dy, last[1] + dx)
                if n in pool and n not in prev:
                    prev[n] = last
                    q.append(n)
        return last, prev

    paths = []
    while pix:
        seed = next(iter(pix))
        end_a, _ = far(seed, pix)
        end_b, prev = far(end_a, pix)
        path, node = [], end_b
        while node is not None:
            path.append(node)
            node = prev[node]
        if len(path) >= min_len or not paths:
            paths.append(np.array([[x, y] for y, x in path], np.float64))
            pix -= {(y + dy, x + dx) for y, x in path for dy in (-2, -1, 0, 1, 2) for dx in (-2, -1, 0, 1, 2)}
        else:
            pix -= set(prev)                                  # a stub too short to trace on its own
    return paths


def carry_roots(
    prev_mask_aligned: np.ndarray,
    cur_rgb: np.ndarray,
    cache: Optional[RidgeCache],
    stop_mask: Optional[np.ndarray],
    params: TraceParams,
    min_area: int = 15,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int]:
    """Re-trace each previous root on the current frame.

    Returns ([(mask, footprint)] per root found, number of previous roots not found).
    """
    n, lbl, st, _ = cv2.connectedComponentsWithStats(prev_mask_aligned.astype(np.uint8), connectivity=8)
    found, gone = [], 0
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            continue
        comp = lbl == i
        half_w = float(cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3).max())
        r = int(np.clip(round(1.5 * half_w + 2), 4, max(4, params.brush_radius)))
        root = np.zeros(comp.shape, bool)
        foot = np.zeros(comp.shape, np.uint8)
        scores = []
        for path in _skeleton_paths(comp):
            pts = path[:: 3] if len(path) > 6 else path[[0, -1]] if len(path) > 1 else path
            p = TraceParams(**{**params.__dict__, "brush_radius": r})
            res = trace_root(cur_rgb, pts, p, stop_mask=stop_mask, cache=cache)
            scores.append(res.score)
            root |= res.mask > 0
            ip = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(foot, [ip], False, 1, thickness=2 * r + 1)
        if scores and max(scores) >= MIN_ROOT_SCORE and root.any():
            found.append((root, foot > 0))
        else:
            gone += 1
    return found, gone


__all__ = ["FrameKey", "parse_frame", "earlier_frames", "window_frames", "find_previous_annotation", "align",
           "warp_mask", "carry_roots", "MIN_ALIGN_CONF"]
