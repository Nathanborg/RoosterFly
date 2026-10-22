"""Editable annotation of one image for the root annotator (Qt-free, undoable).

What is on screen as "root" is

    active = (base & ~erased) | traces | pen

* ``base`` -- the model's mask at the current threshold, or a saved annotation
  that was reloaded (``base_override``), which the threshold then no longer changes;
* ``traces`` -- one object per root the smart brush traced (``origin`` "user"),
  or carried over / pasted from another date (``origin`` "carried").  Each keeps
  ``votes``: how many of its strokes covered each pixel -- its anchors, and how
  often the user insisted on a spot;
* ``pen`` -- exact pen strokes; ``erased`` -- base pixels the user removed.

Strokes are routed to a root: a stroke mostly on an existing trace *refines*
it (the tracer re-measures the root under the stroke, inside the brush, see
``root_tracer.trace_root(existing=)``), one partly on it *extends* it,
otherwise it starts a *new* root.  The brush only adds; the eraser trims.

The smart eraser removes the full width of the root under the stroke and any
part of a trace the cut disconnects from where that root was painted (an
over-followed tail goes in one stroke, the root itself stays); for model
segments it removes the stroke's cross-section plus small cut-off pieces.
``component`` mode removes whole connected roots, ``exact`` only the brush area.

``paste`` replaces the annotation with a shape moved from another date (the
model's prediction is set aside, as for a reopened saved mask); ``empty``
leaves no roots at all.

Memory: a trace is stored cropped to its bounding box, and ``pen`` / ``erased``
are replaced, never changed in place, so undo steps share arrays instead of
copying full frames (a 40-step history on a 6 MP scan held ~1 GB before).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

REFINE_FRAC = 0.6       # stroke samples on a trace for the stroke to refine it
EXTEND_FRAC = 0.15      # ... to extend it

Box = Tuple[int, int, int, int]      # y0, y1, x0, x1


def _disk(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    return cv2.dilate(m.astype(np.uint8), _disk(r)).astype(bool)


def _pieces(m: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    return n, lbl, st[:, cv2.CC_STAT_AREA]


def _keep_anchored(m: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Pieces of ``m`` touching ``anchors``; the largest piece when none does."""
    n, lbl, area = _pieces(m)
    if n <= 2:
        return m
    ids = np.unique(lbl[m & anchors])
    ids = ids[ids > 0]
    if len(ids) == 0:
        ids = [int(np.argmax(area[1:])) + 1]
    return np.isin(lbl, ids)


def _bbox(m: np.ndarray) -> Optional[Box]:
    x, y, w, h = cv2.boundingRect(np.ascontiguousarray(m, dtype=np.uint8))
    return None if w == 0 or h == 0 else (y, y + h, x, x + w)


def _union(a: Optional[Box], b: Optional[Box]) -> Optional[Box]:
    if a is None or b is None:
        return a or b
    return min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])


def _grow(b: Box, r: int, shape: Tuple[int, int]) -> Box:
    return max(0, b[0] - r), min(shape[0], b[1] + r), max(0, b[2] - r), min(shape[1], b[3] + r)


def _cut(full: np.ndarray, b: Box) -> np.ndarray:
    return full[b[0]:b[1], b[2]:b[3]]


@dataclass(frozen=True)
class Trace:
    sub: np.ndarray             # bool, the root inside ``box``
    votes: np.ndarray           # uint8, box-sized
    box: Box
    origin: str = "user"        # "user" | "carried"

    @classmethod
    def make(cls, mask: np.ndarray, votes: np.ndarray, origin: str = "user") -> Optional["Trace"]:
        """From full-frame arrays: cropped to where the root or its strokes are."""
        b = _bbox(mask | (votes > 0))
        if b is None:
            return None
        return cls(np.ascontiguousarray(_cut(mask, b), dtype=bool), np.ascontiguousarray(_cut(votes, b), dtype=np.uint8),
                   b, origin)

    def placed(self, b: Box, what: str = "sub") -> np.ndarray:
        """``sub`` or ``votes`` on the (larger) box ``b``."""
        src = self.sub if what == "sub" else self.votes
        out = np.zeros((b[1] - b[0], b[3] - b[2]), src.dtype)
        y, x = self.box[0] - b[0], self.box[2] - b[2]
        out[y:y + src.shape[0], x:x + src.shape[1]] = src
        return out

    def mask(self, shape: Tuple[int, int]) -> np.ndarray:
        out = np.zeros(shape, bool)
        _cut(out, self.box)[:] = self.sub
        return out


class AnnotationState:
    MAX_UNDO = 40

    def __init__(self, shape: Tuple[int, int]):
        H, W = int(shape[0]), int(shape[1])
        self.shape = (H, W)
        self.traces: List[Trace] = []
        self.pen = np.zeros((H, W), bool)
        self.erased = np.zeros((H, W), bool)
        self.base_override: Optional[np.ndarray] = None
        self.dirty = False                      # edited since loaded / saved
        self.note = ""                          # where the starting point came from
        self.rev = 0                            # bumped on every change of what is on screen (cache key)
        self._undo: list = []
        self._redo: list = []

    # -- undo -----------------------------------------------------------------
    def _snap(self):                            # arrays are never changed in place: references suffice
        return (list(self.traces), self.pen, self.erased, self.base_override, self.dirty)

    def _restore(self, snap):
        traces, self.pen, self.erased, self.base_override, self.dirty = snap
        self.traces = list(traces)
        self.rev += 1

    def checkpoint(self):
        """Call before every edit: the state it leaves becomes one undo step."""
        self._undo.append(self._snap())
        del self._undo[: -self.MAX_UNDO]
        self._redo.clear()

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._snap())
        self._restore(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._snap())
        self._restore(self._redo.pop())
        return True

    # -- masks ----------------------------------------------------------------
    def trace_mask(self, origin: Optional[str] = None, box: Optional[Box] = None) -> np.ndarray:
        """Union of the traces (of one ``origin``), on the frame or only inside ``box``."""
        b = box or (0, self.shape[0], 0, self.shape[1])
        m = np.zeros((b[1] - b[0], b[3] - b[2]), bool)
        for t in self.traces:
            if origin is not None and t.origin != origin:
                continue
            y0, y1, x0, x1 = max(b[0], t.box[0]), min(b[1], t.box[1]), max(b[2], t.box[2]), min(b[3], t.box[3])
            if y0 < y1 and x0 < x1:
                m[y0 - b[0]:y1 - b[0], x0 - b[2]:x1 - b[2]] |= t.sub[y0 - t.box[0]:y1 - t.box[0], x0 - t.box[2]:x1 - t.box[2]]
        return m

    def user_mask(self) -> np.ndarray:
        return self.pen | self.trace_mask()

    def base(self, model_mask: Optional[np.ndarray]) -> np.ndarray:
        if self.base_override is not None:
            return self.base_override
        return np.zeros(self.shape, bool) if model_mask is None else model_mask.astype(bool)

    def active(self, model_mask: Optional[np.ndarray]) -> np.ndarray:
        return (self.base(model_mask) & ~self.erased) | self.user_mask()

    def has_edits(self) -> bool:
        return bool(self.traces) or self.pen.any() or self.erased.any() or self.base_override is not None

    # -- loading ----------------------------------------------------------------
    def _reset(self, base_override: Optional[np.ndarray]):
        self.traces, self.base_override = [], base_override
        self.pen = np.zeros(self.shape, bool)
        self.erased = np.zeros(self.shape, bool)
        self.rev += 1

    def load_saved(self, mask: np.ndarray, note: str = ""):
        self._reset(mask.astype(bool))
        self.dirty, self.note = False, note

    def clear(self):
        """Back to the model's mask alone (drops a reloaded saved annotation too)."""
        self._reset(None)
        self.dirty = True

    def empty(self):
        """No roots at all: the model's mask is set aside too."""
        self._reset(np.zeros(self.shape, bool))
        self.dirty, self.note = True, "emptied"

    def paste(self, mask: np.ndarray, note: str = ""):
        """The annotation becomes exactly ``mask`` (a shape moved from another date), one editable
        trace per connected root; the model's mask is set aside.  Every pixel is an anchor, so the
        smart eraser trims a pasted root where it cuts and keeps both sides."""
        self._reset(np.zeros(self.shape, bool))
        n, lbl, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h = (int(v) for v in st[i, :4])
            sub = lbl[y:y + h, x:x + w] == i
            self.traces.append(Trace(sub, sub.astype(np.uint8), (y, y + h, x, x + w), "carried"))
        self.dirty, self.note = True, note
        self.rev += 1

    def add_roots(self, mask: np.ndarray, existing: np.ndarray, note: str = "", near_px: int = 6,
                  far_px: int = 20, min_px: int = 40) -> int:
        """Add what ``mask`` (a shape moved from another date) has that the image does not show yet:
        its parts farther than ``near_px`` from ``existing`` (what is on screen), as pieces that reach at
        least ``far_px`` away (a new root, or a root's extension) -- a moved copy lying a few pixels beside
        a root already drawn is not added -- of at least ``min_px``.  Returns the pieces added."""
        ex = existing.astype(bool)
        missing = mask.astype(bool) & ~_dilate(ex, near_px)
        dist = cv2.distanceTransform((~ex).astype(np.uint8), cv2.DIST_L2, 3) if ex.any() else None
        n, lbl, st, _ = cv2.connectedComponentsWithStats(missing.astype(np.uint8), connectivity=8)
        added = 0
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < min_px:
                continue
            x, y, w, h = (int(v) for v in st[i, :4])
            sub = lbl[y:y + h, x:x + w] == i
            if dist is not None and float(dist[y:y + h, x:x + w][sub].max()) < far_px:
                continue
            self.traces.append(Trace(sub, sub.astype(np.uint8), (y, y + h, x, x + w), "carried"))
            added += 1
        if added:
            self.dirty = True
            self.note = note or self.note
            self.rev += 1
        return added

    # -- strokes ----------------------------------------------------------------
    def route_stroke(self, pts, r: int) -> Tuple[str, Optional[int]]:
        """('new' | 'extend' | 'refine', trace index) for a smart-brush stroke."""
        if not self.traces:
            return "new", None
        H, W = self.shape
        p = np.asarray(pts, np.float64).reshape(-1, 2)
        if len(p) > 1:
            seg = np.hypot(*np.diff(p, axis=0).T)
            cum = np.r_[0.0, np.cumsum(seg)]
            t = np.linspace(0.0, cum[-1], max(2, int(cum[-1] // 2) + 1))
            p = np.c_[np.interp(t, cum, p[:, 0]), np.interp(t, cum, p[:, 1])]
        xs = np.clip(np.round(p[:, 0]).astype(int), 0, W - 1)
        ys = np.clip(np.round(p[:, 1]).astype(int), 0, H - 1)
        reach = max(3, r // 2)
        best, best_i = 0.0, None
        for i, tr in enumerate(self.traces):
            b = _grow(tr.box, reach, self.shape)                 # only the trace's own neighbourhood
            inside = (ys >= b[0]) & (ys < b[1]) & (xs >= b[2]) & (xs < b[3])
            if not inside.any():
                continue
            near = _dilate(tr.placed(b), reach)
            frac = float(near[ys[inside] - b[0], xs[inside] - b[2]].sum()) / len(xs)
            if frac > best:
                best, best_i = frac, i
        if best_i is None or best < EXTEND_FRAC:
            return "new", None
        if best < REFINE_FRAC:
            return "extend", best_i
        return "refine", best_i

    def apply_trace(self, mode: str, idx: Optional[int], mask: np.ndarray, footprint: np.ndarray,
                    r: int, origin: str = "user"):
        """Merge a traced ``mask`` (HxW) for a stroke with brush area ``footprint``."""
        fp = footprint.astype(bool)
        m = mask.astype(bool) & ~(self.erased & ~fp)        # following never undoes the user's erasures
        votes = fp.astype(np.uint8)
        if mode == "new" or idx is None:
            t = Trace.make(m, votes, origin)
            if t is not None:
                self.traces.append(t)
        else:
            t = self.traces[idx]
            if mode == "extend":
                b = _union(t.box, _bbox(m | fp))
                sub = t.placed(b) | _cut(m, b)
                org = t.origin
            else:                                           # refine: grow near the stroke (the brush adds, the eraser trims)
                grow = m & _dilate(fp, max(2, int(r)))
                b = _union(t.box, _bbox(grow | fp))
                sub = t.placed(b) | _cut(grow, b)
                org = "user"
            v = np.minimum(t.placed(b, "votes").astype(np.int16) + _cut(votes, b), 255).astype(np.uint8)
            self.traces[idx] = Trace(sub, v, b, org)
        self.dirty = True
        self.rev += 1

    def add_pen(self, footprint: np.ndarray):
        self.pen = self.pen | footprint.astype(bool)
        self.dirty = True
        self.rev += 1

    # -- eraser -------------------------------------------------------------------
    def erase(self, footprint: np.ndarray, model_mask: Optional[np.ndarray], mode: str = "smart") -> int:
        """Erase under a stroke; ``mode`` "smart" | "component" | "exact".  Returns pixels removed."""
        fp = footprint.astype(bool)
        act = self.active(model_mask)
        before = int(act.sum())
        E = fp.copy()
        if mode != "exact":
            hit = act & fp
            if hit.any():                                    # the full width of the root under the stroke
                dt = cv2.distanceTransform(act.astype(np.uint8), cv2.DIST_L2, 3)
                k = int(np.ceil(1.3 * float(dt[hit].max()))) + 1
                E |= act & _dilate(fp, k)
        if mode == "component":
            n, lbl, _ = _pieces(act)
            ids = np.unique(lbl[act & fp])
            E |= np.isin(lbl, ids[ids > 0])

        kept: List[Trace] = []
        for t in self.traces:
            Eb = _cut(E, t.box)
            if not (t.sub & Eb).any():
                kept.append(t)
                continue
            m = t.sub & ~Eb
            if mode == "smart" and m.any():
                m = _keep_anchored(m, t.votes > 0)           # an over-followed tail goes with the cut
            if m.sum() >= 4:
                kept.append(Trace(m, t.votes, t.box, t.origin))
        self.traces = kept
        self.pen = self.pen & ~E

        base = self.base(model_mask) & ~self.erased
        cut = E & base
        if mode == "smart" and cut.any():                    # small pieces the cut leaves behind
            n0, lbl0, _ = _pieces(base)
            for cid in np.unique(lbl0[cut]):
                if cid == 0:
                    continue
                rest = (lbl0 == cid) & ~E
                n, lbl, area = _pieces(rest)
                if n <= 2:
                    continue
                big = area[1:].max()
                for j in range(1, n):
                    if area[j] < 0.5 * big:
                        cut |= lbl == j
        self.erased = self.erased | cut
        self.dirty = True
        self.rev += 1
        return before - int(self.active(model_mask).sum())


__all__ = ["Trace", "AnnotationState"]
