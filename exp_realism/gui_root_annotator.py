"""RoosterFly -- interactive minirhizotron root segmentation & annotation studio.

Features:
  1. Instant Startup: Window opens immediately (< 0.5s) with non-blocking model & dataset loading.
  2. Loads the best trained U-Net model (v8_full_real_and_syn_mitb3 or user-selected checkpoint)
     with its exact weights, CDF histogram matching reference, and colour index stats.
  3. Loads single minirhizotron scans or entire folders (e.g. Panama All GIG) with rapid browsing.
  4. Background prediction queue (one thread owns the model): the image on screen first, then
     the next few images (look-ahead), then optionally the whole folder ("Pre-predict Folder").
     Every map is cached on disk (16-bit PNG under _annotations/_prob_cache/<run>), so an image
     predicted once opens instantly.  `python -m exp_realism.annotator_backend --dir ...` fills
     the same cache headless (e.g. on a GPU).
  5. Side-by-side synchronized view:
     - Left: Original RGB minirhizotron scan (with boundary contours and user annotations).
     - Right: Model continuous Probability Heatmap (Jet) or Blended Overlay with dynamic threshold cut.
     - Panning and zooming are 100% synchronized across both canvases.
  6. Probability Threshold Slider:
     - Positioned directly on the probability side; dynamically adjusts the root segmentation
       in real-time (< 2 ms) for that specific image.
  7. Root-Following Smart Brush (Red) -- see exp_realism/root_tracer.py:
     - Paint (or just click) on a missed root: the stroke snaps onto the root's crest
       (multi-scale ridge filter, light or dark roots), then the trace follows the root past
       both ends of the stroke until the ridge fades, bridges short gaps, and stops when it
       runs into an already segmented root.
     - The width comes from the root's own contrast (half maximum against the local soil),
       so the result is a root-shaped tube, never a square block.
     - Shift+drag traces the stroke only (no following); "Stop at" sets how faint a ridge
       the trace keeps following.
  8. Probability Tracer: the same brush run on the model probability map instead of the image.
     Brushing a root again refines it (inside the brush the edges relax each time, so it grows to
     what you paint; the brush only adds); Smart Eraser removes the root's width under the stroke and any tail it cuts
     off (Shift: exact brush area, Ctrl: whole connected root).
  8b. Dates of the same tube & window (T61_L004_<date>): the newest earlier date with an annotation
     is aligned to the frame and its roots are re-traced on it (orange, editable); the cyan outline
     shows the previous date.  See exp_realism/temporal.py.
  8d. Image slider: drag through the folder (raw frames flash by, the image under the handle loads
     fully on release); arrow / Page keys step; "Go to #" jumps to a frame number.
  8e. "Train on my corrections": fine-tunes the model's decoder (encoder frozen) on the saved masks,
     here on the CPU in the background or through a generated Colab notebook on a GPU; the new run
     appears in the Model list (↻ refreshes it).  See exp_realism/finetune_corrections.py.
  8f. "Root simulator": measures the saved masks (root widths, length per frame, colour against
     the soil), fits the v4 simulator to them on the annotated frames' own soil (roots inpainted
     away) and opens the simulator (exp_realism.gui_app) prefilled; the fine-tune can add such
     synthetic frames (see exp_realism/synth_from_annotations.py).
  8c. Saved annotations reload when an image is reopened; edits survive Prev/Next in a session and
     closing asks before discarding unsaved ones.
  9. Real-time Traits Readout:
     - Live calculation of Root Length (mm & cm), Surface Area (mm²), Mean Diameter (mm),
       and Coverage (%) using physical camera calibration (px/mm).
  10. Export & Batch Annotation:
      - Saves binary mask (.png), overlay (.jpg), and traits (.csv).
      - "Save & Next" one-click workflow for rapid dataset curation.

Usage:
    python -m exp_realism.gui_root_annotator
    # or
    python exp_realism/gui_root_annotator.py --dir "Minirhizotron images/Panama/All GIG"
    # pre-predict the whole folder in the background while annotating, 5 images look-ahead:
    python -m exp_realism.gui_root_annotator --dir "..." --predict-all --lookahead 5
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure repository root is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from exp_realism.annotator_backend import (  # noqa: E402
    cache_tag,
    list_images,
    load_cached_prob,
    load_model_bundle,
    mask_border,
    predict_maps,
    centre_cache_path,
    predict_prob,
    prob_cache_path,
    read_rgb,
    save_cached_prob,
)
from exp_realism.root_tracer import ColourCue, RidgeCache, TraceParams, trace_root  # noqa: E402
from exp_realism.annotation_state import AnnotationState  # noqa: E402
from exp_realism import temporal  # noqa: E402
from exp_realism import root_transfer  # noqa: E402

APP_ICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "roosterfly_icon.png")
from exp_realism import finetune_corrections as finetune  # noqa: E402

# Qt5 / Qt6 / PySide6 compatibility shim
QT_API = None
try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtCore import Qt, QThread, pyqtSignal as Signal, QPoint, QRectF
    QT_API = "PyQt5"
except ImportError:
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
        from PyQt6.QtCore import Qt, QThread, pyqtSignal as Signal, QPoint, QRectF
        QT_API = "PyQt6"
    except ImportError:
        try:
            from PySide6 import QtCore, QtGui, QtWidgets
            from PySide6.QtCore import Qt, QThread, Signal, QPoint, QRectF
            QT_API = "PySide6"
        except ImportError as exc:
            raise ImportError(
                "PyQt5, PyQt6, or PySide6 is required. Install with: pip install PyQt5"
            ) from exc


# ==============================================================================
# Helper Utilities
# ==============================================================================

def _align_center():
    return Qt.AlignmentFlag.AlignCenter if hasattr(Qt, "AlignmentFlag") else Qt.AlignCenter


def _smooth_transform():
    return (
        Qt.TransformationMode.SmoothTransformation
        if hasattr(Qt, "TransformationMode")
        else Qt.SmoothTransformation
    )


def _horizontal():
    return Qt.Orientation.Horizontal if hasattr(Qt, "Orientation") else Qt.Horizontal


def _to_u8_rgb(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return np.ascontiguousarray(img)
    return np.ascontiguousarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))


def _smooth_stroke(pts, radius: int) -> List[Tuple[int, int]]:
    """Mouse samples resampled every 1.5 px and averaged over ~``radius`` px (5-11 samples, ends kept):
    the hand's jitter -- and the coarse steps of a zoomed-out view -- no longer show in what is drawn."""
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(p) < 3:
        return [(int(round(x)), int(round(y))) for x, y in p]
    keep = np.r_[True, np.any(np.abs(np.diff(p, axis=0)) > 1e-9, axis=1)]
    p = p[keep]
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.r_[0.0, np.cumsum(seg)]
    if len(p) < 3 or cum[-1] < 3.0:
        return [(int(round(x)), int(round(y))) for x, y in p]
    t = np.linspace(0.0, cum[-1], max(2, int(cum[-1] / 1.5) + 1))
    q = np.c_[np.interp(t, cum, p[:, 0]), np.interp(t, cum, p[:, 1])]
    k = int(np.clip(round(radius / 1.5), 5, 11)) | 1
    k = min(k, len(q) if len(q) % 2 else len(q) - 1)
    if k >= 3:
        pad = np.pad(q, ((k // 2, k // 2), (0, 0)), mode="edge")
        ker = np.ones(k) / k
        sm = np.stack([np.convolve(pad[:, j], ker, mode="valid") for j in range(2)], 1)
        sm[0], sm[-1] = q[0], q[-1]
        q = sm
    out = [(int(round(x)), int(round(y))) for x, y in q]
    return [pt for i, pt in enumerate(out) if i == 0 or pt != out[i - 1]]


def _shift_held() -> bool:
    mod = Qt.KeyboardModifier.ShiftModifier if hasattr(Qt, "KeyboardModifier") else Qt.ShiftModifier
    return bool(QtWidgets.QApplication.keyboardModifiers() & mod)


def _box_of(m: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """(y0, y1, x0, x1) around the non-zero pixels of ``m``; None when there are none."""
    x, y, w, h = cv2.boundingRect(np.ascontiguousarray(m, dtype=np.uint8))
    return None if w == 0 or h == 0 else (y, y + h, x, x + w)


def _rgb_to_qimage(rgb: np.ndarray) -> QtGui.QImage:
    rgb_c = _to_u8_rgb(rgb)
    h, w, ch = rgb_c.shape
    bytes_per_line = ch * w
    fmt = (
        QtGui.QImage.Format.Format_RGB888
        if hasattr(QtGui.QImage, "Format")
        else QtGui.QImage.Format_RGB888
    )
    return QtGui.QImage(rgb_c.data, w, h, bytes_per_line, fmt).copy()


# ==============================================================================
# Background Prediction Queue
# ==============================================================================

PRIO_CURRENT, PRIO_LOOKAHEAD, PRIO_BATCH = 0, 1, 2
SEP_VIEW = "Separate roots (model centre lines)"
CENTRE_TASK = "|centre"   # queue key suffix: the centre-line map of an image whose prediction is cached


class PredictionService(QThread):
    """The one thread that runs the model.

    Requests are served by priority -- the image on screen, then the next few
    images (look-ahead), then a whole-folder pre-prediction -- in request order
    within a priority.  Every map is written to the disk cache, and a cached
    map is loaded instead of re-predicted.
    """

    result_ready = Signal(str, str, object, float, bool)   # tag, path, prob, seconds, from_disk
    centre_ready = Signal(str, str, object)                 # tag, path, centre-line map (model's 2nd head)
    failed = Signal(str, str)                               # path, message
    batch_progress = Signal(int, int, float)                # done, total, seconds per predicted image

    def __init__(self, cache_dir: str = ""):
        super().__init__()
        self.cache_dir = cache_dir
        self._cv = threading.Condition()
        self._tasks: Dict[str, Tuple[int, int]] = {}        # path -> (priority, sequence)
        self._seq = 0
        self._bundle: Optional[Dict[str, Any]] = None
        self._tta = False
        self._tag = ""
        self._stop = False
        self._paused = False
        self._batch: set = set()
        self._batch_total = 0
        self._batch_done = 0
        self._pred_secs: List[float] = []

    def configure(self, bundle: Dict[str, Any], tta: bool):
        with self._cv:
            self._bundle = bundle
            self._tta = bool(tta)
            self._tag = cache_tag(bundle["run_name"], tta)
            self._cv.notify_all()

    def _put(self, path: str, prio: int):
        cur = self._tasks.get(path)
        if cur is None or prio < cur[0]:
            self._seq += 1
            self._tasks[path] = (prio, self._seq)

    def focus(self, current: Optional[str], lookahead: List[str]):
        """Serve ``current`` next and keep ``lookahead`` queued; older focus requests are demoted."""
        with self._cv:
            keep = set(lookahead) | ({current, current + CENTRE_TASK} if current else set())
            for p, (prio, _) in list(self._tasks.items()):
                if prio < PRIO_BATCH and p not in keep:
                    if p in self._batch:
                        self._seq += 1
                        self._tasks[p] = (PRIO_BATCH, self._seq)
                    else:
                        del self._tasks[p]
            if current:
                self._put(current, PRIO_CURRENT)
            for p in lookahead:
                self._put(p, PRIO_LOOKAHEAD)
            self._cv.notify_all()

    def start_batch(self, paths: List[str]):
        with self._cv:
            self._batch = set(paths)
            self._batch_total = len(self._batch)
            self._batch_done = 0
            self._pred_secs = []
            for p in paths:
                self._put(p, PRIO_BATCH)
            self._cv.notify_all()
        self.batch_progress.emit(0, len(paths), 0.0)

    def cancel_batch(self):
        with self._cv:
            for p, (prio, _) in list(self._tasks.items()):
                if prio == PRIO_BATCH:
                    del self._tasks[p]
            self._batch = set()
            self._batch_total = self._batch_done = 0

    def pause(self, paused: bool):
        """Hold the queue (after the image in progress), e.g. while a fine-tune needs the CPU."""
        with self._cv:
            self._paused = bool(paused)
            self._cv.notify_all()

    def shutdown(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self.wait(20000)

    def run(self):
        while True:
            with self._cv:
                while not self._stop and (self._paused or not self._tasks or self._bundle is None):
                    self._cv.wait()
                if self._stop:
                    return
                path = min(self._tasks, key=self._tasks.get)
                prio, _ = self._tasks.pop(path)
                bundle, tta, tag = self._bundle, self._tta, self._tag
            try:
                self._serve(path, prio, bundle, tta, tag)
            except Exception as exc:
                self._count_batch(path, None)
                self.failed.emit(path, f"{type(exc).__name__}: {exc}")

    def _serve(self, path: str, prio: int, bundle: Dict[str, Any], tta: bool, tag: str):
        if path.endswith(CENTRE_TASK):                      # centre lines of an image cached before they were kept
            self._serve_centre(path[: -len(CENTRE_TASK)], bundle, tta, tag)
            return
        cpath = prob_cache_path(path, tag, self.cache_dir)
        t0 = time.time()
        two_heads = bundle.get("n_out", 1) > 1
        if prio == PRIO_BATCH and os.path.exists(cpath) and (
                not two_heads or os.path.exists(centre_cache_path(path, tag, self.cache_dir))):
            self._count_batch(path, None)                   # already cached (with its centre lines): nothing to do
            return
        prob = load_cached_prob(cpath)
        centre = load_cached_prob(centre_cache_path(path, tag, self.cache_dir)) if (two_heads and prob is not None) else None
        from_disk = prob is not None
        rgb = None
        if prob is None:
            rgb = read_rgb(path)
            if rgb is None:
                raise IOError(f"cannot read {path}")
            prob, centre = predict_maps(bundle, rgb, tta)
            save_cached_prob(cpath, prob)                   # the cache keeps the raw maps
            if centre is not None:
                save_cached_prob(centre_cache_path(path, tag, self.cache_dir), centre)
        if bundle.get("border_mask"):
            rgb = rgb if rgb is not None else read_rgb(path)
            if rgb is not None and rgb.shape[:2] == prob.shape:
                prob = mask_border(prob, rgb)
                if centre is not None:
                    centre = mask_border(centre, rgb)
        secs = time.time() - t0
        self._count_batch(path, None if from_disk else secs)
        if prio < PRIO_BATCH:
            self.result_ready.emit(tag, path, prob, secs, from_disk)
            if centre is not None:
                self.centre_ready.emit(tag, path, centre)
            elif two_heads and prio == PRIO_CURRENT:        # predicted before the centre lines were cached
                with self._cv:
                    self._put(path + CENTRE_TASK, PRIO_LOOKAHEAD)
                    self._cv.notify_all()

    def _serve_centre(self, path: str, bundle: Dict[str, Any], tta: bool, tag: str):
        rgb = read_rgb(path)
        if rgb is None:
            raise IOError(f"cannot read {path}")
        prob, centre = predict_maps(bundle, rgb, tta)
        if centre is None:
            return
        save_cached_prob(centre_cache_path(path, tag, self.cache_dir), centre)
        if bundle.get("border_mask") and rgb.shape[:2] == centre.shape:
            centre = mask_border(centre, rgb)
        self.centre_ready.emit(tag, path, centre)

    def _count_batch(self, path: str, secs: Optional[float]):
        with self._cv:
            if path not in self._batch:
                return
            self._batch.discard(path)
            self._batch_done += 1
            if secs is not None:
                self._pred_secs = (self._pred_secs + [secs])[-50:]
            done, total = self._batch_done, self._batch_total
            rate = float(np.mean(self._pred_secs)) if self._pred_secs else 0.0
        self.batch_progress.emit(done, total, rate)


class CarryOverWorker(QThread):
    """Aligns the previous date's annotation to the frame and (optionally) re-traces its roots."""

    done = Signal(str, object, object, str)                 # path, [(mask, footprint)], aligned prev mask, message

    def __init__(self, path: str, rgb: np.ndarray, prev_path: str, prev_mask: np.ndarray,
                 cache: Optional[RidgeCache], params: TraceParams, carry: bool, stop_mask: Optional[np.ndarray],
                 prob: Optional[np.ndarray] = None):
        super().__init__()
        self.path, self.rgb, self.prev_path, self.prev_mask = path, rgb, prev_path, prev_mask
        self.cache, self.params, self.carry, self.stop_mask, self.prob = cache, params, carry, stop_mask, prob

    def run(self):
        key = temporal.parse_frame(self.prev_path)
        when = key.date if key else os.path.basename(self.prev_path)
        try:
            prev_rgb = read_rgb(self.prev_path)
            if prev_rgb is None:
                raise IOError(f"cannot read {self.prev_path}")
            # registered like "Accept prev. date": whole frame (phase correlation + StackReg), then each root
            res = root_transfer.transfer(prev_rgb, self.prev_mask, self.rgb, dst_prob=self.prob)
            if not res.placed:
                self.done.emit(self.path, [], None,
                               f"Previous date {when} is annotated but does not line up with this frame "
                               f"({res.message}) - nothing carried over.")
                return
            warped = res.mask
            shift = res.reg.describe(self.rgb.shape[:2])
            if not self.carry:
                self.done.emit(self.path, [], warped, f"Previous date {when} aligned ({shift}): cyan outline.")
                return
            found, gone = temporal.carry_roots(warped, self.rgb, self.cache, self.stop_mask, self.params)
            msg = (f"↪ Carried {len(found)} root(s) from {when} ({shift})"
                   + (f"; {gone} previous root(s) not found - gone?" if gone else "")
                   + ".  Orange = carried: check them, erase what died, trace what is new.")
            self.done.emit(self.path, found, warped, msg)
        except Exception as exc:
            self.done.emit(self.path, [], None, f"Carry-over from {when} failed: {exc}")


class ExportWorker(QThread):
    """Writes the root-traits CSV (``root_export``): one row per image or per separated root, off the GUI thread."""

    progress = Signal(int, int)
    done = Signal(str, int, str)                            # csv path, rows written, error

    def __init__(self, out_path: str, jobs: List[dict], px_per_mm: float, per_root: bool = False,
                 centre_lines: bool = True):
        super().__init__()
        self.out_path, self.jobs, self.px_per_mm = out_path, jobs, px_per_mm
        self.per_root, self.centre_lines = per_root, centre_lines
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        import csv
        from exp_realism import root_export, root_separation
        n = 0
        try:
            os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
            with open(self.out_path, "w", newline="", encoding="utf-8") as fh:
                cols = root_export.ROOT_COLUMNS if self.per_root else root_export.COLUMNS
                wr = csv.DictWriter(fh, fieldnames=cols)
                wr.writeheader()
                for i, j in enumerate(self.jobs):
                    if self._stop:
                        break
                    if i % 10 == 0:
                        self.progress.emit(i, len(self.jobs))
                    mask, thr = j.get("mask"), j.get("threshold")
                    if mask is None and j.get("mask_path"):
                        m = cv2.imread(j["mask_path"], cv2.IMREAD_GRAYSCALE)
                        mask = None if m is None else m > 127
                    elif mask is None and j.get("prob_path"):
                        p = load_cached_prob(j["prob_path"])
                        if p is not None:
                            mask = p >= thr
                            e = j.get("edge", 0)
                            if e > 0:
                                mask[:e] = mask[-e:] = False
                                mask[:, :e] = mask[:, -e:] = False
                    source = j["source"] if mask is not None else "not predicted yet"
                    sep, centre_src = None, ""
                    if mask is not None and (self.per_root or self.centre_lines):
                        model_c = j.get("centre")
                        if model_c is None and j.get("centre_path"):
                            model_c = load_cached_prob(j["centre_path"])
                        if model_c is not None and model_c.shape != mask.shape:
                            model_c = None
                        centre_src = "model centre lines" if model_c is not None else "mask shape"
                        sep = root_separation.separate_roots(mask, root_separation.centre_map(mask, model_c))
                    thr_out = thr if j["source"] == "model prediction" else None
                    if self.per_root:
                        if mask is not None:
                            for row in root_export.root_table(j["path"], j["index"], sep, source, self.px_per_mm,
                                                              centre_src):
                                wr.writerow(row)
                                n += 1
                    else:
                        wr.writerow(root_export.image_row(j["path"], j["index"], mask, source, self.px_per_mm, thr_out,
                                                          sep, f"centre lines ({centre_src})"))
                        n += 1
            self.progress.emit(len(self.jobs), len(self.jobs))
            self.done.emit(self.out_path, n, "stopped" if self._stop else "")
        except Exception as exc:
            self.done.emit(self.out_path, n, f"{type(exc).__name__}: {exc}")


class ExportDialog(QtWidgets.QDialog):
    """Rows (per image / per root), which images, and where the length comes from."""

    def __init__(self, parent, default_path: str):
        super().__init__(parent)
        self.setWindowTitle("Export root traits (CSV)")
        form = QtWidgets.QFormLayout(self)
        self.rows = QtWidgets.QComboBox()
        self.rows.addItems(["One row per image", "One row per root (separated along the model's centre lines)"])
        self.scope = QtWidgets.QComboBox()
        self.scope.addItems(["Annotated images only (saved or edited)",
                             "All images (the model's prediction where not annotated)"])
        self.length = QtWidgets.QComboBox()
        self.length.addItems(["Model centre lines (touching roots separated)", "Mask skeleton"])
        self.length.setToolTip("Centre lines: the roots are separated along the model's centre-line map, so a clump\n"
                               "of touching roots is measured root by root (no spurious skeleton spokes). Images\n"
                               "predicted before the map was kept use the mask's own shape instead.")
        self.rows.currentIndexChanged.connect(lambda i: self.length.setEnabled(i == 0))
        form.addRow("Rows:", self.rows)
        form.addRow("Images:", self.scope)
        form.addRow("Length from:", self.length)
        path_row = QtWidgets.QHBoxLayout()
        self.path = QtWidgets.QLineEdit(default_path)
        self.path.setMinimumWidth(420)
        browse = QtWidgets.QPushButton("…")
        browse.setFixedWidth(30)
        browse.clicked.connect(self._browse)
        path_row.addWidget(self.path)
        path_row.addWidget(browse)
        form.addRow("Save to:", path_row)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Export")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def _browse(self):
        out, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save root traits", self.path.text(), "CSV (*.csv)")
        if out:
            self.path.setText(out)

    def choices(self) -> Dict[str, Any]:
        return {"per_root": self.rows.currentIndex() == 1, "all_images": self.scope.currentIndex() == 1,
                "centre_lines": self.rows.currentIndex() == 1 or self.length.currentIndex() == 0,
                "path": self.path.text().strip()}


class SeparationWorker(QThread):
    """Separates the roots on screen (``root_separation``) off the GUI thread."""

    done = Signal(object, object, str)                      # key, (Separation, length px, model map used), error

    def __init__(self, key: tuple, mask: np.ndarray, model_centre: Optional[np.ndarray]):
        super().__init__()
        self.key, self.mask, self.model_centre = key, mask, model_centre

    def run(self):
        from exp_realism import root_separation as RS
        try:
            mc = self.model_centre if (self.model_centre is not None and self.model_centre.shape == self.mask.shape) else None
            sep = RS.separate_roots(self.mask, RS.centre_map(self.mask, mc))
            self.done.emit(self.key, (sep, RS.centre_length_px(sep), mc is not None), "")
        except Exception as exc:
            self.done.emit(self.key, None, f"{type(exc).__name__}: {exc}")


class _ElidedLabel(QtWidgets.QLabel):
    """One-line label that never widens the window: text that does not fit ends in "…" and the full
    text is its tooltip.  (Long status messages had pushed the window's minimum width past the screen.)"""

    def __init__(self, text: str = ""):
        super().__init__()
        self._full = ""
        self.setText(text)

    def setText(self, text: str):
        self._full = str(text)
        self.setToolTip(self._full if len(self._full) > 50 else "")
        self._refit()
        self.updateGeometry()

    def text(self) -> str:
        return self._full

    def _refit(self):
        mode = Qt.TextElideMode.ElideRight if hasattr(Qt, "TextElideMode") else Qt.ElideRight
        super().setText(self.fontMetrics().elidedText(self._full, mode, max(12, self.contentsRect().width())))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refit()

    def sizeHint(self) -> QtCore.QSize:
        fm = self.fontMetrics()
        w = fm.horizontalAdvance(self._full) if hasattr(fm, "horizontalAdvance") else fm.width(self._full)
        m = self.contentsMargins()
        return QtCore.QSize(w + m.left() + m.right() + 24, super().sizeHint().height())

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(60, super().minimumSizeHint().height())


class TransferWorker(QThread):
    """Moves an annotation onto other frames with ``root_transfer`` (one or several targets, off the GUI thread)."""

    result = Signal(object, object, str)                    # target dict, TransferResult or None, error
    progress = Signal(int, int)

    def __init__(self, src_rgb: Optional[np.ndarray], src_path: str, src_mask: np.ndarray, targets: List[dict],
                 mode: str = "affine", local: bool = True, search_px: int = 24):
        super().__init__()
        self.src_rgb, self.src_path, self.src_mask, self.targets = src_rgb, src_path, src_mask, targets
        self.mode, self.local, self.search_px = mode, local, search_px
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        src = self.src_rgb
        for i, t in enumerate(self.targets):
            if self._stop:
                return
            self.progress.emit(i, len(self.targets))
            try:
                if src is None:
                    src = read_rgb(self.src_path)
                    if src is None:
                        raise IOError(f"cannot read {self.src_path}")
                dst = t.get("rgb")
                dst = read_rgb(t["path"]) if dst is None else dst
                if dst is None:
                    raise IOError(f"cannot read {t['path']}")
                prob = t.get("prob")                         # the target's prediction: root evidence for the snap
                if prob is None and t.get("prob_path"):
                    prob = load_cached_prob(t["prob_path"])
                res = root_transfer.transfer(src, self.src_mask, dst, self.mode, self.local, self.search_px,
                                             dst_prob=prob)
                self.result.emit(t, res, "")
            except Exception as exc:
                self.result.emit(t, None, f"{type(exc).__name__}: {exc}")
        self.progress.emit(len(self.targets), len(self.targets))


class RootFinderDialog(QtWidgets.QDialog):
    """CanoPie's "Find canopy in other images", for roots: this image's annotation looked for in the other
    dates of its tube & window, at the exact spot (frame registration + a fit per root), then applied."""

    COLS = ("Use", "#", "Date", "Has", "Result", "Roots", "Match", "Move")

    def __init__(self, studio: "RootAnnotatorStudio", src_path: str, src_rgb: np.ndarray, src_mask: np.ndarray):
        super().__init__(studio)
        self.studio, self.src_path, self.src_rgb, self.src_mask = studio, src_path, src_rgb, src_mask
        self.worker: Optional[TransferWorker] = None
        self.results: Dict[str, Any] = {}
        self.setWindowTitle(f"Find these roots in other dates - {os.path.basename(src_path)}")
        self.setModal(False)
        self.resize(820, 520)
        lay = QtWidgets.QVBoxLayout(self)
        n_roots = max(0, cv2.connectedComponents(src_mask.astype(np.uint8))[0] - 1)
        key = temporal.parse_frame(src_path)
        lay.addWidget(QtWidgets.QLabel(
            f"Source: <b>{os.path.basename(src_path)}</b> - {n_roots} root segment(s) as shown now.<br>"
            + (f"Other dates of tube {key.tube}, window {key.loc}{' (' + key.site.strip('_-') + ')' if key.site.strip('_-') else ''}:"
               if key else "Neighbouring frames (no tube / window / date in the file name):")))

        files = studio.image_files
        if key is not None:
            cands = [p for p in temporal.window_frames(src_path, files) if p != src_path]
        else:
            i = files.index(src_path) if src_path in files else 0
            cands = [p for p in files[max(0, i - 10): i + 11] if p != src_path]
        self.paths = cands
        self.table = QtWidgets.QTableWidget(len(cands), len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        when = key.when if key else None
        for r, p in enumerate(cands):
            k = temporal.parse_frame(p)
            has = studio._own_annotation_kind(p)
            use = QtWidgets.QTableWidgetItem()
            later = (k is not None and when is not None and k.when > when) if key else True
            use.setCheckState(Qt.Checked if (later and not has) else Qt.Unchecked)
            vals = [str(files.index(p) + 1), (k.date if k else ""), has or "-", "", "", "", ""]
            self.table.setItem(r, 0, use)
            for c, v in enumerate(vals, start=1):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(self._open_row)
        lay.addWidget(self.table, 1)

        opts = QtWidgets.QHBoxLayout()
        opts.addWidget(QtWidgets.QLabel("Registration:"))
        self.mode_combo = QtWidgets.QComboBox()
        for m in root_transfer.MODES:
            self.mode_combo.addItem(root_transfer.MODE_LABELS[m], m)
        self.mode_combo.setCurrentIndex(root_transfer.MODES.index("affine"))
        self.mode_combo.setToolTip("Whole-frame fit after the phase-correlation shift (pystackreg, as in CanoPie).\n"
                                   "Affine also absorbs small zoom / shear between sessions.")
        opts.addWidget(self.mode_combo)
        self.local_check = QtWidgets.QCheckBox("Refine each root")
        self.local_check.setChecked(True)
        self.local_check.setToolTip("Match a crop around each root again (template match, then StackReg rigid),\n"
                                    "kept only where it fits better than the frame's transform.")
        opts.addWidget(self.local_check)
        opts.addWidget(QtWidgets.QLabel("search ±"))
        self.search_spin = QtWidgets.QSpinBox()
        self.search_spin.setRange(4, 120)
        self.search_spin.setValue(24)
        self.search_spin.setSuffix(" px")
        opts.addWidget(self.search_spin)
        opts.addStretch(1)
        opts.addWidget(QtWidgets.QLabel("Dates with their own polygons:"))
        self.own_combo = QtWidgets.QComboBox()
        self.own_combo.addItem("Add missing roots", "add")
        self.own_combo.addItem("Replace", "replace")
        self.own_combo.addItem("Skip", "skip")
        self.own_combo.setToolTip("Add missing roots: keep that date's polygons and add only the found roots (or parts)\n"
                                  "not already annotated there. Replace: its annotation becomes the found roots.")
        opts.addWidget(self.own_combo)
        self.save_check = QtWidgets.QCheckBox("Save to disk")
        self.save_check.setToolTip("Off: the results become unsaved edits of those images, to review before saving.")
        opts.addWidget(self.save_check)
        lay.addLayout(opts)

        bar = QtWidgets.QHBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setVisible(False)
        bar.addWidget(self.progress, 1)
        self.info = QtWidgets.QLabel("Tick the dates to search, then Find.")
        bar.addWidget(self.info, 2)
        self.btn_find = QtWidgets.QPushButton("🔎 Find")
        self.btn_find.clicked.connect(self._find)
        self.btn_stop = QtWidgets.QPushButton("■ Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_apply = QtWidgets.QPushButton("✓ Apply ticked results")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._apply)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        for b in (self.btn_find, self.btn_stop, self.btn_apply, btn_close):
            bar.addWidget(b)
        lay.addLayout(bar)

    def _rows_checked(self) -> List[int]:
        return [r for r in range(self.table.rowCount()) if self.table.item(r, 0).checkState() == Qt.Checked]

    def _find(self):
        try:
            rows = self._rows_checked()
            if not rows:
                self.info.setText("Tick at least one date.")
                return
            targets = []
            for r in rows:
                prob, prob_path = self.studio._prob_for(self.paths[r])
                targets.append({"path": self.paths[r], "row": r, "prob": prob, "prob_path": prob_path})
            self.worker = TransferWorker(self.src_rgb, self.src_path, self.src_mask, targets,
                                         self.mode_combo.currentData(), self.local_check.isChecked(),
                                         self.search_spin.value())
            self.worker.result.connect(self._on_result)
            self.worker.progress.connect(self._on_progress)
            self.worker.finished.connect(self._on_finished)
            self.progress.setRange(0, len(targets))
            self.progress.setValue(0)
            self.progress.setVisible(True)
            self.btn_find.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_apply.setEnabled(False)
            self.info.setText(f"Searching {len(targets)} date(s) ...")
            self.worker.start()
        except Exception as exc:
            self.info.setText(f"Could not start: {exc}")

    def _stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.info.setText("Stopping after the current date ...")

    def _on_progress(self, i: int, n: int):
        self.progress.setValue(i)

    def _on_result(self, t: dict, res, err: str):
        try:
            r = t["row"]
            if res is None:
                vals = ("error", "", "", err[:60])
            else:
                self.results[t["path"]] = res
                status = ("found" if res.placed else ("no match" if res.reg is not None and not res.reg.matched else "none"))
                low = sum(m.status == "low_score" for m in res.roots) if res.placed else 0
                vals = (status + (f", {low} weak" if low else ""), f"{res.placed}/{res.n_total}",
                        f"{res.reg.ncc:.2f}" if res.reg is not None else "",
                        root_transfer.describe_transform(res.reg.M, res.mask.shape) if res.reg is not None else "")
            for c, v in enumerate(vals, start=4):
                self.table.item(r, c).setText(v)
            if res is None or not res.placed:
                self.table.item(r, 0).setCheckState(Qt.Unchecked)
            self.table.resizeColumnsToContents()
        except Exception as exc:
            self.info.setText(f"Result not shown: {exc}")

    def _on_finished(self):
        self.progress.setVisible(False)
        self.btn_find.setEnabled(True)
        self.btn_stop.setEnabled(False)
        found = sum(1 for res in self.results.values() if res.placed)
        self.btn_apply.setEnabled(found > 0)
        self.info.setText(f"Found the roots on {found} date(s). Untick any you do not want, then Apply "
                          f"(double-click a row to look at that image first).")

    def _apply(self):
        try:
            done, skipped, saved = 0, [], 0
            for r in self._rows_checked():
                p = self.paths[r]
                res = self.results.get(p)
                if res is None or not res.placed:
                    continue
                mode = "replace"
                if self.studio._own_annotation_kind(p):
                    mode = self.own_combo.currentData()
                    if mode == "skip":
                        skipped.append(os.path.basename(p))
                        continue
                n = self.studio._paste_into(p, res.mask, f"found from {os.path.basename(self.src_path)}", mode)
                done += 1
                if self.save_check.isChecked() and self.studio._save_path_state(p):
                    saved += 1
                self.table.item(r, 4).setText(("applied" if mode == "replace" else f"added {n} missing")
                                              + (" + saved" if self.save_check.isChecked() else ""))
                self.table.item(r, 0).setCheckState(Qt.Unchecked)
            msg = (f"Applied to {done} date(s)" + (f", saved {saved}" if self.save_check.isChecked() else
                                                   " (unsaved edits: review, then save)")
                   + (f"; skipped {len(skipped)} with their own polygons" if skipped else ""))
            self.info.setText(msg)
            self.studio.status_info_lbl.setText("🔎 " + msg + ".  Ctrl+Z on an image undoes it there.")
            self.studio._refresh_counter()
        except Exception as exc:
            self.info.setText(f"Apply failed: {exc}")

    def _open_row(self, r: int, _c: int):
        try:
            p = self.paths[r]
            if p in self.studio.image_files:
                self.studio._load_image_by_index(self.studio.image_files.index(p))
        except Exception as exc:
            self.info.setText(f"Could not open: {exc}")

    def closeEvent(self, e):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(5000)
        super().closeEvent(e)


class _GotoSpinBox(QtWidgets.QSpinBox):
    """Frame-number box: Enter jumps there instead of firing the window's Save & Next shortcut."""

    def event(self, e):
        st = QtCore.QEvent.Type.ShortcutOverride if hasattr(QtCore.QEvent, "Type") else QtCore.QEvent.ShortcutOverride
        keys = ((Qt.Key.Key_Return, Qt.Key.Key_Enter) if hasattr(Qt, "Key") else (Qt.Key_Return, Qt.Key_Enter))
        if e.type() == st and e.key() in keys:
            e.accept()
            return True
        return super().event(e)


class SimFitWorker(QThread):
    """Measures the saved masks, writes the inpainted soils and fits the simulator to the annotations."""

    progress = Signal(str)
    done = Signal(object)                                   # dict: cfg, soil_paths, report, stats, error

    def __init__(self, project_root: str, folder: str, pairs: List[Tuple[str, str]], refit: bool):
        super().__init__()
        self.project_root, self.folder, self.pairs, self.refit = project_root, folder, pairs, refit

    def run(self):
        from exp_realism import synth_from_annotations as SY
        out = {"error": ""}
        try:
            sim_dir = os.path.join(self.folder, "_annotations", "_sim")
            soil_dir = os.path.join(sim_dir, "soil")
            os.makedirs(soil_dir, exist_ok=True)
            frames = []
            for i, (img, m) in enumerate(self.pairs):
                rgb = read_rgb(img)
                mk = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
                if rgb is None or mk is None or mk.shape != rgb.shape[:2]:
                    continue
                frames.append((os.path.splitext(os.path.basename(img))[0], rgb, mk > 127))
                if i % 10 == 0:
                    self.progress.emit(f"Reading your annotations {i + 1}/{len(self.pairs)} ...")
            soil_paths = []
            for stem, rgb, m in frames:                      # annotated roots inpainted away
                p = os.path.join(soil_dir, stem + ".png")
                if self.refit or not os.path.exists(p):
                    cv2.imwrite(p, cv2.cvtColor(SY.inpaint_roots(rgb, m)[0], cv2.COLOR_RGB2BGR))
                soil_paths.append(p)
            fitted = os.path.join(sim_dir, "realism_config_fitted.json")
            newest_mask = max(os.path.getmtime(m) for _, m in self.pairs)
            if not self.refit and os.path.exists(fitted) and os.path.getmtime(fitted) > newest_mask:
                with open(fitted, "r") as fh:
                    out.update(cfg=json.load(fh), reused=True)
                with open(os.path.join(sim_dir, "fit_report.json"), "r") as fh:
                    out.update(report=json.load(fh))
            else:
                self.progress.emit("Measuring root widths, lengths and colours ...")
                stats = SY.measure_annotations([(rgb, m) for _, rgb, m in frames])
                gen = SY.SynthGenerator(self.project_root, frames,
                                        base_cfg=SY.default_base_config(self.project_root, self.folder), stats=stats)
                t0 = time.time()
                gen.fit(progress=lambda what, i, n: self.progress.emit(
                    f"Fitting the simulator to your roots ({what}) {i}/{n} · ETA "
                    f"{_fmt_duration((time.time() - t0) / i * (n - i))}"))
                cfg = gen.cfg.to_dict()
                report = dict(gen.fit_report, stats={k: v for k, v in stats.items() if k not in ("color_deltas", "length_px")})
                for name, obj in (("realism_config_fitted.json", cfg), ("fit_report.json", report)):
                    with open(os.path.join(sim_dir, name), "w") as fh:
                        json.dump(obj, fh, indent=1)
                out.update(cfg=cfg, report=report, reused=False)
            out.update(soil_paths=soil_paths, sim_dir=sim_dir)
        except Exception as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"
        self.done.emit(out)


class FinetuneDialog(QtWidgets.QDialog):
    """Train the model on the saved masks: here (CPU, background) or via a Colab notebook (GPU)."""

    def __init__(self, parent, n_pairs: int, n_with_roots: int, runs: List[str], base: str,
                 cpu_minutes_per_step: float, cpu_cache_minutes: float,
                 synth_sources: Optional[List[Tuple[str, str]]] = None, synth_minutes_each: float = 0.1,
                 synth_fit_minutes: float = 5.0):
        super().__init__(parent)
        self._synth_each, self._synth_fit = synth_minutes_each, synth_fit_minutes
        self._n_train = max(0, n_pairs - max(5, round(finetune.VAL_FRAC * n_pairs)))
        self.setWindowTitle("Train the model on your corrections")
        self.choice = ""
        self._per_step, self._cache_min = cpu_minutes_per_step, cpu_cache_minutes
        lay = QtWidgets.QVBoxLayout(self)
        ok = n_pairs >= finetune.MIN_IMAGES
        head = QtWidgets.QLabel(
            f"<b>{n_pairs} corrected images</b> in this folder ({n_with_roots} with roots, "
            f"{n_pairs - n_with_roots} without - those teach 'no root here').<br>"
            "The encoder stays frozen; the decoder and head (3.3M of 47M weights) are retrained. "
            "Whole tube windows are held out to measure the gain; the current model is never changed - "
            "a new model appears in the Model list.<br><br>"
            "<b>Masks must be complete</b>: a root left unannotated is learnt as soil. "
            "About 30-50 images already adapt the model to this site; 100-200 is solid."
            + ("" if ok else f"<br><br><span style='color:#bf616a'>At least {finetune.MIN_IMAGES} saved masks "
                                "are needed.</span>"))
        head.setWordWrap(True)
        head.setMinimumWidth(520)
        lay.addWidget(head)

        form = QtWidgets.QFormLayout()
        self.base_combo = QtWidgets.QComboBox()
        self.base_combo.addItems(runs)
        if base in runs:
            self.base_combo.setCurrentText(base)
        self.base_combo.setToolTip("Fine-tunes always start from an original run (not from an earlier fine-tune).")
        form.addRow("Start from model:", self.base_combo)
        self.steps_spin = QtWidgets.QSpinBox()
        self.steps_spin.setRange(50, 20000)
        self.steps_spin.setSingleStep(100)
        self.steps_spin.setValue(finetune.DEFAULTS["cached"]["steps"])
        self.steps_spin.valueChanged.connect(self._update_estimate)
        form.addRow("Training steps (CPU):", self.steps_spin)
        self.synth_spin = QtWidgets.QSpinBox()
        self.synth_spin.setRange(0, 5)
        self.synth_spin.setValue(0)
        self.synth_spin.setSuffix(" per real image")
        self.synth_spin.setToolTip(
            "Simulated roots matched to yours (widths, length per frame, colour) drawn on the soil of your\n"
            "training frames, with your annotated roots inpainted away and ignored. Held-out windows are\n"
            "never used for them. 40% of every batch then comes from them. 0 = real images only.")
        self.synth_spin.valueChanged.connect(self._update_estimate)
        form.addRow("Synthetic images:", self.synth_spin)
        self.synth_source = QtWidgets.QComboBox()
        for label, key in (synth_sources or [("Fit to your annotations (in the job)", "fit")]):
            self.synth_source.addItem(label, key)
        self.synth_source.currentIndexChanged.connect(self._update_estimate)
        form.addRow("Simulator settings:", self.synth_source)
        self.estimate_lbl = QtWidgets.QLabel("")
        form.addRow("Here on this CPU:", self.estimate_lbl)
        form.addRow("On Colab (GPU):", QtWidgets.QLabel(
            f"{finetune.DEFAULTS['full']['steps']} steps with full augmentation, ~10-15 min on a T4"))
        lay.addLayout(form)
        self._update_estimate()

        btns = QtWidgets.QHBoxLayout()
        b_cpu = QtWidgets.QPushButton("▶ Train here (CPU)")
        b_cpu.clicked.connect(lambda: self._done("cpu"))
        b_colab = QtWidgets.QPushButton("☁ Create Colab notebook (GPU)")
        b_colab.clicked.connect(lambda: self._done("colab"))
        b_cancel = QtWidgets.QPushButton("Cancel")
        b_cancel.clicked.connect(self.reject)
        for b in (b_cpu, b_colab):
            b.setEnabled(ok)
            btns.addWidget(b)
        btns.addStretch(1)
        btns.addWidget(b_cancel)
        lay.addLayout(btns)

    def _update_estimate(self):
        m = self._cache_min + self.steps_spin.value() * self._per_step + 1.0
        if hasattr(self, "synth_spin") and self.synth_spin.value() > 0:
            m += self.synth_spin.value() * self._n_train * self._synth_each
            if self.synth_source.currentData() == "fit":
                m += self._synth_fit
        self.estimate_lbl.setText(f"~{m:.0f} min (prediction pauses meanwhile; you can keep annotating)")

    def _done(self, choice: str):
        self.choice = choice
        self.accept()


SYNTH_SECONDS_EACH = 5.0       # simulate one frame (~2.6 s) + encode it (~2 s), CPU
SYNTH_FIT_MINUTES = 4.0        # simulator fit by test renders (~230 s), CPU


def _mb(name: str):
    box = QtWidgets.QMessageBox
    return getattr(box.StandardButton, name) if hasattr(box, "StandardButton") else getattr(box, name)


def _fmt_duration(sec: float) -> str:
    sec = int(max(0, sec))
    if sec >= 3600:
        return f"{sec // 3600}h {sec % 3600 // 60:02d}m"
    if sec >= 60:
        return f"{sec // 60}m {sec % 60:02d}s"
    return f"{sec}s"


# ==============================================================================
# Synchronized Interactive Canvas Widget
# ==============================================================================

class SyncInteractiveCanvas(QtWidgets.QWidget):
    """Interactive canvas supporting synchronized pan/zoom, drawing, and custom overlays."""

    sig_viewport_changed = Signal(float, float, float)
    sig_stroke_completed = Signal(object, str)
    sig_stroke_progress = Signal(object, str)              # points so far, tool (throttled while dragging)
    sig_mouse_moved_pixel = Signal(int, int, float)

    def __init__(self, is_right_canvas: bool = False, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.is_right_canvas = is_right_canvas
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus if hasattr(Qt, "FocusPolicy") else Qt.StrongFocus)

        self._base_rgb: Optional[np.ndarray] = None
        self._rendered_view: Optional[QtGui.QImage] = None
        self._prob_map: Optional[np.ndarray] = None

        self._zoom: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0

        self._panning: bool = False
        self._last_mouse_pos = QtCore.QPoint()
        self._active_tool: str = "brush"
        self._brush_radius: int = 10
        self._drawing: bool = False
        self._current_stroke_pts: List[Tuple[int, int]] = []
        self._stroke_no_follow: bool = False
        self._stroke_ctrl: bool = False

        self._cursor_img_pos: Optional[Tuple[int, int]] = None
        self._preview_img: Optional[QtGui.QImage] = None
        self._preview_off: Tuple[int, int] = (0, 0)
        self._progress_timer = QtCore.QTimer(self)
        self._progress_timer.setSingleShot(True)
        self._progress_timer.setInterval(70)
        self._progress_timer.timeout.connect(self._emit_progress)

        self.setStyleSheet("background-color: #14171c; border: 1px solid #2e3440; border-radius: 6px;")

    def set_base_image(self, rgb: np.ndarray):
        self._base_rgb = _to_u8_rgb(rgb)
        self._fit_to_window()
        self.update()

    def set_rendered_view(self, qimg: Optional[QtGui.QImage]):
        self._rendered_view = qimg
        self.update()

    def set_prob_map(self, prob: Optional[np.ndarray]):
        self._prob_map = prob

    def set_preview(self, mask: Optional[np.ndarray]):
        """Snapped trace of the stroke being drawn (yellow), or None.  Only its bounding box is kept
        (a full-frame RGBA layer per mouse move was most of the preview's cost on big scans)."""
        if mask is None or not mask.any():
            self._preview_img = None
        else:
            x, y, w, h = cv2.boundingRect(np.ascontiguousarray(mask, dtype=np.uint8))
            rgba = np.zeros((h, w, 4), np.uint8)
            rgba[mask[y:y + h, x:x + w] > 0] = (255, 215, 0, 115)          # see-through: the root stays visible
            fmt = (QtGui.QImage.Format.Format_RGBA8888 if hasattr(QtGui.QImage, "Format")
                   else QtGui.QImage.Format_RGBA8888)
            self._preview_img = QtGui.QImage(rgba.data, w, h, 4 * w, fmt).copy()
            self._preview_off = (x, y)
        self.update()

    def _emit_progress(self):
        if self._drawing and self._current_stroke_pts:
            self.sig_stroke_progress.emit(self._stroke_points(), self._active_tool)

    def _stroke_points(self) -> List[Tuple[int, int]]:
        """The stroke so far with hand / pixel jitter averaged out (what is drawn, painted and traced)."""
        return _smooth_stroke(self._current_stroke_pts, self._brush_radius)

    def set_tool(self, tool_name: str):
        self._active_tool = tool_name
        self.update()

    def set_brush_radius(self, radius: int):
        self._brush_radius = max(1, int(radius))
        self.update()

    def set_viewport(self, zoom: float, offset_x: float, offset_y: float):
        self._zoom = zoom
        self._offset_x = offset_x
        self._offset_y = offset_y
        self.update()

    def _fit_to_window(self):
        if self._base_rgb is None:
            return
        H, W = self._base_rgb.shape[:2]
        vw = max(self.width() - 16, 64)
        vh = max(self.height() - 16, 64)
        self._zoom = min(vw / W, vh / H)
        self._offset_x = (self.width() - W * self._zoom) / 2.0
        self._offset_y = (self.height() - H * self._zoom) / 2.0
        self.sig_viewport_changed.emit(self._zoom, self._offset_x, self._offset_y)

    def _screen_to_image(self, sx: float, sy: float) -> Tuple[int, int]:
        ix = int((sx - self._offset_x) / self._zoom)
        iy = int((sy - self._offset_y) / self._zoom)
        return ix, iy

    # --------------------------------------------------------------------------
    # Mouse Events
    # --------------------------------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent):
        pos = event.position() if hasattr(event, "position") else event.pos()
        btn = event.button()

        if btn in (
            Qt.MouseButton.RightButton if hasattr(Qt, "MouseButton") else Qt.RightButton,
            Qt.MouseButton.MiddleButton if hasattr(Qt, "MouseButton") else Qt.MiddleButton,
        ) or self._active_tool == "pan":
            self._panning = True                             # wheel (middle) or right button held: drag the view
            self._last_mouse_pos = QtCore.QPoint(int(pos.x()), int(pos.y()))
            self.setCursor(Qt.CursorShape.ClosedHandCursor if hasattr(Qt, "CursorShape") else Qt.ClosedHandCursor)
            return

        if btn == (Qt.MouseButton.LeftButton if hasattr(Qt, "MouseButton") else Qt.LeftButton):
            if self._base_rgb is not None:
                ix, iy = self._screen_to_image(pos.x(), pos.y())
                H, W = self._base_rgb.shape[:2]
                if 0 <= ix < W and 0 <= iy < H:
                    self._drawing = True
                    self._current_stroke_pts = [(ix, iy)]
                    shift = (
                        Qt.KeyboardModifier.ShiftModifier
                        if hasattr(Qt, "KeyboardModifier")
                        else Qt.ShiftModifier
                    )
                    ctrl = (
                        Qt.KeyboardModifier.ControlModifier
                        if hasattr(Qt, "KeyboardModifier")
                        else Qt.ControlModifier
                    )
                    self._stroke_no_follow = (event.modifiers() & shift) == shift
                    self._stroke_ctrl = (event.modifiers() & ctrl) == ctrl
                    self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        pos = event.position() if hasattr(event, "position") else event.pos()

        if self._base_rgb is not None:
            ix, iy = self._screen_to_image(pos.x(), pos.y())
            H, W = self._base_rgb.shape[:2]
            if 0 <= ix < W and 0 <= iy < H:
                self._cursor_img_pos = (ix, iy)
                pval = float(self._prob_map[iy, ix]) if self._prob_map is not None else 0.0
                self.sig_mouse_moved_pixel.emit(ix, iy, pval)
            else:
                self._cursor_img_pos = None

        if self._panning:
            dx = pos.x() - self._last_mouse_pos.x()
            dy = pos.y() - self._last_mouse_pos.y()
            self._offset_x += dx
            self._offset_y += dy
            self._last_mouse_pos = QtCore.QPoint(int(pos.x()), int(pos.y()))
            self.sig_viewport_changed.emit(self._zoom, self._offset_x, self._offset_y)
            self.update()
            return

        if self._drawing and self._base_rgb is not None:
            ix, iy = self._screen_to_image(pos.x(), pos.y())
            self._current_stroke_pts.append((ix, iy))
            if not self._progress_timer.isActive():
                self._progress_timer.start()
            self.update()
            return

        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if self._panning:
            self._panning = False
            self.unsetCursor()

        if self._drawing and self._base_rgb is not None:
            self._drawing = False
            H, W = self._base_rgb.shape[:2]
            stroke_mask = np.zeros((H, W), dtype=np.uint8)
            thick = max(1, self._brush_radius * 2)
            pts = self._stroke_points()
            if len(pts) == 1:
                cv2.circle(stroke_mask, pts[0], self._brush_radius, 1, -1)
            else:
                for i in range(len(pts) - 1):
                    cv2.line(stroke_mask, pts[i], pts[i + 1], 1, thickness=thick)
                for pt in pts:
                    cv2.circle(stroke_mask, pt, self._brush_radius, 1, -1)

            self._current_stroke_pts = []
            self._progress_timer.stop()
            self._preview_img = None
            self.update()
            stroke = {"mask": stroke_mask, "pts": pts, "no_follow": self._stroke_no_follow,
                      "shift": self._stroke_no_follow, "ctrl": self._stroke_ctrl}
            self.sig_stroke_completed.emit(stroke, self._active_tool)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        if self._base_rgb is None:
            return
        delta = event.angleDelta().y()
        if delta == 0 or self._panning:                      # a wheel press / tilt, or dragging: no zoom jump
            return
        factor = 1.15 if delta > 0 else 0.87

        pos = event.position() if hasattr(event, "position") else event.pos()
        mx, my = pos.x(), pos.y()

        old_zoom = self._zoom
        new_zoom = float(np.clip(old_zoom * factor, 0.1, 40.0))

        self._offset_x = mx - (mx - self._offset_x) * (new_zoom / old_zoom)
        self._offset_y = my - (my - self._offset_y) * (new_zoom / old_zoom)
        self._zoom = new_zoom

        self.sig_viewport_changed.emit(self._zoom, self._offset_x, self._offset_y)
        self.update()

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self.update()

    # --------------------------------------------------------------------------
    # Paint Rendering
    # --------------------------------------------------------------------------
    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(
            QtGui.QPainter.RenderHint.Antialiasing
            if hasattr(QtGui.QPainter, "RenderHint")
            else QtGui.QPainter.Antialiasing
        )

        if self._base_rgb is None:
            painter.setPen(QtGui.QColor("#4c566a"))
            painter.drawText(
                self.rect(),
                _align_center(),
                "No Image Loaded\n(Click 'Open Image...' or Drag & Drop)",
            )
            return

        H, W = self._base_rgb.shape[:2]
        dest_rect = QtCore.QRectF(
            self._offset_x,
            self._offset_y,
            W * self._zoom,
            H * self._zoom,
        )

        # 1. Draw View Image
        if self._rendered_view is not None:
            painter.drawImage(dest_rect, self._rendered_view)
        else:
            qimg_base = _rgb_to_qimage(self._base_rgb)
            painter.drawImage(dest_rect, qimg_base)

        # 2. In-progress stroke: a translucent band exactly the brush's width (the image stays visible) and the
        #    stabilised path as a thin line; fainter still once the snapped smart-brush trace shows
        if self._drawing and len(self._current_stroke_pts) > 0:
            snapped = self._preview_img is not None
            rgb = (60, 180, 255) if self._active_tool == "eraser" else (255, 60, 60)
            band = QtGui.QColor(*rgb, 35 if snapped else 70)
            r = self._brush_radius * self._zoom
            pts_scr = [QtCore.QPointF(x * self._zoom + self._offset_x, y * self._zoom + self._offset_y)
                       for x, y in self._stroke_points()]
            no_pen = Qt.PenStyle.NoPen if hasattr(Qt, "PenStyle") else Qt.NoPen
            no_brush = Qt.BrushStyle.NoBrush if hasattr(Qt, "BrushStyle") else Qt.NoBrush
            if len(pts_scr) == 1:
                painter.setPen(no_pen)
                painter.setBrush(QtGui.QBrush(band))
                painter.drawEllipse(pts_scr[0], r, r)
            else:
                cap = Qt.PenCapStyle.RoundCap if hasattr(Qt, "PenCapStyle") else Qt.RoundCap
                join = Qt.PenJoinStyle.RoundJoin if hasattr(Qt, "PenJoinStyle") else Qt.RoundJoin
                solid = Qt.PenStyle.SolidLine if hasattr(Qt, "PenStyle") else Qt.SolidLine
                poly = QtGui.QPolygonF(pts_scr)
                painter.setBrush(no_brush)
                painter.setPen(QtGui.QPen(band, max(2.0, 2.0 * r), solid, cap, join))   # one path: no alpha build-up
                painter.drawPolyline(poly)
                painter.setPen(QtGui.QPen(QtGui.QColor(*rgb, 210), 1.3, solid, cap, join))
                painter.drawPolyline(poly)
            painter.setBrush(no_brush)

        if self._drawing and self._preview_img is not None:
            px, py = self._preview_off
            painter.drawImage(QtCore.QRectF(self._offset_x + px * self._zoom, self._offset_y + py * self._zoom,
                                            self._preview_img.width() * self._zoom,
                                            self._preview_img.height() * self._zoom), self._preview_img)

        # 3. Brush Cursor Circle Preview
        if self._cursor_img_pos is not None and self._active_tool in (
            "brush",
            "eraser",
            "pen",
        ):
            cx = self._cursor_img_pos[0] * self._zoom + self._offset_x
            cy = self._cursor_img_pos[1] * self._zoom + self._offset_y
            r = self._brush_radius * self._zoom
            c_color = (
                QtGui.QColor(255, 60, 60, 230)
                if self._active_tool in ("brush", "pen")
                else QtGui.QColor(80, 200, 255, 230)
            )
            painter.setPen(
                QtGui.QPen(
                    c_color,
                    1.5,
                    Qt.PenStyle.DashLine
                    if hasattr(Qt, "PenStyle")
                    else Qt.DashLine,
                )
            )
            painter.setBrush(
                Qt.BrushStyle.NoBrush
                if hasattr(Qt, "BrushStyle")
                else Qt.NoBrush
            )
            painter.drawEllipse(QtCore.QPointF(cx, cy), r, r)


# ==============================================================================
# Main Application Window
# ==============================================================================

class RootAnnotatorStudio(QtWidgets.QMainWindow):
    """Main Application Window for Interactive Minirhizotron Root Segmentation & Annotation."""

    def __init__(
        self,
        project_root: str,
        initial_model: str = "v8_full_real_and_syn_mitb3",
        initial_dir: str = "",
        initial_image: str = "",
        px_per_mm: float = 11.81,
        cache_dir: str = "",
        lookahead: int = 3,
        predict_all: bool = False,
    ):
        super().__init__()
        self.project_root = project_root
        self.px_per_mm = float(px_per_mm)
        self.active_model_name = initial_model
        self.initial_dir = initial_dir
        self.initial_image = initial_image
        self.cache_dir = cache_dir
        self.initial_lookahead = int(lookahead)
        self.initial_predict_all = bool(predict_all)

        self.model: Optional[Any] = None
        self.bundle: Optional[Dict[str, Any]] = None
        self.train_cfg: Dict[str, Any] = {}
        self.device: str = "cpu"  # Initial placeholder, updated on startup

        self.image_files: List[str] = []
        self.current_img_idx: int = -1
        self.current_rgb: Optional[np.ndarray] = None
        self.current_prob: Optional[np.ndarray] = None

        # Annotation of the image on screen, and of every image edited this session (by path)
        self.state: Optional[AnnotationState] = None
        self._state_path: Optional[str] = None
        self._states: Dict[str, AnnotationState] = {}
        self._saved_paths: set = set()
        # Previous date of the same window, aligned to this frame (cyan outline)
        self.prev_overlay: Optional[np.ndarray] = None
        self._carry_msg = ""
        self._carry_workers: List[CarryOverWorker] = []
        # tone / channel the live preview settled on: the stroke's final trace keeps them
        self._preview_lock: Dict[str, str] = {}
        self._nav_previewing = False                     # the canvases show a slider preview frame
        self._ft_proc: Optional[QtCore.QProcess] = None    # fine-tune running in the background
        self._ft_job, self._ft_run, self._ft_report = "", "", None
        self._ft_buf, self._ft_tail, self._ft_stopped = "", [], False
        self._sim_window = None                          # the simulator opened from here
        self._sim_worker: Optional[SimFitWorker] = None
        self._active_tool = "brush"
        self._tool_sizes: Dict[str, int] = {"draw": 10, "eraser": 10}

        # Probability maps already predicted this session: (cache tag, path) -> prob
        self._prob_mem: "OrderedDict[Tuple[str, str], np.ndarray]" = OrderedDict()
        self._prob_mem_budget = 512 * 1024 * 1024
        # Ridge maps of the image on screen (warmed in the background) for the root tracer
        self._ridge_cache: Optional[RidgeCache] = None
        self._prob_ridge_cache: Optional[RidgeCache] = None
        # Root / soil colour of the image on screen, learnt from the user's strokes
        self._colour_cue: Optional[ColourCue] = None
        self._preview_colour: Optional[ColourCue] = None
        self._preview_ctx: Optional[Tuple[np.ndarray, np.ndarray]] = None   # (segmented, prior) while a stroke is drawn
        self._heat_cache: Tuple[Any, Optional[np.ndarray]] = (None, None)  # heatmap blend of the frame on screen
        self._model_mask_cache: Optional[tuple] = None    # (prob, threshold, edge margin, mask)
        self._active_cache: Optional[tuple] = None        # (state, state.rev, model mask, active uint8)
        self._view_cache: Optional[tuple] = None          # (settings key, left rgb, right rgb) for partial redraws
        self._transfer_worker: Optional[TransferWorker] = None   # copy to prev/next, accept prev. date
        self._status_hold_until = 0.0
        self._export_worker: Optional[ExportWorker] = None
        self.current_centre: Optional[np.ndarray] = None   # the model's centre-line map of the image on screen
        self._centre_mem: "OrderedDict[Tuple[str, str], np.ndarray]" = OrderedDict()
        self._sep_result: Optional[tuple] = None           # (key, (Separation, length px, model map used))
        self._sep_worker: Optional[SeparationWorker] = None
        self._sep_wanted = False
        self._sep_view: Tuple[Any, Optional[np.ndarray]] = (None, None)
        self._edits_gen = 0                              # bumped by Delete all / Discard all: older results are dropped
        self._finder: Optional[RootFinderDialog] = None

        self._pred = PredictionService(cache_dir)
        self._pred.result_ready.connect(self._on_prediction_ready)
        self._pred.centre_ready.connect(self._on_centre_ready)
        self._pred.failed.connect(self._on_prediction_failed)
        self._pred.batch_progress.connect(self._on_batch_progress)
        self._pred.start()

        self.setWindowTitle("RoosterFly - Minirhizotron Root Annotation Studio")
        if os.path.exists(APP_ICON):
            self.setWindowIcon(QtGui.QIcon(APP_ICON))
        self.resize(1540, 960)
        self.setAcceptDrops(True)

        self._init_theme()
        self._build_ui()
        self.status_info_lbl.setText("Starting studio window ...")

        # Instant startup: window appears in <0.5s, then loads model & dataset
        QtCore.QTimer.singleShot(10, self._deferred_startup)

    def _init_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1a1e24;
                color: #e5e9f0;
                font-family: "Segoe UI", "Inter", sans-serif;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #2e3440;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #88c0d0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #2e3440;
                color: #eceff4;
                border: 1px solid #3b4252;
                border-radius: 4px;
                padding: 5px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #434c5e;
                border-color: #88c0d0;
            }
            QPushButton:pressed {
                background-color: #4c566a;
            }
            QPushButton:checked {
                background-color: #5e81ac;
                border-color: #81a1c1;
                color: #ffffff;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #242933;
                border: 1px solid #3b4252;
                border-radius: 4px;
                padding: 4px 8px;
                color: #eceff4;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #2e3440;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #bf616a;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #eceff4;
                border: 1px solid #d8dee9;
                width: 16px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #88c0d0;
            }
            QStatusBar {
                background-color: #14171c;
                border-top: 1px solid #2e3440;
                color: #d8dee9;
            }
        """)

    # --------------------------------------------------------------------------
    # UI Layout Construction
    # --------------------------------------------------------------------------
    def _build_ui(self):
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # 1. Top Ribbon: File, Navigation, Model & Tool Controls
        top_ribbon = QtWidgets.QHBoxLayout()

        btn_open_img = QtWidgets.QPushButton("📁 Open Image...")
        btn_open_img.setShortcut("Ctrl+O")
        btn_open_img.clicked.connect(self._on_open_image_dialog)
        top_ribbon.addWidget(btn_open_img)

        btn_open_dir = QtWidgets.QPushButton("📂 Open Folder...")
        btn_open_dir.setShortcut("Ctrl+F")
        btn_open_dir.clicked.connect(self._on_open_dir_dialog)
        top_ribbon.addWidget(btn_open_dir)

        self.btn_prev = QtWidgets.QPushButton("⏮ Prev (A)")
        self.btn_prev.setShortcut("A")
        self.btn_prev.clicked.connect(self._on_prev_image)
        top_ribbon.addWidget(self.btn_prev)

        self.btn_next = QtWidgets.QPushButton("Next (D) ⏭")
        self.btn_next.setShortcut("D")
        self.btn_next.clicked.connect(self._on_next_image)
        top_ribbon.addWidget(self.btn_next)

        self.file_counter_lbl = _ElidedLabel("No Images Loaded")
        self.file_counter_lbl.setStyleSheet("font-weight: bold; color: #88c0d0; padding: 0 8px;")
        top_ribbon.addWidget(self.file_counter_lbl)

        top_ribbon.addSpacing(15)

        top_ribbon.addWidget(QtWidgets.QLabel("Model:"))
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItem(self.active_model_name)
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        top_ribbon.addWidget(self.model_combo)
        btn_models = QtWidgets.QPushButton("↻")
        btn_models.setFixedWidth(30)
        btn_models.setToolTip("Re-read the model list (e.g. after a fine-tune finished on Colab).")
        btn_models.clicked.connect(self._on_refresh_models)
        top_ribbon.addWidget(btn_models)

        self.model_status_lbl = QtWidgets.QLabel("[LOADING]")
        self.model_status_lbl.setStyleSheet("color: #ebcb8b; font-weight: bold;")
        top_ribbon.addWidget(self.model_status_lbl)

        top_ribbon.addSpacing(15)

        self.tta_check = QtWidgets.QCheckBox("TTA (8-Aug)")
        self.tta_check.setToolTip("Test-Time Augmentation. Slower on CPU.")
        self.tta_check.setChecked(False)
        self.tta_check.toggled.connect(self._on_repredict_needed)
        top_ribbon.addWidget(self.tta_check)


        top_ribbon.addStretch(1)

        btn_save = QtWidgets.QPushButton("💾 Save (Ctrl+S)")
        btn_save.setShortcut("Ctrl+S")
        btn_save.setStyleSheet("background-color: #434c5e; color: #a3be8c; border: 1px solid #a3be8c; font-weight: bold;")
        btn_save.clicked.connect(self._on_save_current)
        top_ribbon.addWidget(btn_save)

        btn_save_next = QtWidgets.QPushButton("💾 Save & Next (Enter)")
        btn_save_next.setShortcut("Return")
        btn_save_next.setStyleSheet("background-color: #2e3440; color: #ebcb8b; border: 1px solid #ebcb8b; font-weight: bold;")
        btn_save_next.clicked.connect(self._on_save_and_next)
        top_ribbon.addWidget(btn_save_next)

        self.btn_export = QtWidgets.QPushButton("📊 Export CSV…")
        self.btn_export.setToolTip(
            "One row per image: name, site / tube / window / date / time / session, and the roots' length, projected\n"
            "and surface area, volume, mean diameter, coverage, length density, endpoints, branch points and length\n"
            "by diameter class (0 for images without roots). Annotated images only, or all images with the model's\n"
            "prediction where you have not annotated. Unsaved edits are exported as they are on screen.")
        self.btn_export.clicked.connect(self._on_export_csv)
        top_ribbon.addWidget(self.btn_export)

        main_layout.addLayout(top_ribbon)

        # 1b. Navigation: slide through the folder
        nav_ribbon = QtWidgets.QHBoxLayout()
        nav_ribbon.addWidget(QtWidgets.QLabel("🎞 Browse:"))
        self.nav_slider = QtWidgets.QSlider(_horizontal())
        self.nav_slider.setRange(0, 0)
        self.nav_slider.setTracking(True)
        self.nav_slider.setToolTip(
            "Drag to fly through the folder (raw frames preview while dragging; the frame under the handle\n"
            "loads with its prediction on release). Click it, then Left/Right step 1, PageUp/PageDown step 25."
        )
        self.nav_slider.setStyleSheet("QSlider::sub-page:horizontal { background: #5e81ac; border-radius: 3px; }")
        self.nav_slider.valueChanged.connect(self._on_nav_value_changed)
        self.nav_slider.sliderReleased.connect(self._on_nav_released)
        nav_ribbon.addWidget(self.nav_slider, stretch=1)

        nav_ribbon.addWidget(QtWidgets.QLabel("Go to #"))
        self.nav_spin = _GotoSpinBox()
        self.nav_spin.setRange(1, 1)
        self.nav_spin.setKeyboardTracking(False)
        self.nav_spin.setFixedWidth(90)
        self.nav_spin.setToolTip("Type a frame number and press Enter.")
        self.nav_spin.editingFinished.connect(lambda: self._goto_index(self.nav_spin.value() - 1))
        nav_ribbon.addWidget(self.nav_spin)
        self.nav_total_lbl = QtWidgets.QLabel("/ 0")
        nav_ribbon.addWidget(self.nav_total_lbl)
        nav_ribbon.addSpacing(15)
        transfer_tip = ("\nThe frames are registered first (phase correlation, then StackReg as in CanoPie) and each root is\n"
                        "matched again around itself, so the shapes land on the exact spot. Nothing is placed when the\n"
                        "two frames do not line up. The target's annotation is replaced (Ctrl+Z there undoes it).")
        for text, key, step, tip in (("⇤ Copy to prev", "Ctrl+Shift+A", -1, "Copy the roots shown here onto the previous image."),
                                     ("Copy to next ⇥", "Ctrl+Shift+D", 1, "Copy the roots shown here onto the next image.")):
            b = QtWidgets.QPushButton(text)
            b.setShortcut(key)
            b.setToolTip(f"{tip} and go there ({key}; Shift+click: copy but stay here).{transfer_tip}\n"
                         "If the two images do not line up (another window, a re-inserted tube) it offers to copy\n"
                         "at the same position instead.")
            b.clicked.connect(lambda _c=False, s=step: self._on_copy_to(s))
            nav_ribbon.addWidget(b)
        btn_accept = QtWidgets.QPushButton("⬇ Accept prev. date (P)")
        btn_accept.setShortcut("P")
        btn_accept.setToolTip("Take the roots of the newest earlier date of this tube & window that has an annotation\n"
                              "(the cyan outline) as this image's annotation, exactly as they were." + transfer_tip)
        btn_accept.clicked.connect(self._on_accept_prev)
        nav_ribbon.addWidget(btn_accept)
        btn_find = QtWidgets.QPushButton("🔎 Find in other dates…")
        btn_find.setToolTip("Look for this image's roots in the other dates of its tube & window and apply them\n"
                            "there (like CanoPie's 'Find canopy in other images'): a table of the matches first.")
        btn_find.clicked.connect(self._on_find_in_dates)
        nav_ribbon.addWidget(btn_find)
        nav_ribbon.addSpacing(15)
        self.btn_finetune = QtWidgets.QPushButton("🧠 Train on my corrections…")
        self.btn_finetune.setToolTip("Adapt the model to this site with the masks you saved (CPU here, or a Colab notebook).")
        self.btn_finetune.clicked.connect(self._on_finetune_clicked)
        nav_ribbon.addWidget(self.btn_finetune)
        self.btn_simulator = QtWidgets.QPushButton("🌱 Root simulator…")
        self.btn_simulator.setToolTip(
            "Open the root simulator fitted to the roots you annotated (widths, length per frame, colour),\n"
            "with your frames' soil (annotated roots inpainted away) as backgrounds. Shift+click refits.")
        self.btn_simulator.clicked.connect(self._on_simulator_clicked)
        nav_ribbon.addWidget(self.btn_simulator)
        main_layout.addLayout(nav_ribbon)

        # while dragging: show the raw frame shortly after the handle pauses; keys: load after a pause
        self._nav_preview_timer = QtCore.QTimer(self)
        self._nav_preview_timer.setSingleShot(True)
        self._nav_preview_timer.setInterval(60)
        self._nav_preview_timer.timeout.connect(self._show_nav_preview)
        self._nav_load_timer = QtCore.QTimer(self)
        self._nav_load_timer.setSingleShot(True)
        self._nav_load_timer.setInterval(180)
        self._nav_load_timer.timeout.connect(lambda: self._goto_index(self.nav_slider.value()))
        self._traits_timer = QtCore.QTimer(self)                # trait readout once edits pause
        self._traits_timer.setSingleShot(True)
        self._traits_timer.setInterval(250)
        self._traits_timer.timeout.connect(self._refresh_traits)

        # 2. Tool & Snapping Toolbar
        tool_ribbon = QtWidgets.QHBoxLayout()

        self.tool_btn_group = QtWidgets.QButtonGroup(self)
        self.btn_brush = QtWidgets.QPushButton("🖌 Smart Brush (B)")
        self.btn_brush.setCheckable(True)
        self.btn_brush.setChecked(True)
        self.btn_brush.setShortcut("B")
        self.btn_brush.setStyleSheet("QPushButton:checked { background-color: #bf616a; color: white; }")
        self.btn_brush.clicked.connect(lambda: self._set_active_tool("brush"))
        self.tool_btn_group.addButton(self.btn_brush)
        tool_ribbon.addWidget(self.btn_brush)

        self.btn_eraser = QtWidgets.QPushButton("🧹 Smart Eraser (E)")
        self.btn_eraser.setCheckable(True)
        self.btn_eraser.setShortcut("E")
        self.btn_eraser.setStyleSheet("QPushButton:checked { background-color: #5e81ac; color: white; }")
        self.btn_eraser.setToolTip(
            "Smart Eraser: removes the full width of the root under the stroke, and any part of a traced\n"
            "root the cut disconnects from where you painted it (trims an over-followed tail).\n"
            "Shift = exact brush area only.  Ctrl = the whole connected root."
        )
        self.btn_eraser.clicked.connect(lambda: self._set_active_tool("eraser"))
        self.tool_btn_group.addButton(self.btn_eraser)
        tool_ribbon.addWidget(self.btn_eraser)

        self.btn_pen = QtWidgets.QPushButton("✏ Manual Pen (M)")
        self.btn_pen.setCheckable(True)
        self.btn_pen.setShortcut("M")
        self.btn_pen.setToolTip("Exact drawing without automatic boundary snapping.")
        self.btn_pen.clicked.connect(lambda: self._set_active_tool("pen"))
        self.tool_btn_group.addButton(self.btn_pen)
        tool_ribbon.addWidget(self.btn_pen)

        self.btn_pan = QtWidgets.QPushButton("✋ Pan (H)")
        self.btn_pan.setCheckable(True)
        self.btn_pan.setShortcut("H")
        self.btn_pan.setToolTip("Pan canvas with left click (Right-click or Middle-click also pans anytime).")
        self.btn_pan.clicked.connect(lambda: self._set_active_tool("pan"))
        self.tool_btn_group.addButton(self.btn_pan)
        tool_ribbon.addWidget(self.btn_pan)

        tool_ribbon.addSpacing(15)

        self.size_lbl = QtWidgets.QLabel("Brush Size:")
        self.size_lbl.setToolTip("The eraser keeps its own size: switch tools and the slider shows that tool's size.\n"
                                 "[ / ] shrink / grow the active tool.")
        tool_ribbon.addWidget(self.size_lbl)
        self.brush_size_slider = QtWidgets.QSlider(_horizontal())
        self.brush_size_slider.setRange(2, 60)
        self.brush_size_slider.setValue(10)
        self.brush_size_slider.setFixedWidth(130)
        self.brush_size_slider.valueChanged.connect(self._on_brush_size_changed)
        tool_ribbon.addWidget(self.brush_size_slider)

        self.brush_size_spin = QtWidgets.QSpinBox()
        self.brush_size_spin.setRange(2, 60)
        self.brush_size_spin.setValue(10)
        self.brush_size_spin.valueChanged.connect(self.brush_size_slider.setValue)
        tool_ribbon.addWidget(self.brush_size_spin)
        for key, step in (("[", -1), ("]", 1)):
            shortcut_cls = QtWidgets.QShortcut if hasattr(QtWidgets, "QShortcut") else QtGui.QShortcut
            sc = shortcut_cls(QtGui.QKeySequence(key), self)
            sc.activated.connect(lambda st=step: self.brush_size_slider.setValue(
                self.brush_size_slider.value() + st * max(1, self.brush_size_slider.value() // 8)))

        tool_ribbon.addSpacing(15)

        btn_undo = QtWidgets.QPushButton("↶ Undo (Ctrl+Z)")
        btn_undo.setShortcut("Ctrl+Z")
        btn_undo.clicked.connect(self._on_undo)
        tool_ribbon.addWidget(btn_undo)

        btn_redo = QtWidgets.QPushButton("↷ Redo (Ctrl+Y)")
        btn_redo.setShortcut("Ctrl+Y")
        btn_redo.clicked.connect(self._on_redo)
        tool_ribbon.addWidget(btn_redo)

        btn_clear = QtWidgets.QPushButton("Clear Edits")
        btn_clear.setToolTip("Back to the model's prediction alone: drops your edits, carried roots and a\n"
                             "reloaded saved annotation of this image (Ctrl+Z undoes it).")
        btn_clear.clicked.connect(self._on_clear_user_edits)
        tool_ribbon.addWidget(btn_clear)

        btn_delete = QtWidgets.QToolButton()
        btn_delete.setText("🗑 Delete all")
        btn_delete.setToolTip("Delete every edit of this folder (project): saved masks, unsaved edits, carried roots -\n"
                              "every image shows the model's prediction only. Asks first; the saved files are moved\n"
                              "to _annotations/_deleted_<date-time> (move them back to restore).\n"
                              "The arrow: discard only the unsaved edits, or empty just this image.")
        btn_delete.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        action_cls = getattr(QtWidgets, "QAction", None) or QtGui.QAction
        menu = QtWidgets.QMenu(btn_delete)
        act_folder = action_cls("Delete all edits of this folder (model prediction only)…", self)
        act_folder.triggered.connect(self._on_delete_folder_edits)
        act_discard = action_cls("Discard unsaved edits on all images…", self)
        act_discard.triggered.connect(self._on_discard_all)
        act_empty = action_cls("Empty this image (no roots, model set aside too)", self)
        act_empty.triggered.connect(self._on_empty_image)
        for act in (act_folder, act_discard, act_empty):
            menu.addAction(act)
        btn_delete.setMenu(menu)
        btn_delete.clicked.connect(self._on_delete_folder_edits)
        tool_ribbon.addWidget(btn_delete)

        btn_fit = QtWidgets.QPushButton("🔍 Fit View")
        btn_fit.setShortcut("Space")
        btn_fit.clicked.connect(self._on_fit_views)
        tool_ribbon.addWidget(btn_fit)

        tool_ribbon.addStretch(1)
        main_layout.addLayout(tool_ribbon)

        # 2b. Root tracer settings (Smart Brush)
        tracer_ribbon = QtWidgets.QHBoxLayout()

        tracer_ribbon.addWidget(QtWidgets.QLabel("🧵 Smart Brush on:"))
        self.snap_mode_combo = QtWidgets.QComboBox()
        self.snap_mode_combo.addItems([
            "Image ridges (finds missed roots)",
            "Model probability map",
            "Exact Pen (No Snapping)",
        ])
        self.snap_mode_combo.setToolTip(
            "Image ridges: snaps the stroke onto the root's crest in the image itself and follows it -\n"
            "works where the model saw nothing.\n"
            "Model probability map: same tracer run on the model's (sub-threshold) probability.\n"
            "Exact Pen: paints the stroke as drawn."
        )
        self.snap_mode_combo.currentIndexChanged.connect(self._warm_prob_tracer)
        tracer_ribbon.addWidget(self.snap_mode_combo)

        tracer_ribbon.addSpacing(6)

        self.follow_combo = QtWidgets.QComboBox()
        self.follow_combo.addItems(["Follow root past stroke", "Stroke only"])
        self.follow_combo.setToolTip(
            "Follow: keeps tracing the root beyond both ends of your stroke (a single click traces\n"
            "the root both ways) until it fades, bends away, or joins an already segmented root.\n"
            "Hold Shift while painting for 'Stroke only' on that stroke."
        )
        tracer_ribbon.addWidget(self.follow_combo)

        tracer_ribbon.addSpacing(6)

        tracer_ribbon.addWidget(QtWidgets.QLabel("Roots:"))
        self.tone_combo = QtWidgets.QComboBox()
        self.tone_combo.addItems(["Auto tone", "Light (pale) roots", "Dark roots"])
        self.tone_combo.setToolTip("Whether roots are brighter or darker than the soil. Auto picks light unless dark is clearly stronger.")
        tracer_ribbon.addWidget(self.tone_combo)

        self.colour_check = QtWidgets.QCheckBox("🎨 Colour")
        self.colour_check.setChecked(True)
        self.colour_check.setToolTip(
            "Use root and soil colour as an extra cue. The colours are learnt on each image from your own\n"
            "annotation of it - roots you brushed or a reloaded saved mask, the soil beside them, what you\n"
            "erased - plus each stroke's own crest and the soil next to it, so it gets better as you go.\n"
            "Following prefers root-coloured steps and stops sooner on soil-coloured ones; the width is\n"
            "measured to where the colour turns to soil. Brightness and the ridge still lead."
        )
        tracer_ribbon.addWidget(self.colour_check)

        tracer_ribbon.addSpacing(6)

        tracer_ribbon.addWidget(QtWidgets.QLabel("Stop at:"))
        self.follow_tol_spin = QtWidgets.QDoubleSpinBox()
        self.follow_tol_spin.setRange(0.10, 0.95)
        self.follow_tol_spin.setSingleStep(0.05)
        self.follow_tol_spin.setDecimals(2)
        self.follow_tol_spin.setValue(0.40)
        self.follow_tol_spin.setToolTip(
            "Following stops where the root's ridge drops below this fraction of its strength under\n"
            "your stroke. Lower = follows fainter roots further; higher = stops sooner."
        )
        tracer_ribbon.addWidget(self.follow_tol_spin)

        tracer_ribbon.addSpacing(6)

        tracer_ribbon.addWidget(QtWidgets.QLabel("Max follow:"))
        self.max_follow_spin = QtWidgets.QSpinBox()
        self.max_follow_spin.setRange(0, 3000)
        self.max_follow_spin.setSingleStep(50)
        self.max_follow_spin.setValue(300)
        self.max_follow_spin.setSuffix(" px")
        self.max_follow_spin.setToolTip("Longest distance traced past each end of a stroke.")
        tracer_ribbon.addWidget(self.max_follow_spin)

        tracer_ribbon.addSpacing(6)

        tracer_ribbon.addWidget(QtWidgets.QLabel("Bridge gaps:"))
        self.max_gap_spin = QtWidgets.QSpinBox()
        self.max_gap_spin.setRange(0, 100)
        self.max_gap_spin.setValue(20)
        self.max_gap_spin.setSuffix(" px")
        self.max_gap_spin.setToolTip("Jump over a faded stretch up to this long when the root resumes in line beyond it.")
        tracer_ribbon.addWidget(self.max_gap_spin)

        tracer_ribbon.addStretch(1)

        tracer_ribbon.addWidget(QtWidgets.QLabel("Predict ahead:"))
        self.lookahead_spin = QtWidgets.QSpinBox()
        self.lookahead_spin.setRange(0, 20)
        self.lookahead_spin.setValue(max(0, min(20, self.initial_lookahead)))
        self.lookahead_spin.setToolTip(
            "Predict this many upcoming images in the background while you check the current one,\n"
            "so 'Next' opens instantly. Predictions are cached on disk and reused next session."
        )
        self.lookahead_spin.valueChanged.connect(lambda _v: self._request_predictions())
        tracer_ribbon.addWidget(self.lookahead_spin)

        self.btn_prepredict = QtWidgets.QPushButton("⚡ Pre-predict Folder")
        self.btn_prepredict.setCheckable(True)
        self.btn_prepredict.setToolTip(
            "Predict every image of the folder in the background (starting from the current one)\n"
            "and cache the maps on disk - with the model's centre-line maps, also for images predicted before\n"
            "they were kept (for 'Separate roots' and centre-line lengths in the export). Images you open are\n"
            "always served first. Click again to stop."
        )
        self.btn_prepredict.toggled.connect(self._on_prepredict_toggled)
        tracer_ribbon.addWidget(self.btn_prepredict)
        main_layout.addLayout(tracer_ribbon)

        # 3. Dual Side-by-Side Synchronized Canvases
        dual_canvas_layout = QtWidgets.QHBoxLayout()
        dual_canvas_layout.setSpacing(8)

        # LEFT PANEL: RGB Image
        left_vbox = QtWidgets.QVBoxLayout()
        left_title_bar = QtWidgets.QHBoxLayout()
        lbl_left_title = QtWidgets.QLabel("📷 RGB Minirhizotron Image")
        lbl_left_title.setStyleSheet("font-weight: bold; color: #eceff4; font-size: 13px;")
        left_title_bar.addWidget(lbl_left_title)

        left_title_bar.addStretch(1)
        self.show_contours_check = QtWidgets.QCheckBox("Contours (green)")
        self.show_contours_check.setChecked(True)
        self.show_contours_check.toggled.connect(self._update_overlays)
        left_title_bar.addWidget(self.show_contours_check)

        self.show_user_drawings_check = QtWidgets.QCheckBox("Edits (red) / carried (orange)")
        self.show_user_drawings_check.setChecked(True)
        self.show_user_drawings_check.toggled.connect(self._update_overlays)
        left_title_bar.addWidget(self.show_user_drawings_check)

        self.carry_check = QtWidgets.QCheckBox("Carry prev. date")
        self.carry_check.setChecked(True)
        self.carry_check.setToolTip(
            "On a frame with no annotation yet, re-trace the roots of the newest earlier date of the same\n"
            "tube & window (aligned to this frame) as a starting point."
        )
        left_title_bar.addWidget(self.carry_check)

        self.show_prev_check = QtWidgets.QCheckBox("Prev. date (cyan)")
        self.show_prev_check.setChecked(True)
        self.show_prev_check.toggled.connect(self._update_overlays)
        left_title_bar.addWidget(self.show_prev_check)

        left_vbox.addLayout(left_title_bar)

        self.canvas_left = SyncInteractiveCanvas(is_right_canvas=False)
        left_vbox.addWidget(self.canvas_left, stretch=1)
        dual_canvas_layout.addLayout(left_vbox, stretch=1)

        # RIGHT PANEL: Probability Map & Threshold Slider
        right_vbox = QtWidgets.QVBoxLayout()
        right_title_bar = QtWidgets.QHBoxLayout()
        lbl_right_title = QtWidgets.QLabel("🔮 Model Probability Map & Segmentation")
        lbl_right_title.setStyleSheet("font-weight: bold; color: #88c0d0; font-size: 13px;")
        right_title_bar.addWidget(lbl_right_title)

        right_title_bar.addStretch(1)
        right_title_bar.addWidget(QtWidgets.QLabel("Display View:"))
        self.view_mode_combo = QtWidgets.QComboBox()
        self.view_mode_combo.addItems([
            "Probability Heatmap (Jet)",
            "Blended RGB Overlay",
            "Binary Mask (B&W)",
            SEP_VIEW,
        ])
        self.view_mode_combo.setToolTip(
            f"{SEP_VIEW}: every root in its own colour with its centre line (white), separated where roots touch\n"
            "or cross along the model's centre-line map (its second output), recomputed after each edit.")
        self.view_mode_combo.currentIndexChanged.connect(self._update_overlays)
        right_title_bar.addWidget(self.view_mode_combo)

        right_title_bar.addWidget(QtWidgets.QLabel("Opacity:"))
        self.opacity_slider = QtWidgets.QSlider(_horizontal())
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(65)
        self.opacity_slider.setFixedWidth(80)
        self.opacity_slider.valueChanged.connect(self._update_overlays)
        right_title_bar.addWidget(self.opacity_slider)

        right_vbox.addLayout(right_title_bar)

        self.canvas_right = SyncInteractiveCanvas(is_right_canvas=True)
        right_vbox.addWidget(self.canvas_right, stretch=1)

        thresh_bar = QtWidgets.QHBoxLayout()
        lbl_thresh = QtWidgets.QLabel("⚡ Probability Threshold (Cutoff for this image):")
        lbl_thresh.setStyleSheet("font-weight: bold; color: #ebcb8b; font-size: 12px;")
        thresh_bar.addWidget(lbl_thresh)

        self.thresh_slider = QtWidgets.QSlider(_horizontal())
        self.thresh_slider.setRange(0, 100)
        self.thresh_slider.setValue(45)
        self.thresh_slider.valueChanged.connect(self._on_thresh_slider_changed)
        thresh_bar.addWidget(self.thresh_slider, stretch=1)

        self.thresh_spin = QtWidgets.QDoubleSpinBox()
        self.thresh_spin.setRange(0.00, 1.00)
        self.thresh_spin.setSingleStep(0.01)
        self.thresh_spin.setDecimals(2)
        self.thresh_spin.setValue(0.45)
        self.thresh_spin.valueChanged.connect(self._on_thresh_spin_changed)
        thresh_bar.addWidget(self.thresh_spin)

        thresh_bar.addSpacing(12)
        thresh_bar.addWidget(QtWidgets.QLabel("Ignore edges:"))
        self.edge_margin_spin = QtWidgets.QSpinBox()
        self.edge_margin_spin.setRange(0, 60)
        self.edge_margin_spin.setValue(0)
        self.edge_margin_spin.setSuffix(" px")
        self.edge_margin_spin.setToolTip(
            "Drop the model's predictions within this many pixels of the frame edge (edge and corner\n"
            "artefacts). Your own edits are kept. The camera's border bars are already masked when the\n"
            "model's inference settings say BORDER_MASK.")
        self.edge_margin_spin.valueChanged.connect(lambda _v: self._update_overlays())
        thresh_bar.addWidget(self.edge_margin_spin)

        right_vbox.addLayout(thresh_bar)
        dual_canvas_layout.addLayout(right_vbox, stretch=1)

        main_layout.addLayout(dual_canvas_layout, stretch=1)

        # 4. Synchronize Canvases
        self.canvas_left.sig_viewport_changed.connect(self.canvas_right.set_viewport)
        self.canvas_right.sig_viewport_changed.connect(self.canvas_left.set_viewport)

        self.canvas_left.sig_stroke_completed.connect(self._on_stroke_completed)
        self.canvas_right.sig_stroke_completed.connect(self._on_stroke_completed)
        self.canvas_left.sig_stroke_progress.connect(self._on_stroke_progress)
        self.canvas_right.sig_stroke_progress.connect(self._on_stroke_progress)

        self.canvas_left.sig_mouse_moved_pixel.connect(self._on_mouse_pixel_info)
        self.canvas_right.sig_mouse_moved_pixel.connect(self._on_mouse_pixel_info)

        # 5. Bottom Traits Readout & Status Bar
        traits_box = QtWidgets.QGroupBox("Live Root Trait Measurements & Diagnostics")
        traits_layout = QtWidgets.QHBoxLayout(traits_box)
        traits_layout.setContentsMargins(10, 4, 10, 6)

        self.trait_len_lbl = QtWidgets.QLabel("Total Length: -- mm (-- cm)")
        self.trait_len_lbl.setStyleSheet("font-weight: bold; color: #a3be8c;")
        traits_layout.addWidget(self.trait_len_lbl)

        traits_layout.addSpacing(20)
        self.trait_area_lbl = QtWidgets.QLabel("Surface Area: -- mm²")
        self.trait_area_lbl.setStyleSheet("font-weight: bold; color: #88c0d0;")
        traits_layout.addWidget(self.trait_area_lbl)

        traits_layout.addSpacing(20)
        self.trait_diam_lbl = QtWidgets.QLabel("Mean Diameter: -- mm")
        self.trait_diam_lbl.setStyleSheet("font-weight: bold; color: #ebcb8b;")
        traits_layout.addWidget(self.trait_diam_lbl)

        traits_layout.addSpacing(20)
        self.trait_cov_lbl = QtWidgets.QLabel("Coverage: -- %")
        self.trait_cov_lbl.setStyleSheet("font-weight: bold; color: #d08770;")
        traits_layout.addWidget(self.trait_cov_lbl)

        traits_layout.addSpacing(20)
        self.trait_count_lbl = QtWidgets.QLabel("Root Segments: --")
        self.trait_count_lbl.setStyleSheet("font-weight: bold; color: #b48ead;")
        traits_layout.addWidget(self.trait_count_lbl)

        traits_layout.addStretch(1)

        traits_layout.addWidget(QtWidgets.QLabel("Length from:"))
        self.length_src_combo = QtWidgets.QComboBox()
        self.length_src_combo.addItems(["Centre lines", "Mask skeleton"])
        self.length_src_combo.setToolTip(
            "Centre lines: roots separated along the model's centre-line map and measured one by one - a clump of\n"
            "touching roots is not one blob whose skeleton grows spurious spokes (shapes drawn by hand use their\n"
            "own shape). Mask skeleton: the skeleton of the mask as it is.")
        self.length_src_combo.currentIndexChanged.connect(lambda _i: self._traits_timer.start())
        traits_layout.addWidget(self.length_src_combo)
        traits_layout.addSpacing(12)

        traits_layout.addWidget(QtWidgets.QLabel("Resolution:"))
        self.px_per_mm_spin = QtWidgets.QDoubleSpinBox()
        self.px_per_mm_spin.setRange(0.1, 500.0)
        self.px_per_mm_spin.setValue(self.px_per_mm)
        self.px_per_mm_spin.setSuffix(" px/mm")
        self.px_per_mm_spin.valueChanged.connect(self._on_px_per_mm_changed)
        traits_layout.addWidget(self.px_per_mm_spin)

        main_layout.addWidget(traits_box)

        self.status_bar = self.statusBar()
        self.status_info_lbl = _ElidedLabel("Ready")
        self.status_pixel_lbl = QtWidgets.QLabel("X: --  Y: --  |  Model Prob: --")
        self.lookahead_lbl = QtWidgets.QLabel("")
        self.lookahead_lbl.setStyleSheet("color: #88c0d0; padding: 0 8px;")
        self.batch_bar = QtWidgets.QProgressBar()
        self.batch_bar.setFixedWidth(360)
        self.batch_bar.setTextVisible(True)
        self.batch_bar.setVisible(False)
        self.status_bar.addWidget(self.status_info_lbl, stretch=1)
        self.status_bar.addPermanentWidget(self.batch_bar)
        self.ft_bar = QtWidgets.QProgressBar()
        self.ft_bar.setFixedWidth(380)
        self.ft_bar.setVisible(False)
        self.ft_stop_btn = QtWidgets.QPushButton("■ Stop training")
        self.ft_stop_btn.setVisible(False)
        self.ft_stop_btn.clicked.connect(self._on_finetune_stop)
        self.status_bar.addPermanentWidget(self.ft_bar)
        self.status_bar.addPermanentWidget(self.ft_stop_btn)
        self.status_bar.addPermanentWidget(self.lookahead_lbl)
        self.status_bar.addPermanentWidget(self.status_pixel_lbl)

    # --------------------------------------------------------------------------
    # Deferred Startup Initialization (Non-blocking instant window launch)
    # --------------------------------------------------------------------------
    def _deferred_startup(self):
        self.status_info_lbl.setText("⏳ Initializing PyTorch & loading trained model weights ...")
        QtWidgets.QApplication.processEvents()

        # Check CUDA
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_status_lbl.setText(f"[{self.device.upper()}]")

        self._populate_model_combo()
        self._load_model(self.active_model_name)

        if self.initial_dir and os.path.isdir(self.initial_dir):
            self._load_directory(self.initial_dir)
        elif self.initial_image and os.path.isfile(self.initial_image):
            self._load_single_image(self.initial_image)
        else:
            default_gig = os.path.join(self.project_root, "Minirhizotron images", "Panama", "All GIG")
            if os.path.isdir(default_gig):
                self._load_directory(default_gig)

        if self.initial_predict_all and self.image_files:
            self.btn_prepredict.setChecked(True)

    def closeEvent(self, event: QtGui.QCloseEvent):
        if self._ft_running():
            ans = QtWidgets.QMessageBox.question(
                self, "Training running", "A fine-tune is still running. Stop it and close?",
                _mb("Yes") | _mb("No"), _mb("No"))
            if ans != _mb("Yes"):
                event.ignore()
                return
            self._on_finetune_stop()
        self._stash_state()
        dirty = [p for p, st in self._states.items() if st.dirty]
        if dirty:
            names = "\n".join(os.path.basename(p) for p in dirty[:8]) + ("\n..." if len(dirty) > 8 else "")
            ans = QtWidgets.QMessageBox.question(
                self, "Unsaved annotations",
                f"{len(dirty)} image(s) have unsaved edits:\n{names}\n\nSave them before closing?",
                _mb("Save") | _mb("Discard") | _mb("Cancel"), _mb("Save"),
            )
            if ans == _mb("Cancel"):
                event.ignore()
                return
            if ans == _mb("Save"):
                failed = [p for p in dirty if not self._save_path_state(p)]
                if failed:
                    QtWidgets.QMessageBox.warning(self, "Save failed", "Could not save:\n" + "\n".join(failed))
                    event.ignore()
                    return
        self.status_info_lbl.setText("Stopping background prediction ...")
        QtWidgets.QApplication.processEvents()
        self._pred.shutdown()
        super().closeEvent(event)

    # --------------------------------------------------------------------------
    # Model Loading
    # --------------------------------------------------------------------------
    def _populate_model_combo(self):
        realsoil_dir = os.path.join(self.project_root, "segmentation", "realsoil")
        runs = []
        if os.path.isdir(realsoil_dir):
            try:
                for d in os.listdir(realsoil_dir):
                    if d.startswith("v") or "lodo" in d or "full" in d:
                        runs.append(d)
            except Exception:
                pass
        if not runs:
            runs = [self.active_model_name]
        runs.sort()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for r in runs:
            self.model_combo.addItem(r)
        idx = self.model_combo.findText(self.active_model_name)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.model_combo.blockSignals(False)

    def _on_refresh_models(self):
        self._populate_model_combo()
        self.status_info_lbl.setText(f"Model list refreshed ({self.model_combo.count()} models).")

    # --------------------------------------------------------------------------
    # Fine-tune on the saved masks
    # --------------------------------------------------------------------------
    def _ft_running(self) -> bool:
        if self._ft_proc is None:
            return False
        idle = (QtCore.QProcess.ProcessState.NotRunning if hasattr(QtCore.QProcess, "ProcessState")
                else QtCore.QProcess.NotRunning)
        return self._ft_proc.state() != idle

    def _on_finetune_clicked(self):
        if self._ft_running():
            self.status_info_lbl.setText("A fine-tune is already running.")
            return
        if not self.image_files:
            return
        if self.state is not None and self.state.dirty:
            ans = QtWidgets.QMessageBox.question(
                self, "Unsaved edits", "Save the edits of this image first, so they are part of the training?",
                _mb("Save") | _mb("Discard") | _mb("Cancel"), _mb("Save"))
            if ans == _mb("Cancel"):
                return
            if ans == _mb("Save") and not self._on_save_current():
                return
        folder = os.path.dirname(self.image_files[0])
        pairs = finetune.find_corrected_pairs(folder)
        with_roots = 0
        for _, m in pairs:
            mk = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
            with_roots += int(mk is not None and (mk > 127).any())
        runs = [self.model_combo.itemText(i) for i in range(self.model_combo.count())
                if "ft_" not in self.model_combo.itemText(i)]
        base = finetune.base_of(self.project_root, self.active_model_name)
        secs = self._pred._pred_secs if self._pred._pred_secs else [2.5]
        enc = 0.85 * float(np.mean(secs))                              # encoder share of one prediction
        n_val = max(5, round(finetune.VAL_FRAC * len(pairs)))
        cache_min = ((len(pairs) - n_val) * 3 + n_val) * enc / 60.0
        dlg = FinetuneDialog(self, len(pairs), with_roots, runs or [self.active_model_name], base,
                             cpu_minutes_per_step=2.0 / 60.0, cpu_cache_minutes=cache_min,   # ~2 s/step measured
                             synth_sources=self._synth_sources(folder),
                             synth_minutes_each=SYNTH_SECONDS_EACH / 60.0, synth_fit_minutes=SYNTH_FIT_MINUTES)
        if dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec():
            base = dlg.base_combo.currentText()
            try:
                rel = os.path.relpath(folder, self.project_root)
            except ValueError:                              # another drive than the project
                rel = folder
            n_syn = dlg.synth_spin.value()
            syn_cfg, syn_fitted = self._synth_config(dlg.synth_source.currentData(), folder) if n_syn else (None, False)
            try:
                job = finetune.create_job(self.project_root, base, pairs, source=rel, synth_per_real=n_syn,
                                          synth_config=syn_cfg, synth_fitted=syn_fitted)
                nb_job, nb_copy = finetune.write_colab_notebook(self.project_root, job)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Could not prepare the training job", str(exc))
                return
            if dlg.choice == "cpu":
                self._start_finetune(job, dlg.steps_spin.value())
            else:
                self._show_colab_instructions(job, nb_copy)

    def _on_simulator_clicked(self):
        if self._sim_worker is not None and self._sim_worker.isRunning():
            self.status_info_lbl.setText("The simulator fit is still running ...")
            return
        if not self.image_files:
            return
        folder = os.path.dirname(self.image_files[0])
        pairs = finetune.find_corrected_pairs(folder)
        if not any((cv2.imread(m, cv2.IMREAD_GRAYSCALE) > 127).any() for _, m in pairs[:200] if os.path.exists(m)):
            QtWidgets.QMessageBox.information(self, "Root simulator", "Save some annotated frames with roots first: "
                                              "the simulator is fitted to them.")
            return
        mods = QtWidgets.QApplication.keyboardModifiers()
        shift = Qt.KeyboardModifier.ShiftModifier if hasattr(Qt, "KeyboardModifier") else Qt.ShiftModifier
        refit = (mods & shift) == shift
        self._sim_worker = SimFitWorker(self.project_root, folder, pairs, refit)
        self._sim_worker.progress.connect(self.status_info_lbl.setText)
        self._sim_worker.done.connect(self._on_sim_fit_done)
        self.btn_simulator.setEnabled(False)
        self.status_info_lbl.setText(f"🌱 Preparing the simulator from {len(pairs)} annotated frames ...")
        self._sim_worker.start()

    def _on_sim_fit_done(self, out: dict):
        self.btn_simulator.setEnabled(True)
        if out.get("error"):
            self.status_info_lbl.setText(f"Simulator fit failed: {out['error']}")
            QtWidgets.QMessageBox.warning(self, "Root simulator", out["error"])
            return
        rep = out.get("report") or {}
        try:                                             # an exception in a Qt slot would abort the app
            from exp_realism.gui_app import RootRealismStudioWindow
            from exp_realism.realism_params import RealismConfig
            cfg = RealismConfig.from_dict(out["cfg"])
            w = RootRealismStudioWindow(repo_root=self.project_root)
            w.prefill(cfg, out["soil_paths"], out_dir=os.path.join(out["sim_dir"], "dataset"),
                      note="Fitted to your annotations — Click '▶ RUN / UPDATE ROOT PREVIEW'")
            w.show()
        except Exception as exc:
            self.status_info_lbl.setText(f"Could not open the simulator: {exc}")
            QtWidgets.QMessageBox.warning(self, "Root simulator", f"{type(exc).__name__}: {exc}")
            return
        self._sim_window = w
        st = rep.get("stats") or {}
        self.status_info_lbl.setText(
            f"🌱 Simulator {'(previous fit) ' if out.get('reused') else ''}matched to your roots: width px "
            f"real p50 {(st.get('width_px_q') or {}).get('p50', '?')} vs simulated "
            f"{(rep.get('width_px_synthetic') or {}).get('p50', '?')} · diameter scale {rep.get('diam_scale', '?')} · "
            f"colour gain {rep.get('color_gain', '?')} · {len(out['soil_paths'])} soils (Shift+click to refit).")

    def _synth_sources(self, folder: str) -> List[Tuple[str, str]]:
        src = [("Fit to your annotations (in the job)", "fit")]
        if os.path.exists(os.path.join(folder, "_annotations", "_sim", "realism_config_fitted.json")):
            src.insert(0, ("Saved fit from 🌱 Root simulator", "saved"))
        if self._sim_window is not None and self._sim_window.isVisible():
            src.insert(0, ("Open simulator window's current settings", "window"))
        return src

    def _synth_config(self, key: str, folder: str) -> Tuple[Optional[dict], bool]:
        from exp_realism.synth_from_annotations import default_base_config
        if key == "window" and self._sim_window is not None:
            return self._sim_window.cfg.to_dict(), True
        if key == "saved":
            with open(os.path.join(folder, "_annotations", "_sim", "realism_config_fitted.json"), "r") as fh:
                return json.load(fh), True
        return default_base_config(self.project_root, folder) or None, False

    def _show_colab_instructions(self, job: str, notebook: str):
        rel_nb = os.path.relpath(notebook, self.project_root)
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Train on Colab (GPU)")
        box.setText(
            f"The training job and its notebook are on your Google Drive:\n\n  {rel_nb}\n\n"
            "1. Open drive.google.com -> My Drive -> Raiz_deep_learning -> notebooks\n"
            "2. Right-click the notebook -> Open with -> Google Colaboratory\n"
            "3. Runtime -> Change runtime type -> GPU, then Runtime -> Run all\n"
            "4. When it finishes (about 10-15 min), press ↻ next to 'Model' here and pick the new model.\n\n"
            "Drive needs a few minutes to upload the job folder before Colab can see it.")
        open_drive = box.addButton("Open Google Drive", QtWidgets.QMessageBox.ButtonRole.ActionRole
                                   if hasattr(QtWidgets.QMessageBox, "ButtonRole") else QtWidgets.QMessageBox.ActionRole)
        box.addButton(_mb("Ok"))
        box.exec_() if hasattr(box, "exec_") else box.exec()
        if box.clickedButton() is open_drive:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl("https://drive.google.com/drive/my-drive"))
        self.status_info_lbl.setText(f"Colab notebook ready: {rel_nb}")

    def _start_finetune(self, job: str, steps: int):
        self._ft_job = job
        with open(os.path.join(job, "config.json"), "r") as fh:
            self._ft_run = json.load(fh)["run_name"]
        self._ft_report, self._ft_buf, self._ft_tail, self._ft_stopped = None, "", [], False
        proc = QtCore.QProcess(self)
        proc.setProgram(sys.executable)
        proc.setArguments(["-u", "-m", "exp_realism.finetune_corrections", "--job", job, "--root", self.project_root,
                           "--device", "cpu", "--mode", "cached", "--steps", str(int(steps))])
        proc.setWorkingDirectory(self.project_root)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        proc.setProcessEnvironment(env)
        proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels
                                   if hasattr(QtCore.QProcess, "ProcessChannelMode") else QtCore.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_finetune_output)
        proc.finished.connect(self._on_finetune_finished)
        self._ft_proc = proc
        self._pred.pause(True)
        self.btn_finetune.setEnabled(False)
        self.ft_bar.setRange(0, 0)
        self.ft_bar.setFormat("Fine-tune: loading ...")
        self.ft_bar.setVisible(True)
        self.ft_stop_btn.setVisible(True)
        self.status_info_lbl.setText(f"🧠 Training {self._ft_run} in the background (predictions paused).")
        proc.start()

    def _on_finetune_output(self):
        self._ft_buf += bytes(self._ft_proc.readAllStandardOutput()).decode("utf-8", "replace")
        *lines, self._ft_buf = self._ft_buf.split("\n")
        for line in lines:
            line = line.rstrip()
            if not line.startswith("FT "):
                if line:
                    self._ft_tail = (self._ft_tail + [line])[-25:]
                continue
            try:
                ev = json.loads(line[3:])
            except ValueError:
                continue
            ph = ev.get("phase")
            eta = f" · ETA {_fmt_duration(ev['eta_s'])}" if "eta_s" in ev else ""
            if ph == "cache":
                self.ft_bar.setRange(0, ev["n"])
                self.ft_bar.setValue(ev["i"])
                self.ft_bar.setFormat(f"Fine-tune: features {ev['i']}/{ev['n']}{eta}")
            elif ph in ("synth_fit", "synth"):
                self.ft_bar.setRange(0, ev["n"])
                self.ft_bar.setValue(ev["i"])
                what = "fitting the simulator" if ph == "synth_fit" else "synthetic frames"
                self.ft_bar.setFormat(f"Fine-tune: {what} {ev['i']}/{ev['n']}{eta}")
            elif ph == "train":
                self.ft_bar.setRange(0, ev["steps"])
                self.ft_bar.setValue(ev["step"])
                self.ft_bar.setFormat(f"Fine-tune: step {ev['step']}/{ev['steps']} · loss {ev['loss']:.3f}{eta}")
            elif ph == "base" and ev.get("score"):
                self.status_info_lbl.setText(f"🧠 Held-out windows, current model: F1 {ev['score']['f1']:.2f}")
            elif ph == "val":
                self.status_info_lbl.setText(f"🧠 Step {ev['step']}: held-out F1 {ev['score']['f1']:.2f} "
                                             f"(threshold {ev['score']['thresh']})")
            elif ph == "done":
                self._ft_report = ev.get("report")
            elif ph == "error":
                self._ft_tail.append(ev.get("msg", ""))

    def _on_finetune_stop(self):
        if self._ft_running():
            self._ft_stopped = True
            self._ft_proc.kill()
            self._ft_proc.waitForFinished(5000)

    def _on_finetune_finished(self, *_):
        self._pred.pause(False)
        self.btn_finetune.setEnabled(True)
        self.ft_bar.setVisible(False)
        self.ft_stop_btn.setVisible(False)
        partial = os.path.join(self.project_root, "segmentation", "realsoil", "." + self._ft_run + ".partial")
        if os.path.isdir(partial):
            shutil.rmtree(partial, ignore_errors=True)
        rep = self._ft_report
        if self._ft_stopped or rep is None:
            msg = "Training stopped." if self._ft_stopped else "Training failed:\n\n" + "\n".join(self._ft_tail[-12:])
            self.status_info_lbl.setText(msg.splitlines()[0])
            if not self._ft_stopped:
                QtWidgets.QMessageBox.warning(self, "Fine-tune failed", msg)
            return
        self._populate_model_combo()
        b, n = rep.get("base") or {}, rep.get("new") or {}
        if b and n:
            txt = (f"Held-out: {rep['n_val']} images from tube windows not used for training.\n\n"
                   f"  current model ({rep['base_run']}): F1 {b['f1']:.2f}  (precision {b['precision']:.2f}, "
                   f"recall {b['recall']:.2f}, best threshold {b['thresh']})\n"
                   f"  new model: F1 {n['f1']:.2f}  (precision {n['precision']:.2f}, recall {n['recall']:.2f}, "
                   f"threshold {n['thresh']})\n\n"
                   f"Trained on {rep['n_train']} images"
                   + (f" + {rep['n_synthetic']} synthetic" if rep.get("n_synthetic") else "")
                   + f" in {rep['minutes']} min.")
        else:
            txt = f"Trained on {rep['n_train']} images in {rep['minutes']} min (no held-out images to compare)."
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle(f"New model: {rep['run']}")
        box.setText(txt + "\n\nSwitch the annotator to the new model?")
        switch = box.addButton("Switch to new model", QtWidgets.QMessageBox.ButtonRole.AcceptRole
                               if hasattr(QtWidgets.QMessageBox, "ButtonRole") else QtWidgets.QMessageBox.AcceptRole)
        box.addButton("Keep current", QtWidgets.QMessageBox.ButtonRole.RejectRole
                      if hasattr(QtWidgets.QMessageBox, "ButtonRole") else QtWidgets.QMessageBox.RejectRole)
        box.exec_() if hasattr(box, "exec_") else box.exec()
        self.status_info_lbl.setText(f"✓ New model {rep['run']} is in the Model list.")
        if box.clickedButton() is switch:
            i = self.model_combo.findText(rep["run"])
            if i >= 0:
                self.model_combo.setCurrentIndex(i)

    def _on_model_combo_changed(self, idx: int):
        run_name = self.model_combo.currentText()
        if run_name and run_name != self.active_model_name:
            self._load_model(run_name)
            self._on_repredict_needed()

    def _load_model(self, run_name: str):
        try:
            self.status_info_lbl.setText(f"Loading checkpoint {run_name} ...")
            QtWidgets.QApplication.processEvents()

            bundle = load_model_bundle(self.project_root, run_name, self.device)
            self.bundle = bundle
            self.model = bundle["model"]
            self.train_cfg = bundle["train_cfg"]
            if bundle["thresh"] is not None:
                self.thresh_spin.setValue(bundle["thresh"])

            self.active_model_name = run_name
            self._pred.configure(bundle, self.tta_check.isChecked())
            self.model_status_lbl.setText(f"[{run_name} | {self.device.upper()}]")
            self.status_info_lbl.setText(f"✓ Loaded {run_name} successfully ({self.device.upper()}).")
        except Exception as exc:
            run_dir = os.path.join(self.project_root, "segmentation", "realsoil", run_name)
            hint = ("" if os.path.isdir(run_dir) else
                    f"\n\nNo model folder {run_dir}.\nPut a trained run there (seg_best.pt, train_config.json, "
                    "prep_stats.json, inference_settings.json - see segmentation/realsoil/README.md), or start "
                    "with --root <project folder that holds segmentation/realsoil>. The tools work without a "
                    "model; predictions do not.")
            self.status_info_lbl.setText(f"Error loading model: {exc}")
            QtWidgets.QMessageBox.warning(self, "Model Loading Error", str(exc) + hint)

    # --------------------------------------------------------------------------
    # File & Directory Operations
    # --------------------------------------------------------------------------
    def _on_open_image_dialog(self):
        fpath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Minirhizotron Image", self.project_root, "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )
        if fpath:
            self._load_single_image(fpath)

    def _on_open_dir_dialog(self):
        dpath = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Minirhizotron Folder", self.project_root)
        if dpath:
            self._load_directory(dpath)

    def _load_directory(self, folder_path: str):
        try:
            files = list_images(folder_path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Folder Error", str(exc))
            return
        if not files:
            QtWidgets.QMessageBox.information(self, "No Images", f"No supported image files found in:\n{folder_path}")
            return
        self._set_image_list(files)
        self._load_image_by_index(0)

    def _load_single_image(self, fpath: str):
        files = list_images(os.path.dirname(fpath))
        self._set_image_list(files if files else [fpath])
        try:
            self.current_img_idx = self.image_files.index(fpath)
        except ValueError:
            self._set_image_list([fpath])
            self.current_img_idx = 0
        self._load_image_by_index(self.current_img_idx)

    def _set_image_list(self, files: List[str]):
        if files != self.image_files and self.btn_prepredict.isChecked():
            self.btn_prepredict.setChecked(False)           # a folder batch belongs to its folder
        self.image_files = files
        self.current_img_idx = 0
        n = max(1, len(files))
        for wdg, lo, hi in ((self.nav_slider, 0, n - 1), (self.nav_spin, 1, n)):
            wdg.blockSignals(True)
            wdg.setRange(lo, hi)
            wdg.blockSignals(False)
        self.nav_slider.setPageStep(25)
        self.nav_total_lbl.setText(f"/ {len(files):,}")

    def _stash_state(self):
        """Keep the edits of the image on screen for this session (Prev/Next brings them back)."""
        if self._state_path and self.state is not None and (self.state.dirty or self.state.has_edits()):
            self._states[self._state_path] = self.state

    def _load_image_by_index(self, idx: int):
        if not self.image_files or idx < 0 or idx >= len(self.image_files):
            return
        self._stash_state()
        self.current_img_idx = idx
        fpath = self.image_files[idx]

        self.btn_prev.setEnabled(idx > 0)
        self.btn_next.setEnabled(idx < len(self.image_files) - 1)
        self._nav_load_timer.stop()
        for wdg, v in ((self.nav_slider, idx), (self.nav_spin, idx + 1)):
            wdg.blockSignals(True)
            wdg.setValue(v)
            wdg.blockSignals(False)

        bgr = cv2.imread(fpath)
        if bgr is None:
            self.status_info_lbl.setText(f"Error reading image: {fpath}")
            return
        self.current_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = self.current_rgb.shape[:2]

        st = self._states.get(fpath)
        if st is None:
            st = AnnotationState((H, W))
            saved = self._read_saved_mask(fpath)
            if saved is not None and saved.shape == (H, W):
                st.load_saved(saved, note="saved")
                self._saved_paths.add(fpath)
        self.state, self._state_path = st, fpath
        self.prev_overlay, self._carry_msg = None, ""
        self.current_prob = None
        self.current_centre = None
        self.canvas_left.set_prob_map(None)
        self.canvas_right.set_prob_map(None)

        # Ridge maps for the smart brush, computed while the user looks at the image
        if self._ridge_cache is not None:
            self._ridge_cache.cancel()
        self._ridge_cache = RidgeCache.from_rgb(self.current_rgb)
        self._prob_ridge_cache = None
        self._colour_cue = ColourCue(self.current_rgb)
        threading.Thread(target=lambda rc=self._ridge_cache, cc=self._colour_cue: (rc.warm(), cc.warm()),
                         daemon=True).start()

        self.canvas_left.set_base_image(self.current_rgb)
        self.canvas_right.set_base_image(self.current_rgb)
        self.canvas_left._fit_to_window()
        self._update_overlays()
        self._refresh_counter()

        self._request_predictions()
        if st.base_override is not None and not st.dirty:
            self.status_info_lbl.setText("📂 Reopened your saved annotation of this image (the threshold no longer "
                                         "changes it; 'Clear Edits' goes back to the model).")
        self._start_carry_over()

    def _goto_index(self, idx: int):
        self._nav_preview_timer.stop()
        self._nav_load_timer.stop()
        if 0 <= idx < len(self.image_files) and (idx != self.current_img_idx or self._nav_previewing):
            self._load_image_by_index(idx)
        self._nav_previewing = False

    def _on_nav_value_changed(self, idx: int):
        if not self.image_files:
            return
        name = os.path.basename(self.image_files[idx])
        key = temporal.parse_frame(name)
        where = f"   (tube {key.tube}, window {key.loc}, {key.date})" if key else ""
        self.file_counter_lbl.setText(f"Image {idx + 1} / {len(self.image_files)}:  {name}{where}")
        self.nav_spin.blockSignals(True)
        self.nav_spin.setValue(idx + 1)
        self.nav_spin.blockSignals(False)
        if self.nav_slider.isSliderDown():
            self._nav_preview_timer.start()                 # restarts while the handle keeps moving
        else:
            self._nav_load_timer.start()                    # keys / clicks on the groove: load after a pause

    def _show_nav_preview(self):
        """Raw frame under the slider handle (no prediction, no edits) while dragging."""
        idx = self.nav_slider.value()
        if not self.nav_slider.isSliderDown() or not (0 <= idx < len(self.image_files)):
            return
        bgr = cv2.imread(self.image_files[idx])
        if bgr is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._nav_previewing = True
        for c in (self.canvas_left, self.canvas_right):
            c.set_prob_map(None)
            if c._base_rgb is None or c._base_rgb.shape != rgb.shape:
                c.set_base_image(rgb)
            else:
                c._base_rgb = rgb                            # keep the zoom while flicking through frames
            c.set_rendered_view(_rgb_to_qimage(rgb))

    def _on_nav_released(self):
        self._goto_index(self.nav_slider.value())

    def _on_prev_image(self):
        if self.current_img_idx > 0:
            self._load_image_by_index(self.current_img_idx - 1)

    def _on_next_image(self):
        if self.current_img_idx < len(self.image_files) - 1:
            self._load_image_by_index(self.current_img_idx + 1)

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            fpath = urls[0].toLocalFile()
            if os.path.isdir(fpath):
                self._load_directory(fpath)
            elif os.path.isfile(fpath):
                self._load_single_image(fpath)

    # --------------------------------------------------------------------------
    # Prediction Handling (background queue + memory / disk cache)
    # --------------------------------------------------------------------------
    def _current_path(self) -> Optional[str]:
        if 0 <= self.current_img_idx < len(self.image_files):
            return self.image_files[self.current_img_idx]
        return None

    def _current_tag(self) -> str:
        return cache_tag(self.active_model_name, self.tta_check.isChecked())

    def _lookahead_paths(self) -> List[str]:
        i = self.current_img_idx
        return self.image_files[i + 1: i + 1 + self.lookahead_spin.value()] if i >= 0 else []

    def _remember_prob(self, tag: str, path: str, prob: np.ndarray):
        key = (tag, path)
        self._prob_mem.pop(key, None)
        self._prob_mem[key] = prob
        used = sum(p.nbytes for p in self._prob_mem.values())
        while used > self._prob_mem_budget and len(self._prob_mem) > 1:
            _, old = self._prob_mem.popitem(last=False)
            used -= old.nbytes

    def _request_predictions(self):
        """Show the current image's map if known, else queue it first; queue the look-ahead."""
        path = self._current_path()
        if path is None or self.bundle is None or self.current_rgb is None:
            return
        tag = self._current_tag()
        ahead = [p for p in self._lookahead_paths() if (tag, p) not in self._prob_mem]
        prob = self._prob_mem.get((tag, path))
        if prob is not None:
            self._prob_mem.move_to_end((tag, path))
            centre = self._centre_mem.get((tag, path))
            if centre is not None:
                self.current_centre = centre
            self._pred.focus(None if centre is not None else path, ahead)   # the service brings the centre lines
            if self.current_prob is not prob:
                self._apply_prediction(prob, "ready (predicted ahead)")
        else:
            self.status_info_lbl.setText("⏳ Predicting root probability map (background) ...")
            self._pred.focus(path, ahead)
        self._update_lookahead_lbl()

    def _on_prediction_ready(self, tag: str, path: str, prob: np.ndarray, secs: float, from_disk: bool):
        self._remember_prob(tag, path, prob)
        if tag == self._current_tag() and path == self._current_path() and self.current_prob is None:
            self._apply_prediction(prob, "loaded from disk cache" if from_disk else f"predicted in {secs:.2f}s")
        self._update_lookahead_lbl()

    def _apply_prediction(self, prob_map: np.ndarray, how: str):
        self.current_prob = prob_map
        self._prob_ridge_cache = None
        self._warm_prob_tracer()
        self.canvas_left.set_prob_map(prob_map)
        self.canvas_right.set_prob_map(prob_map)

        th = self.thresh_spin.value()
        px_above = int((prob_map >= th).sum())
        cov = (px_above / float(prob_map.size)) * 100.0

        if time.time() >= self._status_hold_until:              # a copy's result stays readable a moment
            self.status_info_lbl.setText(
                f"✓ Prediction {how} | {px_above:,} root px ({cov:.2f}% coverage) at threshold {th:.2f}"
                + (f"   |   {self._carry_msg}" if self._carry_msg else "")
            )
        self._update_overlays()

    def _on_prediction_failed(self, path: str, err_msg: str):
        if path == self._current_path():
            self.status_info_lbl.setText(f"Inference failed: {err_msg}")
            QtWidgets.QMessageBox.critical(self, "Inference Error", err_msg)
        else:
            self.status_info_lbl.setText(f"Background prediction failed for {os.path.basename(path)}: {err_msg}")

    def _on_repredict_needed(self):
        if self.bundle is not None:
            self._pred.configure(self.bundle, self.tta_check.isChecked())
        if self.current_rgb is not None:
            self.current_prob = None
            self.current_centre = None
            self.canvas_left.set_prob_map(None)
            self.canvas_right.set_prob_map(None)
            self._update_overlays()
            self._request_predictions()
        if self.btn_prepredict.isChecked():                 # restart the folder pass for the new model / TTA
            self._on_prepredict_toggled(True)

    def _update_lookahead_lbl(self):
        ahead = self._lookahead_paths()
        if not ahead:
            self.lookahead_lbl.setText("")
            return
        tag = self._current_tag()
        ready = sum((tag, p) in self._prob_mem for p in ahead)
        self.lookahead_lbl.setText(f"Next {len(ahead)}: {ready} ready")

    def _on_prepredict_toggled(self, checked: bool):
        if not checked:
            self._pred.cancel_batch()
            self.batch_bar.setVisible(False)
            self.status_info_lbl.setText("Folder pre-prediction stopped (maps predicted so far stay cached).")
            return
        if not self.image_files or self.bundle is None:
            self.btn_prepredict.blockSignals(True)
            self.btn_prepredict.setChecked(False)
            self.btn_prepredict.blockSignals(False)
            return
        i = max(0, self.current_img_idx)
        self._pred.start_batch(self.image_files[i:] + self.image_files[:i])   # from here on, then wrap
        self.batch_bar.setVisible(True)
        where = self.cache_dir or os.path.join(os.path.dirname(self.image_files[0]), "_annotations", "_prob_cache")
        self.status_info_lbl.setText(f"⚡ Pre-predicting {len(self.image_files):,} images in the background -> {where}")

    def _on_batch_progress(self, done: int, total: int, sec_per_img: float):
        if total <= 0 or not self.btn_prepredict.isChecked():
            return
        self.batch_bar.setRange(0, total)
        self.batch_bar.setValue(done)
        eta = f" · ETA {_fmt_duration(sec_per_img * (total - done))}" if sec_per_img > 0 else ""
        rate = f" · {sec_per_img:.1f} s/img" if sec_per_img > 0 else ""
        self.batch_bar.setFormat(f"Pre-predicted {done:,}/{total:,}{rate}{eta}")
        if done >= total:
            self.btn_prepredict.blockSignals(True)
            self.btn_prepredict.setChecked(False)
            self.btn_prepredict.blockSignals(False)
            self.batch_bar.setVisible(False)
            self.status_info_lbl.setText(f"✓ All {total:,} images of the folder are predicted and cached.")

    # --------------------------------------------------------------------------
    # Interactive Threshold & Overlay Updates
    # --------------------------------------------------------------------------
    def _on_thresh_slider_changed(self, val: int):
        th = float(val) / 100.0
        self.thresh_spin.blockSignals(True)
        self.thresh_spin.setValue(th)
        self.thresh_spin.blockSignals(False)
        self._update_overlays()

    def _on_thresh_spin_changed(self, val: float):
        self.thresh_slider.blockSignals(True)
        self.thresh_slider.setValue(int(round(val * 100)))
        self.thresh_slider.blockSignals(False)
        self._update_overlays()

    def _model_mask(self) -> Optional[np.ndarray]:
        """The model's mask at the threshold (cached; callers must not change it in place)."""
        if self.current_prob is None:
            return None
        thr, e = self.thresh_spin.value(), self.edge_margin_spin.value()
        c = self._model_mask_cache
        if c is not None and c[0] is self.current_prob and c[1] == thr and c[2] == e:
            return c[3]
        m = self.current_prob >= thr
        if e > 0:                                        # predictions this close to the frame edge are dropped
            m[:e] = False
            m[-e:] = False
            m[:, :e] = False
            m[:, -e:] = False
        self._model_mask_cache = (self.current_prob, thr, e, m)
        return m

    def _get_active_binary_mask(self) -> Optional[np.ndarray]:
        """What is on screen as root, uint8 (cached per edit; callers must not change it in place)."""
        if self.current_rgb is None or self.state is None:
            return None
        mm = self._model_mask()
        c = self._active_cache
        if c is not None and c[0] is self.state and c[1] == self.state.rev and c[2] is mm:
            return c[3]
        a = self.state.active(mm).astype(np.uint8)
        self._active_cache = (self.state, self.state.rev, mm, a)
        return a

    def _heat_blend(self, opacity: float) -> np.ndarray:
        """The frame blended with the jet-coloured probability (cached: changes only with the map / opacity)."""
        c = self._heat_cache
        if not (c[0] is not None and c[0][0] is self.current_prob and c[0][1] is self.current_rgb and c[0][2] == opacity):
            prob_u8 = (np.clip(self.current_prob, 0.0, 1.0) * 255.0).astype(np.uint8)
            jet_rgb = cv2.cvtColor(cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            self._heat_cache = ((self.current_prob, self.current_rgb, opacity),
                                cv2.addWeighted(self.current_rgb, 1.0 - opacity, jet_rgb, opacity, 0.0))
        return self._heat_cache[1]

    def _render_box(self, box: Tuple[int, int, int, int], mask: Optional[np.ndarray], view_mode: str,
                    opacity: float, show_contours: bool, show_user: bool) -> Tuple[np.ndarray, np.ndarray]:
        """(left, right) view pixels of ``box`` (y0, y1, x0, x1)."""
        y0, y1, x0, x1 = box
        rgb = self.current_rgb[y0:y1, x0:x1]
        m = (mask[y0:y1, x0:x1] > 0) if mask is not None else None
        has = m is not None and bool(m.any())

        def outline(img, mm, col):
            cnt, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)   # holes too: soil enclosed by roots
            cv2.drawContours(img, cnt, -1, col, 1)

        def tint(img, mm, col, keep):
            img[mm] = np.clip(img[mm] * keep + np.array(col, dtype=np.float32) * (1.0 - keep), 0, 255).astype(np.uint8)

        # 1. RIGHT CANVAS: Model Probability Map & Segmentation
        if view_mode == SEP_VIEW:
            img = self._separation_image()
            if img is not None and img.shape[:2] == self.current_rgb.shape[:2]:
                right = img[y0:y1, x0:x1].copy()
            else:                                                # still separating: the roots in grey meanwhile
                right = rgb.copy()
                if has:
                    tint(right, m, (210, 210, 210), 0.5)
        elif self.current_prob is None:
            right = rgb.copy()
        elif view_mode == "Binary Mask (B&W)":
            right = cv2.cvtColor(m.astype(np.uint8) * 255 if m is not None else np.zeros(rgb.shape[:2], np.uint8),
                                 cv2.COLOR_GRAY2RGB)
        elif view_mode == "Blended RGB Overlay":
            right = rgb.copy()
            if has:
                tint(right, m, (255, 30, 30), 1.0 - opacity)
                outline(right, m, (255, 230, 50))
        else:                                                    # Default: Probability Heatmap (Jet)
            right = self._heat_blend(opacity)[y0:y1, x0:x1].copy()
            if has:
                tint(right, m, (255, 20, 20), 0.4)
                outline(right, m, (255, 255, 255))

        # 2. LEFT CANVAS: Original RGB Image + Root Contours / Red Annotations
        left = rgb.copy()
        if show_user and self.state is not None:
            for tm, col in ((self.state.trace_mask("carried", box), (255, 150, 0)),
                            (self.state.pen[y0:y1, x0:x1] | self.state.trace_mask("user", box), (255, 30, 30))):
                if tm.any():                                     # a light tint and an outline: the root stays visible
                    tint(left, tm, col, 0.6)
                    outline(left, tm, col)
        if self.prev_overlay is not None and self.show_prev_check.isChecked():
            outline(left, self.prev_overlay[y0:y1, x0:x1], (0, 230, 255))
        if show_contours and has:
            outline(left, m, (40, 240, 110))
        return left, right

    def _update_overlays(self, region: Optional[Tuple[int, int, int, int]] = None):
        """Redraw both views.  ``region`` (y0, y1, x0, x1): only that part changed (a stroke), so only it is
        re-rendered when the view settings are the same -- a full redraw took ~0.5 s on 6 MP scans.
        Also a slot of checkboxes / sliders, whose bool / int argument means "redraw everything"."""
        if self.current_rgb is None:
            return
        if not (isinstance(region, tuple) and len(region) == 4):
            region = None
        if self.view_mode_combo.currentText() == SEP_VIEW:      # separation is global: redraw it whole
            region = None
            self._request_separation()

        H, W = self.current_rgb.shape[:2]
        mask = self._get_active_binary_mask()
        if mask is not None:
            self._traits_timer.start()          # skeleton length after a pause: ~0.2 s per redraw on big scans

        view_mode = self.view_mode_combo.currentText()
        opacity = float(self.opacity_slider.value()) / 100.0
        show_contours = self.show_contours_check.isChecked()
        show_user = self.show_user_drawings_check.isChecked()
        objs = (self.current_rgb, self.current_prob,
                self.prev_overlay if self.show_prev_check.isChecked() else None)
        settings = (view_mode, opacity, show_contours, show_user)
        vc = self._view_cache
        partial = (region is not None and vc is not None and vc[1] == settings
                   and all(a is b for a, b in zip(vc[0], objs)))
        if partial:
            y0, y1, x0, x1 = region
            g = (max(0, y0 - 4), min(H, y1 + 4), max(0, x0 - 4), min(W, x1 + 4))   # margin: outlines stay exact
            left_g, right_g = self._render_box(g, mask, view_mode, opacity, show_contours, show_user)
            iy0, iy1, ix0, ix1 = max(0, y0), min(H, y1), max(0, x0), min(W, x1)
            sl = (slice(iy0 - g[0], iy1 - g[0]), slice(ix0 - g[2], ix1 - g[2]))
            left, right = vc[2], vc[3]
            left[iy0:iy1, ix0:ix1] = left_g[sl]
            right[iy0:iy1, ix0:ix1] = right_g[sl]
        else:
            left, right = self._render_box((0, H, 0, W), mask, view_mode, opacity, show_contours, show_user)
            self._view_cache = (objs, settings, left, right)
        self.canvas_right.set_rendered_view(_rgb_to_qimage(right))
        self.canvas_left.set_rendered_view(_rgb_to_qimage(left))

    # --------------------------------------------------------------------------
    # Smart Brush Stroke Snapping & Undo/Redo Engine
    # --------------------------------------------------------------------------
    def _prob_tracer_cache(self) -> Optional[RidgeCache]:
        if self.current_prob is None:
            return None
        if self._prob_ridge_cache is None:
            # sqrt stretches the low end: on these frames missed roots sit at p ~ 0.005-0.05
            p = np.sqrt(np.clip(self.current_prob, 0.0, 1.0)) * 255.0
            self._prob_ridge_cache = RidgeCache({"prob": p})
        return self._prob_ridge_cache

    def _warm_prob_tracer(self, *_):
        if self.snap_mode_combo.currentText().startswith("Model probability"):
            cache = self._prob_tracer_cache()
            if cache is not None:
                threading.Thread(target=cache.warm, daemon=True).start()

    def _trace_params(self, follow: bool, tone: Optional[str] = None) -> TraceParams:
        return TraceParams(
            brush_radius=self.brush_size_slider.value(),
            follow=follow,
            follow_tol=self.follow_tol_spin.value(),
            max_follow_px=self.max_follow_spin.value(),
            max_gap_px=self.max_gap_spin.value(),
            tone=tone or ("auto", "light", "dark")[self.tone_combo.currentIndex()],
        )

    def _colour_for_stroke(self) -> Optional[ColourCue]:
        """The image's colour cue, its samples taken from the user's annotation now (None when off)."""
        cue, st = self._colour_cue, self.state
        if cue is None or st is None or not self.colour_check.isChecked():
            return None
        root = st.trace_mask("user") | st.pen
        if st.base_override is not None:                     # a saved annotation is the user's too
            root |= st.base_override & ~st.erased
        soil = st.erased.copy()                              # what the model got wrong
        if root.any():
            disk = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            active = self._get_active_binary_mask()
            beside = cv2.dilate(root.astype(np.uint8), disk(12)) > 0
            soil |= beside & ~(cv2.dilate(active, disk(3)) > 0)
        cue.set_samples(root, soil)
        return cue

    def _stroke_footprint(self, pts, r: int) -> np.ndarray:
        H, W = self.current_rgb.shape[:2]
        m = np.zeros((H, W), np.uint8)
        ip = np.asarray(pts, np.int32).reshape(-1, 1, 2)
        if len(ip) == 1:
            cv2.circle(m, tuple(ip[0, 0]), r, 1, -1)
        else:
            cv2.polylines(m, [ip], False, 1, thickness=2 * r + 1)
        return m > 0

    def _on_stroke_progress(self, pts, tool: str):
        """Live: show where the stroke drawn so far snaps to (no following yet)."""
        if (tool != "brush" or self.current_rgb is None or self.state is None
                or self.snap_mode_combo.currentText().startswith("Exact")):
            return
        if len(pts) <= 2 or self._preview_ctx is None:
            # what is segmented does not change while the mouse is down: build it once per stroke
            self._preview_lock = {}
            self._preview_colour = self._colour_for_stroke()
            self._preview_ctx = (self._get_active_binary_mask() > 0,
                                 self.state.base(self._model_mask()) & ~self.state.erased)
        existing, prior = self._preview_ctx
        use_prob = self.snap_mode_combo.currentText().startswith("Model probability")
        cache = self._prob_tracer_cache() if use_prob else self._ridge_cache
        if cache is None:
            return
        r = self.brush_size_slider.value()
        params = self._trace_params(False, "light" if use_prob else None)
        params.tone, params.channel = self._preview_lock.get("tone", params.tone), self._preview_lock.get("channel", "")
        try:
            res = trace_root(self.current_rgb, pts, params, cache=cache, footprint=self._stroke_footprint(pts, r),
                             existing=existing, prior=prior, preview=True,
                             width_channels=self._ridge_cache.channels if (use_prob and self._ridge_cache) else None,
                             colour=self._preview_colour if self.colour_check.isChecked() else None)
        except Exception:
            return
        p = np.asarray(pts, np.float64)
        if not self._preview_lock and len(p) > 1 and float(np.hypot(*(p[-1] - p[0]))) >= 12:
            self._preview_lock = {"tone": res.tone, "channel": res.channel}
        canvas = self.sender() if isinstance(self.sender(), SyncInteractiveCanvas) else self.canvas_left
        canvas.set_preview(res.mask)

    def _on_stroke_completed(self, stroke: Dict[str, Any], tool: str):
        try:
            self._apply_stroke(stroke, tool)
        finally:
            self._preview_lock = {}
            self._preview_ctx = None

    def _apply_stroke(self, stroke: Dict[str, Any], tool: str):
        fp = stroke["mask"] > 0
        st = self.state
        if self.current_rgb is None or st is None or not fp.any():
            return
        snap_mode = self.snap_mode_combo.currentText()
        brush_rad = self.brush_size_slider.value()

        if tool == "eraser":
            mode = "exact" if stroke.get("shift") else ("component" if stroke.get("ctrl") else "smart")
            before = self._get_active_binary_mask()
            st.checkpoint()
            n = st.erase(fp, self._model_mask(), mode)
            self.status_info_lbl.setText(
                f"🧹 Erased {n} px ({mode})   (Shift: exact brush area · Ctrl: whole root · Ctrl+Z: undo)")
            after = self._get_active_binary_mask()
            self._after_edit(_box_of(before != after) if before is not None and after is not None else None)
            return

        if tool == "pen" or snap_mode == "Exact Pen (No Snapping)":
            st.checkpoint()
            st.add_pen(fp)
            self._after_edit(_box_of(fp))
            return

        use_prob = snap_mode.startswith("Model probability")
        cache = self._prob_tracer_cache() if use_prob else self._ridge_cache
        fallback = ""
        if use_prob and cache is None:
            cache, fallback = self._ridge_cache, " (prediction not ready: traced on the image)"
            use_prob = False
        route, idx = st.route_stroke(stroke["pts"], brush_rad)
        follow = self.follow_combo.currentIndex() == 0 and not stroke["no_follow"] and route != "refine"
        params = self._trace_params(follow, "light" if use_prob else None)
        params.tone, params.channel = self._preview_lock.get("tone", params.tone), self._preview_lock.get("channel", "")
        active = self._get_active_binary_mask()
        t0 = time.time()
        res = trace_root(self.current_rgb, stroke["pts"], params, stop_mask=active, cache=cache, footprint=fp,
                         existing=active > 0, prior=st.base(self._model_mask()) & ~st.erased,
                         width_channels=self._ridge_cache.channels if (use_prob and self._ridge_cache) else None,
                         colour=self._colour_for_stroke())
        st.checkpoint()
        st.apply_trace(route, idx, res.mask, fp, brush_rad)
        region = _box_of(res.mask.astype(bool) | fp)
        if idx is not None and idx < len(st.traces):            # a refined carried root turns red all along
            b = st.traces[idx].box
            region = b if region is None else (min(region[0], b[0]), max(region[1], b[1]),
                                               min(region[2], b[2]), max(region[3], b[3]))
        what = {"new": "New root", "extend": "Extended root", "refine": "Refined root"}[route]
        hint = ("brushing over a root re-measures it there · paint a missing part to add it · Shift+eraser trims" if route == "refine"
                else "Shift+drag: stroke only · brush over it to widen")
        self.status_info_lbl.setText(
            f"✓ {what}: {res.note}{fallback} · {(time.time() - t0) * 1000:.0f} ms   ({hint} · Ctrl+Z: undo)")
        self._after_edit(region)

    def _after_edit(self, region: Optional[Tuple[int, int, int, int]] = None):
        self._update_overlays(region)
        self._refresh_counter()

    def _on_undo(self):
        if self.state is not None and self.state.undo():
            self.status_info_lbl.setText("Undo applied.")
            self._after_edit()

    def _on_redo(self):
        if self.state is not None and self.state.redo():
            self.status_info_lbl.setText("Redo applied.")
            self._after_edit()

    def _on_clear_user_edits(self):
        if self.state is None:
            return
        self.state.checkpoint()
        self.state.clear()
        self.status_info_lbl.setText("Back to the model's prediction alone (Ctrl+Z to undo).")
        self._after_edit()

    def _on_empty_image(self):
        if self.state is None:
            return
        self.state.checkpoint()
        self.state.empty()
        self.status_info_lbl.setText("🗑 This image now has no roots - the model's prediction is set aside too "
                                     "(Ctrl+Z undoes it; 'Clear Edits' brings the model back).")
        self._after_edit()

    def _on_discard_all(self):
        try:
            self._stash_state()
            dirty = [p for p, st in self._states.items() if st.dirty]
            if not dirty:
                self.status_info_lbl.setText("No image has unsaved edits.")
                return
            names = "\n".join(os.path.basename(p) for p in dirty[:8]) + ("\n..." if len(dirty) > 8 else "")
            ans = QtWidgets.QMessageBox.question(
                self, "Discard unsaved edits",
                f"Discard the unsaved edits of {len(dirty)} image(s)?\n{names}\n\n"
                "Masks already saved on disk are kept. This cannot be undone.",
                _mb("Yes") | _mb("No"), _mb("No"))
            if ans != _mb("Yes"):
                return
            self._edits_gen += 1
            for p in dirty:
                self._states.pop(p, None)
            cur = self._current_path()
            if cur in dirty and self.current_rgb is not None:   # back to its saved mask, or the model
                st = AnnotationState(self.current_rgb.shape[:2])
                saved = self._read_saved_mask(cur)
                if saved is not None and saved.shape == st.shape:
                    st.load_saved(saved, note="saved")
                self.state = st
            self._after_edit()
            self.status_info_lbl.setText(f"Discarded the unsaved edits of {len(dirty)} image(s).")
        except Exception as exc:
            self.status_info_lbl.setText(f"Discard failed: {exc}")

    def _on_export_csv(self):
        """Root traits of every image to a CSV (see ``root_export``), in the background."""
        try:
            if self._export_worker is not None and self._export_worker.isRunning():
                self._export_worker.stop()                      # the button reads "Stop export" meanwhile
                return
            if not self.image_files:
                return
            folder = os.path.dirname(self.image_files[0])
            default = os.path.join(folder, "_annotations", f"roots_export_{time.strftime('%Y%m%d-%H%M')}.csv")
            dlg = ExportDialog(self, default)
            if not (dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec()):
                return
            opt = dlg.choices()
            out = opt["path"]
            if not out:
                return
            if opt["per_root"] and not out.lower().endswith("_roots.csv"):
                out = out[:-4] + "_roots.csv" if out.lower().endswith(".csv") else out + "_roots.csv"
            self._stash_state()
            thr, edge, tag = self.thresh_spin.value(), self.edge_margin_spin.value(), self._current_tag()
            jobs = []
            for i, p in enumerate(self.image_files):
                st = self.state if p == self._state_path else self._states.get(p)
                saved = self._annotation_files(p)[2]
                base = {"path": p, "index": i + 1, "centre": self._centre_mem.get((tag, p)),
                        "centre_path": centre_cache_path(p, tag, self.cache_dir)}
                if p == self._current_path() and self.current_centre is not None:
                    base["centre"] = self.current_centre
                if st is not None and st.dirty:                 # as on screen, unsaved
                    m = self._get_active_binary_mask() > 0 if p == self._state_path else self._annotation_of(p)
                    jobs.append({**base, "source": "unsaved edits", "mask": m})
                elif os.path.exists(saved):
                    jobs.append({**base, "source": "saved", "mask_path": saved})
                elif opt["all_images"]:
                    mem = self._prob_mem.get((tag, p))
                    job = {**base, "source": "model prediction", "threshold": thr, "edge": edge}
                    if mem is not None:
                        m = mem >= thr
                        if edge > 0:
                            m[:edge] = m[-edge:] = False
                            m[:, :edge] = m[:, -edge:] = False
                        job["mask"] = m
                    else:
                        job["prob_path"] = prob_cache_path(p, tag, self.cache_dir)
                    jobs.append(job)
            if not jobs:
                self.status_info_lbl.setText("Nothing to export: no image of this folder is annotated yet.")
                return
            w = ExportWorker(out, jobs, self.px_per_mm, opt["per_root"], opt["centre_lines"])
            w.progress.connect(lambda i, n: self.status_info_lbl.setText(f"📊 Exporting root traits ... {i}/{n} images"))
            w.done.connect(self._on_export_done)
            self._export_worker = w
            self.btn_export.setText("■ Stop export")
            w.start()
        except Exception as exc:
            self.status_info_lbl.setText(f"Export failed: {exc}")

    def _on_export_done(self, path: str, n: int, err: str):
        self.btn_export.setText("📊 Export CSV…")
        if err and err != "stopped":
            self.status_info_lbl.setText(f"✗ Export failed after {n} rows: {err}")
            QtWidgets.QMessageBox.warning(self, "Export failed", f"{err}\n\n{path}")
        else:
            what = "root row(s)" if path.lower().endswith("_roots.csv") else "image(s)"
            self.status_info_lbl.setText(f"📊 Exported {n} {what} to {path}" + (" (stopped early)" if err else ""))

    def _on_delete_folder_edits(self):
        """Every edit of the folder goes: saved masks / overlays / traits table / simulator fit are moved to
        ``_annotations/_deleted_<date-time>`` and this session's edits are dropped -- the model's prediction
        is all that is left on every image.  The prediction cache is kept."""
        try:
            if not self.image_files:
                return
            folder = os.path.dirname(self.image_files[0])
            ann_dir = os.path.join(folder, "_annotations")
            self._stash_state()
            saved = [p for p in self.image_files if os.path.exists(self._annotation_files(p)[2])]
            edited = [p for p, st in self._states.items() if st.has_edits() or st.dirty]
            extras = [os.path.join(ann_dir, n) for n in ("traits_summary.csv", "_sim")
                      if os.path.exists(os.path.join(ann_dir, n))]
            if not saved and not edited and not extras:
                self.status_info_lbl.setText("This folder has no edits: every image already shows the model's prediction.")
                return
            unsaved = sum(1 for p in edited if self._states[p].dirty)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            ans = QtWidgets.QMessageBox.question(
                self, "Delete all edits of this folder",
                f"Delete every edit of\n{folder}?\n\n"
                f"• {len(saved)} saved annotation(s) (mask, overlay)\n"
                f"• {unsaved} image(s) with unsaved edits\n"
                + ("• traits_summary.csv\n" if any(e.endswith("traits_summary.csv") for e in extras) else "")
                + ("• the simulator fit made from your masks (_sim)\n" if any(e.endswith("_sim") for e in extras) else "")
                + f"\nEvery image will show the model's prediction only. The saved files are moved to\n"
                  f"_annotations/_deleted_{stamp} (move them back to restore). The prediction cache is kept.",
                _mb("Yes") | _mb("No"), _mb("No"))
            if ans != _mb("Yes"):
                return
            backup = os.path.join(ann_dir, f"_deleted_{stamp}")
            moved, failed = 0, []
            if saved or extras:
                os.makedirs(backup, exist_ok=True)
            for p in saved:
                for f in self._annotation_files(p)[2:]:
                    if os.path.exists(f):
                        try:
                            shutil.move(f, os.path.join(backup, os.path.basename(f)))
                            moved += 1
                        except OSError as exc:
                            failed.append(f"{os.path.basename(f)}: {exc}")
            for f in extras:
                try:
                    shutil.move(f, os.path.join(backup, os.path.basename(f)))
                    moved += 1
                except OSError as exc:
                    failed.append(f"{os.path.basename(f)}: {exc}")
            self._edits_gen += 1                         # a carry-over / copy still running must not bring them back
            self._states.clear()
            self._saved_paths.clear()
            if self.current_rgb is not None:                    # the image on screen: the model alone
                self.state = AnnotationState(self.current_rgb.shape[:2])
                self.prev_overlay, self._carry_msg = None, ""
            self._after_edit()
            msg = (f"🗑 Deleted all edits of this folder: {len(saved)} saved annotation(s) and {unsaved} unsaved "
                   f"edit(s); only the model's prediction is shown." + (f" Files moved to {backup}." if moved else ""))
            if failed:
                msg += f"  ⚠ {len(failed)} file(s) could not be moved (open elsewhere?)."
                QtWidgets.QMessageBox.warning(self, "Some files were not moved", "\n".join(failed[:12]))
            self.status_info_lbl.setText(msg)
        except Exception as exc:
            self.status_info_lbl.setText(f"Delete all failed: {exc}")

    # --------------------------------------------------------------------------
    # Moving annotations between images (root_transfer: StackReg as in CanoPie)
    # --------------------------------------------------------------------------
    def _own_annotation_kind(self, path: str) -> str:
        """'unsaved edits' / 'saved' when ``path`` has an annotation of its own, else ''."""
        st = self.state if path == self._state_path else self._states.get(path)
        if st is not None and st.dirty:
            return "unsaved edits"
        if os.path.exists(self._annotation_files(path)[2]):
            return "saved"
        return "edits" if (st is not None and st.has_edits()) else ""

    def _state_of(self, path: str, shape: Tuple[int, int]) -> AnnotationState:
        """The editable annotation of any image (the one on screen, one kept this session, or a fresh one)."""
        if path == self._state_path and self.state is not None:
            return self.state
        st = self._states.get(path)
        if st is None:
            st = AnnotationState(shape)
            saved = self._read_saved_mask(path)
            if saved is not None and saved.shape == st.shape:
                st.load_saved(saved, note="saved")
            self._states[path] = st
        return st

    def _paste_into(self, path: str, mask: np.ndarray, note: str, mode: str = "replace") -> int:
        """Put ``mask`` on ``path`` (undoable on that image): ``replace`` its annotation, or ``add`` the roots
        it does not show yet.  Returns the roots / pieces put there."""
        shown = None
        if mode == "add":                                    # what the image shows now, before the step
            shown = self._get_active_binary_mask() > 0 if path == self._state_path else self._annotation_of(path)
        st = self._state_of(path, mask.shape[:2])
        st.checkpoint()
        if mode == "add":
            n = st.add_roots(mask, shown if shown is not None else np.zeros(mask.shape[:2], bool), note)
        else:
            st.paste(mask, note)
            n = len(st.traces)
        if path == self._state_path:
            self._after_edit()
        return n

    def _ask_add_or_replace(self, what: str) -> Optional[str]:
        """'add' / 'replace' / None (cancel) for putting transferred roots on an image that has its own."""
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("It already has an annotation")
        box.setText(f"{what}\n\nAdd missing roots: keep its polygons and add only the transferred roots (or parts) "
                    "that are not there yet.\nReplace: its annotation becomes the transferred roots.\n"
                    "Either way Ctrl+Z on that image undoes it.")
        role = QtWidgets.QMessageBox.ButtonRole if hasattr(QtWidgets.QMessageBox, "ButtonRole") else QtWidgets.QMessageBox
        b_add = box.addButton("Add missing roots", role.AcceptRole)
        b_rep = box.addButton("Replace", role.DestructiveRole)
        box.addButton("Cancel", role.RejectRole)
        box.setDefaultButton(b_add)
        box.exec_() if hasattr(box, "exec_") else box.exec()
        return {id(b_add): "add", id(b_rep): "replace"}.get(id(box.clickedButton()))

    def _prob_for(self, path: str) -> Tuple[Optional[np.ndarray], str]:
        """(probability map in memory, else its cache file) of ``path`` for the current model."""
        if path == self._current_path() and self.current_prob is not None:
            return self.current_prob, ""
        p = self._prob_mem.get((self._current_tag(), path))
        return (p, "") if p is not None else (None, prob_cache_path(path, self._current_tag(), self.cache_dir))

    def _start_transfer(self, src_path: str, src_rgb: Optional[np.ndarray], src_mask: np.ndarray,
                        targets: List[dict], msg: str):
        if self._transfer_worker is not None and self._transfer_worker.isRunning():
            self.status_info_lbl.setText("A transfer is still running - try again in a moment.")
            return
        w = TransferWorker(src_rgb, src_path, src_mask, targets)
        w.result.connect(lambda t, res, err, gen=self._edits_gen:
                         self._on_transfer_result(t, res, err) if gen == self._edits_gen else None)
        w.finished.connect(lambda w=w: setattr(self, "_transfer_worker", None) if self._transfer_worker is w else None)
        self._transfer_worker = w
        self.status_info_lbl.setText(msg)
        w.start()

    def _on_copy_to(self, step: int):
        """Copy what this image shows onto the previous / next image, registered to the exact spot."""
        try:
            path = self._current_path()
            if path is None or self.state is None or self.current_rgb is None:
                return
            idx = self.current_img_idx + step
            if not 0 <= idx < len(self.image_files):
                self.status_info_lbl.setText("There is no image there.")
                return
            target = self.image_files[idx]
            mask = self._get_active_binary_mask() > 0
            if not mask.any():
                self.status_info_lbl.setText("Nothing to copy: this image shows no roots.")
                return
            has = self._own_annotation_kind(target)
            mode = "replace"
            if has:
                mode = self._ask_add_or_replace(f"Image #{idx + 1} ({os.path.basename(target)}) already has {has}.")
                if mode is None:
                    return
            prob, prob_path = self._prob_for(target)
            k0, k1 = temporal.parse_frame(path), temporal.parse_frame(target)
            other = (" (another tube / window)"
                     if (k0 is not None and k1 is not None and k0.window != k1.window) else "")
            n = max(0, cv2.connectedComponents(mask.astype(np.uint8))[0] - 1)
            self._start_transfer(path, self.current_rgb, mask,
                                 [{"path": target, "index": idx, "kind": "copy", "go": not _shift_held(), "mode": mode,
                                   "prob": prob, "prob_path": prob_path, "src_mask": mask}],
                                 f"{'⇥' if step > 0 else '⇤'} Registering and copying {n} root segment(s) to "
                                 f"#{idx + 1}{other} ...")
        except Exception as exc:
            self.status_info_lbl.setText(f"Copy failed: {exc}")

    def _on_accept_prev(self):
        """Put the previous annotated date's roots on this image, registered to the exact spot (replaces)."""
        try:
            path = self._current_path()
            if path is None or self.state is None or self.current_rgb is None:
                return
            prev, prev_mask = temporal.find_previous_annotation(path, self.image_files, self._annotation_of)
            if prev is None:
                self.status_info_lbl.setText("No earlier date of this tube & window has an annotation to accept.")
                return
            k = temporal.parse_frame(prev)
            has = self._own_annotation_kind(path)
            mode = "replace"
            if has:
                mode = self._ask_add_or_replace(f"This image already has {has}.")
                if mode is None:
                    return
            self._start_transfer(prev, None, prev_mask,
                                 [{"path": path, "rgb": self.current_rgb, "index": self.current_img_idx,
                                   "kind": "accept", "from": k.date if k else os.path.basename(prev),
                                   "mode": mode, "prob": self.current_prob, "src_mask": prev_mask}],
                                 f"⬇ Registering the roots of {k.date if k else os.path.basename(prev)} onto this image ...")
        except Exception as exc:
            self.status_info_lbl.setText(f"Accept failed: {exc}")

    def _on_transfer_result(self, t: dict, res, err: str):
        try:
            num, name = t["index"] + 1, os.path.basename(t["path"])
            if res is None:
                self.status_info_lbl.setText(f"✗ Transfer to #{num} failed: {err}")
                return
            mask, info = res.mask, res.message
            if not res.placed:
                same_size = t.get("src_mask") is not None and t["src_mask"].shape == res.mask.shape
                if not (same_size and res.reg is not None and not res.reg.matched):
                    self._hold_status(f"Nothing placed on #{num} {name}: {res.message}.")
                    return
                ans = QtWidgets.QMessageBox.question(           # e.g. another window, or a re-inserted tube
                    self, "The images do not line up",
                    f"#{num} {name} does not line up with this image ({res.message}).\n\n"
                    "Copy the roots at the same position anyway (no registration)?",
                    _mb("Yes") | _mb("No"), _mb("Yes"))
                if ans != _mb("Yes"):
                    self._hold_status(f"Not copied to #{num}: the images do not line up.")
                    return
                mask, info = t["src_mask"], "copied at the same position (the images do not line up)"
            src = "date " + t["from"] if t["kind"] == "accept" else os.path.basename(self._current_path() or "")
            mode = t.get("mode", "replace")
            n = self._paste_into(t["path"], mask, f"transferred from {src}", mode)
            verb = "⬇ Accepted the roots of " + t["from"] if t["kind"] == "accept" else f"⇥ Copied to #{num} {name}"
            if mode == "add":
                verb += (f" - added {n} missing piece(s), its own polygons kept" if n else
                         " - every copied root is already annotated there, nothing added ('Replace' overwrites)")
            if t.get("go") and t["index"] != self.current_img_idx:
                self._load_image_by_index(t["index"])         # show where it went
            self._hold_status(f"{verb}: {info}.  Orange = transferred: erase what is gone, brush what is new "
                              "(Ctrl+Z undoes).")
            self._refresh_counter()
        except Exception as exc:
            self.status_info_lbl.setText(f"Transfer not applied: {exc}")

    def _on_find_in_dates(self):
        try:
            path = self._current_path()
            if path is None or self.current_rgb is None:
                return
            mask = self._get_active_binary_mask() > 0
            if not mask.any():
                self.status_info_lbl.setText("Nothing to look for: this image shows no roots.")
                return
            if self._finder is not None:
                self._finder.close()
            self._finder = RootFinderDialog(self, path, self.current_rgb, mask)
            self._finder.show()
        except Exception as exc:
            self.status_info_lbl.setText(f"Could not open the finder: {exc}")

    # --------------------------------------------------------------------------
    # Previous date of the same tube & window
    # --------------------------------------------------------------------------
    def _annotation_of(self, path: str) -> Optional[np.ndarray]:
        """Annotation of another frame: this session's edits if any, else its saved mask."""
        st = self._states.get(path)
        if st is not None and st.has_edits():
            prob = self._prob_mem.get((self._current_tag(), path))
            if prob is None:
                prob = load_cached_prob(prob_cache_path(path, self._current_tag(), self.cache_dir))
            return st.active(None if prob is None else prob >= self.thresh_spin.value())
        return self._read_saved_mask(path)

    def _start_carry_over(self):
        path = self._current_path()
        if path is None or self.current_rgb is None or not (self.carry_check.isChecked() or self.show_prev_check.isChecked()):
            return
        prev, prev_mask = temporal.find_previous_annotation(path, self.image_files, self._annotation_of)
        if prev is None:
            return
        carry = self.carry_check.isChecked() and not self.state.has_edits()
        worker = CarryOverWorker(path, self.current_rgb, prev, prev_mask, self._ridge_cache,
                                 self._trace_params(True), carry, self._model_mask(), self.current_prob)
        worker.done.connect(lambda path, found, aligned, msg, gen=self._edits_gen:
                            self._on_carry_done(path, found, aligned, msg) if gen == self._edits_gen else None)
        worker.finished.connect(lambda w=worker: self._carry_workers.remove(w) if w in self._carry_workers else None)
        self._carry_workers.append(worker)
        key = temporal.parse_frame(prev)
        self.status_info_lbl.setText(f"↪ Aligning the previous date ({key.date if key else os.path.basename(prev)}) ...")
        worker.start()

    def _on_carry_done(self, path: str, found: list, aligned: Optional[np.ndarray], msg: str):
        if path != self._current_path() or self.state is None:
            return
        self.prev_overlay = aligned
        if found:
            dirty = self.state.dirty
            self.state.checkpoint()
            for m, fp in found:
                self.state.apply_trace("new", None, m, fp, self.brush_size_slider.value(), origin="carried")
            self.state.dirty = dirty                     # automatic: nothing to save until you edit or save
        self._carry_msg = msg
        if time.time() >= self._status_hold_until:              # a copy's result stays readable a moment
            self.status_info_lbl.setText(msg)
        self._after_edit()

    def _hold_status(self, msg: str, seconds: float = 6.0):
        """Show ``msg`` and keep background messages (date carry-over) from replacing it for a while."""
        self.status_info_lbl.setText(msg)
        self._status_hold_until = time.time() + seconds

    # --------------------------------------------------------------------------
    # Tools, Sizes & View Helpers
    # --------------------------------------------------------------------------
    @staticmethod
    def _size_key(tool: str) -> str:
        return "eraser" if tool == "eraser" else "draw"          # brush and pen share a size

    def _set_active_tool(self, tool_name: str):
        old, new = self._size_key(self._active_tool), self._size_key(tool_name)
        self._tool_sizes[old] = self.brush_size_slider.value()
        self._active_tool = tool_name
        self.canvas_left.set_tool(tool_name)
        self.canvas_right.set_tool(tool_name)
        if new != old:
            self.brush_size_slider.setValue(self._tool_sizes[new])
        self.size_lbl.setText("Eraser Size:" if new == "eraser" else "Brush Size:")
        self.status_info_lbl.setText(f"Active Tool: {tool_name.upper()}  (size {self.brush_size_slider.value()})")

    def _on_brush_size_changed(self, val: int):
        self.brush_size_spin.blockSignals(True)
        self.brush_size_spin.setValue(val)
        self.brush_size_spin.blockSignals(False)
        self.canvas_left.set_brush_radius(val)
        self.canvas_right.set_brush_radius(val)

    def _on_fit_views(self):
        self.canvas_left._fit_to_window()

    def _on_px_per_mm_changed(self, val: float):
        self.px_per_mm = float(val)
        self._refresh_traits()

    def _on_mouse_pixel_info(self, x: int, y: int, prob_val: float):
        self.status_pixel_lbl.setText(f"X: {x:4d}  Y: {y:4d}  |  Model Prob: {prob_val:.3f}")

    # --------------------------------------------------------------------------
    # Real-Time Root Trait Calculations
    # --------------------------------------------------------------------------
    def _refresh_traits(self):
        try:
            mask = self._get_active_binary_mask()
            if mask is not None:
                self._compute_and_display_traits(mask)
                if self._length_from_centre():                  # separated roots: shown when ready
                    if self._current_separation() is not None:
                        self._show_separation_traits()
                    else:
                        self._request_separation()
        except Exception as exc:                                # a slot must never raise (PyQt aborts)
            self.status_info_lbl.setText(f"Traits not updated: {exc}")

    # --------------------------------------------------------------------------
    # Separated roots (the model's centre-line map)
    # --------------------------------------------------------------------------
    def _length_from_centre(self) -> bool:
        return self.length_src_combo.currentIndex() == 0

    def _sep_key(self) -> tuple:
        """What a separation was made from: image, edits, model mask (threshold) and centre-line map."""
        mm = self._model_mask()
        return (self._current_path(), id(self.state), self.state.rev if self.state is not None else -1,
                id(mm), id(self.current_centre))

    def _current_separation(self) -> Optional[tuple]:
        r = self._sep_result
        return r[1] if (r is not None and self.state is not None and r[0] == self._sep_key()) else None

    def _request_separation(self):
        """Separate the roots on screen in the background (each edit / threshold / new map starts it again)."""
        try:
            if self.current_rgb is None or self.state is None or self._current_separation() is not None:
                return
            if self._sep_worker is not None and self._sep_worker.isRunning():
                self._sep_wanted = True                         # again with the latest state when it is done
                return
            mask = self._get_active_binary_mask()
            if mask is None:
                return
            w = SeparationWorker(self._sep_key(), mask > 0, self.current_centre)
            w.done.connect(self._on_separation_done)
            self._sep_worker, self._sep_wanted = w, False
            w.start()
        except Exception as exc:
            self.status_info_lbl.setText(f"Root separation not started: {exc}")

    def _on_separation_done(self, key: tuple, res, err: str):
        try:
            if res is not None:
                self._sep_result = (key, res)
            elif err:
                self.status_info_lbl.setText(f"Root separation failed: {err}")
            if self.state is not None and key == self._sep_key():
                if self.view_mode_combo.currentText() == SEP_VIEW:
                    self._update_overlays()
                    if res is not None and time.time() >= self._status_hold_until:
                        sep, _, model_used = res
                        self.status_info_lbl.setText(
                            f"✂ {sep.n} roots separated "
                            + ("along the model's centre lines" if model_used else
                               "by the mask's shape (the model's centre-line map is not here yet)")
                            + " - white: centre lines, each colour one root.")
                if self._length_from_centre():
                    self._show_separation_traits()
            elif self._sep_wanted or (self.view_mode_combo.currentText() == SEP_VIEW or self._length_from_centre()):
                self._sep_wanted = False
                self._request_separation()                      # the screen changed meanwhile
        except Exception as exc:
            self.status_info_lbl.setText(f"Root separation not shown: {exc}")

    def _show_separation_traits(self):
        r = self._current_separation()
        mask = self._get_active_binary_mask()
        if r is None or mask is None:
            return
        sep, length_px, model_used = r
        L = length_px / self.px_per_mm
        area = float((mask > 0).sum()) / self.px_per_mm ** 2
        self.trait_len_lbl.setText(f"Total Length: {L:.1f} mm ({L / 10.0:.2f} cm)")
        self.trait_len_lbl.setToolTip("Along the roots' centre lines, separated "
                                      + ("by the model's centre-line map." if model_used else "by the mask's shape."))
        self.trait_diam_lbl.setText(f"Mean Diameter: {(area / L) if L > 0 else 0.0:.2f} mm")
        self.trait_count_lbl.setText(f"Roots: {sep.n} separated")

    def _separation_image(self) -> Optional[np.ndarray]:
        """The frame with every separated root coloured (cached per separation)."""
        r = self._current_separation()
        if r is None:
            return None
        if self._sep_view[0] is not r:
            from exp_realism import root_separation as RS
            self._sep_view = (r, RS.colour_image(self.current_rgb, r[0]))
        return self._sep_view[1]

    def _on_centre_ready(self, tag: str, path: str, centre: np.ndarray):
        try:
            key = (tag, path)
            self._centre_mem.pop(key, None)
            self._centre_mem[key] = centre
            while len(self._centre_mem) > 1 and sum(c.nbytes for c in self._centre_mem.values()) > 256 * 1024 * 1024:
                self._centre_mem.popitem(last=False)
            if tag == self._current_tag() and path == self._current_path():
                self.current_centre = centre
                if self.view_mode_combo.currentText() == SEP_VIEW or self._length_from_centre():
                    self._request_separation()
                if self.view_mode_combo.currentText() == SEP_VIEW:
                    self._update_overlays()
        except Exception as exc:
            self.status_info_lbl.setText(f"Centre lines not applied: {exc}")

    def _compute_and_display_traits(self, mask: np.ndarray):
        m = mask > 0
        px_count = int(m.sum())
        H, W = mask.shape[:2]
        cov_pct = (px_count / float(H * W)) * 100.0

        area_mm2 = px_count / (self.px_per_mm ** 2)

        length_mm = 0.0
        if px_count > 0:
            try:
                from skimage.morphology import skeletonize
                sk = skeletonize(m)
                length_mm = float(sk.sum()) / self.px_per_mm
            except Exception:
                length_mm = np.sqrt(px_count) / self.px_per_mm

        length_cm = length_mm / 10.0
        mean_diam_mm = (area_mm2 / length_mm) if length_mm > 0 else 0.0

        num_roots = 0
        if px_count > 0:
            num_roots, _ = cv2.connectedComponents(m.astype(np.uint8))
            num_roots = max(0, num_roots - 1)

        self.trait_len_lbl.setText(f"Total Length: {length_mm:.1f} mm ({length_cm:.2f} cm)")
        self.trait_area_lbl.setText(f"Surface Area: {area_mm2:.1f} mm²")
        self.trait_diam_lbl.setText(f"Mean Diameter: {mean_diam_mm:.2f} mm")
        self.trait_cov_lbl.setText(f"Coverage: {cov_pct:.2f} %")
        self.trait_count_lbl.setText(f"Root Segments: {num_roots}")

    # --------------------------------------------------------------------------
    # Saving & Export
    # --------------------------------------------------------------------------
    @staticmethod
    def _annotation_files(path: str) -> Tuple[str, str, str, str]:
        stem = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(os.path.dirname(path), "_annotations")
        return stem, out_dir, os.path.join(out_dir, f"{stem}_mask.png"), os.path.join(out_dir, f"{stem}_overlay.jpg")

    def _read_saved_mask(self, path: str) -> Optional[np.ndarray]:
        mask_path = self._annotation_files(path)[2]
        if not os.path.exists(mask_path):
            return None
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return None if m is None else m > 127

    def _refresh_counter(self):
        path = self._current_path()
        if path is None:
            return
        tags = []
        if self.state is not None:
            if self.state.dirty:
                tags.append("● unsaved edits")
            elif path in self._saved_paths:
                tags.append("✓ saved")
            if self.state.trace_mask("carried").any():
                tags.append("↪ carried / transferred from another date")
        self.file_counter_lbl.setText(
            f"Image {self.current_img_idx + 1} / {len(self.image_files)}:  {os.path.basename(path)}"
            + (f"   {'  '.join(tags)}" if tags else ""))

    def _traits_row(self, stem: str, mask: np.ndarray) -> str:
        m = mask > 0
        px_count = int(m.sum())
        H, W = mask.shape[:2]
        cov_pct = (px_count / float(H * W)) * 100.0
        area_mm2 = px_count / (self.px_per_mm ** 2)

        sk_px = 0
        try:
            from skimage.morphology import skeletonize
            sk_px = int(skeletonize(m).sum()) if m.any() else 0
        except Exception:
            sk_px = int(np.sqrt(px_count) * self.px_per_mm)

        length_mm = sk_px / self.px_per_mm
        diam_mm = (area_mm2 / length_mm) if length_mm > 0 else 0.0
        return f'"{stem}",{length_mm:.3f},{length_mm/10.0:.3f},{area_mm2:.3f},{cov_pct:.4f},{diam_mm:.4f},{self.thresh_spin.value():.2f}\n'

    @staticmethod
    def _upsert_traits(out_dir: str, stem: str, row: str) -> str:
        """One row per image in traits_summary.csv (a re-save replaces the old row). Returns a warning or ''."""
        csv_path = os.path.join(out_dir, "traits_summary.csv")
        header = "image,root_len_mm,root_len_cm,root_area_mm2,root_coverage_pct,mean_diam_mm,threshold\n"
        lines: List[str] = []
        try:
            if os.path.exists(csv_path):
                with open(csv_path, "r", encoding="utf-8") as f:
                    lines = [ln if ln.endswith("\n") else ln + "\n" for ln in f.read().splitlines()]
            body = [ln for ln in lines[1:] if ln.strip() and ln.split(",", 1)[0].strip('"') != stem]
            tmp = csv_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.writelines([lines[0] if lines else header] + body + [row])
            os.replace(tmp, csv_path)
            return ""
        except PermissionError:
            return "traits_summary.csv is open in another program (Excel?) - its row was not updated"
        except OSError as exc:
            return f"traits_summary.csv not updated: {exc}"

    def _save_mask(self, path: str, rgb: np.ndarray, mask: np.ndarray) -> Tuple[bool, str]:
        """Write <stem>_mask.png, _overlay.jpg and the traits row. Returns (mask written, message)."""
        stem, out_dir, mask_path, overlay_path = self._annotation_files(path)
        try:
            os.makedirs(out_dir, exist_ok=True)
            if not cv2.imwrite(mask_path, (mask > 0).astype(np.uint8) * 255):
                return False, f"could not write {mask_path}"
            vis = rgb.copy().astype(np.float32)
            m = mask > 0
            if m.any():
                red = np.array([255, 30, 30], dtype=np.float32)
                vis[m] = vis[m] * 0.45 + red * 0.55
            vis_u8 = np.clip(vis, 0, 255).astype(np.uint8)
            if m.any():
                cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)   # holes too: soil enclosed by roots
                cv2.drawContours(vis_u8, cnt, -1, (255, 230, 50), 1)
            cv2.imwrite(overlay_path, cv2.cvtColor(vis_u8, cv2.COLOR_RGB2BGR))
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        self._saved_paths.add(path)
        return True, self._upsert_traits(out_dir, stem, self._traits_row(stem, mask))

    def _save_path_state(self, path: str) -> bool:
        """Save a session-edited image that is not on screen (used when closing)."""
        st = self._states.get(path)
        rgb = read_rgb(path)
        if st is None or rgb is None:
            return False
        prob = self._prob_mem.get((self._current_tag(), path))
        if prob is None:
            prob = load_cached_prob(prob_cache_path(path, self._current_tag(), self.cache_dir))
        ok, _ = self._save_mask(path, rgb, st.active(None if prob is None else prob >= self.thresh_spin.value()))
        if ok:
            st.dirty = False
        return ok

    def _on_save_current(self) -> bool:
        path = self._current_path()
        mask = self._get_active_binary_mask()
        if path is None or mask is None:
            return False
        ok, msg = self._save_mask(path, self.current_rgb, mask)
        if not ok:
            self.status_info_lbl.setText(f"✗ Save failed: {msg}")
            QtWidgets.QMessageBox.warning(self, "Save failed", msg)
            return False
        self.state.dirty = False
        self._states[path] = self.state
        n_roots = max(0, cv2.connectedComponents(mask)[0] - 1)
        out_dir = self._annotation_files(path)[1]
        self.status_info_lbl.setText(f"✓ Saved {os.path.basename(path)} ({n_roots} root segments) to {out_dir}"
                                     + (f"   ⚠ {msg}" if msg else ""))
        if msg:
            QtWidgets.QMessageBox.warning(self, "Traits not updated", msg)
        self._refresh_counter()
        return True

    def _on_save_and_next(self):
        if self._on_save_current():
            self._on_next_image()


# ==============================================================================
# Entry Point CLI
# ==============================================================================

def main():
    print("[RoosterFly] Launching the studio window...", flush=True)

    parser = argparse.ArgumentParser(description="RoosterFly - interactive minirhizotron root annotation studio")
    parser.add_argument("--root", type=str, default=str(_REPO_ROOT), help="Project root folder")
    parser.add_argument("--model", type=str, default="v8_full_real_and_syn_mitb3", help="Initial run name in segmentation/realsoil")
    parser.add_argument("--dir", type=str, default="", help="Initial image folder")
    parser.add_argument("--image", type=str, default="", help="Initial image file")
    parser.add_argument("--px-per-mm", type=float, default=11.81, help="Camera resolution in px/mm")
    parser.add_argument("--lookahead", type=int, default=3, help="Upcoming images predicted in the background")
    parser.add_argument("--predict-all", action="store_true", help="Start pre-predicting the whole folder at launch")
    parser.add_argument("--cache-dir", type=str, default="",
                        help="Probability cache root (default: <image folder>/_annotations/_prob_cache)")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("RoosterFly")
    if os.path.exists(APP_ICON):
        app.setWindowIcon(QtGui.QIcon(APP_ICON))
    if sys.platform == "win32":                         # own taskbar icon instead of python.exe's
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("RoosterFly.Studio")
        except Exception:
            pass

    studio = RootAnnotatorStudio(
        project_root=args.root,
        initial_model=args.model,
        initial_dir=args.dir,
        initial_image=args.image,
        px_per_mm=args.px_per_mm,
        cache_dir=args.cache_dir,
        lookahead=args.lookahead,
        predict_all=args.predict_all,
    )
    studio.show()
    print("[RoosterFly] Studio window visible on screen.", flush=True)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
