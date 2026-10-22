"""Interactive PyQt Researcher Studio for Environment-Specific Synthetic Root Generation.

Allows a researcher working with a specific soil type, camera, and plant species to:
  1. Load root-free real soil images from their own environment (or browse the PRMI soil bank).
  2. Optionally load a real root reference image from their experiment for side-by-side matching.
  3. Adjust root architecture, epidermal texture/color, soil-root physical interaction
     (aggregate burial, adherent soil film, contact crevice shadow), and camera optics
     without any UI lag, then click "▶ RUN / UPDATE ROOT PREVIEW" to render in a background thread.
  4. Configure full dataset generation rules in the "Dataset & Randomization" tab:
     - Soil selection: Current soil only, Checked soil types (Cotton, Papaya, Peanut, Sesame,
       Sunflower, Switchgrass), or All 296 soils + flip/exposure augmentation.
     - Species selection: Current species only, or Random mix of checked species.
     - Parameter randomization: Exact GUI sliders (0% jitter), Controlled % jitter around GUI
       sliders per category, or Full Domain Randomization across all realistic ranges.
  5. Export a tailored synthetic training dataset (`rgb/`, `mask/`, `diam/`, `alpha/`,
     `traits.csv`, `realism_config.json`, `dataset_gen_config.json`) to train a U-Net.

Usage:
    python -m exp_realism.gui_app
    # or
    python exp_realism/gui_app.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure repository root is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Qt5 / Qt6 / PySide6 compatibility shim
QT_API = None
try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtCore import Qt, QThread, pyqtSignal as Signal
    QT_API = "PyQt5"
except ImportError:
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
        from PyQt6.QtCore import Qt, QThread, pyqtSignal as Signal
        QT_API = "PyQt6"
    except ImportError:
        try:
            from PySide6 import QtCore, QtGui, QtWidgets
            from PySide6.QtCore import Qt, QThread, Signal
            QT_API = "PySide6"
        except ImportError as exc:
            raise ImportError(
                "PyQt5, PyQt6, or PySide6 is required to run the GUI. "
                "Install with: pip install PyQt5"
            ) from exc

from exp_realism.engine import RealismSimulator
from exp_realism.realism_params import (
    PRESET_ENVIRONMENTS,
    DatasetGenConfig,
    RealismConfig,
    sample_randomized_config,
)
from exp_realism.soil_interaction import compute_preview_root_colors


# ==============================================================================
# Helpers for Qt6 / Qt5 enum differences & image conversion
# ==============================================================================

def _align_center():
    return Qt.AlignmentFlag.AlignCenter if hasattr(Qt, "AlignmentFlag") else Qt.AlignCenter


def _keep_aspect():
    return (
        Qt.AspectRatioMode.KeepAspectRatio
        if hasattr(Qt, "AspectRatioMode")
        else Qt.KeepAspectRatio
    )


def _smooth_transform():
    return (
        Qt.TransformationMode.SmoothTransformation
        if hasattr(Qt, "TransformationMode")
        else Qt.SmoothTransformation
    )


def _horizontal():
    return Qt.Orientation.Horizontal if hasattr(Qt, "Orientation") else Qt.Horizontal


def _to_u8_rgb(img: np.ndarray) -> np.ndarray:
    """Convert float [0,1] or uint8 RGB array to contiguous uint8 RGB."""
    if img.dtype == np.uint8:
        return np.ascontiguousarray(img)
    return np.ascontiguousarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))


def _rgb_to_qpixmap(rgb: np.ndarray) -> QtGui.QPixmap:
    """Convert RGB array (H, W, 3) to QPixmap."""
    rgb_c = _to_u8_rgb(rgb)
    h, w, ch = rgb_c.shape
    bytes_per_line = ch * w
    fmt = (
        QtGui.QImage.Format.Format_RGB888
        if hasattr(QtGui.QImage, "Format")
        else QtGui.QImage.Format_RGB888
    )
    qimg = QtGui.QImage(rgb_c.data, w, h, bytes_per_line, fmt).copy()
    return QtGui.QPixmap.fromImage(qimg)


def _make_mask_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    diam_mm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create a diagnostic segmentation & diameter overlay on top of the RGB frame."""
    vis = _to_u8_rgb(rgb).astype(np.float32)
    m = mask > 0
    if np.any(m):
        overlay = vis.copy()
        if diam_mm is not None and np.any(diam_mm > 0):
            p95 = max(float(np.percentile(diam_mm[m], 95)), 0.25)
            d_norm = np.clip(diam_mm / p95, 0.0, 1.0)
            overlay[:, :, 0] = np.clip(40 + 215 * d_norm, 0, 255)
            overlay[:, :, 1] = np.clip(225 - 85 * d_norm, 0, 255)
            overlay[:, :, 2] = np.clip(115 + 125 * (1.0 - d_norm), 0, 255)
        else:
            overlay[:, :, 0] = 50
            overlay[:, :, 1] = 235
            overlay[:, :, 2] = 130
        vis[m] = 0.55 * vis[m] + 0.45 * overlay[m]

        u8 = (m.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis_u8 = np.clip(vis, 0, 255).astype(np.uint8)
        cv2.drawContours(vis_u8, contours, -1, (255, 230, 60), 1)
        return vis_u8
    return np.clip(vis, 0, 255).astype(np.uint8)


# ==============================================================================
# Background Workers for Non-Blocking Preview Rendering & Dataset Export
# ==============================================================================

class PreviewRenderWorker(QThread):
    """Runs v4 (and optional v3) preview rendering in a background thread so UI never freezes."""

    finished_ok = Signal(dict, object, float)
    failed = Signal(str)

    def __init__(
        self,
        sim: RealismSimulator,
        cfg: RealismConfig,
        soil_index: int,
        stand_idx: int,
        frame_idx: int,
        also_render_v3: bool,
    ):
        super().__init__()
        self.sim = sim
        self.cfg = cfg
        self.soil_index = soil_index
        self.stand_idx = stand_idx
        self.frame_idx = frame_idx
        self.also_render_v3 = also_render_v3

    def run(self):
        try:
            t0 = time.perf_counter()
            v4_res = self.sim.render_v4_frame(
                self.cfg,
                soil_index_or_rgb=self.soil_index,
                stand_idx=self.stand_idx,
                frame_idx=self.frame_idx,
            )
            v3_res = None
            if self.also_render_v3:
                v3_res = self.sim.render_legacy_v3_frame(
                    self.cfg,
                    soil_index_or_rgb=v4_res["background"],
                    stand_idx=self.stand_idx,
                    frame_idx=self.frame_idx,
                )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.finished_ok.emit(v4_res, v3_res, dt_ms)
        except Exception as exc:
            self.failed.emit(str(exc))


class DatasetExportWorker(QThread):
    progress = Signal(int, int, str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        sim: RealismSimulator,
        cfg: RealismConfig,
        gen_cfg: DatasetGenConfig,
        output_dir: str,
    ):
        super().__init__()
        self.sim = sim
        self.cfg = cfg
        self.gen_cfg = gen_cfg
        self.output_dir = output_dir

    def run(self):
        try:
            self.sim.generate_environment_dataset(
                cfg=self.cfg,
                out_dir=self.output_dir,
                n_systems=self.gen_cfg.n_systems,
                frames_per_system=self.gen_cfg.frames_per_system,
                empty_frame_frac=self.gen_cfg.empty_frame_frac,
                gen_cfg=self.gen_cfg,
                progress_cb=lambda done, total, fid: self.progress.emit(done, total, fid),
            )
            self.finished_ok.emit(str(self.output_dir))
        except Exception as exc:
            self.failed.emit(str(exc))


# ==============================================================================
# Interactive Zoomable Image Canvas
# ==============================================================================

class ZoomableImageLabel(QtWidgets.QLabel):
    """Displays an RGB image with click-to-set ROI center and 2x loupe zoom."""

    roi_changed = Signal(float, float)  # normalized (u, v) in [0, 1]

    def __init__(self, title: str = ""):
        super().__init__()
        self.title = title
        self._rgb: Optional[np.ndarray] = None
        self._zoom_2x: bool = False
        self._roi_uv: Tuple[float, float] = (0.5, 0.5)
        self.setAlignment(_align_center())
        self.setMinimumSize(280, 320)
        self.setStyleSheet(
            "QLabel { background-color: #14171c; border: 1px solid #2e3440; border-radius: 6px; }"
        )
        self.setCursor(
            Qt.CursorShape.CrossCursor
            if hasattr(Qt, "CursorShape")
            else Qt.CrossCursor
        )

    def set_image(self, rgb: np.ndarray):
        self._rgb = _to_u8_rgb(rgb)
        self._refresh_pixmap()

    def set_zoom_mode(self, enabled: bool):
        self._zoom_2x = bool(enabled)
        self._refresh_pixmap()

    def set_roi_uv(self, u: float, v: float):
        self._roi_uv = (float(np.clip(u, 0.1, 0.9)), float(np.clip(v, 0.1, 0.9)))
        self._refresh_pixmap()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if self._rgb is not None and self.width() > 0 and self.height() > 0:
            pos = event.position() if hasattr(event, "position") else event.pos()
            u = float(pos.x()) / max(float(self.width()), 1.0)
            v = float(pos.y()) / max(float(self.height()), 1.0)
            self.roi_changed.emit(float(np.clip(u, 0.1, 0.9)), float(np.clip(v, 0.1, 0.9)))
        super().mousePressEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self):
        if self._rgb is None:
            return
        img = self._rgb
        H, W = img.shape[:2]
        if self._zoom_2x:
            u, v = self._roi_uv
            ch, cw = max(H // 2, 64), max(W // 2, 64)
            cx = int(np.clip(u * W, cw // 2, W - cw // 2))
            cy = int(np.clip(v * H, ch // 2, H - ch // 2))
            crop = img[cy - ch // 2 : cy + ch // 2, cx - cw // 2 : cx + cw // 2]
            img = cv2.resize(crop, (W, H), interpolation=cv2.INTER_CUBIC)

        pix = _rgb_to_qpixmap(img)
        scaled = pix.scaled(
            max(self.width() - 8, 64),
            max(self.height() - 8, 64),
            _keep_aspect(),
            _smooth_transform(),
        )
        self.setPixmap(scaled)


# ==============================================================================
# Slider Helper Widget
# ==============================================================================

class ParamSliderRow(QtWidgets.QWidget):
    value_changed = Signal(str, float)

    def __init__(
        self,
        field_name: str,
        label_text: str,
        min_val: float,
        max_val: float,
        initial_val: float,
        step: float = 0.01,
        tooltip: str = "",
    ):
        super().__init__()
        self.field_name = field_name
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.step = float(step)
        self.is_int = step >= 1.0 and float(min_val).is_integer() and float(max_val).is_integer()

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        self.label = QtWidgets.QLabel(label_text)
        self.label.setMinimumWidth(175)
        if tooltip:
            self.label.setToolTip(tooltip)
            self.setToolTip(tooltip)

        self.slider = QtWidgets.QSlider(_horizontal())
        n_steps = max(int(round((self.max_val - self.min_val) / self.step)), 1)
        self.slider.setRange(0, n_steps)

        self.val_label = QtWidgets.QLabel()
        self.val_label.setMinimumWidth(52)
        self.val_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            if hasattr(Qt, "AlignmentFlag")
            else Qt.AlignRight | Qt.AlignVCenter
        )

        layout.addWidget(self.label)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.val_label)

        self.set_value(initial_val, emit=False)
        self.slider.valueChanged.connect(self._on_slider_moved)

    def _on_slider_moved(self, idx: int):
        val = self.min_val + idx * self.step
        val = float(np.clip(val, self.min_val, self.max_val))
        if self.is_int:
            val = float(int(round(val)))
            self.val_label.setText(f"{int(val)}")
        else:
            self.val_label.setText(f"{val:.2f}")
        self.value_changed.emit(self.field_name, val)

    def set_value(self, val: float, emit: bool = False):
        val_c = float(np.clip(val, self.min_val, self.max_val))
        idx = int(round((val_c - self.min_val) / self.step))
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        if self.is_int:
            self.val_label.setText(f"{int(round(val_c))}")
        else:
            self.val_label.setText(f"{val_c:.2f}")
        if emit:
            self.value_changed.emit(self.field_name, val_c)


# ==============================================================================
# Live Root Color Swatch Widget
# ==============================================================================

class RootColorSwatchWidget(QtWidgets.QWidget):
    """Visual real-time color swatch preview showing the young tip and mature root colors."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.tip_color = QtGui.QColor(242, 238, 225)
        self.mature_color = QtGui.QColor(185, 150, 105)
        self.tip_hex = "#F2EEE1"
        self.mat_hex = "#B99669"

    def update_colors(self, tip_rgb_u8: np.ndarray, mat_rgb_u8: np.ndarray):
        tr, tg, tb = int(tip_rgb_u8[0]), int(tip_rgb_u8[1]), int(tip_rgb_u8[2])
        mr, mg, mb = int(mat_rgb_u8[0]), int(mat_rgb_u8[1]), int(mat_rgb_u8[2])
        self.tip_color = QtGui.QColor(tr, tg, tb)
        self.mature_color = QtGui.QColor(mr, mg, mb)
        self.tip_hex = f"#{tr:02X}{tg:02X}{tb:02X}"
        self.mat_hex = f"#{mr:02X}{mg:02X}{mb:02X}"
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        antialiasing = (
            QtGui.QPainter.RenderHint.Antialiasing
            if hasattr(QtGui.QPainter, "RenderHint")
            else QtGui.QPainter.Antialiasing
        )
        painter.setRenderHint(antialiasing)

        w, h = self.width(), self.height()
        box_w = 105
        box_h = h - 10

        # Background frame
        painter.setPen(QtGui.QPen(QtGui.QColor(48, 54, 66), 1))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(22, 26, 33)))
        painter.drawRoundedRect(1, 1, w - 2, h - 2, 6, 6)

        # 1. Tip Swatch (Left)
        x1, y1 = 6, 5
        painter.setPen(QtGui.QPen(QtGui.QColor(70, 80, 95), 1.5))
        painter.setBrush(QtGui.QBrush(self.tip_color))
        painter.drawRoundedRect(x1, y1, box_w, box_h, 4, 4)

        luma_tip = (
            self.tip_color.red() * 299
            + self.tip_color.green() * 587
            + self.tip_color.blue() * 114
        ) / 1000
        text_col_tip = QtGui.QColor(15, 15, 15) if luma_tip > 130 else QtGui.QColor(245, 245, 245)
        painter.setPen(text_col_tip)
        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(x1 + 6, y1 + 15, "Young Tip")
        font.setBold(False)
        font.setPointSize(7)
        painter.setFont(font)
        painter.drawText(x1 + 6, y1 + 31, self.tip_hex)

        # 2. Mature Swatch (Right)
        x2, y2 = w - box_w - 6, 5
        painter.setPen(QtGui.QPen(QtGui.QColor(70, 80, 95), 1.5))
        painter.setBrush(QtGui.QBrush(self.mature_color))
        painter.drawRoundedRect(x2, y2, box_w, box_h, 4, 4)

        luma_mat = (
            self.mature_color.red() * 299
            + self.mature_color.green() * 587
            + self.mature_color.blue() * 114
        ) / 1000
        text_col_mat = QtGui.QColor(15, 15, 15) if luma_mat > 130 else QtGui.QColor(245, 245, 245)
        painter.setPen(text_col_mat)
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(x2 + 6, y2 + 15, "Mature Base")
        font.setBold(False)
        font.setPointSize(7)
        painter.setFont(font)
        painter.drawText(x2 + 6, y2 + 31, self.mat_hex)

        # 3. Age Gradient Bar connecting them (Center)
        gx1 = x1 + box_w + 8
        gx2 = x2 - 8
        gw = gx2 - gx1
        if gw > 24:
            grad = QtGui.QLinearGradient(gx1, y1 + box_h // 2, gx2, y1 + box_h // 2)
            grad.setColorAt(0.0, self.tip_color)
            grad.setColorAt(1.0, self.mature_color)
            painter.setPen(QtGui.QPen(QtGui.QColor(55, 65, 80), 1))
            painter.setBrush(QtGui.QBrush(grad))
            painter.drawRoundedRect(gx1, y1 + box_h // 2 - 5, gw, 10, 3, 3)

            painter.setPen(QtGui.QColor(165, 175, 190))
            font.setPointSize(7)
            font.setBold(False)
            painter.setFont(font)
            align = (
                Qt.AlignmentFlag.AlignCenter
                if hasattr(Qt, "AlignmentFlag")
                else Qt.AlignCenter
            )
            painter.drawText(QtCore.QRectF(gx1, y1, gw, box_h // 2 - 6), align, "Age Ramp ──>")


# ==============================================================================
# Main Studio Window
# ==============================================================================

class RootRealismStudioWindow(QtWidgets.QMainWindow):
    """Main PyQt studio window for interactive soil-specific synthetic root tuning."""

    VALID_SPECIES = [
        "switchgrass",
        "cotton",
        "papaya",
        "sunflower",
        "peanut",
        "sesame",
        "chicory",
        "glycine_max",
        "zea_mays",
        "generic_fibrous",
    ]

    SOIL_GROUPS = [
        "Cotton",
        "Papaya",
        "Peanut",
        "Sesame",
        "Sunflower",
        "Switchgrass",
    ]

    def __init__(self, repo_root: Optional[str] = None):
        super().__init__()
        self.setWindowTitle(
            "Minirhizotron Root Realism Studio (v4) — Environment-Specific U-Net Generator"
        )
        self.resize(1580, 940)
        self._apply_dark_theme()

        self.sim = RealismSimulator(repo_root=repo_root, lazy_root_bank=True)
        first_preset = next(iter(PRESET_ENVIRONMENTS.values()))
        self.cfg = RealismConfig(**first_preset)
        self.gen_cfg = DatasetGenConfig()
        self.soil_index: int = 0
        self.stand_idx: int = 0
        self.frame_idx: int = 0
        self.reference_root_rgb: Optional[np.ndarray] = None

        self.sliders: Dict[str, ParamSliderRow] = {}
        self.gen_sliders: Dict[str, ParamSliderRow] = {}
        self.soil_group_checks: Dict[str, QtWidgets.QCheckBox] = {}
        self.soil_item_checks: Dict[int, QtWidgets.QListWidgetItem] = {}
        self.species_checks: Dict[str, QtWidgets.QCheckBox] = {}
        self.selected_soil_indices: Set[int] = set()
        self.custom_soil_folder: str = ""
        self._last_v4_result: Optional[Dict[str, Any]] = None
        self._last_v3_result: Optional[Dict[str, Any]] = None
        self._preview_worker: Optional[PreviewRenderWorker] = None

        self._build_ui()
        self._populate_soil_list()
        self._show_raw_soil_preview()

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    def _apply_dark_theme(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #1c2026;
                color: #e5e9f0;
                font-family: "Segoe UI", "Inter", sans-serif;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #333a47;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
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
                border: 1px solid #4c566a;
                border-radius: 5px;
                padding: 6px 10px;
                color: #eceff4;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3b4252;
                border-color: #88c0d0;
            }
            QPushButton#runBtn {
                background-color: #1d4ed8;
                border: 1px solid #60a5fa;
                color: #ffffff;
                font-size: 13px;
                font-weight: bold;
                padding: 9px 14px;
                border-radius: 6px;
            }
            QPushButton#runBtn:hover {
                background-color: #2563eb;
            }
            QPushButton#runBtn:disabled {
                background-color: #334155;
                border-color: #475569;
                color: #94a3b8;
            }
            QPushButton#randBtn {
                background-color: #5b21b6;
                border: 1px solid #a78bfa;
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
                padding: 9px 12px;
                border-radius: 6px;
            }
            QPushButton#randBtn:hover {
                background-color: #6d28d9;
            }
            QPushButton#primaryBtn {
                background-color: #2d6a4f;
                border: 1px solid #52b788;
                color: #ffffff;
                font-size: 13px;
                padding: 8px 14px;
            }
            QPushButton#primaryBtn:hover {
                background-color: #40916c;
            }
            QComboBox, QSpinBox, QLineEdit {
                background-color: #242933;
                border: 1px solid #434c5e;
                border-radius: 4px;
                padding: 4px 8px;
                color: #eceff4;
            }
            QTabWidget::pane {
                border: 1px solid #333a47;
                border-radius: 5px;
                background: #20252d;
            }
            QTabBar::tab {
                background: #242933;
                border: 1px solid #333a47;
                padding: 7px 9px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                color: #d8dee9;
            }
            QTabBar::tab:selected {
                background: #2e3440;
                color: #88c0d0;
                font-weight: bold;
                border-bottom-color: #88c0d0;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #2e3440;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #5e81ac;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #88c0d0;
                border: 1px solid #2e3440;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QProgressBar {
                border: 1px solid #434c5e;
                border-radius: 4px;
                text-align: center;
                background-color: #242933;
            }
            QProgressBar::chunk {
                background-color: #52b788;
            }
            """
        )

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # ================== LEFT COLUMN: Soil & Controls ==================
        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(540)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # 1. Environment & Soil Bank Group
        env_group = QtWidgets.QGroupBox("1. Target Environment & Real Soil Source")
        env_layout = QtWidgets.QVBoxLayout(env_group)

        # Preset selector
        preset_row = QtWidgets.QHBoxLayout()
        preset_row.addWidget(QtWidgets.QLabel("Environment Preset:"))
        self.preset_combo = QtWidgets.QComboBox()
        for name in PRESET_ENVIRONMENTS.keys():
            self.preset_combo.addItem(name)
        self.preset_combo.currentTextChanged.connect(self._on_preset_selected)
        preset_row.addWidget(self.preset_combo, stretch=1)
        env_layout.addLayout(preset_row)

        # Species / root architecture prior selector
        crop_row = QtWidgets.QHBoxLayout()
        crop_row.addWidget(QtWidgets.QLabel("Species Prior:"))
        self.crop_combo = QtWidgets.QComboBox()
        for c in self.VALID_SPECIES:
            self.crop_combo.addItem(c)
        self.crop_combo.currentTextChanged.connect(self._on_species_changed)
        crop_row.addWidget(self.crop_combo)

        crop_row.addWidget(QtWidgets.QLabel("Camera Pose:"))
        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_spin.setRange(0, 44)
        self.frame_spin.setValue(self.frame_idx)
        self.frame_spin.valueChanged.connect(self._on_frame_changed)
        crop_row.addWidget(self.frame_spin)

        rand_seed_btn = QtWidgets.QPushButton("Next Pose")
        rand_seed_btn.clicked.connect(self._on_random_roots)
        crop_row.addWidget(rand_seed_btn)
        env_layout.addLayout(crop_row)

        # Soil image selector
        soil_row = QtWidgets.QHBoxLayout()
        soil_row.addWidget(QtWidgets.QLabel("Real Soil Patch:"))
        self.soil_combo = QtWidgets.QComboBox()
        self.soil_combo.currentIndexChanged.connect(self._on_soil_index_changed)
        soil_row.addWidget(self.soil_combo, stretch=1)

        next_soil_btn = QtWidgets.QPushButton("Random Soil")
        next_soil_btn.clicked.connect(self._on_random_soil)
        soil_row.addWidget(next_soil_btn)
        env_layout.addLayout(soil_row)

        # Custom soil folder + Multi-file soil loader + Reference real-root image buttons
        folder_row = QtWidgets.QHBoxLayout()
        load_soil_btn = QtWidgets.QPushButton("Load Soil Folder...")
        load_soil_btn.setToolTip("Load all root-free soil images from a folder (replaces current soil list).")
        load_soil_btn.clicked.connect(self._on_load_custom_soil_folder)
        folder_row.addWidget(load_soil_btn)

        add_soil_files_btn = QtWidgets.QPushButton("+ Add Soil Files...")
        add_soil_files_btn.setToolTip("Select one or multiple soil image files (.png/.jpg) to add to the loaded soils.")
        add_soil_files_btn.clicked.connect(self._on_add_custom_soil_files)
        folder_row.addWidget(add_soil_files_btn)

        load_ref_btn = QtWidgets.QPushButton("Load Root Ref...")
        load_ref_btn.setToolTip("Load a real root reference photo for visual side-by-side comparison.")
        load_ref_btn.clicked.connect(self._on_load_reference_image)
        folder_row.addWidget(load_ref_btn)
        env_layout.addLayout(folder_row)

        left_layout.addWidget(env_group)

        # Action Row: Prominent RUN button + Test Random Sample button
        action_row = QtWidgets.QHBoxLayout()
        self.run_preview_btn = QtWidgets.QPushButton("▶  RUN / UPDATE ROOT PREVIEW")
        self.run_preview_btn.setObjectName("runBtn")
        self.run_preview_btn.clicked.connect(self.render_current_scene)
        action_row.addWidget(self.run_preview_btn, stretch=2)

        self.rand_sample_btn = QtWidgets.QPushButton("🎲 Sample Random Config")
        self.rand_sample_btn.setObjectName("randBtn")
        self.rand_sample_btn.setToolTip(
            "Draw a random parameter & soil sample using your settings in the 'Dataset & Random' tab!"
        )
        self.rand_sample_btn.clicked.connect(self._on_sample_random_config)
        action_row.addWidget(self.rand_sample_btn, stretch=1)
        left_layout.addLayout(action_row)

        # 2. Tabbed Parameter Sliders + 5th Tab for Dataset & Randomization Control
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_architecture_tab(), "Architecture")
        tabs.addTab(self._build_epidermis_tab(), "Epidermis")
        tabs.addTab(self._build_soil_physics_tab(), "Soil Physics")
        tabs.addTab(self._build_optics_tab(), "Optics")
        tabs.addTab(self._build_dataset_randomization_tab(), "Dataset & Random")
        left_layout.addWidget(tabs, stretch=1)

        # Preset Save / Load buttons
        cfg_btn_row = QtWidgets.QHBoxLayout()
        save_cfg_btn = QtWidgets.QPushButton("Save Config JSON...")
        save_cfg_btn.clicked.connect(self._on_save_config)
        load_cfg_btn = QtWidgets.QPushButton("Load Config JSON...")
        load_cfg_btn.clicked.connect(self._on_load_config)
        cfg_btn_row.addWidget(save_cfg_btn)
        cfg_btn_row.addWidget(load_cfg_btn)
        left_layout.addLayout(cfg_btn_row)

        # 3. Dataset Generation Group
        export_group = QtWidgets.QGroupBox(
            "3. Export Environment-Specific U-Net Training Dataset"
        )
        export_layout = QtWidgets.QVBoxLayout(export_group)

        count_row = QtWidgets.QHBoxLayout()
        count_row.addWidget(QtWidgets.QLabel("3D Stands:"))
        self.n_sys_spin = QtWidgets.QSpinBox()
        self.n_sys_spin.setRange(1, 500)
        self.n_sys_spin.setValue(self.gen_cfg.n_systems)
        count_row.addWidget(self.n_sys_spin)

        count_row.addWidget(QtWidgets.QLabel("Frames/Stand:"))
        self.n_frames_spin = QtWidgets.QSpinBox()
        self.n_frames_spin.setRange(1, 500)
        self.n_frames_spin.setValue(self.gen_cfg.frames_per_system)
        count_row.addWidget(self.n_frames_spin)

        count_row.addWidget(QtWidgets.QLabel("Output:"))
        self.out_dir_edit = QtWidgets.QLineEdit("datasets/custom_env_v4")
        count_row.addWidget(self.out_dir_edit, stretch=1)
        browse_out_btn = QtWidgets.QPushButton("...")
        browse_out_btn.setFixedWidth(30)
        browse_out_btn.clicked.connect(self._on_browse_output_dir)
        count_row.addWidget(browse_out_btn)
        export_layout.addLayout(count_row)

        run_row = QtWidgets.QHBoxLayout()
        run_row.addWidget(QtWidgets.QLabel("Run Tag:"))
        self.run_tag_edit = QtWidgets.QLineEdit()
        self.run_tag_edit.setPlaceholderText("auto (r01, r02...) or custom (e.g. wet_soil)")
        self.run_tag_edit.setToolTip(
            "Unique run prefix in filenames (e.g. r01_s000_0000.png).\n"
            "Leave blank to auto-detect and increment (r01, r02, r03...)."
        )
        run_row.addWidget(self.run_tag_edit, stretch=1)

        self.append_runs_check = QtWidgets.QCheckBox("Keep & Append Runs")
        self.append_runs_check.setChecked(True)
        self.append_runs_check.setToolTip(
            "When checked, new runs are saved alongside previous runs in the same folder\n"
            "without overwriting files, and new rows are appended to traits.csv."
        )
        run_row.addWidget(self.append_runs_check)
        export_layout.addLayout(run_row)

        self.generate_btn = QtWidgets.QPushButton(
            "Generate Dataset (Uses 'Dataset & Random' Tab Rules)"
        )
        self.generate_btn.setObjectName("primaryBtn")
        self.generate_btn.clicked.connect(self._on_export_dataset)
        export_layout.addWidget(self.generate_btn)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        export_layout.addWidget(self.progress_bar)

        left_layout.addWidget(export_group)
        main_layout.addWidget(left_panel)

        # ================== RIGHT COLUMN: Multi-View Canvases ==================
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # Top toolbar for view mode & zoom
        toolbar = QtWidgets.QHBoxLayout()
        self.zoom_check = QtWidgets.QCheckBox(
            "2x Detail Loupe (Click any image to pan inspection window)"
        )
        self.zoom_check.setStyleSheet("font-weight: bold; color: #a3be8c;")
        self.zoom_check.toggled.connect(self._on_zoom_toggled)
        toolbar.addWidget(self.zoom_check)

        self.compare_v3_check = QtWidgets.QCheckBox("Also Render Legacy v3 Comparison (Slower)")
        self.compare_v3_check.setChecked(False)
        toolbar.addWidget(self.compare_v3_check)

        toolbar.addStretch(1)
        self.status_label = QtWidgets.QLabel("Adjust sliders, then click '▶ RUN / UPDATE ROOT PREVIEW'")
        self.status_label.setStyleSheet("color: #ebcb8b; font-weight: bold; font-size: 13px;")
        toolbar.addWidget(self.status_label)
        right_layout.addLayout(toolbar)

        # 3 Side-by-side image panels
        canvas_grid = QtWidgets.QHBoxLayout()
        canvas_grid.setSpacing(10)

        c1, self.panel_v4, self.title_v4 = self._create_titled_canvas(
            "Realistic v4 Simulation (Continuous Spline + Soil Physics)"
        )
        c2, self.panel_mask, self.title_mask = self._create_titled_canvas(
            "U-Net Ground Truth (Mask + Diameter Heatmap)"
        )
        c3, self.panel_ref, self.title_ref = self._create_titled_canvas(
            "Selected Real Soil / Real Root Reference"
        )

        for canvas in (self.panel_v4, self.panel_mask, self.panel_ref):
            canvas.roi_changed.connect(self._on_canvas_roi_changed)

        canvas_grid.addWidget(c1, stretch=1)
        canvas_grid.addWidget(c2, stretch=1)
        canvas_grid.addWidget(c3, stretch=1)
        right_layout.addLayout(canvas_grid, stretch=1)

        # Bottom trait & training command bar
        self.info_bar = QtWidgets.QLabel()
        self.info_bar.setWordWrap(True)
        self.info_bar.setStyleSheet(
            "background-color: #242933; border: 1px solid #333a47; "
            "border-radius: 5px; padding: 8px; color: #d8dee9; font-family: Consolas, monospace;"
        )
        right_layout.addWidget(self.info_bar)

        main_layout.addWidget(right_panel, stretch=1)

    def _create_titled_canvas(
        self, title_text: str
    ) -> Tuple[QtWidgets.QWidget, ZoomableImageLabel, QtWidgets.QLabel]:
        container = QtWidgets.QWidget(self)
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(4)

        lbl = QtWidgets.QLabel(title_text, container)
        lbl.setAlignment(_align_center())
        lbl.setStyleSheet(
            "background-color: #242933; color: #eceff4; font-weight: bold; "
            "padding: 5px; border-radius: 4px; border: 1px solid #333a47;"
        )
        canvas = ZoomableImageLabel(title_text)
        vbox.addWidget(lbl)
        vbox.addWidget(canvas, stretch=1)
        return container, canvas, lbl

    # ------------------------------------------------------------------
    # Parameter Slider Tabs (Mapped 1:1 to RealismConfig & DatasetGenConfig)
    # ------------------------------------------------------------------
    def _add_slider(
        self,
        layout: QtWidgets.QVBoxLayout,
        field_name: str,
        label_text: str,
        min_val: float,
        max_val: float,
        step: float = 0.01,
        tooltip: str = "",
    ):
        init_val = float(getattr(self.cfg, field_name))
        row = ParamSliderRow(
            field_name=field_name,
            label_text=label_text,
            min_val=min_val,
            max_val=max_val,
            initial_val=init_val,
            step=step,
            tooltip=tooltip,
        )
        row.value_changed.connect(self._on_param_slider_changed)
        layout.addWidget(row)
        self.sliders[field_name] = row

    def _add_gen_slider(
        self,
        layout: QtWidgets.QVBoxLayout,
        field_name: str,
        label_text: str,
        min_val: float,
        max_val: float,
        step: float = 0.02,
        tooltip: str = "",
    ):
        init_val = float(getattr(self.gen_cfg, field_name))
        row = ParamSliderRow(
            field_name=field_name,
            label_text=label_text,
            min_val=min_val,
            max_val=max_val,
            initial_val=init_val,
            step=step,
            tooltip=tooltip,
        )
        row.value_changed.connect(self._on_gen_slider_changed)
        layout.addWidget(row)
        self.gen_sliders[field_name] = row

    def _build_architecture_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(page)
        vbox.setSpacing(4)

        # Stand Density & Growth Age
        self._add_slider(
            vbox, "root_keep", "Visible Root Density", 0.05, 1.00, 0.02,
            "Fraction of connected root subtrees visible at the tube window."
        )
        self._add_slider(
            vbox, "days", "Plant Growth Age (days)", 10.0, 50.0, 2.0,
            "Duration of 3D root system growth before tube projection."
        )

        # Diameter & Thinness Tuning Group Box
        diam_box = QtWidgets.QGroupBox("Root Diameter & Thinness Tuning")
        d_vbox = QtWidgets.QVBoxLayout(diam_box)
        d_vbox.setSpacing(6)

        # Live physical diameter readout
        self.diam_readout_label = QtWidgets.QLabel()
        self.diam_readout_label.setStyleSheet(
            "background-color: #222730; border: 1px solid #3b4252; border-radius: 5px; padding: 6px; font-size: 11px;"
        )
        d_vbox.addWidget(self.diam_readout_label)

        # 4 Quick Diameter Preset Buttons
        btn_grid = QtWidgets.QGridLayout()
        btn_grid.setSpacing(4)
        diam_presets = [
            ("🧵 Ultra-Thin (< 0.15mm)", self._apply_diam_ultra_thin),
            ("🌾 Fine Grass (~0.25mm)", self._apply_diam_fine),
            ("🌿 Balanced (~0.50mm)", self._apply_diam_balanced),
            ("🥕 Thick / Fleshy (> 1.2mm)", self._apply_diam_thick),
        ]
        for idx, (title, handler) in enumerate(diam_presets):
            btn = QtWidgets.QPushButton(title)
            btn.setStyleSheet("font-size: 11px; padding: 4px 6px; font-weight: bold;")
            btn.clicked.connect(handler)
            btn_grid.addWidget(btn, idx // 2, idx % 2)
        d_vbox.addLayout(btn_grid)

        # Diameter Sliders
        self._add_slider(
            d_vbox, "diam_scale", "Master Diameter Scale", 0.15, 2.50, 0.02,
            "Global scaling multiplier for all roots (set to 0.25-0.45 for ultra-thin capillary roots)."
        )
        self._add_slider(
            d_vbox, "diam_lat_mult", "Lateral Diameter Mult", 0.10, 2.50, 0.02,
            "Multiplier on fine lateral branches (set low for 1-2 px hair-thin laterals)."
        )
        self._add_slider(
            d_vbox, "diam_axis_mult", "Main Axis Diameter Mult", 0.15, 2.50, 0.02,
            "Multiplier on primary/taproot axis diameters."
        )
        self._add_slider(
            d_vbox, "diam_min_mm", "Minimum Diameter Floor (mm)", 0.02, 0.30, 0.01,
            "Hard floor on finest root tips in mm (0.03 mm ~ 0.35 px on CID scanner)."
        )
        vbox.addWidget(diam_box)

        # Curvature & Micro-Geometry
        self._add_slider(
            vbox, "tortuosity_mult", "Biological Tortuosity", 0.60, 2.20, 0.05,
            "3D growth meandering around soil peds."
        )
        self._add_slider(
            vbox, "spline_smooth_mm", "C2 Spline Smoothing (mm)", 0.20, 4.00, 0.10,
            "Eliminates polygonal kinks via arc-length cubic B-spline smoothing."
        )
        self._add_slider(
            vbox, "junction_collar_flare", "Branch Collar Flare", 0.0, 0.85, 0.05,
            "Natural basal thickening where a lateral root emerges from its parent."
        )
        self._add_slider(
            vbox, "edge_roughness_px", "Organic Edge Waviness (px)", 0.0, 1.20, 0.05,
            "Sub-pixel biological diameter modulation along the root boundary."
        )
        vbox.addStretch(1)
        return page

    def _build_epidermis_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(page)
        vbox.setSpacing(4)

        # Quick Root Color & Brightness Tuning Box
        preset_box = QtWidgets.QGroupBox("Root Color & Brightness Tuning")
        p_vbox = QtWidgets.QVBoxLayout(preset_box)
        p_vbox.setSpacing(6)

        # Live visual color swatch preview (Young Tip -> Mature Root)
        self.color_swatch = RootColorSwatchWidget()
        p_vbox.addWidget(self.color_swatch)

        # 5 Quick Color Preset Buttons
        btn_grid = QtWidgets.QGridLayout()
        btn_grid.setSpacing(4)
        presets = [
            ("🌟 Pure White", self._apply_color_pure_white),
            ("🌾 Ivory / Cream", self._apply_color_cream),
            ("🍂 Golden Tan", self._apply_color_golden_tan),
            ("🪵 Rich Brown", self._apply_color_rich_brown),
            ("🌰 Dark Woody", self._apply_color_dark_woody),
        ]
        for idx, (title, handler) in enumerate(presets):
            btn = QtWidgets.QPushButton(title)
            btn.setStyleSheet("font-size: 11px; padding: 4px 6px; font-weight: bold;")
            btn.clicked.connect(handler)
            btn_grid.addWidget(btn, idx // 3, idx % 3)
        p_vbox.addLayout(btn_grid)
        vbox.addWidget(preset_box)

        # Sliders: Color & Brightness
        self._add_slider(
            vbox, "root_brightness", "Root Brightness", 0.30, 2.00, 0.02,
            "Overall epidermal brightness (0.3 = dark/subdued, 1.0 = balanced, 1.8 = glowing bright)."
        )
        self._add_slider(
            vbox, "root_whiteness", "Whiteness / Chalky Luma", 0.0, 1.0, 0.02,
            "Lifts roots towards pure chalky/pearly white and suppresses tan/brown aging."
        )
        self._add_slider(
            vbox, "root_brownness", "Brownness / Suberization", 0.0, 1.0, 0.02,
            "Deepens roots into rich suberized earthy brown and woody bark (can sit darker than soil)."
        )
        self._add_slider(
            vbox, "root_warmth", "Epidermal Warmth", 0.0, 1.0, 0.02,
            "Warm golden/tan vs neutral cool white."
        )
        self._add_slider(
            vbox, "root_red_tint", "Red-Brown / Mahogany Tint", -1.0, 1.0, 0.05,
            "Color shift: -1 = olive/yellow cream, 0 = natural tan/brown, +1 = reddish brown / mahogany."
        )
        self._add_slider(
            vbox, "root_saturation", "Color Saturation", 0.0, 2.0, 0.05,
            "0 = monochrome achromatic white/grey/black, 1 = natural, 2 = rich saturated color."
        )
        self._add_slider(
            vbox, "contrast_target_n", "Target Contrast vs Soil", 0.10, 1.40, 0.02,
            "Target signal-to-noise contrast above the real soil background."
        )
        self._add_slider(
            vbox, "root_age_tint", "Age Progression Tint", 0.0, 1.0, 0.02,
            "Color difference between young meristem tip and mature suberized base."
        )
        self._add_slider(
            vbox, "dark_root_frac", "Senescing Dark Root Share", 0.0, 0.60, 0.02,
            "Fraction of older dark-brown/black senescing roots in the stand."
        )
        self._add_slider(
            vbox, "cylinder_shading", "3D Cylinder Shading", 0.0, 1.0, 0.02,
            "3D Lambertian + rim darkening across the tubular root profile."
        )
        self._add_slider(
            vbox, "stele_strength", "Central Vascular Stele", 0.0, 0.65, 0.02,
            "Internal vascular cylinder optical signature inside translucent roots."
        )
        self._add_slider(
            vbox, "real_texture_mix", "Real Root Strip Texture", 0.0, 1.0, 0.05,
            "Blend weight of real PRMI root epidermal texture strips."
        )
        self._add_slider(
            vbox, "root_hairs_density", "Root Hair Density (/mm)", 0.0, 4.5, 0.1,
            "Density of curved root hairs along the maturation zone."
        )
        vbox.addStretch(1)
        return page

    def _build_soil_physics_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(page)
        vbox.setSpacing(4)
        self._add_slider(
            vbox, "soil_burial_strength", "Real-Ped Burial Strength", 0.0, 0.90, 0.02,
            "Strength of root burial behind actual foreground soil aggregates and crevices."
        )
        self._add_slider(
            vbox, "soil_aggregate_scale_mm", "Aggregate Scale (mm)", 0.6, 4.0, 0.1,
            "Characteristic spatial scale of soil peds occluding the root."
        )
        self._add_slider(
            vbox, "surface_soil_film", "Opaque Adherent Soil Film", 0.0, 0.95, 0.02,
            "Coats the root surface with real soil color/particles WITHOUT ghost transparency."
        )
        self._add_slider(
            vbox, "flank_soil_crust", "Cylinder Flank Soil Crust", 0.0, 0.95, 0.02,
            "Extra soil encrustation along cylinder flanks where the root curves into soil."
        )
        self._add_slider(
            vbox, "contact_shadow_strength", "Soil-Root Crevice Shadow", 0.0, 0.75, 0.02,
            "Dark air-gap / crevice shadow along the root-soil contact boundary."
        )
        self._add_slider(
            vbox, "contact_shadow_width_px", "Crevice Shadow Width (px)", 1.0, 7.0, 0.2,
            "Width of the contact crevice shadow in pixels."
        )
        self._add_slider(
            vbox, "water_droplet_occl", "Tube Condensation Occlusion", 0.0, 1.0, 0.05,
            "Allows specular water droplets on the tube wall to sit in front of roots."
        )
        vbox.addStretch(1)
        return page

    def _build_optics_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(page)
        vbox.setSpacing(4)
        self._add_slider(
            vbox, "local_illum_match", "Match Soil Illumination", 0.0, 1.0, 0.05,
            "Inherit spatial LED lighting & vignetting from the real soil patch."
        )
        self._add_slider(
            vbox, "local_color_cast_match", "Camera White-Balance Lock", 0.0, 1.0, 0.05,
            "Inherit camera color cast / white balance from the real soil patch."
        )
        self._add_slider(
            vbox, "match_soil_noise", "Match Soil Sensor Grain", 0.0, 1.0, 0.05,
            "Inject high-frequency sensor grain matching the real soil patch."
        )
        self._add_slider(
            vbox, "depth_fade_mm", "Depth Recession Scale (mm)", 0.3, 3.0, 0.1,
            "Depth behind the tube wall over which roots blur and fade into soil."
        )
        self._add_slider(
            vbox, "depth_blur_max_px", "Depth Defocus Blur (px)", 0.0, 4.5, 0.1,
            "Maximum defocus blur for deeper root stretches."
        )
        self._add_slider(
            vbox, "psf_sigma_px", "Camera Optical PSF (px)", 0.0, 2.0, 0.05,
            "Optical point-spread function sigma of the minirhizotron scanner."
        )
        vbox.addStretch(1)
        return page

    def _build_dataset_randomization_tab(self) -> QtWidgets.QWidget:
        """Tab giving the user complete control over soil selection, species mix, and parameter jitter."""
        page = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(page)
        vbox.setSpacing(6)

        # --- A. Soil Selection Mode ---
        soil_box = QtWidgets.QGroupBox("A. Soil Selection for Dataset Export & Random Sampling")
        s_layout = QtWidgets.QVBoxLayout(soil_box)

        s_mode_row = QtWidgets.QHBoxLayout()
        s_mode_row.addWidget(QtWidgets.QLabel("Soil Sampling:"))
        self.soil_mode_combo = QtWidgets.QComboBox()
        self.soil_mode_combo.addItem("Randomize Across All Checked Soils Below", "all_soils")
        self.soil_mode_combo.addItem("Selected Soils from Checklist Below (Pick Any)", "custom_selection")
        self.soil_mode_combo.addItem("Only Checked Soil Types Below (PRMI Crops)", "selected_groups")
        self.soil_mode_combo.addItem("Current Selected Soil Patch Only (1 Soil)", "current_soil_only")
        self.soil_mode_combo.currentIndexChanged.connect(self._sync_gen_cfg_from_widgets)
        s_mode_row.addWidget(self.soil_mode_combo, stretch=1)
        s_layout.addLayout(s_mode_row)

        # Soil group checkboxes (2 rows of 3)
        g_grid = QtWidgets.QGridLayout()
        for i, gname in enumerate(self.SOIL_GROUPS):
            chk = QtWidgets.QCheckBox(gname)
            chk.setChecked(True)
            chk.toggled.connect(self._sync_gen_cfg_from_widgets)
            self.soil_group_checks[gname] = chk
            g_grid.addWidget(chk, i // 3, i % 3)
        s_layout.addLayout(g_grid)

        # Multi-Soil Checklist: Allows picking specific soils among any loaded soils
        self.soil_list_widget = QtWidgets.QListWidget()
        self.soil_list_widget.setMaximumHeight(150)
        self.soil_list_widget.setToolTip(
            "Check [✓] all soils you want the simulator to randomize across.\n"
            "Click any soil name to preview it immediately."
        )
        self.soil_list_widget.setStyleSheet(
            "QListWidget { background-color: #1a1e24; border: 1px solid #333a47; border-radius: 4px; }"
            "QListWidget::item { padding: 2px 4px; color: #eceff4; }"
            "QListWidget::item:hover { background-color: #2e3440; }"
        )
        self.soil_list_widget.itemChanged.connect(self._on_soil_list_item_changed)
        self.soil_list_widget.itemClicked.connect(self._on_soil_list_item_clicked)
        s_layout.addWidget(self.soil_list_widget)

        # Soil checklist action buttons
        btn_soil_row = QtWidgets.QHBoxLayout()
        btn_sel_all = QtWidgets.QPushButton("Select All")
        btn_sel_all.setFixedHeight(24)
        btn_sel_all.clicked.connect(lambda: self._set_all_soil_items_checked(True))
        btn_soil_row.addWidget(btn_sel_all)

        btn_sel_none = QtWidgets.QPushButton("Deselect All")
        btn_sel_none.setFixedHeight(24)
        btn_sel_none.clicked.connect(lambda: self._set_all_soil_items_checked(False))
        btn_soil_row.addWidget(btn_sel_none)

        btn_add_more = QtWidgets.QPushButton("+ Add Soil Files...")
        btn_add_more.setFixedHeight(24)
        btn_add_more.setToolTip("Select multiple soil image files (.png/.jpg) to add to this list.")
        btn_add_more.clicked.connect(self._on_add_custom_soil_files)
        btn_soil_row.addWidget(btn_add_more)

        self.soil_sel_count_label = QtWidgets.QLabel("0 soils active")
        self.soil_sel_count_label.setStyleSheet("color: #a3be8c; font-weight: bold;")
        btn_soil_row.addWidget(self.soil_sel_count_label, stretch=1)
        s_layout.addLayout(btn_soil_row)

        self.soil_flip_check = QtWidgets.QCheckBox("Random Horizontal / Vertical Soil Flips")
        self.soil_flip_check.setChecked(True)
        self.soil_flip_check.toggled.connect(self._sync_gen_cfg_from_widgets)
        s_layout.addWidget(self.soil_flip_check)

        self._add_gen_slider(
            s_layout, "soil_exposure_jitter", "Soil Exposure/Contrast Jitter", 0.0, 0.25, 0.01,
            "Randomly varies soil brightness/contrast across generated frames for lighting robustness."
        )
        vbox.addWidget(soil_box)

        # --- B. Species Selection Mode ---
        sp_box = QtWidgets.QGroupBox("B. Root System Species Sampling")
        sp_layout = QtWidgets.QVBoxLayout(sp_box)

        sp_mode_row = QtWidgets.QHBoxLayout()
        sp_mode_row.addWidget(QtWidgets.QLabel("Species Mode:"))
        self.species_mode_combo = QtWidgets.QComboBox()
        self.species_mode_combo.addItem("Current Selected Species Only", "current_only")
        self.species_mode_combo.addItem("Random Mix of Checked Species Below", "selected_species")
        self.species_mode_combo.currentIndexChanged.connect(self._sync_gen_cfg_from_widgets)
        sp_mode_row.addWidget(self.species_mode_combo, stretch=1)
        sp_layout.addLayout(sp_mode_row)

        sp_grid = QtWidgets.QGridLayout()
        for i, sp_name in enumerate(self.VALID_SPECIES):
            chk = QtWidgets.QCheckBox(sp_name)
            chk.setChecked(True)
            chk.toggled.connect(self._sync_gen_cfg_from_widgets)
            self.species_checks[sp_name] = chk
            sp_grid.addWidget(chk, i // 4, i % 4)
        sp_layout.addLayout(sp_grid)
        vbox.addWidget(sp_box)

        # --- C. Parameter Randomization Mode ---
        pm_box = QtWidgets.QGroupBox("C. Parameter Randomization / Jitter")
        pm_layout = QtWidgets.QVBoxLayout(pm_box)

        pm_row = QtWidgets.QHBoxLayout()
        pm_row.addWidget(QtWidgets.QLabel("Parameter Mode:"))
        self.param_mode_combo = QtWidgets.QComboBox()
        self.param_mode_combo.addItem("Controlled Jitter Around GUI Sliders", "jitter_around_gui")
        self.param_mode_combo.addItem("Exact GUI Sliders (0% Variation)", "exact_gui")
        self.param_mode_combo.addItem("Full Domain Randomization (All Ranges)", "full_random")
        self.param_mode_combo.currentIndexChanged.connect(self._sync_gen_cfg_from_widgets)
        pm_row.addWidget(self.param_mode_combo, stretch=1)
        pm_layout.addLayout(pm_row)

        self._add_gen_slider(
            pm_layout, "arch_jitter", "Architecture & Density Jitter", 0.0, 1.0, 0.05,
            "Random variation around your Architecture tab sliders (0 = locked, 1 = wide variation)."
        )
        self._add_gen_slider(
            pm_layout, "epidermis_jitter", "Epidermis, Color & Hair Jitter", 0.0, 1.0, 0.05,
            "Random variation around your Epidermis tab sliders (0 = locked, 1 = wide variation)."
        )
        self._add_gen_slider(
            pm_layout, "soil_physics_jitter", "Soil Burial & Film Jitter", 0.0, 1.0, 0.05,
            "Random variation around your Soil Physics tab sliders (0 = locked, 1 = wide variation)."
        )
        self._add_gen_slider(
            pm_layout, "optics_jitter", "Camera & Blur Jitter", 0.0, 1.0, 0.05,
            "Random variation around your Optics tab sliders (0 = locked, 1 = wide variation)."
        )
        self._add_gen_slider(
            pm_layout, "empty_frame_frac", "Empty (Root-Free) Soil Fraction", 0.0, 0.50, 0.05,
            "Fraction of generated training frames that contain 0 roots (negative examples for U-Net)."
        )
        vbox.addWidget(pm_box)
        vbox.addStretch(1)
        return page

    # ------------------------------------------------------------------
    # Event Handlers (Zero-Lag: Only update state, user clicks Run!)
    # ------------------------------------------------------------------
    def _mark_dirty(self, msg: str = "Settings changed — Click '▶ RUN / UPDATE ROOT PREVIEW'"):
        self.status_label.setText(msg)
        self.status_label.setStyleSheet("color: #ebcb8b; font-weight: bold; font-size: 13px;")

    def _sync_gen_cfg_from_widgets(self):
        self.gen_cfg.soil_mode = str(self.soil_mode_combo.currentData())
        self.gen_cfg.selected_soil_groups = tuple(
            g for g, chk in self.soil_group_checks.items() if chk.isChecked()
        )
        self.gen_cfg.selected_soil_indices = tuple(sorted(self.selected_soil_indices))
        self.gen_cfg.custom_soil_folder = str(self.custom_soil_folder)
        self.gen_cfg.current_soil_index = int(self.soil_index)
        self.gen_cfg.soil_flip_augment = bool(self.soil_flip_check.isChecked())

        self.gen_cfg.species_mode = str(self.species_mode_combo.currentData())
        self.gen_cfg.selected_species = tuple(
            sp for sp, chk in self.species_checks.items() if chk.isChecked()
        )
        self.gen_cfg.param_mode = str(self.param_mode_combo.currentData())

        if hasattr(self, "run_tag_edit"):
            self.gen_cfg.run_version = self.run_tag_edit.text().strip()
        if hasattr(self, "append_runs_check"):
            self.gen_cfg.append_runs = bool(self.append_runs_check.isChecked())
        self._update_soil_sel_count_label()

    def _on_soil_list_item_changed(self, item: QtWidgets.QListWidgetItem):
        idx = item.data(QtCore.Qt.UserRole)
        if idx is not None:
            if item.checkState() == QtCore.Qt.Checked:
                self.selected_soil_indices.add(int(idx))
            else:
                self.selected_soil_indices.discard(int(idx))
            self._sync_gen_cfg_from_widgets()

    def _on_soil_list_item_clicked(self, item: QtWidgets.QListWidgetItem):
        idx = item.data(QtCore.Qt.UserRole)
        if idx is not None and 0 <= int(idx) < self.soil_combo.count():
            self.soil_combo.setCurrentIndex(int(idx))

    def _set_all_soil_items_checked(self, checked: bool):
        if not hasattr(self, "soil_list_widget"):
            return
        self.soil_list_widget.blockSignals(True)
        for i in range(self.soil_list_widget.count()):
            item = self.soil_list_widget.item(i)
            item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
            idx = item.data(QtCore.Qt.UserRole)
            if idx is not None:
                if checked:
                    self.selected_soil_indices.add(int(idx))
                else:
                    self.selected_soil_indices.discard(int(idx))
        self.soil_list_widget.blockSignals(False)
        self._sync_gen_cfg_from_widgets()

    def _select_current_soil_in_list(self):
        if not hasattr(self, "soil_list_widget"):
            return
        self.soil_list_widget.blockSignals(True)
        self.selected_soil_indices.clear()
        for i in range(self.soil_list_widget.count()):
            item = self.soil_list_widget.item(i)
            idx = item.data(QtCore.Qt.UserRole)
            is_cur = bool(idx == self.soil_index)
            item.setCheckState(QtCore.Qt.Checked if is_cur else QtCore.Qt.Unchecked)
            if is_cur and idx is not None:
                self.selected_soil_indices.add(int(idx))
        self.soil_list_widget.blockSignals(False)
        self._sync_gen_cfg_from_widgets()

    def _update_soil_sel_count_label(self):
        if hasattr(self, "soil_sel_count_label"):
            n_tot = len(self.sim.soil_paths)
            if hasattr(self, "soil_mode_combo") and self.soil_mode_combo.currentData() == "current_soil_only":
                self.soil_sel_count_label.setText(f"1/{n_tot} soil active (current only)")
            else:
                n_sel = len(self.selected_soil_indices)
                self.soil_sel_count_label.setText(f"{n_sel}/{n_tot} soils active for random sampling")

    def _on_gen_slider_changed(self, field_name: str, val: float):
        setattr(self.gen_cfg, field_name, val)

    def _on_sample_random_config(self):
        """Draw a random parameter & soil sample using the user's Dataset & Randomization rules."""
        self._sync_gen_cfg_from_widgets()
        rng = np.random.default_rng(int(time.time() * 1000) % 2**31)

        # Pick a soil from the user's configured soil pool (preferring a different soil than current)
        cand_soils = self.sim.get_filtered_soil_indices(self.gen_cfg)
        if cand_soils:
            other_soils = [s for s in cand_soils if int(s) != int(self.soil_index)]
            pool = other_soils if other_soils else cand_soils
            self.soil_index = int(pool[int(rng.integers(0, len(pool)))])
            self.soil_combo.blockSignals(True)
            self.soil_combo.setCurrentIndex(self.soil_index)
            self.soil_combo.blockSignals(False)

        # Sample a random config according to gen_cfg
        self.cfg = sample_randomized_config(self.cfg, self.gen_cfg, rng, for_new_stand=True)
        self._sync_widgets_from_cfg()
        self._show_raw_soil_preview(force_canvas_update=True)
        self.render_current_scene()

    def _populate_soil_list(self):
        self.soil_combo.blockSignals(True)
        self.soil_combo.clear()
        if hasattr(self, "soil_list_widget"):
            self.soil_list_widget.blockSignals(True)
            self.soil_list_widget.clear()

        self.selected_soil_indices = set(range(len(self.sim.soil_paths)))
        for idx, p in enumerate(self.sim.soil_paths):
            name = Path(p).name
            self.soil_combo.addItem(f"[{idx:03d}] {name}")
            if hasattr(self, "soil_list_widget"):
                item = QtWidgets.QListWidgetItem(f"[{idx:03d}] {name}")
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked)
                item.setData(QtCore.Qt.UserRole, idx)
                self.soil_list_widget.addItem(item)

        self.soil_combo.setCurrentIndex(
            min(self.soil_index, max(self.soil_combo.count() - 1, 0))
        )
        self.soil_combo.blockSignals(False)
        if hasattr(self, "soil_list_widget"):
            self.soil_list_widget.blockSignals(False)
        self._sync_gen_cfg_from_widgets()

    def _show_raw_soil_preview(self, force_canvas_update: bool = False):
        """Immediately show the selected real soil patch (< 10ms) without running root growth."""
        if not self.sim.soil_paths:
            return
        try:
            bg = self.sim.get_soil_rgb(self.soil_index)
            if self._last_v4_result is None or force_canvas_update:
                self.panel_v4.set_image(bg)
                self.panel_mask.set_image(bg)
            if self.reference_root_rgb is not None:
                self.title_ref.setText("Target Real Root Reference Image (Match Visually)")
                self.panel_ref.set_image(self.reference_root_rgb)
            else:
                self.title_ref.setText("Selected Real Soil Patch (Root-Free Background)")
                self.panel_ref.set_image(bg)
            soil_name = Path(self.sim.soil_paths[self.soil_index % len(self.sim.soil_paths)]).name
            cand_count = len(self.sim.get_filtered_soil_indices(self.gen_cfg))
            self.info_bar.setText(
                f"Active Preview Soil: {soil_name}  |  Random Sampling Pool: {cand_count}/{len(self.sim.soil_paths)} soils\n"
                f"Adjust sliders or use 'Dataset & Random' tab, then click '▶ RUN / UPDATE ROOT PREVIEW'."
            )
            self._update_color_swatch()
        except Exception:
            pass

    def _sync_widgets_from_cfg(self):
        self.crop_combo.blockSignals(True)
        sp_norm = self.sim.normalize_species(self.cfg.species)
        idx = self.crop_combo.findText(sp_norm)
        if idx >= 0:
            self.crop_combo.setCurrentIndex(idx)
        self.crop_combo.blockSignals(False)

        for name, slider in self.sliders.items():
            if hasattr(self.cfg, name):
                slider.set_value(float(getattr(self.cfg, name)), emit=False)
        self._update_color_swatch()
        self._update_diameter_readout()

    def _on_preset_selected(self, preset_name: str):
        if preset_name in PRESET_ENVIRONMENTS:
            cur_seed = self.cfg.seed
            self.cfg = RealismConfig(**PRESET_ENVIRONMENTS[preset_name])
            self.cfg.species = self.sim.normalize_species(self.cfg.species)
            self.cfg.seed = cur_seed
            self._sync_widgets_from_cfg()
            self._mark_dirty(f"Preset '{preset_name}' loaded — Click '▶ RUN / UPDATE ROOT PREVIEW'")

    def _on_species_changed(self, species_name: str):
        self.cfg.species = self.sim.normalize_species(species_name)
        self._mark_dirty(f"Species set to '{self.cfg.species}' — Click '▶ RUN / UPDATE ROOT PREVIEW'")

    def _on_frame_changed(self, val: int):
        self.frame_idx = int(val)
        self._mark_dirty(f"Camera pose #{self.frame_idx} selected — Click '▶ RUN / UPDATE ROOT PREVIEW'")

    def _on_random_roots(self):
        self.frame_idx = (self.frame_idx + 1) % 45
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(self.frame_idx)
        self.frame_spin.blockSignals(False)
        self._sync_gen_cfg_from_widgets()
        if self.gen_cfg.soil_mode != "current_soil_only":
            cand = self.sim.get_filtered_soil_indices(self.gen_cfg)
            if len(cand) > 1:
                other = [s for s in cand if int(s) != int(self.soil_index)]
                pool = other if other else cand
                self.soil_index = int(pool[int(np.random.randint(0, len(pool)))])
                self.soil_combo.blockSignals(True)
                self.soil_combo.setCurrentIndex(self.soil_index)
                self.soil_combo.blockSignals(False)
        self.render_current_scene()

    def _on_soil_index_changed(self, idx: int):
        if idx >= 0:
            self.soil_index = int(idx)
            self.gen_cfg.current_soil_index = int(idx)
            self._show_raw_soil_preview(force_canvas_update=True)
            if self._last_v4_result is not None:
                self.render_current_scene()
            else:
                self._mark_dirty("Soil patch changed — Click '▶ RUN / UPDATE ROOT PREVIEW'")

    def _on_random_soil(self):
        self._sync_gen_cfg_from_widgets()
        cand = self.sim.get_filtered_soil_indices(self.gen_cfg)
        if not cand and self.sim.soil_paths:
            cand = list(range(len(self.sim.soil_paths)))
        if cand:
            other = [s for s in cand if int(s) != int(self.soil_index)]
            pool = other if other else cand
            idx = int(pool[int(np.random.randint(0, len(pool)))])
            if idx == self.soil_combo.currentIndex():
                self._on_soil_index_changed(idx)
            else:
                self.soil_combo.setCurrentIndex(idx)

    def _on_param_slider_changed(self, field_name: str, val: float):
        setattr(self.cfg, field_name, val)
        if field_name in (
            "root_brightness",
            "root_whiteness",
            "root_brownness",
            "root_warmth",
            "root_red_tint",
            "root_saturation",
            "contrast_target_n",
            "root_age_tint",
            "dark_root_frac",
        ):
            self._update_color_swatch()
        if field_name in (
            "diam_scale",
            "diam_lat_mult",
            "diam_axis_mult",
            "diam_min_mm",
            "px_per_mm",
        ):
            self._update_diameter_readout()
        self._mark_dirty()

    def _apply_diam_ultra_thin(self):
        self._set_diam_params(scale=0.35, lat_mult=0.40, axis_mult=0.55, min_mm=0.03)
        self._mark_dirty("Preset 'Ultra-Thin Hair Roots' applied — Click '▶ RUN / UPDATE ROOT PREVIEW'")

    def _apply_diam_fine(self):
        self._set_diam_params(scale=0.65, lat_mult=0.85, axis_mult=0.90, min_mm=0.05)
        self._mark_dirty("Preset 'Fine Roots' applied — Click '▶ RUN / UPDATE ROOT PREVIEW'")

    def _apply_diam_balanced(self):
        self._set_diam_params(scale=1.00, lat_mult=1.60, axis_mult=1.35, min_mm=0.05)
        self._mark_dirty("Preset 'Balanced Roots' applied — Click '▶ RUN / UPDATE ROOT PREVIEW'")

    def _apply_diam_thick(self):
        self._set_diam_params(scale=1.60, lat_mult=2.20, axis_mult=2.00, min_mm=0.15)
        self._mark_dirty("Preset 'Thick / Fleshy Roots' applied — Click '▶ RUN / UPDATE ROOT PREVIEW'")

    def _set_diam_params(self, **kwargs):
        mapping = {
            "scale": "diam_scale",
            "lat_mult": "diam_lat_mult",
            "axis_mult": "diam_axis_mult",
            "min_mm": "diam_min_mm",
        }
        for k, v in kwargs.items():
            field = mapping.get(k, k)
            if hasattr(self.cfg, field):
                setattr(self.cfg, field, float(v))
            if field in self.sliders:
                self.sliders[field].set_value(float(v), emit=False)
        self._update_diameter_readout()

    def _update_diameter_readout(self):
        if not hasattr(self, "diam_readout_label"):
            return
        px_per_mm = float(getattr(self.cfg, "px_per_mm", 11.81))
        scale = float(getattr(self.cfg, "diam_scale", 1.0))
        lat_m = float(getattr(self.cfg, "diam_lat_mult", 1.60)) * scale
        axis_m = float(getattr(self.cfg, "diam_axis_mult", 1.35)) * scale
        min_mm = float(getattr(self.cfg, "diam_min_mm", 0.05))

        lat_mm = max(min_mm, 0.10 * lat_m)
        axis_mm = max(min_mm, 0.35 * axis_m)
        lat_px = lat_mm * px_per_mm
        axis_px = axis_mm * px_per_mm

        if lat_px <= 1.8:
            badge = "<span style='color: #a3be8c; font-weight: bold;'>[🧵 Ultra-Thin Hair Roots / 1-2 px]</span>"
        elif lat_px <= 3.5:
            badge = "<span style='color: #88c0d0; font-weight: bold;'>[🌾 Fine Roots]</span>"
        elif lat_px <= 7.0:
            badge = "<span style='color: #81a1c1; font-weight: bold;'>[🌿 Balanced Roots]</span>"
        else:
            badge = "<span style='color: #ebcb8b; font-weight: bold;'>[🥕 Coarse / Fleshy Roots]</span>"

        text = (
            f"{badge}<br>"
            f"<b>Laterals:</b> {lat_mm:.2f} mm (~{lat_px:.1f} px) &nbsp;|&nbsp; "
            f"<b>Main Axis:</b> {axis_mm:.2f} mm (~{axis_px:.1f} px)"
        )
        self.diam_readout_label.setText(text)

    def _apply_color_pure_white(self):
        self._set_color_params(brightness=1.40, whiteness=1.0, brownness=0.0, warmth=0.10, red_tint=0.0, saturation=0.20)
        self._mark_dirty("Preset 'Pure White' applied — Click '▶ RUN / UPDATE ROOT PREVIEW'")

    def _apply_color_cream(self):
        self._set_color_params(brightness=1.12, whiteness=0.35, brownness=0.0, warmth=0.45, red_tint=0.0, saturation=0.85)
        self._mark_dirty("Preset 'Ivory / Cream' applied — Click '▶ RUN / UPDATE ROOT PREVIEW'")

    def _apply_color_golden_tan(self):
        self._set_color_params(brightness=1.00, whiteness=0.0, brownness=0.30, warmth=0.70, red_tint=0.10, saturation=1.25)
        self._mark_dirty("Preset 'Golden Tan' applied — Click '▶ RUN / UPDATE ROOT PREVIEW'")

    def _apply_color_rich_brown(self):
        self._set_color_params(brightness=0.88, whiteness=0.0, brownness=0.85, warmth=0.60, red_tint=0.20, saturation=1.10)
        self._mark_dirty("Preset 'Rich Brown' applied — Click '▶ RUN / UPDATE ROOT PREVIEW'")

    def _apply_color_dark_woody(self):
        self._set_color_params(brightness=0.65, whiteness=0.0, brownness=1.00, warmth=0.40, red_tint=0.10, saturation=0.80)
        self._mark_dirty("Preset 'Dark Woody' applied — Click '▶ RUN / UPDATE ROOT PREVIEW'")

    def _set_color_params(self, **kwargs):
        for k, v in kwargs.items():
            field = "root_" + k if not k.startswith("root_") and hasattr(self.cfg, "root_" + k) else k
            if hasattr(self.cfg, field):
                setattr(self.cfg, field, float(v))
            if field in self.sliders:
                self.sliders[field].set_value(float(v), emit=False)
        self._update_color_swatch()

    def _update_color_swatch(self):
        if hasattr(self, "color_swatch"):
            soil_l = 0.35
            soil_s = 0.08
            if hasattr(self, "sim") and self.sim.soil_paths:
                try:
                    bg = self.sim.get_soil_rgb(self.soil_index)
                    luma = bg[..., 0] * 0.299 + bg[..., 1] * 0.587 + bg[..., 2] * 0.114
                    soil_l = float(luma.mean())
                    soil_s = float(luma.std())
                except Exception:
                    pass
            tip, mat = compute_preview_root_colors(self.cfg, soil_l, soil_s)
            self.color_swatch.update_colors(tip, mat)

    def _on_zoom_toggled(self, checked: bool):
        for canvas in (self.panel_v4, self.panel_mask, self.panel_ref):
            canvas.set_zoom_mode(checked)

    def _on_canvas_roi_changed(self, u: float, v: float):
        for canvas in (self.panel_v4, self.panel_mask, self.panel_ref):
            canvas.set_roi_uv(u, v)

    def prefill(self, cfg: Optional[RealismConfig] = None, soil_paths: Optional[List[str]] = None,
                out_dir: str = "", note: str = ""):
        """Start from given settings and soils, e.g. when the root annotator opens the simulator
        with settings fitted to its annotations and the annotated frames (roots inpainted) as soils."""
        if cfg is not None:
            self.cfg = cfg
            self._sync_widgets_from_cfg()
        if soil_paths:
            valid = self.sim.load_user_soil_folder(list(soil_paths), append=False)
            if valid:
                self.custom_soil_folder = os.path.dirname(soil_paths[0])
                self.soil_index = 0
                self._last_v4_result = None
                if len(valid) > 1 and hasattr(self, "soil_mode_combo"):
                    self.soil_mode_combo.blockSignals(True)
                    self.soil_mode_combo.setCurrentIndex(0)
                    self.soil_mode_combo.setItemText(
                        0, f"Randomize Across All Checked Soils in '{Path(self.custom_soil_folder).name}' ({len(valid)} soils)"
                    )
                    self.soil_mode_combo.blockSignals(False)
                self._populate_soil_list()
                self._show_raw_soil_preview(force_canvas_update=True)
        if out_dir:
            self.out_dir_edit.setText(out_dir)
        if note:
            self._mark_dirty(note)

    def _on_load_custom_soil_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Folder Containing Root-Free Real Soil Images"
        )
        if folder:
            valid = self.sim.load_user_soil_folder(folder, append=False)
            if not valid:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Soil Images Found",
                    f"No .png/.jpg/.tif images found in:\n{folder}",
                )
                return
            self.custom_soil_folder = folder
            self.soil_index = 0
            self._last_v4_result = None
            if len(valid) > 1 and hasattr(self, "soil_mode_combo"):
                self.soil_mode_combo.blockSignals(True)
                self.soil_mode_combo.setCurrentIndex(0)  # Randomize Across All Checked Soils Below
                self.soil_mode_combo.setItemText(
                    0, f"Randomize Across All Checked Soils in '{Path(folder).name}' ({len(valid)} soils)"
                )
                self.soil_mode_combo.blockSignals(False)
            self._populate_soil_list()
            self._show_raw_soil_preview(force_canvas_update=True)
            self._mark_dirty(
                f"Loaded {len(valid)} soils from '{Path(folder).name}' (all {len(valid)} active for random sampling) — Click '▶ RUN / UPDATE ROOT PREVIEW'"
            )

    def _on_add_custom_soil_files(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select One or Multiple Root-Free Soil Images to Add (Ctrl/Shift + Click)",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if files:
            # If currently holding the 296 default PRMI bank soils, ask or replace if user wants only their custom files
            append_mode = bool(self.custom_soil_folder) or len(self.sim.soil_paths) < 250
            valid = self.sim.load_user_soil_folder(files, append=append_mode)
            if not valid:
                return
            self.custom_soil_folder = str(Path(files[0]).parent)
            self.soil_index = max(0, len(valid) - len(files))
            self._last_v4_result = None
            if len(valid) > 1 and hasattr(self, "soil_mode_combo"):
                self.soil_mode_combo.blockSignals(True)
                self.soil_mode_combo.setCurrentIndex(0)
                self.soil_mode_combo.setItemText(
                    0, f"Randomize Across All Checked Soils ({len(valid)} loaded soils)"
                )
                self.soil_mode_combo.blockSignals(False)
            self._populate_soil_list()
            self._show_raw_soil_preview(force_canvas_update=True)
            self._mark_dirty(
                f"Added {len(files)} soil file(s) ({len(valid)} total soils active for random sampling) — Click '▶ RUN / UPDATE ROOT PREVIEW'"
            )

    def _on_load_reference_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Real Root Reference Image (for Visual Matching)",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if path:
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is not None:
                self.reference_root_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                self.compare_v3_check.setChecked(False)
                self.title_ref.setText("Target Real Root Reference Image (Match Visually)")
                self.panel_ref.set_image(self.reference_root_rgb)
                self.status_label.setText(f"Loaded reference image: {Path(path).name}")

    def _on_save_config(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Environment Realism Config JSON",
            "realism_config.json",
            "JSON Files (*.json)",
        )
        if path:
            self.cfg.save_json(path)
            self.status_label.setText(f"Saved config to {Path(path).name}")

    def _on_load_config(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Environment Realism Config JSON",
            "",
            "JSON Files (*.json)",
        )
        if path:
            self.cfg = RealismConfig.load_json(path)
            self.cfg.species = self.sim.normalize_species(self.cfg.species)
            self._sync_widgets_from_cfg()
            self._mark_dirty(f"Loaded {Path(path).name} — Click '▶ RUN / UPDATE ROOT PREVIEW'")

    def _on_browse_output_dir(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Generated Dataset"
        )
        if folder:
            self.out_dir_edit.setText(folder)

    # ------------------------------------------------------------------
    # Non-Blocking Background Rendering Pipeline
    # ------------------------------------------------------------------
    def render_current_scene(self):
        if self._preview_worker is not None and self._preview_worker.isRunning():
            return

        self._sync_gen_cfg_from_widgets()
        cand_soils = self.sim.get_filtered_soil_indices(self.gen_cfg)
        if cand_soils and self.soil_index not in cand_soils:
            self.soil_index = int(cand_soils[0])
            self.soil_combo.blockSignals(True)
            self.soil_combo.setCurrentIndex(self.soil_index)
            self.soil_combo.blockSignals(False)

        self.run_preview_btn.setEnabled(False)
        self.run_preview_btn.setText("⏳  RENDERING ROOTS...")
        soil_name = (
            Path(self.sim.soil_paths[self.soil_index % len(self.sim.soil_paths)]).name
            if self.sim.soil_paths
            else "custom"
        )
        self.status_label.setText(
            f"Rendering preview on '{soil_name}' (Dataset pool: {len(cand_soils)} soils)..."
        )
        self.status_label.setStyleSheet("color: #88c0d0; font-weight: bold; font-size: 13px;")

        self._preview_worker = PreviewRenderWorker(
            sim=self.sim,
            cfg=self.cfg.copy(),
            soil_index=self.soil_index,
            stand_idx=self.stand_idx,
            frame_idx=self.frame_idx,
            also_render_v3=bool(self.compare_v3_check.isChecked()),
        )
        self._preview_worker.finished_ok.connect(self._on_preview_finished)
        self._preview_worker.failed.connect(self._on_preview_failed)
        self._preview_worker.start()

    def _on_preview_finished(self, v4_res: dict, v3_res: object, dt_ms: float):
        self.run_preview_btn.setEnabled(True)
        self.run_preview_btn.setText("▶  RUN / UPDATE ROOT PREVIEW")
        self._last_v4_result = v4_res
        self._last_v3_result = v3_res
        self._update_canvases(dt_ms)

    def _on_preview_failed(self, err_msg: str):
        self.run_preview_btn.setEnabled(True)
        self.run_preview_btn.setText("▶  RUN / UPDATE ROOT PREVIEW")
        self.status_label.setText(f"Render error: {err_msg}")
        self.status_label.setStyleSheet("color: #bf616a; font-weight: bold;")
        QtWidgets.QMessageBox.critical(self, "Render Error", err_msg)

    def _update_canvases(self, render_ms: float = 0.0):
        if self._last_v4_result is None:
            return

        v4_rgb = self._last_v4_result["rgb"]
        v4_mask = self._last_v4_result["mask"]
        v4_diam = self._last_v4_result["diameter_mm"]

        self.panel_v4.set_image(v4_rgb)
        overlay_rgb = _make_mask_overlay(v4_rgb, v4_mask, v4_diam)
        self.panel_mask.set_image(overlay_rgb)

        if self.compare_v3_check.isChecked() and self._last_v3_result is not None:
            self.title_ref.setText("Legacy v3 Simulator (Capsule Seams / Ghost Film)")
            self.panel_ref.set_image(self._last_v3_result["rgb"])
        elif self.reference_root_rgb is not None:
            self.title_ref.setText("Target Real Root Reference Image (Match Visually)")
            self.panel_ref.set_image(self.reference_root_rgb)
        else:
            self.title_ref.setText("Selected Real Soil Patch (Root-Free Background)")
            self.panel_ref.set_image(self._last_v4_result["background"])

        root_px = int((v4_mask > 0).sum())
        cov_pct = 100.0 * root_px / max(v4_mask.size, 1)
        len2d_mm = float(self._last_v4_result.get("len2d_mm", 0.0))
        mean_d_mm = float(v4_diam[v4_mask > 0].mean()) if root_px > 0 else 0.0
        soil_name = (
            Path(self.sim.soil_paths[self.soil_index % len(self.sim.soil_paths)]).name
            if self.sim.soil_paths
            else "custom"
        )
        cand_soils = self.sim.get_filtered_soil_indices(self.gen_cfg)
        out_dir = self.out_dir_edit.text().strip() or "datasets/custom_env_v4"
        self.info_bar.setText(
            f"Preview Soil: {soil_name}  |  Dataset Soil Pool: {len(cand_soils)}/{len(self.sim.soil_paths)} soils active  |  "
            f"Species: {self.cfg.species}  |  Length: {len2d_mm:.1f} mm  |  Mean Diam: {mean_d_mm:.2f} mm\n"
            f"Dataset Rules -> Soil Mode: {self.gen_cfg.soil_mode} ({len(cand_soils)} soils)  |  "
            f"Species Mode: {self.gen_cfg.species_mode}  |  "
            f"Param Mode: {self.gen_cfg.param_mode}  |  Output: '{out_dir}'"
        )
        if render_ms > 0:
            self.status_label.setText(
                f"✓ Rendered '{soil_name}' in {render_ms / 1000.0:.2f} s  (Dataset will randomize across {len(cand_soils)} soils)"
            )
            self.status_label.setStyleSheet("color: #a3be8c; font-weight: bold; font-size: 13px;")

    # ------------------------------------------------------------------
    # Dataset Generation
    # ------------------------------------------------------------------
    def _on_export_dataset(self):
        out_dir = self.out_dir_edit.text().strip()
        if not out_dir:
            QtWidgets.QMessageBox.warning(self, "Output Folder", "Please specify an output folder.")
            return

        self._sync_gen_cfg_from_widgets()
        self.gen_cfg.n_systems = int(self.n_sys_spin.value())
        self.gen_cfg.frames_per_system = int(self.n_frames_spin.value())

        cand_soils = self.sim.get_filtered_soil_indices(self.gen_cfg)
        total = self.gen_cfg.n_systems * self.gen_cfg.frames_per_system
        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.status_label.setText(
            f"Generating {total} images randomized across {len(cand_soils)} soils ({self.gen_cfg.param_mode})..."
        )

        self._export_worker = DatasetExportWorker(
            sim=self.sim,
            cfg=self.cfg.copy(),
            gen_cfg=DatasetGenConfig(**self.gen_cfg.__dict__),
            output_dir=out_dir,
        )
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.finished_ok.connect(self._on_export_finished)
        self._export_worker.failed.connect(self._on_export_failed)
        self._export_worker.start()

    def _on_export_progress(self, done: int, total: int, fid: str):
        self.progress_bar.setValue(done)
        self.status_label.setText(f"Exporting {fid} ({done}/{total})...")
        self.info_bar.setText(f"Dataset Generation in Progress ({done}/{total}): {fid}")

    def _on_export_finished(self, out_path: str):
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Dataset exported to {out_path}")
        run_tag_used = self.gen_cfg.run_version or "auto"
        QtWidgets.QMessageBox.information(
            self,
            "Environment Dataset Ready",
            f"Successfully generated dataset in:\n"
            f"{out_path}\n\n"
            f"Sampling & Persistence Settings:\n"
            f"  • Run Version Tag: {run_tag_used}\n"
            f"  • Append Multi-Run: {'Enabled (previous runs kept)' if self.gen_cfg.append_runs else 'Disabled'}\n"
            f"  • Soil Mode: {self.gen_cfg.soil_mode}\n"
            f"  • Species Mode: {self.gen_cfg.species_mode}\n"
            f"  • Parameter Mode: {self.gen_cfg.param_mode}\n\n"
            f"Files created / appended:\n"
            f"  • {out_path}/rgb/ (Realistic composite images)\n"
            f"  • {out_path}/mask/ (Binary U-Net ground-truth masks)\n"
            f"  • {out_path}/diam/ (Root diameter maps in µm)\n"
            f"  • {out_path}/alpha/ (Continuous opacity maps)\n"
            f"  • {out_path}/part_{run_tag_used}_*.zip (Zipped stand chunks for fast Colab loading)\n"
            f"  • {out_path}/traits.csv (Cumulative per-image soil, species & parameters)\n"
            f"  • {out_path}/realism_config_{run_tag_used}.json (Run-specific config backup)",
        )

    def _on_export_failed(self, err_msg: str):
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Export failed")
        QtWidgets.QMessageBox.critical(self, "Dataset Export Error", err_msg)


# ==============================================================================
# Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Launch the Minirhizotron Root Realism Studio (PyQt GUI)."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Initialize window, render one scene, and exit (for automated testing).",
    )
    args = parser.parse_args()

    if args.smoke_test and "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    app = QtWidgets.QApplication(sys.argv)
    win = RootRealismStudioWindow()
    win.show()

    if args.smoke_test:
        QtWidgets.QApplication.processEvents()
        print("GUI smoke test succeeded! Active Qt API:", QT_API)
        win.close()
        return

    exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
    sys.exit(exec_fn())


if __name__ == "__main__":
    main()
