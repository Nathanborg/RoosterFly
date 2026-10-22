"""Camera and tube optical artifacts.

These are cheap to randomise and are often what a real-vs-fake discriminator is
actually latching onto, so omitting them makes the sim2real gap far wider than
the root geometry alone would suggest.

Several are deliberately NOT the textbook version:

* Illumination is **asymmetric with a fixed hotspot**, not a cos^4 radial
  vignette.  Real rigs use two COB LED strips or incandescent lamps, and at least
  one commercial head has a persistent over-exposed corner regardless of exposure.
* Specular reflection off the acrylic is a **hue-shifted, desaturated, structured
  streak**, not an additive white Gaussian blob.
* Condensation shows up as droplet **rims** -- bright arcs -- which are a known
  false-positive source.  Filled discs do not look like water and do not fool
  anything.
* Width-dependent contrast loss is produced by convolving with a **measured PSF**
  rather than an ad-hoc grey-level pull.  Contrast-vs-width *is* the system MTF,
  so doing it with the PSF gives correct edge profiles and correct behaviour at
  crossings for free, and keeps the model self-consistent.  The two published
  calibration points come from printed black lines on white -- the opposite
  contrast polarity against a different background luminance -- so extrapolating
  them to roots on soil has no support.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class OpticsStyle:
    vignette_strength: float = 0.35
    hotspot_xy: Tuple[float, float] = (0.2, 0.15)
    hotspot_strength: float = 0.18
    hotspot_sigma: float = 0.35
    specular_prob: float = 0.35
    specular_strength: float = 0.25
    condensation_prob: float = 0.30
    n_droplets: int = 40
    scratch_prob: float = 0.45
    n_scratches: int = 6
    psf_sigma_px: float = 0.9
    chroma_px: float = 0.0
    read_noise: float = 0.006
    shot_noise: float = 0.010
    jpeg_quality: Optional[int] = 92

    @classmethod
    def sample(cls, rng, px_per_mm: float = 35.0) -> "OpticsStyle":
        return cls(
            vignette_strength=float(rng.uniform(0.10, 0.55)),
            hotspot_xy=(float(rng.uniform(0.05, 0.95)),
                        float(rng.uniform(0.05, 0.95))),
            hotspot_strength=float(rng.uniform(0.0, 0.30)),
            hotspot_sigma=float(rng.uniform(0.15, 0.55)),
            specular_prob=0.35,
            specular_strength=float(rng.uniform(0.10, 0.45)),
            condensation_prob=0.30,
            n_droplets=int(rng.integers(0, 120)),
            scratch_prob=0.45,
            n_scratches=int(rng.integers(0, 14)),
            # PSF scales with resolution: a fixed px sigma would be a different
            # physical blur on a 35 px/mm Bartz and a 212 px/mm AMR
            psf_sigma_px=float(rng.uniform(0.5, 1.8)) * max(1.0, px_per_mm / 35.0) ** 0.5,
            chroma_px=float(rng.uniform(0.0, 0.8)),
            read_noise=float(rng.uniform(0.002, 0.014)),
            shot_noise=float(rng.uniform(0.004, 0.020)),
            jpeg_quality=int(rng.integers(70, 97)),
        )

    def without_artefacts(self, psf_max_px: Optional[float] = None) -> "OpticsStyle":
        """This style with every tube/lens artefact switched off.

        Keeps only what a contact scanner (CI-600) has: the PSF, optionally
        capped at ``psf_max_px``, and shot/read noise.  Vignette, hotspot,
        specular glare, condensation rims, scratches and the chromatic shift go
        to zero and JPEG is dropped (the CI-600 writes TIFF).  Draws nothing
        from any rng, so ``OpticsStyle.sample(rng).without_artefacts()``
        consumes exactly the draws ``render_frame`` would have.
        """
        psf = self.psf_sigma_px if psf_max_px is None else min(self.psf_sigma_px,
                                                              float(psf_max_px))
        return replace(self, vignette_strength=0.0, hotspot_strength=0.0,
                       specular_prob=0.0, specular_strength=0.0,
                       condensation_prob=0.0, n_droplets=0,
                       scratch_prob=0.0, n_scratches=0,
                       chroma_px=0.0, jpeg_quality=None, psf_sigma_px=psf)


# --------------------------------------------------------------------------
# illumination
# --------------------------------------------------------------------------
def illumination_field(shape, style: OpticsStyle) -> np.ndarray:
    """Asymmetric illumination: radial falloff plus a fixed off-centre hotspot."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.hypot((yy - cy) / max(cy, 1), (xx - cx) / max(cx, 1))
    field = 1.0 - style.vignette_strength * np.clip(r, 0, 1.6) ** 2

    hy = style.hotspot_xy[1] * (h - 1)
    hx = style.hotspot_xy[0] * (w - 1)
    s = max(style.hotspot_sigma, 1e-3) * max(h, w)
    field = field + style.hotspot_strength * np.exp(
        -(((yy - hy) ** 2 + (xx - hx) ** 2) / (2.0 * s * s)))
    return np.clip(field, 0.05, 2.0).astype(np.float32)


def apply_specular(img, rng, style: OpticsStyle):
    """A hue-shifted, desaturated arc following the light source."""
    if rng.random() > style.specular_prob:
        return img
    h, w = img.shape[:2]
    layer = np.zeros((h, w), np.float32)
    n = int(rng.integers(1, 3))
    for _ in range(n):
        cx = float(rng.uniform(0, w))
        cy = float(rng.uniform(0, h))
        ax = float(rng.uniform(0.15, 0.7)) * w
        ay = float(rng.uniform(0.02, 0.12)) * h
        ang = float(rng.uniform(0, math.pi))
        tmp = np.zeros((h, w), np.float32)
        cv2.ellipse(tmp, (int(cx), int(cy)), (int(ax), int(ay)),
                    math.degrees(ang), 0, 360, 1.0, -1)
        tmp = cv2.GaussianBlur(tmp, (0, 0), max(ax, ay) * 0.35)
        layer = np.maximum(layer, tmp)
    m = layer.max()
    if m > 1e-6:
        layer /= m
    a = style.specular_strength * layer
    # push toward a desaturated, slightly cool highlight rather than pure white
    tint = np.array([0.93, 0.96, 1.0], np.float32)
    return np.clip(img * (1 - a[..., None]) + a[..., None] * tint, 0, 1)


def apply_condensation(img, rng, style: OpticsStyle):
    """Droplet RIMS -- bright arcs with darker centres."""
    if rng.random() > style.condensation_prob or style.n_droplets <= 0:
        return img
    h, w = img.shape[:2]
    rim = np.zeros((h, w), np.float32)
    core = np.zeros((h, w), np.float32)
    for _ in range(style.n_droplets):
        cx, cy = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        r = int(max(2, rng.gamma(2.2, 3.0)))
        cv2.circle(rim, (cx, cy), r, 1.0, max(1, r // 4))
        cv2.circle(core, (cx, cy), max(1, r - r // 3), 1.0, -1)
    rim = cv2.GaussianBlur(rim, (0, 0), 0.8)
    core = cv2.GaussianBlur(core, (0, 0), 1.5)
    out = img * (1.0 - 0.18 * core[..., None])
    out = out + 0.30 * rim[..., None]
    # a light fog over the whole frame
    if rng.random() < 0.4:
        out = out * 0.93 + 0.07 * float(rng.uniform(0.5, 0.9))
    return np.clip(out, 0, 1)


def apply_scratches(img, rng, style: OpticsStyle):
    """Persistent wall defects.  On a real tube these accumulate with age."""
    if rng.random() > style.scratch_prob or style.n_scratches <= 0:
        return img
    h, w = img.shape[:2]
    layer = np.zeros((h, w), np.float32)
    for _ in range(style.n_scratches):
        x0, y0 = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        ang = float(rng.uniform(0, math.pi))
        L = float(rng.uniform(0.1, 0.9)) * max(h, w)
        x1 = int(x0 + L * math.cos(ang))
        y1 = int(y0 + L * math.sin(ang))
        cv2.line(layer, (x0, y0), (x1, y1), float(rng.uniform(0.3, 1.0)),
                 int(rng.integers(1, 3)))
    layer = cv2.GaussianBlur(layer, (0, 0), 0.7)
    sign = 1.0 if rng.random() < 0.6 else -1.0
    return np.clip(img + sign * 0.10 * layer[..., None], 0, 1)


# --------------------------------------------------------------------------
# sensor chain
# --------------------------------------------------------------------------
def apply_psf(img, sigma_px: float):
    """System point spread function.

    Applying this to the COMPOSITED frame is what produces width-dependent
    contrast loss: a 2 px root loses far more of its contrast to the PSF than a
    20 px root does, automatically and self-consistently, with correct edge
    profiles and correct behaviour where roots cross.
    """
    if sigma_px <= 1e-3:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma_px)


def apply_chromatic_aberration(img, px: float):
    if px <= 1e-3:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    for c, s in ((0, px), (2, -px)):
        M = np.float32([[1 + s / max(w, 1), 0, -s / 2], [0, 1 + s / max(h, 1), -s / 2]])
        out[:, :, c] = cv2.warpAffine(img[:, :, c], M, (w, h),
                                      borderMode=cv2.BORDER_REFLECT)
    return out


def apply_noise(img, rng, style: OpticsStyle):
    """Shot noise (signal-dependent) plus read noise (additive)."""
    out = img.astype(np.float32)
    if style.shot_noise > 0:
        out = out + rng.normal(0, 1, out.shape).astype(np.float32) * (
            style.shot_noise * np.sqrt(np.clip(out, 0, 1)))
    if style.read_noise > 0:
        out = out + rng.normal(0, style.read_noise, out.shape).astype(np.float32)
    return np.clip(out, 0, 1)


def apply_jpeg(img, quality: Optional[int]):
    """Compress LAST, and only if the simulated system writes JPEG at capture."""
    if quality is None:
        return img
    bgr = np.clip(img[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return (dec[:, :, ::-1].astype(np.float32) / 255.0)


def apply_optics(img, rng, style: OpticsStyle, illum: Optional[np.ndarray] = None):
    """Full artifact chain, in physical order.

    Order matters: the PSF acts on the scene, artifacts on the tube wall sit in
    front of it, the sensor adds noise, and JPEG is applied to what the sensor
    produced -- never before.
    """
    out = apply_psf(img, style.psf_sigma_px)
    out = apply_chromatic_aberration(out, style.chroma_px)
    if illum is None:
        illum = illumination_field(out.shape[:2], style)
    out = np.clip(out * illum[..., None], 0, 1)
    out = apply_scratches(out, rng, style)
    out = apply_condensation(out, rng, style)
    out = apply_specular(out, rng, style)
    out = apply_noise(out, rng, style)
    out = apply_jpeg(out, style.jpeg_quality)
    return out.astype(np.float32)
