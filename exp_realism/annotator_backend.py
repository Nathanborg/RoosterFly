"""Qt-free model backend for the root annotator: load a run, predict, cache.

The annotator (``gui_root_annotator``) predicts in a background queue and
keeps every probability map it computes on disk, so an image predicted once --
while you were checking the previous one, by "Pre-predict Folder", or by this
module run headless -- opens instantly afterwards.

Cache layout (16-bit PNG, prob x 65535; exact enough for thresholds like 0.01):

    <image folder>/_annotations/_prob_cache/<run>[_tta]/<stem>.png
    or, with --cache-dir:  <cache dir>/<run>[_tta]/<image folder name>/<stem>.png

Headless pre-prediction (e.g. on a GPU, without PyQt):

    python -m exp_realism.annotator_backend --dir "Minirhizotron images/Panama/All GIG"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(folder: str) -> List[str]:
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS) and not f.startswith(".")
    ]
    files.sort()
    return files


def cache_tag(run_name: str, tta: bool) -> str:
    return f"{run_name}_tta" if tta else run_name


def prob_cache_path(img_path: str, tag: str, cache_dir: str = "") -> str:
    folder, fname = os.path.split(os.path.abspath(img_path))
    stem = os.path.splitext(fname)[0]
    if cache_dir:
        return os.path.join(cache_dir, tag, os.path.basename(folder), stem + ".png")
    return os.path.join(folder, "_annotations", "_prob_cache", tag, stem + ".png")


def load_cached_prob(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    u16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if u16 is None or u16.dtype != np.uint16 or u16.ndim != 2:
        return None
    return u16.astype(np.float32) / 65535.0


def save_cached_prob(path: str, prob: np.ndarray) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        u16 = np.round(np.clip(prob, 0.0, 1.0) * 65535.0).astype(np.uint16)
        tmp = path[:-4] + ".tmp.png"                    # never leave a half-written cache file
        if not cv2.imwrite(tmp, u16):
            return False
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def load_model_bundle(project_root: str, run_name: str, device: str) -> Dict[str, Any]:
    """The trained U-Net of ``segmentation/realsoil/<run_name>`` with its preprocessing."""
    import torch
    from exp_realism.eval_grid_counted import (
        load_or_compute_prep_stats,
        resolve_checkpoint,
        to_input_factory,
    )

    run_dir, ckpt_path, train_cfg = resolve_checkpoint(project_root, run_name, "seg_best.pt")
    cdf_ref, idx_mean, idx_std = load_or_compute_prep_stats(run_dir, project_root, train_cfg)
    to_input = to_input_factory(
        index_mode=train_cfg.get("INDEX_MODE", "extra-channel"),
        idx_mean=idx_mean,
        idx_std=idx_std,
    )

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    in_ch = 4 if train_cfg.get("INDEX_MODE", "extra-channel") == "extra-channel" else 3
    out_classes = 2 if float(train_cfg.get("AUX_DIST_WEIGHT", 0.25)) > 0 else 1

    # Fast direct builder
    import segmentation_models_pytorch as smp
    arch_clean = train_cfg.get("ARCH", "Unet").strip()
    enc_clean = train_cfg.get("ENCODER", "mit_b3").strip()
    if enc_clean.startswith("mit_") and arch_clean in ("UnetPlusPlus", "MAnet"):
        arch_clean = "Unet"
    cls = getattr(smp, arch_clean, smp.Unet)

    # Direct layout (b) for mit_b3
    if enc_clean.startswith("mit_") and in_ch != 3:
        model = cls(enc_clean, encoder_weights=None, in_channels=3, classes=out_classes)
        if hasattr(model.encoder, "patch_embed1") and hasattr(model.encoder.patch_embed1, "proj"):
            old = model.encoder.patch_embed1.proj
            model.encoder.patch_embed1.proj = torch.nn.Conv2d(
                in_ch, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=old.bias is not None
            )
    else:
        model = cls(enc_clean, encoder_weights=None, in_channels=in_ch, classes=out_classes)

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    thresh, border = None, False
    inf_path = os.path.join(run_dir, "inference_settings.json")
    if os.path.exists(inf_path):
        try:
            with open(inf_path, "r") as fh:
                inf = json.load(fh)
            thresh = float(inf.get("THRESH", 0.45))
            border = bool(inf.get("BORDER_MASK", False))
        except Exception:
            pass

    return {
        "run_name": run_name,
        "model": model,
        "train_cfg": train_cfg,
        "cdf_ref": cdf_ref,
        "to_input": to_input,
        "device": device,
        "thresh": thresh,
        "border_mask": border,
        "n_out": out_classes,                       # 2: root probability + centre-line head
    }


def predict_prob(bundle: Dict[str, Any], rgb: np.ndarray, tta: bool = False) -> np.ndarray:
    """Full-resolution root probability map (float32 HxW) of a uint8 RGB image."""
    return predict_maps(bundle, rgb, tta)[0]


def predict_maps(bundle: Dict[str, Any], rgb: np.ndarray, tta: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """(root probability, centre-line map or None) from one forward pass.

    The centre-line map is the model's second head (trained on the root's distance transform over its
    local maximum: 1 in the middle of every root, falling to its edges) -- it tells touching roots apart."""
    from exp_realism.eval_grid_counted import hist_match
    from mrmodel.tiling import predict_image

    rgb_matched = hist_match(rgb, bundle["cdf_ref"])
    H, W = rgb_matched.shape[:2]
    tile_sz = 768 if max(H, W) > 768 else max(H, W)
    maps = predict_image(
        bundle["model"],
        rgb_matched,
        bundle["to_input"],
        scale=1.0,
        device=bundle["device"],
        tta=tta,
        tile=tile_sz,
        overlap=64,
        batch=2,
        all_channels=True,
    )
    return maps[0], (maps[1] if len(maps) > 1 else None)


def centre_cache_path(img_path: str, tag: str, cache_dir: str = "") -> str:
    """Where the centre-line map of an image is cached (next to its probability map)."""
    return prob_cache_path(img_path, tag + "_cl", cache_dir)


def border_bars(rgb: np.ndarray, dark: float = 0.55, flat: float = 0.15, margin: int = 6, edge: int = 3) -> np.ndarray:
    """Camera border bars: ``eval_grid_counted.border_region`` with dark bars, too, required to be flat.

    The training-time detector counts any run of dark rows/columns at an edge as a bar, so a
    frame whose top is simply darker soil (e.g. RootDetector's CD_T017_L001 scan: rows at 0.47x the
    median but textured, std 0.4-0.5x) lost its top 13 %.  Real bars are near-uniform: a row or
    column counts only when it is dark *and* its spread is below ``flat`` x the median.
    """
    y_ = rgb.astype(np.float32) @ np.float32([0.299, 0.587, 0.114])
    out = np.zeros(y_.shape, bool)
    med = float(np.median(y_)) + 1e-3
    for sl, outer, inner in (((slice(0, edge), slice(None)), y_[:edge], y_[edge:edge + 10]),
                             ((slice(-edge, None), slice(None)), y_[-edge:], y_[-edge - 10:-edge]),
                             ((slice(None), slice(0, edge)), y_[:, :edge], y_[:, edge:edge + 10]),
                             ((slice(None), slice(-edge, None)), y_[:, -edge:], y_[:, -edge - 10:-edge])):
        if abs(float(outer.mean()) - float(inner.mean())) > 0.08 * med:      # a seam right at the edge
            out[sl] = True
    for axis in (0, 1):
        prof = y_.mean(axis=1 - axis)
        spread = y_.std(axis=1 - axis)
        med_p = float(np.median(prof))
        bad = ((prof < dark * med_p) | (prof > 1.6 * med_p)) & (spread < flat * med_p)
        n = len(prof)
        a = int(np.argmin(bad)) if not bad.all() else n
        b = int(np.argmin(bad[::-1])) if not bad.all() else n
        a, b = (a + margin if a > 0 else 0), (b + margin if b > 0 else 0)
        if axis == 0:
            out[:a] = True
            out[n - b:] = True
        else:
            out[:, :a] = True
            out[:, n - b:] = True
    return out


def mask_border(prob: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Zero the camera's border bars (:func:`border_bars`) when the run's ``BORDER_MASK`` is set."""
    b = border_bars(rgb)
    if not b.any():
        return prob
    out = prob.copy()
    out[b] = 0.0
    return out


def read_rgb(path: str) -> Optional[np.ndarray]:
    bgr = cv2.imread(path)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def precompute_folder(project_root: str, run_name: str, folder: str, cache_dir: str = "",
                      tta: bool = False, device: str = "") -> None:
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_model_bundle(project_root, run_name, device)
    tag = cache_tag(run_name, tta)
    files = list_images(folder)
    todo = [f for f in files if not os.path.exists(prob_cache_path(f, tag, cache_dir))]
    print(f"[precompute] {run_name} on {device}: {len(files)} images, {len(files) - len(todo)} cached, "
          f"{len(todo)} to predict", flush=True)
    t0 = time.time()
    for i, f in enumerate(todo, 1):
        rgb = read_rgb(f)
        if rgb is None:
            print(f"[precompute] unreadable: {f}", flush=True)
            continue
        prob, centre = predict_maps(bundle, rgb, tta)
        save_cached_prob(prob_cache_path(f, tag, cache_dir), prob)
        if centre is not None:
            save_cached_prob(centre_cache_path(f, tag, cache_dir), centre)
        if i % 25 == 0 or i == len(todo):
            rate = (time.time() - t0) / i
            eta = rate * (len(todo) - i)
            print(f"[precompute] {i}/{len(todo)}  {rate:.2f} s/img  ETA {eta / 3600:.1f} h", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Pre-predict a folder into the annotator's probability cache")
    ap.add_argument("--root", type=str, default=str(_REPO_ROOT), help="Project root folder")
    ap.add_argument("--model", type=str, default="v8_full_real_and_syn_mitb3")
    ap.add_argument("--dir", type=str, required=True, help="Image folder")
    ap.add_argument("--cache-dir", type=str, default="", help="Cache root (default: <dir>/_annotations/_prob_cache)")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--device", type=str, default="")
    a = ap.parse_args()
    precompute_folder(a.root, a.model, a.dir, a.cache_dir, a.tta, a.device)


if __name__ == "__main__":
    main()
