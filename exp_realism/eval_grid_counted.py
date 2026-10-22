from skimage import exposure
import glob
#!/usr/bin/env python
"""Independent evaluation on Grid_counted_data (Zenodo 3527713 rhizotron dataset):
1. Stages train.zip, val.zip, test.zip into a single fast local directory on Colab/local disk.
2. Automatically loads the most recent trained v4 model checkpoint (or specified run).
3. Evaluates full-resolution photos using sliding-window Hann-tiled prediction.
4. Computes Observed root length (mm) vs Predicted root length (mm), Dice, F1@3px.
5. Produces a summary CSV, publication-quality scatter plot (Predicted vs Observed with R^2 and r),
   and overlay inspection sheets.

Usage:
    python exp_realism/eval_grid_counted.py
    python exp_realism/eval_grid_counted.py --run-name v4_realism_mitb3 --split test
"""
import argparse
import glob
import json
import math
import os
import re
import shutil
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import cv2
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn.functional as F

try:
    from skimage.morphology import skeletonize
except ImportError:
    from scipy import ndimage as _ndi

    def skeletonize(mask):
        m = np.asarray(mask, dtype=bool)
        if not m.any():
            return np.zeros_like(m, dtype=bool)
        dist = _ndi.distance_transform_edt(m)
        mx = _ndi.maximum_filter(dist, size=3)
        return m & (dist == mx) & (dist > 0)


from mrstyle import realdata as RD
from mrmodel import tiling as TL

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _raw_index(x01):
    return x01[..., 1] - x01[..., 2] - 2.0 * x01[..., 0]  # G - B - 2R


def to_input_factory(index_mode: str = "extra-channel", idx_mean: float = -0.05, idx_std: float = 0.25):
    def _to_input(x01):
        rgb = (x01 - MEAN) / STD
        if index_mode == "off":
            ch = rgb
        else:
            idx = ((_raw_index(x01) - idx_mean) / idx_std)[..., None]
            ch = np.concatenate([rgb, idx], 2) if index_mode == "extra-channel" else idx
        return np.ascontiguousarray(ch.transpose(2, 0, 1)).astype(np.float32)
    return _to_input


class MultiChannelAdapterWrapper(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, in_ch: int):
        super().__init__()
        self.in_ch = in_ch
        self.base_model = base_model
        if in_ch != 3:
            self.extra_proj = torch.nn.Conv2d(in_ch, 3, kernel_size=1, bias=True)
            with torch.no_grad():
                self.extra_proj.weight.zero_()
                self.extra_proj.bias.zero_()
                for c in range(min(3, in_ch)):
                    self.extra_proj.weight[c, c, 0, 0] = 1.0
        else:
            self.extra_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.extra_proj is not None:
            x = self.extra_proj(x)
        return self.base_model(x)


def build_segmentation_model(arch: str, encoder: str, in_ch: int, out_classes: int,
                             state_dict: dict = None) -> torch.nn.Module:
    """Rebuild exactly the variant train_v4.py created. train_v4 may produce three layouts:
    (a) native in_channels=in_ch, (b) 3-ch model with patch_embed1.proj widened to in_ch,
    (c) 3-ch model wrapped in MultiChannelAdapterWrapper ('base_model.*' + 'extra_proj.*').
    The checkpoint keys tell us which one to build."""
    import segmentation_models_pytorch as smp
    arch_clean = arch.strip()
    enc_clean = encoder.strip()
    if enc_clean.startswith("mit_") and arch_clean in ("UnetPlusPlus", "MAnet"):
        arch_clean = "Unet"
    cls = getattr(smp, arch_clean, smp.Unet)
    kw = {}
    if enc_clean.startswith("tu-convnext"):
        kw = dict(encoder_depth=4, decoder_channels=(256, 128, 64, 32))

    keys = list(state_dict.keys()) if state_dict else []
    if any(k.startswith("base_model.") for k in keys):          # layout (c)
        m3 = cls(enc_clean, encoder_weights=None, in_channels=3, classes=out_classes, **kw)
        return MultiChannelAdapterWrapper(m3, in_ch)

    try:                                                         # layout (a)
        return cls(enc_clean, encoder_weights=None, in_channels=in_ch, classes=out_classes, **kw)
    except Exception:
        pass

    m3 = cls(enc_clean, encoder_weights=None, in_channels=3, classes=out_classes, **kw)
    if in_ch == 3:
        return m3
    if hasattr(m3.encoder, "patch_embed1") and hasattr(m3.encoder.patch_embed1, "proj"):  # layout (b)
        old = m3.encoder.patch_embed1.proj
        m3.encoder.patch_embed1.proj = torch.nn.Conv2d(
            in_ch, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
            padding=old.padding, bias=old.bias is not None)
        return m3
    return MultiChannelAdapterWrapper(m3, in_ch)



def _str2bool(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate root segmentation model on Grid_counted_data")
    ap.add_argument("--root", default=REPO, help="Project repository root folder")
    ap.add_argument(
        "--grid-dir",
        default="",
        help="Path to Grid_counted_data folder with train.zip, val.zip, test.zip",
    )
    ap.add_argument(
        "--stage-dir",
        default="",
        help="Local staging directory for unzipped photos and annotations",
    )
    ap.add_argument(
        "--run-name",
        default="latest",
        help="Name of the training run under segmentation/realsoil/ or 'latest'",
    )
    ap.add_argument(
        "--checkpoint",
        default="seg_best.pt",
        choices=["seg_best.pt", "seg_last.pt"],
        help="Which checkpoint to evaluate",
    )
    ap.add_argument(
        "--split",
        default="all",
        choices=["all", "test", "val", "train", "test+val"],
        help="Which split(s) from Grid_counted_data to evaluate",
    )
    ap.add_argument(
        "--px-per-mm",
        type=float,
        default=12.0,
        help="Resolution of Grid_counted_data / chicory images (default: 12.0 px/mm)",
    )
    ap.add_argument(
        "--ref-px-per-mm",
        type=float,
        default=11.81,
        help="Reference training resolution (default: 11.81 px/mm)",
    )
    ap.add_argument(
        "--scale-to-ref",
        type=_str2bool,
        default=True,
        help="Scale input images by (ref_px_per_mm / px_per_mm) during inference (True/False)",
    )
    ap.add_argument(
        "--thresh",
        type=float,
        default=-1.0,
        help="Binarization threshold (-1 = the run's validation-selected threshold)",
    )
    ap.add_argument(
        "--tta",
        type=_str2bool,
        default=True,
        help="Test-time augmentation with 4 flips (True/False)",
    )
    ap.add_argument(
        "--n-overlays",
        type=int,
        default=6,
        help="Number of worst/best visual overlays to export",
    )
    return ap.parse_args()


# ==============================================================================
# 1. Staging: Extract train/val/test zips into a single unified directory
# ==============================================================================
def stage_grid_counted(grid_dir: str, stage_dir: str) -> pd.DataFrame:
    """Stage all photo & annotation pairs from train.zip, val.zip, test.zip
    into a single local directory with photos/ and annotations/."""
    os.makedirs(stage_dir, exist_ok=True)
    photos_dir = os.path.join(stage_dir, "photos")
    annots_dir = os.path.join(stage_dir, "annotations")
    os.makedirs(photos_dir, exist_ok=True)
    os.makedirs(annots_dir, exist_ok=True)

    manifest_csv = os.path.join(stage_dir, "manifest.csv")
    if os.path.exists(manifest_csv):
        df_m = pd.read_csv(manifest_csv)
        if len(df_m) > 0 and os.path.exists(df_m.iloc[0]["photo_path"]):
            print(f"Using already staged Grid_counted_data in {stage_dir} ({len(df_m)} pairs).")
            return df_m

    print(f"Staging Grid_counted_data from {grid_dir} -> {stage_dir}...")
    records = []
    splits = ["train", "val", "test"]

    for sp in splits:
        zp = os.path.join(grid_dir, f"{sp}.zip")
        if not os.path.exists(zp):
            # Check if there is an unzipped folder
            unzipped_dir = os.path.join(grid_dir, sp)
            if os.path.isdir(unzipped_dir):
                print(f"  reading unzipped {sp} folder...")
                cand_photos = glob.glob(os.path.join(unzipped_dir, "**", "*_photo.png"), recursive=True)
                for pp in cand_photos:
                    bn = os.path.basename(pp)
                    stem = bn.replace("_photo.png", "").replace(".png", "")
                    ap_cand = pp.replace("photos", "annotations").replace("_photo.png", "_annotation.png")
                    if not os.path.exists(ap_cand):
                        ap_cand = os.path.join(os.path.dirname(os.path.dirname(pp)), "annotations", f"{stem}_annotation.png")
                    if os.path.exists(ap_cand):
                        dst_p = os.path.join(photos_dir, f"{stem}.png")
                        dst_a = os.path.join(annots_dir, f"{stem}.png")
                        if not os.path.exists(dst_p):
                            shutil.copyfile(pp, dst_p)
                        if not os.path.exists(dst_a):
                            shutil.copyfile(ap_cand, dst_a)
                        records.append({
                            "stem": stem,
                            "split": sp,
                            "photo_path": dst_p,
                            "annot_path": dst_a,
                        })
            continue

        print(f"  extracting {sp}.zip...")
        with zipfile.ZipFile(zp) as zf:
            nl = [n for n in zf.namelist() if not n.endswith("/")]
            photo_members = {os.path.basename(n).replace("_photo.png", "").replace(".png", ""): n
                             for n in nl if "photo" in n}
            annot_members = {os.path.basename(n).replace("_annotation.png", "").replace(".png", ""): n
                             for n in nl if "annotation" in n}

            common_stems = sorted(set(photo_members) & set(annot_members))
            print(f"    found {len(common_stems)} paired images in {sp}.zip")
            for stem in common_stems:
                dst_p = os.path.join(photos_dir, f"{stem}.png")
                dst_a = os.path.join(annots_dir, f"{stem}.png")
                if not os.path.exists(dst_p):
                    with open(dst_p, "wb") as f_out:
                        f_out.write(zf.read(photo_members[stem]))
                if not os.path.exists(dst_a):
                    with open(dst_a, "wb") as f_out:
                        f_out.write(zf.read(annot_members[stem]))
                records.append({
                    "stem": stem,
                    "split": sp,
                    "photo_path": dst_p,
                    "annot_path": dst_a,
                })

    df_m = pd.DataFrame(records).drop_duplicates("stem").reset_index(drop=True)
    df_m.to_csv(manifest_csv, index=False)
    print(f"Staged {len(df_m)} total photos across {df_m.split.value_counts().to_dict()} into {stage_dir}")
    return df_m


# ==============================================================================
# 2. Model Resolver: Find and load the most recent trained checkpoint
# ==============================================================================
def resolve_checkpoint(project_root: str, run_name: str, ckpt_name: str):
    realsoil_dir = os.path.join(project_root, "segmentation", "realsoil")
    if run_name == "latest":
        def _ckpt_mtime(d):
            ts = [os.path.getmtime(os.path.join(d, f)) for f in ("seg_best.pt", "seg_last.pt")
                  if os.path.exists(os.path.join(d, f))]
            return max(ts) if ts else -1.0
        runs = [d for d in glob.glob(os.path.join(realsoil_dir, "*"))
                if os.path.isdir(d) and _ckpt_mtime(d) > 0]
        if not runs:
            raise FileNotFoundError(f"No trained runs found under {realsoil_dir}")
        runs.sort(key=_ckpt_mtime)
        run_dir = runs[-1]
    else:
        run_dir = os.path.join(realsoil_dir, run_name)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Run folder {run_dir} does not exist.")

    ckpt_path = os.path.join(run_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        fallback = "seg_last.pt" if ckpt_name == "seg_best.pt" else "seg_best.pt"
        if os.path.exists(os.path.join(run_dir, fallback)):
            ckpt_path = os.path.join(run_dir, fallback)
        else:
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.")

    cfg_path = os.path.join(run_dir, "train_config.json")
    if os.path.exists(cfg_path):
        train_cfg = json.load(open(cfg_path, "r"))
    else:
        # Defaults
        train_cfg = {
            "ARCH": "Unet",
            "ENCODER": "mit_b3",
            "INDEX_MODE": "extra-channel",
            "AUX_DIST_WEIGHT": 0.25,
            "HIST_MATCH": "train+test",
            "SCALE": 1,
            "THRESH": 0.50,
        }

    return run_dir, ckpt_path, train_cfg


def hist_match(img: np.ndarray, cdf_ref: np.ndarray) -> np.ndarray:
    """Exact per-channel CDF histogram matching from train_v4.py / realsoil_train.py."""
    out = np.empty_like(img)
    for c in range(3):
        cnt = np.bincount(img[..., c].ravel(), minlength=256)
        cdf = np.cumsum(cnt) / max(cnt.sum(), 1)
        lut = np.interp(cdf, cdf_ref[c], np.arange(256))
        out[..., c] = np.clip(np.round(lut), 0, 255).astype(np.uint8)[img[..., c]]
    return out


def border_region(rgb: np.ndarray, dark: float = 0.55, margin: int = 6, edge: int = 3) -> np.ndarray:
    """Exact camera border bar detector from train_v4.py / realsoil_train.py."""
    y_ = rgb.astype(np.float32) @ np.float32([0.299, 0.587, 0.114])
    out = np.zeros(y_.shape, bool)
    med = float(np.median(y_)) + 1e-3
    sides = (
        (y_[:edge].mean(), y_[edge : edge + 10].mean(), (slice(0, edge), slice(None))),
        (y_[-edge:].mean(), y_[-edge - 10 : -edge].mean(), (slice(-edge, None), slice(None))),
        (y_[:, :edge].mean(), y_[:, edge : edge + 10].mean(), (slice(None), slice(0, edge))),
        (y_[:, -edge:].mean(), y_[:, -edge - 10 : -edge].mean(), (slice(None), slice(-edge, None))),
    )
    for outer, inner, sl in sides:
        if abs(outer - inner) > 0.08 * med:
            out[sl] = True
    for axis in (0, 1):
        prof = y_.mean(axis=1 - axis)
        spread = y_.std(axis=1 - axis)
        med_p = np.median(prof)
        bad = (prof < dark * med_p) | ((prof > 1.6 * med_p) & (spread < 0.15 * med_p))
        n = len(prof)
        a = int(np.argmin(bad)) if not bad.all() else n
        b = int(np.argmin(bad[::-1])) if not bad.all() else n
        if a > 0:
            a += margin
        if b > 0:
            b += margin
        if axis == 0:
            out[:a] = True
            out[n - b :] = True
        else:
            out[:, :a] = True
            out[:, n - b :] = True
    return out


def load_or_compute_prep_stats(run_dir: str, project_root: str, train_cfg: dict):
    """Load exact CDF_REF, IDX_MEAN, IDX_STD from run_dir/prep_stats.json, or compute them
    from the run's synthetic training frames using the exact same seed & logic as train_v4.py."""
    stats_path = os.path.join(run_dir, "prep_stats.json")
    if os.path.exists(stats_path):
        st = json.load(open(stats_path, "r"))
        return (
            np.asarray(st["CDF_REF"], dtype=np.float64),
            float(st["IDX_MEAN"]),
            float(st["IDX_STD"]),
        )

    # Locate synthetic training frames used by this run
    syn_tag = train_cfg.get("SYN_TAG", "New_folder")
    run_args_path = os.path.join(run_dir, "run_args.json")
    ds_dir_from_args = ""
    if os.path.exists(run_args_path):
        try:
            ra = json.load(open(run_args_path, "r"))
            ds_dir_from_args = ra.get("dataset_dir", "")
        except Exception:
            pass

    cand_syn_dirs = [
        f"/content/realsoil/syn_{syn_tag}",
        os.path.join(project_root, "_work", f"syn_{syn_tag}"),
        ds_dir_from_args,
        os.path.join(project_root, "datasets", "New folder"),
    ]
    syn_dir = None
    for d in cand_syn_dirs:
        if d and os.path.isdir(os.path.join(d, "rgb")):
            syn_dir = d
            break

    if syn_dir is None:
        print("WARNING: Could not find synthetic rgb/ folder to compute CDF_REF; using identity CDF.")
        cdf_ident = np.tile(np.linspace(0.0, 1.0, 256, dtype=np.float64), (3, 1))
        return cdf_ident, -0.05, 0.25

    print(f"  Computing pooled 300-frame CDF_REF & colour-index stats from {syn_dir}...")
    rgb_files = sorted(
        (p for p in glob.glob(os.path.join(syn_dir, "rgb", "*"))
         if os.path.exists(os.path.join(syn_dir, "mask", os.path.splitext(os.path.basename(p))[0] + ".png"))),
        key=lambda p: os.path.splitext(os.path.basename(p))[0],
    )
    seed = int(train_cfg.get("SEED", 2026))
    val_systems = int(train_cfg.get("VAL_SYSTEMS", 6))

    def _sys_id(p):
        m = re.search(r"s(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else (hash(os.path.basename(p)) % 100)

    df_f = pd.DataFrame({"path": rgb_files, "system": [_sys_id(p) for p in rgb_files]})
    all_sys = sorted(df_f.system.unique())
    n_val = min(val_systems, max(0, len(all_sys) - 1)) if val_systems > 0 else 0
    val_set = set(all_sys[-n_val:]) if n_val > 0 else set()
    tr_paths = df_f[~df_f.system.isin(val_set)]["path"]
    if len(tr_paths) == 0:
        tr_paths = df_f["path"]

    cnt = np.zeros((3, 256), dtype=np.float64)
    for p in tr_paths.sample(min(300, len(tr_paths)), random_state=seed):
        im_ = cv2.imread(p)[:, :, ::-1]
        for c in range(3):
            cnt[c] += np.bincount(im_[..., c].ravel(), minlength=256)
    cdf_ref = np.cumsum(cnt, 1) / np.maximum(cnt.sum(1, keepdims=True), 1.0)

    hist_mode = train_cfg.get("HIST_MATCH", "train+test")
    iv = []
    for p in tr_paths.sample(min(300, len(tr_paths)), random_state=seed + 1):
        im_ = cv2.imread(p)[:, :, ::-1]
        if hist_mode == "train+test":
            im_ = hist_match(im_, cdf_ref)
        iv.append(_raw_index(im_.astype(np.float32) / 255.0)[::4, ::4].ravel())
    iv_all = np.concatenate(iv)
    idx_mean, idx_std = float(iv_all.mean()), float(iv_all.std() + 1e-6)

    try:
        json.dump(
            {
                "HIST_MATCH": hist_mode,
                "INDEX_MODE": train_cfg.get("INDEX_MODE", "extra-channel"),
                "IDX_MEAN": idx_mean,
                "IDX_STD": idx_std,
                "CDF_REF": cdf_ref.tolist(),
            },
            open(stats_path, "w"),
        )
    except Exception:
        pass
    return cdf_ref, idx_mean, idx_std


# ==============================================================================
# 3. Prediction & Evaluation Engine
# ==============================================================================
def evaluate_grid_counted():
    args = parse_args()
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    project_root = args.root.rstrip("/\\")
    grid_dir = args.grid_dir
    if not grid_dir:
        cand1 = os.path.join(project_root, "Grid_counted_data")
        cand2 = "/content/drive/MyDrive/Raiz_deep_learning/Grid_counted_data"
        grid_dir = cand1 if os.path.exists(cand1) else cand2

    if not os.path.isdir(grid_dir):
        raise FileNotFoundError(f"Grid_counted_data folder {grid_dir} not found.")

    stage_dir = args.stage_dir
    if not stage_dir:
        stage_dir = "/content/grid_counted_stage" if os.path.isdir("/content") else os.path.join(grid_dir, "_staged")

    # 1. Stage images
    manifest = stage_grid_counted(grid_dir, stage_dir)
    if args.split != "all":
        wanted_splits = args.split.split("+")
        manifest = manifest[manifest.split.isin(wanted_splits)].reset_index(drop=True)
        print(f"Filtered to split(s) {wanted_splits}: {len(manifest)} photos.")

    if len(manifest) == 0:
        raise ValueError(f"No photos match split={args.split}")

    # 2. Resolve Model
    run_dir, ckpt_path, train_cfg = resolve_checkpoint(project_root, args.run_name, args.checkpoint)
    hist_mode = train_cfg.get("HIST_MATCH", "train+test")
    idx_mode = train_cfg.get("INDEX_MODE", "extra-channel")
    border_mask_on = bool(train_cfg.get("BORDER_MASK", True))
    train_scale = float(train_cfg.get("SCALE", 1))

    print("Evaluating Model:")
    print(f"  Run Directory : {run_dir}")
    print(f"  Checkpoint    : {ckpt_path}")
    print(f"  Architecture  : {train_cfg.get('ARCH', 'Unet')} + {train_cfg.get('ENCODER', 'mit_b3')}")
    print(f"  Hist Match    : {hist_mode}")
    print(f"  Index Mode    : {idx_mode}")
    print(f"  Border Mask   : {border_mask_on}")
    print(f"  Device        : {DEV}")

    IN_CH = {"off": 3, "extra-channel": 4, "index-only": 1}.get(idx_mode, 4)
    OUT_CLASSES = 2 if float(train_cfg.get("AUX_DIST_WEIGHT", 0.25)) > 0 else 1

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model = build_segmentation_model(
        train_cfg.get("ARCH", "Unet"),
        train_cfg.get("ENCODER", "mit_b3"),
        IN_CH,
        OUT_CLASSES,
        state_dict=state_dict,
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(DEV).eval()

    # Inference settings saved by train_v4.py (validation-selected threshold, post-processing)
    inf_path = os.path.join(run_dir, "inference_settings.json")
    inf = json.load(open(inf_path, "r")) if os.path.exists(inf_path) else {}
    border_mask_on = bool(inf.get("BORDER_MASK", border_mask_on))
    min_comp_px = int(inf.get("MIN_COMPONENT_PX", 30))

    # Threshold: explicit --thresh > validation-selected THRESH from training > 0.50.
    # (Never use summary.csv 'oracle_thr': it is tuned on the TEST sets and would leak.)
    thr = args.thresh
    thr_src = "--thresh"
    if thr <= 0:
        if "THRESH" in inf:
            thr, thr_src = float(inf["THRESH"]), "inference_settings.json (%s)" % inf.get("THRESH_SOURCE", "?")
        else:
            thr, thr_src = 0.50, "default (no inference_settings.json in run)"
    print(f"  Binarization Thr: {thr:.2f}  [{thr_src}]")
    print(f"  Min component   : {min_comp_px} px | Border mask: {border_mask_on}")

    # Determine scale factor (combines training SCALE and camera resolution ratio)
    scale_factor = ((args.ref_px_per_mm / args.px_per_mm) if args.scale_to_ref else 1.0) * train_scale
    print(f"  Scale Factor    : {scale_factor:.4f} (ref {args.ref_px_per_mm} / chicory {args.px_per_mm} * scale {train_scale:g})")

    # Load or compute exact pooled CDF_REF and colour index stats from training run
    cdf_ref, idx_mean, idx_std = load_or_compute_prep_stats(run_dir, project_root, train_cfg)
    print(f"  Colour Index    : mean={idx_mean:.4f}, std={idx_std:.4f}")
    to_input = to_input_factory(index_mode=idx_mode, idx_mean=idx_mean, idx_std=idx_std)

    def prep_real(img_rgb: np.ndarray) -> np.ndarray:
        if hist_mode in ("train+test", "test"):
            return hist_match(img_rgb, cdf_ref)
        return img_rgb

    # 3. Run Inference & Calculate Metrics
    results = []
    overlays_dir = os.path.join(run_dir, "grid_counted_overlays")
    os.makedirs(overlays_dir, exist_ok=True)

    print("\n--- Running Predictions on Full-Resolution Rhizotron Photos ---")
    t_start = time.time()
    all_pairs = []

    for idx, row in manifest.iterrows():
        t0 = time.time()
        stem = row["stem"]
        split = row["split"]
        rgb = cv2.imread(row["photo_path"])[:, :, ::-1].copy()
        with open(row["annot_path"], "rb") as fh_a:
            an, _valid = RD.decode_annotation(fh_a.read())
        an = np.asarray(an, dtype=bool)
        H, W = rgb.shape[:2]
        if an.shape != (H, W):
            an = cv2.resize(an.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0

        # Ground truth skeleton length
        gt_sk = skeletonize(an)
        obs_len_mm = float(gt_sk.sum()) / args.px_per_mm

        # Apply exact training histogram matching (prep(rgb, "real")) before sliding-window prediction
        rgb_in = prep_real(rgb)
        p = TL.predict_image(
            model,
            rgb_in,
            to_input,
            scale=scale_factor,
            device=DEV,
            tta=args.tta,
            tile=768,
            overlap=128,
            batch=4,
            max_full=1024,
            out_hw=(H, W),
            amp_dtype=torch.float16 if DEV == "cuda" else None,
        )
        if border_mask_on:
            p[border_region(rgb)] = 0.0

        # Binarize and filter small debris (same MIN_COMPONENT_PX as training evaluation)
        m = (p > thr).astype(np.uint8)
        n_cc, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros(n_cc, bool)
        keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_comp_px
        pred_m = keep[lab]

        # Predicted skeleton length
        pred_sk = skeletonize(pred_m)
        pred_len_mm = float(pred_sk.sum()) / args.px_per_mm

        # Pixel metrics
        intersection = (pred_m & an).sum()
        union = (pred_m | an).sum()
        dice = 2.0 * intersection / max(pred_m.sum() + an.sum(), 1)
        iou = float(intersection) / max(union, 1)

        # Tolerance 3px skeleton precision/recall
        dg = cv2.distanceTransform((~gt_sk).astype(np.uint8), cv2.DIST_L2, 5)
        dp = cv2.distanceTransform((~pred_sk).astype(np.uint8), cv2.DIST_L2, 5)
        tp_p = int((dg[pred_sk] <= 3).sum()) if gt_sk.sum() > 0 else 0
        tp_r = int((dp[gt_sk] <= 3).sum()) if pred_sk.sum() > 0 else 0
        prec3 = tp_p / max(pred_sk.sum(), 1)
        rec3 = tp_r / max(gt_sk.sum(), 1)
        f1_3px = (2.0 * prec3 * rec3 / (prec3 + rec3)) if (prec3 + rec3) > 0 else 0.0

        error_mm = pred_len_mm - obs_len_mm
        abs_err_mm = abs(error_mm)
        pct_err = (error_mm / max(obs_len_mm, 1e-2)) * 100.0

        rec = {
            "stem": stem,
            "split": split,
            "width": W,
            "height": H,
            "observed_len_mm": round(obs_len_mm, 2),
            "predicted_len_mm": round(pred_len_mm, 2),
            "error_mm": round(error_mm, 2),
            "abs_error_mm": round(abs_err_mm, 2),
            "pct_error": round(pct_err, 2),
            "dice": round(float(dice), 4),
            "iou": round(float(iou), 4),
            "precision_3px": round(float(prec3), 4),
            "recall_3px": round(float(rec3), 4),
            "f1_3px": round(float(f1_3px), 4),
            "sec": round(time.time() - t0, 2),
        }
        results.append(rec)
        w_t = 520
        h_t = int(round(H * w_t / W))
        all_pairs.append((
            cv2.resize(rgb, (w_t, h_t), interpolation=cv2.INTER_AREA),
            cv2.resize(pred_m.astype(np.uint8) * 255, (w_t, h_t), interpolation=cv2.INTER_AREA) > 40,
            cv2.resize(an.astype(np.uint8) * 255, (w_t, h_t), interpolation=cv2.INTER_AREA) > 40,
            rec,
        ))

        if (idx + 1) % 5 == 0 or (idx + 1) == len(manifest):
            print(
                f"  [{idx+1:2d}/{len(manifest)}] {stem:<30} | Obs: {obs_len_mm:6.1f} mm | Pred: {pred_len_mm:6.1f} mm | Dice: {dice:.3f} | {time.time()-t0:.1f}s",
                flush=True,
            )

    df_res = pd.DataFrame(results)
    csv_out = os.path.join(run_dir, "grid_counted_results.csv")
    df_res.to_csv(csv_out, index=False)
    print(f"\nSaved per-image predictions to {csv_out}")

    # 4. Compute Statistical Metrics: Pearson r, Spearman rho, R^2, MAE, RMSE
    print("\n========================================================")
    print("      GRID_COUNTED ROOT LENGTH EVALUATION SUMMARY       ")
    print("========================================================")
    summary_rows = []
    for sp_name, g in [("ALL", df_res)] + list(df_res.groupby("split")):
        if len(g) < 2:
            continue
        y_true = g["observed_len_mm"].to_numpy()
        y_pred = g["predicted_len_mm"].to_numpy()

        if np.std(y_true) > 1e-6 and np.std(y_pred) > 1e-6:
            r_val, p_val = stats.pearsonr(y_true, y_pred)
            rho_val, _ = stats.spearmanr(y_true, y_pred)
        else:
            r_val, rho_val = np.nan, np.nan

        # R^2 = 1 - SS_res / SS_tot
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2_val = 1.0 - (ss_res / max(ss_tot, 1e-8))

        mae_val = np.mean(np.abs(y_true - y_pred))
        rmse_val = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mean_obs = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        mean_dice = g["dice"].mean()
        mean_f1 = g["f1_3px"].mean()

        summary_rows.append({
            "split": sp_name,
            "n": len(g),
            "Pearson_r": round(float(r_val), 3),
            "R2": round(float(r2_val), 3),
            "Spearman_rho": round(float(rho_val), 3),
            "MAE_mm": round(float(mae_val), 1),
            "RMSE_mm": round(float(rmse_val), 1),
            "Mean_Obs_mm": round(float(mean_obs), 1),
            "Mean_Pred_mm": round(float(mean_pred), 1),
            "Mean_Dice": round(float(mean_dice), 3),
            "Mean_F1_3px": round(float(mean_f1), 3),
        })

    df_sum = pd.DataFrame(summary_rows)
    sum_csv = os.path.join(run_dir, "grid_counted_summary.csv")
    df_sum.to_csv(sum_csv, index=False)
    print(df_sum.to_string(index=False))

    # 5. Publication-Quality Scatter Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=140)
    colors = {"train": "#4C78A8", "val": "#F58518", "test": "#54A24B"}

    for sp_name, g in df_res.groupby("split"):
        ax.scatter(
            g["observed_len_mm"],
            g["predicted_len_mm"],
            label=f"{sp_name.upper()} (n={len(g)})",
            color=colors.get(sp_name, "gray"),
            alpha=0.85,
            edgecolors="k",
            s=55,
            zorder=3,
        )

    # 1:1 diagonal line
    max_val = max(df_res["observed_len_mm"].max(), df_res["predicted_len_mm"].max()) * 1.08
    ax.plot([0, max_val], [0, max_val], "k--", lw=1.2, label="1:1 Perfect Agreement", zorder=2)

    # Overall regression line
    all_x = df_res["observed_len_mm"].to_numpy()
    all_y = df_res["predicted_len_mm"].to_numpy()
    if np.std(all_x) > 1e-6 and np.std(all_y) > 1e-6:
        slope, intercept, r_value, _, _ = stats.linregress(all_x, all_y)
    else:
        slope, intercept, r_value = 1.0, 0.0, 0.0
    reg_x = np.array([0, max_val])
    reg_y = slope * reg_x + intercept
    ax.plot(reg_x, reg_y, color="#E45756", lw=1.8, label=f"Fit: y = {slope:.2f}x + {intercept:.1f}", zorder=2)

    all_r2 = 1.0 - (np.sum((all_x - all_y) ** 2) / np.sum((all_x - np.mean(all_x)) ** 2))
    all_mae = np.mean(np.abs(all_x - all_y))

    stats_text = (
        f"Overall Statistics (N = {len(df_res)}):\n"
        f"  Pearson r = {r_value:.3f}\n"
        f"  R² = {all_r2:.3f}\n"
        f"  MAE = {all_mae:.1f} mm\n"
        f"  Mean Dice = {df_res['dice'].mean():.3f}"
    )
    ax.text(
        0.04, 0.96, stats_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F4F8", edgecolor="#BCCCDC", alpha=0.9),
    )

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("Observed Ground Truth Root Length (mm)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted Model Root Length (mm)", fontsize=11, fontweight="bold")
    ax.set_title(f"Root Length: Predicted vs. Observed on Grid_counted_data\nModel: {train_cfg.get('ARCH', 'Unet')} + {train_cfg.get('ENCODER', 'mit_b3')}", fontsize=11.5, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    scatter_path = os.path.join(run_dir, "grid_counted_scatter.png")
    fig.savefig(scatter_path, dpi=160)
    plt.close(fig)
    print(f"\nSaved correlation scatter plot: {scatter_path}")

    # 6. Visual Overlays for Best & Worst Images
    if args.n_overlays > 0 and len(all_pairs) > 0:
        print(f"Exporting top {args.n_overlays} visual overlays...")
        all_pairs.sort(key=lambda x: x[3]["abs_error_mm"])
        selected_pairs = all_pairs[: args.n_overlays // 2] + all_pairs[-args.n_overlays // 2 :]

        for rgb_im, p_m, gt_m, meta in selected_pairs:
            vis = rgb_im.copy()
            # Green overlay for predictions
            vis[p_m] = (0.4 * vis[p_m] + np.array([0, 180, 0])).astype(np.uint8)
            # Red overlay for ground truth
            gt_overlay = rgb_im.copy()
            gt_overlay[gt_m] = (0.4 * gt_overlay[gt_m] + np.array([180, 0, 0])).astype(np.uint8)

            # Resize to fit side-by-side
            w_disp = 520
            h_disp = int(rgb_im.shape[0] * w_disp / rgb_im.shape[1])
            col1 = cv2.resize(rgb_im, (w_disp, h_disp))
            col2 = cv2.resize(vis, (w_disp, h_disp))
            col3 = cv2.resize(gt_overlay, (w_disp, h_disp))
            sheet = np.hstack([col1, col2, col3])

            header = np.full((32, sheet.shape[1], 3), 240, dtype=np.uint8)
            txt = (f"{meta['split'].upper()} | {meta['stem']} | "
                   f"Observed: {meta['observed_len_mm']:.0f} mm | Predicted: {meta['predicted_len_mm']:.0f} mm | "
                   f"Err: {meta['error_mm']:+.0f} mm | Dice: {meta['dice']:.2f}")
            cv2.putText(header, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
            out_img = np.vstack([header, sheet])
            fn = os.path.join(overlays_dir, f"{meta['split']}_{meta['stem']}_overlay.jpg")
            cv2.imwrite(fn, out_img[:, :, ::-1])

        print(f"Saved visual overlay sheets to {overlays_dir}/")

    print("\nEvaluation complete in %.1f minutes." % ((time.time() - t_start) / 60.0))
    return df_res, df_sum


if __name__ == "__main__":
    evaluate_grid_counted()
