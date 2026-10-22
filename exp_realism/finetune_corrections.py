"""Fine-tune a trained root model on the annotator's corrected masks.

Every mask saved in the annotator (``<folder>/_annotations/<stem>_mask.png``) is a
reviewed label for this site and camera.  This adapts a trained run to them:
the encoder (44M weights, the general feature extractor) stays frozen and in
eval mode; the decoder and segmentation head (3.3M) are retrained with the loss
the run was trained with (``train_v4``: thin-root-weighted BCE + Dice + clDice
+ the centre-line distance head).  The result is a new run folder next to the
base one -- the base is never modified -- that the annotator lists as a model.

Two ways to run the same job:

* ``cached`` (CPU default): the frozen encoder runs once per image for three
  views (as-is, h-flip, v-flip); its feature pyramid is kept on local disk and
  only the decoder trains, on random crops aligned to the encoder's stride.
  Minutes instead of hours on a CPU.
* ``full`` (GPU default): ordinary training with the encoder forward every step
  and flip / rot90 / photometric / scale augmentation.

Held-out tube windows (never split across train/validation, since dates of one
window look alike) give skeleton F1 at 3 px for the base and the new model.

Optionally (``synthetic.per_real`` in the job) simulated roots matched to the
annotations are drawn on the *training* frames' own soil, with the annotated roots
inpainted away and ignored (``exp_realism.synth_from_annotations``); a share of every
batch (``synthetic.share``) then comes from them.  Held-out windows are never used
for them, so they cannot leak into training.

    python -m exp_realism.finetune_corrections --job segmentation/realsoil/_finetune_jobs/<run>

The annotator builds the job (:func:`create_job`) and either runs this as a
background process or writes a Colab notebook (:func:`write_colab_notebook`).
Progress goes to stdout as ``FT {json}`` lines.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

JOBS_DIR = os.path.join("segmentation", "realsoil", "_finetune_jobs")
MIN_IMAGES = 10
DEFAULTS = {
    "cached": {"steps": 600, "batch": 4, "crop": 320, "lr": 1e-4, "eval_every": 100},
    "full": {"steps": 1500, "batch": 8, "crop": 384, "lr": 1e-4, "eval_every": 250},
}
VAL_FRAC = 0.2
TOL_PX = 3
MIN_COMPONENT_PX = 3
# low end included: base models trained elsewhere can be badly under-confident on a new site
THRESHOLDS = np.r_[0.005, 0.01, 0.02, 0.03, np.round(np.arange(0.05, 0.91, 0.05), 2)]
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _emit(**kw):
    print("FT " + json.dumps(kw), flush=True)


# ==============================================================================
# Job folder (built by the annotator; self-contained so Colab needs nothing else)
# ==============================================================================

def find_corrected_pairs(folder: str) -> List[Tuple[str, str]]:
    """(image, saved mask) pairs of an image folder, from its ``_annotations``."""
    ann = os.path.join(folder, "_annotations")
    if not os.path.isdir(ann):
        return []
    images = {os.path.splitext(f)[0]: os.path.join(folder, f) for f in os.listdir(folder)
              if f.lower().endswith(_IMG_EXTS) and not f.startswith(".")}
    pairs = []
    for f in sorted(os.listdir(ann)):
        if f.endswith("_mask.png"):
            stem = f[: -len("_mask.png")]
            if stem in images:
                pairs.append((images[stem], os.path.join(ann, f)))
    return pairs


def base_of(project_root: str, run: str) -> str:
    """The original run a fine-tune came from (fine-tunes are re-done from it, not stacked)."""
    cfg = os.path.join(project_root, "segmentation", "realsoil", run, "train_config.json")
    try:
        with open(cfg, "r") as fh:
            return json.load(fh).get("BASE_RUN") or run
    except (OSError, ValueError):
        return run


def new_run_name(base_run: str, n_images: int) -> str:
    head = base_run.split("_")[0]
    head = head if head.startswith("v") else "v" + head
    return f"{head}ft_{_dt.datetime.now():%Y%m%d-%H%M}_{n_images}img"


def create_job(project_root: str, base_run: str, pairs: Sequence[Tuple[str, str]], source: str = "",
               mode_defaults: str = "cached", synth_per_real: int = 0, synth_config: Optional[Dict] = None,
               synth_fitted: bool = False, **overrides) -> str:
    """Copy the pairs into a new job folder with its config; returns the job folder.

    ``synth_per_real`` > 0 adds that many simulated frames per training frame, with the simulator
    settings ``synth_config`` (``RealismConfig`` dict; fitted to the training frames in the job when
    ``synth_fitted`` is False)."""
    if len(pairs) < MIN_IMAGES:
        raise ValueError(f"{len(pairs)} corrected images: at least {MIN_IMAGES} are needed (30+ recommended).")
    run_name = new_run_name(base_run, len(pairs))
    job = os.path.join(project_root, JOBS_DIR, run_name)
    os.makedirs(os.path.join(job, "images"), exist_ok=True)
    os.makedirs(os.path.join(job, "masks"), exist_ok=True)
    stems = []
    for img, mask in pairs:
        stem = os.path.splitext(os.path.basename(img))[0]
        shutil.copy2(img, os.path.join(job, "images", os.path.basename(img)))
        shutil.copy2(mask, os.path.join(job, "masks", stem + ".png"))
        stems.append(stem)
    cfg = {
        "run_name": run_name,
        "base_run": base_run,
        "stems": stems,
        "source": source,
        "created_at": f"{_dt.datetime.now():%Y-%m-%d %H:%M}",
        "seed": 2026,
        "val_frac": VAL_FRAC,
        "unfreeze_last_stage": False,
        "synthetic": {"per_real": int(synth_per_real), "share": 0.4, "config": synth_config or {},
                      "fitted": bool(synth_fitted)},
    }
    cfg.update(overrides)
    with open(os.path.join(job, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=1)
    return job


def write_colab_notebook(project_root: str, job: str) -> Tuple[str, str]:
    """Colab notebook that runs the job on a GPU; saved in the job folder and in ``notebooks/``."""
    rel_job = os.path.relpath(job, project_root).replace("\\", "/")
    drive_dir = os.path.basename(os.path.abspath(project_root))     # the project folder, at the top of My Drive
    with open(os.path.join(job, "config.json"), "r") as fh:
        cfg = json.load(fh)
    run = cfg["run_name"]
    cells = [
        ("md", f"# Fine-tune `{cfg['base_run']}` on {len(cfg['stems'])} corrected masks\n\n"
               "Made by the root annotator.  **Runtime -> Change runtime type -> GPU**, then **Run all**.\n"
               f"The new model is written to `segmentation/realsoil/{run}` on your Drive; back in the annotator "
               "press ↻ next to *Model* and pick it.\n"),
        ("code", "#@title Mount Drive, install the pinned libraries\n"
                 "from google.colab import drive\n"
                 "drive.mount('/content/drive')\n"
                 f"ROOT = '/content/drive/MyDrive/{drive_dir}'\n"
                 f"JOB = ROOT + '/{rel_job}'\n"
                 "# same versions as the annotator, so the saved weights load there unchanged\n"
                 "!pip -q install segmentation-models-pytorch==0.3.3 timm==0.9.2 \"opencv-python-headless<5\"\n"),
        ("code", "#@title Copy the code to local disk (faster imports than from Drive)\n"
                 "import os, shutil, sys\n"
                 "CODE = '/content/raiz_code'\n"
                 "for pkg in ('mrsim', 'mrstyle', 'mrmodel', 'exp_realism'):\n"
                 "    src = os.path.join(ROOT, pkg)\n"
                 "    if os.path.isdir(src):\n"
                 "        shutil.copytree(src, os.path.join(CODE, pkg), dirs_exist_ok=True,\n"
                 "                        ignore=shutil.ignore_patterns('__pycache__', '*.ipynb', '*.jpg', '*.png'))\n"
                 "print(sorted(os.listdir(CODE)))\n"),
        ("code", "#@title Train (decoder + head; encoder frozen)\n"
                 "UNFREEZE_LAST_STAGE = False  #@param {type:\"boolean\"}\n"
                 "STEPS = 1500  #@param {type:\"integer\"}\n"
                 "%cd /content/raiz_code\n"
                 "!python -m exp_realism.finetune_corrections --root \"$ROOT\" --job \"$JOB\" --device cuda "
                 "--steps $STEPS --unfreeze-last-stage $UNFREEZE_LAST_STAGE\n"),
        ("code", "#@title Result\n"
                 "import json\n"
                 f"rep = json.load(open(ROOT + '/segmentation/realsoil/{run}/finetune_report.json'))\n"
                 "print(json.dumps({k: rep[k] for k in ('base', 'new', 'n_train', 'n_val', 'minutes')}, indent=1))\n"),
    ]
    nb = {
        "cells": [
            ({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)} if kind == "md" else
             {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
              "source": src.splitlines(True)})
            for kind, src in cells
        ],
        "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 0,
    }
    in_job = os.path.join(job, "finetune_colab.ipynb")
    copy = os.path.join(project_root, "notebooks", f"finetune_{run}.ipynb")
    for p in (in_job, copy):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(nb, fh, indent=1)
    return in_job, copy


# ==============================================================================
# Labels, loss and scores (mirroring exp_realism/train_v4.py)
# ==============================================================================

def dist_target(y: np.ndarray) -> np.ndarray:
    """Centre-line distance target of train_v4: distance transform / 7x7 local max, roots only."""
    m = (y > 0.5).astype(np.uint8)
    if not m.any():
        return np.zeros_like(y, dtype=np.float32)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float32)
    local_max = cv2.dilate(dt, np.ones((7, 7), np.uint8))
    return np.clip(np.where(m > 0, dt / np.maximum(local_max, 1.0), 0.0), 0.0, 1.0).astype(np.float32)


def _soft_erode(x):
    import torch.nn.functional as F
    import torch
    return torch.min(-F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1)))


def _soft_skel(x, iters: int = 10):
    import torch.nn.functional as F
    skel = F.relu(x - F.max_pool2d(_soft_erode(x), 3, 1, 1))
    for _ in range(iters):
        x = _soft_erode(x)
        d = F.relu(x - F.max_pool2d(_soft_erode(x), 3, 1, 1))
        skel = skel + F.relu(d - skel * d)
    return skel


def seg_loss(out, y, v, w, dist_gt, cldice_w: float, dist_w: float):
    import torch
    import torch.nn.functional as F
    logit = out[:, :1]
    p = torch.sigmoid(logit)
    bce = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).sum() / w.sum().clamp_min(1.0)
    dice = 1.0 - (2.0 * (p * y * v).sum() + 1.0) / ((p * v).sum() + (y * v).sum() + 1.0)
    sp, sy = _soft_skel(p) * v, _soft_skel(y) * v
    tprec = ((sp * y).sum() + 1.0) / (sp.sum() + 1.0)
    tsens = ((sy * p).sum() + 1.0) / (sy.sum() + 1.0)
    total = bce + dice + cldice_w * (1.0 - 2.0 * tprec * tsens / (tprec + tsens))
    if dist_w > 0 and out.shape[1] >= 2:
        dl = (F.smooth_l1_loss(torch.sigmoid(out[:, 1:2]), dist_gt, reduction="none") * v).sum() / v.sum().clamp_min(1.0)
        total = total + dist_w * dl
    return total


def _binarize(p: np.ndarray, thr: float) -> np.ndarray:
    m = (p > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= MIN_COMPONENT_PX
    return keep[lab]


def score_maps(probs: Sequence[np.ndarray], masks: Sequence[np.ndarray]) -> Dict[str, float]:
    """Best skeleton F1 at 3 px over the threshold scan (counts pooled over images), with its Dice."""
    from skimage.morphology import skeletonize
    gts = [m > 0.5 for m in masks]
    gt_sk = [skeletonize(g) for g in gts]
    dgs = [cv2.distanceTransform((~s).astype(np.uint8), cv2.DIST_L2, 5) for s in gt_sk]
    best = None
    for thr in THRESHOLDS:
        tp_p = n_p = tp_r = n_g = 0
        inter = s_p = s_g = 0
        for p, g, gs, dg in zip(probs, gts, gt_sk, dgs):
            b = _binarize(p, thr)
            ps = skeletonize(b)
            dp = cv2.distanceTransform((~ps).astype(np.uint8), cv2.DIST_L2, 5)
            n_p += int(ps.sum()); n_g += int(gs.sum())
            tp_p += int((dg[ps] <= TOL_PX).sum()) if gs.any() else 0
            tp_r += int((dp[gs] <= TOL_PX).sum()) if ps.any() else 0
            inter += int((b & g).sum()); s_p += int(b.sum()); s_g += int(g.sum())
        P = tp_p / n_p if n_p else (1.0 if n_g == 0 else 0.0)
        R = tp_r / n_g if n_g else 1.0
        F1 = 2 * P * R / (P + R) if P + R > 0 else 0.0
        dice = 2 * inter / (s_p + s_g) if s_p + s_g else 1.0
        if best is None or F1 > best["f1"] + 1e-9:
            best = {"f1": round(F1, 4), "precision": round(P, 4), "recall": round(R, 4),
                    "dice": round(dice, 4), "thresh": float(thr)}
    return best


# ==============================================================================
# Data
# ==============================================================================

def split_by_window(stems: Sequence[str], frac: float, seed: int) -> Tuple[List[str], List[str]]:
    """Hold out whole tube windows (all their dates) for validation."""
    from exp_realism.temporal import parse_frame
    groups: Dict[str, List[str]] = {}
    for s in stems:
        k = parse_frame(s)
        groups.setdefault(f"T{k.tube}_L{k.loc}" if k else s, []).append(s)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    target = max(5, int(round(frac * len(stems))))
    val: List[str] = []
    for k in keys:
        if len(val) >= target or len(val) + len(groups[k]) > len(stems) - MIN_IMAGES // 2:
            continue
        val.extend(groups[k])
    train = [s for s in stems if s not in set(val)]
    return train, val


def _pad32(a: np.ndarray, mode: str) -> np.ndarray:
    h, w = a.shape[:2]
    ph, pw = (-h) % 32, (-w) % 32
    if ph == 0 and pw == 0:
        return a
    pad = ((0, ph), (0, pw)) + ((0, 0),) * (a.ndim - 2)
    return np.pad(a, pad, mode=mode)


class Sample:
    """One training frame (real or simulated): rgb, labels and the valid map (0 = ignored)."""

    def __init__(self, stem: str, rgb: np.ndarray, y: np.ndarray, cdf_ref, to_input, thin_weights,
                 valid: Optional[np.ndarray] = None):
        from exp_realism.eval_grid_counted import hist_match
        self.stem, self.rgb = stem, rgb
        self.h, self.w = rgb.shape[:2]
        self.y = (np.asarray(y) > 0.5).astype(np.float32)
        self.valid = np.ones_like(self.y) if valid is None else np.asarray(valid, np.float32)
        self.matched = hist_match(rgb, cdf_ref)
        self.to_input, self.thin_weights = to_input, thin_weights

    @classmethod
    def from_job(cls, job: str, stem: str, cdf_ref, to_input, thin_weights) -> "Sample":
        img_path = next(os.path.join(job, "images", f) for f in os.listdir(os.path.join(job, "images"))
                        if os.path.splitext(f)[0] == stem)
        rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        m = cv2.imread(os.path.join(job, "masks", stem + ".png"), cv2.IMREAD_GRAYSCALE)
        return cls(stem, rgb, m > 127, cdf_ref, to_input, thin_weights)

    def tensors(self, matched: np.ndarray, y: np.ndarray, valid: Optional[np.ndarray] = None):
        """Padded network input and labels (y, v, w, dist) for an (augmented) frame."""
        x = self.to_input(_pad32(matched, "reflect").astype(np.float32) / 255.0)
        v = _pad32(self.valid if valid is None else valid, "constant")
        yp = _pad32(y, "constant")
        return x, yp, v, self.thin_weights(yp, v), dist_target(yp)


# ==============================================================================
# Training
# ==============================================================================

def _decoder_forward(model, feats):
    """Decoder + head on an encoder pyramid laid out as smp 0.3.3 returns it:
    [input (dropped by the decoder), 0-channel 1/2 dummy, f/4, f/8, f/16, f/32]."""
    return model.segmentation_head(model.decoder(*feats))


def _features(model, x):
    """The stored part of the encoder pyramid: f/4, f/8, f/16, f/32."""
    return model.encoder(x)[2:]


def finetune(job: str, project_root: str, device: str = "auto", mode: str = "auto",
             steps: Optional[int] = None, unfreeze_last_stage: Optional[bool] = None) -> str:
    import torch
    from exp_realism.annotator_backend import load_model_bundle
    from functools import partial
    from mrmodel.segdata import thin_weights

    t_start = time.time()
    with open(os.path.join(job, "config.json"), "r") as fh:
        cfg = json.load(fh)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    mode = ("full" if device == "cuda" else "cached") if mode == "auto" else mode
    hp = dict(DEFAULTS[mode])
    if steps:
        hp["steps"] = int(steps)
    unfreeze = bool(cfg.get("unfreeze_last_stage", False) if unfreeze_last_stage is None else unfreeze_last_stage)
    unfreeze = unfreeze and mode == "full"
    seed = int(cfg.get("seed", 2026))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    base_run, run_name = cfg["base_run"], cfg["run_name"]
    _emit(phase="load", msg=f"loading {base_run} on {device} ({mode})")
    bundle = load_model_bundle(project_root, base_run, device)
    model, train_cfg = bundle["model"], dict(bundle["train_cfg"])
    cldice_w = float(train_cfg.get("CLDICE_WEIGHT", 0.35))
    dist_w = float(train_cfg.get("AUX_DIST_WEIGHT", 0.25))

    # Frozen encoder in eval mode (deterministic features, no DropPath); decoder + head train
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    trainable = [{"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")], "lr": hp["lr"]}]
    if unfreeze:
        last = [p for n, p in model.encoder.named_parameters() if n.split(".")[0] in ("patch_embed4", "block4", "norm4")]
        for p in last:
            p.requires_grad_(True)
        trainable.append({"params": last, "lr": hp["lr"] * 0.1})
    for m in model.decoder.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.momentum = 0.01

    def set_train():
        model.train()
        if not unfreeze:
            model.encoder.eval()

    stems = list(cfg["stems"])
    train_stems, val_stems = split_by_window(stems, float(cfg.get("val_frac", VAL_FRAC)), seed)
    _emit(phase="data", n_train=len(train_stems), n_val=len(val_stems))
    tw = partial(thin_weights, max_w=float(train_cfg.get("THIN_MAX_W", 3.0)))
    samples = {s: Sample.from_job(job, s, bundle["cdf_ref"], bundle["to_input"], tw) for s in stems}

    # -- simulated roots on the training frames' soil (never the held-out windows) -------------------
    syn_cfg = cfg.get("synthetic") or {}
    syn_stems: List[str] = []
    syn_info: Dict = {}
    syn_preview = []
    n_syn = int(syn_cfg.get("per_real", 0)) * len(train_stems)
    if n_syn > 0:
        from exp_realism.synth_from_annotations import SynthGenerator
        frames = [(s, samples[s].rgb, samples[s].y > 0.5) for s in train_stems]
        gen = SynthGenerator(project_root, frames, base_cfg=syn_cfg.get("config") or None, seed=seed)
        if not syn_cfg.get("fitted"):
            t_fit = time.time()
            gen.fit(progress=lambda what, i, n: _emit(phase="synth_fit", what=what, i=i, n=n,
                                                       eta_s=round((time.time() - t_fit) / i * (n - i))))
        t_syn = time.time()
        for j in range(n_syn):
            rgb_s, m_s, v_s, _ = gen.render(j % len(frames), seed=j)
            st_ = f"syn{j:05d}"
            samples[st_] = Sample(st_, rgb_s, m_s, bundle["cdf_ref"], bundle["to_input"], tw, valid=v_s)
            syn_stems.append(st_)
            if len(syn_preview) < 9:
                syn_preview.append((rgb_s, m_s, v_s))
            _emit(phase="synth", i=j + 1, n=n_syn, eta_s=round((time.time() - t_syn) / (j + 1) * (n_syn - j - 1)))
        syn_info = {"n": n_syn, "share": float(syn_cfg.get("share", 0.4)), "fit": gen.fit_report,
                    "config": gen.cfg.to_dict(), "stats": {k: v for k, v in gen.stats.items()
                                                          if k not in ("color_deltas", "length_px")}}
    syn_share = float(syn_cfg.get("share", 0.4)) if syn_stems else 0.0

    opt = torch.optim.AdamW(trainable, weight_decay=1e-4)
    warm = 50
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, i / max(1, hp["steps"])))))
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # -- validation inputs (full frames) -------------------------------------------------
    def val_maps(cache_val=None) -> List[np.ndarray]:
        model.eval()
        out = []
        with torch.no_grad():
            for s in val_stems:
                smp_ = samples[s]
                if cache_val is not None:
                    feats = [torch.from_numpy(np.asarray(f, dtype=np.float32)).to(device) for f in cache_val[s]]
                    feats = [None, torch.empty((1, 0, feats[0].shape[2] * 2, feats[0].shape[3] * 2), device=device)] + feats
                    logits = _decoder_forward(model, feats)
                else:
                    x, *_ = smp_.tensors(smp_.matched, smp_.y)
                    with torch.autocast(device, enabled=use_amp):
                        logits = model(torch.from_numpy(x)[None].to(device))
                p = torch.sigmoid(logits[0, 0].float()).cpu().numpy()[: smp_.h, : smp_.w]
                b = val_border[len(out)]
                if b.any():
                    p = p.copy()
                    p[b] = 0.0
                out.append(p)
        set_train()
        return out

    val_masks = [samples[s].y for s in val_stems]
    # validation scored as the model is used in the annotator: camera border bars masked
    from exp_realism.annotator_backend import border_bars
    val_border = [border_bars(samples[s].rgb) for s in val_stems]

    # -- cached mode: encoder once per view ---------------------------------------------------
    tmp_dir = None
    cache: Dict[Tuple[str, int], List[np.ndarray]] = {}
    labels: Dict[Tuple[str, int], Tuple[np.ndarray, ...]] = {}
    cache_val: Optional[Dict[str, List[np.ndarray]]] = None
    if mode == "cached":
        tmp_dir = tempfile.mkdtemp(prefix=f"rootft_{run_name}_")
        views = ([(s, k) for s in train_stems for k in range(3)] + [(s, 0) for s in syn_stems]
                 + [(s, -1) for s in val_stems])
        cache_val = {}
        t0 = time.time()
        model.eval()
        with torch.no_grad():
            for i, (s, k) in enumerate(views):
                smp_ = samples[s]
                img, y, vv = smp_.matched, smp_.y, smp_.valid
                if k == 1:
                    img, y, vv = img[:, ::-1], y[:, ::-1], vv[:, ::-1]
                elif k == 2:
                    img, y, vv = img[::-1], y[::-1], vv[::-1]
                x, yp, v, w, d = smp_.tensors(np.ascontiguousarray(img), np.ascontiguousarray(y),
                                              np.ascontiguousarray(vv))
                feats = _features(model, torch.from_numpy(x)[None].to(device))
                stored = []
                for j, f in enumerate(feats):
                    arr = f[0].cpu().numpy().astype(np.float16)
                    mm = np.lib.format.open_memmap(os.path.join(tmp_dir, f"{i}_{j}.npy"), mode="w+",
                                                   dtype=np.float16, shape=arr.shape)
                    mm[:] = arr
                    stored.append(mm)
                if k < 0:
                    cache_val[s] = [a[None] for a in stored]
                else:
                    cache[(s, k)] = stored
                    labels[(s, k)] = (yp, v, w, d)
                done = i + 1
                el = time.time() - t0
                _emit(phase="cache", i=done, n=len(views), eta_s=round(el / done * (len(views) - done)))
        set_train()

    # -- base model on the held-out windows ----------------------------------------------------
    base_score = score_maps(val_maps(cache_val), val_masks) if val_stems else None
    _emit(phase="base", score=base_score)

    def batch_cached():
        keys_real = [k for k in cache if not k[0].startswith("syn")]
        keys_syn = [k for k in cache if k[0].startswith("syn")]
        feats_b = [[] for _ in range(4)]
        ys, vs, ws, ds = [], [], [], []
        c = hp["crop"]
        for _ in range(hp["batch"]):
            pool = keys_syn if (keys_syn and rng.random() < syn_share) else keys_real
            key = pool[rng.integers(len(pool))]
            f4 = cache[key][0]
            Hp, Wp = f4.shape[1] * 4, f4.shape[2] * 4
            ch, cw = min(c, Hp), min(c, Wp)
            oy = int(rng.integers(0, (Hp - ch) // 32 + 1)) * 32
            ox = int(rng.integers(0, (Wp - cw) // 32 + 1)) * 32
            for j, sc in enumerate((4, 8, 16, 32)):
                feats_b[j].append(np.asarray(cache[key][j][:, oy // sc:(oy + ch) // sc, ox // sc:(ox + cw) // sc],
                                             dtype=np.float32))
            yp, v, w, d = labels[key]
            for lst, a in ((ys, yp), (vs, v), (ws, w), (ds, d)):
                lst.append(a[oy:oy + ch, ox:ox + cw])
        feats = [torch.from_numpy(np.stack(f)).to(device) for f in feats_b]
        B, _, h4, w4 = feats[0].shape
        feats = [None, torch.empty((B, 0, h4 * 2, w4 * 2), device=device)] + feats      # [input (unused), 1/2 dummy, ...]
        t = [torch.from_numpy(np.stack(a)[:, None]).to(device) for a in (ys, vs, ws, ds)]
        return feats, t

    def batch_full():
        c = hp["crop"]
        xs, ys, vs, ws, ds = [], [], [], [], []
        for _ in range(hp["batch"]):
            pool = syn_stems if (syn_stems and rng.random() < syn_share) else train_stems
            smp_ = samples[pool[rng.integers(len(pool))]]
            img, y, v = smp_.matched.astype(np.float32), smp_.y.copy(), smp_.valid.copy()
            s = float(np.exp(rng.uniform(np.log(0.8), np.log(1.25))))
            if abs(s - 1) > 0.02:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
                y = (cv2.resize(y, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.float32)
                v = cv2.resize(v, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            if rng.random() < 0.5:
                img, y, v = img[:, ::-1], y[:, ::-1], v[:, ::-1]
            if rng.random() < 0.5:
                img, y, v = img[::-1], y[::-1], v[::-1]
            k = int(rng.integers(4))
            img, y, v = np.rot90(img, k), np.rot90(y, k), np.rot90(v, k)
            img = img * float(rng.uniform(0.8, 1.2)) + 255.0 * float(rng.uniform(-0.08, 0.08))      # contrast, brightness
            img = 255.0 * np.power(np.clip(img / 255.0, 0, 1), float(np.exp(rng.uniform(-0.25, 0.25))))  # gamma
            img, y = np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8)), np.ascontiguousarray(y)
            v = np.ascontiguousarray(v)
            ph, pw = max(0, c - img.shape[0]), max(0, c - img.shape[1])
            if ph or pw:
                img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
                y = np.pad(y, ((0, ph), (0, pw)))
                v = np.pad(v, ((0, ph), (0, pw)))
            oy = int(rng.integers(0, img.shape[0] - c + 1))
            ox = int(rng.integers(0, img.shape[1] - c + 1))
            img, y, v = img[oy:oy + c, ox:ox + c], y[oy:oy + c, ox:ox + c], v[oy:oy + c, ox:ox + c]
            xs.append(smp_.to_input(img.astype(np.float32) / 255.0))
            ys.append(y); vs.append(v); ws.append(smp_.thin_weights(y, v)); ds.append(dist_target(y))
        x = torch.from_numpy(np.stack(xs)).to(device)
        t = [torch.from_numpy(np.stack(a)[:, None].astype(np.float32)).to(device) for a in (ys, vs, ws, ds)]
        return x, t

    # -- train ------------------------------------------------------------------------------------
    set_train()
    best_score, best_state, log = None, None, []
    t0 = time.time()
    for step in range(1, hp["steps"] + 1):
        if mode == "cached":
            feats, (y, v, w, d) = batch_cached()
            out = _decoder_forward(model, feats)
        else:
            x, (y, v, w, d) = batch_full()
            with torch.autocast(device, enabled=use_amp):
                if unfreeze:
                    out = model(x)
                else:
                    with torch.no_grad():
                        feats = model.encoder(x)
                    out = model.segmentation_head(model.decoder(*feats))
        loss = seg_loss(out.float(), y, v, w, d, cldice_w, dist_w)
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        sched.step()
        if step % 10 == 0 or step == hp["steps"]:
            el = time.time() - t0
            _emit(phase="train", step=step, steps=hp["steps"], loss=round(float(loss), 4),
                  eta_s=round(el / step * (hp["steps"] - step)))
        if val_stems and (step % hp["eval_every"] == 0 or step == hp["steps"]):
            sc = score_maps(val_maps(cache_val), val_masks)
            log.append({"step": step, **sc})
            _emit(phase="val", step=step, score=sc)
            if best_score is None or sc["f1"] > best_score["f1"]:
                best_score = sc
                best_state = {k: v_.detach().cpu().clone() for k, v_ in model.state_dict().items()}
    if best_state is None:                                      # no validation split: keep the last weights
        best_state = {k: v_.detach().cpu().clone() for k, v_ in model.state_dict().items()}

    # -- write the run (complete folder appears atomically) ------------------------------------
    real = os.path.join(project_root, "segmentation", "realsoil")
    part = os.path.join(real, "." + run_name + ".partial")
    if os.path.isdir(part):
        shutil.rmtree(part)
    os.makedirs(part)
    base_dir = os.path.join(real, base_run)
    import torch as _t
    _t.save(best_state, os.path.join(part, "seg_best.pt"))
    shutil.copy2(os.path.join(base_dir, "prep_stats.json"), os.path.join(part, "prep_stats.json"))
    tc = dict(train_cfg)
    tc.update({"BASE_RUN": base_run, "WEIGHTS": base_run, "FREEZE_ENCODER": not unfreeze,
               "FINETUNE_SYNTHETIC": len(syn_stems), "FINETUNE_SYNTHETIC_SHARE": syn_share,
               "FINETUNE_MODE": mode, "FINETUNE_STEPS": hp["steps"], "FINETUNE_LR": hp["lr"],
               "FINETUNE_IMAGES": len(stems), "FINETUNE_UNFREEZE_LAST_STAGE": unfreeze,
               "FINETUNE_SOURCE": cfg.get("source", "")})
    with open(os.path.join(part, "train_config.json"), "w") as fh:
        json.dump(tc, fh, indent=1)
    inf = {}
    try:
        with open(os.path.join(base_dir, "inference_settings.json"), "r") as fh:
            inf = json.load(fh)
    except (OSError, ValueError):
        pass
    if best_score is not None:
        inf.update({"THRESH": best_score["thresh"], "THRESH_SOURCE": "finetune_val"})
    with open(os.path.join(part, "inference_settings.json"), "w") as fh:
        json.dump(inf, fh, indent=1)
    with open(os.path.join(part, "val_log.csv"), "w") as fh:
        fh.write("step,f1,precision,recall,dice,thresh\n")
        for r in log:
            fh.write(f"{r['step']},{r['f1']},{r['precision']},{r['recall']},{r['dice']},{r['thresh']}\n")
    if syn_stems:
        from exp_realism.synth_from_annotations import montage
        cv2.imwrite(os.path.join(part, "synthetic_examples.jpg"), cv2.cvtColor(montage(syn_preview), cv2.COLOR_RGB2BGR))
        with open(os.path.join(part, "synthetic_settings.json"), "w") as fh:
            json.dump(syn_info, fh, indent=1)
    report = {"run": run_name, "base_run": base_run, "base": base_score, "new": best_score,
              "n_synthetic": len(syn_stems), "synthetic_fit": syn_info.get("fit"),
              "n_train": len(train_stems), "n_val": len(val_stems), "val_stems": val_stems,
              "train_stems": train_stems, "mode": mode, "device": device, "steps": hp["steps"],
              "minutes": round((time.time() - t_start) / 60.0, 1)}
    with open(os.path.join(part, "finetune_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    final = os.path.join(real, run_name)
    if os.path.isdir(final):
        shutil.rmtree(final)
    os.replace(part, final)
    if tmp_dir:
        import gc
        cache.clear(); labels.clear()
        del cache, labels, cache_val
        gc.collect()                                          # Windows keeps memory-mapped files until released
        shutil.rmtree(tmp_dir, ignore_errors=True)
    _emit(phase="done", run=run_name, report=report)
    return final


def main():
    ap = argparse.ArgumentParser(description="Fine-tune a root model on the annotator's corrected masks")
    ap.add_argument("--job", required=True, help="job folder made by the annotator (config.json, images/, masks/)")
    ap.add_argument("--root", default=str(_REPO_ROOT), help="project root (holds segmentation/realsoil)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--mode", default="auto", choices=["auto", "cached", "full"])
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--unfreeze-last-stage", default="", help="true/false (GPU full mode only)")
    a = ap.parse_args()
    job = a.job if os.path.isabs(a.job) else os.path.join(a.root, a.job)
    unf = None if a.unfreeze_last_stage == "" else a.unfreeze_last_stage.strip().lower() in ("1", "true", "yes", "on")
    try:
        finetune(job, a.root, a.device, a.mode, a.steps or None, unf)
    except Exception as exc:
        _emit(phase="error", msg=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
