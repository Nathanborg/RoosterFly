"""Whole-image prediction for notebook 04 v2: padding, sliding window, TTA (torch).

v1 fed every image to the U-Net at its own size.  That worked only because
every v1 image was 640x480: PRMI sides of 510 and 552 px are not multiples of
the encoder's stride 32 (smp either raises or misaligns the skips), and a
3991x1842 grid_counted photo does not fit in GPU memory in one piece.

* :func:`pad_to_multiple` reflect-pads the bottom/right to a multiple of 32
  and :func:`unpad` crops back -- reflection, not zeros, so a root running off
  the edge does not end in a black wall the network never saw in training;
* :func:`predict_tiled`: inputs up to ``max_full`` px are predicted whole
  (padded); larger ones with a Hann-blended sliding window (tile 768,
  overlap 128).  Tile predictions are weighted by a 2-D Hann window (floored
  so the image border, covered by one tile only, still counts), so tile seams
  vanish;
* :func:`tta_flips`: v1's 4-flip test-time augmentation (maps un-flipped
  before averaging);
* :func:`predict_image`: uint8 RGB -> probability map, at any scale.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ImportError:                                    # pragma: no cover
    cv2 = None

FLIPS = ((), (-1,), (-2,), (-1, -2))


def _ceil_to(n: int, m: int) -> int:
    return int(-(-int(n) // int(m)) * int(m))


def pad_to_multiple(x: torch.Tensor, multiple: int = 32,
                    mode: str = "reflect") -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad NxCxHxW at the bottom/right to multiples of ``multiple``.

    Returns ``(padded, (H, W))`` -- the original size for :func:`unpad`.
    Reflection needs the pad to be smaller than the side; tiny inputs fall
    back to edge replication.
    """
    H, W = int(x.shape[-2]), int(x.shape[-1])
    ph, pw = _ceil_to(H, multiple) - H, _ceil_to(W, multiple) - W
    if ph == 0 and pw == 0:
        return x, (H, W)
    m = mode if (mode != "reflect" or (ph < H and pw < W)) else "replicate"
    return F.pad(x, (0, pw, 0, ph), mode=m), (H, W)


def unpad(y: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    return y[..., :int(hw[0]), :int(hw[1])]


def hann2d(h: int, w: int, floor: float = 1e-3, device=None,
           dtype=torch.float32) -> torch.Tensor:
    """Separable 2-D Hann window (h x w), floored at ``floor``."""
    def one(n):
        if n <= 2:
            return torch.ones(n, dtype=dtype, device=device)
        return torch.hann_window(n, periodic=False, dtype=dtype, device=device)
    return torch.outer(one(int(h)), one(int(w))).clamp_min(floor)


def tile_starts(n: int, tile: int, overlap: int) -> List[int]:
    """Starts of ``tile`` windows over ``n`` px with at least ``overlap`` px overlap."""
    n, tile = int(n), int(tile)
    if n <= tile:
        return [0]
    stride = max(1, tile - int(overlap))
    s = list(range(0, n - tile, stride))
    s.append(n - tile)
    return sorted(set(s))


def predict_tiled(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
                  tile: int = 768, overlap: int = 128, multiple: int = 32, batch: int = 4,
                  max_full: int = 1024) -> torch.Tensor:
    """Probability map of a 1xCxHxW input, returned 1xKxHxW.

    ``fn`` maps an NxCxhxw batch (h, w multiples of ``multiple``) to Nx K x h x w
    probabilities (TTA, autocast and sigmoid inside it).  Inputs whose sides
    are all <= ``max_full`` are padded and predicted whole; larger ones are
    cut into ``tile``-px windows (sides rounded up to ``multiple``) blended
    with :func:`hann2d`.
    """
    if x.shape[0] != 1:
        raise ValueError("predict_tiled takes one image (1xCxHxW)")
    H, W = int(x.shape[-2]), int(x.shape[-1])
    if max(H, W) <= max_full:
        xp, hw = pad_to_multiple(x, multiple)
        return unpad(fn(xp), hw)
    if tile % multiple:
        raise ValueError("tile %d is not a multiple of %d" % (tile, multiple))
    th, tw = min(tile, _ceil_to(H, multiple)), min(tile, _ceil_to(W, multiple))
    Hp, Wp = max(H, th), max(W, tw)
    if (Hp, Wp) != (H, W):
        mode = "reflect" if (Hp - H < H and Wp - W < W) else "replicate"
        x = F.pad(x, (0, Wp - W, 0, Hp - H), mode=mode)
    wgt_dev = torch.device("cpu") if (Hp * Wp > 9_000_000) else x.device
    wgt = hann2d(th, tw, device=wgt_dev)
    acc, wsum = None, torch.zeros((Hp, Wp), dtype=torch.float32, device=wgt_dev)
    coords = [(y0, x0) for y0 in tile_starts(Hp, th, overlap) for x0 in tile_starts(Wp, tw, overlap)]
    for i in range(0, len(coords), max(1, int(batch))):
        cs = coords[i:i + batch]
        xb = torch.cat([x[..., y0:y0 + th, x0:x0 + tw] for y0, x0 in cs], 0)
        pb = fn(xb).float().to(wgt_dev)
        if acc is None:
            acc = torch.zeros((pb.shape[1], Hp, Wp), dtype=torch.float32, device=wgt_dev)
        for k, (y0, x0) in enumerate(cs):
            acc[:, y0:y0 + th, x0:x0 + tw] += pb[k] * wgt
            wsum[y0:y0 + th, x0:x0 + tw] += wgt
    out = (acc / wsum.clamp_min(1e-12))[None]
    return out[..., :H, :W]


def tta_flips(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
              flips: Sequence[Tuple[int, ...]] = FLIPS) -> torch.Tensor:
    """Mean of ``fn`` over flipped copies, each map flipped back first (v1 ``infer``)."""
    acc = None
    for f in flips:
        xx = x.flip(f) if f else x
        o = fn(xx).float()
        o = o.flip(f) if f else o
        acc = o if acc is None else acc + o
    return acc / len(flips)


def prob_fn(model: torch.nn.Module, tta: bool = True, amp_dtype: Optional[torch.dtype] = None,
            device=None) -> Callable[[torch.Tensor], torch.Tensor]:
    """``x -> sigmoid(model(x))`` (mean over 4 flips with ``tta``), under autocast
    with ``amp_dtype`` (bf16/fp16 on CUDA; None = float32)."""
    def raw(z):
        if device is not None and z.device != torch.device(device):
            z = z.to(device)
        dev = z.device.type
        with torch.no_grad():
            if amp_dtype is not None and dev in ("cuda", "cpu"):
                with torch.autocast(dev, dtype=amp_dtype):
                    out = model(z)
            else:
                out = model(z)
        return torch.sigmoid(out.float())

    if tta:
        return lambda z: tta_flips(raw, z)
    return raw


def predict_image(model: torch.nn.Module, rgb_u8: np.ndarray, to_input: Callable[[np.ndarray], np.ndarray],
                  scale: float = 1.0, device="cpu", tta: bool = True,
                  amp_dtype: Optional[torch.dtype] = None, tile: int = 768, overlap: int = 128,
                  batch: int = 4, max_full: int = 1024,
                  out_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Probability map (float32, ``out_hw`` or the input's size) of one image.

    The image is resized by ``scale`` (network px per input px: the source's
    ``f`` times SCALE) -- INTER_AREA shrinking, INTER_CUBIC enlarging, as the
    training crops -- normalised with ``to_input`` (HxWx3 0-1 -> CxHxW),
    predicted with :func:`predict_tiled` (+TTA), and resized back: INTER_AREA
    when the map is larger than ``out_hw`` (v1's SCALE=2 path), else linear.
    """
    model.eval()
    im = np.asarray(rgb_u8)
    H0, W0 = im.shape[:2]
    oh, ow = (H0, W0) if out_hw is None else (int(out_hw[0]), int(out_hw[1]))
    s = float(scale)
    if abs(s - 1.0) > 1e-6:
        size = (max(1, int(round(W0 * s))), max(1, int(round(H0 * s))))
        im = cv2.resize(im, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    x_dev = "cpu" if (im.shape[0] * im.shape[1] > 9_000_000) else device
    x = torch.from_numpy(to_input(im.astype(np.float32) / 255.0))[None].to(x_dev)
    p = predict_tiled(prob_fn(model, tta, amp_dtype, device=device), x, tile, overlap, batch=batch,
                      max_full=max_full)[0, 0].float().cpu().numpy()
    if p.shape != (oh, ow):
        interp = cv2.INTER_AREA if p.shape[0] * p.shape[1] > oh * ow else cv2.INTER_LINEAR
        p = cv2.resize(p, (ow, oh), interpolation=interp)
    return p.astype(np.float32)


__all__ = ["FLIPS", "pad_to_multiple", "unpad", "hann2d", "tile_starts", "predict_tiled",
           "tta_flips", "prob_fn", "predict_image"]
