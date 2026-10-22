"""Real scans in the RootDetector layout, and the matched synthetic domain.

The Ginger_roots corpus ships as ``dataset-{A,B,C}.zip``, each holding three
subsets with a ``splitfile.txt`` per subset:

    standingcrop   inputs/000-x0.tiff, annotations/000.svg
    newgrowth      inputs/000-x0.tiff, inputs/000-x1.tiff, annotations/000.svg
    validation     (same pair layout as newgrowth)

Two facts about this layout decide how it can be used, and both are easy to get
wrong silently.

**Which image an SVG belongs to.**  In a standing-crop record the trace marks the
roots in the one image.  In a pair it marks NEW GROWTH between the two dates, so
the traced roots exist in x1 and mostly do not exist in x0.  Selecting x0 tiles
by that trace selects tiles where the roots are *about to appear* -- i.e. largely
root-free ones -- which is the opposite of what a root-appearance target needs.
:func:`index_zip` therefore pairs a trace with x1 by default, and
:func:`trace_contrast` lets the notebook check the claim on the data instead of
trusting this docstring.  Note also that a new-growth trace does not mark the
OLD roots in x1, so an untraced region of x1 is not root-free.

**Scale.**  B and C are CI-600 scans, 2550x2273 px over 21.6 x 19.2 cm = 11.81
px/mm (the files' DPI tag says 100; it is wrong).  A is a different instrument
-- stitched panoramas 2048-14836 px wide with transparent borders, DPI tags of
72 or 300 -- and its px/mm is not known.  Tiles are only ever cut at a known,
matched scale; A is resampled when its scale is supplied and skipped when not.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import zipfile
from dataclasses import asdict, dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:                                    # pragma: no cover
    cv2 = None

SUBSETS = ("newgrowth", "standingcrop", "validation")

# CI-600 at 300 dpi.  Verified from geometry, not from the (wrong) DPI tag.
CI600_PX_PER_MM = 2551 / 216.0


# --------------------------------------------------------------------------
# indexing the zips
# --------------------------------------------------------------------------
@dataclass
class Record:
    dataset: str          # 'A' | 'B' | 'C'
    subset: str           # 'newgrowth' | 'standingcrop' | 'validation'
    key: str              # '000'
    image: str            # zip member (or local path after extract)
    svg: str              # zip member / local path; '' when no valid trace
    role: str             # 'single' | 'pair-x1' | 'pair-x0'


def dataset_letter(zip_path: str) -> str:
    m = re.search(r"dataset-([A-Za-z0-9]+)\.zip$", os.path.basename(zip_path))
    return m.group(1) if m else os.path.splitext(os.path.basename(zip_path))[0]


def parse_splitfile(text: str) -> List[Tuple[List[str], List[str]]]:
    """``[(inputs, svgs), ...]`` -- one entry per non-empty line."""
    out = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if not parts:
            continue
        inputs = [p for p in parts if p.startswith("inputs/")]
        svgs = [p for p in parts if p.lower().endswith(".svg")]
        if inputs:
            out.append((inputs, svgs))
    return out


def _key(member: str) -> str:
    return os.path.basename(member).split("-")[0].split(".")[0]


def index_zip(zip_path: str, subsets: Sequence[str] = SUBSETS,
              pair_image: str = "x1") -> List[Record]:
    """List the images to use from one zip, each with the trace valid FOR IT.

    For a pair, ``pair_image`` names the image the trace is attached to: 'x1'
    (default) -- the later image, the one a new-growth trace describes -- or
    'x0', if the notebook's x0/x1 check shows the trace fits x0 on your data.
    Only one image per pair is used: the other has no valid trace.
    """
    if pair_image not in ("x1", "x0"):
        raise ValueError("pair_image must be 'x1' or 'x0'")
    letter = dataset_letter(zip_path)
    recs: List[Record] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for sub in subsets:
            base = "dataset-%s-%s/" % (letter, sub)
            split = base + "splitfile.txt"
            if split not in names:
                continue
            for inputs, svgs in parse_splitfile(zf.read(split).decode("utf-8", "ignore")):
                inputs = [base + p for p in inputs if base + p in names]
                svg = next((base + s for s in svgs if base + s in names), "")
                if not inputs:
                    continue
                if len(inputs) == 1:
                    recs.append(Record(letter, sub, _key(inputs[0]), inputs[0],
                                       svg, "single"))
                    continue
                x1 = next((p for p in inputs if "-x1." in p), inputs[-1])
                x0 = next((p for p in inputs if "-x0." in p), inputs[0])
                img = x1 if pair_image == "x1" else x0
                recs.append(Record(letter, sub, _key(img), img, svg, "pair-" + pair_image))
    return recs


def extract_records(zip_path: str, records: Sequence[Record], dest: str,
                    verbose: bool = True) -> List[Record]:
    """Extract only the members the records need; return records with local paths."""
    os.makedirs(dest, exist_ok=True)
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        info = {i.filename: i for i in zf.infolist()}
        for r in records:
            loc = {}
            for field in ("image", "svg"):
                m = getattr(r, field)
                if not m:
                    loc[field] = ""
                    continue
                p = os.path.join(dest, m)
                if not (os.path.exists(p) and os.path.getsize(p) == info[m].file_size):
                    zf.extract(m, dest)
                loc[field] = p
            out.append(Record(r.dataset, r.subset, r.key, loc["image"], loc["svg"], r.role))
    if verbose:
        print("  %s: %d images extracted -> %s" % (os.path.basename(zip_path), len(out), dest))
    return out


# --------------------------------------------------------------------------
# images and traces
# --------------------------------------------------------------------------
def load_rgb(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """``(rgb uint8 HxWx3, valid bool HxW)``.  RGBA transparency -> invalid."""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise IOError("cannot read %s" % path)
    if im.dtype != np.uint8:
        im = (im / (np.iinfo(im.dtype).max / 255.0)).astype(np.uint8)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    valid = np.ones(im.shape[:2], bool)
    if im.shape[2] == 4:
        valid = im[:, :, 3] > 0
        im = im[:, :, :3]
    return np.ascontiguousarray(im[:, :, ::-1]), valid


_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def svg_polylines(svg_text: str) -> Tuple[List[np.ndarray], Tuple[float, float]]:
    """Centrelines from a RootDetector SVG, in SVG user units.

    Every path in this corpus is absolute ``M`` + ``C`` with each cubic's control
    points equal to its end points, so the coordinate pairs ARE the polyline.
    Anything else (relative commands, H/V, arcs) is refused rather than guessed.
    """
    m = re.search(r'viewBox="([^"]+)"', svg_text)
    if m:
        vb = [float(v) for v in m.group(1).replace(",", " ").split()]
        size = (vb[2], vb[3])
    else:
        w = re.search(r'\bwidth="([\d.]+)', svg_text)
        h = re.search(r'\bheight="([\d.]+)', svg_text)
        size = (float(w.group(1)), float(h.group(1)))
    polys = []
    for d in re.findall(r'\sd="([^"]+)"', svg_text):
        cmds = set(re.findall(r"[A-Za-z]", d.replace("e", "").replace("E", "")))
        if not cmds <= {"M", "C", "L"}:
            raise ValueError("unsupported SVG path commands %s" % sorted(cmds - {"M", "C", "L"}))
        v = np.array([float(x) for x in _NUM.findall(d)], np.float64)
        if v.size >= 4 and v.size % 2 == 0:
            p = v.reshape(-1, 2)
            keep = np.ones(len(p), bool)
            keep[1:] = np.any(np.abs(np.diff(p, axis=0)) > 1e-9, axis=1)
            p = p[keep]
            if len(p) >= 2:
                polys.append(p)
    return polys, size


def load_trace(svg_path: str, image_shape) -> List[np.ndarray]:
    """Polylines in IMAGE pixel coordinates (scaled by viewBox -> image size)."""
    with open(svg_path, encoding="utf-8", errors="ignore") as fh:
        polys, (vw, vh) = svg_polylines(fh.read())
    h, w = image_shape[:2]
    s = np.array([w / vw, h / vh])
    return [p * s for p in polys]


def rasterize_trace(polys: Sequence[np.ndarray], shape, width_px: int = 1,
                    offset=(0.0, 0.0)) -> np.ndarray:
    """Centreline mask, drawn at ``width_px`` (a corridor, not a root width)."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    ox, oy = offset
    for p in polys:
        q = np.round((p - [ox, oy]) * 16).astype(np.int32)       # 4-bit subpixel
        cv2.polylines(m, [q], False, 1, max(1, int(width_px)), cv2.LINE_8, 4)
    return m.astype(bool)


def segments(polys: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """All polyline segments as ``(midpoints Nx2, lengths N)``."""
    if not polys:
        return np.zeros((0, 2)), np.zeros(0)
    a = np.concatenate([p[:-1] for p in polys])
    b = np.concatenate([p[1:] for p in polys])
    return (a + b) / 2.0, np.linalg.norm(b - a, axis=1)


def trace_length_in_box(mid: np.ndarray, seglen: np.ndarray, x: float, y: float,
                        w: float, h: float) -> float:
    """Traced length (px) whose segment midpoints fall inside the box."""
    if seglen.size == 0:
        return 0.0
    inside = ((mid[:, 0] >= x) & (mid[:, 0] < x + w)
              & (mid[:, 1] >= y) & (mid[:, 1] < y + h))
    return float(seglen[inside].sum())


def trace_contrast(rgb: np.ndarray, polys: Sequence[np.ndarray],
                   width_px: int = 3, ring_px: Tuple[int, int] = (9, 25)) -> float:
    """Mean grey level on the trace minus a ring around it.

    The check that tells you which image of a pair a trace describes: a root that
    is really there differs from its surroundings; one that has not grown yet does
    not.  Sign depends on root polarity (bright or dark roots), so compare
    MAGNITUDES between x0 and x1, not signs.
    """
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    on = rasterize_trace(polys, g.shape, width_px)
    if not on.any():
        return float("nan")
    k_in = np.ones((ring_px[0], ring_px[0]), np.uint8)
    k_out = np.ones((ring_px[1], ring_px[1]), np.uint8)
    u8 = on.astype(np.uint8)
    ring = (cv2.dilate(u8, k_out) > 0) & ~(cv2.dilate(u8, k_in) > 0)
    if not ring.any():
        return float("nan")
    return float(g[on].mean() - g[ring].mean())


def root_width_px(rgb: np.ndarray, polys: Sequence[np.ndarray], step_px: float = 15.0,
                  half_len_px: int = 40, max_samples: int = 3000,
                  seed: int = 0) -> Dict[str, float]:
    """Median apparent root width, from intensity profiles across the trace.

    For estimating an UNKNOWN px/mm by comparison with a known-scale dataset:
    ``px_mm_unknown ~= px_mm_known * width_unknown / width_known``.  That assumes
    the two sets hold roots of similar true diameter -- a strong assumption across
    sites and species -- so treat the result as a first guess to be checked by
    eye, never as a calibration.

    The profile measurement itself is :func:`_fwhm_widths`, shared with the
    synthetic-render check, the PRMI ``fwhm_f`` scale cross-check and the
    width-binned Ginger recall, so all of them use one estimator.
    """
    rng = np.random.default_rng(seed)
    pts, nrm = [], []
    for p in polys:
        seg = np.diff(p, axis=0)
        L = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0], np.cumsum(L)])
        if cum[-1] < 2 * step_px:
            continue
        for s in np.arange(step_px, cum[-1] - step_px, step_px):
            i = min(np.searchsorted(cum, s) - 1, len(seg) - 1)
            if L[i] < 1e-6:
                continue
            t = seg[i] / L[i]
            pts.append(p[i] + t * (s - cum[i]))
            nrm.append([-t[1], t[0]])
    if not pts:
        return {"median_px": float("nan"), "n": 0}
    pts, nrm = np.array(pts), np.array(nrm)
    if len(pts) > max_samples:
        sel = rng.choice(len(pts), max_samples, replace=False)
        pts, nrm = pts[sel], nrm[sel]

    w = _fwhm_widths(rgb, pts, nrm, half_len_px=half_len_px)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return {"median_px": float("nan"), "n": 0}
    return {"median_px": float(np.median(w)), "p25_px": float(np.percentile(w, 25)),
            "p75_px": float(np.percentile(w, 75)), "n": int(len(w))}


def _fwhm_widths(img: np.ndarray, pts: np.ndarray, nrm: np.ndarray,
                 half_len_px: int = 40, min_peak: float = 6.0,
                 blur_sigma: float = 0.8) -> np.ndarray:
    """Apparent root width (px) at each sample: FWHM of the grey profile across it.

    ``img`` is the RAW image (RGB uint8, or grey); the 0.8-px blur that steadies
    the half-maximum crossing is applied here, so every caller measures the same
    way.  ``pts`` are ``(x, y)`` points on a root centreline and ``nrm`` unit
    normals to it, both ``Nx2``.  Returns an array ALIGNED with ``pts``: NaN
    where no root is visible (|peak| < ``min_peak`` grey levels), where the
    centre is not above half maximum, or where the half-maximum region runs off
    the profile.  Polarity-free: bright and dark roots both give a width.

    Annotation-free on purpose: it reads widths from pixels, so it can compare
    datasets whose masks were drawn differently (WinRHIZO boxes vs superpixels)
    and give widths to traces, which have none.
    """
    from scipy.ndimage import map_coordinates

    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    nrm = np.asarray(nrm, np.float64).reshape(-1, 2)
    out = np.full(len(pts), np.nan)
    if len(pts) == 0:
        return out
    a = np.asarray(img)
    g = (cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a).astype(np.float32)
    if blur_sigma and blur_sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), blur_sigma)

    t = np.arange(-half_len_px, half_len_px + 1, 0.5)
    xs = pts[:, None, 0] + nrm[:, None, 0] * t[None]
    ys = pts[:, None, 1] + nrm[:, None, 1] * t[None]
    prof = map_coordinates(g, [ys.ravel(), xs.ravel()], order=1,
                           mode="nearest").reshape(xs.shape)
    edge = np.abs(t) > 0.7 * half_len_px
    bg = np.median(prof[:, edge], axis=1)
    centre = np.abs(t) <= 1.5
    peak = prof[:, centre].mean(axis=1) - bg
    c = len(t) // 2
    for k in range(len(prof)):
        if abs(peak[k]) < min_peak:                 # no visible root here
            continue
        above = np.sign(peak[k]) * (prof[k] - bg[k]) >= 0.5 * abs(peak[k])
        if not above[c]:
            continue
        lo = c
        while lo > 0 and above[lo - 1]:
            lo -= 1
        hi = c
        while hi < len(t) - 1 and above[hi + 1]:
            hi += 1
        if lo == 0 or hi == len(t) - 1:             # ran off the profile
            continue
        out[k] = t[hi] - t[lo] + 0.5
    return out


# --------------------------------------------------------------------------
# tiling
# --------------------------------------------------------------------------
def tile_offsets(n: int, t: int) -> List[int]:
    """Evenly spaced offsets covering ``[0, n)`` with tiles of ``t`` (min overlap)."""
    if n < t:
        return []
    k = int(math.ceil(n / t))
    if k == 1:
        return [0]
    stride = (n - t) / (k - 1)
    return [int(round(i * stride)) for i in range(k)]


def resample_to(rgb: np.ndarray, valid: np.ndarray, polys: List[np.ndarray],
                src_px_per_mm: float, dst_px_per_mm: float,
                allow_upsample: bool = False):
    """Bring an image and its trace to the target scale.  Downsample only."""
    f = dst_px_per_mm / src_px_per_mm
    if abs(f - 1.0) <= 0.02:
        return rgb, valid, polys
    if f > 1.02 and not allow_upsample:
        raise ValueError(
            "image is %.2fx COARSER than the target %.2f px/mm; upsampling real "
            "texture teaches the translator to blur. Re-render the synthetic set "
            "at %.2f px/mm instead, or pass allow_upsample=True." %
            (f, dst_px_per_mm, src_px_per_mm))
    h, w = rgb.shape[:2]
    size = (max(1, int(round(w * f))), max(1, int(round(h * f))))
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
    rgb = cv2.resize(rgb, size, interpolation=interp)
    valid = cv2.resize(valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    return rgb, valid, [p * f for p in polys]


@dataclass
class TileScreen:
    """What makes a tile unusable as a root-appearance target."""
    min_trace_mm: float = 5.0          # traced root length inside the tile
    min_valid_frac: float = 0.98       # RGBA transparency / stitching borders
    min_grey_std: float = 6.0          # featureless: tape, foil, blank wall
    max_clip_frac: float = 0.25        # blown highlights or black borders


def screen_tile(tile: np.ndarray, valid: np.ndarray, trace_mm: float,
                s: TileScreen) -> str:
    """'' if kept, else the reason it was dropped (written to the CSV)."""
    if valid.mean() < s.min_valid_frac:
        return "invalid_pixels"
    g = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    if float(g.std()) < s.min_grey_std:
        return "featureless"
    if float(((g >= 250) | (g <= 4)).mean()) > s.max_clip_frac:
        return "clipped"
    if trace_mm < s.min_trace_mm:
        return "too_little_root"
    return ""


def crop_bright_rows(rgb: np.ndarray, valid: np.ndarray, polys: List[np.ndarray],
                     max_rows: int = 8, min_grey: float = 200.0):
    """Strip scanner border rows (mean grey >= ``min_grey``) from top and bottom.

    98 of 100 CI-600 scans end in two bright rows (grey ~233).  They are not a
    reason to drop the bottom tiles, but they should not be in them either.
    Returns ``(rgb, valid, polys, top_rows_removed)``; polylines are shifted.
    """
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).mean(axis=1)
    H = rgb.shape[0]
    top = 0
    while top < min(max_rows, H) and g[top] >= min_grey:
        top += 1
    bot = 0
    while bot < min(max_rows, H - top) and g[H - 1 - bot] >= min_grey:
        bot += 1
    if top == 0 and bot == 0:
        return rgb, valid, polys, 0
    sl = slice(top, H - bot)
    return rgb[sl], valid[sl], [p - np.array([0.0, top]) for p in polys], top


def export_real_tiles(records: Sequence[Record], out_dir: str,
                      px_per_mm: Dict[str, Optional[float]],
                      target_px_per_mm: float = CI600_PX_PER_MM,
                      tile_wh: Tuple[int, int] = (640, 480),
                      screen: Optional[TileScreen] = None,
                      trace_width_mm: float = 0.4,
                      allow_upsample: bool = False,
                      tape_screen: bool = True,
                      crop_bright: bool = True,
                      dedupe: bool = True,
                      min_scan_trace_contrast: Optional[float] = None,
                      verbose: bool = True,
                      keep_root_poor: bool = False) -> List[dict]:
    """Cut root-bearing real tiles at the synthetic frame size and scale.

    Writes ``rgb/<id>.png`` (the CUT target) and ``trace/<id>.png`` (the traced
    centreline corridor, for overlays and the visibility reference) for kept
    tiles, and a ``tiles.csv`` listing EVERY candidate tile with the reason it
    was dropped -- so the screen can be audited rather than trusted.

    Beyond the basic screen (``screen_tile``):

    * ``tape_screen``: tape, foil, plastic, bare tube wall, seam strings and
      stitching borders (``mrstyle.visibility.screen_reason``).  Run 1 trained
      on ~140 such tiles, and CUT learned to paint tape into synthetic soil.
    * ``crop_bright``: scanner border rows are removed before tiling.
    * ``dedupe``: identical tiles after the first are dropped ('duplicate') --
      the same scan sits in two subsets.  Records are processed in order, so
      list standing-crop records first: their traces mark ALL roots, so their
      copy is the one worth keeping.
    * ``min_scan_trace_contrast``: if the trace does not stand out from its scan
      by at least this many grey levels, the trace does not describe that image
      (most of dataset C: 8 of 10 standing-crop scans score ~1 grey), so every
      tile of the scan is dropped as 'trace_mismatch' -- its trace length, and
      hence its selection, would be meaningless.
    * ``keep_root_poor``: ALSO write ``rgb/`` + ``trace/`` for tiles that passed
      every screen but carry too little trace ('too_little_root').  A
      standing-crop one with (almost) no trace is a root-free real tile, which
      the v3 CUT needs so its target domain stops implying 'real always has
      roots' (the painting cue).  The CSV is byte-for-byte what the default
      writes -- those rows stay 'too_little_root' with an empty hash -- so
      every existing selection is unchanged.  The ones actually written are
      listed in a sidecar ``root_poor.csv`` (id, dataset, subset, key, role,
      trace_mm, hash; atomic, only with this option): tiles.csv cannot say
      which root-poor rows have a file (those removed by dedupe have none) and
      gives them an empty hash, which a hash-based leakage guard needs.  They
      are deduplicated among themselves and against earlier kept tiles (a later
      identical kept tile still gets its own copy).
    """
    screen = screen or TileScreen()
    tw, th = tile_wh
    for sub in ("rgb", "trace"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    rows: List[dict] = []
    skipped: Dict[str, int] = {}
    seen = set()
    seen_poor = set()            # kept apart: adding these to `seen` would turn a
    n_poor = 0                   # later kept copy into 'duplicate' (a CSV change)
    poor_rows: List[dict] = []   # the root-poor tiles written (root_poor.csv)
    if tape_screen or dedupe or keep_root_poor:
        from mrstyle import visibility as V

    for r in records:
        src = px_per_mm.get(r.dataset)
        if src is None:
            skipped[r.dataset] = skipped.get(r.dataset, 0) + 1
            continue
        if not r.svg:
            skipped["no_trace"] = skipped.get("no_trace", 0) + 1
            continue
        rgb, valid = load_rgb(r.image)
        polys = load_trace(r.svg, rgb.shape)
        if crop_bright:
            rgb, valid, polys, _ = crop_bright_rows(rgb, valid, polys)
        scan_tc = float("nan")
        mismatch = False
        if min_scan_trace_contrast is not None:
            scan_tc = trace_contrast(rgb, polys)
            # a scan with an (empty) trace and no roots is root-free, not mismatched
            mismatch = bool(polys) and not (np.isfinite(scan_tc)
                                            and abs(scan_tc) >= min_scan_trace_contrast)
        rgb, valid, polys = resample_to(rgb, valid, polys, src, target_px_per_mm,
                                        allow_upsample=allow_upsample)
        mid, seglen = segments(polys)
        H, W = rgb.shape[:2]
        wpx = max(1, int(round(trace_width_mm * target_px_per_mm)))
        for y in tile_offsets(H, th):
            for x in tile_offsets(W, tw):
                tile = np.ascontiguousarray(rgb[y:y + th, x:x + tw])
                vt = valid[y:y + th, x:x + tw]
                mm = trace_length_in_box(mid, seglen, x, y, tw, th) / target_px_per_mm
                why = "trace_mismatch" if mismatch else screen_tile(tile, vt, mm, screen)
                sc = None
                hsh = ""
                if why in ("", "too_little_root") and tape_screen:
                    # tape overrides 'too little root': a tape tile with a short
                    # trace must not count as a usable near-empty real tile
                    sc = V.tape_score(tile)
                    why = V.screen_reason(sc) or why
                if not why and dedupe:
                    hsh = V.tile_hash(tile)
                    if hsh in seen:
                        why = "duplicate"
                    seen.add(hsh)
                tid = "%s_%s_%s_%s_y%04d_x%04d" % (r.dataset, r.subset[:2], r.key,
                                                   r.role.split("-")[0], y, x)
                rows.append({"id": tid, "dataset": r.dataset, "subset": r.subset,
                             "key": r.key, "role": r.role,
                             "source": os.path.basename(r.image), "x": x, "y": y,
                             "trace_mm": round(mm, 2), "dropped": why,
                             "tape_frac": round(sc["tape_frac"], 5) if sc else "",
                             "band_len": round(sc["band_len"], 3) if sc else "",
                             "black_frac": round(sc["black_frac"], 5) if sc else "",
                             "scan_trace_contrast": (round(scan_tc, 2)
                                                     if np.isfinite(scan_tc) else ""),
                             "hash": hsh})
                if why:
                    if not (keep_root_poor and why == "too_little_root"):
                        continue
                    hp = V.tile_hash(tile)
                    if dedupe:
                        if hp in seen or hp in seen_poor:
                            continue
                        seen_poor.add(hp)
                    n_poor += 1
                    poor_rows.append({"id": tid, "dataset": r.dataset, "subset": r.subset,
                                      "key": r.key, "role": r.role,
                                      "trace_mm": round(mm, 2), "hash": hp})
                cv2.imwrite(os.path.join(out_dir, "rgb", tid + ".png"), tile[:, :, ::-1])
                tr = rasterize_trace(polys, (th, tw), wpx, offset=(x, y))
                cv2.imwrite(os.path.join(out_dir, "trace", tid + ".png"),
                            tr.astype(np.uint8) * 255)

    with open(os.path.join(out_dir, "tiles.csv"), "w", newline="") as fh:
        cols = ["id", "dataset", "subset", "key", "role", "source", "x", "y",
                "trace_mm", "dropped", "tape_frac", "band_len", "black_frac",
                "scan_trace_contrast", "hash"]
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    if keep_root_poor:
        # written after every PNG, then renamed: a listed tile is always on disk
        side = os.path.join(out_dir, "root_poor.csv")
        tmp = side + ".tmp"
        with open(tmp, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["id", "dataset", "subset", "key", "role",
                                                "trace_mm", "hash"])
            wr.writeheader()
            wr.writerows(poor_rows)
        os.replace(tmp, side)

    if verbose:
        kept = [r for r in rows if not r["dropped"]]
        print("real tiles: %d candidates -> %d kept (%dx%d at %.2f px/mm = %.1f x %.1f mm)"
              % (len(rows), len(kept), tw, th, target_px_per_mm,
                 tw / target_px_per_mm, th / target_px_per_mm))
        why: Dict[str, int] = {}
        for r in rows:
            if r["dropped"]:
                why[r["dropped"]] = why.get(r["dropped"], 0) + 1
        for k, v in sorted(why.items(), key=lambda kv: -kv[1]):
            print("   dropped %-16s %d" % (k, v))
        by: Dict[str, int] = {}
        for r in kept:
            by[r["dataset"]] = by.get(r["dataset"], 0) + 1
        print("   kept per dataset:", dict(sorted(by.items())))
        if keep_root_poor:
            print("   root-poor tiles also written (deduplicated): %d" % n_poor)
        for k, v in skipped.items():
            print("   SKIPPED %d images: %s" % (v, "no px/mm for dataset " + k
                                               if k != "no_trace" else "no valid trace"))
    return rows


# --------------------------------------------------------------------------
# synthetic domain at the real scale
# --------------------------------------------------------------------------
SYN_COLS = ["id", "system", "frame_in_system", "visible_length_mm",
            "mean_diameter_mm", "projected_area_mm2", "planar_rld_mm_per_mm2",
            "branching_density_per_mm", "n_fragments", "is_empty",
            "px_per_mm_u", "px_per_mm_v", "mask_threshold", "soil_depth_cm",
            "config_id"]

# v3 traits.csv columns (the agreed contract), written after SYN_COLS by a v3
# config only -- a v2 config's part files are byte-for-byte what they were.
#   species        the stand ('cotton' or 'cotton+switchgrass')
#   species_major  the species owning most of the frame's labelled pixels
#   w2e_*          2 x EDT at the mask skeleton (mrsim.traits.skeleton_widths_px)
#   tw_len_*       visible mm per TRUE-width bin in px (diam/), never w2e bins
#   fine_len_frac  share of skeleton length with w2e <= THIN_W2E (4)
#   mask_len_ratio skeleton length / analytic projected length
SYN_COLS_V3 = ["species", "species_major", "diam_mult_axis", "diam_mult_lat",
               "diam_root_sd", "tortuosity_mult", "mask_mode",
               "w2e_p10", "w2e_p50", "w2e_p90",
               "tw_len_0_2", "tw_len_2_3", "tw_len_3_5", "tw_len_5_8", "tw_len_8p",
               "fine_len_frac", "mean_true_diameter_mm", "root_px_frac", "mask_len_ratio",
               "psf_sigma_px", "contrast_target_n", "soil_sigma", "length_fragments_2d_mm",
               "root_palette_mix", "dark_root_share", "root_luma_med", "root_chroma_med"]
# per-frame palette record (mrsim.scene.palette_stats; blank without a palette)
_PALETTE_KEYS = ("root_palette_mix", "dark_root_share", "root_luma_med", "root_chroma_med")
_V3_TRAIT_KEYS = ("w2e_p10", "w2e_p50", "w2e_p90", "tw_len_0_2", "tw_len_2_3",
                  "tw_len_3_5", "tw_len_5_8", "tw_len_8p", "fine_len_frac",
                  "mean_true_diameter_mm", "root_px_frac")
# The render gate (GATE_V3): gate_v3.csv = id + these + on_mode; merge_parts
# merges them into traits.csv.  Same functions and definitions as the CUT
# final audit (visibility_audit_many with ON_MODE_V3 and THIN_W2E).
GATE_COLS_V3 = ["contrast_n_syn", "contrast_n_thin_syn", "syn_invisible", "syn_invisible_thin"]
GATE_FILE = "gate_v3.csv"
# _systems/sys_NNN.json -> systems.csv
SYSTEM_COLS = ["system", "species", "species_major", "plants", "n_plants", "days",
               "spacing_cm", "row_offset_cm", "diam_mult_axis", "diam_mult_lat",
               "diam_root_sd", "tortuosity_mult", "n_roots", "n_culled", "cap_hit",
               "grow_s", "render_s", "n_frames"]


@dataclass
class SynthConfig:
    out_dir: str = "/content/ginger/synth"
    preset: str = "cid_ci600_640"
    species: str = "generic_fibrous"
    n_systems: int = 60
    frames_per_system: int = 100
    days: float = 60.0
    n_plants: int = 8
    spacing_cm: float = 7.0
    row_offset_cm: float = 0.6
    axial_range_cm: Tuple[float, float] = (0.0, 60.0)
    soil_cover: float = 0.14
    seed: int = 20260923
    # Appearance switches.  The defaults reproduce earlier renders exactly.
    # artifacts=False: no vignette, hotspot, specular glare, condensation rims,
    # scratches, chromatic shift or JPEG, and no pale stone blobs or dark organic
    # flecks in the soil -- only PSF + sensor noise, as on a contact scanner.
    # The rng draws are unchanged up to the optics, so frame 0 of every system
    # keeps its geometry; later frames differ (optics draws fewer numbers).
    artifacts: bool = True
    psf_max_px: Optional[float] = None       # cap on the PSF sigma (artifacts=False only)
    # > 0: rescale each frame's root colour ramp so that EVERY age is at least
    # this far (luma, 0-1) from the soil colour.  Colour only: masks unchanged.
    min_root_contrast: float = 0.0
    root_polarity: str = "bright"            # 'bright' | 'dark' | 'auto'
    # floor on fine-root opacity; used only with min_root_contrast > 0 (or, in
    # v3, contrast_target_n > 0).  Unlike the colour fix this changes alpha, so
    # calibrated masks move slightly (geometric masks do not).
    min_root_opacity: Optional[float] = None
    # Density switches (see mrstyle.density).  0.55 biases frames toward the
    # bottom of the tube, as a camera system is usually aimed; a CI-600 scan is a
    # 360-degree unroll, so its tiles sample every azimuth equally (use 0.0).
    below_tube_bias: float = 0.55
    # The per-plant root cap binds at ~60 days for the default stand: once it is
    # hit no new laterals form, so deep axes stay unbranched.
    max_roots_per_plant: int = 8000
    config_id: str = ""                      # written to every CSV row
    # ---- v3 (trailing; every default reproduces the renders above) --------
    # Species: '' = the single `species` above.  'switchgrass:.2,cotton:.1,...'
    # draws one species per system (stream [seed, sys, 1], see draw_system);
    # with probability mixed_stand_p a second species alternates plant by plant.
    species_mix: str = ""
    mixed_stand_p: float = 0.0
    # {species: {n_plants, days, spacing_cm, row_offset_cm}} from the per-species
    # density calibration (density.best_per_species); missing keys -> the fields above
    species_geom: Optional[Dict[str, Dict[str, float]]] = None
    # per-system tortuosity multiplier ~ tortuosity_mu * LogN(0, tortuosity_sd)
    tortuosity_mu: float = 1.0
    tortuosity_sd: float = 0.0
    # relevance culling margin (cm); None = grow every lateral (v2)
    cull_margin_cm: Optional[float] = None
    # mrsim.diameter.DiameterSpec fields (dict); None = preset widths (v2)
    diameter: Optional[Dict[str, float]] = None
    # bridge visibility dips shorter than this (mm) so roots are not dashed
    vis_bridge_mm: float = 0.0
    # 'geometric' = label everything drawn (min width mask_min_width_px)
    mask_mode: str = "calibrated"
    mask_min_width_px: float = 1.0
    # > 0: per-segment width/PSF-aware contrast floor (contrast_n target)
    contrast_target_n: float = 0.0
    # diam/<id>.png uint16 um (> 0 exactly on the mask); order/<id>.png uint8
    save_diam: bool = False
    save_order: bool = False
    # Root colours PER ROOT from a palette of real root colours ('' = v2's one
    # ramp per frame).  e.g. mrsim.scene.DEFAULT_ROOT_PALETTE.  Stream
    # [seed, sys, 3]; masks never depend on it.  With a contrast floor, roots
    # flagged dark (share dark_root_frac) may sit DARKER than a soil they
    # cannot be brighter than without bleaching.
    root_palette: str = ""
    dark_root_frac: float = 0.15
    # realism of each drawn root (palette mode only; 0 / 1 = flat capsules as
    # before): cylindrical shading, along-root luma texture, palette saturation,
    # soft blurred fade where a root leaves the focal plane (mrsim.scene
    # RootAppearance.shading/texture/chroma_scale/vis_fade)
    root_shading: float = 0.0
    root_texture: float = 0.0
    palette_chroma: float = 1.0
    vis_fade: float = 0.0
    # thick roots not beaded by soil crumbs; no end-on dots; soil film dimming
    # (RootAppearance.cover_relief_px / min_planarity / soil_film)
    cover_relief_px: float = 0.0
    min_planarity: float = 0.0
    soil_film: float = 0.0
    # real-image banks (mrstyle.realbanks): '' = procedural.  soil_bank: folder
    # with soil/*.png real root-free soil frames used as backgrounds (stream
    # [seed, sys, frame, 5]); root_bank: folder with roots.csv + roots/*.png
    # real root strips mapped along roots >= 3 px (root_bank_mix = share of
    # the strip's own colour).  Labels never depend on either.
    soil_bank: str = ""
    root_bank: str = ""
    root_bank_mix: float = 0.7
    # stand variety (stream [seed, sys, 7]; None / 0 = off):
    #   days_range (lo, hi)      stand age drawn per system (young = sparse frames)
    #   n_plants_range (lo, hi)  plants per stand drawn per system (integers, inclusive)
    #   thick_p, thick_mult      share of systems whose AXES are multiplied by
    #                            U(thick_mult) -- very thick main roots (raise the
    #                            diameter max_mm with it)
    # root_keep_range (lo, hi): per FRAME, each root is drawn (and labelled) with
    #   probability k ~ U(lo, hi) (stream [seed, sys, frame, 6]) -- sparse frames
    days_range: Optional[Tuple[float, float]] = None
    n_plants_range: Optional[Tuple[int, int]] = None
    thick_p: float = 0.0
    thick_mult: Tuple[float, float] = (2.0, 3.5)
    root_keep_range: Optional[Tuple[float, float]] = None
    # bright_frame_p: share of frames drawn in the BRIGHT look (stream [seed, sys,
    #   frame, 8]): no soil film, contrast target bright_target_n, opacity floor
    #   0.8 -- the white, high-contrast roots of many PRMI cameras next to the
    #   subtle default look
    bright_frame_p: float = 0.0
    bright_target_n: float = 2.0
    # root_keep_log: draw the per-frame keep share LOG-uniformly in root_keep_range (frames from
    #   nearly empty to full: ~1 mm to ~2000 mm of root)
    # fake_soil_p: share of frames on the procedural soil even with a soil_bank (stream
    #   [seed, sys, frame, 9])
    # soil_jitter / soil_hue / soil_sat: bank-soil colour variation (realbanks.soil_background)
    # root_hairs / root_hair_p: fine root hairs, image only (scene.RootAppearance)
    root_keep_log: bool = False
    fake_soil_p: float = 0.0
    soil_jitter: float = 0.06
    soil_hue: float = 0.0
    soil_sat: float = 0.0
    root_hairs: float = 0.0
    root_hair_p: float = 0.4
    # roots wider than tex_max_mag x their texture strip use the strip's cross profile only
    tex_max_mag: float = 0.0


_V3_FIELDS = ("species_mix", "mixed_stand_p", "species_geom", "tortuosity_mu",
              "tortuosity_sd", "cull_margin_cm", "diameter", "vis_bridge_mm", "mask_mode",
              "mask_min_width_px", "contrast_target_n", "save_diam", "save_order",
              "root_palette", "dark_root_frac", "root_shading", "root_texture",
              "palette_chroma", "vis_fade", "cover_relief_px", "min_planarity", "soil_film",
              "soil_bank", "root_bank", "root_bank_mix", "days_range", "n_plants_range",
              "thick_p", "thick_mult", "root_keep_range", "bright_frame_p", "bright_target_n",
              "root_keep_log", "fake_soil_p", "soil_jitter", "soil_hue", "soil_sat", "root_hairs",
              "root_hair_p", "tex_max_mag")


def is_v3(cfg: SynthConfig) -> bool:
    """True when any v3 knob is off its default (then traits get SYN_COLS_V3,
    _systems/ is written and the width traits are computed)."""
    d = SynthConfig()
    return any(getattr(cfg, f) != getattr(d, f) for f in _V3_FIELDS)


def syn_hash(cfg: SynthConfig) -> str:
    """SYN_HASH: short hash of the config WITHOUT out_dir, n_systems and config_id.

    Systems are independent of how many there are (stream ``seed + 7919*i``),
    so N can be raised later and only the new systems rendered; the folder is
    the same wherever it lives; and config_id is a label derived from this
    hash ('v3_<SYN_HASH>').  frames_per_system stays in: frame-spec draws
    precede the render.
    """
    from mrstyle.realdata import short_hash
    d = asdict(cfg)
    for k in ("out_dir", "n_systems", "config_id"):
        d.pop(k, None)
    # realism knobs added later: left out at their defaults so earlier
    # configs keep their hash (and their renders on Drive stay valid)
    for k, v0 in (("root_shading", 0.0), ("root_texture", 0.0), ("palette_chroma", 1.0),
                  ("vis_fade", 0.0), ("cover_relief_px", 0.0), ("min_planarity", 0.0),
                  ("soil_film", 0.0), ("soil_bank", ""), ("root_bank", ""),
                  ("root_bank_mix", 0.7), ("days_range", None), ("n_plants_range", None),
                  ("thick_p", 0.0), ("thick_mult", (2.0, 3.5)), ("thick_mult", [2.0, 3.5]),
                  ("root_keep_range", None), ("bright_frame_p", 0.0), ("bright_target_n", 2.0),
                  ("root_keep_log", False), ("fake_soil_p", 0.0), ("soil_jitter", 0.06),
                  ("soil_hue", 0.0), ("soil_sat", 0.0), ("root_hairs", 0.0), ("root_hair_p", 0.4),
                  ("tex_max_mag", 0.0)):
        if d.get(k) == v0:
            d.pop(k, None)
    return short_hash(d)


def _pick(weights, u: float) -> int:
    w = np.asarray(weights, np.float64)
    c = np.cumsum(w) / w.sum()
    return int(min(np.searchsorted(c, u, side="right"), len(w) - 1))


def draw_system(cfg: SynthConfig, sys_i: int) -> Dict[str, object]:
    """What a system is made of: species, plants, geometry, multipliers.

    Drawn from its OWN stream ``default_rng([seed, sys_i, 1])``, so the main
    stream ``seed + 7919*sys_i`` (growth, framing, soil, optics) is untouched
    and a v2 config draws nothing that matters.  The same number of values is
    drawn whatever the outcome (species, mixed?, second species, tortuosity,
    two diameter normals), so systems stay comparable across settings.

    Returns: system, species (the stand, 'a' or 'a+b'), species_major (the
    first species), species2, plants (one name per plant; with a mixed stand
    the second species takes every other plant), n_plants / days / spacing_cm
    / row_offset_cm (``species_geom`` of the first species, else cfg),
    tortuosity_mult, diam_mult_axis, diam_mult_lat, diam_root_sd.
    """
    from mrsim.diameter import DiameterSpec, system_multipliers
    from mrsim.params import parse_mix
    rng = np.random.default_rng([int(cfg.seed), int(sys_i), 1])
    u = rng.random(3)
    z_tort = float(rng.standard_normal())
    z_sys = rng.standard_normal(2)
    mix = parse_mix(cfg.species_mix)
    sp, sp2 = cfg.species, ""
    if mix:
        names = list(mix)
        sp = names[_pick([mix[n] for n in names], float(u[0]))]
        if len(names) > 1 and float(u[1]) < float(cfg.mixed_stand_p):
            rest = [n for n in names if n != sp]
            sp2 = rest[_pick([mix[n] for n in rest], float(u[2]))]
    geom = dict(n_plants=cfg.n_plants, days=cfg.days, spacing_cm=cfg.spacing_cm,
                row_offset_cm=cfg.row_offset_cm)
    sg = (cfg.species_geom or {}).get(sp)
    if sg:
        for k in geom:
            if k in sg and sg[k] is not None:
                geom[k] = int(sg[k]) if k == "n_plants" else float(sg[k])
    thick = 1.0
    if cfg.days_range is not None or cfg.n_plants_range is not None or cfg.thick_p > 0:
        # stand variety on its own stream: three draws whatever is switched on
        vr = np.random.default_rng([int(cfg.seed), int(sys_i), 7]).random(4)
        if cfg.days_range is not None:
            lo, hi = (float(v) for v in cfg.days_range)
            geom["days"] = lo + (hi - lo) * float(vr[0])
        if cfg.n_plants_range is not None:
            lo, hi = (int(v) for v in cfg.n_plants_range)
            geom["n_plants"] = lo + min(int(vr[1] * (hi - lo + 1)), hi - lo)
        if float(vr[2]) < float(cfg.thick_p):
            lo, hi = (float(v) for v in cfg.thick_mult)
            thick = lo + (hi - lo) * float(vr[3])
    n = int(geom["n_plants"])
    plants = [sp2 if (sp2 and i % 2 == 1) else sp for i in range(n)]
    stand: List[str] = []
    for p in plants:
        if p not in stand:
            stand.append(p)
    spec = DiameterSpec.from_any(cfg.diameter)
    m_axis, m_lat = system_multipliers(spec, z=z_sys) if spec is not None else (1.0, 1.0)
    out = dict(system=int(sys_i), species="+".join(stand), species_major=sp,
               species2=sp2 if len(stand) > 1 else "", plants=plants,
               tortuosity_mult=float(cfg.tortuosity_mu * math.exp(cfg.tortuosity_sd * z_tort)),
               diam_mult_axis=float(m_axis * thick), diam_mult_lat=float(m_lat),
               diam_root_sd=float(spec.root_sd) if spec is not None else 0.0)
    out.update(geom)
    return out


def _build_stand(cfg: SynthConfig, sys_i: int, rng, tube, cam):
    """Grow, project and (v3) re-width one system: ``(system, frame, proj, draw)``.

    Shared by the render and the contrast pilot, through the one growth path
    (``mrsim.scene.grow_and_project``) the density and diameter calibrations
    also use.  For a v2 config this is exactly the old grow_stand +
    project_system on ``rng``.  Diameters use their own stream
    ``[seed, sys_i, 2]`` (per-root, then along-root noise), so a diameter spec
    never moves geometry, framing or soil.
    """
    from mrsim.diameter import DiameterSpec, apply_diameter, root_z
    from mrsim.params import get_species, scale_tortuosity
    from mrsim.scene import grow_and_project
    d = draw_system(cfg, sys_i)
    tort = d["tortuosity_mult"]
    cache: Dict[str, object] = {}

    def preset(name):
        if name not in cache:
            p = get_species(name)
            if tort != 1.0:
                p = scale_tortuosity(p, tort)
            cache[name] = p
        return cache[name]

    plant = preset(d["plants"][0])
    plants = [preset(n) for n in d["plants"]] if len(set(d["plants"])) > 1 else None
    system, frame, proj = grow_and_project(
        plant, tube, cam, d["days"], rng, n_plants=d["n_plants"],
        spacing_cm=d["spacing_cm"], row_offset_cm=tube.r_out + d["row_offset_cm"],
        max_roots_per_plant=cfg.max_roots_per_plant, plants=plants,
        cull_margin_cm=cfg.cull_margin_cm, axial_range_cm=tuple(cfg.axial_range_cm),
        vis_bridge_mm=cfg.vis_bridge_mm)
    if cfg.diameter is not None:
        spec = DiameterSpec.from_any(cfg.diameter)
        drng = np.random.default_rng([int(cfg.seed), int(sys_i), 2])
        proj = apply_diameter(proj, spec, d["diam_mult_axis"], d["diam_mult_lat"],
                              root_z(len(system.roots), drng), rng_along=drng)
    d["palette"] = draw_palette(cfg, sys_i, system)
    return system, frame, proj, d


def draw_palette(cfg: SynthConfig, sys_i: int, system):
    """Per-root colours of system ``sys_i`` (None without ``cfg.root_palette``).

    Own stream ``[seed, sys_i, 3]``: a palette never moves geometry, framing,
    soil, optics or the masks, and a root keeps its colour in every frame.
    """
    if not cfg.root_palette:
        return None
    from mrsim.scene import sample_root_palette
    n = 1 + max([int(r.index) for r in system.roots] + [0])
    return sample_root_palette(n, cfg.root_palette,
                               np.random.default_rng([int(cfg.seed), int(sys_i), 3]),
                               dark_root_frac=cfg.dark_root_frac)


_LUMA3 = np.array([0.299, 0.587, 0.114])


def _polarity_sign(polarity: str, appear, style) -> float:
    if polarity == "dark":
        return -1.0
    if polarity == "auto":
        return 1.0 if float(np.asarray(appear.young_rgb) @ _LUMA3) >= \
            float(style.base_rgb() @ _LUMA3) else -1.0
    return 1.0


def _frame_appearance(cfg: SynthConfig, appear, style, floor: float, target: float,
                      palette=None):
    """The per-frame root appearance: v2's global luma floor (``floor > 0``),
    then v3's per-segment contrast target (``target > 0``, with the opacity
    floor ``min_root_opacity``).  With a per-root ``palette`` the floor is a
    hue-keeping luma floor per segment instead (``min_delta``), applied
    together with the target.  Draws nothing."""
    if palette is not None:
        appear = replace(appear, palette=palette, min_delta=float(max(floor, 0.0)),
                         shading=float(cfg.root_shading), texture=float(cfg.root_texture),
                         chroma_scale=float(cfg.palette_chroma), vis_fade=float(cfg.vis_fade),
                         cover_relief_px=float(cfg.cover_relief_px),
                         min_planarity=float(cfg.min_planarity), soil_film=float(cfg.soil_film),
                         tex_mix=float(cfg.root_bank_mix))
        if cfg.root_bank:
            from mrstyle.realbanks import load_root_bank
            appear = replace(appear, root_bank=load_root_bank(cfg.root_bank))
        if cfg.min_root_opacity is not None and (floor > 0 or target > 0):
            appear = replace(appear, min_opacity=max(appear.min_opacity,
                                                     float(cfg.min_root_opacity)))
        return appear.with_frame_context(
            target_n=float(max(target, 0.0)),
            polarity=-1.0 if cfg.root_polarity == "dark" else 1.0)
    if floor > 0:
        appear = appear.with_min_contrast(style.base_rgb(), floor, cfg.root_polarity,
                                          cfg.min_root_opacity)
    if target > 0:
        if cfg.min_root_opacity is not None and not floor > 0:
            appear = replace(appear, min_opacity=max(appear.min_opacity,
                                                     float(cfg.min_root_opacity)))
        appear = appear.with_frame_context(
            target_n=float(target), polarity=_polarity_sign(cfg.root_polarity, appear, style))
    return appear


def _render_kw(cfg: SynthConfig) -> Dict[str, object]:
    if not is_v3(cfg):
        return {}
    return dict(mask_mode=cfg.mask_mode, mask_min_width_px=cfg.mask_min_width_px,
                width_traits=True)


def _out_subdirs(cfg: SynthConfig) -> List[str]:
    subs = ["rgb", "alpha", "mask", "_done", "_parts"]
    if is_v3(cfg):
        subs.append("_systems")
    if cfg.save_diam:
        subs.append("diam")
    if cfg.save_order:
        subs.append("order")
    return subs


def diam_um_map(fr) -> np.ndarray:
    """``diam/`` content: uint16 micrometres, >= 1 on EVERY mask pixel, 0 off it.

    From ``FrameResult.diameter_label_mm`` (max-cover segment, nearest-filled;
    a crossing takes the root covering the pixel most, ties the nearer root),
    or for a calibrated mask from the coverage-weighted map filled the same
    way.  Rounded, clipped to [1, 65535] um on the mask.
    """
    from mrsim.raster import diameter_on_mask
    d = fr.diameter_label_mm
    if d is None:
        d = diameter_on_mask(fr.diameter_mm, fr.mask)
    um = np.clip(np.rint(np.asarray(d, np.float64) * 1000.0), 1, 65535)
    return np.where(fr.mask, um, 0).astype(np.uint16)


def order_map(fr, system) -> np.ndarray:
    """``order/`` content: uint8 root order + 1 on the mask (0 off it), from the
    pixel's instance (the root contributing most); mask pixels without one
    take the nearest labelled pixel's order."""
    from mrsim.raster import diameter_on_mask
    inst = np.asarray(fr.instance)
    orders = np.asarray([r.order for r in system.roots], np.float32)
    o = np.zeros(inst.shape, np.float32)
    ok = inst >= 0
    o[ok] = orders[inst[ok]] + 1.0
    o = diameter_on_mask(o, fr.mask)
    return np.clip(np.rint(o), 0, 255).astype(np.uint8)


def _species_major(fr, system, plants: Sequence[str]) -> str:
    """Species owning most labelled pixels (instance -> root -> plant)."""
    rp = getattr(system, "root_plant", None)
    inst = np.asarray(fr.instance)
    sel = np.asarray(fr.mask, bool) & (inst >= 0)
    if rp is None or not sel.any() or len(set(plants)) < 2:
        return str(plants[0])
    cnt = np.bincount(np.asarray(rp)[inst[sel]], minlength=len(plants))
    tot: Dict[str, int] = {}
    for i, name in enumerate(plants):
        tot[name] = tot.get(name, 0) + int(cnt[i]) if i < len(cnt) else tot.get(name, 0)
    return max(tot, key=lambda k: (tot[k], -list(tot).index(k)))


def _v3_row(cfg: SynthConfig, fr, draw: Dict[str, object], system) -> Dict[str, object]:
    o, m = fr.traits_observable, fr.meta
    len2d = float(o.get("length_fragments_2d_mm") or 0.0)
    row = {"species": draw["species"],
           "species_major": _species_major(fr, system, draw["plants"]),
           "diam_mult_axis": draw["diam_mult_axis"], "diam_mult_lat": draw["diam_mult_lat"],
           "diam_root_sd": draw["diam_root_sd"], "tortuosity_mult": draw["tortuosity_mult"],
           "mask_mode": cfg.mask_mode,
           "mask_len_ratio": (float(o["visible_length_mm"]) / len2d if len2d > 0
                              else float("nan")),
           "psf_sigma_px": m.get("psf_sigma_px"), "contrast_target_n": cfg.contrast_target_n,
           "soil_sigma": m.get("soil_sigma", float("nan")), "length_fragments_2d_mm": len2d}
    for c in _V3_TRAIT_KEYS:
        row[c] = o.get(c)
    for c in _PALETTE_KEYS:
        row[c] = m.get(c, "")
    return row


def _write_csv_atomic(path: str, fieldnames: Sequence[str], rows) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(fieldnames), restval="")
        wr.writeheader()
        wr.writerows(rows)
    os.replace(tmp, path)


def _render_system(args) -> Tuple[int, int, float]:
    """Worker: grow one stand, render its frames, write PNGs + a CSV part.

    Write order (resume-safe): every PNG, then ``_parts/sys_NNN.csv``, then
    (v3) ``_systems/sys_NNN.json``, then the ``_done`` flag LAST -- a system
    without its flag is rendered again from scratch.  Creates its own
    output folders (v3 adds diam/, order/, _systems/).
    """
    import time
    cfg_d, sys_i = args
    # one thread per worker: the pool already uses every core, and OpenCV's own
    # threads in N workers oversubscribe the CPU (slower, not faster)
    cv2.setNumThreads(1)
    cfg = SynthConfig(**cfg_d)
    done_flag = os.path.join(cfg.out_dir, "_done", "sys_%03d" % sys_i)
    if os.path.exists(done_flag):
        return sys_i, 0, 0.0

    from mrsim import optics as O
    from mrsim import soil as S
    from mrsim.scene import RootAppearance, render_frame, sample_frame_specs
    from mrsim.tube import VisibilityModel, get_preset

    v3 = is_v3(cfg)
    for sub in _out_subdirs(cfg):
        os.makedirs(os.path.join(cfg.out_dir, sub), exist_ok=True)
    t0 = time.time()
    tube, cam = get_preset(cfg.preset)
    rng = np.random.default_rng(cfg.seed + 7919 * sys_i)
    system, frame, proj, draw = _build_stand(cfg, sys_i, rng, tube, cam)
    grow_s = time.time() - t0
    specs = sample_frame_specs(cfg.frames_per_system, cam, tube, rng,
                               below_tube_bias=cfg.below_tube_bias,
                               axial_range_cm=tuple(cfg.axial_range_cm))
    vis = VisibilityModel()
    rkw = _render_kw(cfg)
    rows = []
    style = tex = None
    for j, spec in enumerate(specs):
        if j % 25 == 0:
            style = S.SoilStyle.sample(rng, cam.px_per_mm_u)
            tex = S.TextureCache.build((cam.height_px, cam.width_px), rng, style,
                                       cover=cfg.soil_cover)
            if not cfg.artifacts:
                # built with the same draws (so the occlusion field is the same);
                # the inclusions are just not drawn into the soil
                tex.stones = None
                tex.organics = None
        bright = (cfg.bright_frame_p > 0 and float(np.random.default_rng(
            [int(cfg.seed), int(sys_i), int(j), 8]).random()) < cfg.bright_frame_p)
        appear = _frame_appearance(cfg, RootAppearance.sample(rng), style,
                                   cfg.min_root_contrast,
                                   cfg.bright_target_n if bright else cfg.contrast_target_n,
                                   palette=draw.get("palette"))
        if bright:
            appear = replace(appear, soil_film=0.0, min_opacity=max(appear.min_opacity, 0.8))
        ostyle = None                       # None: render_frame samples it (as before)
        if not cfg.artifacts:
            # drawn at the point render_frame would draw it, then stripped
            ostyle = O.OpticsStyle.sample(rng, cam.px_per_mm_u).without_artefacts(
                cfg.psf_max_px)
        if cfg.root_keep_range is not None:
            lo, hi = (float(v) for v in cfg.root_keep_range)
            k_rng = np.random.default_rng([int(cfg.seed), int(sys_i), int(j), 6])
            u = float(k_rng.random())
            keep = (math.exp(math.log(max(lo, 1e-4)) + u * (math.log(hi) - math.log(max(lo, 1e-4))))
                    if cfg.root_keep_log else lo + (hi - lo) * u)
            appear = replace(appear, root_keep=keep, keep_seed=int(k_rng.integers(1 << 30)))
        if cfg.root_hairs > 0:
            appear = replace(appear, root_hairs=float(cfg.root_hairs), root_hair_p=float(cfg.root_hair_p))
        if cfg.tex_max_mag > 0:
            appear = replace(appear, tex_max_mag=float(cfg.tex_max_mag))
        bg = None
        fake = cfg.fake_soil_p > 0 and float(np.random.default_rng(
            [int(cfg.seed), int(sys_i), int(j), 9]).random()) < cfg.fake_soil_p
        if cfg.soil_bank and not fake:
            from mrstyle.realbanks import soil_background as _bank_bg
            bg = _bank_bg(cfg.soil_bank, np.random.default_rng([cfg.seed, sys_i, j, 5]),
                          wh=(cam.width_px, cam.height_px), jitter=cfg.soil_jitter,
                          hue=cfg.soil_hue, sat=cfg.soil_sat)
        fr = render_frame(system, tube, cam, spec, rng, vis=vis,
                          appear=appear, soil_style=style, optics_style=ostyle,
                          proj=proj, frame=frame, tex=tex, cover=cfg.soil_cover,
                          background=bg, **rkw)
        fid = "s%03d_%04d" % (sys_i, j)
        cv2.imwrite(os.path.join(cfg.out_dir, "rgb", fid + ".png"),
                    (np.clip(fr.rgb, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])
        cv2.imwrite(os.path.join(cfg.out_dir, "alpha", fid + ".png"),
                    (np.clip(fr.alpha, 0, 1) * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(cfg.out_dir, "mask", fid + ".png"),
                    fr.mask.astype(np.uint8) * 255)
        if cfg.save_diam:
            cv2.imwrite(os.path.join(cfg.out_dir, "diam", fid + ".png"), diam_um_map(fr))
        if cfg.save_order:
            cv2.imwrite(os.path.join(cfg.out_dir, "order", fid + ".png"), order_map(fr, system))
        o = fr.traits_observable
        row = {"id": fid, "system": sys_i, "frame_in_system": j}
        for c in SYN_COLS[3:10]:
            row[c] = o.get(c)
        for c in SYN_COLS[10:14]:
            row[c] = fr.meta.get(c)
        row["config_id"] = cfg.config_id
        if v3:
            row.update(_v3_row(cfg, fr, draw, system))
        rows.append(row)

    part = os.path.join(cfg.out_dir, "_parts", "sys_%03d.csv" % sys_i)
    _write_csv_atomic(part, SYN_COLS + (SYN_COLS_V3 if v3 else []), rows)
    if v3:
        n_pl = int(draw["n_plants"])
        rp = getattr(system, "root_plant", None)
        per_plant = (np.bincount(np.asarray(rp), minlength=n_pl) if rp is not None
                     else np.array([len(system.roots)]))
        info = {k: draw[k] for k in ("system", "species", "species_major", "n_plants", "days",
                                     "spacing_cm", "row_offset_cm", "diam_mult_axis",
                                     "diam_mult_lat", "diam_root_sd", "tortuosity_mult")}
        info.update(plants=";".join(draw["plants"]), n_roots=len(system.roots),
                    n_culled=int(getattr(system, "n_culled", 0)),
                    cap_hit=bool(np.any(per_plant >= cfg.max_roots_per_plant)),
                    grow_s=round(grow_s, 2), render_s=round(time.time() - t0 - grow_s, 2),
                    n_frames=len(rows))
        js = os.path.join(cfg.out_dir, "_systems", "sys_%03d.json" % sys_i)
        with open(js + ".tmp", "w") as fh:
            json.dump(info, fh, indent=1, sort_keys=True)
        os.replace(js + ".tmp", js)
    open(done_flag, "w").close()
    return sys_i, len(rows), time.time() - t0


def render_synthetic(cfg: SynthConfig, workers: Optional[int] = None,
                     systems: Optional[Iterable[int]] = None,
                     verbose: bool = True) -> str:
    """Render the synthetic domain, one process per stand, resumable.

    Systems already marked done are skipped, so a disconnected session resumes
    where it stopped.  Returns the merged ``traits.csv`` path.  Workers are
    SPAWNED (``density.spawn_pool``), never forked: safe after CUDA or OpenCV
    threads were started in the kernel.
    """
    import time
    from concurrent.futures import as_completed
    from mrstyle.density import spawn_pool

    for sub in _out_subdirs(cfg):
        os.makedirs(os.path.join(cfg.out_dir, sub), exist_ok=True)
    todo = list(systems) if systems is not None else list(range(cfg.n_systems))
    todo = [i for i in todo
            if not os.path.exists(os.path.join(cfg.out_dir, "_done", "sys_%03d" % i))]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if verbose:
        print("synthetic: %d systems to render (%d frames each) on %d workers"
              % (len(todo), cfg.frames_per_system, workers))
    t0 = time.time()
    n_done = 0
    if todo:
        with spawn_pool(min(workers, len(todo))) as ex:
            futs = [ex.submit(_render_system, (asdict(cfg), i)) for i in todo]
            for f in as_completed(futs):
                i, n, dt = f.result()
                n_done += 1
                if verbose:
                    el = time.time() - t0
                    eta = el / n_done * (len(todo) - n_done)
                    print("  system %3d: %d frames in %.1f min | %d/%d | ETA %.0f min"
                          % (i, n, dt / 60, n_done, len(todo), eta / 60), flush=True)
    return merge_parts(cfg.out_dir)


def _csv_header(path: str) -> List[str]:
    with open(path, newline="") as fh:
        return next(csv.reader(fh), [])


def read_gate_v3(out_dir: str) -> Dict[str, Dict[str, str]]:
    """``{id: {gate column: value}}`` from ``out_dir/gate_v3.csv`` ({} if none)."""
    p = os.path.join(out_dir, GATE_FILE)
    if not os.path.exists(p):
        return {}
    with open(p, newline="") as fh:
        return {r["id"]: {k: v for k, v in r.items() if k != "id" and k is not None}
                for r in csv.DictReader(fh)}


def merge_parts(out_dir: str) -> str:
    """``traits.csv`` from every ``_parts/sys_*.csv`` (atomic).

    Columns are the UNION of the parts: SYN_COLS first, then the SYN_COLS_V3
    present, then any others in first-seen order; a part lacking a column
    leaves it blank.  So v2 parts and v3 parts merge (v2 parts alone give the
    old file byte for byte).  If GATE_V3 has written ``gate_v3.csv`` its
    columns are merged in by id (blank for frames it has not seen yet), and
    ``_systems/*.json`` become ``systems.csv``.
    """
    parts = sorted(glob.glob(os.path.join(out_dir, "_parts", "sys_*.csv")))
    path = os.path.join(out_dir, "traits.csv")
    extra: List[str] = []
    for p in parts:
        for c in _csv_header(p):
            if c not in SYN_COLS and c not in extra:
                extra.append(c)
    cols = list(SYN_COLS) + [c for c in SYN_COLS_V3 if c in extra] + \
        [c for c in extra if c not in SYN_COLS_V3]
    gate = read_gate_v3(out_dir)
    if gate:
        gcols: List[str] = []
        for v in gate.values():
            for c in v:
                if c not in gcols:
                    gcols.append(c)
        cols += [c for c in gcols if c not in cols]
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as fo:
        wr = csv.DictWriter(fo, fieldnames=cols, restval="")
        wr.writeheader()
        for p in parts:
            with open(p, newline="") as fi:
                for r in csv.DictReader(fi):
                    r.pop(None, None)
                    if gate:
                        r.update(gate.get(r.get("id"), {}))
                    wr.writerow(r)
    os.replace(tmp, path)
    merge_systems(out_dir)
    return path


def merge_systems(out_dir: str) -> Optional[str]:
    """``systems.csv`` from ``_systems/sys_*.json`` (atomic); None if there are none."""
    files = sorted(glob.glob(os.path.join(out_dir, "_systems", "sys_*.json")))
    if not files:
        return None
    rows = []
    for f in files:
        with open(f) as fh:
            rows.append(json.load(fh))
    cols = list(SYSTEM_COLS) + sorted({k for r in rows for k in r} - set(SYSTEM_COLS))
    path = os.path.join(out_dir, "systems.csv")
    _write_csv_atomic(path, cols, rows)
    return path


# --------------------------------------------------------------------------
# incremental Drive mirrors: one part zip per render batch
# --------------------------------------------------------------------------
PART_SUBDIRS = ("rgb", "alpha", "mask", "diam", "order")


def zip_systems(out_dir: str, systems: Iterable[int], zip_path: str,
                subdirs: Sequence[str] = PART_SUBDIRS) -> List[int]:
    """One STORED zip holding ONLY ``systems`` (the finished ones): their
    ``<sub>/sNNN_*.png``, ``_parts/sys_NNN.csv``, ``_systems/sys_NNN.json`` and,
    LAST, their ``_done/sys_NNN`` flags.  Written to a temp file, then renamed.

    v2 re-zipped every frame after every batch -- O(n^2) bytes, ~30 GB of
    copying at 10k frames.  A part per batch is written once.  Because the
    done flags are the last members, a restore interrupted halfway leaves
    those systems unflagged, and they are simply rendered again.  Returns the
    systems actually zipped.
    """
    done = [int(s) for s in systems
            if os.path.exists(os.path.join(out_dir, "_done", "sys_%03d" % int(s)))]
    prefixes = tuple("s%03d_" % s for s in done)
    tmp = zip_path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for sub in subdirs:
            d = os.path.join(out_dir, sub)
            if not os.path.isdir(d) or not prefixes:
                continue
            for n in sorted(os.listdir(d)):
                if n.endswith(".png") and n.startswith(prefixes):
                    zf.write(os.path.join(d, n), sub + "/" + n)
        for s in done:
            for rel in ("_parts/sys_%03d.csv" % s, "_systems/sys_%03d.json" % s):
                p = os.path.join(out_dir, *rel.split("/"))
                if os.path.isfile(p):
                    zf.write(p, rel)
        for s in done:
            zf.write(os.path.join(out_dir, "_done", "sys_%03d" % s), "_done/sys_%03d" % s)
    os.replace(tmp, zip_path)
    return done


def restore_parts(parts, dest: str, verbose: bool = True) -> List[int]:
    """Unpack part zips (a glob pattern or a list of paths) into ``dest``.

    A part whose done flags are all present locally is skipped.  Each part's
    done flags are extracted after everything else in it, so an interrupted
    restore never marks a system done without its frames.  Returns the
    systems restored.
    """
    paths = sorted(glob.glob(parts)) if isinstance(parts, str) else list(parts)
    out: List[int] = []
    for p in paths:
        with zipfile.ZipFile(p) as zf:
            names = zf.namelist()
            flags = [n for n in names if n.startswith("_done/")]
            if flags and all(os.path.exists(os.path.join(dest, *n.split("/"))) for n in flags):
                continue
            for n in names:
                if not n.startswith("_done/"):
                    zf.extract(n, dest)
            for n in flags:
                zf.extract(n, dest)
        sys_ids = [int(n.split("_")[-1]) for n in flags]
        out += sys_ids
        if verbose:
            print("  restored %d systems from %s" % (len(sys_ids), os.path.basename(p)))
    return sorted(out)


# --------------------------------------------------------------------------
# contrast pilot and the render gate
# --------------------------------------------------------------------------
def _pilot_stand(args):
    """Worker: one stand, the SAME frames rendered at every (floor, target)."""
    import copy
    cfg_d, stand, floors, n_frames = args[:4]
    targets, width_bins, on_mode = (args[4:7] if len(args) > 4
                                    else ((0.0,), None, "band"))
    cfg = SynthConfig(**cfg_d)

    from mrsim import optics as O
    from mrsim import soil as S
    from mrsim.scene import RootAppearance, render_frame, sample_frame_specs
    from mrsim.tube import VisibilityModel, get_preset
    from mrstyle import visibility as V

    tube, cam = get_preset(cfg.preset)
    rng = np.random.default_rng(cfg.seed + 7919 * stand)
    system, frame, proj, draw = _build_stand(cfg, stand, rng, tube, cam)
    specs = sample_frame_specs(n_frames, cam, tube, rng,
                               below_tube_bias=cfg.below_tube_bias,
                               axial_range_cm=tuple(cfg.axial_range_cm))
    vis = VisibilityModel()
    rkw = _render_kw(cfg)
    plain = (tuple(float(t) for t in targets) == (0.0,) and width_bins is None
             and on_mode == "band" and not is_v3(cfg))
    rows = []
    for j, spec in enumerate(specs):
        # Each frame gets its own soil (visibility depends on the soil colour far
        # more than on the stand), then one generator state replayed for every
        # floor: the floor changes colours only, so geometry, soil and optics
        # stay identical across floors.
        frng = np.random.default_rng(cfg.seed * 31 + 1000003 * stand + j)
        style = S.SoilStyle.sample(frng, cam.px_per_mm_u)
        tex = S.TextureCache.build((cam.height_px, cam.width_px), frng, style,
                                   cover=cfg.soil_cover)
        if not cfg.artifacts:
            tex.stones = None
            tex.organics = None
        state = copy.deepcopy(frng.bit_generator.state)
        for floor in floors:
            for target in targets:
                fr_rng = np.random.default_rng()
                fr_rng.bit_generator.state = copy.deepcopy(state)
                appear = _frame_appearance(cfg, RootAppearance.sample(fr_rng), style,
                                           float(floor), float(target),
                                           palette=draw.get("palette"))
                ostyle = None
                if not cfg.artifacts:
                    ostyle = O.OpticsStyle.sample(fr_rng, cam.px_per_mm_u).without_artefacts(
                        cfg.psf_max_px)
                fr = render_frame(system, tube, cam, spec, fr_rng, vis=vis, appear=appear,
                                  soil_style=style, optics_style=ostyle, proj=proj,
                                  frame=frame, tex=tex, cover=cfg.soil_cover, **rkw)
                rgb8 = (np.clip(fr.rgb, 0, 1) * 255).astype(np.uint8)
                a8 = (np.clip(fr.alpha, 0, 1) * 255).astype(np.uint8)
                empty = not bool(fr.mask.any())
                v = ({"contrast": float("nan"), "contrast_n": float("nan")} if empty else
                     V.frame_visibility(rgb8, V.regions_from_mask(fr.mask, a8,
                                                                  on_mode=on_mode)))
                row = {"stand": stand, "floor": float(floor), "frame": j,
                       "contrast": v["contrast"], "contrast_n": v["contrast_n"],
                       "visible_length_mm": fr.traits_observable.get("visible_length_mm"),
                       "is_empty": empty, "mask_px": int(fr.mask.sum())}
                if not plain:
                    row["target"] = float(target)
                    row["species"] = draw["species"]
                    if width_bins is not None:
                        if empty:
                            for lab in V.width_bin_labels(width_bins):
                                row["cn_" + lab] = float("nan")
                                row["non_" + lab] = 0
                            row["contrast_n_thin"] = float("nan")
                            row["n_on_thin"] = 0
                        else:
                            vb = V.visibility_by_width(rgb8, fr.mask, None, a8, bins=width_bins)
                            for k, val in vb.items():
                                if k.startswith(("cn_", "non_")) or k in (
                                        "contrast_n_thin", "n_on_thin"):
                                    row[k] = val
                rows.append(row)
    return rows


def pilot_visibility(cfg: SynthConfig, floors: Sequence[float], n_frames: int = 30,
                     stands: Sequence[int] = (0, 1, 2), workers: Optional[int] = None,
                     verbose: bool = True, targets: Sequence[float] = (0.0,),
                     width_bins: Optional[Sequence[float]] = None,
                     on_mode: str = "band"):
    """Render a few stands at several root-soil contrast floors and measure visibility.

    The floor is set BEFORE the full render, on evidence: run 1 left 59% of
    synthetic frames with roots less visible than 90% of real ones, and a floor
    tuned on a different metric than the audit's can still leave many frames
    failing the gate.  Every floor renders the same frames, so the table
    compares floors, not random draws.  Workers are spawned, never forked.

    v3 (defaults reproduce the v2 table exactly): ``targets`` -- per-segment
    contrast targets (``contrast_target_n``) crossed with the floors, the same
    frames for every pair; ``width_bins`` (e.g. REPORT_BINS_W2E) adds per-bin
    ``cn_*`` / ``non_*`` and the thin-root ``contrast_n_thin`` from
    ``visibility.visibility_by_width`` with widths from the mask, as the gate
    and the CUT audit measure them; ``on_mode`` (ON_MODE_V3 for v3) for the
    frame-level contrast.  A non-default call adds ``target`` and ``species``
    columns.  Pick the pair with :func:`pick_floor_target`.
    """
    import pandas as pd
    from concurrent.futures import as_completed
    from mrstyle.density import spawn_pool
    workers = workers or max(1, min(len(stands), (os.cpu_count() or 2) - 1))
    extra = (tuple(float(t) for t in targets),
             None if width_bins is None else tuple(float(b) for b in width_bins), on_mode)
    rows = []
    with spawn_pool(workers) as ex:
        futs = [ex.submit(_pilot_stand, (asdict(cfg), s, list(floors), n_frames) + extra)
                for s in stands]
        for n, f in enumerate(as_completed(futs), 1):
            rows += f.result()
            if verbose:
                print("  pilot stand %d/%d done" % (n, len(stands)), flush=True)
    return pd.DataFrame(rows)


def pick_floor_target(pil, T_ABS: float, T_MED: float, max_invisible: float = 0.10,
                      min_on: int = 30, min_frames: int = 5,
                      min_mm: Optional[float] = None):
    """``(floor, target, table)`` from a width-binned pilot table.

    Rule: within every width bin, among frames whose bin has >= ``min_on``
    labelled on-pixels (bins with fewer than ``min_frames`` such frames are
    not judged), the share with ``cn < T_ABS`` must be <= ``max_invisible``
    -- thin roots included, which is the point.  Among the pairs that pass,
    the one whose frame-level median contrast_n is closest to the real median
    ``T_MED`` wins (thick roots no longer over-contrasty); ties go to the
    smaller floor, then the smaller target.  If none passes, the pair with the
    smallest worst-bin share is returned (``table.ok`` is all False).
    """
    import pandas as pd
    p = pil.copy()
    if "target" not in p:
        p["target"] = 0.0
    labs = [c[3:] for c in p.columns if c.startswith("cn_")]
    rows = []
    for (fl, tg), g in p.groupby(["floor", "target"], sort=True):
        r = {"floor": float(fl), "target": float(tg), "n": int(len(g))}
        worst = 0.0
        for lab in labs:
            on = pd.to_numeric(g["non_" + lab], errors="coerce").fillna(0) >= min_on
            r["n_" + lab] = int(on.sum())
            if on.sum() < min_frames:
                r["inv_" + lab] = float("nan")
                continue
            cn = pd.to_numeric(g.loc[on, "cn_" + lab], errors="coerce")
            inv = float((~(cn >= T_ABS)).mean())
            r["inv_" + lab] = inv
            worst = max(worst, inv)
        fr = g[~g.is_empty.astype(bool)]
        if min_mm is not None:
            fr = fr[pd.to_numeric(fr.visible_length_mm, errors="coerce").fillna(0) >= min_mm]
        cnf = pd.to_numeric(fr.contrast_n, errors="coerce")
        r["median_cn"] = float(cnf.median()) if len(cnf) else float("nan")
        r["inv_frame"] = float((~(cnf >= T_ABS)).mean()) if len(cnf) else float("nan")
        r["worst_bin"] = worst
        r["ok"] = worst <= max_invisible
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab["dist_T_MED"] = (tab.median_cn - T_MED).abs()
    ok = tab[tab.ok]
    if len(ok):
        best = ok.sort_values(["dist_T_MED", "floor", "target"]).iloc[0]
    else:
        best = tab.sort_values(["worst_bin", "dist_T_MED", "floor", "target"]).iloc[0]
    return float(best.floor), float(best.target), tab


def gate_flags(df, T_ABS: float):
    """(Re)compute the gate flags from the stored contrasts, exactly as
    ``visibility_audit_many`` defines them:

    * ``syn_invisible`` = not empty and NOT (contrast_n_syn finite and >= T_ABS)
      -- an unmeasurable non-empty frame counts as invisible;
    * ``syn_invisible_thin`` = contrast_n_thin_syn finite and < T_ABS.

    Recomputed on every write, so a changed T_ABS can never leave stale flags.
    """
    d = df.copy()
    cn = d["contrast_n_syn"].astype(float)
    ct = d["contrast_n_thin_syn"].astype(float)
    empty = d["is_empty"].astype(str).str.lower().isin(("true", "1", "1.0"))
    d["syn_invisible"] = (~empty) & ~(cn >= T_ABS)
    d["syn_invisible_thin"] = np.isfinite(ct) & (ct < T_ABS)
    return d


def render_gate_v3(syn_dir: str, ids: Sequence[str], T_ABS: float,
                   workers: Optional[int] = None, verbose: bool = True,
                   bins: Optional[Sequence[float]] = None):
    """GATE_V3: how visible are the synthetic roots BEFORE styling, thin ones too?

    Uses the functions the CUT final audit uses, with the same settings, so
    the gate and the audit cannot disagree:

    * ``contrast_n_syn`` -- ``visibility.synthetic_visibility(on_mode=ON_MODE_V3)``
      (frame-level 'on' over labelled root pixels);
    * ``contrast_n_thin_syn`` -- ``visibility.visibility_by_width`` (widths
      from the mask, ``thin_w2e=THIN_W2E``), i.e. ``thin_regions``;
    * flags by :func:`gate_flags`.

    Also returned (detail, not in gate_v3.csv): ``is_empty``, ``n_on_thin``,
    per-bin ``cn_*`` / ``non_*`` (REPORT_BINS_W2E), ``thick_len_frac`` (share
    of skeleton length with w2e > 24 -- the roots CUT's geometry check
    struggled with), ``n_skel`` and ``on_mode``.  Threads only.
    """
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    from mrsim.traits import REPORT_BINS_W2E, THIN_W2E, skeleton_widths_px
    from mrstyle import visibility as V
    from mrstyle.widths import THICK_W2E
    bins = REPORT_BINS_W2E if bins is None else bins
    ids = list(ids)
    sv = V.synthetic_visibility(syn_dir, ids, workers=workers, verbose=verbose,
                                on_mode=V.ON_MODE_V3)
    labs = V.width_bin_labels(bins)

    def one(sid):
        mask = cv2.imread(os.path.join(syn_dir, "mask", sid + ".png"), cv2.IMREAD_GRAYSCALE) > 127
        r = {"id": sid}
        if not mask.any():
            r.update(contrast_n_thin_syn=float("nan"), n_on_thin=0,
                     thick_len_frac=float("nan"), n_skel=0)
            for lab in labs:
                r["cn_" + lab] = float("nan")
                r["non_" + lab] = 0
            return r
        rgb = cv2.imread(os.path.join(syn_dir, "rgb", sid + ".png"), cv2.IMREAD_COLOR)[:, :, ::-1]
        ap = os.path.join(syn_dir, "alpha", sid + ".png")
        alpha = cv2.imread(ap, cv2.IMREAD_GRAYSCALE) if os.path.exists(ap) else None
        vb = V.visibility_by_width(np.ascontiguousarray(rgb), mask, None, alpha, bins=bins,
                                   thin_w2e=THIN_W2E)
        w = skeleton_widths_px(mask)
        r.update(contrast_n_thin_syn=vb["contrast_n_thin"], n_on_thin=vb["n_on_thin"],
                 thick_len_frac=float(np.mean(w > THICK_W2E)) if w.size else float("nan"),
                 n_skel=int(w.size))
        for lab in labs:
            r["cn_" + lab] = vb["cn_" + lab]
            r["non_" + lab] = vb["non_" + lab]
        return r

    with ThreadPoolExecutor(max_workers=workers or max(1, os.cpu_count() or 2)) as ex:
        det = pd.DataFrame(list(ex.map(one, ids)))
    g = sv[["id", "contrast_n", "is_empty"]].rename(columns={"contrast_n": "contrast_n_syn"})
    g = g.merge(det, on="id", how="left")
    g = gate_flags(g, T_ABS)
    g["on_mode"] = V.ON_MODE_V3
    return g


def write_gate_v3(out_dir: str, gate, T_ABS: float) -> str:
    """Write ``gate_v3.csv`` (id + GATE_COLS_V3 + on_mode, the contract) and
    ``gate_v3_detail.csv`` (everything), merged with rows already there (new
    ids added, re-measured ids replaced) and flags recomputed with ``T_ABS``;
    ``gate_v3.json`` records T_ABS, on_mode and THIN_W2E.  Atomic."""
    import pandas as pd
    from mrsim.traits import THIN_W2E
    from mrstyle import visibility as V
    from mrstyle.realdata import write_csv_atomic, write_json_atomic
    det_p = os.path.join(out_dir, "gate_v3_detail.csv")
    new = gate.copy()
    if os.path.exists(det_p):
        old = pd.read_csv(det_p)
        old = old[~old.id.astype(str).isin(set(new.id.astype(str)))]
        new = pd.concat([old, new], ignore_index=True, sort=False)
    new = gate_flags(new, T_ABS).sort_values("id").reset_index(drop=True)
    write_csv_atomic(new, det_p)
    p = os.path.join(out_dir, GATE_FILE)
    write_csv_atomic(new[["id"] + GATE_COLS_V3 + ["on_mode"]], p)
    write_json_atomic({"T_ABS": float(T_ABS), "on_mode": V.ON_MODE_V3,
                       "thin_w2e": float(THIN_W2E), "n_frames": int(len(new))},
                      os.path.join(out_dir, "gate_v3.json"))
    return p


# --------------------------------------------------------------------------
# geometry audit: did the translator move, add or erase roots?
# --------------------------------------------------------------------------
def ridge_map(rgb01: np.ndarray, sigmas=(1.0, 2.0, 3.5)) -> np.ndarray:
    """Polarity-free ridge response (bright OR dark filaments), multi-scale.

    ``max over sigma of sigma^2 * max|Hessian eigenvalue|`` from OpenCV Gaussian
    + Sobel derivatives: ~75x faster than two skimage ``sato`` passes, which made
    a 6,000-frame audit cost ~750 CPU-minutes.  One Hessian serves both
    polarities, because negating the image only negates its eigenvalues.
    The audit only compares this map with itself (quantile-matched IoU, ratios),
    so its absolute scale does not matter.
    """
    g = cv2.cvtColor(np.asarray(rgb01, np.float32), cv2.COLOR_RGB2GRAY)
    out = np.zeros_like(g)
    for sg in sigmas:
        bl = cv2.GaussianBlur(g, (0, 0), sg)
        dxx = cv2.Sobel(bl, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(bl, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(bl, cv2.CV_32F, 1, 1, ksize=3)
        lam = np.abs(0.5 * (dxx + dyy)) + np.sqrt((0.5 * (dxx - dyy)) ** 2 + dxy ** 2)
        np.maximum(out, (sg * sg) * lam, out=out)
    return out


def _ridge_iou_from_map(r: np.ndarray, mask: np.ndarray) -> float:
    k = int(mask.sum())
    if k == 0:
        return float("nan")
    flat = r.ravel()
    thr = np.partition(flat, flat.size - k)[flat.size - k]
    pred = r >= thr
    return float((pred & mask).sum()) / float((pred | mask).sum())


def ridge_iou(rgb01: np.ndarray, mask: np.ndarray, sigmas=(1.0, 2.0, 3.5)) -> float:
    """IoU between the mask and the ridge map thresholded to the MASK'S AREA.

    Quantile matching makes the score blind to global contrast and colour, which
    is exactly what styling is supposed to change, and sensitive to where the
    filaments are, which it is not.
    """
    return _ridge_iou_from_map(ridge_map(rgb01, sigmas), mask)


def geometry_retention(before01: np.ndarray, after01: np.ndarray, mask: np.ndarray,
                       sigmas=(1.0, 2.0, 3.5), halo_px: int = 7) -> Dict[str, float]:
    """Two guard rails on one frame.

    ``retention = ridge_iou(after) / ridge_iou(before)``: ~1 means the filaments
    are still where the mask says; well below 1 means roots were erased, moved or
    buried under new structure.

    ``bg_gain``: 99th-percentile ridge response AWAY from the mask (outside a
    ``halo_px`` halo), after / before.  Painted-in roots show up here and nowhere
    else -- and it is the only check that works on EMPTY frames, where retention
    is undefined.  A global contrast change scales every frame's gain alike, so
    judge it relative to the median gain of the set, not against 1.

    Guard rails, not quality scores: neither can tell a faithful restyle from one
    that leaves the image unchanged.
    """
    rb = ridge_map(before01, sigmas)
    return _retention_from_maps(rb, ridge_map(after01, sigmas), mask,
                                _background(mask, halo_px))


def _background(mask: np.ndarray, halo_px: int = 7) -> np.ndarray:
    k = np.ones((2 * halo_px + 1,) * 2, np.uint8)
    return ~(cv2.dilate(mask.astype(np.uint8), k) > 0)


def _retention_from_maps(rb: np.ndarray, ra: np.ndarray, mask: np.ndarray,
                         bg: np.ndarray) -> Dict[str, float]:
    b = _ridge_iou_from_map(rb, mask)
    a = _ridge_iou_from_map(ra, mask)
    ret = a / b if (b == b and b > 1e-6) else float("nan")
    if bg.sum() < 100:
        gain = float("nan")
    else:
        pb = float(np.percentile(rb[bg], 99))
        pa = float(np.percentile(ra[bg], 99))
        gain = pa / pb if pb > 1e-9 else float("nan")
    return {"iou_before": b, "iou_after": a, "retention": ret, "bg_gain": gain}


def _read01(path: str) -> np.ndarray:
    im = cv2.imread(path)
    if im is None:
        raise IOError("cannot read %s" % path)
    return im[:, :, ::-1].astype(np.float32) / 255.0


def audit_many(syn_dir: str, styled: Dict[str, str], ids: Sequence[str],
               workers: Optional[int] = None, every: int = 500,
               verbose: bool = True) -> Dict[str, List[dict]]:
    """Audit several styled versions of the same frames in ONE pass.

    ``styled`` maps a method name to its folder of ``<id>.png``.  The synthetic
    frame's ridge map is computed once and shared by every method.

    Threads, not processes: by this point the notebook kernel has initialised
    TensorFlow/CUDA and used OpenCV's thread pool, and FORKING a process in that
    state can deadlock the children -- a cell that simply never finishes.
    OpenCV and NumPy release the GIL for the heavy work, so threads still scale.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    def one(sid):
        rb = ridge_map(_read01(os.path.join(syn_dir, "rgb", sid + ".png")))
        mask = cv2.imread(os.path.join(syn_dir, "mask", sid + ".png"), 0) > 127
        bg = _background(mask)
        res = {}
        for name, d in styled.items():
            ra = ridge_map(_read01(os.path.join(d, sid + ".png")))
            r = _retention_from_maps(rb, ra, mask, bg)
            r["id"] = sid
            r["mask_frac"] = float(mask.mean())
            res[name] = r
        return res

    workers = workers or max(1, os.cpu_count() or 2)
    out: Dict[str, List[dict]] = {k: [] for k in styled}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, r in enumerate(ex.map(one, ids), 1):
            for k, v in r.items():
                out[k].append(v)
            if verbose and (n % every == 0 or n == len(ids)):
                el = time.time() - t0
                print("  audited %d/%d frames  (%.0f s, ETA %.0f s)"
                      % (n, len(ids), el, el / n * (len(ids) - n)))
    return out


def audit_styled(syn_dir: str, sty_dir: str, ids: Sequence[str],
                 workers: Optional[int] = None) -> List[dict]:
    """Single-method convenience wrapper around :func:`audit_many`."""
    return audit_many(syn_dir, {"x": sty_dir}, ids, workers)["x"]


# --------------------------------------------------------------------------
# moving many small files to and from Drive
# --------------------------------------------------------------------------
def zip_dirs(zip_path: str, parts: Dict[str, str], files: Sequence[str] = (),
             ids: Optional[Sequence[str]] = None) -> str:
    """Pack ``{arc_prefix: src_dir}`` (+ loose ``files``) into ONE zip.

    Thousands of small PNGs copied onto the Drive mount one by one take far
    longer than the same bytes as a single file.  PNGs are stored, not deflated:
    they are already compressed.  ``ids`` restricts to ``<id>.png`` in each dir.
    """
    tmp = zip_path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
        for prefix, src in parts.items():
            names = ([i + ".png" for i in ids] if ids is not None
                     else sorted(os.listdir(src)))
            for n in names:
                p = os.path.join(src, n)
                if os.path.isfile(p):
                    zf.write(p, prefix + "/" + n)
        for f in files:
            if os.path.isfile(f):
                zf.write(f, os.path.basename(f), zipfile.ZIP_DEFLATED)
    os.replace(tmp, zip_path)
    return zip_path


def restore_zip(zip_path: str, dest: str) -> bool:
    """Unpack a Drive mirror onto local disk if it exists.  Returns True if done."""
    if not os.path.exists(zip_path):
        return False
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    return True


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def overlay(rgb: np.ndarray, m: np.ndarray, color=(255, 40, 40), a: float = 0.55) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    out[m] = out[m] * (1 - a) + np.array(color, np.float32) * a
    return out.astype(np.uint8)


def grid(images: Sequence[np.ndarray], cols: int, pad: int = 4) -> np.ndarray:
    h, w = images[0].shape[:2]
    rows = int(math.ceil(len(images) / cols))
    out = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), 255, np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        im = im if im.ndim == 3 else np.dstack([im] * 3)
        out[y:y + h, x:x + w] = im[:h, :w]
    return out
