"""Root visibility: ONE measure for real tiles, synthetic frames and styled frames.

The first CUT run was judged with a ridge-filter IoU against the mask.  That
score was blind to the failure that mattered: its baseline agreement was only
0.19 (soil texture responds as strongly as roots), and it kept every frame whose
roots had faded into the soil.  It also mixed two different problems that need
different fixes:

* roots that were already invisible in the synthetic render (59% of frames in
  run 1 -- a renderer problem, fixed in mrsim by forcing roots brighter than
  their soil), and
* roots that CUT faded (about 9% of frames whose roots were visible -- a
  translation problem, guarded by the root loss in the CUT patch).

This module measures visibility directly, the same way everywhere, so the render
gate, the audit and the thresholds cannot disagree:

    centre     skeleton of the synthetic mask, or of a real tile's trace corridor
    on         centre dilated to 3 px (the width-resolved v3 measures, and the
               frame-level gate with on_mode='mask', keep only the labelled
               root pixels of it, so 1-px roots are not judged on the soil
               beside them)
    ring       centre dilated 25 px minus centre dilated 9 px (~4-12 px away),
               minus every known root pixel
    far        centre dilated 81 px minus centre dilated 31 px (~15-40 px away)

    contrast   = median(grey[on]) - median(grey[ring])         grey levels 0-255
    contrast_n = contrast / std(grey[ring])                   comparable across soils
    cnr        = contrast / std(high-pass residual on ring)   fine clutter that hides
                                                              thin roots (report only)
    halo       = median(grey[ring]) - median(grey[far])       a translator can "keep"
                                                              a root by darkening the
                                                              soil next to it instead

Thresholds are not hard-coded judgements: :func:`thresholds_from_reference`
derives them from the real tiles (defaults below are run 1's dataset-B
standing-crop values: p10 0.64, median 1.33, 99.5% bright roots).

The same module holds the screen that keeps tape, foil, plastic, bare tube wall,
seam strings and stitching borders OUT of the real target set -- and, applied to
styled frames, catches CUT painting those objects INTO synthetic ones (29% of
run-1 CUT frames versus 7% of their inputs).
"""
from __future__ import annotations

import hashlib
import math
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import cv2
except ImportError:                                    # pragma: no cover
    cv2 = None

# the one width convention (w2e = 2 x exact EDT at the skeleton) and its bins
from mrsim.traits import REPORT_BINS_W2E, THIN_W2E                  # noqa: E402

# defaults = run-1 real dataset-B standing crop (214 tiles, 11.81 px/mm)
T_ABS_DEFAULT = 0.64        # p10: below this a root is less visible than 90% of real ones
T_MED_DEFAULT = 1.33        # median real root
T_REL_DEFAULT = 0.5         # styled keeps < half its synthetic contrast -> faded
HALO_MAX_GREY = 6.0         # new dark halo deeper than this -> flagged
BG_GAIN_MAX = 1.6           # ABSOLUTE: a threshold relative to the set's own median
                            # drifts upward exactly when painting becomes common
TAPE_FRAC_MAX = 0.005
BAND_LEN_MIN = 0.60
BAND_DB_MAX = 0.0
BLACK_FRAC_MAX = 0.005

MIN_ON, MIN_RING, MIN_FAR = 30, 200, 200

# Ridge-filter scales.  RIDGE_SIGMAS is what every existing number (painting
# thresholds, v2 retention) was measured with.  Their filter support only
# reaches the EDGES of a root wider than ~8 px, so retention on wide roots
# measured edge placement, not the root: v2 frames with mean diameter > 0.4 mm
# failed the geometry check 91% of the time.  RET_SIGMAS_V3 adds scales that
# respond to the body of a 12-30 px root; it is opt-in for retention only, so
# the painting thresholds keep their calibration.
RIDGE_SIGMAS = (1.0, 2.0, 3.5)
RET_SIGMAS_V3 = (1.0, 2.0, 3.5, 6.0, 10.0)

# What 'on' means for a synthetic frame's FRAME-LEVEL measure (regions_from_mask,
# synthetic_visibility, visibility_audit_many).  'band' (default) is v2's 3-px
# band around the centreline and keeps every v2 number bit-identical.  'mask'
# keeps only the labelled root pixels of that band: for roots 3 px or wider the
# two agree (the band lies inside the root), but for a 1-px root the band is two
# thirds soil, so a crisp 1-px root as bright per pixel as a 7-px one read
# contrast_n 0.58 instead of 1.80 -- and syn_invisible / flag_invisible removed
# exactly the thin-majority frames v3 exists to keep.  ON_MODE_V3 is the one
# value the v3 render gate (GATE_V3 -> gate_v3.csv) and the CUT final audit must
# BOTH pass, so the gate and the audit cannot disagree.  Real tiles have 1-px
# trace corridors, not width masks: regions_from_trace stays band-only.
ON_MODES = ("band", "mask")
ON_MODE_V3 = "mask"

_K = {}


def _k(kind: str, a: int, b: Optional[int] = None):
    key = (kind, a, b)
    if key not in _K:
        if kind == "disc":
            _K[key] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a, a))
        elif kind == "rect":                            # a wide, b tall
            _K[key] = np.ones((b, a), np.uint8)
        else:                                           # square
            _K[key] = np.ones((a, a), np.uint8)
    return _K[key]


def grey(rgb_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb_u8), cv2.COLOR_RGB2GRAY).astype(np.float32)


def _read_rgb(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise IOError("cannot read %s" % path)
    return np.ascontiguousarray(im[:, :, ::-1])


# --------------------------------------------------------------------------
# regions
# --------------------------------------------------------------------------
def regions_from_centre(centre: np.ndarray, known: Optional[np.ndarray] = None):
    """``(on, ring, far)`` boolean maps around a 1-px centreline."""
    c = np.asarray(centre).astype(np.uint8)
    on = cv2.dilate(c, _k("sq", 3)) > 0
    ring = (cv2.dilate(c, _k("sq", 25)) > 0) & ~(cv2.dilate(c, _k("sq", 9)) > 0)
    far = (cv2.dilate(c, _k("sq", 81)) > 0) & ~(cv2.dilate(c, _k("sq", 31)) > 0)
    if known is not None:
        kn = np.asarray(known, bool)
        ring &= ~kn
        far &= ~kn
    return on, ring, far


def _check_on_mode(on_mode: str) -> None:
    if on_mode not in ON_MODES:
        raise ValueError("on_mode must be one of %s, not %r" % (ON_MODES, on_mode))


def regions_from_mask(mask: np.ndarray, alpha: Optional[np.ndarray] = None,
                      alpha_min: int = 8, on_mode: str = "band"):
    """Regions for a synthetic frame.

    ``known`` also takes in faint roots that fall below the mask threshold
    (``alpha >= alpha_min`` on a uint8 alpha), so a ring pixel is soil, not a
    neighbouring root the mask chose not to label.

    ``on_mode='mask'`` (v3, see ON_MODE_V3) cuts ``on`` down to the labelled
    root pixels, exactly as the width-resolved measures do
    (:func:`_root_regions`); ``ring`` and ``far`` are unchanged.  The default
    'band' is v2's measure.
    """
    from skimage.morphology import skeletonize
    _check_on_mode(on_mode)
    m = np.asarray(mask) > 0
    known = m if alpha is None else (m | (np.asarray(alpha) >= alpha_min))
    known = cv2.dilate(known.astype(np.uint8), _k("sq", 3)) > 0
    if on_mode == "mask":
        return _root_regions(skeletonize(m), m, known)
    return regions_from_centre(skeletonize(m), known)


def regions_from_trace(trace: np.ndarray):
    """Regions for a real tile from its trace corridor image.

    Untraced roots (new-growth traces mark only NEW roots) can sit in the ring
    and pull real contrast down, so thresholds derived from real tiles are, if
    anything, lenient.
    """
    from skimage.morphology import skeletonize
    t = np.asarray(trace) > 0
    known = cv2.dilate(t.astype(np.uint8), _k("sq", 3)) > 0
    return regions_from_centre(skeletonize(t), known)


# --------------------------------------------------------------------------
# regions per root width (thin roots measured on their own)
# --------------------------------------------------------------------------
def _fmt_edge(x: float) -> str:
    return "inf" if math.isinf(x) else ("%g" % x).replace(".", "p")


def width_bin_labels(bins: Sequence[float] = REPORT_BINS_W2E) -> List[str]:
    """Column-safe labels for ``[lo, hi)`` bins: (2, 3, 5) -> ['w2_3', 'w3_5']."""
    b = [float(x) for x in bins]
    return ["w%s_%s" % (_fmt_edge(lo), _fmt_edge(hi)) for lo, hi in zip(b[:-1], b[1:])]


def _width_parts(mask, width_w2e, alpha, alpha_min):
    """``(mask, skeleton, width map in w2e, known-root map)`` shared by the width regions."""
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    m = np.asarray(mask) > 0
    known = m if alpha is None else (m | (np.asarray(alpha) >= alpha_min))
    known = cv2.dilate(known.astype(np.uint8), _k("sq", 3)) > 0
    skel = skeletonize(m) if m.any() else m
    if width_w2e is None:
        w = 2.0 * ndi.distance_transform_edt(m)        # = mrsim.traits.skeleton_widths_px
    else:
        w = np.asarray(width_w2e, np.float64)
        if w.shape != m.shape:
            raise ValueError("width_w2e %s does not match mask %s" % (w.shape, m.shape))
    return m, skel, w, known


def _root_regions(centre, m, known):
    """:func:`regions_from_centre` with ``on`` cut down to the LABELLED root pixels.

    The frame-level ``on`` (centreline dilated to 3 px) is the root itself for
    any root 3 px or wider -- which is what T_ABS was calibrated on (real roots
    ~8 px wide).  For a 1-px root that band is two thirds soil, so its median is
    the SOIL beside the root: a crisp 1-px root as bright per pixel as a 7-px one
    read contrast_n 0.58 (below T_ABS 0.64, 'invisible') against 1.80 for the
    thick root on the same busy soil.  The width-resolved measures exist to judge
    exactly those roots, so they take ``on`` over mask pixels only.  For roots
    >= 3 px nothing changes (the band lies inside the root, apart from a pixel
    poking out at a tip or a diagonal step).  The frame-level regions_from_mask
    uses this only with ``on_mode='mask'`` (ON_MODE_V3); its default is v2's
    band, so every v2 number stays bit-identical.  A side effect that is
    wanted: a 1-px root moved by 1 px no longer sits in its own 'on' set.

    ``ring`` and ``far`` keep their definitions.  MIN_ON now counts the pixels
    the median is actually taken over (a 1-px root needs ~30 px of length): a
    median over fewer labelled pixels is too noisy to fail a frame on, and NaN
    never fires a flag.
    """
    on, ring, far = regions_from_centre(centre, known)
    return on & m, ring, far


def regions_by_width(mask: np.ndarray, width_w2e: Optional[np.ndarray] = None,
                     alpha: Optional[np.ndarray] = None,
                     bins: Sequence[float] = REPORT_BINS_W2E,
                     alpha_min: int = 8) -> "OrderedDict[str, tuple]":
    """``{label: (on, ring, far)}``: the visibility regions of each width bin.

    A frame's single contrast is dominated by its widest, brightest roots, so a
    translator (or a renderer) can erase every thin root while the frame still
    passes.  Here the centreline is split by width -- ``w2e`` from the mask by
    default, or a caller's ``width_w2e`` map in the same units -- and each part
    gets its own regions.  ``known`` is ALWAYS the full mask (+ faint alpha), so
    a thin root's ring never counts a neighbouring thick root as soil.

    ``on`` is the 3-px band around the bin's centreline INTERSECTED WITH THE
    MASK (see :func:`_root_regions`): a 1-px root is measured on its own pixels,
    not on the soil beside it.

    Bins are ``[lo, hi)``; skeleton pixels outside every bin are ignored.  An
    empty mask gives empty regions (``frame_visibility`` then returns NaN).
    """
    b = [float(x) for x in bins]
    m, skel, w, known = _width_parts(mask, width_w2e, alpha, alpha_min)
    out: "OrderedDict[str, tuple]" = OrderedDict()
    for lab, lo, hi in zip(width_bin_labels(b), b[:-1], b[1:]):
        out[lab] = _root_regions(skel & (w >= lo) & (w < hi), m, known)
    return out


def thin_regions(mask: np.ndarray, width_w2e: Optional[np.ndarray] = None,
                 alpha: Optional[np.ndarray] = None, thin_w2e: float = THIN_W2E,
                 alpha_min: int = 8):
    """Regions of the thin roots only: centreline where ``w2e <= thin_w2e``.

    The same 'thin' as the simulator's fine share and 04's R_thin (inclusive,
    so a 3-px root at w2e 4 counts).  Used for ``contrast_n_thin``.  ``on`` is
    restricted to labelled root pixels (:func:`_root_regions`), so a crisp 1-px
    root and a 7-px root of the same per-pixel brightness read about the same
    contrast_n, comparable with T_ABS.
    """
    m, skel, w, known = _width_parts(mask, width_w2e, alpha, alpha_min)
    return _root_regions(skel & (w <= float(thin_w2e)), m, known)


# --------------------------------------------------------------------------
# the measure
# --------------------------------------------------------------------------
def _grey_parts(rgb_u8: np.ndarray):
    g = grey(rgb_u8)
    return g, g - cv2.GaussianBlur(g, (0, 0), 3.0)


def _visibility_from_grey(g, resid, regions) -> Dict[str, float]:
    on, ring, far = regions
    n_on, n_ring = int(on.sum()), int(ring.sum())
    out = {"contrast": float("nan"), "contrast_n": float("nan"), "cnr": float("nan"),
           "halo": float("nan"), "n_on": n_on}
    if n_on < MIN_ON or n_ring < MIN_RING:
        return out
    gr = g[ring]
    mr = float(np.median(gr))
    c = float(np.median(g[on])) - mr
    out["contrast"] = c
    out["contrast_n"] = c / max(float(gr.std()), 1e-3)
    out["cnr"] = c / max(float(resid[ring].std()), 1e-3)
    if int(far.sum()) >= MIN_FAR:
        out["halo"] = mr - float(np.median(g[far]))
    return out


def frame_visibility(rgb_u8: np.ndarray, regions) -> Dict[str, float]:
    """``contrast, contrast_n, cnr, halo, n_on`` for one image (NaN if unmeasurable)."""
    on, ring, _ = regions
    if int(on.sum()) < MIN_ON or int(ring.sum()) < MIN_RING:
        return _visibility_from_grey(None, None, regions)   # NaNs, no image work
    g, resid = _grey_parts(rgb_u8)
    return _visibility_from_grey(g, resid, regions)


def visibility_by_width(rgb_u8: np.ndarray, mask: np.ndarray,
                        width_w2e: Optional[np.ndarray] = None,
                        alpha: Optional[np.ndarray] = None,
                        bins: Sequence[float] = REPORT_BINS_W2E,
                        thin_w2e: Optional[float] = THIN_W2E,
                        alpha_min: int = 8) -> Dict[str, float]:
    """Visibility per width bin plus the thin-root summary, as flat columns.

    Per bin label ``L`` (see :func:`width_bin_labels`): ``cn_L`` (contrast_n),
    ``c_L`` (contrast, grey levels) and ``non_L`` (on-pixels).  With
    ``thin_w2e``: ``contrast_n_thin``, ``contrast_thin``, ``n_on_thin`` from
    :func:`thin_regions`.  NaN where a part has fewer than MIN_ON on-pixels.

    The render gate and the CUT audit must call this with the SAME width source
    (``width_w2e=None``: w2e from the mask) so their thin numbers agree.
    """
    regs = regions_by_width(mask, width_w2e, alpha, bins, alpha_min)
    g, resid = _grey_parts(rgb_u8)
    out: Dict[str, float] = {}
    for lab, reg in regs.items():
        v = _visibility_from_grey(g, resid, reg)
        out["cn_" + lab] = v["contrast_n"]
        out["c_" + lab] = v["contrast"]
        out["non_" + lab] = v["n_on"]
    if thin_w2e is not None:
        v = _visibility_from_grey(g, resid, thin_regions(mask, width_w2e, alpha,
                                                         thin_w2e, alpha_min))
        out["contrast_n_thin"] = v["contrast_n"]
        out["contrast_thin"] = v["contrast"]
        out["n_on_thin"] = v["n_on"]
    return out


# --------------------------------------------------------------------------
# painted-in structure away from the roots
# --------------------------------------------------------------------------
def ridge_map(rgb01: np.ndarray, sigmas=(1.0, 2.0, 3.5)) -> np.ndarray:
    """Polarity-free multi-scale ridge response (cv2 Hessian, ~75x faster than sato)."""
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


BG_CLEAR_PX = 16   # background = this far from any root CENTRELINE


def centreline(mask: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize
    m = np.asarray(mask) > 0
    return skeletonize(m) if m.any() else m


def background(centre: np.ndarray, clear_px: int = BG_CLEAR_PX) -> np.ndarray:
    """Pixels farther than ``clear_px`` from a root centreline.

    Measured from the CENTRELINE, not the mask edge, and far enough (a 1 mm root
    is ~12 px wide at 11.81 px/mm, plus ~10 px of ridge-filter support): with a
    7 px margin from the mask the top percentile of the "background" was the
    roots' own filter response spilling over, identical for smooth and busy soil.
    The same definition serves synthetic masks and real traces (a thin corridor).
    """
    k = np.ones((2 * clear_px + 1, 2 * clear_px + 1), np.uint8)
    return ~(cv2.dilate((np.asarray(centre) > 0).astype(np.uint8), k) > 0)


def background_from_mask(mask: np.ndarray, clear_px: int = BG_CLEAR_PX) -> np.ndarray:
    """Pixels farther than ``clear_px`` from any root PIXEL (mask or centreline).

    :func:`background` measures from the centreline, so the body of a root
    wider than ~2 x ``clear_px`` (32 px; v3 renders and PRMI masks reach 30+)
    counted as background: its own ridge response then read as 'painted' soil
    and inflated the busy-soil thresholds of thick-root styles.  For a full
    width mask (synthetic frames, PRMI) measure from the mask instead.  The
    centreline lies inside the mask, so ``mask | centreline`` is the mask.
    Real Ginger traces are 1-px corridors and keep :func:`background`.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    k = np.ones((2 * clear_px + 1, 2 * clear_px + 1), np.uint8)
    return ~(cv2.dilate(m, k) > 0)


def bg_ridge(ridge: np.ndarray, centre: np.ndarray, clear_px: int = BG_CLEAR_PX,
             bg: Optional[np.ndarray] = None) -> float:
    """99th-percentile ridge strength away from the roots.

    Compared against the SAME statistic on real tiles (away from their traces),
    this asks "is there more root-like structure in the soil than real soil has?"
    -- a question with an absolute answer, unlike a before/after ratio, which
    depends on how smooth the synthetic soil was to begin with (v2's clean soil
    has ~1/3 the background ridge of run 1's artefact-laden soil).

    ``bg`` overrides the background map (e.g. :func:`background_from_mask`).
    """
    if bg is None:
        bg = background(centre, clear_px)
    if bg.sum() < 100:
        return float("nan")
    return float(np.percentile(ridge[bg], 99))


def ridge_iou(ridge: np.ndarray, mask: np.ndarray) -> float:
    """IoU of the mask with the ridge map thresholded to the MASK'S AREA.

    Blind to global contrast (styling's job), sensitive to WHERE filaments are:
    a shifted or widened root lowers it.  Used as a ratio after / before.
    """
    m = np.asarray(mask) > 0
    k = int(m.sum())
    if k == 0:
        return float("nan")
    flat = ridge.ravel()
    thr = np.partition(flat, flat.size - k)[flat.size - k]
    pred = ridge >= thr
    return float((pred & m).sum()) / float((pred | m).sum())


def bg_gain(rb_before: np.ndarray, ra_after: np.ndarray, centre: np.ndarray,
            clear_px: int = BG_CLEAR_PX, bg: Optional[np.ndarray] = None) -> float:
    """99th-percentile ridge strength AWAY from the mask, after / before.

    The only check that works on empty frames: anything root-like that appears
    where the mask says there is nothing was painted in.  ``bg`` overrides the
    background map, as in :func:`bg_ridge`.
    """
    if bg is None:
        bg = background(centre, clear_px)
    if bg.sum() < 100:
        return float("nan")
    pb = float(np.percentile(rb_before[bg], 99))
    pa = float(np.percentile(ra_after[bg], 99))
    return pa / pb if pb > 1e-9 else float("nan")


# --------------------------------------------------------------------------
# tape / foil / plastic / tube wall / seam string / stitching border
# --------------------------------------------------------------------------
def _vmax(rgb):
    return cv2.max(cv2.max(rgb[:, :, 0], rgb[:, :, 1]), rgb[:, :, 2])


def pale_mask(rgb, lab=None, v=None, v_cool=100, a_cool=0, b_cool=4,
              v_white=140, chroma_white=6):
    """Bright pixels that are COOL (wall, grey/blue tape) or NEUTRAL (white tape).

    Soil objects that are pale -- sand, aggregates, white roots, hyphae,
    droplets -- are warm (b* above ~6) or thin; that, not brightness, is what
    separates them from tape.
    """
    if lab is None:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    if v is None:
        v = _vmax(rgb)
    a = lab[:, :, 1].astype(np.int16) - 128
    b = lab[:, :, 2].astype(np.int16) - 128
    cool = (v >= v_cool) & (a <= a_cool) & (b <= b_cool)
    white = (v >= v_white) & (a * a + b * b <= chroma_white * chroma_white)
    return cool | white


def _band(g, v, bstar, tophat=41, thr=25, vmin=90, min_len=101, margin=3, win=13):
    """(width fraction covered by a thin bright horizontal line, its b* - tile b*)."""
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, _k("rect", 1, tophat))
    bm = ((th >= thr) & (v >= vmin)).astype(np.uint8)
    bm[:margin] = 0
    bm[bm.shape[0] - margin:] = 0                       # scanner-border rows are not bands
    bm = cv2.morphologyEx(bm, cv2.MORPH_OPEN, _k("rect", min_len, 1))
    cov = cv2.dilate(bm, _k("rect", 1, win)).mean(axis=1)
    L = float(cov.max())
    if L == 0.0:
        return 0.0, 0.0
    r = int(cov.argmax())
    lo, hi = max(0, r - win // 2), min(bm.shape[0], r + win // 2 + 1)
    sel = bm[lo:hi] > 0
    return L, float(np.median(bstar[lo:hi][sel]) - np.median(bstar))


def _black_border(g, mx, vmax=3, ring_px=7, ring_min=40, min_px=300):
    """Area fraction of edge-touching flat-black regions with a BRIGHT surround.

    A hard cut (stitching border), not a dark vignette over real soil.
    """
    m = (mx <= vmax).astype(np.uint8)
    if m.sum() < min_px:
        return 0.0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _k("rect", 5, 5))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    H, W = m.shape
    tot = 0
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < min_px or not (x == 0 or y == 0 or x + w == W or y + h == H):
            continue
        x0, y0 = max(0, x - ring_px), max(0, y - ring_px)
        x1, y1 = min(W, x + w + ring_px), min(H, y + h + ring_px)
        comp = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = (cv2.dilate(comp, _k("rect", 2 * ring_px + 1, 2 * ring_px + 1)) > 0) & (comp == 0)
        if ring.any() and float(np.median(g[y0:y1, x0:x1][ring])) >= ring_min:
            tot += a
    return float(tot) / m.size


def tape_score(rgb_u8: np.ndarray) -> Dict[str, float]:
    """Screen features for one tile (uint8 RGB at ~11.8 px/mm).

    ``tape_frac`` is the fraction of the tile left in the pale mask after an
    opening with a 21-px (~1.8 mm) disc: roots and specks are thinner than that,
    tape is not.  Tuned on 474 hand-labelled tiles (precision/recall 0.98).
    """
    rgb = np.ascontiguousarray(rgb_u8)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb_u8 must be HxWx3 uint8")
    H = rgb.shape[0]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mx = _vmax(rgb)
    raw = pale_mask(rgb, lab, mx)
    m = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_OPEN, _k("disc", 21))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    cc = float(st[1:, cv2.CC_STAT_AREA].max()) / m.size if n > 1 else 0.0
    band_len, band_db = _band(g, mx, lab[:, :, 2].astype(np.int16) - 128)
    rows = g.mean(axis=1)
    top = 0
    while top < min(8, H) and rows[top] >= 200:
        top += 1
    bot = 0
    while bot < min(8, H) and rows[H - 1 - bot] >= 200:
        bot += 1
    return {"tape_frac": float(m.mean()), "tape_cc": cc, "pale_frac": float(raw.mean()),
            "band_len": band_len, "band_db": band_db, "black_frac": _black_border(g, mx),
            "border_rows": int(top + bot), "spec_frac": float((mx >= 250).mean())}


def screen_reason(score: Dict[str, float], tape_frac_max: float = TAPE_FRAC_MAX,
                  band_len_min: float = BAND_LEN_MIN, band_db_max: float = BAND_DB_MAX,
                  black_frac_max: float = BLACK_FRAC_MAX) -> str:
    """'' to keep, else 'tape' | 'band' | 'black_border'.

    The band must also be COOLER than its tile: in dataset A, horizontal roots
    and rows of droplets are long bright lines too, but warm.
    """
    if score["tape_frac"] >= tape_frac_max:
        return "tape"
    if score["band_len"] >= band_len_min and score["band_db"] < band_db_max:
        return "band"
    if score["black_frac"] >= black_frac_max:
        return "black_border"
    return ""


def tile_hash(rgb_u8: np.ndarray) -> str:
    """Exact-duplicate key.  The same scan sits in two subsets (e.g. B newgrowth
    003-x1 is standingcrop 004-x0): 192 run-1 target tiles were copies."""
    a = np.ascontiguousarray(rgb_u8)
    h = hashlib.sha1(a.tobytes())
    h.update(repr(a.shape).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------
# the real reference
# --------------------------------------------------------------------------
def _pool(fn, items, workers=None, every=None, label="", verbose=True):
    import time
    from concurrent.futures import ThreadPoolExecutor
    out = []
    t0 = time.time()
    # Threads, not processes: by the time the audit runs the kernel has loaded
    # TensorFlow and used OpenCV's thread pool, and FORKING in that state can
    # deadlock the children -- a cell that simply never finishes.  cv2 and numpy
    # release the GIL for the heavy work, so threads still scale.
    with ThreadPoolExecutor(max_workers=workers or max(1, os.cpu_count() or 2)) as ex:
        for n, r in enumerate(ex.map(fn, items), 1):
            out.append(r)
            if verbose and every and (n % every == 0 or n == len(items)):
                el = time.time() - t0
                print("  %s %d/%d (%.0f s, ETA %.0f s)"
                      % (label, n, len(items), el, el / n * (len(items) - n)))
    return out


def real_reference(real_dir: str, ids: Sequence[str], workers: Optional[int] = None,
                   verbose: bool = True):
    """Visibility of real roots: one row per tile from ``real_dir/{rgb,trace}/<id>.png``."""
    import pandas as pd

    def one(tid):
        rgb = _read_rgb(os.path.join(real_dir, "rgb", tid + ".png"))
        tr = cv2.imread(os.path.join(real_dir, "trace", tid + ".png"), cv2.IMREAD_GRAYSCALE)
        d = frame_visibility(rgb, regions_from_trace(tr > 0))
        d["bg_ridge"] = bg_ridge(ridge_map(rgb.astype(np.float32) / 255.0),
                                 centreline(tr > 0))
        d["id"] = tid
        return d

    return pd.DataFrame(_pool(one, list(ids), workers, every=200,
                              label="real reference", verbose=verbose))


def thresholds_from_reference(df, abs_q: float = 0.10) -> Dict[str, float]:
    """T_ABS = the ``abs_q`` quantile of real contrast_n (signed), T_MED its median."""
    v = np.asarray(df["contrast_n"], float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no measurable real tiles")
    q = np.quantile(v, [abs_q, 0.10, 0.25, 0.50, 0.75, 0.90])
    out = {"T_ABS": float(q[0]), "T_MED": float(q[3]), "p10": float(q[1]),
           "p25": float(q[2]), "p50": float(q[3]), "p75": float(q[4]),
           "p90": float(q[5]), "n": int(v.size), "frac_bright": float((v > 0).mean())}
    if "bg_ridge" in df:
        b = np.asarray(df["bg_ridge"], float)
        b = b[np.isfinite(b)]
        # soil busier than 95% of real soil backgrounds -> something was painted in
        out["BG_RIDGE_MAX"] = float(np.quantile(b, 0.95)) if b.size else float("nan")
    return out


# --------------------------------------------------------------------------
# synthetic-only (render gate) and synthetic-vs-styled (audit)
# --------------------------------------------------------------------------
def _read_frame(syn_dir, sid):
    mask = cv2.imread(os.path.join(syn_dir, "mask", sid + ".png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError("no mask for %s in %s" % (sid, syn_dir))
    ap = os.path.join(syn_dir, "alpha", sid + ".png")
    alpha = cv2.imread(ap, cv2.IMREAD_GRAYSCALE) if os.path.exists(ap) else None
    return _read_rgb(os.path.join(syn_dir, "rgb", sid + ".png")), mask > 127, alpha


def synthetic_visibility(syn_dir: str, ids: Sequence[str], workers: Optional[int] = None,
                         verbose: bool = True, on_mode: str = "band"):
    """Visibility of the synthetic roots BEFORE styling (the render gate).

    ``on_mode`` as in :func:`regions_from_mask`.  The v3 gate passes
    ON_MODE_V3 ('mask'), the same value as the CUT final audit; a non-default
    mode is recorded in an ``on_mode`` column so a gate table says how it was
    measured.  The default ('band', no extra column) is v2's table exactly.
    """
    import pandas as pd
    _check_on_mode(on_mode)

    def one(sid):
        rgb, mask, alpha = _read_frame(syn_dir, sid)
        if not mask.any():
            d = {"contrast": float("nan"), "contrast_n": float("nan"),
                 "cnr": float("nan"), "halo": float("nan"), "n_on": 0}
        else:
            d = frame_visibility(rgb, regions_from_mask(mask, alpha, on_mode=on_mode))
        d["id"] = sid
        d["is_empty"] = not bool(mask.any())
        if on_mode != "band":
            d["on_mode"] = on_mode
        return d

    return pd.DataFrame(_pool(one, list(ids), workers, every=1000,
                              label="synthetic visibility", verbose=verbose))


def _finite(x) -> bool:
    return x is not None and np.isfinite(x)


def visibility_audit_many(syn_dir: str, styled: Dict[str, str], ids: Sequence[str],
                          T_ABS: float = T_ABS_DEFAULT, T_MED: float = T_MED_DEFAULT,
                          T_REL: float = T_REL_DEFAULT, halo_max: float = HALO_MAX_GREY,
                          bg_gain_max: float = BG_GAIN_MAX,
                          bg_ridge_max: Optional[float] = None, bg_gain_min: float = 1.2,
                          workers: Optional[int] = None,
                          every: int = 500, verbose: bool = True,
                          bg_mode: str = "centre",
                          ret_sigmas: Optional[Sequence[float]] = None,
                          thin_w2e: Optional[float] = None,
                          on_mode: str = "band") -> Dict[str, List[dict]]:
    """Audit several styled versions of the same frames in one pass.

    ``syn_dir`` holds ``rgb/``, ``mask/`` and ``alpha/``; ``styled`` maps a method
    name to its folder of ``<id>.png``.  The synthetic frame's regions, contrast,
    ridge map and tape score are computed once and shared by every method.

    A frame is kept only if none of these fire:

    flag_invisible     |contrast_n| < T_ABS: less visible than 90% of real roots
    flag_faded         the synthetic roots were visible and styling kept less than
                       T_REL of their contrast, ending below the real median --
                       a frame brought DOWN to typical real contrast is not faded
    flag_halo          a new dark halo deeper than ``halo_max`` grey
    flag_inverted      the synthetic roots were visible and bright, and styling
                       turned them dark (99.5% of real roots are brighter than soil)
    flag_hallucinated  tape/foil/white masses present after styling but not before
    flag_painted       with ``bg_ridge_max`` (from the real reference): the soil is
                       busier than 95% of real soil AND busier than before styling
                       (bg_gain > ``bg_gain_min``); without it, bg_gain > bg_gain_max

    ``retention`` (ridge IoU after / before) is returned for a geometry check
    calibrated on FDA, which cannot move roots -- the caller sets that threshold.

    ``syn_invisible`` (signed: synthetic contrast_n < T_ABS) is reported, not a
    styling flag -- it is the renderer's failure, and such frames should already
    have been excluded from training by the render gate.

    Opt-in v3 options (the defaults reproduce the v2 numbers exactly):

    ``bg_mode='mask'``  background for bg_ridge / bg_gain from
                        :func:`background_from_mask`, so the body of a root
                        wider than ~32 px is not read as painted soil.
    ``ret_sigmas``      ridge scales for ``retention`` only (e.g.
                        RET_SIGMAS_V3); painting keeps RIDGE_SIGMAS, the scales
                        its thresholds were calibrated with.
    ``thin_w2e``        also measure the thin roots alone (:func:`thin_regions`):
                        ``contrast_n_thin_syn``, ``contrast_n_thin``,
                        ``syn_invisible_thin`` (signed, renderer) and
                        ``flag_invisible_thin`` (|contrast_n_thin| < T_ABS after
                        styling).  Not part of ``keep``: the caller decides
                        (v3 train_ok = keep & ~flag_invisible_thin & ...).
    ``on_mode='mask'``  the frame-level regions (shared by the synthetic and the
                        styled measure) take ``on`` over labelled root pixels
                        only (:func:`regions_from_mask`).  With the v2 band a
                        frame of crisp 1-px roots read about a third of the
                        contrast of a 7-px frame of the same per-pixel
                        brightness, so syn_invisible and flag_invisible -- both
                        in v3's train_ok -- dropped thin-majority frames.  This
                        changes contrast*, ratio, cnr and every flag built on
                        them; halo (ring vs far) is unchanged.  The v3 audit
                        passes ON_MODE_V3, the value GATE_V3 used; rows then
                        carry an ``on_mode`` column.
    """
    if bg_mode not in ("centre", "mask"):
        raise ValueError("bg_mode must be 'centre' or 'mask', not %r" % (bg_mode,))
    _check_on_mode(on_mode)
    rsig = None
    if ret_sigmas is not None and tuple(float(s) for s in ret_sigmas) != RIDGE_SIGMAS:
        rsig = tuple(float(s) for s in ret_sigmas)
    nan_v = {"contrast": float("nan"), "contrast_n": float("nan"),
             "cnr": float("nan"), "halo": float("nan"), "n_on": 0}

    def one(sid):
        syn, mask, alpha = _read_frame(syn_dir, sid)
        empty = not bool(mask.any())
        reg = None if empty else regions_from_mask(mask, alpha, on_mode=on_mode)
        v0 = (frame_visibility(syn, reg) if reg is not None else dict(nan_v))
        rb = ridge_map(syn.astype(np.float32) / 255.0)
        centre = centreline(mask)
        bgm = background_from_mask(mask) if bg_mode == "mask" else None
        br0 = bg_ridge(rb, centre, bg=bgm)
        if empty:
            iou0 = float("nan")
        elif rsig is None:
            iou0 = ridge_iou(rb, mask)
        else:
            iou0 = ridge_iou(ridge_map(syn.astype(np.float32) / 255.0, rsig), mask)
        regt = None
        vt0 = dict(nan_v)
        if thin_w2e is not None and not empty:
            regt = thin_regions(mask, None, alpha, thin_w2e)
            vt0 = frame_visibility(syn, regt)
        tf0 = tape_score(syn)["tape_frac"]
        syn_vis = _finite(v0["contrast_n"]) and v0["contrast_n"] >= T_ABS
        res = {}
        for name, d in styled.items():
            im = _read_rgb(os.path.join(d, sid + ".png"))
            v1 = (frame_visibility(im, reg) if reg is not None else dict(nan_v))
            ra = ridge_map(im.astype(np.float32) / 255.0)
            gain = bg_gain(rb, ra, centre, bg=bgm)
            br1 = bg_ridge(ra, centre, bg=bgm)
            if empty:
                iou1 = float("nan")
            elif rsig is None:
                iou1 = ridge_iou(ra, mask)
            else:
                iou1 = ridge_iou(ridge_map(im.astype(np.float32) / 255.0, rsig), mask)
            tf1 = tape_score(im)["tape_frac"]
            c0, c1 = v0["contrast"], v1["contrast"]
            ratio = (abs(c1) / abs(c0)) if (_finite(c0) and _finite(c1) and abs(c0) > 1e-6) \
                else float("nan")
            r = {"id": sid, "is_empty": empty,
                 "contrast_syn": c0, "contrast_n_syn": v0["contrast_n"],
                 "contrast": c1, "contrast_n": v1["contrast_n"], "ratio": ratio,
                 "cnr_syn": v0["cnr"], "cnr": v1["cnr"],
                 "halo_syn": v0["halo"], "halo": v1["halo"],
                 "bg_gain": gain, "bg_ridge_syn": br0, "bg_ridge": br1,
                 "retention": (iou1 / iou0) if (_finite(iou0) and iou0 > 1e-6)
                 else float("nan"),
                 "tape_frac_syn": tf0, "tape_frac": tf1,
                 "syn_invisible": (not empty) and not syn_vis}
            r["flag_invisible"] = (not empty) and _finite(v1["contrast_n"]) and \
                abs(v1["contrast_n"]) < T_ABS
            r["flag_faded"] = (not empty) and syn_vis and _finite(ratio) and \
                ratio < T_REL and _finite(v1["contrast_n"]) and v1["contrast_n"] < T_MED
            r["flag_halo"] = (not empty) and _finite(v1["halo"]) and _finite(v0["halo"]) and \
                (v1["halo"] - v0["halo"]) < -halo_max
            r["flag_inverted"] = (not empty) and syn_vis and _finite(c1) and c0 > 0 and c1 < 0
            r["flag_hallucinated"] = tf1 >= TAPE_FRAC_MAX and tf0 < TAPE_FRAC_MAX
            if bg_ridge_max is not None and _finite(bg_ridge_max):
                r["flag_painted"] = _finite(br1) and br1 > bg_ridge_max and \
                    _finite(gain) and gain > bg_gain_min
            else:
                r["flag_painted"] = _finite(gain) and gain > bg_gain_max
            r["keep"] = not (r["flag_invisible"] or r["flag_faded"] or r["flag_halo"]
                             or r["flag_inverted"] or r["flag_hallucinated"]
                             or r["flag_painted"])
            if thin_w2e is not None:
                vt1 = frame_visibility(im, regt) if regt is not None else dict(nan_v)
                r["contrast_n_thin_syn"] = vt0["contrast_n"]
                r["contrast_n_thin"] = vt1["contrast_n"]
                r["n_on_thin"] = vt0["n_on"]
                r["syn_invisible_thin"] = _finite(vt0["contrast_n"]) and \
                    vt0["contrast_n"] < T_ABS
                r["flag_invisible_thin"] = _finite(vt1["contrast_n"]) and \
                    abs(vt1["contrast_n"]) < T_ABS
            if on_mode != "band":
                r["on_mode"] = on_mode
            res[name] = r
        return res

    rows = _pool(one, list(ids), workers, every=every, label="audited", verbose=verbose)
    out: Dict[str, List[dict]] = {k: [] for k in styled}
    for r in rows:
        for k, v in r.items():
            out[k].append(v)
    return out


def pick_graded(df, col: str, levels: Sequence[float], n_per_level: int = 2) -> List[str]:
    """Ids whose ``col`` is nearest each level -- for a graded contact sheet from
    which a person, not a percentile, decides what 'clearly visible' means."""
    d = df[np.isfinite(np.asarray(df[col], float))]
    chosen: List[str] = []
    for lv in levels:
        cand = d.iloc[np.argsort(np.abs(np.asarray(d[col], float) - lv))]
        k = 0
        for sid in cand["id"]:
            if sid in chosen:
                continue
            chosen.append(sid)
            k += 1
            if k >= n_per_level:
                break
    return chosen
