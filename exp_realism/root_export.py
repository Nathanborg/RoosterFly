"""One CSV row of root traits per image (the annotator's "Export CSV").

Traits come from :func:`mrsim.traits.frame_traits` (step-weighted skeleton
length, length-weighted mean diameter from the distance transform, length by
diameter class, endpoints and apparent branch points), plus surface area
(pi x d x l) and volume (pi / 4 x d^2 x l) summed along the skeleton.  An image
without roots gets zeros.  The image metadata (site, tube, window, date, time,
session) is read from the file name (:func:`temporal.parse_frame`).
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np

from exp_realism import temporal

DIAM_EDGES_MM = (0.0, 0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 1e9)


def _class_label(lo: float, hi: float) -> str:
    return f"len_mm_d{lo:g}-{hi:g}mm" if hi < 1e8 else f"len_mm_d{lo:g}mm+"


COLUMNS: List[str] = (
    ["image_index", "image", "site", "tube", "window", "date", "time", "session", "source", "threshold",
     "width_px", "height_px", "px_per_mm",
     "root_segments", "root_length_mm", "root_length_cm", "projected_area_mm2", "surface_area_mm2",
     "volume_mm3", "mean_diameter_mm", "root_coverage_pct", "root_length_density_mm_per_cm2",
     "n_endpoints", "n_branch_points"]
    + [_class_label(lo, hi) for lo, hi in zip(DIAM_EDGES_MM[:-1], DIAM_EDGES_MM[1:])]
)


def image_row(path: str, index: int, mask: Optional[np.ndarray], source: str, px_per_mm: float,
              threshold: Optional[float] = None) -> Dict[str, object]:
    """Row for one image; ``mask`` None means nothing to measure (traits left empty)."""
    key = temporal.parse_frame(path)
    row: Dict[str, object] = {c: "" for c in COLUMNS}
    row.update(image_index=index, image=os.path.basename(path), source=source,
               threshold="" if threshold is None else round(float(threshold), 4), px_per_mm=px_per_mm)
    if key is not None:
        row.update(site=key.site.strip("_- "), tube=key.tube, window=key.loc, date=key.date, time=key.time,
                   session=key.session)
    if mask is None:
        return row
    from mrsim.traits import diameter_map_px, frame_traits
    from skimage.morphology import skeletonize
    import cv2

    m = np.asarray(mask).astype(bool)
    H, W = m.shape
    frame_mm2 = (H / px_per_mm) * (W / px_per_mm)
    row.update(width_px=W, height_px=H)
    if not m.any():
        row.update({c: 0 for c in COLUMNS[13:]})
        return row
    t = frame_traits(m, px_per_mm, px_per_mm, frame_mm2)
    L = float(t["visible_length_mm"])
    skel = skeletonize(m)
    d = diameter_map_px(m)[skel] / px_per_mm                 # diameter at each centreline pixel (mm)
    li = L / max(len(d), 1)                                  # length each centreline pixel stands for
    classes = t["length_by_diameter_class"]["length_mm"]
    row.update(
        root_segments=max(0, cv2.connectedComponents(m.astype(np.uint8), connectivity=8)[0] - 1),
        root_length_mm=round(L, 3), root_length_cm=round(L / 10.0, 4),
        projected_area_mm2=round(float(t["projected_area_mm2"]), 3),
        surface_area_mm2=round(float(math.pi * d.sum() * li), 3),
        volume_mm3=round(float(math.pi / 4.0 * (d ** 2).sum() * li), 4),
        mean_diameter_mm=round(float(t["mean_diameter_mm"]), 4) if t["mean_diameter_mm"] == t["mean_diameter_mm"] else 0,
        root_coverage_pct=round(100.0 * float(m.mean()), 4),
        root_length_density_mm_per_cm2=round(float(t["planar_rld_mm_per_mm2"]) * 100.0, 4),
        n_endpoints=int(t["n_endpoints"]), n_branch_points=int(t["n_apparent_branch_points"]),
    )
    for (lo, hi), v in zip(zip(DIAM_EDGES_MM[:-1], DIAM_EDGES_MM[1:]), classes):
        row[_class_label(lo, hi)] = round(float(v), 3)
    return row


__all__ = ["COLUMNS", "DIAM_EDGES_MM", "image_row"]
