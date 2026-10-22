"""CSV rows of root traits: one per image, or one per separated root (the annotator's "Export CSV").

Per image, traits come from :func:`mrsim.traits.frame_traits` (step-weighted
skeleton length, length-weighted mean diameter from the distance transform,
length by diameter class, endpoints and apparent branch points), plus surface
area (pi x d x l) and volume (pi / 4 x d^2 x l) summed along the skeleton.

With a :class:`root_separation.Separation` (roots separated along the model's
centre lines) the length, diameter, surface, volume and diameter classes are
taken from the separated roots instead: where roots touch, the skeleton of the
blob they form grows spurious spokes and its width is the blob's, not a root's.
Per root, each separated root gets its own row.

An image without roots gets zeros.  The image metadata (site, tube, window,
date, time, session) is read from the file name (:func:`temporal.parse_frame`).
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


META = ["image_index", "image", "site", "tube", "window", "date", "time", "session", "source"]
COLUMNS: List[str] = (
    META + ["threshold", "width_px", "height_px", "px_per_mm", "length_source",
            "root_segments", "roots_separated", "root_length_mm", "root_length_cm", "projected_area_mm2",
            "surface_area_mm2", "volume_mm3", "mean_diameter_mm", "root_coverage_pct",
            "root_length_density_mm_per_cm2", "n_endpoints", "n_branch_points"]
    + [_class_label(lo, hi) for lo, hi in zip(DIAM_EDGES_MM[:-1], DIAM_EDGES_MM[1:])]
)
FIRST_TRAIT = COLUMNS.index("root_segments")
ROOT_COLUMNS: List[str] = META + ["centre_source", "px_per_mm", "root_id", "length_mm", "diameter_mm", "area_mm2",
                                  "surface_area_mm2", "volume_mm3", "centroid_x_px", "centroid_y_px"]


def _meta(path: str, index: int, source: str, columns: List[str]) -> Dict[str, object]:
    key = temporal.parse_frame(path)
    row: Dict[str, object] = {c: "" for c in columns}
    row.update(image_index=index, image=os.path.basename(path), source=source)
    if key is not None:
        row.update(site=key.site.strip("_- "), tube=key.tube, window=key.loc, date=key.date, time=key.time,
                   session=key.session)
    return row


def image_row(path: str, index: int, mask: Optional[np.ndarray], source: str, px_per_mm: float,
              threshold: Optional[float] = None, sep=None, length_source: str = "mask skeleton") -> Dict[str, object]:
    """Row for one image; ``mask`` None means nothing to measure (traits left empty).  With ``sep`` the
    length-based traits come from the separated roots (``length_source`` says from which map)."""
    row = _meta(path, index, source, COLUMNS)
    row.update(threshold="" if threshold is None else round(float(threshold), 4), px_per_mm=px_per_mm)
    if mask is None:
        return row
    from mrsim.traits import diameter_map_px, frame_traits
    from skimage.morphology import skeletonize
    import cv2

    m = np.asarray(mask).astype(bool)
    H, W = m.shape
    frame_mm2 = (H / px_per_mm) * (W / px_per_mm)
    row.update(width_px=W, height_px=H, length_source=length_source if sep is not None else "mask skeleton")
    if not m.any():
        row.update({c: 0 for c in COLUMNS[FIRST_TRAIT:]})
        return row
    t = frame_traits(m, px_per_mm, px_per_mm, frame_mm2)
    if sep is not None:
        from exp_realism.root_separation import root_rows
        rr = root_rows(sep, px_per_mm)
        Ls = np.array([r["length_mm"] for r in rr], float)
        ds = np.array([r["diameter_mm"] for r in rr], float)
        L = float(Ls.sum())
        surface = float(math.pi * (ds * Ls).sum())
        volume = float(math.pi / 4.0 * (ds ** 2 * Ls).sum())
        mean_d = float((ds * Ls).sum() / L) if L > 0 else 0.0
        classes = [float(Ls[(ds >= lo) & (ds < hi)].sum()) for lo, hi in zip(DIAM_EDGES_MM[:-1], DIAM_EDGES_MM[1:])]
        n_sep = len(rr)
    else:
        L = float(t["visible_length_mm"])
        skel = skeletonize(m)
        d = diameter_map_px(m)[skel] / px_per_mm             # diameter at each centreline pixel (mm)
        li = L / max(len(d), 1)                              # length each centreline pixel stands for
        surface = float(math.pi * d.sum() * li)
        volume = float(math.pi / 4.0 * (d ** 2).sum() * li)
        mean_d = float(t["mean_diameter_mm"]) if t["mean_diameter_mm"] == t["mean_diameter_mm"] else 0.0
        classes = t["length_by_diameter_class"]["length_mm"]
        n_sep = ""
    row.update(
        root_segments=max(0, cv2.connectedComponents(m.astype(np.uint8), connectivity=8)[0] - 1),
        roots_separated=n_sep,
        root_length_mm=round(L, 3), root_length_cm=round(L / 10.0, 4),
        projected_area_mm2=round(float(t["projected_area_mm2"]), 3),
        surface_area_mm2=round(surface, 3), volume_mm3=round(volume, 4), mean_diameter_mm=round(mean_d, 4),
        root_coverage_pct=round(100.0 * float(m.mean()), 4),
        root_length_density_mm_per_cm2=round(L / frame_mm2 * 100.0, 4),
        n_endpoints=int(t["n_endpoints"]), n_branch_points=int(t["n_apparent_branch_points"]),
    )
    for (lo, hi), v in zip(zip(DIAM_EDGES_MM[:-1], DIAM_EDGES_MM[1:]), classes):
        row[_class_label(lo, hi)] = round(float(v), 3)
    return row


def root_table(path: str, index: int, sep, source: str, px_per_mm: float, centre_source: str) -> List[Dict[str, object]]:
    """One row per separated root of an image (one row with root_id 0 and zeros when it has none)."""
    from exp_realism.root_separation import root_rows
    base = _meta(path, index, source, ROOT_COLUMNS)
    base.update(centre_source=centre_source, px_per_mm=px_per_mm)
    rows = []
    for r in (root_rows(sep, px_per_mm) if sep is not None else []):
        d, L = float(r["diameter_mm"]), float(r["length_mm"])
        rows.append({**base, **r, "surface_area_mm2": round(math.pi * d * L, 4),
                     "volume_mm3": round(math.pi / 4.0 * d * d * L, 5)})
    if not rows:
        rows.append({**base, "root_id": 0, "length_mm": 0, "diameter_mm": 0, "area_mm2": 0, "surface_area_mm2": 0,
                     "volume_mm3": 0, "centroid_x_px": "", "centroid_y_px": ""})
    return rows


__all__ = ["COLUMNS", "ROOT_COLUMNS", "DIAM_EDGES_MM", "image_row", "root_table"]
