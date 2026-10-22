"""Simulation engine with fast 3D stand caching, legacy v3 comparison, and dataset export."""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import time
from dataclasses import asdict, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from mrsim import optics as O
from mrsim import soil as S
from mrsim import traits as T
from mrsim.diameter import DiameterSpec, apply_diameter, root_z

# Ensure mrsim.traits.skeletonize and mrsim.traits.ndi exist even without scikit-image (e.g. canopie-env)
from scipy import ndimage as _ndi
if not hasattr(T, "ndi"):
    T.ndi = _ndi
if not hasattr(T, "skeletonize"):
    def _fallback_skeletonize(mask: np.ndarray) -> np.ndarray:
        m = mask.astype(bool)
        if not m.any():
            return m
        dist = _ndi.distance_transform_edt(m)
        max_d = _ndi.maximum_filter(dist, size=3)
        return m & (dist >= max_d - 1e-4) & (dist > 0)

    T.skeletonize = _fallback_skeletonize
from mrsim.params import get_species, scale_tortuosity
from mrsim.scene import (
    DEFAULT_ROOT_PALETTE,
    FrameSpec,
    RootAppearance,
    grow_and_project,
    render_frame,
    sample_frame_specs,
    sample_root_palette,
)
from mrsim.tube import VisibilityModel, get_preset
from mrstyle import realbanks as RB

from .realism_params import DatasetGenConfig, RealismConfig, sample_randomized_config
from .soil_interaction import SoilContext, analyze_real_soil, composite_realism_frame
from .spline_roots import extract_smooth_ribbons

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class RealismSimulator:
    """Interactive and batch simulator for environment-specific root-on-real-soil synthesis."""

    def __init__(self, repo_root: Optional[str] = None, lazy_root_bank: bool = True):
        if repo_root is None:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.repo_root = repo_root
        self.bank_dir = os.path.join(repo_root, "realdata", "realbank_v1")
        self.root_bank = None
        self._root_bank_loaded = False
        if not lazy_root_bank:
            self.ensure_root_bank()

        self.soil_paths: List[str] = []
        soil_dir = os.path.join(self.bank_dir, "soil")
        if os.path.isdir(soil_dir):
            # Fast non-recursive directory listing instead of slow recursive glob on Google Drive
            self.soil_paths = sorted(
                os.path.join(soil_dir, f)
                for f in os.listdir(soil_dir)
                if f.lower().endswith(IMG_EXT)
            )

        # Cache of grown 3D systems keyed by growth geometry parameters
        self._stand_cache: Dict[tuple, tuple] = {}
        self._ribbon_cache: Dict[tuple, tuple] = {}
        self._soil_img_cache: Dict[str, np.ndarray] = {}

    @staticmethod
    def normalize_species(name: str) -> str:
        from mrsim.params import SPECIES
        key = str(name).strip().lower()
        aliases = {
            "soybean": "glycine_max",
            "maize": "zea_mays",
            "corn": "zea_mays",
            "wheat": "generic_fibrous",
            "ginger": "chicory",
        }
        key = aliases.get(key, key)
        if key not in SPECIES:
            return "switchgrass"
        return key

    def ensure_root_bank(self):
        """Load the real root strip bank on demand (with single-file pickle cache to avoid Google Drive I/O lag)."""
        if self._root_bank_loaded:
            return self.root_bank
        self._root_bank_loaded = True
        cache_pkl = os.path.join(os.path.dirname(__file__), "previews", "_root_bank_cache.pkl")
        if os.path.isfile(cache_pkl):
            try:
                import pickle
                with open(cache_pkl, "rb") as fh:
                    self.root_bank = pickle.load(fh)
                return self.root_bank
            except Exception:
                pass
        if os.path.isdir(os.path.join(self.bank_dir, "roots")):
            try:
                self.root_bank = RB.load_root_bank(self.bank_dir)
                os.makedirs(os.path.dirname(cache_pkl), exist_ok=True)
                import pickle
                with open(cache_pkl, "wb") as fh:
                    pickle.dump(self.root_bank, fh, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                self.root_bank = None
        return self.root_bank

    def load_user_soil_folder(
        self,
        folder: object,
        user_px_per_mm: float = 11.81,
        target_px_per_mm: float = 11.81,
        append: bool = False,
    ) -> List[str]:
        """Load soil images from a folder, a single file, or a list of files/folders."""
        if isinstance(folder, (list, tuple, set)):
            raw_items = list(folder)
        else:
            raw_items = [str(folder)]

        files: List[str] = []
        for item in raw_items:
            if os.path.isfile(item):
                if item.lower().endswith(IMG_EXT):
                    files.append(item)
            elif os.path.isdir(item):
                files.extend(
                    sorted(
                        p
                        for p in glob.glob(os.path.join(item, "**", "*"), recursive=True)
                        if p.lower().endswith(IMG_EXT)
                    )
                )

        valid_paths: List[str] = list(self.soil_paths) if append else []
        existing_set = set(valid_paths)
        if not append:
            self._soil_img_cache.clear()

        f_scale = float(target_px_per_mm) / max(float(user_px_per_mm), 1e-3)
        for p in files:
            if p in existing_set:
                continue
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is None:
                continue
            rgb = im[:, :, ::-1]
            if rgb.shape[0] > 520 or rgb.shape[1] > 680:
                rgb = RB.trim_borders(rgb)
            if abs(f_scale - 1.0) > 0.02:
                rgb = cv2.resize(
                    rgb,
                    None,
                    fx=f_scale,
                    fy=f_scale,
                    interpolation=cv2.INTER_AREA if f_scale < 1 else cv2.INTER_CUBIC,
                )
            crop = RB._cover_crop(rgb, wh=(640, 480), rng=np.random.default_rng(0), max_up=2.5)
            if crop is None:
                crop = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_CUBIC)
            self._soil_img_cache[p] = crop.astype(np.float32) / 255.0
            valid_paths.append(p)
            existing_set.add(p)

        if valid_paths:
            self.soil_paths = valid_paths
        return valid_paths

    def get_soil_rgb(self, index_or_path, flip_seed: Optional[int] = None) -> np.ndarray:
        """Return a 640x480 float32 RGB [0, 1] real soil image by index or file path."""
        if isinstance(index_or_path, int):
            if not self.soil_paths:
                raise RuntimeError("No soil images loaded.")
            path = self.soil_paths[index_or_path % len(self.soil_paths)]
        else:
            path = str(index_or_path)

        if path not in self._soil_img_cache:
            im = cv2.imread(path, cv2.IMREAD_COLOR)
            if im is None:
                raise IOError(f"Cannot read soil image: {path}")
            rgb = im[:, :, ::-1]
            if rgb.shape[1] != 640 or rgb.shape[0] != 480:
                crop = RB._cover_crop(rgb, wh=(640, 480), rng=np.random.default_rng(0), max_up=2.5)
                rgb = crop if crop is not None else cv2.resize(rgb, (640, 480))
            self._soil_img_cache[path] = rgb.astype(np.float32) / 255.0

        out = self._soil_img_cache[path].copy()
        if flip_seed is not None:
            rng = np.random.default_rng(flip_seed)
            if rng.random() < 0.5:
                out = out[:, ::-1]
            if rng.random() < 0.5:
                out = out[::-1, :]
        return np.ascontiguousarray(out)

    def _get_or_grow_stand(self, cfg: RealismConfig, stand_idx: int = 0):
        """Grow and project a 3D root stand, cached by growth parameters for instant GUI updates."""
        sp_name = self.normalize_species(cfg.species)
        key = (
            sp_name,
            round(float(cfg.days), 1),
            int(cfg.n_plants),
            round(float(cfg.spacing_cm), 2),
            round(float(cfg.row_offset_cm), 2),
            round(float(cfg.tortuosity_mult), 2),
            int(cfg.seed),
            int(stand_idx),
        )
        if key not in self._stand_cache:
            tube, cam = get_preset("cid_ci600_640")
            rng = np.random.default_rng(int(cfg.seed) + 7919 * int(stand_idx))
            plant = get_species(sp_name)
            if abs(cfg.tortuosity_mult - 1.0) > 1e-3:
                plant = scale_tortuosity(plant, cfg.tortuosity_mult)
            system, frame, proj0 = grow_and_project(
                plant,
                tube,
                cam,
                float(cfg.days),
                rng,
                n_plants=int(cfg.n_plants),
                spacing_cm=float(cfg.spacing_cm),
                row_offset_cm=tube.r_out + float(cfg.row_offset_cm),
                max_roots_per_plant=6500,
                cull_margin_cm=1.0,
                axial_range_cm=(0.0, 18.0),
                vis_bridge_mm=3.5,
            )
            specs = sample_frame_specs(
                45, cam, tube, rng, below_tube_bias=0.0, axial_range_cm=(0.0, 18.0)
            )
            # Score candidate specs by near-wall root length actually inside the camera's (u, v) FOV
            half_w_mm = 0.5 * cam.width_px / cam.px_per_mm_u
            half_h_mm = 0.5 * cam.height_px / cam.px_per_mm_v
            r_in_mm = tube.r_in * 10.0
            circ_mm = float(proj0.circumference_mm) if len(proj0) > 0 else (2.0 * np.pi * r_in_mm)
            near_wall = proj0.depth_mm[:, 0] < 1.25 if len(proj0) > 0 else np.zeros(0, dtype=bool)
            scored_specs = []
            for sp in specs:
                if len(proj0) == 0:
                    break
                v0 = sp.axial_cm * 10.0
                u0 = r_in_mm * sp.phi_center
                dv = np.abs(proj0.v_mm[:, 0] - v0)
                du = np.abs(((proj0.u_mm[:, 0] - u0 + 0.5 * circ_mm) % circ_mm) - 0.5 * circ_mm)
                in_fov = near_wall & (dv < half_h_mm) & (du < half_w_mm)
                score = float(proj0.seg_len3d_mm[in_fov].sum())
                if score > 8.0:
                    scored_specs.append((score, sp))
            scored_specs.sort(key=lambda item: -item[0])
            non_empty_specs = [sp for _, sp in scored_specs] if scored_specs else list(specs)
            if len(self._stand_cache) > 12:
                self._stand_cache.pop(next(iter(self._stand_cache)))
            self._stand_cache[key] = (system, frame, proj0, tube, cam, non_empty_specs)

        system, frame, proj0, tube, cam, specs = self._stand_cache[key]

        # Apply diameter multipliers & smooth along-root diameter variation quickly on proj0
        d_scale = float(getattr(cfg, "diam_scale", 1.0))
        axis_m = float(cfg.diam_axis_mult) * d_scale
        lat_m = float(cfg.diam_lat_mult) * d_scale
        min_m = float(getattr(cfg, "diam_min_mm", 0.05))
        max_m = float(getattr(cfg, "diam_max_mm", 3.20))
        dspec = DiameterSpec(
            axis_mu=axis_m,
            axis_sys_sd=0.0,
            lat_mu=lat_m,
            lat_sys_sd=0.0,
            root_sd=float(cfg.diam_root_sd),
            along_sd=float(cfg.diam_along_sd),
            along_corr_mm=float(cfg.diam_along_corr_mm),
            min_mm=min_m,
            max_mm=max_m,
        )
        drng = np.random.default_rng([int(cfg.seed), int(stand_idx), 2])
        proj = apply_diameter(
            proj0,
            dspec,
            axis_m,
            lat_m,
            root_z(len(system.roots), drng),
            rng_along=drng,
        )
        return system, frame, proj, tube, cam, specs

    def render_v4_frame(
        self,
        cfg: RealismConfig,
        soil_index_or_rgb=0,
        stand_idx: int = 0,
        frame_idx: int = 0,
    ) -> dict:
        """Render a single frame using the experimental v4 Realism pipeline."""
        if isinstance(soil_index_or_rgb, np.ndarray):
            bg_rgb = soil_index_or_rgb
        else:
            bg_rgb = self.get_soil_rgb(soil_index_or_rgb)

        sp_name = self.normalize_species(cfg.species)
        rib_key = (
            sp_name,
            round(float(cfg.days), 1),
            int(cfg.n_plants),
            round(float(cfg.tortuosity_mult), 2),
            round(float(cfg.diam_axis_mult), 2),
            round(float(cfg.diam_lat_mult), 2),
            # every diameter setting: the ribbons carry the radii (diam_scale was missing here,
            # so changing only it returned the previous diameters)
            round(float(getattr(cfg, "diam_scale", 1.0)), 3),
            round(float(getattr(cfg, "diam_min_mm", 0.05)), 3),
            round(float(getattr(cfg, "diam_max_mm", 3.20)), 3),
            round(float(cfg.diam_root_sd), 3),
            round(float(cfg.diam_along_sd), 3),
            round(float(cfg.diam_along_corr_mm), 2),
            round(float(cfg.root_keep), 3),
            round(float(cfg.spline_smooth_mm), 2),
            round(float(cfg.junction_collar_flare), 2),
            bool(cfg.connected_topology),
            int(cfg.seed),
            int(stand_idx),
            int(frame_idx),
        )
        if rib_key not in self._ribbon_cache:
            system, frame, proj, tube, cam, specs = self._get_or_grow_stand(cfg, stand_idx=stand_idx)
            spec = specs[int(frame_idx) % len(specs)]
            rng_r = np.random.default_rng([int(cfg.seed), int(stand_idx), int(frame_idx), 44])
            vis = VisibilityModel()
            ribbons, len3d, len2d = extract_smooth_ribbons(
                system, proj, cam, spec, tube, vis, cfg, rng_r
            )
            if len(self._ribbon_cache) > 24:
                self._ribbon_cache.pop(next(iter(self._ribbon_cache)))
            self._ribbon_cache[rib_key] = (ribbons, len3d, len2d, spec)

        ribbons, len3d, len2d, spec = self._ribbon_cache[rib_key]
        rng = np.random.default_rng([int(cfg.seed), int(stand_idx), int(frame_idx), 55])
        rb = self.ensure_root_bank() if cfg.use_real_root_bank else None
        out = composite_realism_frame(
            ribbons,
            bg_rgb,
            cfg,
            root_bank=rb,
            rng=rng,
        )
        out["len3d_mm"] = len3d
        out["len2d_mm"] = len2d
        out["spec"] = spec
        out["background"] = bg_rgb
        return out

    def render_legacy_v3_frame(
        self,
        cfg: RealismConfig,
        soil_index_or_rgb=0,
        stand_idx: int = 0,
        frame_idx: int = 0,
    ) -> dict:
        """Render the exact same 3D stand, frame pose, and soil using the legacy v3 pipeline.

        Used for direct side-by-side 'Before (v3) vs After (v4)' comparison.
        """
        system, frame, proj, tube, cam, specs = self._get_or_grow_stand(cfg, stand_idx=stand_idx)
        spec = specs[int(frame_idx) % len(specs)]

        if isinstance(soil_index_or_rgb, np.ndarray):
            bg_rgb = soil_index_or_rgb
        else:
            bg_rgb = self.get_soil_rgb(soil_index_or_rgb)

        rng = np.random.default_rng([int(cfg.seed), int(stand_idx), int(frame_idx), 44])
        vis = VisibilityModel()
        style = S.SoilStyle.sample(rng, cam.px_per_mm_u)
        tex = S.TextureCache.build((cam.height_px, cam.width_px), rng, style, cover=0.14)
        tex.stones = None
        tex.organics = None

        n_roots = 1 + max([int(r.index) for r in system.roots] + [0])
        palette = sample_root_palette(
            n_roots,
            DEFAULT_ROOT_PALETTE,
            np.random.default_rng([int(cfg.seed), int(stand_idx), 3]),
            dark_root_frac=cfg.dark_root_frac,
        )
        appear = RootAppearance.sample(rng)
        appear = replace(
            appear,
            palette=palette,
            min_delta=0.03,
            shading=0.5,
            texture=0.15,
            chroma_scale=0.55,
            vis_fade=0.3,
            cover_relief_px=4.0,
            min_planarity=0.35,
            soil_film=0.75,
            root_bank=(self.ensure_root_bank() if cfg.use_real_root_bank else None),
            tex_mix=0.7,
            min_opacity=0.35,
            root_keep=cfg.root_keep,
            keep_seed=int(cfg.seed + 17 * frame_idx),
            root_hairs=cfg.root_hairs_density,
            root_hair_p=cfg.root_hairs_prob,
            tex_max_mag=1.5,
        ).with_frame_context(target_n=0.4, polarity=1.0)

        ostyle = O.OpticsStyle.sample(rng, cam.px_per_mm_u).without_artefacts(1.0)
        fr = render_frame(
            system,
            tube,
            cam,
            spec,
            rng,
            vis=vis,
            appear=appear,
            soil_style=style,
            optics_style=ostyle,
            proj=proj,
            frame=frame,
            tex=tex,
            cover=0.14,
            background=bg_rgb.copy(),
            mask_mode="geometric",
            mask_min_width_px=1.0,
            width_traits=False,
        )
        return dict(
            rgb=fr.rgb,
            alpha=fr.alpha,
            mask=fr.mask,
            diameter_mm=fr.diameter_label_mm,
            background=bg_rgb,
        )

    def get_filtered_soil_indices(self, gen_cfg: DatasetGenConfig) -> List[int]:
        """Return candidate soil indices matching `gen_cfg.soil_mode` and user selection."""
        n_total = len(self.soil_paths)
        if n_total == 0:
            return [0]
        if gen_cfg.soil_mode == "current_soil_only":
            return [int(gen_cfg.current_soil_index) % n_total]
        sel = [int(i) for i in getattr(gen_cfg, "selected_soil_indices", ()) if 0 <= int(i) < n_total]
        if gen_cfg.soil_mode == "selected_groups" and gen_cfg.selected_soil_groups:
            groups_lower = [g.lower() for g in gen_cfg.selected_soil_groups]
            matched = []
            for idx, p in enumerate(self.soil_paths):
                bname = os.path.basename(p).lower()
                if any(g in bname for g in groups_lower):
                    matched.append(idx)
            if matched:
                if sel:
                    inter = [i for i in matched if i in set(sel)]
                    return inter if inter else matched
                return matched
        if sel:
            return sel
        return list(range(n_total))

    @staticmethod
    def detect_next_run_version(out_dir: str) -> str:
        """Scan out_dir for existing runs and return the next sequential run tag (e.g. 'r01', 'r02')."""
        if not os.path.isdir(out_dir):
            return "r01"
        found_nums = set()
        rgb_dir = os.path.join(out_dir, "rgb")
        if os.path.isdir(rgb_dir):
            for fn in os.listdir(rgb_dir):
                if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                m = re.match(r"^(?:r|run|v)(\d+)_", fn, re.IGNORECASE)
                if m:
                    found_nums.add(int(m.group(1)))
                elif re.match(r"^s\d+_\d+", fn, re.IGNORECASE):
                    found_nums.add(1)

        for fn in os.listdir(out_dir):
            if fn.lower().startswith("part_") and fn.lower().endswith(".zip"):
                m = re.match(r"^part_(?:r|run|v)?(\d+)_", fn, re.IGNORECASE)
                if m:
                    found_nums.add(int(m.group(1)))
                elif re.match(r"^part_\d+\.zip$", fn, re.IGNORECASE):
                    found_nums.add(1)

        traits_path = os.path.join(out_dir, "traits.csv")
        if os.path.isfile(traits_path):
            try:
                with open(traits_path, "r", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        rv = str(row.get("run_version", "")).strip()
                        if rv:
                            m = re.match(r"^(?:r|run|v)?(\d+)$", rv, re.IGNORECASE)
                            if m:
                                found_nums.add(int(m.group(1)))
                        else:
                            fid = str(row.get("id", "")).strip()
                            m = re.match(r"^(?:r|run|v)(\d+)_", fid, re.IGNORECASE)
                            if m:
                                found_nums.add(int(m.group(1)))
                            elif re.match(r"^s\d+_\d+", fid, re.IGNORECASE):
                                found_nums.add(1)
            except Exception:
                pass

        if not found_nums:
            return "r01"
        return f"r{max(found_nums) + 1:02d}"

    def generate_environment_dataset(
        self,
        cfg: RealismConfig,
        out_dir: str,
        n_systems: int = 10,
        frames_per_system: int = 20,
        empty_frame_frac: float = 0.25,
        gen_cfg: Optional[DatasetGenConfig] = None,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> str:
        """Generate a complete environment-specific training dataset (rgb, mask, diam, traits.csv).

        Supports full user control over:
          - Multi-run persistence: automatically tags files (e.g. r01, r02...) so multiple runs
            can be exported into the SAME folder without overwriting existing frames, part zips, or traits.csv.
          - Which soils are used (Current soil only, Selected soil groups, or All soils)
          - Which species are used (Current species only, or Random mix of selected species)
          - How parameters are varied (Exact GUI sliders, Controlled % jitter around GUI sliders,
            or Full Domain Randomization across all realistic biological/optical ranges)
        """
        if gen_cfg is None:
            gen_cfg = DatasetGenConfig(
                n_systems=int(n_systems),
                frames_per_system=int(frames_per_system),
                empty_frame_frac=float(empty_frame_frac),
            )
        else:
            n_systems = int(gen_cfg.n_systems)
            frames_per_system = int(gen_cfg.frames_per_system)
            empty_frame_frac = float(gen_cfg.empty_frame_frac)

        for sub in ("rgb", "mask", "diam", "alpha"):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

        # Multi-run detection and version tag determination
        append_runs = bool(getattr(gen_cfg, "append_runs", True))
        req_tag = str(getattr(gen_cfg, "run_version", "")).strip()

        if append_runs:
            if req_tag and req_tag.lower() not in ("auto", "default"):
                run_tag = re.sub(r"[^A-Za-z0-9_-]+", "_", req_tag).strip("_")
                if not run_tag:
                    run_tag = self.detect_next_run_version(out_dir)
            else:
                run_tag = self.detect_next_run_version(out_dir)
        else:
            if req_tag and req_tag.lower() not in ("auto", "default"):
                run_tag = re.sub(r"[^A-Za-z0-9_-]+", "_", req_tag).strip("_")
            else:
                run_tag = ""

        # Update gen_cfg with resolved run_version
        gen_cfg.run_version = run_tag

        # Save run-specific configuration backups and update latest config pointers
        if run_tag:
            cfg.save_json(os.path.join(out_dir, f"realism_config_{run_tag}.json"))
            with open(os.path.join(out_dir, f"dataset_gen_config_{run_tag}.json"), "w", encoding="utf-8") as fh:
                json.dump(asdict(gen_cfg), fh, indent=2)

        cfg.save_json(os.path.join(out_dir, "realism_config.json"))
        with open(os.path.join(out_dir, "dataset_gen_config.json"), "w", encoding="utf-8") as fh:
            json.dump(asdict(gen_cfg), fh, indent=2)

        # Load existing traits.csv rows if appending
        traits_path = os.path.join(out_dir, "traits.csv")
        existing_rows: List[Dict[str, object]] = []
        if append_runs and os.path.isfile(traits_path):
            try:
                with open(traits_path, "r", newline="", encoding="utf-8") as fh:
                    reader = csv.DictReader(fh)
                    for r in reader:
                        if not r.get("run_version"):
                            m = re.match(r"^(?:r|run|v)?(\d+)_", str(r.get("id", "")), re.IGNORECASE)
                            r["run_version"] = f"r{int(m.group(1)):02d}" if m else "legacy"
                        existing_rows.append(r)
            except Exception as exc:
                print(f"[Engine] Warning: could not parse existing traits.csv to append: {exc}")

        candidate_soils = self.get_filtered_soil_indices(gen_cfg)
        total_frames = int(n_systems) * int(frames_per_system)
        rows: List[Dict[str, object]] = []
        done = 0
        rng_master = np.random.default_rng(cfg.seed)

        for sys_i in range(int(n_systems)):
            # Sample stand-level config (species, days, tortuosity, etc.)
            sys_cfg = sample_randomized_config(cfg, gen_cfg, rng_master, for_new_stand=True)

            for f_j in range(int(frames_per_system)):
                soil_idx = int(candidate_soils[int(rng_master.integers(0, len(candidate_soils)))])
                flip_seed = int(cfg.seed + 1009 * sys_i + f_j) if gen_cfg.soil_flip_augment else None
                bg = self.get_soil_rgb(soil_idx, flip_seed=flip_seed)
                soil_file = (
                    os.path.basename(self.soil_paths[soil_idx % len(self.soil_paths)])
                    if self.soil_paths
                    else "custom"
                )

                # Optional soil brightness/contrast jitter for extra U-Net robustness
                if gen_cfg.soil_exposure_jitter > 1e-4:
                    ej = float(gen_cfg.soil_exposure_jitter)
                    gain = float(rng_master.uniform(1.0 - ej, 1.0 + ej))
                    bias = float(rng_master.uniform(-0.5 * ej, 0.5 * ej))
                    bg = np.clip(bg * gain + bias, 0.01, 0.99).astype(np.float32)

                is_empty_target = rng_master.random() < float(empty_frame_frac)
                if is_empty_target:
                    f_cfg = sys_cfg.copy(root_keep=0.0)
                    res = dict(
                        rgb=bg,
                        alpha=np.zeros(bg.shape[:2], dtype=np.float32),
                        mask=np.zeros(bg.shape[:2], dtype=bool),
                        diameter_mm=np.zeros(bg.shape[:2], dtype=np.float32),
                        len2d_mm=0.0,
                    )
                else:
                    # Sample frame-level appearance / burial / optics parameters
                    f_cfg = sample_randomized_config(sys_cfg, gen_cfg, rng_master, for_new_stand=False)
                    # preserve the stand's 3D growth parameters so stand_cache hits
                    f_cfg.species = sys_cfg.species
                    f_cfg.days = sys_cfg.days
                    f_cfg.tortuosity_mult = sys_cfg.tortuosity_mult
                    res = self.render_v4_frame(
                        f_cfg, soil_index_or_rgb=bg, stand_idx=sys_i, frame_idx=f_j
                    )

                if run_tag:
                    fid = f"{run_tag}_s{sys_i:03d}_{f_j:04d}"
                else:
                    fid = f"s{sys_i:03d}_{f_j:04d}"

                rgb_u8 = (np.clip(res["rgb"], 0.0, 1.0) * 255.0).astype(np.uint8)
                mask_u8 = res["mask"].astype(np.uint8) * 255
                alpha_u8 = (np.clip(res["alpha"], 0.0, 1.0) * 255.0).astype(np.uint8)
                diam_um = np.where(
                    res["mask"],
                    np.clip(np.rint(res["diameter_mm"] * 1000.0), 1, 65535),
                    0,
                ).astype(np.uint16)

                cv2.imwrite(os.path.join(out_dir, "rgb", fid + ".png"), rgb_u8[:, :, ::-1])
                cv2.imwrite(os.path.join(out_dir, "mask", fid + ".png"), mask_u8)
                cv2.imwrite(os.path.join(out_dir, "alpha", fid + ".png"), alpha_u8)
                cv2.imwrite(os.path.join(out_dir, "diam", fid + ".png"), diam_um)

                vis_mm = float(res["mask"].sum()) / max(f_cfg.px_per_mm * 4.0, 1.0)
                rows.append(
                    dict(
                        id=fid,
                        run_version=run_tag or "r01",
                        system=sys_i,
                        frame_in_system=f_j,
                        soil_file=soil_file,
                        species=f_cfg.species,
                        param_mode=gen_cfg.param_mode,
                        root_keep=round(float(f_cfg.root_keep), 3),
                        diam_axis_mult=round(float(f_cfg.diam_axis_mult), 3),
                        soil_burial=round(float(f_cfg.soil_burial_strength), 3),
                        soil_film=round(float(f_cfg.surface_soil_film), 3),
                        contrast_n=round(float(f_cfg.contrast_target_n), 3),
                        visible_length_mm=round(float(res.get("len2d_mm", vis_mm)), 2),
                        is_empty=bool(res["mask"].sum() < 20),
                        mask_px=int(res["mask"].sum()),
                    )
                )
                done += 1
                if progress_cb is not None:
                    progress_cb(done, total_frames, f"{fid} | Soil: {soil_file}")

            # Write part_{sys_i:03d}.zip (or part_{run_tag}_{sys_i:03d}.zip)
            # using ZIP_STORED so Colab can unpack 1000s of frames in seconds
            import zipfile

            if run_tag:
                part_zip = os.path.join(out_dir, f"part_{run_tag}_{sys_i:03d}.zip")
            else:
                part_zip = os.path.join(out_dir, f"part_{sys_i:03d}.zip")

            with zipfile.ZipFile(part_zip, "w", compression=zipfile.ZIP_STORED) as zf:
                for f_j in range(frames_per_system):
                    cur_fid = f"{run_tag}_s{sys_i:03d}_{f_j:04d}" if run_tag else f"s{sys_i:03d}_{f_j:04d}"
                    rp = os.path.join(out_dir, "rgb", cur_fid + ".png")
                    mp = os.path.join(out_dir, "mask", cur_fid + ".png")
                    if os.path.exists(rp) and os.path.exists(mp):
                        zf.write(rp, f"rgb/{cur_fid}.png")
                        zf.write(mp, f"mask/{cur_fid}.png")

            # Flush traits.csv incrementally after every 3D stand
            fieldnames = [
                "id",
                "run_version",
                "system",
                "frame_in_system",
                "soil_file",
                "species",
                "param_mode",
                "root_keep",
                "diam_axis_mult",
                "soil_burial",
                "soil_film",
                "contrast_n",
                "visible_length_mm",
                "is_empty",
                "mask_px",
            ]
            cur_ids = {r["id"] for r in rows}
            combined_rows = [r for r in existing_rows if r.get("id") not in cur_ids] + rows

            with open(traits_path, "w", newline="", encoding="utf-8") as fh:
                wr = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
                wr.writeheader()
                wr.writerows(combined_rows)

        return os.path.join(out_dir, "traits.csv")


def pack_dataset_to_zip_parts(dataset_dir: str, chunk_size: int = 250) -> int:
    """Pack an existing unpacked v4 dataset folder (rgb/ and mask/) into part_XXX.zip chunks
    using ZIP_STORED so Google Colab can extract thousands of frames in <10 seconds."""
    import glob
    import zipfile

    rgb_files = sorted(glob.glob(os.path.join(dataset_dir, "rgb", "*.png")))
    pairs = []
    for rp in rgb_files:
        bn = os.path.basename(rp)
        mp = os.path.join(dataset_dir, "mask", bn)
        if os.path.exists(mp):
            pairs.append((bn, rp, mp))

    n_parts = 0
    for idx in range(0, len(pairs), chunk_size):
        chunk = pairs[idx : idx + chunk_size]
        part_path = os.path.join(dataset_dir, f"part_{n_parts:03d}.zip")
        with zipfile.ZipFile(part_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for bn, rp, mp in chunk:
                zf.write(rp, f"rgb/{bn}")
                zf.write(mp, f"mask/{bn}")
        n_parts += 1
    return n_parts


