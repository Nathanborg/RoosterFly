"""Configuration and environment presets for the experimental v4 root realism simulator."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Dict, Optional, Tuple


@dataclass
class RealismConfig:
    """All tunable parameters for environment-specific root simulation on real soil.

    Grouped into:
    1. Plant & Architecture (3D growth, diameters, tortuosity, connected subtrees)
    2. Root Epidermis & Appearance (color, 3D cylinder shading, stele, hairs, edge roughness)
    3. Soil-Root Physical Interaction (image-driven aggregate burial, surface soil film, contact shadow)
    4. Camera & Optical Matching (local illumination, white balance, depth defocus, PSF)
    """

    # --- 1. Plant & Architecture ---------------------------------------------
    species: str = "switchgrass"
    days: float = 28.0
    n_plants: int = 3
    spacing_cm: float = 6.0
    row_offset_cm: float = 1.0
    px_per_mm: float = 11.81

    # Connected subtree density control (0.02 = sparse single root, 1.0 = full stand)
    root_keep: float = 0.35
    connected_topology: bool = True      # keep parent-lateral hierarchy connected (no orphan stubs)

    # Diameter & morphology multipliers
    diam_scale: float = 1.0              # global diameter scale (0.2 = ultra-thin hair roots, 1.0 = normal, 2.5 = thick)
    diam_axis_mult: float = 1.35         # multiplier on order-0 main axes
    diam_lat_mult: float = 1.60          # multiplier on lateral branches
    diam_root_sd: float = 0.28           # log-normal variation between individual roots
    diam_along_sd: float = 0.06          # gentle organic swelling/narrowing along a root
    diam_along_corr_mm: float = 12.0     # long correlation length -> zero sausage beading
    diam_min_mm: float = 0.05            # minimum diameter floor in mm (0.03 mm ~ 0.35 px on CID scanner)
    diam_max_mm: float = 3.20

    # Curvature & junction geometry
    tortuosity_mult: float = 1.35        # overall biological waviness
    spline_smooth_mm: float = 1.80       # C2 smoothing scale to remove piecewise-linear kinks
    junction_collar_flare: float = 0.35  # basal thickening where a lateral emerges from parent

    # --- 2. Root Epidermis & Appearance --------------------------------------
    # Color & suberization (0 = young white/cream, 1 = mature tan/brown)
    root_age_tint: float = 0.35          # shift along young -> suberized ramp
    root_warmth: float = 0.45            # warm cream/tan vs neutral grey-white
    root_brightness: float = 1.05        # overall brightness relative to auto-contrast
    root_whiteness: float = 0.0          # boost pure chalky white / hyaline roots (0 = natural, 1 = stark white)
    root_brownness: float = 0.0          # deep suberization / tannin brown (0 = natural, 1 = deep woody brown)
    root_red_tint: float = 0.0           # red-brown / mahogany shift (-1 = olive/yellow, +1 = reddish brown)
    root_saturation: float = 1.0         # color saturation (0 = monochrome/achromatic, 1 = natural, 2 = vivid)
    contrast_target_n: float = 0.55      # target signal-to-noise contrast above soil std
    dark_root_frac: float = 0.12         # share of older/senescing dark roots

    # 3D cylindrical & internal anatomy
    cylinder_shading: float = 0.55       # 3D Lambertian + rim darkening across cylinder
    stele_strength: float = 0.22         # subtle central vascular cylinder (stele) optical core
    translucent_fine_mm: float = 0.22    # roots thinner than this show true cortical translucency
    min_fine_opacity: float = 0.48       # minimum opacity of the finest hyaline roots

    # Epidermal micro-texture & root hairs
    use_real_root_bank: bool = True      # blend real PRMI root texture strips at 1:1 longitudinal scale
    real_texture_mix: float = 0.55       # blend weight of real root bank epidermal detail
    striation_strength: float = 0.14     # longitudinal cortical cell-file striations
    mottle_strength: float = 0.16        # along-root suberin/epidermal blotchiness
    edge_roughness_px: float = 0.45      # organic micro-irregularity of the root boundary
    root_hairs_density: float = 1.8      # hairs per mm on each side (0 = off)
    root_hairs_prob: float = 0.45        # fraction of roots bearing root hairs
    root_hairs_length_mm: float = 0.35   # mean hair length in mm

    # --- 3. Soil-Root Physical Interaction -----------------------------------
    # Real-soil-driven burial (uses actual soil aggregates/pores instead of random noise)
    soil_burial_strength: float = 0.45   # fraction/strength of burial under real soil aggregates
    soil_aggregate_scale_mm: float = 1.8 # characteristic aggregate size in mm
    water_droplet_occl: float = 0.75     # let specular water droplets on tube wall sit in front

    # Opaque surface soil film (mottled dirt adhering to root, NOT ghost transparency)
    surface_soil_film: float = 0.50      # amount of local soil color/particles clinging to root
    flank_soil_crust: float = 0.45       # extra soil encrustation along cylinder flanks (|t| -> 1)

    # Contact crevice / air-gap shadow around root at the tube wall
    contact_shadow_strength: float = 0.32  # dark air-gap/crevice shadow along root-soil boundary
    contact_shadow_width_px: float = 2.8   # width of the contact crevice in pixels
    contact_shadow_asym: float = 0.25      # directional lighting asymmetry (-1..1)

    # --- 4. Camera & Optical Matching ----------------------------------------
    depth_fade_mm: float = 1.2           # depth behind tube wall where root blurs & dives into soil
    depth_blur_max_px: float = 2.4       # max defocus blur for deeper root stretches
    local_illum_match: float = 0.75      # inherit spatial lighting & vignetting from real soil image
    local_color_cast_match: float = 0.45 # inherit camera white-balance/tint from real soil image
    psf_sigma_px: float = 0.65           # camera optical point-spread function blur
    match_soil_noise: float = 0.50       # add sensor noise matching the real soil patch's grain

    seed: int = 2026

    # --- 5. Measured root colours (optional) -----------------------------------
    # Real roots' colour relative to the soil right around them (RGB root - ring, in [-1, 1]),
    # e.g. measured on annotated frames.  When set, each root draws one of these instead of the
    # colour sliders above; root_color_gain makes up for translucency/burial/blur that soften
    # the drawn colour before it is seen.
    root_color_deltas: list = field(default_factory=list)
    root_color_gain: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RealismConfig":
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "RealismConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def copy(self, **changes) -> "RealismConfig":
        return replace(self, **changes)


PRESET_ENVIRONMENTS: Dict[str, Dict[str, object]] = {
    "Balanced Field Scanner (Default)": dict(
        species="switchgrass",
        root_keep=0.35,
        diam_axis_mult=1.35,
        diam_lat_mult=1.60,
        tortuosity_mult=1.35,
        root_age_tint=0.35,
        cylinder_shading=0.55,
        stele_strength=0.22,
        soil_burial_strength=0.45,
        surface_soil_film=0.50,
        contact_shadow_strength=0.32,
        root_hairs_density=1.8,
    ),
    "Fine Grass Roots in Sandy/Loam Soil (Switchgrass / Fibrous)": dict(
        species="switchgrass",
        root_keep=0.45,
        diam_axis_mult=1.10,
        diam_lat_mult=1.30,
        tortuosity_mult=1.50,
        root_age_tint=0.25,
        root_warmth=0.40,
        contrast_target_n=0.50,
        cylinder_shading=0.45,
        stele_strength=0.28,
        soil_burial_strength=0.50,
        surface_soil_film=0.55,
        contact_shadow_strength=0.25,
        root_hairs_density=2.2,
    ),
    "Coarse Bright Roots in Dark Organic Soil (Papaya / Cotton)": dict(
        species="papaya",
        root_keep=0.30,
        diam_axis_mult=1.65,
        diam_lat_mult=1.80,
        tortuosity_mult=1.20,
        root_age_tint=0.20,
        root_brightness=1.20,
        contrast_target_n=0.95,
        cylinder_shading=0.65,
        stele_strength=0.15,
        soil_burial_strength=0.35,
        surface_soil_film=0.30,
        contact_shadow_strength=0.42,
        root_hairs_density=3.0,
        root_hairs_prob=0.65,
    ),
    "Subtle Suberized Roots in Wet Clay/Silty Soil (Ginger / Chicory)": dict(
        species="chicory",
        root_keep=0.35,
        diam_axis_mult=1.25,
        diam_lat_mult=1.45,
        tortuosity_mult=1.40,
        root_age_tint=0.60,
        root_warmth=0.60,
        root_brightness=0.92,
        contrast_target_n=0.38,
        dark_root_frac=0.25,
        cylinder_shading=0.50,
        soil_burial_strength=0.55,
        surface_soil_film=0.65,
        contact_shadow_strength=0.28,
        root_hairs_density=0.8,
        psf_sigma_px=0.85,
    ),
    "Dense Branching Dicot Stand (Sunflower / Soybean / Peanut)": dict(
        species="sunflower",
        root_keep=0.55,
        diam_axis_mult=1.45,
        diam_lat_mult=1.55,
        tortuosity_mult=1.35,
        junction_collar_flare=0.45,
        root_age_tint=0.30,
        cylinder_shading=0.55,
        stele_strength=0.25,
        soil_burial_strength=0.42,
        surface_soil_film=0.45,
        contact_shadow_strength=0.35,
        root_hairs_density=1.5,
    ),
}


@dataclass
class DatasetGenConfig:
    """Controls how soils, species, and parameters are sampled when exporting a dataset."""

    # Soil selection: "all_soils" | "current_soil_only" | "selected_groups" | "custom_selection"
    soil_mode: str = "all_soils"
    selected_soil_groups: Tuple[str, ...] = (
        "Cotton",
        "Papaya",
        "Peanut",
        "Sesame",
        "Sunflower",
        "Switchgrass",
    )
    selected_soil_indices: Tuple[int, ...] = ()  # Specific soil indices selected via multi-select checklist
    custom_soil_folder: str = ""                # Path to active custom soil folder if loaded
    current_soil_index: int = 0
    soil_flip_augment: bool = True
    soil_exposure_jitter: float = 0.06  # random brightness/contrast jitter on soil background

    # Species selection: "current_only" | "selected_species"
    species_mode: str = "current_only"
    selected_species: Tuple[str, ...] = (
        "switchgrass",
        "cotton",
        "papaya",
        "sunflower",
        "peanut",
        "sesame",
        "chicory",
        "glycine_max",
        "zea_mays",
    )

    # Parameter sampling mode:
    #   "exact_gui"         -> Use exact slider values from GUI (0% parameter jitter)
    #   "jitter_around_gui" -> Center on GUI sliders and apply per-group jitter [0..1]
    #   "full_random"       -> Sample across full realistic domain ranges
    param_mode: str = "jitter_around_gui"

    # Per-category jitter amounts (0.0 = locked to GUI sliders, 1.0 = wide variation)
    arch_jitter: float = 0.30
    epidermis_jitter: float = 0.30
    soil_physics_jitter: float = 0.30
    optics_jitter: float = 0.20

    # Dataset size & negative frame share
    n_systems: int = 8
    frames_per_system: int = 15
    empty_frame_frac: float = 0.20

    # Multi-run persistence & naming
    run_version: str = ""       # e.g. "r01", "wet_soil", or "" for auto-detect (r01, r02, ...)
    append_runs: bool = True    # keep existing runs in folder without overwriting, append to traits.csv


def sample_randomized_config(
    base_cfg: RealismConfig,
    gen_cfg: DatasetGenConfig,
    rng,
    for_new_stand: bool = False,
) -> RealismConfig:
    """Sample a RealismConfig according to the user's DatasetGenConfig rules."""
    import numpy as np

    cfg = base_cfg.copy()

    # 1. Species selection
    if gen_cfg.species_mode == "selected_species" and len(gen_cfg.selected_species) > 0:
        idx = int(rng.integers(0, len(gen_cfg.selected_species)))
        cfg.species = str(gen_cfg.selected_species[idx])

    mode = gen_cfg.param_mode
    if mode == "exact_gui":
        return cfg

    def _jit_lin(val: float, span: float, j: float, lo: float, hi: float) -> float:
        if j <= 1e-4:
            return float(np.clip(val, lo, hi))
        delta = float(rng.uniform(-span * j, span * j))
        return float(np.clip(val + delta, lo, hi))

    def _jit_mult(val: float, max_log: float, j: float, lo: float, hi: float) -> float:
        if j <= 1e-4:
            return float(np.clip(val, lo, hi))
        factor = float(np.exp(rng.uniform(-max_log * j, max_log * j)))
        return float(np.clip(val * factor, lo, hi))

    if mode == "full_random":
        # Full domain randomization across broad realistic biological & optical intervals
        if for_new_stand:
            cfg.days = float(rng.uniform(14.0, 44.0))
            cfg.tortuosity_mult = float(rng.uniform(0.85, 1.85))
        cfg.root_keep = float(np.clip(np.exp(rng.uniform(np.log(0.08), np.log(0.85))), 0.05, 1.0))
        cfg.diam_scale = float(np.clip(np.exp(rng.uniform(np.log(0.35), np.log(1.85))), 0.20, 2.20))
        cfg.diam_axis_mult = float(rng.uniform(0.40, 2.10))
        cfg.diam_lat_mult = float(rng.uniform(0.30, 2.10))
        cfg.diam_min_mm = float(rng.uniform(0.02, 0.10))
        cfg.spline_smooth_mm = float(rng.uniform(0.80, 2.80))
        cfg.junction_collar_flare = float(rng.uniform(0.10, 0.65))
        cfg.edge_roughness_px = float(rng.uniform(0.15, 0.85))

        cfg.root_brightness = float(rng.uniform(0.50, 1.65))
        cfg.root_whiteness = float(rng.uniform(0.0, 0.85)) if rng.random() < 0.35 else 0.0
        cfg.root_brownness = float(rng.uniform(0.0, 0.85)) if rng.random() < 0.35 else 0.0
        cfg.root_red_tint = float(rng.uniform(-0.4, 0.4))
        cfg.root_saturation = float(rng.uniform(0.4, 1.5))
        cfg.contrast_target_n = float(rng.uniform(0.28, 1.05))
        cfg.root_age_tint = float(rng.uniform(0.08, 0.85))
        cfg.root_warmth = float(rng.uniform(0.15, 0.85))
        cfg.dark_root_frac = float(rng.uniform(0.0, 0.40))
        cfg.cylinder_shading = float(rng.uniform(0.25, 0.80))
        cfg.stele_strength = float(rng.uniform(0.05, 0.45))
        cfg.real_texture_mix = float(rng.uniform(0.20, 0.85))
        cfg.root_hairs_density = float(rng.uniform(0.0, 3.5))

        cfg.soil_burial_strength = float(rng.uniform(0.15, 0.72))
        cfg.soil_aggregate_scale_mm = float(rng.uniform(0.9, 3.2))
        cfg.surface_soil_film = float(rng.uniform(0.15, 0.78))
        cfg.flank_soil_crust = float(rng.uniform(0.15, 0.75))
        cfg.contact_shadow_strength = float(rng.uniform(0.10, 0.55))
        cfg.contact_shadow_width_px = float(rng.uniform(1.5, 4.8))
        cfg.water_droplet_occl = float(rng.uniform(0.20, 0.90))

        cfg.local_illum_match = float(rng.uniform(0.45, 0.95))
        cfg.local_color_cast_match = float(rng.uniform(0.20, 0.75))
        cfg.match_soil_noise = float(rng.uniform(0.25, 0.85))
        cfg.depth_fade_mm = float(rng.uniform(0.6, 2.2))
        cfg.depth_blur_max_px = float(rng.uniform(0.8, 3.5))
        cfg.psf_sigma_px = float(rng.uniform(0.35, 1.25))
        return cfg

    # Default: "jitter_around_gui" -> controlled variation centered on GUI sliders
    ja = float(np.clip(gen_cfg.arch_jitter, 0.0, 1.0))
    je = float(np.clip(gen_cfg.epidermis_jitter, 0.0, 1.0))
    js = float(np.clip(gen_cfg.soil_physics_jitter, 0.0, 1.0))
    jo = float(np.clip(gen_cfg.optics_jitter, 0.0, 1.0))

    if for_new_stand:
        cfg.days = _jit_mult(cfg.days, 0.40, ja, 10.0, 50.0)
        cfg.tortuosity_mult = _jit_mult(cfg.tortuosity_mult, 0.35, ja, 0.60, 2.20)

    cfg.root_keep = _jit_mult(cfg.root_keep, 0.85, ja, 0.04, 1.00)
    cfg.diam_scale = _jit_mult(cfg.diam_scale, 0.35, ja, 0.15, 2.50)
    cfg.diam_axis_mult = _jit_mult(cfg.diam_axis_mult, 0.38, ja, 0.15, 2.50)
    cfg.diam_lat_mult = _jit_mult(cfg.diam_lat_mult, 0.38, ja, 0.10, 2.50)
    cfg.diam_min_mm = _jit_lin(cfg.diam_min_mm, 0.02, ja, 0.02, 0.25)
    cfg.spline_smooth_mm = _jit_lin(cfg.spline_smooth_mm, 0.80, ja, 0.20, 4.00)
    cfg.junction_collar_flare = _jit_lin(cfg.junction_collar_flare, 0.25, ja, 0.0, 0.85)
    cfg.edge_roughness_px = _jit_lin(cfg.edge_roughness_px, 0.35, ja, 0.0, 1.20)

    cfg.root_brightness = _jit_lin(cfg.root_brightness, 0.35, je, 0.30, 2.00)
    cfg.root_whiteness = _jit_lin(cfg.root_whiteness, 0.25, je, 0.0, 1.0)
    cfg.root_brownness = _jit_lin(cfg.root_brownness, 0.25, je, 0.0, 1.0)
    cfg.root_red_tint = _jit_lin(cfg.root_red_tint, 0.20, je, -1.0, 1.0)
    cfg.root_saturation = _jit_lin(cfg.root_saturation, 0.25, je, 0.0, 2.0)
    cfg.contrast_target_n = _jit_mult(cfg.contrast_target_n, 0.50, je, 0.15, 1.40)
    cfg.root_age_tint = _jit_lin(cfg.root_age_tint, 0.35, je, 0.0, 1.0)
    cfg.root_warmth = _jit_lin(cfg.root_warmth, 0.30, je, 0.0, 1.0)
    cfg.dark_root_frac = _jit_lin(cfg.dark_root_frac, 0.20, je, 0.0, 0.60)
    cfg.cylinder_shading = _jit_lin(cfg.cylinder_shading, 0.25, je, 0.0, 1.0)
    cfg.stele_strength = _jit_lin(cfg.stele_strength, 0.20, je, 0.0, 0.65)
    cfg.real_texture_mix = _jit_lin(cfg.real_texture_mix, 0.30, je, 0.0, 1.0)
    cfg.root_hairs_density = _jit_lin(cfg.root_hairs_density, 1.40, je, 0.0, 4.5)

    cfg.soil_burial_strength = _jit_lin(cfg.soil_burial_strength, 0.30, js, 0.0, 0.90)
    cfg.soil_aggregate_scale_mm = _jit_lin(cfg.soil_aggregate_scale_mm, 0.90, js, 0.6, 4.0)
    cfg.surface_soil_film = _jit_lin(cfg.surface_soil_film, 0.32, js, 0.0, 0.95)
    cfg.flank_soil_crust = _jit_lin(cfg.flank_soil_crust, 0.32, js, 0.0, 0.95)
    cfg.contact_shadow_strength = _jit_lin(cfg.contact_shadow_strength, 0.22, js, 0.0, 0.75)
    cfg.contact_shadow_width_px = _jit_lin(cfg.contact_shadow_width_px, 1.50, js, 1.0, 7.0)
    cfg.water_droplet_occl = _jit_lin(cfg.water_droplet_occl, 0.30, js, 0.0, 1.0)

    cfg.local_illum_match = _jit_lin(cfg.local_illum_match, 0.25, jo, 0.0, 1.0)
    cfg.local_color_cast_match = _jit_lin(cfg.local_color_cast_match, 0.25, jo, 0.0, 1.0)
    cfg.match_soil_noise = _jit_lin(cfg.match_soil_noise, 0.25, jo, 0.0, 1.0)
    cfg.depth_fade_mm = _jit_lin(cfg.depth_fade_mm, 0.60, jo, 0.3, 3.0)
    cfg.depth_blur_max_px = _jit_lin(cfg.depth_blur_max_px, 1.10, jo, 0.0, 4.5)
    cfg.psf_sigma_px = _jit_lin(cfg.psf_sigma_px, 0.45, jo, 0.0, 2.0)

    return cfg

