"""Real-soil-driven burial, contact shadows, opaque epidermal texturing, and optical matching.

Replaces the legacy simulator's uncorrelated procedural occlusion and flat alpha-blending
with physically grounded interactions between the 3D root cylinder, the real soil image,
and the minirhizotron tube wall.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

from mrsim import raster as R

from .realism_params import RealismConfig
from .spline_roots import RootRibbon, evaluate_ribbon_local_box

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclass
class SoilContext:
    """Extracted physical & optical properties of a real root-free soil image."""

    bg_rgb: np.ndarray          # (H, W, 3) float32 in [0, 1]
    luma_mean: float
    luma_std: float
    noise_sigma: float          # estimated high-frequency camera sensor noise std
    illum_field: np.ndarray     # (H, W) float32 normalized smooth illumination map (~0.5..1.5)
    color_cast: np.ndarray      # (H, W, 3) float32 local chromatic tint ratio
    soil_local_rgb: np.ndarray  # (H, W, 3) float32 medium-scale local soil color for surface films
    soil_fine_grain: np.ndarray # (H, W, 3) float32 high-frequency soil grain for crusting
    burial_field: np.ndarray    # (H, W) float32 continuous soil-aggregate prominence field
    droplet_mask: np.ndarray    # (H, W) float32 [0, 1] specular water droplets on tube wall


def analyze_real_soil(
    bg_rgb: np.ndarray,
    cfg: RealismConfig,
    rng: np.random.Generator,
) -> SoilContext:
    """Analyze a real soil background image to drive burial, lighting, and surface films."""
    bg = np.clip(np.asarray(bg_rgb, dtype=np.float32), 0.0, 1.0)
    h, w = bg.shape[:2]
    luma = bg @ _LUMA
    luma_mean = float(max(luma.mean(), 0.02))
    luma_std = float(max(luma.std(), 0.01))

    # 1. Estimate high-frequency sensor noise via median absolute deviation of Laplacian
    lap = cv2.Laplacian(luma, cv2.CV_32F, ksize=3)
    noise_sigma = float(np.median(np.abs(lap)) / 4.2)
    noise_sigma = float(np.clip(noise_sigma, 0.002, 0.035))

    # 2. Smooth spatial illumination & vignetting map (captures LED hotspots & dark corners)
    sig_illum = max(32.0, 0.14 * max(h, w))
    luma_smooth = cv2.GaussianBlur(luma, (0, 0), sig_illum)
    illum_raw = luma_smooth / max(luma_mean, 1e-3)
    illum_field = (1.0 + float(cfg.local_illum_match) * (illum_raw - 1.0)).astype(np.float32)
    illum_field = np.clip(illum_field, 0.35, 1.75)

    # 3. Local camera white-balance / color-cast map
    bg_smooth = cv2.GaussianBlur(bg, (0, 0), sig_illum)
    cast_raw = bg_smooth / np.maximum(luma_smooth[..., None], 0.03)
    # Normalize cast so frame-mean luma of cast is 1
    cast_luma = cast_raw @ _LUMA
    cast_raw /= np.maximum(cast_luma[..., None], 0.1)
    color_cast = (1.0 + float(cfg.local_color_cast_match) * (cast_raw - 1.0)).astype(np.float32)
    color_cast = np.clip(color_cast, 0.65, 1.45)

    # 4. Medium-scale soil color and fine soil grain (for surface soil film on roots)
    soil_local_rgb = cv2.GaussianBlur(bg, (0, 0), 4.5)
    soil_blur_small = cv2.GaussianBlur(bg, (0, 0), 1.4)
    soil_fine_grain = (bg - soil_blur_small).astype(np.float32)

    # 5. Detect true specular water droplets / condensation rims on the glass wall
    # True water droplets are bright, near-achromatic (low saturation) specular highlights
    luma_med = cv2.GaussianBlur(luma, (0, 0), 12.0)
    local_diff = luma - luma_med
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy)
    grad_env = cv2.GaussianBlur(grad_mag, (0, 0), 2.0)

    c_max = np.max(bg, axis=2)
    c_min = np.min(bg, axis=2)
    sat = (c_max - c_min) / np.maximum(c_max, 1e-3)
    achromatic = np.clip((0.25 - sat) / 0.15, 0.0, 1.0)
    bright_gate = np.clip((luma - max(0.62, luma_mean + 2.2 * luma_std)) / 0.15, 0.0, 1.0)

    spec_score = (local_diff / max(luma_std, 0.02)) * (grad_env / max(float(grad_env.mean()), 1e-3))
    droplet_mask = (np.clip((spec_score - 2.5) / 2.5, 0.0, 1.0) * achromatic * bright_gate).astype(np.float32)
    if droplet_mask.any():
        droplet_mask = cv2.GaussianBlur(droplet_mask, (0, 0), 1.0)

    # 6. Real-soil-driven aggregate burial field
    # Combines actual soil clod prominence with sharp pixel-level boundary gating so bright clods
    # on dark soil never dilate a dark halo around themselves when occluding a root!
    agg_px = max(6.0, float(cfg.soil_aggregate_scale_mm) * float(cfg.px_per_mm))
    luma_fine = cv2.GaussianBlur(luma, (0, 0), 1.0)
    luma_agg = cv2.GaussianBlur(luma, (0, 0), 0.30 * agg_px)
    luma_broad = cv2.GaussianBlur(luma, (0, 0), 1.6 * agg_px)

    # Gate aggregate prominence by fine-scale luminance so clod footprints never bleed outside clod edges
    clod_core = np.minimum(luma_fine, luma_agg)
    clod_relief = (clod_core - luma_broad) / max(luma_std, 0.015)
    pos_gate = np.clip(clod_relief + 0.15, 0.0, 1.0)
    tex_energy = cv2.GaussianBlur(grad_mag, (0, 0), 0.25 * agg_px)
    tex_norm = ((tex_energy - tex_energy.mean()) / max(float(tex_energy.std()), 1e-4)) * pos_gate

    # Add a smooth random phase modulation so the same soil image can bury slightly different
    # subsets of aggregates across different synthetic frames while staying locked to real clods
    gh = max(4, int(round(h / (0.8 * agg_px))))
    gw = max(4, int(round(w / (0.8 * agg_px))))
    coarse_mod = rng.normal(0.0, 1.0, size=(gh, gw)).astype(np.float32)
    mod_full = cv2.resize(coarse_mod, (w, h), interpolation=cv2.INTER_CUBIC)
    mod_full = cv2.GaussianBlur(mod_full, (0, 0), 0.5 * agg_px)
    mod_full /= max(float(mod_full.std()), 1e-4)

    burial_field = (0.62 * clod_relief + 0.16 * tex_norm + 0.22 * mod_full * pos_gate).astype(np.float32)
    burial_field = (burial_field - burial_field.mean()) / max(float(burial_field.std()), 1e-5)

    return SoilContext(
        bg_rgb=bg,
        luma_mean=luma_mean,
        luma_std=luma_std,
        noise_sigma=noise_sigma,
        illum_field=illum_field,
        color_cast=color_cast,
        soil_local_rgb=soil_local_rgb,
        soil_fine_grain=soil_fine_grain,
        burial_field=burial_field,
        droplet_mask=droplet_mask,
    )


def _sample_root_base_colors(
    rid: int,
    cfg: RealismConfig,
    soil_ctx: SoilContext,
    force_not_dark: bool = False,
    local_rgb: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Deterministic per-root (tip_rgb, mature_rgb, is_dark) adapted to the soil context.

    With ``cfg.root_color_deltas`` (measured real roots), the root takes one measured
    root-minus-soil colour on top of the soil around it (``local_rgb``, else the frame mean).
    """
    deltas = getattr(cfg, "root_color_deltas", None)
    if deltas:
        rng_p = np.random.default_rng([int(cfg.seed), int(rid), 405])
        d = np.asarray(deltas[int(rng_p.integers(len(deltas)))], np.float32)[:3]
        d = d * float(getattr(cfg, "root_color_gain", 1.0))
        if local_rgb is not None:
            base = np.asarray(local_rgb, np.float32)
        elif getattr(soil_ctx, "bg_rgb", None) is not None:
            base = soil_ctx.bg_rgb.reshape(-1, 3).mean(0).astype(np.float32)
        else:                                                # colour swatch preview: neutral soil of that luma
            base = np.full(3, float(getattr(soil_ctx, "luma_mean", 0.35)), np.float32)
        old_c = np.clip(base + d, 0.02, 0.98).astype(np.float32)
        tip_c = np.clip(base + 1.1 * d + 0.02, 0.02, 0.98).astype(np.float32)   # tips a touch paler
        return tip_c, old_c, bool(float(d @ _LUMA) < 0.0)
    rng_r = np.random.default_rng([int(cfg.seed), int(rid), 404])
    u_age = float(rng_r.uniform(-0.15, 0.15))
    u_warm = float(rng_r.uniform(-0.10, 0.10))
    u_dark = float(rng_r.random())

    whiteness = float(np.clip(getattr(cfg, "root_whiteness", 0.0), 0.0, 1.0))
    brownness = float(np.clip(getattr(cfg, "root_brownness", 0.0), 0.0, 1.0))
    red_tint = float(np.clip(getattr(cfg, "root_red_tint", 0.0), -1.0, 1.0))
    sat = float(np.clip(getattr(cfg, "root_saturation", 1.0), 0.0, 2.5))
    brightness = float(np.clip(getattr(cfg, "root_brightness", 1.05), 0.25, 2.20))

    # A root can be dark if dark_root_frac triggers or if brownness is very high (unless whiteness is requested)
    if force_not_dark or whiteness > 0.25:
        is_dark = False
    else:
        is_dark = (u_dark < float(cfg.dark_root_frac) and soil_ctx.luma_mean > 0.22) or (brownness > 0.70 and rng_r.random() < 0.40)

    warmth = float(np.clip(cfg.root_warmth + u_warm, 0.0, 1.0))
    age_tint = float(np.clip(cfg.root_age_tint + u_age, 0.0, 1.0))

    # 1. Base biological chroma vectors (Young meristem tip vs mature zone)
    tip_c = np.array([
        0.93 + 0.04 * warmth,
        0.91 - 0.03 * warmth,
        0.85 - 0.16 * warmth,
    ], dtype=np.float32)
    old_c = np.array([
        0.75 + 0.05 * warmth - 0.28 * age_tint,
        0.63 - 0.05 * warmth - 0.28 * age_tint,
        0.47 - 0.16 * warmth - 0.24 * age_tint,
    ], dtype=np.float32)

    # 2. Whiteness boost: blend toward chalky pure white (0.98, 0.98, 0.97)
    if whiteness > 0.0:
        pure_white_tip = np.array([0.98, 0.98, 0.97], dtype=np.float32)
        pure_white_old = np.array([0.95, 0.94, 0.93], dtype=np.float32)
        tip_c = (1.0 - whiteness) * tip_c + whiteness * pure_white_tip
        old_c = (1.0 - whiteness) * old_c + whiteness * pure_white_old

    # 3. Brownness boost: deepen suberization into rich earthy chocolate/woody brown
    if brownness > 0.0:
        woody_tip = np.array([0.76, 0.62, 0.42], dtype=np.float32)
        woody_old = np.array([0.42, 0.26, 0.14], dtype=np.float32)
        tip_c = (1.0 - brownness) * tip_c + brownness * woody_tip
        old_c = (1.0 - brownness) * old_c + brownness * woody_old

    # 4. Red-Brown / Mahogany tint (-1 = olive/yellowish, +1 = reddish brown)
    if abs(red_tint) > 1e-3:
        for c_arr in (tip_c, old_c):
            c_arr[0] = np.clip(c_arr[0] + 0.10 * red_tint, 0.05, 0.99)
            c_arr[1] = np.clip(c_arr[1] - 0.05 * red_tint, 0.05, 0.99)
            c_arr[2] = np.clip(c_arr[2] - 0.08 * red_tint, 0.05, 0.99)

    # 5. Saturation control (0 = monochrome/achromatic, 1 = natural, 2 = rich vivid)
    if sat != 1.0:
        for c_arr in (tip_c, old_c):
            lum = float(c_arr @ _LUMA)
            c_arr[:] = np.clip(lum + sat * (c_arr - lum), 0.02, 0.99)

    # 6. Adapt luminance to soil_ctx & root_brightness
    soil_l = soil_ctx.luma_mean
    soil_s = max(soil_ctx.luma_std, 0.03)
    delta_l = float(cfg.contrast_target_n) * soil_s * brightness

    if is_dark:
        # Senescing / dark suberized root: sits darker than pale/medium soil
        target_old_l = max(0.06, soil_l - max(0.08, 0.85 * delta_l))
        target_tip_l = max(0.10, target_old_l + 0.08)
    else:
        # Standard root:
        # When brownness is high, allow root to be darker than soil if brightness is low
        # When whiteness is high, root stays bright and high-contrast
        lift_factor = max(0.0, 1.0 - 1.25 * brownness)
        min_lift = soil_l + max(0.02, delta_l) * lift_factor
        base_old_l = float(old_c @ _LUMA) * brightness
        base_tip_l = float(tip_c @ _LUMA) * brightness
        if whiteness > 0.0:
            base_tip_l = max(base_tip_l, 0.85 * brightness)
            base_old_l = max(base_old_l, 0.75 * brightness)
        target_old_l = float(np.clip(max(base_old_l, min_lift) if brownness < 0.5 else base_old_l, 0.08, 0.98))
        target_tip_l = float(np.clip(max(base_tip_l, target_old_l + 0.04), 0.12, 0.99))

    def _set_luma(c: np.ndarray, tl: float) -> np.ndarray:
        cur_l = float(c @ _LUMA)
        ch = c - cur_l
        # Scale chroma slightly down at extreme luma to stay in [0.02, 0.98] without hue shift
        scale = 1.0
        for v in ch:
            if v > 1e-6 and tl + v > 0.98:
                scale = min(scale, (0.98 - tl) / v)
            elif v < -1e-6 and tl + v < 0.02:
                scale = min(scale, (0.02 - tl) / v)
        scale = max(0.0, scale)
        return np.clip(tl + scale * ch, 0.02, 0.98).astype(np.float32)

    return _set_luma(tip_c, target_tip_l), _set_luma(old_c, target_old_l), is_dark


def compute_preview_root_colors(
    cfg: RealismConfig,
    soil_luma_mean: float = 0.35,
    soil_luma_std: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (tip_rgb_uint8, mature_rgb_uint8) for UI swatch display."""
    class _DummyCtx:
        luma_mean = float(soil_luma_mean)
        luma_std = float(soil_luma_std)

    tip, mat, _ = _sample_root_base_colors(0, cfg, _DummyCtx(), force_not_dark=True)  # type: ignore
    tip_u8 = np.clip(tip * 255.0 + 0.5, 0, 255).astype(np.uint8)
    mat_u8 = np.clip(mat * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return tip_u8, mat_u8


def _synthesize_ribbon_texture(
    rib: RootRibbon,
    box_data: dict,
    soil_ctx: SoilContext,
    ysl: slice,
    xsl: slice,
    cfg: RealismConfig,
    root_bank: Optional[object],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize realistic 3D cylindrical root epidermis, stele, and surface soil film.

    Returns `(root_rgb_box, alpha_box, geom_box)` over `(ysl, xsl)`.
    """
    cov_alpha = box_data["cov_alpha"]
    cov_geom = box_data["cov_geom"]
    t = box_data["t_coord"]          # in [-1, 1] across root
    s_px = box_data["s_px"]          # arc-length in px along root
    r_px = box_data["r_px"]          # local radius in px
    diam_mm = box_data["diam_mm"]
    depth_mm = box_data["depth_mm"]
    age_days = box_data["age_days"]
    bh, bw = cov_alpha.shape

    local_rgb = None
    if getattr(cfg, "root_color_deltas", None):
        local_rgb = np.median(soil_ctx.soil_local_rgb[ysl, xsl].reshape(-1, 3), axis=0)
    tip_rgb, old_rgb, _ = _sample_root_base_colors(rib.root_id, cfg, soil_ctx, local_rgb=local_rgb)

    # 1. Age progression along the root (paler meristematic tip -> browner mature base)
    age_f = np.clip(age_days / 24.0, 0.0, 1.0) ** 0.75
    if rib.is_apical_tip:
        # Near the physical tip, keep unsuberized pale coloration
        total_s = float(rib.arc_s[-1])
        dist_tip_mm = np.maximum(total_s - s_px, 0.0) / max(cfg.px_per_mm, 1.0)
        tip_zone = np.clip(dist_tip_mm / 3.5, 0.0, 1.0)
        age_f = age_f * tip_zone

    # If user configured high whiteness, keep the entire root white along its length
    whiteness = float(getattr(cfg, "root_whiteness", 0.0))
    if whiteness > 0.0:
        age_f = age_f * (1.0 - 0.85 * whiteness)

    base_col = (
        tip_rgb[None, None, :] * (1.0 - age_f[..., None])
        + old_rgb[None, None, :] * age_f[..., None]
    ).astype(np.float32)

    # 2. 3D Cylindrical Shading + Central Vascular Stele (Internal Anatomy)
    abs_t = np.clip(np.abs(t), 0.0, 1.0)
    # Lambertian cylinder profile with directional lighting bias matching shadow_asym
    light_t = np.clip(t - 0.18 * float(cfg.contact_shadow_asym), -1.0, 1.0)
    cyl_lambert = np.sqrt(np.clip(1.0 - light_t * light_t, 0.0, 1.0))

    # Gentle 3D cylinder roundness (capped so edges never turn into a dark black outline!)
    rim_shade = 1.0 - 0.36 * float(cfg.cylinder_shading) * (1.0 - cyl_lambert)

    # Central vascular stele: subtle warm denser core (|t| < 0.32) without dark side troughs
    if cfg.stele_strength > 0.0:
        stele_core = np.exp(-0.5 * (t / 0.28) ** 2)
        stele_vis = np.clip(1.2 - r_px / 14.0, 0.25, 1.0)
        anat_profile = 1.0 + float(cfg.stele_strength) * stele_vis * (0.28 * stele_core - 0.08 * (1.0 - stele_core))
    else:
        stele_core = np.zeros_like(t)
        anat_profile = np.ones_like(t)

    shade_3d = (rim_shade * anat_profile).astype(np.float32)

    # 3. Real Root Bank Epidermal Texture (at 1:1 longitudinal scale, no magnification bands!)
    tex_mod = np.ones((bh, bw, 3), dtype=np.float32)
    med_w_px = float(np.median(2.0 * r_px))
    if cfg.use_real_root_bank and root_bank is not None and len(root_bank) > 0 and med_w_px >= 2.5:
        u_pick = float(((rib.root_id * 2654435761) & 0xFFFFFFFF) / 2**32)
        tex_id = root_bank.pick(med_w_px, u_pick)
        strip = root_bank.strips[tex_id]
        ls, ks = strip.shape[:2]

        # Sample longitudinal coordinate at 1:1 physical pixel scale (no stretching!)
        row_f = np.mod(s_px + float((rib.root_id * 37) % ls), 2.0 * ls)
        row_idx = np.where(row_f < ls, row_f, 2.0 * ls - 1.0 - row_f).astype(np.int32).clip(0, ls - 1)

        # Sample inner 72% of strip columns to avoid donor soil border
        col_lo, col_hi = 0.16 * (ks - 1), 0.84 * (ks - 1)
        col_idx = np.rint(col_lo + 0.5 * (t + 1.0) * (col_hi - col_lo)).astype(np.int32).clip(0, ks - 1)

        raw_tex = strip[row_idx, col_idx]
        prof_1d = root_bank.profiles[tex_id][col_idx]
        prof_l = np.maximum(prof_1d @ _LUMA, 0.05)[..., None]
        detail_ratio = np.clip(raw_tex / prof_l, 0.65, 1.42)

        mix = float(cfg.real_texture_mix) * np.clip((med_w_px - 2.0) / 3.0, 0.2, 1.0)
        tex_mod = (1.0 - mix) + mix * detail_ratio

    # 4. Procedural Longitudinal Cortical Striations & Suberin Mottling
    if cfg.striation_strength > 0.0 or cfg.mottle_strength > 0.0:
        ph0 = float((rib.root_id * 1.17) % (2 * math.pi))
        ph1 = float((rib.root_id * 2.63) % (2 * math.pi))
        stri = (
            0.6 * np.sin(9.5 * t + 0.06 * s_px + ph0)
            + 0.4 * np.sin(17.0 * t - 0.09 * s_px + ph1)
        )
        mottle = (
            0.55 * np.sin(0.045 * s_px + 1.8 * t + ph1)
            + 0.45 * np.cos(0.019 * s_px - 1.2 * t + ph0)
        )
        epiderm_mult = np.exp(
            float(cfg.striation_strength) * 0.45 * stri
            + float(cfg.mottle_strength) * 0.55 * mottle
        ).astype(np.float32)
    else:
        epiderm_mult = np.ones_like(t)

    root_col = base_col * shade_3d[..., None] * tex_mod * epiderm_mult[..., None]
    root_col = np.clip(root_col, 0.02, 0.98)

    # 5. Real-Soil Aggregate Burial (Coherent Longitudinal Stretches + Flank Encroachment)
    # A cylindrical root has bending stiffness: its spine (|t| < 0.7) only buries behind coherent
    # soil peds larger than the root diameter (never small Swiss-cheese holes inside a thick root!).
    burial_sub = soil_ctx.burial_field[ysl, xsl]
    soil_exact = soil_ctx.bg_rgb[ysl, xsl]
    soil_loc = soil_ctx.soil_local_rgb[ysl, xsl]
    soil_grain = soil_ctx.soil_fine_grain[ysl, xsl]

    if cfg.soil_burial_strength > 0.01:
        w_px = 2.0 * r_px
        # Smooth burial field at a scale proportional to root width so small specks cannot punch interior holes
        ped_sigma = float(np.clip(1.6 * med_w_px, 3.5, 22.0))
        burial_ped = cv2.GaussianBlur(burial_sub, (0, 0), ped_sigma)
        # Flanks (|t| > 0.65) can still feel finer crumb encroachment
        flank_zone = np.clip((abs_t - 0.60) / 0.40, 0.0, 1.0)
        burial_eff = burial_ped * (1.0 - 0.45 * flank_zone) + burial_sub * (0.45 * flank_zone)

        relief = 1.0 / (1.0 + (w_px / 6.5) ** 2.0)
        q_base = float(np.clip(1.0 - 0.50 * float(cfg.soil_burial_strength), 0.18, 0.98))
        thr_base = float(np.quantile(burial_ped, q_base))
        spine_resist = np.clip((w_px - 6.0) / 8.0, 0.0, 0.75) * (1.0 - flank_zone)
        thr = thr_base + 0.30 * (1.0 - relief) + 0.35 * spine_resist
        raw_buried = 0.5 - 0.5 * np.tanh((thr - burial_eff) / 0.22)

        # Gate burial by local aggregate brightness so recessed dark pores never occlude roots
        soil_loc_luma = soil_loc @ _LUMA
        fg_floor = soil_ctx.luma_mean - 0.10 * soil_ctx.luma_std
        fg_gate = np.clip((soil_loc_luma - fg_floor) / max(0.40 * soil_ctx.luma_std, 0.02), 0.0, 1.0)
        buried_frac = np.clip(raw_buried * fg_gate, 0.0, 1.0)
        trans_burial = (1.0 - buried_frac).astype(np.float32)
    else:
        trans_burial = np.ones_like(burial_sub)

    if cfg.water_droplet_occl > 0.0:
        drop_sub = soil_ctx.droplet_mask[ysl, xsl]
        trans_burial = trans_burial * (1.0 - 0.75 * float(cfg.water_droplet_occl) * drop_sub)

    # 6. Opaque Surface Soil Film & Flank Crusting (Replaces Ghost Transparency!)
    soil_crust_rgb = np.clip(0.75 * soil_loc + 0.25 * np.maximum(soil_loc, soil_exact) + 0.55 * soil_grain, 0.02, 0.98)

    patch_wave = 0.5 + 0.5 * np.sin(0.07 * s_px + 2.5 * t + float(rib.root_id))
    flank_f = (abs_t ** 2.2) * float(cfg.flank_soil_crust) * (0.35 + 0.65 * patch_wave)
    depth_f = np.clip(depth_mm / max(cfg.depth_fade_mm, 0.3), 0.0, 1.0) * 0.40
    burial_fringe = (1.0 - trans_burial) * 0.60
    film_f = float(cfg.surface_soil_film) * (0.25 + 0.75 * patch_wave) * np.clip(3.5 / np.maximum(r_px, 1.5), 0.3, 1.0)

    crust_weight = np.clip(flank_f + depth_f + burial_fringe + 0.50 * film_f, 0.0, 0.85).astype(np.float32)
    root_col = root_col * (1.0 - crust_weight[..., None]) + soil_crust_rgb * crust_weight[..., None]

    # 7. Apply Local Camera Illumination & Color Cast from the Real Soil Patch
    illum_sub = soil_ctx.illum_field[ysl, xsl]
    cast_sub = soil_ctx.color_cast[ysl, xsl]
    root_col = np.clip(root_col * illum_sub[..., None] * cast_sub, 0.01, 0.99)

    # 8. Material Opacity + Cortical Flank Feathering:
    # Fine roots are slightly more translucent in their outer cortex (|t| > 0.45) than in their central stele!
    fine_lim = max(float(cfg.translucent_fine_mm), 0.05)
    fine_t = np.clip(diam_mm / (1.8 * fine_lim), 0.0, 1.0)
    core_opacity = float(cfg.min_fine_opacity) + (0.96 - float(cfg.min_fine_opacity)) * (fine_t ** 0.7)
    # Softly feather cylinder grazing edges (|t| -> 1) so root boundary blends organically into soil
    flank_feather = 1.0 - (0.32 - 0.14 * fine_t) * (abs_t ** 2.4)
    mat_opacity = core_opacity * flank_feather

    alpha_box = (cov_alpha * mat_opacity * trans_burial).astype(np.float32)
    geom_box = np.where(trans_burial >= 0.42, cov_geom, 0.0).astype(np.float32)

    return root_col.astype(np.float32), alpha_box, geom_box


def _render_organic_root_hairs(
    comp: np.ndarray,
    ribbons: List[RootRibbon],
    burial_field: np.ndarray,
    illum_field: np.ndarray,
    cfg: RealismConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw fine, slightly curved, translucent root hairs emerging from the root epidermis.

    Root hairs only appear in open soil pores (not on top of foreground soil clods) and
    stay out of the binary segmentation mask (since annotators trace the root cylinder).
    """
    if cfg.root_hairs_density <= 0.0 or not ribbons:
        return comp

    h, w = comp.shape[:2]
    hair_col = np.zeros((h, w, 3), dtype=np.float32)
    hair_alpha = np.zeros((h, w), dtype=np.float32)

    px_mm = float(cfg.px_per_mm)
    burial_thr = float(np.quantile(burial_field, np.clip(1.0 - 0.5 * cfg.soil_burial_strength, 0.2, 0.95)))

    for rib in ribbons:
        rng_h = np.random.default_rng([int(cfg.seed), int(rib.root_id), 707])
        if rng_h.random() > float(cfg.root_hairs_prob):
            continue
        med_r = float(np.median(rib.radii_px))
        if med_r < 0.7 or med_r > 9.0:
            continue

        total_len_mm = float(rib.arc_s[-1]) / max(px_mm, 1.0)
        if total_len_mm < 0.8:
            continue

        tip_c, old_c, is_dark = _sample_root_base_colors(rib.root_id, cfg, SoilContext(
            bg_rgb=comp, luma_mean=float((comp @ _LUMA).mean()), luma_std=0.1, noise_sigma=0.01,
            illum_field=illum_field, color_cast=np.ones_like(comp), soil_local_rgb=comp,
            soil_fine_grain=np.zeros_like(comp), burial_field=burial_field, droplet_mask=np.zeros((h, w), np.float32)
        ))
        if is_dark:
            continue
        h_rgb = np.clip(0.75 * tip_c + 0.25 * old_c, 0.15, 0.96)

        # Root hair zone is typically behind the meristematic tip (1.5 mm from tip back to mature zone)
        n_hairs = int(rng_h.poisson(float(cfg.root_hairs_density) * total_len_mm * 2.0))
        n_hairs = min(n_hairs, 220)
        if n_hairs <= 0:
            continue

        pts = rib.pts
        nrms = rib.normals
        arc_s = rib.arc_s
        radii = rib.radii_px
        m = len(pts)

        s_min = 0.0
        s_max = float(arc_s[-1]) - (1.2 * px_mm if rib.is_apical_tip else 0.0)
        if s_max <= s_min + 2.0:
            continue

        layer_a = np.zeros((h, w), dtype=np.uint8)
        for _ in range(n_hairs):
            s_val = float(rng_h.uniform(s_min, s_max))
            idx = int(np.clip(np.searchsorted(arc_s, s_val) - 1, 0, m - 2))
            p0 = pts[idx]
            nrm = nrms[idx]
            r_here = float(radii[idx])

            side = 1.0 if rng_h.random() < 0.5 else -1.0
            ang = float(rng_h.normal(0.0, 0.35))
            ca, sa = math.cos(ang), math.sin(ang)
            dx = side * (nrm[0] * ca - nrm[1] * sa)
            dy = side * (nrm[0] * sa + nrm[1] * ca)

            bx = float(p0[0] + dx * r_here * 0.92)
            by = float(p0[1] + dy * r_here * 0.92)
            ix, iy = int(round(bx)), int(round(by))
            if not (0 <= ix < w and 0 <= iy < h):
                continue
            # Do not spawn hairs under foreground soil aggregates
            if burial_field[iy, ix] > burial_thr:
                continue

            hlen = float(rng_h.gamma(2.2, 0.5 * float(cfg.root_hairs_length_mm) * px_mm))
            hlen = min(max(hlen, 1.5), 1.4 * px_mm)

            # Slight organic curl at hair midpoint
            curl = float(rng_h.normal(0.0, 0.25)) * hlen
            mx = bx + 0.5 * hlen * dx - 0.5 * curl * dy
            my = by + 0.5 * hlen * dy + 0.5 * curl * dx
            ex = bx + hlen * dx - curl * dy
            ey = by + hlen * dy + curl * dx

            op = int(rng_h.uniform(35, 95))
            poly = np.array([
                [int(round(bx * 4)), int(round(by * 4))],
                [int(round(mx * 4)), int(round(my * 4))],
                [int(round(ex * 4)), int(round(ey * 4))],
            ], dtype=np.int32)
            cv2.polylines(layer_a, [poly], False, op, 1, cv2.LINE_AA, shift=2)

        if layer_a.any():
            af = (layer_a.astype(np.float32) / 255.0) * illum_field
            need = af > hair_alpha
            hair_alpha = np.maximum(hair_alpha, af)
            for c in range(3):
                hair_col[..., c] = np.where(need, h_rgb[c] * illum_field, hair_col[..., c])

    if not hair_alpha.any():
        return comp
    return np.clip(comp * (1.0 - hair_alpha[..., None]) + hair_col * hair_alpha[..., None], 0.0, 1.0).astype(np.float32)


def composite_realism_frame(
    ribbons: List[RootRibbon],
    bg_rgb: np.ndarray,
    cfg: RealismConfig,
    root_bank: Optional[object],
    rng: np.random.Generator,
    soil_ctx: Optional[SoilContext] = None,
) -> dict:
    """Render and composite all smooth root ribbons onto a real soil background."""
    if soil_ctx is None:
        soil_ctx = analyze_real_soil(bg_rgb, cfg, rng)

    h, w = soil_ctx.bg_rgb.shape[:2]
    alpha_acc = np.zeros((h, w), dtype=np.float32)
    rgb_acc = np.zeros((h, w, 3), dtype=np.float32)
    trans = np.ones((h, w), dtype=np.float32)

    cover = np.zeros((h, w), dtype=np.float32)
    cover_diam_mm = np.zeros((h, w), dtype=np.float32)
    blur_map = np.zeros((h, w), dtype=np.float32)
    shadow_acc = np.zeros((h, w), dtype=np.float32)
    instance = np.full((h, w), -1, dtype=np.int32)
    order_map = np.zeros((h, w), dtype=np.int32)

    # Sort ribbons front-to-back by median depth so nearer roots occlude deeper ones
    sorted_ribbons = sorted(ribbons, key=lambda r: float(np.median(r.depth_mm)))

    for rib in sorted_ribbons:
        res = evaluate_ribbon_local_box(
            rib,
            width=w,
            height=h,
            edge_roughness_px=cfg.edge_roughness_px,
            shadow_width_px=cfg.contact_shadow_width_px,
            shadow_asym=cfg.contact_shadow_asym,
            depth_fade_mm=cfg.depth_fade_mm,
            depth_blur_max_px=cfg.depth_blur_max_px,
            cover_min_radius_px=0.5,
        )
        if res is None:
            continue
        ysl, xsl, box_data = res

        root_col, a_box, geom_box = _synthesize_ribbon_texture(
            rib, box_data, soil_ctx, ysl, xsl, cfg, root_bank
        )

        # Accumulate contact shadow outside the root
        shadow_acc[ysl, xsl] = np.maximum(shadow_acc[ysl, xsl], box_data["shadow"])

        # Update geometric mask coverage
        c_sub = cover[ysl, xsl]
        take_g = geom_box > c_sub
        if take_g.any():
            cover[ysl, xsl] = np.where(take_g, geom_box, c_sub)
            cover_diam_mm[ysl, xsl] = np.where(take_g, box_data["diam_mm"], cover_diam_mm[ysl, xsl])

        if not a_box.any():
            continue

        t_view = trans[ysl, xsl]
        contrib = a_box * t_view

        alpha_acc[ysl, xsl] += contrib
        rgb_acc[ysl, xsl] += contrib[..., None] * root_col
        blur_map[ysl, xsl] += contrib * box_data["blur_px"]

        take_i = contrib > 0.15
        instance[ysl, xsl] = np.where(take_i & (instance[ysl, xsl] < 0), np.int32(rib.root_id), instance[ysl, xsl])
        order_map[ysl, xsl] = np.where(take_i & (order_map[ysl, xsl] == 0), np.int32(rib.order + 1), order_map[ysl, xsl])

        trans[ysl, xsl] = t_view * (1.0 - a_box)

    alpha_acc = np.clip(alpha_acc, 0.0, 1.0)
    lit = alpha_acc > 1e-5
    root_layer = np.zeros_like(soil_ctx.bg_rgb)
    root_layer[lit] = rgb_acc[lit] / alpha_acc[lit, None]
    blur_norm = np.zeros((h, w), dtype=np.float32)
    blur_norm[lit] = blur_map[lit] / alpha_acc[lit]

    # 1. Apply Pore-Weighted Contact Crevice Shadow to the soil background around roots
    bg_shaded = soil_ctx.bg_rgb.copy()
    if cfg.contact_shadow_strength > 0.0 and shadow_acc.any():
        # Contact shadows are deepest where the root bridges over darker soil pores/gaps
        soil_luma = soil_ctx.bg_rgb @ _LUMA
        pore_mod = np.clip(1.25 - 0.55 * (soil_luma / max(soil_ctx.luma_mean, 0.08)), 0.35, 1.15)
        sh_eff = shadow_acc * pore_mod * np.clip(1.0 - 0.85 * alpha_acc, 0.0, 1.0)
        sh_mult = 1.0 - 0.55 * float(cfg.contact_shadow_strength) * np.clip(sh_eff, 0.0, 1.0)
        bg_shaded = bg_shaded * sh_mult[..., None]

    # 2. Composite root layer onto shaded soil
    comp = bg_shaded * (1.0 - alpha_acc[..., None]) + root_layer * alpha_acc[..., None]

    # 3. Render organic curved root hairs into open soil pores
    if cfg.root_hairs_density > 0.0 and sorted_ribbons:
        comp = _render_organic_root_hairs(
            comp, sorted_ribbons, soil_ctx.burial_field, soil_ctx.illum_field, cfg, rng
        )

    # 4. Apply depth-dependent defocus blur & camera PSF to the root contribution only
    # (The real soil background already has the camera's native PSF and sensor noise!)
    psf_sig = float(cfg.psf_sigma_px)
    if psf_sig > 0.05 or blur_norm.max() > 0.25:
        comp_psf = cv2.GaussianBlur(comp, (0, 0), max(psf_sig, 0.4))
        comp_deep = cv2.GaussianBlur(comp, (0, 0), max(psf_sig + 1.6, 1.5))
        deep_w = np.clip(blur_norm / max(float(cfg.depth_blur_max_px), 0.5), 0.0, 1.0)[..., None]
        blurred_root = comp_psf * (1.0 - deep_w) + comp_deep * deep_w

        # Blend blurred root + edge transition (dilated alpha band) so background soil stays 100% sharp!
        edge_band = cv2.GaussianBlur(alpha_acc, (0, 0), max(psf_sig + 0.8, 0.8))
        blend_w = np.clip(edge_band * 1.15, 0.0, 1.0)[..., None]
        comp = bg_shaded * (1.0 - blend_w) + blurred_root * blend_w

    # 5. Add matched sensor grain over the rendered roots (so roots aren't cleaner than the soil!)
    if cfg.match_soil_noise > 0.0:
        grain_sd = soil_ctx.noise_sigma * float(cfg.match_soil_noise)
        noise = rng.normal(0.0, grain_sd, size=comp.shape).astype(np.float32)
        root_zone = cv2.GaussianBlur(alpha_acc, (0, 0), 1.2)[..., None]
        comp = np.clip(comp + noise * np.sqrt(np.clip(root_zone, 0.0, 1.0)), 0.0, 1.0)

    mask = cover >= 0.5
    diam_label = R.diameter_on_mask(cover_diam_mm, mask)

    return dict(
        rgb=np.clip(comp, 0.0, 1.0).astype(np.float32),
        alpha=alpha_acc,
        mask=mask,
        diameter_mm=diam_label,
        order=order_map,
        instance=instance,
        soil_ctx=soil_ctx,
        ribbons=sorted_ribbons,
    )
