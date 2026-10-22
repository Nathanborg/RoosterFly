"""Root diameter realism, applied to an already-projected system.

v2 frames were 78% 1-2 px roots (2 x EDT at the skeleton), while real roots
are ~8 px wide (Ginger B FWHM) and PRMI Switchgrass masks have a median w2e of
~10 px.  Two causes: the absolute taper in ``grow.radius_at`` (fixed in the v3
presets) and one fixed diameter per root type, so every lateral2 of every
plant had the same width.  Real root systems vary at four levels, and each one
is a separate multiplier here:

* per STAND, per class (axes = order 0, laterals = order >= 1): soil,
  genotype and age move a whole system thicker or thinner -- lognormal
  ``axis_mu * exp(axis_sys_sd * z)`` and the same for laterals;
* per ROOT: lognormal ``exp(root_sd * z_root)`` (median 1);
* ALONG a root: slow swelling and thinning, a stationary AR(1) process in
  arc length with correlation length ``along_corr_mm`` and log-sd ``along_sd``;
* then clipped to ``[min_mm, max_mm]``: a floor (~1 px) so no root is drawn
  thinner than the pixel it is labelled with, and ``max_mm = 2.5`` (~30 px)
  in v3, because 3-5 mm taproots are 35-60 px wide -- wider than any PRMI p90
  -- and the CUT geometry audit failed 91% of v2 frames with wide roots.

Why on the PROJECTED system: diameter does not feed back into growth (radius
is drawn at birth and only read by the renderer), so a new width needs no
regrowth.  Width enters the render only through the visibility model
(``p_visible`` rises with diameter: a thicker root stays visible deeper) and
opacity.  That is what makes a 48-spec calibration grid cheap: one growth per
stand, every spec rendered from the same projection with the same random
numbers.

Random streams (all separate from the main ``seed + 7919*sys`` stream, so a
diameter spec never changes geometry, soil or framing):
``[seed, sys, 1]`` -> the system multipliers (drawn in ``ginger.draw_system``);
``[seed, sys, 2]`` -> per-root ``z_root`` then the along-root noise.  An
identity spec returns the projection object itself.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class DiameterSpec:
    """Diameter multipliers.  The defaults are the identity (no change)."""

    axis_mu: float = 1.0          # order-0 roots (primary + basal/nodal/crown)
    axis_sys_sd: float = 0.0      # per-stand log-sd of the axis multiplier
    lat_mu: float = 1.0           # order >= 1
    lat_sys_sd: float = 0.0
    root_sd: float = 0.0          # per-root lognormal log-sd
    along_sd: float = 0.0         # along-root log-sd (slow swelling/thinning)
    along_corr_mm: float = 4.0    # its correlation length (3D arc length)
    min_mm: float = 0.0           # floor (v3: 0.08 mm ~ 1 px); 0 = none
    max_mm: float = 0.0           # cap (v3: 2.5 mm ~ 30 px); 0 = none

    @classmethod
    def from_any(cls, spec) -> Optional["DiameterSpec"]:
        """A spec, a dict of its fields (as stored in SynthConfig / config.json)
        or None.  Unknown keys raise, so a typo cannot silently do nothing."""
        if spec is None or isinstance(spec, cls):
            return spec
        names = {f.name for f in fields(cls)}
        bad = set(spec) - names
        if bad:
            raise KeyError("unknown DiameterSpec fields: %s" % sorted(bad))
        return cls(**{k: float(v) for k, v in spec.items()})

    def to_dict(self) -> dict:
        return asdict(self)

    def is_identity(self) -> bool:
        return (self.axis_mu == 1.0 and self.lat_mu == 1.0 and self.axis_sys_sd == 0.0
                and self.lat_sys_sd == 0.0 and self.root_sd == 0.0
                and self.along_sd == 0.0 and self.min_mm <= 0.0 and self.max_mm <= 0.0)


def system_multipliers(spec: DiameterSpec, rng=None,
                       z: Optional[Sequence[float]] = None) -> Tuple[float, float]:
    """``(m_axis, m_lat)`` for one stand.

    Always two standard normals, drawn from ``rng`` unless ``z`` is given --
    the calibration grid passes the SAME ``z`` to every spec (common random
    numbers), so specs differ by their parameters, not by their draws.
    """
    if z is None:
        z = rng.standard_normal(2)
    z0, z1 = float(z[0]), float(z[1])
    return (float(spec.axis_mu * math.exp(spec.axis_sys_sd * z0)),
            float(spec.lat_mu * math.exp(spec.lat_sys_sd * z1)))


def root_z(n_roots: int, rng) -> np.ndarray:
    """One standard normal per root id (``len(system.roots)`` of them)."""
    return rng.standard_normal(int(max(n_roots, 0)))


def _along_factor(proj, spec: DiameterSpec, rng) -> Tuple[np.ndarray, np.ndarray]:
    """``exp(along_sd * x)`` at the start and end node of every segment.

    ``x`` is a unit-variance AR(1) process along each root (segments of a
    root are contiguous in a ProjectedSystem), lag-1 correlation
    ``exp(-ds / along_corr_mm)`` with ``ds`` the median segment length.  It is
    computed for all roots in ONE ``lfilter`` pass over the concatenated
    stream, then each root's carry-over from the previous root is removed and
    a stationary start added:

        y_k = x_k + rho^(k-k0+1) * (z_root - x_(k0-1))

    so ``Var(y) = 1`` from the first segment on and roots are independent.
    A segment's END uses the next segment's value when that segment belongs
    to the same root, so joints stay continuous.
    """
    from scipy.signal import lfilter
    n = len(proj)
    rid = np.asarray(proj.root_id)
    ds = float(np.median(proj.seg_len3d_mm)) if n else 0.2
    rho = math.exp(-max(ds, 1e-6) / max(float(spec.along_corr_mm), 1e-6))
    b = math.sqrt(max(1.0 - rho * rho, 0.0))
    e = rng.standard_normal(n)
    x = lfilter([b], [1.0, -rho], e)
    start = np.ones(n, bool)
    start[1:] = rid[1:] != rid[:-1]
    idx = np.arange(n)
    k0 = np.maximum.accumulate(np.where(start, idx, 0))
    prev = np.where(k0 > 0, x[np.maximum(k0 - 1, 0)], 0.0)
    z0 = rng.standard_normal(int(start.sum()))
    g = np.cumsum(start) - 1
    with np.errstate(under="ignore"):
        y = x + rho ** (idx - k0 + 1).astype(np.float64) * (z0[g] - prev)
    f = np.exp(float(spec.along_sd) * y)
    same_next = np.zeros(n, bool)
    same_next[:-1] = ~start[1:]
    f_end = np.where(same_next, np.roll(f, -1), f)
    return f, f_end


def apply_diameter(proj, spec: Optional[DiameterSpec], m_axis: float = 1.0,
                   m_lat: float = 1.0, z_root: Optional[np.ndarray] = None,
                   rng_along=None):
    """``proj`` with ``diam_mm`` scaled by ``m_class * exp(root_sd*z) * along``,
    then clipped to ``[min_mm, max_mm]`` (each only when > 0).

    ``z_root``: :func:`root_z` output, indexed by ``proj.root_id`` (required
    when ``root_sd > 0``).  ``rng_along``: a Generator (or an int seed) for
    the along-root noise (required when ``along_sd > 0``); pass a fresh one
    from the same seed for every spec to keep common random numbers.
    Returns ``proj`` itself for an identity spec with unit multipliers, else
    a copy (every other field shared, unchanged).
    """
    spec = DiameterSpec.from_any(spec)
    if spec is None or (spec.is_identity() and m_axis == 1.0 and m_lat == 1.0):
        return proj
    if len(proj) == 0:
        return proj
    d = np.asarray(proj.diam_mm, np.float64).copy()
    order = np.asarray(proj.order)
    mult = np.where(order == 0, float(m_axis), float(m_lat))
    if spec.root_sd > 0:
        if z_root is None:
            raise ValueError("root_sd > 0 needs z_root (root_z(len(system.roots), rng))")
        zr = np.asarray(z_root, np.float64)[np.asarray(proj.root_id)]
        mult = mult * np.exp(float(spec.root_sd) * zr)
    d *= mult[:, None]
    if spec.along_sd > 0:
        if rng_along is None:
            raise ValueError("along_sd > 0 needs rng_along")
        rng = (np.random.default_rng(rng_along) if not hasattr(rng_along, "standard_normal")
               else rng_along)
        f0, f1 = _along_factor(proj, spec, rng)
        d[:, 0] *= f0
        d[:, 1] *= f1
    lo = float(spec.min_mm) if spec.min_mm > 0 else None
    hi = float(spec.max_mm) if spec.max_mm > 0 else None
    if lo is not None or hi is not None:
        d = np.clip(d, lo, hi)
    return replace(proj, diam_mm=d.astype(np.float32))
