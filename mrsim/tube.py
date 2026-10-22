"""Minirhizotron tube geometry, projection and visibility.

The three things this module exists to get right:

1. **A tube has three radii, not one.**  The sensor rides the INNER wall; roots
   grow on the OUTER wall.  Collapsing them introduces a ~9% scale error in the
   circumferential axis ONLY (``R_in/R_out``), so circumferential roots read 9%
   short and 9% thin while axial roots read correct.  That propagates into every
   diameter bin, SRL, and the orientation histogram.

2. **Scale is not a scalar.**  Even before refraction, an orthographic unwrap
   makes circumferential scale vary by ``cos(phi)`` across the frame.  We emit
   ``(scale_u, scale_v)`` and a per-pixel scale map, never one number.

3. **The 0.78 mm depth of field is a fitted detection probability, not an optical
   cutoff.**  Taylor et al. (2013) recovered it by non-linear regression from
   roots extracted off tube surfaces, and needed a *separate* diameter
   under-estimation correction -- i.e. the real relationship is a soft,
   diameter-dependent detection probability.  Implementing it as a hard geometric
   threshold produces a bimodal visible/invisible population where reality has a
   broad partially-visible band.  We model p(visible | depth, diameter) and
   report the realised effective DOF as a validation output.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

CM_PER_MM = 0.1
MM_PER_CM = 10.0


# --------------------------------------------------------------------------
# hardware presets
# --------------------------------------------------------------------------
@dataclass
class TubeParams:
    """Tube geometry.  All radii in cm; wall thickness derived, not assumed."""

    id_cm: float           # inner diameter  (sensor rides here)
    od_cm: float           # outer diameter  (roots grow here)
    length_cm: float = 183.0
    angle_from_vertical_deg: float = 30.0
    n_wall: float = 1.49   # acrylic / CAB refractive index

    @property
    def r_in(self) -> float:
        return 0.5 * self.id_cm

    @property
    def r_out(self) -> float:
        return 0.5 * self.od_cm

    @property
    def wall_cm(self) -> float:
        return self.r_out - self.r_in

    @property
    def circumferential_scale(self) -> float:
        """``R_in / R_out`` -- the anisotropic foreshortening factor."""
        return self.r_in / self.r_out


@dataclass
class CameraParams:
    """Imaging geometry.  ``px_per_mm`` is DERIVED from the field of view."""

    name: str
    width_px: int
    height_px: int
    fov_w_mm: float        # circumferential extent of the frame
    fov_h_mm: float        # axial extent
    unwrap: str = "cylindrical"   # 'cylindrical' (scanner) | 'orthographic'
    jpeg_quality: Optional[int] = 92

    @property
    def px_per_mm_u(self) -> float:
        return self.width_px / self.fov_w_mm

    @property
    def px_per_mm_v(self) -> float:
        return self.height_px / self.fov_h_mm


# NOTE: every preset below must be confirmed against the user's own corpus in
# notebook 00 before it is trusted.  Published metadata for these systems is
# inconsistent (the VSI MS-190 is variously documented at 98, 102 and 148 px/mm),
# and file DPI tags are unreliable.
PRESETS = {
    # Bartz BTC-100X: 640x480 over 18 x 13.5 mm = 28 um/px.  CAB tube.
    "bartz_btc100x": (
        TubeParams(id_cm=5.3975, od_cm=5.715, length_cm=183.0,
                   angle_from_vertical_deg=30.0),
        CameraParams("bartz_btc100x", 640, 480, 18.0, 13.5, unwrap="orthographic"),
    ),
    # CID CI-600 scanner: 21.6 x 19.6 cm strip at 300 dpi.  Contact sensor on the
    # inner wall, so the unwrap is cylindrical and geometrically clean.
    "cid_ci600_300dpi": (
        TubeParams(id_cm=6.35, od_cm=6.985, length_cm=105.0,
                   angle_from_vertical_deg=30.0),
        CameraParams("cid_ci600_300dpi", 2551, 2315, 216.0, 196.0,
                     unwrap="cylindrical"),
    ),
    # The same CI-600 scan at the same 11.81 px/mm, cut into 640x480 frames: the
    # pixel size of every other synthetic set, the scale of the real scans.  This
    # is what the style-transfer domains are built on -- a 640x480 tile of a real
    # scan and a frame of this preset show a root at the same width in pixels.
    "cid_ci600_640": (
        TubeParams(id_cm=6.35, od_cm=6.985, length_cm=105.0,
                   angle_from_vertical_deg=30.0),
        CameraParams("cid_ci600_640", 640, 480,
                     640 / (2551 / 216.0), 480 / (2315 / 196.0),
                     unwrap="cylindrical"),
    ),
    # Rhizosystems AMR.  Spec gives FOV 3.01 x 2.26 mm at 100x with a selectable
    # sensor readout, so px/mm depends on which readout is used:
    #   640x480   -> 212.6 px/mm (4.70 um/px)
    #   1600x1200 -> 531.6 px/mm (1.88 um/px)
    # This is microscope-scale, ~7x a Bartz, and it inverts the usual thin-root
    # problem: at 212 px/mm a 0.2 mm root is 42 px wide.  CONFIRM WHICH READOUT
    # YOUR CORPUS USES IN NOTEBOOK 00 -- it is the highest-variance unknown.
    "rhizosystems_amr_640": (
        TubeParams(id_cm=6.35, od_cm=6.985, angle_from_vertical_deg=30.0),
        CameraParams("rhizosystems_amr_640", 640, 480, 3.01, 2.26,
                     unwrap="orthographic"),
    ),
    "rhizosystems_amr_1600": (
        TubeParams(id_cm=6.35, od_cm=6.985, angle_from_vertical_deg=30.0),
        CameraParams("rhizosystems_amr_1600", 1600, 1200, 3.01, 2.26,
                     unwrap="orthographic"),
    ),
}


def get_preset(name: str) -> Tuple[TubeParams, CameraParams]:
    if name not in PRESETS:
        raise KeyError("unknown preset %r; available: %s" % (name, sorted(PRESETS)))
    tube, cam = PRESETS[name]
    return (TubeParams(**vars(tube)), CameraParams(**vars(cam)))


# --------------------------------------------------------------------------
# tube frame
# --------------------------------------------------------------------------
@dataclass
class TubeFrame:
    """Orthonormal frame attached to the tube.

    ``e_down`` is the component of gravity perpendicular to the tube axis, so
    ``phi = 0`` is the BOTTOM of the tube.  That matters: unslurried tubes carry
    ~55% more root length below than above (126.4 vs 81.5 cm, winter wheat), and
    an isotropic azimuthal density is wrong in both scanner and matrix frames.
    """

    origin: np.ndarray
    axis: np.ndarray       # unit vector along the tube, pointing down-slope
    e_down: np.ndarray     # phi = 0
    e_side: np.ndarray     # phi = +pi/2
    tube: TubeParams

    @classmethod
    def build(cls, tube: TubeParams, origin=None, azimuth_deg: float = 0.0):
        th = math.radians(tube.angle_from_vertical_deg)
        az = math.radians(azimuth_deg)
        axis = np.array([math.sin(th) * math.cos(az),
                         math.sin(th) * math.sin(az),
                         -math.cos(th)])
        axis /= np.linalg.norm(axis)
        g = np.array([0.0, 0.0, -1.0])
        d = g - (g @ axis) * axis
        n = np.linalg.norm(d)
        if n < 1e-9:                      # perfectly horizontal tube
            d = np.array([1.0, 0.0, 0.0]) - (np.array([1.0, 0.0, 0.0]) @ axis) * axis
            n = np.linalg.norm(d)
        e_down = d / n
        e_side = np.cross(axis, e_down)
        if origin is None:
            origin = np.zeros(3)
        return cls(np.asarray(origin, float), axis, e_down, e_side, tube)

    def to_local(self, xyz: np.ndarray):
        """World -> (axial [cm], phi [rad], rho [cm]).

        ``phi`` is measured from the bottom of the tube, increasing toward
        ``e_side``.  ``rho`` is distance from the tube axis.
        """
        p = np.atleast_2d(np.asarray(xyz, float)) - self.origin
        along = p @ self.axis
        radial = p - np.outer(along, self.axis)
        x = radial @ self.e_down
        y = radial @ self.e_side
        rho = np.hypot(x, y)
        phi = np.arctan2(y, x)
        return along, phi, rho

    # -- the one-sided wall barrier ---------------------------------------
    def confinement(self, w_barrier: float = 40.0, w_attract: float = 0.7,
                    band_cm: float = 0.18):
        """Penalty for the tropism objective.

        A *symmetric* distance penalty (``w * |rho - R_out|``) is attractive from
        both sides, so roots grow straight through the tube wall and then get
        culled -- which silently truncates them.  This is one-sided: a hard
        barrier inside the wall, an attraction in a contact band just outside, and
        nothing beyond.

        ``w_attract`` is deliberately comparable to the tropism objectives (which
        are O(1)), because the root-tube artifact is strong and real: roots that
        reach the wall preferentially grow ALONG it.  Too weak an attraction and
        each root only brushes the wall, so every frame shows short stubby blobs
        instead of the long continuous runs a real frame shows -- and a blob
        skeletonises into a branchy network far longer than its own centreline.
        This is also the mechanism behind the documented +55% root length below
        vs above unslurried tubes, so it should be calibrated, not left at a
        guess.
        """
        r_out = self.tube.r_out

        def penalty(xyz: np.ndarray) -> float:
            _, _, rho = self.to_local(xyz)
            rho = float(rho[0])
            if rho < r_out:
                return w_barrier * (r_out - rho) ** 2 + w_barrier * 0.1
            d = rho - r_out
            if d < band_cm:
                return -w_attract * (1.0 - d / band_cm)
            return 0.0

        # The barrier can only act through the tropism argmin, which means every
        # root needs extra candidate directions while it is near the wall.  Doing
        # that everywhere costs ~6x on a whole stand, of which all but a sliver is
        # far from the tube and unaffected.  This predicate lets the growth loop
        # pay the cost only where it changes the outcome.
        reach = max(band_cm * 3.0, 0.6)

        def near(xyz: np.ndarray) -> bool:
            _, _, rho = self.to_local(xyz)
            return abs(float(rho[0]) - r_out) < reach

        penalty.near = near
        return penalty

    # -- relevance culling --------------------------------------------------
    def relevance(self, reach_cm, margin_cm: float = 1.0,
                  axial_range_cm: Optional[Tuple[float, float]] = None):
        """Predicate ``keep(child_subType, base_xyz) -> bool`` for ``grow(keep_branch=)``.

        A lateral (with everything that will ever branch off it) stays within
        its reach of its base: arc length bounds straight-line distance.  If
        the base is farther than ``reach + margin`` from the tube's OUTER wall
        -- or, with ``axial_range_cm``, that far outside the axial window the
        frames are cut from -- no part of it can ever be imaged, so it need not
        be grown.  The test is one-sided: a base at or inside the wall is kept.

        ``reach_cm``: ``{subType: cm}``, a callable ``subType -> cm`` (e.g.
        :meth:`mrsim.params.PlantParams.reach_cm`) or one number.
        ``margin_cm`` covers the visible depth behind the wall (4 mm), the
        tube's tilt across a frame and reach estimates that are 3-sigma rather
        than hard bounds.  ``axial_range_cm`` is in the same tube-axis
        coordinate as ``FrameSpec.axial_cm``; widen it by half a frame height.
        """
        r_out = self.tube.r_out
        if callable(reach_cm):
            reach_of = reach_cm
        elif isinstance(reach_cm, dict):
            table = {int(k): float(v) for k, v in reach_cm.items()}
            far = max(table.values()) if table else 0.0

            def reach_of(st):
                return table.get(int(st), far)
        else:
            const = float(reach_cm)

            def reach_of(st):
                return const
        m = float(margin_cm)
        lo_hi = None if axial_range_cm is None else (float(axial_range_cm[0]),
                                                      float(axial_range_cm[1]))

        def keep(sub_type, xyz) -> bool:
            along, _, rho = self.to_local(xyz)
            lim = float(reach_of(sub_type)) + m
            if float(rho[0]) - r_out > lim:
                return False
            if lo_hi is not None:
                a = float(along[0])
                if a < lo_hi[0] - lim or a > lo_hi[1] + lim:
                    return False
            return True

        return keep


# --------------------------------------------------------------------------
# visibility
# --------------------------------------------------------------------------
@dataclass
class VisibilityModel:
    """p(detected | depth behind the outer wall, root diameter).

    ``p = exp(-(depth / lam(d))**beta)`` with ``lam(d) = lam0 * (d/d_ref)**gamma``.
    The integral of a stretched exponential is ``lam * Gamma(1 + 1/beta)``, so the
    *effective* depth of field for a root of diameter d is available in closed
    form and ``lam0`` can be calibrated to reproduce the published 0.78 mm when
    averaged over the corpus diameter distribution.
    """

    lam0_mm: float = 0.62
    beta: float = 1.6
    gamma: float = 0.35
    d_ref_mm: float = 0.35

    def lam(self, diam_mm):
        d = np.maximum(np.asarray(diam_mm, float), 1e-4)
        return self.lam0_mm * (d / self.d_ref_mm) ** self.gamma

    def p_visible(self, depth_mm, diam_mm):
        z = np.maximum(np.asarray(depth_mm, float), 0.0)
        return np.exp(-((z / self.lam(diam_mm)) ** self.beta))

    def effective_dof_mm(self, diam_mm) -> np.ndarray:
        """Closed-form effective DOF: ``lam * Gamma(1 + 1/beta)``."""
        return self.lam(diam_mm) * math.gamma(1.0 + 1.0 / self.beta)

    def calibrate(self, diam_mm, target_dof_mm: float = 0.78) -> "VisibilityModel":
        """Set ``lam0`` so the corpus-averaged effective DOF hits ``target``."""
        cur = float(np.mean(self.effective_dof_mm(diam_mm)))
        if cur <= 0:
            return self
        self.lam0_mm *= target_dof_mm / cur
        return self


# --------------------------------------------------------------------------
# projection
# --------------------------------------------------------------------------
def project(frame: TubeFrame, cam: CameraParams, xyz_cm: np.ndarray):
    """Project world points onto the unrolled image plane.

    Returns ``(u_mm, v_mm, depth_mm, scale_u)`` where ``u`` is circumferential,
    ``v`` axial, ``depth`` is distance behind the OUTER wall (into soil), and
    ``scale_u`` is the local circumferential scale factor d(u)/d(arc at R_out).

    The ``R_in/R_out`` factor is what makes circumferential roots read short.
    """
    along_cm, phi, rho_cm = frame.to_local(xyz_cm)
    tube = frame.tube
    k = tube.circumferential_scale                     # R_in / R_out

    if cam.unwrap == "cylindrical":
        u_mm = (tube.r_in * MM_PER_CM) * phi
        scale_u = np.full_like(phi, k)
    elif cam.unwrap == "orthographic":
        u_mm = (tube.r_in * MM_PER_CM) * np.sin(phi)
        scale_u = k * np.cos(phi)
    else:
        raise ValueError("unknown unwrap %r" % cam.unwrap)

    v_mm = along_cm * MM_PER_CM
    depth_mm = (rho_cm - tube.r_out) * MM_PER_CM
    return u_mm, v_mm, depth_mm, scale_u


def facing_camera(phi: np.ndarray, phi_center: float, half_width_rad: float):
    """Boolean mask: is this point on the arc the camera is looking at?

    A matrix camera sees a small azimuthal window, not the whole circumference.
    """
    d = np.abs(np.angle(np.exp(1j * (phi - phi_center))))
    return d <= half_width_rad


def frame_window(cam: CameraParams, tube: TubeParams):
    """Half-width of the imaged arc, in radians, and whether the scan wraps.

    For a scanner the strip can exceed the tube circumference, producing a
    duplicated seam band; the overlap must be computed at the INNER radius, where
    the sensor actually travels.
    """
    circ_in_mm = 2.0 * math.pi * tube.r_in * MM_PER_CM
    half = (cam.fov_w_mm / 2.0) / (tube.r_in * MM_PER_CM)
    wraps = cam.fov_w_mm > circ_in_mm
    overlap_mm = max(0.0, cam.fov_w_mm - circ_in_mm)
    return half, wraps, overlap_mm, circ_in_mm
