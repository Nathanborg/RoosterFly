"""3D stochastic root-system growth, CRootBox/CPlantBox formulation.

Deliberately NOT a fractal/IFS generator.  Real root systems are not statistically
self-similar over the available scale range (Bouda et al. 2016: <2 decades, local
log-log slope varying >25% of D), and an IFS has no time axis, no gravitropism, no
diameter-dependent elongation and no per-root stochasticity -- which are exactly
what make rendered roots look real.  Box-counting D is used as a *validation*
statistic in :mod:`mrsim.traits`, never as a generator.

Five details that are easy to get silently wrong, and are unit-tested in
``tests/test_grow.py``:

1. ``lb`` is the BASAL (proximal) zone, ``la`` the APICAL (distal) zone.
2. ``lmax = lb + la + (nob-1)*ln`` -- the four are not independent.
3. The heading rotation must be applied in the LOCAL orthonormal frame.
4. Both sigma and the trial count scale with ``sqrt(dx)``, which is what makes
   tortuosity independent of the chosen segment length.
5. Exotropism minimises the angle to the CURRENT heading, not the birth heading.
   Against the birth heading it silently becomes a fixed-direction tropism and
   every lateral runs unnaturally straight.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .params import (
    GF_LINEAR,
    GF_NEGEXP,
    TROPISM_EXO,
    TROPISM_GRAVI,
    TROPISM_HYDRO,
    TROPISM_PLAGIO,
    PlantParams,
    RootTypeParams,
)

GRAVITY = np.array([0.0, 0.0, -1.0])


# --------------------------------------------------------------------------
# linear algebra
# --------------------------------------------------------------------------
def _basis3(hx: float, hy: float, hz: float):
    """Two unit vectors perpendicular to a unit heading, as plain scalars.

    Written out longhand rather than with ``np.cross``/``np.linalg.norm``: on
    3-vectors those are dominated by numpy's dispatch overhead, and this is the
    innermost loop of the whole simulator (it ran at ~40% of total growth time).
    """
    ax, ay, az = abs(hx), abs(hy), abs(hz)
    if ax <= ay and ax <= az:
        tx, ty, tz = 1.0, 0.0, 0.0
    elif ay <= az:
        tx, ty, tz = 0.0, 1.0, 0.0
    else:
        tx, ty, tz = 0.0, 0.0, 1.0
    ux = hy * tz - hz * ty
    uy = hz * tx - hx * tz
    uz = hx * ty - hy * tx
    n = math.sqrt(ux * ux + uy * uy + uz * uz)
    if n < 1e-15:
        return 0.0, 1.0, 0.0, 0.0, 0.0, 1.0
    ux, uy, uz = ux / n, uy / n, uz / n
    vx = hy * uz - hz * uy
    vy = hz * ux - hx * uz
    vz = hx * uy - hy * ux
    return ux, uy, uz, vx, vy, vz


def ons(h) -> np.ndarray:
    """Orthonormal basis whose FIRST COLUMN is ``h`` (CRootBox Matrix3d::ons).

    The heading update must be expressed in this frame.  Applying the rotation in
    world coordinates instead makes every root curve toward a fixed world axis.
    """
    hx, hy, hz = float(h[0]), float(h[1]), float(h[2])
    n = math.sqrt(hx * hx + hy * hy + hz * hz)
    if n <= 0:
        hx, hy, hz = 0.0, 0.0, -1.0
    else:
        hx, hy, hz = hx / n, hy / n, hz / n
    ux, uy, uz, vx, vy, vz = _basis3(hx, hy, hz)
    return np.array([[hx, ux, vx], [hy, uy, vy], [hz, uz, vz]])


def _rot_ab_s(hx: float, hy: float, hz: float, alpha: float, beta: float):
    """Scalar form of :func:`rot_ab`; returns a unit 3-tuple."""
    ux, uy, uz, vx, vy, vz = _basis3(hx, hy, hz)
    c = math.cos(alpha)
    s = math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    a1, a2 = s * cb, s * sb
    ox = hx * c + ux * a1 + vx * a2
    oy = hy * c + uy * a1 + vy * a2
    oz = hz * c + uz * a1 + vz * a2
    n = math.sqrt(ox * ox + oy * oy + oz * oz)
    if n < 1e-15:
        return hx, hy, hz
    return ox / n, oy / n, oz / n


def rot_ab(h, alpha: float, beta: float) -> np.ndarray:
    """Rotate heading ``h`` by polar ``alpha`` / azimuth ``beta`` in its own frame."""
    hx, hy, hz = float(h[0]), float(h[1]), float(h[2])
    n = math.sqrt(hx * hx + hy * hy + hz * hz)
    if n > 0:
        hx, hy, hz = hx / n, hy / n, hz / n
    return np.array(_rot_ab_s(hx, hy, hz, alpha, beta))


# --------------------------------------------------------------------------
# growth functions (CRootBox growth.h)
# --------------------------------------------------------------------------
def length_at_age(age: float, r: float, k: float, gf: int) -> float:
    """Length reached at ``age``.  ``r`` is the INITIAL rate, not a constant."""
    if age <= 0:
        return 0.0
    if gf == GF_LINEAR:
        return float(min(k, r * age))
    return float(k * (1.0 - np.exp(-(r / k) * age)))   # negative exponential


def age_at_length(length: float, r: float, k: float, gf: int) -> float:
    """Inverse of :func:`length_at_age`; needed to date lateral emergence."""
    if length <= 0:
        return 0.0
    if gf == GF_LINEAR:
        return float(length / r)
    if length >= k:
        return float("inf")
    return float(-(k / r) * np.log(1.0 - length / k))


# --------------------------------------------------------------------------
# tropism
# --------------------------------------------------------------------------
def _objective_s(kind: int, nx: float, ny: float, nz: float,
                 cx: float, cy: float, cz: float,
                 gsa: Optional[float]) -> float:
    """Tropism objective, scalar form.  Lower is better."""
    if kind == TROPISM_GRAVI:
        if gsa is not None:
            # gravitropic set-point angle: roots hold a non-vertical angle and
            # actively return to it after displacement.  Pure gravitropism
            # collapses everything to vertical; pure exotropism freezes the
            # initial angle and cannot recover.
            ang = math.acos(max(-1.0, min(1.0, -nz)))
            return abs(ang - gsa)
        return 0.5 * (nz + 1.0)
    if kind == TROPISM_PLAGIO:
        return abs(nz)
    if kind == TROPISM_EXO:
        # angle to the CURRENT heading -- a curvature penalty producing smooth
        # meanders.  Using the birth heading here is bug (5).
        return math.acos(max(-1.0, min(1.0, nx * cx + ny * cy + nz * cz))) / math.pi
    if kind == TROPISM_HYDRO:
        return -nz
    return 0.0


def _objective(kind: int, h_new, h_cur, gsa: Optional[float]) -> float:
    return _objective_s(kind, float(h_new[0]), float(h_new[1]), float(h_new[2]),
                        float(h_cur[0]), float(h_cur[1]), float(h_cur[2]), gsa)


def heading_step(h_cur: np.ndarray, dx: float, sigma: float, n: float,
                 kind: int, gsa: Optional[float], rng,
                 constraint=None) -> np.ndarray:
    """One tropism step: try ``n*sqrt(dx)`` candidates, keep the argmin.

    This is random optimisation, not vector addition.  Both ``sigma`` and the
    trial count carry ``sqrt(dx)``; without it, halving dx doubles the tortuosity
    per centimetre.
    """
    sq = math.sqrt(max(dx, 1e-12))
    cx, cy, cz = float(h_cur[0]), float(h_cur[1]), float(h_cur[2])
    nrm = math.sqrt(cx * cx + cy * cy + cz * cz)
    if nrm > 0:
        cx, cy, cz = cx / nrm, cy / nrm, cz / nrm

    # max(1, ...) guard: int(floor(n*sqrt(dx))) collapses to 0 at the small dx
    # this simulator uses, which would silently disable the tropism entirely.
    n_trials = int(math.floor(n * sq))
    if n > 0:
        n_trials = max(1, n_trials)
    if constraint is not None:
        # A constraint can only act through the argmin, so a root type with
        # tropismN = 0 (which is most fine laterals -- they are deliberately a
        # pure random walk) would ignore it completely: it would wander through
        # the tube wall and be culled at projection time.  Give every root enough
        # candidates for the barrier and the contact band to bite, without
        # otherwise changing its tropism.
        n_trials = max(n_trials, 6)

    # draw all candidate angles at once: one rng call beats n_trials+1 calls
    k = n_trials + 1
    alphas = rng.normal(size=k) * (sigma * sq)
    betas = rng.uniform(0.0, 2.0 * math.pi, size=k)

    best_h = None
    best_v = float("inf")
    for i in range(k):
        tx, ty, tz = _rot_ab_s(cx, cy, cz, float(alphas[i]), float(betas[i]))
        v = _objective_s(kind, tx, ty, tz, cx, cy, cz, gsa)
        if constraint is not None:
            v += constraint(np.array([tx, ty, tz]))
        if v < best_v:
            best_v, best_h = v, (tx, ty, tz)
    return np.array(best_h)


# --------------------------------------------------------------------------
# taper
# --------------------------------------------------------------------------
def radius_at(s: float, total_len: float, p: RootTypeParams) -> float:
    """Radius at arc position ``s`` along a root of current length ``total_len``.

    Constant absolute decrement per cm over most of the root, plus a short apical
    taper zone.  A linear wedge normalised by lmax would give a 2 cm lateral the
    same base:tip ratio as a 60 cm axis.
    """
    r = p.a - p.taper_per_cm * s
    r = max(r, p.a * p.tip_radius_frac)
    az = p.apical_taper_cm
    if az > 0 and total_len - s < az:
        f = max(0.0, (total_len - s) / az)
        r *= p.tip_radius_frac + (1.0 - p.tip_radius_frac) * f
    return max(r, 1e-4)


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------
@dataclass
class Root:
    """One root axis as a polyline, with per-node radius/age.

    ``birth_day`` / ``rlt`` are carried even though v1 renders live roots only --
    the renderer reads colour and edge sharpness from an age function, so adding
    the v2 senescence trajectory does not require changing the data model.
    """

    subType: int
    order: int
    params: RootTypeParams
    parent: Optional[int] = None
    parent_s: float = 0.0            # arc position on the parent [cm]
    birth_day: float = 0.0
    nodes: List[np.ndarray] = field(default_factory=list)
    node_age: List[float] = field(default_factory=list)   # day each node was laid
    node_radius: List[float] = field(default_factory=list)
    node_s: List[float] = field(default_factory=list)     # running arc length [cm]
    heading: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    h0: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    length: float = 0.0
    alive: bool = True
    emerged_laterals: int = 0
    next_branch_s: float = 0.0
    index: int = -1

    def polyline(self) -> np.ndarray:
        return np.asarray(self.nodes, dtype=float)

    def radii(self) -> np.ndarray:
        return np.asarray(self.node_radius, dtype=float)


@dataclass
class RootSystem:
    roots: List[Root] = field(default_factory=list)
    species: str = ""
    days: float = 0.0
    seed_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # laterals NOT spawned because ``keep_branch`` judged them unable to reach
    # the tube (relevance culling); 0 when culling is off
    n_culled: int = 0
    # plant index of every root (``grow_stand``); None for a single plant.  A
    # mixed-species stand needs it to say which species a pixel belongs to.
    root_plant: Optional[np.ndarray] = None

    def total_length(self) -> float:
        return float(sum(r.length for r in self.roots))

    def n_tips(self) -> int:
        return len(self.roots)

    def n_branch_points(self) -> int:
        return sum(1 for r in self.roots if r.parent is not None)


# --------------------------------------------------------------------------
# the growth loop
# --------------------------------------------------------------------------
def _spawn(system: RootSystem, plant: PlantParams, subType: int, order: int,
           pos: np.ndarray, heading: np.ndarray, day: float, rng,
           parent: Optional[int] = None, parent_s: float = 0.0) -> Root:
    p = plant.get(subType).realise(rng)
    r = Root(
        subType=subType, order=order, params=p, parent=parent, parent_s=parent_s,
        birth_day=day, nodes=[np.asarray(pos, dtype=float).copy()],
        node_age=[day], node_radius=[p.a],
        heading=np.asarray(heading, dtype=float).copy(),
        h0=np.asarray(heading, dtype=float).copy(),
        next_branch_s=p.lb,
    )
    r.index = len(system.roots)
    system.roots.append(r)
    return r


def _lateral_heading(parent_h: np.ndarray, theta: float, rng) -> np.ndarray:
    """Insertion at polar ``theta``, azimuth uniform on [0, 2pi).

    Omitting the uniform azimuth collapses the system into a plane or a regular
    helix -- one of the most visible synthetic tells.
    """
    beta = float(rng.uniform(0.0, 2.0 * np.pi))
    return rot_ab(parent_h, theta, beta)


def grow(plant: PlantParams, days: float, rng, dt: float = 1.0,
         max_roots: int = 20000, confine=None, seed_offset=None,
         keep_branch=None) -> RootSystem:
    """Grow a root system for ``days``.

    Parameters
    ----------
    confine : callable or None
        ``confine(xyz) -> float`` penalty added to the tropism objective, used by
        :mod:`mrsim.tube` to implement the one-sided tube-wall barrier.  A
        symmetric penalty would let roots grow through the tube wall.
    keep_branch : callable or None
        ``keep_branch(child_subType, base_xyz) -> bool``, asked once per lateral
        just before it is spawned (relevance culling, see
        :meth:`mrsim.tube.TubeFrame.relevance`).  A refused lateral still
        counts as emerged -- the parent's branching schedule advances exactly as
        if it had grown -- but it is never created, so it costs nothing and
        does not count against ``max_roots``.  In a stand most laterals form
        tens of cm from the tube; growing them is where the time went, and
        with the cap at 8000 roots per plant the cap bound at ~60 days and deep
        axes stayed bare.  ``None`` (default) is the original loop exactly.
    """
    system = RootSystem(species=plant.name, days=days)
    # v3 presets: re-taper every node laid in a step (PlantParams.exact_taper)
    exact_taper = bool(getattr(plant, "exact_taper", False))
    seed = np.array([0.0, 0.0, -abs(plant.seed_depth_cm)])
    if seed_offset is not None:
        seed = seed + np.asarray(seed_offset, dtype=float)
    system.seed_pos = seed

    _spawn(system, plant, 1, 0, seed, np.array([0.0, 0.0, -1.0]), 0.0, rng)

    n_basal = 0
    t = 0.0
    while t < days:
        t = min(days, t + dt)

        # shoot-borne / nodal roots emitted over time (cereals are dominated by
        # these; modelling a cereal as taproot+laterals is topologically wrong)
        if plant.basal_type is not None and n_basal < plant.max_basal:
            due = plant.first_basal + n_basal * plant.delay_basal
            while n_basal < plant.max_basal and due <= t:
                bp = plant.get(plant.basal_type)
                h = _lateral_heading(np.array([0.0, 0.0, -1.0]), bp.theta, rng)
                _spawn(system, plant, plant.basal_type, 0, seed, h, due, rng)
                n_basal += 1
                due = plant.first_basal + n_basal * plant.delay_basal

        # iterate over a snapshot: roots born this timestep start next timestep
        n_now = len(system.roots)
        for i in range(n_now):
            r = system.roots[i]
            if not r.alive:
                continue
            age = t - r.birth_day
            if age <= 0:
                continue
            p = r.params
            if age > p.rlt:
                r.alive = False
                continue

            target = length_at_age(age, p.r, p.lmax, p.gf)
            target = min(target, p.lmax)
            grow_by = target - r.length
            if grow_by <= 1e-9:
                continue

            # subdivide the increment into dx steps so tortuosity is resolved
            n_steps = max(1, int(np.ceil(grow_by / p.dx)))
            step = grow_by / n_steps
            n_before = len(r.nodes)
            near_fn = getattr(confine, "near", None) if confine is not None else None
            for _ in range(n_steps):
                cons = None
                if confine is not None:
                    pos_now = r.nodes[-1]
                    if near_fn is None or near_fn(pos_now):
                        def cons(h_try, _pos=pos_now, _st=step):
                            return confine(_pos + _st * h_try)

                r.heading = heading_step(r.heading, step, p.tropismS, p.tropismN,
                                         p.tropismT, p.gsa, rng, constraint=cons)
                new_pos = r.nodes[-1] + step * r.heading
                r.length += step
                r.nodes.append(new_pos)
                r.node_age.append(t)
                r.node_s.append(r.length)
                r.node_radius.append(radius_at(r.length, r.length, p))

            # retro-taper: node radii depend on the CURRENT total length,
            # because the apical taper zone travels with the tip
            _retaper(r, since=(n_before if exact_taper else None))

            # --- lateral emergence -------------------------------------------
            if p.successors and len(system.roots) < max_roots:
                limit = max(0.0, r.length - p.la)
                while (r.next_branch_s <= limit
                       and r.emerged_laterals < p.n_branches()):
                    s = r.next_branch_s
                    born = r.birth_day + age_at_length(s, p.r, p.lmax, p.gf) + p.ldelay
                    if born > t:
                        break
                    child = _pick_successor(p.successors, rng)
                    if child is not None:
                        pos = _point_at_arc(r, s)
                        if keep_branch is not None and not keep_branch(child, pos):
                            # culled: no heading or realisation draws, no root
                            system.n_culled += 1
                        else:
                            cp = plant.get(child)
                            h = _lateral_heading(r.heading, cp.theta, rng)
                            _spawn(system, plant, child, r.order + 1, pos, h, born, rng,
                                   parent=r.index, parent_s=s)
                    r.emerged_laterals += 1
                    # lnk lets inter-lateral distance drift along the root; real
                    # branching density peaks at intermediate positions rather
                    # than giving a tell-tale evenly spaced comb.
                    ln_eff = max(1e-2, p.ln * (1.0 + p.lnk * s))
                    r.next_branch_s = s + ln_eff

    return system


def _retaper(r: Root, since: Optional[int] = None) -> None:
    """Refresh node radii near the tip.

    ``radius_at`` depends on the current total length ONLY through the apical
    taper zone, so nodes further back than ``apical_taper_cm`` are already final.
    Recomputing the whole polyline every timestep made growth O(n^2) in nodes.

    That premise holds only for nodes that were refreshed while they were in
    the zone.  A node is created AS the tip (radius = tip radius) and a step
    usually lays more than one apical zone of new root, so the nodes laid
    early in the step are already behind the zone at this call and keep the
    tip radius for good.  ``since`` (v3, ``PlantParams.exact_taper``) also
    refreshes every node from that index -- the nodes laid this step -- which
    keeps growth O(n) and makes every radius what ``radius_at`` says.
    ``None`` keeps the original behaviour exactly (frozen presets).
    """
    n = len(r.nodes)
    if n < 2:
        r.node_radius = [r.params.a]
        r.node_s = [0.0]
        return
    if len(r.node_s) != n:                     # safety: rebuild if out of sync
        pts = np.asarray(r.nodes, dtype=float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        r.node_s = [0.0] + list(np.cumsum(seg))
    total = float(r.node_s[-1])
    zone = max(r.params.apical_taper_cm, 0.0) * 1.05
    i = n - 1
    while i >= 0 and (total - r.node_s[i]) <= zone:
        r.node_radius[i] = radius_at(float(r.node_s[i]), total, r.params)
        i -= 1
    if since is not None:
        for k in range(max(int(since), 1), i + 1):
            r.node_radius[k] = radius_at(float(r.node_s[k]), total, r.params)


def _pick_successor(successors, rng) -> Optional[int]:
    u = float(rng.random())
    acc = 0.0
    for sub, prob in successors:
        acc += prob
        if u < acc:
            return sub
    return None


def _point_at_arc(r: Root, s: float) -> np.ndarray:
    """Position at arc length ``s`` along a root, by linear interpolation."""
    pts = np.asarray(r.nodes, dtype=float)
    if len(pts) == 1:
        return pts[0].copy()
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    if s <= 0:
        return pts[0].copy()
    if s >= cum[-1]:
        return pts[-1].copy()
    j = int(np.searchsorted(cum, s) - 1)
    j = max(0, min(j, len(seg) - 1))
    f = (s - cum[j]) / max(seg[j], 1e-12)
    return pts[j] + f * (pts[j + 1] - pts[j])


# --------------------------------------------------------------------------
# sub-dx refinement
# --------------------------------------------------------------------------
def subdivide(pts: np.ndarray, radii: np.ndarray, max_seg_cm: float,
              sigma: float, rng) -> Tuple[np.ndarray, np.ndarray]:
    """Refine a polyline to <= ``max_seg_cm`` with a Brownian bridge on heading.

    Growth dx is 0.25-1 cm; at 35-212 px/mm that is 350-2000 px of perfectly
    straight capsule, so roots would render as visible polygons with identically
    zero sub-dx tortuosity.  The bridge variance is set by the SAME tropismS, so
    sub-dx statistics stay consistent with supra-dx ones rather than invented.
    """
    pts = np.asarray(pts, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if len(pts) < 2:
        return pts, radii

    out_p = [pts[0]]
    out_r = [radii[0]]
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i + 1]
        r0, r1 = radii[i], radii[i + 1]
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L <= max_seg_cm or L < 1e-12:
            out_p.append(p1)
            out_r.append(r1)
            continue
        k = int(np.ceil(L / max_seg_cm))
        h = d / L
        basis = ons(h)
        u, v = basis[:, 1], basis[:, 2]
        # Brownian bridge: lateral displacement pinned to zero at both ends
        ts = np.linspace(0.0, 1.0, k + 1)
        amp = sigma * np.sqrt(L)
        du = rng.normal(0.0, amp, size=k + 1).cumsum()
        dv = rng.normal(0.0, amp, size=k + 1).cumsum()
        du -= ts * du[-1]
        dv -= ts * dv[-1]
        scale = L / max(k, 1) * 0.5
        for j in range(1, k + 1):
            base = p0 + d * ts[j]
            off = (du[j] * u + dv[j] * v) * scale
            out_p.append(base + off)
            out_r.append(r0 + (r1 - r0) * ts[j])
    return np.asarray(out_p), np.asarray(out_r)
