"""Parameter sets for the minirhizotron simulator.

Two independent parameter families:

* :class:`RootTypeParams` / :class:`PlantParams` -- root system architecture,
  following the CRootBox/CPlantBox formulation.  Verified against
  ``src/structural/rootparameter.h`` rather than the published tables, because
  at least one rendered version of the CRootBox Annals of Botany Table 3 has
  ``la`` and ``lb`` swapped.

* :class:`TubeParams` / :class:`CameraParams` (in :mod:`mrsim.tube`) -- hardware.

Units are centimetres and days throughout the growth model.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# tropism types (CRootBox enum order)
# --------------------------------------------------------------------------
TROPISM_PLAGIO = 0
TROPISM_GRAVI = 1
TROPISM_EXO = 2
TROPISM_HYDRO = 3

GF_NEGEXP = 1
GF_LINEAR = 2


@dataclass
class RootTypeParams:
    """Per-developmental-order root parameters, with paired standard deviations.

    Every length/rate/angle has a ``*s`` sibling.  These are sampled ONCE per root
    instance at birth -- that per-root realisation, not per-segment noise, is the
    main source of realistic variety between roots.
    """

    name: str = "root"
    subType: int = 1

    # --- zones along the root -------------------------------------------------
    # lb = BASAL zone   (proximal: root base -> first lateral)
    # la = APICAL zone  (distal:   last lateral -> tip)
    # Getting these backwards clusters laterals at the wrong end of every root.
    lb: float = 0.0
    lbs: float = 0.0
    la: float = 10.0
    las: float = 0.0
    ln: float = 1.0          # inter-lateral distance
    lns: float = 0.0
    lnk: float = 0.0         # slope: lets ln drift along the root
    lmax: float = 50.0       # maximal root length
    lmaxs: float = 0.0

    # --- growth ---------------------------------------------------------------
    r: float = 1.0           # INITIAL elongation rate [cm/day] (not constant)
    rs: float = 0.0
    gf: int = GF_NEGEXP
    rlt: float = 1e9         # root life time [days]
    rlts: float = 0.0
    ldelay: float = 5.0      # lateral emergence delay [days]
    ldelays: float = 1.0

    # --- geometry -------------------------------------------------------------
    a: float = 0.05          # radius [cm]
    a_dev: float = 0.0
    theta: float = 1.22      # insertion angle to parent [rad] (~70 deg)
    thetas: float = 0.0

    # taper: absolute decrement per cm, plus a short apical taper zone.
    # NOT a linear wedge normalised by lmax -- that would give a 2 cm lateral the
    # same base:tip ratio as a 60 cm axis.
    taper_per_cm: float = 0.0015
    apical_taper_cm: float = 0.5
    tip_radius_frac: float = 0.45

    # --- tropism --------------------------------------------------------------
    tropismT: int = TROPISM_GRAVI
    tropismN: float = 1.0    # number of trials (strength); 0 -> pure random walk
    tropismS: float = 0.2    # tortuosity sigma; effective units rad*cm^-0.5
    gsa: Optional[float] = None  # gravitropic set-point angle [rad from gravity]

    # --- branching ------------------------------------------------------------
    successors: Sequence[Tuple[int, float]] = ()   # (child subType, probability)

    dx: float = 0.25         # axial growth resolution [cm]
    dxMin: float = 0.05

    # Number of branches, fixed at realisation.  Stored rather than recomputed:
    # re-deriving it from the snapped lmax round-trips through a floor() that can
    # land one branch low on float error, silently breaking the lmax relation.
    nob: Optional[int] = None

    def realise(self, rng) -> "RootTypeParams":
        """Draw one per-root realisation of the stochastic parameters.

        Enforces ``lmax = lb + la + (nob-1)*ln``: sampling the four independently
        produces roots whose branching zone overruns lmax, or roots with no
        laterals at all.  lb/la/ln are kept as drawn and lmax is snapped.
        """
        def pos(mu, sd, floor=0.0):
            if sd <= 0:
                return max(floor, float(mu))
            return max(floor, float(rng.normal(mu, sd)))

        lb = pos(self.lb, self.lbs)
        la = pos(self.la, self.las, floor=1e-3)
        # positive floor on ln: a near-zero ln spins the lateral-placement loop
        ln = pos(self.ln, self.lns, floor=1e-2)
        lmax = pos(self.lmax, self.lmaxs, floor=la + lb)

        if ln > 0 and lmax > lb + la:
            nob = int(np.floor((lmax - lb - la) / ln)) + 1
        else:
            nob = 0
        if nob > 0:
            lmax = lb + la + (nob - 1) * ln

        return replace(
            self,
            lb=lb, la=la, ln=ln, lmax=lmax, nob=nob,
            r=pos(self.r, self.rs, floor=1e-4),
            rlt=pos(self.rlt, self.rlts, floor=1e-3),
            a=pos(self.a, self.a_dev, floor=1e-4),
            theta=pos(self.theta, self.thetas),
            ldelay=pos(self.ldelay, self.ldelays),
        )

    def n_branches(self) -> int:
        if self.nob is not None:
            return self.nob
        if self.ln <= 0 or self.lmax <= self.lb + self.la:
            return 0
        return int(np.floor((self.lmax - self.lb - self.la) / self.ln)) + 1


@dataclass
class PlantParams:
    """A species preset: one RootTypeParams per developmental order."""

    name: str = "generic"
    types: Dict[int, RootTypeParams] = field(default_factory=dict)
    basal_type: Optional[int] = None    # shoot-borne / nodal roots (cereals)
    first_basal: float = 7.0            # day of first basal root
    delay_basal: float = 7.0            # days between basal roots
    max_basal: int = 0
    seed_depth_cm: float = 3.0

    # NOT a dataclass field (no annotation), so asdict() of the frozen presets
    # is unchanged.  True (set by every v3 preset) makes grow() re-taper EVERY
    # node laid in a timestep, not only those inside the apical zone: the
    # original _retaper left nodes laid early in a step -- more than one apical
    # zone behind the new tip -- at the radius they were created with, which
    # is the TIP radius, so most of a lateral's length was drawn at
    # tip_radius_frac * a (v1 laterals: length-weighted median ~0.4 a).
    exact_taper = False

    def get(self, subType: int) -> RootTypeParams:
        return self.types[subType]

    def reach_cm(self, subType: int, _seen: Optional[frozenset] = None) -> float:
        """How far from its base a root of ``subType`` -- with all its
        descendants -- can possibly extend [cm].

        ``lmax + 3*lmaxs`` bounds the root's own arc length with probability
        0.9987 (a realised lmax is drawn from N(lmax, lmaxs) and then snapped
        DOWN to the branching relation), and a lateral can start at most that
        far along it, so the longest successor reach is added.  Arc length
        bounds straight-line distance, so relevance culling
        (:meth:`mrsim.tube.TubeFrame.relevance`) that skips a lateral whose base
        is farther than this from the tube wall (plus a margin) removes only
        roots none of which could ever be imaged.  Successor cycles are cut (a
        type already on the path adds nothing), so the recursion always ends.
        """
        seen = _seen or frozenset()
        if subType in seen or subType not in self.types:
            return 0.0
        p = self.types[subType]
        own = float(p.lmax) + 3.0 * float(p.lmaxs)
        nxt = seen | {subType}
        kids = [self.reach_cm(int(c), nxt) for c, prob in p.successors if prob > 0]
        return own + (max(kids) if kids else 0.0)


# --------------------------------------------------------------------------
# species presets
# --------------------------------------------------------------------------
def _maize() -> PlantParams:
    """Maize-like: shoot-borne crown roots dominate, not taproot-plus-laterals."""
    t1 = RootTypeParams(
        name="primary", subType=1, lb=2.0, lbs=0.5, la=4.0, las=1.0,
        ln=0.8, lns=0.2, lmax=90.0, lmaxs=15.0,
        r=3.0, rs=0.5, a=0.13, a_dev=0.02, theta=0.0,
        tropismT=TROPISM_GRAVI, tropismN=4.0, tropismS=0.35,
        gsa=np.deg2rad(20.0), successors=((2, 1.0),),
        taper_per_cm=0.0008, apical_taper_cm=0.8,
    )
    t2 = RootTypeParams(
        name="lateral1", subType=2, lb=0.5, lbs=0.2, la=1.5, las=0.5,
        ln=0.4, lns=0.15, lmax=12.0, lmaxs=4.0,
        r=1.2, rs=0.3, a=0.05, a_dev=0.012, theta=np.deg2rad(75.0), thetas=0.25,
        tropismT=TROPISM_EXO, tropismN=2.0, tropismS=0.6,
        rlt=60.0, rlts=15.0, successors=((3, 0.6),),
        taper_per_cm=0.0025, apical_taper_cm=0.3,
    )
    t3 = RootTypeParams(
        name="lateral2", subType=3, lb=0.2, la=0.8, las=0.3,
        ln=0.25, lns=0.1, lmax=3.0, lmaxs=1.2,
        r=0.5, rs=0.15, a=0.022, a_dev=0.006, theta=np.deg2rad(80.0), thetas=0.3,
        tropismT=TROPISM_EXO, tropismN=0.0, tropismS=0.9,
        rlt=35.0, rlts=10.0, successors=(),
        taper_per_cm=0.004, apical_taper_cm=0.15,
    )
    tb = RootTypeParams(
        name="crown", subType=4, lb=1.5, lbs=0.5, la=4.0, las=1.0,
        ln=0.7, lns=0.2, lmax=70.0, lmaxs=12.0,
        r=2.6, rs=0.5, a=0.11, a_dev=0.02, theta=np.deg2rad(60.0), thetas=0.2,
        tropismT=TROPISM_GRAVI, tropismN=3.0, tropismS=0.4,
        gsa=np.deg2rad(45.0), successors=((2, 1.0),),
        taper_per_cm=0.0009, apical_taper_cm=0.8,
    )
    return PlantParams(
        name="zea_mays", types={1: t1, 2: t2, 3: t3, 4: tb},
        basal_type=4, first_basal=7.0, delay_basal=4.0, max_basal=14,
    )


def _soybean() -> PlantParams:
    t1 = RootTypeParams(
        name="taproot", subType=1, lb=1.0, lbs=0.3, la=3.0, las=1.0,
        ln=0.6, lns=0.2, lmax=70.0, lmaxs=12.0,
        r=2.2, rs=0.4, a=0.20, a_dev=0.03, theta=0.0,
        tropismT=TROPISM_GRAVI, tropismN=5.0, tropismS=0.25,
        gsa=np.deg2rad(5.0), successors=((2, 1.0),),
        taper_per_cm=0.0012, apical_taper_cm=0.8,
    )
    t2 = RootTypeParams(
        name="lateral1", subType=2, lb=0.4, la=1.2, las=0.4,
        ln=0.35, lns=0.12, lmax=16.0, lmaxs=5.0,
        r=1.0, rs=0.25, a=0.04, a_dev=0.01, theta=np.deg2rad(80.0), thetas=0.25,
        tropismT=TROPISM_PLAGIO, tropismN=1.5, tropismS=0.7,
        rlt=55.0, rlts=15.0, successors=((3, 0.5),),
        taper_per_cm=0.0028, apical_taper_cm=0.25,
    )
    t3 = RootTypeParams(
        name="lateral2", subType=3, lb=0.15, la=0.6, las=0.25,
        ln=0.22, lns=0.08, lmax=2.5, lmaxs=1.0,
        r=0.42, rs=0.12, a=0.018, a_dev=0.005, theta=np.deg2rad(82.0), thetas=0.3,
        tropismT=TROPISM_EXO, tropismN=0.0, tropismS=1.0,
        rlt=30.0, rlts=8.0, taper_per_cm=0.0045, apical_taper_cm=0.12,
    )
    tb = RootTypeParams(
        name="basal", subType=4, lb=1.0, la=3.0, ln=0.6, lmax=45.0, lmaxs=10.0,
        r=1.8, rs=0.4, a=0.10, a_dev=0.02, theta=np.deg2rad(90.0), thetas=0.15,
        tropismT=TROPISM_PLAGIO, tropismN=2.0, tropismS=0.5,
        successors=((2, 1.0),), taper_per_cm=0.0015, apical_taper_cm=0.6,
    )
    return PlantParams(
        name="glycine_max", types={1: t1, 2: t2, 3: t3, 4: tb},
        basal_type=4, first_basal=9.0, delay_basal=6.0, max_basal=5,
    )


def _generic_fibrous() -> PlantParams:
    """Species-agnostic dense fibrous system (perennial grass-like)."""
    t1 = RootTypeParams(
        name="axis", subType=1, lb=1.0, lbs=0.4, la=3.0, las=1.0,
        ln=0.5, lns=0.2, lmax=60.0, lmaxs=15.0,
        r=2.0, rs=0.6, a=0.055, a_dev=0.015, theta=0.0,
        tropismT=TROPISM_GRAVI, tropismN=3.0, tropismS=0.45,
        gsa=np.deg2rad(35.0), successors=((2, 1.0),),
        taper_per_cm=0.0010, apical_taper_cm=0.6,
    )
    t2 = RootTypeParams(
        name="lateral1", subType=2, lb=0.3, la=1.0, las=0.4,
        ln=0.3, lns=0.12, lmax=8.0, lmaxs=3.0,
        r=0.9, rs=0.25, a=0.025, a_dev=0.007, theta=np.deg2rad(70.0), thetas=0.3,
        tropismT=TROPISM_EXO, tropismN=1.0, tropismS=0.8,
        rlt=50.0, rlts=15.0, successors=((3, 0.5),),
        taper_per_cm=0.003, apical_taper_cm=0.2,
    )
    t3 = RootTypeParams(
        name="lateral2", subType=3, lb=0.1, la=0.5, las=0.2,
        ln=0.2, lns=0.07, lmax=2.0, lmaxs=0.8,
        r=0.4, rs=0.12, a=0.013, a_dev=0.004, theta=np.deg2rad(80.0), thetas=0.35,
        tropismT=TROPISM_EXO, tropismN=0.0, tropismS=1.1,
        rlt=28.0, rlts=8.0, taper_per_cm=0.005, apical_taper_cm=0.1,
    )
    return PlantParams(
        name="generic_fibrous", types={1: t1, 2: t2, 3: t3},
        basal_type=1, first_basal=5.0, delay_basal=3.0, max_basal=10,
    )


# --------------------------------------------------------------------------
# v3 presets: realistic taper, more species
# --------------------------------------------------------------------------
# The three presets above are FROZEN (every v1/v2 render depends on them).  They
# share one bug that the v3 width study measured: ``taper_per_cm`` is an
# absolute decrement about as large as the radius itself, so a 2 cm lateral2
# of generic_fibrous reaches its tip floor (0.45 a) after ~1.4 cm and 24-49% of
# every root's length is drawn at the floor.  Length-weighted median diameters
# came out at 0.16 / 0.26 / 0.50 mm for nominal 0.26 / 0.50 / 1.10 mm, and 78%
# of synthetic root length was 1-2 px wide.  The presets below use a taper 3-5x
# smaller and a tip floor of 0.6-0.7, so a root keeps its width until a short
# apical zone (0.1-1 cm) -- which is what root anatomy shows.
#
# Order-0 roots (taproot / primary / nodal / basal) have a NOMINAL diameter of
# at most 2.5 mm (a <= 0.125 cm).  Real chicory or sunflower taproots are
# thicker near the crown, but what a tube 5-60 cm deep images is the thinner
# part, and a 3-5 mm root is 35-60 px wide at 11.81 px/mm -- wider than any
# PRMI p90 (<= 36 px) and exactly where the CUT geometry audit failed 91% of
# v2 frames.  ``DiameterSpec.max_mm = 2.5`` caps the realised widths as well.
#
# They also set ``exact_taper``: the v1 growth loop left every node laid early
# in a timestep at the TIP radius (see PlantParams.exact_taper), which on its
# own put most lateral length at tip_radius_frac * a.
#
# Diameters below are quoted in mm (= 20 * a).  These are plausible starting
# values, not fitted ones: the width calibration (mrstyle.widths) corrects the
# overall scale through the diameter multipliers.
_V3_TIP = 0.65            # tip radius floor as a fraction of a (was 0.45)


def _v3_plant(p: PlantParams) -> PlantParams:
    """Mark a v3 preset: exact re-tapering in grow() (see PlantParams.exact_taper)."""
    p.exact_taper = True
    return p


def _lat(name, sub, a, lmax, lmaxs, ln, lns, la, las, lb, r, rs, theta_deg, succ=(),
         taper=0.0008, apical=0.2, tropismT=TROPISM_EXO, tropismN=1.0, tropismS=0.8,
         gsa_deg=None, rlt=60.0, rlts=15.0, a_dev_frac=0.25, thetas=0.3):
    """A lateral root type with the v3 (realistic) taper."""
    return RootTypeParams(
        name=name, subType=sub, lb=lb, la=la, las=las, ln=ln, lns=lns,
        lmax=lmax, lmaxs=lmaxs, r=r, rs=rs, a=a, a_dev=a * a_dev_frac,
        theta=np.deg2rad(theta_deg), thetas=thetas,
        tropismT=tropismT, tropismN=tropismN, tropismS=tropismS,
        gsa=None if gsa_deg is None else np.deg2rad(gsa_deg),
        rlt=rlt, rlts=rlts, successors=tuple(succ),
        taper_per_cm=taper, apical_taper_cm=apical, tip_radius_frac=_V3_TIP,
    )


def _fibrous_v3() -> PlantParams:
    """generic_fibrous with the taper fixed: the default for unknown grasses."""
    t1 = RootTypeParams(
        name="axis", subType=1, lb=1.0, lbs=0.4, la=3.0, las=1.0,
        ln=0.5, lns=0.2, lmax=60.0, lmaxs=15.0,
        r=2.0, rs=0.6, a=0.045, a_dev=0.010, theta=0.0,
        tropismT=TROPISM_GRAVI, tropismN=3.0, tropismS=0.45,
        gsa=np.deg2rad(35.0), successors=((2, 1.0),),
        taper_per_cm=0.00025, apical_taper_cm=0.6, tip_radius_frac=_V3_TIP,
    )
    t2 = _lat("lateral1", 2, a=0.020, lmax=8.0, lmaxs=3.0, ln=0.3, lns=0.12,
              la=1.0, las=0.4, lb=0.3, r=0.9, rs=0.25, theta_deg=70.0,
              succ=((3, 0.5),), taper=0.0007, apical=0.2, tropismN=1.0,
              tropismS=0.8, rlt=50.0)
    t3 = _lat("lateral2", 3, a=0.010, lmax=2.0, lmaxs=0.8, ln=0.2, lns=0.07,
              la=0.5, las=0.2, lb=0.1, r=0.4, rs=0.12, theta_deg=80.0,
              taper=0.0012, apical=0.12, tropismN=0.0, tropismS=1.1, rlt=28.0,
              rlts=8.0, thetas=0.35)
    return _v3_plant(PlantParams(name="fibrous_v3", types={1: t1, 2: t2, 3: t3},
                       basal_type=1, first_basal=5.0, delay_basal=3.0, max_basal=12))


def _switchgrass() -> PlantParams:
    """Many long, steep nodal axes plus fine laterals (PRMI Switchgrass is the
    only PRMI crop imaged at the synthetic 11.81 px/mm)."""
    t1 = RootTypeParams(
        name="primary", subType=1, lb=1.0, lbs=0.3, la=3.0, las=1.0,
        ln=0.4, lns=0.15, lmax=120.0, lmaxs=20.0,
        r=2.5, rs=0.5, a=0.040, a_dev=0.008, theta=0.0,
        tropismT=TROPISM_GRAVI, tropismN=4.0, tropismS=0.35,
        gsa=np.deg2rad(15.0), successors=((2, 1.0),),
        taper_per_cm=0.0002, apical_taper_cm=0.8, tip_radius_frac=_V3_TIP,
    )
    t2 = _lat("lateral1", 2, a=0.015, lmax=6.0, lmaxs=2.0, ln=0.25, lns=0.08,
              la=0.8, las=0.3, lb=0.3, r=0.8, rs=0.2, theta_deg=70.0,
              succ=((3, 0.5),), taper=0.0006, apical=0.2, tropismN=1.0,
              tropismS=0.8, rlt=50.0)
    t3 = _lat("lateral2", 3, a=0.008, lmax=1.5, lmaxs=0.6, ln=0.2, lns=0.07,
              la=0.4, las=0.15, lb=0.1, r=0.35, rs=0.1, theta_deg=80.0,
              taper=0.001, apical=0.1, tropismN=0.0, tropismS=1.1, rlt=30.0,
              rlts=8.0, thetas=0.35)
    tb = RootTypeParams(
        name="nodal", subType=4, lb=1.5, lbs=0.5, la=4.0, las=1.0,
        ln=0.4, lns=0.15, lmax=150.0, lmaxs=25.0,
        r=2.5, rs=0.5, a=0.045, a_dev=0.008, theta=np.deg2rad(30.0), thetas=0.15,
        tropismT=TROPISM_GRAVI, tropismN=3.0, tropismS=0.4,
        gsa=np.deg2rad(20.0), successors=((2, 1.0),),
        taper_per_cm=0.0002, apical_taper_cm=0.8, tip_radius_frac=_V3_TIP,
    )
    return _v3_plant(PlantParams(name="switchgrass", types={1: t1, 2: t2, 3: t3, 4: tb},
                       basal_type=4, first_basal=6.0, delay_basal=2.0, max_basal=30))


def _taproot(a, lmax, lmaxs, r, gsa_deg=5.0, taper=0.0006, tip=0.5, ln=0.6, lb=1.0,
             la=3.0, tropismN=5.0, tropismS=0.25, succ=((2, 1.0),)):
    """A dicot taproot: straight, gravitropic, thinning slowly with depth."""
    return RootTypeParams(
        name="taproot", subType=1, lb=lb, lbs=0.3, la=la, las=1.0,
        ln=ln, lns=0.2 * ln, lmax=lmax, lmaxs=lmaxs,
        r=r, rs=0.2 * r, a=a, a_dev=0.1 * a, theta=0.0,
        tropismT=TROPISM_GRAVI, tropismN=tropismN, tropismS=tropismS,
        gsa=np.deg2rad(gsa_deg), successors=tuple(succ),
        taper_per_cm=taper, apical_taper_cm=1.0, tip_radius_frac=tip,
    )


def _chicory() -> PlantParams:
    """Deep straight fleshy taproot with long oblique laterals (grid_counted /
    Zenodo chicory).  Nominal taproot 2.4 mm: thicker at the crown in reality,
    but the tube images its thinner, deeper part."""
    t1 = _taproot(a=0.120, lmax=200.0, lmaxs=30.0, r=2.0, gsa_deg=0.0, taper=0.0008,
                  tip=0.5, ln=0.8, tropismN=6.0, tropismS=0.2)
    t2 = _lat("lateral1", 2, a=0.025, lmax=40.0, lmaxs=10.0, ln=0.5, lns=0.15,
              la=2.0, las=0.6, lb=0.5, r=1.2, rs=0.3, theta_deg=75.0,
              succ=((3, 0.6),), taper=0.0002, apical=0.4, tropismT=TROPISM_GRAVI,
              tropismN=1.5, tropismS=0.5, gsa_deg=40.0, rlt=80.0, rlts=20.0)
    t3 = _lat("lateral2", 3, a=0.010, lmax=4.0, lmaxs=1.5, ln=0.3, lns=0.1,
              la=0.8, las=0.3, lb=0.2, r=0.5, rs=0.15, theta_deg=80.0,
              taper=0.0008, apical=0.15, tropismN=0.0, tropismS=1.0, rlt=40.0,
              rlts=10.0)
    return _v3_plant(PlantParams(name="chicory", types={1: t1, 2: t2, 3: t3},
                       basal_type=None, first_basal=7.0, delay_basal=7.0, max_basal=0))


def _cotton() -> PlantParams:
    """Taproot dicot with plagiotropic laterals in the topsoil."""
    t1 = _taproot(a=0.115, lmax=150.0, lmaxs=25.0, r=2.2, taper=0.0009, tip=0.4)
    t2 = _lat("lateral1", 2, a=0.035, lmax=35.0, lmaxs=10.0, ln=0.5, lns=0.15,
              la=2.0, las=0.6, lb=0.5, r=1.2, rs=0.3, theta_deg=85.0,
              succ=((3, 0.6),), taper=0.0003, apical=0.4, tropismT=TROPISM_GRAVI,
              tropismN=1.5, tropismS=0.5, gsa_deg=70.0, rlt=80.0, rlts=20.0)
    t3 = _lat("lateral2", 3, a=0.014, lmax=6.0, lmaxs=2.0, ln=0.35, lns=0.1,
              la=0.8, las=0.3, lb=0.2, r=0.6, rs=0.15, theta_deg=80.0,
              succ=((5, 0.4),), taper=0.0006, apical=0.15, tropismN=0.5,
              tropismS=0.9, rlt=45.0, rlts=12.0)
    t5 = _lat("lateral3", 5, a=0.008, lmax=1.5, lmaxs=0.6, ln=0.2, lns=0.07,
              la=0.4, las=0.15, lb=0.1, r=0.35, rs=0.1, theta_deg=80.0,
              taper=0.001, apical=0.1, tropismN=0.0, tropismS=1.1, rlt=30.0,
              rlts=8.0)
    return _v3_plant(PlantParams(name="cotton", types={1: t1, 2: t2, 3: t3, 5: t5},
                       basal_type=None, first_basal=7.0, delay_basal=7.0, max_basal=0))


def _sunflower() -> PlantParams:
    """Strong taproot with dense second- and third-order laterals."""
    t1 = _taproot(a=0.120, lmax=170.0, lmaxs=25.0, r=2.5, taper=0.0008, tip=0.45, ln=0.5)
    t2 = _lat("lateral1", 2, a=0.040, lmax=50.0, lmaxs=12.0, ln=0.3, lns=0.1,
              la=2.0, las=0.6, lb=0.5, r=1.4, rs=0.3, theta_deg=75.0,
              succ=((3, 0.7),), taper=0.0003, apical=0.4, tropismT=TROPISM_GRAVI,
              tropismN=1.5, tropismS=0.5, gsa_deg=60.0, rlt=80.0, rlts=20.0)
    t3 = _lat("lateral2", 3, a=0.015, lmax=8.0, lmaxs=2.5, ln=0.3, lns=0.1,
              la=1.0, las=0.3, lb=0.2, r=0.7, rs=0.2, theta_deg=80.0,
              succ=((5, 0.5),), taper=0.0006, apical=0.2, tropismN=0.5,
              tropismS=0.9, rlt=45.0, rlts=12.0)
    t5 = _lat("lateral3", 5, a=0.008, lmax=1.5, lmaxs=0.6, ln=0.2, lns=0.07,
              la=0.4, las=0.15, lb=0.1, r=0.35, rs=0.1, theta_deg=80.0,
              taper=0.001, apical=0.1, tropismN=0.0, tropismS=1.1, rlt=30.0,
              rlts=8.0)
    return _v3_plant(PlantParams(name="sunflower", types={1: t1, 2: t2, 3: t3, 5: t5},
                       basal_type=None, first_basal=7.0, delay_basal=7.0, max_basal=0))


def _peanut() -> PlantParams:
    """Taproot with relatively thick, long, near-horizontal laterals."""
    t1 = _taproot(a=0.100, lmax=120.0, lmaxs=20.0, r=1.8, taper=0.0007, tip=0.5, ln=0.5)
    t2 = _lat("lateral1", 2, a=0.030, lmax=30.0, lmaxs=8.0, ln=0.4, lns=0.12,
              la=1.5, las=0.5, lb=0.4, r=1.0, rs=0.25, theta_deg=85.0,
              succ=((3, 0.5),), taper=0.0003, apical=0.3, tropismT=TROPISM_GRAVI,
              tropismN=1.5, tropismS=0.5, gsa_deg=70.0, rlt=70.0, rlts=18.0)
    t3 = _lat("lateral2", 3, a=0.012, lmax=4.0, lmaxs=1.5, ln=0.3, lns=0.1,
              la=0.6, las=0.2, lb=0.15, r=0.5, rs=0.15, theta_deg=80.0,
              taper=0.0008, apical=0.15, tropismN=0.0, tropismS=1.0, rlt=40.0,
              rlts=10.0)
    return _v3_plant(PlantParams(name="peanut", types={1: t1, 2: t2, 3: t3},
                       basal_type=None, first_basal=7.0, delay_basal=7.0, max_basal=0))


def _sesame() -> PlantParams:
    """Thin taproot with profuse, very fine laterals (PRMI Sesame has the
    narrowest masks of all PRMI crops)."""
    t1 = _taproot(a=0.080, lmax=110.0, lmaxs=20.0, r=1.8, taper=0.0006, tip=0.5, ln=0.4)
    t2 = _lat("lateral1", 2, a=0.018, lmax=15.0, lmaxs=5.0, ln=0.25, lns=0.08,
              la=1.0, las=0.4, lb=0.3, r=0.9, rs=0.25, theta_deg=75.0,
              succ=((3, 0.7),), taper=0.0005, apical=0.2, tropismN=1.0,
              tropismS=0.8, rlt=55.0, rlts=15.0)
    t3 = _lat("lateral2", 3, a=0.007, lmax=2.0, lmaxs=0.8, ln=0.15, lns=0.05,
              la=0.4, las=0.15, lb=0.1, r=0.4, rs=0.12, theta_deg=80.0,
              taper=0.0008, apical=0.1, tropismN=0.0, tropismS=1.1, rlt=30.0,
              rlts=8.0)
    return _v3_plant(PlantParams(name="sesame", types={1: t1, 2: t2, 3: t3},
                       basal_type=None, first_basal=7.0, delay_basal=7.0, max_basal=0))


def _papaya() -> PlantParams:
    """Woody, coarse adventitious roots with few laterals (PRMI Papaya masks
    have a median w2e of 18 px)."""
    t1 = _taproot(a=0.120, lmax=120.0, lmaxs=20.0, r=1.5, taper=0.0004, tip=0.6,
                  ln=1.5, la=4.0, tropismN=3.0, tropismS=0.35, gsa_deg=20.0)
    t2 = _lat("lateral1", 2, a=0.050, lmax=40.0, lmaxs=10.0, ln=1.2, lns=0.4,
              la=3.0, las=1.0, lb=1.0, r=1.0, rs=0.25, theta_deg=60.0,
              succ=((3, 0.5),), taper=0.0002, apical=0.6, tropismT=TROPISM_GRAVI,
              tropismN=1.5, tropismS=0.45, gsa_deg=45.0, rlt=120.0, rlts=30.0)
    t3 = _lat("lateral2", 3, a=0.020, lmax=6.0, lmaxs=2.0, ln=0.6, lns=0.2,
              la=1.0, las=0.3, lb=0.3, r=0.5, rs=0.15, theta_deg=75.0,
              taper=0.0005, apical=0.2, tropismN=0.5, tropismS=0.8, rlt=60.0,
              rlts=15.0)
    tb = RootTypeParams(
        name="adventitious", subType=4, lb=2.0, lbs=0.5, la=4.0, las=1.0,
        ln=1.5, lns=0.4, lmax=90.0, lmaxs=15.0,
        r=1.4, rs=0.3, a=0.115, a_dev=0.012, theta=np.deg2rad(55.0), thetas=0.2,
        tropismT=TROPISM_GRAVI, tropismN=2.0, tropismS=0.4,
        gsa=np.deg2rad(40.0), successors=((2, 1.0),),
        taper_per_cm=0.0004, apical_taper_cm=0.8, tip_radius_frac=_V3_TIP,
    )
    return _v3_plant(PlantParams(name="papaya", types={1: t1, 2: t2, 3: t3, 4: tb},
                       basal_type=4, first_basal=8.0, delay_basal=4.0, max_basal=6))


def _v3_taper(plant: PlantParams, name: str, factor: float = 4.0,
              max_axis_a: float = 0.12) -> PlantParams:
    """A frozen v1 preset with the taper fixed: ``taper_per_cm / factor``, tip
    floor 0.65, and order-0 radii capped at ``max_axis_a`` (2.4 mm).  Every
    other parameter -- architecture, rates, angles -- is the original."""
    p = copy.deepcopy(plant)
    order0 = {1} | ({p.basal_type} if p.basal_type is not None else set())
    for st, t in list(p.types.items()):
        a = min(t.a, max_axis_a) if st in order0 else t.a
        a_dev = t.a_dev * (a / t.a) if t.a > 0 else t.a_dev
        p.types[st] = replace(t, taper_per_cm=t.taper_per_cm / factor,
                              tip_radius_frac=_V3_TIP, a=a, a_dev=a_dev)
    p.name = name
    return _v3_plant(p)


def _zea_mays_v3() -> PlantParams:
    return _v3_taper(_maize(), "zea_mays_v3")


def _glycine_max_v3() -> PlantParams:
    return _v3_taper(_soybean(), "glycine_max_v3")


SPECIES = {
    "zea_mays": _maize,
    "glycine_max": _soybean,
    "generic_fibrous": _generic_fibrous,
    # v3: realistic taper (the three above stay byte-identical)
    "fibrous_v3": _fibrous_v3,
    "switchgrass": _switchgrass,
    "chicory": _chicory,
    "cotton": _cotton,
    "sunflower": _sunflower,
    "peanut": _peanut,
    "sesame": _sesame,
    "papaya": _papaya,
    "zea_mays_v3": _zea_mays_v3,
    "glycine_max_v3": _glycine_max_v3,
}

# the presets whose taper/width were fixed for v3 (tests and notebooks use this)
V3_SPECIES = ("fibrous_v3", "switchgrass", "chicory", "cotton", "sunflower", "peanut",
              "sesame", "papaya", "zea_mays_v3", "glycine_max_v3")

SPECIES_NOTES = {
    "generic_fibrous": "v1 default, FROZEN; taper drives laterals to 1-2 px",
    "zea_mays": "v1 maize, FROZEN; same taper bug",
    "glycine_max": "v1 soybean, FROZEN; same taper bug",
    "fibrous_v3": "generic_fibrous with the taper fixed (unknown grasses)",
    "switchgrass": "30 steep nodal axes + fine laterals; PRMI Switchgrass (11.81 px/mm)",
    "chicory": "deep straight taproot, long oblique laterals; grid_counted / Zenodo",
    "cotton": "taproot dicot, near-horizontal laterals in the topsoil",
    "sunflower": "strong taproot, dense 2nd/3rd-order laterals",
    "peanut": "taproot, relatively thick and long laterals",
    "sesame": "thin taproot, profuse very fine laterals (narrowest PRMI masks)",
    "papaya": "woody coarse adventitious roots, few laterals (PRMI Papaya p50 18 px)",
    "zea_mays_v3": "zea_mays with the taper fixed (crown roots dominate)",
    "glycine_max_v3": "glycine_max with the taper fixed",
}


def get_species(name: str) -> PlantParams:
    if name not in SPECIES:
        raise KeyError("unknown species %r; available: %s" % (name, sorted(SPECIES)))
    return copy.deepcopy(SPECIES[name]())


def parse_mix(spec) -> Dict[str, float]:
    """``'switchgrass:.2, cotton:.1'`` (or a dict) -> weights normalised to 1.

    Order of first appearance is kept (draws are made against it, so the same
    string always gives the same systems); a repeated name adds up; a name
    without ``:w`` weighs 1.  ``''``/None -> ``{}`` (no mix: the single
    ``species`` of the config is used).  An unknown species raises KeyError
    at parse time, before anything is rendered with a typo.
    """
    if spec is None:
        return {}
    if isinstance(spec, dict):
        items = [(str(k), float(v)) for k, v in spec.items()]
    else:
        items = []
        for tok in str(spec).split(","):
            tok = tok.strip()
            if not tok:
                continue
            name, _, w = tok.partition(":")
            items.append((name.strip(), float(w) if w.strip() else 1.0))
    out: Dict[str, float] = {}
    for name, w in items:
        if name not in SPECIES:
            raise KeyError("unknown species %r in mix; available: %s" % (name, sorted(SPECIES)))
        if not (w >= 0.0):
            raise ValueError("negative or NaN weight for %r" % name)
        out[name] = out.get(name, 0.0) + w
    tot = sum(out.values())
    if not out:
        return {}
    if tot <= 0:
        raise ValueError("species mix %r has zero total weight" % (spec,))
    return {k: v / tot for k, v in out.items()}


def scale_tortuosity(plant: PlantParams, mult: float) -> PlantParams:
    """Copy of ``plant`` with every root type's ``tropismS`` times ``mult``.

    v2 roots were visibly too straight.  tropismS is the per-step heading
    sigma, so this changes curviness WITHOUT changing the number of random
    draws (heading_step draws the same candidates, only scaled), the growth
    rates or the branching -- and the sub-dx Brownian bridge in
    :func:`mrsim.grow.subdivide`, which reads the same tropismS, scales with it.
    """
    p = copy.deepcopy(plant)
    m = float(mult)
    for st, t in list(p.types.items()):
        p.types[st] = replace(t, tropismS=float(t.tropismS) * m)
    return p
