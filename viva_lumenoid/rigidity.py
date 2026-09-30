"""Rigidity-percolation sweep (follow-up FP1): does raising crosslink
connectivity turn the elastic modulus on?

Stage-1 finds the v1 network is floppy (modulus ~0) with mean coordination
z ≈ 1.2. This sweep raises the max crosslinks per end (a *bundling* proxy — the
spec notes 5–7 protomers bundle into a strand, adding junctions) and measures the
ensemble modulus vs z. The prediction from Maxwell counting is that a central-
force network stays floppy until z crosses the 2D isostatic point z = 4.

The result (see the bm-v3 study) is that this topology cannot reach z = 4 —
NC1×k + 7S×m saturates geometrically below it — so the modulus stays ~0 across the
achievable range. Rigidity therefore requires *junction bending* (or true
bundling bonds), not more central-force crosslinks: the constraint the vertex-model
coupling must respect.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from .params import CollagenParams
from .stage1 import run_stage1
from .ensemble import mean_coordination, RIGIDITY_THRESHOLD_2D


@dataclass
class ConnectivityPoint:
    nc1_max: int
    svns_max: int
    make_prob: float
    z_mean: float
    modulus_mean: float
    modulus_ci95: float
    n_crosslinks_mean: float


@dataclass
class RigidityResult:
    points: list
    rigidity_threshold: float
    reached_rigid: bool

    def summary(self) -> dict[str, Any]:
        return {'points': [asdict(p) for p in self.points],
                'rigidity_threshold': self.rigidity_threshold,
                'reached_rigid': self.reached_rigid}


# (nc1_max, 7S_max, make_prob) rungs of increasing connectivity.
DEFAULT_RUNGS = [(1, 3, 0.12), (2, 4, 0.25), (3, 5, 0.40), (5, 8, 0.60)]


def run_connectivity_sweep(rungs=None, seeds=None,
                           base: CollagenParams | None = None,
                           sample_dt: float = 8.0) -> RigidityResult:
    rungs = rungs or DEFAULT_RUNGS
    seeds = seeds or [11, 22, 33]
    b = base or CollagenParams(n_rods=200, box_xy=20.0,
                               assemble_steps=8000, hold_steps=8000)
    points = []
    for nc1, sv, mp in rungs:
        mods, zs, xls = [], [], []
        for sd in seeds:
            p = CollagenParams(**{**b.to_dict(),
                                  'nc1_max_crosslinks': nc1, 'svns_max_crosslinks': sv,
                                  'make_prob': mp, 'seed': sd})
            r = run_stage1(p, sample_dt=sample_dt)
            mods.append(r.elastic_modulus_lj)
            xls.append(r.n_crosslinks_assembled)
            zs.append(mean_coordination(r.n_crosslinks_assembled, b.n_rods))
        mods = np.array(mods)
        sem = mods.std(ddof=1) / np.sqrt(len(seeds)) if len(seeds) > 1 else float('nan')
        points.append(ConnectivityPoint(
            nc1_max=nc1, svns_max=sv, make_prob=mp,
            z_mean=float(np.mean(zs)), modulus_mean=float(mods.mean()),
            modulus_ci95=float(1.96 * sem), n_crosslinks_mean=float(np.mean(xls))))
    reached = any(p.z_mean >= RIGIDITY_THRESHOLD_2D for p in points)
    return RigidityResult(points=points, rigidity_threshold=RIGIDITY_THRESHOLD_2D,
                          reached_rigid=reached)
