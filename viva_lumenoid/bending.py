"""Junction-bending sweep (FP1): does bending rigidify the floppy network?

Turns on junction bending — harmonic angle restraints pinned to each crosslink's
AS-FORMED geometry (CollagenNetworkProcess._rebuild_restraints) — and measures the
**athermal** elastic modulus (energy minimisation, deterministic) vs the bending
stiffness bending_k.

What it establishes (see the bm-v4 study):
  * bending_k = 0 (floppy) → modulus ≈ 0, measured cleanly and reproducibly
    (the athermal minimisation removes the thermal chaos that plagued the dynamic
    measurement).
  * bending_k > 0 → a LARGE mechanical response (|E| jumps to tens–hundreds): the
    network is no longer floppy. But the stiff, pre-stressed, disordered network
    has many near-degenerate minima, so the modulus VALUE is ill-conditioned /
    not reproducible at v1 network size (large scatter, occasional pathological
    negative minima). A clean reproducible modulus needs a larger network (self-
    averaging) and a pre-stress-free reference — a scoped v2 task.

So FP1 confirms bending's role qualitatively (it dominates the mechanics) but
does not yet yield a calibratable modulus; the honest statistics (median, robust
spread, fraction with a sane positive value) are reported rather than a number.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from .params import CollagenParams
from .collagen import CollagenNetworkProcess


def _assemble_frozen(params: CollagenParams):
    """Build a process, assemble the crosslinked network, freeze it (pinning the
    junction-bending restraints when bending is on). Returns the live process."""
    from process_bigraph import allocate_core
    proc = CollagenNetworkProcess(config={'params': params.to_dict()}, core=allocate_core())
    proc.initial_state()
    t_assemble = params.assemble_steps * params.timestep
    for _ in range(max(1, int(round(t_assemble / 8.0)))):
        proc.update({'strain_rate': 0.0, 'make_on': 1.0, 'break_on': 1.0}, 8.0)
    proc.update({'strain_rate': 0.0, 'make_on': 0.0, 'break_on': 0.0}, 8.0)  # freeze + pin
    return proc


@dataclass
class BendingPoint:
    bending_k: float
    modulus_median: float
    modulus_iqr: float
    frac_positive: float             # fraction of seeds with a sane positive modulus
    reproducible: bool               # IQR small relative to |median| (a usable value)
    per_seed: list


@dataclass
class BendingResult:
    points: list
    large_response: bool                 # |median| grows by orders of magnitude above floppy

    def summary(self) -> dict[str, Any]:
        return {'points': [asdict(p) for p in self.points], 'large_response': self.large_response}


DEFAULT_KS = [0.0, 40.0, 80.0]


def run_bending_sweep(bending_ks=None, seeds=None,
                      base: CollagenParams | None = None) -> BendingResult:
    bending_ks = DEFAULT_KS if bending_ks is None else bending_ks
    seeds = seeds or [11, 22, 33, 44]
    b = base or CollagenParams(n_rods=200, box_xy=20.0)
    points = []
    for k in bending_ks:
        # bending needs the clamped dt≈0.002 → scale assemble steps for the same LJ time
        asm = int(round(60.0 / 0.002)) if k > 0 else 6000
        dt = 0.002 if k > 0 else 0.01
        mods = []
        for sd in seeds:
            p = CollagenParams(**{**b.to_dict(), 'bending_k': k, 'timestep': dt,
                                  'assemble_steps': asm, 'seed': sd})
            proc = _assemble_frozen(p)
            try:
                mods.append(proc.athermal_modulus())
            finally:
                proc.close()
        mods = np.array(mods)
        med = float(np.median(mods))
        q1, q3 = np.percentile(mods, [25, 75])
        iqr = float(q3 - q1)
        points.append(BendingPoint(
            bending_k=k, modulus_median=med, modulus_iqr=iqr,
            frac_positive=float(np.mean(mods > 0)),
            reproducible=bool(abs(med) > 1e-3 and iqr < 0.5 * abs(med)),
            per_seed=[float(x) for x in mods]))
    floppy = abs(points[0].modulus_median)
    rigid = max(abs(p.modulus_median) for p in points)
    return BendingResult(points=points, large_response=bool(rigid > 10 * (floppy + 0.01)))
