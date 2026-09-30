"""Seed-ensemble harness — turn single-seed, noise-limited readouts into
statistics with confidence intervals, the rigorous form of the stage-1 result.

Reports the elastic modulus as a mean ± 95% CI over seeds (so its sign is
defensible, not a coin-flip), the ensemble-averaged stress relaxation and its τ
(or an honest below-detection verdict), and the network's mean coordination z —
the number that explains WHY the modulus is ~0: a 2-bead-rod network with NC1×1 +
7S×3 connectivity has z ≤ 3, below the 2D central-force rigidity threshold z = 4,
so it is floppy regardless of crosslink density (a real, spec-consistent finding).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from .params import CollagenParams
from .stage1 import run_stage1, _relaxation_time


RIGIDITY_THRESHOLD_2D = 4.0  # Maxwell central-force isostatic point in 2D


def mean_coordination(n_crosslinks: int, n_rods: int) -> float:
    """Mean bead coordination z. Beads = 2·n_rods; bonds = n_rods (intra) +
    n_crosslinks; z = 2·bonds / beads = 1 + n_crosslinks/n_rods. Max (all ends
    bonded: n_rods/2 NC1 + 3n_rods/2 7S = 2·n_rods crosslinks) → z = 3 < 4."""
    return 1.0 + n_crosslinks / max(1, n_rods)


@dataclass
class EnsembleResult:
    n_seeds: int
    modulus_mean: float
    modulus_ci95: float          # half-width of the 95% CI
    modulus_consistent_with_zero: bool
    tau_ensemble_lj: float
    tau_resolved: bool
    n_crosslinks_mean: float
    coordination_z: float
    is_floppy: bool              # z below the 2D rigidity threshold
    viscous_trace_mean: list
    viscous_t: list
    per_seed_modulus: list

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop('viscous_trace_mean'); d.pop('viscous_t')
        return d


def stage1_ensemble(params: CollagenParams | None = None,
                    seeds: list[int] | None = None,
                    sample_dt: float = 8.0) -> EnsembleResult:
    p0 = params or CollagenParams()
    seeds = seeds or [11, 22, 33, 44, 55, 66, 77, 88]

    moduli, xlinks, vis_traces = [], [], []
    for sd in seeds:
        p = CollagenParams(**{**p0.to_dict(), 'seed': sd})
        r = run_stage1(p, sample_dt=sample_dt)
        moduli.append(r.elastic_modulus_lj)
        xlinks.append(r.n_crosslinks_assembled)
        ph = np.array(r.trace['phase']); sg = np.array(r.trace['sigma'])
        vis_traces.append(sg[ph == 'hold_viscous'])

    moduli = np.array(moduli)
    n = len(seeds)
    sem = moduli.std(ddof=1) / np.sqrt(n) if n > 1 else float('nan')
    ci95 = 1.96 * sem
    mod_mean = float(moduli.mean())

    m = min(len(v) for v in vis_traces)
    mean_vis = np.mean([v[:m] for v in vis_traces], axis=0)
    tv = np.arange(m) * sample_dt
    tau = _relaxation_time(tv, mean_vis)

    xl_mean = float(np.mean(xlinks))
    z = mean_coordination(xl_mean, p0.n_rods)

    return EnsembleResult(
        n_seeds=n,
        modulus_mean=mod_mean,
        modulus_ci95=float(ci95),
        modulus_consistent_with_zero=bool(abs(mod_mean) <= ci95) if np.isfinite(ci95) else True,
        tau_ensemble_lj=float(tau),
        tau_resolved=bool(np.isfinite(tau)),
        n_crosslinks_mean=xl_mean,
        coordination_z=float(z),
        is_floppy=bool(z < RIGIDITY_THRESHOLD_2D),
        viscous_trace_mean=[float(x) for x in mean_vis],
        viscous_t=[float(x) for x in tv],
        per_seed_modulus=[float(x) for x in moduli],
    )
