"""Stage 2: the growing substrate (flat). Impose an equibiaxial in-plane strain
rate ε̇ and read off the steady-state stress σ(ε̇).

The spec's stage 2 inverts Barrientos 2026 (who grow the network at fixed stress
and read the growth rate): here we impose ε̇ and read σ. At slow rates the network
remodels fast enough to flow — a viscous response with an effective viscosity
η = σ/ε̇ that is roughly rate-independent. As ε̇ rises past the crosslink turnover
rate the network can no longer relax between deformation increments and the
response turns elastic (σ climbs faster than linearly, η rises).

Why flat is enough (spec): a patch of a sphere of radius R growing at Ṙ is
stretched at ε̇ = Ṙ/R in every in-plane direction, so a flat biaxially stretching
slab IS the growing shell to leading order while R ≫ mesh (tens of µm vs ~0.1 µm).

Chemistry is ON throughout the stretch (the growing, remodelling substrate),
unlike stage 1 where the ramp is run with the crosslink chemistry frozen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .collagen import CollagenNetworkProcess
from .params import CollagenParams


@dataclass
class RatePoint:
    strain_rate: float
    sigma_steady: float          # steady-state in-plane network stress
    sigma_std: float             # spread over the measurement window
    eta_effective: float         # σ/ε̇
    n_crosslinks: int


@dataclass
class Stage2Result:
    points: list[RatePoint]
    viscosity_lowrate: float     # η fit at the slow-rate (viscous) end
    crossover_rate: float        # ε̇ where the response departs from viscous (η rises)
    trace: dict[str, list] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            'rates': [p.strain_rate for p in self.points],
            'sigma_steady': [p.sigma_steady for p in self.points],
            'eta_effective': [p.eta_effective for p in self.points],
            'n_crosslinks': [p.n_crosslinks for p in self.points],
            'viscosity_lowrate': self.viscosity_lowrate,
            'crossover_rate': self.crossover_rate,
        }


def _steady_stress(proc, rate, params, sample_dt, measure_strain, settle_strain):
    """Stretch at `rate` with chemistry on; return (mean σ, std σ, n_crosslinks)
    over the window after `settle_strain` up to `measure_strain`."""
    sigmas = []
    xlinks = last = None
    total_strain = settle_strain + measure_strain
    n = max(1, int(round((total_strain / rate) / sample_dt)))
    for _ in range(n):
        # growing, remodelling substrate: make + break both on (turnover)
        last = proc.update(
            {'strain_rate': rate, 'make_on': 1.0, 'break_on': 1.0}, sample_dt)
        if last['strain'] >= settle_strain:
            sigmas.append(last['sigma_inplane'])
        xlinks = last['n_crosslinks']
    sigmas = np.array(sigmas) if sigmas else np.array([last['sigma_inplane']])
    return float(np.mean(sigmas)), float(np.std(sigmas)), int(xlinks)


def run_stage2(params: CollagenParams | None = None,
               rates: list[float] | None = None,
               sample_dt: float = 8.0,
               settle_strain: float = 0.05,
               measure_strain: float = 0.15,
               working_directory: str = '') -> Stage2Result:
    """Sweep ε̇ and return σ(ε̇), the low-rate viscosity, and the crossover rate."""
    from process_bigraph import allocate_core

    p = params or CollagenParams()
    if rates is None:
        # a decade-and-a-bit around the default rate
        rates = [5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2]

    t_assemble = p.assemble_steps * p.timestep
    points: list[RatePoint] = []
    for rate in rates:
        proc = CollagenNetworkProcess(
            config={'params': p.to_dict(), 'working_directory': working_directory},
            core=allocate_core())
        proc.initial_state()
        # assemble the network to steady state (make + break on, no strain)
        n_as = max(1, int(round(t_assemble / sample_dt)))
        for _ in range(n_as):
            proc.update({'strain_rate': 0.0, 'make_on': 1.0, 'break_on': 1.0}, sample_dt)
        sig, std, xl = _steady_stress(proc, rate, p, sample_dt,
                                      measure_strain, settle_strain)
        proc.close()
        points.append(RatePoint(strain_rate=rate, sigma_steady=sig, sigma_std=std,
                                eta_effective=sig / rate if rate else float('nan'),
                                n_crosslinks=xl))

    r = np.array([pt.strain_rate for pt in points])
    eta = np.array([pt.eta_effective for pt in points])
    # low-rate viscosity: median η over the slowest third of the sweep
    k = max(1, len(points) // 3)
    viscosity_lowrate = float(np.median(eta[:k]))
    # crossover: first rate where η exceeds 1.5x the low-rate plateau
    crossover = float('nan')
    for pt in points:
        if pt.eta_effective > 1.5 * viscosity_lowrate:
            crossover = pt.strain_rate
            break

    return Stage2Result(
        points=points,
        viscosity_lowrate=viscosity_lowrate,
        crossover_rate=crossover,
        trace={'rates': list(r), 'sigma_steady': [pt.sigma_steady for pt in points],
               'eta_effective': [pt.eta_effective for pt in points]},
    )
