"""Stage 1: stretch -> hold -> switch chemistry on; report elastic modulus and
remodelling (stress-relaxation) time.

This is the v1 deliverable the spec names: "a standalone collagen IV network ...
run until it reports an elastic modulus and a remodelling time for the AICS
vertex model." We drive ``CollagenNetworkProcess`` directly through the four
phases and reduce the stress trace to two numbers.

The modulus is reported in LJ stress units AND converted to pascals with the
thermal energy scale (kT/a**3 ≈ 2.5 Pa) — but flagged, because whether v1 can
report a real modulus at all is still open (spec v1-decision #4). Treat the LJ
value / the ratio as primary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .collagen import CollagenNetworkProcess
from .params import CollagenParams


@dataclass
class Stage1Result:
    elastic_modulus_lj: float
    elastic_modulus_Pa: float
    relaxation_time_lj: float
    final_strain: float
    n_crosslinks_assembled: int
    sigma_prestretch_baseline: float
    sigma_elastic_plateau: float
    sigma_viscous_start: float
    sigma_viscous_end: float
    stress_noise: float          # thermal noise floor of the bond-virial stress
    relaxation_noise_limited: bool  # True when tau is below the noise floor at this size
    trace: dict[str, list]  # t, strain, sigma, n_crosslinks, phase
    snapshot: dict = None    # final network: positions, atom_types, box (for the mesh viz)
    frames: list = None      # time-ordered spatial snapshots for the simulation movie

    def summary(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d.pop('trace')
        d.pop('snapshot', None)
        d.pop('frames', None)
        return d


def _smooth(y: np.ndarray, w: int = 3) -> np.ndarray:
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode='same')


def _relaxation_time(t: np.ndarray, sigma: np.ndarray) -> float:
    """Estimate tau from sigma(t) = A*exp(-(t-t0)/tau) + C over the viscous hold.

    Robust, scipy-free: smooth, then fit ln(sigma - C) vs t after subtracting a
    baseline C (the final stress). Returns NaN when the decay is below the noise
    floor — an honest "not resolved at this network size", which is the calibration
    question the spec raises (decisions #4/#8/#10), not a silent zero.
    """
    if len(t) < 4:
        return float('nan')
    t = t - t[0]
    s = _smooth(sigma)
    C = float(s[-1])
    excess = s - C
    s0 = float(excess[0])
    # signal must clear the trace's own noise to be a real decay
    noise = float(np.std(sigma - _smooth(sigma)))
    if s0 <= max(2.0 * noise, 1e-6):
        return float('nan')
    mask = excess > 0.05 * s0
    if mask.sum() >= 3:
        A = np.vstack([t[mask], np.ones(mask.sum())]).T
        slope, _ = np.linalg.lstsq(A, np.log(excess[mask]), rcond=None)[0]
        if slope < 0:
            return float(-1.0 / slope)
    target = s0 / np.e
    below = np.where(excess <= target)[0]
    return float(t[below[0]]) if len(below) else float('nan')


def run_stage1(params: CollagenParams | None = None,
               strain_rate: float | None = None,
               sample_dt: float = 5.0,
               working_directory: str = '',
               capture_frames: bool = False,
               frame_stride: int = 2) -> Stage1Result:
    """Run the four-phase stage-1 protocol and return the reduced readouts.

    With ``capture_frames=True`` the spatial state (positions, box, atom types)
    is collected every ``frame_stride`` samples into ``result.frames`` — the
    trajectory that drives the simulation movie (viz.network_movie_figure).
    """
    p = params or CollagenParams()
    rate = strain_rate if strain_rate is not None else p.strain_rate

    from process_bigraph import allocate_core
    proc = CollagenNetworkProcess(
        config={'params': p.to_dict(), 'working_directory': working_directory},
        core=allocate_core())

    # phase durations in LJ time
    t_assemble = p.assemble_steps * p.timestep
    t_stretch = p.target_strain / rate
    t_hold = p.hold_steps * p.timestep

    trace = {'t': [], 'strain': [], 'sigma': [], 'n_crosslinks': [], 'phase': []}
    frames: list = []
    _ctr = {'i': 0}

    def drive(duration, strain_rate_val, make_on, break_on, phase, t_offset):
        n = max(1, int(round(duration / sample_dt)))
        last = None
        for _ in range(n):
            last = proc.update(
                {'strain_rate': strain_rate_val, 'make_on': make_on,
                 'break_on': break_on}, sample_dt)
            t_offset += sample_dt
            trace['t'].append(t_offset)
            trace['strain'].append(last['strain'])
            trace['sigma'].append(last['sigma_inplane'])
            trace['n_crosslinks'].append(last['n_crosslinks'])
            trace['phase'].append(phase)
            if capture_frames and _ctr['i'] % max(1, frame_stride) == 0:
                frames.append({
                    't': t_offset, 'strain': float(last['strain']),
                    'sigma': float(last['sigma_inplane']),
                    'n_crosslinks': int(last['n_crosslinks']), 'phase': phase,
                    'positions': np.asarray(last['positions'], dtype=float).round(3),
                    'atom_types': np.asarray(last['atom_types']),
                    'box': list(last['box_dimensions']),
                })
            _ctr['i'] += 1
        return last, t_offset

    proc.initial_state()
    t = 0.0
    # assemble to the make/break steady state (both on → genuine turnover count)
    assembled, t = drive(t_assemble, 0.0, 1.0, 1.0, 'assemble', t)
    n_xlinks = assembled['n_crosslinks']
    # Equilibrate FROZEN so the pre-stress settles before we measure a reference
    # (spec #8 flags a too-short equilibration as one of the stage-1 errors).
    relaxed, t = drive(t_hold, 0.0, 0.0, 0.0, 'relax', t)
    baseline = float(relaxed['sigma_inplane'])  # settled pre-stretch reference
    _, t = drive(t_stretch, rate, 0.0, 0.0, 'stretch', t)
    elastic_last, t = drive(t_hold, 0.0, 0.0, 0.0, 'hold_elastic', t)
    # viscous relaxation: BREAKAGE ONLY (make off) so formation doesn't confound τ
    viscous_last, t = drive(t_hold, 0.0, 0.0, 1.0, 'hold_viscous', t)
    snapshot = {
        'positions': viscous_last['positions'],
        'atom_types': viscous_last['atom_types'],
        'box': viscous_last['box_dimensions'],
        'n_rods': p.n_rods,
    }
    proc.close()

    ph = np.array(trace['phase'])
    tt = np.array(trace['t'])
    sig = np.array(trace['sigma'])
    stn = np.array(trace['strain'])

    el = ph == 'hold_elastic'
    sigma_plateau = float(np.median(sig[el])) if el.any() else float('nan')
    final_strain = float(elastic_last['strain'])
    # Elastic modulus = tangent slope of the stress-strain curve during the
    # active stretch ramp (baseline pre-stress removed). A linear fit is more
    # robust than a single plateau point, which creeps as the un-crosslinked
    # rods relax. Near-zero/very-soft is expected: the published network's own
    # linear modulus is ≈0.03 Pa (spec decision #4).
    stm = ph == 'stretch'
    if stm.sum() >= 2 and np.ptp(stn[stm]) > 1e-6:
        A = np.vstack([stn[stm], np.ones(stm.sum())]).T
        slope, _ = np.linalg.lstsq(A, sig[stm] - baseline, rcond=None)[0]
        modulus_lj = float(slope)
    else:
        modulus_lj = (sigma_plateau - baseline) / final_strain if final_strain else float('nan')

    vis = ph == 'hold_viscous'
    tau = _relaxation_time(tt[vis], sig[vis]) if vis.sum() >= 4 else float('nan')
    stress_noise = float(np.std(sig[vis] - _smooth(sig[vis]))) if vis.any() else float('nan')

    return Stage1Result(
        elastic_modulus_lj=modulus_lj,
        elastic_modulus_Pa=modulus_lj * p.kT_per_a3_Pa,
        relaxation_time_lj=tau,
        final_strain=final_strain,
        n_crosslinks_assembled=int(n_xlinks),
        sigma_prestretch_baseline=baseline,
        sigma_elastic_plateau=sigma_plateau,
        sigma_viscous_start=float(sig[vis][0]) if vis.any() else float('nan'),
        sigma_viscous_end=float(sig[vis][-1]) if vis.any() else float('nan'),
        stress_noise=stress_noise,
        relaxation_noise_limited=bool(np.isnan(tau)),
        trace=trace,
        snapshot=snapshot,
        frames=(frames if capture_frames else None),
    )


def stage1_composite_spec(params: CollagenParams | None = None) -> dict:
    """process-bigraph composite document wiring controller -> network -> emitter.

    This is the dashboard/`/viva-run` form of stage 1: a Composite that runs the
    staged protocol on a shared timeline. The direct ``run_stage1`` driver above
    is the analysis path; this is the reproducible spec.
    """
    p = params or CollagenParams()
    rate = p.strain_rate
    return {
        'controller': {
            '_type': 'process', 'address': 'local:StagedStretchController',
            'config': {
                'strain_rate': rate,
                't_assemble': p.assemble_steps * p.timestep,
                't_relax': p.hold_steps * p.timestep,
                't_stretch': p.target_strain / rate,
                't_hold_elastic': p.hold_steps * p.timestep,
            },
            'inputs': {},
            'outputs': {'strain_rate': ['strain_rate'], 'make_on': ['make_on'],
                        'break_on': ['break_on'], 'phase': ['phase']},
        },
        'network': {
            '_type': 'process', 'address': 'local:CollagenNetworkProcess',
            'config': {'params': p.to_dict()},
            'inputs': {'strain_rate': ['strain_rate'], 'make_on': ['make_on'],
                       'break_on': ['break_on']},
            'outputs': {
                'sigma_inplane': ['sigma_inplane'], 'strain': ['strain'],
                'n_crosslinks': ['n_crosslinks'], 'pxx': ['pxx'], 'pyy': ['pyy'],
                'volume': ['volume'], 'positions': ['positions'],
                'atom_types': ['atom_types'], 'box_dimensions': ['box_dimensions'],
            },
        },
        'emitter': {
            '_type': 'step', 'address': 'local:ram-emitter',
            'config': {'emit': {
                'sigma_inplane': 'float', 'strain': 'float', 'n_crosslinks': 'integer',
                'phase': 'string'}},
            'inputs': {'sigma_inplane': ['sigma_inplane'], 'strain': ['strain'],
                       'n_crosslinks': ['n_crosslinks'], 'phase': ['phase']},
            'outputs': {},
        },
    }
