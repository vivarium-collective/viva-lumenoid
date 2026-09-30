"""Workspace Visualization Steps — the aesthetic, diagnostic figures wired for
the vivarium-workbench.

Each subclass accumulates the network process's per-tick state across a composite
run and renders the multi-panel Plotly figure from ``viva_lumenoid.viz``. They
follow the viva-superpowers new-style contract (``accumulate`` + ``render``).

For the direct-driver path (``run_stage1`` / ``run_stage2``) the same figures are
available as pure functions in ``viva_lumenoid.viz`` — these classes are the
composable, dashboard-rendered wrappers.
"""
from __future__ import annotations

from types import SimpleNamespace

from viva_superpowers.visualization import Visualization

from . import viz


class Stage1Diagnostic(Visualization):
    """Stage-1 diagnostic dashboard: stress vs time (by phase), stress-strain +
    modulus fit, crosslink assembly/remodelling, and the final network mesh.

    Wire its inputs to the network process's outputs plus the controller phase.
    """

    config_schema = {
        **Visualization.config_schema,
        'title': {'_type': 'string', '_default': 'Stage 1 — modulus & remodelling'},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tr = {'t': [], 'sigma': [], 'strain': [], 'n_crosslinks': [], 'phase': []}
        self._t = 0.0
        self._last = None

    def inputs(self):
        return {
            'sigma_inplane': 'float', 'strain': 'float', 'n_crosslinks': 'integer',
            'phase': 'string', 'positions': 'list', 'atom_types': 'list',
            'box_dimensions': 'list', 'time': 'float',
        }

    def accumulate(self, state):
        self._t = float(state.get('time', self._t + 1.0))
        self._tr['t'].append(self._t)
        self._tr['sigma'].append(float(state.get('sigma_inplane', 0.0)))
        self._tr['strain'].append(float(state.get('strain', 0.0)))
        self._tr['n_crosslinks'].append(int(state.get('n_crosslinks', 0)))
        self._tr['phase'].append(state.get('phase', ''))
        self._last = state

    def render(self) -> str:
        s = self._last or {}
        # Reconstruct enough of a Stage1Result for the figure function.
        from .stage1 import _relaxation_time, _smooth
        import numpy as np
        ph = np.array(self._tr['phase']); tt = np.array(self._tr['t'])
        sig = np.array(self._tr['sigma']); stn = np.array(self._tr['strain'])
        base = float(np.median(sig[ph == 'relax'])) if (ph == 'relax').any() else (
            float(sig[0]) if len(sig) else 0.0)
        stm = ph == 'stretch'
        if stm.sum() >= 2 and np.ptp(stn[stm]) > 1e-6:
            A = np.vstack([stn[stm], np.ones(stm.sum())]).T
            modulus = float(np.linalg.lstsq(A, sig[stm] - base, rcond=None)[0][0])
        else:
            modulus = 0.0
        vis = ph == 'hold_viscous'
        tau = _relaxation_time(tt[vis], sig[vis]) if vis.sum() >= 4 else float('nan')
        result = SimpleNamespace(
            trace=self._tr,
            elastic_modulus_lj=modulus,
            relaxation_time_lj=tau,
            relaxation_noise_limited=bool(np.isnan(tau)),
            sigma_prestretch_baseline=base,
            n_crosslinks_assembled=(max(self._tr['n_crosslinks']) if self._tr['n_crosslinks'] else 0),
            snapshot=({'positions': s.get('positions'), 'atom_types': s.get('atom_types'),
                       'box': s.get('box_dimensions'), 'n_rods': None}
                      if s.get('positions') else None),
        )
        return viz.figure_to_html(viz.stage1_figure(result), self.config.get('title', ''))


class Stage2Diagnostic(Visualization):
    """Stage-2 diagnostic dashboard: σ(ε̇), effective viscosity η = σ/ε̇, and
    crosslinks vs strain rate. Rendered from a completed sweep (see viva_lumenoid.viz)."""

    config_schema = {
        **Visualization.config_schema,
        'title': {'_type': 'string', '_default': 'Stage 2 — σ(ε̇) growing substrate'},
    }

    def inputs(self):
        # A sweep is not a single timeline; the workbench renders this from the
        # study's stored stage-2 result. Kept minimal for registration.
        return {'sigma_inplane': 'float', 'strain': 'float', 'time': 'float'}

    def accumulate(self, state):
        pass

    def render(self) -> str:
        return ("<div style='font:14px system-ui;padding:24px;color:#6b7280'>"
                "Stage-2 σ(ε̇) sweep — render via "
                "<code>viva_lumenoid.viz.stage2_figure(run_stage2())</code>.</div>")
