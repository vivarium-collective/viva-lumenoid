"""Stage-2 sweep and the diagnostic figures build from real runs."""
from __future__ import annotations

import math

import pytest

pytest.importorskip("lammps", reason="LAMMPS Python bindings required")
pytest.importorskip("plotly", reason="plotly required for figures")

from viva_lumenoid import CollagenParams, run_stage1
from viva_lumenoid.stage2 import run_stage2
from viva_lumenoid.viz import stage1_figure, stage2_figure, figure_to_html


def _tiny():
    return CollagenParams(n_rods=40, box_xy=10.0, assemble_steps=1000,
                          hold_steps=1000, make_prob=0.15)


def test_stage2_sweep_runs():
    r = run_stage2(_tiny(), rates=[1e-3, 4e-3, 1.6e-2], sample_dt=20.0)
    assert len(r.points) == 3
    for p in r.points:
        assert p.strain_rate > 0
        assert math.isfinite(p.sigma_steady)
        assert math.isfinite(p.eta_effective)
    assert math.isfinite(r.viscosity_lowrate)


def test_stage1_figure_builds():
    r = run_stage1(_tiny(), sample_dt=20.0)
    fig = stage1_figure(r)
    assert len(fig.data) >= 4          # multiple diagnostic panels
    html = figure_to_html(fig)
    assert "Plotly" in html and "<html" in html.lower()


def test_stage2_figure_builds():
    r = run_stage2(_tiny(), rates=[1e-3, 4e-3, 1.6e-2], sample_dt=20.0)
    fig = stage2_figure(r)
    assert len(fig.data) >= 3
    assert "Plotly" in figure_to_html(fig)
