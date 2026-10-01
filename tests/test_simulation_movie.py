"""Frame capture + the animated simulation-movie figure."""
from __future__ import annotations

import pytest

from viva_lumenoid import CollagenParams, run_stage1
from viva_lumenoid import viz


def test_capture_frames_collects_spatial_trajectory():
    r = run_stage1(CollagenParams(n_rods=60, assemble_steps=1500, hold_steps=1500),
                   sample_dt=20.0, capture_frames=True, frame_stride=1)
    assert r.frames and len(r.frames) >= 4
    f = r.frames[0]
    for key in ("t", "strain", "sigma", "phase", "positions", "box", "atom_types"):
        assert key in f
    # positions are 2 beads per rod
    assert len(f["positions"]) == 2 * 60
    # the protocol's phases are represented across the trajectory
    phases = {fr["phase"] for fr in r.frames}
    assert "assemble" in phases


def test_default_run_captures_no_frames():
    r = run_stage1(CollagenParams(n_rods=40, assemble_steps=800, hold_steps=800),
                   sample_dt=40.0)
    assert r.frames is None            # opt-in only; no movie overhead by default


def test_movie_figure_has_animation_frames():
    r = run_stage1(CollagenParams(n_rods=50, assemble_steps=1200, hold_steps=1200),
                   sample_dt=20.0, capture_frames=True, frame_stride=1)
    fig = viz.network_movie_figure(r, "test")
    assert len(fig.frames) == len(r.frames)
    assert len(fig.data) >= 6          # rods, NC1, 7S, box, stress line, cursor
    html = viz.figure_to_html(fig)
    assert "Simulation movie" in html and "play" in html


def test_movie_requires_frames():
    r = run_stage1(CollagenParams(n_rods=40, assemble_steps=800, hold_steps=800),
                   sample_dt=40.0)
    with pytest.raises(ValueError):
        viz.network_movie_figure(r)
