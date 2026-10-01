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


# --- coherent clips (movie.capture_clip) ---------------------------------- #
import numpy as np
from viva_lumenoid.movie import capture_clip


def _median_step(frames):
    steps = []
    for a, b in zip(frames[:-1], frames[1:]):
        pa = np.asarray(a["positions"])[:, :2]
        pb = np.asarray(b["positions"])[:, :2]
        steps.append(np.median(np.hypot(*(pb - pa).T)))
    return float(np.mean(steps))


def test_clip_motion_is_coherent_static_box():
    # a finely-sampled relax clip moves a small fraction of a bead spacing/frame
    c = capture_clip(CollagenParams(n_rods=80, box_xy=16.0, assemble_steps=3000),
                     mode="relax", n_frames=25, dt=0.15)
    assert len(c.frames) == 25
    step = _median_step(c.frames)
    assert step < 0.5, f"motion not coherent: {step:.2f} a/frame"   # vs ~6 a raw
    # unwrap+detrend keeps the network bounded (no index-scramble blow-up)
    xext = max(np.ptp(np.asarray(f["positions"])[:, 0]) for f in c.frames)
    assert xext < 3 * 16.0


def test_clip_stretch_grows_the_box():
    c = capture_clip(CollagenParams(n_rods=80, box_xy=16.0, assemble_steps=3000),
                     mode="stretch", n_frames=25, dt=0.06, strain_rate=0.02)
    assert c.frames[-1]["box"][0] > c.frames[0]["box"][0]     # substrate grew
    # extent tracks the box (no unwrap blow-up under the deforming box)
    assert np.ptp(np.asarray(c.frames[-1]["positions"])[:, 0]) < 1.6 * c.frames[-1]["box"][0]


def test_clip_feeds_the_movie_figure():
    c = capture_clip(CollagenParams(n_rods=60, box_xy=16.0, assemble_steps=2500),
                     mode="assemble", n_frames=20, dt=0.15)
    fig = viz.network_movie_figure(c, "assembly")
    assert len(fig.frames) == 20
    assert "Simulation movie" in viz.figure_to_html(fig)


def test_clip_rejects_unknown_mode():
    with pytest.raises(ValueError):
        capture_clip(mode="nope")
