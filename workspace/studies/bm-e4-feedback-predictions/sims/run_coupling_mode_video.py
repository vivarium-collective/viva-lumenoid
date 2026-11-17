#!/usr/bin/env python
"""E4 followup req-e4-dynamic-viz: a dynamic view of the growth-driven two-way
coupling — the realized basal geometry emerging and lagging under BM reaction,
contrasted with the uncoupled (inert BM) control.

Reviewer 2026-11-09: "needs more dynamic visualizations." E4's primary path is the
REDUCED BM two-way feedback (the explicit-LAMMPS path stays a separate verification
gate). This renders the real BM-coupled growth in two coupling modes — two-way (BM
reacts + remodels) vs uncoupled (inert BM) — as cell populations whose basal
footprint and height evolve, so the coupling-mode difference is visible.

Usage: python workspace/studies/bm-e4-feedback-predictions/sims/run_coupling_mode_video.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

HERE = Path(__file__).resolve()
WS_ROOT = HERE.parents[4]
sys.path.insert(0, str(WS_ROOT))
STUDY_DIR = HERE.parents[1]
OUT = STUDY_DIR / "bm-e4-coupling-mode-video"

from viva_lumenoid.bm_coupling import run_bm_coupled_sheet  # noqa: E402

GROWTH = dict(rate=0.03, vol_rate=0.05, steps=48, dt=0.25, capture_frames=True)
MODES = [
    ("two-way (BM reacts)", "#059669", dict(mode="relaxing", bm_stiffness=50.0, bm_relaxation=14.0)),
    ("uncoupled (inert BM)", "#9ca3af", dict(mode="no-bm")),
]


def main():
    runs = [(name, color, run_bm_coupled_sheet(**GROWTH, **kw)) for name, color, kw in MODES]
    for name, _, r in runs:
        print(f"{name:24s} basal-area fold {r.a_mean[-1]/r.a_mean[0]:.3f}  "
              f"gap {r.gap_final:+.2f}  height {100*r.height_gain:+.0f}%  BMtens {r.bm_tension_mean[-1]:.2f}")

    from simulariumio import (AgentData, CameraData, DisplayData, MetaData,
                              TrajectoryData, UnitData, TrajectoryConverter)
    from simulariumio.constants import DISPLAY_TYPE, VIZ_TYPE

    frames = [r.frames for _, _, r in runs]
    T = min(len(f) for f in frames); nf = len(frames[0][0]["x"]); ng = len(runs); n = ng * nf
    SCALE, ZSCALE = 9.0, 60.0
    span = (frames[0][0]["x"].max() - frames[0][0]["x"].min()) or 1.0
    offs = [(i - (ng - 1) / 2) * span * 1.5 for i in range(ng)]
    times = np.array([frames[0][t]["t"] for t in range(T)], dtype=float)
    positions = np.zeros((T, n, 3)); radii = np.zeros((T, n))
    for t in range(T):
        for g, fr_all in enumerate(frames):
            fr = fr_all[t]; base = g * nf
            positions[t, base:base+nf, 0] = (fr["x"] + offs[g]) * SCALE
            positions[t, base:base+nf, 1] = fr["y"] * SCALE
            positions[t, base:base+nf, 2] = fr["height"] * ZSCALE
            radii[t, base:base+nf] = np.sqrt(np.clip(fr["area"], 1e-6, None) / np.pi) * SCALE
    positions -= positions.reshape(-1, 3).mean(axis=0)
    types = [sum(([MODES[g][0]] * nf for g in range(ng)), []) for _ in range(T)]
    display = {name: DisplayData(name=name, display_type=DISPLAY_TYPE.SPHERE, color=color)
               for name, color, _ in MODES}
    agent = AgentData(times=times, n_agents=np.full(T, n, dtype=float),
                      viz_types=np.full((T, n), VIZ_TYPE.DEFAULT),
                      unique_ids=np.tile(np.arange(n, dtype=float), (T, 1)),
                      types=types, positions=positions, radii=radii, display_data=display)
    extent = float(np.abs(positions).max()) or 1.0
    camera = CameraData(position=np.array([0.0, -1.3*extent, 1.0*extent]),
                        look_at_position=np.zeros(3), up_vector=np.array([0.0, 0.0, 1.0]), fov_degrees=55.0)
    traj = TrajectoryData(meta_data=MetaData(box_size=np.array([2.6*extent]*3), camera_defaults=camera,
                          trajectory_title="E4 - two-way BM coupling vs uncoupled (reduced path)"),
                          agent_data=agent, time_units=UnitData("s"), spatial_units=UnitData("um"))
    TrajectoryConverter(traj).save(str(OUT))
    print("wrote", str(OUT) + ".simularium")


if __name__ == "__main__":
    main()
