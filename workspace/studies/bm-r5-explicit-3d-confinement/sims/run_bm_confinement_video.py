#!/usr/bin/env python
"""R5 followup req-r5-simularium-confinement-video: a Simularium trajectory of
cells growing TALLER under a stiff/slow BM vs spreading on a soft/absent BM.

Reviewer 2026-11-09: "Could we visualize a video of cells ... growing taller under
confinement or spreading over softer bm?" This drives the real BM-coupled growth
(viva_lumenoid.bm_coupling.run_bm_coupled_sheet, capture_frames) and renders each
cell as a sphere whose footprint radius tracks sqrt(basal area) and whose height
(z) tracks h=V0/A. Two populations side by side: FREE (no BM, spreads, stays flat)
and CONFINED (stiff/slow BM, compacts and rises). Play it in the workbench
Simularium viewer.

Usage: python workspace/studies/bm-r5-explicit-3d-confinement/sims/run_bm_confinement_video.py
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
OUT = STUDY_DIR / "bm-r5-confinement-video"   # .simularium appended by the writer

from viva_lumenoid.bm_coupling import run_bm_coupled_sheet  # noqa: E402

GROWTH = dict(rate=0.03, vol_rate=0.05, steps=48, dt=0.25, capture_frames=True)


def build():
    free = run_bm_coupled_sheet(mode="no-bm", **GROWTH)
    conf = run_bm_coupled_sheet(mode="relaxing", bm_stiffness=60.0, bm_relaxation=16.0, **GROWTH)
    return free, conf


def write_video(free, conf, out_stem):
    from simulariumio import (AgentData, CameraData, DisplayData, MetaData,
                              TrajectoryData, UnitData, TrajectoryConverter)
    from simulariumio.constants import DISPLAY_TYPE, VIZ_TYPE

    ff, cf = free.frames, conf.frames
    T = min(len(ff), len(cf))
    nf = len(ff[0]["x"])          # cells per population
    n = 2 * nf                    # free + confined
    SCALE = 9.0                   # world scale
    ZSCALE = 60.0                 # exaggerate height so thickening reads
    # Separate the two populations in x.
    span = (ff[0]["x"].max() - ff[0]["x"].min()) or 1.0
    off = span * 1.4

    times = np.array([ff[t]["t"] for t in range(T)], dtype=float)
    positions = np.zeros((T, n, 3))
    radii = np.zeros((T, n))
    for t in range(T):
        for grp, (fr, dx) in enumerate(((ff[t], -off), (cf[t], +off))):
            base = grp * nf
            x = (fr["x"] + dx) * SCALE
            y = fr["y"] * SCALE
            z = fr["height"] * ZSCALE
            r = np.sqrt(np.clip(fr["area"], 1e-6, None) / np.pi) * SCALE
            positions[t, base:base + nf, 0] = x
            positions[t, base:base + nf, 1] = y
            positions[t, base:base + nf, 2] = z
            radii[t, base:base + nf] = r
    center = positions.reshape(-1, 3).mean(axis=0)
    positions = positions - center

    types = [["free cell"] * nf + ["confined cell"] * nf for _ in range(T)]
    display = {
        "free cell": DisplayData(name="free cell (no BM)", display_type=DISPLAY_TYPE.SPHERE, color="#9ca3af"),
        "confined cell": DisplayData(name="confined cell (stiff/slow BM)", display_type=DISPLAY_TYPE.SPHERE, color="#1d4ed8"),
    }
    agent = AgentData(
        times=times, n_agents=np.full(T, n, dtype=float),
        viz_types=np.full((T, n), VIZ_TYPE.DEFAULT),
        unique_ids=np.tile(np.arange(n, dtype=float), (T, 1)),
        types=types, positions=positions, radii=radii, display_data=display)
    extent = float(np.abs(positions).max()) or 1.0
    camera = CameraData(position=np.array([0.0, -1.3 * extent, 1.1 * extent]),
                        look_at_position=np.zeros(3), up_vector=np.array([0.0, 0.0, 1.0]),
                        fov_degrees=55.0)
    traj = TrajectoryData(
        meta_data=MetaData(box_size=np.array([2.6 * extent] * 3), camera_defaults=camera,
                           trajectory_title="R5 — cells thicken under a stiff BM vs spread free"),
        agent_data=agent, time_units=UnitData("s"), spatial_units=UnitData("um"))
    TrajectoryConverter(traj).save(str(out_stem))
    return str(out_stem) + ".simularium"


def main():
    free, conf = build()
    path = write_video(free, conf, OUT)
    print("free  height gain: %+.1f%%  (spreads, stays flat)" % (100 * free.height_gain))
    print("confined height gain: %+.1f%%  gap %.2f  (compacts, rises)" % (100 * conf.height_gain, conf.gap_final))
    print("wrote", path)


if __name__ == "__main__":
    main()
