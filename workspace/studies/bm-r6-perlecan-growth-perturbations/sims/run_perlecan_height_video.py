#!/usr/bin/env python
"""R6 followup req-r6-taller-shorter-viz: a dynamic view of cells growing taller /
shorter across the perlecan conditions.

Reviewer 2026-11-09: "visualization of the cells growing taller[/]shorter would be
helpful." R6 reproduces Fig 7 — perlecan knockdown taller > control > overexpression
shorter — because perlecan promotes collagen-IV bond breakage, so knockdown LENGTHENS
the collagen lifetime (a stiffer, slower-relaxing BM) and overexpression SHORTENS it
(a softer, faster-relaxing BM). This maps each condition onto the R4 BM-coupling
(stiffness, relaxation) and renders the three cell populations growing under confined
volume growth, so the taller>control>shorter ordering is VISIBLE as cell height.

Reduced units; the mapping condition->(stiffness,tau) is the qualitative Fig-7
direction, not a calibrated lifetime->modulus fit.

Usage: python workspace/studies/bm-r6-perlecan-growth-perturbations/sims/run_perlecan_height_video.py
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
OUT = STUDY_DIR / "bm-r6-perlecan-height-video"

from viva_lumenoid.bm_coupling import run_bm_coupled_sheet  # noqa: E402

GROWTH = dict(rate=0.03, vol_rate=0.05, steps=48, dt=0.25, capture_frames=True)
# perlecan condition -> (BM stiffness, relaxation tau). Knockdown: long collagen
# lifetime -> stiff, slow BM. Overexpression: short lifetime -> soft, fast BM.
CONDITIONS = [
    ("perlecan KD (taller)",  "#1d4ed8", dict(mode="relaxing", bm_stiffness=80.0, bm_relaxation=30.0)),
    ("control",               "#6b7280", dict(mode="relaxing", bm_stiffness=30.0, bm_relaxation=12.0)),
    ("perlecan OE (shorter)", "#dc2626", dict(mode="relaxing", bm_stiffness=6.0,  bm_relaxation=4.0)),
]


def main():
    runs = [(name, color, run_bm_coupled_sheet(**GROWTH, **kw)) for name, color, kw in CONDITIONS]
    for name, _, r in runs:
        print(f"{name:24s} height gain {100*r.height_gain:+5.1f}%  gap {r.gap_final:.2f}")
    # assert the ordering is taller > control > shorter
    hg = [r.height_gain for _, _, r in runs]
    print("ordering taller>control>shorter:", hg[0] > hg[1] > hg[2])

    from simulariumio import (AgentData, CameraData, DisplayData, MetaData,
                              TrajectoryData, UnitData, TrajectoryConverter)
    from simulariumio.constants import DISPLAY_TYPE, VIZ_TYPE

    frames = [r.frames for _, _, r in runs]
    T = min(len(f) for f in frames)
    nf = len(frames[0][0]["x"])
    ng = len(runs)
    n = ng * nf
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

    types = [sum(([CONDITIONS[g][0]] * nf for g in range(ng)), []) for _ in range(T)]
    display = {name: DisplayData(name=name, display_type=DISPLAY_TYPE.SPHERE, color=color)
               for name, color, _ in CONDITIONS}
    agent = AgentData(times=times, n_agents=np.full(T, n, dtype=float),
                      viz_types=np.full((T, n), VIZ_TYPE.DEFAULT),
                      unique_ids=np.tile(np.arange(n, dtype=float), (T, 1)),
                      types=types, positions=positions, radii=radii, display_data=display)
    extent = float(np.abs(positions).max()) or 1.0
    camera = CameraData(position=np.array([0.0, -1.3*extent, 1.0*extent]),
                        look_at_position=np.zeros(3), up_vector=np.array([0.0, 0.0, 1.0]), fov_degrees=55.0)
    traj = TrajectoryData(meta_data=MetaData(box_size=np.array([2.6*extent]*3), camera_defaults=camera,
                          trajectory_title="R6 — perlecan KD (tall) > control > OE (short)"),
                          agent_data=agent, time_units=UnitData("s"), spatial_units=UnitData("um"))
    TrajectoryConverter(traj).save(str(OUT))
    print("wrote", str(OUT) + ".simularium")


if __name__ == "__main__":
    main()
