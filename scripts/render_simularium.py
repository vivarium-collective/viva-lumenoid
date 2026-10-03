#!/usr/bin/env python
"""Write a .simularium trajectory for each study so it can be browsed in the
workbench's Simularium viewer (Analysis tab → Simularium Viewer).

Reuses the coherent movie clips (finely-sampled, unwrapped, drift-removed) as the
trajectory, converts the NC1/7S beads to Simularium point agents, and writes
``workspace/studies/<study>/simularium/<study>.simularium``. The workbench
Analysis tab auto-discovers any ``.simularium`` under a study dir.

    python scripts/render_simularium.py
"""
from __future__ import annotations

import os

from viva_lumenoid import CollagenParams
from viva_lumenoid.movie import capture_clip
from viva_lumenoid.simularium import write_trajectory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIES = os.path.join(ROOT, "workspace", "studies")

# (study, clip mode, params) — matches the simulation movies, tuned per study.
JOBS = [
    ("bm-v1-stage1-modulus-remodelling", "remodel",
     CollagenParams(n_rods=250, box_xy=13.0, assemble_steps=22000)),
    ("bm-v2-stress-vs-strainrate", "stretch",
     CollagenParams(n_rods=250, box_xy=13.0, assemble_steps=20000)),
    ("bm-v3-junction-bending-rigidity", "relax",
     CollagenParams(n_rods=250, box_xy=13.0, assemble_steps=20000,
                    nc1_max_crosslinks=5, svns_max_crosslinks=8, make_prob=0.27)),
    ("bm-v4-junction-bending", "relax",
     CollagenParams(n_rods=250, box_xy=13.0, assemble_steps=20000, bending_k=4.0)),
    ("bm-v5-porosity-bundling", "assemble",
     CollagenParams(n_rods=400, box_xy=13.0, assemble_steps=20000)),
]


def main():
    for study, mode, params in JOBS:
        d = os.path.join(STUDIES, study, "simularium")
        os.makedirs(d, exist_ok=True)
        kw = {"dt": 0.06, "strain_rate": 0.02} if mode == "stretch" else {"dt": 0.15}
        clip = capture_clip(params, mode=mode, n_frames=80, **kw)
        out = write_trajectory(clip.frames, params,
                               os.path.join(d, study),  # suffix added by writer
                               title=f"{study} — {mode}")
        print(f"  {study}: {os.path.basename(out)} "
              f"({len(clip.frames)} frames, {os.path.getsize(out)//1024} KB)")
    print("Simularium trajectories written.")


if __name__ == "__main__":
    main()
