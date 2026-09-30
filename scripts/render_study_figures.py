#!/usr/bin/env python
"""Regenerate the committed diagnostic figures for each study.

Runs the stage-1 and stage-2 drivers (seeded, so reproducible) and writes the
self-contained interactive Plotly HTML into each study's ``viz/`` dir, where the
study declares it as ``address: html:<file>``. The vivarium-workbench publish
step inlines these into the read-only dashboard bundle.

    python scripts/render_study_figures.py
"""
from __future__ import annotations

import os

from viva_lumenoid import run_stage1, CollagenParams
from viva_lumenoid.stage2 import run_stage2
from viva_lumenoid.viz import stage1_figure, stage2_figure, save_html

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIES = os.path.join(ROOT, "workspace", "studies")


def main():
    # Stage 1
    s1 = os.path.join(STUDIES, "bm-v1-stage1-modulus-remodelling", "viz")
    os.makedirs(s1, exist_ok=True)
    print("Running stage 1 …")
    r1 = run_stage1(CollagenParams(n_rods=200, box_xy=20.0,
                                   assemble_steps=6000, hold_steps=12000),
                    sample_dt=8.0)
    save_html(stage1_figure(r1), os.path.join(s1, "stage1_diagnostic.html"),
              "Stage 1 — modulus & remodelling")
    print(f"  E ≈ {r1.elastic_modulus_lj:.3f} kT/a³, "
          f"τ {'noise-limited' if r1.relaxation_noise_limited else f'{r1.relaxation_time_lj:.0f}'}")

    # Stage 2
    s2 = os.path.join(STUDIES, "bm-v2-stress-vs-strainrate", "viz")
    os.makedirs(s2, exist_ok=True)
    print("Running stage 2 sweep …")
    r2 = run_stage2(CollagenParams(n_rods=150, box_xy=18.0, assemble_steps=4000),
                    rates=[5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2], sample_dt=10.0)
    save_html(stage2_figure(r2), os.path.join(s2, "stage2_diagnostic.html"),
              "Stage 2 — σ(ε̇) growing substrate")
    print(f"  low-rate η ≈ {r2.viscosity_lowrate:.0f}")
    print("Figures written.")


if __name__ == "__main__":
    main()
