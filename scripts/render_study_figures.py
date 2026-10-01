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
from viva_lumenoid.rigidity import run_connectivity_sweep
from viva_lumenoid.bending import run_bending_sweep
from viva_lumenoid.stage5 import run_porosity_bundling
from viva_lumenoid.viz import (stage1_figure, stage2_figure, rigidity_figure,
                               bending_figure, porosity_bundling_figure,
                               evidence_map_figure, network_movie_figure,
                               network_movie_3d_figure, save_html)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIES = os.path.join(ROOT, "workspace", "studies")


def render_movie(study, fname, title, params, mode, fname_3d=None, **clip_kw):
    """Capture a finely-sampled, coherent clip once and write the 2D movie, plus
    an optional rotatable 3D view of the same clip.

    Uses movie.capture_clip (small dt + periodic-unwrap + COM-drift removal) so
    the motion is continuous, not the teleporting of coarse analysis snapshots.
    """
    from viva_lumenoid.movie import capture_clip
    d = os.path.join(STUDIES, study, "viz")
    os.makedirs(d, exist_ok=True)
    c = capture_clip(params, mode=mode, **clip_kw)
    save_html(network_movie_figure(c, title), os.path.join(d, fname), title)
    msg = f"  movie {study}/{fname}: {c.summary}"
    if fname_3d:
        save_html(network_movie_3d_figure(c, title), os.path.join(d, fname_3d), title)
        msg += f"  + 3D {fname_3d}"
    print(msg)
    return c


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

    # Stage 3 (follow-up): rigidity-percolation sweep
    s3 = os.path.join(STUDIES, "bm-v3-junction-bending-rigidity", "viz")
    os.makedirs(s3, exist_ok=True)
    print("Running rigidity sweep …")
    r3 = run_connectivity_sweep()
    save_html(rigidity_figure(r3), os.path.join(s3, "rigidity_diagnostic.html"),
              "Rigidity sweep — modulus vs z")
    print(f"  z 1.24→{max(p.z_mean for p in r3.points):.2f}; reached_rigid={r3.reached_rigid}")

    # Stage 4 (FP1): junction-bending sweep (athermal modulus)
    s4 = os.path.join(STUDIES, "bm-v4-junction-bending", "viz")
    os.makedirs(s4, exist_ok=True)
    print("Running junction-bending sweep (athermal; dt=0.002) …")
    r4 = run_bending_sweep(seeds=[11, 22, 33, 44, 55])
    save_html(bending_figure(r4), os.path.join(s4, "bending_diagnostic.html"),
              "FP1 — junction bending")
    print(f"  floppy median {r4.points[0].modulus_median:+.3f}; "
          f"large_response={r4.large_response}")

    # Stage 5 (bm-v5): porosity & bundling geometric readouts + evidence map
    s5 = os.path.join(STUDIES, "bm-v5-porosity-bundling", "viz")
    os.makedirs(s5, exist_ok=True)
    print("Running porosity & bundling density sweep …")
    r5 = run_porosity_bundling(densities=(150, 250, 350, 500), assemble_steps=8000)
    save_html(porosity_bundling_figure(r5),
              os.path.join(s5, "porosity_bundling.html"),
              "Stage 5 — porosity & bundling")
    # Evidence map overlays the model's own measured values (modulus from stage 1,
    # pore median + mean strand from stage 5) on the measured literature bands.
    measured = {
        "modulus_Pa": float(getattr(r1, "elastic_modulus_Pa", r1.elastic_modulus_lj * 2.5)),
        "pore_median_nm": r5.median_pore_nm,
        "mean_strand": r5.mean_strand,
    }
    save_html(evidence_map_figure(measured),
              os.path.join(s5, "evidence_map.html"),
              "Evidence map — readouts vs comparators")
    print(f"  pore median {r5.median_pore_nm:.0f} nm · mean strand {r5.mean_strand:.2f} "
          f"(max {r5.max_strand})")

    # Simulation movies — the actual spatial state unfolding, COHERENTLY: each is
    # a finely-sampled clip (small dt, periodic-unwrapped, COM-drift removed) of
    # the mode that matters for that study, so the motion is continuous.
    print("Rendering simulation movies …")
    render_movie("bm-v1-stage1-modulus-remodelling", "stage1_movie.html",
                 "stage 1 — remodelling: crosslinks break, the network relaxes",
                 CollagenParams(n_rods=140, box_xy=18.0, assemble_steps=7000),
                 mode="remodel", n_frames=80, dt=0.15, fname_3d="stage1_movie_3d.html")
    render_movie("bm-v2-stress-vs-strainrate", "stage2_movie.html",
                 "stage 2 — the growing substrate stretches (equibiaxial)",
                 CollagenParams(n_rods=140, box_xy=18.0, assemble_steps=6000),
                 mode="stretch", n_frames=80, dt=0.06, strain_rate=0.02,
                 fname_3d="stage2_movie_3d.html")
    render_movie("bm-v3-junction-bending-rigidity", "rigidity_movie.html",
                 "rigidity — high-connectivity network jiggling (NC1×5 + 7S×8)",
                 CollagenParams(n_rods=180, box_xy=20.0, assemble_steps=8000,
                                nc1_max_crosslinks=5, svns_max_crosslinks=8, make_prob=0.6),
                 mode="relax", n_frames=80, dt=0.15, fname_3d="rigidity_movie_3d.html")
    render_movie("bm-v4-junction-bending", "bending_movie.html",
                 "junction bending — rods pinned at crosslink angles",
                 CollagenParams(n_rods=140, box_xy=18.0, assemble_steps=6000,
                                bending_k=50.0),
                 mode="relax", n_frames=80, dt=0.15, fname_3d="bending_movie_3d.html")
    # bm-v5 stays 2D-only: its study-charts payload (assembly movie + porosity +
    # evidence map) is already near the 16 MB per-file publish limit.
    render_movie("bm-v5-porosity-bundling", "assembly_movie.html",
                 "assembly — a porous collagen IV mesh forms",
                 CollagenParams(n_rods=300, box_xy=20.0),
                 mode="assemble", n_frames=75, dt=0.15)
    print("Figures written.")


if __name__ == "__main__":
    main()
