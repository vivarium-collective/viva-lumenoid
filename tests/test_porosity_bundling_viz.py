"""Porosity + bundling geometric readouts and the stage-5 / evidence-map figures.

Unit-level and fast: the geometry readouts run on synthetic networks (no LAMMPS),
and the figures render from a hand-built result object, so the whole file is
sub-second and deterministic.
"""
from __future__ import annotations

import numpy as np

from viva_lumenoid import geometry as geo
from viva_lumenoid import comparators as C
from viva_lumenoid.stage5 import PorosityBundlingResult, DensityPoint
from viva_lumenoid import viz


def _parallel_bundle(n_rods, x0, y0, box, dx=0.3):
    """n near-parallel horizontal rods stacked at lateral spacing dx (one bundle)."""
    pos = []
    for k in range(n_rods):
        y = y0 + k * dx
        pos.append([x0, y, 0.0])
        pos.append([x0 + 1.0, y, 0.0])  # rod length ~1 a, horizontal
    return np.array(pos)


def test_bundling_detects_a_lateral_bundle():
    box = [20.0, 20.0, 2.0]
    # one tight 6-rod bundle + scattered singletons
    bundle = _parallel_bundle(6, 2.0, 2.0, box, dx=0.3)
    singles = np.array([[10.0, 10.0, 0.0], [11.0, 10.0, 0.0],
                        [5.0, 15.0, 0.0], [6.0, 15.0, 0.0]])
    pos = np.vstack([bundle, singles])
    b = geo.bundling(pos, box, bead_spacing_nm=114.0)
    assert b.max_size >= 6          # the 6-rod bundle is found as one strand
    assert b.frac_bundled > 0.0
    assert b.lateral_cutoff_a == 59.5 / 114.0


def test_porosity_reports_nm_pore_sizes():
    box = [20.0, 20.0, 2.0]
    rng = np.random.default_rng(0)
    # a sparse scatter of short rods leaves large pores
    n = 60
    centres = rng.uniform(1, 19, size=(n, 2))
    pos = []
    for cx, cy in centres:
        pos.append([cx, cy, 0.0]); pos.append([cx + 0.8, cy, 0.0])
    p = geo.porosity(np.array(pos), box, bead_spacing_nm=114.0, grid_px=120)
    assert p.pore_diam_nm.size > 0
    assert p.median_nm > 0
    assert 0.0 <= p.occupied_fraction <= 1.0


def test_comparators_are_well_formed():
    for c in C.all_comparators():
        assert c.lo <= c.value <= c.hi, c.key
        assert c.units and c.source
    assert C.MAXWELL_Z_2D == 4.0 and C.Z_NC1_7S_MAX < C.MAXWELL_Z_2D


def _toy_result():
    diam = np.array([40, 55, 60, 72, 90, 110, 130.0])
    pts = [DensityPoint(n, cov, n / 400, med, med + 5, ms, fb, mx)
           for n, cov, med, ms, fb, mx in [
               (150, 0.018, 130, 1.03, 0.1, 2),
               (500, 0.056, 76, 1.17, 0.18, 7)]]
    snap = {"positions": np.array([[2.0, 2.0, 0], [3.0, 2.0, 0],
                                   [5.0, 5.0, 0], [6.0, 5.0, 0]]),
            "box": [20.0, 20.0, 2.0], "atom_types": np.array([1, 2, 1, 2]),
            "n_rods": 2}
    return PorosityBundlingResult(
        pore_diam_nm=diam, strand_sizes=np.array([7, 2, 1, 1]),
        bundling_hist={1: 40, 2: 5, 7: 1}, canonical_n_rods=500,
        median_pore_nm=76.0, mean_pore_nm=85.0, p90_pore_nm=130.0,
        coverage=0.056, mean_strand=1.17, max_strand=7, frac_bundled=0.18,
        frac_in_5_7=0.02, points=pts, snapshot=snap, summary="toy")


def test_porosity_bundling_figure_renders():
    fig = viz.porosity_bundling_figure(_toy_result())
    assert len(fig.data) >= 4
    html = viz.figure_to_html(fig)
    assert "Stage 5" in html and len(html) > 1000


def test_evidence_map_overlays_model_values():
    fig = viz.evidence_map_figure(measured={
        "modulus_Pa": 0.03, "pore_median_nm": 76.0, "mean_strand": 1.17})
    # 4 comparator forests + at least one red model overlay
    assert len(fig.data) >= 5
    html = viz.figure_to_html(fig)
    assert "Evidence map" in html
