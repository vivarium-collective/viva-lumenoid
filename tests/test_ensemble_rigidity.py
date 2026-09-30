"""Seed-ensemble statistics and the rigidity (mean-coordination) finding."""
from __future__ import annotations

import math

import pytest

pytest.importorskip("lammps", reason="LAMMPS Python bindings required")

from viva_lumenoid import CollagenParams, stage1_ensemble, mean_coordination
from viva_lumenoid.ensemble import RIGIDITY_THRESHOLD_2D
from viva_lumenoid.rigidity import run_connectivity_sweep


def _tiny():
    return CollagenParams(n_rods=40, box_xy=10.0, assemble_steps=1000,
                          hold_steps=1000, make_prob=0.15)


def test_mean_coordination_and_floppy_bound():
    # NC1×1 + 7S×3 caps crosslinks at 2·n_rods → z_max = 3 < 4 (always floppy).
    z_max = mean_coordination(2 * 200, 200)
    assert math.isclose(z_max, 3.0)
    assert z_max < RIGIDITY_THRESHOLD_2D


def test_stage1_ensemble_reports_ci():
    r = stage1_ensemble(_tiny(), seeds=[1, 2, 3])
    d = r.summary()
    assert d["n_seeds"] == 3
    assert math.isfinite(d["modulus_mean"])
    assert d["coordination_z"] >= 1.0
    assert d["is_floppy"] is True            # tiny under-connected network
    assert len(r.per_seed_modulus) == 3


def test_connectivity_sweep_stays_below_threshold():
    r = run_connectivity_sweep(rungs=[(1, 3, 0.15), (3, 5, 0.4)],
                               seeds=[1, 2], base=_tiny())
    assert len(r.points) == 2
    # z rises with connectivity but this topology can't reach rigidity
    assert r.points[1].z_mean >= r.points[0].z_mean
    assert r.reached_rigid is False
