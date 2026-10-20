"""Locking contract for bm-r3-network-expansion-supply.

R3 reproduces only the remodeling-vs-frozen half of Barrientos Fig 2B-C: the
collagen IV network forms crosslinks when the chemistry is ON and none when it
is FROZEN. (The third Fig-2B-C supply mode -- remodeling WITH material
insertion -- is a composite gap: viva_lumenoid.composites.stage2_sweep has no
insertion process, so it is intentionally NOT tested here.)

Runs are tiny (40 rods, 4 growth steps) and use the faithful released angle set
(use_real_angles) so the small network percolates enough to form crosslinks
when the chemistry is on.
"""
from __future__ import annotations

import pytest

pytest.importorskip("lammps", reason="LAMMPS Python bindings required")

from bigraph_schema import allocate_core

from viva_lumenoid.collagen import CollagenNetworkProcess
from viva_lumenoid.params import CollagenParams


def _run(make_on: float, break_on: float) -> dict:
    """A bounded growth-loading run of the collagen network in one supply mode."""
    params = CollagenParams(
        n_rods=40,
        box_xy=8.0,
        slab_thickness=4.0,
        assemble_steps=300,
        hold_steps=200,
        use_real_angles=True,
        bending_k=4.0,
    )
    bm = CollagenNetworkProcess(config={"params": params.to_dict()}, core=allocate_core())
    bm.initial_state()
    out: dict = {}
    for _ in range(4):
        out = bm.update(
            {"strain_rate": 0.001, "make_on": make_on, "break_on": break_on}, 1.0
        )
    bm.close()
    return out


def test_remodeling_forms_crosslinks():
    """Chemistry ON (make_on = break_on = 1) -> the network forms crosslinks."""
    out = _run(make_on=1.0, break_on=1.0)
    assert out["n_crosslinks"] > 0, (
        f"remodeling ON should form crosslinks, got {out['n_crosslinks']}"
    )


def test_frozen_has_no_remodeling():
    """Chemistry OFF (make_on = break_on = 0) -> no crosslinks form (topology frozen)."""
    out = _run(make_on=0.0, break_on=0.0)
    assert out["n_crosslinks"] == 0, (
        f"frozen network should form no crosslinks, got {out['n_crosslinks']}"
    )


def test_remodeling_distinguishes_from_frozen():
    """The two supply modes the composite CAN express are distinguishable:
    crosslinks appear only when the chemistry is on."""
    on = _run(make_on=1.0, break_on=1.0)["n_crosslinks"]
    frozen = _run(make_on=0.0, break_on=0.0)["n_crosslinks"]
    assert on > frozen == 0, f"expected ON > FROZEN == 0, got ON={on}, FROZEN={frozen}"
