"""Locking contract for bm-r3-network-expansion-supply.

R3 now reproduces the FULL three-mode Barrientos Fig 2B-C supply ordering, thanks
to the material-insertion mechanism added to CollagenNetworkProcess (the
``insert_on`` control port + the ``insert_per_step`` parameter; ``_insert_rods``
creates fresh two-bead NC1/7S rods mid-run). The three supply modes:

  * no-remodeling             : chemistry off, no insertion -> frozen, dilutes most
  * remodeling-without-insert : make+break on, no insertion -> fixed material
  * remodeling-with-insertion : make+break on + insertion   -> material supplied

Under growth (expansion), connectivity orders with-insertion > without-insertion
> no-remodeling. These tests lock (1) the insertion mechanism (exact atom-count
growth; inserted rods crosslink in), and (2) the three-mode ordering.

Runs are bounded (40 rods, 4 assemble + 8 growth steps) and use the faithful
released angle set (use_real_angles) so the small network percolates. Skips
cleanly without the LAMMPS Python bindings.
"""
from __future__ import annotations

import pytest

pytest.importorskip("lammps", reason="LAMMPS Python bindings required")

from bigraph_schema import allocate_core

from viva_lumenoid.collagen import CollagenNetworkProcess
from viva_lumenoid.params import CollagenParams


def _proc(insert_per_step: int = 3, n_rods: int = 40, seed: int = 12345):
    params = CollagenParams(
        n_rods=n_rods, box_xy=8.0, slab_thickness=4.0, seed=seed,
        assemble_steps=300, hold_steps=200,
        use_real_angles=True, bending_k=4.0, insert_per_step=insert_per_step,
    )
    bm = CollagenNetworkProcess(config={"params": params.to_dict()}, core=allocate_core())
    bm.initial_state()
    return bm


def _grow(mode: str, assemble: int = 4, grow: int = 8, strain_rate: float = 0.003):
    """Run one supply mode: shared assembly, then growth. Returns (rods, crosslinks, density)."""
    make, brk, ins = {
        "frozen": (0.0, 0.0, 0.0),
        "no_insert": (1.0, 1.0, 0.0),
        "with_insert": (1.0, 1.0, 1.0),
    }[mode]
    bm = _proc()
    for _ in range(assemble):
        bm.update({"strain_rate": 0.0, "make_on": 1.0, "break_on": 1.0, "insert_on": 0.0}, 1.0)
    out: dict = {}
    for _ in range(grow):
        out = bm.update({"strain_rate": strain_rate, "make_on": make, "break_on": brk,
                         "insert_on": ins}, 1.0)
    rods = bm._n_rods_current
    bm.close()
    area = 8.0 * 8.0 * (1.0 + out["strain"]) ** 2
    return rods, out["n_crosslinks"], out["n_crosslinks"] / area


@pytest.fixture(scope="module")
def modes():
    return {m: _grow(m) for m in ("frozen", "no_insert", "with_insert")}


def test_insertion_grows_atoms_deterministically():
    """insert_per_step=k grows the atom count by EXACTLY 2k per insert step; insert_on=0 is flat."""
    bm = _proc(insert_per_step=3)
    nat0 = bm._lmp.get_natoms()
    for i in range(5):
        out = bm.update({"strain_rate": 0.0, "make_on": 1.0, "break_on": 1.0, "insert_on": 1.0}, 1.0)
        assert out["num_atoms"] == nat0 + 2 * 3 * (i + 1), (
            f"step {i+1}: expected {nat0 + 6*(i+1)} atoms, got {out['num_atoms']}")
    bm.close()
    # control: no insertion -> atom count flat
    bm2 = _proc(insert_per_step=3)
    n0 = bm2._lmp.get_natoms()
    for _ in range(5):
        out = bm2.update({"strain_rate": 0.0, "make_on": 1.0, "break_on": 1.0, "insert_on": 0.0}, 1.0)
    assert out["num_atoms"] == n0, f"insert_on=0 should hold atoms flat at {n0}, got {out['num_atoms']}"
    bm2.close()


def test_inserted_rods_crosslink_in(modes):
    """Under insertion the crosslink count grows (inserted rods become crosslink-eligible)."""
    # with-insertion ends with more crosslinks than the without-insertion mode on fixed material
    assert modes["with_insert"][1] > modes["no_insert"][1] > modes["frozen"][1], (
        f"crosslink ordering with>without>frozen failed: {[modes[m][1] for m in ('with_insert','no_insert','frozen')]}")


def test_rod_count_grows_only_with_insertion(modes):
    """Material (rod) count increases ONLY in the insertion mode; the others keep fixed material."""
    rods = {m: modes[m][0] for m in modes}
    assert rods["with_insert"] > rods["no_insert"] == rods["frozen"], (
        f"expected rods(with_insert) > rods(no_insert) == rods(frozen), got {rods}")


def test_three_supply_modes_ordered(modes):
    """Fig 2B-C ordering: crosslink-per-area density with-insertion > without-insertion > frozen."""
    d = {m: modes[m][2] for m in modes}
    assert d["with_insert"] > d["no_insert"] > d["frozen"], (
        f"expected density with_insert > no_insert > frozen, got {d}")


def test_remodeling_vs_frozen(modes):
    """Retained building block: a remodeling network holds more crosslinks than a frozen one."""
    assert modes["no_insert"][1] > modes["frozen"][1]
