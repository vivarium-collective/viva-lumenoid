"""Stage-1 collagen IV network: the model runs on real LAMMPS and the staged
protocol produces the two v1 readouts.

Tiny networks / short phases keep the suite fast (a few seconds). The tests
assert structure and qualitative physics, not calibrated numbers — calibration
is the investigation's open work (spec decisions #4/#8/#10).
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("lammps", reason="LAMMPS Python bindings required")

from viva_lumenoid import CollagenParams, build_network_data, run_stage1
from viva_lumenoid.collagen import CollagenNetworkProcess


def _tiny() -> CollagenParams:
    return CollagenParams(n_rods=40, box_xy=10.0, assemble_steps=1000,
                          hold_steps=1000, make_every=100, make_prob=0.15,
                          strain_rate=4e-3, target_strain=0.12)


def test_network_data_counts():
    p = CollagenParams(n_rods=25)
    data = build_network_data(p)
    assert f"{2 * p.n_rods} atoms" in data       # two beads per rod
    assert f"{p.n_rods} bonds" in data           # one intra-rod bond per rod
    assert "2 atom types" in data and "3 bond types" in data
    assert "Atoms  # molecular" in data and "Bonds" in data


def test_process_assembles_and_responds():
    from process_bigraph import allocate_core
    p = _tiny()
    proc = CollagenNetworkProcess(config={"params": p.to_dict()}, core=allocate_core())
    s0 = proc.initial_state()
    assert s0["num_atoms"] == 2 * p.n_rods
    assert s0["n_crosslinks"] == 0                # bare rods, no crosslinks yet
    # formation on with no strain -> crosslinks form
    for _ in range(10):
        s = proc.update({"strain_rate": 0.0, "make_on": 1.0, "break_on": 1.0}, 10.0)
    assert s["n_crosslinks"] > 0
    # strain the box (frozen chemistry) -> box grows
    lx0 = s["box_dimensions"][0]
    for _ in range(5):
        s = proc.update({"strain_rate": p.strain_rate, "make_on": 0.0, "break_on": 0.0}, 10.0)
    assert s["box_dimensions"][0] > lx0
    assert s["strain"] > 0
    proc.close()


def test_offrate_breaks_crosslinks_no_densification():
    """Regression guard: break_on (force-independent off-rate) must REDUCE the
    crosslink count — the old `fix bond/break` overstretch model let the network
    only densify. Under make+break the count stays bounded (steady state); under
    break-only it decays."""
    from process_bigraph import allocate_core
    p = _tiny()
    proc = CollagenNetworkProcess(config={"params": p.to_dict()}, core=allocate_core())
    proc.initial_state()
    for _ in range(12):  # make + break -> steady state, not runaway
        s = proc.update({"strain_rate": 0.0, "make_on": 1.0, "break_on": 1.0}, 20.0)
    steady = s["n_crosslinks"]
    assert steady < 3 * p.n_rods            # bounded, not saturated/overflowed
    n0 = s["n_crosslinks"]
    for _ in range(8):                       # break only -> decays
        s = proc.update({"strain_rate": 0.0, "make_on": 0.0, "break_on": 1.0}, 20.0)
    assert s["n_crosslinks"] < n0
    proc.close()


def test_run_stage1_smoke():
    r = run_stage1(_tiny(), sample_dt=20.0)
    d = r.summary()
    # all readout fields present
    for k in ("elastic_modulus_lj", "elastic_modulus_Pa", "relaxation_time_lj",
              "final_strain", "n_crosslinks_assembled", "sigma_prestretch_baseline",
              "stress_noise", "relaxation_noise_limited"):
        assert k in d
    # a network assembled and the box reached a finite strain
    assert d["n_crosslinks_assembled"] > 0
    assert d["final_strain"] > 0
    assert math.isfinite(d["elastic_modulus_lj"])
    # Pa conversion is the kT/a^3 scale applied to the LJ modulus
    assert math.isclose(d["elastic_modulus_Pa"],
                        d["elastic_modulus_lj"] * CollagenParams().kT_per_a3_Pa,
                        rel_tol=1e-6)
    # trace covers every phase, including the equilibration + both holds
    phases = set(r.trace["phase"])
    assert {"assemble", "relax", "stretch", "hold_elastic", "hold_viscous"} <= phases


def test_composite_spec_shape():
    from viva_lumenoid.stage1 import stage1_composite_spec
    spec = stage1_composite_spec(_tiny())
    assert set(spec) == {"controller", "network", "emitter"}
    assert spec["network"]["address"] == "local:CollagenNetworkProcess"
    assert spec["controller"]["address"] == "local:StagedStretchController"
    # controller drives the network's control ports
    assert spec["network"]["inputs"]["strain_rate"] == ["strain_rate"]
    assert spec["network"]["inputs"]["make_on"] == ["make_on"]
    assert spec["network"]["inputs"]["break_on"] == ["break_on"]
