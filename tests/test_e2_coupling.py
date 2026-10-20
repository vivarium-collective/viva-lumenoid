"""E2 — actin-adhesion-collagen force transfer.

The actomyosin cortex (LAMMPS seam) pulls through reversible basal linkers
(BasalAdhesion) on a collagen IV BM patch (CollagenNetworkProcess). Verifies the
occupancy-gated force transfer; the LAMMPS parts skip without the binding.
"""
import pytest


def test_actin_adhesion_bm_registered():
    import viva_lumenoid.composites  # noqa: F401
    from process_bigraph.composite_generator import _REGISTRY
    assert "viva_lumenoid.composites.actin_adhesion_bm" in _REGISTRY


def test_adhesion_occupancy_kinetics():
    # Pure-process: no engine needed. Occupancy rises toward 1 when k_off=0.
    from bigraph_schema import allocate_core
    from viva_lumenoid.adhesion import BasalAdhesion
    adh = BasalAdhesion(config={"k_on": 1.0, "k_off": 0.0}, core=allocate_core())
    o1 = adh.update({"cortex_tension": 1.0}, 0.1)["linker_occupancy"]
    o2 = adh.update({"cortex_tension": 1.0}, 0.1)["linker_occupancy"]
    assert 0.0 <= o1 <= o2 <= 1.0 and o2 > o1


def test_adhesion_transmits_occupancy_gated_traction():
    from bigraph_schema import allocate_core
    from viva_lumenoid.adhesion import BasalAdhesion
    adh = BasalAdhesion(config={"k_on": 0.5, "k_off": 0.1, "coupling_gain": 1e-3},
                        core=allocate_core())
    out = adh.update({"cortex_tension": 2.0}, 1.0)
    assert out["traction"] == pytest.approx(out["linker_occupancy"] * 2.0)
    assert out["bm_strain_rate"] == pytest.approx(1e-3 * out["traction"])


def test_e2_force_transfer_runs():
    pytest.importorskip("lammps")
    from bigraph_schema import allocate_core
    from pbg_lammps.model import LammpsModelProcess
    import viva_lumenoid.actomyosin  # noqa: F401
    from viva_lumenoid.adhesion import BasalAdhesion
    from viva_lumenoid.collagen import CollagenNetworkProcess
    from viva_lumenoid.params import CollagenParams

    cortex = LammpsModelProcess(
        config={"model": "actomyosin-cortex", "params": {"n": 5, "contractility": 1.5}},
        core=allocate_core())
    adh = BasalAdhesion(config={"k_on": 0.5, "k_off": 0.1, "coupling_gain": 1e-3},
                        core=allocate_core())
    bm = CollagenNetworkProcess(
        config={"params": CollagenParams(n_rods=30, box_xy=10.0, slab_thickness=2.0,
                                         assemble_steps=200, hold_steps=100).to_dict()},
        core=allocate_core())
    cortex.initial_state()
    bm.initial_state()
    ct = cortex.update({}, 1.0)["cortical_tension"]
    a = adh.update({"cortex_tension": ct}, 1.0)
    b = bm.update({"strain_rate": a["bm_strain_rate"],
                   "make_on": a["make_on"], "break_on": a["break_on"]}, 1.0)
    cortex.close()
    bm.close()
    assert "sigma_inplane" in b  # the collagen BM responded to the transmitted load
