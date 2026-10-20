"""E4 / P3 — bidirectional LAMMPS collagen-BM <-> Tyssue epithelium coupling.

The capstone: the epithelial basal surface stretches an explicit LAMMPS collagen
BM, whose stress feeds back as a basal tension. Skips without the full stack.
"""
import pytest


def test_coupled_lumenoid_registered():
    import viva_lumenoid.composites  # noqa: F401
    from process_bigraph.composite_generator import _REGISTRY
    assert "viva_lumenoid.composites.coupled_lumenoid" in _REGISTRY


def test_lammps_tyssue_coupling_runs():
    for mod in ("lammps", "tyssue", "vivarium_tyssue", "tables", "quantities"):
        pytest.importorskip(mod)
    from bigraph_schema import allocate_core
    from viva_lumenoid.multiscale_coupling import LammpsTyssueCoupling

    p = LammpsTyssueCoupling(
        config={"bm_rods": 30, "strain_gain": 20.0, "tension_gain": 0.02},
        core=allocate_core())
    outs = [p.update({}, 0.02) for _ in range(3)]
    p.close()

    for o in outs:
        assert {"basal_area", "basal_strain_rate", "bm_sigma",
                "basal_tension", "lumen_pressure"} <= set(o)
    # the epithelium moved (basal area changed) AND the LAMMPS BM carried stress —
    # i.e. the two engines are actually coupled, not idle.
    assert len({round(o["basal_area"], 3) for o in outs}) > 1
    assert any(abs(o["bm_sigma"]) > 1e-6 for o in outs)
