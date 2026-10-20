"""Actomyosin cortex (E1) over the viva-lammps model seam.

Proves the collagen network's sibling is a LammpsModel PLUGIN driven by the
generic LammpsModelProcess — same engine, different model. Skips cleanly when the
LAMMPS python binding is unavailable.
"""
import pytest


def _run(eps, n=6, steps=3):
    from bigraph_schema import allocate_core
    from pbg_lammps.model import LammpsModelProcess
    import viva_lumenoid.actomyosin  # noqa: F401  (registers 'actomyosin-cortex')

    p = LammpsModelProcess(
        config={"model": "actomyosin-cortex", "params": {"n": n, "contractility": eps}},
        core=allocate_core(),
    )
    p.initial_state()
    out = {}
    for _ in range(steps):
        out = p.update({}, 1.0)
    p.close()
    return out


def test_actomyosin_cortex_runs():
    pytest.importorskip("lammps")
    out = _run(1.0, n=5, steps=2)
    assert "cortical_tension" in out
    assert out["num_atoms"] == 25  # 5x5 square lattice patch


def test_contractility_raises_cortical_tension():
    pytest.importorskip("lammps")
    # Higher myosin-activity proxy (LJ well depth) => higher cortical tension.
    assert _run(3.0)["cortical_tension"] > _run(1.0)["cortical_tension"]


def test_cortical_tension_monotone_in_contractility():
    pytest.importorskip("lammps")
    # E1 primary contract: cortical tension rises strictly along the full
    # contractility sweep {0.5, 1.0, 2.0, 3.0} (the myosin-activity proxy).
    tensions = [_run(eps)["cortical_tension"] for eps in (0.5, 1.0, 2.0, 3.0)]
    assert all(lo < hi for lo, hi in zip(tensions, tensions[1:])), tensions


def test_model_is_registered():
    pytest.importorskip("pbg_lammps")
    from pbg_lammps.model import get_model
    import viva_lumenoid.actomyosin  # noqa: F401
    assert get_model("actomyosin-cortex").__name__ == "ActomyosinCortexModel"
