"""Tyssue composites (E3 closed lumenoid) over the vivarium-tyssue engine.

Proves the EulerSolver-over-a-geometry wiring resolves and steps from
viva-lumenoid, using the merged LumenoidGeometry + a vendored lumenoid mesh.
Skips cleanly when the optional tyssue runtime stack is absent (CI without it).
"""
import pytest


def _need_tyssue():
    for mod in ("tyssue", "vivarium_tyssue", "tables", "quantities"):
        pytest.importorskip(mod)


def test_closed_lumenoid_registered():
    import viva_lumenoid.composites  # noqa: F401
    from process_bigraph.composite_generator import _REGISTRY
    assert "viva_lumenoid.composites.closed_lumenoid" in _REGISTRY


def _build_and_step(cid, step=True):
    import viva_lumenoid.composites  # noqa: F401  (registers generators)
    from process_bigraph.composite_generator import (
        _REGISTRY, apply_core_extensions, build_generator,
    )
    from bigraph_schema import allocate_core
    from process_bigraph import Composite

    entry = _REGISTRY[f"viva_lumenoid.composites.{cid}"]
    core = apply_core_extensions(entry, allocate_core())  # registers the tyssue engine
    doc = build_generator(entry, None, core=core)
    state = doc["state"] if isinstance(doc, dict) and "state" in doc else doc
    comp = Composite({"state": state}, core=core)
    node = comp.state["Tyssue"]["instance"]
    if step:
        comp.update({}, 0.01)  # one Euler step must not raise
    return node


def test_closed_lumenoid_runs():
    _need_tyssue()
    node = _build_and_step("closed_lumenoid")
    assert node.eptm.Nc > 0 and node.eptm.Nf > 0  # lumenoid mesh loaded + stepped


def test_tyssue_3d_monolayer_runs():
    _need_tyssue()
    node = _build_and_step("tyssue_3d_monolayer")
    assert node.eptm.Nc > 0 and node.eptm.Nf > 0  # 3D monolayer loaded + stepped


def test_tyssue_2d_sheet_runs():
    # R4 steps thanks to the numpy-2.x 2D-cross shim in tyssue_coupling.
    _need_tyssue()
    node = _build_and_step("tyssue_2d_sheet")
    assert node.eptm.Nf > 0
