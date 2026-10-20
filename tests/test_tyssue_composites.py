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


def _build_and_step(cid, step=True, n_steps=1):
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
        for _ in range(n_steps):
            comp.update({}, 0.01)  # each Euler step must not raise
    return node


def _finite_positive(df, col):
    """A dataframe column is finite and strictly positive across all rows."""
    import numpy as np
    assert col in df.columns, f"missing column {col!r}"
    v = np.asarray(df[col], dtype=float)
    assert np.isfinite(v).all(), f"{col}: non-finite values present"
    assert (v > 0).all(), f"{col}: non-positive values present"
    return v


def test_closed_lumenoid_runs():
    _need_tyssue()
    node = _build_and_step("closed_lumenoid")
    assert node.eptm.Nc > 0 and node.eptm.Nf > 0  # lumenoid mesh loaded + stepped
    # E3 evidence contract: a closed shell with finite, positive per-cell geometry.
    _finite_positive(node.eptm.cell_df, "area")
    _finite_positive(node.eptm.cell_df, "vol")


def test_tyssue_3d_monolayer_runs():
    _need_tyssue()
    node = _build_and_step("tyssue_3d_monolayer")
    assert node.eptm.Nc > 0 and node.eptm.Nf > 0  # 3D monolayer loaded + stepped
    # R5 evidence contract: explicit per-cell volumes emerge finite and positive.
    _finite_positive(node.eptm.cell_df, "area")
    _finite_positive(node.eptm.cell_df, "vol")


def test_tyssue_2d_sheet_runs():
    # R4 steps thanks to the numpy-2.x 2D-cross shim in tyssue_coupling.
    _need_tyssue()
    node = _build_and_step("tyssue_2d_sheet")
    assert node.eptm.Nf > 0
    # R4 evidence contract: every cell-face carries a finite, positive basal area
    # and an explicit finite height degree of freedom.
    _finite_positive(node.eptm.face_df, "area")
    import numpy as np
    heights = np.asarray(node.eptm.face_df["height"], dtype=float)
    assert np.isfinite(heights).all(), "face height: non-finite values present"


def test_tyssue_composites_step_stably():
    """Stepping several Euler steps keeps per-cell geometry finite (no NaN/blow-up)."""
    _need_tyssue()
    import numpy as np
    for cid, df_attr, cols in (
        ("tyssue_2d_sheet", "face_df", ("area",)),
        ("tyssue_3d_monolayer", "cell_df", ("area", "vol")),
        ("closed_lumenoid", "cell_df", ("area", "vol")),
    ):
        node = _build_and_step(cid, n_steps=5)
        df = getattr(node.eptm, df_attr)
        for col in cols:
            v = np.asarray(df[col], dtype=float)
            assert np.isfinite(v).all(), f"{cid}.{df_attr}.{col}: NaN/inf after 5 steps"
            assert (v > 0).all(), f"{cid}.{df_attr}.{col}: non-positive after 5 steps"
