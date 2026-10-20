"""R6 -- turnover->relaxation->shape inference pipeline.

Pure-process contracts for study bm-r6-perlecan-growth-perturbations. These lock
the Barrientos Fig 7 intervention ordering. Perlecan promotes collagen IV bond
breakage, so perlecan KNOCKDOWN lengthens the collagen lifetime -> slower BM
relaxation -> more confinement -> TALLER cell, and OVEREXPRESSION shortens it ->
faster relaxation -> shorter cell. perlecan_factor is the lifetime multiplier:
< 1 overexpression, > 1 knockdown. Also locks the linear inferred-relaxation
scaling, the inferred/predicted output split, and the growth-rate effect.
"""


def _run(perlecan=1.0, lifetime=5.0, doubling=5.0, steps=40, dt=0.2):
    from viva_lumenoid.turnover_shape import TurnoverShapeProcess
    from bigraph_schema import allocate_core
    p = TurnoverShapeProcess(
        config={"collagen_lifetime": lifetime, "perlecan_factor": perlecan,
                "doubling_time": doubling}, core=allocate_core())
    out = {}
    for _ in range(steps):
        out = p.update({}, dt)
    return out


def _height(perlecan, steps=40, dt=0.2):
    return _run(perlecan=perlecan, steps=steps, dt=dt)["cell_height"]


def test_turnover_shape_pipeline_registered():
    import viva_lumenoid.composites  # noqa: F401
    from process_bigraph.composite_generator import _REGISTRY
    assert "viva_lumenoid.composites.turnover_shape_pipeline" in _REGISTRY


def test_perlecan_intervention_ordering():
    # behavior_test: perlecan-intervention-ordering (primary)
    # Barrientos Fig 7: knockdown (longer lifetime, factor>1) -> taller >
    # control > overexpression (shorter lifetime, factor<1) -> shorter.
    assert _height(25.0) > _height(1.0) > _height(0.05)


def test_inferred_relaxation_scales_with_perlecan():
    # behavior_test: inferred-relaxation-scales-with-perlecan (primary)
    # inferred_relaxation = collagen_lifetime * perlecan_factor (strictly increasing, linear).
    relax = [_run(perlecan=pf)["inferred_relaxation"] for pf in (0.3, 1.0, 2.0, 3.0)]
    assert relax == sorted(relax)
    for pf in (0.3, 1.0, 2.0, 3.0):
        assert _run(perlecan=pf)["inferred_relaxation"] == 5.0 * pf


def test_inferred_and_predicted_distinct():
    # behavior_test: inferred-and-predicted-distinct (secondary)
    out = _run(perlecan=2.0)
    assert "inferred_relaxation" in out and "cell_height" in out
    # the inferred (turnover) quantity is the relaxation time, not the predicted height
    assert out["inferred_relaxation"] == 10.0
    assert out["inferred_relaxation"] != out["cell_height"]


def test_faster_growth_spreads():
    # behavior_test: faster-growth-spreads (secondary)
    fast = _run(perlecan=1.0, doubling=2.5)["cell_height"]
    control = _run(perlecan=1.0, doubling=5.0)["cell_height"]
    assert fast < control
