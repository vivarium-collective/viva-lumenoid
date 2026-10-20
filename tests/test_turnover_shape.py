"""R6 — turnover->relaxation->shape inference pipeline.

Pure-process: the perlecan intervention ordering must reproduce (gain -> longer
collagen lifetime -> slower BM relaxation -> more confinement -> taller cell).
"""


def _height(perlecan, steps=40, dt=0.2):
    from viva_lumenoid.turnover_shape import TurnoverShapeProcess
    from bigraph_schema import allocate_core
    p = TurnoverShapeProcess(
        config={"collagen_lifetime": 5.0, "perlecan_factor": perlecan,
                "doubling_time": 5.0}, core=allocate_core())
    out = {}
    for _ in range(steps):
        out = p.update({}, dt)
    return out["cell_height"]


def test_turnover_shape_pipeline_registered():
    import viva_lumenoid.composites  # noqa: F401
    from process_bigraph.composite_generator import _REGISTRY
    assert "viva_lumenoid.composites.turnover_shape_pipeline" in _REGISTRY


def test_perlecan_intervention_ordering():
    # perlecan gain > control > loss (Barrientos Fig 7 ordering).
    assert _height(2.0) > _height(1.0) > _height(0.3)
