"""R1 — reduced-order homogeneous tissue baseline (prism cell + viscoelastic BM).

Pure-process (no engine): the confinement regime must separate — a stiff,
slowly-relaxing BM confines the cell (taller) vs a soft, fast-relaxing one.
"""


def _final_height(stiffness, relaxation, steps=40, dt=0.2):
    from viva_lumenoid.tissue_prism import TissuePrismProcess
    from bigraph_schema import allocate_core
    p = TissuePrismProcess(
        config={"bm_stiffness": stiffness, "bm_relaxation": relaxation,
                "doubling_time": 5.0}, core=allocate_core())
    out = {}
    for _ in range(steps):
        out = p.update({}, dt)
    return out["cell_height"]


def test_tissue_prism_baseline_registered():
    import viva_lumenoid.composites  # noqa: F401
    from process_bigraph.composite_generator import _REGISTRY
    assert "viva_lumenoid.composites.tissue_prism_baseline" in _REGISTRY


def test_confinement_regime_separates():
    confined = _final_height(5.0, 50.0)   # stiff, slow-relaxing BM
    spread = _final_height(0.1, 1.0)      # soft, fast-relaxing BM
    assert confined > spread > 0.0        # confinement -> taller cell
