"""R1 -- reduced-order homogeneous tissue baseline (prism cell + viscoelastic BM).

Pure-process (no engine) contracts for study bm-r1-published-tissue-baseline.
These lock the Fig 1G confinement regime: a stiff, slowly-relaxing BM confines the
cell (taller) vs a soft, fast-relaxing one, with cell height rising monotonically
along BOTH the stiffness and the relaxation axes of the regime map.
"""

STIFFS = [0.1, 0.5, 1.0, 2.0, 5.0]
RELAXES = [1.0, 5.0, 20.0, 50.0]


def _final(stiffness, relaxation, doubling=5.0, steps=40, dt=0.2):
    from viva_lumenoid.tissue_prism import TissuePrismProcess
    from bigraph_schema import allocate_core
    p = TissuePrismProcess(
        config={"bm_stiffness": stiffness, "bm_relaxation": relaxation,
                "doubling_time": doubling}, core=allocate_core())
    out = {}
    for _ in range(steps):
        out = p.update({}, dt)
    return out


def _final_height(stiffness, relaxation, steps=40, dt=0.2):
    return _final(stiffness, relaxation, steps=steps, dt=dt)["cell_height"]


def test_tissue_prism_baseline_registered():
    import viva_lumenoid.composites  # noqa: F401
    from process_bigraph.composite_generator import _REGISTRY
    assert "viva_lumenoid.composites.tissue_prism_baseline" in _REGISTRY


def test_confinement_regime_separates():
    # behavior_test: confinement-regime-separates (primary)
    confined = _final_height(5.0, 50.0)   # stiff, slow-relaxing BM
    spread = _final_height(0.1, 1.0)      # soft, fast-relaxing BM
    assert confined > spread > 0.0        # confinement -> taller cell
    assert confined / spread > 1.5        # ~1.66x in the recorded run


def test_height_monotonic_in_stiffness():
    # behavior_test: height-monotonic-in-stiffness (primary)
    for r in RELAXES:
        heights = [_final_height(s, r) for s in STIFFS]
        assert heights == sorted(heights), f"non-monotone in stiffness at relaxation={r}: {heights}"


def test_height_monotonic_in_relaxation():
    # behavior_test: height-monotonic-in-relaxation (primary)
    for s in STIFFS:
        heights = [_final_height(s, r) for r in RELAXES]
        assert heights == sorted(heights), f"non-monotone in relaxation at stiffness={s}: {heights}"


def test_confinement_tracks_bm_stress():
    # behavior_test: confinement-tracks-bm-stress (secondary, mechanism)
    confined = _final(5.0, 50.0)
    spread = _final(0.1, 1.0)
    # the taller (confined) cell carries the larger BM reaction stress
    assert confined["bm_stress"] > spread["bm_stress"]
    assert confined["cell_height"] > spread["cell_height"]
