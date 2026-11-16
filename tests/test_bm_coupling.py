"""R4 explicit BM <-> epithelium coupling (the reviewer's #1 ask, 2026-11-09).

The BM is a face energy inside the tyssue vertex solver, so its tension is
carried continuously across cells through shared vertices (unlike the retired
per-cell Maxwell surrogate). These tests pin the coupled-sheet physics:

  * basal area responds to the preferred (growing) area (the engine gap R4 flagged
    — CrossSectionGeometry didn't respond — is fixed by driving the mesh with
    PlanarGeometry);
  * a stiff BM confines the realized area below the preferred (the confinement);
  * with a growing cell volume, that confinement drives the volume into cell
    HEIGHT (absolute thickening) — more for a stiffer/slower BM;
  * a BM-stiffness gradient gives a SMOOTH, monotone area/height gradient across
    the sheet (the neighbour coupling the per-cell elements could not produce).

Skips cleanly when the optional tyssue runtime stack is absent (CI without it).
"""
import numpy as np
import pytest


def _need_tyssue():
    for mod in ("tyssue", "vivarium_tyssue", "tables"):
        pytest.importorskip(mod)


def _run(**kw):
    _need_tyssue()
    from viva_lumenoid.bm_coupling import run_bm_coupled_sheet
    return run_bm_coupled_sheet(steps=kw.pop("steps", 24), **kw)


def test_bm_effector_registers_into_engine_map():
    _need_tyssue()
    from bigraph_schema import allocate_core
    # The BM effector must be resolvable by the EulerSolver after registration.
    from viva_lumenoid.bm_coupling import register_bm_coupling, bm_effector
    register_bm_coupling(allocate_core())
    from vivarium_tyssue.maps import EFFECTORS_MAP
    assert EFFECTORS_MAP.get("BMFaceAreaElasticity") is bm_effector()


def test_area_responds_to_preferred_without_bm():
    # No BM: the realized basal area tracks the growing preferred area (the engine
    # responds to prefered_area — the gap R4 flagged is gone).
    r = _run(mode="no-bm", rate=0.04)
    assert r.a_mean[-1] > r.a_mean[0] * 1.2          # it grew substantially
    assert r.gap_final == pytest.approx(0.0, abs=0.03)  # tracks the target
    assert r.bm_tension_mean[-1] == pytest.approx(0.0, abs=1e-9)


def test_stiff_bm_confines_area_and_carries_tension():
    free = _run(mode="no-bm", rate=0.04)
    stiff = _run(mode="elastic", rate=0.04, bm_stiffness=40.0)
    assert stiff.gap_final > 0.1                      # real confinement
    assert stiff.gap_final > free.gap_final + 0.1     # much more than free
    assert stiff.a_mean[-1] < free.a_mean[-1]         # confined below free
    assert stiff.bm_tension_mean[-1] > 0.0            # BM carries reaction tension


def test_confined_volume_growth_increases_height():
    # Growing cell volume + BM-confined basal area -> the volume goes into HEIGHT.
    free = _run(mode="no-bm", rate=0.04, vol_rate=0.05)
    stiff = _run(mode="elastic", rate=0.04, vol_rate=0.05, bm_stiffness=40.0)
    assert stiff.height_gain > 0.1                    # absolute thickening
    assert stiff.height_gain > free.height_gain + 0.1  # BM drives the thickening
    assert stiff.height_mean[-1] > stiff.height_mean[0]


def test_relaxing_bm_confines_less_than_elastic():
    elastic = _run(mode="elastic", rate=0.04, bm_stiffness=40.0)
    relaxing = _run(mode="relaxing", rate=0.04, bm_stiffness=40.0, bm_relaxation=12.0)
    # A remodelling BM relaxes its rest area, so it confines no more than a frozen one.
    assert relaxing.gap_final <= elastic.gap_final + 1e-6
    assert relaxing.bm_tension_mean[-1] > 0.0


def test_stiffness_gradient_gives_smooth_neighbour_coupled_gradient():
    # Continuous BM tension across shared vertices -> stiffer faces are smaller and
    # taller, monotonically in the stiffness coordinate (not independent jumps).
    r = _run(mode="elastic", rate=0.04, vol_rate=0.05, bm_stiffness=40.0,
             stiffness_gradient=20.0)
    # Rank (Spearman) monotonicity — the confinement saturates at high stiffness,
    # so the field is monotone but nonlinear; rank correlation is the right metric.
    from scipy.stats import spearmanr
    ca = spearmanr(r.stiffness, r.area_final_cells).statistic
    ch = spearmanr(r.stiffness, r.height_final_cells).statistic
    assert ca < -0.9          # stiffer BM -> smaller basal area (monotone)
    assert ch > 0.9           # stiffer BM -> taller cell (monotone)
    # Neighbour coupling = a smooth, monotone field (not independent per-cell
    # jumps): area is monotone non-increasing in the stiffness coordinate (within
    # a small numerical tolerance), and the stiff end is clearly confined below
    # the soft end.
    a_by_k = r.area_final_cells[np.argsort(r.stiffness)]
    # No significant up-jump (tolerates tiny ties between equal-stiffness faces);
    # Spearman above already pins the overall monotonicity.
    assert np.max(np.diff(a_by_k)) <= 0.02 * a_by_k.mean()
    assert a_by_k[0] - a_by_k[-1] > 0.1 * a_by_k[0]              # soft end >> stiff end
