"""Coupled growth experiments (R4 reduced-order cross-section, R5 real Tyssue
monolayer) — the meeting-aligned preferred->realized->confinement chain.

R4 is engine-free (fast). R5 steps a real Tyssue monolayer and is marked slow.
"""
import numpy as np
import pytest

from viva_lumenoid.growth_runs import run_sheet_growth_2d


# --- R4: reduced-order 2D cross-section ------------------------------------

def test_sheet_growth_happens_and_lags_under_stiff_bm():
    r = run_sheet_growth_2d(bm_stiffness=5.0, bm_relaxation=50.0, steps=60)
    assert r.a_mean[-1] > r.a_mean[0]              # growth happened
    assert r.a_mean[-1] < r.a_pref[-1]             # realized lags the target
    assert r.gap_final > 0.05                      # a real confinement gap
    assert r.stress_mean[-1] > 0.0                 # BM carries reaction stress


def test_sheet_no_growth_is_flat():
    r = run_sheet_growth_2d(mode="none", bm_stiffness=5.0, steps=40)
    assert r.a_mean[-1] == pytest.approx(r.a_mean[0], abs=1e-3)
    assert abs(r.stress_mean[-1]) < 1e-3


def test_sheet_free_bm_realizes_more_than_stiff():
    free = run_sheet_growth_2d(bm_stiffness=0.0, steps=60)
    stiff = run_sheet_growth_2d(bm_stiffness=5.0, bm_relaxation=50.0, steps=60)
    assert free.a_mean[-1] > stiff.a_mean[-1]      # BM-off spreads more
    assert free.height_mean[-1] < stiff.height_mean[-1]  # stiff BM -> taller cells


def test_sheet_stiffness_gradient_makes_height_gradient():
    r = run_sheet_growth_2d(bm_stiffness=3.0, stiffness_gradient=1.5,
                            bm_relaxation=50.0, steps=60)
    # stiffer end confines more -> smaller area -> taller reconstructed height
    hi_stiff = r.height_cells_final[np.argmax(r.stiffness)]
    lo_stiff = r.height_cells_final[np.argmin(r.stiffness)]
    assert hi_stiff > lo_stiff


def test_sheet_reconstructed_height_is_fixed_volume():
    r = run_sheet_growth_2d(bm_stiffness=2.0, steps=30, v0=1.0)
    # h = V0 / A at every step (declared fixed-volume reconstruction)
    assert r.height_mean[-1] == pytest.approx(float(np.mean(1.0 / r.a_cells[-1])))


# --- R5: real Tyssue monolayer (slow) --------------------------------------

@pytest.mark.slow
def test_monolayer_growth_runs_and_grows():
    from viva_lumenoid.growth_runs import run_monolayer_growth_3d
    r = run_monolayer_growth_3d(rate=0.4, bm_stiffness=0.5, steps=16)
    assert len(r.t) == 17
    assert r.v_real[-1] > r.v_real[0]              # realized volume grew
    assert r.v_pref[-1] > r.v_pref[0]              # target grew
    assert r.gap_final > 0.0                       # realized lags the target
    assert np.all(np.isfinite(r.height_mean))
    assert r.heights_final.size > 0


@pytest.mark.slow
def test_monolayer_free_vs_stiff_bm():
    from viva_lumenoid.growth_runs import run_monolayer_growth_3d
    free = run_monolayer_growth_3d(rate=0.4, bm_stiffness=0.0, steps=18)
    stiff = run_monolayer_growth_3d(rate=0.4, bm_stiffness=5.0,
                                    bm_relaxation=50.0, steps=18)
    # a stiff BM confines basal spreading relative to the free control, and
    # carries a positive reaction stress the free control does not.
    assert stiff.basal_area[-1] < free.basal_area[-1]
    assert stiff.stress[-1] > free.stress[-1]
    assert stiff.height_mean[-1] >= free.height_mean[-1]  # confinement -> taller (directional)
