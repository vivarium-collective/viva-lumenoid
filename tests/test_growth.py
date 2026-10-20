"""Preferred (target) growth + reduced-order BM reaction — the meeting-aligned
cell<->BM feedback primitive (viva_lumenoid.growth).

Locks the 2026-10-20 review's modeling constraint: growth is an increase in the
PREFERRED (target) geometry, NOT an imposed realized geometry; the realized
geometry LAGS the target because a viscoelastic BM resists the expansion, and
that lag IS the confinement. Growth is distinct from cell division.
"""
import math

import pytest

from viva_lumenoid.growth import PreferredGrowth, ReducedBM


# --- PreferredGrowth: the target law ---------------------------------------

def test_no_growth_is_constant():
    g = PreferredGrowth(x0=1.0, mode="none")
    assert g.target(0.0) == 1.0
    assert g.target(10.0) == 1.0


def test_linear_growth():
    g = PreferredGrowth(x0=2.0, mode="linear", rate=0.1)
    assert g.target(0.0) == pytest.approx(2.0)
    assert g.target(5.0) == pytest.approx(2.0 * (1.0 + 0.1 * 5.0))  # 3.0


def test_exponential_growth():
    g = PreferredGrowth(x0=1.0, mode="exponential", rate=0.1)
    assert g.target(0.0) == pytest.approx(1.0)
    assert g.target(5.0) == pytest.approx(math.exp(0.5))


def test_growth_is_monotonic_increasing():
    for mode in ("linear", "exponential"):
        g = PreferredGrowth(x0=1.0, mode=mode, rate=0.05)
        ts = [g.target(t) for t in range(0, 20)]
        assert ts == sorted(ts)
        assert ts[-1] > ts[0]


def test_division_is_off():
    # growth != division: this primitive never adds cells (proliferation OFF).
    g = PreferredGrowth(x0=1.0, mode="exponential", rate=0.1)
    assert g.divides is False


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        PreferredGrowth(x0=1.0, mode="proliferate")


# --- ReducedBM: the viscoelastic reaction ----------------------------------

def test_bm_zero_stress_at_rest():
    bm = ReducedBM(x0=1.0, stiffness=1.0, relaxation=5.0)
    assert bm.react(1.0, dt=0.1) == pytest.approx(0.0)


def test_bm_stress_rises_with_strain():
    bm = ReducedBM(x0=1.0, stiffness=2.0, relaxation=1e9)  # no relaxation
    s = bm.react(1.5, dt=0.1)
    assert s == pytest.approx(2.0 * 0.5)  # stiffness * (x - rest)


def test_stiffer_bm_reacts_harder():
    soft = ReducedBM(x0=1.0, stiffness=0.5, relaxation=1e9)
    stiff = ReducedBM(x0=1.0, stiffness=5.0, relaxation=1e9)
    assert stiff.react(1.5, 0.1) > soft.react(1.5, 0.1)


def test_zero_stiffness_is_free_control():
    # no-BM / free-growth control: stiffness 0 -> no reaction at any strain.
    bm = ReducedBM(x0=1.0, stiffness=0.0, relaxation=5.0)
    assert bm.react(3.0, dt=0.1) == pytest.approx(0.0)


def test_fast_relaxation_sheds_stress():
    # a fast-relaxing BM lets its rest state chase the area -> sheds stress;
    # a slow-relaxing BM holds stress (confines).
    fast = ReducedBM(x0=1.0, stiffness=1.0, relaxation=0.2)
    slow = ReducedBM(x0=1.0, stiffness=1.0, relaxation=50.0)
    for _ in range(20):
        s_fast = fast.react(1.5, dt=0.1)
        s_slow = slow.react(1.5, dt=0.1)
    assert s_slow > s_fast
    assert s_fast < 0.5 * s_slow  # fast has largely relaxed


# --- The mechanistic acceptance the meeting asked for ----------------------

def _confinement(stiffness, relaxation, rate=0.05, steps=60, dt=0.2, mobility=0.3):
    """Single reduced cell: realized area relaxes toward a growing preferred
    target, resisted by a reduced BM.  Returns (A_pref, A_realized, stress)."""
    g = PreferredGrowth(x0=1.0, mode="exponential", rate=rate)
    bm = ReducedBM(x0=1.0, stiffness=stiffness, relaxation=relaxation)
    A = 1.0
    t = 0.0
    stress = 0.0
    for _ in range(steps):
        t += dt
        A_pref = g.target(t)
        stress = bm.react(A, dt)
        A += mobility * ((A_pref - A) - stress) * dt
        A = max(A, 1e-6)
    return g.target(t), A, stress


def test_no_growth_gives_no_bm_strain():
    # zero preferred growth -> no growth-driven BM strain, realized stays ~initial.
    g = PreferredGrowth(x0=1.0, mode="none")
    bm = ReducedBM(x0=1.0, stiffness=5.0, relaxation=50.0)
    A = 1.0
    for _ in range(60):
        s = bm.react(A, 0.2)
        A += 0.3 * ((g.target(0.0) - A) - s) * 0.2
    assert A == pytest.approx(1.0, abs=1e-3)
    assert s == pytest.approx(0.0, abs=1e-3)


def test_realized_is_not_hardcoded_to_preferred():
    # THE meeting test: with a stiff/slow BM the realized area is strictly LESS
    # than the preferred target -> realized emerges from the mechanical balance,
    # it does not track the target by construction.
    A_pref, A_real, stress = _confinement(stiffness=5.0, relaxation=50.0)
    assert A_real < A_pref
    assert (A_pref - A_real) / A_pref > 0.05   # a real, not numerical, gap
    assert stress > 0.0


def test_bm_off_realizes_growth():
    # BM-off (free) -> preferred growth is largely realized: no BM reaction, and
    # the only remaining lag is the finite-mobility chase of a moving target, so
    # the gap is far smaller than under a stiff/slow BM.
    pref0, free, stress0 = _confinement(stiffness=0.0, relaxation=5.0)
    pref1, stiff, _ = _confinement(stiffness=5.0, relaxation=50.0)
    assert stress0 == pytest.approx(0.0)
    free_gap = (pref0 - free) / pref0
    stiff_gap = (pref1 - stiff) / pref1
    assert free_gap < 0.5 * stiff_gap   # free growth is much more fully realized
    assert free > stiff                 # and reaches a larger realized area


def test_stiff_bm_widens_the_gap():
    # stiff BM -> bigger preferred-vs-realized gap than soft BM.
    _, soft, _ = _confinement(stiffness=0.5, relaxation=50.0)
    _, stiff, _ = _confinement(stiffness=5.0, relaxation=50.0)
    assert stiff < soft  # stiffer confines more -> less realized area
