"""P3 rigor — coupling controls + convergence for the LAMMPS<->Tyssue coupling.

Uses the reduced-order (effective) BM so these run without LAMMPS; the tyssue
epithelium is still real, so the tyssue stack is required.
"""
import pytest


def _need():
    for mod in ("tyssue", "vivarium_tyssue", "tables", "quantities"):
        pytest.importorskip(mod)


def _run(coupling_mode="two-way", substeps=1, n=15, dt=0.02):
    from bigraph_schema import allocate_core
    from viva_lumenoid.multiscale_coupling import LammpsTyssueCoupling
    p = LammpsTyssueCoupling(config={
        "bm_mode": "reduced", "coupling_mode": coupling_mode, "bm_substeps": substeps,
        "strain_gain": 20.0, "tension_gain": 0.02,
        "bm_modulus": 50.0, "bm_relaxation": 10.0}, core=allocate_core())
    out = {}
    for _ in range(n):
        out = p.update({}, dt)
    p.close()
    return out


def test_coupling_controls():
    _need()
    two_way = _run("two-way")
    one_way = _run("one-way")
    uncoupled = _run("uncoupled")
    assert two_way["basal_tension"] != 0.0     # two-way: BM reaction fed back (sign may vary)
    assert one_way["basal_tension"] == 0.0     # one-way: no reaction fed back
    assert uncoupled["bm_sigma"] == 0.0        # uncoupled: epithelium doesn't drive the BM


def test_bm_subcycling_converges():
    _need()
    s1 = _run("two-way", substeps=1)["bm_sigma"]
    s2 = _run("two-way", substeps=2)["bm_sigma"]
    s4 = _run("two-way", substeps=4)["bm_sigma"]
    s8 = _run("two-way", substeps=8)["bm_sigma"]
    # successive-refinement error shrinks as the BM is subcycled (convergence).
    assert abs(s8 - s4) < abs(s2 - s1)


# --- EXPLICIT-LAMMPS-BM path (the real collagen BM in the loop) -------------
def _run_explicit(coupling_mode="two-way", substeps=1, n=16, dt=0.02, bm_rods=40):
    from bigraph_schema import allocate_core
    from viva_lumenoid.multiscale_coupling import LammpsTyssueCoupling
    p = LammpsTyssueCoupling(config={
        "bm_mode": "explicit", "coupling_mode": coupling_mode, "bm_substeps": substeps,
        "bm_rods": bm_rods, "strain_gain": 20.0, "tension_gain": 0.02}, core=allocate_core())
    out = {}
    for _ in range(n):
        out = p.update({}, dt)
    p.close()
    return out


def test_explicit_bm_loop_runs_with_directionality():
    """The REAL LAMMPS-collagen-BM <-> Tyssue loop runs bounded in the stable regime
    (40 rods, coarse subcycling) with correct feedback directionality. Locks the robust
    explicit-path claims (F4); the explicit subcycle DIVERGENCE (F5) is a recorded finding,
    not asserted as a stable contract."""
    _need()
    pytest.importorskip("lammps")
    two_way = _run_explicit("two-way")
    one_way = _run_explicit("one-way")
    # explicit loop stays bounded at 40 rods / 16 steps (120 rods/24 steps diverges — see F5)
    assert abs(two_way["bm_sigma"]) < 1000.0
    # feedback directionality holds on the explicit path
    assert two_way["basal_tension"] != 0.0     # two-way feeds the real collagen reaction back
    assert one_way["basal_tension"] == 0.0     # one-way feeds none back (exactly zero)
