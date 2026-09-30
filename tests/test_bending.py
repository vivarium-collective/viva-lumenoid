"""Junction bending (FP1): k=0 is a clean no-op; k>0 adds restraints and rigidifies."""
from __future__ import annotations

import pytest

pytest.importorskip("lammps", reason="LAMMPS Python bindings required")

from process_bigraph import allocate_core

from viva_lumenoid import CollagenParams, run_stage1


def test_bending_zero_is_noop_on_topology():
    """bending_k=0 must not create restraints or change the assemble behaviour."""
    from viva_lumenoid.collagen import CollagenNetworkProcess
    p = CollagenParams(n_rods=40, box_xy=10.0, bending_k=0.0)
    proc = CollagenNetworkProcess(config={"params": p.to_dict()}, core=allocate_core())
    proc.initial_state()
    for _ in range(6):
        proc.update({"strain_rate": 0.0, "make_on": 1.0, "break_on": 1.0}, 20.0)
    # freeze transition would pin restraints only when bending_k>0
    proc.update({"strain_rate": 0.0, "make_on": 0.0, "break_on": 0.0}, 20.0)
    assert proc._restrain_on is False
    proc.close()


def test_bending_pins_restraints_when_on():
    """bending_k>0 pins junction restraints at the assemble→frozen transition."""
    from viva_lumenoid.collagen import CollagenNetworkProcess
    p = CollagenParams(n_rods=60, box_xy=12.0, bending_k=30.0, timestep=0.002)
    proc = CollagenNetworkProcess(config={"params": p.to_dict()}, core=allocate_core())
    proc.initial_state()
    for _ in range(8):  # assemble (make+break) — restraints NOT created yet
        proc.update({"strain_rate": 0.0, "make_on": 1.0, "break_on": 1.0}, 8.0)
    assert proc._restrain_on is False
    proc.update({"strain_rate": 0.0, "make_on": 0.0, "break_on": 0.0}, 8.0)  # freeze
    assert proc._restrain_on is True     # pinned once at the transition
    proc.close()


@pytest.mark.slow
def test_bending_changes_mechanics():
    """End-to-end: the floppy baseline is ~0 and bending produces a large
    mechanical response (|E| >> floppy). The bending VALUE is ill-conditioned at
    this size (bm-v4), so we assert the order-of-magnitude change, not a number."""
    from viva_lumenoid.bending import run_bending_sweep
    r = run_bending_sweep(bending_ks=[0.0, 80.0], seeds=[11, 22, 33],
                          base=CollagenParams(n_rods=120, box_xy=16.0))
    floppy = abs(r.points[0].modulus_median)
    rigid = max(abs(m) for m in r.points[1].per_seed)
    assert floppy < 0.1        # floppy modulus is cleanly ~0
    assert rigid > 1.0         # bending produces a large mechanical response
