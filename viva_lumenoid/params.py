"""Published parameter set for the v1 coarse-grained collagen IV network.

Every value here is a *clean-room* transcription from the Basement-membrane v1
specification (AICS-Lumenoids Roam graph, exported 2026-09-30) and the two
papers it rests on — Meadowcroft 2025 (nonequilibrium collagen IV) and
Barrientos 2026 (basement-membrane growth). These are the numbers as stated in
the spec's parameter sheet, not a byte-for-byte copy of the authors' input
files. The authors' LAMMPS input scripts are now public
(github.com/Billie1717/BasementMembraneTurnoverSims, Zenodo
10.5281/zenodo.20719515, commit ab4c004) and are the ground truth to cross-check
these against — a verification task that has not yet been done. Where the spec
records a convention uncertainty or an unresolved choice, the comment says so.

Units are Lennard-Jones (LJ) reduced units, matching the released input scripts:
  * length unit  σ  = the two-bead spacing a ≈ 114 nm (protomer 376 nm end-to-end)
  * energy unit  ε  = kT  (so one stress unit is kT/a**3 ≈ 2-3 Pa at a ≈ 114-125 nm)
  * the network's own linear Young's modulus in the paper is ≈0.03 Pa — i.e.
    10**3-10**4x softer than reconstituted BM — so v1 moduli are reported as a
    RATIO / in LJ units by default (spec v1-decision #4, energy scale still open).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class CollagenParams:
    # ---- geometry (spec: "has: geometrical structure") -------------------
    bead_spacing_nm: float = 114.0     # a; supplement mapping a ≈ 114 nm (±6% convention)
    protomer_length_nm: float = 376.0  # end-to-end; two-bead rod spans this
    persistence_length_nm: float = 40.0  # semiflexible chain, l_p ≈ 39 nm / ~360 nm contour
    fibril_diameter_nm: float = 8.0    # 5-11 nm individual fibril (readout comparator only)

    # In LJ units the bead spacing IS the length unit -> bond length = 1.0.
    bond_length: float = 1.0
    # Intra-rod bond stiffness: stiff so the protomer stays rod-like. Flexibility
    # in the real model lives at junctions (soft angles); v1-minimal folds it into
    # a finite (not infinite) rod bond and omits dynamic junction angles — a named
    # simplification (spec: LAMMPS puts l_p as soft angles where rods join).
    bond_k_intra: float = 200.0

    # ---- connectivity (spec: NC1 end binds 1, 7S end binds 3) -------------
    # atom types: 1 = NC1 end bead, 2 = 7S end bead.
    # bond types:  1 = intra-rod, 2 = NC1-NC1 crosslink, 3 = 7S-7S crosslink.
    nc1_max_crosslinks: int = 1
    svns_max_crosslinks: int = 3       # "7S" end
    crosslink_k: float = 100.0         # crosslink bond stiffness
    crosslink_r0: float = 0.30         # crosslink rest length (< bond_length)
    crosslink_cutoff: float = 0.35     # Rmin for bond/create (form when within this)
    crosslink_break_cutoff: float = 1.20  # Rmax for bond/break

    # ---- junction bending (FP1; spec: l_p enters as "soft angles where rods join")
    # A harmonic angle (partner–end–partner) at each crosslink junction gives the
    # network BENDING rigidity, the constraint that lifts a sub-isostatic (z < 4)
    # central-force network out of the floppy regime into a measurable modulus.
    # bending_k = 0 reproduces the v1 floppy central-force network exactly.
    bending_k: float = 0.0             # harmonic angle stiffness (energy/rad^2)
    bending_theta0: float = 180.0      # rest angle (deg); 180 = rods collinear across the junction

    # ---- crosslink kinetics (spec: MP make rate, break locked at FactorMult) --
    # `MP` sets bond formation; breaking is locked to it at FactorMult = 0.66,
    # giving a make/break ratio of 1.52 in the released input scripts. Calibrated by
    # matching the relaxation time to the ≈20 h collagen IV lifetime (EVD).
    make_every: int = 100              # Nevery for fix bond/create (LJ steps)
    # per-attempt formation probability (MP proxy), tuned so the assemble phase
    # reaches a percolating-but-unsaturated mesh (~0.5 crosslink/rod) without
    # overflowing LAMMPS's special-neighbor list.
    make_prob: float = 0.12
    factor_mult: float = 0.66          # break rate = factor_mult relative to make
    break_every: int = 100
    # Force-INDEPENDENT crosslink off-rate (per LJ time). Breaking is modelled by
    # deleting a random fraction 1-exp(-off_rate*Δt) of crosslink bonds each
    # interval — NOT LAMMPS `fix bond/break`, which only breaks over-stretched
    # bonds (a force-DEPENDENT overstretch model that either dissolves or
    # saturates the network, never a controllable rate). The remodelling time is
    # τ ≈ 1/off_rate; this is the knob calibrated to the ≈20 h collagen IV
    # lifetime (spec #10). v1 keeps it force-independent (catch-bonds are v2).
    off_rate: float = 0.02

    # ---- pair interactions (spec v1-decision #2: released input scripts have eps=0) --
    # Every pair coefficient is zero in the released input scripts (lj/cut 0.0 0.0), so
    # rods pass through each other — NO excluded volume in v1. Bonds carry all
    # the mechanics. We mirror that with pair_style zero.
    excluded_volume: bool = False

    # ---- thermodynamics / integrator -------------------------------------
    temperature: float = 1.0           # kT energy unit
    langevin_damp: float = 1.0         # overdamped-ish network dynamics
    timestep: float = 0.01             # released input scripts use 0.01 (Cell Methods 0.001,
                                       # supplement 0.002 — v1-decision open question)

    # ---- box / population (quasi-2D slab; spec: rods lie in-plane) --------
    n_rods: int = 200                  # small default so stage 1 runs in ~seconds
    box_xy: float = 20.0               # in units of a (σ)
    slab_thickness: float = 2.0        # thin z; quasi-2D
    seed: int = 12345

    # ---- stage-1 protocol (spec: stretch, hold, then chemistry on) --------
    assemble_steps: int = 8000         # form the initial crosslinked network
    target_strain: float = 0.15        # equibiaxial in-plane strain to reach
    strain_rate: float = 1.5e-3        # ε̇ per LJ time (stage 2 sweeps this)
    hold_steps: int = 16000            # relaxation window (spec asks >= 7 tau)

    # ---- unit conversions (energy scale still open, spec v1-decision #4) ---
    kT_per_a3_Pa: float = 2.5          # 1 LJ stress unit ≈ 2-3 Pa at a ≈ 114-125 nm

    def __post_init__(self):
        # process-bigraph's map[float] config type coerces every value to float;
        # coerce the count/step/seed fields back to int so numpy and LAMMPS get
        # the integers they require.
        for f in ('nc1_max_crosslinks', 'svns_max_crosslinks', 'make_every',
                  'break_every', 'n_rods', 'seed', 'assemble_steps', 'hold_steps'):
            setattr(self, f, int(round(getattr(self, f))))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# The single source of truth other modules import.
DEFAULT = CollagenParams()
