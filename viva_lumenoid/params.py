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
    # CROSS-CHECK CORRECTION (2026-10-01): the released rod backbone bond is
    # `bond_coeff 1 200.0 3.0` — the two beads of a protomer sit 3σ apart. So the
    # LJ length unit is σ = protomer / 3 = 376 / 3 ≈ 125 nm (NOT 114, which assumed
    # a 1σ rod). bead_spacing_nm is that σ→nm factor.
    bead_spacing_nm: float = 125.0     # σ = protomer/3 ≈ 125 nm (released rod = 3σ)
    protomer_length_nm: float = 376.0  # end-to-end; two-bead rod spans 3σ
    persistence_length_nm: float = 40.0  # semiflexible chain, l_p ≈ 39 nm / ~360 nm contour
    fibril_diameter_nm: float = 8.0    # 5-11 nm individual fibril (readout comparator only)

    # Rod backbone bond: k=200, r0=3.0 (released `bond_coeff 1 200.0 3.0`).
    bond_length: float = 3.0           # released rod backbone rest length (3σ)
    bond_k_intra: float = 200.0        # released rod backbone stiffness (matches)

    # ---- connectivity (spec: NC1 end binds 1, 7S end binds 3) -------------
    # atom types: 1 = NC1 end bead, 2 = 7S end bead.
    # bond types:  1 = intra-rod, 2 = NC1-NC1 crosslink, 3 = 7S-7S crosslink.
    # Released crosslink bonds: `bond_coeff 2/3 6.0 0.5` (NC1 and 7S), formation
    # cutoff MD=0.65, breaking cutoff BD=1.35 (NC1); the released 7S break cutoff
    # is 0.95 — our single break cutoff uses the NC1 value (7S-specific cutoff is a
    # remaining fidelity gap). (CROSS-CHECK CORRECTION 2026-10-01.)
    nc1_max_crosslinks: int = 1
    svns_max_crosslinks: int = 3       # "7S" end (released max_bonds_7s = 3)
    crosslink_k: float = 6.0           # released crosslink bond stiffness
    crosslink_r0: float = 0.5          # released crosslink rest length (0.5σ ≈ 62 nm)
    crosslink_cutoff: float = 0.65     # released MD (form when within this)
    crosslink_break_cutoff: float = 1.35  # released BD (NC1); 7S BD=0.95 not yet split

    # ---- junction bending (harmonic angle at each crosslink junction) --------
    # CROSS-CHECK CORRECTION (2026-10-01, see references/cross-check-vs-released-
    # code.md): the released code is NOT a central-force network. It sets an NC1
    # junction angle `angle_coeff 1 4.0 180.0` (KangNC1 = 4.0, rest 180°) plus two
    # 7S angles (k=1.0 @155°, k=1.0 @60°). So bending is a core feature of the
    # published model, not an add-on, and `bending_k = 0` does NOT reproduce it —
    # it omits it. The default stays 0 only to keep existing runs reproducible;
    # the released NC1 value is 4.0 and a faithful run must set it (and add the 7S
    # angles). The floppy/central-force results (bm-v1/bm-v3/bm-v4) are artifacts
    # of this omission, pending a re-run with the released angle set.
    bending_k: float = 0.0             # harmonic angle stiffness; RELEASED NC1 value = 4.0
    bending_theta0: float = 180.0      # rest angle (deg); released NC1 angle = 180°

    # ---- crosslink kinetics (spec: MP make rate, break locked at FactorMult) --
    # `MP` sets bond formation; breaking is locked to it at FactorMult = 0.66,
    # giving a make/break ratio of 1.52 in the released input scripts. Calibrated by
    # matching the relaxation time to the ≈20 h collagen IV lifetime (EVD).
    make_every: int = 500              # released Nevery (bond reaction interval)
    # per-attempt formation probability. Released MP = 0.12 / 0.18 / 0.27; we use
    # the low value 0.12.
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
    temperature: float = 1.0           # kT energy unit (released fix langevin 1.0)
    langevin_damp: float = 0.1         # released Langevin damp (fix langevin ... 0.1)
    timestep: float = 0.01             # released input scripts use 0.01 (Cell Methods 0.001,
                                       # supplement 0.002 — v1-decision open question)

    # ---- box / population (quasi-2D slab; spec: rods lie in-plane) --------
    # Density retuned for the corrected 3σ rod: ~250 rods in a 15σ box percolates
    # (~1 crosslink/rod). (CROSS-CHECK CORRECTION 2026-10-01.)
    n_rods: int = 250                  # percolating default for the 3σ-rod geometry
    box_xy: float = 15.0               # in units of σ
    slab_thickness: float = 2.0        # thin z; quasi-2D
    seed: int = 12345

    # ---- stage-1 protocol (spec: stretch, hold, then chemistry on) --------
    assemble_steps: int = 8000         # form the initial crosslinked network
    target_strain: float = 0.15        # equibiaxial in-plane strain to reach
    strain_rate: float = 1.5e-3        # ε̇ per LJ time (stage 2 sweeps this)
    hold_steps: int = 16000            # relaxation window (spec asks >= 7 tau)

    # ---- unit conversions (energy scale still open, spec v1-decision #4) ---
    kT_per_a3_Pa: float = 2.2          # 1 LJ stress unit = kT/σ³ ≈ 2.2 Pa at σ ≈ 125 nm

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
