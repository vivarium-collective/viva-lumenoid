# Cross-check: clean-room viva-lumenoid vs the authors' released code

**Date:** 2026-10-01 · **Released code:** [github.com/Billie1717/BasementMembraneTurnoverSims](https://github.com/Billie1717/BasementMembraneTurnoverSims),
Zenodo [10.5281/zenodo.20719515](https://doi.org/10.5281/zenodo.20719515), cloned at commit `ab4c004`
(bibliography `barrientos2026code`). Source read: each stage's `parameters.md` and
`build_*_annotated.py` (the annotated LAMMPS input writers).

This is the verification flagged as a decision when the code became public. It was
prompted by a direct question — *do we trust the clean-room build?* — and the
answer is: **partially.** The high-level structure matches; several quantitative
mechanics parameters do not, and one divergence overturns a conclusion.

## Headline finding

**The released model is bending-stabilized, not a central-force network.** Its
force field includes harmonic *angle* terms at the crosslink junctions:

```
angle_coeff 1  4.0   180.0   # NC1 junction angle (KangNC1) — enforces NC1-NC1 linearity
angle_coeff 2  1.0   155.0   # 7S junction angle
angle_coeff 3  1.0    60.0   # 7S orthogonal angle (closed 7S trimers)
```

Our clean-room model defaults `bending_k = 0` and frames junction bending as a
*new addition* (bm-v4 / FP1), on the premise (bm-v3) that the released model is a
central-force network that is floppy because z ≤ 3 < 4. **That premise is wrong.**
The released model has angular rigidity by construction, so it is not expected to
be floppy, and bm-v1's "modulus ≈ 0" is an artifact of our omitting the angles —
not a property of the published model. bm-v3's rigidity-percolation argument and
bm-v4's "bending is the missing ingredient" are reframed: bending was never
missing from the paper; it was missing from our transcription.

## Parameter comparison

| Quantity | Released code | Clean-room (CollagenParams) | Match |
|---|---|---|---|
| Integration timestep | 0.01 | 0.01 | ✓ |
| Temperature (kT) | 1.0 | 1.0 | ✓ |
| Langevin damp | **0.1** | **1.0** | ✗ (10×; affects dynamics/diffusion) |
| Pair interactions | WCA ε=0 (no excluded volume) | ε=0 (`excluded_volume=False`) | ✓ |
| Rod backbone bond k, r0 | 200.0, **3.0** | 200.0, **1.0** | k ✓, **r0 ✗** |
| NC1 crosslink bond k, r0 | **6.0, 0.5** | **100.0, 0.30** | ✗ |
| 7S crosslink bond k, r0 | **6.0, 0.5** | **100.0, 0.30** | ✗ |
| NC1 junction angle | **k=4.0, θ₀=180°** | **absent (`bending_k=0`)** | **✗ major** |
| 7S junction angles | **k=1.0 @155°, k=1.0 @60°** | **absent** | **✗ major** |
| NC1 max bonds | 1 (pairwise) | 1 | ✓ |
| 7S max bonds | 3 | 3 | ✓ |
| Make probability MP | 0.12 / 0.18 / 0.27 | 0.12 | ✓ (low value) |
| Break/make ratio FactorMult | 0.66 | 0.66 | ✓ |
| Nevery (bond-react attempts) | **500** (assembly), 5000 (stretch) | **100** (`make_every`) | ✗ |
| Bond-formation cutoff | **0.65** (NC1 & 7S) | **0.35** (`crosslink_cutoff`) | ✗ |
| Bond-break cutoff | **1.35** (NC1), **0.95** (7S) | **1.20** (single) | ✗ |
| Break mechanism | `bond/react`, distance-gated, P=FactorMult·MP | random-fraction deletion at `off_rate=0.02` | ✗ (different model) |
| Monomer exchange | **GCE** insert/delete (κ_ρ=0.01, N_preferred) | **none** (fixed `n_rods`) | ✗ |
| Length unit σ | **≈125 nm** (protomer 376 nm = 3σ) | 114 nm, but rod = **1σ** | ✗ geometry |
| Stretch protocol | 20 discrete biaxial steps, ΔL=50 total, Nevery=5000 | continuous equibiaxial to ε=0.15 | ✗ |

Notes:
- The crosslink rest length r0 = 0.5σ ≈ 62 nm is close to the measured 59.5 nm
  cell-deposited protomer spacing — a point in the released model's favour.
- Because the rod is 3σ in the released model but 1σ in ours, the rod length
  relative to crosslink spacing and box differs threefold; bm-v5's porosity
  numbers in nm are not a like-for-like comparison.
- The break mechanism differs in kind: the paper breaks a bond stochastically
  once it is stretched past the BD cutoff (distance-gated); we delete a random
  fraction per interval at a fixed rate. Both give turnover, but the remodelling
  physics is not the same.

## What matches (the build is not worthless)

The topology and the dimensionless regime are faithful: two-bead rods; NC1 binds
1, 7S binds 3; no excluded volume (ε=0); kT=1, dt=0.01; FactorMult=0.66; MP in the
published range; rod-backbone stiffness k=200. The investigation's *qualitative*
infrastructure (LAMMPS-backed network, staged protocol, readouts) is sound.

## Per-study impact

- **bm-v1 (modulus & remodelling):** the "modulus ≈ 0 / floppy" result is an
  artifact of the missing angles and the 1σ (vs 3σ) rod, not the paper. Re-run
  with the released angle set + geometry before trusting any modulus.
- **bm-v3 (rigidity percolation):** the premise (central-force network) is false;
  the released model is angle-stabilized. The z-vs-rigidity argument still holds
  as general physics but does not describe the published model.
- **bm-v4 (junction bending as FP1):** bending is not a new ingredient — the paper
  has KangNC1=4.0 and two 7S angles. Re-scope as "match the released angle set",
  not "add bending".
- **bm-v2 (σ(ε̇)):** affected by the angle, bond-stiffness, break-mechanism and
  stretch-protocol divergences; treat as exploratory until re-run.
- **bm-v5 (porosity & bundling):** pore sizes in nm use the wrong length mapping
  (1σ rod); bundling is also affected by the absent 7S angles that organise trimers.

## Recommended next step

**DONE 2026-10-01** — `CollagenParams` corrected to the released scalar values
(bond k/r0, rod r0=3.0, cutoffs, damp=0.1, Nevery=500, σ≈125 nm) and all studies
re-run. Outcome: a very-soft modulus consistent with the published ≈0.03 Pa; the
central-force / bending-as-new explanations retired.

## Engine note (corrected 2026-10-02)

Both the released code and viva-lumenoid run on **LAMMPS** — this is not an engine
difference. The released code drives it with `fix bond/react` (the REACTION
package) + molecule templates, which create the junction angles atomically when a
bond forms, plus a custom-patched GCE (Nucleation/Death) for monomer exchange.
viva-lumenoid drives the *same* engine with `fix bond/create` + a Python-side
off-rate, which does not create angles. **Our LAMMPS build already has
`fix bond/react` available**, so the remaining fidelity gap is not the engine but:

  1. **Junction angles active during assembly** — **CLOSED (2026-10-02).** Real
     `angle_style harmonic` angles are now created incrementally as NC1 crosslinks
     form (`create_bonds single/angle`, NC1 180° k=4.0; `CollagenParams.
     use_real_angles`), so the network equilibrates compatible with them rather
     than having them pinned on afterwards. This fixes the ill-conditioning: the
     modulus becomes **reproducible and positive** across seeds (e.g. per-seed
     36.7 / 27.6 / 37.5 LJ) instead of the sign-flipping as-formed-restraint result
     (0.9 / −2.5 / −0.5). So the released model, run faithfully, is **not floppy** —
     it has a well-defined finite modulus.
  2. **Per-angle deletion** — **CLOSED (2026-10-02).** Breaking a crosslink now
     re-syncs the angles (delete-all + recreate, `_rebuild_real_angles`), so the
     stiff faithful network **remodels** in the viscous hold. This gives a
     reproducible remodelling time (τ ≈ 40 τ) alongside the modulus — the
     elastic/viscous separation the investigation set out to do (both were
     noise-limited before).
  2b. **Crosslink density** — **CLOSED (2026-10-02).** The authors' dense initial
     config (`Assembly/input/dataNucType9_dense`) is 3126 protomers in a
     32.27×32.27×12.0 box → **areal density 3.0 rods/σ², a 12σ-thick slab**
     (several layers), V=12500. Matching that density + thickness in our builder,
     **make+break and make-only AGREE** (≈2.6 crosslinks/rod) and the modulus is
     reproducible (E ≈ 3 LJ ≈ 6 Pa). The earlier sparsity and the 79-vs-7 Pa
     spread were artifacts of a too-thin (2σ), too-sparse geometry. The remaining
     refinements are the GCE-maintained monomer pool and the distance-gated break
     mechanism (for exact turnover), not the gross density.
  2c. **Absolute Pa / energy scale** — open by the spec's own decision #4: the
     modulus is reported in reduced units (E ≈ 3 LJ); ~6 Pa uses kT/σ³ ≈ 2.2 Pa.
     The published ≈0.03 Pa figure assumes the authors' own energy scale (s_E),
     so an absolute Pa comparison is only meaningful once that is fixed.
  3. **The custom GCE** (Nucleation/Death with the density penalty) — still a gap;
     it needs the authors' patched `fix_bond_react`.
  4. **The 7S-specific break cutoff** (0.95), the 7S angles (155°/60°), and the
     dense initial configuration.

Smoldyn is not used by the released model; it appears only in the AICS spec
(decision #2) as a *future* engine option for stage 3 (surface binders / flexible
fibers).
