# Processes, composites & engines — modular architecture

_Companion to the investigation upgrade plan (Barrientos et al., Cell 2026 →
multiscale epithelial morphogenesis). Status as of 2026-10-19._

**Principle.** A small set of **general, reusable processes**; **thin composites**
that only *select and parameterise* processes (never duplicate wiring); one
**builder** per engine family. Studies pick a composite + params; composites pick
processes + protocol. Effective (reduced-order) and explicit (molecular) BM are
**interchangeable alternatives**, never additive parallel forces.

---

## 1. viva-lumenoid composites — DONE (this pass)

The five collagen composites were five near-identical copies of one wiring. They
are now **thin, config-driven** selections over a single shared builder:

- `viva_lumenoid/composites/builders.py :: collagen_network_spec(params, *, emit, chemistry_during_stretch)`
  — the one place the `StagedStretchController → CollagenNetworkProcess → emitter`
  graph is wired.
- `viva_lumenoid/composites/__init__.py` — five thin generators that differ ONLY
  by `CollagenParams` + protocol/emit:

  | composite | differs by | role |
  |---|---|---|
  | `stage1_composite` | staged (chemistry off during stretch) | elastic vs viscous separation |
  | `stage2_sweep` | **growth** (chemistry on during stretch) | σ(ε̇) growth rheology (R2) |
  | `connectivity_sweep` | `bending_k = 0` (central force) | rigidity counterfactual (bm-v3, adversarial) |
  | `bending_sweep` | `bending_k > 0`, `use_real_angles` | junction bending (bm-v4) |
  | `porosity_bundling` | `emit="geometry"` | geometric readouts (bm-v5) |

- Process generalization: `StagedStretchController` gained one config knob,
  `chemistry_during_stretch` (default `False` = exact prior behaviour), so the
  **same** controller serves both the staged stage-1 protocol and the growth
  rheology of stage 2 / R2.
- `stage1.stage1_composite_spec` now delegates to the builder (single source of
  wiring). All 42 workspace tests pass; all five composites resolve `ready`.

**Rule going forward:** a new collagen composite = one more thin generator, not a
new copy of the graph.

---

## 2. viva-lammps — PROPOSED generalization (cross-repo)

Today `CollagenNetworkProcess` is collagen-specific (its LAMMPS script +
`build_network_data` + readout live in `viva_lumenoid/collagen.py`). The actomyosin
cortex (E1) is a *second* molecular LAMMPS model; we do not want a second bespoke
process. Generalize the engine wrapper in **viva-lammps** so one process hosts many
models:

- A general `LammpsModelProcess` owning the LAMMPS lifecycle (init → `run N` →
  read observables → close) and the port contract, parameterised by a **model
  plugin**: `(data_builder, script_builder, observable_readers, port_spec)`.
- `CollagenNetworkProcess` becomes the *collagen plugin* over that base;
  `ActomyosinCortexProcess` (E1) becomes an *actin plugin* (filaments / crosslinks
  / motors) over the same base — same init/run/read seam, different model.
- Keeps the real-LAMMPS bridge and the `strain_rate / make_on / break_on` style
  port convention; adds a registry so composites address `local:<plugin>`.

Deliverable: collagen and actin are two plugins of **one** general LAMMPS process,
reused across the R2/R3 (collagen) and E1/E2 (actin + collagen) composites.

---

## 3. vivarium-tyssue + geometry classes — PROPOSED (cross-repo)

The epithelial vertex model is `vivarium-tyssue` (imported as an optional dev
sibling engine). The coupling hierarchy needs **pluggable geometry classes** behind
one `TyssueProcess` (geometry, division, cell-volume, epithelial mechanical energy):

| geometry class | plan level | used by |
|---|---|---|
| `PlanarSheet2D` (+ BM traction/area law; height via a prism relation) | 2D planar | R4 |
| `CrossSection2D` (height-resolving profile, basal adhesion) | 2D cross-section | diagnostic |
| `Monolayer3D` (apical/basal/lateral surfaces + volume constraints) | 3D monolayer | R5 |
| `ClosedLumenoid3D` (closed curved shell + lumen volume/pressure + BM shell) | 3D lumenoid | E3, E4 |

`CrossSection2D` and `ClosedLumenoid3D` are the ones most likely **new** to
vivarium-tyssue; `PlanarSheet2D`/`Monolayer3D` may map onto existing tyssue models
(verify before building). Each geometry exposes the **basal DOFs** the BM couples to.

---

## 4. Coupling contract (LAMMPS ↔ Tyssue) — PROPOSED

Mechanical contract: epithelial **basal geometry/displacements → BM**; BM returns
**reaction forces/traction → basal DOFs**. Validate with prescribed-displacement
and imposed-traction tests before enabling feedback; check dimensions, force &
work balance, mesh-refinement independence and coupling-step convergence. The BM is
either the reduced-order constitutive law (calibrated from R2) **or** the explicit
LAMMPS collagen patch — one or the other, selected per composite, never summed.

---

## 5. Study → composite → process map

| study | composite (id) | processes (existing ✓ / new ◻) |
|---|---|---|
| bm-v1..v5 | `…composites.{stage1_composite,stage2_sweep,connectivity_sweep,bending_sweep,porosity_bundling}` | ✓ CollagenNetworkProcess, ✓ StagedStretchController |
| R1 | `…composites.tissue_prism_baseline` | ◻ reduced-order tissue/BM (effective) |
| R2, R3 | `…composites.stage2_sweep` (reuse) | ✓ collagen network |
| R4 | `…composites.tyssue_2d_sheet` | ◻ TyssueProcess[`PlanarSheet2D`] + effective BM |
| R5 | `…composites.tyssue_3d_monolayer` | ◻ TyssueProcess[`Monolayer3D`] |
| R6 | `…composites.turnover_shape_pipeline` | ◻ inference→shape pipeline over R2/R5 |
| E1 | `…composites.actomyosin_cortex` | ◻ LAMMPS actin plugin |
| E2 | `…composites.actin_adhesion_bm` | ◻ actin + ✓ collagen + ◻ adhesion mapping |
| E3 | `…composites.closed_lumenoid` | ◻ TyssueProcess[`ClosedLumenoid3D`] + BM shell |
| E4 | `…composites.coupled_lumenoid` | ◻ full coupling + controls |

Composite ids for the ◻ rows are **planned** (design targets); they resolve as
unresolved-composite lint gaps until the processes above exist — the honest signal
that a study is designed but not yet implemented.

---

## Open decisions (mirrored into investigation `decisions_needed`)

- Converge all collagen composites on the one general builder — **done**.
- viva-lammps: build the general `LammpsModelProcess` + plugin seam (collagen, actin).
- vivarium-tyssue: which geometry classes are new vs existing; add `CrossSection2D`,
  `ClosedLumenoid3D`.
- Coupling: which vertex-model term receives the BM (basal tension / mobility /
  elastic spring) — spec #5.
