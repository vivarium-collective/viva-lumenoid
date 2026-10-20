# Figure conventions (basement-membrane-v1)

Adopted from the 2026-10-20 scientific figure review. Every study figure follows
these so the report leads with honest, quantitative paper reproduction and never
equates a smoke test, a deterministic transfer equation, or a geometry screenshot
with biological reproduction.

## 1. Evidence status — required on every main figure

Record `evidence_status:` in the figure's `visualizations[].config` (and state it
in the title/caption). One of:

| status | meaning | examples |
|---|---|---|
| `observed` | a real measured simulation result, honestly scoped | R2 dense τ-vs-residence |
| `reproduced` | quantitatively matches a specific paper figure, shown with an overlay/comparison + deviation | R1 regime map vs Fig 1G (once overlaid) |
| `engineered-diagnostic` | a verification / smoke / control / unit-test figure — NOT biological evidence | E1 LJ patch, E2 deterministic transfer, geometry thumbnails, parameter ladders |
| `hypothesis` | a prospective / illustrative figure for a result not yet produced | three-arm schematics before the data exist |
| `contradicted` | the result disagrees with the paper; shown as a flagged departure, not a pass | R6 perlecan ordering |

Never label an `engineered-diagnostic` or `hypothesis` figure as reproduction.

## 2. Main vs supplement

Tag each figure `tier: main` or `tier: supplement`. Keep **1–2 question-driven
main figures per study**; demote to supplement: movies/animations (retain ONE
optional online animation at most), parameter/`bending_k` ladders, provenance
"cross-check" panels, near-duplicate diagnostics, and standalone geometry/volume
histograms. A mesh or geometry screenshot is `engineered-diagnostic`, `supplement`.

## 3. Captions are scientific statements

A caption states what the figure shows, its evidence status, the comparison, and
the scope/caveat — not the figure's name. Include:
- **Axes + units** explicitly (e.g. "network bond-virial in-plane stress, reduced LJ units", not "stress"). Name the stress component (total vs network vs hydrostatic virial).
- **Uncertainty**: seed ensembles with error bars / bands where the model is stochastic; say "single deterministic run" where it is not.
- **Fit windows**: when a slope/τ/modulus is fit, state the interval it is fit over.
- **Reference overlays**: when comparing to the paper, overlay or cite the specific figure + panel and report deviation; do not infer reproduction from monotonicity alone.
- **Provenance**: released-code LAMMPS vs reduced-order analytic model vs engineered proxy.

## 4. Units & calibration

Reduced (LJ) units are stated as such; any kT/σ³ → Pa or LJ-time → hours
conversion carries the unresolved-energy-scale caveat. Do not present a reduced-unit
number as a physical one.

## 5. Narrative order

Lead with the quantitative paper reproductions (R1 Fig 1, R2/R3 Fig 2, R6 Fig 7),
then the spatial Tyssue studies (R4/R5), then the actin/coupling extensions
(E1–E4). Extensions are explicitly framed as extensions, not paper reproductions.
