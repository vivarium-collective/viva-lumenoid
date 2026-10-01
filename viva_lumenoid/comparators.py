"""Measured literature comparators for the collagen IV basement-membrane v1 model.

Single source of truth for every experimental number the model is held against.
Each value is transcribed from the AICS-Lumenoids "Basement membrane — v1"
specification (Roam export 2026-09-30) and its diagrammatic board — the two
documents in ``workspace/references/``. A readout the simulation produces is
*validated* by landing in one of these bands (or honestly missing it); the
visualizations draw these as reference bands so a chart answers "is the model
right?" and not just "what did it do?".

Nothing here is simulation output — these are the comparators. Keeping them in
one module means the study acceptance bands, the Plotly reference bands, and the
evidence-map figure all cite the SAME numbers with the SAME provenance.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Comparator:
    """One measured quantity the model is compared to."""
    key: str
    label: str              # short human label
    value: float            # central value (or band midpoint)
    lo: float               # lower bound of the plausible/measured band
    hi: float               # upper bound
    units: str
    method: str             # how it was measured
    source: str             # citation key / provenance from the spec
    note: str = ""

    @property
    def half_width(self) -> float:
        return 0.5 * (self.hi - self.lo)


# --------------------------------------------------------------------------- #
# Elastic modulus — the "modulus ladder" (spec: has: geometrical structure,
# "Effective elastic modulus"; v1-decision #4 energy scale).
# Spans 6+ orders: the published CG model is ~0.03 Pa; real BM 10^3–10^5 Pa.
# --------------------------------------------------------------------------- #
MODULUS_PUBLISHED_CG = Comparator(
    "modulus_published_cg", "Meadowcroft CG model (own linear E)",
    0.03, 0.02, 0.05, "Pa", "coarse-grained sim, linear Young's modulus",
    "@meadowcroft2025nonequilibrium supplement (AllenCell §9)",
    "The paper's own network is 10^3–10^4× softer than reconstituted BM; "
    "attributed to missing components, thin-sheet geometry, no pre-stress.")
MODULUS_RECONSTITUTED_BM = Comparator(
    "modulus_reconstituted_bm", "reconstituted BM", 275.0, 50.0, 500.0, "Pa",
    "bulk rheology", "@meadowcroft2025nonequilibrium (comparator)", "")
MODULUS_ACINUS_BM = Comparator(
    "modulus_acinus_bm", "MCF10A acinus BM (AFM)", 1025.0, 529.0, 1521.0, "Pa",
    "AFM nanoindentation, shear modulus 1025 ± 496 Pa",
    "@fabris2018nanoscale", "")
MODULUS_SPHEROID_BM = Comparator(
    "modulus_spheroid_bm", "MDA-MB-231 spheroid BM", 82000.0, 60000.0, 100000.0,
    "Pa", "micro-inflation", "@li2021nonlinear", "")

MODULUS_LADDER = [MODULUS_PUBLISHED_CG, MODULUS_RECONSTITUTED_BM,
                  MODULUS_ACINUS_BM, MODULUS_SPHEROID_BM]

# --------------------------------------------------------------------------- #
# Porosity — pore-size distribution (spec: "Porosity → pore-size distribution").
# --------------------------------------------------------------------------- #
PORE_CORNEAL_EM = Comparator(
    "pore_corneal_em", "corneal BM pores (EM)", 72.0, 32.0, 112.0, "nm",
    "low-voltage SEM, exposed BM face, 72 ± 40 nm",
    "@abrams2000nanoscale", "Fibre diameter on the same face: 77 ± 44 nm.")
PORE_MATRIGEL = Comparator(
    "pore_matrigel", "hydrated Matrigel mesh", 40.0, 20.0, 60.0, "nm",
    "mesh-size estimate, hydrated", "spec (Matrigel comparator)", "")

PORE_COMPARATORS = [PORE_CORNEAL_EM, PORE_MATRIGEL]

# --------------------------------------------------------------------------- #
# Bundling — protomers per laterally-associated strand (spec: "Bundling →
# protomers per strand", a READOUT to reproduce, not an input).
# --------------------------------------------------------------------------- #
BUNDLING_PFHR9 = Comparator(
    "bundling_pfhr9", "PFHR-9 scaffold strand", 6.0, 5.0, 7.0, "protomers",
    "immuno-gold SEM of cell-deposited collagen IV scaffold",
    "@pokidysheva2025targeted",
    "The NC1/7S-only model can bundle ONLY through 7S branching (no lateral-"
    "association bond); failing to reach 5–7 is a spec-named failure mode.")

# --------------------------------------------------------------------------- #
# Protomer geometry — sets the model's length scale σ (spec: "has: geometrical
# structure"; the SVG's PROTOMER GEOMETRY + FLEXIBILITY panels).
# --------------------------------------------------------------------------- #
PROTOMER_CONTOUR = Comparator(
    "protomer_contour", "collagen IV contour length", 360.0, 340.0, 380.0, "nm",
    "single-molecule AFM, 360 ± 20 nm (n=262); EM 355 ± 4.9 nm",
    "@alshaer2021sequence / @timpl1981network", "")
PERSISTENCE_LENGTH = Comparator(
    "persistence_length", "persistence length l_p", 39.0, 37.0, 41.0, "nm",
    "single-molecule AFM, 39 ± 2 nm (n=262)", "@alshaer2021sequence",
    "Less than half the ≈90 nm of fibril-forming collagens I/II/III; the rods "
    "are rigid so flexibility enters as soft angles where rods join.")
FIBRIL_DIAMETER = Comparator(
    "fibril_diameter", "individual fibril diameter", 8.0, 5.0, 11.0, "nm",
    "high-resolution SEM of glomerular BM (5–9 / 6–11 nm)",
    "@shirato1991fine", "")
BEAD_SPACING = Comparator(
    "bead_spacing", "two-bead spacing a (σ)", 114.0, 107.0, 125.0, "nm",
    "supplement mapping a ≈ 114 nm; σ candidate 107–125 nm (±6% convention)",
    "spec @meadowcroft2025nonequilibrium supplement", "")

# --------------------------------------------------------------------------- #
# Remodelling time — the viscous part (spec: "Remodelling time → collagen IV
# lifetime ≈20 h"; also what calibrates MP, so it cannot also validate).
# --------------------------------------------------------------------------- #
REMODELLING_TIME = Comparator(
    "remodelling_time", "collagen IV lifetime", 20.0, 3.0, 30.0, "h",
    "fluorescent-timer reporter, Drosophila wing disc (≈20 h control)",
    "@drosophila-collagen-timer (EVD)",
    "Spreads 3–30 h across systems; FRAP half-times measure mobility not "
    "lifetime. Calibrates MP and the time unit — a calibration target, not a test.")

# --------------------------------------------------------------------------- #
# Connectivity — mean coordination z and the 2D rigidity (Maxwell) threshold.
# --------------------------------------------------------------------------- #
MAXWELL_Z_2D = 4.0  # central-force isostatic threshold in 2D
# NC1 end binds 1 partner, 7S end binds 3 → max z ≈ (1+3)/… ≤ 3 per protomer:
# the released NC1/7S topology is sub-isostatic, so a pure central-force network
# is floppy and needs bending (angles) to carry a modulus.
Z_NC1_7S_MAX = 3.0


def all_comparators() -> list[Comparator]:
    """Every comparator as a flat list (for the evidence-map figure)."""
    return [
        PROTOMER_CONTOUR, PERSISTENCE_LENGTH, FIBRIL_DIAMETER, BEAD_SPACING,
        PORE_CORNEAL_EM, PORE_MATRIGEL, BUNDLING_PFHR9,
        MODULUS_PUBLISHED_CG, MODULUS_RECONSTITUTED_BM, MODULUS_ACINUS_BM,
        MODULUS_SPHEROID_BM, REMODELLING_TIME,
    ]
