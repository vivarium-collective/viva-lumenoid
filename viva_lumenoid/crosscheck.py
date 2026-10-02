"""The clean-room vs released-code parameter comparison, as data.

Single source for the cross-check figure and any report that cites it. Mirrors
references/cross-check-vs-released-code.md (released code: Zenodo
10.5281/zenodo.20719515, commit ab4c004). Each row is (quantity, released, ours,
status) where status is "match" or "diverge". `ours` reflects the CORRECTED
CollagenParams (2026-10-01).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Row:
    quantity: str
    released: str
    ours: str
    status: str   # "match" | "diverge"
    note: str = ""


ROWS = [
    Row("Rod topology", "two-bead rod", "two-bead rod", "match"),
    Row("Connectivity", "NC1×1, 7S×3", "NC1×1, 7S×3", "match"),
    Row("Excluded volume", "none (ε=0)", "none (ε=0)", "match"),
    Row("Temperature kT", "1.0", "1.0", "match"),
    Row("Timestep", "0.01", "0.01", "match"),
    Row("FactorMult (break/make)", "0.66", "0.66", "match"),
    Row("Make prob MP", "0.12/0.18/0.27", "0.12", "match", "low value"),
    Row("Rod backbone k, r0", "200, 3.0", "200, 3.0", "match", "corrected"),
    Row("Length unit σ", "≈125 nm", "≈125 nm", "match", "corrected"),
    Row("Crosslink bond k, r0", "6.0, 0.5", "6.0, 0.5", "match", "corrected"),
    Row("Form / break cutoff", "0.65 / 1.35", "0.65 / 1.35", "match", "corrected (7S break 0.95 not split)"),
    Row("Langevin damp", "0.1", "0.1", "match", "corrected"),
    Row("Nevery", "500", "500", "match", "corrected"),
    Row("NC1 junction angle", "k=4.0 @180°", "not during assembly", "diverge", "released: angle_style; ours: as-formed restraint"),
    Row("7S junction angles", "k=1.0 @155°, @60°", "absent", "diverge"),
    Row("Break mechanism", "distance-gated bond/react", "random-fraction deletion", "diverge"),
    Row("Monomer exchange (GCE)", "present", "absent", "diverge"),
    Row("Initial configuration", "dense physiological", "lattice (sparser)", "diverge"),
]


def n_match() -> int:
    return sum(1 for r in ROWS if r.status == "match")


def n_diverge() -> int:
    return sum(1 for r in ROWS if r.status == "diverge")
