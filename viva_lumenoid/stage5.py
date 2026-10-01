"""Stage 5 — porosity & bundling readouts (spec: "has: geometrical structure").

The spec lists porosity and bundling as the network's two geometric *readouts to
reproduce* — the model should PRODUCE them from crosslink kinetics, and is held
against measured comparators (pores 72 ± 40 nm corneal EM / 20–60 nm Matrigel;
5–7 protomers per strand in the PFHR-9 scaffold). This study assembles the real
collagen IV network, reads both off the snapshot (:mod:`viva_lumenoid.geometry`),
and sweeps network density so the pore-size scaling can be placed against the
comparator bands.

Two findings this exposes, both spec-anticipated:
  * pore size scales with network density — the comparator is matched at a
    particular coverage, not universally;
  * the NC1/7S-only topology UNDER-bundles (no lateral-association bond), so the
    strand-size distribution falls short of 5–7 — a spec-named failure mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .params import CollagenParams
from .stage1 import run_stage1
from . import geometry as geo


@dataclass
class DensityPoint:
    n_rods: int
    coverage: float            # areal occupied fraction (readout of density)
    density_per_a2: float      # rods per a^2
    median_pore_nm: float
    mean_pore_nm: float
    mean_strand: float
    frac_bundled: float
    max_strand: int


@dataclass
class PorosityBundlingResult:
    # canonical (densest sampled) network — full distributions for the histograms
    pore_diam_nm: np.ndarray = field(repr=False, default=None)
    strand_sizes: np.ndarray = field(repr=False, default=None)
    bundling_hist: dict = field(default_factory=dict)
    canonical_n_rods: int = 0
    median_pore_nm: float = 0.0
    mean_pore_nm: float = 0.0
    p90_pore_nm: float = 0.0
    coverage: float = 0.0
    mean_strand: float = 0.0
    max_strand: int = 0
    frac_bundled: float = 0.0
    frac_in_5_7: float = 0.0
    points: list = field(default_factory=list)          # density sweep
    snapshot: dict = field(repr=False, default=None)     # canonical network mesh
    summary: str = ""


def run_porosity_bundling(params: CollagenParams | None = None,
                          densities=(150, 250, 350, 500),
                          assemble_steps: int = 8000) -> PorosityBundlingResult:
    """Assemble the network at several densities; read porosity + bundling."""
    base = params or CollagenParams()
    points: list[DensityPoint] = []
    canonical = None
    canonical_snapshot = None
    for n in densities:
        p = CollagenParams(**{**base.to_dict(), "n_rods": int(n),
                              "assemble_steps": assemble_steps, "hold_steps": 1000})
        r = run_stage1(p)
        snap = r.snapshot
        por = geo.porosity(snap["positions"], snap["box"], p.bead_spacing_nm)
        bun = geo.bundling(snap["positions"], snap["box"], p.bead_spacing_nm)
        area = float(snap["box"][0]) * float(snap["box"][1])
        points.append(DensityPoint(
            n_rods=int(n), coverage=por.occupied_fraction,
            density_per_a2=int(n) / area,
            median_pore_nm=por.median_nm, mean_pore_nm=por.mean_nm,
            mean_strand=bun.mean_size, frac_bundled=bun.frac_bundled,
            max_strand=bun.max_size))
        # keep the densest network as canonical for the full distributions
        if canonical is None or int(n) >= canonical[0]:
            canonical = (int(n), por, bun)
            canonical_snapshot = snap

    n0, por0, bun0 = canonical
    reached = bun0.frac_in_5_7 > 0.05
    summary = (
        f"Pore median {por0.median_nm:.0f} nm (corneal EM 72±40; Matrigel 20–60) · "
        f"bundling mean {bun0.mean_size:.2f}, max {bun0.max_size} protomers/strand "
        f"({'reaches' if reached else 'UNDER-shoots'} the 5–7 comparator — "
        f"{'' if reached else 'no lateral-association bond, spec-named failure'})")
    return PorosityBundlingResult(
        pore_diam_nm=por0.pore_diam_nm, strand_sizes=bun0.strand_sizes,
        bundling_hist=bun0.size_hist, canonical_n_rods=n0,
        median_pore_nm=por0.median_nm, mean_pore_nm=por0.mean_nm,
        p90_pore_nm=por0.p90_nm, coverage=por0.occupied_fraction,
        mean_strand=bun0.mean_size, max_strand=bun0.max_size,
        frac_bundled=bun0.frac_bundled, frac_in_5_7=bun0.frac_in_5_7,
        points=points, snapshot=canonical_snapshot, summary=summary)
