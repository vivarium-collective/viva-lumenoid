"""Registered composite generators for the basement-membrane studies.

Each study's ``baseline.composite`` names one of these. They are **thin,
config-driven** selections over a single shared builder
(:func:`viva_lumenoid.composites.builders.collagen_network_spec`), which wires the
two reusable workspace processes — ``StagedStretchController`` →
``CollagenNetworkProcess`` → emitter. The composites differ ONLY by
``CollagenParams`` and a protocol / emit choice; none duplicate the wiring, so a
new collagen composite is one more thin generator here, not a copy of the graph.
``core_extensions=[register_viva_lumenoid]`` builds each against a core that knows
the two processes.
"""
from __future__ import annotations

from viva_superpowers.composite_generator import composite_generator

from ..params import CollagenParams
from ..collagen import register_viva_lumenoid
from .builders import collagen_network_spec


def _params(**kw) -> CollagenParams:
    """CollagenParams from the generator's Configure kwargs (unknown keys dropped)."""
    allowed = set(CollagenParams().to_dict())
    return CollagenParams(**{k: v for k, v in kw.items() if k in allowed})


# Configurable params exposed in the workbench "Configure" panel. Shared across
# the collagen composites; bending_sweep overrides the bending_k default below.
_COMMON_PARAMS = {
    "n_rods": {"type": "integer", "default": 200,
               "description": "Number of two-bead collagen IV rods."},
    "box_xy": {"type": "float", "default": 20.0,
               "description": "In-plane box size (units of the bead spacing a)."},
    "make_prob": {"type": "float", "default": 0.12,
                  "description": "Per-attempt crosslink formation probability (MP proxy)."},
    "off_rate": {"type": "float", "default": 0.02,
                 "description": "Force-independent crosslink off-rate (per LJ time)."},
    "bending_k": {"type": "float", "default": 0.0,
                  "description": "Junction bending stiffness (0 = floppy central-force network)."},
    "seed": {"type": "integer", "default": 12345, "description": "RNG seed."},
}

_BENDING_PARAMS = dict(_COMMON_PARAMS)
_BENDING_PARAMS["bending_k"] = {
    "type": "float", "default": 50.0,
    "description": "Junction bending stiffness (fix restrain, as-formed angles); >0 rigidifies.",
}


@composite_generator(
    name="stage1_composite",
    description="Stage 1 — staged protocol (assemble → relax → stretch FROZEN → "
                "hold elastic → hold viscous). Separates a reproducible elastic "
                "modulus from a crosslink-turnover remodelling time.",
    parameters=_COMMON_PARAMS,
    core_extensions=[register_viva_lumenoid],
)
def stage1_composite(core=None, **kw) -> dict:
    # Staged stage-1: the network is loaded FROZEN (chemistry off during stretch).
    return collagen_network_spec(_params(**kw), emit="rheology",
                                 chemistry_during_stretch=False)


@composite_generator(
    name="stage2_sweep",
    description="Stage 2 / R2 — growth rheology: the collagen IV network on a "
                "growing (biaxially stretching) substrate with the crosslink "
                "chemistry ON during the stretch; the σ(ε̇) sweep runs this at a "
                "range of strain rates.",
    parameters=_COMMON_PARAMS,
    core_extensions=[register_viva_lumenoid],
)
def stage2_sweep(core=None, **kw) -> dict:
    # Growth rheology: remodel WHILE strained (chemistry on during stretch).
    return collagen_network_spec(_params(**kw), emit="rheology",
                                 chemistry_during_stretch=True)


@composite_generator(
    name="connectivity_sweep",
    description="Rigidity (bm-v3, adversarial) — the CENTRAL-FORCE counterfactual: "
                "bending_k forced to 0, crosslink connectivity (z) swept to test "
                "rigidity percolation. A control that should NOT reproduce the "
                "released angle-stabilized model.",
    parameters=_COMMON_PARAMS,
    core_extensions=[register_viva_lumenoid],
)
def connectivity_sweep(core=None, **kw) -> dict:
    # Central-force control: force bending off regardless of the Configure value.
    return collagen_network_spec(_params(**{**kw, "bending_k": 0.0}), emit="rheology")


@composite_generator(
    name="bending_sweep",
    description="FP1 (bm-v4) — the collagen IV network WITH junction bending "
                "(released NC1 value, as-formed angle restraints created during "
                "assembly). The athermal modulus is swept vs bending_k.",
    parameters=_BENDING_PARAMS,
    core_extensions=[register_viva_lumenoid],
)
def bending_sweep(core=None, **kw) -> dict:
    # Bending active (bending_k>0) with faithful as-formed angles.
    return collagen_network_spec(_params(**{**kw, "use_real_angles": True}),
                                 emit="rheology")


@composite_generator(
    name="porosity_bundling",
    description="Stage 5 (bm-v5) — the assembled collagen IV network read for its "
                "geometric readouts (pore-size distribution, protomers-per-strand) "
                "against the measured BM comparators; emits the assembly "
                "observables, not the stress trace.",
    parameters=_COMMON_PARAMS,
    core_extensions=[register_viva_lumenoid],
)
def porosity_bundling(core=None, **kw) -> dict:
    # Geometric readout of the assembled network.
    return collagen_network_spec(_params(**kw), emit="geometry")
