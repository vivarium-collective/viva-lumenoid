"""Registered composite generators for the basement-membrane studies.

Each study's ``baseline.composite`` names one of these. They make the collagen IV
network a real, *discoverable* composite (StagedStretchController → the LAMMPS-
backed CollagenNetworkProcess → emitter), so the workbench's Composites/Registry
tabs resolve them and show their processes — instead of the "composite not found
in registry" warning that a bare Python spec-dict produced.

The generators reuse the spec builders in ``viva_lumenoid.stage1`` and register
this workspace's processes via ``core_extensions`` so the composite builds against
a core that knows ``CollagenNetworkProcess`` / ``StagedStretchController``.
"""
from __future__ import annotations

from viva_superpowers.composite_generator import composite_generator

from ..params import CollagenParams
from ..stage1 import stage1_composite_spec
from ..collagen import register_viva_lumenoid


def _params(**kw) -> CollagenParams:
    allowed = set(CollagenParams().to_dict())
    return CollagenParams(**{k: v for k, v in kw.items() if k in allowed})


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


@composite_generator(
    name="stage1_composite",
    description="Stage 1 — collagen IV network staged protocol "
                "(assemble → relax → stretch → hold elastic → hold viscous). "
                "StagedStretchController drives the LAMMPS-backed CollagenNetworkProcess.",
    parameters=_COMMON_PARAMS,
    core_extensions=[register_viva_lumenoid],
)
def stage1_composite(core=None, **kw) -> dict:
    return stage1_composite_spec(_params(**kw))


@composite_generator(
    name="stage2_sweep",
    description="Stage 2 — the collagen IV network on a growing (biaxially "
                "stretching) substrate; the σ(ε̇) sweep runs this composite at a "
                "range of strain rates with the crosslink chemistry on.",
    parameters=_COMMON_PARAMS,
    core_extensions=[register_viva_lumenoid],
)
def stage2_sweep(core=None, **kw) -> dict:
    return stage1_composite_spec(_params(**kw))


@composite_generator(
    name="connectivity_sweep",
    description="Rigidity (bm-v3) — the collagen IV network whose crosslink "
                "connectivity (z) is swept to test rigidity percolation.",
    parameters=_COMMON_PARAMS,
    core_extensions=[register_viva_lumenoid],
)
def connectivity_sweep(core=None, **kw) -> dict:
    return stage1_composite_spec(_params(**kw))


_BENDING_PARAMS = dict(_COMMON_PARAMS)
_BENDING_PARAMS["bending_k"] = {
    "type": "float", "default": 50.0,
    "description": "Junction bending stiffness (fix restrain, as-formed angles); >0 rigidifies.",
}


@composite_generator(
    name="bending_sweep",
    description="FP1 (bm-v4) — the collagen IV network WITH junction bending "
                "(fix restrain pinned to each crosslink's as-formed angle). The "
                "athermal modulus is swept vs bending_k.",
    parameters=_BENDING_PARAMS,
    core_extensions=[register_viva_lumenoid],
)
def bending_sweep(core=None, **kw) -> dict:
    return stage1_composite_spec(_params(**kw))


@composite_generator(
    name="porosity_bundling",
    description="Stage 5 (bm-v5) — the assembled collagen IV network, read for "
                "its geometric readouts: pore-size distribution (porosity) and "
                "protomers-per-strand (bundling), against the measured BM comparators.",
    parameters=_COMMON_PARAMS,
    core_extensions=[register_viva_lumenoid],
)
def porosity_bundling(core=None, **kw) -> dict:
    return stage1_composite_spec(_params(**kw))
