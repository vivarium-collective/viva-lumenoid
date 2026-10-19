"""Shared, config-driven composite builders for the collagen IV network.

ONE builder — :func:`collagen_network_spec` — wires the two reusable workspace
processes into a process-bigraph composite document::

    StagedStretchController  ->  CollagenNetworkProcess  ->  ram-emitter

Every collagen composite in this workspace (stage 1, stage 2 / R2, the rigidity
and bending controls, the geometric readout) is a THIN selection of
``CollagenParams`` + a protocol / emit choice over this one builder, so the
processes are reused and no composite carries duplicated wiring. Adding a new
collagen composite is one more thin generator in ``__init__.py``, not a new copy
of the graph. New engines (Tyssue vertex model, actomyosin) follow the same
discipline: a small set of general processes, composites that only select +
parameterise them.
"""
from __future__ import annotations

from typing import Any

from ..params import CollagenParams

# The network process's full output port set (the store paths it writes). A
# composite's emitter reads a SUBSET of these (plus `phase` from the controller).
NETWORK_OUTPUTS: dict[str, list[str]] = {
    "sigma_inplane": ["sigma_inplane"], "strain": ["strain"],
    "n_crosslinks": ["n_crosslinks"], "pxx": ["pxx"], "pyy": ["pyy"],
    "volume": ["volume"], "positions": ["positions"],
    "atom_types": ["atom_types"], "box_dimensions": ["box_dimensions"],
}

# Named emit sets — which observables the RAM emitter records. Kept to simple
# scalar/string observables so every composite resolves + runs cheaply; array
# geometry (positions/atom_types/box) is read by the viz/analysis path, not emitted.
EMIT_SETS: dict[str, dict[str, str]] = {
    # stress trace + strain + crosslink count + protocol phase (stages 1 & 2).
    "rheology": {"sigma_inplane": "float", "strain": "float",
                 "n_crosslinks": "integer", "phase": "string"},
    # assembled-network readout (porosity / bundling studies): no stress trace.
    "geometry": {"n_crosslinks": "integer", "strain": "float", "phase": "string"},
}


def collagen_network_spec(
    params: "CollagenParams | None" = None,
    *,
    emit: "str | dict[str, str]" = "rheology",
    chemistry_during_stretch: bool = False,
) -> dict[str, Any]:
    """Build the collagen-network composite document (controller→network→emitter).

    Parameters
    ----------
    params:
        A :class:`CollagenParams` (defaults when ``None``). ALL network + protocol
        knobs live here — ``bending_k``, ``use_real_angles``, ``make_prob``,
        ``off_rate``, density (``n_rods`` / ``box_xy`` / ``slab_thickness``),
        ``strain_rate``, ``assemble_steps``, ``hold_steps``, ``target_strain`` …
        Composites differ by these values, never by the wiring.
    emit:
        An :data:`EMIT_SETS` key (``"rheology"`` | ``"geometry"``) or an explicit
        ``{store_path: type}`` mapping of what the emitter records.
    chemistry_during_stretch:
        ``False`` (default) = staged stage-1 protocol, the network is loaded
        frozen during the stretch. ``True`` = growth rheology (stage 2 / R2): the
        crosslink chemistry stays on so the network remodels while strained.
    """
    p = params or CollagenParams()
    rate = p.strain_rate
    emit_map = EMIT_SETS[emit] if isinstance(emit, str) else dict(emit)
    emit_inputs = {k: [k] for k in emit_map}
    return {
        "controller": {
            "_type": "process", "address": "local:StagedStretchController",
            "config": {
                "strain_rate": rate,
                "t_assemble": p.assemble_steps * p.timestep,
                "t_relax": p.hold_steps * p.timestep,
                "t_stretch": (p.target_strain / rate) if rate else 0.0,
                "t_hold_elastic": p.hold_steps * p.timestep,
                "chemistry_during_stretch": bool(chemistry_during_stretch),
            },
            "inputs": {},
            "outputs": {"strain_rate": ["strain_rate"], "make_on": ["make_on"],
                        "break_on": ["break_on"], "phase": ["phase"]},
        },
        "network": {
            "_type": "process", "address": "local:CollagenNetworkProcess",
            "config": {"params": p.to_dict()},
            "inputs": {"strain_rate": ["strain_rate"], "make_on": ["make_on"],
                       "break_on": ["break_on"]},
            "outputs": dict(NETWORK_OUTPUTS),
        },
        "emitter": {
            "_type": "step", "address": "local:ram-emitter",
            "config": {"emit": emit_map},
            "inputs": emit_inputs,
            "outputs": {},
        },
    }
