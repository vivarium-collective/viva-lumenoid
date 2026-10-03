"""Simularium adapters for the collagen IV network.

Turns a run's bead positions into a ``.simularium`` trajectory so any study
output can be browsed in the Simularium viewer — the workbench's Analysis tab
auto-surfaces the "Simularium Viewer" tool for every study that has a
``.simularium`` file under its dir.

This is the collagen model's end of the generic viva seam (viva-simularium):
each timestep becomes a list of point agents ``{type, x, y, z, radius}``, with
the NC1 and 7S beads as the two agent types, converted to nm via the bead
spacing. ``write_simularium`` (viva-simularium + simulariumio) does the
encoding. A ``molecule_positions`` output on CollagenNetworkProcess emits the
same shape for the live SimulariumAnalysis path.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .params import CollagenParams

# NC1 (type 1) and 7S (type 2) beads, coloured to match the diagnostic figures.
DISPLAY = {
    "NC1": {"color": "#4C78A8"},
    "7S": {"color": "#F58518"},
}


def agents_from_positions(positions, atom_types, nm: float, radius_nm: float) -> list[dict]:
    """One timestep's point agents ({type, x, y, z, radius}) in nm."""
    pos = np.asarray(positions, dtype=float)
    types = np.asarray(atom_types)
    n = min(len(pos), len(types))
    out = []
    for i in range(n):
        p = pos[i]
        out.append({
            "type": "NC1" if int(types[i]) == 1 else "7S",
            "x": float(p[0]) * nm, "y": float(p[1]) * nm,
            "z": float(p[2]) * nm, "radius": radius_nm,
        })
    return out


def _radius_nm(params: CollagenParams) -> float:
    # a fibril radius, floored for visibility at the box's nm scale.
    return max(0.12 * params.bead_spacing_nm, params.fibril_diameter_nm)


def write_trajectory(frames: list[dict], params: CollagenParams, output_path: str,
                     title: str = "") -> str:
    """Write a ``.simularium`` from captured frames (positions + box + t).

    ``frames`` is the list produced by run_stage1(capture_frames=True) / the
    movie clips: each has ``positions``, ``atom_types``, ``box``, ``t``.
    """
    from viva_simularium import write_simularium
    if not frames:
        raise ValueError("no frames to write")
    nm = params.bead_spacing_nm
    radius = _radius_nm(params)
    agent_frames = [agents_from_positions(f["positions"], f["atom_types"], nm, radius)
                    for f in frames]
    times = [float(f.get("t", i)) for i, f in enumerate(frames)]
    box = [float(b) * nm for b in frames[-1]["box"]]
    # time is in LJ (reduced) units; simulariumio's time_unit must be a pint unit,
    # so we leave it at the writer's default and note the LJ scale in the title.
    out = write_simularium(
        times, agent_frames, box, output_path,
        display={k: {"color": v["color"], "radius": radius} for k, v in DISPLAY.items()},
        spatial_unit="nm", title=(title + " (time in LJ units)").strip(), fmt="binary",
        default_radius=radius)
    return str(out)


def write_snapshot(snapshot: dict, params: CollagenParams, output_path: str,
                   title: str = "") -> str:
    """Write a single-frame ``.simularium`` from a final-network snapshot."""
    frame = {"positions": snapshot["positions"], "atom_types": snapshot["atom_types"],
             "box": snapshot["box"], "t": 0.0}
    return write_trajectory([frame], params, output_path, title=title)


def molecule_positions(positions, atom_types, box, nm: float,
                       radius_nm: float = 15.0) -> list[dict]:
    """The ``molecule_positions`` emit shape (point agents in nm) for the live
    SimulariumAnalysis path — same contract viva_smoldyn emits."""
    return agents_from_positions(positions, atom_types, nm, radius_nm)
