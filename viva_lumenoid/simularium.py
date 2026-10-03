"""Simularium adapters for the collagen IV network.

Builds a ``.simularium`` trajectory that shows the actual basement-membrane
**mesh**, not a bag of points: each collagen IV protomer is drawn as a *rod
fiber* (its NC1 end → 7S end), each crosslink as a *fiber* joining two rod ends,
and small spheres mark the NC1 (binds 1) and 7S (binds 3) ends so the two
chemistries stay colour-coded. This uses Simularium's fiber + sphere agent types
together, with an explicit camera framed on the network (simulariumio's default
camera is tuned for a ~100-unit box and sits inside our ~1600 nm box, which is
what made earlier point-only files render as a few huge foreground balls).

Topology comes from the run's bond list (``bonds`` = ``[type, id1, id2]``; type
1 = intra-rod, 2 = NC1–NC1, 3 = 7S–7S) mapped back to bead positions via the
per-frame ``atom_ids``. Positions are converted LJ σ → nm and recentred on the
network centroid so the mesh sits at the origin where the viewer expects it.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .params import CollagenParams

# NC1 / 7S beads coloured to match the diagnostic figures; rods a recessive
# slate; crosslinks a bright green so the junctions (the network's load path)
# read clearly.
DISPLAY = {
    "collagen rod": {"color": "#8893A5", "display": "FIBER"},
    "crosslink":    {"color": "#4FA65B", "display": "FIBER"},
    "NC1":          {"color": "#4C78A8", "display": "SPHERE"},
    "7S":           {"color": "#F58518", "display": "SPHERE"},
}

# viz_type codes (simulariumio.constants.VIZ_TYPE)
_DEFAULT, _FIBER = 1000.0, 1001.0


def _radii_nm(params: CollagenParams) -> dict[str, float]:
    """Mark/fiber sizes in nm, scaled to the rod length so the mesh reads at any
    box size: thin rods, slightly thicker crosslinks, small end nodes (7S the
    larger hub, since it binds 3)."""
    rod_len = max(params.bond_length * params.bead_spacing_nm, 1.0)
    return {
        "collagen rod": max(3.0, 0.03 * rod_len),
        "crosslink":    max(4.0, 0.04 * rod_len),
        "NC1":          max(5.0, 0.06 * rod_len),
        "7S":           max(6.0, 0.075 * rod_len),
    }


def _frame_agents(frame: dict, nm: float, radii: dict[str, float],
                  n_rods: int) -> list[dict]:
    """One timestep's agents: rod + crosslink fibers (from bonds) and NC1/7S
    end-node spheres. Coordinates are in nm (not yet recentred)."""
    pos = np.asarray(frame["positions"], dtype=float)[:, :3] * nm
    types = np.asarray(frame["atom_types"]).astype(int)
    n = len(pos)
    ids = frame.get("atom_ids")
    if ids is not None and len(ids) == n:
        idmap = {int(a): i for i, a in enumerate(np.asarray(ids))}
    else:  # assume atoms are in id order (1-indexed)
        idmap = {i + 1: i for i in range(n)}

    agents: list[dict] = []

    def fiber(kind: str, i1: int, i2: int) -> None:
        p1, p2 = pos[i1], pos[i2]
        agents.append({
            "type": kind, "viz": _FIBER, "pos": (0.0, 0.0, 0.0),
            "radius": radii[kind],
            "sub": [float(p1[0]), float(p1[1]), float(p1[2]),
                    float(p2[0]), float(p2[1]), float(p2[2])],
        })

    bonds = frame.get("bonds") or []
    if bonds:
        for b in bonds:
            bt, a1, a2 = int(b[0]), int(b[1]), int(b[2])
            i1, i2 = idmap.get(a1), idmap.get(a2)
            if i1 is None or i2 is None:
                continue
            fiber("collagen rod" if bt == 1 else "crosslink", i1, i2)
    else:
        # No bond list (e.g. a bare snapshot): draw the rod backbones only, by the
        # build-order pairing (rod i = beads id 2i+1, 2i+2).
        for i in range(n_rods):
            i1, i2 = idmap.get(2 * i + 1), idmap.get(2 * i + 2)
            if i1 is not None and i2 is not None:
                fiber("collagen rod", i1, i2)

    for i in range(n):
        kind = "NC1" if types[i] == 1 else "7S"
        agents.append({
            "type": kind, "viz": _DEFAULT,
            "pos": (float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])),
            "radius": radii[kind], "sub": [],
        })
    return agents


def _build_network_trajectory(frames: list[dict], params: CollagenParams,
                              title: str):
    """Assemble a simulariumio TrajectoryData (fibers + spheres + camera)."""
    from simulariumio import (
        AgentData, CameraData, DisplayData, MetaData, TrajectoryData, UnitData,
    )
    from simulariumio.constants import DISPLAY_TYPE

    nm = params.bead_spacing_nm
    radii = _radii_nm(params)
    per_frame = [_frame_agents(f, nm, radii, params.n_rods) for f in frames]
    T = len(per_frame)
    width = max((len(a) for a in per_frame), default=1)

    # Recentre on the network centroid (mean node position over all frames) so the
    # mesh sits at the origin — the viewer's box is origin-centred.
    node_pts = np.array([a["pos"] for fr in per_frame for a in fr
                         if a["viz"] == _DEFAULT], dtype=float)
    centroid = node_pts.mean(axis=0) if len(node_pts) else np.zeros(3)

    viz = np.full((T, width), _DEFAULT)
    uids = np.zeros((T, width))
    posm = np.zeros((T, width, 3))
    rad = np.zeros((T, width))
    nsub = np.zeros((T, width))
    sub = np.zeros((T, width, 6))
    types_ll: list[list[str]] = []
    n_agents = np.zeros(T)

    for t, agents in enumerate(per_frame):
        n_agents[t] = len(agents)
        row: list[str] = []
        for i, a in enumerate(agents):
            viz[t, i] = a["viz"]
            uids[t, i] = i
            rad[t, i] = a["radius"]
            row.append(a["type"])
            if a["sub"]:
                s = np.asarray(a["sub"], dtype=float)
                s[0::3] -= centroid[0]
                s[1::3] -= centroid[1]
                s[2::3] -= centroid[2]
                nsub[t, i] = len(s)
                sub[t, i, :len(s)] = s
            else:
                posm[t, i] = np.asarray(a["pos"], dtype=float) - centroid
        types_ll.append(row)

    # Camera: frame the whole recentred cloud (node bbox) head-on down +z.
    if len(node_pts):
        rc = node_pts - centroid
        span = np.maximum(rc.max(axis=0) - rc.min(axis=0), 1.0)
        box = [float(max(span[0], span[1]) * 1.1)] * 2 + [float(max(span[2], 1.0) * 2)]
        dist = (max(span[0], span[1]) / 2) / math.tan(math.radians(75 / 2)) * 1.3
    else:
        box, dist = [1000.0, 1000.0, 200.0], 1500.0
    camera = CameraData(
        position=np.array([0.0, 0.0, float(dist)]),
        look_at_position=np.array([0.0, 0.0, 0.0]),
        up_vector=np.array([0.0, 1.0, 0.0]), fov_degrees=75.0)

    display_data = {
        name: DisplayData(
            name=name,
            display_type=(DISPLAY_TYPE.FIBER if spec["display"] == "FIBER"
                          else DISPLAY_TYPE.SPHERE),
            radius=radii[name], color=spec["color"])
        for name, spec in DISPLAY.items()
    }

    times = np.array([float(f.get("t", i)) for i, f in enumerate(frames)])
    agent_data = AgentData(
        times=times, n_agents=n_agents, viz_types=viz, unique_ids=uids,
        types=types_ll, positions=posm, radii=rad,
        n_subpoints=nsub, subpoints=sub, display_data=display_data)
    return TrajectoryData(
        meta_data=MetaData(
            box_size=np.array(box), camera_defaults=camera,
            scale_factor=1.0,
            trajectory_title=(title + " (time in LJ units)").strip()),
        agent_data=agent_data,
        time_units=UnitData("s"), spatial_units=UnitData("nm"))


def write_trajectory(frames: list[dict], params: CollagenParams, output_path: str,
                     title: str = "") -> str:
    """Write a network ``.simularium`` from captured frames.

    ``frames`` is the list produced by ``capture_clip`` / ``run_stage1`` (each has
    ``positions``, ``atom_types``, ``box``, ``t``, and — for crosslinks — ``bonds``
    + ``atom_ids``). Written as JSON so the workbench's inline preview can read it;
    the full Allen viewer (where it opens by default) handles JSON and binary.
    """
    from pathlib import Path

    from simulariumio import JsonWriter

    if not frames:
        raise ValueError("no frames to write")
    traj = _build_network_trajectory(frames, params, title)
    stem = Path(output_path)
    if stem.suffix == ".simularium":
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    JsonWriter.save(traj, str(stem), validate_ids=False)
    return str(stem.with_suffix(".simularium"))


def write_snapshot(snapshot: dict, params: CollagenParams, output_path: str,
                   title: str = "") -> str:
    """Write a single-frame network ``.simularium`` from a final-network snapshot.

    A bare snapshot has no bond list, so only the rod backbones are drawn (by
    build-order pairing) — pass a frame with ``bonds`` to also show crosslinks.
    """
    frame = {"positions": snapshot["positions"], "atom_types": snapshot["atom_types"],
             "box": snapshot["box"], "t": 0.0,
             "atom_ids": snapshot.get("atom_ids"), "bonds": snapshot.get("bonds")}
    return write_trajectory([frame], params, output_path, title=title)


# -- kept for back-compat (point-agent emit shape) ------------------------- #
def agents_from_positions(positions, atom_types, nm: float, radius_nm: float) -> list[dict]:
    """One timestep's NC1/7S point agents ({type, x, y, z, radius}) in nm."""
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


def molecule_positions(positions, atom_types, box, nm: float,
                       radius_nm: float = 15.0) -> list[dict]:
    """The ``molecule_positions`` emit shape (point agents in nm) for a live
    SimulariumAnalysis path — same contract viva_smoldyn emits."""
    return agents_from_positions(positions, atom_types, nm, radius_nm)
