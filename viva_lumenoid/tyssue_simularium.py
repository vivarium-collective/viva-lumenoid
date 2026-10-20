"""Simularium trajectories for the Tyssue epithelial composites.

Captures the epithelium's vertex positions over a short run and writes a
``.simularium`` (vertices as spheres) so the workbench's Simularium Viewer can
play the tissue relaxing/growing. Companion to ``viva_lumenoid.simularium`` (the
collagen-network adapter). Vertices are the spatial agents; the mesh moves, the
agent count is constant.
"""
from __future__ import annotations

import pathlib

import numpy as np


def tyssue_frames(composite_id: str, steps: int = 20, dt: float = 0.02) -> list[dict]:
    """Build the Tyssue composite, step it, capture vertex positions per step."""
    import viva_lumenoid.composites  # noqa: F401  (registers generators)
    from process_bigraph.composite_generator import (
        _REGISTRY, apply_core_extensions, build_generator,
    )
    from bigraph_schema import allocate_core
    from process_bigraph import Composite

    entry = _REGISTRY[composite_id]
    core = apply_core_extensions(entry, allocate_core())
    doc = build_generator(entry, None, core=core)
    state = doc["state"] if isinstance(doc, dict) and "state" in doc else doc
    comp = Composite({"state": state}, core=core)
    node = comp.state["Tyssue"]["instance"]
    eptm = node.eptm
    coords = list(eptm.coords)

    frames = []
    for i in range(steps):
        comp.update({}, dt)
        pos = eptm.vert_df[coords].to_numpy(dtype=float)
        if pos.shape[1] == 2:
            pos = np.column_stack([pos, np.zeros(len(pos))])
        frames.append({"t": i * dt, "positions": pos})
    return frames


def write_tyssue_trajectory(frames: list[dict], output_path: str,
                            title: str = "tyssue epithelium") -> str:
    """Write a ``.simularium`` from captured vertex-position frames (spheres)."""
    from simulariumio import (
        AgentData, CameraData, DisplayData, MetaData, TrajectoryData, UnitData,
        TrajectoryConverter,
    )
    from simulariumio.constants import DISPLAY_TYPE, VIZ_TYPE

    if not frames:
        raise ValueError("no frames to write")
    T = len(frames)
    nv = frames[0]["positions"].shape[0]
    times = np.array([f["t"] for f in frames], dtype=float)

    positions = np.zeros((T, nv, 3), dtype=float)
    for t, f in enumerate(frames):
        positions[t, : f["positions"].shape[0], :] = f["positions"]
    # Center the cloud and scale so the scene fits the viewer's default camera.
    center = positions.reshape(-1, 3).mean(axis=0)
    positions = positions - center
    extent = float(np.abs(positions).max()) or 1.0
    s = 80.0 / extent
    positions = positions * s

    n_agents = np.full(T, nv, dtype=float)
    viz_types = np.full((T, nv), VIZ_TYPE.DEFAULT, dtype=float)
    unique_ids = np.tile(np.arange(nv, dtype=float), (T, 1))
    types = [["vertex"] * nv for _ in range(T)]
    radii = np.full((T, nv), 1.6, dtype=float)

    display = {"vertex": DisplayData(name="vertex", display_type=DISPLAY_TYPE.SPHERE,
                                     radius=1.6, color="#4f7cff")}
    agent = AgentData(times=times, n_agents=n_agents, viz_types=viz_types,
                      unique_ids=unique_ids, types=types, positions=positions,
                      radii=radii, display_data=display)
    box = np.array([2.4 * extent * s] * 3)
    camera = CameraData(position=np.array([0.0, 0.0, 160.0]),
                        look_at_position=np.zeros(3), up_vector=np.array([0.0, 1.0, 0.0]),
                        fov_degrees=55.0)
    traj = TrajectoryData(
        meta_data=MetaData(box_size=box, camera_defaults=camera, trajectory_title=title),
        agent_data=agent, time_units=UnitData("s"), spatial_units=UnitData("um"))

    out = pathlib.Path(output_path)
    stem = out.with_suffix("") if out.suffix == ".simularium" else out
    TrajectoryConverter(traj).save(str(stem))
    return str(stem) + ".simularium"
