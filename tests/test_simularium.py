"""Simularium converter — turn the collagen network into a .simularium mesh."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("simulariumio", reason="simulariumio required")

from viva_lumenoid import CollagenParams
from viva_lumenoid.simularium import agents_from_positions, write_trajectory


def _net_frame(n_rods, t):
    """A frame with interleaved NC1/7S beads, intra-rod bonds, and one crosslink."""
    n = 2 * n_rods
    rng = np.random.default_rng(t)
    pos = rng.uniform(0, 10, size=(n, 3))
    atom_types = np.tile([1, 2], n_rods)          # NC1, 7S, NC1, 7S, ...
    atom_ids = np.arange(1, n + 1)
    bonds = [[1, 2 * i + 1, 2 * i + 2] for i in range(n_rods)]   # intra-rod
    bonds.append([2, 1, 3])                       # one NC1–NC1 crosslink
    return {"positions": pos, "atom_types": atom_types, "atom_ids": atom_ids,
            "bonds": bonds, "box": [10.0, 10.0, 2.0], "t": float(t)}


def test_agents_from_positions_types_and_nm():
    pos = np.array([[1.0, 2.0, 0.5], [1.5, 2.0, 0.5]])
    types = np.array([1, 2])  # NC1, 7S
    ag = agents_from_positions(pos, types, nm=125.0, radius_nm=15.0)
    assert [a["type"] for a in ag] == ["NC1", "7S"]
    assert ag[0]["x"] == 125.0 and ag[0]["radius"] == 15.0   # converted to nm


def test_write_trajectory_builds_a_network_mesh(tmp_path):
    frames = [_net_frame(10, t) for t in range(5)]
    out = write_trajectory(frames, CollagenParams(), str(tmp_path / "traj"), title="test")
    assert out.endswith(".simularium")
    doc = json.load(open(out))          # JSON so the inline preview can read it
    ti = doc["trajectoryInfo"]
    assert len(doc["spatialData"]["bundleData"]) == 5

    # all four mesh agent types, with fibers as FIBER and ends as SPHERE
    geom = {v["name"]: v["geometry"]["displayType"] for v in ti["typeMapping"].values()}
    assert geom == {"collagen rod": "FIBER", "crosslink": "FIBER",
                    "NC1": "SPHERE", "7S": "SPHERE"}

    # an explicit camera is set (not simulariumio's default-box one)
    assert ti["cameraDefault"]["position"]["z"] > 0

    # frame 0 holds both fibers (viz 1001, with subpoints) and spheres (1000)
    a = doc["spatialData"]["bundleData"][0]["data"]
    i, viz_counts, fiber_nsub = 0, {}, None
    while i < len(a):
        viz = a[i]; nsub = int(a[i + 10])
        viz_counts[viz] = viz_counts.get(viz, 0) + 1
        if viz == 1001.0 and fiber_nsub is None:
            fiber_nsub = nsub
        i += 11 + nsub
    assert viz_counts.get(1001.0, 0) == 11      # 10 rods + 1 crosslink
    assert viz_counts.get(1000.0, 0) == 20      # 10 NC1 + 10 7S end nodes
    assert fiber_nsub == 6                       # two xyz endpoints
