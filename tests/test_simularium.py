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

    # the four mesh agent types plus the per-frame simulation-box outline,
    # with fibers as FIBER and ends as SPHERE
    geom = {v["name"]: v["geometry"]["displayType"] for v in ti["typeMapping"].values()}
    assert geom == {"collagen rod": "FIBER", "crosslink": "FIBER",
                    "NC1": "SPHERE", "7S": "SPHERE",
                    "simulation box": "FIBER"}

    # an explicit camera is set (not simulariumio's default-box one)
    assert ti["cameraDefault"]["position"]["z"] > 0

    # frame 0 holds both fibers (viz 1001, with subpoints) and spheres (1000)
    a = doc["spatialData"]["bundleData"][0]["data"]
    i, viz_counts, rod_nsub = 0, {}, None
    while i < len(a):
        viz = a[i]; nsub = int(a[i + 10])
        viz_counts[viz] = viz_counts.get(viz, 0) + 1
        if viz == 1001.0 and nsub == 6 and rod_nsub is None:
            rod_nsub = nsub
        i += 11 + nsub
    assert viz_counts.get(1001.0, 0) == 12      # 10 rods + 1 crosslink + 1 box
    assert viz_counts.get(1000.0, 0) == 20      # 10 NC1 + 10 7S end nodes
    assert rod_nsub == 6                         # two xyz endpoints (rod/crosslink)


def _stretch_frame(n_rods, t, box_w):
    """A frame carrying σ/ε traces and a growing box, with the bead cloud filling
    the box (as a real stretched network does — beads scale with the box, so the
    simularium global centroid/scale track the deformation)."""
    n = 2 * n_rods
    base = np.random.default_rng(0).uniform(0, 1, size=(n, 3))   # fixed shape
    pos = base * [box_w, box_w, 2.0]                             # fills the box
    f = {
        "positions": pos, "atom_types": np.tile([1, 2], n_rods),
        "atom_ids": np.arange(1, n + 1),
        "bonds": [[1, 2 * i + 1, 2 * i + 2] for i in range(n_rods)] + [[2, 1, 3]],
        "box": [box_w, box_w, 2.0], "t": float(t),
        "strain": 0.01 * t, "sigma": 0.4 + 0.02 * t, "n_crosslinks": n_rods // 2,
    }
    return f


def test_simularium_box_grows_with_strain(tmp_path):
    """The simulation-box outline is rebuilt per frame, so under stretch the box
    widens frame-to-frame — the reviewer's "box itself showing the strain"."""
    frames = [_stretch_frame(10, t, box_w=10.0 + 0.5 * t) for t in range(6)]
    out = write_trajectory(frames, CollagenParams(), str(tmp_path / "stretch"),
                           title="stretch")
    doc = json.load(open(out))
    box_tid = next(int(k) for k, v in doc["trajectoryInfo"]["typeMapping"].items()
                   if v["name"] == "simulation box")

    def box_span(frame):
        a = frame["data"]; i = 0
        while i < len(a):
            tid = int(a[i + 1]); nsub = int(a[i + 10])
            if tid == box_tid:
                xs = a[i + 11:i + 11 + nsub][0::3]
                return max(xs) - min(xs)
            i += 11 + nsub
        return None

    bundle = doc["spatialData"]["bundleData"]
    first, last = box_span(bundle[0]), box_span(bundle[-1])
    assert first and last and last > first * 1.2      # box visibly widened


def test_simularium_embeds_diagnostic_plots(tmp_path):
    """Frames carrying σ/ε traces get the stress-vs-strain (and time-series) plots
    embedded in the .simularium — the reviewer's "add the plots to the simularium
    visualizations themselves"."""
    frames = [_stretch_frame(10, t, box_w=10.0 + 0.5 * t) for t in range(6)]
    out = write_trajectory(frames, CollagenParams(), str(tmp_path / "plots"),
                           title="plots")
    doc = json.load(open(out))
    titles = [p["layout"]["title"] for p in doc["plotData"]["data"]]
    assert any("stress vs strain" in t.lower() for t in titles)
    assert len(titles) == 3      # σ(ε), σ/ε(t), crosslinks(t)


def test_simularium_snapshot_has_no_plots(tmp_path):
    """A bare snapshot (no σ/ε traces) writes cleanly with an empty plot panel."""
    frames = [_net_frame(10, 0)]
    out = write_trajectory(frames, CollagenParams(), str(tmp_path / "snap"),
                           title="snap")
    doc = json.load(open(out))
    assert doc.get("plotData", {}).get("data", []) == []
