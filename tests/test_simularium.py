"""Simularium converter — turn collagen bead positions into a .simularium file."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("simulariumio", reason="simulariumio required")
pytest.importorskip("viva_simularium", reason="viva-simularium required")

from viva_lumenoid import CollagenParams
from viva_lumenoid.simularium import agents_from_positions, write_trajectory


def test_agents_from_positions_types_and_nm():
    pos = np.array([[1.0, 2.0, 0.5], [1.5, 2.0, 0.5]])
    types = np.array([1, 2])  # NC1, 7S
    ag = agents_from_positions(pos, types, nm=125.0, radius_nm=15.0)
    assert [a["type"] for a in ag] == ["NC1", "7S"]
    assert ag[0]["x"] == 125.0 and ag[0]["radius"] == 15.0   # converted to nm


def test_write_trajectory_produces_a_simularium_file(tmp_path):
    rng = np.random.default_rng(0)
    frames = []
    for t in range(5):
        pos = rng.uniform(0, 10, size=(20, 3))
        frames.append({"positions": pos, "atom_types": np.tile([1, 2], 10),
                       "box": [10.0, 10.0, 2.0], "t": float(t)})
    out = write_trajectory(frames, CollagenParams(), str(tmp_path / "traj"), title="test")
    assert out.endswith(".simularium")
    import json
    doc = json.load(open(out))   # JSON format — the workbench viewer parses this
    assert "trajectoryInfo" in doc and "bundleData" in doc["spatialData"]
    assert len(doc["spatialData"]["bundleData"]) == 5
    tm = doc["trajectoryInfo"]["typeMapping"]
    assert {v["name"] for v in tm.values()} == {"NC1", "7S"}
