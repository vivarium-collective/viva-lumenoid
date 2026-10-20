"""The tyssue Simularium writer produces a .simularium from vertex frames.

Engine-free (uses only numpy + simulariumio), so it runs in CI.
"""
import numpy as np


def test_write_tyssue_trajectory(tmp_path):
    from viva_lumenoid.tyssue_simularium import write_tyssue_trajectory
    frames = [{"t": float(i), "positions": np.random.rand(12, 3)} for i in range(4)]
    out = write_tyssue_trajectory(frames, str(tmp_path / "traj.simularium"), "test")
    import os
    assert out.endswith(".simularium") and os.path.getsize(out) > 0
