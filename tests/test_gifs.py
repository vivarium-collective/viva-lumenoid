"""The gif renderers produce a non-empty .gif from synthetic frames.

Engine-free (numpy + matplotlib/Pillow only), so it runs in CI; the real captures
(which need LAMMPS/tyssue) are exercised by scripts/gen_gifs.py, not here.
"""
import numpy as np


def test_collagen_network_gif(tmp_path):
    from viva_lumenoid.gifs import collagen_network_gif
    frames = [{"t": float(i),
               "positions": np.random.rand(12, 3),
               "atom_types": np.array([1, 2] * 6),
               "atom_ids": np.arange(1, 13),
               "bonds": [[1, 1, 2], [2, 1, 3], [3, 4, 6]]} for i in range(3)]
    out = collagen_network_gif(frames, str(tmp_path / "c.gif"), "test")
    import os
    assert out.endswith(".gif") and os.path.getsize(out) > 0


def test_tyssue_mesh_gif(tmp_path):
    from viva_lumenoid.gifs import tyssue_mesh_gif
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    frames = [{"t": i * 0.1, "positions": np.random.rand(3, 3), "edges": edges} for i in range(3)]
    out = tyssue_mesh_gif(frames, str(tmp_path / "t.gif"), "test")
    import os
    assert os.path.getsize(out) > 0
