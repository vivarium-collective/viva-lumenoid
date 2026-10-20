"""Generate a .simularium trajectory for every study that has spatial agents, so
the workbench Simularium Viewer connects to it.

  collagen network (R2, R3)            -> viva_lumenoid.simularium.write_trajectory
  tyssue epithelium (R4, R5, E3)       -> viva_lumenoid.tyssue_simularium
  actomyosin cortex (E1, E2)           -> LAMMPS bead positions
  LAMMPS<->Tyssue coupling (E4)        -> the coupled epithelium's vertices

R1 and R6 are reduced-order (0-D) models with no spatial agents, so they have no
trajectory. Writes <study>/<study>.simularium (the viewer globs for *.simularium).

Run:  python scripts/gen_simularium.py
"""
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STUD = ROOT / "workspace" / "studies"

import numpy as np  # noqa: E402


def _out(slug):
    d = STUD / slug
    d.mkdir(parents=True, exist_ok=True)
    return str(d / f"{slug}.simularium")


def gen_collagen(slug, title, **params_kw):
    from viva_lumenoid.stage1 import run_stage1
    from viva_lumenoid.params import CollagenParams
    from viva_lumenoid.simularium import write_trajectory
    p = CollagenParams(n_rods=90, box_xy=8.0, slab_thickness=3.0, assemble_steps=1500,
                       hold_steps=2500, use_real_angles=True, bending_k=4.0, **params_kw)
    r = run_stage1(p, capture_frames=True, frame_stride=3)
    return write_trajectory(r.frames, p, _out(slug), title)


def gen_tyssue(slug, composite_id, title):
    from viva_lumenoid.tyssue_coupling import _patch_numpy_cross_2d
    from viva_lumenoid.tyssue_simularium import tyssue_frames, write_tyssue_trajectory
    _patch_numpy_cross_2d()
    return write_tyssue_trajectory(tyssue_frames(composite_id, steps=18, dt=0.02), _out(slug), title)


def _bead_frames(process, n, getter, steps):
    process.initial_state() if hasattr(process, "initial_state") else None
    frames = []
    for i in range(steps):
        process.update({}, 1.0)
        frames.append({"t": float(i), "positions": getter()})
    return frames


def gen_actomyosin(slug, title, contractility=1.5, steps=14):
    from bigraph_schema import allocate_core
    from pbg_lammps.model import LammpsModelProcess
    import viva_lumenoid.actomyosin  # noqa: F401
    from viva_lumenoid.tyssue_simularium import write_tyssue_trajectory
    p = LammpsModelProcess(config={"model": "actomyosin-cortex",
                                   "params": {"n": 8, "contractility": contractility}},
                           core=allocate_core())
    frames = _bead_frames(p, 8, lambda: np.array(p._lmp.gather_atoms("x", 1, 3)).reshape(-1, 3), steps)
    p.close()
    return write_tyssue_trajectory(frames, _out(slug), title)


def gen_coupled(slug, title, steps=14):
    from bigraph_schema import allocate_core
    from viva_lumenoid.multiscale_coupling import LammpsTyssueCoupling
    from viva_lumenoid.tyssue_simularium import write_tyssue_trajectory
    p = LammpsTyssueCoupling(config={"bm_rods": 40, "strain_gain": 20.0, "tension_gain": 0.02},
                             core=allocate_core())
    frames = []
    for i in range(steps):
        p.update({}, 0.02)
        cd = list(p._es.eptm.coords)
        pos = p._es.eptm.vert_df[cd].to_numpy(float)
        if pos.shape[1] == 2:
            pos = np.column_stack([pos, np.zeros(len(pos))])
        frames.append({"t": i * 0.02, "positions": pos})
    p.close()
    return write_tyssue_trajectory(frames, _out(slug), title)


JOBS = [
    ("bm-r2-collagen-network-rheology", lambda: gen_collagen("bm-r2-collagen-network-rheology", "collagen IV network (growth rheology)", off_rate=0.02)),
    ("bm-r3-network-expansion-supply", lambda: gen_collagen("bm-r3-network-expansion-supply", "collagen IV network (supply)", off_rate=0.02, make_prob=0.18)),
    ("bm-r4-coupled-2d-epithelium", lambda: gen_tyssue("bm-r4-coupled-2d-epithelium", "viva_lumenoid.composites.tyssue_2d_sheet", "2D epithelial sheet")),
    ("bm-r5-explicit-3d-confinement", lambda: gen_tyssue("bm-r5-explicit-3d-confinement", "viva_lumenoid.composites.tyssue_3d_monolayer", "3D monolayer")),
    ("bm-e1-actomyosin-cortex", lambda: gen_actomyosin("bm-e1-actomyosin-cortex", "actomyosin cortex")),
    ("bm-e2-actin-adhesion-collagen", lambda: gen_actomyosin("bm-e2-actin-adhesion-collagen", "actomyosin cortex (E2)")),
    ("bm-e3-closed-3d-lumenoid", lambda: gen_tyssue("bm-e3-closed-3d-lumenoid", "viva_lumenoid.composites.closed_lumenoid", "closed lumenoid")),
    ("bm-e4-feedback-predictions", lambda: gen_coupled("bm-e4-feedback-predictions", "coupled lumenoid (E4)")),
]


def main():
    for slug, fn in JOBS:
        try:
            out = fn()
            import os
            print(f"  ok   {slug:38s} -> {os.path.getsize(out)//1024} KB")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {slug:38s} {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
