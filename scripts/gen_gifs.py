"""Generate an animated .gif for every spatial study and drop it in viz/.

  collagen network (R2, R3)        -> gifs.collagen_network_gif
  actomyosin cortex (E1, E2)       -> gifs.beads_gif
  tyssue epithelium (R4, R5, E3)   -> gifs.tyssue_mesh_gif
  LAMMPS<->Tyssue coupling (E4)    -> gifs.coupling_gif (two-panel, the hero)

R1/R6 are reduced-order (0-D) with no spatial agents -> no gif. Writes
workspace/studies/<slug>/viz/<slug>.gif.

Run:  python scripts/gen_gifs.py
"""
import os
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
STUD = ROOT / "workspace" / "studies"

from viva_lumenoid import gifs  # noqa: E402
from viva_lumenoid.params import CollagenParams  # noqa: E402


def _out(slug):
    # gif: addresses resolve to a bare filename in the STUDY DIR root (not viz/).
    d = STUD / slug
    d.mkdir(parents=True, exist_ok=True)
    return str(d / f"{slug}.gif")


def _collagen(slug, title, **kw):
    fr = gifs.capture_collagen(CollagenParams(n_rods=70, box_xy=8.0, slab_thickness=3.0,
                                              use_real_angles=True, bending_k=4.0, **kw), steps=14)
    return gifs.collagen_network_gif(fr, _out(slug), title)


def _actomyosin(slug, title, contractility=1.5):
    return gifs.beads_gif(gifs.capture_actomyosin(contractility=contractility, steps=14), _out(slug), title)


def _tyssue(slug, cid, title):
    return gifs.tyssue_mesh_gif(gifs.capture_tyssue(cid, steps=16), _out(slug), title)


def _coupling(slug, title):
    return gifs.coupling_gif(gifs.capture_coupling(steps=16, bm_rods=40), _out(slug), title)


JOBS = [
    ("bm-r2-collagen-network-rheology", lambda: _collagen("bm-r2-collagen-network-rheology", "R2 — collagen IV network remodeling", off_rate=0.02)),
    ("bm-r3-network-expansion-supply", lambda: _collagen("bm-r3-network-expansion-supply", "R3 — collagen network (remodeling + supply)", off_rate=0.02, make_prob=0.18)),
    ("bm-r4-coupled-2d-epithelium", lambda: _tyssue("bm-r4-coupled-2d-epithelium", "viva_lumenoid.composites.tyssue_2d_sheet", "R4 — 2D epithelial sheet")),
    ("bm-r5-explicit-3d-confinement", lambda: _tyssue("bm-r5-explicit-3d-confinement", "viva_lumenoid.composites.tyssue_3d_monolayer", "R5 — 3D monolayer")),
    ("bm-e1-actomyosin-cortex", lambda: _actomyosin("bm-e1-actomyosin-cortex", "E1 — actomyosin cortex (LJ bead patch)")),
    ("bm-e2-actin-adhesion-collagen", lambda: _actomyosin("bm-e2-actin-adhesion-collagen", "E2 — actomyosin cortex (coupling input)")),
    ("bm-e3-closed-3d-lumenoid", lambda: _tyssue("bm-e3-closed-3d-lumenoid", "viva_lumenoid.composites.closed_lumenoid", "E3 — closed 3D lumenoid shell")),
    ("bm-e4-feedback-predictions", lambda: _coupling("bm-e4-feedback-predictions", "E4 — LAMMPS collagen BM ↔ Tyssue epithelium")),
]


def main():
    for slug, fn in JOBS:
        try:
            out = fn()
            print(f"  ok   {slug:38s} -> {os.path.getsize(out)//1024} KB")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {slug:38s} {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
