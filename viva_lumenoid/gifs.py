"""Animated GIF visualizations for the spatial lumenoid studies.

Companion to ``viva_lumenoid.simularium`` (interactive viewer) and
``viva_lumenoid.tyssue_simularium``: lightweight, self-contained ``.gif`` movies
that embed directly in a study's visualization (no viewer needed). Extends the
approach in ``vivarium_tyssue.visualizations.tissue_gif`` (frames -> matplotlib
FuncAnimation -> PillowWriter) to the LAMMPS collagen network, and adds a
two-panel COUPLING gif (Tyssue epithelium + explicit LAMMPS collagen BM) for E4.

Renderers take captured frames and write a ``.gif``; capture helpers run the real
composites/processes to produce those frames.
"""
from __future__ import annotations

import pathlib
import tempfile
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# collagen atom types: 1 = NC1 end bead, 2 = 7S end bead
_NC1, _S7 = "#e8734a", "#4f7cff"


def _save_anim(fig, draw, n_frames, out_path, fps=10):
    """Animate draw(i) over n_frames and write a .gif to out_path."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    import matplotlib.pyplot as plt
    anim = FuncAnimation(fig, draw, frames=n_frames, interval=1000 // fps)
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        anim.save(tmp_path, writer=PillowWriter(fps=fps))
        out.write_bytes(pathlib.Path(tmp_path).read_bytes())
    finally:
        plt.close(fig)
        pathlib.Path(tmp_path).unlink(missing_ok=True)
    return str(out)


def _bounds(all_xy, pad_frac=0.06):
    mins, maxs = all_xy.min(axis=0), all_xy.max(axis=0)
    pad = pad_frac * (maxs - mins + 1e-9)
    return mins - pad, maxs + pad


# -- collagen LAMMPS network ------------------------------------------------- #
def capture_collagen(params, steps=16, make_on=1.0, break_on=1.0) -> list[dict]:
    """Step CollagenNetworkProcess, capturing positions/atom_types/bonds/ids."""
    from bigraph_schema import allocate_core
    from .collagen import CollagenNetworkProcess
    proc = CollagenNetworkProcess(config={"params": params.to_dict()}, core=allocate_core())
    proc.initial_state()
    frames = []
    for i in range(steps):
        o = proc.update({"strain_rate": 0.0, "make_on": make_on, "break_on": break_on}, 6.0)
        frames.append({
            "t": float(i),
            "positions": np.asarray(o["positions"], dtype=float),
            "atom_types": np.asarray(o["atom_types"], dtype=int),
            "atom_ids": np.asarray(o["atom_ids"], dtype=int),
            "bonds": [list(map(int, b)) for b in o["bonds"]],
        })
    proc.close()
    return frames


def collagen_network_gif(frames: list[dict], out_path: str,
                         title: str = "collagen IV network") -> str:
    """Animate a collagen network: NC1/7S beads + crosslink bonds (xy projection)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    allxy = np.vstack([f["positions"][:, :2] for f in frames])
    lo, hi = _bounds(allxy)
    fig, ax = plt.subplots(figsize=(5, 5))

    def draw(i):
        ax.clear()
        f = frames[i]
        pos, types = f["positions"], f["atom_types"]
        id2idx = {int(a): k for k, a in enumerate(f["atom_ids"])}
        # crosslink bonds (types 2/3) as grey segments; intra-rod (type 1) skipped for clarity
        for bt, a1, a2 in f["bonds"]:
            if bt in (2, 3) and a1 in id2idx and a2 in id2idx:
                p, q = pos[id2idx[a1]], pos[id2idx[a2]]
                ax.plot([p[0], q[0]], [p[1], q[1]], color="#999", lw=0.5, alpha=0.6, zorder=1)
        for t, col, lbl in ((1, _NC1, "NC1"), (2, _S7, "7S")):
            m = types == t
            if m.any():
                ax.scatter(pos[m, 0], pos[m, 1], s=10, c=col, label=lbl, zorder=2, edgecolors="none")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.set_axis_off()
        nx = sum(1 for bt, *_ in f["bonds"] if bt in (2, 3))
        ax.set_title(f"{title}\nt={f['t']:.0f}  ·  {nx} crosslinks", fontsize=10)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    return _save_anim(fig, draw, len(frames), out_path)


# -- actomyosin cortex (LJ bead patch) --------------------------------------- #
def capture_actomyosin(contractility=1.5, n=8, steps=16) -> list[dict]:
    """Step the actomyosin-cortex LAMMPS model, capturing bead positions."""
    from bigraph_schema import allocate_core
    from pbg_lammps.model import LammpsModelProcess
    import viva_lumenoid.actomyosin  # noqa: F401
    p = LammpsModelProcess(config={"model": "actomyosin-cortex",
                                   "params": {"n": n, "contractility": contractility}},
                           core=allocate_core())
    p.initial_state()
    frames = []
    for i in range(steps):
        p.update({}, 1.0)
        frames.append({"t": float(i),
                       "positions": np.array(p._lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)})
    p.close()
    return frames


def beads_gif(frames: list[dict], out_path: str, title: str = "actomyosin cortex",
              color: str = _NC1) -> str:
    """Animate a bead patch contracting (xy projection)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    allxy = np.vstack([f["positions"][:, :2] for f in frames])
    lo, hi = _bounds(allxy, pad_frac=0.12)
    fig, ax = plt.subplots(figsize=(5, 5))

    def draw(i):
        ax.clear()
        pos = frames[i]["positions"]
        ax.scatter(pos[:, 0], pos[:, 1], s=40, c=color, edgecolors="#333", linewidths=0.4)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"{title}\nt={frames[i]['t']:.0f}  ·  {len(pos)} beads", fontsize=10)

    return _save_anim(fig, draw, len(frames), out_path)


# -- tyssue epithelial mesh -------------------------------------------------- #
def capture_tyssue(composite_id: str, steps=16, dt=0.02) -> list[dict]:
    """Step a Tyssue composite, capturing vertex positions + edges per frame."""
    import viva_lumenoid.composites  # noqa: F401
    from process_bigraph.composite_generator import _REGISTRY, apply_core_extensions, build_generator
    from bigraph_schema import allocate_core
    from process_bigraph import Composite
    from .tyssue_coupling import _patch_numpy_cross_2d
    _patch_numpy_cross_2d()
    e = _REGISTRY[composite_id]; core = apply_core_extensions(e, allocate_core())
    doc = build_generator(e, None, core=core)
    comp = Composite({"state": doc["state"] if "state" in doc else doc}, core=core)
    eptm = comp.state["Tyssue"]["instance"].eptm
    coords = list(eptm.coords)
    edges = eptm.edge_df[["srce", "trgt"]].to_numpy(dtype=int)
    frames = []
    for i in range(steps):
        comp.update({}, dt)
        pos = eptm.vert_df[coords].to_numpy(dtype=float)
        if pos.shape[1] == 2:
            pos = np.column_stack([pos, np.zeros(len(pos))])
        frames.append({"t": i * dt, "positions": pos.copy(), "edges": edges})
    return frames


def tyssue_mesh_gif(frames: list[dict], out_path: str,
                    title: str = "tyssue epithelium") -> str:
    """Animate a tyssue mesh: edges as segments (xy projection)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    allxy = np.vstack([f["positions"][:, :2] for f in frames])
    lo, hi = _bounds(allxy)
    fig, ax = plt.subplots(figsize=(5, 5))

    def draw(i):
        ax.clear()
        f = frames[i]; pos = f["positions"]
        for s, t in f["edges"]:
            ax.plot([pos[s, 0], pos[t, 0]], [pos[s, 1], pos[t, 1]],
                    color="#4f7cff", lw=0.5, alpha=0.8)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"{title}\nt={f['t']:.2f}  ·  {len(pos)} vertices", fontsize=10)

    return _save_anim(fig, draw, len(frames), out_path)


# -- E4 coupling: Tyssue epithelium + explicit LAMMPS collagen BM ------------ #
def capture_coupling(steps=16, dt=0.02, bm_rods=40) -> list[dict]:
    """Step the explicit-BM LammpsTyssueCoupling, capturing BOTH engines."""
    from bigraph_schema import allocate_core
    from .multiscale_coupling import LammpsTyssueCoupling
    p = LammpsTyssueCoupling(config={"bm_mode": "explicit", "coupling_mode": "two-way",
                                     "bm_rods": bm_rods, "strain_gain": 20.0,
                                     "tension_gain": 0.02}, core=allocate_core())
    coords = None
    eptm_edges = None
    frames = []
    for i in range(steps):
        o = p.update({}, dt)
        eptm = p._es.eptm
        if coords is None:
            coords = list(eptm.coords)
            eptm_edges = eptm.edge_df[["srce", "trgt"]].to_numpy(dtype=int)
        vp = eptm.vert_df[coords].to_numpy(dtype=float)
        if vp.shape[1] == 2:
            vp = np.column_stack([vp, np.zeros(len(vp))])
        bm = p._bm._read()
        frames.append({
            "t": i * dt, "vpos": vp.copy(), "vedges": eptm_edges,
            "bpos": np.asarray(bm["positions"], dtype=float),
            "btypes": np.asarray(bm["atom_types"], dtype=int),
            "bids": np.asarray(bm["atom_ids"], dtype=int),
            "bbonds": [list(map(int, b)) for b in bm["bonds"]],
            "basal_tension": float(o.get("basal_tension", 0.0)),
            "bm_sigma": float(o.get("bm_sigma", 0.0)),
        })
    p.close()
    return frames


def coupling_gif(frames: list[dict], out_path: str,
                 title: str = "E4 — LAMMPS collagen BM ↔ Tyssue epithelium") -> str:
    """Two-panel coupling animation: epithelium (left) + explicit collagen BM (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vlo, vhi = _bounds(np.vstack([f["vpos"][:, :2] for f in frames]))
    blo, bhi = _bounds(np.vstack([f["bpos"][:, :2] for f in frames]))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 4.6))

    def draw(i):
        f = frames[i]
        axL.clear(); axR.clear()
        # epithelium
        vp = f["vpos"]
        for s, t in f["vedges"]:
            axL.plot([vp[s, 0], vp[t, 0]], [vp[s, 1], vp[t, 1]], color="#2bb673", lw=0.5, alpha=0.8)
        axL.set_xlim(vlo[0], vhi[0]); axL.set_ylim(vlo[1], vhi[1]); axL.set_aspect("equal")
        axL.set_axis_off()
        axL.set_title(f"epithelium (Tyssue)\nbasal tension = {f['basal_tension']:.3f}", fontsize=9)
        # collagen BM
        bp, bt = f["bpos"], f["btypes"]
        id2idx = {int(a): k for k, a in enumerate(f["bids"])}
        for btype, a1, a2 in f["bbonds"]:
            if btype in (2, 3) and a1 in id2idx and a2 in id2idx:
                p0, q0 = bp[id2idx[a1]], bp[id2idx[a2]]
                axR.plot([p0[0], q0[0]], [p0[1], q0[1]], color="#999", lw=0.5, alpha=0.6)
        for ty, col in ((1, _NC1), (2, _S7)):
            m = bt == ty
            if m.any():
                axR.scatter(bp[m, 0], bp[m, 1], s=10, c=col, edgecolors="none")
        axR.set_xlim(blo[0], bhi[0]); axR.set_ylim(blo[1], bhi[1]); axR.set_aspect("equal")
        axR.set_axis_off()
        axR.set_title(f"collagen IV BM (LAMMPS)\nσ = {f['bm_sigma']:.3f} LJ", fontsize=9)
        fig.suptitle(f"{title}   ·   t = {f['t']:.2f}", fontsize=11)

    return _save_anim(fig, draw, len(frames), out_path, fps=8)
