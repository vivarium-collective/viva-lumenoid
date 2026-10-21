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
            "box": list(o.get("box_dimensions") or []),
        })
    proc.close()
    return frames


# collagen bond types: 1 = intra-rod (the rod), 2 = NC1-NC1 crosslink, 3 = 7S-7S crosslink
_ROD_COL, _XLINK_COL = "#9aa7b4", "#c0392b"


def _min_image_seg(p, q, box):
    """Minimum-image 2D segment from p: periodic boundaries mean a bond between two
    atoms stored on opposite sides of the box is really SHORT across the edge, not a
    line spanning the whole box. Draw it at its true (wrapped) length from p."""
    d = np.asarray(q[:2], float) - np.asarray(p[:2], float)
    if box is not None and len(box) >= 2 and box[0] and box[1]:
        bx = np.array([box[0], box[1]], float)
        d = d - bx * np.round(d / bx)
    return (float(p[0]), float(p[0] + d[0])), (float(p[1]), float(p[1] + d[1]))


def _draw_collagen_network(ax, pos, types, ids, bonds, box, *, bead_s=9, lw=0.9):
    """Draw a collagen network (xy): intra-rod bonds (the rods) + crosslinks +
    NC1/7S beads, every bond at its true minimum-image length (no periodic-wrap
    lines spanning the box)."""
    id2idx = {int(a): k for k, a in enumerate(ids)}
    for bt, a1, a2 in bonds:
        if a1 not in id2idx or a2 not in id2idx:
            continue
        xs, ys = _min_image_seg(pos[id2idx[a1]], pos[id2idx[a2]], box)
        if bt == 1:                       # intra-rod bond = the collagen rod
            ax.plot(xs, ys, color=_ROD_COL, lw=lw * 0.8, alpha=0.7, zorder=1)
        elif bt in (2, 3):                # NC1/7S crosslink = network junction
            ax.plot(xs, ys, color=_XLINK_COL, lw=lw, alpha=0.85, zorder=2)
    for t, col in ((1, _NC1), (2, _S7)):
        m = types == t
        if m.any():
            ax.scatter(pos[m, 0], pos[m, 1], s=bead_s, c=col, zorder=3, edgecolors="none")


def collagen_network_gif(frames: list[dict], out_path: str,
                         title: str = "collagen IV network") -> str:
    """Animate a collagen network: NC1/7S beads + crosslink bonds (xy projection)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    box = next((f.get("box") for f in frames if f.get("box")), None)
    if box and box[0] and box[1]:
        lo, hi = np.array([0.0, 0.0]), np.array([box[0], box[1]])
    else:
        lo, hi = _bounds(np.vstack([f["positions"][:, :2] for f in frames]))
    fig, ax = plt.subplots(figsize=(5, 5))

    def draw(i):
        ax.clear()
        f = frames[i]
        _draw_collagen_network(ax, f["positions"], f["atom_types"], f["atom_ids"],
                               f["bonds"], f.get("box") or box)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.set_axis_off()
        nx = sum(1 for bt, *_ in f["bonds"] if bt in (2, 3))
        ax.set_title(f"{title}\nt={f['t']:.0f}  ·  {nx} crosslinks  "
                     f"(grey = collagen rods, red = crosslinks)", fontsize=9)

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
            "bbox": list(bm.get("box_dimensions") or []),
            "basal_tension": float(o.get("basal_tension", 0.0)),
            "bm_sigma": float(o.get("bm_sigma", 0.0)),
            "basal_strain_rate": float(o.get("basal_strain_rate", 0.0)),
            "basal_area": float(o.get("basal_area", 0.0)),
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
        _draw_collagen_network(axR, f["bpos"], f["btypes"], f["bids"], f["bbonds"],
                               f.get("bbox"), bead_s=10)
        axR.set_xlim(blo[0], bhi[0]); axR.set_ylim(blo[1], bhi[1]); axR.set_aspect("equal")
        axR.set_axis_off()
        axR.set_title(f"collagen IV BM (LAMMPS)\nσ = {f['bm_sigma']:.3f} LJ", fontsize=9)
        fig.suptitle(f"{title}   ·   t = {f['t']:.2f}", fontsize=11)

    return _save_anim(fig, draw, len(frames), out_path, fps=8)


def coupling_loop_gif(frames: list[dict], out_path: str,
                      title: str = "E4 — LAMMPS↔Tyssue coupling loop") -> str:
    """The coupling made EXPLICIT: epithelium (left) and explicit LAMMPS collagen BM
    (right) joined by two bidirectional arrows that pulse with the live signals —
    DOWN = epithelial strain stretching the collagen, UP = collagen stress fed back
    as basal tension — over a co-evolution trace strip with a moving time cursor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    vlo, vhi = _bounds(np.vstack([f["vpos"][:, :2] for f in frames]))
    blo, bhi = _bounds(np.vstack([f["bpos"][:, :2] for f in frames]))
    rate = np.array([f["basal_strain_rate"] for f in frames])
    sig = np.array([f["bm_sigma"] for f in frames])
    ten = np.array([f["basal_tension"] for f in frames])
    tt = np.array([f["t"] for f in frames])

    def _norm(a):
        m = np.max(np.abs(a)) or 1.0
        return a / m
    rn, sn, tn = _norm(rate), _norm(sig), _norm(ten)
    # arrow line-width from the normalised magnitude (visible pulse)
    def _lw(x):
        return 1.0 + 7.0 * abs(float(x))

    fig = plt.figure(figsize=(9.5, 6.4))
    gs = GridSpec(2, 3, height_ratios=[2.1, 1.0], width_ratios=[1, 0.55, 1],
                  hspace=0.28, wspace=0.08)
    axL = fig.add_subplot(gs[0, 0]); axM = fig.add_subplot(gs[0, 1]); axR = fig.add_subplot(gs[0, 2])
    axB = fig.add_subplot(gs[1, :])

    def draw(i):
        f = frames[i]
        for ax in (axL, axM, axR, axB):
            ax.clear()
        # epithelium
        vp = f["vpos"]
        for s, t in f["vedges"]:
            axL.plot([vp[s, 0], vp[t, 0]], [vp[s, 1], vp[t, 1]], color="#2bb673", lw=0.5, alpha=0.8)
        axL.set_xlim(vlo[0], vhi[0]); axL.set_ylim(vlo[1], vhi[1]); axL.set_aspect("equal"); axL.set_axis_off()
        axL.set_title(f"epithelium (Tyssue)\nbasal area = {f['basal_area']:.1f}", fontsize=9)
        # collagen BM (rods + crosslinks, minimum-image)
        _draw_collagen_network(axR, f["bpos"], f["btypes"], f["bids"], f["bbonds"],
                               f.get("bbox"), bead_s=9)
        axR.set_xlim(blo[0], bhi[0]); axR.set_ylim(blo[1], bhi[1]); axR.set_aspect("equal"); axR.set_axis_off()
        axR.set_title(f"collagen IV BM (LAMMPS)\nσ = {f['bm_sigma']:.2f} LJ", fontsize=9)
        # the coupling arrows (middle)
        axM.set_xlim(0, 1); axM.set_ylim(0, 1); axM.set_axis_off()
        axM.annotate("", xy=(0.92, 0.72), xytext=(0.08, 0.72),
                     arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=_lw(rn[i])))
        axM.text(0.5, 0.86, f"ε̇ stretches\ncollagen\n({rate[i]:.2f})", ha="center", va="center",
                 fontsize=8, color="#c0392b")
        axM.annotate("", xy=(0.08, 0.28), xytext=(0.92, 0.28),
                     arrowprops=dict(arrowstyle="-|>", color="#2c6fbb", lw=_lw(tn[i])))
        axM.text(0.5, 0.13, f"σ → basal\ntension\n({f['basal_tension']:.3f})", ha="center", va="center",
                 fontsize=8, color="#2c6fbb")
        # co-evolution trace strip
        axB.plot(tt, rn, color="#c0392b", lw=1.6, label="ε̇ (strain rate)")
        axB.plot(tt, sn, color="#555", lw=1.6, label="σ (collagen stress)")
        axB.plot(tt, tn, color="#2c6fbb", lw=1.6, label="basal tension (feedback)")
        axB.axvline(tt[i], color="#111", lw=1.2, alpha=0.7)
        axB.axhline(0, color="#ddd", lw=0.8)
        axB.set_xlim(tt[0], tt[-1]); axB.set_ylim(-1.15, 1.15)
        axB.set_yticks([]); axB.set_xlabel("time (LJ)", fontsize=8)
        axB.set_title("coupling signals (each normalised) — the round-trip co-evolving", fontsize=9)
        axB.legend(loc="upper center", ncol=3, fontsize=8, framealpha=0.6, bbox_to_anchor=(0.5, 1.5))
        fig.suptitle(f"{title}   ·   two-way, explicit BM   ·   t = {f['t']:.2f}", fontsize=11)

    return _save_anim(fig, draw, len(frames), out_path, fps=6)


# ── E4 coupling OVERLAY: epithelium + collagen BM in ONE space ─────────────── #
# The coupling is through aggregate signals (basal area <-> collagen strain/stress),
# not a per-vertex map, so the overlay PLACES the real collagen network on the real
# epithelial basal surface (a representational co-location) and lets both deform
# from the real run — showing WHERE they connect (the basal membrane) and HOW they
# move together (both breathe as the epithelium grows).
def capture_coupling_overlay(steps=18, dt=0.02, bm_rods=90):
    """Step the explicit-BM coupling; per frame capture the epithelial cross-section
    (ring), the collagen BM state, and the coupling signals."""
    from bigraph_schema import allocate_core
    from viva_lumenoid.multiscale_coupling import LammpsTyssueCoupling
    p = LammpsTyssueCoupling(config={"bm_mode": "explicit", "coupling_mode": "two-way",
                                     "bm_rods": bm_rods, "strain_gain": 20.0,
                                     "tension_gain": 0.02}, core=allocate_core())
    frames = []
    for i in range(steps):
        o = p.update({}, dt)
        eptm = p._es.eptm
        v = eptm.vert_df
        zc = float(v["z"].median())
        half = 0.10 * (float(v["z"].max()) - float(v["z"].min()) + 1e-9)
        band = v[(v["z"] >= zc - half) & (v["z"] <= zc + half)]
        bm = p._bm._read()
        frames.append({
            "t": i * dt,
            "band": band[["x", "y"]].to_numpy(dtype=float),
            "bpos": np.asarray(bm["positions"], float), "btypes": np.asarray(bm["atom_types"], int),
            "bids": np.asarray(bm["atom_ids"], int), "bbonds": [list(map(int, b)) for b in bm["bonds"]],
            "bbox": list(bm.get("box_dimensions") or []),
            "basal_area": float(o.get("basal_area", 0.0)), "bm_sigma": float(o.get("bm_sigma", 0.0)),
            "basal_tension": float(o.get("basal_tension", 0.0)),
        })
    p.close()
    return frames


def _wrap_collagen_onto_ring(f, cx, cy, R, thickness):
    """Map the collagen patch (box coords) onto an annulus of radius R centered at
    (cx,cy): patch-x -> angle, patch-y -> small radial offset outside the basal ring.
    Returns (rod_segs, xlink_segs, bead_xy, bead_types) in ring coordinates."""
    pos, ids, box = f["bpos"], f["bids"], f["bbox"]
    Lx = box[0] if box and box[0] else (pos[:, 0].max() - pos[:, 0].min() + 1e-9)
    Ly = box[1] if box and box[1] else (pos[:, 1].max() - pos[:, 1].min() + 1e-9)
    id2idx = {int(a): k for k, a in enumerate(ids)}

    def to_ring(px, py):
        th = 2 * np.pi * (px / Lx)
        r = R + ((py / Ly) - 0.5) * thickness
        return cx + r * np.cos(th), cy + r * np.sin(th)
    rods, xls = [], []
    for bt, a1, a2 in f["bbonds"]:
        if a1 not in id2idx or a2 not in id2idx:
            continue
        p0, q0 = pos[id2idx[a1]], pos[id2idx[a2]]
        dx = q0[0] - p0[0]
        if Lx:
            dx -= Lx * round(dx / Lx)           # min-image in the patch BEFORE wrapping
        x0, y0 = to_ring(p0[0], p0[1]); x1, y1 = to_ring(p0[0] + dx, q0[1])
        (xls if bt in (2, 3) else rods).append(((x0, x1), (y0, y1)))
    bx, by = to_ring(pos[:, 0], pos[:, 1])
    return rods, xls, (bx, by), f["btypes"]


def coupling_overlay_gif(frames, out_path, title="E4 — epithelium + collagen BM, overlaid at the basal surface"):
    """Movie: the lumenoid cross-section (epithelium) with the real collagen BM wrapped
    as the membrane on its basal (outer) surface — both deforming together."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    allxy = np.vstack([f["band"] for f in frames])
    c0 = allxy.mean(axis=0)
    span = np.abs(allxy - c0).max() * 1.6
    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    return _save_anim(fig, lambda i: _draw_overlay_frame(ax, frames[i], span, title),
                      len(frames), out_path, fps=7)


def _draw_overlay_frame(ax, f, span, title):
    """One overlay frame: lumenoid cross-section (epithelium) + the real collagen BM
    wrapped on its basal (outer) surface, both at their current deformed state."""
    ax.clear()
    band = f["band"]
    cx, cy = band.mean(axis=0)
    rad = np.hypot(band[:, 0] - cx, band[:, 1] - cy)
    R_out = np.percentile(rad, 90)
    R_in = np.percentile(rad, 12)
    thick = 0.14 * R_out
    # epithelium cross-section: cell vertices + lumen (inner) + basal (outer) rings
    ax.scatter(band[:, 0], band[:, 1], s=7, c="#2bb673", alpha=0.6, zorder=2, edgecolors="none")
    ang = np.linspace(0, 2 * np.pi, 120)
    ax.plot(cx + R_in * np.cos(ang), cy + R_in * np.sin(ang), color="#8bd3b0", lw=1.0, alpha=0.7, zorder=1)
    ax.plot(cx + R_out * np.cos(ang), cy + R_out * np.sin(ang), color="#1f9d57", lw=1.6, zorder=3)
    ax.text(cx, cy, "lumen", ha="center", va="center", fontsize=9, color="#777")
    # collagen BM wrapped on the basal (outer) ring
    rods, xls, (bx, by), bt = _wrap_collagen_onto_ring(f, cx, cy, R_out + thick * 0.5, thick)
    for (xs, ys) in rods:
        ax.plot(xs, ys, color=_ROD_COL, lw=0.7, alpha=0.7, zorder=4)
    for (xs, ys) in xls:
        ax.plot(xs, ys, color=_XLINK_COL, lw=0.9, alpha=0.85, zorder=5)
    for ty, col in ((1, _NC1), (2, _S7)):
        m = bt == ty
        if m.any():
            ax.scatter(bx[m], by[m], s=6, c=col, zorder=6, edgecolors="none")
    ax.annotate("collagen IV BM (LAMMPS)\non the basal surface",
                xy=(cx, cy + R_out + thick), xytext=(cx, cy + span * 0.82),
                ha="center", fontsize=9, color=_XLINK_COL,
                arrowprops=dict(arrowstyle="-|>", color=_XLINK_COL, lw=1.2))
    ax.set_xlim(cx - span, cx + span); ax.set_ylim(cy - span, cy + span)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title(f"{title}\nt={f['t']:.2f}  ·  basal area {f['basal_area']:.0f}  ·  "
                 f"collagen σ {f['bm_sigma']:.2f}  ·  basal tension {f['basal_tension']:.3f}", fontsize=9)


def coupling_overlay_image(frames, out_path, frame_idx=None,
                           title="E4 — epithelium + collagen BM, overlaid at the basal surface") -> str:
    """A single representative overlay frame as a PNG (same scene as the movie)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if frame_idx is None:
        frame_idx = int(len(frames) * 0.66)
    allxy = np.vstack([f["band"] for f in frames])
    c0 = allxy.mean(axis=0)
    span = np.abs(allxy - c0).max() * 1.6
    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    _draw_overlay_frame(ax, frames[frame_idx], span, title)
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out)
