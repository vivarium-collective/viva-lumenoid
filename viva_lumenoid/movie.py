"""Coherent simulation-movie clips: finely-sampled, unwrapped, drift-removed.

The network's beads diffuse ~sqrt(4*D*dt) per step; with the analysis sampling
(dt ~8 LJ time) a bead moves ~6 a between frames in a ~20 a box, so a movie made
from those snapshots looks like it teleports. A *coherent* movie needs (1) a fine
frame spacing so per-frame displacement is a small fraction of a bead spacing,
(2) periodic-image UNWRAPPING so a bead crossing the box edge doesn't jump a full
box width, and (3) removal of the whole-network centre-of-mass DRIFT (the system
is unanchored, so its COM random-walks) so the motion you watch is the network
deforming, not the box sliding.

``capture_clip`` runs the real CollagenNetworkProcess: a coarse (uncaptured)
pre-assembly to build a crosslinked network, then ``n_frames`` captured at a fine
``dt`` under one protocol mode. The frames it returns have the same shape the
movie figure consumes (positions, box, atom_types, sigma, strain, phase, t) plus
a ``trace`` for the stress backdrop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .params import CollagenParams

# mode -> (make_on, break_on, strain_rate, phase label)
_MODES = {
    "assemble": (1.0, 1.0, 0.0, "assemble"),
    "relax":    (0.0, 0.0, 0.0, "relax"),
    "stretch":  (0.0, 0.0, None, "stretch"),   # rate filled from params/arg
    "remodel":  (0.0, 1.0, 0.0, "hold_viscous"),
}


@dataclass
class ClipResult:
    frames: list = field(default_factory=list)
    trace: dict = field(default_factory=dict)
    mode: str = ""
    dt: float = 0.0
    summary: str = ""


def _unwrap_and_detrend(frames: list, unwrap: bool = True) -> list:
    """Make the clip's motion continuous (in place): remove whole-network COM
    drift always, and — for a STATIC box — unwrap periodic-image jumps so a bead
    crossing an edge doesn't teleport a box width.

    ``unwrap`` is left off for the stretch mode: its box deforms with a shifting
    origin (``fix deform remap``), so a box-relative unwrap is unreliable; the raw
    remapped coordinates there are already continuous (≈0 real wraps per frame),
    needing only COM-drift removal.
    """
    prev_wrapped = None
    unwrapped = None
    com0 = None
    for fr in frames:
        pos = np.asarray(fr["positions"], dtype=float).copy()
        box = np.asarray(fr["box"][:2], dtype=float)
        xy = pos[:, :2]
        if unwrap:
            if prev_wrapped is None:
                unwrapped = xy.copy()
            else:
                d = xy - prev_wrapped
                d -= box * np.round(d / box)       # minimum-image step
                unwrapped = unwrapped + d
            prev_wrapped = xy.copy()
            xy = unwrapped
        com = xy.mean(axis=0)
        if com0 is None:
            com0 = com.copy()
        pos[:, :2] = xy - (com - com0)             # remove whole-network drift
        fr["positions"] = pos.round(3)
    return frames


def capture_clip(params: CollagenParams | None = None, mode: str = "stretch",
                 n_frames: int = 80, dt: float = 0.15,
                 pre_assemble_steps: int | None = None,
                 strain_rate: float = 0.012) -> ClipResult:
    """Finely-sampled, coherent clip of the real network under one mode."""
    if mode not in _MODES:
        raise ValueError(f"unknown mode {mode!r}; one of {list(_MODES)}")
    p = params or CollagenParams()
    make, brk, rate, phase = _MODES[mode]
    if rate is None:
        rate = strain_rate
    # "assemble" shows the mesh forming, so start from little/no pre-assembly
    pre = (0 if mode == "assemble" else
           (pre_assemble_steps if pre_assemble_steps is not None else p.assemble_steps))

    from process_bigraph import allocate_core
    from .collagen import CollagenNetworkProcess
    proc = CollagenNetworkProcess(config={"params": p.to_dict()}, core=allocate_core())
    proc.initial_state()

    # coarse pre-assembly (NOT captured) — build the crosslinked starting network
    if pre > 0:
        for _ in range(max(1, int(pre * p.timestep / 5.0))):
            proc.update({"strain_rate": 0.0, "make_on": 1.0, "break_on": 1.0}, 5.0)

    frames = []
    t = 0.0
    for _ in range(n_frames):
        last = proc.update({"strain_rate": rate, "make_on": make, "break_on": brk}, dt)
        t += dt
        frames.append({
            "t": t, "strain": float(last["strain"]),
            "sigma": float(last["sigma_inplane"]),
            "n_crosslinks": int(last["n_crosslinks"]), "phase": phase,
            "positions": np.asarray(last["positions"], dtype=float),
            "atom_types": np.asarray(last["atom_types"]),
            "atom_ids": np.asarray(last["atom_ids"]),
            "bonds": last["bonds"],  # [bond_type, atom_id_1, atom_id_2]
            "box": list(last["box_dimensions"]),
        })
    proc.close()
    _unwrap_and_detrend(frames, unwrap=(mode != "stretch"))

    trace = {"t": [f["t"] for f in frames], "sigma": [f["sigma"] for f in frames],
             "strain": [f["strain"] for f in frames],
             "n_crosslinks": [f["n_crosslinks"] for f in frames],
             "phase": [f["phase"] for f in frames]}
    med_step = _median_step(frames)
    return ClipResult(
        frames=frames, trace=trace, mode=mode, dt=dt,
        summary=(f"{mode} clip · {n_frames} frames @ dt={dt} LJ · "
                 f"median step {med_step:.2f} a (coherent)"))


def _median_step(frames) -> float:
    steps = []
    for a, b in zip(frames[:-1], frames[1:]):
        pa = np.asarray(a["positions"])[:, :2]
        pb = np.asarray(b["positions"])[:, :2]
        if pa.shape == pb.shape:
            steps.append(np.median(np.hypot(*(pb - pa).T)))
    return float(np.mean(steps)) if steps else 0.0
