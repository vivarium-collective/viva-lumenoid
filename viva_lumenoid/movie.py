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
    drift always, and unwrap periodic-image jumps so a bead crossing an edge
    doesn't teleport a box width.

    The coherence step wraps every bead to its **nearest periodic image of the
    network centroid**, in box-fractional coordinates. This is what a DEFORMING
    box needs:

    - *Teleport-free.* A bead that crosses the periodic boundary is drawn at the
      image closest to the (smoothly moving) network, so it no longer jumps a
      full box width between frames — the artefact the reviewer saw as the
      stress-vs-strain movie "jumping across timepoints".
    - *Bounded.* Unlike a true unwrap (which accumulates the unbounded random-walk
      of a freely-diffusing floppy bead and blows the extent far outside the box),
      the min-image-about-centroid keeps the whole network within one box, so the
      camera stays framed as the substrate stretches.
    - *Strain-faithful.* Fractional coordinates are mapped back through *this*
      frame's box, so the cloud grows with the box and the strain reads off the
      geometry.

    Fractional (reduced) coordinates make this box-size-independent, so it is
    correct whether the box is static or deforming. ``unwrap=False`` keeps the raw
    coordinates (only COM-drift removed).
    """
    com0 = None
    for fr in frames:
        pos = np.asarray(fr["positions"], dtype=float).copy()
        box = np.asarray(fr["box"][:2], dtype=float)
        xy = pos[:, :2]
        if unwrap:
            frac = xy / box                        # reduced (box-fractional) coords
            com_frac = frac.mean(axis=0)
            rel = frac - com_frac
            rel -= np.round(rel)                   # nearest image of the centroid
            xy = (com_frac + rel) * box            # back to real coords in this box
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
    # Unwrap every mode (per-frame box handles the deforming stretch box too) so a
    # bead crossing the periodic boundary doesn't teleport a box width between frames.
    _unwrap_and_detrend(frames, unwrap=True)

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
