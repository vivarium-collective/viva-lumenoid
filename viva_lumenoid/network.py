"""Generate the initial coarse-grained collagen IV network as a LAMMPS data file.

One collagen IV protomer = one two-bead rod (spec: "one protomer = one two-bead
rod, the NC1 end binds 1 partner and the 7S end binds 3"). The rod lies in the
slab plane (quasi-2D); flexibility and crosslinking are added by the input
script at runtime. This module only lays down the un-crosslinked rods — the
crosslinks form dynamically during the stage-1 "assemble" phase via
``fix bond/create``.

atom types: 1 = NC1 end bead, 2 = 7S end bead
bond types: 1 = intra-rod (present here), 2 = NC1-NC1, 3 = 7S-7S (formed later)
"""
from __future__ import annotations

import numpy as np

from .params import CollagenParams


def build_network_data(params: CollagenParams) -> str:
    """Return LAMMPS data-file text for ``n_rods`` two-bead rods in a slab.

    Deterministic given ``params.seed`` so a study run is reproducible.
    """
    rng = np.random.default_rng(params.seed)
    n = params.n_rods
    L = params.box_xy
    half = params.bond_length / 2.0
    zmid = params.slab_thickness / 2.0

    # Rod centers uniformly in the xy plane; a thin z jitter keeps it 3D but
    # quasi-2D. In-plane orientation is uniform on the circle.
    cx = rng.uniform(0.0, L, n)
    cy = rng.uniform(0.0, L, n)
    cz = zmid + rng.uniform(-0.1, 0.1, n)
    theta = rng.uniform(0.0, np.pi, n)
    dx = half * np.cos(theta)
    dy = half * np.sin(theta)

    atoms = []   # (id, mol, type, x, y, z)
    bonds = []   # (id, type, a1, a2)
    aid = 0
    for i in range(n):
        a1 = aid + 1  # NC1 end, type 1
        a2 = aid + 2  # 7S end,  type 2
        atoms.append((a1, i + 1, 1, cx[i] - dx[i], cy[i] - dy[i], cz[i]))
        atoms.append((a2, i + 1, 2, cx[i] + dx[i], cy[i] + dy[i], cz[i]))
        bonds.append((i + 1, 1, a1, a2))  # intra-rod bond, type 1
        aid += 2

    n_atoms = len(atoms)
    n_bonds = len(bonds)

    lines: list[str] = []
    lines.append("LAMMPS data file - viva-lumenoid collagen IV network (v1 clean-room)")
    lines.append("")
    lines.append(f"{n_atoms} atoms")
    lines.append(f"{n_bonds} bonds")
    lines.append("2 atom types")
    lines.append("3 bond types")
    lines.append("")
    lines.append(f"0.0 {L:.6f} xlo xhi")
    lines.append(f"0.0 {L:.6f} ylo yhi")
    lines.append(f"0.0 {params.slab_thickness:.6f} zlo zhi")
    lines.append("")
    lines.append("Masses")
    lines.append("")
    lines.append("1 1.0")
    lines.append("2 1.0")
    lines.append("")
    lines.append("Atoms  # molecular")
    lines.append("")
    for (a, mol, t, x, y, z) in atoms:
        lines.append(f"{a} {mol} {t} {x:.6f} {y:.6f} {z:.6f}")
    lines.append("")
    lines.append("Bonds")
    lines.append("")
    for (b, t, a1, a2) in bonds:
        lines.append(f"{b} {t} {a1} {a2}")
    lines.append("")
    return "\n".join(lines)
