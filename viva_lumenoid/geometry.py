"""Geometric readouts off a real collagen IV network snapshot: porosity and
bundling — the spec's two "readouts to reproduce" (has: geometrical structure).

Both are computed from the network's bead positions + box (no bond topology
needed), under the periodic box, and converted to nanometres through the bead
spacing a. They are *readouts*, not inputs: the model is accountable to the
pore-size and bundling comparators in :mod:`viva_lumenoid.comparators`.

  * **Porosity** — the pore-size distribution. Rods are rasterised into the
    periodic plane; the Euclidean distance transform of the void gives, at every
    empty point, the radius of the largest empty disc centred there. The local
    maxima of that field are the maximal inscribed pores (classic
    maximal-covering-radius pore sizing). Pore *diameter* = 2 × radius.

  * **Bundling** — protomers per laterally-associated strand. Two rods bundle
    when their (minimum-image) midpoints sit within a lateral cutoff set by the
    measured cell-deposited nearest-neighbour spacing (≈59.5 nm ≈ 0.5 a) AND run
    nearly parallel. Connected components of that graph are strands; the strand
    size distribution is compared to the 5–7 protomers/strand comparator. The
    released NC1/7S-only model has no lateral-association bond, so it is expected
    to UNDER-bundle — a spec-named failure the readout makes visible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _rods_from_positions(positions, box):
    """(midpoints, angles[0,π), lengths) for each 2-bead rod, min-image unwrapped.

    Returns rods whose min-image length is sane (drops any degenerate pair).
    """
    pos = np.asarray(positions, dtype=float)[:, :2]
    lx, ly = float(box[0]), float(box[1])
    n = (len(pos) // 2) * 2
    a = pos[0:n:2]
    b = pos[1:n:2]
    # minimum-image displacement a->b (rods may straddle the periodic boundary)
    d = b - a
    d[:, 0] -= lx * np.round(d[:, 0] / lx)
    d[:, 1] -= ly * np.round(d[:, 1] / ly)
    length = np.hypot(d[:, 0], d[:, 1])
    mid = a + 0.5 * d
    mid[:, 0] = np.mod(mid[:, 0], lx)
    mid[:, 1] = np.mod(mid[:, 1], ly)
    ang = np.mod(np.arctan2(d[:, 1], d[:, 0]), np.pi)  # undirected orientation
    good = length > 1e-6
    return mid[good], ang[good], length[good], d[good], a[good], (lx, ly)


# --------------------------------------------------------------------------- #
# porosity
# --------------------------------------------------------------------------- #
@dataclass
class PorosityReadout:
    pore_diam_a: np.ndarray            # pore diameters in units of a (σ)
    pore_diam_nm: np.ndarray           # pore diameters in nm
    median_nm: float
    mean_nm: float
    p90_nm: float
    occupied_fraction: float           # areal coverage by rod material
    grid: np.ndarray = field(repr=False, default=None)     # distance field (a)
    extent_a: tuple = (0.0, 0.0)


def porosity(positions, box, bead_spacing_nm: float = 114.0,
             grid_px: int = 220, samples_per_rod: int = 6) -> PorosityReadout:
    """Pore-size distribution via the maximal-inscribed-disc (distance-transform)
    method, under the periodic box."""
    from scipy import ndimage

    mid, ang, length, d, a, (lx, ly) = _rods_from_positions(positions, box)
    px = grid_px
    py = max(1, int(round(grid_px * ly / lx)))
    dx, dy = lx / px, ly / py

    mask = np.zeros((py, px), dtype=bool)
    # rasterise each rod as a short polyline of sample points
    ts = np.linspace(0.0, 1.0, samples_per_rod)
    for a0, dd in zip(a, d):
        xs = a0[0] + ts * dd[0]
        ys = a0[1] + ts * dd[1]
        ix = np.mod((xs / dx).astype(int), px)
        iy = np.mod((ys / dy).astype(int), py)
        mask[iy, ix] = True
    occupied = float(mask.mean())

    # periodic distance transform: tile 3x3, transform, crop centre tile
    tiled = np.tile(mask, (3, 3))
    dist = ndimage.distance_transform_edt(~tiled, sampling=(dy, dx))  # in a-units
    core = dist[py:2 * py, px:2 * px]

    # local maxima of the distance field = maximal inscribed pore radii
    mx = ndimage.maximum_filter(core, size=max(3, px // 40))
    peaks = (core == mx) & (core > max(dx, dy))
    radii_a = core[peaks]
    if radii_a.size == 0:
        radii_a = np.array([core.max()]) if core.size else np.array([0.0])
    diam_a = np.sort(2.0 * radii_a)
    diam_nm = diam_a * bead_spacing_nm
    return PorosityReadout(
        pore_diam_a=diam_a, pore_diam_nm=diam_nm,
        median_nm=float(np.median(diam_nm)), mean_nm=float(np.mean(diam_nm)),
        p90_nm=float(np.percentile(diam_nm, 90)),
        occupied_fraction=occupied, grid=core, extent_a=(lx, ly))


# --------------------------------------------------------------------------- #
# bundling
# --------------------------------------------------------------------------- #
@dataclass
class BundlingReadout:
    strand_sizes: np.ndarray           # protomers per strand (component sizes)
    size_hist: dict                    # {size: count}
    mean_size: float
    max_size: int
    frac_bundled: float                # fraction of rods in a strand of >= 2
    frac_in_5_7: float                 # fraction of rods in strands of size 5-7
    lateral_cutoff_a: float


def bundling(positions, box, bead_spacing_nm: float = 114.0,
             nn_spacing_nm: float = 59.5, angle_tol_deg: float = 22.0
             ) -> BundlingReadout:
    """Protomers per laterally-associated strand (connected components of the
    near-parallel, laterally-close rod graph)."""
    from scipy.spatial import cKDTree
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    mid, ang, length, d, a, (lx, ly) = _rods_from_positions(positions, box)
    nrod = len(mid)
    if nrod == 0:
        return BundlingReadout(np.array([]), {}, 0.0, 0, 0.0, 0.0, 0.0)
    cutoff_a = nn_spacing_nm / bead_spacing_nm        # ≈0.52 a (cell-deposited)
    ang_tol = np.deg2rad(angle_tol_deg)

    tree = cKDTree(mid, boxsize=[lx, ly])
    pairs = tree.query_pairs(r=cutoff_a, output_type="ndarray")
    rows, cols = [], []
    for i, j in pairs:
        dth = abs(ang[i] - ang[j])
        dth = min(dth, np.pi - dth)                    # undirected angle diff
        if dth <= ang_tol:
            rows.append(i); cols.append(j)
    if rows:
        n = nrod
        data = np.ones(len(rows))
        g = csr_matrix((data, (rows, cols)), shape=(n, n))
        n_comp, labels = connected_components(g, directed=False)
    else:
        labels = np.arange(nrod)
    sizes = np.bincount(labels)
    hist = {}
    for s in sizes:
        hist[int(s)] = hist.get(int(s), 0) + 1
    frac_bundled = float((sizes[sizes >= 2].sum()) / nrod)
    in57 = sizes[(sizes >= 5) & (sizes <= 7)].sum()
    return BundlingReadout(
        strand_sizes=np.sort(sizes)[::-1], size_hist=hist,
        mean_size=float(sizes.mean()), max_size=int(sizes.max()),
        frac_bundled=frac_bundled, frac_in_5_7=float(in57 / nrod),
        lateral_cutoff_a=cutoff_a)
