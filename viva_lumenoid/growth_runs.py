"""Coupled growth experiments for the meeting-aligned revision (R4, R5).

These are the "simplest two-way feedback experiment" the 2026-10-20 review asked
for: a PREFERRED (target) cell geometry grows, the realized geometry lags it
because a viscoelastic BM resists expansion, and we measure the realized
expansion + cell height across BM stiffness / relaxation regimes. Growth is NOT
cell division (no cells are added).

Two realizations, both built on the shared ``growth`` primitives:

- ``run_sheet_growth_2d`` — a reduced-order 2D CROSS-SECTION: a row of prism
  cells, each a ``PreferredGrowth`` target + a per-cell ``ReducedBM`` reaction.
  Height is RECONSTRUCTED from a fixed-volume assumption (h = V0 / A). This is the
  effective-BM alternative; the resolved vertex-model 2D sheet
  (``tyssue_2d_sheet`` / CrossSectionGeometry) stays a static mesh at the engine
  level (its area does not respond to prefered_area — a documented limitation), so
  the causal confinement experiment is run here instead and labelled as such.

- ``run_monolayer_growth_3d`` — REAL Tyssue: a 3D monolayer (MonolayerGeometry)
  whose preferred CELL VOLUME is ramped so basal area + cell height emerge from the
  vertex model, with a ``ReducedBM`` reaction fed back as added basal contractility.
  Cell height is the emergent apical-basal vertex z-extent (not reconstructed).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .growth import PreferredGrowth, ReducedBM


# --------------------------------------------------------------------------
# R4 — reduced-order 2D cross-section of prism cells
# --------------------------------------------------------------------------

@dataclass
class SheetGrowthResult:
    t: np.ndarray                 # time
    a_pref: np.ndarray            # preferred (target) basal area A0(t), shared
    a_mean: np.ndarray            # realized mean basal area A(t)
    a_cells: np.ndarray           # realized per-cell area, shape (steps+1, n_cells)
    height_mean: np.ndarray       # reconstructed mean cell height h(t) = V0/A
    stress_mean: np.ndarray       # mean BM reaction stress
    x: np.ndarray                 # per-cell x position (for the spatial row)
    stiffness: np.ndarray         # per-cell BM stiffness (heterogeneity)
    height_cells_final: np.ndarray  # per-cell reconstructed height at the end
    gap_final: float              # (A0 - A_mean)/A0 at the end — the confinement
    mode: str = ""
    label: str = ""


def run_sheet_growth_2d(
    *,
    n_cells: int = 12,
    mode: str = "exponential",
    rate: float = 0.04,
    bm_stiffness: float = 1.0,
    bm_relaxation: float = 10.0,
    stiffness_gradient: float = 0.0,
    mobility: float = 0.3,
    steps: int = 60,
    dt: float = 0.2,
    v0: float = 1.0,
    a0: float = 1.0,
    label: str = "",
) -> SheetGrowthResult:
    """Grow a row of prism cells against a viscoelastic BM.

    ``bm_stiffness=0`` is the free / no-BM control. ``stiffness_gradient`` tilts the
    per-cell stiffness linearly across the row (0 → uniform), so a BM-stiffness
    gradient produces a spatial height gradient. Height is reconstructed as
    ``v0 / area`` (fixed-volume assumption, declared).
    """
    x = np.linspace(0.0, 1.0, n_cells)
    # per-cell BM stiffness: uniform + an optional linear gradient across the row
    stiff = bm_stiffness * (1.0 + stiffness_gradient * (x - 0.5))
    stiff = np.clip(stiff, 0.0, None)
    growth = PreferredGrowth(a0, mode=mode, rate=rate)
    bms = [ReducedBM(a0, stiffness=float(s), relaxation=bm_relaxation) for s in stiff]

    A = np.full(n_cells, a0, dtype=float)
    t_arr, a_pref, a_mean, a_cells, h_mean, s_mean = [], [], [], [], [], []
    t = 0.0
    for _ in range(steps + 1):
        A_pref = growth.target(t)
        stress = np.array([bm.react(A[i], dt) for i, bm in enumerate(bms)])
        t_arr.append(t)
        a_pref.append(A_pref)
        a_mean.append(float(A.mean()))
        a_cells.append(A.copy())
        h_mean.append(float(np.mean(v0 / A)))
        s_mean.append(float(stress.mean()))
        # realized area relaxes toward the (growing) target, resisted by BM reaction
        A = A + mobility * ((A_pref - A) - stress) * dt
        A = np.clip(A, 1e-6, None)
        t += dt

    a_mean_arr = np.array(a_mean)
    a_pref_arr = np.array(a_pref)
    gap_final = float((a_pref_arr[-1] - a_mean_arr[-1]) / a_pref_arr[-1])
    return SheetGrowthResult(
        t=np.array(t_arr), a_pref=a_pref_arr, a_mean=a_mean_arr,
        a_cells=np.array(a_cells), height_mean=np.array(h_mean),
        stress_mean=np.array(s_mean), x=x, stiffness=stiff,
        height_cells_final=v0 / np.array(a_cells)[-1],
        gap_final=gap_final, mode=mode, label=label)


# --------------------------------------------------------------------------
# R5 — real Tyssue 3D monolayer, preferred-volume growth
# --------------------------------------------------------------------------

@dataclass
class MonolayerGrowthResult:
    t: np.ndarray
    v_pref: np.ndarray            # preferred (target) total cell volume
    v_real: np.ndarray            # realized total cell volume
    basal_area: np.ndarray        # mean basal face area
    height_mean: np.ndarray       # mean per-cell apical-basal z-extent
    stress: np.ndarray            # BM reaction stress fed back
    heights_final: np.ndarray = field(default_factory=lambda: np.array([]))
    gap_final: float = 0.0
    mode: str = ""
    label: str = ""


def _build_monolayer(core=None, vol_elasticity: float = 3.0):
    from bigraph_schema import allocate_core
    from vivarium_tyssue.processes.eulersolver import EulerSolver
    from .tyssue_coupling import register_tyssue
    tcore = register_tyssue(core or allocate_core())
    es = EulerSolver(config=dict(
        name="MonoGrowth", eptm="workspace/datasets/monolayer_box.hf5",
        tissue_type="Monolayer",
        parameters={"cell_df": {"vol_elasticity": float(vol_elasticity),
                                "prefered_vol": 1.0, "is_alive": 1.0},
                    "face_df": {"area_elasticity": 1.0, "prefered_area": 1.0,
                                "contractility": 0.05, "is_alive": 1.0},
                    "edge_df": {"line_tension": 0.0, "is_active": 1.0},
                    "vert_df": {"viscosity": 1.0, "is_alive": 1.0}},
        geom="MonolayerGeometry",
        effectors=["CellVolumeElasticity", "FaceAreaElasticity",
                   "FaceContractility", "LineTension"],
        ref_effector="FaceAreaElasticity", factory="model_factory",
        settings={"threshold_length": 0.03}, auto_reconnect=False,
        max_displacement=0.2, bounds={}, output_columns={}, maps={},
        backend="python", substeps=1), core=tcore)
    return es


def _cell_heights(eptm) -> np.ndarray:
    e, v = eptm.edge_df, eptm.vert_df
    hs = []
    for _c, grp in e.groupby("cell"):
        vids = np.unique(np.concatenate([grp["srce"].values, grp["trgt"].values]))
        zz = v.loc[vids, "z"].values
        hs.append(float(zz.max() - zz.min()))
    return np.array(hs)


def run_monolayer_growth_3d(
    *,
    mode: str = "exponential",
    rate: float = 0.03,
    bm_stiffness: float = 0.3,
    bm_relaxation: float = 10.0,
    base_contractility: float = 0.05,
    vol_elasticity: float = 3.0,
    steps: int = 24,
    dt: float = 0.05,
    prerelax: int = 12,
    label: str = "",
) -> MonolayerGrowthResult:
    """Ramp the preferred cell VOLUME and basal AREA of a real Tyssue monolayer
    (isotropic preferred growth); a ReducedBM reaction on the basal area is fed
    back as added basal contractility that resists basal spreading.

    Without a BM (``bm_stiffness=0``, the free control) the growth spreads the
    cells basally; a stiff/slow BM confines the basal spreading, so (under the
    near-incompressible ``vol_elasticity``) the volume is driven into cell HEIGHT.
    ``vol_elasticity`` is the incompressibility knob (higher = more volume-
    conserving). Cell height is the emergent apical-basal vertex z-extent (NOT
    reconstructed); the realized volume emerges from the vertex model.
    """
    es = _build_monolayer(vol_elasticity=vol_elasticity)
    e = es.eptm
    basal_mask = (e.face_df["segment"] == "basal")

    def v_real():
        return float(np.nansum(e.cell_df["vol"].values))

    def a_basal():
        return float(e.face_df.loc[basal_mask, "area"].mean())

    nc = int(e.Nc)
    t = 0.0
    # pre-relax to equilibrium (the mesh starts over-inflated) so the growth run is
    # dominated by growth, not by the initial settling transient.
    for _ in range(prerelax):
        es.update({"behaviors": [], "global_time": t}, dt)
        t += dt
    # baseline the targets on the SETTLED realized geometry so they track upward
    # from where the mesh equilibrated (not the over-inflated initial state).
    v_start_per_cell = v_real() / nc
    a_start = a_basal()
    grow_v = PreferredGrowth(v_start_per_cell, mode=mode, rate=rate)
    grow_a = PreferredGrowth(a_start, mode=mode, rate=rate)
    bm = ReducedBM(a_start, stiffness=bm_stiffness, relaxation=bm_relaxation)

    rec_t, rec_vp, rec_vr, rec_ab, rec_h, rec_s = [], [], [], [], [], []
    for _ in range(steps + 1):
        e.cell_df["prefered_vol"] = grow_v.target(t)
        e.face_df.loc[basal_mask, "prefered_area"] = grow_a.target(t)
        a = a_basal()
        stress = bm.react(a, dt)
        # BM reaction resists basal spreading -> added basal contractility (confines)
        e.face_df.loc[basal_mask, "contractility"] = base_contractility + max(stress, 0.0)
        rec_t.append(t)
        rec_vp.append(grow_v.target(t) * nc)
        rec_vr.append(v_real())
        rec_ab.append(a)
        rec_h.append(float(np.nanmean(_cell_heights(e))))
        rec_s.append(float(stress))
        es.update({"behaviors": [], "global_time": t}, dt)
        t += dt

    vp = np.array(rec_vp)
    vr = np.array(rec_vr)
    gap = float((vp[-1] - vr[-1]) / vp[-1]) if vp[-1] else 0.0
    return MonolayerGrowthResult(
        t=np.array(rec_t), v_pref=vp, v_real=vr, basal_area=np.array(rec_ab),
        height_mean=np.array(rec_h), stress=np.array(rec_s),
        heights_final=_cell_heights(e), gap_final=gap, mode=mode, label=label)
