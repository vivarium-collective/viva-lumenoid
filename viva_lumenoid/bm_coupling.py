"""Explicit basement-membrane ↔ epithelium coupling as a Tyssue face energy.

The reviewer's #1 ask (2026-11-09, Saurabh Mogre): R4's "coupled growth" was 12
INDEPENDENT per-cell scalar Maxwell ODEs with no neighbour coupling — an
engine-and-wiring check, not a coupled sheet. A real coupled sheet needs the BM
tension carried continuously across cells through the vertex-model force balance.

This module supplies exactly that, as the coupling contract the investigation
specced: the BM is an **energy on each basal face** inside the tyssue vertex
solver —

    E_f    = (K_BM / 2) · (A_f − A_rest)² / A_rest          # per basal face
    T_f    = K_BM · (A_f − A_rest) / A_rest, clipped ≥ 0     # tension (buckles in
                                                              # compression → no
                                                              # restoring force)
    dA_rest/dt = (A_f − A_rest) / τ   (+ supply)             # slow turnover state

Because neighbouring faces **share vertices**, each face's BM tension enters the
same vertex force balance, so the BM tension is continuous across the sheet (the
per-cell Maxwell elements could not do this). In the homogeneous limit (uniform
faces, uniform K_BM, uniform A_rest) the sheet reduces to R1's prism law: a stiff
/ slow BM makes the realized basal area lag the preferred (growing) area, and —
at fixed cell volume — that confinement raises cell height (h = V0 / A).

`BMFaceAreaElasticity` plugs into vivarium-tyssue's `EulerSolver` (it is injected
into `EFFECTORS_MAP` by `register_bm_coupling`), so a composite can list it in its
`effectors`. `run_bm_coupled_sheet` drives the REAL tyssue vertex model (not a
0-D surrogate) through a preferred-area growth ramp against the BM, across the
no-BM / elastic / relaxing coupling modes and a BM-stiffness gradient, and is what
the R4 study now runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- tyssue compat shims (same ones register_tyssue applies) so this module can
# build a sheet standalone (tests, the run driver) without a full vivarium core.
_TYSSUE_READY = False


def _ensure_tyssue():
    global _TYSSUE_READY
    if _TYSSUE_READY:
        return
    try:
        from vivarium_tyssue._tyssue_compat import apply_tyssue_compat
        apply_tyssue_compat()
    except Exception:  # noqa: BLE001 — best-effort, matches register_tyssue
        pass
    from .tyssue_coupling import _patch_numpy_cross_2d
    _patch_numpy_cross_2d()
    _TYSSUE_READY = True


# --------------------------------------------------------------------------
# The BM face-energy effector
# --------------------------------------------------------------------------
def _bm_effector_class():
    """Build the effector class lazily (needs tyssue imported + patched)."""
    _ensure_tyssue()
    from tyssue.dynamics import effectors as _eff
    from tyssue.dynamics.effectors import AbstractEffector
    from tyssue.dynamics.planar_gradients import area_grad as area_grad2d
    from tyssue.dynamics.sheet_gradients import area_grad as area_grad3d
    from tyssue.utils import to_nd

    class BMFaceAreaElasticity(AbstractEffector):
        """Basement-membrane energy on each (basal) face: ½·K·(A−A₀)²/A₀ with the
        tension clipped ≥ 0 (a fibrous sheet buckles rather than pushing back in
        compression). Shared vertices make the tension continuous across cells."""

        dimensionless = False
        magnitude = "bm_stiffness"
        label = "BM face-area elasticity (clipped)"
        element = "face"
        specs = {
            "face": {
                "is_alive": 1,
                "area": 1.0,
                "bm_stiffness": 0.0,      # K_BM (0 = no BM / free control)
                "bm_rest_area": 1.0,      # A_rest (relaxes on turnover)
            }
        }
        spatial_ref = "bm_rest_area", _eff.units.area

        @staticmethod
        def energy(eptm):
            fd = eptm.face_df
            da = fd["area"] - fd["bm_rest_area"]
            a0 = fd["bm_rest_area"].clip(lower=1e-9)
            stretched = (da > 0).astype(float)       # no energy under compression
            return 0.5 * fd["bm_stiffness"] * fd["is_alive"] * stretched * da ** 2 / a0

        @staticmethod
        def gradient(eptm):
            fd = eptm.face_df
            da = fd["area"] - fd["bm_rest_area"]
            a0 = fd["bm_rest_area"].clip(lower=1e-9)
            # tension T_f = K (A − A₀)/A₀, clipped ≥ 0
            kg_ = (fd["bm_stiffness"] * fd["is_alive"] * da / a0).clip(lower=0.0)
            kg = to_nd(eptm.upcast_face(kg_), len(eptm.coords))
            if len(eptm.coords) == 2:
                grad_s, grad_t = area_grad2d(eptm)
            else:
                grad_s, grad_t = area_grad3d(eptm)
            grad_s = kg * grad_s
            grad_t = kg * grad_t
            grad_s.columns = ["g" + u for u in eptm.coords]
            grad_t.columns = ["g" + u for u in eptm.coords]
            return grad_s, grad_t

    return BMFaceAreaElasticity


# Module-level singleton so the class identity is stable across registrations.
_BM_EFFECTOR = None


def bm_effector():
    global _BM_EFFECTOR
    if _BM_EFFECTOR is None:
        _BM_EFFECTOR = _bm_effector_class()
    return _BM_EFFECTOR


def register_bm_coupling(core):
    """Core extension: register the tyssue engine AND inject the BM face-energy
    effector into vivarium-tyssue's ``EFFECTORS_MAP`` so a composite can list
    ``BMFaceAreaElasticity`` in its ``effectors``."""
    from .tyssue_coupling import register_tyssue
    core = register_tyssue(core)
    from vivarium_tyssue.maps import effectors_maps
    effectors_maps.EFFECTORS_MAP["BMFaceAreaElasticity"] = bm_effector()
    # EulerSolver reads the map object by reference at build time; also patch the
    # already-imported top-level alias if the maps package re-exported it.
    try:
        import vivarium_tyssue.maps as _m
        _m.EFFECTORS_MAP["BMFaceAreaElasticity"] = bm_effector()
    except Exception:  # noqa: BLE001
        pass
    return core


# --------------------------------------------------------------------------
# The real engine-coupled growth driver (replaces the 0-D per-cell surrogate)
# --------------------------------------------------------------------------
@dataclass
class BMCoupledResult:
    mode: str
    t: np.ndarray                   # macro time
    a_pref: np.ndarray              # preferred (target) basal area A0(t)
    a_mean: np.ndarray              # realized mean basal area A(t) (emergent)
    a_cells: np.ndarray             # per-cell realized area, (steps+1, n_faces)
    height_mean: np.ndarray         # mean cell height h = V0 / A (prism, fixed vol)
    bm_tension_mean: np.ndarray     # mean BM face tension (clipped ≥ 0)
    gap_final: float                # (A0 − A_mean)/A0 at the end — the confinement
    height_gain: float              # h_final / h_initial − 1 (thickening)
    x: np.ndarray                   # per-face x (spatial axis)
    stiffness: np.ndarray           # per-face BM stiffness (heterogeneity)
    area_final_cells: np.ndarray    # per-face realized area at the end
    height_final_cells: np.ndarray  # per-face height at the end
    frames: list | None = None      # per-step {t,x,y,area,height} (capture_frames)


def _build_sheet():
    """Load the real R4 cross-section mesh and return (sheet, PlanarGeometry)."""
    _ensure_tyssue()
    from tyssue import Sheet, PlanarGeometry
    from tyssue.io.hdf5 import load_datasets
    import importlib.resources as _  # noqa: F401
    from pathlib import Path

    here = Path(__file__).resolve()
    # datasets live at <workspace>/workspace/datasets/cross_section.hf5; walk up.
    for up in here.parents:
        cand = up / "workspace" / "datasets" / "cross_section.hf5"
        if cand.exists():
            dsets = load_datasets(str(cand))
            break
    else:
        raise FileNotFoundError("cross_section.hf5 not found above " + str(here))
    sh = Sheet("bm_sheet", dsets)
    sh.face_df["is_alive"] = 1
    sh.vert_df["viscosity"] = 1.0
    PlanarGeometry.update_all(sh)
    return sh, PlanarGeometry


def run_bm_coupled_sheet(
    *,
    mode: str = "relaxing",
    growth_mode: str = "exponential",
    rate: float = 0.04,
    bm_stiffness: float = 20.0,
    bm_relaxation: float = 12.0,
    stiffness_gradient: float = 0.0,
    line_tension: float = 0.02,
    area_elasticity: float = 1.0,
    v0: float = 1.0,
    vol_rate: float = 0.0,
    steps: int = 40,
    dt: float = 0.25,
    capture_frames: bool = False,
) -> BMCoupledResult:
    """Grow a REAL tyssue sheet's preferred basal area against an explicit BM face
    energy, letting the realized area + cell height emerge from the vertex model.

    ``mode``:
      * ``no-bm``   — bm_stiffness forced to 0 (free growth control).
      * ``elastic`` — BM on, rest area FROZEN (no turnover): purely elastic
        confinement (the realized area lags the growing preferred area and stays
        lagged).
      * ``relaxing``— BM on, rest area relaxes with ``bm_relaxation`` τ
        (viscoelastic): confinement partially relaxes as the BM remodels.

    ``stiffness_gradient`` > 0 tilts bm_stiffness across x (0 at the soft end,
    ``bm_stiffness·(1+gradient)`` at the stiff end) to test neighbour coupling: a
    continuous BM gives a SMOOTH area/height gradient across the shared vertices.
    """
    from tyssue.solvers.quasistatic import QSSolver
    from .growth import PreferredGrowth

    sh, G = _build_sheet()
    G.update_all(sh)
    n = sh.Nf
    a_init = float(sh.face_df["area"].mean())
    xc = sh.face_df["x"].to_numpy()
    rngx = (xc.max() - xc.min()) or 1.0

    sh.face_df["area_elasticity"] = area_elasticity
    sh.edge_df["line_tension"] = line_tension

    k_free = 0.0 if mode == "no-bm" else bm_stiffness
    if stiffness_gradient and mode != "no-bm":
        # Span soft -> stiff across x: the soft end is ~free (BM off), the stiff
        # end is bm_stiffness*(1+gradient). A continuous BM then gives a smooth
        # area/height gradient between them.
        frac = (xc - xc.min()) / rngx
        stiff = k_free * (1.0 + stiffness_gradient) * frac
    else:
        stiff = np.full(n, k_free)
    sh.face_df["bm_stiffness"] = stiff
    sh.face_df["bm_rest_area"] = a_init if mode != "no-bm" else 1e9

    model = _coupled_model()
    solver = QSSolver()
    grow = PreferredGrowth(a_init, mode=growth_mode, rate=rate)
    # Preferred cell volume can grow independently of basal area. When it does and
    # the BM confines basal spreading, the extra volume goes into cell HEIGHT
    # (h = V0(t) / A) — the absolute height-increase regime the reviewer asks for.
    grow_vol = PreferredGrowth(v0, mode=(growth_mode if vol_rate else "none"), rate=vol_rate)

    ts, aprefs, ameans, hmeans, tens, acells = [], [], [], [], [], []
    frames = [] if capture_frames else None
    v0_last = v0
    for i in range(steps + 1):
        t = i * dt
        a_pref = grow.target(t)
        v0_last = grow_vol.target(t)
        sh.face_df["prefered_area"] = a_pref
        if i > 0:
            # BM turnover: rest area relaxes toward the realized area (viscoelastic).
            if mode == "relaxing":
                A = sh.face_df["area"]
                sh.face_df["bm_rest_area"] = (
                    sh.face_df["bm_rest_area"]
                    + (A - sh.face_df["bm_rest_area"]) / max(bm_relaxation, 1e-9) * dt
                )
            G.update_all(sh)
            solver.find_energy_min(sh, G, model)
            G.update_all(sh)
        A = sh.face_df["area"].to_numpy()
        da = A - sh.face_df["bm_rest_area"].to_numpy()
        a0 = np.clip(sh.face_df["bm_rest_area"].to_numpy(), 1e-9, None)
        T = np.clip(stiff * da / a0, 0.0, None)
        ts.append(t); aprefs.append(a_pref); ameans.append(float(A.mean()))
        hmeans.append(float(np.mean(v0_last / A))); tens.append(float(T.mean()))
        acells.append(A.copy())
        if frames is not None:
            G.update_all(sh)
            fx = sh.face_df["x"].to_numpy(); fy = sh.face_df["y"].to_numpy()
            frames.append({"t": t, "x": fx.copy(), "y": fy.copy(),
                           "area": A.copy(), "height": (v0_last / A).copy()})

    acells = np.asarray(acells)
    ameans = np.asarray(ameans)
    hmeans = np.asarray(hmeans)
    a_final = acells[-1]
    gap_final = float((aprefs[-1] - ameans[-1]) / aprefs[-1])
    height_gain = float(hmeans[-1] / hmeans[0] - 1.0)
    return BMCoupledResult(
        mode=mode, t=np.asarray(ts), a_pref=np.asarray(aprefs), a_mean=ameans,
        a_cells=acells, height_mean=hmeans, bm_tension_mean=np.asarray(tens),
        gap_final=gap_final, height_gain=height_gain, x=xc, stiffness=stiff,
        area_final_cells=a_final, height_final_cells=v0_last / a_final,
        frames=frames,
    )


def _coupled_model():
    _ensure_tyssue()
    from tyssue.dynamics import effectors, factory
    return factory.model_factory(
        [effectors.FaceAreaElasticity, effectors.LineTension, bm_effector()],
        effectors.FaceAreaElasticity,
    )


def run_all_modes(**kw) -> dict:
    """Convenience: run no-bm / elastic / relaxing and return a dict of results."""
    return {m: run_bm_coupled_sheet(mode=m, **kw) for m in ("no-bm", "elastic", "relaxing")}
