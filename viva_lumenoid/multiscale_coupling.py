"""LAMMPS collagen-BM  <->  Tyssue epithelium coupling (the P3 capstone).

The mechanical contract the whole campaign is built toward: the epithelial basal
geometry drives the basement membrane, and the BM's reaction stress feeds back on
the epithelium. This process owns BOTH engines and closes the loop each step:

  1. step the Tyssue epithelium (EulerSolver over a LumenoidGeometry) -> its basal
     BM-shell surface area changes;
  2. that basal area change is a strain rate imposed on the LAMMPS collagen IV BM
     patch (CollagenNetworkProcess), which remodels and returns a stress;
  3. the BM stress feeds back as an added basal contractility (tension) on the
     epithelium's basal faces — so a stiff/slow-relaxing BM confines the cells.

A reduced-order BM (an effective modulus) and this explicit LAMMPS BM are
interchangeable; this is the explicit-BM realization. v0 single-lumenoid patch,
bidirectional, demonstrating the force round-trip — not yet convergence-verified.
"""
from __future__ import annotations

from process_bigraph import Process


_EULER_EFFECTORS = ["CellVolumeElasticity", "FaceAreaElasticity",
                    "FaceContractility", "LineTension"]


class LammpsTyssueCoupling(Process):
    """Bidirectional collagen-BM (LAMMPS) <-> epithelium (Tyssue) coupling."""

    config_schema = {
        "eptm": {"_type": "string", "_default": "workspace/datasets/lumenoid.hf5"},
        "bm_rods": {"_type": "integer", "_default": 40},
        "strain_gain": {"_type": "float", "_default": 20.0},     # basal strain-rate -> BM strain-rate
        "tension_gain": {"_type": "float", "_default": 0.02},    # BM stress -> added basal contractility
        "base_contractility": {"_type": "float", "_default": 0.05},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self._t = 0.0
        self._A_prev = None
        self._es = None
        self._bm = None

    # --- lazy build of both engines (heavy; done on first use) ---
    def _build(self):
        if self._es is not None:
            return
        from vivarium_tyssue.processes.eulersolver import EulerSolver
        from .tyssue_coupling import register_tyssue
        from .collagen import CollagenNetworkProcess
        from .params import CollagenParams
        from bigraph_schema import allocate_core

        tcore = register_tyssue(allocate_core())
        c = self.config
        self._es = EulerSolver(config=dict(
            name="CoupledLumenoid", eptm=c["eptm"], tissue_type="Monolayer",
            parameters={"cell_df": {"vol_elasticity": 1.0, "prefered_vol": 1.0, "is_alive": 1.0},
                        "face_df": {"area_elasticity": 1.0, "prefered_area": 1.0,
                                    "contractility": c["base_contractility"], "is_alive": 1.0},
                        "edge_df": {"line_tension": 0.0, "is_active": 1.0},
                        "vert_df": {"viscosity": 1.0, "is_alive": 1.0}},
            geom="LumenoidGeometry", effectors=_EULER_EFFECTORS,
            ref_effector="FaceAreaElasticity", factory="model_factory",
            settings={"threshold_length": 0.03, "lumen_side": "basal",
                      "bm_side": "basal", "lumen_elasticity": 1.0},
            auto_reconnect=False, max_displacement=0.2, bounds={},
            output_columns={}, maps={}, backend="python", substeps=1),
            core=tcore)
        self._basal_mask = (self._es.eptm.face_df["segment"] == "basal")
        self._bm = CollagenNetworkProcess(
            config={"params": CollagenParams(n_rods=int(c["bm_rods"]), box_xy=10.0,
                                             slab_thickness=2.0, assemble_steps=600,
                                             hold_steps=300).to_dict()},
            core=allocate_core())
        self._bm.initial_state()

    def inputs(self):
        return {}

    def outputs(self):
        return {
            "basal_area": "overwrite[float]",        # epithelial basal BM-shell surface
            "basal_strain_rate": "overwrite[float]",  # how fast the epithelium stretches the BM
            "bm_sigma": "overwrite[float]",           # collagen BM stress (LAMMPS)
            "basal_tension": "overwrite[float]",      # BM reaction fed back as basal contractility
            "lumen_pressure": "overwrite[float]",
        }

    def update(self, state, interval):
        self._build()
        c = self.config

        # 1. step the epithelium; read its basal BM-shell surface area.
        self._es.update({"behaviors": [], "global_time": self._t}, interval)
        self._t += interval
        A = float(self._es.eptm.settings.get("bm_surface_area", 0.0))
        rate = ((A - self._A_prev) / self._A_prev / interval
                if self._A_prev else 0.0)
        self._A_prev = A

        # 2. the basal strain drives the collagen BM (LAMMPS); read its stress.
        bm_out = self._bm.update({"strain_rate": c["strain_gain"] * rate,
                                  "make_on": 1.0, "break_on": 1.0}, interval)
        sigma = float(bm_out.get("sigma_inplane", 0.0))

        # 3. feed the BM reaction back as an added basal contractility (tension).
        tension = c["tension_gain"] * sigma
        self._es.eptm.face_df.loc[self._basal_mask, "contractility"] = (
            c["base_contractility"] + tension)

        return {
            "basal_area": A,
            "basal_strain_rate": rate,
            "bm_sigma": sigma,
            "basal_tension": tension,
            "lumen_pressure": float(self._es.eptm.settings.get("lumen_pressure", 0.0)),
        }

    def close(self):
        bm = getattr(self, "_bm", None)
        if bm is not None:
            bm.close()
            self._bm = None


def register_coupling(core):
    """Core extension: make ``local:LammpsTyssueCoupling`` resolvable."""
    if "LammpsTyssueCoupling" not in core.link_registry:
        core.register_link("LammpsTyssueCoupling", LammpsTyssueCoupling)
    return core
