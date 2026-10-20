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
        # BM realization (P3 interchangeable alternatives, never additive):
        #   "explicit" = the LAMMPS collagen network; "reduced" = an effective
        #   viscoelastic BM (E_eff + relaxation), no LAMMPS. Controls fall out of
        #   the gains: strain_gain=0 -> uncoupled; tension_gain=0 -> one-way.
        "bm_mode": {"_type": "string", "_default": "explicit"},
        "bm_modulus": {"_type": "float", "_default": 50.0},      # reduced BM elastic modulus
        "bm_relaxation": {"_type": "float", "_default": 10.0},   # reduced BM stress-relaxation time
        # P3 coupling controls (an explicit enum, because bigraph config drops
        # 0.0-valued gains): "two-way" (full), "one-way" (BM driven, no reaction
        # fed back), "uncoupled" (BM inert — the epithelium doesn't drive it).
        "coupling_mode": {"_type": "string", "_default": "two-way"},
        # Subcycle the BM this many steps per epithelial macro-step (plan section 6:
        # subcycled BM equilibration). Advances BM stability without shrinking the
        # exchange interval.
        "bm_substeps": {"_type": "integer", "_default": 1},
        # Epithelial PREFERRED growth (the 2026-10-20 review's constraint): basal
        # displacement should be driven by a growing preferred cell geometry, not by
        # contractility alone. "none" (default -> backward-compatible, no growth),
        # "linear" or "exponential" ramp the preferred cell volume + basal area so
        # the REALIZED basal area emerges from the mechanics (it is NOT hard-coded to
        # the target). Growth is NOT cell division (no cells are added).
        "growth_mode": {"_type": "string", "_default": "none"},
        "growth_rate": {"_type": "float", "_default": 0.0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self._t = 0.0
        self._A_prev = None
        self._es = None
        self._bm = None
        self._bm_strain = 0.0   # reduced-BM accumulated strain
        self._bm_rest = 0.0     # reduced-BM relaxing rest strain (Maxwell element)
        self._grow_v = None     # preferred cell-volume growth driver
        self._grow_a = None     # preferred basal-area growth driver
        self._A_base = None     # realized basal area at growth onset (for folds)

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
        # Preferred-growth drivers (division OFF): ramp the preferred cell volume +
        # basal area from their baseline (1.0). "none" leaves them constant.
        from .growth import PreferredGrowth
        self._grow_v = PreferredGrowth(1.0, mode=c["growth_mode"], rate=c["growth_rate"])
        self._grow_a = PreferredGrowth(1.0, mode=c["growth_mode"], rate=c["growth_rate"])
        # Explicit (LAMMPS) BM only; the reduced BM is a cheap analytic element.
        if c["bm_mode"] == "explicit":
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
            "preferred_area_fold": "overwrite[float]",  # target basal-area growth (A0/A0_base)
            "realized_area_fold": "overwrite[float]",   # realized basal-area growth (A/A_base)
        }

    def update(self, state, interval):
        self._build()
        c = self.config

        # 0. epithelial PREFERRED growth: ramp the preferred cell volume + basal
        #    area so the basal displacement is growth-driven. The realized basal
        #    area emerges from the vertex model + BM reaction below (not hard-coded).
        if c["growth_mode"] != "none" and c["growth_rate"] != 0.0:
            self._es.eptm.cell_df["prefered_vol"] = self._grow_v.target(self._t)
            self._es.eptm.face_df.loc[self._basal_mask, "prefered_area"] = \
                self._grow_a.target(self._t)

        # 1. step the epithelium; read its basal BM-shell surface area.
        self._es.update({"behaviors": [], "global_time": self._t}, interval)
        self._t += interval
        A = float(self._es.eptm.settings.get("bm_surface_area", 0.0))
        if self._A_base is None:
            self._A_base = A or 1.0
        rate = ((A - self._A_prev) / self._A_prev / interval
                if self._A_prev else 0.0)
        self._A_prev = A

        # 2. the basal strain drives the BM; read its stress (explicit or reduced).
        #    "uncoupled" control: the epithelium does NOT drive the BM.
        strain_rate_bm = 0.0 if c["coupling_mode"] == "uncoupled" else c["strain_gain"] * rate
        nsub = max(1, int(c["bm_substeps"]))
        sub_dt = interval / nsub
        if c["bm_mode"] == "reduced":
            # Maxwell viscoelastic effective BM, subcycled.
            for _ in range(nsub):
                self._bm_strain += strain_rate_bm * sub_dt
                self._bm_rest += (self._bm_strain - self._bm_rest) / max(c["bm_relaxation"], 1e-9) * sub_dt
            sigma = c["bm_modulus"] * (self._bm_strain - self._bm_rest)
        else:
            sigma = 0.0
            for _ in range(nsub):
                bm_out = self._bm.update({"strain_rate": strain_rate_bm,
                                          "make_on": 1.0, "break_on": 1.0}, sub_dt)
                sigma = float(bm_out.get("sigma_inplane", 0.0))

        # 3. feed the BM reaction back as an added basal contractility (tension).
        #    "one-way" control: no reaction fed back to the epithelium.
        tension = 0.0 if c["coupling_mode"] == "one-way" else c["tension_gain"] * sigma
        self._es.eptm.face_df.loc[self._basal_mask, "contractility"] = (
            c["base_contractility"] + tension)

        return {
            "basal_area": A,
            "basal_strain_rate": rate,
            "bm_sigma": sigma,
            "basal_tension": tension,
            "lumen_pressure": float(self._es.eptm.settings.get("lumen_pressure", 0.0)),
            "preferred_area_fold": self._grow_a.target(self._t),
            "realized_area_fold": (A / self._A_base) if self._A_base else 1.0,
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
