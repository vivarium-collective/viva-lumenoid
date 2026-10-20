"""Registered composite generators for the basement-membrane studies.

Each study's ``baseline.composite`` names one of these. They are **thin,
config-driven** selections over a single shared builder
(:func:`viva_lumenoid.composites.builders.collagen_network_spec`), which wires the
two reusable workspace processes — ``StagedStretchController`` →
``CollagenNetworkProcess`` → emitter. The composites differ ONLY by
``CollagenParams`` and a protocol / emit choice; none duplicate the wiring, so a
new collagen composite is one more thin generator here, not a copy of the graph.
``core_extensions=[register_viva_lumenoid]`` builds each against a core that knows
the two processes.
"""
from __future__ import annotations

from viva_superpowers.composite_generator import composite_generator

from ..params import CollagenParams
from ..collagen import register_viva_lumenoid
from .builders import collagen_network_spec, lammps_model_spec, tyssue_eptm_spec


def _register_actomyosin(core):
    """Lazy core extension: import the actomyosin plugin (→ pbg_lammps) only at
    composite-BUILD time, so this module imports cleanly when the optional
    viva-lammps engine isn't installed (CI without the engines extra)."""
    from ..actomyosin import register_actomyosin as _r
    return _r(core)


def _register_tyssue(core):
    """Lazy core extension: register the vivarium-tyssue engine only at
    composite-BUILD time (keeps this module import-clean without the optional
    tyssue engine)."""
    from ..tyssue_coupling import register_tyssue as _r
    return _r(core)


def _register_e2(core):
    """Lazy core extension for the E2 coupling: the LAMMPS model seam (cortex) +
    the collagen network + controller + the basal-adhesion linker process."""
    from ..actomyosin import register_actomyosin
    from ..collagen import register_viva_lumenoid
    from ..adhesion import register_adhesion
    core = register_actomyosin(core)   # LammpsModelProcess + actomyosin-cortex plugin
    core = register_viva_lumenoid(core)  # CollagenNetworkProcess + StagedStretchController
    core = register_adhesion(core)       # BasalAdhesion
    return core


def _register_coupling(core):
    """Lazy core extension: the LAMMPS<->Tyssue coupling process (owns both engines
    internally; the composite core only needs the coupling node + an emitter)."""
    from ..multiscale_coupling import register_coupling
    return register_coupling(core)


def _params(**kw) -> CollagenParams:
    """CollagenParams from the generator's Configure kwargs (unknown keys dropped)."""
    allowed = set(CollagenParams().to_dict())
    return CollagenParams(**{k: v for k, v in kw.items() if k in allowed})


# Configurable params exposed in the workbench "Configure" panel. Shared across
# the collagen composites; bending_sweep overrides the bending_k default below.
_COMMON_PARAMS = {
    "n_rods": {"type": "integer", "default": 200,
               "description": "Number of two-bead collagen IV rods."},
    "box_xy": {"type": "float", "default": 20.0,
               "description": "In-plane box size (units of the bead spacing a)."},
    "make_prob": {"type": "float", "default": 0.12,
                  "description": "Per-attempt crosslink formation probability (MP proxy)."},
    "off_rate": {"type": "float", "default": 0.02,
                 "description": "Force-independent crosslink off-rate (per LJ time)."},
    "bending_k": {"type": "float", "default": 0.0,
                  "description": "Junction bending stiffness (0 = floppy central-force network)."},
    "seed": {"type": "integer", "default": 12345, "description": "RNG seed."},
}

_BENDING_PARAMS = dict(_COMMON_PARAMS)
_BENDING_PARAMS["bending_k"] = {
    "type": "float", "default": 50.0,
    "description": "Junction bending stiffness (fix restrain, as-formed angles); >0 rigidifies.",
}


@composite_generator(
    name="stage1_composite",
    description="Stage 1 — staged protocol (assemble → relax → stretch FROZEN → "
                "hold elastic → hold viscous). Separates a reproducible elastic "
                "modulus from a crosslink-turnover remodelling time.",
    parameters=_COMMON_PARAMS,
    core_extensions=[register_viva_lumenoid],
)
def stage1_composite(core=None, **kw) -> dict:
    # Staged stage-1: the network is loaded FROZEN (chemistry off during stretch).
    return collagen_network_spec(_params(**kw), emit="rheology",
                                 chemistry_during_stretch=False)


@composite_generator(
    name="stage2_sweep",
    description="Stage 2 / R2 — growth rheology: the collagen IV network on a "
                "growing (biaxially stretching) substrate with the crosslink "
                "chemistry ON during the stretch; the σ(ε̇) sweep runs this at a "
                "range of strain rates.",
    parameters=_COMMON_PARAMS,
    core_extensions=[register_viva_lumenoid],
)
def stage2_sweep(core=None, **kw) -> dict:
    # Growth rheology: remodel WHILE strained (chemistry on during stretch).
    return collagen_network_spec(_params(**kw), emit="rheology",
                                 chemistry_during_stretch=True)


@composite_generator(
    name="connectivity_sweep",
    description="Rigidity (bm-v3, adversarial) — the CENTRAL-FORCE counterfactual: "
                "bending_k forced to 0, crosslink connectivity (z) swept to test "
                "rigidity percolation. A control that should NOT reproduce the "
                "released angle-stabilized model.",
    parameters=_COMMON_PARAMS,
    core_extensions=[register_viva_lumenoid],
)
def connectivity_sweep(core=None, **kw) -> dict:
    # Central-force control: force bending off regardless of the Configure value.
    return collagen_network_spec(_params(**{**kw, "bending_k": 0.0}), emit="rheology")


@composite_generator(
    name="bending_sweep",
    description="FP1 (bm-v4) — the collagen IV network WITH junction bending "
                "(released NC1 value, as-formed angle restraints created during "
                "assembly). The athermal modulus is swept vs bending_k.",
    parameters=_BENDING_PARAMS,
    core_extensions=[register_viva_lumenoid],
)
def bending_sweep(core=None, **kw) -> dict:
    # Bending active (bending_k>0) with faithful as-formed angles.
    return collagen_network_spec(_params(**{**kw, "use_real_angles": True}),
                                 emit="rheology")


@composite_generator(
    name="porosity_bundling",
    description="Stage 5 (bm-v5) — the assembled collagen IV network read for its "
                "geometric readouts (pore-size distribution, protomers-per-strand) "
                "against the measured BM comparators; emits the assembly "
                "observables, not the stress trace.",
    parameters=_COMMON_PARAMS,
    core_extensions=[register_viva_lumenoid],
)
def porosity_bundling(core=None, **kw) -> dict:
    # Geometric readout of the assembled network.
    return collagen_network_spec(_params(**kw), emit="geometry")


# --- Extension campaign composites (reuse the engine seams) ---

_CORTEX_PARAMS = {
    "n": {"type": "integer", "default": 8,
          "description": "Cortex patch size (lattice replications per side)."},
    "contractility": {"type": "float", "default": 1.0,
                      "description": "Myosin activity proxy (LJ well depth); higher = more cortical tension."},
    "temperature": {"type": "float", "default": 0.2, "description": "Langevin temperature (LJ units)."},
    "seed": {"type": "integer", "default": 12345, "description": "RNG seed."},
}

_CORTEX_EMIT = {"cortical_tension": "float", "pressure_xx": "float",
                "pressure_yy": "float", "temperature": "float"}


@composite_generator(
    name="actomyosin_cortex",
    description="E1 (extension) — a coarse 2D actomyosin cortex on the viva-lammps "
                "model seam (LammpsModelProcess + the actomyosin-cortex plugin). "
                "Contractility (LJ ε, a myosin-activity proxy) tunes the cortical "
                "tension that will map onto a Tyssue cortical-energy parameter.",
    parameters=_CORTEX_PARAMS,
    core_extensions=[_register_actomyosin],
)
def actomyosin_cortex(core=None, **kw) -> dict:
    params = {k: v for k, v in kw.items() if k in _CORTEX_PARAMS}
    return lammps_model_spec("actomyosin-cortex", params, emit=_CORTEX_EMIT)


_LUMENOID_PARAMS = {
    "contractility": {"type": "float", "default": 0.05,
                      "description": "Apical/basal face contractility of the lumenoid cells."},
    "lumen_elasticity": {"type": "float", "default": 1.0,
                         "description": "Lumen pressure stiffness: P = lumen_elasticity*(V0 - V)."},
}


@composite_generator(
    name="closed_lumenoid",
    description="E3 (extension) — a closed 3D lumenoid on the vivarium-tyssue "
                "engine (EulerSolver over the merged LumenoidGeometry + lumenoid "
                "mesh). Tracks a Laplace lumen pressure on top of the enclosed "
                "lumen volume and exposes the basal surface as the basement-"
                "membrane coupling shell (the surface the collagen BM attaches to).",
    parameters=_LUMENOID_PARAMS,
    core_extensions=[_register_tyssue],
)
def closed_lumenoid(core=None, **kw) -> dict:
    contractility = float(kw.get("contractility", 0.05))
    lumen_elasticity = float(kw.get("lumen_elasticity", 1.0))
    parameters = {
        "cell_df": {"vol_elasticity": 1.0, "prefered_vol": 1.0, "is_alive": 1.0},
        "face_df": {"area_elasticity": 1.0, "prefered_area": 1.0,
                    "contractility": contractility, "is_alive": 1.0},
        "edge_df": {"line_tension": 0.0, "is_active": 1.0},
        "vert_df": {"viscosity": 1.0, "is_alive": 1.0},
    }
    return tyssue_eptm_spec(
        "Lumenoid", "workspace/datasets/lumenoid.hf5", "LumenoidGeometry", "Monolayer",
        effectors=["CellVolumeElasticity", "FaceAreaElasticity",
                   "FaceContractility", "LineTension"],
        ref_effector="FaceAreaElasticity", parameters=parameters,
        settings={"threshold_length": 0.03, "lumen_side": "basal",
                  "bm_side": "basal", "lumen_elasticity": lumen_elasticity},
    )


_SHEET_PARAMS = {
    "line_tension": {"type": "float", "default": 0.1,
                     "description": "Junction line tension of the 2D sheet."},
    "prefered_area": {"type": "float", "default": 1.0,
                      "description": "Target cell area."},
}


@composite_generator(
    name="tyssue_2d_sheet",
    description="R4 — coupled 2D epithelium on the vivarium-tyssue engine: a "
                "polygonal sheet relaxed with the merged CrossSectionGeometry "
                "(explicit per-cell height DOF for the prism/height relation). The "
                "homogeneous reference for the confinement response (vs R1).",
    parameters=_SHEET_PARAMS,
    core_extensions=[_register_tyssue],
)
def tyssue_2d_sheet(core=None, **kw) -> dict:
    # 2D sheet: area + line-tension only (FaceContractility's gradient needs 3D).
    parameters = {
        "face_df": {"area_elasticity": 1.0,
                    "prefered_area": float(kw.get("prefered_area", 1.0)), "is_alive": 1.0},
        "edge_df": {"line_tension": float(kw.get("line_tension", 0.1)), "is_active": 1.0},
        "vert_df": {"viscosity": 1.0, "is_alive": 1.0},
    }
    return tyssue_eptm_spec(
        "Sheet2D", "workspace/datasets/cross_section.hf5", "CrossSectionGeometry", "Sheet",
        effectors=["FaceAreaElasticity", "LineTension"],
        ref_effector="FaceAreaElasticity", parameters=parameters,
        settings={"threshold_length": 0.03},
    )


_MONOLAYER_PARAMS = {
    "contractility": {"type": "float", "default": 0.05,
                      "description": "Apical/basal face contractility."},
    "vol_elasticity": {"type": "float", "default": 1.0,
                       "description": "Cell volume elasticity (resists height change)."},
}


@composite_generator(
    name="tyssue_3d_monolayer",
    description="R5 — explicit 3D cell confinement: a Tyssue 3D monolayer "
                "(MonolayerGeometry) with apical/basal/lateral surfaces and "
                "cell-volume targets, so cell height + basal area emerge "
                "explicitly (no prism approximation).",
    parameters=_MONOLAYER_PARAMS,
    core_extensions=[_register_tyssue],
)
def tyssue_3d_monolayer(core=None, **kw) -> dict:
    parameters = {
        "cell_df": {"vol_elasticity": float(kw.get("vol_elasticity", 1.0)),
                    "prefered_vol": 1.0, "is_alive": 1.0},
        "face_df": {"area_elasticity": 1.0, "prefered_area": 1.0,
                    "contractility": float(kw.get("contractility", 0.05)), "is_alive": 1.0},
        "edge_df": {"line_tension": 0.0, "is_active": 1.0},
        "vert_df": {"viscosity": 1.0, "is_alive": 1.0},
    }
    return tyssue_eptm_spec(
        "Monolayer3D", "workspace/datasets/monolayer_box.hf5", "MonolayerGeometry", "Monolayer",
        effectors=["CellVolumeElasticity", "FaceAreaElasticity",
                   "FaceContractility", "LineTension"],
        ref_effector="FaceAreaElasticity", parameters=parameters,
        settings={"threshold_length": 0.03},
    )


_E2_PARAMS = {
    "contractility": {"type": "float", "default": 1.5,
                      "description": "Cortex myosin-activity proxy (drives cortex tension)."},
    "k_on": {"type": "float", "default": 0.5, "description": "Basal-linker binding rate."},
    "k_off": {"type": "float", "default": 0.1, "description": "Basal-linker unbinding rate."},
    "coupling_gain": {"type": "float", "default": 1.0e-4,
                      "description": "Transmitted traction -> BM strain rate."},
    "n_rods": {"type": "integer", "default": 40,
               "description": "Collagen rods in the single-cell BM patch."},
}


@composite_generator(
    name="actin_adhesion_bm",
    description="E2 (extension) — actin-adhesion-collagen force transfer: an "
                "actomyosin cortex (LAMMPS seam) pulls through reversible basal "
                "linkers (BasalAdhesion) on a collagen IV BM patch "
                "(CollagenNetworkProcess). Measures traction, linker occupancy and "
                "the BM deformation on a single-cell patch.",
    parameters=_E2_PARAMS,
    core_extensions=[_register_e2],
)
def actin_adhesion_bm(core=None, **kw) -> dict:
    from ..params import CollagenParams
    cortex_params = {"n": 6, "contractility": float(kw.get("contractility", 1.5))}
    adhesion = {"k_on": float(kw.get("k_on", 0.5)), "k_off": float(kw.get("k_off", 0.1)),
                "coupling_gain": float(kw.get("coupling_gain", 1.0e-4)),
                "make_on": 1.0, "break_on": 1.0}
    bm_params = CollagenParams(n_rods=int(kw.get("n_rods", 40)), box_xy=10.0,
                               slab_thickness=2.0, assemble_steps=800, hold_steps=400).to_dict()
    emit = {"cortex_tension": "float", "traction": "float",
            "linker_occupancy": "float", "bm_sigma": "float", "bm_strain": "float"}
    return {
        "cortex": {
            "_type": "process", "address": "local:LammpsModelProcess",
            "config": {"model": "actomyosin-cortex", "params": cortex_params},
            "inputs": {},
            "outputs": {"cortical_tension": ["cortex_tension"]},
        },
        "adhesion": {
            "_type": "process", "address": "local:BasalAdhesion",
            "config": adhesion,
            "inputs": {"cortex_tension": ["cortex_tension"]},
            "outputs": {"traction": ["traction"], "linker_occupancy": ["linker_occupancy"],
                        "bm_strain_rate": ["bm_strain_rate"],
                        "make_on": ["make_on"], "break_on": ["break_on"]},
        },
        "bm": {
            "_type": "process", "address": "local:CollagenNetworkProcess",
            "config": {"params": bm_params},
            "inputs": {"strain_rate": ["bm_strain_rate"], "make_on": ["make_on"],
                       "break_on": ["break_on"]},
            "outputs": {"sigma_inplane": ["bm_sigma"], "strain": ["bm_strain"],
                        "n_crosslinks": ["bm_xlinks"]},
        },
        "emitter": {
            "_type": "step", "address": "local:ram-emitter",
            "config": {"emit": emit},
            "inputs": {k: [k] for k in emit},
            "outputs": {},
        },
    }


_COUPLED_PARAMS = {
    "bm_rods": {"type": "integer", "default": 40,
                "description": "Collagen rods in the explicit LAMMPS BM patch."},
    "strain_gain": {"type": "float", "default": 20.0,
                    "description": "Epithelial basal strain-rate -> BM strain-rate."},
    "tension_gain": {"type": "float", "default": 0.02,
                     "description": "BM stress -> added basal contractility (reaction force)."},
}

_COUPLED_EMIT = {"basal_area": "float", "basal_strain_rate": "float",
                 "bm_sigma": "float", "basal_tension": "float", "lumen_pressure": "float"}


@composite_generator(
    name="coupled_lumenoid",
    description="E4 / P3 capstone — bidirectional LAMMPS collagen-BM <-> Tyssue "
                "epithelium coupling: the lumenoid's basal BM-shell surface stretches "
                "an explicit LAMMPS collagen IV BM patch, whose stress feeds back as a "
                "basal tension on the epithelium. The force round-trip the multiscale "
                "campaign is built toward (explicit-BM realization).",
    parameters=_COUPLED_PARAMS,
    core_extensions=[_register_coupling],
)
def coupled_lumenoid(core=None, **kw) -> dict:
    cfg = {
        "eptm": "workspace/datasets/lumenoid.hf5",
        "bm_rods": int(kw.get("bm_rods", 40)),
        "strain_gain": float(kw.get("strain_gain", 20.0)),
        "tension_gain": float(kw.get("tension_gain", 0.02)),
        "base_contractility": 0.05,
    }
    return {
        "coupling": {
            "_type": "process", "address": "local:LammpsTyssueCoupling",
            "config": cfg,
            "inputs": {},
            "outputs": {k: [k] for k in _COUPLED_EMIT},
        },
        "emitter": {
            "_type": "step", "address": "local:ram-emitter",
            "config": {"emit": _COUPLED_EMIT},
            "inputs": {k: [k] for k in _COUPLED_EMIT},
            "outputs": {},
        },
    }


def _register_tissue_prism(core):
    """Lazy core extension for the reduced-order tissue baseline (R1)."""
    from ..tissue_prism import register_tissue_prism
    return register_tissue_prism(core)


_PRISM_PARAMS = {
    "bm_stiffness": {"type": "float", "default": 1.0,
                     "description": "BM area elasticity (stiffness)."},
    "bm_relaxation": {"type": "float", "default": 5.0,
                      "description": "BM relaxation time tau_bm."},
    "doubling_time": {"type": "float", "default": 5.0,
                      "description": "Proliferation timescale (cell doubling)."},
}

_PRISM_EMIT = {"basal_area": "float", "cell_height": "float",
               "bm_stress": "float", "confinement": "float"}


@composite_generator(
    name="tissue_prism_baseline",
    description="R1 — reduced-order homogeneous tissue mechanics: a fixed-volume "
                "prism cell whose basal area is set by proliferation against a "
                "viscoelastic BM (stiffness + relaxation). Cell height = volume/area; "
                "the paper's Fig 1 confinement regimes emerge from BM stiffness vs the "
                "relaxation/doubling ratio. The effective-BM reference for R4/R5.",
    parameters=_PRISM_PARAMS,
    core_extensions=[_register_tissue_prism],
)
def tissue_prism_baseline(core=None, **kw) -> dict:
    cfg = {k: v for k, v in kw.items() if k in _PRISM_PARAMS}
    return {
        "tissue": {
            "_type": "process", "address": "local:TissuePrismProcess",
            "config": cfg,
            "inputs": {},
            "outputs": {k: [k] for k in _PRISM_EMIT},
        },
        "emitter": {
            "_type": "step", "address": "local:ram-emitter",
            "config": {"emit": _PRISM_EMIT},
            "inputs": {k: [k] for k in _PRISM_EMIT},
            "outputs": {},
        },
    }


def _register_turnover_shape(core):
    """Lazy core extension for the turnover->shape pipeline (R6)."""
    from ..turnover_shape import register_turnover_shape
    return register_turnover_shape(core)


_R6_PARAMS = {
    "collagen_lifetime": {"type": "float", "default": 5.0,
                          "description": "Inferred collagen IV molecular lifetime."},
    "perlecan_factor": {"type": "float", "default": 1.0,
                        "description": "Perlecan modulation: >1 stabilizes (longer lifetime), <1 destabilizes."},
    "doubling_time": {"type": "float", "default": 5.0,
                      "description": "Growth-rate condition (cell doubling time)."},
}

_R6_EMIT = {"inferred_relaxation": "float", "basal_area": "float",
            "cell_height": "float", "confinement": "float"}


@composite_generator(
    name="turnover_shape_pipeline",
    description="R6 — turnover->relaxation->shape inference pipeline: an inferred "
                "collagen IV lifetime (modulated by perlecan) sets the BM relaxation "
                "time, which the reduced-order prism model turns into a predicted "
                "cell shape. Inferred (relaxation) and predicted (shape) are kept "
                "distinct; conditions are control / perlecan gain-loss / growth rate.",
    parameters=_R6_PARAMS,
    core_extensions=[_register_turnover_shape],
)
def turnover_shape_pipeline(core=None, **kw) -> dict:
    cfg = {k: v for k, v in kw.items() if k in _R6_PARAMS}
    return {
        "pipeline": {
            "_type": "process", "address": "local:TurnoverShapeProcess",
            "config": cfg,
            "inputs": {},
            "outputs": {k: [k] for k in _R6_EMIT},
        },
        "emitter": {
            "_type": "step", "address": "local:ram-emitter",
            "config": {"emit": _R6_EMIT},
            "inputs": {k: [k] for k in _R6_EMIT},
            "outputs": {},
        },
    }
