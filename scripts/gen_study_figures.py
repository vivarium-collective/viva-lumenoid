"""Render each new study's figure to workspace/studies/<slug>/viz/<name>.html and
wire a `visualizations:` block into its study.yaml. Run:
    python scripts/gen_study_figures.py
"""
import sys, warnings, pathlib
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from viva_lumenoid.viz import save_html
from viva_lumenoid.study_figures import (
    STUDY_FIGURES, EXTRA_STUDY_FIGURES, COMPANION_FIGURES, V5_EXTRA_FIGURES,
    driving_schema_figure,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
STUD = ROOT / "workspace" / "studies"
INV = ROOT / "workspace" / "investigations" / "basement-membrane-v1"

TITLES = {
    "r1_regime": "Confinement regimes: cell height vs BM stiffness × relaxation (Fig 1G reproduction)",
    "r2_rheology": "Collagen IV network assembly + stress (small real run)",
    "r3_supply": "Crosslink supply vs remodelling (small real run)",
    "r4_confinement": "R4 main — 2D sheet static substrate vs R1 confinement benchmark (observed)",
    "r4_sheet": "R4 supplement — 12-cell planar Tyssue mesh thumbnail (engineered-diagnostic, z=0)",
    "r4_bm_hypothesis": "R4 hypothesis — BM-stiffness-dependent confinement (coupling not in composite)",
    "r5_monolayer": "R5 main — closed 3D monolayer render + per-cell area distribution (engineered-diagnostic)",
    "r5_volume_hist": "R5 supplement — cell-volume distribution appendix (engineered-diagnostic)",
    "r5_bm_hypothesis": "R5 hypothesis — matched BM controls (coupling not in composite)",
    "r6_intervention": "Intervention ordering on cell height (Fig 7 reproduction)",
    "e1_cortical_tension": "E1 — LJ-patch tension: tunability + seed/size sensitivity (engineered-diagnostic)",
    "e1_tension_components": "E1 (supplement) — pressure components of the LJ-patch tension",
    "e2_force_transfer": "E2 — force-transfer coupling unit test (engineered-diagnostic)",
    "e2_bm_response": "E2 (supplement) — actin→adhesion→collagen BM bounded smoke",
    "e3_lumenoid": "E3 — closed 3D lumenoid shell: geometry verification (engineered-diagnostic)",
    "e3_lumen_dynamics": "E3 (supplement) — lumen/BM state static at rest (gap: no filling drive)",
    "e4_feedback": "E4 — coupling-control time courses: two-way / one-way / uncoupled (engineered-diagnostic)",
    "e4_subcycle_convergence": "E4 (supplement) — BM subcycling convergence",
    "bm_v4_faithful_angles": "Junction angles formed during assembly stabilise the modulus (real LAMMPS)",
    "bm_v5_porosity": "Simulated pore size reaches the corneal-EM reference at density (real geometry)",
    "bm_v5_bundling": "The NC1/7S-only network under-bundles — no lateral-association bond (real geometry)",
}


def main():
    results = []
    # Investigation-level DRIVING SCHEMA (the Barrientos three-model framework).
    (INV / "viz").mkdir(parents=True, exist_ok=True)
    try:
        save_html(driving_schema_figure(), str(INV / "viz" / "driving_schema.html"),
                  "Driving schema — collagen IV turnover → BM relaxation → cell shape")
        results.append(("[investigation]", "driving_schema", "schema", "ok"))
    except Exception as e:
        results.append(("[investigation]", "driving_schema", "schema", f"FAIL {type(e).__name__}: {e}"))
    all_figs = [(slug, name, fn, kind) for slug, (name, fn, kind) in STUDY_FIGURES.items()]
    all_figs += list(EXTRA_STUDY_FIGURES)
    all_figs += [(slug, name, fn, kind) for slug, (name, fn, kind) in COMPANION_FIGURES.items()]
    all_figs += [(slug, name, fn, "data") for slug, items in V5_EXTRA_FIGURES.items()
                 for name, fn in items]
    for slug, name, fn, kind in all_figs:
        vdir = STUD / slug / "viz"
        vdir.mkdir(parents=True, exist_ok=True)
        try:
            fig = fn()
            save_html(fig, str(vdir / f"{name}.html"), TITLES.get(name, name))
            results.append((slug, name, kind, "ok"))
        except Exception as e:
            results.append((slug, name, kind, f"FAIL {type(e).__name__}: {e}"))
    # Extra MAIN figures for a study that has more than one (v5: the split story).
    for slug, extras in V5_EXTRA_FIGURES.items():
        vdir = STUD / slug / "viz"
        vdir.mkdir(parents=True, exist_ok=True)
        for name, fn in extras:
            try:
                save_html(fn(), str(vdir / f"{name}.html"), TITLES[name])
                results.append((slug, name, "data", "ok"))
            except Exception as e:
                results.append((slug, name, "data", f"FAIL {type(e).__name__}: {e}"))
    for slug, name, kind, status in results:
        print(f"  [{kind:9s}] {slug:38s} {name}.html  -> {status}")
    return results


if __name__ == "__main__":
    main()
