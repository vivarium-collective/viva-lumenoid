"""Render each new study's figure to workspace/studies/<slug>/viz/<name>.html and
wire a `visualizations:` block into its study.yaml. Run:
    python scripts/gen_study_figures.py
"""
import sys, warnings, pathlib
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from viva_lumenoid.viz import save_html
from viva_lumenoid.study_figures import STUDY_FIGURES, driving_schema_figure

ROOT = pathlib.Path(__file__).resolve().parent.parent
STUD = ROOT / "workspace" / "studies"
INV = ROOT / "workspace" / "investigations" / "basement-membrane-v1"

TITLES = {
    "r1_regime": "Confinement regimes: cell height vs BM stiffness × relaxation (Fig 1G reproduction)",
    "r2_rheology": "Collagen IV network assembly + stress (small real run)",
    "r3_supply": "Crosslink supply vs remodelling (small real run)",
    "r4_sheet": "Relaxed 2D epithelial sheet (real tyssue run)",
    "r5_monolayer": "Explicit 3D cell confinement (real tyssue run)",
    "r6_intervention": "Intervention ordering on cell height (Fig 7 reproduction)",
    "e1_cortical_tension": "Cortical tension vs myosin contractility (real LAMMPS)",
    "e2_force_transfer": "Actin → adhesion → collagen BM force transfer (schematic)",
    "e3_lumenoid": "Closed 3D lumenoid shell (real tyssue run)",
    "e4_feedback": "Growth↔turnover↔relaxation↔confinement loop (schematic)",
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
    for slug, (name, fn, kind) in STUDY_FIGURES.items():
        vdir = STUD / slug / "viz"
        vdir.mkdir(parents=True, exist_ok=True)
        try:
            fig = fn()
            save_html(fig, str(vdir / f"{name}.html"), TITLES[name])
            results.append((slug, name, kind, "ok"))
        except Exception as e:
            results.append((slug, name, kind, f"FAIL {type(e).__name__}: {e}"))
    for slug, name, kind, status in results:
        print(f"  [{kind:9s}] {slug:38s} {name}.html  -> {status}")
    return results


if __name__ == "__main__":
    main()
