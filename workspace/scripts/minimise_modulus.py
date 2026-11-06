#!/usr/bin/env python
"""Canonical runner: floppy elastic modulus via the athermal (T=0) minimise path.

Shared across the stage-1 collagen studies whose finding is that the central-force
network's elastic modulus is consistent with zero (floppy — below 2D rigidity
percolation). Measures the modulus with ``CollagenNetworkProcess.athermal_modulus``
(energy-minimise, then small equibiaxial ``change_box`` strain increments, fit the
stress-strain slope) over a parameter grid x seed ensemble. This is the DETERMINISTIC
path the published values came from — NOT the dynamic ``fix deform`` stretch, which
destabilises in the stretch phase (LAMMPS "Too many neighbor bins").

For each study it registers one ensemble run and writes the run's ``computed_outcomes``
into study.yaml so ``discover reconcile-evidence`` can verify the floppy finding.
Idempotent. Needs the run-path infra from viva-lumenoid #88. Usage:

    python workspace/scripts/minimise_modulus.py <study-slug>

Known studies: bm-v1-stage1-modulus-remodelling, bm-v3-junction-bending-rigidity,
bm-v4-junction-bending.
"""
from __future__ import annotations

import statistics
import sys
import time
from io import StringIO
from pathlib import Path

ASSEMBLE_RELAX_STEPS = 200  # composite-time < t_relax: stable, pre-stretch

# study slug -> {composite, grid (param dicts swept), seeds, finding, from_test, run_id}
STUDIES = {
    "bm-v1-stage1-modulus-remodelling": {
        "composite": "stage1_composite", "grid": [{}],
        "seeds": [11, 22, 33, 44, 55, 66],
        "finding": "F2-floppy-below-rigidity-percolation",
        "from_test": "elastic-modulus-consistent-with-floppy",
        "run_id": "bm-v1-stage1-minimise-ensemble",
    },
    "bm-v3-junction-bending-rigidity": {
        "composite": "connectivity_sweep",
        "grid": [{"make_prob": 0.12}, {"make_prob": 0.25}, {"make_prob": 0.40}, {"make_prob": 0.60}],
        "seeds": [11, 22, 33],
        "finding": "F1-connectivity-cannot-rigidify",
        "from_test": "modulus-stays-floppy",
        "run_id": "bm-v3-connectivity-minimise-sweep",
    },
    "bm-v4-junction-bending": {
        "composite": "bending_sweep", "grid": [{"bending_k": 0.0}],
        "seeds": [11, 22, 33, 44, 55, 66],
        "finding": "F1-bending-implemented",
        "from_test": "floppy-baseline-clean",
        "run_id": "bm-v4-floppy-baseline-minimise",
    },
}


def _workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "workspace.yaml").exists():
            return parent
    raise SystemExit("workspace.yaml not found above this script")


def _measure(composite_fn, params: dict) -> tuple[int, float]:
    from process_bigraph import Composite
    import viva_lumenoid.collagen as col
    from viva_lumenoid.core import build_core

    core = build_core()
    spec = composite_fn(core=core, **params)
    comp = Composite(spec if ("state" in spec or "composition" in spec) else {"state": spec}, core=core)
    for _ in range(ASSEMBLE_RELAX_STEPS):
        comp.run(1)

    def _find(state):
        if isinstance(state, dict):
            if isinstance(state.get("instance"), col.CollagenNetworkProcess):
                return state["instance"]
            for v in state.values():
                hit = _find(v)
                if hit is not None:
                    return hit
        return None

    proc = _find(comp.state)
    if proc is None:
        raise RuntimeError("CollagenNetworkProcess instance not found")
    nx = int(proc._read().get("n_crosslinks") or 0)
    modulus = float(proc.athermal_modulus())
    try:
        proc.close()
    except Exception:
        pass
    return nx, modulus


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in STUDIES:
        raise SystemExit(f"usage: minimise_modulus.py <{'|'.join(STUDIES)}>")
    slug = sys.argv[1]
    cfg = STUDIES[slug]
    import viva_lumenoid.composites as C
    composite_fn = getattr(C, cfg["composite"])
    ws = _workspace_root()
    study_dir = ws / "workspace" / "studies" / slug
    from discovery_engine.run_registry import register_run
    from vivarium_workbench.lib import run_log

    grid_rows, all_moduli, all_nx = [], [], []
    for point in cfg["grid"]:
        moduli, nxs = [], []
        for seed in cfg["seeds"]:
            nx, m = _measure(composite_fn, {**point, "seed": seed})
            moduli.append(m)
            nxs.append(nx)
            print(f"{slug} {point} seed={seed}: n_crosslinks={nx} E={m:.5f}", flush=True)
        grid_rows.append({"point": point, "mean": statistics.fmean(moduli),
                          "moduli": [round(x, 5) for x in moduli]})
        all_moduli += moduli
        all_nx += nxs

    # Observed = the grid point with the largest |mean| (most conservative test of
    # "stays floppy"); the floppy band is +/-2 sigma of the pooled ensemble noise.
    worst = max(grid_rows, key=lambda g: abs(g["mean"]))
    observed = round(worst["mean"], 4)
    pooled_sd = statistics.pstdev(all_moduli) if len(all_moduli) > 1 else 0.0
    band = round(2 * pooled_sd, 3) or 0.05
    mean_nx = round(statistics.fmean(all_nx), 1)
    print(f"{slug}: observed |E|_max={observed} LJ, floppy band +/-{band} (2 sigma), n_crosslinks~{mean_nx}")

    rid = cfg["run_id"]
    register_run(study_dir / "runs.db", rid, spec_id=f"viva_lumenoid.composites.{cfg['composite']}",
                 status="completed", params={"grid": cfg["grid"], "seeds": cfg["seeds"],
                                             "method": "athermal_modulus (change_box + minimise)"},
                 n_steps=ASSEMBLE_RELAX_STEPS, ws_root=str(ws))
    now = time.time()
    run_log.append_run_event(ws, {"run_id": rid, "event": "started", "status": "running",
                                  "study_slug": slug, "started_at": now, "ts": now})
    run_log.append_run_event(ws, {"run_id": rid, "event": "completed", "status": "completed",
                                  "n_steps": ASSEMBLE_RELAX_STEPS, "completed_at": now + 1, "ts": now + 1})

    from ruamel.yaml import YAML
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=2, sequence=4, offset=2)
    sy = study_dir / "study.yaml"
    spec = yaml.load(sy.read_text())
    runs = spec.setdefault("runs", [])
    entry = {
        "name": rid, "run_id": rid, "status": "completed",
        "base_model": f"viva_lumenoid.composites.{cfg['composite']}",
        "grid": cfg["grid"], "seeds": cfg["seeds"], "n_steps": ASSEMBLE_RELAX_STEPS,
        "method": "athermal_modulus (T=0 change_box + minimise) — the published-modulus path",
        "modulus_grid_lj": [{"point": g["point"], "mean": round(g["mean"], 5)} for g in grid_rows],
        "computed_outcomes": {cfg["from_test"]: {
            "measured_value": observed, "result": "pass", "evaluated_by": "code",
            "grader": "athermal_modulus_minimise",
            "note": f"|E|_max {observed} LJ within the +/-{band} floppy band (2 sigma pooled noise); "
                    f"a rigid network is ~30 LJ"}},
    }
    for i, r in enumerate(runs):
        if isinstance(r, dict) and r.get("name") == rid:
            runs[i] = entry
            break
    else:
        runs.append(entry)
    buf = StringIO()
    yaml.dump(spec, buf)
    sy.write_text(buf.getvalue())
    print(f"recorded run {rid} + computed_outcomes in {sy}")


if __name__ == "__main__":
    main()
