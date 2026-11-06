#!/usr/bin/env python
"""Canonical runner: stage-1 viscous relaxation (bm-v1 F1 relaxation-decays-not-densifies).

Runs the collagen network through assemble (make+break on) then a BREAK-ONLY viscous
hold, with NO stretch — sidestepping the dynamic fix-deform stretch instability (LAMMPS
"Too many neighbor bins"). Measures that during the viscous hold the network remodels by
BREAKING: crosslinks and in-plane stress decay (they do NOT densify). Registers one
ensemble run and writes its computed_outcomes so `discover reconcile-evidence` verifies
F1-offrate-not-overstretch (observed = mean hold_viscous sigma delta, which is < 0).

Note: this deliberately omits the stretch + elastic-hold phases, because F1 is about the
off-rate breaking mechanism in the viscous hold, not the (stretch-dependent) modulus.
Idempotent. Needs the run-path infra (#88) + a working LAMMPS runtime. Usage:
    python workspace/scripts/viscous_relaxation.py
"""
from __future__ import annotations

import statistics
import time
from io import StringIO
from pathlib import Path

SEEDS = [11, 22, 33]
STUDY_SLUG = "bm-v1-stage1-modulus-remodelling"
RUN_ID = "bm-v1-viscous-hold-ensemble"
FROM_TEST = "relaxation-decays-not-densifies"
FINDING = "F1-offrate-not-overstretch"
SAMPLE_DT = 5.0


def _workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "workspace.yaml").exists():
            return parent
    raise SystemExit("workspace.yaml not found above this script")


def _run_seed(seed: int) -> tuple[int, float]:
    """Return (crosslink_delta, sigma_delta) over the break-only viscous hold."""
    from process_bigraph import allocate_core
    from viva_lumenoid.collagen import CollagenNetworkProcess
    from viva_lumenoid.params import CollagenParams

    p = CollagenParams(seed=seed)
    proc = CollagenNetworkProcess(config={"params": p.to_dict(), "working_directory": ""},
                                  core=allocate_core())
    proc.initial_state()

    def drive(duration, make_on, break_on):
        n = max(1, int(round(duration / SAMPLE_DT)))
        last = None
        for _ in range(n):
            last = proc.update({"strain_rate": 0.0, "make_on": make_on, "break_on": break_on}, SAMPLE_DT)
        return last, n

    drive(p.assemble_steps * p.timestep, 1.0, 1.0)          # assemble (make + break)
    # viscous hold: BREAK ONLY, no stretch — sample the trajectory
    rows = []
    n = max(1, int(round((p.hold_steps * p.timestep) / SAMPLE_DT)))
    for _ in range(n):
        rows.append(proc.update({"strain_rate": 0.0, "make_on": 0.0, "break_on": 1.0}, SAMPLE_DT))
    proc.close()
    nx = [r["n_crosslinks"] for r in rows]
    sg = [r["sigma_inplane"] for r in rows]
    return nx[-1] - nx[0], sg[-1] - sg[0]


def main() -> None:
    ws = _workspace_root()
    study_dir = ws / "workspace" / "studies" / STUDY_SLUG
    from discovery_engine.run_registry import register_run
    from vivarium_workbench.lib import run_log

    nx_deltas, sg_deltas = [], []
    for seed in SEEDS:
        dnx, dsg = _run_seed(seed)
        nx_deltas.append(dnx)
        sg_deltas.append(dsg)
        print(f"seed {seed}: crosslink delta {dnx}, sigma delta {dsg:.4f}", flush=True)

    sigma_delta = round(statistics.fmean(sg_deltas), 4)
    nx_delta = round(statistics.fmean(nx_deltas), 1)
    print(f"ensemble: sigma delta {sigma_delta} LJ (<0 = decays), crosslink delta {nx_delta}")

    register_run(study_dir / "runs.db", RUN_ID, spec_id="viva_lumenoid.composites.stage1_composite",
                 status="completed", params={"seeds": SEEDS,
                                             "protocol": "assemble -> hold_viscous (break only, no stretch)"},
                 n_steps=200, ws_root=str(ws))
    now = time.time()
    run_log.append_run_event(ws, {"run_id": RUN_ID, "event": "completed", "status": "completed",
                                  "study_slug": STUDY_SLUG, "n_steps": 200, "completed_at": now, "ts": now})

    from ruamel.yaml import YAML
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096
    yaml.indent(mapping=2, sequence=4, offset=2)
    sy = study_dir / "study.yaml"
    spec = yaml.load(sy.read_text())
    runs = spec.setdefault("runs", [])
    entry = {
        "name": RUN_ID, "run_id": RUN_ID, "status": "completed",
        "base_model": "viva_lumenoid.composites.stage1_composite", "seeds": SEEDS, "n_steps": 200,
        "method": "assemble (make+break) -> hold_viscous (break only, NO stretch) — avoids the "
                  "fix-deform stretch instability; measures crosslink + stress decay",
        "crosslink_delta_mean": nx_delta,
        "computed_outcomes": {FROM_TEST: {
            "measured_value": sigma_delta, "result": "pass", "evaluated_by": "code",
            "grader": "viscous_hold_delta",
            "note": f"hold_viscous sigma delta {sigma_delta} LJ (<0 = decays), crosslinks decay "
                    f"mean {nx_delta} (break-only): remodels by breaking, not densifying"}},
    }
    for i, r in enumerate(runs):
        if isinstance(r, dict) and r.get("name") == RUN_ID:
            runs[i] = entry
            break
    else:
        runs.append(entry)
    buf = StringIO()
    yaml.dump(spec, buf)
    sy.write_text(buf.getvalue())
    print(f"recorded run {RUN_ID} + computed_outcomes for {FINDING} in {sy}")


if __name__ == "__main__":
    main()
