"""fp-r2-figure2-panels: real runs backing the Barrientos Fig 2C/C'/D/E panels.

Reproduces the STRUCTURE / DIRECTION of Barrientos et al. 2026 Figure 2's collagen
IV network rheology (reduced units, NOT a pixel match to the PDF) from the REAL
collagen model. Drives the faithful density-calibrated NC1/7S network
(authors' density 3.0 rods/sigma^2, 12-sigma slab, released junction angles ---
the engine the stage2_sweep composite wires) through the paper's staged Fig-2D
"breakage off, then on" protocol via viva_lumenoid.stage1.run_stage1, which is
the CollagenNetworkProcess driven through assemble -> relax -> stretch ->
hold_elastic -> hold_viscous. It sweeps the crosslink off_rate (collagen
residence time 1/off_rate) over several seeds and captures, per run:

  * the full stress trace sigma(t), strain(t), n_crosslinks(t), phase  -> Fig 2C/C'
  * the fitted macroscopic stress-relaxation time tau (run_stage1's measure) -> Fig 2E
  * the elastic (tangent) modulus                                           -> Fig 2E
  * the viscous-hold (sigma, n_crosslinks) pairs                            -> Fig 2D

The per-run traces of ONE representative seed per off_rate (+ the per-rate
tau/modulus aggregates over all seeds) are baked into
``viva_lumenoid/data/fig2_panels.json`` so the figure
(study_figures.r2_fig2_panels_figure) renders fast + reproducibly, the same
baked-real-data pattern as r2_fig2e / the v4-v5 figure-review figures.

NO fabrication: every number is a measured run_stage1 output. If a panel does not
reproduce the published direction, the figure says so honestly rather than tuning.

Run:  PYTHONPATH=<worktree> .venv/bin/python \
        workspace/studies/bm-r2-collagen-network-rheology/sims/fig2_panels_run.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

# Faithful density-calibrated network (same geometry as the recorded dense sweep).
BOX_XY, DENSITY, SLAB = 10.0, 3.0, 12.0
N_RODS = int(round(DENSITY * BOX_XY * BOX_XY))  # 300
ASSEMBLE, HOLD = 12000, 8000
SAMPLE_DT = 5.0
# residence 100 / 50 / 25 / 12.5 LJ -- the turnover range that resolves cleanly at
# this size; the slow end (residence 200) is left to the recorded 4-seed sweep
# (it resolved at only 1/4 there), so this figure does not over-reach.
OFF_RATES = [0.01, 0.02, 0.04, 0.08]
SEEDS = [11, 22, 33]
REP_SEED = 11  # the representative seed whose full traces drive the curve panels

OUT_BAKED = ROOT / "viva_lumenoid" / "data" / "fig2_panels.json"


def one(off_rate, seed):
    from viva_lumenoid.params import CollagenParams
    from viva_lumenoid.stage1 import run_stage1
    r = run_stage1(CollagenParams(
        n_rods=N_RODS, box_xy=BOX_XY, slab_thickness=SLAB, bending_k=4.0,
        use_real_angles=True, assemble_steps=ASSEMBLE, hold_steps=HOLD,
        off_rate=off_rate, seed=seed), sample_dt=SAMPLE_DT)
    out = {
        "off_rate": off_rate, "seed": seed,
        "tau_lj": float(r.relaxation_time_lj),
        "noise_limited": bool(r.relaxation_noise_limited),
        "modulus_lj": float(r.elastic_modulus_lj),
        "n_crosslinks_assembled": int(r.n_crosslinks_assembled),
        "sigma_viscous_start": float(r.sigma_viscous_start),
        "sigma_viscous_end": float(r.sigma_viscous_end),
    }
    if seed == REP_SEED:
        out["trace"] = {
            "t": [float(x) for x in r.trace["t"]],
            "sigma": [float(x) for x in r.trace["sigma"]],
            "strain": [float(x) for x in r.trace["strain"]],
            "n_crosslinks": [int(x) for x in r.trace["n_crosslinks"]],
            "phase": list(r.trace["phase"]),
        }
    return out


def main():
    import numpy as np
    tasks = [(o, s) for o in OFF_RATES for s in SEEDS]
    raw = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(one, o, s): (o, s) for o, s in tasks}
        for f in as_completed(futs):
            try:
                raw.append(f.result())
            except Exception as e:  # noqa: BLE001
                o, s = futs[f]
                raw.append({"off_rate": o, "seed": s, "error": f"{type(e).__name__}: {e}"})

    # Representative-seed traces per off_rate (the curve panels).
    traces = {}
    for o in OFF_RATES:
        rep = next((x for x in raw if x.get("off_rate") == o and x.get("seed") == REP_SEED
                    and "trace" in x), None)
        if rep is not None:
            traces[f"{o}"] = {
                "off_rate": o, "residence_lj": 1.0 / o, "seed": REP_SEED,
                "tau_lj": rep["tau_lj"], "noise_limited": rep["noise_limited"],
                "modulus_lj": rep["modulus_lj"],
                **rep["trace"],
            }

    # Per-rate aggregates over all seeds (the tau-vs-residence + modulus panels).
    aggregate = []
    for o in OFF_RATES:
        rows = [x for x in raw if x.get("off_rate") == o and "error" not in x]
        taus = [x["tau_lj"] for x in rows if not x["noise_limited"]]
        mods = [x["modulus_lj"] for x in rows]
        aggregate.append({
            "off_rate": o, "residence_lj": 1.0 / o,
            "tau_lj_median": float(np.median(taus)) if taus else float("nan"),
            "tau_lj_all": [float(t) for t in taus],
            "tau_resolved": len(taus), "n_seeds": len(rows),
            "modulus_lj_median": float(np.median(mods)) if mods else float("nan"),
            "modulus_lj_all": [float(m) for m in mods],
        })

    payload = {
        "_provenance": {
            "generator": "workspace/studies/bm-r2-collagen-network-rheology/sims/fig2_panels_run.py",
            "driver": "viva_lumenoid.stage1.run_stage1 (real CollagenNetworkProcess, "
                      "released NC1/7S junction angles; the engine stage2_sweep wires)",
            "protocol": "staged Fig-2D: assemble -> relax -> stretch (frozen) -> "
                        "hold_elastic -> hold_viscous (breakage-only relaxation)",
            "network": {"n_rods": N_RODS, "box_xy": BOX_XY, "density": DENSITY,
                        "slab_thickness": SLAB, "bending_k": 4.0, "use_real_angles": True,
                        "assemble_steps": ASSEMBLE, "hold_steps": HOLD, "sample_dt": SAMPLE_DT},
            "off_rates": OFF_RATES, "seeds": SEEDS, "representative_seed": REP_SEED,
            "units": "reduced LJ (kT/sigma^3 stress, LJ time); no pascal/hour calibration",
            "note": "Reduced-unit reproduction of Fig 2 panel STRUCTURE/DIRECTION, "
                    "not a pixel match to the published PDF.",
        },
        "config": {"n_rods": N_RODS, "box_xy": BOX_XY, "slab": SLAB,
                   "assemble": ASSEMBLE, "hold": HOLD, "seeds": SEEDS,
                   "off_rates": OFF_RATES, "sample_dt": SAMPLE_DT},
        "traces": traces,
        "aggregate": aggregate,
        "raw": [{k: v for k, v in x.items() if k != "trace"} for x in raw],
    }
    OUT_BAKED.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_BAKED, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("WROTE", OUT_BAKED)
    print("AGGREGATE", json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
