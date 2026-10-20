"""fp-r2-dense-multiseed: resolve tau vs collagen residence time (Barrientos Fig 2E).

Runs the faithful density-calibrated collagen IV network (authors' density 3.0
rods/sigma^2, 12-sigma slab, released NC1/7S angles) at ~300 rods, multi-seed,
sweeping the crosslink off_rate (collagen residence time 1/off_rate) and fitting
the macroscopic stress-relaxation time tau at each. Reuses the exact tau measure
from run_stage1 (relaxation_time_lj / relaxation_noise_limited / elastic_modulus_lj).
Parallel across (off_rate, seed).

The recorded result (run r2-dense-multiseed-sweep in bm-r2/study.yaml): tau rises
monotonically with residence (37.3 -> 101.6 LJ over residence 12.5 -> 200),
resolved 4/4 seeds up to residence 100; modulus turnover-independent at 3.19 LJ.

Run:  python scripts/r2_dense_sweep.py [out.json]
"""
import json
import pathlib
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

BOX_XY, DENSITY, SLAB = 10.0, 3.0, 12.0
N_RODS = int(round(DENSITY * BOX_XY * BOX_XY))  # 300
ASSEMBLE, HOLD = 12000, 8000
OFF_RATES = [0.005, 0.01, 0.02, 0.04, 0.08]  # residence 200/100/50/25/12.5 LJ
SEEDS = [11, 22, 33, 44]


def one(off_rate, seed):
    from viva_lumenoid.params import CollagenParams
    from viva_lumenoid.stage1 import run_stage1
    r = run_stage1(CollagenParams(
        n_rods=N_RODS, box_xy=BOX_XY, slab_thickness=SLAB, bending_k=4.0,
        use_real_angles=True, assemble_steps=ASSEMBLE, hold_steps=HOLD,
        off_rate=off_rate, seed=seed))
    return {"off_rate": off_rate, "seed": seed, "tau_lj": float(r.relaxation_time_lj),
            "noise_limited": bool(r.relaxation_noise_limited),
            "modulus_lj": float(r.elastic_modulus_lj)}


def main(out="r2_dense_sweep.json"):
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
    agg = []
    for o in OFF_RATES:
        rows = [x for x in raw if x.get("off_rate") == o and "error" not in x]
        taus = [x["tau_lj"] for x in rows if not x["noise_limited"]]
        mods = [x["modulus_lj"] for x in rows]
        agg.append({"off_rate": o, "residence_lj": 1.0 / o,
                    "tau_lj_median": float(np.median(taus)) if taus else float("nan"),
                    "tau_resolved": len(taus), "n_seeds": len(rows),
                    "modulus_lj_median": float(np.median(mods)) if mods else float("nan")})
    with open(out, "w") as fh:
        json.dump({"config": {"n_rods": N_RODS, "box_xy": BOX_XY, "slab": SLAB,
                              "assemble": ASSEMBLE, "hold": HOLD, "seeds": SEEDS},
                   "aggregate": agg, "raw": raw}, fh, indent=2)
    print("DONE", json.dumps(agg))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "r2_dense_sweep.json")
