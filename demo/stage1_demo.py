"""Stage-1 demo: run the collagen IV network through stretch -> hold -> chemistry
on, print the elastic modulus + remodelling time, and save the stress trace.

    python demo/stage1_demo.py

Requires the workspace venv with `lammps` importable (see README).
"""
from __future__ import annotations

import json
import os

from viva_lumenoid import CollagenParams, run_stage1


def main():
    params = CollagenParams()  # the study's default v1 config
    print("Running stage-1 protocol (assemble -> relax -> stretch -> hold x2)...")
    r = run_stage1(params, sample_dt=8.0)

    print("\n=== Stage-1 readouts (clean-room v1, LJ units) ===")
    print(json.dumps(r.summary(), indent=2))
    m = r.elastic_modulus_lj
    print(f"\nElastic modulus : {m:.3f} kT/a^3  (~{r.elastic_modulus_Pa:.2f} Pa at kT/a^3=2.5)")
    if r.relaxation_noise_limited:
        print("Remodelling time: noise-limited at this network size "
              "(spec decision #10 — needs a larger network / seed ensemble)")
    else:
        print(f"Remodelling time: {r.relaxation_time_lj:.1f} LJ time")

    # Save the stress trace next to this script.
    out = os.path.join(os.path.dirname(__file__), "stage1_stress_trace.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = r.trace["t"]
        sig = r.trace["sigma"]
        ph = r.trace["phase"]
        colors = {"assemble": "#888", "relax": "#4C78A8", "stretch": "#F58518",
                  "hold_elastic": "#54A24B", "hold_viscous": "#E45756"}
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for phase, c in colors.items():
            xs = [t[i] for i in range(len(t)) if ph[i] == phase]
            ys = [sig[i] for i in range(len(t)) if ph[i] == phase]
            ax.plot(xs, ys, ".-", color=c, label=phase, ms=5)
        ax.set_xlabel("LJ time")
        ax.set_ylabel("network stress  sigma_inplane  (kT/a^3)")
        ax.set_title("Stage 1 — collagen IV network stress through the staged protocol")
        ax.legend(fontsize=8, ncol=3)
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"\nStress trace saved to {out}")
    except Exception as e:  # matplotlib optional
        print(f"\n(plot skipped: {e})")


if __name__ == "__main__":
    main()
