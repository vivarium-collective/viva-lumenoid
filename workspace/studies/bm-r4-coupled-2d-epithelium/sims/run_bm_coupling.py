#!/usr/bin/env python
"""R4 canonical run: the explicit BM<->epithelium coupling sweep.

Runs the REAL tyssue vertex model (viva_lumenoid.bm_coupling.run_bm_coupled_sheet)
through a preferred-area + preferred-volume growth ramp against the BM face energy,
across the no-bm / elastic / relaxing coupling modes and a BM-stiffness gradient,
and renders the 4-panel coupled-growth figure + a summary JSON.

This replaces the retired per-cell-Maxwell surrogate: the BM tension is now carried
continuously across cells through the shared vertices of the vertex-model sheet.

Usage:  python workspace/studies/bm-r4-coupled-2d-epithelium/sims/run_bm_coupling.py [out.json]
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Make the worktree's viva_lumenoid win over any editable install at canonical.
HERE = Path(__file__).resolve()
WS_ROOT = HERE.parents[4]  # <root>/workspace/studies/<slug>/sims/run_...py
sys.path.insert(0, str(WS_ROOT))

from viva_lumenoid.bm_coupling import run_bm_coupled_sheet  # noqa: E402

STUDY_DIR = HERE.parents[1]
VIZ = STUDY_DIR / "viz" / "r4_bm_coupling.html"

# Fixed growth target so the modes are comparable; vol grows so confinement can
# raise absolute cell height.
COMMON = dict(rate=0.04, vol_rate=0.05, steps=36, dt=0.25)
BM = dict(bm_stiffness=40.0, bm_relaxation=12.0)


def run():
    modes = {}
    for m in ("no-bm", "elastic", "relaxing"):
        modes[m] = run_bm_coupled_sheet(mode=m, **COMMON, **BM)
    grad = run_bm_coupled_sheet(mode="elastic", stiffness_gradient=20.0, **COMMON, **BM)
    return modes, grad


def summary(modes, grad):
    return {
        "experiment": "explicit BM face-energy coupling on a tyssue vertex sheet",
        "growth": {"area_rate": COMMON["rate"], "vol_rate": COMMON["vol_rate"],
                   "steps": COMMON["steps"], "dt": COMMON["dt"]},
        "bm": BM,
        "modes": {
            m: {
                "confinement_gap_final": round(r.gap_final, 4),
                "height_gain": round(r.height_gain, 4),
                "bm_tension_final": round(float(r.bm_tension_mean[-1]), 4),
                "realized_area_final": round(float(r.a_mean[-1]), 4),
                "preferred_area_final": round(float(r.a_pref[-1]), 4),
            } for m, r in modes.items()
        },
        "neighbour_coupling_gradient": {
            "note": "stiffness 0 (soft) -> high (stiff) across x; smooth monotone field",
            "area_soft_vs_stiff": [round(float(grad.area_final_cells[grad.stiffness.argmin()]), 4),
                                   round(float(grad.area_final_cells[grad.stiffness.argmax()]), 4)],
            "height_soft_vs_stiff": [round(float(grad.height_final_cells[grad.stiffness.argmin()]), 4),
                                     round(float(grad.height_final_cells[grad.stiffness.argmax()]), 4)],
        },
        "verdict": (
            "A stiff/slow BM confines basal-area growth (gap %.2f vs %.2f free) and "
            "drives growing cell volume into cell height (+%.0f%% vs +%.0f%% free); the "
            "BM tension is continuous across cells (smooth stiffness-gradient field). "
            "Reduces to R1's prism law in the homogeneous limit."
            % (modes["elastic"].gap_final, modes["no-bm"].gap_final,
               100 * modes["elastic"].height_gain, 100 * modes["no-bm"].height_gain)
        ),
    }


def figure(modes, grad):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    colors = {"no-bm": "#6b7280", "elastic": "#2563eb", "relaxing": "#059669"}
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "A · Basal area: preferred (dashed) vs realized (solid) — the gap is confinement",
            "B · Cell height (prism h=V0/A) — confinement drives thickening",
            "C · BM reaction tension (continuous across cells)",
            "D · Spatial field: stiffer BM → smaller area, taller cell (neighbour coupling)",
        ),
        vertical_spacing=0.16, horizontal_spacing=0.1,
    )
    any_r = next(iter(modes.values()))
    fig.add_trace(go.Scatter(x=any_r.t, y=any_r.a_pref, mode="lines",
                             line=dict(color="#111827", dash="dash"), name="preferred area A0(t)"),
                  row=1, col=1)
    for m, r in modes.items():
        fig.add_trace(go.Scatter(x=r.t, y=r.a_mean, mode="lines", line=dict(color=colors[m]),
                                 name=f"{m} realized"), row=1, col=1)
        fig.add_trace(go.Scatter(x=r.t, y=r.height_mean, mode="lines", line=dict(color=colors[m]),
                                 name=f"{m} height", showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=r.t, y=r.bm_tension_mean, mode="lines", line=dict(color=colors[m]),
                                 name=f"{m} BM tension", showlegend=False), row=2, col=1)
    order = grad.stiffness.argsort()
    fig.add_trace(go.Scatter(x=grad.stiffness[order], y=grad.area_final_cells[order],
                             mode="lines+markers", line=dict(color="#2563eb"),
                             name="basal area"), row=2, col=2)
    fig.add_trace(go.Scatter(x=grad.stiffness[order], y=grad.height_final_cells[order],
                             mode="lines+markers", line=dict(color="#dc2626"),
                             name="cell height", yaxis="y2"), row=2, col=2)
    fig.update_xaxes(title_text="time", row=1, col=1)
    fig.update_xaxes(title_text="time", row=1, col=2)
    fig.update_xaxes(title_text="time", row=2, col=1)
    fig.update_xaxes(title_text="BM stiffness K_BM (face)", row=2, col=2)
    fig.update_yaxes(title_text="basal area", row=1, col=1)
    fig.update_yaxes(title_text="cell height h = V0/A", row=1, col=2)
    fig.update_yaxes(title_text="BM tension", row=2, col=1)
    fig.update_yaxes(title_text="area (blue) / height (red)", row=2, col=2)
    fig.update_layout(
        template="plotly_white", height=760, width=1040,
        title=("R4 · Explicit BM↔epithelium coupling (tyssue face energy) — "
               "realized geometry emerges and lags the preferred under BM resistance"),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0),
        margin=dict(t=120, l=70, r=30, b=60),
    )
    return fig


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else (STUDY_DIR / "sims" / "r4_bm_coupling_summary.json")
    modes, grad = run()
    s = summary(modes, grad)
    out.write_text(json.dumps(s, indent=2))
    VIZ.parent.mkdir(parents=True, exist_ok=True)
    figure(modes, grad).write_html(str(VIZ), include_plotlyjs="cdn", full_html=True)
    print(json.dumps(s, indent=2))
    print(f"\nwrote figure  -> {VIZ}")
    print(f"wrote summary -> {out}")


if __name__ == "__main__":
    main()
