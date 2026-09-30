"""Aesthetic, diagnostic Plotly figures for the basement-membrane studies.

Pure functions: each takes a run's result object (or its trace/snapshot) and
returns a ``plotly.graph_objects.Figure`` you can inspect to *diagnose the
model*, not just admire it. Colours are a colourblind-safe categorical set;
hover is on everywhere; the network snapshot shows the actual mesh.

    from viva_lumenoid import run_stage1
    from viva_lumenoid.viz import stage1_figure, save_html
    save_html(stage1_figure(run_stage1()), "stage1.html")
"""
from __future__ import annotations

from typing import Any

import numpy as np

# Colourblind-safe categorical palette (Tableau/Vega-derived; validated order).
BLUE, ORANGE, GREEN, RED, PURPLE, TEAL, GOLD = (
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2", "#EECA3B")

PHASE_COLOR = {
    "assemble": "#9AA0A6",       # grey — building the network
    "relax": BLUE,               # settling the pre-stress
    "stretch": ORANGE,           # loading (chemistry frozen)
    "hold_elastic": GREEN,       # elastic plateau
    "hold_viscous": RED,         # remodelling / relaxation
}
PHASE_ORDER = ["assemble", "relax", "stretch", "hold_elastic", "hold_viscous"]

_INK = "#1f2328"
_MUTED = "#6b7280"
_GRID = "rgba(120,130,145,0.18)"


def _rolling(y, w=5):
    y = np.asarray(y, dtype=float)
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="same")


def _layout(fig, height=760):
    fig.update_layout(
        template="plotly_white",
        height=height,
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif",
                  size=13, color=_INK),
        margin=dict(l=64, r=28, t=64, b=56),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=11)),
        hovermode="closest",
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor=_GRID, zeroline=False, linecolor=_GRID,
                     ticks="outside", tickcolor=_GRID)
    fig.update_yaxes(gridcolor=_GRID, zeroline=False, linecolor=_GRID,
                     ticks="outside", tickcolor=_GRID)
    return fig


# --------------------------------------------------------------------------- #
# Stage 1
# --------------------------------------------------------------------------- #
def _network_segments(snapshot: dict):
    """Rod segments (x, y with None breaks), skipping periodic-wrap artifacts."""
    pos = np.asarray(snapshot["positions"], dtype=float)
    box = snapshot["box"]
    lx, ly = box[0], box[1]
    xs, ys, ex, ey, etypes = [], [], [], [], []
    types = snapshot.get("atom_types")
    n = len(pos) // 2 * 2
    for i in range(0, n, 2):
        a, b = pos[i], pos[i + 1]
        # rods are ~1 bead-spacing long; anything longer is a periodic-wrap streak
        if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 > 9.0:
            continue
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
    if types is not None:
        types = np.asarray(types)
        ex = pos[:, 0]
        ey = pos[:, 1]
        etypes = types
    return xs, ys, ex, ey, etypes, (lx, ly)


def stage1_figure(result: Any) -> "object":
    """2x2 diagnostic dashboard for a stage-1 run."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    tr = result.trace
    t = np.asarray(tr["t"]); sig = np.asarray(tr["sigma"])
    strain = np.asarray(tr["strain"]); xl = np.asarray(tr["n_crosslinks"])
    ph = np.asarray(tr["phase"])

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Network stress through the staged protocol",
            "Stress–strain during stretch  →  elastic modulus",
            "Crosslinks (network assembly & remodelling)",
            "Final network mesh (rods)"),
        vertical_spacing=0.13, horizontal_spacing=0.11,
        specs=[[{}, {}], [{}, {}]])

    # (1,1) stress vs time — faint raw markers + a bold smoothed trend per phase,
    # so the signal reads through the thermal noise (a diagnostic in itself).
    for phase in PHASE_ORDER:
        m = ph == phase
        if not m.any():
            continue
        c = PHASE_COLOR[phase]
        fig.add_trace(go.Scatter(
            x=t[m], y=sig[m], mode="markers", name=phase,
            marker=dict(size=5, color=c, opacity=0.35), legendgroup=phase,
            hovertemplate=f"<b>{phase}</b><br>t=%{{x:.0f}}<br>σ=%{{y:.3f}}<extra></extra>"),
            row=1, col=1)
        if m.sum() >= 3:
            fig.add_trace(go.Scatter(
                x=t[m], y=_rolling(sig[m]), mode="lines",
                line=dict(color=c, width=2.5), legendgroup=phase, showlegend=False,
                hoverinfo="skip"), row=1, col=1)
    # baseline line
    fig.add_hline(y=result.sigma_prestretch_baseline, line=dict(color=_MUTED, width=1, dash="dot"),
                  row=1, col=1)

    # (1,2) stress vs strain during stretch + fit line
    sm = ph == "stretch"
    if sm.any():
        fig.add_trace(go.Scatter(
            x=strain[sm], y=sig[sm], mode="markers", name="stretch σ(ε)",
            marker=dict(size=8, color=ORANGE, line=dict(width=1, color="white")),
            showlegend=False,
            hovertemplate="ε=%{x:.3f}<br>σ=%{y:.3f}<extra></extra>"), row=1, col=2)
        # fit line through baseline + slope
        xx = np.linspace(0, float(strain[sm].max()), 20)
        yy = result.sigma_prestretch_baseline + result.elastic_modulus_lj * xx
        fig.add_trace(go.Scatter(
            x=xx, y=yy, mode="lines", name="tangent modulus",
            line=dict(color=RED, width=2, dash="dash"), showlegend=False,
            hovertemplate="fit<extra></extra>"), row=1, col=2)
        # zoom the y-axis to the data so the (very soft) slope is legible
        lo, hi = float(sig[sm].min()), float(sig[sm].max())
        pad = max(0.05, 0.25 * (hi - lo))
        fig.update_yaxes(range=[lo - pad, hi + pad], row=1, col=2)

    # (2,1) crosslinks vs time (phase-coloured markers on a grey line)
    fig.add_trace(go.Scatter(
        x=t, y=xl, mode="lines", line=dict(color=_MUTED, width=1.5),
        showlegend=False, hoverinfo="skip"), row=2, col=1)
    for phase in PHASE_ORDER:
        m = ph == phase
        if not m.any():
            continue
        fig.add_trace(go.Scatter(
            x=t[m], y=xl[m], mode="markers",
            marker=dict(size=6, color=PHASE_COLOR[phase]), showlegend=False,
            legendgroup=phase,
            hovertemplate=f"<b>{phase}</b><br>t=%{{x:.0f}}<br>crosslinks=%{{y}}<extra></extra>"),
            row=2, col=1)

    # (2,2) network mesh
    if result.snapshot:
        xs, ys, ex, ey, etypes, (lx, ly) = _network_segments(result.snapshot)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", line=dict(color="rgba(80,90,105,0.55)", width=1.2),
            name="rods", showlegend=False, hoverinfo="skip"), row=2, col=2)
        if len(ex):
            nc1 = etypes == 1
            fig.add_trace(go.Scatter(
                x=ex[nc1], y=ey[nc1], mode="markers", name="NC1",
                marker=dict(size=4, color=BLUE), legendgroup="nc1",
                hovertemplate="NC1 end<extra></extra>"), row=2, col=2)
            fig.add_trace(go.Scatter(
                x=ex[~nc1], y=ey[~nc1], mode="markers", name="7S",
                marker=dict(size=4, color=ORANGE), legendgroup="7s",
                hovertemplate="7S end<extra></extra>"), row=2, col=2)
        # Box stays ~square under equibiaxial stretch, so matching ranges gives a
        # near-1:1 aspect without a cross-subplot scaleanchor (which would couple
        # this panel's scale to another subplot's axis).
        fig.update_xaxes(range=[0, lx], row=2, col=2, constrain="domain")
        fig.update_yaxes(range=[0, ly], row=2, col=2, constrain="domain")

    fig.update_xaxes(title_text="LJ time", row=1, col=1)
    fig.update_yaxes(title_text="σ (kT/a³)", row=1, col=1)
    fig.update_xaxes(title_text="strain ε", row=1, col=2)
    fig.update_yaxes(title_text="σ (kT/a³)", row=1, col=2)
    fig.update_xaxes(title_text="LJ time", row=2, col=1)
    fig.update_yaxes(title_text="crosslink bonds", row=2, col=1)
    fig.update_xaxes(title_text="x (a)", row=2, col=2)
    fig.update_yaxes(title_text="y (a)", row=2, col=2)

    m = result.elastic_modulus_lj
    tau = ("noise-limited" if result.relaxation_noise_limited
           else f"{result.relaxation_time_lj:.0f} τ")
    _layout(fig)
    fig.update_layout(title=dict(
        text=(f"<b>Stage 1 — collagen IV modulus &amp; remodelling</b>"
              f"   <span style='font-size:12px;color:{_MUTED}'>"
              f"E ≈ {m:.2f} kT/a³ (~{m*2.5:.1f} Pa) · remodelling: {tau} · "
              f"{result.n_crosslinks_assembled} crosslinks</span>"),
        x=0.01, xanchor="left"))
    return fig


# --------------------------------------------------------------------------- #
# Stage 2
# --------------------------------------------------------------------------- #
def stage2_figure(result: Any) -> "object":
    """1x3 diagnostic dashboard for a stage-2 ε̇ sweep."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    rates = np.asarray([p.strain_rate for p in result.points])
    sig = np.asarray([p.sigma_steady for p in result.points])
    sigstd = np.asarray([p.sigma_std for p in result.points])
    eta = np.asarray([p.eta_effective for p in result.points])
    xl = np.asarray([p.n_crosslinks for p in result.points])

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Steady stress σ(ε̇)", "Effective viscosity η = σ/ε̇",
                        "Crosslinks vs strain rate"),
        horizontal_spacing=0.08)

    fig.add_trace(go.Scatter(
        x=rates, y=sig, error_y=dict(type="data", array=sigstd, color=_MUTED, thickness=1),
        mode="lines+markers", line=dict(color=BLUE, width=2),
        marker=dict(size=9, color=BLUE, line=dict(width=1, color="white")),
        name="σ_steady", showlegend=False,
        hovertemplate="ε̇=%{x:.1e}<br>σ=%{y:.3f}<extra></extra>"), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=rates, y=eta, mode="lines+markers", line=dict(color=PURPLE, width=2),
        marker=dict(size=9, color=PURPLE, line=dict(width=1, color="white")),
        name="η", showlegend=False,
        hovertemplate="ε̇=%{x:.1e}<br>η=%{y:.1f}<extra></extra>"), row=1, col=2)
    fig.add_hline(y=result.viscosity_lowrate, line=dict(color=GREEN, width=1.5, dash="dash"),
                  annotation_text="low-rate η", annotation_position="top left",
                  row=1, col=2)
    if np.isfinite(result.crossover_rate):
        fig.add_vline(x=result.crossover_rate, line=dict(color=RED, width=1.5, dash="dot"),
                      annotation_text="crossover", row=1, col=2)

    fig.add_trace(go.Scatter(
        x=rates, y=xl, mode="lines+markers", line=dict(color=ORANGE, width=2),
        marker=dict(size=9, color=ORANGE, line=dict(width=1, color="white")),
        name="crosslinks", showlegend=False,
        hovertemplate="ε̇=%{x:.1e}<br>crosslinks=%{y}<extra></extra>"), row=1, col=3)

    fig.update_xaxes(type="log", title_text="ε̇ (1/τ)", row=1, col=1)
    fig.update_yaxes(title_text="σ (kT/a³)", row=1, col=1)
    fig.update_xaxes(type="log", title_text="ε̇ (1/τ)", row=1, col=2)
    fig.update_yaxes(type="log", title_text="η (kT·τ/a³)", row=1, col=2)
    fig.update_xaxes(type="log", title_text="ε̇ (1/τ)", row=1, col=3)
    fig.update_yaxes(title_text="crosslink bonds", row=1, col=3)

    _layout(fig, height=430)
    regime = ("viscous plateau" if np.isfinite(result.crossover_rate)
              else "elastic-dominated across sampled rates")
    fig.update_layout(title=dict(
        text=(f"<b>Stage 2 — σ(ε̇) on the growing substrate</b>"
              f"   <span style='font-size:12px;color:{_MUTED}'>"
              f"low-rate η ≈ {result.viscosity_lowrate:.0f} · {regime}</span>"),
        x=0.01, xanchor="left"))
    return fig


# --------------------------------------------------------------------------- #
# Rigidity-percolation sweep (follow-up)
# --------------------------------------------------------------------------- #
def rigidity_figure(result: Any) -> "object":
    """Elastic modulus vs mean coordination z, against the 2D rigidity threshold."""
    import plotly.graph_objects as go

    pts = result.points
    z = np.asarray([p.z_mean for p in pts])
    mod = np.asarray([p.modulus_mean for p in pts])
    ci = np.asarray([p.modulus_ci95 for p in pts])
    zc = result.rigidity_threshold

    fig = go.Figure()
    # shade the floppy region z < zc
    fig.add_vrect(x0=min(0.9, float(z.min()) - 0.2), x1=zc, fillcolor="rgba(76,120,168,0.06)",
                  line_width=0, annotation_text="floppy (z < 4)",
                  annotation_position="top left", annotation_font_size=11)
    fig.add_vline(x=zc, line=dict(color=RED, width=2, dash="dash"),
                  annotation_text="rigidity threshold z = 4", annotation_position="top right")
    fig.add_hline(y=0.0, line=dict(color=_MUTED, width=1, dash="dot"))
    fig.add_trace(go.Scatter(
        x=z, y=mod, error_y=dict(type="data", array=ci, color=_MUTED, thickness=1.2),
        mode="lines+markers", line=dict(color=GREEN, width=2),
        marker=dict(size=11, color=GREEN, line=dict(width=1.2, color="white")),
        name="modulus", showlegend=False,
        customdata=np.stack([[p.nc1_max for p in pts], [p.svns_max for p in pts]], axis=-1),
        hovertemplate=("z=%{x:.2f}<br>E=%{y:.3f} kT/a³"
                       "<br>NC1×%{customdata[0]} 7S×%{customdata[1]}<extra></extra>")))
    fig.update_xaxes(title_text="mean coordination z", range=[min(0.9, float(z.min()) - 0.2),
                                                              max(zc + 0.5, float(z.max()) + 0.3)])
    fig.update_yaxes(title_text="elastic modulus E (kT/a³)")
    _layout(fig, height=460)
    reached = "reached rigidity" if result.reached_rigid else "never reaches z = 4 → stays floppy"
    fig.update_layout(title=dict(
        text=(f"<b>Rigidity sweep — modulus vs connectivity</b>   "
              f"<span style='font-size:12px;color:{_MUTED}'>"
              f"raising NC1/7S crosslinks per end: {reached}</span>"),
        x=0.01, xanchor="left"))
    return fig


# --------------------------------------------------------------------------- #
def figure_to_html(fig, title: str = "viva-lumenoid") -> str:
    import plotly.io as pio
    return pio.to_html(fig, full_html=True, include_plotlyjs="cdn",
                       config={"displaylogo": False}, default_height="100%")


def save_html(fig, path: str, title: str = "viva-lumenoid") -> str:
    with open(path, "w") as f:
        f.write(figure_to_html(fig, title))
    return path
