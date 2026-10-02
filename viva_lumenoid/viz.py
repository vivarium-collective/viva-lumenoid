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

from . import comparators as C

# Colourblind-safe categorical palette (Tableau/Vega-derived; validated order).
BLUE, ORANGE, GREEN, RED, PURPLE, TEAL, GOLD = (
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2", "#EECA3B")

# Comparator bands render in a reserved neutral/good hue so a measured literature
# range never reads as "another data series": soft green fill + a dashed centre.
_BAND_FILL = "rgba(84,162,75,0.10)"
_BAND_LINE = "rgba(84,162,75,0.55)"


def _comparator_band(fig, comp, orient="v", row=None, col=None, label=True):
    """Shade a measured-literature band (lo..hi) with a centre line + label."""
    kw = {}
    if row is not None:
        kw = dict(row=row, col=col)
    if orient == "v":
        fig.add_vrect(x0=comp.lo, x1=comp.hi, fillcolor=_BAND_FILL, line_width=0,
                      **kw)
        fig.add_vline(x=comp.value, line=dict(color=_BAND_LINE, width=1.5, dash="dash"),
                      annotation_text=(comp.label if label else None),
                      annotation_position="top", annotation_font_size=10, **kw)
    else:
        fig.add_hrect(y0=comp.lo, y1=comp.hi, fillcolor=_BAND_FILL, line_width=0,
                      **kw)
        fig.add_hline(y=comp.value, line=dict(color=_BAND_LINE, width=1.5, dash="dash"),
                      annotation_text=(comp.label if label else None),
                      annotation_position="right", annotation_font_size=10, **kw)
    return fig

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
              f"E ≈ {m:.2f} kT/σ³ (~{m*2.2:.1f} Pa) · remodelling: {tau} · "
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
                  annotation_text="rigidity threshold z = 4 (Maxwell, 2D)",
                  annotation_position="top right")
    # where the released NC1×1 + 7S×3 topology tops out: z ≤ 3, sub-isostatic
    fig.add_vline(x=C.Z_NC1_7S_MAX, line=dict(color=ORANGE, width=1.5, dash="dot"),
                  annotation_text="NC1×1 + 7S×3 → z ≤ 3", annotation_position="bottom right",
                  annotation_font_size=10)
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
    fig.update_xaxes(title_text="mean coordination z")
    fig.update_yaxes(title_text="elastic modulus E (kT/σ³)")
    # SUPERSEDED banner — this central-force framing does not describe the released model
    fig.add_annotation(xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                       text="SUPERSEDED", font=dict(size=46, color="rgba(228,87,86,0.14)"),
                       textangle=-18)
    fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.04, xanchor="right",
                       showarrow=False, align="right", font=dict(size=10, color=RED),
                       text=("cross-check 2026-10-01: the released model has junction<br>"
                             "angles (not central-force) — this angles-off sweep does<br>"
                             "not describe the published model"))
    _layout(fig, height=460)
    fig.update_layout(title=dict(
        text=(f"<b>Rigidity sweep — angles-off network (superseded explanation)</b>   "
              f"<span style='font-size:12px;color:{_MUTED}'>"
              f"with angles off the network is sub-isostatic &amp; soft; the released "
              f"model has angles and is soft anyway</span>"),
        x=0.01, xanchor="left"))
    return fig


# --------------------------------------------------------------------------- #
# Junction-bending sweep (FP1 resolution)
# --------------------------------------------------------------------------- #
def bending_figure(result: Any) -> "object":
    """Elastic modulus vs junction-bending stiffness: floppy → rigid."""
    import plotly.graph_objects as go

    pts = result.points
    k = np.asarray([p.bending_k for p in pts])
    med = np.asarray([p.modulus_median for p in pts])
    iqr = np.asarray([p.modulus_iqr for p in pts])

    fig = go.Figure()
    fig.add_hline(y=0.0, line=dict(color=_MUTED, width=1, dash="dot"),
                  annotation_text="floppy (E ≈ 0)", annotation_position="bottom right")
    # per-seed points (shows the occasional numerical outlier honestly)
    for p in pts:
        fig.add_trace(go.Scatter(
            x=[p.bending_k] * len(p.per_seed), y=p.per_seed, mode="markers",
            marker=dict(size=7, color=BLUE, opacity=0.35), showlegend=False,
            hovertemplate="seed E=%{y:.2f}<extra></extra>"))
    # robust median trend with IQR
    fig.add_trace(go.Scatter(
        x=k, y=med, error_y=dict(type="data", array=iqr / 2, color=_MUTED, thickness=1.2),
        mode="lines+markers", line=dict(color=GREEN, width=2.5),
        marker=dict(size=12, color=GREEN, line=dict(width=1.4, color="white")),
        name="median E", showlegend=False,
        hovertemplate="bending_k=%{x}<br>median E=%{y:.2f} kT/a³<extra></extra>"))
    # mark the RELEASED NC1 bending stiffness (KangNC1 = 4.0)
    fig.add_vline(x=4.0, line=dict(color=ORANGE, width=2, dash="dash"),
                  annotation_text="released NC1 value (k=4.0)",
                  annotation_position="top", annotation_font_size=11)
    fig.update_xaxes(title_text="junction bending stiffness  bending_k")
    fig.update_yaxes(title_text="elastic modulus E (kT/σ³)")
    _layout(fig, height=470)
    repro = any(getattr(p, "reproducible", False) for p in pts)
    verdict = ("clean, reproducible modulus" if repro else
               "even at the released k=4.0 the response is ILL-CONDITIONED — no "
               "reproducible modulus at v1 size")
    fig.update_layout(title=dict(
        text=(f"<b>Bending sweep around the released k=4.0 (not the stiffness source)</b>   "
              f"<span style='font-size:12px;color:{_MUTED}'>{verdict}; "
              f"median (line) + per-seed points</span>"),
        x=0.01, xanchor="left"))
    return fig


# --------------------------------------------------------------------------- #
# Simulation movie — the spatial network state unfolding, synced to a metric
# --------------------------------------------------------------------------- #
def _frame_mesh(frame):
    """(rod_xs, rod_ys, nc1_xy, s7_xy, box_xy) for one captured frame."""
    pos = np.asarray(frame["positions"], dtype=float)
    lx, ly = float(frame["box"][0]), float(frame["box"][1])
    types = np.asarray(frame.get("atom_types")) if frame.get("atom_types") is not None else None
    n = (len(pos) // 2) * 2
    xs, ys = [], []
    for i in range(0, n, 2):
        a, b = pos[i], pos[i + 1]
        if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 > 9.0:   # periodic-wrap streak
            continue
        xs += [a[0], b[0], None]
        ys += [a[1], b[1], None]
    nc1 = s7 = (np.array([]), np.array([]))
    if types is not None:
        m = types[:len(pos)] == 1
        nc1 = (pos[:len(types)][m][:, 0], pos[:len(types)][m][:, 1])
        s7 = (pos[:len(types)][~m][:, 0], pos[:len(types)][~m][:, 1])
    box_xy = ([0, lx, lx, 0, 0], [0, 0, ly, ly, 0])
    return xs, ys, nc1, s7, box_xy


def network_movie_figure(result: Any, title: str = "collagen IV network") -> "object":
    """An animated player of the network's spatial state (left) synced to the
    stress trace (right) — play/pause + a phase-labelled time slider.

    ``result`` must have ``frames`` (from run_stage1(capture_frames=True)); the
    box outline + cursor are coloured by the current protocol phase so the
    assemble → relax → stretch → hold → remodel structure reads at a glance.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    frames = result.frames or []
    if not frames:
        raise ValueError("result has no captured frames (run with capture_frames=True)")
    t_all = np.asarray([f["t"] for f in frames])
    sig_all = np.asarray([f["sigma"] for f in frames])
    # full (dense) stress trace for the static backdrop line
    tr = result.trace
    tt, sg, phs = np.asarray(tr["t"]), np.asarray(tr["sigma"]), np.asarray(tr["phase"])
    maxbox = max(float(f["box"][0]) for f in frames)
    # auto-range the mesh panel to the actual data extent (unwrapping can carry a
    # few beads just outside the box — show them rather than clip) unioned w/ box
    _allx = np.concatenate([np.asarray(f["positions"])[:, 0] for f in frames])
    _ally = np.concatenate([np.asarray(f["positions"])[:, 1] for f in frames])
    _lo = min(float(_allx.min()), float(_ally.min()), 0.0) - 0.8
    _hi = max(float(_allx.max()), float(_ally.max()), maxbox) + 0.8

    def phase_color(ph):
        return PHASE_COLOR.get(ph, _MUTED)

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.6, 0.4],
        subplot_titles=("Network — quasi-2D slab, top-down (rods + crosslink ends)",
                        "Network stress through the clip"),
        horizontal_spacing=0.09)

    f0 = frames[0]
    xs, ys, nc1, s7, box_xy = _frame_mesh(f0)
    # --- base data (indices matter for frame .traces mapping) ---
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                  line=dict(color="rgba(80,90,105,0.55)", width=1.2),
                  name="rods", showlegend=False, hoverinfo="skip"), row=1, col=1)      # 0
    fig.add_trace(go.Scatter(x=nc1[0], y=nc1[1], mode="markers", name="NC1 end",
                  marker=dict(size=4, color=BLUE), hoverinfo="skip"), row=1, col=1)     # 1
    fig.add_trace(go.Scatter(x=s7[0], y=s7[1], mode="markers", name="7S end",
                  marker=dict(size=4, color=ORANGE), hoverinfo="skip"), row=1, col=1)   # 2
    fig.add_trace(go.Scatter(x=box_xy[0], y=box_xy[1], mode="lines",
                  line=dict(color=phase_color(f0["phase"]), width=2.5),
                  name="box", showlegend=False, hoverinfo="skip"), row=1, col=1)        # 3
    # static stress backdrop + phase-coloured markers
    fig.add_trace(go.Scatter(x=tt, y=sg, mode="lines",
                  line=dict(color="rgba(120,130,145,0.45)", width=1.2),
                  name="σ(t)", showlegend=False, hoverinfo="skip"), row=1, col=2)       # 4
    fig.add_trace(go.Scatter(x=[f0["t"]], y=[f0["sigma"]], mode="markers",
                  marker=dict(size=13, color=phase_color(f0["phase"]),
                              line=dict(width=1.5, color="white")),
                  name="now", showlegend=False,
                  hovertemplate="t=%{x:.0f}<br>σ=%{y:.3f}<extra></extra>"), row=1, col=2)  # 5

    go_frames = []
    for i, f in enumerate(frames):
        xs, ys, nc1, s7, box_xy = _frame_mesh(f)
        c = phase_color(f["phase"])
        go_frames.append(go.Frame(name=str(i), traces=[0, 1, 2, 3, 5], data=[
            go.Scatter(x=xs, y=ys),
            go.Scatter(x=nc1[0], y=nc1[1]),
            go.Scatter(x=s7[0], y=s7[1]),
            go.Scatter(x=box_xy[0], y=box_xy[1], line=dict(color=c, width=2.5)),
            go.Scatter(x=[f["t"]], y=[f["sigma"]], marker=dict(color=c, size=13)),
        ]))
    fig.frames = go_frames

    steps = [dict(method="animate", label=f"{f['phase']} ε={f['strain']:.2f}",
                  args=[[str(i)], dict(mode="immediate",
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0))])
             for i, f in enumerate(frames)]
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=1.14,
            xanchor="left", yanchor="top", pad=dict(t=0, r=8),
            buttons=[
                dict(label="▶ play", method="animate", args=[None, dict(
                    frame=dict(duration=90, redraw=True), fromcurrent=True,
                    transition=dict(duration=0))]),
                dict(label="❚❚ pause", method="animate", args=[[None], dict(
                    mode="immediate", frame=dict(duration=0, redraw=False),
                    transition=dict(duration=0))])])],
        sliders=[dict(active=0, x=0.0, y=0, len=1.0, pad=dict(t=28, b=8),
                      currentvalue=dict(prefix="phase: ", font=dict(size=12)),
                      steps=steps)])

    fig.update_xaxes(range=[_lo, _hi], row=1, col=1, constrain="domain",
                     title_text="x (a)")
    fig.update_yaxes(range=[_lo, _hi], row=1, col=1, constrain="domain",
                     title_text="y (a)")
    fig.update_xaxes(title_text="LJ time", row=1, col=2)
    fig.update_yaxes(title_text="σ (kT/a³)", row=1, col=2)
    _layout(fig, height=560)
    fig.update_layout(
        margin=dict(l=56, r=24, t=96, b=72),
        title=dict(text=(f"<b>Simulation movie — {title}</b>   "
                   f"<span style='font-size:12px;color:{_MUTED}'>"
                   f"{len(frames)} frames · quasi-2D slab (z≈2a thick) shown "
                   f"top-down · finely sampled + drift-removed · press ▶ play</span>"),
                   x=0.01, xanchor="left"))
    return fig


# --------------------------------------------------------------------------- #
# True 3D view — the quasi-2D slab in 3D, rotatable and animated
# --------------------------------------------------------------------------- #
def _frame_mesh_3d(frame):
    """(rod x/y/z with None breaks, nc1 xyz, s7 xyz) for one frame, in 3D."""
    pos = np.asarray(frame["positions"], dtype=float)
    types = np.asarray(frame.get("atom_types")) if frame.get("atom_types") is not None else None
    n = (len(pos) // 2) * 2
    xs, ys, zs = [], [], []
    for i in range(0, n, 2):
        a, b = pos[i], pos[i + 1]
        if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 > 9.0:   # periodic-wrap streak
            continue
        xs += [a[0], b[0], None]; ys += [a[1], b[1], None]; zs += [a[2], b[2], None]
    nc1 = s7 = (np.array([]), np.array([]), np.array([]))
    if types is not None:
        m = types[:len(pos)] == 1
        p = pos[:len(types)]
        nc1 = (p[m][:, 0], p[m][:, 1], p[m][:, 2])
        s7 = (p[~m][:, 0], p[~m][:, 1], p[~m][:, 2])
    return xs, ys, zs, nc1, s7


def _box_wireframe_3d(lx, ly, lz):
    """12 edges of the slab box as x/y/z line lists with None breaks."""
    c = [(0, 0, 0), (lx, 0, 0), (lx, ly, 0), (0, ly, 0),
         (0, 0, lz), (lx, 0, lz), (lx, ly, lz), (0, ly, lz)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    xs, ys, zs = [], [], []
    for i, j in edges:
        xs += [c[i][0], c[j][0], None]
        ys += [c[i][1], c[j][1], None]
        zs += [c[i][2], c[j][2], None]
    return xs, ys, zs


def network_movie_3d_figure(result: Any, title: str = "collagen IV network",
                            z_exaggerate: float = 2.0) -> "object":
    """A rotatable, animated **3D** view of the quasi-2D slab — rods + crosslink
    ends in 3D with the box wireframe, so the slab's z-thickness (and the several
    layers through it) are visible. Drag to orbit; press ▶ to play.

    ``z_exaggerate`` stretches the (very thin) z axis for visibility only — the
    subtitle says so; x/y are to scale.
    """
    import plotly.graph_objects as go
    frames = result.frames or []
    if not frames:
        raise ValueError("result has no captured frames (run with capture_frames=True)")
    lx = max(float(f["box"][0]) for f in frames)
    ly = max(float(f["box"][1]) for f in frames)
    lz = max(float(f["box"][2]) for f in frames)

    def traces_for(f):
        xs, ys, zs, nc1, s7 = _frame_mesh_3d(f)
        bx, by, bz = _box_wireframe_3d(float(f["box"][0]), float(f["box"][1]), float(f["box"][2]))
        return [
            go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                         line=dict(color="rgba(80,90,105,0.55)", width=2),
                         name="rods", showlegend=False, hoverinfo="skip"),
            go.Scatter3d(x=nc1[0], y=nc1[1], z=nc1[2], mode="markers", name="NC1 end",
                         marker=dict(size=2.6, color=BLUE), hoverinfo="skip"),
            go.Scatter3d(x=s7[0], y=s7[1], z=s7[2], mode="markers", name="7S end",
                         marker=dict(size=2.6, color=ORANGE), hoverinfo="skip"),
            go.Scatter3d(x=bx, y=by, z=bz, mode="lines",
                         line=dict(color="rgba(120,130,145,0.6)", width=2),
                         showlegend=False, hoverinfo="skip"),
        ]

    fig = go.Figure(data=traces_for(frames[0]))
    fig.frames = [go.Frame(name=str(i), data=traces_for(f)) for i, f in enumerate(frames)]
    steps = [dict(method="animate", label=f"{f['phase']} ε={f['strain']:.2f}",
                  args=[[str(i)], dict(mode="immediate",
                        frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
             for i, f in enumerate(frames)]
    fig.update_layout(
        template="plotly_white", height=620,
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", size=13, color=_INK),
        margin=dict(l=0, r=0, t=96, b=72),
        scene=dict(
            xaxis_title="x (a)", yaxis_title="y (a)", zaxis_title="z (a)",
            aspectmode="manual",
            aspectratio=dict(x=1, y=ly / lx, z=max(0.14, z_exaggerate * lz / lx)),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.1))),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=1.1,
            xanchor="left", yanchor="top",
            buttons=[
                dict(label="▶ play", method="animate", args=[None, dict(
                    frame=dict(duration=100, redraw=True), fromcurrent=True,
                    transition=dict(duration=0))]),
                dict(label="❚❚ pause", method="animate", args=[[None], dict(
                    mode="immediate", frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(active=0, x=0.0, y=0, len=1.0, pad=dict(t=24, b=8),
                      currentvalue=dict(prefix="phase: ", font=dict(size=12)), steps=steps)],
        title=dict(text=(f"<b>3D view — {title}</b>   "
                   f"<span style='font-size:12px;color:{_MUTED}'>"
                   f"the real 3D quasi-2D slab · drag to orbit · z ×{z_exaggerate:g} "
                   f"for visibility (x/y to scale) · press ▶ play</span>"),
                   x=0.01, xanchor="left"))
    return fig


# --------------------------------------------------------------------------- #
# Stage 5 — porosity & bundling (geometric readouts vs measured comparators)
# --------------------------------------------------------------------------- #
def porosity_bundling_figure(result: Any) -> "object":
    """2x2: pore-size distribution + density scaling + strand-size bundling +
    the pore map (distance field) — each against its measured comparator."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from . import geometry as geo

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Pore-size distribution vs measured BM",
            "Pore size scales with network density",
            "Bundling — protomers per strand (model under-bundles)",
            "Pore map — maximal inscribed empty discs"),
        vertical_spacing=0.14, horizontal_spacing=0.11)

    # (1,1) pore-size histogram vs corneal-EM + Matrigel bands
    diam = np.asarray(result.pore_diam_nm)
    _comparator_band(fig, C.PORE_CORNEAL_EM, "v", row=1, col=1)
    _comparator_band(fig, C.PORE_MATRIGEL, "v", row=1, col=1, label=False)
    fig.add_trace(go.Histogram(
        x=diam, nbinsx=40, marker=dict(color=BLUE, line=dict(width=0.5, color="white")),
        name="pores", showlegend=False,
        hovertemplate="pore ≈%{x:.0f} nm<br>%{y} discs<extra></extra>"), row=1, col=1)
    fig.add_vline(x=result.median_pore_nm, line=dict(color=RED, width=2),
                  annotation_text=f"median {result.median_pore_nm:.0f} nm",
                  annotation_position="top right", row=1, col=1)

    # (1,2) median pore vs density (coverage), with comparator bands
    cov = np.asarray([p.coverage * 100 for p in result.points])
    medp = np.asarray([p.median_pore_nm for p in result.points])
    _comparator_band(fig, C.PORE_CORNEAL_EM, "h", row=1, col=2)
    fig.add_trace(go.Scatter(
        x=cov, y=medp, mode="lines+markers", line=dict(color=PURPLE, width=2),
        marker=dict(size=11, color=PURPLE, line=dict(width=1, color="white")),
        name="median pore", showlegend=False,
        customdata=[p.n_rods for p in result.points],
        hovertemplate="coverage %{x:.1f}%<br>%{customdata} rods<br>median %{y:.0f} nm<extra></extra>"),
        row=1, col=2)

    # (2,1) bundling strand-size histogram vs the 5-7 band
    hist = result.bundling_hist or {}
    sizes = sorted(hist)
    counts = [hist[s] for s in sizes]
    _comparator_band(fig, C.BUNDLING_PFHR9, "v", row=2, col=1)
    fig.add_trace(go.Bar(
        x=sizes, y=counts, marker=dict(color=TEAL), name="strands", showlegend=False,
        hovertemplate="%{x} protomers/strand<br>%{y} strands<extra></extra>"), row=2, col=1)
    fig.add_annotation(row=2, col=1, x=0.98, y=0.92, xref="x domain", yref="y domain",
                       xanchor="right", showarrow=False, font=dict(size=11, color=_MUTED),
                       text=(f"mean {result.mean_strand:.2f} · max {result.max_strand}"
                             f" · {result.frac_in_5_7:.0%} in 5–7"))

    # (2,2) pore map: distance field heatmap + rod overlay (recomputed from mesh)
    if result.snapshot:
        por = geo.porosity(result.snapshot["positions"], result.snapshot["box"],
                           grid_px=180)
        grid = por.grid
        lx, ly = por.extent_a
        fig.add_trace(go.Heatmap(
            z=grid, x=np.linspace(0, lx, grid.shape[1]), y=np.linspace(0, ly, grid.shape[0]),
            colorscale="Tealrose", showscale=False, zsmooth="best",
            hovertemplate="empty-disc radius %{z:.2f} a<extra></extra>"), row=2, col=2)
        xs, ys, *_ , (bx, by) = _network_segments(result.snapshot)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", line=dict(color="rgba(20,24,30,0.45)", width=1.0),
            showlegend=False, hoverinfo="skip"), row=2, col=2)
        fig.update_xaxes(range=[0, lx], row=2, col=2, constrain="domain")
        fig.update_yaxes(range=[0, ly], row=2, col=2, constrain="domain")

    fig.update_xaxes(title_text="pore diameter (nm)", row=1, col=1)
    fig.update_yaxes(title_text="count", row=1, col=1)
    fig.update_xaxes(title_text="areal coverage (%)", row=1, col=2)
    fig.update_yaxes(title_text="median pore (nm)", row=1, col=2)
    fig.update_xaxes(title_text="protomers per strand", row=2, col=1, dtick=1)
    fig.update_yaxes(title_text="strand count", type="log", row=2, col=1)
    fig.update_xaxes(title_text="x (a)", row=2, col=2)
    fig.update_yaxes(title_text="y (a)", row=2, col=2)

    _layout(fig)
    fig.update_layout(title=dict(
        text=(f"<b>Stage 5 — porosity &amp; bundling</b>   "
              f"<span style='font-size:12px;color:{_MUTED}'>"
              f"pores reach the corneal-EM band at high density; bundling "
              f"under-shoots 5–7 (no lateral bond)</span>"),
        x=0.01, xanchor="left"))
    return fig


# --------------------------------------------------------------------------- #
# Evidence map — every v1 readout against its measured comparator + provenance
# --------------------------------------------------------------------------- #
def evidence_map_figure(measured: dict | None = None) -> "object":
    """A forest plot of the spec's measured comparators, faceted by unit, with
    the model's own measured values overlaid where available.

    ``measured`` may carry: modulus_Pa, pore_median_nm, mean_strand,
    remodelling_h — each drawn as a red diamond against its literature band.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    measured = measured or {}

    panels = [
        ("Geometry (nm)", [C.PROTOMER_CONTOUR, C.PERSISTENCE_LENGTH,
                           C.BEAD_SPACING, C.FIBRIL_DIAMETER], "linear", None),
        ("Pores (nm)", [C.PORE_CORNEAL_EM, C.PORE_MATRIGEL], "linear",
         ("pore_median_nm", "model pore median")),
        ("Modulus (Pa)", C.MODULUS_LADDER, "log",
         ("modulus_Pa", "model E")),
        ("Bundling / time", [C.BUNDLING_PFHR9, C.REMODELLING_TIME], "linear", None),
    ]
    fig = make_subplots(rows=1, cols=len(panels), horizontal_spacing=0.055,
                        subplot_titles=[p[0] for p in panels])

    for ci, (_, comps, scale, overlay) in enumerate(panels, start=1):
        ylabels = [c.label for c in comps]
        yy = list(range(len(comps)))
        xmid = [c.value for c in comps]
        xlo = [c.value - c.lo for c in comps]
        xhi = [c.hi - c.value for c in comps]
        fig.add_trace(go.Scatter(
            x=xmid, y=yy, mode="markers",
            error_x=dict(type="data", symmetric=False, array=xhi, arrayminus=xlo,
                         color=_BAND_LINE, thickness=2, width=6),
            marker=dict(size=10, color=GREEN, line=dict(width=1, color="white")),
            showlegend=False,
            customdata=[[c.method, c.source] for c in comps],
            hovertemplate="%{text}: %{x}<br>%{customdata[0]}<br><i>%{customdata[1]}</i><extra></extra>",
            text=ylabels), row=1, col=ci)
        # overlay the model's measured value for this panel, if provided
        if overlay and overlay[0] in measured:
            mv = measured[overlay[0]]
            # place the model marker on whichever comparator row shares the quantity
            yrow = len(comps) - 1
            fig.add_trace(go.Scatter(
                x=[mv], y=[yrow + 0.0], mode="markers",
                marker=dict(size=15, color=RED, symbol="diamond",
                            line=dict(width=1.2, color="white")),
                name=overlay[1], showlegend=False,
                hovertemplate=f"<b>{overlay[1]}</b>: %{{x:.3g}}<extra></extra>"), row=1, col=ci)
        fig.update_xaxes(type=scale, row=1, col=ci)
        fig.update_yaxes(tickmode="array", tickvals=yy, ticktext=ylabels,
                         row=1, col=ci, tickfont=dict(size=10))

    _layout(fig, height=430)
    fig.update_layout(title=dict(
        text=("<b>Evidence map — what each v1 readout is held against</b>   "
              f"<span style='font-size:12px;color:{_MUTED}'>measured literature "
              f"bands (green) from the spec; model values (red ◆) where run</span>"),
        x=0.01, xanchor="left"))
    return fig


# --------------------------------------------------------------------------- #
# Modulus ladder — where the corrected v1 sits among measured BM moduli
# --------------------------------------------------------------------------- #
def modulus_ladder_figure(model_pa: float | None = None,
                          model_lo_pa: float | None = None,
                          model_hi_pa: float | None = None) -> "object":
    """A log-scale ladder (Pa) placing the model's corrected modulus against the
    measured comparators: the published CG model (≈0.03 Pa) through real BM
    (10³–10⁵ Pa). Shows the v1 network lives in the published *very-soft* regime,
    10³–10⁵× below real basement membrane."""
    import plotly.graph_objects as go

    comps = C.MODULUS_LADDER
    ys = list(range(len(comps)))
    fig = go.Figure()
    for c, y in zip(comps, ys):
        fig.add_trace(go.Scatter(
            x=[c.lo, c.hi], y=[y, y], mode="lines",
            line=dict(color=_BAND_LINE, width=10), opacity=0.5,
            showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[c.value], y=[y], mode="markers+text",
            marker=dict(size=13, color=GREEN, line=dict(width=1, color="white")),
            text=[f"  {c.label}"], textposition="middle right",
            textfont=dict(size=11), showlegend=False,
            hovertemplate=f"{c.label}: {c.value:g} Pa<br>{c.method}<extra></extra>"))
    # the model's corrected modulus (a band if lo/hi given; else a point)
    if model_pa is not None:
        my = len(comps)
        if model_lo_pa is not None and model_hi_pa is not None:
            lo = max(1e-3, abs(model_lo_pa)); hi = max(lo * 1.2, abs(model_hi_pa))
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[my, my], mode="lines",
                line=dict(color=RED, width=10), opacity=0.5, showlegend=False,
                hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[max(1e-3, abs(model_pa))], y=[my], mode="markers+text",
            marker=dict(size=15, color=RED, symbol="diamond", line=dict(width=1.2, color="white")),
            text=["  corrected v1 (|E|, noise-limited)"], textposition="middle right",
            textfont=dict(size=11, color=RED), showlegend=False,
            hovertemplate=f"corrected v1: ~{abs(model_pa):.2g} Pa (consistent with 0)<extra></extra>"))
        ys = ys + [my]
    fig.update_xaxes(type="log", title_text="elastic modulus (Pa, log scale)",
                     range=[-2.2, 5.2])
    fig.update_yaxes(showticklabels=False, range=[-0.6, len(ys) - 0.4])
    _layout(fig, height=360)
    fig.update_layout(
        margin=dict(l=20, r=20, t=64, b=52),
        title=dict(text=("<b>Modulus ladder — the v1 network is in the published very-soft regime</b>   "
                   f"<span style='font-size:12px;color:{_MUTED}'>corrected v1 ≈ "
                   f"published CG (0.03 Pa); real BM is 10³–10⁵× stiffer</span>"),
                   x=0.01, xanchor="left"))
    return fig


# --------------------------------------------------------------------------- #
# Cross-check — clean-room vs released code (headline result)
# --------------------------------------------------------------------------- #
def cross_check_figure(model_pa: float | None = None,
                       model_lo_pa: float | None = None,
                       model_hi_pa: float | None = None) -> "object":
    """Headline figure: the parameter comparison (matched vs divergent) next to
    the modulus ladder, so the cross-check verdict and its consequence read
    together. Green = matches the released code; red = diverges."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from . import crosscheck as X

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.56, 0.44],
        specs=[[{"type": "table"}, {"type": "xy"}]],
        subplot_titles=(f"Parameters: {X.n_match()} match · {X.n_diverge()} diverge",
                        "Consequence — modulus vs measured BM"),
        horizontal_spacing=0.06)

    rows = X.ROWS
    match_fill = "rgba(84,162,75,0.14)"
    div_fill = "rgba(228,87,86,0.14)"
    status_txt = ["✓ match" if r.status == "match" else "✗ diverge" for r in rows]
    cell_fill = [[match_fill if r.status == "match" else div_fill for r in rows]] * 4
    fig.add_trace(go.Table(
        columnwidth=[26, 26, 26, 16],
        header=dict(values=["<b>Quantity</b>", "<b>Released</b>", "<b>Clean-room</b>", "<b>Status</b>"],
                    fill_color="#f1f3f5", align="left", font=dict(size=11, color=_INK)),
        cells=dict(values=[[r.quantity for r in rows], [r.released for r in rows],
                           [r.ours for r in rows], status_txt],
                   fill_color=cell_fill, align="left",
                   font=dict(size=10, color=_INK), height=22)), row=1, col=1)

    # right: the modulus ladder (reuse the comparator bands + model point)
    comps = C.MODULUS_LADDER
    for i, c in enumerate(comps):
        fig.add_trace(go.Scatter(x=[c.lo, c.hi], y=[i, i], mode="lines",
                      line=dict(color=_BAND_LINE, width=9), opacity=0.5,
                      showlegend=False, hoverinfo="skip"), row=1, col=2)
        fig.add_trace(go.Scatter(x=[c.value], y=[i], mode="markers+text",
                      marker=dict(size=11, color=GREEN, line=dict(width=1, color="white")),
                      text=[f"  {c.label}"], textposition="middle right",
                      textfont=dict(size=10), showlegend=False,
                      hovertemplate=f"{c.label}: {c.value:g} Pa<extra></extra>"), row=1, col=2)
    if model_pa is not None:
        my = len(comps)
        if model_lo_pa is not None and model_hi_pa is not None:
            fig.add_trace(go.Scatter(x=[max(1e-3, abs(model_lo_pa)), max(2e-3, abs(model_hi_pa))],
                          y=[my, my], mode="lines", line=dict(color=RED, width=9),
                          opacity=0.5, showlegend=False, hoverinfo="skip"), row=1, col=2)
        fig.add_trace(go.Scatter(x=[max(1e-3, abs(model_pa))], y=[my], mode="markers+text",
                      marker=dict(size=14, color=RED, symbol="diamond", line=dict(width=1.2, color="white")),
                      text=["  corrected v1"], textposition="middle right",
                      textfont=dict(size=10, color=RED), showlegend=False,
                      hovertemplate=f"corrected v1 ~{abs(model_pa):.2g} Pa (consistent with 0)<extra></extra>"),
                      row=1, col=2)
    fig.update_xaxes(type="log", title_text="modulus (Pa, log)", range=[-2.2, 5.2], row=1, col=2)
    fig.update_yaxes(showticklabels=False, range=[-0.6, len(comps) + 0.6], row=1, col=2)
    _layout(fig, height=520)
    fig.update_layout(
        title=dict(text=("<b>Cross-check — clean-room vs the released code</b>   "
                   f"<span style='font-size:12px;color:{_MUTED}'>structure + "
                   f"central-force params now match; angles/GCE/break-mechanism diverge; "
                   f"both are very soft</span>"), x=0.01, xanchor="left"))
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
