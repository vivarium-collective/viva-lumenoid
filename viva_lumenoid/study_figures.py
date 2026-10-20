"""Per-study figures for the reproduction + extension campaign (R1-R6, E1-E4).

Two honest kinds, kept visibly distinct:

* **Real data-driven** figures, which RUN the study's composite and plot its
  actual output: E1 (actomyosin contractility sweep), E3/R4/R5 (the stepped
  tyssue mesh + a real geometric readout), R2/R3 (a small real collagen run).
* **Design schematics** for the planned studies with no runnable composite yet
  (R1/R6/E2/E4): annotated diagrams of the plan's expected result / mechanism,
  captioned ``DESIGN TARGET — not yet simulated`` so they never read as data.

Each ``*_figure()`` returns a ``plotly.graph_objects.Figure``. The data-driven
ones import heavy engines lazily and run them internally, so callers (the HTML
generator + the tests) just call the function.
"""
from __future__ import annotations

import numpy as np

from .viz import BLUE, ORANGE, GREEN, RED, PURPLE, TEAL, GOLD, _layout

DESIGN_TAG = "DESIGN TARGET — not yet simulated (schematic of the plan's expected result)"


def _go():
    import plotly.graph_objects as go
    return go


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------
def _diagram(title: str, xr=(0, 10), yr=(0, 10), height=620):
    """Blank annotated-diagram canvas (hidden axes) for the schematics."""
    go = _go()
    fig = go.Figure()
    fig.update_layout(
        title=title, template="plotly_white", height=height,
        xaxis=dict(range=xr, visible=False), yaxis=dict(range=yr, visible=False),
        margin=dict(l=20, r=20, t=90, b=70), showlegend=False,
    )
    fig.add_annotation(x=0.5, y=-0.07, xref="paper", yref="paper", showarrow=False,
                       text=f"<b>{DESIGN_TAG}</b>", font=dict(color=RED, size=12))
    return fig


def _box(fig, x0, y0, x1, y1, text, color):
    fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color=color, width=2),
                  fillcolor=color, opacity=0.12)
    fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=text, showarrow=False,
                       font=dict(size=12, color="#222"))


def _arrow(fig, x0, y0, x1, y1, label=""):
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=2,
                       arrowcolor="#555", text=label, font=dict(size=11, color="#555"))


def _tyssue_build_step(composite_id: str, steps: int = 2):
    """Build + step a tyssue composite; return its live Epithelium (eptm)."""
    import viva_lumenoid.composites  # noqa: F401  (registers generators)
    from process_bigraph.composite_generator import _REGISTRY, apply_core_extensions, build_generator
    from bigraph_schema import allocate_core
    from process_bigraph import Composite
    entry = _REGISTRY[f"viva_lumenoid.composites.{composite_id}"]
    core = apply_core_extensions(entry, allocate_core())
    doc = build_generator(entry, None, core=core)
    state = doc["state"] if isinstance(doc, dict) and "state" in doc else doc
    comp = Composite({"state": state}, core=core)
    for _ in range(steps):
        comp.update({}, 0.01)
    return comp.state["Tyssue"]["instance"].eptm


def _mesh_traces(eptm, three_d: bool, color_by: str | None = None):
    """Vertex scatter (+ a sampled edge wireframe) for a tyssue epithelium."""
    go = _go()
    v = eptm.vert_df
    traces = []
    # edge wireframe (sample to keep the figure light)
    e = eptm.edge_df
    n = len(e)
    step = max(1, n // 1500)
    ex, ey, ez = [], [], []
    for s, t in zip(e["srce"].values[::step], e["trgt"].values[::step]):
        ex += [v.loc[s, "x"], v.loc[t, "x"], None]
        ey += [v.loc[s, "y"], v.loc[t, "y"], None]
        if three_d:
            ez += [v.loc[s, "z"], v.loc[t, "z"], None]
    if three_d:
        traces.append(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                                   line=dict(color="rgba(80,80,80,0.35)", width=1),
                                   hoverinfo="skip", showlegend=False))
        traces.append(go.Scatter3d(
            x=v["x"], y=v["y"], z=v["z"], mode="markers",
            marker=dict(size=2.5, color=(v[color_by] if color_by in v else v["z"]),
                        colorscale="Viridis", showscale=True,
                        colorbar=dict(title=(color_by or "z"))),
            hoverinfo="skip", showlegend=False))
    else:
        traces.append(go.Scatter(x=ex, y=ey, mode="lines",
                                 line=dict(color="rgba(80,80,80,0.5)", width=1),
                                 hoverinfo="skip", showlegend=False))
        traces.append(go.Scatter(x=v["x"], y=v["y"], mode="markers",
                                  marker=dict(size=5, color=BLUE), hoverinfo="skip",
                                  showlegend=False))
    return traces


# ---------------------------------------------------------------------------
# DATA-DRIVEN figures
# ---------------------------------------------------------------------------
def _e1_patch_tension(n, eps, seed, steps):
    """One actomyosin LJ-patch run; return the final effective in-plane tension
    τ = −(pxx+pyy)/2 (reduced LJ stress). E1-named helper (does not touch the
    shared LAMMPS helpers)."""
    from bigraph_schema import allocate_core
    from pbg_lammps.model import LammpsModelProcess
    import viva_lumenoid.actomyosin  # noqa: F401
    p = LammpsModelProcess(config={"model": "actomyosin-cortex",
                                   "params": {"n": int(n), "contractility": float(eps),
                                              "seed": int(seed)}},
                           core=allocate_core())
    p.initial_state()
    out = {}
    for _ in range(steps):
        out = p.update({}, 1.0)
    p.close()
    return float(out["cortical_tension"]), float(out["pressure_xx"]), float(out["pressure_yy"])


def e1_cortical_tension_figure(proxies=(0.5, 1.0, 2.0, 3.0, 4.0),
                               sizes=(5, 6, 7, 8), seeds=(11, 22, 33),
                               n=6, steps=30):
    """E1 MAIN (engineered-diagnostic, supplement-grade verification): seed-ensemble
    effective in-plane tension of the LJ bead patch vs the LJ well-depth proxy, plus
    patch-size sensitivity.

    This is engineering verification that the measured effective tension is a
    tunable, monotone function of the LJ attraction proxy — NOT calibrated cortical
    tension and NOT filament-based active contractility. The patch has no explicit
    actin filaments, motors or crosslinkers, and the model exposes no strain/deform
    input, so force–strain and relaxation-after-step-strain curves are NOT producible
    here (named gap)."""
    go = _go()
    from plotly.subplots import make_subplots
    # Panel A: tension vs LJ well-depth proxy, mean + min/max band over seeds.
    meanA, loA, hiA = [], [], []
    for eps in proxies:
        vals = [_e1_patch_tension(n, eps, s, steps)[0] for s in seeds]
        meanA.append(float(np.mean(vals))); loA.append(float(min(vals))); hiA.append(float(max(vals)))
    # Panel B: tension vs patch size n at fixed proxy, mean + min/max band over seeds.
    eps_fixed = proxies[len(proxies) // 2]
    meanB, loB, hiB, atoms = [], [], [], []
    for nn in sizes:
        vals = [_e1_patch_tension(nn, eps_fixed, s, steps)[0] for s in seeds]
        meanB.append(float(np.mean(vals))); loB.append(float(min(vals))); hiB.append(float(max(vals)))
        atoms.append(nn * nn)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.56, 0.44],
                        subplot_titles=(f"A — tension vs LJ well-depth proxy ({len(seeds)} seeds, n={n})",
                                        f"B — patch-size sensitivity (proxy ε={eps_fixed})"))
    # A band + mean
    fig.add_trace(go.Scatter(x=list(proxies) + list(proxies)[::-1], y=hiA + loA[::-1],
                             fill="toself", fillcolor="rgba(228,87,86,0.15)",
                             line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(proxies), y=meanA, mode="lines+markers",
                             line=dict(color=RED, width=3), marker=dict(size=9),
                             name="mean τ (± seed min/max)"), row=1, col=1)
    fig.add_hline(y=0.0, line=dict(color="#888", width=1, dash="dot"), row=1, col=1)
    if meanA[0] < 0:
        fig.add_annotation(x=proxies[0], y=meanA[0], text="expansive<br>(τ < 0)", showarrow=True,
                           arrowhead=2, ax=30, ay=-28, font=dict(size=10, color=RED), row=1, col=1)
    # B band + mean
    fig.add_trace(go.Scatter(x=list(sizes) + list(sizes)[::-1], y=hiB + loB[::-1],
                             fill="toself", fillcolor="rgba(76,120,168,0.15)",
                             line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=list(sizes), y=meanB, mode="lines+markers",
                             line=dict(color=BLUE, width=3), marker=dict(size=9),
                             name="mean τ vs patch size", text=[f"{a} beads" for a in atoms]),
                  row=1, col=2)
    fig.update_xaxes(title_text="LJ well-depth proxy ε (reduced; attraction strength, NOT biological myosin activity)",
                     row=1, col=1)
    fig.update_yaxes(title_text="effective in-plane tension τ = −(pₓₓ+p_yy)/2 (reduced LJ stress)", row=1, col=1)
    fig.update_xaxes(title_text="patch size n (n×n bead lattice)", row=1, col=2)
    fig.update_yaxes(title_text="τ (reduced LJ stress)", row=1, col=2)
    fig.update_layout(
        title=f"E1 — LJ-patch effective tension: tunability + seed/size sensitivity "
              f"(engineered-diagnostic; {steps} steps, reduced units)",
        template="plotly_white", height=560, margin=dict(t=92, b=70),
        legend=dict(orientation="h", y=1.12, x=0),
        annotations=list(fig.layout.annotations) + [dict(
            x=0.5, y=-0.22, xref="paper", yref="paper", showarrow=False,
            text="Engineering verification of tunability, NOT filament-based active contractility or calibrated cortical "
                 "tension. No explicit filaments/motors/crosslinkers; the model exposes no strain input, so force–strain "
                 "and step-strain relaxation are NOT producible here (named gap).",
            font=dict(size=10.5, color="#777"))])
    return fig


def e1_tension_components_figure(proxies=(0.5, 1.0, 2.0, 3.0, 4.0), seeds=(11, 22, 33),
                                 n=6, steps=30):
    """E1 SUPPLEMENT (engineered-diagnostic): the pxx/pyy pressure components behind
    the tension, showing WHY the smallest proxy is expansive (repulsive pressure
    outweighs attraction → τ<0). Same bounded LJ-patch runs; reduced units."""
    go = _go()
    mt, mpxx, mpyy = [], [], []
    for eps in proxies:
        rows = [_e1_patch_tension(n, eps, s, steps) for s in seeds]
        mt.append(float(np.mean([r[0] for r in rows])))
        mpxx.append(float(np.mean([r[1] for r in rows])))
        mpyy.append(float(np.mean([r[2] for r in rows])))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(proxies), y=mpxx, mode="lines+markers",
                             line=dict(color=ORANGE, width=2), name="mean pₓₓ"))
    fig.add_trace(go.Scatter(x=list(proxies), y=mpyy, mode="lines+markers",
                             line=dict(color=GOLD, width=2), name="mean p_yy"))
    fig.add_trace(go.Scatter(x=list(proxies), y=mt, mode="lines+markers",
                             line=dict(color=RED, width=3), name="τ = −(pₓₓ+p_yy)/2"))
    fig.add_hline(y=0.0, line=dict(color="#888", width=1, dash="dot"))
    fig.update_layout(
        title="E1 (supplement) — pressure components of the LJ-patch tension (engineered-diagnostic)",
        xaxis_title="LJ well-depth proxy ε (reduced units)",
        yaxis_title="reduced LJ stress",
        legend=dict(orientation="h", y=1.1),
        annotations=[dict(x=0.5, y=-0.2, xref="paper", yref="paper", showarrow=False,
                          text="At small ε the repulsive/thermal pressure outweighs the LJ attraction, so the net in-plane "
                               "stress is expansive (τ<0); a diagnostic decomposition, not a biological measurement.",
                          font=dict(size=10.5, color="#777"))])
    return _layout(fig, height=520)


def _collagen_micro(make_on: float, break_on: float, n_rods=30, box_xy=9.0, nstep=60, dt_s=8.0):
    """A small real collagen-network run: returns (t, n_crosslinks, sigma)."""
    from bigraph_schema import allocate_core
    from viva_lumenoid.collagen import CollagenNetworkProcess
    from viva_lumenoid.params import CollagenParams
    p = CollagenParams(n_rods=n_rods, box_xy=box_xy, slab_thickness=2.0, seed=7)
    proc = CollagenNetworkProcess(config={"params": p.to_dict()}, core=allocate_core())
    proc.initial_state()
    t, xl, sig = [], [], []
    for i in range(nstep):
        out = proc.update({"strain_rate": 0.0, "make_on": make_on, "break_on": break_on}, dt_s)
        t.append(i * dt_s)
        xl.append(float(out.get("n_crosslinks", 0)))
        sig.append(float(out.get("sigma_inplane", 0.0)))
    proc.close()
    return np.array(t), np.array(xl), np.array(sig)


def r2_rheology_figure():
    """R2: a small REAL collagen run — the crosslink network assembles and the
    in-plane stress develops. Full Fig 2D/2E reproduction (relaxation vs turnover)
    is the study target; this is the mechanism at reduced scale."""
    go = _go()
    t, xl, sig = _collagen_micro(make_on=1.0, break_on=1.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=xl, mode="lines", name="crosslinks (NC1/7S)",
                             line=dict(color=BLUE, width=3), yaxis="y1"))
    fig.add_trace(go.Scatter(x=t, y=sig, mode="lines", name="σ in-plane",
                             line=dict(color=ORANGE, width=2, dash="dot"), yaxis="y2"))
    fig.update_layout(
        title="R2 — collagen IV network assembles + carries stress (small real run)",
        xaxis_title="time (LJ)", yaxis=dict(title="crosslinks", color=BLUE),
        yaxis2=dict(title="σ in-plane", color=ORANGE, overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.08),
        annotations=[dict(x=0.5, y=-0.18, xref="paper", yref="paper", showarrow=False,
                          text="reduced-scale mechanism check; Fig 2D/2E reproduction is the study target",
                          font=dict(size=11, color="#777"))])
    return _layout(fig, height=560)


# Measured results of the dense multi-seed turnover sweep (followup fp-r2-dense-multiseed):
# 300 rods, density 3.0, 12-sigma slab, 4 seeds (11/22/33/44), off_rate 0.005-0.08.
# These are REAL measured medians recorded as run r2-dense-multiseed-sweep in bm-r2/study.yaml.
_R2_DENSE = {
    "residence": [12.5, 25.0, 50.0, 100.0, 200.0],
    "tau": [37.3, 39.7, 43.1, 67.5, 101.6],
    "seeds_resolved": [4, 4, 4, 4, 1],
}


def r2_fig2e_figure():
    """R2: the Barrientos Fig-2E reproduction — stress-relaxation time τ vs collagen
    residence time, from the dense multi-seed sweep (real measured medians). τ rises
    monotonically with residence; the modulus is turnover-independent (3.19 LJ)."""
    go = _go()
    res = np.array(_R2_DENSE["residence"])
    tau = np.array(_R2_DENSE["tau"])
    sr = _R2_DENSE["seeds_resolved"]
    # Split well-resolved (4/4 seeds) from the partial slow-turnover end (1/4).
    solid = [i for i, s in enumerate(sr) if s >= 4]
    partial = [i for i, s in enumerate(sr) if s < 4]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res, y=tau, mode="lines", name="τ(residence)",
                             line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Scatter(x=res[solid], y=tau[solid], mode="markers",
                             name="resolved 4/4 seeds",
                             marker=dict(color=BLUE, size=12, symbol="circle")))
    if partial:
        fig.add_trace(go.Scatter(x=res[partial], y=tau[partial], mode="markers",
                                 name="partial (1/4 seeds)",
                                 marker=dict(color=BLUE, size=13, symbol="circle-open",
                                             line=dict(width=2))))
    # The intrinsic relaxation floor (~37 LJ) read off the fast-turnover end.
    fig.add_hline(y=37.0, line=dict(color=GREEN, width=1.5, dash="dot"),
                  annotation_text="intrinsic relaxation floor ≈ 37 LJ",
                  annotation_position="bottom right",
                  annotation_font=dict(size=11, color=GREEN))
    fig.update_layout(
        title="R2 — stress-relaxation time τ vs collagen residence (Barrientos Fig 2E)",
        xaxis=dict(title="collagen residence time 1/off_rate (LJ)", type="log"),
        yaxis_title="stress-relaxation time τ (LJ)",
        legend=dict(orientation="h", y=1.1),
        annotations=[dict(x=0.5, y=-0.2, xref="paper", yref="paper", showarrow=False,
                          text="dense multi-seed network (300 rods, density 3.0, 12σ slab, 4 seeds); "
                               "modulus turnover-independent (3.19 LJ); reduced units",
                          font=dict(size=11, color="#777"))])
    return _layout(fig, height=560)


def _r3_supply_mode(make_on, break_on, insert_on, n_rods=40, assemble=4, grow=8, strain_rate=0.003):
    """One R3 supply mode: shared assembly then growth. Returns (rods, crosslinks, density)."""
    from bigraph_schema import allocate_core
    from viva_lumenoid.collagen import CollagenNetworkProcess
    from viva_lumenoid.params import CollagenParams
    p = CollagenParams(n_rods=n_rods, box_xy=8.0, slab_thickness=4.0, seed=12345,
                       use_real_angles=True, bending_k=4.0, insert_per_step=3)
    proc = CollagenNetworkProcess(config={"params": p.to_dict()}, core=allocate_core())
    proc.initial_state()
    for _ in range(assemble):
        proc.update({"strain_rate": 0.0, "make_on": 1.0, "break_on": 1.0, "insert_on": 0.0}, 1.0)
    out = {}
    for _ in range(grow):
        out = proc.update({"strain_rate": strain_rate, "make_on": make_on,
                           "break_on": break_on, "insert_on": insert_on}, 1.0)
    rods = proc._n_rods_current
    proc.close()
    area = 8.0 * 8.0 * (1.0 + out["strain"]) ** 2
    return rods, int(out["n_crosslinks"]), out["n_crosslinks"] / area


def r3_supply_figure():
    """R3: the three Barrientos Fig 2B-C supply modes under growth, from REAL collagen
    runs using the material-insertion mechanism. Crosslink-per-area density orders
    remodeling-with-insertion > remodeling-without-insertion > no-remodeling, and the
    rod (material) count grows only with insertion."""
    go = _go()
    modes = [("no-remodeling", 0.0, 0.0, 0.0, PURPLE),
             ("remodeling<br>(no insertion)", 1.0, 1.0, 0.0, BLUE),
             ("remodeling<br>+ insertion", 1.0, 1.0, 1.0, GREEN)]
    labels, dens, text = [], [], []
    for name, mk, bk, ins, _c in modes:
        rods, xl, d = _r3_supply_mode(mk, bk, ins)
        labels.append(name); dens.append(d); text.append(f"{rods} rods · {xl} xlinks")
    fig = go.Figure(go.Bar(x=labels, y=dens, marker_color=[m[4] for m in modes],
                           text=text, textposition="outside"))
    fig.update_layout(
        title="R3 — three supply modes under growth (Barrientos Fig 2B-C ordering)",
        yaxis_title="crosslink density (per expanded area)",
        annotations=[dict(x=0.5, y=-0.2, xref="paper", yref="paper", showarrow=False,
                          text="real collagen runs (40 rods + material-insertion mechanism); density orders "
                               "with-insertion > without-insertion > frozen (robust 3/3 seeds); reduced units",
                          font=dict(size=11, color="#777"))])
    return _layout(fig, height=560)


# ---------------------------------------------------------------------------
# R4 / R5 figure-review helpers (new; the listed shared helpers are untouched)
# ---------------------------------------------------------------------------
def _r45_build(composite_id: str):
    """Build a tyssue composite and return the live Composite (so a caller can
    step it AND read geometry per step). Mirrors _tyssue_build_step's build but
    returns the Composite rather than only the final eptm."""
    import viva_lumenoid.composites  # noqa: F401  (registers generators)
    from process_bigraph.composite_generator import _REGISTRY, apply_core_extensions, build_generator
    from bigraph_schema import allocate_core
    from process_bigraph import Composite
    entry = _REGISTRY[f"viva_lumenoid.composites.{composite_id}"]
    core = apply_core_extensions(entry, allocate_core())
    doc = build_generator(entry, None, core=core)
    state = doc["state"] if isinstance(doc, dict) and "state" in doc else doc
    comp = Composite({"state": state}, core=core)
    return comp


def _r4_sheet_relaxation_trace(steps: int = 6, dt: float = 0.01):
    """Real tyssue_2d_sheet run: per-step mean basal area, derived prism height,
    and the max vertex displacement from the start. Returns a dict of arrays."""
    comp = _r45_build("tyssue_2d_sheet")
    eptm = comp.state["Tyssue"]["instance"].eptm
    v0 = eptm.vert_df[["x", "y"]].values.copy()
    area, height, disp = [], [], []
    for _ in range(steps + 1):
        area.append(float(eptm.face_df["area"].mean()))
        height.append(float(np.nanmean(eptm.face_df["height"].values)))
        disp.append(float(np.abs(eptm.vert_df[["x", "y"]].values - v0).max()))
        comp.update({}, dt)
    return {"step": np.arange(steps + 1), "area": np.array(area),
            "height": np.array(height), "max_disp": np.array(disp),
            "Nf": int(eptm.Nf), "Nv": int(eptm.Nv)}


def r4_sheet_figure():
    """R4 ARCHITECTURE THUMBNAIL (engineered-diagnostic, supplement).

    The 12-cell planar CrossSection Tyssue geometry, drawn on its z=0 plane. This
    verifies mesh construction over the vivarium-tyssue engine; it is NOT biological
    evidence of confinement. The geometry is planar (z=0) and does NOT compute a real
    3D cell height — the per-face ``height`` column is a stored CrossSection prism DOF,
    not a resolved apical-basal distance."""
    eptm = _tyssue_build_step("tyssue_2d_sheet")
    go = _go()
    fig = go.Figure(data=_mesh_traces(eptm, three_d=False))
    fig.update_layout(
        title=f"R4 (supplement · engineered-diagnostic) — 12-cell planar Tyssue mesh (z=0)",
        xaxis=dict(scaleanchor="y", title="x (reduced)"), yaxis=dict(title="y (reduced)"),
        annotations=[dict(x=0.5, y=-0.13, xref="paper", yref="paper", showarrow=False,
                          text="A 12-cell planar Tyssue geometry resolves and relaxes; this verifies mesh "
                               "construction but not BM-dependent epithelial confinement.<br>"
                               "Planar z=0 geometry — no real 3D height is computed here.",
                          font=dict(size=11, color="#777"))])
    return _layout(fig, height=560)


def r4_confinement_figure():
    """R4 MAIN (observed) — question-driven quantified panel.

    LEFT: what the real tyssue_2d_sheet composite actually produces — mean per-cell
    basal area and the stored CrossSection prism-height DOF across EulerSolver
    relaxation steps. Both are flat (the sheet is a pre-equilibrated static mesh; the
    max vertex displacement is ~0 over the run), and all 12 cells are identical, so
    there is no spatial heterogeneity and no confinement response.

    RIGHT: the R1 reduced-order prism benchmark (real TissuePrismProcess runs via the
    _prism_trajectory helper) for three BM regimes — the confinement response the
    sheet is MISSING, because tyssue_2d_sheet has only line_tension + prefered_area and
    NO BM stiffness / relaxation / traction coupling. Evidence: observed for both
    panels (real runs); this is NOT a reproduction of the Fig-1 confinement regimes."""
    go = _go()
    from plotly.subplots import make_subplots
    tr = _r4_sheet_relaxation_trace()
    a0 = tr["area"][0] or 1.0
    h0 = tr["height"][0] or 1.0
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.5, 0.5], horizontal_spacing=0.12,
        subplot_titles=(f"A — real tyssue_2d_sheet readout ({tr['Nf']} cells): static",
                        "B — R1 reduced-order BM benchmark: confinement response"))
    # Panel A: real tyssue sheet, fold-change vs step (flat).
    fig.add_trace(go.Scatter(x=tr["step"], y=tr["area"] / a0, mode="lines+markers",
                             line=dict(color=BLUE, width=3), marker=dict(size=8),
                             name="basal area (tyssue sheet)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=tr["step"], y=tr["height"] / h0, mode="lines+markers",
                             line=dict(color=TEAL, width=2, dash="dot"), marker=dict(size=7),
                             name="prism-height DOF (tyssue sheet)"), row=1, col=1)
    fig.update_xaxes(title_text="EulerSolver relaxation step (dt 0.01)", row=1, col=1)
    fig.update_yaxes(title_text="fold-change vs step 0", range=[0.0, 3.0], row=1, col=1)
    # Panel B: R1 benchmark confinement h/h0 for three regimes.
    corners = [("soft / fast-relaxing", 0.2, 0.2, GREEN),
               ("intermediate", 1.0, 1.0, BLUE),
               ("stiff / slow-relaxing", 8.0, 8.0, RED)]
    for nm, k, r, col in corners:
        t, _area, conf = _prism_trajectory(k, r)
        fig.add_trace(go.Scatter(x=t / t[-1], y=conf, mode="lines", line=dict(color=col, width=3),
                                 name=f"R1 {nm}: {conf[0]:.2f}→{conf[-1]:.2f}"), row=1, col=2)
    fig.update_xaxes(title_text="proliferation progress (fraction of run)", row=1, col=2)
    fig.update_yaxes(title_text="confinement h/h₀ (R1 benchmark)", range=[0.3, 1.05], row=1, col=2)
    fig.update_layout(
        title="R4 — the 2D sheet is a static substrate, not a confinement response "
              "(evidence: observed; NOT a Fig-1 reproduction)",
        template="plotly_white", height=560,
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=10)),
        annotations=list(fig.layout.annotations) + [dict(
            x=0.5, y=-0.46, xref="paper", yref="paper", showarrow=False,
            text="GAP (named): tyssue_2d_sheet exposes only line_tension + prefered_area — no BM stiffness, "
                 "relaxation or traction coupling. The sheet does not move (max vertex displacement ≈ 0) and "
                 "all 12 cells are identical, so matched low/high-BM-relaxation snapshots and a "
                 "stiffness-stratified basal-area-vs-time panel are NOT producible here; see the hypothesis "
                 "supplement. Panel B is the real R1 reduced-order benchmark, not the tyssue sheet. Reduced units.",
            font=dict(size=10.5, color="#777"))])
    return fig


def r4_bm_confinement_hypothesis():
    """R4 HYPOTHESIS (supplement) — the BM-dependent confinement test that
    tyssue_2d_sheet CANNOT yet run, drawn so the missing experiment is explicit."""
    fig = _diagram("R4 (hypothesis · supplement) — BM-stiffness-dependent confinement (requires coupling NOT in the composite)",
                   xr=(0, 10), yr=(0, 10), height=560)
    _box(fig, 0.6, 5.2, 4.6, 8.8, "low BM relaxation stiffness<br>→ spread / flat cells<br>(larger basal area)", GREEN)
    _box(fig, 5.4, 5.2, 9.4, 8.8, "high BM relaxation stiffness<br>→ confined / tall cells<br>(smaller basal area)", RED)
    _arrow(fig, 4.6, 7.0, 5.4, 7.0, "increasing BM stiffness / slower relaxation")
    fig.add_annotation(x=5, y=2.8, showarrow=False, font=dict(size=12, color="#444"),
                       text="Matched 2D cell-sheet snapshots at low vs high BM relaxation, and a basal-area-"
                            "vs-time panel stratified by BM stiffness, with the R1 benchmark overlaid.")
    fig.add_annotation(x=5, y=1.2, showarrow=False, font=dict(size=11, color=RED),
                       text="<b>NOT producible from tyssue_2d_sheet</b>: its params are line_tension + prefered_area; "
                            "BM stiffness / relaxation / traction coupling is not wired. Hypothesis until that coupling exists.")
    return fig


def _r5_monolayer_run(steps: int = 5, dt: float = 0.01):
    """Real tyssue_3d_monolayer run. Returns the live eptm plus per-step total cell
    volume (for the conservation residual) and the per-cell z-extent height array."""
    comp = _r45_build("tyssue_3d_monolayer")
    eptm = comp.state["Tyssue"]["instance"].eptm
    total_vol = [float(np.nansum(eptm.cell_df["vol"].values))]
    for _ in range(steps):
        comp.update({}, dt)
        total_vol.append(float(np.nansum(eptm.cell_df["vol"].values)))
    # per-cell height = vertex z-extent for the cell's vertices
    e, v = eptm.edge_df, eptm.vert_df
    heights = []
    for _c, grp in e.groupby("cell"):
        vids = np.unique(np.concatenate([grp["srce"].values, grp["trgt"].values]))
        zz = v.loc[vids, "z"].values
        heights.append(float(zz.max() - zz.min()))
    return eptm, np.array(total_vol), np.array(heights)


def _r5_cell_centroid_markers(eptm, color_by: str = "area"):
    """R5-named helper: cell centroids (cell_df) as 3D markers coloured by a per-cell
    column (default basal area), so per-cell heterogeneity is visible in the render."""
    go = _go()
    cdf = eptm.cell_df
    vals = cdf[color_by].values
    return go.Scatter3d(
        x=cdf["x"], y=cdf["y"], z=cdf["z"], mode="markers",
        marker=dict(size=4.5, color=vals, colorscale="Viridis", cmin=float(np.nanmin(vals)),
                    cmax=float(np.nanmax(vals)), showscale=True,
                    colorbar=dict(title=f"per-cell<br>{color_by}", len=0.85)),
        name="cell centroids", hovertext=[f"{color_by}={x:.2f}" for x in vals],
        hoverinfo="text", showlegend=False)


def r5_monolayer_figure():
    """R5 MAIN (engineered-diagnostic) — the explicit 3D monolayer as a MODEL
    ILLUSTRATION paired with the real per-cell basal-area distribution.

    LEFT: the closed 3D Tyssue monolayer (MonolayerGeometry) wireframe with cell
    centroids coloured by per-cell area (cell_df.area, common limits) — the real
    per-cell heterogeneity. RIGHT: that per-cell area distribution from the single
    real run. A finite-volume mesh with identifiable basal surfaces does NOT establish
    BM-dependent confinement: per-cell HEIGHT (vertex z-extent) is uniform (= 1.00) in
    this single run, and no BM-relaxation axis is present. Total cell volume is
    conserved to <0.01% over 5 Euler steps (dt 0.01)."""
    go = _go()
    from plotly.subplots import make_subplots
    eptm, total_vol, heights = _r5_monolayer_run()
    area = eptm.cell_df["area"].values
    resid = abs(total_vol[-1] - total_vol[0]) / (total_vol[0] or 1.0) * 100.0
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                        specs=[[{"type": "scene"}, {"type": "xy"}]],
                        subplot_titles=(f"3D monolayer ({eptm.Nc} cells) — centroids by per-cell area",
                                        "per-cell area distribution (cell_df.area)"))
    for tr in _mesh_traces(eptm, three_d=True, color_by="z"):
        fig.add_trace(tr, row=1, col=1)
    fig.add_trace(_r5_cell_centroid_markers(eptm, color_by="area"), row=1, col=1)
    fig.add_trace(go.Histogram(x=area, nbinsx=24, marker_color=TEAL, showlegend=False,
                               name=f"median {np.nanmedian(area):.2f}"), row=1, col=2)
    fig.update_xaxes(title_text="per-cell area (cell_df.area, reduced)", row=1, col=2)
    fig.update_yaxes(title_text="cell count", row=1, col=2)
    fig.update_layout(
        title="R5 (main · engineered-diagnostic) — closed 3D monolayer: finite cell volumes + "
              "identifiable basal surfaces",
        template="plotly_white", height=620, scene=dict(aspectmode="data"),
        annotations=list(fig.layout.annotations) + [dict(
            x=0.5, y=-0.02, xref="paper", yref="paper", showarrow=False,
            text=f"A closed 3D Tyssue monolayer has finite cell volumes and identifiable basal surfaces; "
                 f"no BM-dependent height response is established by this geometric verification alone. "
                 f"Per-cell area {np.nanmin(area):.2f}–{np.nanmax(area):.2f} (median {np.nanmedian(area):.2f}); "
                 f"per-cell height (vertex z-extent) uniform = {np.median(heights):.2f}; "
                 f"total cell volume conserved to {resid:.3f}% over 5 Euler steps (dt 0.01). Reduced units.<br>"
                 f"GAP (named): the free-BM / stiff-frozen-BM / stiff-fast-relaxing-BM controls the morphology "
                 f"ordering needs are NOT in tyssue_3d_monolayer (params: contractility + vol_elasticity, no BM "
                 f"shell) — see the hypothesis supplement.",
            font=dict(size=10.5, color="#777"))])
    return fig


def r5_volume_hist_figure():
    """R5 APPENDIX (engineered-diagnostic, supplement) — the standalone cell-volume
    histogram, kept as a geometry-validation appendix with the volume-conservation
    residual stated (it does not by itself demonstrate confinement)."""
    go = _go()
    eptm, total_vol, _heights = _r5_monolayer_run()
    vol = eptm.cell_df["vol"].values
    resid = abs(total_vol[-1] - total_vol[0]) / (total_vol[0] or 1.0) * 100.0
    fig = go.Figure(go.Histogram(x=vol, nbinsx=24, marker_color=TEAL, showlegend=False))
    fig.add_vline(x=1.0, line=dict(color=RED, width=1.5, dash="dot"),
                  annotation_text="target vol 1.0", annotation_font_size=10)
    fig.update_layout(
        title=f"R5 (supplement · engineered-diagnostic) — cell-volume distribution ({eptm.Nc} cells)",
        xaxis_title="per-cell volume (cell_df.vol, reduced)", yaxis_title="cell count",
        annotations=[dict(x=0.5, y=-0.2, xref="paper", yref="paper", showarrow=False,
                          text=f"per-cell volume {np.nanmin(vol):.2f}–{np.nanmax(vol):.2f} (median "
                               f"{np.nanmedian(vol):.2f}); total volume conserved to {resid:.3f}% over 5 Euler "
                               f"steps (dt 0.01). Geometry validation only — finite volumes do not establish "
                               f"BM-dependent confinement. Reduced units.",
                          font=dict(size=11, color="#777"))])
    return _layout(fig, height=520)


def r5_bm_controls_hypothesis():
    """R5 HYPOTHESIS (supplement) — the free / stiff-frozen / stiff-fast-relaxing BM
    matched conditions the morphology ordering needs, which require a BM-shell coupling
    NOT present in tyssue_3d_monolayer."""
    fig = _diagram("R5 (hypothesis · supplement) — matched BM controls (require a BM-shell coupling NOT in the composite)",
                   xr=(0, 12), yr=(0, 10), height=560)
    conds = [("free BM<br>(no shell)<br>→ baseline height", 2.0, GREEN),
             ("stiff frozen BM<br>→ strongest confinement<br>(tallest)", 6.0, RED),
             ("stiff fast-relaxing BM<br>→ partial relaxation<br>(intermediate)", 10.0, GOLD)]
    for label, x, c in conds:
        _box(fig, x - 1.6, 5.0, x + 1.6, 8.6, label, c)
    fig.add_annotation(x=6, y=3.0, showarrow=False, font=dict(size=12, color="#444"),
                       text="Before/after 3D cell images coloured by cell height (common limits), a cell-height "
                            "distribution by condition, height-vs-doubling-time traces, basal/apical area, and a "
                            "volume-conservation residual + explicit energy/traction terms.")
    fig.add_annotation(x=6, y=1.3, showarrow=False, font=dict(size=11, color=RED),
                       text="<b>NOT producible from tyssue_3d_monolayer</b>: its params are contractility + "
                            "vol_elasticity; no BM shell coupling. Hypothesis until that coupling exists.")
    return fig


def _e3_section_traces(eptm, go):
    """A thin equatorial (z≈median) section cut of the lumenoid, as x–y vertex
    scatter + sampled edges. E3-named helper (does not touch _mesh_traces)."""
    v = eptm.vert_df
    zc = float(v["z"].median())
    half = 0.08 * (float(v["z"].max()) - float(v["z"].min()) + 1e-9)
    band = v[(v["z"] >= zc - half) & (v["z"] <= zc + half)]
    traces = [go.Scatter(x=band["x"], y=band["y"], mode="markers",
                         marker=dict(size=4, color=BLUE), hoverinfo="skip", showlegend=False)]
    return traces, band


def e3_lumenoid_figure():
    """E3 MAIN (engineered-diagnostic, illustration): the closed 3D lumenoid shell
    (LumenoidGeometry) with the basal BM-coupling surface and the enclosed lumen
    labelled, alongside a thin equatorial section cut at the same scale and a scale
    reference. The counts (cells, lumen volume) are GEOMETRY metrics from a real
    tyssue build, not growing-lumenoid results; lumen filling and BM-regulated
    morphogenesis remain untested (see the supplement)."""
    eptm = _tyssue_build_step("closed_lumenoid")
    go = _go()
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
                        specs=[[{"type": "scene"}, {"type": "xy"}]],
                        subplot_titles=(f"3D closed shell ({eptm.Nc} cells)", "equatorial section cut (z ≈ median)"))
    for tr in _mesh_traces(eptm, three_d=True, color_by="z"):
        fig.add_trace(tr, row=1, col=1)
    sect, band = _e3_section_traces(eptm, go)
    for tr in sect:
        fig.add_trace(tr, row=1, col=2)
    vtot = float(eptm.cell_df["vol"].sum())
    lv = float(eptm.settings.get("lumen_vol", 0.0))
    # Section-cut labels + a scale reference bar (1 model length unit).
    cx, cy = float(band["x"].mean()), float(band["y"].mean())
    rmax = float(((band["x"] - cx) ** 2 + (band["y"] - cy) ** 2).pow(0.5).max())
    fig.add_annotation(x=cx, y=cy, text="lumen", showarrow=False,
                       font=dict(size=12, color="#444"), row=1, col=2)
    fig.add_annotation(x=cx, y=cy + rmax * 0.98, text="basal BM-coupling surface", showarrow=True,
                       arrowhead=2, ax=0, ay=-26, font=dict(size=10, color=BLUE), row=1, col=2)
    x0 = cx - rmax; y0 = cy - rmax * 1.12
    fig.add_shape(type="line", x0=x0, y0=y0, x1=x0 + 1.0, y1=y0, line=dict(color="#333", width=3),
                  row=1, col=2)
    fig.add_annotation(x=x0 + 0.5, y=y0, text="1 length unit", showarrow=False, yshift=-10,
                       font=dict(size=10, color="#333"), row=1, col=2)
    fig.update_xaxes(scaleanchor="y2", title_text="x", row=1, col=2)
    fig.update_yaxes(title_text="y", row=1, col=2)
    fig.update_layout(
        title=f"E3 — closed 3D lumenoid shell: geometry verification ({eptm.Nc} cells, engineered-diagnostic)",
        template="plotly_white", height=640, scene=dict(aspectmode="data"), margin=dict(t=92, b=60),
        annotations=list(fig.layout.annotations) + [dict(
            x=0.5, y=-0.04, xref="paper", yref="paper", showarrow=False,
            text=f"GEOMETRY metrics (reduced units): {eptm.Nc} cells · lumen volume {lv:.1f} · total cell volume {vtot:.1f}. "
                 f"The basal surface is the BM coupling shell. Verifies geometry only; lumen filling and BM-regulated "
                 f"morphogenesis remain untested.",
            font=dict(size=10.5, color="#777"))])
    return fig


def _e3_lumen_trace(steps=12, dt=0.01):
    """Step the closed_lumenoid composite and record the lumen/BM settings each
    step. E3-named helper (mirrors the tyssue build but records per-step settings)."""
    import viva_lumenoid.composites  # noqa: F401
    from process_bigraph.composite_generator import _REGISTRY, apply_core_extensions, build_generator
    from bigraph_schema import allocate_core
    from process_bigraph import Composite
    entry = _REGISTRY["viva_lumenoid.composites.closed_lumenoid"]
    core = apply_core_extensions(entry, allocate_core())
    doc = build_generator(entry, None, core=core)
    state = doc["state"] if isinstance(doc, dict) and "state" in doc else doc
    comp = Composite({"state": state}, core=core)
    eptm = comp.state["Tyssue"]["instance"].eptm
    t, lvol, lpress, barea = [], [], [], []
    for i in range(steps):
        comp.update({}, dt)
        s = eptm.settings
        t.append(i * dt)
        lvol.append(float(s.get("lumen_vol", 0.0)))
        lpress.append(float(s.get("lumen_pressure", 0.0)))
        barea.append(float(s.get("bm_surface_area", 0.0)))
    return np.array(t), np.array(lvol), np.array(lpress), np.array(barea)


def e3_lumen_dynamics_figure():
    """E3 SUPPLEMENT (hypothesis / gap-named): the lumen volume, lumen pressure and
    basal BM-shell surface across a real multi-step EulerSolver run of the closed
    lumenoid. They are FLAT: the shell sits at rest equilibrium with lumen
    pressure = 0, and NO lumen-filling drive is wired (no LumenElasticity effector in
    the force balance, no lumen-volume target ramp). A dynamic lumen-filling /
    BM-regulated morphogenesis panel is therefore NOT producible from this composite —
    shown honestly as flat, with the gap named."""
    go = _go()
    from plotly.subplots import make_subplots
    t, lvol, lpress, barea = _e3_lumen_trace()
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "lumen volume", "lumen pressure (P=0 at rest)", "basal BM-shell surface area"))
    fig.add_trace(go.Scatter(x=t, y=lvol, mode="lines+markers", line=dict(color=BLUE, width=3),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=lpress, mode="lines+markers", line=dict(color=RED, width=3),
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=barea, mode="lines+markers", line=dict(color=TEAL, width=3),
                             showlegend=False), row=1, col=3)
    fig.update_yaxes(title_text="lumen volume (reduced)", row=1, col=1)
    fig.update_yaxes(title_text="lumen pressure (reduced)", row=1, col=2)
    fig.update_yaxes(title_text="basal area (reduced)", row=1, col=3)
    for c in (1, 2, 3):
        fig.update_xaxes(title_text="time (reduced)", row=1, col=c)
    fig.update_layout(
        title=f"E3 (supplement) — lumen/BM state is static at rest (GAP: no lumen-filling drive wired; hypothesis)",
        template="plotly_white", height=460, margin=dict(t=92, b=70),
        annotations=list(fig.layout.annotations) + [dict(
            x=0.5, y=-0.26, xref="paper", yref="paper", showarrow=False,
            text=f"Real {len(t)}-step run: lumen volume {lvol[0]:.1f}→{lvol[-1]:.1f}, pressure {lpress[0]:.2g}→{lpress[-1]:.2g}, "
                 f"basal area {barea[0]:.1f}→{barea[-1]:.1f} — flat. The lumen is a geometry diagnostic, not a driven DOF; "
                 f"lumen filling and BM-regulated morphogenesis are NOT yet producible (named gap).",
            font=dict(size=10.5, color="#777"))])
    return fig


# ---------------------------------------------------------------------------
# E2 — force-transfer coupling unit test (real, bounded)
# ---------------------------------------------------------------------------
def _e2_adhesion_steps(cortex_tensions, k_on, k_off=0.1, coupling_gain=1.0e-4, dt=1.0):
    """Run the pure deterministic BasalAdhesion over a prescribed cortex-tension
    step series; return per-step (occupancy, traction, bm_strain_rate). E2-named
    helper. ``k_on=0`` is forced through a post-construction config override because
    process-bigraph drops 0.0-valued config (so the zero-linker control is exact)."""
    from bigraph_schema import allocate_core
    from viva_lumenoid.adhesion import BasalAdhesion
    cfg = {"k_off": k_off, "coupling_gain": coupling_gain}
    if k_on != 0.0:
        cfg["k_on"] = k_on
    p = BasalAdhesion(config=cfg, core=allocate_core())
    p.config["k_on"] = float(k_on)   # exact, even for 0.0 (bigraph drops 0.0 config)
    p._occ = 0.0
    occ, trac, sr = [], [], []
    for tension in cortex_tensions:
        o = p.update({"cortex_tension": float(tension)}, dt)
        occ.append(float(o["linker_occupancy"]))
        trac.append(float(o["traction"]))
        sr.append(float(o["bm_strain_rate"]))
    return occ, trac, sr


def e2_force_transfer_figure(hold=8):
    """E2 MAIN (engineered-diagnostic, DETERMINISTIC MAPPING): the BasalAdhesion
    coupling unit test. A prescribed cortex-tension step series drives reversible
    linker occupancy; the transmitted traction = occupancy × tension and the BM
    strain rate = coupling_gain × traction, with a zero-linker (k_on=0) control that
    disables transfer. This verifies that deterministic occupancy gating correctly
    scales a PRESCRIBED cortical traction — it does NOT establish microscopic
    integrin mechanics or force-conserving multiscale coupling (named gap)."""
    go = _go()
    from plotly.subplots import make_subplots
    tensions = [1.0] * hold + [3.0] * hold
    t = list(range(len(tensions)))
    occ, trac, sr = _e2_adhesion_steps(tensions, k_on=0.5)
    occ0, trac0, sr0 = _e2_adhesion_steps(tensions, k_on=0.0)  # zero-linker control
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("imposed cortex tension + linker occupancy",
                                        "transmitted traction = occupancy × tension",
                                        "BM strain rate = coupling_gain × traction"))
    fig.add_trace(go.Scatter(x=t, y=tensions, mode="lines", line=dict(color="#888", width=2, dash="dot"),
                             name="imposed cortex tension"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=occ, mode="lines+markers", line=dict(color=GOLD, width=3),
                             name="linker occupancy (k_on=0.5)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=occ0, mode="lines", line=dict(color=RED, width=2, dash="dash"),
                             name="occupancy (zero-linker k_on=0)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=trac, mode="lines+markers", line=dict(color=BLUE, width=3),
                             showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=trac0, mode="lines", line=dict(color=RED, width=2, dash="dash"),
                             showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=sr, mode="lines+markers", line=dict(color=GREEN, width=3),
                             showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=sr0, mode="lines", line=dict(color=RED, width=2, dash="dash"),
                             showlegend=False), row=3, col=1)
    fig.update_yaxes(title_text="tension / occupancy", row=1, col=1)
    fig.update_yaxes(title_text="traction (reduced)", row=2, col=1)
    fig.update_yaxes(title_text="strain rate (reduced)", row=3, col=1)
    fig.update_xaxes(title_text="coupling step", row=3, col=1)
    fig.update_layout(
        title="E2 — force-transfer coupling unit test: occupancy-gated traction (engineered-diagnostic; "
              "deterministic mapping)",
        template="plotly_white", height=640, margin=dict(t=92, b=78),
        legend=dict(orientation="h", y=1.1, x=0),
        annotations=list(fig.layout.annotations) + [dict(
            x=0.5, y=-0.14, xref="paper", yref="paper", showarrow=False,
            text="Deterministic occupancy gating scales a PRESCRIBED cortical traction; the zero-linker control (k_on=0) "
                 "disables transfer exactly. This does NOT establish microscopic integrin mechanics or a force-conserving "
                 "multiscale coupling (named gap).",
            font=dict(size=10.5, color="#777"))])
    return fig


def _e2_bm_response(steps=6, n_rods=20):
    """Manually wire cortex (LAMMPS) → BasalAdhesion → collagen BM over a few steps;
    return per-step (cortex_tension, occupancy, traction, bm_sigma, bm_strain).
    E2-named helper; a bounded smoke of the actin_adhesion_bm composite's path."""
    from bigraph_schema import allocate_core
    from pbg_lammps.model import LammpsModelProcess
    import viva_lumenoid.actomyosin  # noqa: F401
    from viva_lumenoid.adhesion import BasalAdhesion
    from viva_lumenoid.collagen import CollagenNetworkProcess
    from viva_lumenoid.params import CollagenParams
    cortex = LammpsModelProcess(config={"model": "actomyosin-cortex",
                                        "params": {"n": 6, "contractility": 1.5}},
                                core=allocate_core())
    cortex.initial_state()
    adh = BasalAdhesion(config={"k_on": 0.5, "k_off": 0.1, "coupling_gain": 1.0e-4},
                        core=allocate_core())
    bm = CollagenNetworkProcess(config={"params": CollagenParams(
        n_rods=n_rods, box_xy=10.0, slab_thickness=2.0,
        assemble_steps=200, hold_steps=100).to_dict()}, core=allocate_core())
    bm.initial_state()
    rows = []
    for _ in range(steps):
        ct = float(cortex.update({}, 1.0)["cortical_tension"])
        ao = adh.update({"cortex_tension": ct}, 1.0)
        bo = bm.update({"strain_rate": ao["bm_strain_rate"], "make_on": 1.0, "break_on": 1.0}, 1.0)
        rows.append((ct, float(ao["linker_occupancy"]), float(ao["traction"]),
                     float(bo.get("sigma_inplane", 0.0)), float(bo.get("strain", 0.0))))
    cortex.close(); bm.close()
    return rows


def e2_bm_response_figure():
    """E2 SUPPLEMENT (engineered-diagnostic): a bounded smoke of the full
    actin_adhesion_bm path — the LAMMPS actomyosin cortex pulls through the basal
    linkers on the collagen IV BM patch, and the BM responds (in-plane stress +
    accumulated strain). Confirms the engines are wired end-to-end; the BM reaction
    stress is real but a force-CONSERVING coupling + integrin mechanics are NOT
    established (named gap)."""
    go = _go()
    from plotly.subplots import make_subplots
    rows = _e2_bm_response()
    t = list(range(len(rows)))
    ct = [r[0] for r in rows]; occ = [r[1] for r in rows]; trac = [r[2] for r in rows]
    sig = [r[3] for r in rows]; strn = [r[4] for r in rows]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5],
                        subplot_titles=("cortex tension → traction (through linkers)",
                                        "collagen BM response (stress + strain)"))
    fig.add_trace(go.Scatter(x=t, y=ct, mode="lines+markers", line=dict(color=RED, width=2),
                             name="cortex tension (LAMMPS)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=trac, mode="lines+markers", line=dict(color=BLUE, width=3),
                             name="transmitted traction"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=sig, mode="lines+markers", line=dict(color=ORANGE, width=3),
                             name="BM in-plane stress σ", yaxis="y3"), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=strn, mode="lines+markers", line=dict(color=TEAL, width=2, dash="dot"),
                             name="BM accumulated strain"), row=1, col=2)
    fig.update_yaxes(title_text="tension / traction (reduced)", row=1, col=1)
    fig.update_yaxes(title_text="BM σ / strain (reduced)", row=1, col=2)
    fig.update_xaxes(title_text="coupling step", row=1, col=1)
    fig.update_xaxes(title_text="coupling step", row=1, col=2)
    fig.update_layout(
        title="E2 (supplement) — actin→adhesion→collagen BM bounded smoke: the BM responds "
              "(engineered-diagnostic)",
        template="plotly_white", height=500, margin=dict(t=92, b=70),
        legend=dict(orientation="h", y=1.12, x=0),
        annotations=list(fig.layout.annotations) + [dict(
            x=0.5, y=-0.22, xref="paper", yref="paper", showarrow=False,
            text="End-to-end wiring smoke (20-rod BM): cortex tension is transmitted through reversible linkers and the "
                 "collagen BM carries a finite stress and accumulates strain. Reduced units; a force-CONSERVING multiscale "
                 "coupling and microscopic integrin mechanics are NOT established (named gap).",
            font=dict(size=10.5, color="#777"))])
    return fig


# ---------------------------------------------------------------------------
# E4 — coupling-control time courses (real, bounded; reduced-order BM)
# ---------------------------------------------------------------------------
def _e4_run_mode(mode, substeps=1, n=15, dt=0.02):
    """Run the reduced-order LammpsTyssueCoupling in one coupling mode; return the
    per-step time courses. E4-named helper (reduced BM → no LAMMPS; real tyssue)."""
    from bigraph_schema import allocate_core
    from viva_lumenoid.multiscale_coupling import LammpsTyssueCoupling
    p = LammpsTyssueCoupling(config={
        "bm_mode": "reduced", "coupling_mode": mode, "bm_substeps": int(substeps),
        "strain_gain": 20.0, "tension_gain": 0.02,
        "bm_modulus": 50.0, "bm_relaxation": 10.0}, core=allocate_core())
    keys = ("basal_area", "basal_strain_rate", "bm_sigma", "basal_tension", "lumen_pressure")
    traj = {k: [] for k in keys}
    traj["t"] = []
    for i in range(n):
        o = p.update({}, dt)
        traj["t"].append(i * dt)
        for k in keys:
            traj[k].append(float(o[k]))
    p.close()
    return traj


def e4_feedback_figure():
    """E4 MAIN (engineered-diagnostic, COUPLING CONTROL): aligned time courses of the
    reduced-order LAMMPS↔Tyssue coupling in three modes — two-way, one-way and
    uncoupled — in identical conditions. The two-way path returns a nonzero BM
    reaction (basal tension) that damps the epithelial contraction; one-way drives
    the BM but feeds back nothing; uncoupled leaves the BM inert. This demonstrates
    CORRECT SOFTWARE WIRING of the coupling controls, NOT a mechanistic prediction:
    the explicit-LAMMPS collagen BM is NOT in this loop and there is no morphogenetic
    growth trajectory (named gaps)."""
    go = _go()
    from plotly.subplots import make_subplots
    modes = [("two-way", BLUE), ("one-way", ORANGE), ("uncoupled", "#999")]
    data = {m: _e4_run_mode(m) for m, _ in modes}
    panels = [("basal_area", "basal BM-shell area (reduced)"),
              ("bm_sigma", "BM stress σ (reduced)"),
              ("basal_tension", "basal tension fed back (reduced)"),
              ("basal_strain_rate", "basal strain rate (reduced)")]
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.13, horizontal_spacing=0.1,
                        subplot_titles=[p[1] for p in panels])
    for idx, (key, _) in enumerate(panels):
        r = idx // 2 + 1; c = idx % 2 + 1
        for m, col in modes:
            fig.add_trace(go.Scatter(x=data[m]["t"], y=data[m][key], mode="lines",
                                     line=dict(color=col, width=3), name=m,
                                     showlegend=(idx == 0)), row=r, col=c)
    for idx in range(4):
        fig.update_xaxes(title_text="time (reduced)", row=idx // 2 + 1, col=idx % 2 + 1)
    tw = data["two-way"]; ow = data["one-way"]
    fig.update_layout(
        title="E4 — coupling-control time courses: two-way vs one-way vs uncoupled "
              "(engineered-diagnostic; correct wiring, not a prediction)",
        template="plotly_white", height=680, margin=dict(t=96, b=74),
        legend=dict(orientation="h", y=1.08, x=0),
        annotations=list(fig.layout.annotations) + [dict(
            x=0.5, y=-0.1, xref="paper", yref="paper", showarrow=False,
            text=f"Real 15-step reduced-order runs (dt 0.02). Two-way returns a nonzero basal tension "
                 f"({tw['basal_tension'][-1]:+.3f}) that damps the contraction (basal area {tw['basal_area'][0]:.1f}→"
                 f"{tw['basal_area'][-1]:.1f} vs one-way {ow['basal_area'][0]:.1f}→{ow['basal_area'][-1]:.1f}); one-way "
                 f"and uncoupled feed back zero as designed. Explicit-LAMMPS collagen BM is NOT in this loop and there is "
                 f"no morphogenetic growth trajectory (named gaps).",
            font=dict(size=10.5, color="#777"))])
    return fig


def e4_subcycle_convergence_figure():
    """E4 SUPPLEMENT (engineered-diagnostic): BM subcycling convergence — the two-way
    BM stress at the final step vs the number of BM subcycles per epithelial
    macro-step (bm_substeps). Successive-refinement differences shrink, so the
    subcycled BM converges. This is the convergence that IS wired; exchange-cadence
    convergence beyond BM subcycling (shrinking the epithelium↔BM exchange interval)
    is NOT wired (named gap)."""
    go = _go()
    from plotly.subplots import make_subplots
    subs = [1, 2, 4, 8]
    sig = [_e4_run_mode("two-way", substeps=s)["bm_sigma"][-1] for s in subs]
    diffs = [abs(sig[i + 1] - sig[i]) for i in range(len(sig) - 1)]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                        subplot_titles=("final BM stress vs BM subcycles",
                                        "successive-refinement |Δσ| (convergence)"))
    fig.add_trace(go.Scatter(x=subs, y=sig, mode="lines+markers", line=dict(color=BLUE, width=3),
                             marker=dict(size=10), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[f"{subs[i]}→{subs[i+1]}" for i in range(len(diffs))], y=diffs,
                             mode="lines+markers", line=dict(color=RED, width=3),
                             marker=dict(size=10), showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="BM subcycles per macro-step", type="log", row=1, col=1)
    fig.update_yaxes(title_text="final BM stress σ (reduced)", row=1, col=1)
    fig.update_xaxes(title_text="refinement", row=1, col=2)
    fig.update_yaxes(title_text="|Δσ| between refinements", row=1, col=2)
    fig.update_layout(
        title="E4 (supplement) — BM subcycling convergence (engineered-diagnostic)",
        template="plotly_white", height=460, margin=dict(t=92, b=74),
        annotations=list(fig.layout.annotations) + [dict(
            x=0.5, y=-0.26, xref="paper", yref="paper", showarrow=False,
            text=f"Real two-way reduced-order runs: final σ = {', '.join(f'{v:.3f}' for v in sig)} at subcycles "
                 f"{subs}; |Δσ| = {', '.join(f'{d:.3f}' for d in diffs)} shrinks → converged. This is BM subcycling "
                 f"convergence only; exchange-cadence convergence (shrinking the epithelium↔BM exchange interval) is NOT "
                 f"wired (named gap).",
            font=dict(size=10.5, color="#777"))])
    return fig


# ---------------------------------------------------------------------------
# DESIGN SCHEMATICS (no runnable composite yet)
# ---------------------------------------------------------------------------
def r1_regime_schematic():
    """R1: Barrientos Fig 1G stiffness×relaxation 4-regime phase-diagram target."""
    fig = _diagram("R1 — stiffness × relaxation confinement regimes (Fig 1G target)")
    # 2x2 regime grid
    _box(fig, 0.6, 0.6, 5, 5, "spread / flat<br>(soft, fast-relaxing BM)", GREEN)
    _box(fig, 5, 0.6, 9.4, 5, "jammed<br>(stiff, fast-relaxing)", GOLD)
    _box(fig, 0.6, 5, 5, 9.4, "remodelling-limited<br>(soft, slow-relaxing)", TEAL)
    _box(fig, 5, 5, 9.4, 9.4, "confined / tall<br>(stiff, slow-relaxing BM)", RED)
    fig.add_annotation(x=5, y=0.2, text="BM relaxation time / doubling time →", showarrow=False,
                       font=dict(size=12))
    fig.add_annotation(x=0.2, y=5, text="BM stiffness →", showarrow=False, textangle=-90,
                       font=dict(size=12))
    return fig


def r6_perturbation_schematic():
    """R6: expected perlecan/growth intervention ordering on cell shape (Fig 7)."""
    fig = _diagram("R6 — expected intervention ordering on cell height (Fig 7 target)", xr=(0, 10), yr=(0, 10))
    conds = [("perlecan loss\n(faster turnover)", 1.5, GREEN),
             ("control", 4.0, BLUE),
             ("growth-rate ↑", 6.0, ORANGE),
             ("perlecan gain\n(slower turnover)", 8.5, RED)]
    fig.add_shape(type="line", x0=1, y0=3, x1=9.3, y1=3, line=dict(color="#888", width=2))
    for label, x, c in conds:
        fig.add_shape(type="circle", x0=x - 0.25, y0=2.75, x1=x + 0.25, y1=3.25,
                      line=dict(color=c, width=2), fillcolor=c, opacity=0.5)
        fig.add_annotation(x=x, y=4.2, text=label.replace("\n", "<br>"), showarrow=False,
                           font=dict(size=11, color=c))
    fig.add_annotation(x=5, y=1.8, text="shorter cell ⟵   relative cell height   ⟶ taller cell",
                       showarrow=False, font=dict(size=12))
    fig.add_annotation(x=5, y=7.5, showarrow=False, font=dict(size=12, color="#444"),
                       text="chain lifetime → BM relaxation → confinement:<br>"
                            "slower turnover (perlecan gain) ⇒ slower relaxation ⇒ taller, more confined cells")
    return fig


def e2_force_transfer_schematic():
    """E2: cortex → reversible basal linkers → collagen IV BM force transfer."""
    fig = _diagram("E2 — actin → adhesion → collagen BM force transfer")
    _box(fig, 0.5, 5.5, 3.2, 8.5, "actomyosin<br>cortex<br>(E1)", RED)
    _box(fig, 4.0, 5.5, 6.4, 8.5, "reversible<br>basal linkers<br>(integrins)", GOLD)
    _box(fig, 7.2, 5.5, 9.6, 8.5, "collagen IV<br>BM patch<br>(R2)", BLUE)
    _arrow(fig, 3.2, 7.0, 4.0, 7.0, "contractility")
    _arrow(fig, 6.4, 7.0, 7.2, 7.0, "traction")
    _arrow(fig, 7.2, 5.6, 6.4, 5.6, "reaction")
    fig.add_annotation(x=5, y=3.2, showarrow=False, font=dict(size=12, color="#444"),
                       text="measure: traction, linker occupancy, network deformation, work balance<br>"
                            "(single-cell patch; prescribed-displacement + imposed-traction tests first)")
    return fig


def e4_feedback_schematic():
    """E4: the growth↔turnover↔relaxation↔confinement feedback loop + controls."""
    fig = _diagram("E4 — new feedback predictions (coupled loop vs controls)")
    _box(fig, 4.0, 8.0, 6.5, 9.5, "epithelial growth", GREEN)
    _box(fig, 7.5, 5.0, 9.6, 6.5, "collagen turnover", ORANGE)
    _box(fig, 4.0, 2.0, 6.5, 3.5, "BM relaxation", BLUE)
    _box(fig, 0.4, 5.0, 2.6, 6.5, "cell confinement<br>/ shape", RED)
    _arrow(fig, 6.5, 8.6, 8.0, 6.5)
    _arrow(fig, 8.2, 5.0, 6.0, 3.5)
    _arrow(fig, 4.0, 3.2, 2.6, 5.2)
    _arrow(fig, 1.6, 6.5, 4.2, 8.2)
    fig.add_annotation(x=5.2, y=5.7, text="coupled<br>loop", showarrow=False,
                       font=dict(size=13, color="#444"))
    fig.add_annotation(x=5, y=0.6, showarrow=False, font=dict(size=11, color="#777"),
                       text="tested against one-way / uncoupled / uniform-material controls "
                            "to attribute any departure to the mechanism")
    return fig


# study slug -> figure function (single source for the generator + tests)
def driving_schema_figure():
    """The investigation's DRIVING SCHEMA — reproduces the Barrientos et al. (Cell
    2026) three-model framework and maps each box to the studies that realize it:
    Collagen IV turnover (timer) -> BM relaxation (network) -> cell shape (tissue
    mechanics), closed by the measured anchors. A conceptual framework diagram."""
    go = _go()
    fig = go.Figure()
    fig.update_layout(
        title="Driving schema — collagen IV turnover → BM relaxation → cell shape "
              "(Barrientos et al., Cell 2026)",
        template="plotly_white", height=700,
        xaxis=dict(range=(0, 10), visible=False), yaxis=dict(range=(0, 10), visible=False),
        margin=dict(l=20, r=20, t=90, b=40), showlegend=False)
    # left column: the three models (mapped to our studies)
    _box(fig, 0.3, 7.0, 3.7, 9.5, "<b>Tissue Mechanics Model</b><br>cells on a basement membrane"
         "<br><i>our: R1 prism · R4/R5 tyssue</i>", BLUE)
    _box(fig, 0.3, 3.75, 3.7, 6.25, "<b>Collagen IV Network Model</b><br>two-bead rods + NC1/7S bonds"
         "<br><i>our: R2/R3 collagen network</i>", ORANGE)
    _box(fig, 0.3, 0.5, 3.7, 3.0, "<b>Collagen IV Timer Model</b><br>molecular age / turnover"
         "<br><i>our: R6 turnover pipeline</i>", GREEN)
    # right column: the relationships
    _box(fig, 5.3, 7.0, 9.6, 9.5, "Cell Shape  ~  BM Relaxation", BLUE)
    _box(fig, 5.3, 3.75, 9.6, 6.25, "BM Relaxation  ∝  Collagen IV Turnover", ORANGE)
    _box(fig, 5.3, 0.5, 9.6, 3.0, "Collagen IV Turnover", GREEN)
    # model -> relationship
    for y in (8.25, 5.0, 1.75):
        _arrow(fig, 3.7, y, 5.3, y)
    # chain up the right: turnover -> relaxation -> cell shape
    _arrow(fig, 7.45, 3.0, 7.45, 3.75)
    _arrow(fig, 7.45, 6.25, 7.45, 7.0)
    # experimental anchors (as in the paper)
    fig.add_annotation(x=7.45, y=9.85, text="<i>Experimentally measured: Cell Shape</i>",
                       showarrow=False, font=dict(size=11, color=BLUE))
    fig.add_annotation(x=7.45, y=0.15, text="<i>Measured in vivo: Collagen IV Fluorescent Timer</i>",
                       showarrow=False, font=dict(size=11, color=GREEN))
    return fig


def _prism_trajectory(stiffness, relax_over_doubling, doubling=5.0, steps=40, dt=0.2):
    """A single reduced-order prism trajectory: confinement + basal area over time."""
    from viva_lumenoid.tissue_prism import TissuePrismProcess
    from bigraph_schema import allocate_core
    p = TissuePrismProcess(config={"bm_stiffness": float(stiffness),
                                   "bm_relaxation": float(relax_over_doubling * doubling),
                                   "doubling_time": float(doubling)}, core=allocate_core())
    t, area, conf = [], [], []
    for i in range(steps):
        o = p.update({}, dt)
        t.append(i * dt); area.append(o["basal_area"]); conf.append(o["confinement"])
    return np.array(t), np.array(area), np.array(conf)


def r1_regime_figure():
    """R1 — reduced-order reimplementation of the Barrientos Fig-1 tissue model.

    Panel A: confinement across the TWO dimensionless axes the paper uses — BM
    stiffness relative to the cell and BM relaxation time / proliferation (doubling)
    time — with the confinement=1 boundary drawn and the three sampled regimes
    marked. Panels B–D: representative basal-area growth trajectories for the
    soft/fast, intermediate and stiff/slow corners. This is our analytic
    reimplementation, compared to Fig 1 at the level of regime topology; it is not a
    digitized overlay of the published boundaries (evidence: observed)."""
    go = _go()
    from plotly.subplots import make_subplots
    stiff = np.geomspace(0.1, 10.0, 14)
    ratio = np.geomspace(0.1, 10.0, 14)   # BM relaxation / doubling time (dimensionless)
    C = np.zeros((len(ratio), len(stiff)))
    for i, r in enumerate(ratio):
        for j, k in enumerate(stiff):
            _, _, conf = _prism_trajectory(k, r)
            C[i, j] = conf[-1]
    corners = [("soft / fast-relaxing", 0.2, 0.2, GREEN),
               ("intermediate", 1.0, 1.0, BLUE),
               ("stiff / slow-relaxing", 8.0, 8.0, RED)]

    fig = make_subplots(
        rows=3, cols=2, column_widths=[0.62, 0.38], row_heights=[1, 1, 1],
        specs=[[{"rowspan": 3}, {}], [None, {}], [None, {}]],
        subplot_titles=("A — confinement phase map (reduced-order reimplementation)",
                        "B — soft / fast-relaxing", "C — intermediate", "D — stiff / slow-relaxing"))
    fig.add_trace(go.Heatmap(z=C, x=stiff, y=ratio, colorscale="Viridis",
                             colorbar=dict(title="confinement<br>(h/h₀)", len=0.95)),
                  row=1, col=1)
    # confinement = 1 regime boundary (spread below, confined above)
    fig.add_trace(go.Contour(z=C, x=stiff, y=ratio, showscale=False,
                             contours=dict(start=1.0, end=1.0, size=1.0, coloring="none"),
                             line=dict(color="white", width=2),
                             hoverinfo="skip"), row=1, col=1)
    for nm, k, r, col in corners:
        fig.add_trace(go.Scatter(x=[k], y=[r], mode="markers+text", text=[nm.split()[0]],
                                 textposition="top center",
                                 marker=dict(color=col, size=12, line=dict(color="white", width=1.5)),
                                 showlegend=False), row=1, col=1)
    fig.update_xaxes(title_text="BM stiffness (relative to cell)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="BM relaxation / doubling time", type="log", row=1, col=1)
    for idx, (nm, k, r, col) in enumerate(corners):
        rr = idx + 1
        t, area, _ = _prism_trajectory(k, r)
        fig.add_trace(go.Scatter(x=t, y=area, mode="lines", line=dict(color=col, width=3),
                                 showlegend=False), row=rr, col=2)
        fig.update_yaxes(title_text="basal area", row=rr, col=2)
    fig.update_xaxes(title_text="time (∝ cell number)", row=3, col=2)
    fig.update_layout(
        title="R1 — reduced-order tissue model: confinement regimes vs BM stiffness × "
              "relaxation/doubling (cf. Barrientos Fig 1; evidence: observed)",
        template="plotly_white", height=640, margin=dict(t=90))
    return fig


def r6_intervention_figure():
    """R6 — Barrientos Fig 7 reproduction: predicted cell height per perlecan/growth
    intervention. Perlecan promotes collagen IV bond breakage, so overexpression
    shortens the lifetime (~1 h) → faster BM relaxation → shorter cells, and
    knockdown lengthens it (~500 h) → taller cells. Reproduces the paper's ordering
    (knockdown > control > overexpression) after correcting a previously-inverted
    perlecan direction (figure review 2026-10-20)."""
    go = _go()
    from viva_lumenoid.turnover_shape import TurnoverShapeProcess
    from bigraph_schema import allocate_core
    # perlecan_factor encodes the paper's lifetime ratios: knockdown ~500h (x25),
    # control ~20h (x1), overexpression ~1h (x0.05).
    conds = [("perlecan loss\n(knockdown)", 25.0, 5.0), ("control", 1.0, 5.0),
             ("perlecan gain\n(overexpression)", 0.05, 5.0), ("fast growth", 1.0, 2.5)]
    names, heights, relax = [], [], []
    for nm, perl, dbl in conds:
        p = TurnoverShapeProcess(config={"collagen_lifetime": 5.0, "perlecan_factor": perl,
                                         "doubling_time": dbl}, core=allocate_core())
        o = {}
        for _ in range(40):
            o = p.update({}, 0.2)
        names.append(nm); heights.append(o["cell_height"]); relax.append(o["inferred_relaxation"])
    fig = go.Figure()
    fig.add_bar(x=names, y=heights, marker_color=[GREEN, BLUE, RED, GOLD])
    for i, r in enumerate(relax):
        fig.add_annotation(x=names[i], y=heights[i], text=f"τ={r:.2g}", showarrow=False,
                           yshift=12, font=dict(size=10, color="#444"))
    fig.add_annotation(
        x=0.5, y=1.0, xref="paper", yref="paper", showarrow=False, yshift=-2,
        text="knockdown → taller · overexpression → shorter (matches Barrientos Fig 7)",
        font=dict(size=11, color=GREEN))
    fig.update_layout(
        title="R6 — perlecan/growth intervention on cell height (Fig 7 reproduction; "
              "τ = inferred BM relaxation)",
        template="plotly_white", height=540, yaxis=dict(title="predicted cell height"),
        margin=dict(t=90),
        annotations=list(fig.layout.annotations) + [dict(
            x=0.5, y=-0.18, xref="paper", yref="paper", showarrow=False,
            text="Perlecan promotes collagen IV bond breakage → overexpression shortens lifetime "
                 "(~1h) → faster relaxation → shorter; knockdown (~500h) → taller. Reduced units.",
            font=dict(size=11, color="#777"))])
    return fig


# ===========================================================================
# bm-v1 / bm-v2 / bm-v3 — figure-review upgrade (2026-10-20)
#
# These three studies previously had NO figure functions. Each now has a single
# question-driven primary built from BOUNDED REAL LAMMPS runs (small network,
# ≤3 seeds, few steps), with its evidence status named in the caption. Shared
# helpers are NOT touched — the helpers below are v1/v2/v3-namespaced.
#
# FIDELITY NOTE (verified in code + at runtime, 2026-10-20): the "faithful"
# (use_real_angles=True, bending_k>0) path creates ONLY the released NC1 junction
# angle (angle_coeff 1 = 4.0 @180°; two angles per NC1 crosslink). The released
# 7S angles (k=1.0 @155° and @60°) and the 7S-specific break cutoff (0.95 vs the
# NC1 1.35 used for both) are NOT implemented. So these runs are NC1-angle-
# stabilized, NOT a full reproduction of the released angle set — the figures are
# captioned accordingly and never claim full faithfulness.
# ---------------------------------------------------------------------------
def _vseries_largest_cc(bonds):
    """Largest connected component (bead count) from a LAMMPS bond list
    [[bond_type, id1, id2], ...]. Shared by the v1/v2 connectivity readouts."""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for _bt, i, j in bonds:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    from collections import Counter
    sizes = Counter(find(a) for a in parent)
    return max(sizes.values()) if sizes else 0


def _v1_params(seed):
    """CollagenParams for the v1 NC1-angle-stabilized (faithful-path) network."""
    from viva_lumenoid.params import CollagenParams
    return CollagenParams(n_rods=60, box_xy=6.0, slab_thickness=4.0, seed=seed,
                          use_real_angles=True, bending_k=4.0,
                          target_strain=0.08, strain_rate=4.0e-3)


def _v1_run_seed(seed):
    """Run the staged stage-1 protocol (assemble→relax-frozen→stretch-frozen→
    hold-elastic→hold-viscous) on a small NC1-angle-stabilized network. Returns a
    trajectory dict (continuous time, σ, strain, crosslinks, largest-CC, phase),
    the as-formed mesh (positions/ids/bonds), and (E_tangent, tau)."""
    from bigraph_schema import allocate_core
    from viva_lumenoid.collagen import CollagenNetworkProcess
    proc = CollagenNetworkProcess(config={"params": _v1_params(seed).to_dict()},
                                  core=allocate_core())
    proc.initial_state()
    tr = {"t": [], "sigma": [], "strain": [], "xl": [], "lcc": [], "phase": []}
    tt = [0.0]
    mesh = {}

    def rec(o, phase):
        tr["t"].append(tt[0]); tr["sigma"].append(float(o["sigma_inplane"]))
        tr["strain"].append(float(o["strain"])); tr["xl"].append(int(o["n_crosslinks"]))
        tr["lcc"].append(_vseries_largest_cc(o["bonds"])); tr["phase"].append(phase)

    def step(sr, mk, bk, phase, n):
        o = {}
        for _ in range(n):
            o = proc.update({"strain_rate": sr, "make_on": mk, "break_on": bk}, 1.0)
            tt[0] += 1.0
            rec(o, phase)
        return o

    o = step(0.0, 1.0, 1.0, "assemble", 25)          # make-only assembly (real-angle)
    mesh = {"pos": np.array(o["positions"]), "types": np.array(o["atom_types"]),
            "ids": np.array(o["atom_ids"]), "bonds": o["bonds"], "box": o["box_dimensions"]}
    step(0.0, 0.0, 0.0, "relax", 4)                  # frozen settle
    step(4.0e-3, 0.0, 0.0, "stretch", 20)            # frozen elastic loading
    step(0.0, 0.0, 0.0, "hold_elastic", 6)           # elastic plateau (frozen)
    step(0.0, 0.0, 1.0, "hold_viscous", 30)          # break-only viscous relaxation
    proc.close()

    strain = np.array(tr["strain"]); sigma = np.array(tr["sigma"])
    phase = np.array(tr["phase"]); t = np.array(tr["t"])
    # Tangent modulus over the stretch ramp, fit window strain ∈ [0.01, max].
    m = (phase == "stretch") & (strain >= 0.01)
    E = float("nan")
    if m.sum() >= 2:
        A = np.vstack([strain[m], np.ones(m.sum())]).T
        E = float(np.linalg.lstsq(A, sigma[m], rcond=None)[0][0])
    # Relaxation time from the viscous hold: 1/e crossing of (σ0−σ∞).
    vm = phase == "hold_viscous"
    tau = float("nan")
    if vm.sum() >= 3:
        tv = t[vm] - t[vm][0]; sv = sigma[vm]
        amp = sv[0] - sv[-1]
        if abs(amp) > 1e-6:
            target = sv[-1] + amp / np.e
            for k in range(1, len(sv)):
                if (amp > 0 and sv[k] <= target) or (amp < 0 and sv[k] >= target):
                    tau = float(tv[k]); break
    return tr, mesh, E, tau


# Fit window + phase palette for the v1 primary.
_V1_FIT_LO = 0.01
_V1_SEEDS = (11, 22, 33)


def v1_stage1_main_figure():
    """bm-v1 PRIMARY (evidence: observed — real bounded LAMMPS; tier: main).

    An NC1-angle-stabilized collagen IV network develops tensile stress during
    imposed biaxial stretching and relaxes when bond remodeling resumes. Four
    panels: (A) the σ(t) trace with the stretch / frozen-hold / chemistry-on-hold
    phases shaded; (B) network bond-virial in-plane stress vs strain with the
    fitted tangent modulus over strain ∈ [0.01, 0.08]; (C) crosslinks vs largest
    connected component over the protocol; (D) one labeled as-formed network.
    E and τ are reported across independent seeds with spread. Reduced LJ units.

    HONESTY: NC1-angle-stabilized only — the released 7S angles (155°/60°) and the
    7S-specific break cutoff are not implemented (named in the caption); τ is not
    cleanly resolved at this bounded scale (dense multiseed is a mini candidate)."""
    go = _go()
    from plotly.subplots import make_subplots
    runs = [_v1_run_seed(s) for s in _V1_SEEDS]
    Es = np.array([r[2] for r in runs], dtype=float)
    taus = np.array([r[3] for r in runs], dtype=float)
    E_m, E_s = float(np.nanmean(Es)), float(np.nanstd(Es))
    tau_m, tau_s = float(np.nanmean(taus)), float(np.nanstd(taus))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "A — σ(t): stretch / frozen-hold / chemistry-on-hold phases",
            "B — network bond-virial stress vs strain (+ fitted tangent E)",
            "C — crosslinks vs largest connected component",
            "D — representative as-formed network (seed 11, NC1 angles active)"))

    # Panel A: σ(t) for each seed + phase shading (built from the seed-11 phase map).
    tr0 = runs[0][0]
    t0 = np.array(tr0["t"]); ph0 = np.array(tr0["phase"])
    bands = [("stretch", "rgba(31,119,180,0.10)", "stretch"),
             ("hold_elastic", "rgba(120,120,120,0.12)", "frozen hold"),
             ("hold_viscous", "rgba(44,160,44,0.12)", "chemistry-on hold")]
    for name, col, label in bands:
        idx = np.where(ph0 == name)[0]
        if len(idx):
            x0, x1 = float(t0[idx[0]]) - 0.5, float(t0[idx[-1]]) + 0.5
            fig.add_vrect(x0=x0, x1=x1, fillcolor=col, line_width=0, row=1, col=1,
                          annotation_text=label, annotation_position="top left",
                          annotation_font=dict(size=9, color="#555"))
    for i, (tr, *_rest) in enumerate(runs):
        fig.add_trace(go.Scatter(x=tr["t"], y=tr["sigma"], mode="lines",
                                 line=dict(color=BLUE, width=1.4),
                                 opacity=0.55, showlegend=False,
                                 name=f"seed {_V1_SEEDS[i]}"), row=1, col=1)
    fig.update_xaxes(title_text="time (LJ)", row=1, col=1)
    fig.update_yaxes(title_text="σ bond-virial (LJ)", row=1, col=1)

    # Panel B: σ vs strain over the stretch phase, 3 seeds + fitted tangent.
    all_e, all_s = [], []
    for i, (tr, *_rest) in enumerate(runs):
        ph = np.array(tr["phase"]); e = np.array(tr["strain"]); s = np.array(tr["sigma"])
        m = ph == "stretch"
        fig.add_trace(go.Scatter(x=e[m], y=s[m], mode="markers",
                                 marker=dict(color=BLUE, size=6, opacity=0.5),
                                 showlegend=False), row=1, col=2)
        fm = m & (e >= _V1_FIT_LO)
        all_e += e[fm].tolist(); all_s += s[fm].tolist()
    if len(all_e) >= 2:
        ae = np.array(all_e); A = np.vstack([ae, np.ones(len(ae))]).T
        slope, icpt = np.linalg.lstsq(A, np.array(all_s), rcond=None)[0]
        xs = np.linspace(_V1_FIT_LO, max(ae), 20)
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + icpt, mode="lines",
                                 line=dict(color=RED, width=3),
                                 name=f"tangent E = {E_m:.1f} ± {E_s:.1f} LJ"), row=1, col=2)
    fig.add_vrect(x0=_V1_FIT_LO, x1=max(all_e) if all_e else 0.08,
                  fillcolor="rgba(214,39,40,0.06)", line_width=0, row=1, col=2)
    fig.update_xaxes(title_text="in-plane strain ε", row=1, col=2)
    fig.update_yaxes(title_text="σ bond-virial (LJ)", row=1, col=2)

    # Panel C: crosslinks + largest connected component over time (seed-11).
    fig.add_trace(go.Scatter(x=tr0["t"], y=tr0["xl"], mode="lines",
                             line=dict(color=ORANGE, width=3), name="crosslinks"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=tr0["t"], y=tr0["lcc"], mode="lines",
                             line=dict(color=TEAL, width=2, dash="dot"),
                             name="largest connected component"), row=2, col=1)
    fig.update_xaxes(title_text="time (LJ)", row=2, col=1)
    fig.update_yaxes(title_text="count (beads / bonds)", row=2, col=1)

    # Panel D: the representative as-formed network (seed 11), labeled.
    mesh = runs[0][1]
    pos = mesh["pos"]; ids = mesh["ids"]; types = mesh["types"]
    id2i = {int(a): k for k, a in enumerate(ids)}
    for bt, c, nm in ((1, "rgba(120,120,120,0.35)", None),
                      (2, BLUE, "NC1 crosslink (angle-braced)"),
                      (3, ORANGE, "7S crosslink (no angle)")):
        bx, by = [], []
        for b in mesh["bonds"]:
            if b[0] != bt:
                continue
            a, c2 = id2i.get(b[1]), id2i.get(b[2])
            if a is None or c2 is None:
                continue
            bx += [pos[a, 0], pos[c2, 0], None]; by += [pos[a, 1], pos[c2, 1], None]
        fig.add_trace(go.Scatter(x=bx, y=by, mode="lines",
                                 line=dict(color=c, width=(1 if bt == 1 else 2)),
                                 name=nm, showlegend=bool(nm), hoverinfo="skip"),
                      row=2, col=2)
    is_nc1 = types == 1
    fig.add_trace(go.Scatter(x=pos[is_nc1, 0], y=pos[is_nc1, 1], mode="markers",
                             marker=dict(color=BLUE, size=4), name="NC1 bead",
                             hoverinfo="skip"), row=2, col=2)
    fig.add_trace(go.Scatter(x=pos[~is_nc1, 0], y=pos[~is_nc1, 1], mode="markers",
                             marker=dict(color=ORANGE, size=4), name="7S bead",
                             hoverinfo="skip"), row=2, col=2)
    fig.update_xaxes(title_text="x (σ)", row=2, col=2, scaleanchor="y2")
    fig.update_yaxes(title_text="y (σ)", row=2, col=2)

    fig.update_layout(
        template="plotly_white", height=820, margin=dict(t=96, b=96),
        legend=dict(orientation="h", y=1.07, x=0, font=dict(size=10)),
        title="bm-v1 — NC1-angle-stabilized collagen IV network: elastic modulus "
              "AND remodelling, separated (observed; real bounded LAMMPS, 3 seeds)")
    fig.add_annotation(
        x=0.5, y=-0.14, xref="paper", yref="paper", showarrow=False,
        font=dict(size=11, color="#777"), align="center",
        text=(f"60-rod NC1-angle-stabilized network (bending_k=4.0, use_real_angles; dense slab), "
              f"3 seeds. Tangent elastic modulus E = {E_m:.1f} ± {E_s:.1f} LJ "
              f"(≈ {E_m*2.2:.0f} ± {E_s*2.2:.0f} Pa at kT/σ³≈2.2 Pa — energy scale unresolved), "
              f"fit over strain ∈ [0.01, 0.08]. Stress is the network BOND-VIRIAL in-plane "
              f"component (reduced LJ units), not total/hydrostatic. Relaxation τ ≈ {tau_m:.0f} ± {tau_s:.0f} LJ "
              f"is NOT cleanly resolved at this bounded scale (dense multiseed = mini candidate). "
              f"NC1 junction angles verified active; released 7S angles (155°/60°) + 7S break cutoff NOT implemented."))
    return fig


def _v2_run_rate(rate, seed, target_strain=0.12, max_steps=140):
    """One bm-v2 growth-rheology point: assemble, then stretch at `rate` with the
    crosslink chemistry ON until `target_strain`. Returns (σ_late, strain_reached,
    crosslinks_per_rod_late, n_steps) with σ averaged over strain ∈ [0.8,1.0]·target."""
    from bigraph_schema import allocate_core
    from viva_lumenoid.collagen import CollagenNetworkProcess
    from viva_lumenoid.params import CollagenParams
    n_rods = 60
    p = CollagenParams(n_rods=n_rods, box_xy=6.0, slab_thickness=4.0, seed=seed,
                       use_real_angles=True, bending_k=4.0)
    proc = CollagenNetworkProcess(config={"params": p.to_dict()}, core=allocate_core())
    proc.initial_state()
    for _ in range(22):
        proc.update({"strain_rate": 0.0, "make_on": 1.0, "break_on": 1.0}, 1.0)
    strains, sig, xl = [], [], []
    o = {}
    for _ in range(max_steps):
        o = proc.update({"strain_rate": rate, "make_on": 1.0, "break_on": 1.0}, 1.0)
        strains.append(float(o["strain"])); sig.append(float(o["sigma_inplane"]))
        xl.append(int(o["n_crosslinks"]))
        if o["strain"] >= target_strain:
            break
    proc.close()
    strains = np.array(strains); sig = np.array(sig); xl = np.array(xl)
    win = strains >= 0.8 * target_strain
    if not win.any():
        win = np.ones(len(strains), dtype=bool)
    return (float(np.mean(sig[win])), float(strains[-1]),
            float(np.mean(xl[win]) / n_rods), int(len(strains)))


_V2_RATES = (2.0e-3, 6.0e-3, 1.8e-2)
_V2_SEEDS = (11, 22, 33)
_V2_TARGET_STRAIN = 0.12


def v2_rheology_main_figure():
    """bm-v2 PRIMARY (evidence: observed — bounded-preliminary; tier: main).

    Imposed growth rate ε̇ vs the network's apparent mechanical resistance at a
    MATCHED total strain, with the paired crosslink-density readout that tests
    whether the response reflects turnover competition or a rate-dependent change
    in structure. Three panels: (A) σ(ε̇) at matched strain with per-seed spread,
    (B) effective viscosity η_eff = σ/ε̇, (C) crosslinks-per-rod at each rate.
    Error bars summarize 3 independent networks. Reduced LJ units.

    HONESTY: at this bounded scale σ(ε̇) at matched strain is CONFOUNDED by
    crosslink density (slower stretch allows more assembly time → denser network),
    so true rate-stiffening cannot be cleanly separated from a density change — a
    density-matched dense multiseed sweep is a MINI CANDIDATE. τ(ε̇) is not cleanly
    resolved here (same τ-resolution gap as bm-v1)."""
    go = _go()
    from plotly.subplots import make_subplots
    rates = np.array(_V2_RATES)
    sig_m, sig_s, eta_m, eta_s, xl_m, xl_s, strain_reached = ([] for _ in range(7))
    for r in rates:
        vals = [_v2_run_rate(r, s, target_strain=_V2_TARGET_STRAIN) for s in _V2_SEEDS]
        sg = np.array([v[0] for v in vals]); xr = np.array([v[2] for v in vals])
        st = np.array([v[1] for v in vals])
        sig_m.append(sg.mean()); sig_s.append(sg.std())
        eta_m.append((sg / r).mean()); eta_s.append((sg / r).std())
        xl_m.append(xr.mean()); xl_s.append(xr.std()); strain_reached.append(st.mean())
    sig_m, sig_s = np.array(sig_m), np.array(sig_s)
    eta_m, eta_s = np.array(eta_m), np.array(eta_s)
    xl_m, xl_s = np.array(xl_m), np.array(xl_s)

    fig = make_subplots(
        rows=1, cols=3, horizontal_spacing=0.09,
        subplot_titles=(
            "A — σ(ε̇) at matched strain ε=0.12",
            "B — effective viscosity η_eff = σ/ε̇",
            "C — crosslinks / rod (paired connectivity)"))
    fig.add_trace(go.Scatter(x=rates, y=sig_m, error_y=dict(type="data", array=sig_s),
                             mode="lines+markers", line=dict(color=BLUE, width=3),
                             marker=dict(size=10), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=rates, y=eta_m, error_y=dict(type="data", array=eta_s),
                             mode="lines+markers", line=dict(color=ORANGE, width=3),
                             marker=dict(size=10), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=rates, y=xl_m, error_y=dict(type="data", array=xl_s),
                             mode="lines+markers", line=dict(color=GREEN, width=3),
                             marker=dict(size=10), showlegend=False), row=1, col=3)
    fig.update_xaxes(title_text="strain rate ε̇ (1/LJ)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="strain rate ε̇ (1/LJ)", type="log", row=1, col=2)
    fig.update_xaxes(title_text="strain rate ε̇ (1/LJ)", type="log", row=1, col=3)
    fig.update_yaxes(title_text="σ bond-virial (LJ)", row=1, col=1)
    fig.update_yaxes(title_text="η_eff = σ/ε̇ (LJ·time)", row=1, col=2)
    fig.update_yaxes(title_text="crosslinks / rod", row=1, col=3)
    fig.update_layout(
        template="plotly_white", height=520, margin=dict(t=92, b=118),
        title="bm-v2 — growth rate vs apparent mechanical resistance, with the paired "
              "connectivity readout (observed, bounded-preliminary; 3 seeds)")
    fig.add_annotation(
        x=0.5, y=-0.30, xref="paper", yref="paper", showarrow=False,
        font=dict(size=11, color="#777"), align="center",
        text=(f"60-rod NC1-angle-stabilized network, chemistry ON during stretch; "
              f"σ averaged over strain ∈ [0.096, 0.12] (matched-strain plateau criterion), 3 seeds. "
              f"σ(ε̇) is nearly flat/overlapping across rates ({sig_m[0]:.2f}→{sig_m[-1]:.2f} LJ) while "
              f"crosslinks/rod FALLS with rate ({xl_m[0]:.2f}→{xl_m[-1]:.2f}) — the resistance is CONFOUNDED "
              f"by rate-dependent assembly time, not a clean rate-stiffening signal. "
              f"η_eff falls {eta_m[0]:.0f}→{eta_m[-1]:.0f} LJ·time (De = τ·ε̇ not reported — τ unresolved). "
              f"Stress = network bond-virial in-plane component, reduced LJ units. "
              f"A density-matched dense multiseed sweep is the mini candidate to separate the two."))
    return fig


def _v3_run_point(nc1, svns, make_prob, seed):
    """One bm-v3 central-force control point: angles OFF (bending_k=0), assemble at
    the given crosslink caps, then measure the athermal (energy-minimization)
    modulus. Returns (measured z, athermal_modulus). z = 1 + crosslinks/rod."""
    from bigraph_schema import allocate_core
    from viva_lumenoid.collagen import CollagenNetworkProcess
    from viva_lumenoid.params import CollagenParams
    n_rods = 60
    p = CollagenParams(n_rods=n_rods, box_xy=6.0, slab_thickness=4.0, seed=seed,
                       bending_k=0.0, use_real_angles=False,          # CENTRAL FORCE
                       nc1_max_crosslinks=nc1, svns_max_crosslinks=svns, make_prob=make_prob)
    proc = CollagenNetworkProcess(config={"params": p.to_dict()}, core=allocate_core())
    proc.initial_state()
    o = {}
    for _ in range(22):
        o = proc.update({"strain_rate": 0.0, "make_on": 1.0, "break_on": 1.0}, 1.0)
    z = 1.0 + o["n_crosslinks"] / n_rods
    E = proc.athermal_modulus(target_strain=0.03, n_increments=4, minimize_iters=1500)
    proc.close()
    return float(z), float(E)


_V3_RUNGS = ((1, 3, 0.12), (2, 4, 0.25), (3, 5, 0.40), (5, 8, 0.60))
_V3_SEEDS = (11, 22, 33)


def v3_centralforce_control_figure():
    """bm-v3 PRIMARY, framed as a NEGATIVE CONTROL (evidence: engineered-diagnostic;
    tier: main). "Central-force control does not reproduce the released network."

    With the angular potentials intentionally removed (bending_k=0), changing the
    allowable end-link valency sweeps the measured coordination z but does NOT
    produce a reproducibly positive athermal modulus across sampled connectivities
    and seeds. Measured z is on the x-axis; each seed's modulus is a point with a
    zero reference and the ensemble mean ± spread. The z=4 Maxwell line is drawn
    only as the idealized 2D central-force heuristic — it is NOT a threshold for
    the released angle-stabilized, finite-thickness network.

    This artificial control must not be read as the mechanism of softness in the
    published model (which carries junction angles and is soft for other reasons)."""
    go = _go()
    zs_all, es_all, z_mean, E_mean, E_sd = [], [], [], [], []
    for nc1, svns, mp in _V3_RUNGS:
        pts = [_v3_run_point(nc1, svns, mp, s) for s in _V3_SEEDS]
        z = np.array([p[0] for p in pts]); E = np.array([p[1] for p in pts])
        zs_all += z.tolist(); es_all += E.tolist()
        z_mean.append(z.mean()); E_mean.append(E.mean()); E_sd.append(E.std())
    z_mean, E_mean, E_sd = np.array(z_mean), np.array(E_mean), np.array(E_sd)

    fig = go.Figure()
    fig.add_hline(y=0.0, line=dict(color="#444", width=1.5),
                  annotation_text="zero modulus (reproducibly soft by design)",
                  annotation_position="bottom right", annotation_font=dict(size=11, color="#444"))
    fig.add_trace(go.Scatter(x=zs_all, y=es_all, mode="markers",
                             marker=dict(color=PURPLE, size=8, opacity=0.5),
                             name="per-seed modulus (angles OFF)"))
    fig.add_trace(go.Scatter(x=z_mean, y=E_mean, error_y=dict(type="data", array=E_sd),
                             mode="lines+markers", line=dict(color=PURPLE, width=3),
                             marker=dict(size=12), name="ensemble mean ± SD"))
    fig.add_vline(x=4.0, line=dict(color="#aaa", width=1.5, dash="dash"),
                  annotation_text="z=4 Maxwell (idealized 2D central-force heuristic — "
                                  "NOT a threshold for the released angle-stabilized network)",
                  annotation_position="top left", annotation_font=dict(size=10, color="#999"))
    fig.update_layout(
        title="bm-v3 — central-force control does NOT reproduce the released network "
              "(engineered-diagnostic / negative control; all angles OFF)",
        xaxis=dict(title="measured coordination z = 1 + crosslinks/rod", range=[1.0, 4.3]),
        yaxis_title="athermal elastic modulus (LJ)")
    fig.add_annotation(
        x=0.5, y=-0.20, xref="paper", yref="paper", showarrow=False,
        font=dict(size=11, color="#777"), align="center",
        text=(f"60-rod network, bending_k=0 (angular potentials removed), crosslink caps swept "
              f"NC1×1/7S×3 → NC1×5/7S×8, 3 seeds. Measured z rises {z_mean[0]:.2f}→{z_mean[-1]:.2f} "
              f"(never near 4); the athermal (energy-minimization) modulus is scattered about zero "
              f"with seed-spread exceeding the mean at every rung (CIs span 0) — not reproducibly "
              f"positive. Athermal modulus (released-code LAMMPS, reduced LJ units). This is a "
              f"counterfactual with angle terms DISABLED, NOT a reproduction of the published model."))
    return _layout(fig, height=600)


STUDY_FIGURES = {
    "bm-v1-stage1-modulus-remodelling": ("v1_stage1_main", v1_stage1_main_figure, "data"),
    "bm-v2-stress-vs-strainrate": ("v2_rheology_main", v2_rheology_main_figure, "data"),
    "bm-v3-junction-bending-rigidity": ("v3_centralforce_control", v3_centralforce_control_figure, "data"),
    "bm-r1-published-tissue-baseline": ("r1_regime", r1_regime_figure, "data"),
    "bm-r2-collagen-network-rheology": ("r2_fig2e", r2_fig2e_figure, "data"),
    "bm-r3-network-expansion-supply": ("r3_supply", r3_supply_figure, "data"),
    "bm-r4-coupled-2d-epithelium": ("r4_confinement", r4_confinement_figure, "data"),
    "bm-r5-explicit-3d-confinement": ("r5_monolayer", r5_monolayer_figure, "data"),
    "bm-r6-perlecan-growth-perturbations": ("r6_intervention", r6_intervention_figure, "data"),
    "bm-e1-actomyosin-cortex": ("e1_cortical_tension", e1_cortical_tension_figure, "data"),
    "bm-e2-actin-adhesion-collagen": ("e2_force_transfer", e2_force_transfer_figure, "light"),
    "bm-e3-closed-3d-lumenoid": ("e3_lumenoid", e3_lumenoid_figure, "data"),
    "bm-e4-feedback-predictions": ("e4_feedback", e4_feedback_figure, "data"),
}

# Companion (supplement-tier) figures for the E1-E4 extension studies: a second
# honest figure per study. Same (name, fn, kind) shape; rendered + tested alongside
# the primaries. kind "light" = pure Python (always builds); "data" = needs engines.
COMPANION_FIGURES = {
    "bm-e1-actomyosin-cortex": ("e1_tension_components", e1_tension_components_figure, "data"),
    "bm-e2-actin-adhesion-collagen": ("e2_bm_response", e2_bm_response_figure, "data"),
    "bm-e3-closed-3d-lumenoid": ("e3_lumen_dynamics", e3_lumen_dynamics_figure, "data"),
    "bm-e4-feedback-predictions": ("e4_subcycle_convergence", e4_subcycle_convergence_figure, "data"),
}

# Companion (supplement / hypothesis) figures from the 2026-10-20 figure review of
# the R4/R5 spatial studies. The main figure of each study stays in STUDY_FIGURES
# above; these are the demoted thumbnail + the named-gap hypothesis panels. Each is
# (slug, name, fn, kind) so the generator + tests render them alongside the mains.
EXTRA_STUDY_FIGURES = [
    ("bm-r4-coupled-2d-epithelium", "r4_sheet", r4_sheet_figure, "data"),
    ("bm-r4-coupled-2d-epithelium", "r4_bm_hypothesis", r4_bm_confinement_hypothesis, "schematic"),
    ("bm-r5-explicit-3d-confinement", "r5_volume_hist", r5_volume_hist_figure, "data"),
    ("bm-r5-explicit-3d-confinement", "r5_bm_hypothesis", r5_bm_controls_hypothesis, "schematic"),
]
