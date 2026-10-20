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
def e1_cortical_tension_figure(contractilities=(0.5, 1.0, 2.0, 3.0, 4.0), n=6, steps=5):
    """E1: real cortical-tension vs myosin contractility, from the actomyosin
    LAMMPS plugin over the LammpsModelProcess seam."""
    go = _go()
    from bigraph_schema import allocate_core
    from pbg_lammps.model import LammpsModelProcess
    import viva_lumenoid.actomyosin  # noqa: F401
    xs, ys = [], []
    for eps in contractilities:
        p = LammpsModelProcess(config={"model": "actomyosin-cortex",
                                        "params": {"n": n, "contractility": float(eps)}},
                               core=allocate_core())
        p.initial_state()
        out = {}
        for _ in range(steps):
            out = p.update({}, 1.0)
        p.close()
        xs.append(float(eps)); ys.append(float(out["cortical_tension"]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers",
                             line=dict(color=RED, width=3), marker=dict(size=9),
                             name="cortical tension"))
    fig.update_layout(title="E1 — cortical tension rises with myosin contractility (real LAMMPS)",
                      xaxis_title="contractility (LJ well depth ε — myosin-activity proxy)",
                      yaxis_title="cortical tension τ = −(pxx+pyy)/2")
    return _layout(fig, height=560)


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


def r3_supply_figure():
    """R3: a small REAL collagen run comparing remodelling (make+break) vs
    make-only — the ordering that motivates 'supply is needed for sustained
    growth'. Full Fig 2B-C reproduction is the study target."""
    go = _go()
    t, xl_rb, _ = _collagen_micro(make_on=1.0, break_on=1.0)
    _, xl_mo, _ = _collagen_micro(make_on=1.0, break_on=0.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=xl_rb, mode="lines", name="remodelling (make+break)",
                             line=dict(color=GREEN, width=3)))
    fig.add_trace(go.Scatter(x=t, y=xl_mo, mode="lines", name="make-only (no turnover)",
                             line=dict(color=PURPLE, width=3, dash="dash")))
    fig.update_layout(title="R3 — crosslink supply vs remodelling (small real run)",
                      xaxis_title="time (LJ)", yaxis_title="crosslinks",
                      legend=dict(orientation="h", y=1.08),
                      annotations=[dict(x=0.5, y=-0.18, xref="paper", yref="paper", showarrow=False,
                          text="reduced-scale mechanism check; Fig 2B-C ordering is the study target",
                          font=dict(size=11, color="#777"))])
    return _layout(fig, height=560)


def r4_sheet_figure():
    """R4: the actual 2D epithelial sheet after relaxation (CrossSectionGeometry)."""
    eptm = _tyssue_build_step("tyssue_2d_sheet")
    go = _go()
    fig = go.Figure(data=_mesh_traces(eptm, three_d=False))
    fig.update_layout(title=f"R4 — relaxed 2D epithelial sheet ({eptm.Nf} cells, real tyssue run)",
                      xaxis=dict(scaleanchor="y", title="x"), yaxis=dict(title="y"))
    return _layout(fig, height=620)


def r5_monolayer_figure():
    """R5: the actual stepped 3D monolayer + its real cell-volume distribution."""
    eptm = _tyssue_build_step("tyssue_3d_monolayer")
    go = _go()
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                        specs=[[{"type": "scene"}, {"type": "xy"}]],
                        subplot_titles=(f"3D monolayer ({eptm.Nc} cells)", "cell-volume distribution"))
    for tr in _mesh_traces(eptm, three_d=True, color_by="z"):
        fig.add_trace(tr, row=1, col=1)
    vol = eptm.cell_df["vol"].values
    fig.add_trace(go.Histogram(x=vol, nbinsx=24, marker_color=TEAL, showlegend=False), row=1, col=2)
    fig.update_layout(title="R5 — explicit 3D cell confinement (real tyssue run)", height=620)
    fig.update_xaxes(title_text="cell volume", row=1, col=2)
    return fig


def e3_lumenoid_figure():
    """E3: the actual closed lumenoid shell (LumenoidGeometry) + cell volumes."""
    eptm = _tyssue_build_step("closed_lumenoid")
    go = _go()
    fig = go.Figure(data=_mesh_traces(eptm, three_d=True, color_by="z"))
    vtot = float(eptm.cell_df["vol"].sum())
    fig.update_layout(
        title=f"E3 — closed 3D lumenoid shell ({eptm.Nc} cells, real tyssue run)",
        scene=dict(aspectmode="data"),
        annotations=[dict(x=0.5, y=-0.02, xref="paper", yref="paper", showarrow=False,
                          text=f"total cell volume {vtot:.1f} · basal surface = the BM coupling shell",
                          font=dict(size=11, color="#777"))])
    return _layout(fig, height=640)


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


def r1_regime_figure():
    """R1 — Fig 1G reproduction (REAL): final cell height over BM stiffness x
    relaxation/doubling, from the reduced-order prism model. The confinement regime
    topology emerges — tall (confined) in the stiff, slowly-relaxing corner."""
    go = _go()
    from viva_lumenoid.tissue_prism import TissuePrismProcess
    from bigraph_schema import allocate_core
    stiff = np.geomspace(0.1, 10.0, 12)
    relax = np.geomspace(0.5, 50.0, 12)   # doubling fixed at 5 -> y is relaxation/doubling
    H = np.zeros((len(relax), len(stiff)))
    for i, tau in enumerate(relax):
        for j, k in enumerate(stiff):
            p = TissuePrismProcess(config={"bm_stiffness": float(k), "bm_relaxation": float(tau),
                                           "doubling_time": 5.0}, core=allocate_core())
            o = {}
            for _ in range(40):
                o = p.update({}, 0.2)
            H[i, j] = o["cell_height"]
    fig = go.Figure(data=go.Heatmap(z=H, x=stiff, y=relax / 5.0, colorscale="Viridis",
                                    colorbar=dict(title="cell<br>height")))
    fig.update_layout(
        title="R1 — confinement regimes: cell height vs BM stiffness × relaxation/doubling "
              "(Fig 1G reproduction)",
        template="plotly_white", height=560,
        xaxis=dict(title="BM stiffness", type="log"),
        yaxis=dict(title="BM relaxation / doubling time", type="log"))
    return fig


def r6_intervention_figure():
    """R6 — Fig 7 reproduction (REAL): predicted cell height per perlecan/growth
    intervention from the turnover->shape pipeline (inferred relaxation annotated).
    Reproduces the intervention rank-order: perlecan gain -> taller > control > loss."""
    go = _go()
    from viva_lumenoid.turnover_shape import TurnoverShapeProcess
    from bigraph_schema import allocate_core
    conds = [("perlecan loss", 0.3, 5.0), ("control", 1.0, 5.0),
             ("perlecan gain", 2.0, 5.0), ("fast growth", 1.0, 2.5)]
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
        fig.add_annotation(x=names[i], y=heights[i], text=f"τ={r:.1f}", showarrow=False,
                           yshift=12, font=dict(size=10, color="#444"))
    fig.update_layout(
        title="R6 — intervention ordering: predicted cell height by condition "
              "(Fig 7 reproduction; τ = inferred BM relaxation)",
        template="plotly_white", height=520, yaxis=dict(title="predicted cell height"))
    return fig


STUDY_FIGURES = {
    "bm-r1-published-tissue-baseline": ("r1_regime", r1_regime_figure, "data"),
    "bm-r2-collagen-network-rheology": ("r2_rheology", r2_rheology_figure, "data"),
    "bm-r3-network-expansion-supply": ("r3_supply", r3_supply_figure, "data"),
    "bm-r4-coupled-2d-epithelium": ("r4_sheet", r4_sheet_figure, "data"),
    "bm-r5-explicit-3d-confinement": ("r5_monolayer", r5_monolayer_figure, "data"),
    "bm-r6-perlecan-growth-perturbations": ("r6_intervention", r6_intervention_figure, "data"),
    "bm-e1-actomyosin-cortex": ("e1_cortical_tension", e1_cortical_tension_figure, "data"),
    "bm-e2-actin-adhesion-collagen": ("e2_force_transfer", e2_force_transfer_schematic, "schematic"),
    "bm-e3-closed-3d-lumenoid": ("e3_lumenoid", e3_lumenoid_figure, "data"),
    "bm-e4-feedback-predictions": ("e4_feedback", e4_feedback_schematic, "schematic"),
}
