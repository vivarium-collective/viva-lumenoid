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
    """R6 — CONTRADICTION (figure review): the model's perlecan ordering (gain ->
    taller) is OPPOSITE the paper's reported morphology (overexpression -> shorter).
    Shown as a flagged contradiction, not a Fig 7 reproduction, pending an audit of
    the perlecan -> collagen-turnover direction. The growth-rate result is correct."""
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
    # Flag the perlecan contradiction directly on the figure.
    fig.add_annotation(
        x="perlecan gain", y=heights[2], ax=0, ay=-48, showarrow=True, arrowhead=2,
        arrowcolor=RED, text="model: gain → taller<br><b>paper: overexpression → SHORTER</b>",
        font=dict(size=11, color=RED), align="center")
    fig.update_layout(
        title="R6 — perlecan ordering CONTRADICTS the paper (evidence: contradicted; "
              "τ = inferred BM relaxation)",
        template="plotly_white", height=540, yaxis=dict(title="predicted cell height"),
        annotations=list(fig.layout.annotations) + [dict(
            x=0.5, y=-0.16, xref="paper", yref="paper", showarrow=False,
            text="Perlecan direction is an unaudited model assumption (perlecan stabilizes collagen); "
                 "the paper reports the opposite. Growth-rate axis (fast growth → flatter) is correct.",
            font=dict(size=11, color="#777"))])
    return fig


STUDY_FIGURES = {
    "bm-r1-published-tissue-baseline": ("r1_regime", r1_regime_figure, "data"),
    "bm-r2-collagen-network-rheology": ("r2_fig2e", r2_fig2e_figure, "data"),
    "bm-r3-network-expansion-supply": ("r3_supply", r3_supply_figure, "data"),
    "bm-r4-coupled-2d-epithelium": ("r4_confinement", r4_confinement_figure, "data"),
    "bm-r5-explicit-3d-confinement": ("r5_monolayer", r5_monolayer_figure, "data"),
    "bm-r6-perlecan-growth-perturbations": ("r6_intervention", r6_intervention_figure, "data"),
    "bm-e1-actomyosin-cortex": ("e1_cortical_tension", e1_cortical_tension_figure, "data"),
    "bm-e2-actin-adhesion-collagen": ("e2_force_transfer", e2_force_transfer_schematic, "schematic"),
    "bm-e3-closed-3d-lumenoid": ("e3_lumenoid", e3_lumenoid_figure, "data"),
    "bm-e4-feedback-predictions": ("e4_feedback", e4_feedback_schematic, "schematic"),
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
