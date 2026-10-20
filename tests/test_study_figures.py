"""Each new study (R1-R6, E1-E4) has a figure that builds. Data-driven figures
skip cleanly without the heavy engine stack (CI); schematics always build."""
import pytest
from viva_lumenoid.study_figures import (
    STUDY_FIGURES, EXTRA_STUDY_FIGURES, COMPANION_FIGURES, V5_EXTRA_FIGURES,
)

# The v4/v5 figure-review figures draw REAL results baked into
# data/figrev_v45.json, so they build without the live engines (unlike the other
# "data" figures, which run LAMMPS/tyssue). List them so the test does not
# importorskip on an engine they do not need.
_BAKED_DATA_NAMES = {"bm_v4_faithful_angles", "bm_v5_porosity", "bm_v5_bundling"}


@pytest.mark.parametrize("slug", list(STUDY_FIGURES))
def test_study_figure_builds(slug):
    name, fn, kind = STUDY_FIGURES[slug]
    if kind == "data" and name not in _BAKED_DATA_NAMES:
        pytest.importorskip("lammps")
        pytest.importorskip("tyssue")
    fig = fn()
    assert fig is not None and hasattr(fig, "data")


@pytest.mark.parametrize("slug,name,fn,kind", EXTRA_STUDY_FIGURES,
                         ids=[f"{s}:{n}" for s, n, _f, _k in EXTRA_STUDY_FIGURES])
def test_extra_study_figure_builds(slug, name, fn, kind):
    # Supplement / hypothesis companion figures from the R4/R5 figure review.
    if kind == "data":
        pytest.importorskip("lammps")
        pytest.importorskip("tyssue")
    fig = fn()
    assert fig is not None and hasattr(fig, "data")


@pytest.mark.parametrize("slug", list(COMPANION_FIGURES))
def test_companion_figure_builds(slug):
    # Supplement companion figures from the E1-E4 extension figure review.
    name, fn, kind = COMPANION_FIGURES[slug]
    if kind == "data":
        pytest.importorskip("lammps")
        pytest.importorskip("tyssue")
    fig = fn()
    assert fig is not None and hasattr(fig, "data")


@pytest.mark.parametrize("slug", list(V5_EXTRA_FIGURES))
def test_v5_extra_figures_build(slug):
    # The second MAIN figure for v5 (the porosity/bundling split) — baked data.
    for name, fn in V5_EXTRA_FIGURES[slug]:
        fig = fn()
        assert fig is not None and hasattr(fig, "data")


def test_driving_schema_builds():
    # Investigation-level framework diagram (no engine needed).
    from viva_lumenoid.study_figures import driving_schema_figure
    fig = driving_schema_figure()
    assert len(fig.layout.shapes) >= 6  # three model + three relationship boxes
