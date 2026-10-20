"""Each new study (R1-R6, E1-E4) has a figure that builds. Data-driven figures
skip cleanly without the heavy engine stack (CI); schematics always build."""
import pytest
from viva_lumenoid.study_figures import STUDY_FIGURES, EXTRA_STUDY_FIGURES, COMPANION_FIGURES


@pytest.mark.parametrize("slug", list(STUDY_FIGURES))
def test_study_figure_builds(slug):
    name, fn, kind = STUDY_FIGURES[slug]
    if kind == "data":
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


def test_driving_schema_builds():
    # Investigation-level framework diagram (no engine needed).
    from viva_lumenoid.study_figures import driving_schema_figure
    fig = driving_schema_figure()
    assert len(fig.layout.shapes) >= 6  # three model + three relationship boxes
