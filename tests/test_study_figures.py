"""Each new study (R1-R6, E1-E4) has a figure that builds. Data-driven figures
skip cleanly without the heavy engine stack (CI); schematics always build."""
import pytest
from viva_lumenoid.study_figures import STUDY_FIGURES


@pytest.mark.parametrize("slug", list(STUDY_FIGURES))
def test_study_figure_builds(slug):
    name, fn, kind = STUDY_FIGURES[slug]
    if kind == "data":
        pytest.importorskip("lammps")
        pytest.importorskip("tyssue")
    fig = fn()
    assert fig is not None and hasattr(fig, "data")
