"""Turnover -> relaxation -> shape inference pipeline (R6).

Closes the reproduction campaign (Barrientos Fig 3-7): a collagen IV molecular
lifetime (INFERRED from the turnover data, and modulated by perlecan) sets the BM
relaxation time, which the reduced-order prism model (R1) turns into a PREDICTED
cell shape. The two are kept distinct — ``inferred_relaxation`` is read off the
turnover inference, ``cell_height`` / ``confinement`` are the mechanical
prediction. Conditions: control, perlecan gain/loss (changes the lifetime), and
growth-rate change (changes the doubling time).

Perlecan DESTABILIZES the collagen IV network ("perlecan promotes bond breakage",
Barrientos et al. Cell 2026): perlecan overexpression SHORTENS the collagen IV
lifetime (measured ~1 h vs ~20 h control), giving FASTER BM relaxation and
SHORTER/fatter cells; perlecan knockdown LENGTHENS the lifetime (~500 h), giving
slower relaxation and TALLER/thinner cells. So ``perlecan_factor`` < 1 models
overexpression and > 1 models knockdown. (An earlier version had this inverted and
predicted the opposite ordering; corrected against the paper 2026-10-20.)
"""
from __future__ import annotations

from process_bigraph import Process

from .tissue_prism import TissuePrismProcess


class TurnoverShapeProcess(Process):
    """Inferred collagen lifetime (x perlecan) -> BM relaxation -> prism shape."""

    config_schema = {
        "collagen_lifetime": {"_type": "float", "_default": 5.0},  # inferred collagen IV lifetime
        "perlecan_factor": {"_type": "float", "_default": 1.0},    # lifetime multiplier: <1 = overexpression (shortens), >1 = knockdown (lengthens)
        "doubling_time": {"_type": "float", "_default": 5.0},      # growth-rate condition
        "bm_stiffness": {"_type": "float", "_default": 1.0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        c = self.config
        # INFERRED: perlecan modulates the collagen IV lifetime, which sets the BM
        # relaxation time (turnover -> relaxation).
        self._relaxation = c["collagen_lifetime"] * c["perlecan_factor"]
        # PREDICTED: run the reduced-order mechanics with that relaxation.
        self._prism = TissuePrismProcess(config={
            "bm_stiffness": c["bm_stiffness"],
            "bm_relaxation": self._relaxation,
            "doubling_time": c["doubling_time"],
        }, core=core)

    def inputs(self):
        return {}

    def outputs(self):
        return {
            "inferred_relaxation": "overwrite[float]",  # inferred (turnover) quantity
            "basal_area": "overwrite[float]",
            "cell_height": "overwrite[float]",          # predicted (mechanics) quantities
            "confinement": "overwrite[float]",
        }

    def update(self, state, interval):
        o = self._prism.update({}, interval)
        return {
            "inferred_relaxation": self._relaxation,
            "basal_area": o["basal_area"],
            "cell_height": o["cell_height"],
            "confinement": o["confinement"],
        }


def register_turnover_shape(core):
    """Core extension: make ``local:TurnoverShapeProcess`` resolvable."""
    if "TurnoverShapeProcess" not in core.link_registry:
        core.register_link("TurnoverShapeProcess", TurnoverShapeProcess)
    return core
