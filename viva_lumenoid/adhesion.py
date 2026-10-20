"""Basal adhesion — reversible linkers transferring cortex tension to the BM (E2).

The force-transfer seam of the actin-adhesion-collagen coupling: an active
actomyosin cortex (a ``LammpsModelProcess`` plugin) pulls, through reversible
basal linkers, on a collagen IV BM patch (``CollagenNetworkProcess``). This
process is the linker layer in between — it carries a linker **occupancy** with
simple on/off kinetics, and transmits a **traction** = occupancy x cortex_tension
that drives a BM strain rate (the cortex deforming the BM). It also holds the BM
crosslink chemistry on so the BM remodels under the transmitted load.

Minimal single-cell-patch v0 (plan E2): measures traction, linker occupancy and
the BM deformation; a starting point before multicellular coupling, not a
molecularly-resolved adhesion.
"""
from __future__ import annotations

from process_bigraph import Process


class BasalAdhesion(Process):
    """Reversible basal linkers: cortex tension -> (occupancy-gated) BM traction."""

    config_schema = {
        "k_on": {"_type": "float", "_default": 0.5},      # linker binding rate
        "k_off": {"_type": "float", "_default": 0.1},     # linker unbinding rate
        "coupling_gain": {"_type": "float", "_default": 1.0e-4},  # traction -> strain_rate
        # Keep the BM crosslink chemistry on so it remodels under the load.
        "make_on": {"_type": "float", "_default": 1.0},
        "break_on": {"_type": "float", "_default": 1.0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self._occ = 0.0  # linker occupancy in [0, 1]

    def inputs(self):
        return {"cortex_tension": "float"}

    def outputs(self):
        return {
            "traction": "overwrite[float]",
            "linker_occupancy": "overwrite[float]",
            "bm_strain_rate": "overwrite[float]",
            "make_on": "overwrite[float]",
            "break_on": "overwrite[float]",
        }

    def update(self, state, interval):
        c = self.config
        tension = float((state or {}).get("cortex_tension", 0.0))
        # Reversible binding: d(occ)/dt = k_on*(1-occ) - k_off*occ.
        self._occ += (c["k_on"] * (1.0 - self._occ) - c["k_off"] * self._occ) * interval
        self._occ = max(0.0, min(1.0, self._occ))
        traction = self._occ * tension
        return {
            "traction": traction,
            "linker_occupancy": self._occ,
            "bm_strain_rate": c["coupling_gain"] * traction,
            "make_on": c["make_on"],
            "break_on": c["break_on"],
        }


def register_adhesion(core):
    """Core extension: make ``local:BasalAdhesion`` resolvable."""
    if "BasalAdhesion" not in core.link_registry:
        core.register_link("BasalAdhesion", BasalAdhesion)
    return core
