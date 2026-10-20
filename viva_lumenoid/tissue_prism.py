"""Reduced-order homogeneous tissue mechanics (R1).

The paper's P1 baseline (Barrientos Fig 1): a representative fixed-volume PRISM
cell whose basal area is set by proliferation (a growing preferred area) pushing
against a VISCOELASTIC basement membrane (a stiffness + a relaxation time). Cell
height = volume / basal area. The four stiffness-relaxation regimes emerge from BM
stiffness vs the relaxation/doubling ratio: a stiff, slowly-relaxing BM confines
the cell (it cannot spread, so it grows tall); a soft or fast-relaxing BM lets it
spread (stays flat). Homogeneous (non-spatial) — the reference the spatial Tyssue
studies (R4/R5) are compared against; the effective-BM alternative to the explicit
LAMMPS BM.
"""
from __future__ import annotations

import math

from process_bigraph import Process


class TissuePrismProcess(Process):
    """Fixed-volume prism cell + viscoelastic BM -> cell height / confinement."""

    config_schema = {
        "volume": {"_type": "float", "_default": 1.0},
        "area0": {"_type": "float", "_default": 1.0},
        "bm_stiffness": {"_type": "float", "_default": 1.0},     # BM area elasticity
        "bm_relaxation": {"_type": "float", "_default": 5.0},    # tau_bm (rest-area relaxation)
        "doubling_time": {"_type": "float", "_default": 5.0},    # proliferation timescale
        "mobility": {"_type": "float", "_default": 0.2},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        a0 = self.config["area0"]
        self._A = a0        # current basal area
        self._A_pref = a0   # preferred area (grows with proliferation)
        self._A_bm = a0     # BM rest area (viscoelastic: relaxes toward A)

    def inputs(self):
        return {}

    def outputs(self):
        return {
            "basal_area": "overwrite[float]",
            "cell_height": "overwrite[float]",
            "bm_stress": "overwrite[float]",
            "confinement": "overwrite[float]",  # height relative to the initial height
        }

    def update(self, state, interval):
        c = self.config
        # Viscoelastic BM: its rest area relaxes toward the current area.
        self._A_bm += (self._A - self._A_bm) / max(c["bm_relaxation"], 1e-9) * interval
        bm_stress = c["bm_stiffness"] * (self._A - self._A_bm)
        # Proliferation grows the preferred (target) basal area.
        self._A_pref *= math.exp(interval / max(c["doubling_time"], 1e-9))
        # Basal area relaxes toward preferred, resisted by the BM reaction stress.
        self._A += c["mobility"] * ((self._A_pref - self._A) - bm_stress) * interval
        self._A = max(self._A, 1e-6)
        height = c["volume"] / self._A
        confinement = height / (c["volume"] / c["area0"])
        return {
            "basal_area": self._A,
            "cell_height": height,
            "bm_stress": bm_stress,
            "confinement": confinement,
        }


def register_tissue_prism(core):
    """Core extension: make ``local:TissuePrismProcess`` resolvable."""
    if "TissuePrismProcess" not in core.link_registry:
        core.register_link("TissuePrismProcess", TissuePrismProcess)
    return core
