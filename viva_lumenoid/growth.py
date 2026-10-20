"""Preferred (target) growth + reduced-order BM reaction — the cell<->BM feedback
primitive shared by the spatial/coupled studies (R4, R5, E4).

The modeling constraint from the 2026-10-20 modeling review (Eran/Tasnif/Saurabh):
growth is an increase in the PREFERRED (target) cell area/volume, NOT an imposed
realized geometry. The realized geometry LAGS the target because a viscoelastic
basement membrane resists the expansion — and that lag IS the confinement. Growth
here is distinct from cell DIVISION (proliferation): no cells are added; a single
cell's target simply grows.

Two pure, engine-free primitives so the dynamics are testable without Tyssue/LAMMPS:

- ``PreferredGrowth`` — the time-varying target law A0(t) / V0(t).
- ``ReducedBM`` — a Maxwell viscoelastic BM (stiffness + relaxation) whose reaction
  stress resists a realized geometry's excess over its relaxing rest state.

The spatial studies feed ``PreferredGrowth.target(t)`` into a Tyssue model's
``prefered_area`` / ``prefered_vol`` each macro-step (so the realized geometry
relaxes toward it through the engine's own area/volume elasticity), and apply the
``ReducedBM`` reaction as an added basal tension that resists spreading. This
mirrors ``TissuePrism``'s ``_A_pref`` / ``_A`` / ``_A_bm`` split and the reduced
path in ``LammpsTyssueCoupling``, factored out so all three studies share one
auditable contract.
"""
from __future__ import annotations

import math


class PreferredGrowth:
    """Time-varying preferred (target) geometry — growth, not division.

    Parameters
    ----------
    x0 : float
        The initial preferred area/volume.
    mode : {"none", "linear", "exponential"}
        ``none`` holds the target constant (the no-growth control); ``linear``
        gives ``x0*(1 + rate*t)``; ``exponential`` gives ``x0*exp(rate*t)``.
    rate : float
        Growth rate (per unit time). ``0.0`` is equivalent to ``mode="none"``.

    Proliferation (cell division) is OFF: ``divides`` is always ``False``. This is
    deliberate — the MVP grows a single cell's target geometry; faithful
    reproduction of the paper's proliferation is a separate extension.
    """

    MODES = ("none", "linear", "exponential")

    def __init__(self, x0: float, mode: str = "none", rate: float = 0.0):
        if mode not in self.MODES:
            raise ValueError(
                f"unknown growth mode {mode!r}; pick one of {self.MODES}")
        self.x0 = float(x0)
        self.mode = mode
        self.rate = float(rate)
        self.divides = False  # explicit: proliferation (cell division) is OFF

    def target(self, t: float) -> float:
        """The preferred (target) area/volume at time ``t``."""
        if self.mode == "none" or self.rate == 0.0:
            return self.x0
        if self.mode == "linear":
            return self.x0 * (1.0 + self.rate * t)
        return self.x0 * math.exp(self.rate * t)


class ReducedBM:
    """Reduced-order viscoelastic basement membrane (a Maxwell element).

    Resists a realized geometry ``x``'s excess over its relaxing rest state::

        rest  += (x - rest) / tau * dt      # stress relaxation
        stress = stiffness * (x - rest)     # elastic reaction

    Controls fall out of the parameters: ``stiffness=0`` is a free / no-BM control;
    a large ``relaxation`` (tau) is a slow-relaxing, confining BM; a small one is a
    fast-relaxing, spreading BM. The reaction is the reduced-order, analytic
    alternative to the explicit LAMMPS collagen BM (never additive with it).
    """

    def __init__(self, x0: float, stiffness: float = 1.0, relaxation: float = 5.0):
        self.rest = float(x0)
        self.stiffness = float(stiffness)
        self.relaxation = float(relaxation)

    def react(self, x: float, dt: float) -> float:
        """Advance the rest state by ``dt`` and return the current reaction stress."""
        self.rest += (x - self.rest) / max(self.relaxation, 1e-9) * dt
        return self.stiffness * (x - self.rest)
