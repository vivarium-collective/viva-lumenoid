"""Actomyosin cortex — a LAMMPS model plugin over the viva-lammps seam (E1).

This is the collagen network's sibling: instead of a bespoke Process it is a
:class:`pbg_lammps.model.LammpsModel` plugin, so the SAME general
``LammpsModelProcess`` runs both the collagen BM and the actomyosin cortex
(``local:LammpsModelProcess`` with ``config.model`` selecting the plugin). See
``docs/architecture-processes-composites-engines.md``.

v0 coarse cortex: a 2D periodic patch of beads under Langevin dynamics whose LJ
well depth stands in for **myosin contractility** — a higher ``contractility``
raises the cortical tension the patch carries. The readout is the in-plane
cortical tension τ = -(pxx+pyy)/2 (a line-tension proxy) plus its components, so
the tension/relaxation can later be mapped onto a Tyssue cortical-energy
parameter. It is explicitly a NEW extension (the paper does not simulate actin)
and a starting point, not a filament-resolved model.
"""
from __future__ import annotations

from typing import Any

from pbg_lammps.model import LammpsModel, LammpsModelProcess, register_model


class ActomyosinCortexModel(LammpsModel):
    """2D active LJ patch; ``contractility`` (LJ ε) tunes the cortical tension.

    params (all optional):
        n            : lattice replications per side (default 8 → ~n*n beads)
        density      : 2D number density (default 0.8)
        contractility: LJ well depth ε = myosin activity proxy (default 1.0)
        cutoff       : LJ pair cutoff (default 2.5)
        temperature  : Langevin temperature (default 0.2)
        damp         : Langevin damping time (default 1.0)
        timestep     : integration timestep (default 0.005)
        seed         : RNG seed (default 12345)
    """

    def build_data(self, params: dict) -> tuple[Any, str]:
        # Atoms created inline via create_atoms; no data file.
        return None, params.get("working_directory", "")

    def build_script(self, params: dict, data_path: Any) -> str:
        n = int(params.get("n", 8))
        density = float(params.get("density", 0.8))
        eps = float(params.get("contractility", 1.0))
        cutoff = float(params.get("cutoff", 2.5))
        temp = float(params.get("temperature", 0.2))
        damp = float(params.get("damp", 1.0))
        dt = float(params.get("timestep", 0.005))
        seed = int(params.get("seed", 12345))
        lines = [
            "units lj",
            "dimension 2",
            "atom_style atomic",
            "boundary p p p",
            f"lattice sq {density}",
            f"region box block 0 {n} 0 {n} -0.1 0.1",
            "create_box 1 box",
            "create_atoms 1 box",
            "mass 1 1.0",
            f"pair_style lj/cut {cutoff}",
            # LJ well depth = contractility (myosin activity proxy).
            f"pair_coeff 1 1 {eps} 1.0",
            "pair_modify shift yes",
            f"velocity all create {temp} {seed} dist gaussian mom yes",
            f"timestep {dt}",
            "fix integ all nve",
            f"fix lang all langevin {temp} {temp} {damp} {seed}",
            "fix flat all enforce2d",
        ]
        return "\n".join(lines) + "\n"

    def ports(self) -> tuple[dict, dict]:
        inputs: dict = {}
        outputs = {
            "cortical_tension": "overwrite[float]",   # τ = -(pxx+pyy)/2
            "pressure_xx": "overwrite[float]",
            "pressure_yy": "overwrite[float]",
            "temperature": "overwrite[float]",
            "num_atoms": "overwrite[integer]",
        }
        return inputs, outputs

    def read_observables(self, lmp) -> dict:
        pxx = float(lmp.get_thermo("pxx"))
        pyy = float(lmp.get_thermo("pyy"))
        return {
            "cortical_tension": -0.5 * (pxx + pyy),
            "pressure_xx": pxx,
            "pressure_yy": pyy,
            "temperature": float(lmp.get_thermo("temp")),
            "num_atoms": int(lmp.get_natoms()),
        }


register_model("actomyosin-cortex", ActomyosinCortexModel)


def register_actomyosin(core):
    """Core extension: make ``local:LammpsModelProcess`` resolvable + ensure the
    actomyosin-cortex plugin is registered (import side effect above)."""
    if "LammpsModelProcess" not in core.link_registry:
        core.register_link("LammpsModelProcess", LammpsModelProcess)
    return core
