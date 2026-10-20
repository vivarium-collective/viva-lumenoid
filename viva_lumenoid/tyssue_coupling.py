"""Tyssue vertex-model engine registration for the viva-lumenoid composites.

Registers the vivarium-tyssue engine (schema types + EulerSolver + the geometry
classes incl. the merged CrossSectionGeometry / LumenoidGeometry) INTO the
workspace core so a Tyssue composite — ``EulerSolver`` over a geometry + an
``eptm`` mesh — resolves and runs. Mirrors ``vivarium_tyssue.core.build_core``.

Lazy by design: ``vivarium_tyssue`` (and ``tyssue``) are imported only when the
core extension runs at composite-BUILD time, so ``viva_lumenoid.composites``
imports cleanly when the optional engine isn't installed (CI without it).
"""
from __future__ import annotations


def register_tyssue(core):
    """Core extension: make ``local:EulerSolver`` + the tyssue types resolvable."""
    # Best-effort numpy/topology compat shims (same as vivarium_tyssue.core).
    for mod, fn in (
        ("vivarium_tyssue._tyssue_compat", "apply_tyssue_compat"),
        ("vivarium_tyssue.behaviors.tyssue_patches", "apply_tyssue_topology_patches"),
    ):
        try:
            import importlib
            getattr(importlib.import_module(mod), fn)()
        except Exception:  # noqa: BLE001 — shims are best-effort
            pass

    from vivarium_tyssue.data_types import register_types
    from vivarium_tyssue.processes import register_processes

    core = register_types(core)
    core = register_processes(core)
    return core
