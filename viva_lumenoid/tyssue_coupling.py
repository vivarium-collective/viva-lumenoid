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


def _patch_numpy_cross_2d():
    """tyssue's 2D geometry uses ``np.cross`` on 2D vectors, which numpy 2.x
    removed (raises "must be 3-dimensional"). Restore the old behaviour — a 2D
    cross returns the scalar z-component — so 2D sheets (R4) build/step. 3D cross
    is untouched. Idempotent."""
    import numpy as _np
    if getattr(_np, "_viva_cross2d_patched", False):
        return
    _orig = _np.cross

    def _cross(a, b, *args, **kwargs):
        aa, bb = _np.asarray(a), _np.asarray(b)
        if aa.shape and bb.shape and aa.shape[-1] == 2 and bb.shape[-1] == 2 \
                and not args and not kwargs:
            return aa[..., 0] * bb[..., 1] - aa[..., 1] * bb[..., 0]
        return _orig(a, b, *args, **kwargs)

    _np.cross = _cross
    _np._viva_cross2d_patched = True


def register_tyssue(core):
    """Core extension: make ``local:EulerSolver`` + the tyssue types resolvable."""
    _patch_numpy_cross_2d()
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
