"""viva_lumenoid — v1 coarse-grained collagen IV basement-membrane network.

Clean-room process-bigraph implementation of the AICS-Lumenoids "Basement
membrane — v1 model specification": a standalone collagen IV network, coarse-
grained at the fiber scale, run on a moving boundary until it reports an elastic
modulus and a remodelling time for the AICS vertex model.
"""
from .params import CollagenParams, DEFAULT
from .network import build_network_data
from .collagen import (
    CollagenNetworkProcess,
    StagedStretchController,
    register_viva_lumenoid,
    base_script,
)
from .stage1 import run_stage1, stage1_composite_spec, Stage1Result
from .stage2 import run_stage2, Stage2Result
from .ensemble import stage1_ensemble, EnsembleResult, mean_coordination
from .rigidity import run_connectivity_sweep, RigidityResult
from .bending import run_bending_sweep, BendingResult

__all__ = [
    'CollagenParams', 'DEFAULT', 'build_network_data',
    'CollagenNetworkProcess', 'StagedStretchController', 'register_viva_lumenoid',
    'base_script', 'run_stage1', 'stage1_composite_spec', 'Stage1Result',
    'run_stage2', 'Stage2Result',
    'stage1_ensemble', 'EnsembleResult', 'mean_coordination',
    'run_connectivity_sweep', 'RigidityResult',
    'run_bending_sweep', 'BendingResult',
]

# Register the workspace's composite generators (discoverable by the workbench so
# each study's baseline.composite resolves in the Composites/Registry tabs).
from . import composites  # noqa: E402,F401
