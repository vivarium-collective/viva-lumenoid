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

__all__ = [
    'CollagenParams', 'DEFAULT', 'build_network_data',
    'CollagenNetworkProcess', 'StagedStretchController', 'register_viva_lumenoid',
    'base_script', 'run_stage1', 'stage1_composite_spec', 'Stage1Result',
]
