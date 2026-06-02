"""RDT-Dx: Recursive Depth Transformer for Medical LLM.

A medical diagnostic LLM that enhances complex case reasoning by reusing
intermediate Transformer layers (Core) across multiple iterations with
Bridge-based state calibration, instead of scaling parameter count.
"""

from .rdt_fixed import RDTFixed16, create_rdt_fixed_16, RDTOutput, LoRALinear
from .bridge import BridgeV2, BridgeRegistry
from .aggregation import (
    aggregate_last4_mean,
    aggregate_last,
    aggregate_mean_all,
    GatedMeanAggregation,
    get_aggregation,
)

__all__ = [
    "RDTFixed16",
    "create_rdt_fixed_16",
    "RDTOutput",
    "LoRALinear",
    "BridgeV2",
    "BridgeRegistry",
    "aggregate_last4_mean",
    "aggregate_last",
    "aggregate_mean_all",
    "GatedMeanAggregation",
    "get_aggregation",
]
__version__ = "0.1.0"
