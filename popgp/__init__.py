"""POPGP: Phase-Ordered Pre-Geometric Projection framework."""

import json

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy/torch scalar types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
        except ImportError:
            pass
        return super().default(obj)


def validation_json(report: dict) -> str:
    """Serialize a validation report to JSON, handling numpy/torch types."""
    return json.dumps(report, indent=2, cls=_NumpyEncoder)


from popgp.config import (
    BackendConfig,
    PiGeomConfig,
    PiLocConfig,
    PiResConfig,
    PiTimeConfig,
    SimulationConfig,
    SimulatorConfig,
    SubstrateConfig,
)
from popgp.simulator import Simulator, SimulatorResult

# Phase 1 modules
from popgp.coarse_grain import optimize_cells, enumerate_partitions
from popgp.capacity import cut_capacity, check_capacity_bound

__all__ = [
    # Primary API
    "Simulator",
    "SimulatorConfig",
    "SimulatorResult",
    # Config components
    "SubstrateConfig",
    "PiResConfig",
    "PiLocConfig",
    "PiGeomConfig",
    "PiTimeConfig",
    "SimulationConfig",
    "BackendConfig",
    # Phase 1: Cell selection
    "optimize_cells",
    "enumerate_partitions",
    "cut_capacity",
    "check_capacity_bound",
    # Utilities
    "validation_json",
]
