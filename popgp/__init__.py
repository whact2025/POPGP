"""POPGP: Phase-Ordered Pre-Geometric Projection framework."""

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
]
