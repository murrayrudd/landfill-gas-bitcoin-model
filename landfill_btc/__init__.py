"""Landfill-gas-to-Bitcoin-mining Monte Carlo model (Python reproduction)."""

from .model import (
    DEFAULT_RIGS,
    MODERN_RIGS,
    FinancialParams,
    LandfillParams,
    MitigationParams,
    RigSpec,
    Scenario,
    SimulationResult,
    infer_catchment,
    landfill_forward,
    mitigation_value,
    rigs_deployed,
    simulate,
    summary_stats,
)

__all__ = [
    "DEFAULT_RIGS",
    "MODERN_RIGS",
    "FinancialParams",
    "LandfillParams",
    "MitigationParams",
    "RigSpec",
    "Scenario",
    "SimulationResult",
    "infer_catchment",
    "landfill_forward",
    "mitigation_value",
    "rigs_deployed",
    "simulate",
    "summary_stats",
]

__version__ = "1.0.0"
