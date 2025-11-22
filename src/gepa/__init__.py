# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from .adapters import default_adapter
from .api import optimize
from .core.adapter import EvaluationBatch, GEPAAdapter
from .core.result import GEPAResult
from .examples import aime
from .strategies.eval_policy import FullEvaluationPolicy, RandomSplitEvaluationPolicy
from .utils.stop_condition import (
    CompositeStopper,
    FileStopper,
    MaxMetricCallsStopper,
    NoImprovementStopper,
    ScoreThresholdStopper,
    SignalStopper,
    StopperProtocol,
    TimeoutStopCondition,
)

from importlib import import_module

__all__ = [
    "CompositeStopper",
    "FileStopper",
    "MaxMetricCallsStopper",
    "NoImprovementStopper",
    "ScoreThresholdStopper",
    "SignalStopper",
    "StopperProtocol",
    "TimeoutStopCondition",
]

def __getattr__(name: str):
    """Lazily import names from gepa.utils.stop_condition to avoid circular imports."""
    if name in __all__:
        mod = import_module("gepa.utils.stop_condition")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")