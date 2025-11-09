"""
PulseOS Framework - Production-Grade Survival-Pressure Learning System

A technically sophisticated implementation of patent-specified adaptive learning
algorithms with full performance optimization and production-ready architecture.

This framework implements:
- Performance Threshold Detection Circuit (PTDC)
- Nonlinear Gradient Computation Module (NGCM)
- Adaptive Parameter Controller (APC)
- State Persistence and Rollback Subsystem (SPRS)
"""

__version__ = "1.0.0"

from pulseos.runtime import Runtime, Config
from pulseos.agent import Agent, SurvivalConstraint
from pulseos.circuits.ptdc import PerformanceThresholdDetectionCircuit
from pulseos.circuits.ngcm import NonlinearGradientComputationModule
from pulseos.circuits.apc import AdaptiveParameterController
from pulseos.persistence.snapshot import StateSnapshot, SnapshotManager

__all__ = [
    "Runtime",
    "Config",
    "Agent",
    "SurvivalConstraint",
    "PerformanceThresholdDetectionCircuit",
    "NonlinearGradientComputationModule",
    "AdaptiveParameterController",
    "StateSnapshot",
    "SnapshotManager",
]

