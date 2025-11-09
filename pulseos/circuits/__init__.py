"""Circuits package - Core patent-specified algorithms"""

from pulseos.circuits.ptdc import PerformanceThresholdDetectionCircuit
from pulseos.circuits.ngcm import NonlinearGradientComputationModule
from pulseos.circuits.apc import AdaptiveParameterController

__all__ = [
    "PerformanceThresholdDetectionCircuit",
    "NonlinearGradientComputationModule",
    "AdaptiveParameterController"
]

