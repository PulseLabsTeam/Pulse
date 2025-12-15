"""Optimization package - Performance optimizations"""

from pulseos.optimization.cache import (
    MemoryPool,
    VectorizationUtils,
    CacheOptimizer,
    ZeroCopyBuffer
)
from pulseos.optimization.gradient_cache import GradientCache, CacheMetrics
from pulseos.optimization.hardware import (
    HardwareEmulationLayer,
    HardwareProfile,
    HardwareMode
)

__all__ = [
    "MemoryPool",
    "VectorizationUtils",
    "CacheOptimizer",
    "ZeroCopyBuffer",
    "GradientCache",
    "CacheMetrics",
    "HardwareEmulationLayer",
    "HardwareProfile",
    "HardwareMode"
]

