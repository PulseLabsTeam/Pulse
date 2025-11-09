"""
Enhanced Telemetry with Detailed Metrics

Provides comprehensive metrics including gradient history, cache metrics,
and convergence tracking as specified in review requirements.
"""

import time
from typing import Dict, List, Optional, Any, Deque
from dataclasses import dataclass, field
from collections import deque
import json
import numpy as np


@dataclass
class GradientHistoryPoint:
    """Gradient computation history point"""
    timestamp: float
    delta: float
    gradient: float
    sigmoid: float
    cache_hit: bool


@dataclass
class CacheMetricsPoint:
    """Cache metrics snapshot"""
    timestamp: float
    hit_rate: float
    hits: int
    misses: int
    evictions: int
    memory_bytes: int


@dataclass
class ConvergencePoint:
    """Convergence tracking point"""
    timestamp: float
    step: int
    survival_signal: float
    converged_agents: int
    total_agents: int
    convergence_rate: float


class EnhancedMetricsCollector:
    """
    Enhanced metrics collector with detailed tracking.
    
    Tracks:
    - Gradient computation history
    - Cache performance metrics
    - Convergence progression
    - Performance trends
    """
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize enhanced metrics collector.
        
        Args:
            max_history: Maximum history points to retain
        """
        self.max_history = max_history
        
        # Gradient history
        self.gradient_history: Deque[GradientHistoryPoint] = deque(maxlen=max_history)
        
        # Cache metrics history
        self.cache_metrics_history: Deque[CacheMetricsPoint] = deque(maxlen=max_history)
        
        # Convergence tracking
        self.convergence_history: Deque[ConvergencePoint] = deque(maxlen=max_history)
        
        # Performance metrics
        self.step_durations: Deque[float] = deque(maxlen=max_history)
        self.survival_signals: Deque[float] = deque(maxlen=max_history)
        self.alpha_values: Deque[float] = deque(maxlen=max_history)
        self.epsilon_values: Deque[float] = deque(maxlen=max_history)
        
        # Statistics
        self.start_time = time.time()
        self.total_steps = 0
    
    def record_gradient(
        self,
        delta: float,
        gradient: float,
        sigmoid: float,
        cache_hit: bool
    ) -> None:
        """
        Record gradient computation.
        
        Args:
            delta: Delta value
            gradient: Computed gradient
            sigmoid: Computed sigmoid
            cache_hit: Whether this was a cache hit
        """
        point = GradientHistoryPoint(
            timestamp=time.time(),
            delta=delta,
            gradient=gradient,
            sigmoid=sigmoid,
            cache_hit=cache_hit
        )
        self.gradient_history.append(point)
    
    def record_cache_metrics(
        self,
        hit_rate: float,
        hits: int,
        misses: int,
        evictions: int,
        memory_bytes: int
    ) -> None:
        """
        Record cache performance metrics.
        
        Args:
            hit_rate: Current cache hit rate
            hits: Total cache hits
            misses: Total cache misses
            evictions: Total evictions
            memory_bytes: Memory usage in bytes
        """
        point = CacheMetricsPoint(
            timestamp=time.time(),
            hit_rate=hit_rate,
            hits=hits,
            misses=misses,
            evictions=evictions,
            memory_bytes=memory_bytes
        )
        self.cache_metrics_history.append(point)
    
    def record_convergence(
        self,
        step: int,
        survival_signal: float,
        converged_agents: int,
        total_agents: int
    ) -> None:
        """
        Record convergence progress.
        
        Args:
            step: Current step
            survival_signal: Current survival signal
            converged_agents: Number of converged agents
            total_agents: Total number of agents
        """
        point = ConvergencePoint(
            timestamp=time.time(),
            step=step,
            survival_signal=survival_signal,
            converged_agents=converged_agents,
            total_agents=total_agents,
            convergence_rate=converged_agents / total_agents if total_agents > 0 else 0.0
        )
        self.convergence_history.append(point)
    
    def record_step(
        self,
        step: int,
        duration: float,
        survival_signal: float,
        alpha: float,
        epsilon: float
    ) -> None:
        """
        Record step metrics.
        
        Args:
            step: Step number
            duration: Step duration in seconds
            survival_signal: Survival signal value
            alpha: Learning rate
            epsilon: Exploration rate
        """
        self.total_steps = step
        self.step_durations.append(duration)
        self.survival_signals.append(survival_signal)
        self.alpha_values.append(alpha)
        self.epsilon_values.append(epsilon)
    
    def get_gradient_statistics(self) -> Dict[str, Any]:
        """Get gradient computation statistics."""
        if not self.gradient_history:
            return {}
        
        gradients = [p.gradient for p in self.gradient_history]
        cache_hits = sum(1 for p in self.gradient_history if p.cache_hit)
        cache_misses = len(self.gradient_history) - cache_hits
        
        return {
            "total_computations": len(self.gradient_history),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": cache_hits / len(self.gradient_history) if self.gradient_history else 0.0,
            "gradient_mean": np.mean(gradients),
            "gradient_std": np.std(gradients),
            "gradient_min": np.min(gradients),
            "gradient_max": np.max(gradients)
        }
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        if not self.cache_metrics_history:
            return {}
        
        hit_rates = [p.hit_rate for p in self.cache_metrics_history]
        memory_usage = [p.memory_bytes for p in self.cache_metrics_history]
        
        return {
            "average_hit_rate": np.mean(hit_rates),
            "current_hit_rate": hit_rates[-1] if hit_rates else 0.0,
            "hit_rate_std": np.std(hit_rates),
            "average_memory_bytes": np.mean(memory_usage),
            "peak_memory_bytes": np.max(memory_usage) if memory_usage else 0
        }
    
    def get_convergence_statistics(self) -> Dict[str, Any]:
        """Get convergence statistics."""
        if not self.convergence_history:
            return {}
        
        convergence_rates = [p.convergence_rate for p in self.convergence_history]
        survival_signals = [p.survival_signal for p in self.convergence_history]
        
        return {
            "current_convergence_rate": convergence_rates[-1] if convergence_rates else 0.0,
            "average_convergence_rate": np.mean(convergence_rates),
            "convergence_rate_std": np.std(convergence_rates),
            "average_survival_signal": np.mean(survival_signals),
            "convergence_points": len(self.convergence_history)
        }
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get overall performance statistics."""
        if not self.step_durations:
            return {}
        
        return {
            "total_steps": self.total_steps,
            "uptime_seconds": time.time() - self.start_time,
            "average_step_duration_ms": np.mean(self.step_durations) * 1000,
            "min_step_duration_ms": np.min(self.step_durations) * 1000,
            "max_step_duration_ms": np.max(self.step_durations) * 1000,
            "average_survival_signal": np.mean(self.survival_signals) if self.survival_signals else 0.0,
            "average_alpha": np.mean(self.alpha_values) if self.alpha_values else 0.0,
            "average_epsilon": np.mean(self.epsilon_values) if self.epsilon_values else 0.0
        }
    
    def export_comprehensive_report(self) -> str:
        """Export comprehensive metrics report as JSON."""
        report = {
            "timestamp": time.time(),
            "gradient_statistics": self.get_gradient_statistics(),
            "cache_statistics": self.get_cache_statistics(),
            "convergence_statistics": self.get_convergence_statistics(),
            "performance_statistics": self.get_performance_statistics()
        }
        
        return json.dumps(report, indent=2)
    
    def clear(self) -> None:
        """Clear all metrics."""
        self.gradient_history.clear()
        self.cache_metrics_history.clear()
        self.convergence_history.clear()
        self.step_durations.clear()
        self.survival_signals.clear()
        self.alpha_values.clear()
        self.epsilon_values.clear()
        self.total_steps = 0
        self.start_time = time.time()

