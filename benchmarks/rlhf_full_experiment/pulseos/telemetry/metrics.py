"""
Metrics Collection System

Collects and exports runtime metrics for monitoring and analysis.
Supports Prometheus and OpenTelemetry export formats.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import json


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects runtime metrics for telemetry.
    
    Features:
    - High-performance metric collection
    - Configurable retention
    - Export to Prometheus/OpenTelemetry formats
    """
    
    def __init__(self, max_retention: int = 10000):
        """
        Initialize metrics collector.
        
        Args:
            max_retention: Maximum number of data points to retain
        """
        self.max_retention = max_retention
        
        # Metric storage
        self.metrics: Dict[str, deque] = {
            "survival_signal": deque(maxlen=max_retention),
            "alpha": deque(maxlen=max_retention),
            "epsilon": deque(maxlen=max_retention),
            "gradient": deque(maxlen=max_retention),
            "step_duration": deque(maxlen=max_retention),
            "agent_count": deque(maxlen=max_retention)
        }
        
        # Statistics
        self.total_steps = 0
        self.start_time = time.time()
    
    def record_step(
        self,
        step: int,
        duration: float,
        survival_signal: float,
        alpha: float,
        epsilon: float,
        gradient: float,
        agent_count: int
    ) -> None:
        """
        Record metrics for a step.
        
        Args:
            step: Step number
            duration: Step duration in seconds
            survival_signal: Survival signal value
            alpha: Learning rate
            epsilon: Exploration rate
            gradient: Gradient value
            agent_count: Number of agents
        """
        timestamp = time.time()
        
        self.metrics["survival_signal"].append(MetricPoint(timestamp, survival_signal))
        self.metrics["alpha"].append(MetricPoint(timestamp, alpha))
        self.metrics["epsilon"].append(MetricPoint(timestamp, epsilon))
        self.metrics["gradient"].append(MetricPoint(timestamp, gradient))
        self.metrics["step_duration"].append(MetricPoint(timestamp, duration))
        self.metrics["agent_count"].append(MetricPoint(timestamp, agent_count))
        
        self.total_steps += 1
    
    def get_metric(self, metric_name: str) -> List[MetricPoint]:
        """Get metric data points."""
        return list(self.metrics.get(metric_name, deque()))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregated statistics."""
        stats = {
            "total_steps": self.total_steps,
            "uptime_seconds": time.time() - self.start_time
        }
        
        for metric_name, points in self.metrics.items():
            if points:
                values = [p.value for p in points]
                stats[f"{metric_name}_mean"] = sum(values) / len(values)
                stats[f"{metric_name}_min"] = min(values)
                stats[f"{metric_name}_max"] = max(values)
                stats[f"{metric_name}_count"] = len(values)
        
        return stats
    
    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus metrics text format
        """
        lines = []
        
        for metric_name, points in self.metrics.items():
            if points:
                # Use most recent value
                latest = points[-1]
                lines.append(
                    f"pulseos_{metric_name} {latest.value} {int(latest.timestamp * 1000)}"
                )
        
        return "\n".join(lines)
    
    def export_json(self) -> str:
        """
        Export metrics as JSON.
        
        Returns:
            JSON string
        """
        data = {
            "timestamp": time.time(),
            "metrics": {
                name: [
                    {"timestamp": p.timestamp, "value": p.value}
                    for p in points
                ]
                for name, points in self.metrics.items()
            },
            "statistics": self.get_statistics()
        }
        
        return json.dumps(data, indent=2)
    
    def clear(self) -> None:
        """Clear all metrics."""
        for metric in self.metrics.values():
            metric.clear()
        self.total_steps = 0
        self.start_time = time.time()

