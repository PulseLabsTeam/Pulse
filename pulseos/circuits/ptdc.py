"""
Performance Threshold Detection Circuit (PTDC)

Hardware-optimized threshold detection with sub-millisecond latency.
Implements patent-specified normalization and parallel comparison operations.
"""

import time
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from collections import deque


@dataclass
class ThresholdConfig:
    """Configuration for threshold detection"""
    threshold: float
    normalization_window: int = 100
    detection_interval: float = 0.001  # Sub-millisecond
    enable_normalization: bool = True


class PerformanceThresholdDetectionCircuit:
    """
    Performance Threshold Detection Circuit (PTDC)
    
    Implements hardware-optimized comparison operations with:
    - Normalization circuit: M_norm(t) = M_t / M_initial
    - Parallel comparison array for multiple metrics
    - Sub-millisecond latency detection
    - Configurable threshold registers with hot-reload capability
    """
    
    def __init__(
        self,
        threshold: float,
        normalization_window: int = 100,
        detection_interval: float = 0.001
    ):
        """
        Initialize PTDC.
        
        Args:
            threshold: Performance threshold value
            normalization_window: Window size for normalization baseline
            detection_interval: Minimum time between detections (seconds)
        """
        self.config = ThresholdConfig(
            threshold=threshold,
            normalization_window=normalization_window,
            detection_interval=detection_interval
        )
        
        # Normalization baseline storage
        self.initial_metrics: Dict[str, float] = {}
        self.metric_history: Dict[str, deque] = {}
        
        # Threshold registers (hot-reloadable)
        self.threshold_registers: Dict[str, float] = {}
        self.last_detection_time: Dict[str, float] = {}
        
        # Performance tracking
        self.detection_count: int = 0
        self.total_detection_time: float = 0.0
        
    def set_threshold(self, agent_id: str, threshold: float) -> None:
        """
        Set threshold for a specific agent (hot-reload capability).
        
        Args:
            agent_id: Agent identifier
            threshold: New threshold value
        """
        self.threshold_registers[agent_id] = threshold
    
    def get_threshold(self, agent_id: str) -> float:
        """Get current threshold for an agent."""
        return self.threshold_registers.get(agent_id, self.config.threshold)
    
    def register_agent(self, agent_id: str, initial_metric: float) -> None:
        """
        Register an agent and initialize normalization baseline.
        
        Args:
            agent_id: Agent identifier
            initial_metric: Initial performance metric value
        """
        self.initial_metrics[agent_id] = initial_metric
        self.metric_history[agent_id] = deque(maxlen=self.config.normalization_window)
        self.metric_history[agent_id].append(initial_metric)
        self.threshold_registers[agent_id] = self.config.threshold
        self.last_detection_time[agent_id] = 0.0
    
    def normalize_metric(self, agent_id: str, metric: float) -> float:
        """
        Normalize metric using baseline: M_norm(t) = M_t / M_initial
        
        Args:
            agent_id: Agent identifier
            metric: Raw metric value
            
        Returns:
            Normalized metric value
        """
        if agent_id not in self.initial_metrics:
            self.register_agent(agent_id, metric)
        
        initial = self.initial_metrics[agent_id]
        
        if initial == 0.0:
            return 1.0 if metric == 0.0 else float('inf')
        
        return metric / initial
    
    def evaluate(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """
        Evaluate all metrics against thresholds with parallel comparison.
        
        Implements hardware-optimized parallel comparison array.
        Uses numpy vectorization for sub-millisecond latency.
        
        Args:
            metrics: Dictionary mapping agent_id to current metric value
            
        Returns:
            Dictionary mapping agent_id to threshold status (True = meeting threshold)
        """
        detection_start = time.perf_counter()
        
        if not metrics:
            return {}
        
        # Prepare vectors for parallel comparison
        agent_ids = list(metrics.keys())
        metric_values = np.array([metrics[aid] for aid in agent_ids])
        
        # Normalize metrics if enabled
        if self.config.enable_normalization:
            initial_values = np.array([
                self.initial_metrics.get(aid, metrics[aid])
                for aid in agent_ids
            ])
            # Avoid division by zero
            initial_values = np.where(initial_values == 0, 1.0, initial_values)
            normalized_metrics = metric_values / initial_values
        else:
            normalized_metrics = metric_values
        
        # Get thresholds for parallel comparison
        thresholds = np.array([
            self.get_threshold(aid) for aid in agent_ids
        ])
        
        # Parallel comparison array (vectorized)
        # Hardware-optimized: single vectorized operation
        threshold_status = normalized_metrics >= thresholds
        
        # Update history
        current_time = time.time()
        for agent_id, metric in metrics.items():
            if agent_id not in self.metric_history:
                self.register_agent(agent_id, metric)
            
            # Rate limiting: only update if enough time has passed
            if current_time - self.last_detection_time[agent_id] >= self.config.detection_interval:
                self.metric_history[agent_id].append(metric)
                self.last_detection_time[agent_id] = current_time
        
        # Convert to dictionary
        result = {
            agent_id: bool(status)
            for agent_id, status in zip(agent_ids, threshold_status)
        }
        
        # Performance tracking
        detection_duration = time.perf_counter() - detection_start
        self.detection_count += 1
        self.total_detection_time += detection_duration
        
        return result
    
    def evaluate_single(self, agent_id: str, metric: float) -> bool:
        """
        Evaluate single metric against threshold.
        
        Args:
            agent_id: Agent identifier
            metric: Current metric value
            
        Returns:
            True if metric meets threshold, False otherwise
        """
        if agent_id not in self.initial_metrics:
            self.register_agent(agent_id, metric)
        
        normalized = self.normalize_metric(agent_id, metric)
        threshold = self.get_threshold(agent_id)
        
        return normalized >= threshold
    
    def get_average_detection_latency(self) -> float:
        """Get average detection latency in seconds."""
        if self.detection_count == 0:
            return 0.0
        return self.total_detection_time / self.detection_count
    
    def get_statistics(self) -> Dict[str, any]:
        """Get PTDC statistics."""
        return {
            "detection_count": self.detection_count,
            "average_latency_ms": self.get_average_detection_latency() * 1000,
            "registered_agents": len(self.initial_metrics),
            "normalization_enabled": self.config.enable_normalization
        }
    
    def get_state(self) -> Dict[str, any]:
        """Get current state for snapshot/restore."""
        return {
            "initial_metrics": self.initial_metrics.copy(),
            "threshold_registers": self.threshold_registers.copy(),
            "config": {
                "threshold": self.config.threshold,
                "normalization_window": self.config.normalization_window,
                "detection_interval": self.config.detection_interval
            }
        }
    
    def restore_state(self, state: Dict[str, any]) -> None:
        """Restore state from snapshot."""
        self.initial_metrics = state.get("initial_metrics", {}).copy()
        self.threshold_registers = state.get("threshold_registers", {}).copy()
        
        config_data = state.get("config", {})
        self.config.threshold = config_data.get("threshold", self.config.threshold)
        self.config.normalization_window = config_data.get(
            "normalization_window", self.config.normalization_window
        )
        self.config.detection_interval = config_data.get(
            "detection_interval", self.config.detection_interval
        )
        
        # Rebuild history deques
        self.metric_history = {
            aid: deque(maxlen=self.config.normalization_window)
            for aid in self.initial_metrics.keys()
        }

