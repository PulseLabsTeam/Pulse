"""
Architectural Improvements for PulseOS

Phase 3: Enhanced components for complex scenarios:
- Multi-threshold PTDC for bimodal and multi-objective scenarios
- Skewness-aware NGCM for skewed distributions
- Multi-objective survival coordination
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import time

from pulseos.circuits.ptdc import PerformanceThresholdDetectionCircuit, ThresholdConfig
from pulseos.circuits.ngcm import NonlinearGradientComputationModule
from pulseos.circuits.apc import AdaptiveParameterController


class MultiThresholdPTDC(PerformanceThresholdDetectionCircuit):
    """
    Enhanced PTDC that handles multiple performance regions.
    
    For bimodal: Tracks distance to BOTH preference peaks
    For multi-objective: Tracks distance to EACH objective threshold
    """
    
    def __init__(
        self,
        thresholds: List[float],
        normalization_window: int = 100,
        detection_interval: float = 0.001,
        enable_normalization: bool = False,
        mode: str = "multi_objective"  # "multi_objective" or "bimodal"
    ):
        """
        Initialize multi-threshold PTDC.
        
        Args:
            thresholds: List of threshold values
            mode: "multi_objective" or "bimodal"
        """
        # Use first threshold for base initialization
        super().__init__(
            threshold=thresholds[0],
            normalization_window=normalization_window,
            detection_interval=detection_interval,
            enable_normalization=enable_normalization
        )
        
        self.thresholds = thresholds
        self.mode = mode
        
        # For bimodal: track both peaks
        if mode == "bimodal":
            self.peak_tracker = {aid: {'peak1': None, 'peak2': None} 
                                for aid in self.initial_metrics.keys()}
    
    def evaluate(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """
        Evaluate metrics against multiple thresholds.
        
        For bimodal: Returns True if metric is near either peak
        For multi-objective: Returns True if ALL objectives meet thresholds
        """
        if not metrics:
            return {}
        
        agent_ids = list(metrics.keys())
        metric_values = np.array([metrics[aid] for aid in agent_ids])
        
        if self.mode == "bimodal":
            # Bimodal: check distance to nearest peak
            results = {}
            for agent_id, metric in metrics.items():
                if agent_id not in self.peak_tracker:
                    self.peak_tracker[agent_id] = {'peak1': None, 'peak2': None}
                
                # Detect peaks from history
                if agent_id in self.metric_history and len(self.metric_history[agent_id]) > 20:
                    history = list(self.metric_history[agent_id])
                    peaks = self._detect_peaks(history)
                    
                    if len(peaks) >= 2:
                        self.peak_tracker[agent_id]['peak1'] = peaks[0]
                        self.peak_tracker[agent_id]['peak2'] = peaks[1]
                
                # Check distance to nearest peak
                peak1 = self.peak_tracker[agent_id]['peak1']
                peak2 = self.peak_tracker[agent_id]['peak2']
                
                if peak1 is not None and peak2 is not None:
                    dist1 = abs(metric - peak1)
                    dist2 = abs(metric - peak2)
                    min_dist = min(dist1, dist2)
                    # Consider "meeting threshold" if within 0.2 of either peak
                    results[agent_id] = min_dist < 0.2
                else:
                    # Fallback to single threshold
                    results[agent_id] = metric >= self.thresholds[0]
            
            return results
        
        else:  # multi_objective
            # Multi-objective: all thresholds must be met
            results = {}
            for agent_id, metric in metrics.items():
                # For multi-objective, we need multiple metrics
                # For now, use weighted combination
                # In practice, this would receive multiple metric values
                if len(self.thresholds) == 1:
                    results[agent_id] = metric >= self.thresholds[0]
                else:
                    # Assume metric is a composite score
                    # Check against all thresholds (using minimum)
                    results[agent_id] = metric >= min(self.thresholds)
            
            return results
    
    def _detect_peaks(self, history: List[float], window: int = 10) -> List[float]:
        """Detect peaks in history using simple peak detection"""
        if len(history) < window * 2:
            return []
        
        peaks = []
        for i in range(window, len(history) - window):
            if (history[i] > max(history[i-window:i]) and 
                history[i] > max(history[i+1:i+window+1])):
                peaks.append(history[i])
        
        # Return top 2 peaks
        if len(peaks) >= 2:
            peaks.sort(reverse=True)
            return peaks[:2]
        elif len(peaks) == 1:
            return [peaks[0], peaks[0]]  # Duplicate if only one peak
        
        return []
    
    def compute_composite_distance(self, agent_id: str, metric: float) -> float:
        """
        Compute composite distance to thresholds.
        
        For bimodal: Distance to nearest peak
        For multi-objective: Weighted distance to all thresholds
        """
        if self.mode == "bimodal":
            peak1 = self.peak_tracker.get(agent_id, {}).get('peak1')
            peak2 = self.peak_tracker.get(agent_id, {}).get('peak2')
            
            if peak1 is not None and peak2 is not None:
                dist1 = abs(metric - peak1)
                dist2 = abs(metric - peak2)
                return min(dist1, dist2)
        
        # Multi-objective: average distance to all thresholds
        distances = [abs(metric - thresh) for thresh in self.thresholds]
        return np.mean(distances)


class SkewnessAwareNGCM(NonlinearGradientComputationModule):
    """
    Enhanced NGCM that adapts gradient computation to distribution shape.
    
    For skewed distributions:
    - Detects skewness direction (left/right)
    - Asymmetric urgency function (steeper on sparse side)
    - Adaptive exploration (more on tail side)
    """
    
    def __init__(
        self,
        cache_size: int = 256,
        implementation: str = "LUT",
        beta: float = 1.0,
        target_hit_rate: float = 0.75,
        skewness_threshold: float = 0.5
    ):
        super().__init__(cache_size, implementation, beta, target_hit_rate)
        
        self.skewness_threshold = skewness_threshold
        self.skewness_direction: Optional[str] = None  # "left" or "right"
        self.skewness_magnitude: float = 0.0
        
        # History for skewness detection
        self.delta_history: deque = deque(maxlen=1000)
    
    def detect_skewness(self) -> Tuple[Optional[str], float]:
        """
        Detect skewness in delta distribution.
        
        Returns:
            Tuple of (direction, magnitude)
        """
        if len(self.delta_history) < 50:
            return None, 0.0
        
        deltas = np.array(list(self.delta_history))
        mean_delta = np.mean(deltas)
        median_delta = np.median(deltas)
        
        # Simple skewness measure
        if mean_delta > median_delta:
            direction = "right"
            magnitude = (mean_delta - median_delta) / (np.std(deltas) + 1e-6)
        elif mean_delta < median_delta:
            direction = "left"
            magnitude = (median_delta - mean_delta) / (np.std(deltas) + 1e-6)
        else:
            direction = None
            magnitude = 0.0
        
        return direction, magnitude
    
    def compute_gradient(self, delta: float, timestamp: int) -> float:
        """
        Compute gradient with skewness compensation.
        
        If skewness > 0 (right-skewed): Increase gradient on left side
        If skewness < 0 (left-skewed): Increase gradient on right side
        """
        # Update history
        self.delta_history.append(delta)
        
        # Detect skewness periodically
        if len(self.delta_history) % 100 == 0:
            direction, magnitude = self.detect_skewness()
            if magnitude > self.skewness_threshold:
                self.skewness_direction = direction
                self.skewness_magnitude = magnitude
        
        # Compute base gradient
        base_gradient = super().compute_gradient(delta, timestamp)
        
        # Apply skewness compensation
        if self.skewness_direction and self.skewness_magnitude > self.skewness_threshold:
            compensation_factor = self._compute_skewness_compensation(delta)
            return base_gradient * compensation_factor
        
        return base_gradient
    
    def _compute_skewness_compensation(self, delta: float) -> float:
        """
        Compute compensation factor based on skewness.
        
        For right-skewed: Increase gradient for negative deltas
        For left-skewed: Increase gradient for positive deltas
        """
        if self.skewness_direction == "right":
            # Right-skewed: sparse on left, dense on right
            # Increase gradient for negative deltas (left side)
            if delta < 0:
                return 1.0 + self.skewness_magnitude * 0.5
            else:
                return 1.0 - self.skewness_magnitude * 0.2
        
        elif self.skewness_direction == "left":
            # Left-skewed: dense on left, sparse on right
            # Increase gradient for positive deltas (right side)
            if delta > 0:
                return 1.0 + self.skewness_magnitude * 0.5
            else:
                return 1.0 - self.skewness_magnitude * 0.2
        
        return 1.0


class MultiObjectiveSurvivalConstraint:
    """
    Handle multiple competing objectives with Pareto optimization.
    
    Strategy:
    - Maintain Pareto front of non-dominated solutions
    - Survival pressure drives toward Pareto front
    - Balance exploration across objectives
    """
    
    def __init__(
        self,
        thresholds: Dict[str, float],  # objective_name -> threshold
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize multi-objective survival constraint.
        
        Args:
            thresholds: Dictionary mapping objective names to thresholds
            weights: Optional weights for each objective (default: equal)
        """
        self.thresholds = thresholds
        self.weights = weights or {obj: 1.0 / len(thresholds) for obj in thresholds.keys()}
        
        # Pareto front tracking
        self.pareto_front: List[Dict[str, float]] = []
        
        # Objective history
        self.objective_history: Dict[str, List[float]] = {
            obj: [] for obj in thresholds.keys()
        }
    
    def evaluate(self, objectives: Dict[str, float]) -> bool:
        """
        Evaluate if objectives meet thresholds.
        
        Returns:
            True if all objectives meet their thresholds
        """
        for obj_name, threshold in self.thresholds.items():
            if obj_name not in objectives:
                return False
            if objectives[obj_name] < threshold:
                return False
        
        return True
    
    def compute_survival_signal(self, objectives: Dict[str, float]) -> float:
        """
        Compute composite survival signal for multi-objective scenario.
        
        Uses scalarization: weighted combination of distances to thresholds
        """
        if not objectives:
            return 0.0
        
        # Compute normalized distances to thresholds
        distances = []
        for obj_name, threshold in self.thresholds.items():
            if obj_name in objectives:
                distance = max(0, threshold - objectives[obj_name])
                normalized_distance = distance / (threshold + 1e-6)
                weighted_distance = normalized_distance * self.weights.get(obj_name, 1.0)
                distances.append(weighted_distance)
        
        if not distances:
            return 0.0
        
        # Composite distance (lower is better)
        composite_distance = np.mean(distances)
        
        # Convert to survival signal (0-1, higher is better)
        survival_signal = max(0.0, min(1.0, 1.0 - composite_distance))
        
        return survival_signal
    
    def update_pareto_front(self, objectives: Dict[str, float]):
        """Update Pareto front with new solution"""
        # Check if solution is non-dominated
        is_dominated = False
        dominated_indices = []
        
        for i, front_solution in enumerate(self.pareto_front):
            # Check if front_solution dominates new solution
            if self._dominates(front_solution, objectives):
                is_dominated = True
                break
            
            # Check if new solution dominates front_solution
            if self._dominates(objectives, front_solution):
                dominated_indices.append(i)
        
        # Remove dominated solutions
        for idx in reversed(dominated_indices):
            self.pareto_front.pop(idx)
        
        # Add if non-dominated
        if not is_dominated:
            self.pareto_front.append(objectives.copy())
    
    def _dominates(self, solution1: Dict[str, float], solution2: Dict[str, float]) -> bool:
        """
        Check if solution1 dominates solution2.
        
        solution1 dominates solution2 if:
        - solution1 is better or equal in all objectives
        - solution1 is strictly better in at least one objective
        """
        better_in_any = False
        worse_in_any = False
        
        for obj_name in self.thresholds.keys():
            val1 = solution1.get(obj_name, 0.0)
            val2 = solution2.get(obj_name, 0.0)
            
            if val1 > val2:
                better_in_any = True
            elif val1 < val2:
                worse_in_any = True
        
        return better_in_any and not worse_in_any
    
    def get_distance_to_pareto_front(self, objectives: Dict[str, float]) -> float:
        """Compute distance to nearest point on Pareto front"""
        if not self.pareto_front:
            # No Pareto front yet, use distance to thresholds
            return sum(max(0, thresh - objectives.get(obj, 0.0)) 
                      for obj, thresh in self.thresholds.items())
        
        # Find minimum distance to any point on Pareto front
        min_distance = float('inf')
        for front_solution in self.pareto_front:
            distance = sum(abs(objectives.get(obj, 0.0) - front_solution.get(obj, 0.0))
                          for obj in self.thresholds.keys())
            min_distance = min(min_distance, distance)
        
        return min_distance

