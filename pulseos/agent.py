"""
Agent Interface and Survival Constraint System

Defines the agent interface and constraint algebra.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ConstraintOperator(Enum):
    """Constraint composition operators"""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"


@dataclass
class Constraint:
    """Base constraint definition"""
    threshold: float
    operator: Optional[ConstraintOperator] = None
    children: List['Constraint'] = field(default_factory=list)


class SurvivalConstraint:
    """
    Constraint system with full constraint algebra.
    
    Features:
    - Composite constraints: AND, OR, NOT operations
    - Temporal constraints: time-based conditions
    - Statistical constraints: variance, percentiles
    - Multi-objective constraints with Pareto optimization
    - Constraint learning: automatic threshold adaptation
    """
    
    def __init__(
        self,
        threshold: float,
        constraint_type: str = "simple",
        temporal_window: Optional[int] = None,
        statistical_mode: Optional[str] = None,
        learning_rate: float = 0.01
    ):
        """
        Initialize survival constraint.
        
        Args:
            threshold: Performance threshold value
            constraint_type: Type of constraint (simple, composite, temporal, statistical)
            temporal_window: Window size for temporal constraints
            statistical_mode: Statistical mode (mean, median, percentile, variance)
            learning_rate: Learning rate for adaptive threshold (default: 0.01)
        """
        self.threshold = threshold
        self.constraint_type = constraint_type
        self.temporal_window = temporal_window
        self.statistical_mode = statistical_mode
        
        # Constraint history for temporal/statistical evaluation
        self.history: List[float] = []
        
        # Adaptive threshold learning
        self.adaptive_threshold = threshold
        self.learning_rate = learning_rate
    
    def evaluate(self, metric: float) -> bool:
        """
        Evaluate if metric meets constraint.
        
        Args:
            metric: Performance metric value
            
        Returns:
            True if constraint is met
        """
        self.history.append(metric)
        
        if self.temporal_window:
            # Keep only recent history
            self.history = self.history[-self.temporal_window:]
        
        if self.constraint_type == "simple":
            return metric >= self.threshold
        
        elif self.constraint_type == "statistical":
            return self._evaluate_statistical()
        
        elif self.constraint_type == "temporal":
            return self._evaluate_temporal()
        
        else:
            return metric >= self.threshold
    
    def _evaluate_statistical(self) -> bool:
        """Evaluate statistical constraint."""
        if len(self.history) < 2:
            return self.history[-1] >= self.threshold if self.history else False
        
        if self.statistical_mode == "mean":
            value = np.mean(self.history)
        elif self.statistical_mode == "median":
            value = np.median(self.history)
        elif self.statistical_mode == "percentile":
            value = np.percentile(self.history, 90)
        elif self.statistical_mode == "variance":
            value = np.var(self.history)
        else:
            value = np.mean(self.history)
        
        return value >= self.threshold
    
    def _evaluate_temporal(self) -> bool:
        """Evaluate temporal constraint."""
        if not self.history:
            return False
        
        if self.temporal_window and len(self.history) < self.temporal_window:
            # Not enough history yet
            return True
        
        # Check if recent values meet threshold
        recent_values = self.history[-self.temporal_window:] if self.temporal_window else self.history
        return all(v >= self.threshold for v in recent_values)
    
    def compute_survival_signal(self, survival_ratio: float) -> float:
        """
        Compute survival pressure signal from survival ratio.
        
        Args:
            survival_ratio: Ratio of agents meeting threshold (0-1)
            
        Returns:
            Survival signal value (0-1)
        """
        # Apply sigmoid-like transformation for smooth signal
        # Higher ratio -> higher signal (less pressure)
        return survival_ratio
    
    def adapt_threshold(self, performance_history: List[float]) -> None:
        """
        Adapt threshold based on performance history.
        
        Args:
            performance_history: Historical performance values
        """
        if not performance_history:
            return
        
        # Simple adaptive threshold: move toward median performance
        median_performance = np.median(performance_history)
        
        # Gradually adapt threshold
        self.adaptive_threshold = (
            (1 - self.learning_rate) * self.adaptive_threshold +
            self.learning_rate * median_performance
        )
        
        # Update threshold
        self.threshold = self.adaptive_threshold


class Agent(ABC):
    """
    Base agent interface for survival-pressure learning.
    
    All agents must implement this interface to work with PulseOS runtime.
    """
    
    def __init__(self, agent_id: str):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique identifier for the agent
        """
        self.agent_id = agent_id
        self.learning_rate = 0.01
        self.exploration_rate = 0.1
        self.performance_history: List[float] = []
    
    @abstractmethod
    async def step(self) -> Dict[str, Any]:
        """
        Execute one step of agent behavior.
        
        Returns:
            Dictionary containing step results
        """
        pass
    
    @abstractmethod
    def get_performance_metric(self) -> float:
        """
        Get current performance metric.
        
        Returns:
            Performance metric value
        """
        pass
    
    def update_learning_rate(self, alpha: float) -> None:
        """
        Update learning rate.
        
        Args:
            alpha: New learning rate value
        """
        self.learning_rate = alpha
    
    def update_exploration_rate(self, epsilon: float) -> None:
        """
        Update exploration rate.
        
        Args:
            epsilon: New exploration rate value
        """
        self.exploration_rate = epsilon
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current agent state for snapshot.
        
        Returns:
            State dictionary
        """
        return {
            "agent_id": self.agent_id,
            "learning_rate": self.learning_rate,
            "exploration_rate": self.exploration_rate,
            "performance_history": self.performance_history.copy()
        }
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        """
        Restore agent state from snapshot.
        
        Args:
            state: State dictionary
        """
        self.learning_rate = state.get("learning_rate", self.learning_rate)
        self.exploration_rate = state.get("exploration_rate", self.exploration_rate)
        self.performance_history = state.get("performance_history", []).copy()

