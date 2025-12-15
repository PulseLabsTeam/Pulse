"""
PPO Baseline Survival Constraint

Survival constraint that compares agent performance to PPO baseline.
Agent survives if it beats or matches PPO baseline performance.
"""

import numpy as np
from typing import Dict, Optional, List
from pulseos.agent import SurvivalConstraint


class PPOBaselineSurvivalConstraint(SurvivalConstraint):
    """
    Survival constraint based on PPO baseline comparison.
    
    Instead of comparing to a fixed threshold, compares agent's Sharpe ratio
    to the PPO baseline Sharpe ratio. Agent survives if:
    - agent_sharpe >= ppo_baseline_sharpe
    
    This prevents gaming by doing nothing (0% return won't beat PPO baseline)
    and forces active competitive performance.
    """
    
    def __init__(
        self,
        ppo_baseline_sharpe: float,
        constraint_type: str = "simple",
        temporal_window: Optional[int] = None,
        statistical_mode: Optional[str] = None,
        learning_rate: float = 0.01,
        margin: float = 0.0,  # Optional margin - agent must beat baseline by this amount
        threshold_percentile: Optional[float] = None,  # Percentile of baseline to use as threshold (e.g., 0.1 for 10th percentile)
        threshold_fixed: Optional[float] = None,  # Fixed threshold value (overrides percentile if set)
        use_rolling_baseline: bool = False,  # STRATEGY 5: Use rolling baseline of agent's performance
        rolling_window: int = 20,  # STRATEGY 5: Window size for rolling baseline
        adaptive_threshold_start: Optional[float] = None,  # STRATEGY 5: Starting threshold for adaptive schedule
        adaptive_threshold_episodes: int = 500  # STRATEGY 5: Episodes to reach full baseline
    ):
        """
        Initialize PPO baseline survival constraint.
        
        Args:
            ppo_baseline_sharpe: Average PPO baseline Sharpe ratio to compare against
            constraint_type: Type of constraint evaluation (simple, statistical, temporal)
            temporal_window: Window size for temporal constraints
            statistical_mode: Statistical mode (mean, median, percentile, variance)
            learning_rate: Learning rate for adaptive threshold
            margin: Optional margin - agent must beat baseline by this amount (default: 0.0)
            threshold_percentile: Percentile of baseline to use as threshold (0.0-1.0, e.g., 0.1 for 10th percentile)
            threshold_fixed: Fixed threshold value (overrides percentile if set)
        """
        # Calculate effective threshold based on percentile or fixed value
        if threshold_fixed is not None:
            effective_threshold = threshold_fixed
        elif threshold_percentile is not None:
            # Calculate threshold as percentile of baseline
            # For example, 10th percentile means threshold = baseline * (1 - percentile)
            # So 10th percentile of 3.6 baseline = 3.6 * 0.9 = 3.24
            # But we want it lower, so we subtract: baseline - (baseline * percentile)
            # Actually, let's think: if baseline is 3.6 and we want 10th percentile,
            # we want threshold = baseline * (1 - percentile) = 3.6 * 0.9 = 3.24
            # But that's still high. Let's use: baseline - (baseline * percentile * 0.5)
            # Or simpler: threshold = baseline * (1 - percentile)
            effective_threshold = ppo_baseline_sharpe * (1.0 - threshold_percentile)
        else:
            effective_threshold = ppo_baseline_sharpe
        
        # Use calculated threshold
        super().__init__(
            threshold=effective_threshold,
            constraint_type=constraint_type,
            temporal_window=temporal_window,
            statistical_mode=statistical_mode,
            learning_rate=learning_rate
        )
        self.ppo_baseline_sharpe = ppo_baseline_sharpe
        self.margin = margin
        self.threshold_percentile = threshold_percentile
        self.threshold_fixed = threshold_fixed
        self.effective_threshold = effective_threshold + margin
        
        # STRATEGY 5: Rolling baseline and adaptive threshold support
        self.use_rolling_baseline = use_rolling_baseline
        self.rolling_window = rolling_window
        self.adaptive_threshold_start = adaptive_threshold_start
        self.adaptive_threshold_episodes = adaptive_threshold_episodes
        self.current_episode = 0  # Track current episode for adaptive threshold
        
        # Track agent Sharpe ratios (raw values, not normalized)
        self.agent_sharpe_history: Dict[str, List[float]] = {}
    
    def evaluate_sharpe(self, agent_id: str, sharpe_ratio: float, episode: Optional[int] = None) -> bool:
        """
        Evaluate if agent's Sharpe ratio meets survival constraint.
        
        Args:
            agent_id: Agent identifier
            sharpe_ratio: Agent's current Sharpe ratio
            episode: Current episode number (for adaptive threshold)
            
        Returns:
            True if agent survives (sharpe >= ppo_baseline + margin), False otherwise
        """
        # Update current episode for adaptive threshold
        if episode is not None:
            self.current_episode = episode
        
        # Track Sharpe ratio history
        if agent_id not in self.agent_sharpe_history:
            self.agent_sharpe_history[agent_id] = []
        self.agent_sharpe_history[agent_id].append(sharpe_ratio)
        
        # Keep only recent history if temporal window is set
        if self.temporal_window:
            self.agent_sharpe_history[agent_id] = \
                self.agent_sharpe_history[agent_id][-self.temporal_window:]
        
        # STRATEGY 5: Compute adaptive threshold if enabled
        effective_threshold = self.effective_threshold
        if self.adaptive_threshold_start is not None:
            # Adaptive threshold: starts at adaptive_threshold_start, gradually increases to baseline
            progress = min(1.0, self.current_episode / self.adaptive_threshold_episodes)
            adaptive_threshold = self.adaptive_threshold_start + (self.ppo_baseline_sharpe - self.adaptive_threshold_start) * progress
            effective_threshold = adaptive_threshold + self.margin
        
        # STRATEGY 5: Use rolling baseline if enabled
        if self.use_rolling_baseline and len(self.agent_sharpe_history[agent_id]) >= self.rolling_window:
            # Use rolling baseline: agent must be in top 50% of recent performance
            recent_performance = self.agent_sharpe_history[agent_id][-self.rolling_window:]
            median_performance = np.median(recent_performance)
            # Survive if current performance >= median of recent performance
            return sharpe_ratio >= median_performance
        
        # Simple comparison: agent must beat PPO baseline (with optional margin)
        if self.constraint_type == "simple":
            return sharpe_ratio >= effective_threshold
        
        elif self.constraint_type == "statistical":
            return self._evaluate_statistical_sharpe(agent_id, effective_threshold)
        
        elif self.constraint_type == "temporal":
            return self._evaluate_temporal_sharpe(agent_id, effective_threshold)
        
        else:
            return sharpe_ratio >= effective_threshold
    
    def _evaluate_statistical_sharpe(self, agent_id: str, effective_threshold: float = None) -> bool:
        """Evaluate statistical constraint on Sharpe ratios."""
        if effective_threshold is None:
            effective_threshold = self.effective_threshold
            
        if agent_id not in self.agent_sharpe_history or len(self.agent_sharpe_history[agent_id]) < 2:
            if agent_id in self.agent_sharpe_history and len(self.agent_sharpe_history[agent_id]) > 0:
                return self.agent_sharpe_history[agent_id][-1] >= effective_threshold
            return False
        
        history = self.agent_sharpe_history[agent_id]
        
        if self.statistical_mode == "mean":
            value = np.mean(history)
        elif self.statistical_mode == "median":
            value = np.median(history)
        elif self.statistical_mode == "percentile":
            value = np.percentile(history, 90)
        elif self.statistical_mode == "variance":
            value = np.var(history)
        else:
            value = np.mean(history)
        
        return value >= effective_threshold
    
    def _evaluate_temporal_sharpe(self, agent_id: str, effective_threshold: float = None) -> bool:
        """Evaluate temporal constraint on Sharpe ratios."""
        if effective_threshold is None:
            effective_threshold = self.effective_threshold
            
        if agent_id not in self.agent_sharpe_history or not self.agent_sharpe_history[agent_id]:
            return False
        
        history = self.agent_sharpe_history[agent_id]
        
        if self.temporal_window and len(history) < self.temporal_window:
            # Not enough history yet - be lenient
            return True
        
        # Check if recent values meet threshold
        recent_values = history[-self.temporal_window:] if self.temporal_window else history
        return all(v >= effective_threshold for v in recent_values)
    
    def update_baseline(self, new_ppo_baseline_sharpe: float) -> None:
        """
        Update PPO baseline Sharpe ratio (e.g., if running multiple PPO trials).
        
        Args:
            new_ppo_baseline_sharpe: New PPO baseline Sharpe ratio
        """
        self.ppo_baseline_sharpe = new_ppo_baseline_sharpe
        
        # Recalculate threshold based on percentile or fixed value
        if self.threshold_fixed is not None:
            effective_threshold = self.threshold_fixed
        elif self.threshold_percentile is not None:
            effective_threshold = new_ppo_baseline_sharpe * (1.0 - self.threshold_percentile)
        else:
            effective_threshold = new_ppo_baseline_sharpe
        
        self.threshold = effective_threshold
        self.effective_threshold = effective_threshold + self.margin
    
    def get_survival_status(self, agent_id: str) -> Dict[str, any]:
        """
        Get detailed survival status for an agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Dictionary with survival status information
        """
        if agent_id not in self.agent_sharpe_history or not self.agent_sharpe_history[agent_id]:
            return {
                "alive": False,
                "current_sharpe": None,
                "ppo_baseline": self.ppo_baseline_sharpe,
                "margin": self.margin,
                "meets_threshold": False,
                "recent_sharpe": None
            }
        
        current_sharpe = self.agent_sharpe_history[agent_id][-1]
        recent_sharpe = np.mean(self.agent_sharpe_history[agent_id][-10:]) if len(self.agent_sharpe_history[agent_id]) >= 10 else current_sharpe
        meets_threshold = current_sharpe >= self.effective_threshold
        
        return {
            "alive": meets_threshold,
            "current_sharpe": current_sharpe,
            "ppo_baseline": self.ppo_baseline_sharpe,
            "effective_threshold": self.effective_threshold,
            "threshold_percentile": self.threshold_percentile,
            "threshold_fixed": self.threshold_fixed,
            "margin": self.margin,
            "meets_threshold": meets_threshold,
            "recent_sharpe": recent_sharpe,
            "performance_gap": current_sharpe - self.ppo_baseline_sharpe,
            "threshold_gap": current_sharpe - self.effective_threshold
        }


