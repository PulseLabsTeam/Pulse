"""
PulseOS-Specific Enhancements

Novel features that leverage PulseOS architecture for improved performance.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class MultiScaleSurvivalEvaluator:
    """
    Multi-Scale Survival Evaluation
    
    Evaluates survival at multiple time scales (short, medium, long).
    Agent survives if meeting threshold at ANY scale (more forgiving).
    Encourages both short-term performance and long-term stability.
    """
    
    def __init__(
        self,
        short_window: int = 5,
        medium_window: int = 20,
        long_window: int = 50,
        threshold: float = None
    ):
        """
        Initialize multi-scale survival evaluator.
        
        Args:
            short_window: Short-term evaluation window (episodes)
            medium_window: Medium-term evaluation window (episodes)
            long_window: Long-term evaluation window (episodes)
            threshold: Survival threshold (if None, uses PPO baseline)
        """
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.threshold = threshold
        
        # Track performance at each scale
        self.short_term_history: List[float] = []
        self.medium_term_history: List[float] = []
        self.long_term_history: List[float] = []
    
    def evaluate(
        self,
        current_sharpe: float,
        ppo_baseline: float,
        threshold: Optional[float] = None
    ) -> Tuple[bool, Dict[str, any]]:
        """
        Evaluate survival at multiple time scales.
        
        Args:
            current_sharpe: Current Sharpe ratio
            ppo_baseline: PPO baseline Sharpe ratio
            threshold: Override threshold (if None, uses ppo_baseline or self.threshold)
            
        Returns:
            Tuple of (survives, details_dict)
        """
        effective_threshold = threshold or self.threshold or ppo_baseline
        
        # Update histories
        self.short_term_history.append(current_sharpe)
        self.medium_term_history.append(current_sharpe)
        self.long_term_history.append(current_sharpe)
        
        # Keep only recent history
        if len(self.short_term_history) > self.short_window:
            self.short_term_history.pop(0)
        if len(self.medium_term_history) > self.medium_window:
            self.medium_term_history.pop(0)
        if len(self.long_term_history) > self.long_window:
            self.long_term_history.pop(0)
        
        # Evaluate at each scale
        short_term_avg = np.mean(self.short_term_history) if self.short_term_history else current_sharpe
        medium_term_avg = np.mean(self.medium_term_history) if self.medium_term_history else current_sharpe
        long_term_avg = np.mean(self.long_term_history) if self.long_term_history else current_sharpe
        
        short_term_survives = short_term_avg >= effective_threshold
        medium_term_survives = medium_term_avg >= effective_threshold
        long_term_survives = long_term_avg >= effective_threshold
        
        # Agent survives if meeting threshold at ANY scale
        survives = short_term_survives or medium_term_survives or long_term_survives
        
        details = {
            "survives": survives,
            "short_term_avg": short_term_avg,
            "medium_term_avg": medium_term_avg,
            "long_term_avg": long_term_avg,
            "short_term_survives": short_term_survives,
            "medium_term_survives": medium_term_survives,
            "long_term_survives": long_term_survives,
            "threshold": effective_threshold
        }
        
        return survives, details


class PerformanceTrajectoryReward:
    """
    Performance Trajectory Rewards
    
    Rewards agents not just for current performance, but for improvement trajectory.
    Agents showing positive momentum get survival signal boost.
    Prevents agents from getting stuck in local optima.
    """
    
    def __init__(self, momentum_window: int = 20, trajectory_window: int = 50):
        """
        Initialize performance trajectory reward calculator.
        
        Args:
            momentum_window: Window for computing momentum (episodes)
            trajectory_window: Window for trajectory analysis (episodes)
        """
        self.momentum_window = momentum_window
        self.trajectory_window = trajectory_window
        self.performance_history: List[float] = []
    
    def compute_trajectory_bonus(
        self,
        current_performance: float
    ) -> Tuple[float, Dict[str, any]]:
        """
        Compute trajectory-based bonus.
        
        Args:
            current_performance: Current performance metric
            
        Returns:
            Tuple of (bonus, details_dict)
        """
        self.performance_history.append(current_performance)
        
        # Keep only recent history
        if len(self.performance_history) > self.trajectory_window:
            self.performance_history.pop(0)
        
        bonus = 0.0
        details = {
            "momentum": None,
            "trajectory": None,
            "acceleration": None
        }
        
        if len(self.performance_history) >= self.momentum_window:
            # Compute momentum (rate of change)
            recent = np.mean(self.performance_history[-self.momentum_window:])
            prev = np.mean(self.performance_history[-2*self.momentum_window:-self.momentum_window]) if len(self.performance_history) >= 2*self.momentum_window else self.performance_history[0]
            
            momentum = recent - prev
            details["momentum"] = momentum
            
            # Compute trajectory (overall trend)
            if len(self.performance_history) >= self.trajectory_window:
                early = np.mean(self.performance_history[:self.momentum_window])
                late = np.mean(self.performance_history[-self.momentum_window:])
                trajectory = (late - early) / (self.trajectory_window - self.momentum_window) if self.trajectory_window > self.momentum_window else 0.0
                details["trajectory"] = trajectory
                
                # Compute acceleration (rate of change of momentum)
                mid = np.mean(self.performance_history[self.momentum_window:2*self.momentum_window]) if len(self.performance_history) >= 2*self.momentum_window else recent
                acceleration = (recent - mid) - (mid - prev)
                details["acceleration"] = acceleration
                
                # Bonus based on trajectory and acceleration
                if trajectory > 0.01:  # Positive trajectory
                    bonus += 0.05 * min(1.0, trajectory * 10)  # Up to 0.05 bonus
                if acceleration > 0.005:  # Positive acceleration (improving faster)
                    bonus += 0.03 * min(1.0, acceleration * 20)  # Up to 0.03 bonus
            else:
                # Use momentum for shorter histories
                if momentum > 0.05:  # Positive momentum
                    bonus += 0.05 * min(1.0, momentum * 10)  # Up to 0.05 bonus
        
        return bonus, details


class AdaptiveLearningRateModulator:
    """
    Adaptive Learning Rate Modulation
    
    Uses NGCM gradient magnitude to modulate learning rate more aggressively.
    When gradient is high (rapidly changing survival signal), increase learning rate.
    When gradient is low (stable), reduce learning rate for fine-tuning.
    """
    
    def __init__(
        self,
        base_lr: float = 0.01,
        min_lr: float = 0.001,
        max_lr: float = 0.05,
        gradient_history_window: int = 10
    ):
        """
        Initialize adaptive learning rate modulator.
        
        Args:
            base_lr: Base learning rate
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            gradient_history_window: Window for gradient history
        """
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.gradient_history_window = gradient_history_window
        self.gradient_history: List[float] = []
    
    def modulate_learning_rate(
        self,
        current_gradient: float,
        survival_signal: float
    ) -> Tuple[float, Dict[str, any]]:
        """
        Modulate learning rate based on gradient magnitude and survival signal.
        
        Args:
            current_gradient: Current gradient magnitude from NGCM
            survival_signal: Current survival signal
            
        Returns:
            Tuple of (modulated_lr, details_dict)
        """
        self.gradient_history.append(abs(current_gradient))
        
        # Keep only recent history
        if len(self.gradient_history) > self.gradient_history_window:
            self.gradient_history.pop(0)
        
        # Compute gradient statistics
        if len(self.gradient_history) >= 3:
            avg_gradient = np.mean(self.gradient_history)
            gradient_std = np.std(self.gradient_history)
        else:
            avg_gradient = abs(current_gradient)
            gradient_std = 0.0
        
        # Modulate based on gradient magnitude
        # High gradient = rapidly changing = increase LR
        # Low gradient = stable = decrease LR
        gradient_factor = 1.0
        if avg_gradient > 0.5:  # High gradient (rapidly changing)
            gradient_factor = 1.5  # Increase LR by 50%
        elif avg_gradient > 0.3:  # Moderate gradient
            gradient_factor = 1.2  # Increase LR by 20%
        elif avg_gradient < 0.1:  # Low gradient (stable)
            gradient_factor = 0.7  # Decrease LR by 30%
        elif avg_gradient < 0.2:  # Low-moderate gradient
            gradient_factor = 0.85  # Decrease LR by 15%
        
        # Also consider survival signal
        # When DYING, we want higher LR regardless of gradient
        survival_factor = 1.0
        if survival_signal < 0.3:  # DYING
            survival_factor = 1.3  # Increase LR by 30%
        elif survival_signal > 0.7:  # ALIVE
            survival_factor = 0.9  # Slight decrease for fine-tuning
        
        # Combine factors
        combined_factor = gradient_factor * survival_factor
        
        # Compute modulated learning rate
        modulated_lr = self.base_lr * combined_factor
        modulated_lr = np.clip(modulated_lr, self.min_lr, self.max_lr)
        
        details = {
            "base_lr": self.base_lr,
            "modulated_lr": modulated_lr,
            "gradient_factor": gradient_factor,
            "survival_factor": survival_factor,
            "combined_factor": combined_factor,
            "avg_gradient": avg_gradient,
            "gradient_std": gradient_std
        }
        
        return modulated_lr, details


