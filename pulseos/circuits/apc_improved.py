"""
Adaptive Parameter Controller (APC)

Implements patent-specified adaptive learning rate and exploration rate control
with momentum-based parameter updates.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class APCState:
    """State of Adaptive Parameter Controller"""
    alpha: float
    epsilon: float
    alpha_history: deque = field(default_factory=lambda: deque(maxlen=100))
    epsilon_history: deque = field(default_factory=lambda: deque(maxlen=100))


class ImprovedAdaptiveParameterController:
    """
    Adaptive Parameter Controller (APC)
    
    Implements patent-specified adaptive parameter control:
    - Learning Rate: α(t) = α_base × (1 + γ × G(t) × (1 - S(t)))
    - Exploration Rate: ε(t) = ε_min + (ε_max - ε_min) × (1 - S(t))^κ
    
    Features momentum-based updates and configurable adaptation parameters.
    """
    
    def __init__(
        self,
        alpha_base: float = 0.01,
        alpha_max_change: float = 0.50,  # Increased from 0.10 (50% max change)
        alpha_smooth: float = 0.75,  # Decreased from 0.9 (less smoothing)
        epsilon_min: float = 0.01,
        epsilon_max: float = 0.3,
        epsilon_kappa: float = 2.0,
        gamma: float = 0.5,  # Increased from 0.1 (stronger adaptation signal)
        momentum_decay: float = 0.9  # Momentum decay for accumulated adaptation
    ):
        """
        Initialize APC.
        
        Args:
            alpha_base: Base learning rate
            alpha_max_change: Maximum change per step (default 0.50)
            alpha_smooth: Smoothing factor for EMA (default 0.75)
            epsilon_min: Minimum exploration rate
            epsilon_max: Maximum exploration rate
            epsilon_kappa: Kappa parameter for exploration rate curve
            gamma: Gamma parameter for learning rate adaptation (default 0.5)
            momentum_decay: Momentum decay factor for accumulated adaptation
        """
        self.alpha_base = alpha_base
        self.alpha_max_change = alpha_max_change
        self.alpha_smooth = alpha_smooth
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_kappa = epsilon_kappa
        self.gamma = gamma
        self.momentum_decay = momentum_decay
        
        # Current state
        self.state = APCState(
            alpha=alpha_base,
            epsilon=epsilon_max  # Start with high exploration
        )
        
        # Adaptation momentum for accumulated updates
        self.adaptation_momentum = 0.0
        
        # Statistics
        self.update_count = 0
        self.rate_limit_hits = 0
        self.momentum_updates = 0
        
    def update_parameters(
        self,
        gradient: float,
        survival_signal: float
    ) -> Tuple[float, float]:
        """
        Update adaptive parameters with improved adaptation magnitude.
        
        Uses momentum-based updates to accumulate adaptation over multiple steps.
        
        Args:
            gradient: Gradient value G(t) from NGCM
            survival_signal: Survival signal S(t) from constraint evaluation
            
        Returns:
            Tuple of (alpha, epsilon) updated values
        """
        self.update_count += 1
        
        # Compute new alpha using patent equation
        # α(t) = α_base × (1 + γ × G(t) × (1 - S(t)))
        alpha_raw = self.alpha_base * (
            1.0 + self.gamma * gradient * (1.0 - survival_signal)
        )
        
        # Compute adaptation delta
        adaptation_delta = alpha_raw - self.state.alpha
        
        # Update momentum (accumulate adaptation over multiple steps)
        self.adaptation_momentum = (
            self.momentum_decay * self.adaptation_momentum +
            (1.0 - self.momentum_decay) * adaptation_delta
        )
        
        # Apply momentum to current alpha
        alpha_with_momentum = self.state.alpha + self.adaptation_momentum
        
        # Apply rate limiting
        alpha_change = alpha_with_momentum - self.state.alpha
        max_allowed_change = abs(self.state.alpha * self.alpha_max_change)
        
        if abs(alpha_change) > max_allowed_change:
            self.rate_limit_hits += 1
            alpha_change = np.sign(alpha_change) * max_allowed_change
            alpha_with_momentum = self.state.alpha + alpha_change
            self.adaptation_momentum = alpha_change
        else:
            self.momentum_updates += 1
        
        # Apply exponential moving average smoothing
        alpha_smoothed = (
            self.alpha_smooth * self.state.alpha +
            (1.0 - self.alpha_smooth) * alpha_with_momentum
        )
        
        # Saturation arithmetic: ensure alpha stays positive
        alpha_new = max(0.0, alpha_smoothed)
        
        # Compute new epsilon using patent equation
        # ε(t) = ε_min + (ε_max - ε_min) × (1 - S(t))^κ
        survival_complement = 1.0 - survival_signal
        epsilon_new = self.epsilon_min + (
            (self.epsilon_max - self.epsilon_min) *
            (survival_complement ** self.epsilon_kappa)
        )
        
        # Clamp epsilon to valid range
        epsilon_new = np.clip(epsilon_new, self.epsilon_min, self.epsilon_max)
        
        # Update state
        self.state.alpha = alpha_new
        self.state.epsilon = epsilon_new
        
        # Update history
        self.state.alpha_history.append(alpha_new)
        self.state.epsilon_history.append(epsilon_new)
        
        return alpha_new, epsilon_new
    
    def increase_exploration(self, factor: float = 1.5) -> None:
        """
        Increase exploration rate (used during rollback recovery).
        
        Args:
            factor: Multiplicative factor for exploration increase
        """
        new_epsilon = min(
            self.epsilon_max,
            self.state.epsilon * factor
        )
        self.state.epsilon = new_epsilon
    
    def decrease_exploration(self, factor: float = 0.8) -> None:
        """
        Decrease exploration rate (used during convergence).
        
        Args:
            factor: Multiplicative factor for exploration decrease
        """
        new_epsilon = max(
            self.epsilon_min,
            self.state.epsilon * factor
        )
        self.state.epsilon = new_epsilon
    
    def get_alpha(self) -> float:
        """Get current learning rate."""
        return self.state.alpha
    
    def get_epsilon(self) -> float:
        """Get current exploration rate."""
        return self.state.epsilon
    
    def get_statistics(self) -> Dict[str, any]:
        """Get APC statistics."""
        alpha_history = list(self.state.alpha_history)
        epsilon_history = list(self.state.epsilon_history)
        
        return {
            "current_alpha": self.state.alpha,
            "current_epsilon": self.state.epsilon,
            "alpha_base": self.alpha_base,
            "update_count": self.update_count,
            "rate_limit_hits": self.rate_limit_hits,
            "rate_limit_rate": (
                self.rate_limit_hits / self.update_count
                if self.update_count > 0 else 0.0
            ),
            "momentum_updates": self.momentum_updates,
            "adaptation_momentum": self.adaptation_momentum,
            "alpha_mean": np.mean(alpha_history) if alpha_history else self.state.alpha,
            "alpha_std": np.std(alpha_history) if alpha_history else 0.0,
            "alpha_change_magnitude": (
                abs(alpha_history[-1] - alpha_history[0]) / alpha_history[0]
                if len(alpha_history) > 1 else 0.0
            ),
            "epsilon_mean": np.mean(epsilon_history) if epsilon_history else self.state.epsilon,
            "epsilon_std": np.std(epsilon_history) if epsilon_history else 0.0
        }
    
    def get_state(self) -> Dict[str, any]:
        """Get current state for snapshot/restore."""
        return {
            "alpha": self.state.alpha,
            "epsilon": self.state.epsilon,
            "alpha_base": self.alpha_base,
            "alpha_max_change": self.alpha_max_change,
            "alpha_smooth": self.alpha_smooth,
            "epsilon_min": self.epsilon_min,
            "epsilon_max": self.epsilon_max,
            "epsilon_kappa": self.epsilon_kappa,
            "gamma": self.gamma,
            "momentum_decay": self.momentum_decay,
            "adaptation_momentum": self.adaptation_momentum,
            "update_count": self.update_count,
            "rate_limit_hits": self.rate_limit_hits,
            "momentum_updates": self.momentum_updates,
            "alpha_history": list(self.state.alpha_history),
            "epsilon_history": list(self.state.epsilon_history)
        }
    
    def restore_state(self, state: Dict[str, any]) -> None:
        """Restore state from snapshot."""
        self.state.alpha = state.get("alpha", self.alpha_base)
        self.state.epsilon = state.get("epsilon", self.epsilon_max)
        
        self.alpha_base = state.get("alpha_base", self.alpha_base)
        self.alpha_max_change = state.get("alpha_max_change", self.alpha_max_change)
        self.alpha_smooth = state.get("alpha_smooth", self.alpha_smooth)
        self.epsilon_min = state.get("epsilon_min", self.epsilon_min)
        self.epsilon_max = state.get("epsilon_max", self.epsilon_max)
        self.epsilon_kappa = state.get("epsilon_kappa", self.epsilon_kappa)
        self.gamma = state.get("gamma", self.gamma)
        self.momentum_decay = state.get("momentum_decay", self.momentum_decay)
        self.adaptation_momentum = state.get("adaptation_momentum", 0.0)
        
        self.update_count = state.get("update_count", 0)
        self.rate_limit_hits = state.get("rate_limit_hits", 0)
        self.momentum_updates = state.get("momentum_updates", 0)
        
        # Restore history
        alpha_history = state.get("alpha_history", [])
        epsilon_history = state.get("epsilon_history", [])
        
        self.state.alpha_history = deque(alpha_history, maxlen=100)
        self.state.epsilon_history = deque(epsilon_history, maxlen=100)


