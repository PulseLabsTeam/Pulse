"""
PulseOS Trading Agent

Trading agent that implements the PulseOS Agent interface for survival-pressure learning.
"""

import numpy as np
from typing import Dict, Any, List
from pulseos import Agent
from trading_env import TradingEnv


class PulseOSTradingAgent(Agent):
    """
    PulseOS-based trading agent with adaptive learning.
    
    Uses survival-pressure learning to adapt trading strategy based on
    performance metrics (Sharpe ratio, returns, drawdown).
    """
    
    def __init__(self, agent_id: str, env: TradingEnv, seed: int = None, initial_weights: Dict[str, np.ndarray] = None, death_penalty_multiplier: float = 100.0):
        """
        Initialize PulseOS trading agent.
        
        Args:
            agent_id: Unique agent identifier
            env: Trading environment
            seed: Random seed for initialization (if None, uses hash of agent_id)
            initial_weights: Optional dict with 'policy_weights', 'value_weights', etc. for warm start
            death_penalty_multiplier: Magnitude of death penalty (default 100.0, negative value applied in DYING state)
        """
        super().__init__(agent_id)
        self.env = env
        
        # Get state and action dimensions
        state = env.reset()
        self.state_dim = len(state)
        self.action_dim = 3  # Hold, Buy, Sell
        
        # Regularization for variance reduction (reduced strength)
        self.entropy_coef = 0.005  # Reduced entropy regularization (was 0.01)
        self.value_coef = 0.5  # Value loss coefficient
        self.policy_reg_coef = 5e-5  # Reduced L2 regularization (was 1e-4)
        
        # Performance tracking for early stopping
        self.performance_window = 30  # Longer window
        self.min_performance_threshold = 0.2  # Lower threshold (was 0.3)
        
        # Adaptive learning rate decay (slower decay)
        self.lr_decay = 0.9998  # Slower decay (was 0.9995)
        self.min_lr = 1e-5
        
        # Weight initialization - support warm start or random initialization
        if initial_weights is not None:
            # Warm start from provided weights
            self.policy_weights = initial_weights.get('policy_weights', None).copy()
            self.policy_bias = initial_weights.get('policy_bias', np.zeros(self.action_dim))
            self.value_weights = initial_weights.get('value_weights', None).copy()
            self.value_bias = initial_weights.get('value_bias', 0.0)
            
            # Add small perturbation for diversity (reduced noise for better consistency)
            if initial_weights.get('add_noise', True):
                noise_scale = initial_weights.get('noise_scale', 0.01)  # Default 1% noise (reduced from 2%)
                self.policy_weights += np.random.randn(*self.policy_weights.shape) * noise_scale * np.std(self.policy_weights)
                self.value_weights += np.random.randn(*self.value_weights.shape) * noise_scale * np.std(self.value_weights)
        else:
            # OPTIMIZED weight initialization: Based on seed 1 analysis
            # Balance between stability (0.3x) and potential (0.5x) -> use 0.35x
            # Seed 1 analysis showed: larger weights (0.5x) worked well, but 0.3x reduces variance
            # Compromise: 0.35x for policy, 0.25x for value (slightly more conservative)
            if seed is not None:
                np.random.seed(seed)
            else:
                np.random.seed(hash(agent_id) % 2**31)  # Seed based on agent_id for reproducibility
            
            # Optimized initialization: Balance stability and potential
            scale = np.sqrt(2.0 / (self.state_dim + self.action_dim))
            
            # OPTIMIZED: Use 0.35x multiplier (compromise between 0.3x stability and 0.5x potential)
            # Seed 1 analysis: 0.5x worked but had high variance, 0.3x is stable but conservative
            # 0.35x provides good balance: more potential than 0.3x, more stable than 0.5x
            self.policy_weights = np.random.randn(self.state_dim, self.action_dim) * scale * 0.35
            
            # OPTIMIZED: Adaptive bias based on seed characteristics
            # Seed 1 had zero bias and succeeded, but small bias helps exploration
            # Use smaller bias (0.005 instead of 0.01) to be closer to seed 1's zero bias
            seed_hash = hash(str(seed)) if seed is not None else hash(agent_id)
            bias_scale = 0.005 * (1.0 + 0.1 * np.sin(seed_hash % 100))  # Slight variation based on seed
            self.policy_bias = np.random.randn(self.action_dim) * bias_scale
            
            # OPTIMIZED: More conservative value initialization (0.25x instead of 0.2x)
            # Slightly larger than before to match policy weight increase
            self.value_weights = np.random.randn(self.state_dim) * scale * 0.25
            self.value_bias = 0.0
            
            np.random.seed()  # Reset seed
        
        # Verify dimensions
        assert self.policy_weights.shape == (self.state_dim, self.action_dim), \
            f"Policy weights shape mismatch: {self.policy_weights.shape} != ({self.state_dim}, {self.action_dim})"
        assert self.value_weights.shape == (self.state_dim,), \
            f"Value weights shape mismatch: {self.value_weights.shape} != ({self.state_dim},)"
        
        # Trading state
        self.current_state = state
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_done = False
        
        # Performance tracking
        self.performance_history = []
        self.episode_metrics = []
        
        # REINFORCE-style learning
        self.gamma = 0.99
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        
        # Adaptive learning parameters (updated by PulseOS runtime)
        self.momentum = 0.0
        self.momentum_decay = 0.9
        
        # Gradient clipping for stability
        self.max_gradient_norm = 0.5  # Tighter clipping for more stability
        
        # Learning rate scaling (base learning rate from PulseOS)
        self.base_lr = 0.01
        self.lr_scale = 1.0
        
        # Gradient accumulation for more stable updates
        self.gradient_buffer = None
        self.gradient_buffer_size = 5  # Increased from 3 for more stable updates
        self.gradient_buffer_count = 0
        
        # Survival reward bonus tracking
        self.survival_bonus_weight = 0.5  # Weight for survival bonus in reward
        self.last_survival_signal = None  # Track last survival signal for reward bonus
        self._policy_update_pending = False  # Track if policy update is pending
        
        # ALPHA-SEEKING: Track how much we're exceeding baseline to reward alpha
        self.alpha_seeking_bonus_weight = 0.3  # Additional reward for exceeding baseline significantly
        
        # Death penalty configuration (configurable for hyperparameter tuning)
        self.death_penalty_multiplier = death_penalty_multiplier
        self.base_death_penalty = death_penalty_multiplier  # Store base value for progressive schedule
        
        # Episode tracking for progressive death penalty
        self.current_episode = 0
        
    def _get_progressive_death_penalty(self) -> float:
        """
        Get progressive death penalty based on current episode and performance.
        
        STRATEGY 1: Eliminate Death Penalty Interference
        - Grace Period: First 100 episodes = ZERO death penalties
        - Minimal Penalties: Episodes 100-300 = 0.01 penalty (10x reduction)
        - Progressive Only: Episodes 300+ = 0.1 penalty (10x reduction from current)
        - Remove penalty caps: Let natural reward signal dominate
        
        Expected Impact: Agents spend more time ALIVE, better learning signal
        """
        # STRATEGY 1: Grace period - ZERO penalties for first 100 episodes
        if self.current_episode < 100:
            return 0.0  # No death penalties during grace period
        
        # STRATEGY 1: Minimal penalties for episodes 100-300
        elif self.current_episode < 300:
            base_penalty = 0.01  # 10x reduction from previous 0.1
        # STRATEGY 1: Progressive penalties for episodes 300+
        else:
            base_penalty = 0.1  # 10x reduction from previous 2.0
        
        # No performance-based scaling - let natural reward signal dominate
        # No learning rate-based adjustment - keep it simple
        # No penalty cap - let natural reward signal dominate
        
        return base_penalty
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability"""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / (np.sum(exp_logits) + 1e-8)
    
    def set_survival_signal(self, survival_signal: float, distance_to_baseline: float = None) -> None:
        """
        Set survival signal for reward bonus calculation.
        
        This is called by the runtime after calculating survival signal.
        The bonus will be added to episode rewards before policy update.
        
        Args:
            survival_signal: Survival signal value (0.0 to 1.0)
            distance_to_baseline: How much we're exceeding baseline (positive = beating, negative = below)
        """
        self.last_survival_signal = survival_signal
        
        # CRITICAL: Add survival bonus to current episode rewards BEFORE policy update
        # This incentivizes the agent to MAINTAIN ALIVE status
        if len(self.episode_rewards) > 0:
            self._add_survival_bonus_to_rewards(survival_signal, distance_to_baseline)
        
        # Now update policy if it was pending
        if self._policy_update_pending:
            self._update_policy()
            self._policy_update_pending = False
    
    def _add_survival_bonus_to_rewards(self, survival_signal: float, distance_to_baseline: float = None) -> None:
        """
        Add survival bonus to episode rewards.
        
        STRATEGY 2: Survival Pressure as Exploration Boost Only
        - Remove death penalties from rewards entirely
        - Use survival signal ONLY for positive rewards when ALIVE (bonus, not penalty removal)
        - Pure positive reinforcement: Reward exceeding baseline, don't penalize below
        
        Expected Impact: Clean learning signal, no penalty interference
        
        Args:
            survival_signal: Survival signal value (0.0 to 1.0)
            distance_to_baseline: How much we're exceeding baseline (positive = beating, negative = below)
        """
        if len(self.episode_rewards) == 0:
            return
        
        # STRATEGY 2: Only add positive rewards when ALIVE
        # No penalties for STRUGGLING or DYING - let natural reward signal dominate
        # Survival pressure is applied via LR/exploration boost in trading_rl_test.py
        
        survival_bonus = 0.0
        
        if survival_signal > 0.7:  # ALIVE - Only case where we add rewards
            # ALPHA-SEEKING: Reward exceeding baseline significantly
            # Use quadratic scaling for high performance to create strong incentive
            
            # Normalized distance above 0.7 threshold (0.0 to 0.3)
            distance_above_threshold = (survival_signal - 0.7) / 0.3
            
            # Base survival bonus (quadratic scaling)
            survival_bonus = self.survival_bonus_weight * (distance_above_threshold ** 2)
            
            # Additional exponential boost for very high performance (0.9+)
            if survival_signal > 0.9:
                extra_boost = self.survival_bonus_weight * 0.5 * ((survival_signal - 0.9) / 0.1) ** 2
                survival_bonus += extra_boost
            
            # ALPHA-SEEKING BONUS: Reward significantly exceeding baseline
            # This encourages risk-taking and alpha-seeking, not just survival
            if distance_to_baseline is not None and distance_to_baseline > 0:
                # STRATEGY 7: Alpha-Seeking Multiplier (Enhanced)
                # Increase reward multiplier when exceeding baseline significantly
                # At +0.5 above baseline: alpha bonus ≈ 0.3 * 0.25 = 0.075
                # At +1.0 above baseline: alpha bonus ≈ 0.3 * 1.0 = 0.3
                # At +2.0 above baseline: alpha bonus ≈ 0.3 * 4.0 = 1.2 (MASSIVE alpha reward!)
                alpha_bonus = self.alpha_seeking_bonus_weight * (distance_to_baseline ** 2)
                survival_bonus += alpha_bonus
            
            # STRATEGY 7: Consistency Bonus
            # Reward agents for consistent performance (low variance)
            if len(self.performance_history) >= 20:
                recent_performance = self.performance_history[-20:]
                performance_std = np.std(recent_performance)
                if performance_std < 0.1:  # Very consistent (low variance)
                    consistency_bonus = 0.1 * (1.0 - performance_std / 0.1)  # Up to 0.1 bonus
                    survival_bonus += consistency_bonus
            
            # STRATEGY 7: Risk-Adjusted Rewards
            # Scale rewards by Sharpe ratio improvement, not just absolute performance
            if distance_to_baseline is not None and distance_to_baseline > 0:
                # Additional bonus proportional to Sharpe improvement
                sharpe_improvement_bonus = 0.05 * min(1.0, distance_to_baseline / 1.0)  # Up to 0.05 bonus
                survival_bonus += sharpe_improvement_bonus
            
            # STRATEGY 7: Momentum Multiplier - 2x rewards when improving rapidly
            # Check if performance is improving rapidly
            if len(self.performance_history) >= 20:
                recent_10 = np.mean(self.performance_history[-10:])
                prev_10 = np.mean(self.performance_history[-20:-10])
                momentum = recent_10 - prev_10
                
                if momentum > 0.1:  # Improving rapidly (>0.1 improvement)
                    momentum_multiplier = 1.0 + min(1.0, momentum * 10.0)  # Up to 2x multiplier
                    survival_bonus *= momentum_multiplier
            
            # STRATEGY 7: Sustained Performance Bonus - Bonus for staying ALIVE >10 episodes
            # Track consecutive ALIVE episodes
            if not hasattr(self, 'consecutive_alive_episodes'):
                self.consecutive_alive_episodes = 0
            
            if survival_signal > 0.7:  # ALIVE
                self.consecutive_alive_episodes += 1
                if self.consecutive_alive_episodes > 10:
                    # Bonus increases with consecutive ALIVE episodes (up to 0.2 bonus)
                    sustained_bonus = 0.02 * min(10, self.consecutive_alive_episodes - 10)  # 0.02 per episode after 10
                    survival_bonus += sustained_bonus
            else:
                self.consecutive_alive_episodes = 0  # Reset counter if not ALIVE
            
            # STRATEGY 7: Trajectory Rewards - Exponential bonus for positive acceleration
            # Check if performance is accelerating (improving at increasing rate)
            if len(self.performance_history) >= 30:
                recent_10 = np.mean(self.performance_history[-10:])
                mid_10 = np.mean(self.performance_history[-20:-10])
                prev_10 = np.mean(self.performance_history[-30:-20])
                
                velocity_1 = mid_10 - prev_10  # First derivative (velocity)
                velocity_2 = recent_10 - mid_10  # Second derivative (acceleration)
                
                if velocity_2 > velocity_1 and velocity_2 > 0.05:  # Positive acceleration and improving
                    # Exponential bonus for acceleration
                    acceleration_bonus = 0.1 * (velocity_2 ** 2)  # Quadratic scaling
                    survival_bonus += acceleration_bonus
            
            # At 0.7 signal: bonus = 0 (just alive)
            # At 0.85 signal: bonus ≈ 0.125
            # At 0.9 signal: bonus ≈ 0.22
            # At 1.0 signal: bonus ≈ 0.75
            # PLUS alpha bonus for exceeding baseline significantly!
        
        # STRATEGY 2: No penalties for STRUGGLING or DYING
        # Survival pressure is applied via LR/exploration boost only
        # Let natural reward signal dominate
        
        # Add bonus to all episode rewards (distributed evenly)
        if survival_bonus > 0:
            bonus_per_step = survival_bonus / len(self.episode_rewards)
            for i in range(len(self.episode_rewards)):
                self.episode_rewards[i] += bonus_per_step
    
    async def step(self) -> Dict[str, Any]:
        """
        Execute one step of trading.
        
        Returns:
            Dictionary with step results
        """
        if self.episode_done:
            # Start new episode
            self.current_state = self.env.reset()
            self.episode_reward = 0.0
            self.episode_steps = 0
            self.episode_done = False
            self.episode_states = []
            self.episode_actions = []
            self.episode_rewards = []
            self.episode_log_probs = []
            # Increment episode counter for progressive death penalty
            self.current_episode += 1
            # Don't reset gradient buffer - keep accumulating across episodes
        
        # Select action
        logits = self.current_state @ self.policy_weights + self.policy_bias
        action_probs = self._softmax(logits)
        
        # Use exploration rate from PulseOS (clamped to reasonable range)
        exploration_rate = max(0.01, min(0.25, self.exploration_rate))
        
        # Better exploration: use epsilon-greedy with temperature scaling
        # Add temperature to make exploration smoother
        temperature = 1.0 + exploration_rate * 0.5  # Slight temperature increase for exploration
        scaled_logits = logits / temperature
        scaled_probs = self._softmax(scaled_logits)
        
        if np.random.random() < exploration_rate:
            # Exploration: sample from temperature-scaled distribution
            action = np.random.choice(self.action_dim, p=scaled_probs)
        else:
            # Exploitation: take best action
            action = np.argmax(action_probs)
        
        log_prob = np.log(action_probs[action] + 1e-8)
        
        # Take step in environment
        next_state, reward, done, info = self.env.step(action)
        
        # Store episode data
        self.episode_states.append(self.current_state.copy())
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_log_probs.append(log_prob)
        
        self.current_state = next_state
        self.episode_reward += reward
        self.episode_steps += 1
        self.episode_done = done
        
        # Update policy if episode is done
        if done:
            # Defer policy update - will be called after survival signal is calculated
            # This allows survival bonus to be added to rewards before policy update
            self._policy_update_pending = True
            metrics = self.env.get_metrics()
            self.episode_metrics.append(metrics)
            self.performance_history.append(self.get_performance_metric())
        
        return {
            "action": action,
            "reward": reward,
            "done": done,
            "episode_reward": self.episode_reward,
            "episode_steps": self.episode_steps,
            **info
        }
    
    def _update_policy(self):
        """Update policy using REINFORCE with adaptive learning rate"""
        if len(self.episode_rewards) == 0:
            return
        
        # Compute returns
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = np.array(returns)
        
        # Normalize returns with better handling
        if len(returns) > 1:
            returns_mean = returns.mean()
            returns_std = returns.std()
            if returns_std > 1e-6:
                returns = (returns - returns_mean) / returns_std
            else:
                returns = returns - returns_mean  # Just center if std too small
        
        # Clip returns to prevent extreme values
        returns = np.clip(returns, -5.0, 5.0)
        
        # Compute policy gradient
        states_array = np.array(self.episode_states)
        actions_array = np.array(self.episode_actions)
        
        # Forward pass
        logits = states_array @ self.policy_weights + self.policy_bias
        action_probs = np.array([self._softmax(l) for l in logits])
        log_probs = np.log(action_probs[np.arange(len(actions_array)), actions_array] + 1e-8)
        
        # Policy gradient (REINFORCE)
        policy_gradient = np.zeros_like(self.policy_weights)
        for i in range(len(states_array)):
            state = states_array[i]
            action = actions_array[i]
            return_val = returns[i]
            
            # Compute gradient for this step
            logits_single = state @ self.policy_weights + self.policy_bias
            probs_single = self._softmax(logits_single)
            
            # Gradient w.r.t. policy weights
            for j in range(self.state_dim):
                for k in range(self.action_dim):
                    if k == action:
                        grad = state[j] * (1 - probs_single[k])
                    else:
                        grad = -state[j] * probs_single[k]
                    policy_gradient[j, k] += grad * return_val
        
        # Normalize gradient
        if len(states_array) > 0:
            policy_gradient /= len(states_array)
        
        # Accumulate gradients for more stable updates
        if self.gradient_buffer is None:
            self.gradient_buffer = policy_gradient.copy()
            self.gradient_buffer_count = 1
        else:
            self.gradient_buffer = 0.7 * self.gradient_buffer + 0.3 * policy_gradient
            self.gradient_buffer_count += 1
        
        # Use accumulated gradient if we have enough samples
        if self.gradient_buffer_count >= self.gradient_buffer_size:
            policy_gradient = self.gradient_buffer.copy()
            self.gradient_buffer_count = 0  # Reset counter
        else:
            # Use current gradient but scale down
            policy_gradient = policy_gradient * (self.gradient_buffer_count / self.gradient_buffer_size)
        
        # Add entropy regularization for exploration
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-8), axis=1).mean()
        entropy_bonus = self.entropy_coef * entropy
        
        # Add L2 regularization on policy weights
        l2_reg = self.policy_reg_coef * np.sum(self.policy_weights ** 2)
        
        # Clip gradients for stability (tighter clipping)
        grad_norm = np.linalg.norm(policy_gradient)
        if grad_norm > self.max_gradient_norm:
            policy_gradient = policy_gradient * (self.max_gradient_norm / grad_norm)
        
        # Update with adaptive learning rate and momentum
        self.momentum = self.momentum_decay * self.momentum + (1 - self.momentum_decay) * policy_gradient
        
        # Apply learning rate decay
        self.learning_rate = max(self.min_lr, self.learning_rate * self.lr_decay)
        
        # Use learning rate from PulseOS runtime with scaling
        # Ensure learning rate is reasonable (not too small or too large)
        effective_lr = max(self.min_lr, min(0.05, self.learning_rate)) * self.lr_scale
        
        # Update policy with regularization (reduced impact)
        # Add small entropy bonus (reduced)
        entropy_bonus_grad = entropy_bonus * np.random.randn(*self.policy_weights.shape) * 0.005  # Reduced (was 0.01)
        policy_update = self.momentum + entropy_bonus_grad
        # Subtract L2 regularization (reduced)
        policy_update -= self.policy_reg_coef * self.policy_weights
        
        self.policy_weights += effective_lr * policy_update
        
        # Update value function with better learning
        values = (states_array @ self.value_weights + self.value_bias).flatten()
        value_error = returns - values
        
        # Clip value errors more aggressively
        value_error = np.clip(value_error, -3.0, 3.0)
        
        # Value function gradient with L2 regularization
        value_gradient = states_array.T @ value_error / len(states_array)
        value_l2_reg = self.policy_reg_coef * self.value_weights
        value_gradient -= value_l2_reg
        
        # Clip value gradient
        value_grad_norm = np.linalg.norm(value_gradient)
        if value_grad_norm > self.max_gradient_norm:
            value_gradient = value_gradient * (self.max_gradient_norm / value_grad_norm)
        
        # Update value function with momentum-like smoothing
        value_update = 0.7 * getattr(self, '_value_momentum', np.zeros_like(value_gradient)) + 0.3 * value_gradient
        self._value_momentum = value_update
        
        # Update value function
        self.value_weights += effective_lr * self.value_coef * value_update
        self.value_bias += effective_lr * 0.1 * np.mean(value_error)
        
        # Clip value bias to prevent extreme values
        self.value_bias = np.clip(self.value_bias, -10.0, 10.0)
    
    def get_sharpe_ratio(self) -> float:
        """
        Get current Sharpe ratio for PPO baseline comparison.
        
        Returns:
            Current Sharpe ratio from most recent episode
        """
        if len(self.episode_metrics) == 0:
            return 0.0
        
        recent_metrics = self.episode_metrics[-1]
        return recent_metrics.get("sharpe_ratio", 0.0)
    
    def get_performance_metric(self) -> float:
        """
        Get current performance metric for PulseOS survival constraint.
        
        Returns:
            Performance metric (0-1 scale) based on Sharpe ratio and returns
        """
        if len(self.episode_metrics) == 0:
            return 0.0
        
        # Use most recent episode metrics
        recent_metrics = self.episode_metrics[-1]
        
        sharpe = recent_metrics.get("sharpe_ratio", 0.0)
        total_return = recent_metrics.get("total_return", 0.0)
        max_drawdown = recent_metrics.get("max_drawdown", 1.0)
        
        # Better performance metric calculation
        # Sharpe ratio is most important for trading
        
        # Normalize Sharpe ratio with better handling of negative values
        # Use sigmoid-like function for better scaling
        if sharpe > 0:
            sharpe_metric = 1.0 / (1.0 + np.exp(-(sharpe - 1.5) / 0.5))
        else:
            # Penalize negative Sharpe more aggressively
            sharpe_metric = max(0.0, 1.0 / (1.0 + np.exp(-(sharpe + 1.0) / 0.3)))
        
        sharpe_metric = np.clip(sharpe_metric, 0.0, 1.0)
        
        # Return metric (log scale for better handling of large returns)
        # Handle zero returns better
        if total_return > 0:
            return_metric = min(1.0, np.log(1 + total_return) / np.log(1 + 0.5))  # Normalize to 50% return
        elif total_return == 0:
            return_metric = 0.3  # Small penalty for zero returns (was 0.0)
        else:
            return_metric = max(0.0, 1.0 + total_return / 0.5)  # Penalize negative returns
        
        return_metric = np.clip(return_metric, 0.0, 1.0)
        
        # Drawdown penalty (lower is better, target: <20%)
        drawdown_metric = np.clip(1.0 - max_drawdown / 0.3, 0.0, 1.0)
        
        # Combined performance metric - heavily favor Sharpe ratio
        # Sharpe ratio is the key metric for trading performance
        performance = 0.7 * sharpe_metric + 0.2 * return_metric + 0.1 * drawdown_metric
        
        return performance
    
    def get_weights(self, noise_scale: float = 0.01) -> Dict[str, np.ndarray]:
        """Get current weights for warm starting other agents"""
        return {
            'policy_weights': self.policy_weights.copy(),
            'policy_bias': self.policy_bias.copy(),
            'value_weights': self.value_weights.copy(),
            'value_bias': self.value_bias,
            'add_noise': True,  # Flag for whether to add noise when warm starting
            'noise_scale': noise_scale  # Noise scale for warm start (default 1%)
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get agent state for snapshot"""
        state = super().get_state()
        state.update({
            "policy_weights": self.policy_weights.tolist(),
            "policy_bias": self.policy_bias.tolist(),
            "value_weights": self.value_weights.tolist(),
            "value_bias": self.value_bias,
            "momentum": self.momentum.tolist() if isinstance(self.momentum, np.ndarray) else self.momentum,
            "episode_metrics": self.episode_metrics[-10:] if self.episode_metrics else []
        })
        return state
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore agent state from snapshot"""
        super().restore_state(state)
        if "policy_weights" in state:
            self.policy_weights = np.array(state["policy_weights"])
        if "policy_bias" in state:
            self.policy_bias = np.array(state["policy_bias"])
        if "value_weights" in state:
            self.value_weights = np.array(state["value_weights"])
        if "value_bias" in state:
            self.value_bias = state["value_bias"]
        if "momentum" in state:
            self.momentum = np.array(state["momentum"]) if isinstance(state["momentum"], list) else state["momentum"]
        if "episode_metrics" in state:
            self.episode_metrics = state["episode_metrics"]

