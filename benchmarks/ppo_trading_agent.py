"""
PPO Baseline Agent for Trading RL

Simple PPO implementation for comparison with PulseOS.
"""

import numpy as np
from typing import Dict, Any, List
from collections import deque
from trading_env import TradingEnv


class PPOTradingAgent:
    """
    Proximal Policy Optimization agent for trading.
    
    Simple PPO implementation with:
    - Policy network (linear)
    - Value network (linear)
    - Clipped objective
    - Advantage estimation
    """
    
    def __init__(self, env: TradingEnv, learning_rate: float = 3e-4):
        """
        Initialize PPO agent.
        
        Args:
            env: Trading environment
            learning_rate: Learning rate for policy/value updates
        """
        self.env = env
        self.learning_rate = learning_rate
        
        # Get state and action dimensions
        state = env.reset()
        self.state_dim = len(state)
        self.action_dim = 3  # Hold, Buy, Sell
        
        # Policy network (linear) - ensure correct dimensions
        self.policy_weights = np.random.randn(self.state_dim, self.action_dim) * 0.1
        self.policy_bias = np.zeros(self.action_dim)
        
        # Value network (linear) - ensure correct dimensions
        self.value_weights = np.random.randn(self.state_dim) * 0.1
        self.value_bias = 0.0
        
        # Verify dimensions
        assert self.policy_weights.shape == (self.state_dim, self.action_dim), \
            f"Policy weights shape mismatch: {self.policy_weights.shape} != ({self.state_dim}, {self.action_dim})"
        assert self.value_weights.shape == (self.state_dim,), \
            f"Value weights shape mismatch: {self.value_weights.shape} != ({self.state_dim},)"
        
        # PPO hyperparameters
        self.gamma = 0.99  # Discount factor
        self.gae_lambda = 0.95  # GAE lambda
        self.clip_epsilon = 0.2  # PPO clip parameter
        self.value_coef = 0.5  # Value loss coefficient
        self.entropy_coef = 0.01  # Entropy bonus
        
        # Training buffers
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        self.episode_values = []
        self.episode_dones = []
        
        # Performance tracking
        self.episode_rewards_history = []
        self.episode_lengths = []
        self.metrics_history = []
        
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability"""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / (np.sum(exp_logits) + 1e-8)
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> tuple:
        """
        Select action using current policy.
        
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: Estimated value
        """
        # Policy forward pass
        logits = state @ self.policy_weights + self.policy_bias
        action_probs = self._softmax(logits)
        
        # Value forward pass
        value = np.dot(state, self.value_weights) + self.value_bias
        
        if deterministic:
            action = np.argmax(action_probs)
        else:
            action = np.random.choice(self.action_dim, p=action_probs)
        
        log_prob = np.log(action_probs[action] + 1e-8)
        
        return action, log_prob, value
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> np.ndarray:
        """Compute Generalized Advantage Estimation"""
        advantages = np.zeros(len(rewards))
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
            
            advantages[t] = last_gae
        
        returns = advantages + values[:-1] if len(values) > len(rewards) else advantages + np.array(values)
        return advantages, returns
    
    def update(self, states: np.ndarray, actions: np.ndarray, old_log_probs: np.ndarray,
               advantages: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """
        Update policy and value networks using PPO.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            old_log_probs: Old log probabilities
            advantages: Computed advantages
            returns: Computed returns
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute new log probs and values
        logits = states @ self.policy_weights + self.policy_bias
        action_probs = np.array([self._softmax(l) for l in logits])
        new_log_probs = np.log(action_probs[np.arange(len(actions)), actions] + 1e-8)
        values = (states @ self.value_weights + self.value_bias).flatten()
        
        # Compute ratios
        ratios = np.exp(new_log_probs - old_log_probs)
        
        # PPO clipped objective
        surr1 = ratios * advantages
        surr2 = np.clip(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -np.mean(np.minimum(surr1, surr2))
        
        # Value loss
        value_loss = 0.5 * np.mean((values - returns) ** 2)
        
        # Entropy bonus
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-8), axis=1).mean()
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Compute gradients (simplified REINFORCE-style)
        # Policy gradient: ∇log π(a|s) * advantage
        policy_grad = np.zeros_like(self.policy_weights)
        for i in range(len(states)):
            state = states[i]
            action = actions[i]
            advantage = advantages[i]
            
            # Compute policy probabilities
            logits = state @ self.policy_weights + self.policy_bias
            probs = self._softmax(logits)
            
            # Policy gradient: (∇log π(a|s)) * advantage
            for j in range(self.state_dim):
                for k in range(self.action_dim):
                    if k == action:
                        grad = state[j] * (1 - probs[k])
                    else:
                        grad = -state[j] * probs[k]
                    policy_grad[j, k] += grad * advantage
        
        policy_grad /= len(states)
        
        # Value gradient: ∇V(s) * (V(s) - return)
        value_grad = states.T @ (values - returns) / len(states)
        
        # Update weights
        self.policy_weights -= self.learning_rate * policy_grad
        self.value_weights -= self.learning_rate * value_grad
        
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "total_loss": total_loss
        }
    
    def train_episode(self) -> Dict[str, Any]:
        """Train for one episode"""
        state = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        # Collect episode data
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []
        
        done = False
        while not done:
            action, log_prob, value = self.select_action(state)
            next_state, reward, done, info = self.env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Compute advantages and returns
        values.append(self.select_action(state, deterministic=True)[2])  # Final value
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # Update policy
        states_array = np.array(states)
        actions_array = np.array(actions)
        old_log_probs_array = np.array(log_probs)
        
        update_info = self.update(states_array, actions_array, old_log_probs_array, advantages, returns)
        
        # Get final metrics
        metrics = self.env.get_metrics()
        
        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "metrics": metrics,
            **update_info
        }
    
    def train(self, max_episodes: int = 10000, target_sharpe: float = 1.5) -> Dict[str, Any]:
        """
        Train agent for multiple episodes.
        
        Returns:
            Dictionary with training results
        """
        results = {
            "episodes": [],
            "rewards": [],
            "sharpe_ratios": [],
            "returns": [],
            "episodes_to_target": None
        }
        
        for episode in range(max_episodes):
            episode_result = self.train_episode()
            
            sharpe = episode_result["metrics"]["sharpe_ratio"]
            total_return = episode_result["metrics"]["total_return"]
            
            results["episodes"].append(episode + 1)
            results["rewards"].append(episode_result["episode_reward"])
            results["sharpe_ratios"].append(sharpe)
            results["returns"].append(total_return)
            
            # Check if target reached
            if results["episodes_to_target"] is None and sharpe >= target_sharpe:
                results["episodes_to_target"] = episode + 1
            
            if (episode + 1) % 50 == 0:  # More frequent updates for shorter test
                recent_sharpe = np.mean(results["sharpe_ratios"][-50:]) if len(results["sharpe_ratios"]) >= 50 else results["sharpe_ratios"][-1]
                print(f"PPO Episode {episode + 1}: Recent Sharpe = {recent_sharpe:.3f}")
        
        return results
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """
        Get current weights for warm start.
        
        Returns:
            Dictionary with policy_weights, policy_bias, value_weights, value_bias
        """
        return {
            "policy_weights": self.policy_weights.copy(),
            "policy_bias": self.policy_bias.copy(),
            "value_weights": self.value_weights.copy(),
            "value_bias": self.value_bias,
            "add_noise": False,  # Don't add noise when transferring from PPO
            "noise_scale": 0.0
        }

