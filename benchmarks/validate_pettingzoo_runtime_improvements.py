"""
Real PettingZoo Benchmark with Runtime Improvements

Tests improved Runtime with actual PettingZoo multi-agent environments.
Compares Population-Only vs Population+Runtime (Improved).

Original baseline results:
- Population-Only: 100% success, 24.3 episodes
- Population+Runtime (old): 100% success, 22.3 episodes (+8%)

Target with improvements:
- Population+Runtime (improved): 100% success, <20 episodes (≥20% improvement)
"""

import asyncio
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Try to import pettingzoo
try:
    from pettingzoo.mpe import simple_tag_v3, simple_adversary_v3, simple_spread_v3
    PETTINGZOO_AVAILABLE = True
except ImportError:
    PETTINGZOO_AVAILABLE = False
    print("Warning: pettingzoo not available. Install with: pip install pettingzoo[mpe]")

from pulseos.runtime import Runtime, Config
from pulseos import Agent, SurvivalConstraint


@dataclass
class PettingZooTrialResult:
    """Results from a single PettingZoo trial"""
    trial: int
    method: str  # "population_only" or "population_runtime_improved"
    environment: str
    convergence_episodes: int
    success: bool
    final_coverage: float
    alpha_changes: List[float]
    adaptation_signals: List[float]
    survival_signals: List[float]
    alpha_change_magnitude: float
    adaptation_signal_variance: float
    learning_curve: List[float]


@dataclass
class PettingZooBenchmarkResult:
    """Results from PettingZoo benchmark"""
    environment: str
    population_only_results: List[PettingZooTrialResult]
    population_runtime_results: List[PettingZooTrialResult]
    avg_convergence_improvement: float  # Percentage improvement
    avg_alpha_change_magnitude: float
    avg_adaptation_signal_variance: float
    success_rate_population_only: float
    success_rate_population_runtime: float


class PettingZooREINFORCEAgent(Agent):
    """REINFORCE agent for PettingZoo environments"""
    
    def __init__(self, agent_id: str, env_name: str, observation_space_size: int, action_space_size: int):
        super().__init__(agent_id)
        self.env_name = env_name
        self.observation_space_size = observation_space_size
        self.action_space_size = action_space_size
        
        # Policy network (simple linear policy)
        scale = np.sqrt(2.0 / (observation_space_size + action_space_size))
        self.policy_weights = np.random.randn(observation_space_size, action_space_size) * scale
        
        # Episode tracking
        self.episode_observations = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        
        # Performance tracking
        self.coverage = 0.0  # Coverage/reward metric
        self.coverage_history = []
        self.episode_rewards_history = []
        self.converged = False
        self.convergence_step = None
        self.target_coverage = 0.6
        
        # REINFORCE parameters
        self.gamma = 0.99
        self.baseline = 0.0
        
    async def step(self) -> Dict[str, Any]:
        """Execute one step (called by environment)"""
        # This will be called by the environment during episode execution
        # For now, return current state
        return {
            "coverage": self.coverage,
            "converged": self.converged
        }
    
    def select_action(self, observation: np.ndarray) -> int:
        """Select action using policy"""
        # Ensure observation is 1D and get its size
        obs_flat = observation.flatten() if len(observation.shape) > 1 else observation
        obs_size = obs_flat.shape[0]
        
        # Resize policy weights if observation size doesn't match
        if obs_size != self.observation_space_size:
            # Reinitialize policy weights with correct size
            scale = np.sqrt(2.0 / (obs_size + self.action_space_size))
            self.policy_weights = np.random.randn(obs_size, self.action_space_size) * scale
            self.observation_space_size = obs_size
        
        # Double-check shapes match before matmul
        if obs_flat.shape[0] != self.policy_weights.shape[0]:
            # Emergency fix: resize policy weights to match observation
            scale = np.sqrt(2.0 / (obs_flat.shape[0] + self.action_space_size))
            self.policy_weights = np.random.randn(obs_flat.shape[0], self.action_space_size) * scale
            self.observation_space_size = obs_flat.shape[0]
        
        # Use the flattened observation for matmul
        logits = obs_flat @ self.policy_weights
        action_probs = self._softmax(logits.flatten())
        return np.random.choice(len(action_probs), p=action_probs)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax with numerical stability"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def record_step(self, observation: np.ndarray, action: int, reward: float):
        """Record step data for REINFORCE"""
        logits = observation @ self.policy_weights
        action_probs = self._softmax(logits.flatten())
        log_prob = np.log(action_probs[action] + 1e-8)
        
        self.episode_observations.append(observation.copy())
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_log_probs.append(log_prob)
    
    def update_policy(self):
        """Update policy using REINFORCE"""
        if len(self.episode_rewards) == 0:
            return
        
        # Compute returns
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = np.array(returns)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        # Update baseline
        self.baseline = 0.9 * self.baseline + 0.1 * np.mean(returns)
        
        # Compute policy gradient
        policy_gradient = np.zeros_like(self.policy_weights)
        
        for i, (obs, action, log_prob, G) in enumerate(zip(
            self.episode_observations,
            self.episode_actions,
            self.episode_log_probs,
            returns
        )):
            advantage = G - self.baseline
            
            # Compute gradient
            logits = obs @ self.policy_weights
            action_probs = self._softmax(logits.flatten())
            
            # Policy gradient: ∇log π(a|s) * advantage
            grad_log_prob = np.zeros(self.action_space_size)
            grad_log_prob[action] = 1.0 - action_probs[action]
            
            policy_gradient += np.outer(obs, grad_log_prob) * advantage
        
        # Update weights with adaptive learning rate
        if len(self.episode_rewards) > 0:
            policy_gradient /= len(self.episode_rewards)
            self.policy_weights += self.learning_rate * policy_gradient
        
        # Update coverage (episode reward)
        episode_reward = sum(self.episode_rewards)
        self.coverage = episode_reward / 100.0  # Normalize to [0, 1]
        self.coverage_history.append(self.coverage)
        self.episode_rewards_history.append(episode_reward)
        
        # Check convergence
        if len(self.coverage_history) >= 10:
            recent_avg = np.mean(self.coverage_history[-10:])
            if recent_avg >= self.target_coverage and not self.converged:
                self.converged = True
                self.convergence_step = len(self.coverage_history)
        
        # Clear episode data
        self.episode_observations = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
    
    def get_performance_metric(self) -> float:
        """Return current coverage as performance metric"""
        return self.coverage if self.coverage_history else 0.0


class PettingZooPopulation:
    """Population management for PettingZoo with Runtime integration"""
    
    def __init__(
        self,
        env_name: str,
        num_agents: Optional[int] = None,  # PettingZoo has fixed agents, this is ignored
        use_runtime: bool = False,
        elimination_rate: float = 0.3,
        elimination_interval: int = 5
    ):
        self.env_name = env_name
        self.num_agents = num_agents
        self.use_runtime = use_runtime
        self.elimination_rate = elimination_rate
        self.elimination_interval = elimination_interval
        
        # Initialize environment
        if not PETTINGZOO_AVAILABLE:
            raise ImportError("PettingZoo not available")
        
        if env_name == "simple_tag_v3":
            self.env = simple_tag_v3.parallel_env(max_cycles=25, render_mode=None)
        elif env_name == "simple_adversary_v3":
            self.env = simple_adversary_v3.parallel_env(max_cycles=25, render_mode=None)
        elif env_name == "simple_spread_v3":
            self.env = simple_spread_v3.parallel_env(max_cycles=25, render_mode=None)
        else:
            raise ValueError(f"Unknown environment: {env_name}")
        
        observations, infos = self.env.reset()
        
        # PettingZoo environments have fixed agents - use them directly
        # Map each environment agent to a REINFORCE agent
        self.agents = {}
        env_agent_names = list(self.env.agents)
        
        # Create one REINFORCE agent per environment agent
        # Get observation/action spaces from actual observations after reset
        for env_agent_name in env_agent_names:
            if env_agent_name not in observations:
                continue
                
            # Get observation size from actual observation
            obs = observations[env_agent_name]
            obs_size = obs.shape[0] if hasattr(obs, 'shape') and len(obs.shape) > 0 else 1
            
            # Get action space
            try:
                action_space = self.env.action_space(env_agent_name)
            except:
                action_space = self.env.action_spaces[env_agent_name]
            
            action_size = action_space.n if hasattr(action_space, 'n') else 1
            
            # Use environment agent name as agent ID
            self.agents[env_agent_name] = PettingZooREINFORCEAgent(
                env_agent_name,
                env_name,
                obs_size,
                action_size
            )
        
        # Store actual number of agents (PettingZoo has fixed number)
        self.num_agents = len(self.agents)
        
        # Initialize Runtime if using it
        self.runtime = None
        if use_runtime:
            constraint = SurvivalConstraint(threshold=0.6)
            # Use improved Runtime configuration
            config = Config(
                alpha_base=0.01,
                alpha_max_change_per_step=0.50,  # Improved from 0.20 (50% max change)
                alpha_smooth=0.75,  # Improved from 0.9 (faster adaptation)
                gamma=0.5,  # Improved from 0.1 (stronger adaptation signal)
                momentum_decay=0.9  # New: momentum-based updates
            )
            self.runtime = Runtime(constraint=constraint, config=config)
            
            # Register all agents
            for agent_id, agent in self.agents.items():
                self.runtime.register_agent(agent_id, agent)
        
        self.episode_count = 0
    
    async def train_episode(self) -> Dict[str, Any]:
        """Train one episode"""
        self.episode_count += 1
        
        # Reset environment
        observations, infos = self.env.reset()
        
        # Run episode - agents map 1:1 with environment agents
        env_agent_names = list(self.env.agents)
        episode_rewards = {agent_id: 0.0 for agent_id in self.agents.keys()}
        done = False
        
        while not done:
            # Get actions from our agents (1:1 mapping with env agents)
            actions = {}
            for env_agent_name in env_agent_names:
                if env_agent_name in observations and env_agent_name in self.agents:
                    agent = self.agents[env_agent_name]
                    obs = observations[env_agent_name]
                    # Ensure we're using the right agent for this observation
                    # Double-check observation size matches agent's expected size
                    obs_flat = obs.flatten() if len(obs.shape) > 1 else obs
                    if obs_flat.shape[0] != agent.observation_space_size:
                        # Agent was initialized with wrong size, fix it
                        agent.observation_space_size = obs_flat.shape[0]
                        scale = np.sqrt(2.0 / (obs_flat.shape[0] + agent.action_space_size))
                        agent.policy_weights = np.random.randn(obs_flat.shape[0], agent.action_space_size) * scale
                    # Pass the flattened observation to select_action
                    action = agent.select_action(obs_flat)
                    actions[env_agent_name] = action
            
            # Step environment
            next_observations, rewards, terminations, truncations, infos = self.env.step(actions)
            
            # Record steps for REINFORCE
            for env_agent_name in env_agent_names:
                if env_agent_name in observations and env_agent_name in actions and env_agent_name in self.agents:
                    agent = self.agents[env_agent_name]
                    obs = observations[env_agent_name]
                    action = actions[env_agent_name]
                    reward = rewards.get(env_agent_name, 0.0)
                    
                    agent.record_step(obs, action, reward)
                    episode_rewards[agent.agent_id] += reward
            
            observations = next_observations
            done = all(terminations.values()) or all(truncations.values())
        
        # Update policies
        for agent in self.agents.values():
            agent.update_policy()
        
        # Run Runtime step if using Runtime
        if self.use_runtime and self.runtime:
            await self.runtime.step()
        
        # Check for elimination/spawning
        if self.episode_count % self.elimination_interval == 0:
            await self._eliminate_and_spawn()
        
        # Compute statistics
        coverages = [agent.coverage for agent in self.agents.values()]
        converged_count = sum(1 for agent in self.agents.values() if agent.converged)
        
        return {
            "episode": self.episode_count,
            "mean_coverage": np.mean(coverages),
            "converged_count": converged_count,
            "converged_rate": converged_count / len(self.agents) if self.agents else 0.0
        }
    
    async def _eliminate_and_spawn(self):
        """Eliminate worst performers and spawn from best (by resetting policies)"""
        # Sort agents by coverage
        agent_performances = [
            (agent_id, agent.coverage)
            for agent_id, agent in self.agents.items()
        ]
        agent_performances.sort(key=lambda x: x[1], reverse=True)
        
        # Get elites (top performers)
        num_elites = max(1, int(len(self.agents) * (1 - self.elimination_rate)))
        elites = agent_performances[:num_elites]
        worst = agent_performances[num_elites:]
        
        # "Eliminate and spawn": Reset worst agents' policies to elite agents' policies
        # This simulates population dynamics while keeping agent IDs fixed (required by PettingZoo)
        for worst_id, _ in worst:
            # Select a random elite as parent
            parent_id, _ = elites[np.random.randint(len(elites))]
            parent_agent = self.agents[parent_id]
            worst_agent = self.agents[worst_id]
            
            # Copy policy weights from parent with small mutation
            worst_agent.policy_weights = parent_agent.policy_weights.copy() + np.random.randn(*parent_agent.policy_weights.shape) * 0.01
            
            # Reset coverage and convergence for "spawned" agent
            worst_agent.coverage = 0.0
            worst_agent.coverage_history = []
            worst_agent.converged = False
            worst_agent.convergence_step = None
            
            # If using Runtime, preserve adapted parameters from parent
            if self.use_runtime and self.runtime:
                # Get parent's adapted parameters if available
                if parent_id in self.runtime.agent_adapted_params:
                    parent_params = self.runtime.agent_adapted_params[parent_id]
                    worst_agent.update_learning_rate(parent_params["alpha"])
                    worst_agent.update_exploration_rate(parent_params["epsilon"])
                    
                    # Update Runtime's tracking for this agent
                    self.runtime.agent_adapted_params[worst_id] = {
                        "alpha": parent_params["alpha"],
                        "epsilon": parent_params["epsilon"],
                        "timestamp": self.runtime.current_step
                    }
    
    def get_convergence_episode(self) -> Optional[int]:
        """Get episode when all agents converged"""
        converged_agents = [
            agent for agent in self.agents.values()
            if agent.converged and agent.convergence_step is not None
        ]
        
        if len(converged_agents) == len(self.agents):
            return max(agent.convergence_step for agent in converged_agents)
        return None
    
    def is_converged(self) -> bool:
        """Check if all agents converged"""
        return all(agent.converged for agent in self.agents.values())


async def run_population_only_trial(
    trial_num: int,
    env_name: str = "simple_tag_v3",
    max_episodes: int = 100
) -> PettingZooTrialResult:
    """Run trial with population dynamics only (no Runtime)"""
    population = PettingZooPopulation(
        env_name=env_name,
        num_agents=20,
        use_runtime=False
    )
    
    learning_curve = []
    
    for episode in range(max_episodes):
        result = await population.train_episode()
        learning_curve.append(result["mean_coverage"])
        
        if population.is_converged():
            break
    
    convergence_episode = population.get_convergence_episode() or max_episodes
    success = population.is_converged()
    
    coverages = [agent.coverage for agent in population.agents.values()]
    final_coverage = np.mean(coverages)
    
    return PettingZooTrialResult(
        trial=trial_num,
        method="population_only",
        environment=env_name,
        convergence_episodes=convergence_episode,
        success=success,
        final_coverage=final_coverage,
        alpha_changes=[],
        adaptation_signals=[],
        survival_signals=[],
        alpha_change_magnitude=0.0,
        adaptation_signal_variance=0.0,
        learning_curve=learning_curve
    )


async def run_population_runtime_trial(
    trial_num: int,
    env_name: str = "simple_tag_v3",
    max_episodes: int = 100
) -> PettingZooTrialResult:
    """Run trial with population dynamics + improved Runtime"""
    population = PettingZooPopulation(
        env_name=env_name,
        num_agents=20,
        use_runtime=True
    )
    
    alpha_changes = []
    adaptation_signals = []
    survival_signals = []
    learning_curve = []
    initial_alpha = population.runtime.apc.get_alpha() if population.runtime else 0.01
    
    for episode in range(max_episodes):
        result = await population.train_episode()
        learning_curve.append(result["mean_coverage"])
        
        # Track Runtime metrics
        if population.runtime:
            stats = population.runtime.get_statistics()
            current_alpha = stats.get("current_alpha", initial_alpha)
            alpha_changes.append(current_alpha)
            
            # Get adaptation signal from performance history
            if population.runtime.performance_history:
                recent = list(population.runtime.performance_history)[-1]
                adaptation_signal = recent.get("adaptation_signal", 0.0)
                survival_signal = recent.get("survival_signal", 0.0)
                adaptation_signals.append(adaptation_signal)
                survival_signals.append(survival_signal)
        
        if population.is_converged():
            break
    
    convergence_episode = population.get_convergence_episode() or max_episodes
    success = population.is_converged()
    
    coverages = [agent.coverage for agent in population.agents.values()]
    final_coverage = np.mean(coverages)
    
    # Compute statistics
    alpha_change_magnitude = (
        abs(alpha_changes[-1] - initial_alpha) / initial_alpha
        if len(alpha_changes) > 0 and initial_alpha > 0 else 0.0
    )
    adaptation_signal_variance = np.var(adaptation_signals) if len(adaptation_signals) > 1 else 0.0
    
    return PettingZooTrialResult(
        trial=trial_num,
        method="population_runtime_improved",
        environment=env_name,
        convergence_episodes=convergence_episode,
        success=success,
        final_coverage=final_coverage,
        alpha_changes=alpha_changes,
        adaptation_signals=adaptation_signals,
        survival_signals=survival_signals,
        alpha_change_magnitude=alpha_change_magnitude,
        adaptation_signal_variance=adaptation_signal_variance,
        learning_curve=learning_curve
    )


async def run_pettingzoo_benchmark(
    env_name: str = "simple_tag_v3",
    num_trials: int = 10,
    max_episodes: int = 100
) -> PettingZooBenchmarkResult:
    """Run PettingZoo benchmark comparing Population-Only vs Population+Runtime"""
    
    print(f"\n{'='*80}")
    print(f"PettingZoo Runtime Improvements Benchmark: {env_name}")
    print(f"{'='*80}")
    print(f"Trials: {num_trials}")
    print(f"Max episodes: {max_episodes}")
    print(f"{'='*80}")
    
    population_only_results = []
    population_runtime_results = []
    
    # Run Population-Only trials
    print(f"\nRunning Population-Only trials...")
    for trial in range(num_trials):
        np.random.seed(42 + trial)
        result = await run_population_only_trial(trial + 1, env_name, max_episodes)
        population_only_results.append(result)
        print(f"  Trial {trial + 1}: {result.convergence_episodes} episodes, "
              f"success={result.success}, coverage={result.final_coverage:.3f}")
    
    # Run Population+Runtime trials
    print(f"\nRunning Population+Runtime (Improved) trials...")
    for trial in range(num_trials):
        np.random.seed(42 + trial)
        result = await run_population_runtime_trial(trial + 1, env_name, max_episodes)
        population_runtime_results.append(result)
        print(f"  Trial {trial + 1}: {result.convergence_episodes} episodes, "
              f"success={result.success}, coverage={result.final_coverage:.3f}, "
              f"alpha_change={result.alpha_change_magnitude:.1%}, "
              f"signal_var={result.adaptation_signal_variance:.3f}")
    
    # Compute statistics
    pop_only_episodes = [r.convergence_episodes for r in population_only_results]
    pop_runtime_episodes = [r.convergence_episodes for r in population_runtime_results]
    
    avg_pop_only = np.mean(pop_only_episodes)
    avg_pop_runtime = np.mean(pop_runtime_episodes)
    
    improvement = ((avg_pop_only - avg_pop_runtime) / avg_pop_only * 100) if avg_pop_only > 0 else 0.0
    
    avg_alpha_change = np.mean([r.alpha_change_magnitude for r in population_runtime_results])
    avg_signal_variance = np.mean([r.adaptation_signal_variance for r in population_runtime_results])
    
    success_rate_pop = np.mean([r.success for r in population_only_results])
    success_rate_runtime = np.mean([r.success for r in population_runtime_results])
    
    result = PettingZooBenchmarkResult(
        environment=env_name,
        population_only_results=population_only_results,
        population_runtime_results=population_runtime_results,
        avg_convergence_improvement=improvement,
        avg_alpha_change_magnitude=avg_alpha_change,
        avg_adaptation_signal_variance=avg_signal_variance,
        success_rate_population_only=success_rate_pop,
        success_rate_population_runtime=success_rate_runtime
    )
    
    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"\nPopulation-Only:")
    print(f"  Convergence: {avg_pop_only:.1f} ± {np.std(pop_only_episodes):.1f} episodes")
    print(f"  Success Rate: {success_rate_pop:.1%}")
    
    print(f"\nPopulation+Runtime (Improved):")
    print(f"  Convergence: {avg_pop_runtime:.1f} ± {np.std(pop_runtime_episodes):.1f} episodes")
    print(f"  Success Rate: {success_rate_runtime:.1%}")
    print(f"  Alpha Change Magnitude: {avg_alpha_change:.1%}")
    print(f"  Adaptation Signal Variance: {avg_signal_variance:.3f}")
    print(f"\n🎯 Improvement: {improvement:.1f}% faster convergence")
    
    # Evaluation
    print(f"\n{'='*80}")
    print("EVALUATION")
    print(f"{'='*80}")
    
    if improvement >= 20:
        print("✅ EXCELLENT: ≥20% improvement - Runtime provides significant value!")
        recommendation = "KEEP_RUNTIME"
    elif improvement >= 10:
        print("⚠️  MODEST: 10-20% improvement - Runtime helps but marginal value")
        recommendation = "KEEP_RUNTIME_MARGINAL"
    else:
        print("❌ LOW: <10% improvement - Runtime doesn't add enough value")
        recommendation = "REMOVE_RUNTIME"
    
    if avg_alpha_change >= 0.30:
        print(f"✅ Parameter changes: {avg_alpha_change:.1%} (target: ≥30%)")
    else:
        print(f"⚠️  Parameter changes: {avg_alpha_change:.1%} (target: ≥30%)")
    
    if avg_signal_variance >= 0.1:
        print(f"✅ Adaptation signal variance: {avg_signal_variance:.3f} (target: ≥0.1)")
    else:
        print(f"⚠️  Adaptation signal variance: {avg_signal_variance:.3f} (target: ≥0.1)")
    
    print(f"{'='*80}")
    
    # Save results
    output_dir = "benchmark_results/pettingzoo_runtime_improvements"
    os.makedirs(output_dir, exist_ok=True)
    
    results_data = {
        "environment": env_name,
        "avg_convergence_improvement": improvement,
        "avg_alpha_change_magnitude": avg_alpha_change,
        "avg_adaptation_signal_variance": avg_signal_variance,
        "success_rate_population_only": success_rate_pop,
        "success_rate_population_runtime": success_rate_runtime,
        "recommendation": recommendation,
        "population_only_results": [asdict(r) for r in population_only_results],
        "population_runtime_results": [asdict(r) for r in population_runtime_results]
    }
    
    json_path = os.path.join(output_dir, f"{env_name}_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nSaved results to {json_path}")
    
    # Create visualization
    create_visualization(result, output_dir)
    
    return result


def create_visualization(result: PettingZooBenchmarkResult, output_dir: str):
    """Create visualization of benchmark results"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Convergence episodes comparison
    ax1 = axes[0, 0]
    pop_only_episodes = [r.convergence_episodes for r in result.population_only_results]
    pop_runtime_episodes = [r.convergence_episodes for r in result.population_runtime_results]
    
    ax1.boxplot([pop_only_episodes, pop_runtime_episodes], 
                tick_labels=['Population-Only', 'Population+Runtime'])
    ax1.set_ylabel('Convergence Episodes')
    ax1.set_title(f'Convergence Speed Comparison\n({result.avg_convergence_improvement:.1f}% improvement)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Learning curves (average)
    ax2 = axes[0, 1]
    max_len = max(
        max(len(r.learning_curve) for r in result.population_only_results),
        max(len(r.learning_curve) for r in result.population_runtime_results)
    )
    
    # Average learning curves
    pop_only_curves = []
    pop_runtime_curves = []
    for i in range(max_len):
        pop_only_values = [r.learning_curve[i] for r in result.population_only_results if i < len(r.learning_curve)]
        pop_runtime_values = [r.learning_curve[i] for r in result.population_runtime_results if i < len(r.learning_curve)]
        if pop_only_values:
            pop_only_curves.append(np.mean(pop_only_values))
        if pop_runtime_values:
            pop_runtime_curves.append(np.mean(pop_runtime_values))
    
    ax2.plot(pop_only_curves, label='Population-Only', linewidth=2)
    ax2.plot(pop_runtime_curves, label='Population+Runtime', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Mean Coverage')
    ax2.set_title('Learning Curves (Average)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Alpha change magnitude
    ax3 = axes[1, 0]
    alpha_changes = [r.alpha_change_magnitude for r in result.population_runtime_results]
    ax3.hist(alpha_changes, bins=10, alpha=0.7, edgecolor='black')
    ax3.axvline(0.30, color='r', linestyle='--', label='Target (30%)')
    ax3.set_xlabel('Alpha Change Magnitude')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Parameter Adaptation Magnitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Adaptation signal variance
    ax4 = axes[1, 1]
    signal_variances = [r.adaptation_signal_variance for r in result.population_runtime_results]
    ax4.hist(signal_variances, bins=10, alpha=0.7, edgecolor='black')
    ax4.axvline(0.1, color='r', linestyle='--', label='Target (0.1)')
    ax4.set_xlabel('Adaptation Signal Variance')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Adaptation Signal Quality')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{result.environment}_results.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"Saved visualization to {plot_path}")


async def main():
    """Run PettingZoo benchmarks"""
    if not PETTINGZOO_AVAILABLE:
        print("PettingZoo not available. Install with: pip install pettingzoo[mpe]")
        return
    
    print("="*80)
    print("PettingZoo Runtime Improvements Validation")
    print("="*80)
    print("\nTesting improved Runtime with real PettingZoo multi-agent environments.")
    print("Target: ≥20% faster convergence (24.3 → <20 episodes)")
    print("="*80)
    
    # Test with simple_tag_v3 (the environment from original results)
    result = await run_pettingzoo_benchmark(
        env_name="simple_tag_v3",
        num_trials=10,
        max_episodes=100
    )
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    
    return result


if __name__ == "__main__":
    asyncio.run(main())

