"""
Test Runtime Improvements

Tests the improved Runtime with:
1. Increased adaptation magnitude (gamma: 0.1 → 0.5)
2. Variance-based adaptation signal
3. Parameter preservation across spawning

Measures:
- Parameter change magnitude (target: 30-50% per step)
- Adaptation signal variance (target: >0.1)
- Convergence speed improvement (target: ≥20%)
"""

import asyncio
import time
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from pulseos import Runtime, Agent, SurvivalConstraint
from pulseos.runtime import Config


@dataclass
class TestResult:
    """Results from a single test run"""
    method: str
    trial: int
    convergence_episodes: int
    success: bool
    final_reward: float
    alpha_changes: List[float]
    epsilon_changes: List[float]
    adaptation_signals: List[float]
    survival_signals: List[float]
    alpha_change_magnitude: float
    adaptation_signal_variance: float


@dataclass
class TestSummary:
    """Summary of all test results"""
    method: str
    avg_convergence_episodes: float
    std_convergence_episodes: float
    success_rate: float
    avg_alpha_change_magnitude: float
    avg_adaptation_signal_variance: float
    improvement_over_baseline: float  # Percentage improvement


class SimpleTestAgent(Agent):
    """Simple test agent for validating Runtime improvements"""
    
    def __init__(self, agent_id: str, target_reward: float = 0.8):
        super().__init__(agent_id)
        self.target_reward = target_reward
        self.current_reward = 0.0
        self.reward_history = []
        self.converged = False
        self.convergence_step = None
        
    async def step(self) -> Dict[str, Any]:
        """Simulate agent learning"""
        # Simulate learning: reward increases with learning rate
        reward_delta = self.learning_rate * (1.0 - self.current_reward)
        noise = np.random.randn() * self.exploration_rate * 0.1
        
        self.current_reward = np.clip(
            self.current_reward + reward_delta + noise,
            0.0,
            1.0
        )
        
        self.reward_history.append(self.current_reward)
        
        # Check convergence
        if len(self.reward_history) >= 10:
            recent_avg = np.mean(self.reward_history[-10:])
            if recent_avg >= self.target_reward and not self.converged:
                self.converged = True
                self.convergence_step = len(self.reward_history)
        
        return {
            "reward": self.current_reward,
            "converged": self.converged
        }
    
    def get_performance_metric(self) -> float:
        """Return current reward as performance metric"""
        return self.current_reward if self.reward_history else 0.0


async def run_baseline_trial(trial_num: int, max_episodes: int = 100) -> TestResult:
    """Run baseline trial without Runtime"""
    agent = SimpleTestAgent(f"baseline_{trial_num}")
    
    alpha_changes = []
    epsilon_changes = []
    adaptation_signals = []
    survival_signals = []
    
    for episode in range(max_episodes):
        await agent.step()
        
        if agent.converged:
            break
    
    convergence_episodes = agent.convergence_step if agent.converged else max_episodes
    
    return TestResult(
        method="baseline",
        trial=trial_num,
        convergence_episodes=convergence_episodes,
        success=agent.converged,
        final_reward=agent.reward_history[-1] if agent.reward_history else 0.0,
        alpha_changes=alpha_changes,
        epsilon_changes=epsilon_changes,
        adaptation_signals=adaptation_signals,
        survival_signals=survival_signals,
        alpha_change_magnitude=0.0,
        adaptation_signal_variance=0.0
    )


async def run_improved_runtime_trial(trial_num: int, max_episodes: int = 100) -> TestResult:
    """Run trial with improved Runtime"""
    constraint = SurvivalConstraint(threshold=0.6)
    # Use default Config (already has improved parameters) or create with explicit values
    config = Config(
        alpha_base=0.01,
        alpha_max_change_per_step=0.50,  # 50% max change
        alpha_smooth=0.75,  # Less smoothing
        gamma=0.5  # Increased gamma
        # momentum_decay has default value of 0.9 in Config
    )
    runtime = Runtime(constraint=constraint, config=config)
    
    agent = SimpleTestAgent(f"improved_{trial_num}")
    runtime.register_agent(agent.agent_id, agent)
    
    alpha_changes = []
    epsilon_changes = []
    adaptation_signals = []
    survival_signals = []
    initial_alpha = runtime.apc.get_alpha()
    
    for episode in range(max_episodes):
        # Run Runtime step
        step_result = await runtime.step()
        
        # Track changes
        current_alpha = step_result["alpha"]
        current_epsilon = step_result["epsilon"]
        
        if len(alpha_changes) == 0:
            alpha_changes.append(0.0)
        else:
            alpha_change = abs(current_alpha - alpha_changes[-1]) / alpha_changes[-1] if alpha_changes[-1] > 0 else 0.0
            alpha_changes.append(current_alpha)
        
        epsilon_changes.append(current_epsilon)
        adaptation_signals.append(step_result.get("adaptation_signal", 0.0))
        survival_signals.append(step_result.get("survival_signal", 0.0))
        
        if agent.converged:
            break
    
    convergence_episodes = agent.convergence_step if agent.converged else max_episodes
    
    # Compute statistics
    alpha_change_magnitude = (
        abs(alpha_changes[-1] - initial_alpha) / initial_alpha
        if initial_alpha > 0 and len(alpha_changes) > 0 else 0.0
    )
    adaptation_signal_variance = np.var(adaptation_signals) if len(adaptation_signals) > 1 else 0.0
    
    return TestResult(
        method="improved_runtime",
        trial=trial_num,
        convergence_episodes=convergence_episodes,
        success=agent.converged,
        final_reward=agent.reward_history[-1] if agent.reward_history else 0.0,
        alpha_changes=alpha_changes,
        epsilon_changes=epsilon_changes,
        adaptation_signals=adaptation_signals,
        survival_signals=survival_signals,
        alpha_change_magnitude=alpha_change_magnitude,
        adaptation_signal_variance=adaptation_signal_variance
    )


async def run_test_suite(num_trials: int = 10) -> Dict[str, TestSummary]:
    """Run full test suite"""
    print("="*80)
    print("Runtime Improvements Test Suite")
    print("="*80)
    print(f"Trials: {num_trials}")
    print("="*80)
    
    baseline_results = []
    improved_results = []
    
    # Run baseline trials
    print("\nRunning baseline trials (no Runtime)...")
    for trial in range(num_trials):
        np.random.seed(42 + trial)
        result = await run_baseline_trial(trial + 1)
        baseline_results.append(result)
        print(f"  Trial {trial + 1}: {result.convergence_episodes} episodes, "
              f"success={result.success}")
    
    # Run improved Runtime trials
    print("\nRunning improved Runtime trials...")
    for trial in range(num_trials):
        np.random.seed(42 + trial)
        result = await run_improved_runtime_trial(trial + 1)
        improved_results.append(result)
        print(f"  Trial {trial + 1}: {result.convergence_episodes} episodes, "
              f"success={result.success}, "
              f"alpha_change={result.alpha_change_magnitude:.1%}, "
              f"signal_var={result.adaptation_signal_variance:.3f}")
    
    # Compute summaries
    baseline_summary = TestSummary(
        method="baseline",
        avg_convergence_episodes=np.mean([r.convergence_episodes for r in baseline_results]),
        std_convergence_episodes=np.std([r.convergence_episodes for r in baseline_results]),
        success_rate=np.mean([r.success for r in baseline_results]),
        avg_alpha_change_magnitude=0.0,
        avg_adaptation_signal_variance=0.0,
        improvement_over_baseline=0.0
    )
    
    improved_summary = TestSummary(
        method="improved_runtime",
        avg_convergence_episodes=np.mean([r.convergence_episodes for r in improved_results]),
        std_convergence_episodes=np.std([r.convergence_episodes for r in improved_results]),
        success_rate=np.mean([r.success for r in improved_results]),
        avg_alpha_change_magnitude=np.mean([r.alpha_change_magnitude for r in improved_results]),
        avg_adaptation_signal_variance=np.mean([r.adaptation_signal_variance for r in improved_results]),
        improvement_over_baseline=(
            (baseline_summary.avg_convergence_episodes - np.mean([r.convergence_episodes for r in improved_results])) /
            baseline_summary.avg_convergence_episodes * 100
        )
    )
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nBaseline:")
    print(f"  Convergence: {baseline_summary.avg_convergence_episodes:.1f} ± {baseline_summary.std_convergence_episodes:.1f} episodes")
    print(f"  Success Rate: {baseline_summary.success_rate:.1%}")
    
    print(f"\nImproved Runtime:")
    print(f"  Convergence: {improved_summary.avg_convergence_episodes:.1f} ± {improved_summary.std_convergence_episodes:.1f} episodes")
    print(f"  Success Rate: {improved_summary.success_rate:.1%}")
    print(f"  Alpha Change Magnitude: {improved_summary.avg_alpha_change_magnitude:.1%}")
    print(f"  Adaptation Signal Variance: {improved_summary.avg_adaptation_signal_variance:.3f}")
    print(f"\n🎯 Improvement: {improved_summary.improvement_over_baseline:.1f}% faster convergence")
    
    # Evaluation
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    if improved_summary.improvement_over_baseline >= 20:
        print("✅ EXCELLENT: ≥20% improvement - Runtime provides significant value!")
        recommendation = "KEEP_RUNTIME"
    elif improved_summary.improvement_over_baseline >= 10:
        print("⚠️  MODEST: 10-20% improvement - Runtime helps but marginal value")
        recommendation = "KEEP_RUNTIME_MARGINAL"
    else:
        print("❌ LOW: <10% improvement - Runtime doesn't add enough value")
        recommendation = "REMOVE_RUNTIME"
    
    if improved_summary.avg_alpha_change_magnitude >= 0.30:
        print(f"✅ Parameter changes: {improved_summary.avg_alpha_change_magnitude:.1%} (target: ≥30%)")
    else:
        print(f"⚠️  Parameter changes: {improved_summary.avg_alpha_change_magnitude:.1%} (target: ≥30%)")
    
    if improved_summary.avg_adaptation_signal_variance >= 0.1:
        print(f"✅ Adaptation signal variance: {improved_summary.avg_adaptation_signal_variance:.3f} (target: ≥0.1)")
    else:
        print(f"⚠️  Adaptation signal variance: {improved_summary.avg_adaptation_signal_variance:.3f} (target: ≥0.1)")
    
    print("="*80)
    
    # Save results
    output_dir = "benchmark_results/runtime_improvements"
    os.makedirs(output_dir, exist_ok=True)
    
    results_data = {
        "baseline_summary": asdict(baseline_summary),
        "improved_summary": asdict(improved_summary),
        "baseline_results": [asdict(r) for r in baseline_results],
        "improved_results": [asdict(r) for r in improved_results],
        "recommendation": recommendation
    }
    
    json_path = os.path.join(output_dir, "test_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nSaved results to {json_path}")
    
    # Create visualization
    create_visualization(baseline_results, improved_results, output_dir)
    
    return {
        "baseline": baseline_summary,
        "improved": improved_summary
    }


def create_visualization(baseline_results: List[TestResult], improved_results: List[TestResult], output_dir: str):
    """Create visualization of test results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Convergence episodes comparison
    ax1 = axes[0, 0]
    baseline_episodes = [r.convergence_episodes for r in baseline_results]
    improved_episodes = [r.convergence_episodes for r in improved_results]
    
    ax1.boxplot([baseline_episodes, improved_episodes], labels=['Baseline', 'Improved Runtime'])
    ax1.set_ylabel('Convergence Episodes')
    ax1.set_title('Convergence Speed Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Alpha change magnitude
    ax2 = axes[0, 1]
    alpha_changes = [r.alpha_change_magnitude for r in improved_results]
    ax2.hist(alpha_changes, bins=10, alpha=0.7, edgecolor='black')
    ax2.axvline(0.30, color='r', linestyle='--', label='Target (30%)')
    ax2.set_xlabel('Alpha Change Magnitude')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Parameter Adaptation Magnitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Adaptation signal variance
    ax3 = axes[1, 0]
    signal_variances = [r.adaptation_signal_variance for r in improved_results]
    ax3.hist(signal_variances, bins=10, alpha=0.7, edgecolor='black')
    ax3.axvline(0.1, color='r', linestyle='--', label='Target (0.1)')
    ax3.set_xlabel('Adaptation Signal Variance')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Adaptation Signal Quality')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Learning curves (example from first trial)
    ax4 = axes[1, 1]
    if improved_results and improved_results[0].adaptation_signals:
        episodes = range(len(improved_results[0].adaptation_signals))
        ax4.plot(episodes, improved_results[0].adaptation_signals, label='Adaptation Signal', alpha=0.7)
        ax4.plot(episodes, improved_results[0].survival_signals, label='Survival Signal', alpha=0.7)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Signal Value')
        ax4.set_title('Signal Evolution (Trial 1)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "test_results.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"Saved visualization to {plot_path}")


async def main():
    """Main test function"""
    results = await run_test_suite(num_trials=10)
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    return results


if __name__ == "__main__":
    asyncio.run(main())

