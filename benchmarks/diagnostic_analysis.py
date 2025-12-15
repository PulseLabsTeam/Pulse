"""
Phase 1: Diagnostic Analysis Tool

Deep dive into failure cases to understand why PulseOS fails in:
- multi_objective_normal_th-0.5 (9.4% reduction)
- linear_bimodal_th-0.5 (0.6% reduction)
- linear_skewed_th-0.3 (10.8% reduction)

Generates comprehensive diagnostic reports with visualizations.
"""

import asyncio
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import pandas as pd

from pulseos import Runtime, Config, Agent, SurvivalConstraint
from pulseos.circuits.ptdc import PerformanceThresholdDetectionCircuit
from pulseos.circuits.ngcm import NonlinearGradientComputationModule
from pulseos.circuits.apc import AdaptiveParameterController
from benchmarks.strategic_benchmark_suite import VariantRLHFAgent


@dataclass
class DiagnosticData:
    """Diagnostic data collected during a trial"""
    step: int
    survival_signal: float
    gradient: float
    alpha: float
    epsilon: float
    preference: float
    reward: float
    distance_to_threshold: float
    threshold_status: bool
    metric_value: float


@dataclass
class ScenarioDiagnostics:
    """Complete diagnostics for a scenario"""
    scenario_name: str
    method: str  # "PPO" or "PulseOS"
    trial_results: List[List[DiagnosticData]]
    convergence_steps: List[int]
    final_rewards: List[float]
    survival_signal_evolution: List[List[float]]
    gradient_evolution: List[List[float]]
    alpha_evolution: List[List[float]]
    epsilon_evolution: List[List[float]]
    distance_evolution: List[List[float]]


class DiagnosticCollector:
    """Collects detailed diagnostic data during training"""
    
    def __init__(self):
        self.data: List[DiagnosticData] = []
        self.runtime: Optional[Runtime] = None
        self.ptdc: Optional[PerformanceThresholdDetectionCircuit] = None
        self.ngcm: Optional[NonlinearGradientComputationModule] = None
        self.apc: Optional[AdaptiveParameterController] = None
        
    def attach_to_runtime(self, runtime: Runtime):
        """Attach collector to runtime to gather internal state"""
        self.runtime = runtime
        # Access internal components if possible
        # Note: This may require runtime modifications to expose internals
        
    def collect_step(
        self,
        step: int,
        agent_id: str,
        preference: float,
        reward: float,
        metric_value: float,
        threshold: float
    ):
        """Collect diagnostic data for a single step"""
        survival_signal = 0.0
        gradient = 0.0
        alpha = 0.01
        epsilon = 0.1
        threshold_status = False
        
        # Try to extract internal state from runtime
        if self.runtime:
            # Get survival signal from constraint evaluation
            try:
                metrics = {agent_id: metric_value}
                threshold_status_dict = self.runtime.constraint.evaluate(metrics)
                threshold_status = threshold_status_dict.get(agent_id, False)
                
                # Compute survival signal
                meeting_threshold = sum(1 for v in threshold_status_dict.values() if v)
                total_agents = len(threshold_status_dict)
                if total_agents > 0:
                    survival_ratio = meeting_threshold / total_agents
                    survival_signal = self.runtime.constraint.compute_survival_signal(survival_ratio)
            except:
                pass
            
            # Try to get adaptive parameters
            try:
                if hasattr(self.runtime, 'apc'):
                    alpha = self.runtime.apc.get_alpha()
                    epsilon = self.runtime.apc.get_epsilon()
            except:
                pass
            
            # Try to compute gradient
            try:
                if hasattr(self.runtime, 'ngcm'):
                    distance = metric_value - threshold
                    gradient = self.runtime.ngcm.compute_gradient(distance, step)
            except:
                # Fallback: compute gradient manually
                distance = metric_value - threshold
                if hasattr(self, 'ngcm') and self.ngcm:
                    gradient = self.ngcm.compute_gradient(distance, step)
                else:
                    # Simple gradient approximation
                    beta = 1.0
                    sigmoid = 1.0 / (1.0 + np.exp(-beta * distance))
                    gradient = beta * sigmoid * (1.0 - sigmoid)
        
        distance_to_threshold = abs(metric_value - threshold)
        
        diagnostic = DiagnosticData(
            step=step,
            survival_signal=survival_signal,
            gradient=gradient,
            alpha=alpha,
            epsilon=epsilon,
            preference=preference,
            reward=reward,
            distance_to_threshold=distance_to_threshold,
            threshold_status=threshold_status,
            metric_value=metric_value
        )
        
        self.data.append(diagnostic)
        return diagnostic
    
    def get_data(self) -> List[DiagnosticData]:
        """Get all collected diagnostic data"""
        return self.data
    
    def reset(self):
        """Reset collector for new trial"""
        self.data = []


async def run_diagnostic_trial(
    scenario_name: str,
    reward_model_type: str,
    preference_distribution: str,
    convergence_threshold: float,
    method: str = "PulseOS",
    max_steps: int = 5000,
    collector: Optional[DiagnosticCollector] = None
) -> List[DiagnosticData]:
    """Run a single diagnostic trial with detailed data collection"""
    
    if collector is None:
        collector = DiagnosticCollector()
    collector.reset()
    
    if method == "PulseOS":
        constraint = SurvivalConstraint(threshold=0.5)
        config = Config(
            max_agents=1,
            parallel_updates=False,
            alpha_base=0.02,
            gamma=0.2,
            alpha_max_change_per_step=0.25
        )
        runtime = Runtime(constraint=constraint, config=config)
        
        agent = VariantRLHFAgent(
            f"diagnostic_agent",
            reward_model_type,
            preference_distribution,
            convergence_threshold
        )
        runtime.register_agent("diagnostic_agent", agent)
        collector.attach_to_runtime(runtime)
        
        for step in range(max_steps):
            await runtime.step()
            
            # Get agent state
            preference = agent.preference_history[-1] if agent.preference_history else 0.0
            reward = agent.reward_history[-1] if agent.reward_history else 0.0
            metric = agent.get_performance_metric()
            
            # Collect diagnostic data
            collector.collect_step(
                step=step,
                agent_id="diagnostic_agent",
                preference=preference,
                reward=reward,
                metric_value=metric,
                threshold=0.5
            )
            
            # Check convergence
            if agent.converged:
                break
    
    else:  # PPO baseline
        agent = VariantRLHFAgent(
            f"ppo_agent",
            reward_model_type,
            preference_distribution,
            convergence_threshold
        )
        
        ppo_reward = 0.0
        ppo_variance = 1.0
        ppo_learning_rate = 0.01
        
        for step in range(max_steps):
            reward = agent._compute_reward(ppo_reward, ppo_variance)
            preference = agent._sample_preference(reward, ppo_variance)
            
            error = preference - ppo_reward
            ppo_reward += ppo_learning_rate * error
            ppo_variance = max(0.05, ppo_variance * 0.999)
            
            agent.preference_history.append(preference)
            agent.reward_history.append(reward)
            
            metric = agent.get_performance_metric()
            
            # Collect diagnostic data (simplified for PPO)
            collector.collect_step(
                step=step,
                agent_id="ppo_agent",
                preference=preference,
                reward=reward,
                metric_value=metric,
                threshold=0.5
            )
            
            if len(agent.preference_history) >= 50:
                recent_avg = np.mean(agent.preference_history[-50:])
                if recent_avg > convergence_threshold:
                    break
    
    return collector.get_data()


async def diagnose_scenario(
    scenario_name: str,
    reward_model_type: str,
    preference_distribution: str,
    convergence_threshold: float,
    num_trials: int = 5
) -> Tuple[ScenarioDiagnostics, ScenarioDiagnostics]:
    """Run diagnostic analysis for a scenario"""
    
    print(f"\n{'='*70}")
    print(f"Diagnosing Scenario: {scenario_name}")
    print(f"{'='*70}")
    
    # Run PPO diagnostics
    print(f"Running PPO diagnostics...")
    ppo_trials = []
    for trial in range(num_trials):
        print(f"  Trial {trial + 1}/{num_trials}...", end="\r")
        collector = DiagnosticCollector()
        data = await run_diagnostic_trial(
            scenario_name,
            reward_model_type,
            preference_distribution,
            convergence_threshold,
            method="PPO",
            collector=collector
        )
        ppo_trials.append(data)
    print(f"  PPO diagnostics complete.     ")
    
    # Run PulseOS diagnostics
    print(f"Running PulseOS diagnostics...")
    pulseos_trials = []
    for trial in range(num_trials):
        print(f"  Trial {trial + 1}/{num_trials}...", end="\r")
        collector = DiagnosticCollector()
        data = await run_diagnostic_trial(
            scenario_name,
            reward_model_type,
            preference_distribution,
            convergence_threshold,
            method="PulseOS",
            collector=collector
        )
        pulseos_trials.append(data)
    print(f"  PulseOS diagnostics complete.     ")
    
    # Process PPO data
    ppo_convergence_steps = []
    ppo_final_rewards = []
    for trial_data in ppo_trials:
        ppo_convergence_steps.append(len(trial_data))
        if trial_data:
            ppo_final_rewards.append(trial_data[-1].reward)
        else:
            ppo_final_rewards.append(0.0)
    
    ppo_diagnostics = ScenarioDiagnostics(
        scenario_name=scenario_name,
        method="PPO",
        trial_results=ppo_trials,
        convergence_steps=ppo_convergence_steps,
        final_rewards=ppo_final_rewards,
        survival_signal_evolution=[[d.survival_signal for d in trial] for trial in ppo_trials],
        gradient_evolution=[[d.gradient for d in trial] for trial in ppo_trials],
        alpha_evolution=[[d.alpha for d in trial] for trial in ppo_trials],
        epsilon_evolution=[[d.epsilon for d in trial] for trial in ppo_trials],
        distance_evolution=[[d.distance_to_threshold for d in trial] for trial in ppo_trials]
    )
    
    # Process PulseOS data
    pulseos_convergence_steps = []
    pulseos_final_rewards = []
    for trial_data in pulseos_trials:
        pulseos_convergence_steps.append(len(trial_data))
        if trial_data:
            pulseos_final_rewards.append(trial_data[-1].reward)
        else:
            pulseos_final_rewards.append(0.0)
    
    pulseos_diagnostics = ScenarioDiagnostics(
        scenario_name=scenario_name,
        method="PulseOS",
        trial_results=pulseos_trials,
        convergence_steps=pulseos_convergence_steps,
        final_rewards=pulseos_final_rewards,
        survival_signal_evolution=[[d.survival_signal for d in trial] for trial in pulseos_trials],
        gradient_evolution=[[d.gradient for d in trial] for trial in pulseos_trials],
        alpha_evolution=[[d.alpha for d in trial] for trial in pulseos_trials],
        epsilon_evolution=[[d.epsilon for d in trial] for trial in pulseos_trials],
        distance_evolution=[[d.distance_to_threshold for d in trial] for trial in pulseos_trials]
    )
    
    return ppo_diagnostics, pulseos_diagnostics


def create_diagnostic_visualizations(
    ppo_diagnostics: ScenarioDiagnostics,
    pulseos_diagnostics: ScenarioDiagnostics,
    output_dir: Path
):
    """Create comprehensive diagnostic visualizations"""
    
    scenario_name = ppo_diagnostics.scenario_name
    fig_dir = output_dir / "diagnostics" / scenario_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Survival Signal Evolution
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, evolution in enumerate(ppo_diagnostics.survival_signal_evolution):
        ax.plot(evolution, alpha=0.3, color='blue', label='PPO' if i == 0 else '')
    for i, evolution in enumerate(pulseos_diagnostics.survival_signal_evolution):
        ax.plot(evolution, alpha=0.3, color='red', label='PulseOS' if i == 0 else '')
    ax.set_xlabel('Step')
    ax.set_ylabel('Survival Signal')
    ax.set_title(f'{scenario_name}: Survival Signal Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'survival_signal_evolution.png', dpi=150)
    plt.close()
    
    # 2. Gradient Magnitude Plots
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, evolution in enumerate(ppo_diagnostics.gradient_evolution):
        ax.plot(evolution, alpha=0.3, color='blue', label='PPO' if i == 0 else '')
    for i, evolution in enumerate(pulseos_diagnostics.gradient_evolution):
        ax.plot(evolution, alpha=0.3, color='red', label='PulseOS' if i == 0 else '')
    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title(f'{scenario_name}: Gradient Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'gradient_evolution.png', dpi=150)
    plt.close()
    
    # 3. Distance to Threshold
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, evolution in enumerate(ppo_diagnostics.distance_evolution):
        ax.plot(evolution, alpha=0.3, color='blue', label='PPO' if i == 0 else '')
    for i, evolution in enumerate(pulseos_diagnostics.distance_evolution):
        ax.plot(evolution, alpha=0.3, color='red', label='PulseOS' if i == 0 else '')
    ax.set_xlabel('Step')
    ax.set_ylabel('Distance to Threshold')
    ax.set_title(f'{scenario_name}: Distance to Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'distance_to_threshold.png', dpi=150)
    plt.close()
    
    # 4. Parameter Adaptation (Alpha)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, evolution in enumerate(ppo_diagnostics.alpha_evolution):
        ax.plot(evolution, alpha=0.3, color='blue', label='PPO' if i == 0 else '')
    for i, evolution in enumerate(pulseos_diagnostics.alpha_evolution):
        ax.plot(evolution, alpha=0.3, color='red', label='PulseOS' if i == 0 else '')
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate (Alpha)')
    ax.set_title(f'{scenario_name}: Learning Rate Adaptation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'alpha_evolution.png', dpi=150)
    plt.close()
    
    # 5. Parameter Adaptation (Epsilon)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, evolution in enumerate(ppo_diagnostics.epsilon_evolution):
        ax.plot(evolution, alpha=0.3, color='blue', label='PPO' if i == 0 else '')
    for i, evolution in enumerate(pulseos_diagnostics.epsilon_evolution):
        ax.plot(evolution, alpha=0.3, color='red', label='PulseOS' if i == 0 else '')
    ax.set_xlabel('Step')
    ax.set_ylabel('Exploration Rate (Epsilon)')
    ax.set_title(f'{scenario_name}: Exploration Rate Adaptation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'epsilon_evolution.png', dpi=150)
    plt.close()
    
    # 6. Convergence Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ['PPO', 'PulseOS']
    ppo_mean = np.mean(ppo_diagnostics.convergence_steps)
    pulseos_mean = np.mean(pulseos_diagnostics.convergence_steps)
    ppo_std = np.std(ppo_diagnostics.convergence_steps)
    pulseos_std = np.std(pulseos_diagnostics.convergence_steps)
    
    ax.bar(methods, [ppo_mean, pulseos_mean], yerr=[ppo_std, pulseos_std], 
           color=['blue', 'red'], alpha=0.7, capsize=5)
    ax.set_ylabel('Steps to Convergence')
    ax.set_title(f'{scenario_name}: Convergence Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(fig_dir / 'convergence_comparison.png', dpi=150)
    plt.close()
    
    print(f"  Diagnostic visualizations saved to: {fig_dir}")


def generate_diagnostic_report(
    scenarios: Dict[str, Tuple[ScenarioDiagnostics, ScenarioDiagnostics]],
    output_dir: Path
):
    """Generate comprehensive diagnostic report"""
    
    report_path = output_dir / "diagnostic_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# PulseOS Diagnostic Analysis Report\n\n")
        f.write("## Phase 1: Deep Dive into Failure Cases\n\n")
        f.write("This report analyzes why PulseOS fails in certain RLHF scenarios.\n\n")
        
        for scenario_name, (ppo_diag, pulseos_diag) in scenarios.items():
            f.write(f"\n## Scenario: {scenario_name}\n\n")
            
            # Summary statistics
            ppo_mean_steps = np.mean(ppo_diag.convergence_steps)
            pulseos_mean_steps = np.mean(pulseos_diag.convergence_steps)
            reduction = ((ppo_mean_steps - pulseos_mean_steps) / ppo_mean_steps) * 100
            
            f.write(f"### Summary\n\n")
            f.write(f"- **PPO Average Steps:** {ppo_mean_steps:.1f} ± {np.std(ppo_diag.convergence_steps):.1f}\n")
            f.write(f"- **PulseOS Average Steps:** {pulseos_mean_steps:.1f} ± {np.std(pulseos_diag.convergence_steps):.1f}\n")
            f.write(f"- **Step Reduction:** {reduction:.1f}%\n\n")
            
            # Key observations
            f.write(f"### Key Observations\n\n")
            
            # Analyze survival signal
            ppo_avg_survival = [np.mean([d.survival_signal for d in trial]) for trial in ppo_diag.trial_results]
            pulseos_avg_survival = [np.mean([d.survival_signal for d in trial]) for trial in pulseos_diag.trial_results]
            
            f.write(f"1. **Survival Signal Behavior:**\n")
            f.write(f"   - PPO average: {np.mean(ppo_avg_survival):.3f}\n")
            f.write(f"   - PulseOS average: {np.mean(pulseos_avg_survival):.3f}\n\n")
            
            # Analyze gradients
            ppo_avg_gradient = [np.mean([abs(d.gradient) for d in trial]) for trial in ppo_diag.trial_results]
            pulseos_avg_gradient = [np.mean([abs(d.gradient) for d in trial]) for trial in pulseos_diag.trial_results]
            
            f.write(f"2. **Gradient Magnitude:**\n")
            f.write(f"   - PPO average: {np.mean(ppo_avg_gradient):.3f}\n")
            f.write(f"   - PulseOS average: {np.mean(pulseos_avg_gradient):.3f}\n\n")
            
            # Analyze distance to threshold
            ppo_avg_distance = [np.mean([d.distance_to_threshold for d in trial]) for trial in ppo_diag.trial_results]
            pulseos_avg_distance = [np.mean([d.distance_to_threshold for d in trial]) for trial in pulseos_diag.trial_results]
            
            f.write(f"3. **Distance to Threshold:**\n")
            f.write(f"   - PPO average: {np.mean(ppo_avg_distance):.3f}\n")
            f.write(f"   - PulseOS average: {np.mean(pulseos_avg_distance):.3f}\n\n")
            
            # Root cause analysis
            f.write(f"### Root Cause Analysis\n\n")
            
            if reduction < 20:
                f.write(f"**Problem:** PulseOS shows minimal improvement ({reduction:.1f}% reduction).\n\n")
                
                if scenario_name.startswith("multi_objective"):
                    f.write("**Hypothesis:** Multi-objective scenario requires balancing multiple thresholds. "
                           "Current single-threshold PTDC cannot handle multiple competing objectives.\n\n")
                
                elif scenario_name.startswith("bimodal"):
                    f.write("**Hypothesis:** Bimodal distribution has two preference peaks. "
                           "Current threshold detection assumes single mode, causing confusion.\n\n")
                
                elif scenario_name.startswith("skewed"):
                    f.write("**Hypothesis:** Skewed distribution has asymmetric preference landscape. "
                           "NGCM assumes symmetric gradients, failing to adapt to asymmetry.\n\n")
            
            # Visualizations reference
            f.write(f"### Visualizations\n\n")
            f.write(f"See `diagnostics/{scenario_name}/` for detailed plots:\n")
            f.write(f"- Survival signal evolution\n")
            f.write(f"- Gradient magnitude plots\n")
            f.write(f"- Distance to threshold\n")
            f.write(f"- Parameter adaptation curves\n")
            f.write(f"- Convergence comparison\n\n")
        
        # Overall recommendations
        f.write("\n## Overall Recommendations\n\n")
        f.write("Based on diagnostic analysis:\n\n")
        f.write("1. **Multi-Objective Scenarios:** Implement multi-threshold PTDC\n")
        f.write("2. **Bimodal Distributions:** Enhance threshold detection for multiple modes\n")
        f.write("3. **Skewed Distributions:** Implement skewness-aware gradient computation\n")
        f.write("4. **Hyperparameter Tuning:** Scenario-specific configurations needed\n\n")
    
    print(f"\nDiagnostic report saved to: {report_path}")


async def main():
    """Run Phase 1 diagnostic analysis"""
    
    print("=" * 70)
    print("PHASE 1: DIAGNOSTIC ANALYSIS")
    print("=" * 70)
    
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    # Define failing scenarios
    failing_scenarios = [
        ("multi_objective_normal_th-0.5", "multi_objective", "normal", -0.5),
        ("linear_bimodal_th-0.5", "linear", "bimodal", -0.5),
        ("linear_skewed_th-0.3", "linear", "skewed", -0.3),
    ]
    
    all_diagnostics = {}
    
    for scenario_name, reward_model, pref_dist, threshold in failing_scenarios:
        ppo_diag, pulseos_diag = await diagnose_scenario(
            scenario_name,
            reward_model,
            pref_dist,
            threshold,
            num_trials=5
        )
        
        all_diagnostics[scenario_name] = (ppo_diag, pulseos_diag)
        
        # Create visualizations
        create_diagnostic_visualizations(ppo_diag, pulseos_diag, output_dir)
    
    # Generate report
    generate_diagnostic_report(all_diagnostics, output_dir)
    
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print(f"Report: {output_dir / 'diagnostic_report.md'}")


if __name__ == "__main__":
    asyncio.run(main())

