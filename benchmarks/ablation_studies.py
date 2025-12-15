"""
Phase 4: Ablation Studies

Tests what components contribute to success by removing/modifying them.
"""

import asyncio
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from pulseos import Runtime, Config, Agent, SurvivalConstraint
from benchmarks.strategic_benchmark_suite import VariantRLHFAgent, run_rlhf_variant_ppo


@dataclass
class AblationResult:
    """Results from an ablation study configuration"""
    configuration: str
    scenario: str
    steps_to_convergence: List[int]
    avg_steps: float
    std_steps: float
    final_rewards: List[float]


class AblationStudy:
    """Run ablation studies to understand component contributions"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, List[AblationResult]] = {}
    
    async def run_ablation_scenario(
        self,
        scenario_name: str,
        reward_model_type: str,
        preference_distribution: str,
        convergence_threshold: float,
        num_trials: int = 5
    ):
        """Run ablation study for a specific scenario"""
        
        print(f"\n{'='*70}")
        print(f"Ablation Study: {scenario_name}")
        print(f"{'='*70}")
        
        configurations = [
            ("Full PulseOS", self._run_full_pulseos),
            ("Without PTDC", self._run_without_ptdc),
            ("Without NGCM", self._run_without_ngcm),
            ("Without APC", self._run_without_apc),
            ("Only Survival Constraint", self._run_only_survival),
            ("PPO with Survival Constraint", self._run_ppo_with_survival),
        ]
        
        scenario_results = []
        
        for config_name, run_func in configurations:
            print(f"\nTesting: {config_name}")
            results = []
            
            for trial in range(num_trials):
                print(f"  Trial {trial + 1}/{num_trials}...", end="\r")
                steps = await run_func(
                    scenario_name,
                    reward_model_type,
                    preference_distribution,
                    convergence_threshold,
                    trial
                )
                results.append(steps)
            
            print(f"  Complete. Average: {np.mean(results):.1f} steps     ")
            
            ablation_result = AblationResult(
                configuration=config_name,
                scenario=scenario_name,
                steps_to_convergence=results,
                avg_steps=np.mean(results),
                std_steps=np.std(results),
                final_rewards=[0.0] * num_trials  # Placeholder
            )
            scenario_results.append(ablation_result)
        
        self.results[scenario_name] = scenario_results
        return scenario_results
    
    async def _run_full_pulseos(
        self,
        scenario_name: str,
        reward_model_type: str,
        preference_distribution: str,
        convergence_threshold: float,
        trial: int
    ) -> int:
        """Run full PulseOS"""
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
            f"full_{trial}",
            reward_model_type,
            preference_distribution,
            convergence_threshold
        )
        runtime.register_agent(f"full_{trial}", agent)
        
        for step in range(5000):
            await runtime.step()
            if agent.converged:
                return step
        
        return 5000
    
    async def _run_without_ptdc(
        self,
        scenario_name: str,
        reward_model_type: str,
        preference_distribution: str,
        convergence_threshold: float,
        trial: int
    ) -> int:
        """Run PulseOS without PTDC (use simple threshold check)"""
        constraint = SurvivalConstraint(threshold=0.5)
        config = Config(
            max_agents=1,
            parallel_updates=False,
            alpha_base=0.02,
            gamma=0.2,
            alpha_max_change_per_step=0.25
        )
        runtime = Runtime(constraint=constraint, config=config)
        
        # Disable PTDC by bypassing threshold detection
        original_evaluate = runtime.ptdc.evaluate
        def simple_evaluate(metrics):
            # Simple threshold check without PTDC logic
            return {aid: m >= 0.5 for aid, m in metrics.items()}
        runtime.ptdc.evaluate = simple_evaluate
        
        agent = VariantRLHFAgent(
            f"no_ptdc_{trial}",
            reward_model_type,
            preference_distribution,
            convergence_threshold
        )
        runtime.register_agent(f"no_ptdc_{trial}", agent)
        
        for step in range(5000):
            await runtime.step()
            if agent.converged:
                return step
        
        return 5000
    
    async def _run_without_ngcm(
        self,
        scenario_name: str,
        reward_model_type: str,
        preference_distribution: str,
        convergence_threshold: float,
        trial: int
    ) -> int:
        """Run PulseOS without NGCM (use simple gradient)"""
        constraint = SurvivalConstraint(threshold=0.5)
        config = Config(
            max_agents=1,
            parallel_updates=False,
            alpha_base=0.02,
            gamma=0.2,
            alpha_max_change_per_step=0.25
        )
        runtime = Runtime(constraint=constraint, config=config)
        
        # Replace NGCM with simple gradient computation
        original_compute = runtime.ngcm.compute_gradient
        def simple_gradient(delta, timestamp):
            return abs(delta) * 0.1  # Simple linear gradient
        runtime.ngcm.compute_gradient = simple_gradient
        
        agent = VariantRLHFAgent(
            f"no_ngcm_{trial}",
            reward_model_type,
            preference_distribution,
            convergence_threshold
        )
        runtime.register_agent(f"no_ngcm_{trial}", agent)
        
        for step in range(5000):
            await runtime.step()
            if agent.converged:
                return step
        
        return 5000
    
    async def _run_without_apc(
        self,
        scenario_name: str,
        reward_model_type: str,
        preference_distribution: str,
        convergence_threshold: float,
        trial: int
    ) -> int:
        """Run PulseOS without APC (use fixed parameters)"""
        constraint = SurvivalConstraint(threshold=0.5)
        config = Config(
            max_agents=1,
            parallel_updates=False,
            alpha_base=0.02,
            gamma=0.2,
            alpha_max_change_per_step=0.25
        )
        runtime = Runtime(constraint=constraint, config=config)
        
        # Replace APC with fixed parameters
        original_update = runtime.apc.update_parameters
        def fixed_params(gradient, survival_signal):
            return 0.02, 0.1  # Fixed alpha, epsilon
        runtime.apc.update_parameters = fixed_params
        
        agent = VariantRLHFAgent(
            f"no_apc_{trial}",
            reward_model_type,
            preference_distribution,
            convergence_threshold
        )
        runtime.register_agent(f"no_apc_{trial}", agent)
        
        for step in range(5000):
            await runtime.step()
            if agent.converged:
                return step
        
        return 5000
    
    async def _run_only_survival(
        self,
        scenario_name: str,
        reward_model_type: str,
        preference_distribution: str,
        convergence_threshold: float,
        trial: int
    ) -> int:
        """Run with only survival constraint (minimal version)"""
        # Just use survival constraint to guide learning
        agent = VariantRLHFAgent(
            f"survival_only_{trial}",
            reward_model_type,
            preference_distribution,
            convergence_threshold
        )
        
        # Simple survival-guided learning
        for step in range(5000):
            reward = agent._compute_reward(agent.reward, agent.variance)
            preference = agent._sample_preference(reward, agent.variance)
            
            # Survival-guided update
            metric = agent.get_performance_metric()
            survival_signal = 1.0 if metric >= 0.5 else 0.0
            
            error = preference - agent.reward
            lr = 0.02 * (1.0 + survival_signal * 0.5)
            agent.reward += lr * error
            
            agent.variance = max(0.05, agent.variance * 0.999)
            agent.preference_history.append(preference)
            agent.reward_history.append(reward)
            
            if len(agent.preference_history) >= 50:
                recent_avg = np.mean(agent.preference_history[-50:])
                if recent_avg > convergence_threshold:
                    return step
        
        return 5000
    
    async def _run_ppo_with_survival(
        self,
        scenario_name: str,
        reward_model_type: str,
        preference_distribution: str,
        convergence_threshold: float,
        trial: int
    ) -> int:
        """Run PPO with survival constraint guidance"""
        agent = VariantRLHFAgent(
            f"ppo_survival_{trial}",
            reward_model_type,
            preference_distribution,
            convergence_threshold
        )
        
        ppo_reward = 0.0
        ppo_variance = 1.0
        ppo_learning_rate = 0.01
        
        for step in range(5000):
            reward = agent._compute_reward(ppo_reward, ppo_variance)
            preference = agent._sample_preference(reward, ppo_variance)
            
            # Survival-guided PPO
            metric = agent.get_performance_metric()
            survival_signal = 1.0 if metric >= 0.5 else 0.0
            
            error = preference - ppo_reward
            lr = ppo_learning_rate * (1.0 + survival_signal * 0.3)
            ppo_reward += lr * error
            ppo_variance = max(0.05, ppo_variance * 0.999)
            
            agent.preference_history.append(preference)
            
            if len(agent.preference_history) >= 50:
                recent_avg = np.mean(agent.preference_history[-50:])
                if recent_avg > convergence_threshold:
                    return step
        
        return 5000
    
    def save_results(self):
        """Save ablation study results"""
        results_path = self.output_dir / "ablation_results.json"
        
        results_dict = {}
        for scenario, ablation_results in self.results.items():
            results_dict[scenario] = [
                {
                    'configuration': r.configuration,
                    'avg_steps': r.avg_steps,
                    'std_steps': r.std_steps,
                    'steps': r.steps_to_convergence
                }
                for r in ablation_results
            ]
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nAblation results saved to: {results_path}")
        return results_path
    
    def generate_report(self):
        """Generate ablation study report"""
        report_path = self.output_dir / "ablation_study_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# PulseOS Ablation Study Report\n\n")
            f.write("## Phase 4: Component Analysis\n\n")
            f.write("This report analyzes what components contribute to PulseOS success.\n\n")
            
            for scenario, ablation_results in self.results.items():
                f.write(f"## Scenario: {scenario}\n\n")
                f.write("| Configuration | Avg Steps | Std Dev | vs Full PulseOS |\n")
                f.write("|---------------|-----------|---------|-----------------|\n")
                
                full_pulseos_avg = None
                for r in ablation_results:
                    if r.configuration == "Full PulseOS":
                        full_pulseos_avg = r.avg_steps
                        break
                
                for r in ablation_results:
                    vs_full = ""
                    if full_pulseos_avg and r.configuration != "Full PulseOS":
                        diff = ((r.avg_steps - full_pulseos_avg) / full_pulseos_avg * 100) if full_pulseos_avg > 0 else 0
                        vs_full = f"{diff:+.1f}%"
                    
                    f.write(f"| {r.configuration} | {r.avg_steps:.1f} | {r.std_steps:.1f} | {vs_full} |\n")
                
                f.write("\n### Key Insights\n\n")
                
                # Find best and worst
                best = min(ablation_results, key=lambda x: x.avg_steps)
                worst = max(ablation_results, key=lambda x: x.avg_steps)
                
                f.write(f"- **Best Configuration:** {best.configuration} ({best.avg_steps:.1f} steps)\n")
                f.write(f"- **Worst Configuration:** {worst.configuration} ({worst.avg_steps:.1f} steps)\n")
                
                if full_pulseos_avg:
                    f.write(f"- **Full PulseOS Performance:** {full_pulseos_avg:.1f} steps\n")
                    
                    # Analyze component contributions
                    no_ptdc = next((r for r in ablation_results if r.configuration == "Without PTDC"), None)
                    no_ngcm = next((r for r in ablation_results if r.configuration == "Without NGCM"), None)
                    no_apc = next((r for r in ablation_results if r.configuration == "Without APC"), None)
                    
                    f.write("\n**Component Impact:**\n")
                    if no_ptdc:
                        impact = ((no_ptdc.avg_steps - full_pulseos_avg) / full_pulseos_avg * 100) if full_pulseos_avg > 0 else 0
                        f.write(f"- Removing PTDC: {impact:+.1f}% change\n")
                    if no_ngcm:
                        impact = ((no_ngcm.avg_steps - full_pulseos_avg) / full_pulseos_avg * 100) if full_pulseos_avg > 0 else 0
                        f.write(f"- Removing NGCM: {impact:+.1f}% change\n")
                    if no_apc:
                        impact = ((no_apc.avg_steps - full_pulseos_avg) / full_pulseos_avg * 100) if full_pulseos_avg > 0 else 0
                        f.write(f"- Removing APC: {impact:+.1f}% change\n")
                
                f.write("\n")
        
        print(f"\nAblation report saved to: {report_path}")
        return report_path


async def run_ablation_studies():
    """Run ablation studies on key scenarios"""
    
    print("=" * 70)
    print("PHASE 4: ABLATION STUDIES")
    print("=" * 70)
    
    output_dir = Path("benchmark_results")
    study = AblationStudy(output_dir)
    
    # Run ablation on failing scenarios
    scenarios = [
        ("multi_objective_normal_th-0.5", "multi_objective", "normal", -0.5),
        ("linear_bimodal_th-0.5", "linear", "bimodal", -0.5),
        ("linear_skewed_th-0.3", "linear", "skewed", -0.3),
    ]
    
    for scenario_name, reward_model, pref_dist, threshold in scenarios:
        await study.run_ablation_scenario(
            scenario_name,
            reward_model,
            pref_dist,
            threshold,
            num_trials=5
        )
    
    study.save_results()
    study.generate_report()
    
    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)
    
    return study.results


if __name__ == "__main__":
    asyncio.run(run_ablation_studies())




