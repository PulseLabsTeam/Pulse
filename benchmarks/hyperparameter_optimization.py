"""
Phase 2: Hyperparameter Optimization Framework

Uses Optuna for automated hyperparameter tuning to find optimal configurations
for each RLHF scenario type.
"""

import asyncio
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

from pulseos import Runtime, Config, Agent, SurvivalConstraint
from benchmarks.strategic_benchmark_suite import VariantRLHFAgent, run_rlhf_variant_pulseos


@dataclass
class HyperparameterConfig:
    """Hyperparameter configuration"""
    survival_threshold: float
    alpha_base: float
    epsilon_min: float
    epsilon_max: float
    gamma: float
    beta: float
    cache_size: int
    alpha_max_change: float = 0.25
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HyperparameterOptimizer:
    """Optimizes PulseOS hyperparameters for specific scenarios"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_configs: Dict[str, HyperparameterConfig] = {}
    
    async def optimize_scenario(
        self,
        scenario_name: str,
        reward_model_type: str,
        preference_distribution: str,
        convergence_threshold: float,
        n_trials: int = 50,
        n_validation_trials: int = 5
    ) -> Tuple[HyperparameterConfig, float]:
        """
        Optimize hyperparameters for a specific scenario.
        
        Returns:
            Tuple of (best_config, best_score)
        """
        
        print(f"\n{'='*70}")
        print(f"Optimizing: {scenario_name}")
        print(f"{'='*70}")
        
        def objective(trial):
            # Define hyperparameter search space
            survival_threshold = trial.suggest_float('survival_threshold', 0.3, 0.95, step=0.05)
            alpha_base = trial.suggest_float('alpha_base', 0.0001, 0.1, log=True)
            epsilon_min = trial.suggest_float('epsilon_min', 0.01, 0.1, step=0.01)
            epsilon_max = trial.suggest_float('epsilon_max', 0.1, 0.7, step=0.1)
            gamma = trial.suggest_float('gamma', 1.0, 10.0, step=1.0)
            beta = trial.suggest_float('beta', 2.0, 20.0, step=2.0)
            cache_size = trial.suggest_int('cache_size', 64, 512, step=64)
            alpha_max_change = trial.suggest_float('alpha_max_change', 0.1, 0.5, step=0.05)
            
            # Run evaluation in a new event loop (to avoid conflicts)
            import threading
            result = [None]
            exception = [None]
            
            def run_async():
                try:
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    score = new_loop.run_until_complete(self._evaluate_config(
                        scenario_name,
                        reward_model_type,
                        preference_distribution,
                        convergence_threshold,
                        survival_threshold,
                        alpha_base,
                        epsilon_min,
                        epsilon_max,
                        gamma,
                        beta,
                        cache_size,
                        alpha_max_change,
                        n_trials=3  # Quick evaluation during optimization
                    ))
                    result[0] = score
                    new_loop.close()
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=run_async)
            thread.start()
            thread.join()
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            study_name=f"optimize_{scenario_name}",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        print(f"Running {n_trials} optimization trials...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best config
        best_params = study.best_params
        best_config = HyperparameterConfig(
            survival_threshold=best_params['survival_threshold'],
            alpha_base=best_params['alpha_base'],
            epsilon_min=best_params['epsilon_min'],
            epsilon_max=best_params['epsilon_max'],
            gamma=best_params['gamma'],
            beta=best_params['beta'],
            cache_size=best_params['cache_size'],
            alpha_max_change=best_params['alpha_max_change']
        )
        
        # Validate best config with more trials
        print(f"\nValidating best configuration with {n_validation_trials} trials...")
        best_score = await self._evaluate_config(
            scenario_name,
            reward_model_type,
            preference_distribution,
            convergence_threshold,
            best_config.survival_threshold,
            best_config.alpha_base,
            best_config.epsilon_min,
            best_config.epsilon_max,
            best_config.gamma,
            best_config.beta,
            best_config.cache_size,
            best_config.alpha_max_change,
            n_trials=n_validation_trials
        )
        
        print(f"Best score: {best_score:.2f}% reduction")
        
        # Save optimization history
        self._save_optimization_history(study, scenario_name)
        
        return best_config, best_score
    
    async def _evaluate_config(
        self,
        scenario_name: str,
        reward_model_type: str,
        preference_distribution: str,
        convergence_threshold: float,
        survival_threshold: float,
        alpha_base: float,
        epsilon_min: float,
        epsilon_max: float,
        gamma: float,
        beta: float,
        cache_size: int,
        alpha_max_change: float,
        n_trials: int = 5
    ) -> float:
        """
        Evaluate a hyperparameter configuration.
        
        Returns:
            Average step reduction percentage
        """
        
        # Run PPO baseline
        from benchmarks.strategic_benchmark_suite import run_rlhf_variant_ppo
        ppo_results = await run_rlhf_variant_ppo(
            reward_model_type,
            preference_distribution,
            convergence_threshold,
            num_trials=n_trials,
            max_steps=5000
        )
        ppo_avg_steps = np.mean([r.steps_to_convergence for r in ppo_results])
        
        # Run PulseOS with config
        # Note: This requires modifying Runtime to accept custom hyperparameters
        # For now, we'll use a workaround by creating custom Config
        constraint = SurvivalConstraint(threshold=survival_threshold)
        config = Config(
            max_agents=1,
            parallel_updates=False,
            alpha_base=alpha_base,
            gamma=gamma,
            alpha_max_change_per_step=alpha_max_change
        )
        
        # Store config for custom agent creation
        # We'll need to modify the agent to use these parameters
        pulseos_results = await self._run_pulseos_with_config(
            reward_model_type,
            preference_distribution,
            convergence_threshold,
            constraint,
            config,
            epsilon_min,
            epsilon_max,
            beta,
            cache_size,
            n_trials
        )
        
        pulseos_avg_steps = np.mean([r.steps_to_convergence for r in pulseos_results])
        
        # Calculate reduction
        if ppo_avg_steps > 0:
            reduction = ((ppo_avg_steps - pulseos_avg_steps) / ppo_avg_steps) * 100
        else:
            reduction = 0.0
        
        return reduction
    
    async def _run_pulseos_with_config(
        self,
        reward_model_type: str,
        preference_distribution: str,
        convergence_threshold: float,
        constraint: SurvivalConstraint,
        config: Config,
        epsilon_min: float,
        epsilon_max: float,
        beta: float,
        cache_size: int,
        n_trials: int
    ):
        """Run PulseOS with custom hyperparameters"""
        # This is a simplified version - full implementation would require
        # modifying Runtime to accept custom NGCM/APC parameters
        results = []
        
        for trial in range(n_trials):
            runtime = Runtime(constraint=constraint, config=config)
            
            agent = VariantRLHFAgent(
                f"pulseos_{trial}",
                reward_model_type,
                preference_distribution,
                convergence_threshold
            )
            runtime.register_agent(f"pulseos_{trial}", agent)
            
            # Run until convergence
            max_steps = 5000
            for step in range(max_steps):
                await runtime.step()
                if agent.converged:
                    break
            
            from benchmarks.strategic_benchmark_suite import TrialResult
            results.append(TrialResult(
                trial=trial + 1,
                method="PulseOS",
                steps_to_convergence=agent.convergence_step if agent.converged else max_steps,
                total_time=0.0,
                final_reward=np.mean(agent.preference_history[-50:]) if agent.preference_history else 0.0,
                convergence_reward=convergence_threshold,
                learning_curve=agent.preference_history.copy(),
                step_times=[],
                variant_name=f"{reward_model_type}_{preference_distribution}"
            ))
        
        return results
    
    def _save_optimization_history(self, study: optuna.Study, scenario_name: str):
        """Save optimization history and visualizations"""
        study_dir = self.output_dir / "optimization" / scenario_name
        study_dir.mkdir(parents=True, exist_ok=True)
        
        # Save study as JSON
        study_json = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'trials': [
                {
                    'number': t.number,
                    'value': t.value,
                    'params': t.params,
                    'state': t.state.name
                }
                for t in study.trials
            ]
        }
        
        with open(study_dir / "study.json", 'w') as f:
            json.dump(study_json, f, indent=2)
        
        # Create visualizations
        try:
            fig1 = plot_optimization_history(study)
            fig1.write_image(str(study_dir / "optimization_history.png"))
            
            fig2 = plot_param_importances(study)
            fig2.write_image(str(study_dir / "param_importances.png"))
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
    
    def save_best_configs(self):
        """Save all best configurations to JSON"""
        configs_dict = {}
        for scenario, config in self.best_configs.items():
            if hasattr(config, 'to_dict'):
                configs_dict[scenario] = config.to_dict()
            elif isinstance(config, dict):
                configs_dict[scenario] = config
            else:
                # Convert to dict manually
                configs_dict[scenario] = {
                    'survival_threshold': getattr(config, 'survival_threshold', 0.5),
                    'alpha_base': getattr(config, 'alpha_base', 0.02),
                    'epsilon_min': getattr(config, 'epsilon_min', 0.01),
                    'epsilon_max': getattr(config, 'epsilon_max', 0.3),
                    'gamma': getattr(config, 'gamma', 0.2),
                    'beta': getattr(config, 'beta', 1.0),
                    'cache_size': getattr(config, 'cache_size', 256),
                    'alpha_max_change': getattr(config, 'alpha_max_change', 0.25)
                }
        
        config_path = self.output_dir / "optimal_hyperparameters.json"
        with open(config_path, 'w') as f:
            json.dump(configs_dict, f, indent=2)
        
        print(f"\nOptimal hyperparameters saved to: {config_path}")
        return config_path


class AdaptiveConfigSelector:
    """
    Automatically select best PulseOS config based on task characteristics.
    
    Analyzes reward distribution and preference data to classify scenario type,
    then selects optimal hyperparameters.
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """Initialize with optimal configs from optimization"""
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                self.config_map = json.load(f)
        else:
            # Default configs (will be replaced after optimization)
            self.config_map = {
                'linear_normal': HyperparameterConfig(
                    survival_threshold=0.5,
                    alpha_base=0.02,
                    epsilon_min=0.01,
                    epsilon_max=0.3,
                    gamma=0.2,
                    beta=1.0,
                    cache_size=256
                ).to_dict(),
                'nonlinear_normal': HyperparameterConfig(
                    survival_threshold=0.5,
                    alpha_base=0.01,
                    epsilon_min=0.05,
                    epsilon_max=0.5,
                    gamma=0.5,
                    beta=5.0,
                    cache_size=256
                ).to_dict(),
                'multi_objective': HyperparameterConfig(
                    survival_threshold=0.7,
                    alpha_base=0.005,
                    epsilon_min=0.1,
                    epsilon_max=0.7,
                    gamma=2.0,
                    beta=10.0,
                    cache_size=512
                ).to_dict(),
                'bimodal': HyperparameterConfig(
                    survival_threshold=0.6,
                    alpha_base=0.01,
                    epsilon_min=0.05,
                    epsilon_max=0.5,
                    gamma=1.0,
                    beta=5.0,
                    cache_size=256
                ).to_dict(),
                'skewed': HyperparameterConfig(
                    survival_threshold=0.4,
                    alpha_base=0.015,
                    epsilon_min=0.05,
                    epsilon_max=0.6,
                    gamma=1.5,
                    beta=8.0,
                    cache_size=256
                ).to_dict()
            }
    
    def detect_scenario_type(
        self,
        reward_samples: List[float],
        preference_data: List[float]
    ) -> str:
        """
        Analyze reward distribution and classify scenario type.
        
        Returns:
            Scenario type: 'linear_normal', 'nonlinear_normal', 'multi_objective',
                          'bimodal', 'skewed'
        """
        # Analyze reward distribution
        reward_std = np.std(reward_samples)
        reward_mean = np.mean(reward_samples)
        
        # Analyze preference distribution
        pref_std = np.std(preference_data)
        pref_mean = np.mean(preference_data)
        
        # Check for bimodality
        from scipy import stats
        try:
            # Simple bimodality test
            hist, bins = np.histogram(preference_data, bins=20)
            peaks = []
            for i in range(1, len(hist) - 1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                    peaks.append(i)
            
            if len(peaks) >= 2:
                return 'bimodal'
        except:
            pass
        
        # Check for skewness
        try:
            skewness = stats.skew(preference_data)
            if abs(skewness) > 0.5:
                return 'skewed'
        except:
            pass
        
        # Check for multi-objective (high variance in preferences)
        if pref_std > reward_std * 1.5:
            return 'multi_objective'
        
        # Check for nonlinear (non-uniform reward distribution)
        if reward_std > reward_mean * 0.5:
            return 'nonlinear_normal'
        
        # Default to linear normal
        return 'linear_normal'
    
    def select_config(self, scenario_type: str) -> HyperparameterConfig:
        """Return optimal hyperparameters for detected scenario type"""
        config_dict = self.config_map.get(scenario_type, self.config_map['linear_normal'])
        return HyperparameterConfig(**config_dict)


async def optimize_all_scenarios(
    output_dir: Path,
    n_trials: int = 50,
    n_validation_trials: int = 5
) -> Dict[str, HyperparameterConfig]:
    """Optimize hyperparameters for all scenarios"""
    
    optimizer = HyperparameterOptimizer(output_dir)
    
    scenarios = [
        ("linear_normal_th-0.5", "linear", "normal", -0.5),
        ("nonlinear_normal_th-0.5", "nonlinear", "normal", -0.5),
        ("multi_objective_normal_th-0.5", "multi_objective", "normal", -0.5),
        ("linear_bimodal_th-0.5", "linear", "bimodal", -0.5),
        ("linear_skewed_th-0.3", "linear", "skewed", -0.3),
    ]
    
    for scenario_name, reward_model, pref_dist, threshold in scenarios:
        best_config, best_score = await optimizer.optimize_scenario(
            scenario_name,
            reward_model,
            pref_dist,
            threshold,
            n_trials=n_trials,
            n_validation_trials=n_validation_trials
        )
        
        optimizer.best_configs[scenario_name] = best_config
    
    # Save all configs
    optimizer.save_best_configs()
    
    return optimizer.best_configs


async def main():
    """Run hyperparameter optimization"""
    
    print("=" * 70)
    print("PHASE 2: HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    # Optimize all scenarios
    best_configs = await optimize_all_scenarios(
        output_dir,
        n_trials=50,
        n_validation_trials=5
    )
    
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print(f"\nOptimal configurations found for {len(best_configs)} scenarios")
    print(f"Results saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    asyncio.run(main())

