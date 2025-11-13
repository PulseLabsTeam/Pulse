"""
Week 4 Day 22-25: Evaluation Harness

Runs comprehensive comparisons across all methods:
- Baseline PPO
- Death Penalty
- Population
- Runtime Adaptation
- Full PulseOS

Performs statistical analysis and generates reports.
"""

import os
import numpy as np
import json
from pathlib import Path
from datasets import load_dataset
from baseline_ppo import BaselinePPORLHFTrainer
from death_penalty import PPOWithDeathPenalty
from population_training import PopulationTrainer
from runtime_adaptation import RuntimeAlphaAdaptation
import asyncio
from full_pulseos import FullPulseOSRLHF
from scipy import stats
import matplotlib.pyplot as plt

def load_config(method):
    """Load configuration for a method."""
    base_config = {
        "model_name": "gpt2",
        "reward_model_path": "pulseos_rlhf/reward_model",
        "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        "learning_rate": 1.41e-5,
        "batch_size": 16,
        "mini_batch_size": 4,
        "ppo_epochs": 4,
        "ppo_clip_epsilon": 0.2,
        "max_length": 512,
        "max_new_tokens": 64,
        "target_reward": -1.0,
        "convergence_window": 20,
        "min_samples": 200,
        "max_samples": 5000,
    }
    
    if method == "death_penalty":
        base_config.update({
            "death_threshold": -2.0,
            "death_penalty": -10.0,
        })
    elif method == "population":
        base_config.update({
            "population_size": 5,
            "elimination_rate": 0.2,
        })
    elif method == "runtime":
        base_config.update({
            "death_threshold": -2.0,
            "death_penalty": -10.0,
            "alpha_base": 0.01,
            "alpha_max_change": 0.50,
            "gamma": 0.5,
        })
    elif method == "full_pulseos":
        base_config.update({
            "death_threshold": -2.0,
            "death_penalty": -10.0,
            "survival_threshold": 0.55,
            "alpha_max_change": 0.50,
            "gamma": 0.5,
            "population_size": 5,
            "elimination_rate": 0.2,
        })
    
    return base_config

async def run_trial(method, trial_num, dataset, config):
    """Run a single trial."""
    print(f"  {method} Trial {trial_num}: Starting...")
    
    if method == "baseline":
        trainer = BaselinePPORLHFTrainer(config)
        results = trainer.train(dataset)
    elif method == "death_penalty":
        trainer = PPOWithDeathPenalty(config)
        results = trainer.train(dataset)
    elif method == "population":
        trainer = PopulationTrainer(config)
        results = trainer.train(dataset, max_generations=10, steps_per_generation=100)
        # Convert to samples
        results["samples_to_convergence"] = results.get("total_samples", results.get("samples_to_convergence", 0))
    elif method == "runtime":
        trainer = RuntimeAlphaAdaptation(config)
        results = trainer.train(dataset)
    elif method == "full_pulseos":
        trainer = FullPulseOSRLHF(config)
        results = await trainer.train(dataset)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"    Completed: {results['samples_to_convergence']} samples, converged={results['converged']}")
    
    return results

async def run_evaluation(num_trials=10):
    """Run comprehensive evaluation across all methods."""
    print("=" * 80)
    print("Week 4 Day 22-25: Comprehensive Evaluation")
    print("=" * 80)
    print(f"Running {num_trials} trials per method")
    print()
    
    methods = ["baseline", "death_penalty", "population", "runtime", "full_pulseos"]
    
    # Load dataset
    print("Loading dataset...")
    try:
        dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        dataset = dataset.select(range(5000))  # More samples for multiple trials
        print(f"✓ Loaded {len(dataset)} samples")
    except Exception as e:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            dataset = load_dataset("Anthropic/hh-rlhf", split="train", token=hf_token)
            dataset = dataset.select(range(5000))
            print(f"✓ Loaded {len(dataset)} samples")
        else:
            raise
    print()
    
    # Run trials for each method
    all_results = {}
    
    for method in methods:
        print(f"Running {method} trials...")
        config = load_config(method)
        method_results = []
        
        for trial in range(num_trials):
            result = await run_trial(method, trial + 1, dataset, config)
            method_results.append(result)
        
        all_results[method] = method_results
        print()
    
    # Statistical analysis
    print("=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)
    
    analysis = {}
    baseline_samples = [r['samples_to_convergence'] for r in all_results['baseline']]
    baseline_mean = np.mean(baseline_samples)
    baseline_std = np.std(baseline_samples)
    
    analysis['baseline'] = {
        'mean': baseline_mean,
        'std': baseline_std,
        'samples': baseline_samples,
    }
    
    print(f"\nBaseline PPO:")
    print(f"  Mean: {baseline_mean:.1f} ± {baseline_std:.1f}")
    print(f"  Samples: {baseline_samples}")
    
    for method in methods[1:]:
        method_samples = [r['samples_to_convergence'] for r in all_results[method]]
        method_mean = np.mean(method_samples)
        method_std = np.std(method_samples)
        
        improvement = ((baseline_mean - method_mean) / baseline_mean) * 100
        
        # T-test vs baseline
        t_stat, p_value = stats.ttest_ind(baseline_samples, method_samples)
        
        # Cohen's d
        pooled_std = np.sqrt((baseline_std**2 + method_std**2) / 2)
        cohens_d = (baseline_mean - method_mean) / pooled_std if pooled_std > 0 else 0
        
        analysis[method] = {
            'mean': method_mean,
            'std': method_std,
            'samples': method_samples,
            'improvement_percent': improvement,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cohens_d': cohens_d,
        }
        
        print(f"\n{method.replace('_', ' ').title()}:")
        print(f"  Mean: {method_mean:.1f} ± {method_std:.1f}")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
        print(f"  Cohen's d: {cohens_d:.3f}")
        print(f"  Samples: {method_samples}")
    
    # Save results
    output_path = Path("pulseos_rlhf/evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'all_results': all_results,
            'analysis': analysis,
        }, f, indent=2)
    
    print(f"\nSaved results to {output_path}")
    
    return all_results, analysis

if __name__ == "__main__":
    asyncio.run(run_evaluation(num_trials=10))


