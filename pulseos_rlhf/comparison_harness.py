"""
Comparison Harness for Baseline vs Death Penalty

Runs multiple trials of each method and compares results.
"""

import os
import numpy as np
from pathlib import Path
from datasets import load_dataset
from baseline_ppo import BaselinePPORLHFTrainer
from death_penalty import PPOWithDeathPenalty
import json
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
    
    return base_config

def run_trial(method, trial_num, dataset, config):
    """Run a single trial."""
    print(f"  {method} Trial {trial_num}: Starting...")
    
    if method == "baseline":
        trainer = BaselinePPORLHFTrainer(config)
    elif method == "death_penalty":
        trainer = PPOWithDeathPenalty(config)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    results = trainer.train(dataset)
    
    print(f"    Completed: {results['samples_to_convergence']} samples, converged={results['converged']}")
    
    return results

def main():
    print("=" * 80)
    print("Comparison Harness: Baseline vs Death Penalty")
    print("=" * 80)
    print()
    
    num_trials = 5
    
    # Load dataset
    print("Loading dataset...")
    try:
        dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        dataset = dataset.select(range(2000))  # More samples for multiple trials
        print(f"✓ Loaded {len(dataset)} samples")
    except Exception as e:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if hf_token:
            dataset = load_dataset("Anthropic/hh-rlhf", split="train", token=hf_token)
            dataset = dataset.select(range(2000))
            print(f"✓ Loaded {len(dataset)} samples")
        else:
            raise
    print()
    
    # Run baseline trials
    print("Running baseline PPO trials...")
    baseline_results = []
    baseline_config = load_config("baseline")
    
    for trial in range(num_trials):
        result = run_trial("baseline", trial + 1, dataset, baseline_config)
        baseline_results.append(result)
    
    print()
    
    # Run death penalty trials
    print("Running death penalty PPO trials...")
    death_penalty_results = []
    death_penalty_config = load_config("death_penalty")
    
    for trial in range(num_trials):
        result = run_trial("death_penalty", trial + 1, dataset, death_penalty_config)
        death_penalty_results.append(result)
    
    print()
    
    # Statistical analysis
    baseline_samples = [r['samples_to_convergence'] for r in baseline_results]
    death_penalty_samples = [r['samples_to_convergence'] for r in death_penalty_results]
    
    baseline_mean = np.mean(baseline_samples)
    baseline_std = np.std(baseline_samples)
    death_penalty_mean = np.mean(death_penalty_samples)
    death_penalty_std = np.std(death_penalty_samples)
    
    improvement = ((baseline_mean - death_penalty_mean) / baseline_mean) * 100
    
    # T-test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(baseline_samples, death_penalty_samples)
    
    # Cohen's d
    pooled_std = np.sqrt((baseline_std**2 + death_penalty_std**2) / 2)
    cohens_d = (baseline_mean - death_penalty_mean) / pooled_std if pooled_std > 0 else 0
    
    # Results
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"\nBaseline PPO:")
    print(f"  Mean samples: {baseline_mean:.1f} ± {baseline_std:.1f}")
    print(f"  Individual: {baseline_samples}")
    
    print(f"\nDeath Penalty PPO:")
    print(f"  Mean samples: {death_penalty_mean:.1f} ± {death_penalty_std:.1f}")
    print(f"  Individual: {death_penalty_samples}")
    
    print(f"\n🎯 IMPROVEMENT: {improvement:.1f}%")
    print(f"\nStatistical Analysis:")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print("=" * 80)
    
    # Save results
    output_path = Path("pulseos_rlhf/comparison_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            "baseline_results": baseline_results,
            "death_penalty_results": death_penalty_results,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "death_penalty_mean": death_penalty_mean,
            "death_penalty_std": death_penalty_std,
            "improvement_percent": improvement,
            "p_value": p_value,
            "cohens_d": cohens_d,
        }, f, indent=2)
    
    print(f"\nSaved results to {output_path}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    methods = ['Baseline PPO', 'Death Penalty PPO']
    means = [baseline_mean, death_penalty_mean]
    stds = [baseline_std, death_penalty_std]
    
    plt.bar(methods, means, yerr=stds, capsize=10, color=['blue', 'red'], alpha=0.7)
    plt.ylabel('Samples to Convergence')
    plt.title(f'Sample Efficiency Comparison\n({improvement:.1f}% improvement)')
    plt.grid(True, alpha=0.3, axis='y')
    
    plot_path = Path("pulseos_rlhf/comparison_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()


