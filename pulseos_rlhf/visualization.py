"""
Week 4 Day 22-25: Visualization

Generates charts and plots for comparison across all methods.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_results():
    """Load evaluation results."""
    results_path = Path("pulseos_rlhf/evaluation_results.json")
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    return data['all_results'], data['analysis']

def plot_comparison(analysis):
    """Plot comparison across all methods."""
    methods = list(analysis.keys())
    means = [analysis[m]['mean'] for m in methods]
    stds = [analysis[m]['std'] for m in methods]
    
    method_labels = [m.replace('_', ' ').title() for m in methods]
    
    plt.figure(figsize=(12, 6))
    plt.bar(method_labels, means, yerr=stds, capsize=10, alpha=0.7, color=['blue', 'red', 'green', 'orange', 'purple'][:len(methods)])
    plt.ylabel('Samples to Convergence')
    plt.title('Sample Efficiency Comparison Across Methods')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    # Add improvement percentages
    baseline_mean = analysis['baseline']['mean']
    for i, method in enumerate(methods[1:], 1):
        improvement = analysis[method]['improvement_percent']
        plt.text(i, means[i] + stds[i] + 50, f'{improvement:.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    output_path = Path("pulseos_rlhf/method_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to {output_path}")

def plot_learning_curves(all_results):
    """Plot learning curves for all methods."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    methods = list(all_results.keys())
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        
        # Plot individual trials
        for trial_result in all_results[method]:
            if 'rewards_history' in trial_result and 'samples_history' in trial_result:
                ax.plot(trial_result['samples_history'], trial_result['rewards_history'], 
                       alpha=0.2, linewidth=1)
        
        # Plot average
        if all_results[method] and 'rewards_history' in all_results[method][0]:
            # Align all histories
            max_len = max(len(r.get('rewards_history', [])) for r in all_results[method])
            avg_history = []
            for i in range(max_len):
                values = [r['rewards_history'][i] for r in all_results[method] 
                         if i < len(r.get('rewards_history', []))]
                if values:
                    avg_history.append(np.mean(values))
            
            if avg_history:
                ax.plot(range(len(avg_history)), avg_history, 'b-', linewidth=2, label='Average')
        
        ax.set_xlabel('Samples')
        ax.set_ylabel('Reward')
        ax.set_title(method.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Remove extra subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    output_path = Path("pulseos_rlhf/learning_curves.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved learning curves to {output_path}")

def plot_improvement_summary(analysis):
    """Plot improvement summary."""
    methods = [m for m in analysis.keys() if m != 'baseline']
    improvements = [analysis[m]['improvement_percent'] for m in methods]
    p_values = [analysis[m]['p_value'] for m in methods]
    significant = [analysis[m]['significant'] for m in methods]
    
    colors = ['green' if sig else 'red' for sig in significant]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(methods)), improvements, color=colors, alpha=0.7)
    
    # Add p-value labels
    for i, (bar, p_val) in enumerate(zip(bars, p_values)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'p={p_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.axhline(y=20, color='green', linestyle='--', linewidth=1, label='20% Target')
    plt.ylabel('Improvement (%)')
    plt.xlabel('Method')
    plt.title('Improvement Over Baseline PPO')
    plt.xticks(range(len(methods)), [m.replace('_', ' ').title() for m in methods], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path("pulseos_rlhf/improvement_summary.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved improvement summary to {output_path}")

def main():
    print("=" * 80)
    print("Generating Visualizations")
    print("=" * 80)
    print()
    
    all_results, analysis = load_results()
    
    print("1. Plotting method comparison...")
    plot_comparison(analysis)
    
    print("2. Plotting learning curves...")
    plot_learning_curves(all_results)
    
    print("3. Plotting improvement summary...")
    plot_improvement_summary(analysis)
    
    print()
    print("=" * 80)
    print("Visualization Complete")
    print("=" * 80)

if __name__ == "__main__":
    main()


