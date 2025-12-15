"""
Hyperparameter Optimization for TRUE PulseOS Death Penalty

Tests different death penalty values to find optimal configuration that beats PPO baseline.
"""

import asyncio
import json
import os
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from datetime import datetime

# Import test functions
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trading_rl_test import run_trading_test, TestResults

@dataclass
class HyperparameterResult:
    """Results from hyperparameter test"""
    death_penalty: float
    ppo_avg_sharpe: float
    pulseos_avg_sharpe: float
    improvement: float  # Percentage improvement over PPO
    pulseos_std: float
    ppo_std: float
    success_rate: float  # % of trials >= 1.5 Sharpe
    high_performance_rate: float  # % of trials >= 3.5 Sharpe
    trials: int
    episodes: int


async def test_death_penalty_config(
    death_penalty: float,
    episodes: int = 200,
    trials: int = 3
) -> HyperparameterResult:
    """
    Test a specific death penalty configuration.
    
    Args:
        death_penalty: Magnitude of death penalty (negative value)
        episodes: Number of episodes per trial
        trials: Number of trials to run
        
    Returns:
        HyperparameterResult with test results
    """
    print("\n" + "=" * 80)
    print(f"Testing Death Penalty: {death_penalty:.1f}")
    print(f"Episodes: {episodes}, Trials: {trials}")
    print("=" * 80)
    
    # Run test with this death penalty
    results = await run_trading_test(
        symbol="SPY",
        num_trials=trials,
        max_episodes=episodes,
        target_sharpe=1.5,
        target_return=0.15,
        test_mode="standard",
        death_penalty_multiplier=death_penalty
    )
    
    # Calculate statistics
    pulseos_sharpes = [r.final_sharpe for r in results.pulseos_results]
    ppo_sharpes = [r.final_sharpe for r in results.ppo_results]
    
    pulseos_avg = np.mean(pulseos_sharpes)
    ppo_avg = np.mean(ppo_sharpes)
    pulseos_std = np.std(pulseos_sharpes)
    ppo_std = np.std(ppo_sharpes)
    
    improvement = ((pulseos_avg - ppo_avg) / ppo_avg) * 100
    success_rate = sum(1 for s in pulseos_sharpes if s >= 1.5) / len(pulseos_sharpes) * 100
    high_performance_rate = sum(1 for s in pulseos_sharpes if s >= 3.5) / len(pulseos_sharpes) * 100
    
    result = HyperparameterResult(
        death_penalty=death_penalty,
        ppo_avg_sharpe=ppo_avg,
        pulseos_avg_sharpe=pulseos_avg,
        improvement=improvement,
        pulseos_std=pulseos_std,
        ppo_std=ppo_std,
        success_rate=success_rate,
        high_performance_rate=high_performance_rate,
        trials=trials,
        episodes=episodes
    )
    
    print(f"\n✅ Results:")
    print(f"   PPO Avg: {ppo_avg:.3f} ± {ppo_std:.3f}")
    print(f"   PulseOS Avg: {pulseos_avg:.3f} ± {pulseos_std:.3f}")
    print(f"   Improvement: {improvement:+.1f}%")
    print(f"   Success Rate: {success_rate:.1f}%")
    print(f"   High Performance Rate: {high_performance_rate:.1f}%")
    
    return result


async def run_hyperparameter_sweep():
    """
    Run systematic hyperparameter sweep to find optimal death penalty.
    """
    print("\n" + "=" * 80)
    print("TRUE PulseOS Hyperparameter Optimization")
    print("=" * 80)
    print("\nTesting death penalty values to find optimal configuration")
    print("Goal: Beat PPO baseline (target: +10% or better)")
    print("=" * 80)
    
    # Test configurations (as suggested by tech review)
    experiments = [
        {"death_penalty": -100.0, "episodes": 200, "trials": 3},  # Baseline (current)
        {"death_penalty": -50.0, "episodes": 200, "trials": 3},
        {"death_penalty": -25.0, "episodes": 200, "trials": 3},
        {"death_penalty": -10.0, "episodes": 200, "trials": 3},
        {"death_penalty": -5.0, "episodes": 200, "trials": 3},
    ]
    
    results = []
    
    for exp in experiments:
        result = await test_death_penalty_config(**exp)
        results.append(result)
        
        # Save intermediate results
        output_dir = "benchmark_results/trading_rl/hyperparameter_sweep"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/sweep_results.json", "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
    
    # Find best configuration
    best_result = max(results, key=lambda r: r.pulseos_avg_sharpe)
    
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SWEEP RESULTS")
    print("=" * 80)
    
    print("\nAll Configurations:")
    print("-" * 80)
    for r in results:
        status = "✅ BEATS PPO" if r.improvement > 0 else "❌ Below PPO"
        print(f"Death Penalty {r.death_penalty:6.1f}: "
              f"PulseOS={r.pulseos_avg_sharpe:.3f} ± {r.pulseos_std:.3f}, "
              f"PPO={r.ppo_avg_sharpe:.3f} ± {r.ppo_std:.3f}, "
              f"Improvement={r.improvement:+.1f}% {status}")
    
    print("\n" + "=" * 80)
    print(f"🏆 BEST CONFIGURATION")
    print("=" * 80)
    print(f"Death Penalty: {best_result.death_penalty:.1f}")
    print(f"PulseOS Avg Sharpe: {best_result.pulseos_avg_sharpe:.3f} ± {best_result.pulseos_std:.3f}")
    print(f"PPO Avg Sharpe: {best_result.ppo_avg_sharpe:.3f} ± {best_result.ppo_std:.3f}")
    print(f"Improvement: {best_result.improvement:+.1f}%")
    print(f"Success Rate: {best_result.success_rate:.1f}%")
    print(f"High Performance Rate: {best_result.high_performance_rate:.1f}%")
    
    if best_result.improvement > 0:
        print(f"\n🎉 SUCCESS! PulseOS beats PPO by {best_result.improvement:.1f}%")
    else:
        print(f"\n⚠️  Still below PPO baseline. Consider:")
        print(f"   1. Testing survival signal relaxation")
        print(f"   2. Increasing episodes to 500-600")
        print(f"   3. Testing intermediate values (e.g., -15.0, -20.0)")
    
    # Save final results
    output_dir = "benchmark_results/trading_rl/hyperparameter_sweep"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{output_dir}/HYPERPARAMETER_SWEEP_REPORT_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write("# TRUE PulseOS Hyperparameter Sweep Results\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Summary\n\n")
        f.write(f"**Best Configuration**: Death Penalty = {best_result.death_penalty:.1f}\n\n")
        f.write(f"**Results**:\n")
        f.write(f"- PulseOS Avg: {best_result.pulseos_avg_sharpe:.3f} ± {best_result.pulseos_std:.3f}\n")
        f.write(f"- PPO Avg: {best_result.ppo_avg_sharpe:.3f} ± {best_result.ppo_std:.3f}\n")
        f.write(f"- Improvement: {best_result.improvement:+.1f}%\n")
        f.write(f"- Success Rate: {best_result.success_rate:.1f}%\n")
        f.write(f"- High Performance Rate: {best_result.high_performance_rate:.1f}%\n\n")
        f.write("## All Configurations\n\n")
        f.write("| Death Penalty | PulseOS Avg | PulseOS Std | PPO Avg | PPO Std | Improvement | Status |\n")
        f.write("|---------------|-------------|-------------|---------|---------|-------------|--------|\n")
        for r in results:
            status = "✅ BEATS PPO" if r.improvement > 0 else "❌ Below PPO"
            f.write(f"| {r.death_penalty:6.1f} | {r.pulseos_avg_sharpe:.3f} | {r.pulseos_std:.3f} | "
                   f"{r.ppo_avg_sharpe:.3f} | {r.ppo_std:.3f} | {r.improvement:+.1f}% | {status} |\n")
    
    print(f"\n✅ Results saved to {report_path}")
    
    return best_result, results


if __name__ == "__main__":
    asyncio.run(run_hyperparameter_sweep())
