"""
Config 2 Trial 6: Optimized Initialization + Seed 6

Run Trial 6 with optimized initialization (based on seed 1 analysis).
"""

import asyncio
import numpy as np
from datetime import datetime
from trading_rl_test import run_pulseos_trial, download_stock_data, run_ppo_trial
from trading_env import TradingEnv
import json
import os

async def run_config2_trial6():
    """
    Run Config 2 trial 6 with seed 6 and optimized initialization.
    """
    print("=" * 80)
    print("CONFIG 2 TRIAL 6: Optimized Initialization + Seed 6")
    print("=" * 80)
    print("\nConfiguration:")
    print("  - NO Warm Start (independent training)")
    print("  - OPTIMIZED Initialization (based on seed 1 analysis):")
    print("    * Policy weights: 0.35x multiplier (compromise between 0.3x and 0.5x)")
    print("    * Adaptive bias: 0.005 scale (smaller, closer to seed 1's zero bias)")
    print("    * Value weights: 0.25x multiplier (slightly larger)")
    print("  - Death Penalty Schedule:")
    print("    * Episodes 0-150: -0.25 (very mild)")
    print("    * Episodes 150-300: -1.0 (moderate)")
    print("    * Episodes 300+: -3.0 (moderate-high)")
    print("  - Survival Signal: Exponential relaxation (aggressive)")
    print("  - Episodes: 1000")
    print("  - Seed: 6")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # Download data
    symbol = "SPY"
    data = download_stock_data(symbol)
    
    # Use same PPO baseline
    print("\n📊 Using PPO Baseline: 3.625 (from previous trials)")
    print("-" * 80)
    ppo_baseline_sharpe = 3.625
    
    # Run Config 2 trial with seed 6 and optimized initialization
    print("\n🚀 Running Config 2 Trial 6 (1000 episodes, seed 6, optimized init)...")
    print("-" * 80)
    print("This will take approximately 10-15 minutes...")
    print("-" * 80)
    
    env = TradingEnv(data, initial_capital=100000.0, commission=0.001)
    
    result = await run_pulseos_trial(
        6, env, 1000, 1.5, 0.15,
        seed=6,  # Seed 6
        ppo_baseline_sharpe=ppo_baseline_sharpe,
        death_penalty_multiplier=-5.0
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    # Results
    pulseos_sharpe = result.final_sharpe
    improvement = ((pulseos_sharpe - ppo_baseline_sharpe) / ppo_baseline_sharpe) * 100
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nPPO Baseline: {ppo_baseline_sharpe:.3f}")
    print(f"\nConfig 2 Trial 6 (seed 6, optimized init):")
    print(f"  Final Sharpe: {pulseos_sharpe:.3f}")
    print(f"  Improvement: {improvement:+.1f}%")
    
    if pulseos_sharpe >= 4.0:
        print(f"\n  ✅ EXCELLENT: Achieved 4.0+ Sharpe!")
    elif pulseos_sharpe >= ppo_baseline_sharpe:
        print(f"\n  ✅ SUCCESS: Beats PPO baseline!")
    elif pulseos_sharpe >= 3.5:
        print(f"\n  ✅ Good: Competitive performance")
    else:
        print(f"\n  ⚠️  Below PPO baseline")
    
    print(f"\nTest Duration: {duration:.1f} minutes")
    
    # Comparison to previous trials
    print("\n" + "=" * 80)
    print("COMPARISON TO PREVIOUS TRIALS")
    print("=" * 80)
    print("\nPrevious Results:")
    print("  Trial 1 (500 ep, unknown seed): 4.259 Sharpe (+15.9% vs PPO)")
    print("  Trial 1 (1000 ep, seed 1, standard init): 3.688 Sharpe (+1.7% vs PPO)")
    print("  Trial 2 (1000 ep, seed 2, standard init): 2.053 Sharpe (-43.4% vs PPO)")
    print("  Trial 3 (1000 ep, seed 3, improved init): 3.077 Sharpe (-15.1% vs PPO)")
    print("  Trial 4 (1000 ep, seed 4, improved init): 3.536 Sharpe (-2.5% vs PPO)")
    print("  Trial 5 (1000 ep, seed 5, optimized init): 3.648 Sharpe (+0.6% vs PPO)")
    print(f"\nTrial 6 (1000 ep, seed 6, optimized init):")
    print(f"  Final Sharpe: {pulseos_sharpe:.3f}")
    print(f"  Improvement: {improvement:+.1f}% vs PPO")
    
    # Statistics across all trials
    all_trials = [
        {"trial": "1 (prev)", "seed": "Unknown", "sharpe": 4.259, "episodes": 500, "init": "Standard"},
        {"trial": "1", "seed": 1, "sharpe": 3.688, "episodes": 1000, "init": "Standard"},
        {"trial": "2", "seed": 2, "sharpe": 2.053, "episodes": 1000, "init": "Standard"},
        {"trial": "3", "seed": 3, "sharpe": 3.077, "episodes": 1000, "init": "Improved"},
        {"trial": "4", "seed": 4, "sharpe": 3.536, "episodes": 1000, "init": "Improved"},
        {"trial": "5", "seed": 5, "sharpe": 3.648, "episodes": 1000, "init": "Optimized"},
        {"trial": "6", "seed": 6, "sharpe": pulseos_sharpe, "episodes": 1000, "init": "Optimized"}
    ]
    
    # Filter to 1000-episode trials for fair comparison
    trials_1000ep = [t for t in all_trials if t["episodes"] == 1000]
    sharpes_1000ep = [t["sharpe"] for t in trials_1000ep]
    
    # Filter by initialization type
    trials_standard_init = [t for t in trials_1000ep if t["init"] == "Standard"]
    trials_improved_init = [t for t in trials_1000ep if t["init"] == "Improved"]
    trials_optimized_init = [t for t in trials_1000ep if t["init"] == "Optimized"]
    
    sharpes_standard_init = [t["sharpe"] for t in trials_standard_init]
    sharpes_improved_init = [t["sharpe"] for t in trials_improved_init]
    sharpes_optimized_init = [t["sharpe"] for t in trials_optimized_init]
    
    print(f"\nStatistics (1000-episode trials):")
    print(f"  Average Sharpe: {np.mean(sharpes_1000ep):.3f} ± {np.std(sharpes_1000ep):.3f}")
    print(f"  Beats PPO: {sum(1 for s in sharpes_1000ep if s > ppo_baseline_sharpe)}/{len(sharpes_1000ep)}")
    print(f"  Excellent (4.0+): {sum(1 for s in sharpes_1000ep if s >= 4.0)}/{len(sharpes_1000ep)}")
    
    if len(sharpes_standard_init) > 0:
        print(f"\nStatistics (standard initialization):")
        print(f"  Average Sharpe: {np.mean(sharpes_standard_init):.3f} ± {np.std(sharpes_standard_init):.3f}")
        print(f"  Beats PPO: {sum(1 for s in sharpes_standard_init if s > ppo_baseline_sharpe)}/{len(sharpes_standard_init)}")
    
    if len(sharpes_improved_init) > 0:
        print(f"\nStatistics (improved initialization):")
        print(f"  Average Sharpe: {np.mean(sharpes_improved_init):.3f} ± {np.std(sharpes_improved_init):.3f}")
        print(f"  Beats PPO: {sum(1 for s in sharpes_improved_init if s > ppo_baseline_sharpe)}/{len(sharpes_improved_init)}")
    
    if len(sharpes_optimized_init) > 0:
        print(f"\nStatistics (optimized initialization):")
        print(f"  Average Sharpe: {np.mean(sharpes_optimized_init):.3f} ± {np.std(sharpes_optimized_init):.3f}")
        print(f"  Beats PPO: {sum(1 for s in sharpes_optimized_init if s > ppo_baseline_sharpe)}/{len(sharpes_optimized_init)}")
        print(f"  Success Rate: {sum(1 for s in sharpes_optimized_init if s > ppo_baseline_sharpe)}/{len(sharpes_optimized_init)} ({sum(1 for s in sharpes_optimized_init if s > ppo_baseline_sharpe)/len(sharpes_optimized_init)*100:.0f}%)")
    
    if pulseos_sharpe > max([t["sharpe"] for t in trials_1000ep if t["trial"] != "6"]):
        print(f"\n  🎉 BEST 1000-episode result so far!")
    elif pulseos_sharpe > 3.688:
        print(f"\n  ✅ Better than Trial 1 (seed 1): +{pulseos_sharpe - 3.688:.3f} Sharpe")
    elif pulseos_sharpe >= ppo_baseline_sharpe:
        print(f"\n  ✅ Beats PPO baseline")
    else:
        print(f"\n  ⚠️  Below PPO baseline")
    
    # Save results
    output_dir = "benchmark_results/trading_rl/config2_large_scale"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_path = f"{output_dir}/config2_1000ep_trial6_seed6_optimized_init_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "test_config": {
                "episodes": 1000,
                "trials": 1,
                "seed": 6,
                "no_warm_start": True,
                "optimized_initialization": True,
                "init_details": {
                    "policy_multiplier": 0.35,
                    "bias_scale": 0.005,
                    "value_multiplier": 0.25
                }
            },
            "ppo_baseline": {
                "final_sharpe": ppo_baseline_sharpe,
                "episodes": 1000
            },
            "pulseos_result": {
                "final_sharpe": float(pulseos_sharpe),
                "improvement_vs_ppo": float(improvement),
                "beats_ppo": bool(pulseos_sharpe > ppo_baseline_sharpe),
                "excellent": bool(pulseos_sharpe >= 4.0),
                "exceeds_previous_best": bool(pulseos_sharpe > 4.259)
            },
            "all_trials": all_trials,
            "statistics_1000ep": {
                "avg_sharpe": float(np.mean(sharpes_1000ep)),
                "std_sharpe": float(np.std(sharpes_1000ep)),
                "beats_ppo_rate": sum(1 for s in sharpes_1000ep if s > ppo_baseline_sharpe) / len(sharpes_1000ep),
                "excellent_rate": sum(1 for s in sharpes_1000ep if s >= 4.0) / len(sharpes_1000ep)
            },
            "statistics_by_init": {
                "standard": {
                    "avg_sharpe": float(np.mean(sharpes_standard_init)) if len(sharpes_standard_init) > 0 else None,
                    "std_sharpe": float(np.std(sharpes_standard_init)) if len(sharpes_standard_init) > 0 else None,
                    "beats_ppo_rate": sum(1 for s in sharpes_standard_init if s > ppo_baseline_sharpe) / len(sharpes_standard_init) if len(sharpes_standard_init) > 0 else None
                },
                "improved": {
                    "avg_sharpe": float(np.mean(sharpes_improved_init)) if len(sharpes_improved_init) > 0 else None,
                    "std_sharpe": float(np.std(sharpes_improved_init)) if len(sharpes_improved_init) > 0 else None,
                    "beats_ppo_rate": sum(1 for s in sharpes_improved_init if s > ppo_baseline_sharpe) / len(sharpes_improved_init) if len(sharpes_improved_init) > 0 else None
                },
                "optimized": {
                    "avg_sharpe": float(np.mean(sharpes_optimized_init)) if len(sharpes_optimized_init) > 0 else None,
                    "std_sharpe": float(np.std(sharpes_optimized_init)) if len(sharpes_optimized_init) > 0 else None,
                    "beats_ppo_rate": sum(1 for s in sharpes_optimized_init if s > ppo_baseline_sharpe) / len(sharpes_optimized_init) if len(sharpes_optimized_init) > 0 else None
                }
            },
            "duration_minutes": duration
        }, f, indent=2)
    
    report_path = f"{output_dir}/config2_1000ep_trial6_seed6_optimized_init_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write("# Config 2 Trial 6: Optimized Initialization + Seed 6\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Test Configuration\n\n")
        f.write("- **Episodes**: 1000\n")
        f.write("- **Seed**: 6\n")
        f.write("- **Optimized Initialization**: Yes (based on seed 1 analysis)\n")
        f.write("  - Policy weights: 0.35x multiplier (compromise between 0.3x and 0.5x)\n")
        f.write("  - Adaptive bias: 0.005 scale (smaller, closer to seed 1's zero bias)\n")
        f.write("  - Value weights: 0.25x multiplier (slightly larger)\n")
        f.write("- **No Warm Start**: Independent training\n\n")
        f.write("## Results\n\n")
        f.write(f"- **PPO Baseline**: {ppo_baseline_sharpe:.3f}\n")
        f.write(f"- **Config 2 Final Sharpe**: {pulseos_sharpe:.3f}\n")
        f.write(f"- **Improvement vs PPO**: {improvement:+.1f}%\n")
        f.write(f"- **Beats PPO**: {'Yes' if pulseos_sharpe > ppo_baseline_sharpe else 'No'}\n")
        f.write(f"- **Exceeds 4.0**: {'Yes' if pulseos_sharpe >= 4.0 else 'No'}\n\n")
        f.write("## All Trials Comparison\n\n")
        f.write("| Trial | Episodes | Seed | Init | Sharpe | Improvement | Status |\n")
        f.write("|-------|----------|------|------|--------|-------------|--------|\n")
        for t in all_trials:
            status = "✅" if t["sharpe"] > ppo_baseline_sharpe else "❌"
            f.write(f"| Trial {t['trial']} | {t['episodes']} | {t['seed']} | {t['init']} | {t['sharpe']:.3f} | {((t['sharpe'] - ppo_baseline_sharpe) / ppo_baseline_sharpe * 100):+.1f}% | {status} |\n")
        f.write(f"\n**Test Duration**: {duration:.1f} minutes\n")
    
    print(f"\n✅ Results saved to:")
    print(f"   JSON: {json_path}")
    print(f"   Report: {report_path}")
    
    return result, ppo_baseline_sharpe

if __name__ == "__main__":
    asyncio.run(run_config2_trial6())



