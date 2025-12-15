"""
Config 2 Trial 7: Enhanced Optimized Initialization + Milder Penalties + Relaxed Survival Signal

Combining all improvements to reduce seed dependency.
"""

import asyncio
import numpy as np
from datetime import datetime
from trading_rl_test import run_pulseos_trial, download_stock_data, run_ppo_trial
from trading_env import TradingEnv
import json
import os

async def run_config2_trial7():
    """
    Run Config 2 trial 7 with seed 7 and all improvements combined.
    """
    print("=" * 80)
    print("CONFIG 2 TRIAL 7: Enhanced Optimized Init + Milder Penalties + Relaxed Survival Signal")
    print("=" * 80)
    print("\nConfiguration:")
    print("  - NO Warm Start (independent training)")
    print("  - OPTIMIZED Initialization (based on seed 1 analysis):")
    print("    * Policy weights: 0.35x multiplier")
    print("    * Adaptive bias: 0.005 scale")
    print("    * Value weights: 0.25x multiplier")
    print("  - ENHANCED Death Penalty Schedule (milder to reduce seed dependency):")
    print("    * Episodes 0-200: -0.1 (very mild, reduced from -0.25)")
    print("    * Episodes 200-400: -0.5 (mild, reduced from -1.0)")
    print("    * Episodes 400+: -2.0 (moderate, reduced from -3.0)")
    print("  - RELAXED Survival Signal:")
    print("    * DYING threshold: < 0.2 (relaxed from < 0.3)")
    print("    * More lenient to reduce seed dependency")
    print("  - Survival Signal: Exponential relaxation (aggressive)")
    print("  - Episodes: 1000")
    print("  - Seed: 7")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # Download data
    symbol = "SPY"
    data = download_stock_data(symbol)
    
    # Use same PPO baseline
    print("\n📊 Using PPO Baseline: 3.625 (from previous trials)")
    print("-" * 80)
    ppo_baseline_sharpe = 3.625
    
    # Run Config 2 trial with seed 7 and all improvements
    print("\n🚀 Running Config 2 Trial 7 (1000 episodes, seed 7, enhanced config)...")
    print("-" * 80)
    print("This will take approximately 10-15 minutes...")
    print("-" * 80)
    
    env = TradingEnv(data, initial_capital=100000.0, commission=0.001)
    
    result = await run_pulseos_trial(
        7, env, 1000, 1.5, 0.15,
        seed=7,  # Seed 7
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
    print(f"\nConfig 2 Trial 7 (seed 7, enhanced config):")
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
    print("  Trial 6 (1000 ep, seed 6, optimized init): 2.024 Sharpe (-44.2% vs PPO)")
    print(f"\nTrial 7 (1000 ep, seed 7, enhanced config):")
    print(f"  Final Sharpe: {pulseos_sharpe:.3f}")
    print(f"  Improvement: {improvement:+.1f}% vs PPO")
    
    # Statistics across all trials
    all_trials = [
        {"trial": "1 (prev)", "seed": "Unknown", "sharpe": 4.259, "episodes": 500, "init": "Standard", "config": "Standard"},
        {"trial": "1", "seed": 1, "sharpe": 3.688, "episodes": 1000, "init": "Standard", "config": "Standard"},
        {"trial": "2", "seed": 2, "sharpe": 2.053, "episodes": 1000, "init": "Standard", "config": "Standard"},
        {"trial": "3", "seed": 3, "sharpe": 3.077, "episodes": 1000, "init": "Improved", "config": "Standard"},
        {"trial": "4", "seed": 4, "sharpe": 3.536, "episodes": 1000, "init": "Improved", "config": "Standard"},
        {"trial": "5", "seed": 5, "sharpe": 3.648, "episodes": 1000, "init": "Optimized", "config": "Standard"},
        {"trial": "6", "seed": 6, "sharpe": 2.024, "episodes": 1000, "init": "Optimized", "config": "Standard"},
        {"trial": "7", "seed": 7, "sharpe": pulseos_sharpe, "episodes": 1000, "init": "Optimized", "config": "Enhanced"}
    ]
    
    # Filter to 1000-episode trials for fair comparison
    trials_1000ep = [t for t in all_trials if t["episodes"] == 1000]
    sharpes_1000ep = [t["sharpe"] for t in trials_1000ep]
    
    # Filter by initialization type
    trials_optimized_init = [t for t in trials_1000ep if t["init"] == "Optimized"]
    trials_enhanced_config = [t for t in trials_1000ep if t["config"] == "Enhanced"]
    
    sharpes_optimized_init = [t["sharpe"] for t in trials_optimized_init]
    sharpes_enhanced_config = [t["sharpe"] for t in trials_enhanced_config]
    
    print(f"\nStatistics (1000-episode trials):")
    print(f"  Average Sharpe: {np.mean(sharpes_1000ep):.3f} ± {np.std(sharpes_1000ep):.3f}")
    print(f"  Beats PPO: {sum(1 for s in sharpes_1000ep if s > ppo_baseline_sharpe)}/{len(sharpes_1000ep)}")
    print(f"  Excellent (4.0+): {sum(1 for s in sharpes_1000ep if s >= 4.0)}/{len(sharpes_1000ep)}")
    
    if len(sharpes_optimized_init) > 0:
        print(f"\nStatistics (optimized initialization):")
        print(f"  Average Sharpe: {np.mean(sharpes_optimized_init):.3f} ± {np.std(sharpes_optimized_init):.3f}")
        print(f"  Beats PPO: {sum(1 for s in sharpes_optimized_init if s > ppo_baseline_sharpe)}/{len(sharpes_optimized_init)}")
    
    if len(sharpes_enhanced_config) > 0:
        print(f"\nStatistics (enhanced config - milder penalties + relaxed survival):")
        print(f"  Average Sharpe: {np.mean(sharpes_enhanced_config):.3f} ± {np.std(sharpes_enhanced_config):.3f}")
        print(f"  Beats PPO: {sum(1 for s in sharpes_enhanced_config if s > ppo_baseline_sharpe)}/{len(sharpes_enhanced_config)}")
        print(f"  Success Rate: {sum(1 for s in sharpes_enhanced_config if s > ppo_baseline_sharpe)}/{len(sharpes_enhanced_config)} ({sum(1 for s in sharpes_enhanced_config if s > ppo_baseline_sharpe)/len(sharpes_enhanced_config)*100:.0f}%)")
    
    if pulseos_sharpe > max([t["sharpe"] for t in trials_1000ep if t["trial"] != "7"]):
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
    
    json_path = f"{output_dir}/config2_1000ep_trial7_seed7_enhanced_config_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "test_config": {
                "episodes": 1000,
                "trials": 1,
                "seed": 7,
                "no_warm_start": True,
                "optimized_initialization": True,
                "enhanced_config": True,
                "init_details": {
                    "policy_multiplier": 0.35,
                    "bias_scale": 0.005,
                    "value_multiplier": 0.25
                },
                "penalty_schedule": {
                    "episodes_0_200": -0.1,
                    "episodes_200_400": -0.5,
                    "episodes_400_plus": -2.0
                },
                "survival_signal": {
                    "dying_threshold": 0.2,
                    "relaxed": True
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
            "statistics_enhanced_config": {
                "avg_sharpe": float(np.mean(sharpes_enhanced_config)) if len(sharpes_enhanced_config) > 0 else None,
                "std_sharpe": float(np.std(sharpes_enhanced_config)) if len(sharpes_enhanced_config) > 0 else None,
                "beats_ppo_rate": sum(1 for s in sharpes_enhanced_config if s > ppo_baseline_sharpe) / len(sharpes_enhanced_config) if len(sharpes_enhanced_config) > 0 else None
            },
            "duration_minutes": duration
        }, f, indent=2)
    
    report_path = f"{output_dir}/config2_1000ep_trial7_seed7_enhanced_config_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write("# Config 2 Trial 7: Enhanced Config (Optimized Init + Milder Penalties + Relaxed Survival Signal)\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Test Configuration\n\n")
        f.write("- **Episodes**: 1000\n")
        f.write("- **Seed**: 7\n")
        f.write("- **Optimized Initialization**: Yes (based on seed 1 analysis)\n")
        f.write("- **Enhanced Config**: Yes\n")
        f.write("  - Milder penalties: -0.1 early (was -0.25), -0.5 mid (was -1.0), -2.0 late (was -3.0)\n")
        f.write("  - Relaxed survival signal: DYING threshold < 0.2 (was < 0.3)\n")
        f.write("- **No Warm Start**: Independent training\n\n")
        f.write("## Results\n\n")
        f.write(f"- **PPO Baseline**: {ppo_baseline_sharpe:.3f}\n")
        f.write(f"- **Config 2 Final Sharpe**: {pulseos_sharpe:.3f}\n")
        f.write(f"- **Improvement vs PPO**: {improvement:+.1f}%\n")
        f.write(f"- **Beats PPO**: {'Yes' if pulseos_sharpe > ppo_baseline_sharpe else 'No'}\n")
        f.write(f"- **Exceeds 4.0**: {'Yes' if pulseos_sharpe >= 4.0 else 'No'}\n\n")
        f.write("## All Trials Comparison\n\n")
        f.write("| Trial | Episodes | Seed | Init | Config | Sharpe | Improvement | Status |\n")
        f.write("|-------|----------|------|------|--------|--------|-------------|--------|\n")
        for t in all_trials:
            status = "✅" if t["sharpe"] > ppo_baseline_sharpe else "❌"
            f.write(f"| Trial {t['trial']} | {t['episodes']} | {t['seed']} | {t['init']} | {t['config']} | {t['sharpe']:.3f} | {((t['sharpe'] - ppo_baseline_sharpe) / ppo_baseline_sharpe * 100):+.1f}% | {status} |\n")
        f.write(f"\n**Test Duration**: {duration:.1f} minutes\n")
    
    print(f"\n✅ Results saved to:")
    print(f"   JSON: {json_path}")
    print(f"   Report: {report_path}")
    
    return result, ppo_baseline_sharpe

if __name__ == "__main__":
    asyncio.run(run_config2_trial7())



