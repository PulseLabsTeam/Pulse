"""
Config 2 Large-Scale Test: Reproducing the 4.259 Sharpe Result

Run 50 trials of Config 2 (no warm start) to determine:
1. Success rate of achieving 4.0+ Sharpe
2. What makes successful runs different
3. Whether we can reproduce the 4.259 Sharpe result
"""

import asyncio
import numpy as np
from datetime import datetime
from trading_rl_test import run_pulseos_trial, download_stock_data
from trading_env import TradingEnv
import json

async def run_config2_large_scale_test():
    """
    Run 50 trials of Config 2 to analyze success patterns.
    """
    print("=" * 80)
    print("CONFIG 2 LARGE-SCALE TEST: Reproducing 4.259 Sharpe Result")
    print("=" * 80)
    print("\nConfiguration:")
    print("  - NO Warm Start (independent training)")
    print("  - Death Penalty Schedule:")
    print("    * Episodes 0-150: -0.25 (very mild)")
    print("    * Episodes 150-300: -1.0 (moderate)")
    print("    * Episodes 300+: -3.0 (moderate-high)")
    print("  - Survival Signal: Exponential relaxation (aggressive)")
    print("  - Episodes: 500")
    print("  - Trials: 50")
    print("=" * 80)
    print("\nGoal: Determine success rate and identify patterns")
    print("=" * 80)
    
    # Download data
    symbol = "SPY"
    data = download_stock_data(symbol)
    
    # Run PPO baseline first
    print("\n📊 Step 1: Running PPO Baseline...")
    print("-" * 80)
    from trading_rl_test import run_ppo_trial
    ppo_results = []
    for trial in range(1, 6):
        env = TradingEnv(data, initial_capital=100000.0, commission=0.001)
        result = await run_ppo_trial(trial, env, 500, 1.5, 0.15)
        ppo_results.append(result)
    
    ppo_baseline_sharpe = np.mean([r.final_sharpe for r in ppo_results])
    print(f"\n✅ PPO Baseline: {ppo_baseline_sharpe:.3f}")
    print("-" * 80)
    
    # Run 50 Config 2 trials
    print("\n🚀 Step 2: Running 50 Config 2 Trials...")
    print("-" * 80)
    
    results = []
    successful_runs = []  # Trials that hit 4.0+ Sharpe
    failed_runs = []  # Trials below 3.0 Sharpe
    
    for trial in range(1, 51):
        print(f"\nTrial {trial}/50...")
        env = TradingEnv(data, initial_capital=100000.0, commission=0.001)
        
        # Use trial number as seed for reproducibility
        seed = trial
        
        result = await run_pulseos_trial(
            trial, env, 500, 1.5, 0.15,
            seed=seed,  # Use trial number as seed
            ppo_baseline_sharpe=ppo_baseline_sharpe,
            death_penalty_multiplier=-5.0  # Base penalty (progressive schedule applies)
        )
        
        results.append({
            "trial": trial,
            "seed": seed,
            "final_sharpe": result.final_sharpe,
            "beats_ppo": result.final_sharpe > ppo_baseline_sharpe,
            "excellent": result.final_sharpe >= 4.0,
            "poor": result.final_sharpe < 3.0
        })
        
        if result.final_sharpe >= 4.0:
            successful_runs.append(trial)
            print(f"  ✅ EXCELLENT: {result.final_sharpe:.3f} Sharpe")
        elif result.final_sharpe >= ppo_baseline_sharpe:
            print(f"  ✅ Good: {result.final_sharpe:.3f} Sharpe (beats PPO)")
        elif result.final_sharpe < 3.0:
            failed_runs.append(trial)
            print(f"  ❌ Poor: {result.final_sharpe:.3f} Sharpe")
        else:
            print(f"  ⚠️  Below PPO: {result.final_sharpe:.3f} Sharpe")
        
        # Progress update every 10 trials
        if trial % 10 == 0:
            current_success_rate = len(successful_runs) / trial * 100
            print(f"\n  Progress: {trial}/50 trials")
            print(f"  Success Rate (4.0+ Sharpe): {current_success_rate:.1f}% ({len(successful_runs)}/{trial})")
    
    # Analyze results
    sharpes = [r["final_sharpe"] for r in results]
    avg_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes)
    
    success_rate = len(successful_runs) / 50 * 100
    beats_ppo_rate = sum(1 for r in results if r["beats_ppo"]) / 50 * 100
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"\nPPO Baseline: {ppo_baseline_sharpe:.3f}")
    print(f"\nConfig 2 Results (50 trials):")
    print(f"  Average Sharpe: {avg_sharpe:.3f} ± {std_sharpe:.3f}")
    print(f"  Improvement vs PPO: {((avg_sharpe - ppo_baseline_sharpe) / ppo_baseline_sharpe) * 100:+.1f}%")
    print(f"  Success Rate (4.0+ Sharpe): {success_rate:.1f}% ({len(successful_runs)}/50)")
    print(f"  Beats PPO Rate: {beats_ppo_rate:.1f}% ({sum(1 for r in results if r['beats_ppo'])}/50)")
    print(f"  Poor Runs (<3.0 Sharpe): {len(failed_runs)}/50")
    
    print(f"\nSuccessful Runs (4.0+ Sharpe):")
    for trial_num in successful_runs:
        sharpe = next(r["final_sharpe"] for r in results if r["trial"] == trial_num)
        print(f"  Trial {trial_num}: {sharpe:.3f} Sharpe")
    
    # Top 10 trials
    sorted_results = sorted(results, key=lambda x: x["final_sharpe"], reverse=True)
    print(f"\nTop 10 Trials:")
    for i, r in enumerate(sorted_results[:10], 1):
        print(f"  {i}. Trial {r['trial']} (seed {r['seed']}): {r['final_sharpe']:.3f} Sharpe")
    
    # Success criteria analysis
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA ANALYSIS")
    print("=" * 80)
    
    if success_rate >= 40:
        print("✅ EXCELLENT: 40%+ success rate → Worth $30M-$50M")
        print("   Ready for other domains testing")
    elif success_rate >= 20:
        print("✅ GOOD: 20-40% success rate → Worth $15M-$30M")
        print("   Need to optimize initialization")
    elif success_rate >= 10:
        print("⚠️  MODERATE: 10-20% success rate → Worth $10M-$20M")
        print("   Need significant optimization")
    else:
        print("❌ POOR: <10% success rate → Worth $5M-$9M")
        print("   Consider Config 3 approach or hybrid")
    
    # Save results
    output_dir = "benchmark_results/trading_rl/config2_large_scale"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    json_path = f"{output_dir}/config2_50trials_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "ppo_baseline": ppo_baseline_sharpe,
            "results": results,
            "statistics": {
                "avg_sharpe": avg_sharpe,
                "std_sharpe": std_sharpe,
                "success_rate": success_rate,
                "beats_ppo_rate": beats_ppo_rate,
                "successful_runs": successful_runs,
                "failed_runs": failed_runs
            }
        }, f, indent=2)
    
    # Save markdown report
    report_path = f"{output_dir}/config2_50trials_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write("# Config 2 Large-Scale Test Results\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **PPO Baseline**: {ppo_baseline_sharpe:.3f}\n")
        f.write(f"- **Average Sharpe**: {avg_sharpe:.3f} ± {std_sharpe:.3f}\n")
        f.write(f"- **Success Rate (4.0+)**: {success_rate:.1f}% ({len(successful_runs)}/50)\n")
        f.write(f"- **Beats PPO Rate**: {beats_ppo_rate:.1f}%\n\n")
        f.write("## Successful Runs (4.0+ Sharpe)\n\n")
        for trial_num in successful_runs:
            sharpe = next(r["final_sharpe"] for r in results if r["trial"] == trial_num)
            f.write(f"- Trial {trial_num} (seed {trial_num}): {sharpe:.3f} Sharpe\n")
        f.write("\n## Top 10 Trials\n\n")
        for i, r in enumerate(sorted_results[:10], 1):
            f.write(f"{i}. Trial {r['trial']} (seed {r['seed']}): {r['final_sharpe']:.3f} Sharpe\n")
    
    print(f"\n✅ Results saved to:")
    print(f"   JSON: {json_path}")
    print(f"   Report: {report_path}")
    
    return results, successful_runs, failed_runs

if __name__ == "__main__":
    asyncio.run(run_config2_large_scale_test())



