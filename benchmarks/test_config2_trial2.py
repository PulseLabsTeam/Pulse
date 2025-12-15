"""
Config 2 Extended Training Test: Trial 2 with Different Seed

Run another 1000-episode trial with seed 2 to test reproducibility.
"""

import asyncio
import numpy as np
from datetime import datetime
from trading_rl_test import run_pulseos_trial, download_stock_data, run_ppo_trial
from trading_env import TradingEnv
import json
import os

async def run_config2_trial2():
    """
    Run Config 2 trial 2 with seed 2 (1000 episodes).
    """
    print("=" * 80)
    print("CONFIG 2 TRIAL 2: 1000 Episodes, Seed 2")
    print("=" * 80)
    print("\nConfiguration:")
    print("  - NO Warm Start (independent training)")
    print("  - Death Penalty Schedule:")
    print("    * Episodes 0-150: -0.25 (very mild)")
    print("    * Episodes 150-300: -1.0 (moderate)")
    print("    * Episodes 300+: -3.0 (moderate-high)")
    print("  - Survival Signal: Exponential relaxation (aggressive)")
    print("  - Episodes: 1000")
    print("  - Seed: 2 (different from Trial 1)")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # Download data
    symbol = "SPY"
    data = download_stock_data(symbol)
    
    # Use same PPO baseline as Trial 1 (or run new one)
    print("\n📊 Using PPO Baseline: 3.625 (from Trial 1)")
    print("-" * 80)
    ppo_baseline_sharpe = 3.625
    
    # Run Config 2 trial with seed 2
    print("\n🚀 Running Config 2 Trial 2 (1000 episodes, seed 2)...")
    print("-" * 80)
    print("This will take approximately 10-15 minutes...")
    print("-" * 80)
    
    env = TradingEnv(data, initial_capital=100000.0, commission=0.001)
    
    result = await run_pulseos_trial(
        2, env, 1000, 1.5, 0.15,
        seed=2,  # Different seed
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
    print(f"\nConfig 2 Trial 2 (seed 2):")
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
    
    # Comparison to Trial 1 and previous best
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print("\nPrevious Results:")
    print("  Config 2 Trial 1 (500 ep, unknown seed): 4.259 Sharpe (+15.9% vs PPO)")
    print("  Config 2 Trial 1 (1000 ep, seed 1): 3.688 Sharpe (+1.7% vs PPO)")
    print(f"\nConfig 2 Trial 2 (1000 ep, seed 2):")
    print(f"  Final Sharpe: {pulseos_sharpe:.3f}")
    print(f"  Improvement: {improvement:+.1f}% vs PPO")
    
    if pulseos_sharpe > 4.259:
        print(f"\n  🎉 EXCEEDED previous best! (+{pulseos_sharpe - 4.259:.3f} Sharpe)")
    elif pulseos_sharpe >= 4.0:
        print(f"\n  ✅ Excellent: Matched/exceeded 4.0 threshold")
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
    
    json_path = f"{output_dir}/config2_1000ep_trial2_seed2_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "test_config": {
                "episodes": 1000,
                "trials": 1,
                "seed": 2,
                "no_warm_start": True
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
            "comparison": {
                "previous_best_500ep": 4.259,
                "trial1_1000ep_seed1": 3.688,
                "trial2_1000ep_seed2": float(pulseos_sharpe),
                "best_so_far": max(4.259, 3.688, float(pulseos_sharpe))
            },
            "duration_minutes": duration
        }, f, indent=2)
    
    report_path = f"{output_dir}/config2_1000ep_trial2_seed2_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write("# Config 2 Trial 2: 1000 Episodes, Seed 2\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Test Configuration\n\n")
        f.write("- **Episodes**: 1000\n")
        f.write("- **Seed**: 2\n")
        f.write("- **No Warm Start**: Independent training\n\n")
        f.write("## Results\n\n")
        f.write(f"- **PPO Baseline**: {ppo_baseline_sharpe:.3f}\n")
        f.write(f"- **Config 2 Final Sharpe**: {pulseos_sharpe:.3f}\n")
        f.write(f"- **Improvement vs PPO**: {improvement:+.1f}%\n")
        f.write(f"- **Beats PPO**: {'Yes' if pulseos_sharpe > ppo_baseline_sharpe else 'No'}\n")
        f.write(f"- **Exceeds 4.0**: {'Yes' if pulseos_sharpe >= 4.0 else 'No'}\n")
        f.write(f"- **Exceeds Previous Best (4.259)**: {'Yes' if pulseos_sharpe > 4.259 else 'No'}\n\n")
        f.write("## Comparison\n\n")
        f.write("| Trial | Episodes | Seed | Sharpe | Improvement |\n")
        f.write("|-------|----------|------|--------|-------------|\n")
        f.write(f"| Trial 1 (prev) | 500 | Unknown | 4.259 | +15.9% |\n")
        f.write(f"| Trial 1 | 1000 | 1 | 3.688 | +1.7% |\n")
        f.write(f"| Trial 2 | 1000 | 2 | {pulseos_sharpe:.3f} | {improvement:+.1f}% |\n")
        f.write(f"\n**Test Duration**: {duration:.1f} minutes\n")
    
    print(f"\n✅ Results saved to:")
    print(f"   JSON: {json_path}")
    print(f"   Report: {report_path}")
    
    return result, ppo_baseline_sharpe

if __name__ == "__main__":
    asyncio.run(run_config2_trial2())



