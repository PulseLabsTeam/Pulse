"""
Config 2 Extended Training Test: Single Trial with 1000 Episodes

Test if extended training (1000 episodes) helps Config 2 achieve better results.
"""

import asyncio
import numpy as np
from datetime import datetime
from trading_rl_test import run_pulseos_trial, download_stock_data, run_ppo_trial
from trading_env import TradingEnv
import json
import os

async def run_config2_extended():
    """
    Run single Config 2 trial with 1000 episodes.
    """
    print("=" * 80)
    print("CONFIG 2 EXTENDED TRAINING TEST: 1000 Episodes")
    print("=" * 80)
    print("\nConfiguration:")
    print("  - NO Warm Start (independent training)")
    print("  - Death Penalty Schedule:")
    print("    * Episodes 0-150: -0.25 (very mild)")
    print("    * Episodes 150-300: -1.0 (moderate)")
    print("    * Episodes 300+: -3.0 (moderate-high)")
    print("  - Survival Signal: Exponential relaxation (aggressive)")
    print("  - Episodes: 1000")
    print("  - Trials: 1")
    print("=" * 80)
    
    start_time = datetime.now()
    
    # Download data
    symbol = "SPY"
    data = download_stock_data(symbol)
    
    # Run PPO baseline (for comparison)
    print("\n📊 Running PPO Baseline (1000 episodes)...")
    print("-" * 80)
    env_ppo = TradingEnv(data, initial_capital=100000.0, commission=0.001)
    ppo_result = await run_ppo_trial(1, env_ppo, 1000, 1.5, 0.15)
    ppo_sharpe = ppo_result.final_sharpe
    print(f"\n✅ PPO Baseline: {ppo_sharpe:.3f}")
    print("-" * 80)
    
    # Run Config 2 trial
    print("\n🚀 Running Config 2 Trial (1000 episodes)...")
    print("-" * 80)
    print("This will take approximately 10-15 minutes...")
    print("-" * 80)
    
    env = TradingEnv(data, initial_capital=100000.0, commission=0.001)
    
    result = await run_pulseos_trial(
        1, env, 1000, 1.5, 0.15,
        seed=1,  # Use seed 1 for reproducibility
        ppo_baseline_sharpe=ppo_sharpe,
        death_penalty_multiplier=-5.0
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    # Results
    pulseos_sharpe = result.final_sharpe
    improvement = ((pulseos_sharpe - ppo_sharpe) / ppo_sharpe) * 100
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nPPO Baseline (1000 episodes):")
    print(f"  Final Sharpe: {ppo_sharpe:.3f}")
    
    print(f"\nConfig 2 Trial (1000 episodes):")
    print(f"  Final Sharpe: {pulseos_sharpe:.3f}")
    print(f"  Improvement: {improvement:+.1f}%")
    
    if pulseos_sharpe >= 4.0:
        print(f"\n  ✅ EXCELLENT: Achieved 4.0+ Sharpe!")
    elif pulseos_sharpe >= ppo_sharpe:
        print(f"\n  ✅ SUCCESS: Beats PPO baseline!")
    elif pulseos_sharpe >= 3.5:
        print(f"\n  ✅ Good: Competitive performance")
    else:
        print(f"\n  ⚠️  Below PPO baseline")
    
    print(f"\nTest Duration: {duration:.1f} minutes")
    
    # Comparison to previous results
    print("\n" + "=" * 80)
    print("COMPARISON TO PREVIOUS RESULTS")
    print("=" * 80)
    print("\nConfig 2 Trial 1 (500 episodes):")
    print("  Final Sharpe: 4.259")
    print("  Improvement: +15.9% vs PPO")
    print("\nConfig 2 Trial 1 (1000 episodes):")
    print(f"  Final Sharpe: {pulseos_sharpe:.3f}")
    print(f"  Improvement: {improvement:+.1f}% vs PPO")
    
    if pulseos_sharpe > 4.259:
        print(f"\n  ✅ EXCEEDED previous best! (+{pulseos_sharpe - 4.259:.3f} Sharpe)")
    elif pulseos_sharpe >= 4.0:
        print(f"\n  ✅ Matched/exceeded 4.0 threshold")
    else:
        print(f"\n  ⚠️  Below previous best of 4.259")
    
    # Save results
    output_dir = "benchmark_results/trading_rl/config2_large_scale"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_path = f"{output_dir}/config2_1000ep_trial1_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "test_config": {
                "episodes": 1000,
                "trials": 1,
                "seed": 1,
                "no_warm_start": True
            },
            "ppo_baseline": {
                "final_sharpe": ppo_sharpe,
                "episodes": 1000
            },
            "pulseos_result": {
                "final_sharpe": float(pulseos_sharpe),
                "improvement_vs_ppo": float(improvement),
                "beats_ppo": bool(pulseos_sharpe > ppo_sharpe),
                "excellent": bool(pulseos_sharpe >= 4.0),
                "exceeds_previous_best": bool(pulseos_sharpe > 4.259)
            },
            "comparison": {
                "previous_best_500ep": 4.259,
                "current_1000ep": pulseos_sharpe,
                "difference": pulseos_sharpe - 4.259
            },
            "duration_minutes": duration
        }, f, indent=2)
    
    report_path = f"{output_dir}/config2_1000ep_trial1_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write("# Config 2 Extended Training Test: 1000 Episodes\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Test Configuration\n\n")
        f.write("- **Episodes**: 1000\n")
        f.write("- **Trials**: 1\n")
        f.write("- **Seed**: 1\n")
        f.write("- **No Warm Start**: Independent training\n\n")
        f.write("## Results\n\n")
        f.write(f"- **PPO Baseline**: {ppo_sharpe:.3f}\n")
        f.write(f"- **Config 2 Final Sharpe**: {pulseos_sharpe:.3f}\n")
        f.write(f"- **Improvement vs PPO**: {improvement:+.1f}%\n")
        f.write(f"- **Beats PPO**: {'Yes' if pulseos_sharpe > ppo_sharpe else 'No'}\n")
        f.write(f"- **Exceeds 4.0**: {'Yes' if pulseos_sharpe >= 4.0 else 'No'}\n")
        f.write(f"- **Exceeds Previous Best (4.259)**: {'Yes' if pulseos_sharpe > 4.259 else 'No'}\n\n")
        f.write("## Comparison\n\n")
        f.write("| Metric | 500 Episodes | 1000 Episodes | Change |\n")
        f.write("|--------|--------------|---------------|--------|\n")
        f.write(f"| Final Sharpe | 4.259 | {pulseos_sharpe:.3f} | {pulseos_sharpe - 4.259:+.3f} |\n")
        f.write(f"| Improvement vs PPO | +15.9% | {improvement:+.1f}% | {improvement - 15.9:+.1f}% |\n")
        f.write(f"\n**Test Duration**: {duration:.1f} minutes\n")
    
    print(f"\n✅ Results saved to:")
    print(f"   JSON: {json_path}")
    print(f"   Report: {report_path}")
    
    return result, ppo_sharpe

if __name__ == "__main__":
    asyncio.run(run_config2_extended())

