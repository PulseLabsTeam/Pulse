"""
Config 2 Batch Test: Run 3 trials at a time for quick iteration

Run batches of 3 trials with 400 episodes each.
Allows for quick feedback and iterative testing.
"""

import asyncio
import numpy as np
from datetime import datetime
from trading_rl_test import run_pulseos_trial, download_stock_data
from trading_env import TradingEnv
import json
import os

async def run_config2_batch(batch_num: int, start_trial: int, ppo_baseline_sharpe: float, data):
    """
    Run a batch of 3 trials.
    
    Args:
        batch_num: Batch number (1, 2, 3, ...)
        start_trial: Starting trial number
        ppo_baseline_sharpe: PPO baseline Sharpe ratio
        data: Stock data
        
    Returns:
        List of trial results
    """
    print(f"\n{'='*80}")
    print(f"BATCH {batch_num}: Trials {start_trial}-{start_trial+2}")
    print(f"{'='*80}")
    
    results = []
    
    for i in range(3):
        trial_num = start_trial + i
        print(f"\nTrial {trial_num}...")
        
        env = TradingEnv(data, initial_capital=100000.0, commission=0.001)
        seed = trial_num
        
        result = await run_pulseos_trial(
            trial_num, env, 400, 1.5, 0.15,  # 400 episodes
            seed=seed,
            ppo_baseline_sharpe=ppo_baseline_sharpe,
            death_penalty_multiplier=-5.0
        )
        
        results.append({
            "trial": trial_num,
            "seed": seed,
            "final_sharpe": result.final_sharpe,
            "beats_ppo": result.final_sharpe > ppo_baseline_sharpe,
            "excellent": result.final_sharpe >= 4.0,
            "good": result.final_sharpe >= 3.5,
            "poor": result.final_sharpe < 3.0
        })
        
        if result.final_sharpe >= 4.0:
            print(f"  ✅ EXCELLENT: {result.final_sharpe:.3f} Sharpe")
        elif result.final_sharpe >= 3.5:
            print(f"  ✅ Good: {result.final_sharpe:.3f} Sharpe")
        elif result.final_sharpe >= ppo_baseline_sharpe:
            print(f"  ✅ Beats PPO: {result.final_sharpe:.3f} Sharpe")
        elif result.final_sharpe < 3.0:
            print(f"  ❌ Poor: {result.final_sharpe:.3f} Sharpe")
        else:
            print(f"  ⚠️  Below PPO: {result.final_sharpe:.3f} Sharpe")
    
    # Batch summary
    sharpes = [r["final_sharpe"] for r in results]
    successful = [r for r in results if r["excellent"]]
    good = [r for r in results if r["good"]]
    
    print(f"\nBatch {batch_num} Summary:")
    print(f"  Average Sharpe: {np.mean(sharpes):.3f}")
    print(f"  Excellent (4.0+): {len(successful)}/3")
    print(f"  Good (3.5+): {len(good)}/3")
    print(f"  Beats PPO: {sum(1 for r in results if r['beats_ppo'])}/3")
    
    return results

async def run_config2_iterative():
    """
    Run Config 2 tests in batches of 3 trials.
    """
    print("=" * 80)
    print("CONFIG 2 ITERATIVE TEST: Batches of 3 Trials")
    print("=" * 80)
    print("\nConfiguration:")
    print("  - NO Warm Start (independent training)")
    print("  - Death Penalty Schedule:")
    print("    * Episodes 0-150: -0.25 (very mild)")
    print("    * Episodes 150-300: -1.0 (moderate)")
    print("    * Episodes 300+: -3.0 (moderate-high)")
    print("  - Survival Signal: Exponential relaxation (aggressive)")
    print("  - Episodes: 400 per trial")
    print("  - Batch Size: 3 trials")
    print("=" * 80)
    
    # Download data
    symbol = "SPY"
    data = download_stock_data(symbol)
    
    # Run PPO baseline (3 trials for speed)
    print("\n📊 Step 1: Running PPO Baseline (3 trials, 400 episodes)...")
    print("-" * 80)
    from trading_rl_test import run_ppo_trial
    ppo_results = []
    for trial in range(1, 4):
        env = TradingEnv(data, initial_capital=100000.0, commission=0.001)
        result = await run_ppo_trial(trial, env, 400, 1.5, 0.15)
        ppo_results.append(result)
    
    ppo_baseline_sharpe = np.mean([r.final_sharpe for r in ppo_results])
    print(f"\n✅ PPO Baseline: {ppo_baseline_sharpe:.3f}")
    print("-" * 80)
    
    # Run batches iteratively
    all_results = []
    batch_num = 1
    start_trial = 1
    
    print("\n🚀 Step 2: Running Config 2 Batches...")
    print("=" * 80)
    print("\nRun batches of 3 trials. After each batch, you can:")
    print("  - Continue with next batch")
    print("  - Stop if results are promising")
    print("  - Stop if results are poor")
    print("=" * 80)
    
    while True:
        # Run batch
        batch_results = await run_config2_batch(batch_num, start_trial, ppo_baseline_sharpe, data)
        all_results.extend(batch_results)
        
        # Cumulative statistics
        total_trials = len(all_results)
        sharpes = [r["final_sharpe"] for r in all_results]
        successful = [r for r in all_results if r["excellent"]]
        good = [r for r in all_results if r["good"]]
        beats_ppo = sum(1 for r in all_results if r["beats_ppo"])
        
        print(f"\n{'='*80}")
        print(f"CUMULATIVE STATISTICS ({total_trials} trials)")
        print(f"{'='*80}")
        print(f"  Average Sharpe: {np.mean(sharpes):.3f} ± {np.std(sharpes):.3f}")
        print(f"  Improvement vs PPO: {((np.mean(sharpes) - ppo_baseline_sharpe) / ppo_baseline_sharpe) * 100:+.1f}%")
        print(f"  Excellent (4.0+): {len(successful)}/{total_trials} ({len(successful)/total_trials*100:.1f}%)")
        print(f"  Good (3.5+): {len(good)}/{total_trials} ({len(good)/total_trials*100:.1f}%)")
        print(f"  Beats PPO: {beats_ppo}/{total_trials} ({beats_ppo/total_trials*100:.1f}%)")
        
        if successful:
            print(f"\n  Successful Trials (4.0+ Sharpe):")
            for r in successful:
                print(f"    Trial {r['trial']}: {r['final_sharpe']:.3f} Sharpe")
        
        # Save results after each batch
        output_dir = "benchmark_results/trading_rl/config2_large_scale"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = f"{output_dir}/config2_batch_{batch_num}_{total_trials}trials_{timestamp}.json"
        
        with open(json_path, "w") as f:
            json.dump({
                "ppo_baseline": ppo_baseline_sharpe,
                "test_config": {
                    "episodes": 400,
                    "batch_size": 3,
                    "total_trials": total_trials
                },
                "results": all_results,
                "statistics": {
                    "avg_sharpe": float(np.mean(sharpes)),
                    "std_sharpe": float(np.std(sharpes)),
                    "success_rate": len(successful) / total_trials * 100,
                    "good_rate": len(good) / total_trials * 100,
                    "beats_ppo_rate": beats_ppo / total_trials * 100,
                    "successful_runs": [r["trial"] for r in successful]
                }
            }, f, indent=2)
        
        print(f"\n✅ Results saved to: {json_path}")
        
        # Ask user if they want to continue
        print(f"\n{'='*80}")
        print(f"Batch {batch_num} complete. {total_trials} total trials.")
        print(f"{'='*80}")
        
        # For automated runs, continue for a few batches
        # User can stop manually if needed
        if batch_num >= 10:  # Stop after 10 batches (30 trials)
            print("\n✅ Completed 10 batches (30 trials). Stopping.")
            break
        
        batch_num += 1
        start_trial += 3
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\nTotal Trials: {len(all_results)}")
    print(f"PPO Baseline: {ppo_baseline_sharpe:.3f}")
    print(f"\nConfig 2 Results:")
    print(f"  Average Sharpe: {np.mean(sharpes):.3f} ± {np.std(sharpes):.3f}")
    print(f"  Success Rate (4.0+): {len(successful)/len(all_results)*100:.1f}% ({len(successful)}/{len(all_results)})")
    print(f"  Good Rate (3.5+): {len(good)/len(all_results)*100:.1f}% ({len(good)}/{len(all_results)})")
    print(f"  Beats PPO: {beats_ppo}/{len(all_results)} ({beats_ppo/len(all_results)*100:.1f}%)")
    
    if successful:
        print(f"\nSuccessful Trials (4.0+ Sharpe):")
        for r in sorted(successful, key=lambda x: x["final_sharpe"], reverse=True):
            print(f"  Trial {r['trial']}: {r['final_sharpe']:.3f} Sharpe")
    
    return all_results

if __name__ == "__main__":
    asyncio.run(run_config2_iterative())



