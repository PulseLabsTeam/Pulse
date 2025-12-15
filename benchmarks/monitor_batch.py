"""
Quick monitor for Config 2 batch test
"""

import json
import os
from pathlib import Path
import numpy as np

def check_batch_progress():
    """Check latest batch results"""
    output_dir = "benchmark_results/trading_rl/config2_large_scale"
    
    if not os.path.exists(output_dir):
        print("No results yet. Test is starting up...")
        return
    
    # Find latest batch file
    batch_files = sorted(Path(output_dir).glob("config2_batch_*.json"), key=os.path.getmtime, reverse=True)
    
    if not batch_files:
        print("No batch results yet. Test is running...")
        return
    
    latest_file = batch_files[0]
    
    with open(latest_file, "r") as f:
        data = json.load(f)
    
    stats = data["statistics"]
    results = data["results"]
    
    print(f"\n{'='*60}")
    print(f"Latest Batch Results")
    print(f"{'='*60}")
    print(f"Total Trials: {len(results)}")
    print(f"PPO Baseline: {data['ppo_baseline']:.3f}")
    print(f"\nConfig 2 Results:")
    print(f"  Average Sharpe: {stats['avg_sharpe']:.3f} ± {stats['std_sharpe']:.3f}")
    print(f"  Success Rate (4.0+): {stats['success_rate']:.1f}% ({len(stats['successful_runs'])}/{len(results)})")
    print(f"  Good Rate (3.5+): {stats['good_rate']:.1f}%")
    print(f"  Beats PPO: {stats['beats_ppo_rate']:.1f}%")
    
    if stats['successful_runs']:
        print(f"\n  Successful Trials (4.0+ Sharpe):")
        for trial_num in stats['successful_runs']:
            sharpe = next(r['final_sharpe'] for r in results if r['trial'] == trial_num)
            print(f"    Trial {trial_num}: {sharpe:.3f} Sharpe")
    
    print(f"\nLatest batch file: {latest_file.name}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    check_batch_progress()



