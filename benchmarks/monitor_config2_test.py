"""
Monitor Config 2 Large-Scale Test Progress

Check progress and results as they come in.
"""

import json
import os
from pathlib import Path
import numpy as np
from datetime import datetime

def check_progress():
    """Check test progress from log file"""
    log_path = "/tmp/config2_large_scale.log"
    
    if not os.path.exists(log_path):
        print("Test not started yet or log file not found.")
        return
    
    with open(log_path, "r") as f:
        lines = f.readlines()
    
    # Count completed trials
    completed_trials = sum(1 for line in lines if "Trial" in line and "completed" in line)
    
    # Find latest results
    results = []
    current_trial = None
    for line in lines:
        if "Trial" in line and "/50" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "Trial":
                    try:
                        current_trial = int(parts[i+1].split("/")[0])
                    except:
                        pass
        if "Sharpe" in line and current_trial:
            try:
                sharpe_str = line.split("Sharpe")[-1].strip()
                sharpe = float(sharpe_str.split()[0])
                results.append({"trial": current_trial, "sharpe": sharpe})
            except:
                pass
    
    print(f"Progress: {completed_trials}/50 trials completed")
    
    if results:
        sharpes = [r["sharpe"] for r in results]
        successful = [r for r in results if r["sharpe"] >= 4.0]
        
        print(f"\nCompleted Trials: {len(results)}")
        print(f"Average Sharpe: {np.mean(sharpes):.3f}")
        print(f"Successful (4.0+): {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
        
        if successful:
            print(f"\nSuccessful Trials:")
            for r in successful:
                print(f"  Trial {r['trial']}: {r['sharpe']:.3f} Sharpe")
    
    # Check for final results
    if "FINAL RESULTS" in "".join(lines):
        print("\n✅ Test completed! Check final results above.")

if __name__ == "__main__":
    check_progress()



