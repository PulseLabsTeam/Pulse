"""
Configuration 2: Even More Gradual Progressive Death Penalty

This test implements Configuration 2 from the tech review:
- Progressive death penalty: -0.25 (episodes 0-150), -1.0 (150-300), -3.0 (300+)
- Exponential survival signal relaxation
- 500 episodes, 5 trials

Expected improvements:
- Even lower variance: std dev < 0.5
- Higher average: 3.5-4.0 Sharpe
- More trials beat PPO: 3-4/5
"""

import asyncio
import numpy as np
from datetime import datetime
from trading_rl_test import run_trading_test

async def main():
    print("=" * 80)
    print("CONFIGURATION 2: Even More Gradual Progressive Death Penalty")
    print("=" * 80)
    print("\nConfiguration:")
    print("  - Death Penalty Schedule:")
    print("    * Episodes 0-150: -0.25 (very mild - allows exploration)")
    print("    * Episodes 150-300: -1.0 (moderate - survival pressure starts)")
    print("    * Episodes 300+: -3.0 (moderate-high - survival pressure)")
    print("  - Survival Signal: Exponential relaxation")
    print("  - Episodes: 500")
    print("  - Trials: 5")
    print("=" * 80)
    print("\nStarting test...\n")
    
    start_time = datetime.now()
    
    # Run test with base death penalty = 5.0 (will be overridden by progressive schedule)
    results = await run_trading_test('SPY', 5, 500, 1.5, 0.15, 'standard', -5.0)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    pulseos_sharpes = [r.final_sharpe for r in results.pulseos_results]
    ppo_sharpes = [r.final_sharpe for r in results.ppo_results]
    
    pulseos_avg = np.mean(pulseos_sharpes)
    ppo_avg = np.mean(ppo_sharpes)
    pulseos_std = np.std(pulseos_sharpes)
    ppo_std = np.std(ppo_sharpes)
    
    improvement = ((pulseos_avg - ppo_avg) / ppo_avg) * 100
    beats_ppo = sum(1 for s in pulseos_sharpes if s > ppo_avg)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"\nPPO Baseline:")
    print(f"  Avg Sharpe: {ppo_avg:.3f} ± {ppo_std:.3f}")
    print(f"\nPulseOS Results:")
    print(f"  Avg Sharpe: {pulseos_avg:.3f} ± {pulseos_std:.3f}")
    print(f"  Improvement: {improvement:+.1f}%")
    print(f"  Std Dev: {pulseos_std:.3f}")
    print(f"  Trials Beating PPO: {beats_ppo}/{len(pulseos_sharpes)}")
    
    print(f"\nIndividual PulseOS Results:")
    for i, sharpe in enumerate(pulseos_sharpes, 1):
        status = "✅ BEATS PPO" if sharpe > ppo_avg else "❌ Below PPO"
        print(f"  Trial {i}: Sharpe={sharpe:.3f} {status}")
    
    # Compare to Config 1
    print("\n" + "=" * 80)
    print("COMPARISON TO CONFIGURATION 1")
    print("=" * 80)
    print("\nConfig 1 (0-100: -0.5, 100-200: -2.0, 200+: -5.0):")
    print("  Avg Sharpe: 2.557 ± 0.662")
    print("  Improvement: -34.0%")
    print("  Trials Beating PPO: 0/5")
    print("\nConfig 2 (0-150: -0.25, 150-300: -1.0, 300+: -3.0):")
    print(f"  Avg Sharpe: {pulseos_avg:.3f} ± {pulseos_std:.3f}")
    print(f"  Improvement: {improvement:+.1f}%")
    print(f"  Trials Beating PPO: {beats_ppo}/{len(pulseos_sharpes)}")
    
    # Success criteria
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA")
    print("=" * 80)
    success_variance = pulseos_std < 0.5
    success_avg = pulseos_avg > 3.5
    success_trials = beats_ppo >= 3
    success_improvement = improvement > 0
    
    print(f"  Variance < 0.5: {'✅' if success_variance else '❌'} ({pulseos_std:.3f})")
    print(f"  Avg Sharpe > 3.5: {'✅' if success_avg else '❌'} ({pulseos_avg:.3f})")
    print(f"  Trials Beating PPO >= 3: {'✅' if success_trials else '❌'} ({beats_ppo}/{len(pulseos_sharpes)})")
    print(f"  Improvement > 0%: {'✅' if success_improvement else '❌'} ({improvement:+.1f}%)")
    
    overall_success = success_variance and success_avg and success_trials and success_improvement
    print(f"\nOverall Success: {'✅ YES' if overall_success else '⚠️  PARTIAL'}")
    
    if overall_success:
        print("\n🎉 CONFIGURATION 2 IS SUCCESSFUL!")
    elif improvement > -20 and pulseos_std < 0.8:
        print("\n✅ Significant improvement! Continue tuning...")
    else:
        print("\n⚠️  Needs more tuning. Consider Configuration 3 (warm start).")
    
    print(f"\nTest Duration: {duration:.1f} minutes")
    
    # Save results
    output_dir = "benchmark_results/trading_rl/progressive_penalty"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{output_dir}/CONFIG2_RESULTS_{timestamp}.md"
    
    with open(report_path, "w") as f:
        f.write("# Configuration 2: Even More Gradual Progressive Death Penalty\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Configuration\n\n")
        f.write("- **Death Penalty Schedule**:\n")
        f.write("  - Episodes 0-150: -0.25 (very mild)\n")
        f.write("  - Episodes 150-300: -1.0 (moderate)\n")
        f.write("  - Episodes 300+: -3.0 (moderate-high)\n")
        f.write("- **Survival Signal**: Exponential relaxation\n")
        f.write("- **Episodes**: 500\n")
        f.write("- **Trials**: 5\n\n")
        f.write("## Results\n\n")
        f.write(f"- **PPO Avg**: {ppo_avg:.3f} ± {ppo_std:.3f}\n")
        f.write(f"- **PulseOS Avg**: {pulseos_avg:.3f} ± {pulseos_std:.3f}\n")
        f.write(f"- **Improvement**: {improvement:+.1f}%\n")
        f.write(f"- **Std Dev**: {pulseos_std:.3f}\n")
        f.write(f"- **Trials Beating PPO**: {beats_ppo}/{len(pulseos_sharpes)}\n\n")
        f.write("## Individual Results\n\n")
        f.write("### PulseOS\n")
        for i, sharpe in enumerate(pulseos_sharpes, 1):
            status = "✅ BEATS PPO" if sharpe > ppo_avg else "❌ Below PPO"
            f.write(f"- Trial {i}: {sharpe:.3f} {status}\n")
    
    print(f"\n✅ Results saved to {report_path}")

if __name__ == "__main__":
    asyncio.run(main())



