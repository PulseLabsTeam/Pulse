"""
Configuration 3: Warm Start from PPO + Progressive Death Penalty

This test implements Configuration 3 from the tech review:
- Warm start: Initialize PulseOS agents from PPO weights
- Progressive death penalty: -0.25 (episodes 0-150), -1.0 (150-300), -3.0 (300+)
- Exponential survival signal relaxation
- 500 episodes, 5 trials

Expected improvements:
- Lower variance: std dev < 0.5 (better initialization)
- Higher average: 3.5-4.0 Sharpe (starting from good baseline)
- More trials beat PPO: 3-4/5 (better consistency)
"""

import asyncio
import numpy as np
from datetime import datetime
from trading_rl_test import run_trading_test, run_ppo_trial, run_pulseos_trial, download_stock_data
from trading_env import TradingEnv

async def main():
    print("=" * 80)
    print("CONFIGURATION 3: Warm Start from PPO + Progressive Death Penalty")
    print("=" * 80)
    print("\nConfiguration:")
    print("  - Warm Start: Initialize PulseOS from PPO weights")
    print("  - Death Penalty Schedule:")
    print("    * Episodes 0-150: -0.25 (very mild - allows exploration)")
    print("    * Episodes 150-300: -1.0 (moderate - survival pressure starts)")
    print("    * Episodes 300+: -3.0 (moderate-high - survival pressure)")
    print("  - Survival Signal: Exponential relaxation")
    print("  - Episodes: 500")
    print("  - Trials: 5")
    print("=" * 80)
    print("\nExpected Results:")
    print("  - Avg Sharpe: 3.5-4.0 (vs current 2.671)")
    print("  - Std Dev: 0.3-0.5 (vs current 0.883)")
    print("  - Trials Beating PPO: 3-4/5 (vs current 1/5)")
    print("=" * 80)
    print("\nStarting test...\n")
    
    start_time = datetime.now()
    
    # Download data
    symbol = "SPY"
    data = download_stock_data(symbol)
    
    # Step 1: Run PPO trials and extract weights
    print("📊 Step 1: Running PPO Baseline Trials and Extracting Weights...")
    print("-" * 80)
    ppo_results = []
    ppo_agents = []
    
    for trial in range(1, 6):  # 5 trials
        env = TradingEnv(data, initial_capital=100000.0, commission=0.001)
        result, agent = await run_ppo_trial(trial, env, 500, 1.5, 0.15, return_agent=True)
        ppo_results.append(result)
        ppo_agents.append(agent)
    
    # Compute PPO baseline
    ppo_baseline_sharpe = np.mean([r.final_sharpe for r in ppo_results])
    print(f"\n✅ PPO Baseline Established: Average Sharpe Ratio = {ppo_baseline_sharpe:.3f}")
    print(f"   Extracted weights from {len(ppo_agents)} PPO agents")
    print("-" * 80)
    
    # Step 2: Run PulseOS trials with warm start from BEST PPO agent
    print("\n🚀 Step 2: Running PulseOS Trials with Warm Start from BEST PPO Agent...")
    print("-" * 80)
    
    # Find best PPO agent
    best_ppo_idx = np.argmax([r.final_sharpe for r in ppo_results])
    best_ppo_agent = ppo_agents[best_ppo_idx]
    best_ppo_sharpe = ppo_results[best_ppo_idx].final_sharpe
    print(f"   Using weights from PPO Trial {best_ppo_idx + 1} (Sharpe: {best_ppo_sharpe:.3f})")
    
    pulseos_results = []
    
    for trial in range(1, 6):  # 5 trials
        # Use BEST PPO agent's weights for all trials (with small noise for diversity)
        ppo_weights = best_ppo_agent.get_weights()
        
        # Add small noise to allow exploration while benefiting from best initialization
        ppo_weights['add_noise'] = True
        ppo_weights['noise_scale'] = 0.01  # 1% noise for diversity
        
        # Create new environment for PulseOS trial
        env = TradingEnv(data, initial_capital=100000.0, commission=0.001)
        
        # Run PulseOS trial with warm start
        result = await run_pulseos_trial(
            trial, env, 500, 1.5, 0.15,
            initial_weights=ppo_weights,
            ppo_baseline_sharpe=ppo_baseline_sharpe,
            death_penalty_multiplier=-5.0  # Base penalty (will be overridden by progressive schedule)
        )
        pulseos_results.append(result)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    # Calculate statistics
    pulseos_sharpes = [r.final_sharpe for r in pulseos_results]
    ppo_sharpes = [r.final_sharpe for r in ppo_results]
    
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
    print(f"\nPulseOS Results (Warm Start from PPO):")
    print(f"  Avg Sharpe: {pulseos_avg:.3f} ± {pulseos_std:.3f}")
    print(f"  Improvement: {improvement:+.1f}%")
    print(f"  Std Dev: {pulseos_std:.3f}")
    print(f"  Trials Beating PPO: {beats_ppo}/{len(pulseos_sharpes)}")
    
    print(f"\nIndividual PulseOS Results:")
    for i, sharpe in enumerate(pulseos_sharpes, 1):
        status = "✅ BEATS PPO" if sharpe > ppo_avg else "❌ Below PPO"
        print(f"  Trial {i}: Sharpe={sharpe:.3f} {status}")
    
    print(f"\nIndividual PPO Results:")
    for i, sharpe in enumerate(ppo_sharpes, 1):
        print(f"  Trial {i}: Sharpe={sharpe:.3f}")
    
    # Compare to previous configurations
    print("\n" + "=" * 80)
    print("COMPARISON TO PREVIOUS CONFIGURATIONS")
    print("=" * 80)
    print("\nConfig 1 (Progressive penalty, no warm start):")
    print("  Avg Sharpe: 2.557 ± 0.662")
    print("  Improvement: -34.0%")
    print("  Trials Beating PPO: 0/5")
    print("\nConfig 2 (Gradual penalty, no warm start):")
    print("  Avg Sharpe: 2.671 ± 0.883")
    print("  Improvement: -27.3%")
    print("  Trials Beating PPO: 1/5")
    print("\nConfig 3 (Gradual penalty + warm start from PPO):")
    print(f"  Avg Sharpe: {pulseos_avg:.3f} ± {pulseos_std:.3f}")
    print(f"  Improvement: {improvement:+.1f}%")
    print(f"  Trials Beating PPO: {beats_ppo}/{len(pulseos_sharpes)}")
    
    # Calculate improvements
    config2_avg = 2.671
    config2_std = 0.883
    avg_improvement = ((pulseos_avg - config2_avg) / config2_avg) * 100
    variance_improvement = ((config2_std - pulseos_std) / config2_std) * 100
    
    print(f"\nImprovements over Config 2:")
    print(f"  Average Improvement: {avg_improvement:+.1f}%")
    print(f"  Variance Reduction: {variance_improvement:+.1f}% (lower is better)")
    print(f"  Trials Beating PPO: +{beats_ppo - 1} (vs Config 2)")
    
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
        print("\n🎉 CONFIGURATION 3 IS SUCCESSFUL!")
        print("   Warm start + progressive penalty enables consistent PPO-beating performance!")
    elif success_improvement and success_trials:
        print("\n✅ Significant improvement! Warm start is working!")
    elif success_improvement:
        print("\n✅ Improvement achieved! Continue tuning...")
    else:
        print("\n⚠️  Needs more tuning. Consider hybrid approaches.")
    
    print(f"\nTest Duration: {duration:.1f} minutes")
    
    # Save results
    output_dir = "benchmark_results/trading_rl/progressive_penalty"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{output_dir}/CONFIG3_WARMSTART_RESULTS_{timestamp}.md"
    
    with open(report_path, "w") as f:
        f.write("# Configuration 3: Warm Start from PPO + Progressive Death Penalty\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Configuration\n\n")
        f.write("- **Warm Start**: Initialize PulseOS from PPO weights\n")
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
        f.write("\n### PPO\n")
        for i, sharpe in enumerate(ppo_sharpes, 1):
            f.write(f"- Trial {i}: {sharpe:.3f}\n")
        f.write("\n## Comparison\n\n")
        f.write("| Configuration | Avg Sharpe | Std Dev | Improvement | Trials Beating PPO |\n")
        f.write("|---------------|------------|---------|-------------|-------------------|\n")
        f.write(f"| Config 1 | 2.557 | 0.662 | -34.0% | 0/5 |\n")
        f.write(f"| Config 2 | 2.671 | 0.883 | -27.3% | 1/5 |\n")
        f.write(f"| **Config 3** | **{pulseos_avg:.3f}** | **{pulseos_std:.3f}** | **{improvement:+.1f}%** | **{beats_ppo}/5** |\n")
    
    print(f"\n✅ Results saved to {report_path}")

if __name__ == "__main__":
    asyncio.run(main())

