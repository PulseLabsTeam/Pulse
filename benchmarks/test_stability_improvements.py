"""
Test Stability Improvements for Trial 2 Performance

This script tests the stability improvements designed to replicate Trial 2's
performance consistently and prevent the boom-bust cycle collapse.

Improvements tested:
1. Momentum/EMA to survival signal (smooth boom-bust cycles)
2. Adaptive temporal window (longer after episode 200)
3. Fixed early restart logic (don't restart improving/ALIVE trials)
4. Performance momentum tracking (prevent degradation)
5. Maintain minimum learning pressure even when ALIVE
"""

import asyncio
import sys
import os
from datetime import datetime

# Add parent directory to path to import from benchmarks
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.trading_rl_test import run_trading_test


async def main():
    """Run stability improvements test with 5 trials"""
    print("=" * 80)
    print("🧪 TESTING STABILITY IMPROVEMENTS")
    print("=" * 80)
    print()
    print("This test will:")
    print("  1. Run PPO trials to establish baseline Sharpe ratio")
    print("  2. Run 5 PulseOS trials with stability improvements:")
    print("     - Momentum/EMA smoothing for survival signal")
    print("     - Adaptive temporal window (longer after episode 200)")
    print("     - Fixed early restart (don't restart ALIVE/improving trials)")
    print("     - Performance momentum tracking")
    print("     - Minimum learning pressure maintenance")
    print("     - Survival reward bonus (aggressive scaling at high performance)")
    print("     - Strong penalties for low survival signal")
    print("  3. Analyze if Trial 2-like performance is replicated consistently")
    print()
    
    # Run test with 5 trials, 600 episodes (same as previous test)
    # This will run PPO trials first to establish baseline
    results = await run_trading_test(
        symbol="SPY",
        num_trials=5,  # 5 trials to validate stability
        max_episodes=600,  # Extended training to observe stability
        target_sharpe=1.5,
        target_return=0.15,
        test_mode="standard"  # Standard mode uses PPO baseline constraint
    )
    
    print("\n" + "=" * 80)
    print("📊 STABILITY IMPROVEMENTS TEST RESULTS")
    print("=" * 80)
    
    # Analyze results
    ppo_baseline = results.ppo_avg_final_sharpe
    pulseos_sharpes = [r.final_sharpe for r in results.pulseos_results]
    pulseos_avg = results.pulseos_avg_final_sharpe
    pulseos_std = results.pulseos_std_final_sharpe if hasattr(results, 'pulseos_std_final_sharpe') else 0.0
    
    print(f"\nPPO Baseline Sharpe: {ppo_baseline:.3f}")
    print(f"\nPulseOS Results ({len(pulseos_sharpes)} trials):")
    print(f"  Average Sharpe: {pulseos_avg:.3f}")
    print(f"  Std Sharpe: {pulseos_std:.3f}")
    print(f"  Range: {min(pulseos_sharpes):.3f} - {max(pulseos_sharpes):.3f}")
    
    # Count ALIVE episodes (beating baseline)
    alive_count = sum(1 for s in pulseos_sharpes if s >= ppo_baseline)
    print(f"  Trials beating baseline: {alive_count}/{len(pulseos_sharpes)} ({alive_count/len(pulseos_sharpes)*100:.1f}%)")
    
    # Check for Trial 2 pattern (consistent ALIVE then collapse)
    print(f"\n📈 Trial-by-Trial Analysis:")
    for i, sharpe in enumerate(pulseos_sharpes, 1):
        status = "✅ ALIVE" if sharpe >= ppo_baseline else "❌ DYING"
        print(f"  Trial {i}: {sharpe:.3f} {status}")
    
    # Success criteria
    print(f"\n🎯 Success Criteria:")
    success_1 = pulseos_avg >= ppo_baseline
    success_2 = alive_count >= 3  # At least 3/5 trials beat baseline
    success_3 = pulseos_std < 1.5  # Reasonable variance
    
    print(f"  Average ≥ Baseline: {'✅' if success_1 else '❌'} ({pulseos_avg:.3f} vs {ppo_baseline:.3f})")
    print(f"  ≥3/5 trials beat baseline: {'✅' if success_2 else '❌'} ({alive_count}/5)")
    print(f"  Std < 1.5: {'✅' if success_3 else '❌'} ({pulseos_std:.3f})")
    
    if success_1 and success_2 and success_3:
        print(f"\n🎉 SUCCESS! Stability improvements working - consistent Trial 2-like performance!")
    elif success_1 and success_2:
        print(f"\n✅ Good progress - Average and consistency criteria met!")
    elif success_1:
        print(f"\n⚠️  Average met but consistency needs work")
    else:
        print(f"\n⚠️  Need further optimization - stability improvements may need tuning")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"benchmark_results/stability_test_{timestamp}.md"
    os.makedirs("benchmark_results", exist_ok=True)
    
    with open(results_file, "w") as f:
        f.write("# Stability Improvements Test Results\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- Trials: 5\n")
        f.write(f"- Episodes per trial: 600\n")
        f.write(f"- PPO Baseline: {ppo_baseline:.3f}\n\n")
        f.write(f"## Results\n\n")
        f.write(f"| Trial | Final Sharpe | Status |\n")
        f.write(f"|-------|-------------|--------|\n")
        for i, sharpe in enumerate(pulseos_sharpes, 1):
            status = "ALIVE" if sharpe >= ppo_baseline else "DYING"
            f.write(f"| {i} | {sharpe:.3f} | {status} |\n")
        f.write(f"\n**Average**: {pulseos_avg:.3f}\n")
        f.write(f"**Std**: {pulseos_std:.3f}\n")
        f.write(f"**Trials beating baseline**: {alive_count}/5\n")
    
    print(f"\n📄 Results saved to: {results_file}")
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

