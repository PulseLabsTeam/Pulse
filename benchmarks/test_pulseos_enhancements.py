"""
Test PulseOS Enhancements

Test suite for new PulseOS-specific enhancements.
"""

import asyncio
import numpy as np
from trading_rl_test import run_trading_test, TestResults, generate_summary_report


async def test_pulseos_enhancements(
    symbol: str = "SPY",
    num_trials: int = 5,
    max_episodes: int = 500,
    test_mode: str = "standard"
) -> TestResults:
    """
    Test PulseOS with all enhancements enabled.
    
    Args:
        symbol: Stock symbol to trade
        num_trials: Number of trials
        max_episodes: Maximum episodes per trial
        test_mode: Test mode (standard, warm_start, etc.)
        
    Returns:
        TestResults with comparison
    """
    print("=" * 80)
    print("🚀 PULSEOS ENHANCEMENTS TEST")
    print("=" * 80)
    print(f"\nTesting with all enhancements enabled:")
    print("  - Phase 1: Lower survival thresholds (10th percentile)")
    print("  - Phase 2: Enhanced survival signals (momentum-aware, recovery bonuses)")
    print("  - Phase 3: Progressive death penalties (performance-based)")
    print("  - Phase 4: PulseOS-specific enhancements (trajectory rewards, LR modulation)")
    print("  - Phase 5: Learning rate warmup and adaptive decay")
    print("  - Phase 6: Reward shaping (consistency bonuses, risk-adjusted rewards)")
    print(f"\nSymbol: {symbol}, Trials: {num_trials}, Episodes: {max_episodes}\n")
    
    # Test with 10th percentile threshold (as suggested in plan)
    results = await run_trading_test(
        symbol=symbol,
        num_trials=num_trials,
        max_episodes=max_episodes,
        test_mode=test_mode,
        threshold_percentile=0.1,  # 10th percentile threshold
        threshold_fixed=None
    )
    
    return results


async def main():
    """Main test execution."""
    print("\n🚀 Starting PulseOS Enhancements Test\n")
    
    # Run test with all enhancements
    results = await test_pulseos_enhancements(
        symbol="SPY",
        num_trials=5,
        max_episodes=500,
        test_mode="standard"
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("📊 PULSEOS ENHANCEMENTS TEST RESULTS")
    print("=" * 80)
    
    ppo_sharpes = [r.final_sharpe for r in results.ppo_results]
    pulseos_sharpes = [r.final_sharpe for r in results.pulseos_results]
    
    ppo_avg = np.mean(ppo_sharpes)
    pulseos_avg = np.mean(pulseos_sharpes)
    pulseos_std = np.std(pulseos_sharpes)
    
    improvement = ((pulseos_avg / ppo_avg) - 1) * 100
    beats_ppo = sum(1 for s in pulseos_sharpes if s >= ppo_avg)
    
    print(f"\nPPO Baseline:")
    print(f"  Average Sharpe: {ppo_avg:.3f}")
    print(f"  Std Dev: {np.std(ppo_sharpes):.3f}")
    
    print(f"\nPulseOS with Enhancements:")
    print(f"  Average Sharpe: {pulseos_avg:.3f}")
    print(f"  Std Dev: {pulseos_std:.3f}")
    print(f"  Improvement: {improvement:+.1f}%")
    print(f"  Beats PPO: {beats_ppo}/{len(pulseos_sharpes)} ({beats_ppo/len(pulseos_sharpes)*100:.1f}%)")
    
    # Success criteria check
    print(f"\n🎯 Success Criteria Check:")
    success_avg = pulseos_avg >= 4.0
    success_std = pulseos_std < 0.4
    success_rate = beats_ppo / len(pulseos_sharpes) >= 0.8
    
    print(f"  Average ≥ 4.0: {'✅' if success_avg else '❌'} ({pulseos_avg:.3f})")
    print(f"  Std Dev < 0.4: {'✅' if success_std else '❌'} ({pulseos_std:.3f})")
    print(f"  80%+ beat PPO: {'✅' if success_rate else '❌'} ({beats_ppo/len(pulseos_sharpes)*100:.1f}%)")
    
    if success_avg and success_std and success_rate:
        print(f"\n🎉 SUCCESS! All criteria met!")
    elif success_avg and success_std:
        print(f"\n✅ Excellent progress - Average and variance criteria met!")
    elif success_avg:
        print(f"\n✅ Good progress - Average criterion met, variance needs work")
    else:
        print(f"\n⚠️  Need further optimization")
    
    # Generate report
    output_dir = "benchmark_results/trading_rl/pulseos_enhancements"
    generate_summary_report(results, output_dir)
    print(f"\n📄 Report saved to {output_dir}")
    
    print("\n" + "=" * 80)
    print("✅ PULSEOS ENHANCEMENTS TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())


