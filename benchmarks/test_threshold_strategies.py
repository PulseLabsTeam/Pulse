"""
Test Threshold Strategies

Comprehensive test suite for different survival threshold strategies.
Tests multiple threshold approaches to find optimal configuration.
"""

import asyncio
import numpy as np
from typing import List, Dict, Optional
from trading_rl_test import run_trading_test, TestResults, generate_summary_report


async def test_threshold_strategies(
    symbol: str = "SPY",
    num_trials: int = 5,
    max_episodes: int = 500,
    test_mode: str = "standard"
) -> Dict[str, TestResults]:
    """
    Test multiple threshold strategies.
    
    Args:
        symbol: Stock symbol to trade
        num_trials: Number of trials per strategy
        max_episodes: Maximum episodes per trial
        test_mode: Test mode (standard, warm_start, etc.)
        
    Returns:
        Dictionary mapping strategy names to test results
    """
    print("=" * 80)
    print("🎯 THRESHOLD STRATEGY TESTING")
    print("=" * 80)
    print(f"\nTesting {num_trials} trials per strategy with {max_episodes} episodes each")
    print(f"Symbol: {symbol}, Test Mode: {test_mode}\n")
    
    results = {}
    
    # Strategy 1: Baseline (no threshold adjustment)
    print("\n" + "=" * 80)
    print("STRATEGY 1: Baseline (PPO baseline as threshold)")
    print("=" * 80)
    baseline_results = await run_trading_test(
        symbol=symbol,
        num_trials=num_trials,
        max_episodes=max_episodes,
        test_mode=test_mode,
        threshold_percentile=None,
        threshold_fixed=None
    )
    results["baseline"] = baseline_results
    
    # Strategy 2: 10th Percentile (~3.0 Sharpe for baseline ~3.6)
    print("\n" + "=" * 80)
    print("STRATEGY 2: 10th Percentile Threshold")
    print("=" * 80)
    percentile_10_results = await run_trading_test(
        symbol=symbol,
        num_trials=num_trials,
        max_episodes=max_episodes,
        test_mode=test_mode,
        threshold_percentile=0.1,
        threshold_fixed=None
    )
    results["percentile_10"] = percentile_10_results
    
    # Strategy 3: 15th Percentile
    print("\n" + "=" * 80)
    print("STRATEGY 3: 15th Percentile Threshold")
    print("=" * 80)
    percentile_15_results = await run_trading_test(
        symbol=symbol,
        num_trials=num_trials,
        max_episodes=max_episodes,
        test_mode=test_mode,
        threshold_percentile=0.15,
        threshold_fixed=None
    )
    results["percentile_15"] = percentile_15_results
    
    # Strategy 4: 20th Percentile
    print("\n" + "=" * 80)
    print("STRATEGY 4: 20th Percentile Threshold")
    print("=" * 80)
    percentile_20_results = await run_trading_test(
        symbol=symbol,
        num_trials=num_trials,
        max_episodes=max_episodes,
        test_mode=test_mode,
        threshold_percentile=0.20,
        threshold_fixed=None
    )
    results["percentile_20"] = percentile_20_results
    
    # Strategy 5: Fixed 2.5 Sharpe
    print("\n" + "=" * 80)
    print("STRATEGY 5: Fixed 2.5 Sharpe Threshold")
    print("=" * 80)
    fixed_25_results = await run_trading_test(
        symbol=symbol,
        num_trials=num_trials,
        max_episodes=max_episodes,
        test_mode=test_mode,
        threshold_percentile=None,
        threshold_fixed=2.5
    )
    results["fixed_2.5"] = fixed_25_results
    
    # Strategy 6: Fixed 3.0 Sharpe
    print("\n" + "=" * 80)
    print("STRATEGY 6: Fixed 3.0 Sharpe Threshold")
    print("=" * 80)
    fixed_30_results = await run_trading_test(
        symbol=symbol,
        num_trials=num_trials,
        max_episodes=max_episodes,
        test_mode=test_mode,
        threshold_percentile=None,
        threshold_fixed=3.0
    )
    results["fixed_3.0"] = fixed_30_results
    
    return results


def compare_strategies(results: Dict[str, TestResults]) -> None:
    """Compare all threshold strategies and print summary."""
    print("\n" + "=" * 80)
    print("📊 THRESHOLD STRATEGY COMPARISON")
    print("=" * 80)
    
    comparison_data = []
    
    for strategy_name, test_results in results.items():
        pulseos_sharpes = [r.final_sharpe for r in test_results.pulseos_results]
        ppo_avg = np.mean([r.final_sharpe for r in test_results.ppo_results])
        
        avg_sharpe = np.mean(pulseos_sharpes)
        std_sharpe = np.std(pulseos_sharpes)
        beats_ppo = sum(1 for s in pulseos_sharpes if s >= ppo_avg)
        beats_ppo_pct = (beats_ppo / len(pulseos_sharpes)) * 100
        improvement = ((avg_sharpe / ppo_avg) - 1) * 100
        
        comparison_data.append({
            "strategy": strategy_name,
            "avg_sharpe": avg_sharpe,
            "std_sharpe": std_sharpe,
            "beats_ppo": beats_ppo,
            "beats_ppo_pct": beats_ppo_pct,
            "improvement_pct": improvement,
            "ppo_baseline": ppo_avg
        })
    
    # Sort by average Sharpe ratio
    comparison_data.sort(key=lambda x: x["avg_sharpe"], reverse=True)
    
    print("\nStrategy Performance Summary:")
    print("-" * 80)
    print(f"{'Strategy':<20} {'Avg Sharpe':<12} {'Std Dev':<10} {'Beats PPO':<12} {'Improvement':<12}")
    print("-" * 80)
    
    for data in comparison_data:
        print(f"{data['strategy']:<20} {data['avg_sharpe']:>10.3f}  {data['std_sharpe']:>8.3f}  "
              f"{data['beats_ppo']:>3}/{len(results[data['strategy']].pulseos_results):<3} "
              f"({data['beats_ppo_pct']:>5.1f}%)  {data['improvement_pct']:>+10.1f}%")
    
    # Find best strategy
    best = comparison_data[0]
    print("\n" + "=" * 80)
    print(f"🏆 BEST STRATEGY: {best['strategy']}")
    print("=" * 80)
    print(f"  Average Sharpe: {best['avg_sharpe']:.3f} (vs PPO baseline: {best['ppo_baseline']:.3f})")
    print(f"  Improvement: {best['improvement_pct']:+.1f}%")
    print(f"  Std Dev: {best['std_sharpe']:.3f}")
    print(f"  Beats PPO: {best['beats_ppo']}/{len(results[best['strategy']].pulseos_results)} ({best['beats_ppo_pct']:.1f}%)")
    
    # Success criteria check
    print("\n🎯 Success Criteria Check:")
    success_avg = best['avg_sharpe'] >= 4.0
    success_std = best['std_sharpe'] < 0.4
    success_rate = best['beats_ppo_pct'] >= 80
    
    print(f"  Average ≥ 4.0: {'✅' if success_avg else '❌'} ({best['avg_sharpe']:.3f})")
    print(f"  Std Dev < 0.4: {'✅' if success_std else '❌'} ({best['std_sharpe']:.3f})")
    print(f"  80%+ beat PPO: {'✅' if success_rate else '❌'} ({best['beats_ppo_pct']:.1f}%)")
    
    if success_avg and success_std and success_rate:
        print("\n🎉 SUCCESS! All criteria met!")
    elif success_avg and success_std:
        print("\n✅ Excellent progress - Average and variance criteria met!")
    elif success_avg:
        print("\n✅ Good progress - Average criterion met, variance needs work")
    else:
        print("\n⚠️  Need further optimization")
    
    return comparison_data


async def main():
    """Main test execution."""
    print("\n🚀 Starting Threshold Strategy Testing\n")
    
    # Run comprehensive threshold strategy tests
    results = await test_threshold_strategies(
        symbol="SPY",
        num_trials=5,
        max_episodes=500,
        test_mode="standard"
    )
    
    # Compare strategies
    comparison = compare_strategies(results)
    
    # Generate reports for each strategy
    print("\n📄 Generating detailed reports...")
    for strategy_name, test_results in results.items():
        output_dir = f"benchmark_results/trading_rl/threshold_strategies/{strategy_name}"
        generate_summary_report(test_results, output_dir)
        print(f"  ✅ {strategy_name}: Report saved to {output_dir}")
    
    print("\n" + "=" * 80)
    print("✅ ALL THRESHOLD STRATEGY TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())


