"""
Quick test script with fewer trials to get results faster
"""
import asyncio
from trading_rl_test import run_trading_test

async def main():
    print("🚀 Quick Test: All Strategy Implementations\n")
    print("=" * 80)
    
    # Quick test with fewer trials
    results = await run_trading_test(
        symbol="SPY",
        num_trials=3,  # Reduced for faster testing
        max_episodes=500,
        target_sharpe=1.5,
        target_return=0.15,
        test_mode="fixed_seeds_warm_start",
        death_penalty_multiplier=100.0,
        threshold_percentile=None,
        threshold_fixed=None
    )
    
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nPPO Average Sharpe: {results.ppo_avg_final_sharpe:.3f}")
    print(f"PulseOS Average Sharpe: {results.pulseos_avg_final_sharpe:.3f}")
    improvement = (results.pulseos_avg_final_sharpe / results.ppo_avg_final_sharpe - 1) * 100 if results.ppo_avg_final_sharpe > 0 else 0
    print(f"Improvement: {improvement:+.1f}%")
    print(f"\nPPO Trials: {len(results.ppo_results)}")
    print(f"PulseOS Trials: {len(results.pulseos_results)}")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(main())


