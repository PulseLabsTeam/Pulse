"""
Test script to validate the new strategy implementations
"""
import asyncio
from trading_rl_test import run_trading_test

async def main():
    print("🚀 Testing All Strategy Implementations\n")
    print("=" * 80)
    
    # Test with Strategy 4 (V6 Replication) which includes all strategies
    # This uses:
    # - Strategy 1: Grace period, minimal penalties
    # - Strategy 2: No death penalties, only positive rewards
    # - Strategy 3: Aggressive filtering
    # - Strategy 4: Warm start with filtering
    # - Strategy 5: Adaptive threshold (via Strategy 6)
    # - Strategy 6: Curriculum learning
    # - Strategy 7: Enhanced recovery bonuses
    
    results = await run_trading_test(
        symbol="SPY",
        num_trials=10,  # Run 10 trials for better statistics
        max_episodes=500,  # 500 episodes per trial
        target_sharpe=1.5,
        target_return=0.15,
        test_mode="fixed_seeds_warm_start",  # Uses Strategy 4 + all others
        death_penalty_multiplier=100.0,
        threshold_percentile=None,  # Use curriculum learning instead
        threshold_fixed=None
    )
    
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nPPO Average Sharpe: {results.ppo_avg_final_sharpe:.3f}")
    print(f"PulseOS Average Sharpe: {results.pulseos_avg_final_sharpe:.3f}")
    print(f"Improvement: {(results.pulseos_avg_final_sharpe / results.ppo_avg_final_sharpe - 1) * 100:+.1f}%")
    print(f"\nPPO Trials: {len(results.ppo_results)}")
    print(f"PulseOS Trials: {len(results.pulseos_results)}")
    
    if results.pulseos_avg_episodes_to_sharpe and results.ppo_avg_episodes_to_sharpe:
        print(f"\nSample Efficiency:")
        print(f"  PPO: {results.ppo_avg_episodes_to_sharpe:.1f} episodes")
        print(f"  PulseOS: {results.pulseos_avg_episodes_to_sharpe:.1f} episodes")
        if results.sample_efficiency_improvement:
            print(f"  Improvement: {results.sample_efficiency_improvement:+.1f}%")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(main())


