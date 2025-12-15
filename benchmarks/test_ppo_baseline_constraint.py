"""
Quick test of PPO Baseline Survival Constraint

This script tests the new PPO baseline survival constraint implementation.
"""

import asyncio
import sys
import os

# Add parent directory to path to import from benchmarks
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.trading_rl_test import run_trading_test


async def main():
    """Run a quick test with PPO baseline survival constraint"""
    print("=" * 80)
    print("🧪 TESTING PPO BASELINE SURVIVAL CONSTRAINT")
    print("=" * 80)
    print()
    print("This test will:")
    print("  1. Run PPO trials to establish baseline Sharpe ratio")
    print("  2. Run PulseOS trials with survival = 'beat PPO baseline'")
    print("  3. Verify that agents must outperform PPO to survive")
    print()
    
    # Run extended test (3 trials each, 600 episodes)
    results = await run_trading_test(
        symbol="SPY",
        num_trials=3,  # More trials for better statistics
        max_episodes=600,  # Extended training to allow agents to learn consistently
        target_sharpe=1.5,
        target_return=0.15,
        test_mode="standard"  # Standard mode uses PPO baseline constraint
    )
    
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS")
    print("=" * 80)
    print(f"PPO Baseline Sharpe: {results.ppo_avg_final_sharpe:.3f}")
    print(f"PulseOS Average Sharpe: {results.pulseos_avg_final_sharpe:.3f}")
    
    if results.pulseos_avg_final_sharpe >= results.ppo_avg_final_sharpe:
        print("✅ SUCCESS: PulseOS beat PPO baseline!")
    else:
        print("⚠️  PulseOS did not beat PPO baseline in this test")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

