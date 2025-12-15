"""
Quick test with minimal episodes to verify everything works
"""

import asyncio
from trading_rl_test import run_trading_test, generate_summary_report

async def quick_test():
    """Run a quick test with minimal episodes"""
    print("Running quick test (1 trial, 100 episodes)...")
    
    results = await run_trading_test(
        symbol="SPY",
        num_trials=1,  # Just 1 trial for quick test
        max_episodes=100,  # Very few episodes
        target_sharpe=1.5,
        target_return=0.15
    )
    
    # Generate report
    report_path = generate_summary_report(results)
    print(f"\n✅ Quick test completed! Results: {report_path}")
    print(f"\nPPO Episodes to Sharpe≥1.5: {results.ppo_avg_episodes_to_sharpe}")
    print(f"PulseOS Episodes to Sharpe≥1.5: {results.pulseos_avg_episodes_to_sharpe}")
    if results.sample_efficiency_improvement:
        print(f"Sample Efficiency Improvement: {results.sample_efficiency_improvement:.1f}%")

if __name__ == "__main__":
    asyncio.run(quick_test())




