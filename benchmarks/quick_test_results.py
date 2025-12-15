import asyncio
import numpy as np
from trading_rl_test import run_trading_test

async def main():
    results = await run_trading_test('SPY', 3, 200, 1.5, 0.15, 'standard', -5.0)
    
    pulseos_sharpes = [r.final_sharpe for r in results.pulseos_results]
    ppo_sharpes = [r.final_sharpe for r in results.ppo_results]
    
    print(f'\n=== RESULTS SUMMARY ===')
    print(f'PPO Avg Sharpe: {np.mean(ppo_sharpes):.3f} ± {np.std(ppo_sharpes):.3f}')
    print(f'PulseOS Avg Sharpe: {np.mean(pulseos_sharpes):.3f} ± {np.std(pulseos_sharpes):.3f}')
    improvement = ((np.mean(pulseos_sharpes) - np.mean(ppo_sharpes)) / np.mean(ppo_sharpes)) * 100
    print(f'Improvement: {improvement:+.1f}%')
    print(f'\nIndividual PulseOS Results:')
    for i, sharpe in enumerate(pulseos_sharpes, 1):
        print(f'  Trial {i}: Sharpe={sharpe:.3f}')
    print(f'\nIndividual PPO Results:')
    for i, sharpe in enumerate(ppo_sharpes, 1):
        print(f'  Trial {i}: Sharpe={sharpe:.3f}')

if __name__ == "__main__":
    asyncio.run(main())



