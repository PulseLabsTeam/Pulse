"""
Test TRUE PulseOS with optimized configuration:
- Death Penalty: -5.0 (best from sweep)
- Survival Signal: Progressive relaxation (implemented)
- Episodes: 500 (extended training)
"""

import asyncio
import numpy as np
from trading_rl_test import run_trading_test

async def main():
    print("=" * 80)
    print("TRUE PulseOS Extended Training Test")
    print("=" * 80)
    print("Configuration:")
    print("  - Death Penalty: -5.0")
    print("  - Survival Signal: Progressive relaxation")
    print("  - Episodes: 500")
    print("  - Trials: 5")
    print("=" * 80)
    
    results = await run_trading_test('SPY', 5, 500, 1.5, 0.15, 'standard', -5.0)
    
    pulseos_sharpes = [r.final_sharpe for r in results.pulseos_results]
    ppo_sharpes = [r.final_sharpe for r in results.ppo_results]
    
    print(f'\n=== FINAL RESULTS ===')
    print(f'PPO Avg Sharpe: {np.mean(ppo_sharpes):.3f} ± {np.std(ppo_sharpes):.3f}')
    print(f'PulseOS Avg Sharpe: {np.mean(pulseos_sharpes):.3f} ± {np.std(pulseos_sharpes):.3f}')
    improvement = ((np.mean(pulseos_sharpes) - np.mean(ppo_sharpes)) / np.mean(ppo_sharpes)) * 100
    print(f'Improvement: {improvement:+.1f}%')
    
    beats_ppo = sum(1 for s in pulseos_sharpes if s > np.mean(ppo_sharpes))
    print(f'Trials beating PPO average: {beats_ppo}/{len(pulseos_sharpes)}')
    
    print(f'\nIndividual PulseOS Results:')
    for i, sharpe in enumerate(pulseos_sharpes, 1):
        status = "✅ BEATS PPO" if sharpe > np.mean(ppo_sharpes) else "❌ Below PPO"
        print(f'  Trial {i}: Sharpe={sharpe:.3f} {status}')

if __name__ == "__main__":
    asyncio.run(main())



