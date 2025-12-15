# Config 2 Trial 5: Optimized Initialization + Seed 5

**Date**: 2025-11-10 15:56:07

## Test Configuration

- **Episodes**: 1000
- **Seed**: 5
- **Optimized Initialization**: Yes (based on seed 1 analysis)
  - Policy weights: 0.35x multiplier (compromise between 0.3x and 0.5x)
  - Adaptive bias: 0.005 scale (smaller, closer to seed 1's zero bias)
  - Value weights: 0.25x multiplier (slightly larger)
- **No Warm Start**: Independent training

## Results

- **PPO Baseline**: 3.625
- **Config 2 Final Sharpe**: 3.648
- **Improvement vs PPO**: +0.6%
- **Beats PPO**: Yes
- **Exceeds 4.0**: No

## All Trials Comparison

| Trial | Episodes | Seed | Init | Sharpe | Improvement | Status |
|-------|----------|------|------|--------|-------------|--------|
| Trial 1 (prev) | 500 | Unknown | Standard | 4.259 | +17.5% | ✅ |
| Trial 1 | 1000 | 1 | Standard | 3.688 | +1.7% | ✅ |
| Trial 2 | 1000 | 2 | Standard | 2.053 | -43.4% | ❌ |
| Trial 3 | 1000 | 3 | Improved | 3.077 | -15.1% | ❌ |
| Trial 4 | 1000 | 4 | Improved | 3.536 | -2.5% | ❌ |
| Trial 5 | 1000 | 5 | Optimized | 3.648 | +0.6% | ✅ |

**Test Duration**: 1.1 minutes
