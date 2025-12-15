# Config 2 Trial 7: Enhanced Config (Optimized Init + Milder Penalties + Relaxed Survival Signal)

**Date**: 2025-11-10 16:01:47

## Test Configuration

- **Episodes**: 1000
- **Seed**: 7
- **Optimized Initialization**: Yes (based on seed 1 analysis)
- **Enhanced Config**: Yes
  - Milder penalties: -0.1 early (was -0.25), -0.5 mid (was -1.0), -2.0 late (was -3.0)
  - Relaxed survival signal: DYING threshold < 0.2 (was < 0.3)
- **No Warm Start**: Independent training

## Results

- **PPO Baseline**: 3.625
- **Config 2 Final Sharpe**: 2.125
- **Improvement vs PPO**: -41.4%
- **Beats PPO**: No
- **Exceeds 4.0**: No

## All Trials Comparison

| Trial | Episodes | Seed | Init | Config | Sharpe | Improvement | Status |
|-------|----------|------|------|--------|--------|-------------|--------|
| Trial 1 (prev) | 500 | Unknown | Standard | Standard | 4.259 | +17.5% | ✅ |
| Trial 1 | 1000 | 1 | Standard | Standard | 3.688 | +1.7% | ✅ |
| Trial 2 | 1000 | 2 | Standard | Standard | 2.053 | -43.4% | ❌ |
| Trial 3 | 1000 | 3 | Improved | Standard | 3.077 | -15.1% | ❌ |
| Trial 4 | 1000 | 4 | Improved | Standard | 3.536 | -2.5% | ❌ |
| Trial 5 | 1000 | 5 | Optimized | Standard | 3.648 | +0.6% | ✅ |
| Trial 6 | 1000 | 6 | Optimized | Standard | 2.024 | -44.2% | ❌ |
| Trial 7 | 1000 | 7 | Optimized | Enhanced | 2.125 | -41.4% | ❌ |

**Test Duration**: 1.1 minutes
