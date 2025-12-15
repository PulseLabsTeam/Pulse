# Config 2 Trial 3: Improved Initialization + Seed 3

**Date**: 2025-11-10 15:50:00

## Test Configuration

- **Episodes**: 1000
- **Seed**: 3
- **Improved Initialization**: Yes
  - Smaller initial weights (0.3x instead of 0.5x)
  - Small random bias to encourage exploration
  - More conservative value initialization
- **No Warm Start**: Independent training

## Results

- **PPO Baseline**: 3.625
- **Config 2 Final Sharpe**: 3.077
- **Improvement vs PPO**: -15.1%
- **Beats PPO**: No
- **Exceeds 4.0**: No

## All Trials Comparison

| Trial | Episodes | Seed | Sharpe | Improvement | Status |
|-------|----------|------|--------|-------------|--------|
| Trial 1 (prev) | 500 | Unknown | 4.259 | +17.5% | ✅ |
| Trial 1 | 1000 | 1 | 3.688 | +1.7% | ✅ |
| Trial 2 | 1000 | 2 | 2.053 | -43.4% | ❌ |
| Trial 3 | 1000 | 3 | 3.077 | -15.1% | ❌ |

**Test Duration**: 1.1 minutes
