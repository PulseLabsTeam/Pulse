# Config 2 Trial 4: Improved Initialization + Seed 4

**Date**: 2025-11-10 15:52:58

## Test Configuration

- **Episodes**: 1000
- **Seed**: 4
- **Improved Initialization**: Yes
  - Smaller initial weights (0.3x instead of 0.5x)
  - Small random bias to encourage exploration
  - More conservative value initialization
- **No Warm Start**: Independent training

## Results

- **PPO Baseline**: 3.625
- **Config 2 Final Sharpe**: 3.536
- **Improvement vs PPO**: -2.5%
- **Beats PPO**: No
- **Exceeds 4.0**: No

## All Trials Comparison

| Trial | Episodes | Seed | Init | Sharpe | Improvement | Status |
|-------|----------|------|------|--------|-------------|--------|
| Trial 1 (prev) | 500 | Unknown | Standard | 4.259 | +17.5% | ✅ |
| Trial 1 | 1000 | 1 | Standard | 3.688 | +1.7% | ✅ |
| Trial 2 | 1000 | 2 | Standard | 2.053 | -43.4% | ❌ |
| Trial 3 | 1000 | 3 | Improved | 3.077 | -15.1% | ❌ |
| Trial 4 | 1000 | 4 | Improved | 3.536 | -2.5% | ❌ |

**Test Duration**: 1.0 minutes
