# Configuration 3: Warm Start from PPO + Progressive Death Penalty

**Date**: 2025-11-10 15:01:30

## Configuration

- **Warm Start**: Initialize PulseOS from PPO weights
- **Death Penalty Schedule**:
  - Episodes 0-150: -0.25 (very mild)
  - Episodes 150-300: -1.0 (moderate)
  - Episodes 300+: -3.0 (moderate-high)
- **Survival Signal**: Exponential relaxation
- **Episodes**: 500
- **Trials**: 5

## Results

- **PPO Avg**: 3.604 ± 0.412
- **PulseOS Avg**: 3.512 ± 0.283
- **Improvement**: -2.6%
- **Std Dev**: 0.283
- **Trials Beating PPO**: 3/5

## Individual Results

### PulseOS
- Trial 1: 3.603 ❌ Below PPO
- Trial 2: 2.949 ❌ Below PPO
- Trial 3: 3.648 ✅ BEATS PPO
- Trial 4: 3.654 ✅ BEATS PPO
- Trial 5: 3.704 ✅ BEATS PPO

### PPO
- Trial 1: 4.238
- Trial 2: 2.938
- Trial 3: 3.631
- Trial 4: 3.637
- Trial 5: 3.576

## Comparison

| Configuration | Avg Sharpe | Std Dev | Improvement | Trials Beating PPO |
|---------------|------------|---------|-------------|-------------------|
| Config 1 | 2.557 | 0.662 | -34.0% | 0/5 |
| Config 2 | 2.671 | 0.883 | -27.3% | 1/5 |
| **Config 3** | **3.512** | **0.283** | **-2.6%** | **3/5** |
