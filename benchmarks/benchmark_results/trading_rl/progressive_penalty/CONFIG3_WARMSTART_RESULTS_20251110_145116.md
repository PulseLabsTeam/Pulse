# Configuration 3: Warm Start from PPO + Progressive Death Penalty

**Date**: 2025-11-10 14:51:16

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

- **PPO Avg**: 3.644 ± 0.365
- **PulseOS Avg**: 2.421 ± 0.422
- **Improvement**: -33.6%
- **Std Dev**: 0.422
- **Trials Beating PPO**: 0/5

## Individual Results

### PulseOS
- Trial 1: 2.067 ❌ Below PPO
- Trial 2: 2.058 ❌ Below PPO
- Trial 3: 2.908 ❌ Below PPO
- Trial 4: 2.104 ❌ Below PPO
- Trial 5: 2.965 ❌ Below PPO

### PPO
- Trial 1: 3.061
- Trial 2: 3.689
- Trial 3: 4.212
- Trial 4: 3.620
- Trial 5: 3.638

## Comparison

| Configuration | Avg Sharpe | Std Dev | Improvement | Trials Beating PPO |
|---------------|------------|---------|-------------|-------------------|
| Config 1 | 2.557 | 0.662 | -34.0% | 0/5 |
| Config 2 | 2.671 | 0.883 | -27.3% | 1/5 |
| **Config 3** | **2.421** | **0.422** | **-33.6%** | **0/5** |
