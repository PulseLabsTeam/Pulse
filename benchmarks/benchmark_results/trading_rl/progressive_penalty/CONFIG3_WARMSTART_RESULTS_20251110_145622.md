# Configuration 3: Warm Start from PPO + Progressive Death Penalty

**Date**: 2025-11-10 14:56:22

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

- **PPO Avg**: 3.490 ± 0.275
- **PulseOS Avg**: 2.017 ± 0.572
- **Improvement**: -42.2%
- **Std Dev**: 0.572
- **Trials Beating PPO**: 0/5

## Individual Results

### PulseOS
- Trial 1: 1.184 ❌ Below PPO
- Trial 2: 1.936 ❌ Below PPO
- Trial 3: 2.980 ❌ Below PPO
- Trial 4: 1.941 ❌ Below PPO
- Trial 5: 2.045 ❌ Below PPO

### PPO
- Trial 1: 3.674
- Trial 2: 2.943
- Trial 3: 3.626
- Trial 4: 3.581
- Trial 5: 3.628

## Comparison

| Configuration | Avg Sharpe | Std Dev | Improvement | Trials Beating PPO |
|---------------|------------|---------|-------------|-------------------|
| Config 1 | 2.557 | 0.662 | -34.0% | 0/5 |
| Config 2 | 2.671 | 0.883 | -27.3% | 1/5 |
| **Config 3** | **2.017** | **0.572** | **-42.2%** | **0/5** |
