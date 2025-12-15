# Enhanced Survival Reward System

## Reward Structure

The survival signal now provides **both rewards and penalties** with aggressive scaling:

### High Performance Rewards (ALIVE: signal > 0.7)

**Quadratic + Exponential Scaling:**
- At 0.7 signal (just ALIVE): bonus = 0
- At 0.85 signal: bonus ≈ 0.125
- At 0.9 signal: bonus ≈ 0.22
- At 1.0 signal (peak): bonus ≈ 0.75 ⚡ **MASSIVE REWARD!**

**Formula:**
```python
if survival_signal > 0.7:
    distance = (survival_signal - 0.7) / 0.3
    bonus = weight * (distance ** 2)
    if survival_signal > 0.9:
        bonus += weight * 0.5 * ((survival_signal - 0.9) / 0.1) ** 2
```

### Struggling Penalties (0.4 < signal ≤ 0.7)

**Quadratic Penalty:**
- At 0.7 signal: penalty = 0
- At 0.55 signal: penalty ≈ -0.045
- At 0.4 signal: penalty ≈ -0.075

**Formula:**
```python
if 0.4 < survival_signal <= 0.7:
    distance = (0.7 - survival_signal) / 0.3
    penalty = -0.3 * weight * (distance ** 1.5)
```

### Dying Penalties (signal ≤ 0.4)

**Exponential Penalty:**
- At 0.4 signal: penalty = 0
- At 0.2 signal: penalty ≈ -0.125
- At 0.0 signal: penalty ≈ -0.5 ⚠️ **MASSIVE PENALTY!**

**Formula:**
```python
if survival_signal <= 0.4:
    distance = (0.4 - survival_signal) / 0.4
    penalty = -1.0 * weight * (distance ** 2)
```

## Why This Works

### Asymmetric Incentive Structure

1. **High rewards for peak performance** → Strong incentive to maintain excellence
2. **Moderate penalties for struggling** → Warning signal before collapse
3. **Severe penalties for dying** → Strong disincentive to fail

### The Complete Picture

```
Survival Signal → Reward/Penalty
─────────────────────────────────
1.0 (peak)      → +0.75 ⚡ MASSIVE REWARD
0.9             → +0.22
0.85            → +0.125
0.7 (alive)     → 0.0
0.55 (struggling) → -0.045 ⚠️ Warning
0.4 (dying)     → -0.075
0.2             → -0.125
0.0 (very dying) → -0.5 ⚠️ MASSIVE PENALTY
```

### Expected Behavior

**Before (weak penalties):**
- Agent can drift slowly without strong disincentive
- Small penalties don't create urgency

**After (strong penalties):**
- Agent strongly motivated to avoid low performance
- Exponential penalties create urgency when struggling
- Creates "performance floor" - agent avoids falling below baseline

## Combined Effect

The system now has:
1. **Strong rewards** for high performance (incentive to excel)
2. **Strong penalties** for low performance (disincentive to fail)
3. **Asymmetric scaling** (penalties grow faster than rewards)

This creates a **performance funnel**:
- Easy to maintain high performance (big rewards)
- Hard to accept low performance (big penalties)
- Creates stable equilibrium at high performance



