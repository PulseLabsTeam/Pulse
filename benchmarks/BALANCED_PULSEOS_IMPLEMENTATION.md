# Balanced PulseOS Implementation: Survival Pressure + Alpha-Seeking

## Core Principle Alignment

**PulseOS Core Principle**: Survival-pressure learning where agents maintain performance above thresholds while adapting learning parameters.

**Trading Domain Requirement**: Needs risk-taking, exploration, and alpha-seeking (exceeding baseline significantly).

## The Balanced Approach

We've implemented a **balanced system** that:
1. ✅ **Maintains survival pressure** (core PulseOS principle)
2. ✅ **Encourages alpha-seeking** (trading domain requirement)
3. ✅ **Allows risk-taking** (less conservative)
4. ✅ **Maintains exploration** (even when ALIVE)

## Key Improvements

### 1. Alpha-Seeking Bonus Reward

**Problem**: Original implementation only rewarded beating baseline, not exceeding it significantly.

**Solution**: Added quadratic alpha bonus for exceeding baseline by large margins:
```python
if distance_to_baseline > 0:
    alpha_bonus = 0.3 * (distance_to_baseline ** 2)
    # At +0.5 above: bonus = 0.075
    # At +1.0 above: bonus = 0.3
    # At +2.0 above: bonus = 1.2 (MASSIVE!)
```

**Impact**: Encourages agents to seek alpha, not just survive.

### 2. Reduced Penalties (Less Conservative)

**Problem**: Original penalties were too harsh, discouraging risk-taking.

**Solution**: Reduced penalty weights:
- **STRUGGLING**: Linear penalty (was quadratic) - less harsh
- **DYING**: Reduced from -1.0 to -0.7 weight - allows more exploration
- **Max penalty**: -0.35 (was -0.5) - less conservative

**Impact**: Allows risk-taking without excessive punishment.

### 3. Higher Learning Rate When ALIVE

**Problem**: Original reduced LR too much when ALIVE (0.5x-0.85x), discouraging exploration.

**Solution**: Keep LR higher when ALIVE:
- **New range**: 0.7x-0.94x (was 0.5x-0.85x)
- **Floor**: 0.7x (was 0.5x)

**Impact**: Maintains exploration even when performing well.

### 4. Higher Exploration When ALIVE

**Problem**: Original reduced exploration too much when ALIVE (0.9x multiplier, floor 0.01), discouraging alpha-seeking.

**Solution**: Keep exploration higher when ALIVE:
- **Multiplier**: 0.95x (was 0.9x)
- **Floor**: 0.05 (was 0.01)

**Impact**: Continues exploring for alpha opportunities even when ALIVE.

## Complete Reward Structure

### ALIVE (signal > 0.7)
- **Survival bonus**: Quadratic scaling (0 to 0.75)
- **Alpha bonus**: Quadratic for exceeding baseline (+0.075 to +1.2+)
- **Total**: Can reach 1.95+ for peak performance with alpha

### STRUGGLING (0.4 < signal ≤ 0.7)
- **Penalty**: Linear scaling (-0.05 to -0.1)
- **Less harsh**: Allows recovery attempts

### DYING (signal ≤ 0.4)
- **Penalty**: Quadratic but reduced (-0.0875 to -0.35)
- **Less conservative**: Allows exploration to recover

## Parameter Changes Summary

| Parameter | Before | After | Impact |
|-----------|--------|-------|--------|
| LR when ALIVE | 0.5x-0.85x | 0.7x-0.94x | More exploration |
| Exploration when ALIVE | 0.9x (floor 0.01) | 0.95x (floor 0.05) | More alpha-seeking |
| STRUGGLING penalty | Quadratic -0.045 to -0.075 | Linear -0.05 to -0.1 | Less harsh |
| DYING penalty | -0.125 to -0.5 | -0.0875 to -0.35 | Less conservative |
| Alpha bonus | None | +0.075 to +1.2+ | Encourages alpha |

## Expected Behavior

**Before (Too Conservative)**:
- Beat baseline → Reduce exploration → Stop seeking alpha → Drift
- Penalties too harsh → Avoid risk-taking → Miss opportunities

**After (Balanced)**:
- Beat baseline → Keep exploring → Seek alpha → Maintain/exceed performance
- Penalties moderate → Allow risk-taking → Find opportunities

## Core Principle Alignment

✅ **Survival Pressure**: Still maintained (penalties for dying)
✅ **Adaptive Parameters**: Still adapt (but less conservatively)
✅ **Constraint Satisfaction**: Still required (must beat baseline)
✅ **Alpha-Seeking**: Now encouraged (bonus for exceeding significantly)
✅ **Risk-Taking**: Now allowed (reduced penalties)

## Testing

This balanced approach should:
1. Maintain survival pressure (core PulseOS)
2. Encourage alpha-seeking (trading requirement)
3. Allow risk-taking (less conservative)
4. Beat PPO baseline consistently

Let's test to see if this balanced approach works better!



