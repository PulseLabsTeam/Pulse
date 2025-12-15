# PPO Baseline Survival Constraint - Implementation Summary

## Overview

This implementation addresses the core problem you identified: **the survival signal was incentivizing agents to "survive by hiding" (doing nothing) rather than "survive by outperforming"**.

## The Problem

**Old Survival Signal:**
- Survival = "don't lose money" (fixed threshold)
- Solution = "don't trade"
- Agent survives perfectly by doing nothing
- This is like grading a student: "You pass if you don't fail any tests"
- Student: "I'll just not take any tests" → Technically passes, learns nothing

## The Solution

**New Survival Signal:**
- Survival = "Must beat PPO baseline performance"
- Agent survives if: `agent_sharpe >= ppo_baseline_sharpe`
- Can't game by not trading (0% return won't beat PPO)
- Forces active competitive performance
- Survival = competitive advantage

## Implementation Details

### 1. New Constraint Class: `PPOBaselineSurvivalConstraint`

**Location:** `benchmarks/ppo_baseline_constraint.py`

**Features:**
- Compares agent's Sharpe ratio directly to PPO baseline Sharpe ratio
- Supports statistical and temporal evaluation modes
- Tracks Sharpe ratio history for each agent
- Provides detailed survival status information

**Key Methods:**
- `evaluate_sharpe(agent_id, sharpe_ratio)` - Check if agent survives based on Sharpe ratio
- `get_survival_status(agent_id)` - Get detailed survival status
- `update_baseline(new_ppo_baseline_sharpe)` - Update baseline if needed

### 2. Modified Trading Test

**Location:** `benchmarks/trading_rl_test.py`

**Changes:**
- PPO trials run FIRST to establish baseline Sharpe ratio
- PPO baseline is computed as average of all PPO trial Sharpe ratios
- PulseOS trials use `PPOBaselineSurvivalConstraint` with PPO baseline
- Survival signal computed based on Sharpe ratio comparison (not normalized metric)
- Logging shows survival status: "ALIVE" (beating baseline) or "DYING" (below baseline)

**Key Modifications:**
- `run_pulseos_trial()` now accepts `ppo_baseline_sharpe` parameter
- Runtime step overrides survival signal when using PPO baseline constraint
- All test modes updated to pass PPO baseline to trials

### 3. Agent Enhancement

**Location:** `benchmarks/pulseos_trading_agent.py`

**Added:**
- `get_sharpe_ratio()` method to expose current Sharpe ratio for comparison

## How It Works

1. **PPO Baseline Establishment:**
   ```
   Run PPO trials → Compute average Sharpe ratio → Use as survival threshold
   ```

2. **Survival Evaluation:**
   ```
   After each episode:
   - Get agent's Sharpe ratio
   - Compare to PPO baseline
   - If agent_sharpe >= ppo_baseline_sharpe → ALIVE (survival_signal = 0.7)
   - If agent_sharpe < ppo_baseline_sharpe → DYING (survival_signal = 0.0)
   ```

3. **Adaptive Learning:**
   ```
   High survival signal (beating baseline) → Lower learning rate, less exploration
   Low survival signal (below baseline) → Higher learning rate, more exploration
   ```

## Testing

### Quick Test

Run the test script:
```bash
cd benchmarks
python test_ppo_baseline_constraint.py
```

### Full Test

Run the main trading test:
```bash
cd benchmarks
python trading_rl_test.py
```

The test will:
1. Run PPO trials to establish baseline
2. Run PulseOS trials with PPO baseline survival constraint
3. Show survival status throughout training
4. Compare final performance

## Expected Behavior

**Before (Old Constraint):**
- Agent could survive by doing nothing (0% return)
- No competitive pressure
- Learning stagnates

**After (PPO Baseline Constraint):**
- Agent MUST beat PPO baseline to survive
- Competitive pressure forces active trading
- Learning adapts based on relative performance

## Example Output

```
✅ PPO Baseline Established: Average Sharpe Ratio = 3.729
   PulseOS agents must beat this baseline to survive!

🚀 Running PulseOS Trials...
  Episode 10: Sharpe=2.5, PPO Baseline=3.729, Survival=DYING, Signal=0.000
  Episode 20: Sharpe=3.8, PPO Baseline=3.729, Survival=ALIVE, Signal=0.700
  Episode 30: Sharpe=4.2, PPO Baseline=3.729, Survival=ALIVE, Signal=0.700
```

## Benefits

1. ✅ **Prevents Gaming:** Can't survive by doing nothing
2. ✅ **Forces Competition:** Must outperform baseline
3. ✅ **Meaningful Pressure:** Survival = competitive advantage
4. ✅ **Adaptive Learning:** Learning rate adjusts based on relative performance
5. ✅ **Clear Metrics:** Easy to see if agent is beating baseline

## Next Steps

1. Run tests to verify implementation
2. Compare results with old constraint
3. Tune parameters (margin, temporal window, etc.)
4. Consider adaptive baseline updates if running multiple PPO trials



