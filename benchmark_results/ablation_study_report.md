# PulseOS Ablation Study Report

## Phase 4: Component Analysis

This report analyzes what components contribute to PulseOS success.

## Scenario: multi_objective_normal_th-0.5

| Configuration | Avg Steps | Std Dev | vs Full PulseOS |
|---------------|-----------|---------|-----------------|
| Full PulseOS | 49.0 | 0.0 |  |
| Without PTDC | 49.0 | 0.0 | +0.0% |
| Without NGCM | 49.0 | 0.0 | +0.0% |
| Without APC | 49.0 | 0.0 | +0.0% |
| Only Survival Constraint | 363.8 | 595.7 | +642.4% |
| PPO with Survival Constraint | 49.0 | 0.0 | +0.0% |

### Key Insights

- **Best Configuration:** Full PulseOS (49.0 steps)
- **Worst Configuration:** Only Survival Constraint (363.8 steps)
- **Full PulseOS Performance:** 49.0 steps

**Component Impact:**
- Removing PTDC: +0.0% change
- Removing NGCM: +0.0% change
- Removing APC: +0.0% change

## Scenario: linear_bimodal_th-0.5

| Configuration | Avg Steps | Std Dev | vs Full PulseOS |
|---------------|-----------|---------|-----------------|
| Full PulseOS | 49.0 | 0.0 |  |
| Without PTDC | 49.0 | 0.0 | +0.0% |
| Without NGCM | 49.0 | 0.0 | +0.0% |
| Without APC | 49.0 | 0.0 | +0.0% |
| Only Survival Constraint | 49.0 | 0.0 | +0.0% |
| PPO with Survival Constraint | 49.0 | 0.0 | +0.0% |

### Key Insights

- **Best Configuration:** Full PulseOS (49.0 steps)
- **Worst Configuration:** Full PulseOS (49.0 steps)
- **Full PulseOS Performance:** 49.0 steps

**Component Impact:**
- Removing PTDC: +0.0% change
- Removing NGCM: +0.0% change
- Removing APC: +0.0% change

## Scenario: linear_skewed_th-0.3

| Configuration | Avg Steps | Std Dev | vs Full PulseOS |
|---------------|-----------|---------|-----------------|
| Full PulseOS | 49.0 | 0.0 |  |
| Without PTDC | 49.0 | 0.0 | +0.0% |
| Without NGCM | 49.0 | 0.0 | +0.0% |
| Without APC | 50.4 | 2.8 | +2.9% |
| Only Survival Constraint | 55.4 | 12.8 | +13.1% |
| PPO with Survival Constraint | 49.0 | 0.0 | +0.0% |

### Key Insights

- **Best Configuration:** Full PulseOS (49.0 steps)
- **Worst Configuration:** Only Survival Constraint (55.4 steps)
- **Full PulseOS Performance:** 49.0 steps

**Component Impact:**
- Removing PTDC: +0.0% change
- Removing NGCM: +0.0% change
- Removing APC: +2.9% change

