# 🚀 Quick Guide: Starting from Cell 2

## Cell 2: Add PulseOS to Path
**Copy this code:**

```
import sys
import os

sys.path.insert(0, '.')
sys.path.insert(0, 'benchmarks')

if os.path.exists('pulseos'):
    sys.path.insert(0, 'pulseos')
    print("✓ PulseOS added to Python path")
    print("✓ Ready for full experiment")
else:
    print("⚠️ pulseos folder not found - check extraction")
```

**Run it** - should complete instantly (few seconds max)

---

## Cell 3: Run FULL Experiment
**Copy this code:**

```
import asyncio
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, 'benchmarks')

from real_llm_rlhf_experiment import run_experiment, ExperimentConfig

# FULL EXPERIMENT - uses updated defaults
config = ExperimentConfig()

print("🚀 Starting FULL RLHF Experiment...")
print(f"Device: {config.device}")
print(f"Trials: {config.num_trials} each (baseline + PulseOS)")
print(f"Max samples per trial: {config.max_samples}")
print(f"Target reward: {config.target_reward}")
print(f"Dataset: {'FULL' if config.dataset_size is None else config.dataset_size}")
print("="*80)
print("⚠️ This will take 2-4 hours. Don't close browser!")
print("="*80)

results = await run_experiment(config)
```

**Run it** - This will take 2-4 hours!

---

## Cell 4: View Results (After Cell 3 completes)
**Copy this code:**

```
print(f"\n🎯 Improvement: {results.improvement_percent:.1f}%")
print(f"Baseline: {results.baseline_mean_samples:.1f} samples")
print(f"PulseOS: {results.pulseos_mean_samples:.1f} samples")

from google.colab import files
import json

with open('results.json', 'w') as f:
    json.dump({
        'improvement': results.improvement_percent,
        'baseline_mean': results.baseline_mean_samples,
        'pulseos_mean': results.pulseos_mean_samples,
        'p_value': results.p_value,
        'significant': results.significant
    }, f, indent=2)

files.download('results.json')
```

---

## Summary:
1. ✅ **Cell 2:** Add PulseOS (instant)
2. ✅ **Cell 3:** Run experiment (2-4 hours)
3. ⏳ **Wait** for Cell 3 to complete
4. ✅ **Cell 4:** View results

**Start with Cell 2!** 🎯


