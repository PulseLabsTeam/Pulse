# 🚀 Quick Validation Test - Cell 5

**Copy this code into Cell 5:**

```
import asyncio
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, 'benchmarks')

# Reload module to clear cache
import importlib
if 'real_llm_rlhf_experiment' in sys.modules:
    importlib.reload(sys.modules['real_llm_rlhf_experiment'])

from real_llm_rlhf_experiment import run_experiment, ExperimentConfig

# QUICK VALIDATION TEST - Smaller scale but still meaningful
config = ExperimentConfig(
    model_name="gpt2",
    dataset_name="Anthropic/hh-rlhf",
    num_trials=1,  # 1 trial each (baseline + PulseOS = 2 total)
    max_samples=10000,  # Still test up to 10k samples per trial
    target_reward=0.7,  # Realistic target
    convergence_window=20,
    batch_size=8,
    max_length=128,
    learning_rate=1.41e-5,
    ppo_epochs=4,
    ppo_clip_epsilon=0.2,
    seed=42,
    output_dir="benchmark_results/real_llm_rlhf",
    device="cuda",
    reward_model_samples=1000,  # Good reward model
    reward_model_epochs=3,
    dataset_size=20000  # 20k samples (meaningful but faster)
)

print("🚀 Starting QUICK VALIDATION Test...")
print(f"Device: {config.device}")
print(f"Trials: {config.num_trials} each (baseline + PulseOS)")
print(f"Max samples per trial: {config.max_samples}")
print(f"Target reward: {config.target_reward}")
print(f"Dataset size: {config.dataset_size} samples")
print("="*80)
print("⚠️ Expected time: 30-60 minutes")
print("="*80)

results = await run_experiment(config)
```

## What This Tests:
- ✅ **20,000 samples** from HH-RLHF dataset
- ✅ **1 trial each** (baseline PPO + PulseOS)
- ✅ **Up to 10,000 samples** per trial to reach target
- ✅ **Target reward: 0.7** (realistic)
- ✅ **Proper reward model** (1000 samples, 3 epochs)

## Expected Runtime:
- **30-60 minutes** (much faster than full experiment)
- Still produces **valuable data** to see if PulseOS is better

## What You'll Get:
- Sample complexity comparison (baseline vs PulseOS)
- Improvement percentage
- Statistical analysis (though limited with 1 trial)
- Quick validation if PulseOS shows promise

**Ready to run!** 🎯


