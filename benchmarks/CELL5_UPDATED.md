# 🚀 Cell 5: Quick Validation Test (Updated - Harder Target)

**Copy this ENTIRE code into Cell 5:**

```
import asyncio
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, 'benchmarks')

# IMPORTANT: Reload module to get updated code with progress prints
import importlib
if 'real_llm_rlhf_experiment' in sys.modules:
    importlib.reload(sys.modules['real_llm_rlhf_experiment'])

from real_llm_rlhf_experiment import run_experiment, ExperimentConfig

# QUICK VALIDATION TEST - Harder target to measure real differences
config = ExperimentConfig(
    model_name="gpt2",
    dataset_name="Anthropic/hh-rlhf",
    num_trials=1,  # 1 trial each
    max_samples=10000,  # Up to 10k samples per trial
    target_reward=0.85,  # Harder target (was 0.7 - too easy!)
    convergence_window=20,
    batch_size=8,
    max_length=128,
    learning_rate=1.41e-5,
    ppo_epochs=4,
    ppo_clip_epsilon=0.2,
    seed=42,
    output_dir="benchmark_results/real_llm_rlhf",
    device="cuda",
    reward_model_samples=1000,
    reward_model_epochs=3,
    dataset_size=20000  # 20k samples
)

print("🚀 Starting QUICK VALIDATION Test...")
print(f"Device: {config.device}")
print(f"Trials: {config.num_trials} each (baseline + PulseOS)")
print(f"Max samples per trial: {config.max_samples}")
print(f"Target reward: {config.target_reward} (harder target)")
print(f"Dataset size: {config.dataset_size} samples")
print("="*80)
print("⚠️ Expected time: 30-60 minutes")
print("="*80)

results = await run_experiment(config)
```

## What Changed:
- ✅ `target_reward=0.85` (was 0.7) - **Harder target**
- ✅ All other settings same
- ✅ Progress prints included

## Expected Results:
- Baseline PPO: Should take 50-200+ samples to converge
- PulseOS: Should show improvement if more efficient
- **Real sample complexity differences!**

**Copy-paste this entire code into Cell 5!** 🎯


