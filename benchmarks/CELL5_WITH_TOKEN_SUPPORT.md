# 🔑 Updated: Cell 5 with Token Support

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

# OPTIONAL: Set Hugging Face token if needed (HH-RLHF is public, usually not needed)
# Uncomment and add your token if you get authentication errors:
# os.environ["HF_TOKEN"] = "hf_your_token_here"

# QUICK VALIDATION TEST
config = ExperimentConfig(
    model_name="gpt2",
    dataset_name="Anthropic/hh-rlhf",
    num_trials=1,  # 1 trial each
    max_samples=10000,
    target_reward=0.7,
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
print(f"Trials: {config.num_trials} each")
print(f"Max samples: {config.max_samples}")
print(f"Dataset size: {config.dataset_size}")
print("="*80)
print("⚠️ Expected time: 30-60 minutes")
print("="*80)

results = await run_experiment(config)
```

## If You Get Authentication Errors:

**Add this BEFORE Cell 5:**

```
import os
from huggingface_hub import login

# Get token from: https://huggingface.co/settings/tokens
HF_TOKEN = "hf_your_token_here"  # Replace with your token

login(token=HF_TOKEN)
os.environ["HF_TOKEN"] = HF_TOKEN
print("✓ Logged into Hugging Face")
```

## Most Likely:
**You DON'T need a token** - HH-RLHF is public. But the code now supports it if needed!

**Try running without a token first!** 🎯


