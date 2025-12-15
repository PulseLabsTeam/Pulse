# 🔄 Steps After Uploading Updated Zip

## Step 1: Extract New Zip File
**Re-run Cell 1** (or create new cell):

```
from google.colab import files
import zipfile
import os
import shutil

# Remove old files
if os.path.exists('benchmarks'):
    shutil.rmtree('benchmarks')
if os.path.exists('pulseos'):
    shutil.rmtree('pulseos')

print("📁 Upload UPDATED 'rlhf_full_experiment.zip':")
uploaded = files.upload()

# Extract
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"✓ Extracted {filename}")
```

**Run it** - Upload `rlhf_full_experiment.zip`

---

## Step 2: Re-add PulseOS to Path
**Re-run Cell 4:**

```
import sys
import os

sys.path.insert(0, '.')
sys.path.insert(0, 'benchmarks')

if os.path.exists('pulseos'):
    sys.path.insert(0, 'pulseos')
    print("✓ PulseOS ready")
```

---

## Step 3: Run Cell 5 (Same Code - Now With Progress Prints!)
**Use the SAME Cell 5 code:**

```
import asyncio
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, 'benchmarks')

# IMPORTANT: Reload module to get updated code
import importlib
if 'real_llm_rlhf_experiment' in sys.modules:
    importlib.reload(sys.modules['real_llm_rlhf_experiment'])

from real_llm_rlhf_experiment import run_experiment, ExperimentConfig

# QUICK VALIDATION TEST
config = ExperimentConfig(
    model_name="gpt2",
    dataset_name="Anthropic/hh-rlhf",
    num_trials=1,
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
    dataset_size=20000
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

**Key addition:** `importlib.reload()` to ensure you get the updated code with progress prints!

---

## What You'll See Now:
- ✅ Progress updates every 10 samples: "Sample 10/10000, Recent reward: X.XXX"
- ✅ Convergence messages when reached
- ✅ Completion times

**Run Steps 1, 2, then 3!** 🎯


