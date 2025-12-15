# 🆕 New File: rlhf_full_experiment.zip

## New File Created!
**`rlhf_full_experiment.zip`** - Fresh file with fixed code

## Steps to Use:

### Step 1: Upload NEW File
**Create Cell 1** - Copy this code:

```
from google.colab import files
import zipfile
import os
import shutil

# Remove ALL old files
if os.path.exists('benchmarks'):
    shutil.rmtree('benchmarks')
if os.path.exists('pulseos'):
    shutil.rmtree('pulseos')
if os.path.exists('colab_files.zip'):
    os.remove('colab_files.zip')

print("📁 Upload NEW 'rlhf_full_experiment.zip':")
uploaded = files.upload()

# Extract
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"✓ Extracted {filename}")
        print(f"✓ Files extracted successfully")
```

**Run it** - Upload `rlhf_full_experiment.zip` from Desktop

---

### Step 2: Install Dependencies
**Cell 2:**
```
!pip install -q torch transformers trl datasets scipy matplotlib numpy
print("✓ Dependencies installed")
```

---

### Step 3: Check GPU
**Cell 3:**
```
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

### Step 4: Add PulseOS
**Cell 4:**
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

### Step 5: Run FULL Experiment
**Cell 5:**
```
import asyncio
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, 'benchmarks')

from real_llm_rlhf_experiment import run_experiment, ExperimentConfig

# FULL EXPERIMENT - EXPLICITLY SET ALL PARAMETERS
config = ExperimentConfig(
    model_name="gpt2",
    dataset_name="Anthropic/hh-rlhf",
    num_trials=10,
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
    dataset_size=None  # FULL dataset
)

print("🚀 Starting FULL RLHF Experiment...")
print(f"Trials: {config.num_trials} each")
print(f"Max samples: {config.max_samples}")
print(f"Target reward: {config.target_reward}")
print("="*80)
print("⚠️ This will take 2-4 hours!")
print("="*80)

results = await run_experiment(config)
```

---

**The new file is on your Desktop: `rlhf_full_experiment.zip`** 🎯


