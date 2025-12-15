# 🚀 Steps to Run Full Experiment

## Step 1: Extract New Zip File
**Run Cell 3 again** (or create new cell):
```
from google.colab import files
import zipfile
import os

# Remove old extracted files
import shutil
if os.path.exists('benchmarks'):
    shutil.rmtree('benchmarks')
if os.path.exists('pulseos'):
    shutil.rmtree('pulseos')

print("📁 Upload NEW 'colab_files.zip':")
uploaded = files.upload()

# Extract new zip
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"✓ Extracted {filename}")
```

## Step 2: Re-add PulseOS to Path
**Run Cell 4 again**:
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
    print("⚠️ pulseos folder not found")
```

## Step 3: Run FULL Experiment
**Create new Cell 5** (or replace old one):
```
import asyncio
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, 'benchmarks')

from real_llm_rlhf_experiment import run_experiment, ExperimentConfig

# FULL EXPERIMENT - uses updated defaults
config = ExperimentConfig()  # All defaults are now full experiment settings!

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

## Step 4: Wait for Completion
- ⏱️ **Expected time:** 2-4 hours
- 📊 **Progress:** You'll see training progress in the output
- ⚠️ **Don't close** the browser tab!

## Step 5: View Results
**After Cell 5 completes, run Cell 6**:
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

## Summary:
1. ✅ Extract new zip (Cell 3)
2. ✅ Re-add PulseOS (Cell 4)
3. ✅ Run full experiment (Cell 5) - **2-4 hours**
4. ⏳ Wait for completion
5. ✅ View results (Cell 6)

**Ready to go!** 🎯


