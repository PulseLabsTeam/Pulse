# 🚀 Complete Guide: Full RLHF Experiment in Google Colab

## Prerequisites
- ✅ Google Colab open (https://colab.research.google.com)
- ✅ GPU enabled (Runtime → Change runtime type → GPU → T4)
- ✅ `colab_files.zip` ready to upload

---

## Step 1: Upload and Extract Files

**Create Cell 1** - Copy this code:

```
from google.colab import files
import zipfile
import os
import shutil

# Remove old files if they exist
if os.path.exists('benchmarks'):
    shutil.rmtree('benchmarks')
if os.path.exists('pulseos'):
    shutil.rmtree('pulseos')

print("📁 Upload 'colab_files.zip':")
uploaded = files.upload()

# Extract
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"✓ Extracted {filename}")
```

**Run it** - Click "Choose Files" and select `colab_files.zip` from your Desktop

---

## Step 2: Install Dependencies

**Create Cell 2** - Copy this code:

```
!pip install -q torch transformers trl datasets scipy matplotlib numpy
print("✓ Dependencies installed")
```

**Run it** - Takes ~1-2 minutes

---

## Step 3: Check GPU

**Create Cell 3** - Copy this code:

```
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU not detected! Enable GPU in Runtime settings")
```

**Run it** - Should show "CUDA available: True" and "GPU: Tesla T4"

---

## Step 4: Add PulseOS to Path

**Create Cell 4** - Copy this code:

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

**Run it** - Should complete instantly

---

## Step 5: Run FULL Experiment

**Create Cell 5** - Copy this code:

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

**Run it** - ⏱️ **This will take 2-4 hours!**
- Don't close the browser tab
- You'll see progress updates
- Wait for it to complete

---

## Step 6: View and Download Results

**After Cell 5 completes, create Cell 6** - Copy this code:

```
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
print(f"\n🎯 Improvement: {results.improvement_percent:.1f}%")
print(f"Baseline PPO: {results.baseline_mean_samples:.1f} samples")
print(f"PulseOS: {results.pulseos_mean_samples:.1f} samples")
print(f"\nStatistical Analysis:")
print(f"  p-value: {results.p_value:.4f}")
print(f"  Significant: {'Yes ✅' if results.significant else 'No ❌'}")
print(f"  Cohen's d: {results.cohens_d:.3f}")

# Valuation assessment
print("\n" + "="*80)
print("VALUATION ASSESSMENT")
print("="*80)
if results.improvement_percent >= 40:
    print(f"✅ EXCELLENT: {results.improvement_percent:.1f}% reduction")
    print("   Valuation: $50M-$150M")
elif results.improvement_percent >= 20:
    print(f"⚠️  GOOD: {results.improvement_percent:.1f}% reduction")
    print("   Valuation: $30M-$70M")
else:
    print(f"⚠️  MODEST: {results.improvement_percent:.1f}% reduction")
    print("   Valuation: $15M-$40M")
print("="*80)

# Download results
from google.colab import files
import json

with open('results.json', 'w') as f:
    json.dump({
        'improvement': results.improvement_percent,
        'baseline_mean': results.baseline_mean_samples,
        'pulseos_mean': results.pulseos_mean_samples,
        'p_value': results.p_value,
        'significant': results.significant,
        'cohens_d': results.cohens_d
    }, f, indent=2)

files.download('results.json')
print("\n✅ Results downloaded!")
```

**Run it** - Downloads results JSON file

---

## Quick Checklist

- [ ] Cell 1: Upload & extract zip
- [ ] Cell 2: Install dependencies
- [ ] Cell 3: Check GPU
- [ ] Cell 4: Add PulseOS to path
- [ ] Cell 5: Run full experiment (2-4 hours)
- [ ] Cell 6: View results

---

## Expected Timeline

- **Setup (Cells 1-4):** ~5 minutes
- **Full Experiment (Cell 5):** 2-4 hours
- **Results (Cell 6):** ~1 minute
- **Total:** ~2-4 hours

---

## Troubleshooting

**"GPU not available":**
→ Runtime → Change runtime type → GPU → Save

**"pulseos folder not found":**
→ Re-run Cell 1 to extract files

**"Module not found":**
→ Re-run Cell 2 to install dependencies

**Experiment taking too long:**
→ Normal! 2-4 hours is expected for full experiment

---

## What You'll Get

- **10 trials** each (baseline + PulseOS)
- **Up to 10,000 samples** per trial
- **Full HH-RLHF dataset** (~160k samples)
- **Statistical significance** (p-value, Cohen's d)
- **Valuation assessment** based on improvement

**Ready to start!** 🎯


