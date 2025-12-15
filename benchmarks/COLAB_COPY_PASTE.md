# 🚀 Google Colab Setup - Copy & Paste Guide

## ⚠️ IMPORTANT: Only copy the CODE, NOT the markdown formatting!

When you see code blocks like:
```
```python
code here
```
```

**ONLY copy the code inside**, NOT the ```python or ``` parts!

---

## Quick Start (5 minutes):

### Step 1: Go to Colab
Visit: **https://colab.research.google.com**
- Sign in with Google (free account)

### Step 2: Create New Notebook
- Click "New notebook"
- Name it: "PulseOS RLHF Experiment"

### Step 3: Enable GPU (IMPORTANT!)
- Click **"Runtime"** → **"Change runtime type"**
- Hardware accelerator: Select **"GPU"**
- Click **"Save"**

### Step 4: Run These Cells (Copy ONLY the code, NOT the markdown):

---

#### **CELL 1: Install Dependencies**
**Copy this code (without the ```python markers):**
```
!pip install -q torch transformers trl datasets scipy matplotlib numpy
```

---

#### **CELL 2: Check GPU**
**Copy this code:**
```
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

#### **CELL 3: Upload Files**
**Copy this code:**
```
from google.colab import files
import zipfile
import os

print("📁 Upload 'colab_files.zip':")
uploaded = files.upload()

# Extract
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"✓ Extracted {filename}")
```

---

#### **CELL 4: Install PulseOS**
**Copy this code:**
```
import sys
import os

# Add paths so Python can find pulseos
sys.path.insert(0, '.')
sys.path.insert(0, 'benchmarks')

# Check if pulseos exists and add to path
if os.path.exists('pulseos'):
    sys.path.insert(0, 'pulseos')
    print("✓ PulseOS added to Python path")
    print("✓ Ready to run experiment")
else:
    print("⚠️ pulseos folder not found - check extraction")
```

---

#### **CELL 5: Run Experiment**
**Copy this code:**
```
import asyncio
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, 'benchmarks')

from real_llm_rlhf_experiment import run_experiment, ExperimentConfig

config = ExperimentConfig(
    device="cuda",  # Use GPU!
    num_trials=1,
    max_samples=20,
    dataset_size=200
)

print("🚀 Starting experiment...")
results = await run_experiment(config)
```

---

#### **CELL 6: View & Download Results**
**Copy this code:**
```
print(f"\n🎯 Improvement: {results.improvement_percent:.1f}%")
print(f"Baseline: {results.baseline_mean_samples:.1f} samples")
print(f"PulseOS: {results.pulseos_mean_samples:.1f} samples")

from google.colab import files
import json

with open('results.json', 'w') as f:
    json.dump({'improvement': results.improvement_percent}, f)

files.download('results.json')
```

---

## Files to Upload:

1. **`colab_files.zip`** - Contains everything you need
   - Location: `benchmarks/colab_files.zip` (or your Desktop)
   - Just upload this one file!

## Common Mistakes:

❌ **WRONG:** Copying ` ```python ` or ` ``` ` markers
✅ **RIGHT:** Only copy the actual code

❌ **WRONG:** Including markdown formatting
✅ **RIGHT:** Just the Python code

## That's It!

- **Cost:** $0 (completely free)
- **Time:** ~5-10 minutes
- **GPU:** Free T4 (much faster than Mac!)

## Troubleshooting:

**"GPU not available":**
→ Make sure you selected GPU in Runtime settings

**"File not found":**
→ Make sure you uploaded `colab_files.zip` and extracted it

**"Module not found":**
→ Run Cell 1 (pip install) first

**"SyntaxError: invalid syntax" on line with ```python:**
→ You copied the markdown markers! Only copy the code inside.

Ready to go! 🚀

