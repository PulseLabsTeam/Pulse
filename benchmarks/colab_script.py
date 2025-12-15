"""
PulseOS RLHF Experiment - Google Colab Version
Copy this entire file into a Colab notebook cell and run it!
"""

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
# !pip install -q torch transformers trl datasets scipy matplotlib numpy

# ============================================================================
# CELL 2: Check GPU
# ============================================================================
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ Enable GPU: Runtime → Change runtime type → GPU")

# ============================================================================
# CELL 3: Upload Files
# ============================================================================
from google.colab import files
import os
import zipfile

print("📁 Upload 'colab_files.zip' file:")
uploaded = files.upload()

# Extract zip
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"✓ Extracted {filename}")
        os.remove(filename)  # Clean up

# ============================================================================
# CELL 4: Install PulseOS
# ============================================================================
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'benchmarks')

# Install pulseos
if os.path.exists('pulseos'):
    !pip install -e . -q
    print("✓ PulseOS installed")
else:
    print("⚠️ pulseos folder not found - check upload")

# ============================================================================
# CELL 5: Run Experiment
# ============================================================================
import asyncio
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from real_llm_rlhf_experiment import run_experiment, ExperimentConfig

config = ExperimentConfig(
    model_name="gpt2",
    num_trials=1,
    max_samples=20,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dataset_size=200,
    reward_model_samples=50,
    reward_model_epochs=1
)

print("🚀 Starting RLHF Experiment...")
print(f"Device: {config.device}")
results = await run_experiment(config)

# ============================================================================
# CELL 6: Display Results
# ============================================================================
print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Baseline PPO: {results.baseline_mean_samples:.1f} samples")
print(f"PulseOS: {results.pulseos_mean_samples:.1f} samples")
print(f"🎯 Improvement: {results.improvement_percent:.1f}%")
print(f"p-value: {results.p_value:.4f}")
print(f"Significant: {'Yes ✅' if results.significant else 'No ❌'}")

# Download results
from google.colab import files
import json

results_file = 'rlhf_results.json'
with open(results_file, 'w') as f:
    json.dump({
        'improvement_percent': results.improvement_percent,
        'baseline_mean': results.baseline_mean_samples,
        'pulseos_mean': results.pulseos_mean_samples,
        'p_value': results.p_value
    }, f, indent=2)

files.download(results_file)
print("\n✅ Results downloaded!")


