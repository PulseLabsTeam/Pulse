# Cell 6: View & Download Results

**Copy this code into a new cell (no markdown markers):**

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

## What This Does:
- Displays the improvement percentage
- Shows baseline vs PulseOS sample counts
- Saves results to JSON file
- Downloads the results file to your computer

## Expected Output:
```
🎯 Improvement: XX.X%
Baseline: XX.X samples
PulseOS: XX.X samples
```

Then it will download `results.json` automatically!


