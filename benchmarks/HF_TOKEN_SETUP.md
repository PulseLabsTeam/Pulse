# 🔑 Hugging Face Token Setup (If Needed)

## Check First:
The HH-RLHF dataset (`Anthropic/hh-rlhf`) is **public** and usually doesn't require a token. But if you get authentication errors, here's how to set it up:

## Option 1: If No Token Needed (Most Likely)
Just run the experiment - it should work without a token.

## Option 2: If Token Required

### Step 1: Get Hugging Face Token
1. Go to: https://huggingface.co/settings/tokens
2. Sign in (or create free account)
3. Click "New token"
4. Name it: "Colab RLHF"
5. Select "Read" permissions
6. Copy the token

### Step 2: Add to Colab Cell (Before Cell 5)
**Create a new cell before Cell 5:**

```
import os
from huggingface_hub import login

# Paste your token here (or use environment variable)
HF_TOKEN = "hf_your_token_here"  # Replace with your actual token

# Login to Hugging Face
login(token=HF_TOKEN)
print("✓ Logged into Hugging Face")
```

**OR use environment variable (more secure):**

```
import os
from huggingface_hub import login

# Set token from environment (set in Colab secrets or manually)
HF_TOKEN = os.environ.get("HF_TOKEN", "hf_your_token_here")
login(token=HF_TOKEN)
print("✓ Logged into Hugging Face")
```

### Step 3: Update Cell 5 to Use Token
Add this to the beginning of Cell 5:

```
import os
from huggingface_hub import login

# Login if token is set
if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])
    print("✓ Using Hugging Face token")
```

## Most Likely Scenario:
**You DON'T need a token** - HH-RLHF is public. But if you get authentication errors, use the steps above.

## Quick Test:
Run this in a cell to check if you need authentication:

```
from datasets import load_dataset
try:
    dataset = load_dataset("Anthropic/hh-rlhf", split="train[:10]")
    print("✓ No token needed - dataset is public!")
except Exception as e:
    print(f"⚠️ Error: {e}")
    print("You may need a Hugging Face token")
```

**Try running without a token first - it should work!** 🎯


