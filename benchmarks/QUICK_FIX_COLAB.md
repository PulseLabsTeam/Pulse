# 🔧 Quick Fix for Colab - Run This Cell

**Create a new cell BEFORE Cell 4 and run this:**

```
# Fix the import error
import os

# Read the file
with open('text_generation_arena.py', 'r') as f:
    content = f.read()

# Fix the import - replace the old import with correct one
old_import = """from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AdamW
)"""

new_import = """from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from torch.optim import AdamW"""

if old_import in content:
    content = content.replace(old_import, new_import)
    with open('text_generation_arena.py', 'w') as f:
        f.write(content)
    print("✓ Fixed import - AdamW now from torch.optim")
else:
    print("⚠️ Import already fixed or file structure different")
    # Try alternative fix
    if "from transformers import" in content and "AdamW" in content:
        lines = content.split('\n')
        new_lines = []
        skip_next = False
        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue
            if 'AdamW' in line and 'transformers' in lines[max(0, i-3):i+1]:
                # Skip AdamW line, add torch.optim import after transformers block
                if i+1 < len(lines) and lines[i+1].strip() == ')':
                    new_lines.append(')')
                    new_lines.append('from torch.optim import AdamW')
                    skip_next = True
                    continue
            new_lines.append(line)
        content = '\n'.join(new_lines)
        with open('text_generation_arena.py', 'w') as f:
            f.write(content)
        print("✓ Fixed import using alternative method")

print("✓ Ready to import!")
```

**Then run Cell 4 again!**


