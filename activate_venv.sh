#!/bin/bash
# Activate virtual environment for PulseOS experiments
# Usage: source activate_venv.sh

cd "$(dirname "$0")/.."
source venv/bin/activate
export TOKENIZERS_PARALLELISM=false
echo "Virtual environment activated!"
echo "Python: $(which python)"
echo "Python version: $(python --version)"


