#!/usr/bin/env bash
# Script to set up a CPU-only PyTorch environment named 'ptmath'
# Usage: bash setup_ptmath_cpu.sh

set -e

python -m venv ptmath
source ptmath/bin/activate

# Upgrade pip and install CPU-only PyTorch packages
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

echo "Environment 'ptmath' created with CPU-only PyTorch."
