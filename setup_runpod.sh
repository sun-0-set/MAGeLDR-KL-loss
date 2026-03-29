#!/bin/bash
# ============================================================================
# RunPod environment setup – run ONCE when you first start the pod.
# Usage:  bash setup_runpod.sh
# ============================================================================
set -euo pipefail

PROJECT_DIR="/workspace/MAGeLDR-KL-loss"
ENV_NAME="jager-cv"

CONDA_PREFIX="/workspace/miniconda3"

echo "=== [1/4] Installing Miniconda (if needed) ==="
# First try to activate conda if it's installed but not on PATH
if ! command -v conda &>/dev/null; then
  for d in "$CONDA_PREFIX" "$HOME/miniconda3" /opt/conda; do
    if [[ -f "$d/bin/conda" ]]; then
      eval "$("$d/bin/conda" shell.bash hook)"
      break
    fi
  done
fi

if ! command -v conda &>/dev/null; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_PREFIX"
  rm /tmp/miniconda.sh
  eval "$("$CONDA_PREFIX/bin/conda" shell.bash hook)"
  conda init bash
  echo "[info] Miniconda installed to $CONDA_PREFIX (persistent)."
else
  echo "[info] Conda already available: $(which conda)"
fi

# Accept Anaconda TOS (required since conda 25.x)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>/dev/null || true

echo "=== [2/4] Creating conda environment '$ENV_NAME' ==="
if conda env list | grep -qw "$ENV_NAME"; then
  echo "[info] Environment '$ENV_NAME' already exists — switching it to Python 3.13."
  conda install -y -n "$ENV_NAME" python=3.13
else
  conda create -y -n "$ENV_NAME" python=3.13
fi
conda activate "$ENV_NAME"

echo "=== [3/4] Refreshing packaging tools ==="
pip install --upgrade pip

echo "=== [4/4] Installing project dependencies ==="
# Python 3.13 wheels are published via the default PyPI torch package.
# GPU use still depends on the host driver being new enough for the bundled CUDA runtime.
pip install -r "$PROJECT_DIR/requirements.txt"

echo ""
echo "============================================="
echo "  Setup complete!  Activate with:"
echo "    conda activate $ENV_NAME"
echo "============================================="
