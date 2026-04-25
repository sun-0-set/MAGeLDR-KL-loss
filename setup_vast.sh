#!/usr/bin/env bash
# ============================================================================
# Vast.ai environment setup - run once per fresh workspace/container.
#
# Expected layout on the instance:
#   /workspace/MAGeLDR-KL-loss
#   /workspace/data/DREsS/DREsS_New_cleaned.tsv
#   /workspace/models/deberta-v3-large
#
# Usage:
#   bash setup_vast.sh
#   PROJECT_DIR=/workspace/MAGeLDR-KL-loss ENV_NAME=jager-cv bash setup_vast.sh
#   TORCH_CUDA=cu128 bash setup_vast.sh  # for machines capped at CUDA 12.8
# ============================================================================
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_ROOT/MAGeLDR-KL-loss}"
ENV_NAME="${ENV_NAME:-jager-cv}"
MINICONDA_DIR="${MINICONDA_DIR:-$WORKSPACE_ROOT/miniconda3}"
ENV_DIR="${ENV_DIR:-$MINICONDA_DIR/envs/$ENV_NAME}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCH_CUDA="${TORCH_CUDA:-cu130}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/$TORCH_CUDA}"

export CONDA_ENVS_PATH="$MINICONDA_DIR/envs"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/tmp/conda-pkgs}"
export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"

echo "=== [1/4] Making conda available ==="
if ! command -v conda >/dev/null 2>&1; then
  for d in "$MINICONDA_DIR" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    if [[ -f "$d/bin/conda" ]]; then
      eval "$("$d/bin/conda" shell.bash hook)"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
  rm /tmp/miniconda.sh
  eval "$("$MINICONDA_DIR/bin/conda" shell.bash hook)"
  conda init bash
  echo "[info] Miniconda installed to $MINICONDA_DIR"
else
  echo "[info] Conda already available: $(which conda)"
fi

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo "=== [2/4] Creating or refreshing env '$ENV_NAME' ==="
if [[ -d "$ENV_DIR/conda-meta" ]]; then
  conda install -y -p "$ENV_DIR" python=3.13
else
  conda create -y -p "$ENV_DIR" python=3.13
fi
conda activate "$ENV_DIR"

echo "=== [3/4] Refreshing pip ==="
pip install --upgrade pip

echo "=== [4/4] Installing PyTorch and project dependencies ==="
echo "[info] torch install: torch==$TORCH_VERSION from $TORCH_INDEX_URL"
pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX_URL"

tmp_requirements="$(mktemp)"
grep -vE '^[[:space:]]*(#|$|torch([[:space:]]|[<>=!~]).*)$' "$PROJECT_DIR/requirements.txt" > "$tmp_requirements"
pip install -r "$tmp_requirements"
rm -f "$tmp_requirements"

python - <<'PY'
import torch
print("[info] torch:", torch.__version__)
print("[info] torch.cuda:", torch.version.cuda)
print("[info] cuda available:", torch.cuda.is_available())
PY

echo
echo "============================================="
echo "  Setup complete"
echo "  Activate with: conda activate $ENV_DIR"
echo "============================================="
