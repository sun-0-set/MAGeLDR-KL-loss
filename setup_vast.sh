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
# ============================================================================
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_ROOT/MAGeLDR-KL-loss}"
ENV_NAME="${ENV_NAME:-jager-cv}"
CONDA_PREFIX="${CONDA_PREFIX:-$WORKSPACE_ROOT/miniconda3}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

echo "=== [1/4] Making conda available ==="
if ! command -v conda >/dev/null 2>&1; then
  for d in "$CONDA_PREFIX" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    if [[ -f "$d/bin/conda" ]]; then
      eval "$("$d/bin/conda" shell.bash hook)"
      break
    fi
  done
fi

if ! command -v conda >/dev/null 2>&1; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_PREFIX"
  rm /tmp/miniconda.sh
  eval "$("$CONDA_PREFIX/bin/conda" shell.bash hook)"
  conda init bash
  echo "[info] Miniconda installed to $CONDA_PREFIX"
else
  echo "[info] Conda already available: $(which conda)"
fi

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

echo "=== [2/4] Creating or refreshing env '$ENV_NAME' ==="
if conda env list | grep -qw "$ENV_NAME"; then
  conda install -y -n "$ENV_NAME" python=3.13
else
  conda create -y -n "$ENV_NAME" python=3.13
fi
conda activate "$ENV_NAME"

echo "=== [3/4] Refreshing pip ==="
pip install --upgrade pip

echo "=== [4/4] Installing PyTorch and project dependencies ==="
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
echo "  Activate with: conda activate $ENV_NAME"
echo "============================================="
