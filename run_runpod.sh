#!/bin/bash
# ============================================================================
# RunPod launch script – equivalent of dress_cv_k6.slurm for a RunPod pod
# with 4 GPUs.
#
# Usage:
#   bash run_runpod.sh          # runs fold 0 (default)
#   bash run_runpod.sh 3        # runs fold 3
#   bash run_runpod.sh all      # runs all 6 folds sequentially
# ============================================================================
set -eo pipefail

# ---- Environment (conda) ---------------------------------------------------

# Source .bashrc without nounset (it may reference unset vars like PS1)
if [[ -f "$HOME/.bashrc" ]]; then
  set +u
  source "$HOME/.bashrc"
  set -u
fi
# Ensure conda is available (find it explicitly if not on PATH)
if ! command -v conda &>/dev/null; then
  for d in /workspace/miniconda3 "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
    if [[ -f "$d/bin/conda" ]]; then
      eval "$("$d/bin/conda" shell.bash hook)"
      break
    fi
  done
fi

if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found. Run setup_runpod.sh first."
  exit 1
fi

conda activate jager-cv

echo "[info] which python: $(which python)"
echo "[info] which torchrun: $(which torchrun)"
python -c "import torch; print('[info] torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ---- Paths ------------------------------------------------------------------

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"
echo "[info] PWD=$PWD"

# ---- CV + data config -------------------------------------------------------

K=6
NUM_GPUS="${NUM_GPUS:-4}"

SPLITS_DIR="$SCRIPT_DIR/../splits/k6_promptcv"
DATA_PATH="$SCRIPT_DIR/../data/DREsS/DREsS_New_cleaned.tsv"

MODEL="$SCRIPT_DIR/../models/deberta-v3-large"

EPOCHS="${EPOCHS:-16}"
SCHED_EPOCHS="${SCHED_EPOCHS:-40}"
ENS_EPOCH_START="${ENS_EPOCH_START:-8}"
ENS_EPOCH_END="${ENS_EPOCH_END:-14}"
BATCH=8
ACCUM=2
MAXLEN=808

COMMON=(--data_path "$DATA_PATH" --model_name "$MODEL" --max_length "$MAXLEN"
        --epochs "$EPOCHS" --sched_epochs "$SCHED_EPOCHS"
        --batch_size "$BATCH" --grad_accum "$ACCUM"
        --num_workers 4 --prefetch_factor 4
        --ens_mode fixed_range
        --ens_epoch_start "$ENS_EPOCH_START" --ens_epoch_end "$ENS_EPOCH_END"
        --ens_stride 1)

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_SHOW_CPP_STACKTRACES=1

JOBS=(
  "jager --joint --mixture --no-conf_gating --no-reassignment --lambda0 3 --lambda_min 0.5 --C 5e-2"
  "jager --joint --mixture --no-conf_gating --no-reassignment --lambda0 3 --lambda_min 0.5 --C 1e-1"
  "jager --joint --mixture --no-conf_gating --no-reassignment --lambda0 3 --lambda_min 0.5 --C 2e-1"
  "jager --joint --mixture --no-conf_gating --no-reassignment --lambda0 3 --lambda_min 1 --C 5e-2"
  "jager --joint --mixture --no-conf_gating --no-reassignment --lambda0 3 --lambda_min 1 --C 1e-1"
  "jager --joint --mixture --no-conf_gating --no-reassignment --lambda0 3 --lambda_min 1 --C 2e-1"
  "jager --joint --mixture --no-conf_gating --no-reassignment --lambda0 5 --lambda_min 0.5 --C 5e-2"
  "jager --joint --mixture --no-conf_gating --no-reassignment --lambda0 5 --lambda_min 0.5 --C 1e-1"
  "jager --joint --mixture --no-conf_gating --no-reassignment --lambda0 5 --lambda_min 0.5 --C 2e-1"
  "jager --joint --mixture --no-conf_gating --no-reassignment --lambda0 5 --lambda_min 1 --C 5e-2"
  "jager --joint --mixture --no-conf_gating --no-reassignment --lambda0 5 --lambda_min 1 --C 1e-1"
  "jager --joint --mixture --no-conf_gating --no-reassignment --lambda0 5 --lambda_min 1 --C 2e-1"
  "jager --joint --mixture --no-conf_gating --no-reassignment --lambda0 10 --lambda_min 0.5 --C 5e-2"
  "jager --joint --mixture --no-conf_gating --no-reassignment --lambda0 10 --lambda_min 0.5 --C 1e-1"
  "jager --joint --mixture --no-conf_gating --no-reassignment --lambda0 10 --lambda_min 0.5 --C 2e-1"
  "jager --joint --mixture --no-conf_gating --no-reassignment --lambda0 10 --lambda_min 1 --C 5e-2"
  "jager --joint --mixture --no-conf_gating --no-reassignment --lambda0 10 --lambda_min 1 --C 1e-1"
  "jager --joint --mixture --no-conf_gating --no-reassignment --lambda0 10 --lambda_min 1 --C 2e-1"
)

# ---- Fold selection ---------------------------------------------------------

FOLD_ARG="${1:-0}"   # default: fold 0

if [[ "$FOLD_ARG" == "all" ]]; then
  FOLDS=($(seq 0 $((K - 1))))
else
  FOLDS=("$FOLD_ARG")
fi

# ---- Basic sanity checks ----------------------------------------------------

echo "[check] SPLITS_DIR=$SPLITS_DIR"
ls -l "$SPLITS_DIR" || { echo "ERROR: cannot ls $SPLITS_DIR"; exit 1; }

count=$(ls -1 "$SPLITS_DIR"/fold*.json 2>/dev/null | wc -l | tr -d ' ')
[[ "$count" == "$K" ]] || { echo "ERROR: expected K=$K splits, found $count"; exit 1; }

[[ -f "$DATA_PATH" ]] || { echo "ERROR: data file missing: $DATA_PATH"; exit 1; }
[[ -d "$MODEL" ]] || echo "[note] MODEL dir not found; HF might download."

splits_name="$(basename "$SPLITS_DIR")"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RESULTS_ROOT="$SCRIPT_DIR/../results/${splits_name}/${RUN_ID}"
mkdir -p "$RESULTS_ROOT"
echo "[info] RESULTS_ROOT=$RESULTS_ROOT"
echo "[info] EPOCHS=$EPOCHS SCHED_EPOCHS=$SCHED_EPOCHS OOF=E${ENS_EPOCH_START}-${ENS_EPOCH_END}"

# ---- Run folds --------------------------------------------------------------

for FOLD in "${FOLDS[@]}"; do

  if (( FOLD < 0 || FOLD >= K )); then
    echo "ERROR: FOLD=$FOLD is outside [0, $((K-1))]"
    exit 1
  fi

  MASTER_PORT=$((29500 + FOLD))

  echo "[info] running fold ${FOLD} on host $(hostname)"

  for job in "${JOBS[@]}"; do
    read -r LOSS ARGS <<<"$job"

    if [[ -n "$ARGS" ]]; then
      TAG="${LOSS}-$(echo "$ARGS" | tr ' ' '-' | tr -s '-')"
    else
      TAG="${LOSS}"
    fi

    SAVE="${RESULTS_ROOT}/${TAG}/fold${FOLD}"
    mkdir -p "$SAVE"
    LOG="$SAVE/train.log"

    echo ">> [fold ${FOLD}] ${TAG} → $LOG"

    set +e
    ARGS_ARR=()
    if [[ -n "$ARGS" ]]; then
      read -r -a ARGS_ARR <<<"$ARGS"
    fi

    torchrun \
      --nproc_per_node="$NUM_GPUS" \
      --master_port="${MASTER_PORT}" \
      "$SCRIPT_DIR/train.py" "${COMMON[@]}" \
        --loss "$LOSS" "${ARGS_ARR[@]}" \
        --split_file "${SPLITS_DIR}/fold${FOLD}.json" \
        --save_dir "$SAVE" \
        2>&1 | tee "$LOG"

    code=$?
    echo "$code" > "$SAVE/.exit_code"

    if [[ "$code" != 0 ]]; then
      echo "!! [fold ${FOLD}] ${TAG} FAILED (code $code). See $LOG"
    fi
    set -e
  done
done

echo "[done] Finished. Results in: $RESULTS_ROOT"
