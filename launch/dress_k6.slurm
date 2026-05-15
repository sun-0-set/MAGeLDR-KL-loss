#!/bin/bash -l
#SBATCH -p a100
#SBATCH --job-name=dress_k6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --array=0-239
#SBATCH --output=../slurm_logs/%x_%A_%a.out
#SBATCH --error=../slurm_logs/%x_%A_%a.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"

K="${K:-6}"
SEED_START="${SEED_START:-42}"
SEED_END="${SEED_END:-81}"
LOSS="${LOSS:-jager}"
RUN_LABEL="${RUN_LABEL:-${LOSS}}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

EPOCHS="${EPOCHS:-18}"
SCHED_EPOCHS="${SCHED_EPOCHS:-$EPOCHS}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MAXLEN="${MAXLEN:-808}"
NUM_WORKERS="${NUM_WORKERS:-0}"

SPLITS_DIR="${SPLITS_DIR:-$REPO_ROOT/../splits/k6_promptcv}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/../data/DREsS/DREsS_New_cleaned.tsv}"
MODEL="${MODEL:-$REPO_ROOT/../models/deberta-v3-large}"
RESULTS_BASE="${RESULTS_BASE:-$REPO_ROOT/../results}"
TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-$REPO_ROOT/../cache/tokens}"

ENS_MODE="${ENS_MODE:-fixed_range}"
ENS_EPOCH_START="${ENS_EPOCH_START:-10}"
ENS_EPOCH_END="${ENS_EPOCH_END:-16}"
ENS_T="${ENS_T:-5}"
ENS_STRIDE="${ENS_STRIDE:-1}"
SAVE_EPOCH_VAL_PREDS="${SAVE_EPOCH_VAL_PREDS:-1}"
HF_OFFLINE="${HF_OFFLINE:-1}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"
CONDA_ENV="${CONDA_ENV:-jager-cv}"

SEED_COUNT=$((SEED_END - SEED_START + 1))
TASK_COUNT=$((SEED_COUNT * K))
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= TASK_COUNT )); then
  echo "TASK_ID=$TASK_ID outside 0-$((TASK_COUNT - 1))"
  exit 1
fi

SEED=$((SEED_START + TASK_ID / K))
FOLD=$((TASK_ID % K))
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
RUN_ID="${RUN_ID:-hpc_dress_k${K}_${RUN_LABEL}_s${SEED_START}_${SEED_END}_${ARRAY_JOB_ID}}"
TAG="${TAG:-${RUN_LABEL}-E${ENS_EPOCH_START}-${ENS_EPOCH_END}}"
RESULTS_ROOT="${RESULTS_ROOT:-$RESULTS_BASE/$(basename "$SPLITS_DIR")/$RUN_ID}"
SAVE="$RESULTS_ROOT/seed${SEED}/${TAG}/fold${FOLD}"
LOG="$SAVE/train.log"
MASTER_PORT="${MASTER_PORT:-$((26000 + TASK_ID))}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'TASK_ID=%s\nSEED=%s\nFOLD=%s\nLOSS=%s\nRUN_ID=%s\nTAG=%s\nSAVE=%s\n' \
    "$TASK_ID" "$SEED" "$FOLD" "$LOSS" "$RUN_ID" "$TAG" "$SAVE"
  exit 0
fi

if [[ -f "$HOME/.bashrc" ]]; then
  set +eu
  source "$HOME/.bashrc" || true
  set -eu
fi

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if [[ "$HF_OFFLINE" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
fi

mkdir -p "$TOKEN_CACHE_DIR" "$SAVE"

common=(
  --data_path "$DATA_PATH"
  --split_file "$SPLITS_DIR/fold${FOLD}.json"
  --model_name "$MODEL"
  --save_dir "$SAVE"
  --seed "$SEED"
  --loss "$LOSS"
  --max_length "$MAXLEN"
  --batch_size "$BATCH"
  --grad_accum "$ACCUM"
  --epochs "$EPOCHS"
  --sched_epochs "$SCHED_EPOCHS"
  --ens_mode "$ENS_MODE"
  --ens_epoch_start "$ENS_EPOCH_START"
  --ens_epoch_end "$ENS_EPOCH_END"
  --ens_t "$ENS_T"
  --ens_stride "$ENS_STRIDE"
  --num_workers "$NUM_WORKERS"
  --prefetch_factor 4
  --token_cache_dir "$TOKEN_CACHE_DIR"
  --wandb_run_name "${RUN_ID}-seed${SEED}-fold${FOLD}-${TAG}"
)

if [[ "$HF_OFFLINE" == "1" ]]; then
  common+=(--hf_offline)
fi
if [[ "$SAVE_EPOCH_VAL_PREDS" == "1" ]]; then
  common+=(--save_epoch_val_preds)
fi
if [[ "$WANDB_ENABLED" == "1" ]]; then
  common+=(--wandb)
fi

extra=()
if [[ -n "$EXTRA_TRAIN_ARGS" ]]; then
  read -r -a extra <<<"$EXTRA_TRAIN_ARGS"
fi

echo "[run] task=$TASK_ID seed=$SEED fold=$FOLD loss=$LOSS tag=$TAG"
echo "[run] save=$SAVE"

torchrun \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  "$REPO_ROOT/train.py" "${common[@]}" "${extra[@]}" \
  >"$LOG" 2>&1
