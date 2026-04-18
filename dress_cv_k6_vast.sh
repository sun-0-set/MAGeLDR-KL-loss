#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./dress_cv_k6_vast.sh <fold|all> [run_id]

Environment overrides:
  NPROC_PER_NODE   Number of GPUs for torchrun (default: 4)
  RESULTS_BASE     Base results directory (default: ../results)
  RESULTS_ROOT     Full results root; overrides RESULTS_BASE/run_id layout
  SPLITS_DIR       CV splits directory (default: ../splits/k6_promptcv)
  DATA_PATH        Dataset path (default: ../data/DREsS/DREsS_New_cleaned.tsv)
  MODEL            Local model directory (default: ../models/deberta-v3-large)
  TOKEN_CACHE_DIR  Reusable token cache dir (default: alongside data file)
  EPOCHS           Actual training epochs (default: 16)
  SCHED_EPOCHS     Scheduler horizon in epochs (default: 40)
  ENS_EPOCH_START  First epoch in fixed OOF ensemble (default: 8)
  ENS_EPOCH_END    Last epoch in fixed OOF ensemble (default: 14)
  BATCH            Per-device batch size (default: 16 for 4x H100 SXM)
  ACCUM            Gradient accumulation steps (default: 1)
  MAXLEN           Max sequence length (default: 808)
  NUM_WORKERS      DataLoader workers (default: 4)
  PREFETCH_FACTOR  DataLoader prefetch factor (default: 4)
  HF_OFFLINE       1 to force local HF assets only (default: 1)
  SAVE_MODEL       1 to save best.pt checkpoints (default: 0)
  GRAD_CKPT        1 to enable --grad_ckpt if memory is tight (default: 0)
  JOB_SET          full (default), ce_only, or ce_sweep
  RESUME           1 to skip jobs with .exit_code == 0 (default: 1)
  WANDB_ENABLED    1 to enable W&B logging (default: 0)
  WANDB_PROJECT    W&B project name (default: jager)
  WANDB_ENTITY     W&B entity/team (default: unset)
  WANDB_MODE       online/offline/disabled (default: online)
  WANDB_GROUP      W&B group; default is RUN_ID
  WANDB_JOB_TYPE   W&B job type (default: train)
  WANDB_TAGS       Extra comma-separated W&B tags
  EXTRA_ARGS       Extra args appended to train.py for all jobs

Examples:
  ./dress_cv_k6_vast.sh 0 vast_20260410
  JOB_SET=ce_only BATCH=8 ACCUM=2 ./dress_cv_k6_vast.sh all vast_ce_20260418
  NPROC_PER_NODE=4 SAVE_MODEL=1 ./dress_cv_k6_vast.sh 3
EOF
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

FOLD="${FOLD:-${1:-}}"
RUN_ID="${RUN_ID:-${2:-}}"

if [[ -z "${FOLD}" ]]; then
  echo "ERROR: fold is required. Run ./dress_cv_k6_vast.sh <fold|all> [run_id]"
  exit 1
fi

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="vast_$(date +%Y%m%d_%H%M%S)"
fi

K="${K:-6}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
SPLITS_DIR="${SPLITS_DIR:-$SCRIPT_DIR/../splits/k6_promptcv}"
DATA_PATH="${DATA_PATH:-$SCRIPT_DIR/../data/DREsS/DREsS_New_cleaned.tsv}"
MODEL="${MODEL:-$SCRIPT_DIR/../models/deberta-v3-large}"
TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-}"
RESULTS_BASE="${RESULTS_BASE:-$SCRIPT_DIR/../results}"
RESULTS_ROOT="${RESULTS_ROOT:-$RESULTS_BASE/$(basename "$SPLITS_DIR")/$RUN_ID}"

EPOCHS="${EPOCHS:-16}"
SCHED_EPOCHS="${SCHED_EPOCHS:-40}"
ENS_EPOCH_START="${ENS_EPOCH_START:-8}"
ENS_EPOCH_END="${ENS_EPOCH_END:-14}"
BATCH="${BATCH:-16}"
ACCUM="${ACCUM:-1}"
MAXLEN="${MAXLEN:-808}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
JAGER_DEBUG="${JAGER_DEBUG:-1}"
HF_OFFLINE="${HF_OFFLINE:-1}"
SAVE_MODEL="${SAVE_MODEL:-0}"
GRAD_CKPT="${GRAD_CKPT:-0}"
JOB_SET="${JOB_SET:-full}"
RESUME="${RESUME:-1}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-jager}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-$RUN_ID}"
WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-train}"
WANDB_TAGS="${WANDB_TAGS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29500}"

COMMON=(
  --data_path "$DATA_PATH"
  --model_name "$MODEL"
  --max_length "$MAXLEN"
  --epochs "$EPOCHS"
  --sched_epochs "$SCHED_EPOCHS"
  --batch_size "$BATCH"
  --grad_accum "$ACCUM"
  --num_workers "$NUM_WORKERS"
  --prefetch_factor "$PREFETCH_FACTOR"
  --ens_mode fixed_range
  --ens_epoch_start "$ENS_EPOCH_START"
  --ens_epoch_end "$ENS_EPOCH_END"
  --ens_stride 1
)

if [[ -n "$TOKEN_CACHE_DIR" ]]; then
  COMMON+=(--token_cache_dir "$TOKEN_CACHE_DIR")
fi

if [[ "$HF_OFFLINE" == "1" ]]; then
  COMMON+=(--hf_offline)
fi

if [[ "$SAVE_MODEL" == "1" ]]; then
  COMMON+=(--save_model)
fi

if [[ "$GRAD_CKPT" == "1" ]]; then
  COMMON+=(--grad_ckpt)
fi

if [[ "$WANDB_ENABLED" == "1" ]]; then
  COMMON+=(--wandb --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" --wandb_group "$WANDB_GROUP" --wandb_job_type "$WANDB_JOB_TYPE")
  if [[ -n "$WANDB_ENTITY" ]]; then
    COMMON+=(--wandb_entity "$WANDB_ENTITY")
  fi
  if [[ -n "$WANDB_TAGS" ]]; then
    COMMON+=(--wandb_tags "$WANDB_TAGS")
  fi
fi

EXTRA_ARGS_ARR=()
if [[ -n "$EXTRA_ARGS" ]]; then
  read -r -a EXTRA_ARGS_ARR <<<"$EXTRA_ARGS"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export JAGeR_DEBUG="$JAGER_DEBUG"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_DIR="${WANDB_DIR:-$RESULTS_ROOT/.wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$RESULTS_ROOT/.wandb_cache}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

case "$JOB_SET" in
  ce_only|ce_sweep)
    JOBS=(
      "ce"
      "ce --ce_label_smoothing 0.05"
      "ce --ce_label_smoothing 0.03"
    )
    ;;
  full)
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
    ;;
  *)
    echo "ERROR: JOB_SET must be 'full', 'ce_only', or 'ce_sweep', got: $JOB_SET"
    exit 1
    ;;
esac

FOLDS=()
if [[ "$FOLD" == "all" ]]; then
  for ((i = 0; i < K; i++)); do
    FOLDS+=("$i")
  done
elif [[ "$FOLD" =~ ^[0-9]+$ ]]; then
  if (( FOLD < 0 || FOLD >= K )); then
    echo "ERROR: FOLD=$FOLD is outside [0, $((K - 1))]"
    exit 1
  fi
  FOLDS=("$FOLD")
else
  echo "ERROR: fold must be an integer or 'all', got: $FOLD"
  exit 1
fi

echo "[info] PWD=$PWD"
echo "[info] RUN_ID=$RUN_ID FOLD=$FOLD NPROC_PER_NODE=$NPROC_PER_NODE"
echo "[info] RESULTS_ROOT=$RESULTS_ROOT"
echo "[info] EPOCHS=$EPOCHS SCHED_EPOCHS=$SCHED_EPOCHS OOF=E${ENS_EPOCH_START}-${ENS_EPOCH_END}"
echo "[info] BATCH=$BATCH ACCUM=$ACCUM JOB_SET=$JOB_SET RESUME=$RESUME"
echo "[info] WANDB_ENABLED=$WANDB_ENABLED WANDB_MODE=$WANDB_MODE WANDB_GROUP=$WANDB_GROUP"
echo "[info] host=$(hostname)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
fi

echo "[check] SPLITS_DIR=$SPLITS_DIR"
ls -l "$SPLITS_DIR" || { echo "ERROR: cannot ls $SPLITS_DIR"; exit 1; }

count=$(ls -1 "$SPLITS_DIR"/fold*.json 2>/dev/null | wc -l | tr -d ' ')
[[ "$count" == "$K" ]] || { echo "ERROR: expected K=$K splits, found $count"; exit 1; }

[[ -f "$DATA_PATH" ]] || { echo "ERROR: data file missing: $DATA_PATH"; exit 1; }
[[ -d "$MODEL" ]] || { echo "ERROR: model dir missing: $MODEL"; exit 1; }
[[ -f "$MODEL/tokenizer.json" ]] || { echo "ERROR: saved fast tokenizer missing: $MODEL/tokenizer.json"; exit 1; }

mkdir -p "$RESULTS_ROOT"
echo "[check] tokenizer.json present at $MODEL/tokenizer.json"

OVERALL_FAIL_COUNT=0

for FOLD in "${FOLDS[@]}"; do
  MASTER_PORT=$((MASTER_PORT_BASE + FOLD))
  FAIL_COUNT=0
  FAIL_SUMMARY="$RESULTS_ROOT/fold${FOLD}.failed_jobs.txt"
  rm -f "$FAIL_SUMMARY"

  echo "[info] running fold ${FOLD} on host $(hostname)"

  for job in "${JOBS[@]}"; do
    read -r LOSS ARGS <<<"$job"

    if [[ -n "${ARGS:-}" ]]; then
      TAG="${LOSS}-$(echo "$ARGS" | tr ' ' '-' | tr -s '-')"
    else
      TAG="${LOSS}"
    fi

    SAVE="${RESULTS_ROOT}/${TAG}/fold${FOLD}"
    mkdir -p "$SAVE"
    LOG="$SAVE/train.log"
    EXIT_CODE_FILE="$SAVE/.exit_code"

    if [[ "$RESUME" == "1" && -f "$EXIT_CODE_FILE" ]]; then
      prev_code="$(tr -d '[:space:]' < "$EXIT_CODE_FILE")"
      if [[ "$prev_code" == "0" ]]; then
        echo ">> [fold ${FOLD}] ${TAG} already completed successfully; skipping"
        continue
      fi
    fi

    echo ">> [fold ${FOLD}] ${TAG} -> $LOG"

    set +e
    ARGS_ARR=()
    if [[ -n "${ARGS:-}" ]]; then
      read -r -a ARGS_ARR <<<"$ARGS"
    fi

    torchrun \
      --nproc_per_node="$NPROC_PER_NODE" \
      --master_port="$MASTER_PORT" \
      "$SCRIPT_DIR/train.py" "${COMMON[@]}" \
        --wandb_run_name "${RUN_ID}-fold${FOLD}-${TAG}" \
        "${EXTRA_ARGS_ARR[@]}" \
        --loss "$LOSS" "${ARGS_ARR[@]}" \
        --split_file "${SPLITS_DIR}/fold${FOLD}.json" \
        --save_dir "$SAVE" \
        >"$LOG" 2>&1

    code=$?
    echo "$code" > "$EXIT_CODE_FILE"

    if [[ "$code" != 0 ]]; then
      FAIL_COUNT=$((FAIL_COUNT + 1))
      OVERALL_FAIL_COUNT=$((OVERALL_FAIL_COUNT + 1))
      printf '%s\t%s\t%s\n' "$TAG" "$code" "$LOG" >> "$FAIL_SUMMARY"
      echo "!! [fold ${FOLD}] ${TAG} FAILED (code $code). See $LOG"
    fi
    set -e
  done

  if (( FAIL_COUNT > 0 )); then
    echo "[summary] Fold ${FOLD} finished with ${FAIL_COUNT} failed run(s). See $FAIL_SUMMARY"
  else
    echo "[done] Fold ${FOLD} finished. Results in: $RESULTS_ROOT"
  fi
done

if (( OVERALL_FAIL_COUNT > 0 )); then
  echo "[summary] Sweep finished with ${OVERALL_FAIL_COUNT} failed run(s). Results in: $RESULTS_ROOT"
  exit 1
fi

echo "[done] Sweep finished successfully. Results in: $RESULTS_ROOT"
