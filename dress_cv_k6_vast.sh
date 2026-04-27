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
  STAGE1_EPOCHS    two_stage_joint bare-joint epochs (default: 16)
  STAGE1_SCHED_EPOCHS
                   two_stage_joint bare-joint scheduler horizon (default: 40)
  STAGE2_EPOCHS    two_stage_joint mixture adaptation epochs (default: 8)
  STAGE2_SCHED_EPOCHS
                   two_stage_joint adaptation scheduler horizon (default: 8)
  ENS_MODE         fixed_range or tail (default: tail for two_stage_joint, fixed_range otherwise)
  ENS_EPOCH_START  First epoch in fixed OOF ensemble (default: 8)
  ENS_EPOCH_END    Last epoch in fixed OOF ensemble (default: 14)
  ENS_T            Number of trailing epochs for tail OOF (default: 5 for two_stage_joint, 10 otherwise)
  BATCH            Per-device batch size (default: 8, matching dress_cv_k6.slurm)
  ACCUM            Gradient accumulation steps (default: 2, matching dress_cv_k6.slurm)
  MAXLEN           Max sequence length (default: 808)
  NUM_WORKERS      DataLoader workers (default: 4)
  PREFETCH_FACTOR  DataLoader prefetch factor (default: 4)
  HF_OFFLINE       1 to force local HF assets only (default: 1)
  SAVE_MODEL       1 to save best.pt checkpoints (default: 0)
  SAVE_EPOCH_VAL_PREDS
                   1 to save epoch_val_preds.npz for all validation epochs (default: 0)
  STAGE2_SAVE_MODEL
                   two_stage_joint: 1 to keep stage-2 best.pt files (default: 0)
  KEEP_STAGE1_BEST two_stage_joint: 1 to keep stage-1 handoff best.pt files after
                   stage-2 fanout completes (default: 0)
  GRAD_CKPT        1 to enable --grad_ckpt if memory is tight (default: 0)
  JOB_SET          full (default; mirrors dress_cv_k6.slurm), ce_only, ce_sweep,
                   two_stage_joint, or leader_ce05_paired
  PAIR_SEEDS       leader_ce05_paired seeds (default: "42 43 44 45 46")
  LEADER_LAMBDA0   leader_ce05_paired JAGeR lambda0 (default: 3)
  LEADER_C         leader_ce05_paired JAGeR C (default: 5e-2)
  CE_SMOOTHING     leader_ce05_paired CE label smoothing (default: 0.05)
  LEADER_ENS_MODE  leader_ce05_paired JAGeR OOF mode (default: tail)
  LEADER_ENS_T     leader_ce05_paired JAGeR tail OOF T (default: 5)
  CE_ENS_MODE      leader_ce05_paired CE OOF mode (default: fixed_range)
  CE_ENS_EPOCH_START
                   leader_ce05_paired CE fixed OOF start (default: 8)
  CE_ENS_EPOCH_END leader_ce05_paired CE fixed OOF end (default: 14)
  PAIR_WINDOW_MATRIX
                   leader_ce05_paired: 1 runs both methods under both OOF
                   windows for paper sensitivity (default: 1)
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
  JOB_SET=two_stage_joint NPROC_PER_NODE=4 ./dress_cv_k6_vast.sh all vast_2stage_20260425
  JOB_SET=leader_ce05_paired PAIR_SEEDS="42 43 44 45 46" ./dress_cv_k6_vast.sh all vast_leader_ce05_s5
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

JOB_SET="${JOB_SET:-full}"
EPOCHS="${EPOCHS:-16}"
SCHED_EPOCHS="${SCHED_EPOCHS:-40}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-16}"
STAGE1_SCHED_EPOCHS="${STAGE1_SCHED_EPOCHS:-40}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-8}"
STAGE2_SCHED_EPOCHS="${STAGE2_SCHED_EPOCHS:-8}"
STAGE2_SAVE_MODEL="${STAGE2_SAVE_MODEL:-0}"
KEEP_STAGE1_BEST="${KEEP_STAGE1_BEST:-0}"
if [[ -z "${ENS_MODE:-}" ]]; then
  if [[ "$JOB_SET" == "two_stage_joint" ]]; then
    ENS_MODE="tail"
  else
    ENS_MODE="fixed_range"
  fi
fi
ENS_EPOCH_START="${ENS_EPOCH_START:-8}"
ENS_EPOCH_END="${ENS_EPOCH_END:-14}"
if [[ -z "${ENS_T:-}" ]]; then
  if [[ "$JOB_SET" == "two_stage_joint" ]]; then
    ENS_T="5"
  else
    ENS_T="10"
  fi
fi
ENS_STRIDE="${ENS_STRIDE:-1}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-2}"
MAXLEN="${MAXLEN:-808}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
JAGER_DEBUG="${JAGER_DEBUG:-1}"
HF_OFFLINE="${HF_OFFLINE:-1}"
SAVE_MODEL="${SAVE_MODEL:-0}"
SAVE_EPOCH_VAL_PREDS="${SAVE_EPOCH_VAL_PREDS:-0}"
GRAD_CKPT="${GRAD_CKPT:-0}"
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
TWO_STAGE_LAMBDA0S="${TWO_STAGE_LAMBDA0S:-1 2 3}"
TWO_STAGE_CS="${TWO_STAGE_CS:-1e-2 2e-2 5e-2}"
PAIR_SEEDS="${PAIR_SEEDS:-42 43 44 45 46}"
LEADER_LAMBDA0="${LEADER_LAMBDA0:-3}"
LEADER_C="${LEADER_C:-5e-2}"
LEADER_EPOCHS="${LEADER_EPOCHS:-$STAGE1_EPOCHS}"
LEADER_SCHED_EPOCHS="${LEADER_SCHED_EPOCHS:-$STAGE1_SCHED_EPOCHS}"
LEADER_ENS_MODE="${LEADER_ENS_MODE:-tail}"
LEADER_ENS_T="${LEADER_ENS_T:-5}"
LEADER_ENS_EPOCH_START="${LEADER_ENS_EPOCH_START:-$ENS_EPOCH_START}"
LEADER_ENS_EPOCH_END="${LEADER_ENS_EPOCH_END:-$ENS_EPOCH_END}"
CE_SMOOTHING="${CE_SMOOTHING:-0.05}"
CE_EPOCHS="${CE_EPOCHS:-$EPOCHS}"
CE_SCHED_EPOCHS="${CE_SCHED_EPOCHS:-$SCHED_EPOCHS}"
CE_ENS_MODE="${CE_ENS_MODE:-fixed_range}"
CE_ENS_T="${CE_ENS_T:-$ENS_T}"
CE_ENS_EPOCH_START="${CE_ENS_EPOCH_START:-8}"
CE_ENS_EPOCH_END="${CE_ENS_EPOCH_END:-14}"
PAIR_WINDOW_MATRIX="${PAIR_WINDOW_MATRIX:-1}"

case "$ENS_MODE" in
  fixed_range|tail) ;;
  *)
    echo "ERROR: ENS_MODE must be fixed_range or tail, got: $ENS_MODE"
    exit 1
    ;;
esac

COMMON_BASE=(
  --data_path "$DATA_PATH"
  --model_name "$MODEL"
  --max_length "$MAXLEN"
  --batch_size "$BATCH"
  --grad_accum "$ACCUM"
  --num_workers "$NUM_WORKERS"
  --prefetch_factor "$PREFETCH_FACTOR"
)

if [[ -n "$TOKEN_CACHE_DIR" ]]; then
  COMMON_BASE+=(--token_cache_dir "$TOKEN_CACHE_DIR")
fi

if [[ "$HF_OFFLINE" == "1" ]]; then
  COMMON_BASE+=(--hf_offline)
fi

if [[ "$SAVE_MODEL" == "1" && "$JOB_SET" != "two_stage_joint" ]]; then
  COMMON_BASE+=(--save_model)
fi

if [[ "$SAVE_EPOCH_VAL_PREDS" == "1" ]]; then
  COMMON_BASE+=(--save_epoch_val_preds)
fi

if [[ "$GRAD_CKPT" == "1" ]]; then
  COMMON_BASE+=(--grad_ckpt)
fi

if [[ "$WANDB_ENABLED" == "1" ]]; then
  COMMON_BASE+=(--wandb --wandb_project "$WANDB_PROJECT" --wandb_mode "$WANDB_MODE" --wandb_group "$WANDB_GROUP" --wandb_job_type "$WANDB_JOB_TYPE")
  if [[ -n "$WANDB_ENTITY" ]]; then
    COMMON_BASE+=(--wandb_entity "$WANDB_ENTITY")
  fi
  if [[ -n "$WANDB_TAGS" ]]; then
    COMMON_BASE+=(--wandb_tags "$WANDB_TAGS")
  fi
fi

make_common_args() {
  local epochs="$1"
  local sched_epochs="$2"
  local ens_mode="${3:-$ENS_MODE}"
  local ens_epoch_start="${4:-$ENS_EPOCH_START}"
  local ens_epoch_end="${5:-$ENS_EPOCH_END}"
  local ens_t="${6:-$ENS_T}"
  local ens_stride="${7:-$ENS_STRIDE}"
  COMMON=("${COMMON_BASE[@]}" --epochs "$epochs" --sched_epochs "$sched_epochs")
  case "$ens_mode" in
    tail)
      COMMON+=(--ens_mode tail --ens_t "$ens_t" --ens_stride "$ens_stride")
      ;;
    fixed_range)
      COMMON+=(--ens_mode fixed_range --ens_epoch_start "$ens_epoch_start" --ens_epoch_end "$ens_epoch_end" --ens_stride "$ens_stride")
      ;;
    *)
      echo "ERROR: ensemble mode must be fixed_range or tail, got: $ens_mode"
      exit 1
      ;;
  esac
}

make_common_args "$EPOCHS" "$SCHED_EPOCHS"

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
    JOBS=()
    for MIXTURE_FLAG in "--no-mixture" "--mixture"; do
      for LAMBDA0 in 1 2 3; do
        for C in 1e-2 2e-2 5e-2; do
          JOBS+=("jager --joint ${MIXTURE_FLAG} --no-conf_gating --no-reassignment --lambda0 ${LAMBDA0} --lambda_min 0.5 --C ${C}")
        done
      done
    done
    ;;
  two_stage_joint)
    JOBS=()
    read -r -a TWO_STAGE_LAMBDA0_ARR <<<"$TWO_STAGE_LAMBDA0S"
    read -r -a TWO_STAGE_C_ARR <<<"$TWO_STAGE_CS"
    ;;
  leader_ce05_paired)
    JOBS=()
    read -r -a PAIR_SEED_ARR <<<"$PAIR_SEEDS"
    ;;
  *)
    echo "ERROR: JOB_SET must be 'full', 'ce_only', 'ce_sweep', 'two_stage_joint', or 'leader_ce05_paired', got: $JOB_SET"
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
if [[ "$ENS_MODE" == "tail" ]]; then
  OOF_DESC="T${ENS_T}"
else
  OOF_DESC="E${ENS_EPOCH_START}-${ENS_EPOCH_END}"
fi
echo "[info] EPOCHS=$EPOCHS SCHED_EPOCHS=$SCHED_EPOCHS OOF=$OOF_DESC"
if [[ "$JOB_SET" == "two_stage_joint" ]]; then
  echo "[info] STAGE1_EPOCHS=$STAGE1_EPOCHS STAGE1_SCHED_EPOCHS=$STAGE1_SCHED_EPOCHS"
  echo "[info] STAGE2_EPOCHS=$STAGE2_EPOCHS STAGE2_SCHED_EPOCHS=$STAGE2_SCHED_EPOCHS"
  echo "[info] STAGE2_SAVE_MODEL=$STAGE2_SAVE_MODEL KEEP_STAGE1_BEST=$KEEP_STAGE1_BEST"
elif [[ "$JOB_SET" == "leader_ce05_paired" ]]; then
  echo "[info] PAIR_SEEDS=$PAIR_SEEDS"
  echo "[info] LEADER lambda0=$LEADER_LAMBDA0 C=$LEADER_C epochs=$LEADER_EPOCHS sched=$LEADER_SCHED_EPOCHS OOF=${LEADER_ENS_MODE}"
  echo "[info] CE smoothing=$CE_SMOOTHING epochs=$CE_EPOCHS sched=$CE_SCHED_EPOCHS OOF=${CE_ENS_MODE}"
  echo "[info] PAIR_WINDOW_MATRIX=$PAIR_WINDOW_MATRIX"
fi
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

  if [[ "$JOB_SET" == "two_stage_joint" ]]; then
    for LAMBDA0 in "${TWO_STAGE_LAMBDA0_ARR[@]}"; do
      for C in "${TWO_STAGE_C_ARR[@]}"; do
        STAGE1_TAG="jager-two-stage-stage1-bare-joint-lambda0-${LAMBDA0}-C-${C}"
        STAGE1_SAVE="${RESULTS_ROOT}/${STAGE1_TAG}/fold${FOLD}"
        STAGE1_LOG="$STAGE1_SAVE/train.log"
        STAGE1_EXIT_CODE_FILE="$STAGE1_SAVE/.exit_code"
        STAGE1_BEST="$STAGE1_SAVE/best.pt"

        mkdir -p "$STAGE1_SAVE"

        STAGE1_READY=0
        if [[ "$RESUME" == "1" && -f "$STAGE1_EXIT_CODE_FILE" ]]; then
          prev_code="$(tr -d '[:space:]' < "$STAGE1_EXIT_CODE_FILE")"
          if [[ "$prev_code" == "0" && -f "$STAGE1_BEST" ]]; then
            echo ">> [fold ${FOLD}] ${STAGE1_TAG} already completed successfully; skipping"
            STAGE1_READY=1
          fi
        fi

        if [[ "$STAGE1_READY" == "0" ]]; then
          echo ">> [fold ${FOLD}] ${STAGE1_TAG} -> $STAGE1_LOG"

          set +e
          make_common_args "$STAGE1_EPOCHS" "$STAGE1_SCHED_EPOCHS"
          torchrun \
            --nproc_per_node="$NPROC_PER_NODE" \
            --master_port="$MASTER_PORT" \
            "$SCRIPT_DIR/train.py" "${COMMON[@]}" \
              --wandb_run_name "${RUN_ID}-fold${FOLD}-${STAGE1_TAG}" \
              "${EXTRA_ARGS_ARR[@]}" \
              --loss jager --joint --no-mixture --no-conf_gating --no-reassignment \
              --lambda0 "$LAMBDA0" --lambda_min 0.5 --C "$C" \
              --save_model \
              --split_file "${SPLITS_DIR}/fold${FOLD}.json" \
              --save_dir "$STAGE1_SAVE" \
              >"$STAGE1_LOG" 2>&1

          code=$?
          echo "$code" > "$STAGE1_EXIT_CODE_FILE"

          if [[ "$code" == "0" && -f "$STAGE1_BEST" ]]; then
            STAGE1_READY=1
          else
            FAIL_COUNT=$((FAIL_COUNT + 1))
            OVERALL_FAIL_COUNT=$((OVERALL_FAIL_COUNT + 1))
            if [[ "$code" == "0" ]]; then
              printf '%s\t%s\t%s\n' "$STAGE1_TAG" "missing-best.pt" "$STAGE1_LOG" >> "$FAIL_SUMMARY"
              echo "!! [fold ${FOLD}] ${STAGE1_TAG} completed but did not write $STAGE1_BEST"
            else
              printf '%s\t%s\t%s\n' "$STAGE1_TAG" "$code" "$STAGE1_LOG" >> "$FAIL_SUMMARY"
              echo "!! [fold ${FOLD}] ${STAGE1_TAG} FAILED (code $code). See $STAGE1_LOG"
            fi
          fi
          set -e
        fi

        for CG in 0 1; do
          for RA in 0 1; do
            STAGE2_TAG="jager-two-stage-stage2-mixture-cg${CG}-ra${RA}-from-bare-lambda0-${LAMBDA0}-C-${C}"
            STAGE2_SAVE="${RESULTS_ROOT}/${STAGE2_TAG}/fold${FOLD}"
            STAGE2_LOG="$STAGE2_SAVE/train.log"
            STAGE2_EXIT_CODE_FILE="$STAGE2_SAVE/.exit_code"
            STAGE2_SAVE_MODEL_ARGS=()
            CG_ARG="--no-conf_gating"
            RA_ARG="--no-reassignment"
            if [[ "$CG" == "1" ]]; then
              CG_ARG="--conf_gating"
            fi
            if [[ "$RA" == "1" ]]; then
              RA_ARG="--reassignment"
            fi
            if [[ "$STAGE2_SAVE_MODEL" == "1" ]]; then
              STAGE2_SAVE_MODEL_ARGS=(--save_model)
            fi

            mkdir -p "$STAGE2_SAVE"

            if [[ "$STAGE1_READY" != "1" ]]; then
              {
                echo "ERROR: stage 2 requires the stage 1 checkpoint."
                echo "Missing or invalid checkpoint: $STAGE1_BEST"
              } >"$STAGE2_LOG"
              echo "1" > "$STAGE2_EXIT_CODE_FILE"
              FAIL_COUNT=$((FAIL_COUNT + 1))
              OVERALL_FAIL_COUNT=$((OVERALL_FAIL_COUNT + 1))
              printf '%s\t%s\t%s\n' "$STAGE2_TAG" "missing-stage1-best.pt" "$STAGE2_LOG" >> "$FAIL_SUMMARY"
              echo "!! [fold ${FOLD}] ${STAGE2_TAG} refused to run; missing $STAGE1_BEST"
              continue
            fi

            if [[ "$RESUME" == "1" && -f "$STAGE2_EXIT_CODE_FILE" ]]; then
              prev_code="$(tr -d '[:space:]' < "$STAGE2_EXIT_CODE_FILE")"
              if [[ "$prev_code" == "0" ]]; then
                echo ">> [fold ${FOLD}] ${STAGE2_TAG} already completed successfully; skipping"
                continue
              fi
            fi

            echo ">> [fold ${FOLD}] ${STAGE2_TAG} -> $STAGE2_LOG"

            set +e
            make_common_args "$STAGE2_EPOCHS" "$STAGE2_SCHED_EPOCHS"
            torchrun \
              --nproc_per_node="$NPROC_PER_NODE" \
              --master_port="$MASTER_PORT" \
              "$SCRIPT_DIR/train.py" "${COMMON[@]}" \
                --wandb_run_name "${RUN_ID}-fold${FOLD}-${STAGE2_TAG}" \
                "${EXTRA_ARGS_ARR[@]}" \
                --loss jager --joint --mixture "$CG_ARG" "$RA_ARG" \
                --lambda0 "$LAMBDA0" --lambda_min 0.5 --C "$C" \
                --init_model_from "$STAGE1_BEST" \
                "${STAGE2_SAVE_MODEL_ARGS[@]}" \
                --split_file "${SPLITS_DIR}/fold${FOLD}.json" \
                --save_dir "$STAGE2_SAVE" \
                >"$STAGE2_LOG" 2>&1

            code=$?
            echo "$code" > "$STAGE2_EXIT_CODE_FILE"

            if [[ "$code" != 0 ]]; then
              FAIL_COUNT=$((FAIL_COUNT + 1))
              OVERALL_FAIL_COUNT=$((OVERALL_FAIL_COUNT + 1))
              printf '%s\t%s\t%s\n' "$STAGE2_TAG" "$code" "$STAGE2_LOG" >> "$FAIL_SUMMARY"
              echo "!! [fold ${FOLD}] ${STAGE2_TAG} FAILED (code $code). See $STAGE2_LOG"
            fi
            set -e
          done
        done

        if [[ "$KEEP_STAGE1_BEST" != "1" && -f "$STAGE1_BEST" ]]; then
          rm -f "$STAGE1_BEST"
          echo ">> [fold ${FOLD}] ${STAGE1_TAG} pruned handoff checkpoint: $STAGE1_BEST"
        fi
      done
    done
  elif [[ "$JOB_SET" == "leader_ce05_paired" ]]; then
    LEADER_BASE_TAG="jager-leader-bare-joint-lambda0-${LEADER_LAMBDA0}-C-${LEADER_C}"
    CE_BASE_TAG="ce-label-smoothing-${CE_SMOOTHING}"
    LEADER_WINDOW_TAG="T${LEADER_ENS_T}"
    if [[ "$LEADER_ENS_MODE" == "fixed_range" ]]; then
      LEADER_WINDOW_TAG="E${LEADER_ENS_EPOCH_START}-${LEADER_ENS_EPOCH_END}"
    fi
    CE_WINDOW_TAG="T${CE_ENS_T}"
    if [[ "$CE_ENS_MODE" == "fixed_range" ]]; then
      CE_WINDOW_TAG="E${CE_ENS_EPOCH_START}-${CE_ENS_EPOCH_END}"
    fi

    for SEED in "${PAIR_SEED_ARR[@]}"; do
      PAIR_JOBS=("leader:leader")
      if [[ "$PAIR_WINDOW_MATRIX" == "1" ]]; then
        PAIR_JOBS+=("leader:ce" "ce05:leader")
      fi
      PAIR_JOBS+=("ce05:ce")

      for PAIR_SPEC in "${PAIR_JOBS[@]}"; do
        METHOD_KEY="${PAIR_SPEC%%:*}"
        WINDOW_KEY="${PAIR_SPEC##*:}"

        case "$METHOD_KEY" in
          leader)
            BASE_TAG="$LEADER_BASE_TAG"
            JOB_ARGS=(
              --loss jager --joint --no-mixture --no-conf_gating --no-reassignment
              --lambda0 "$LEADER_LAMBDA0" --lambda_min 0.5 --C "$LEADER_C"
            )
            EPOCHS_FOR_JOB="$LEADER_EPOCHS"
            SCHED_FOR_JOB="$LEADER_SCHED_EPOCHS"
            ;;
          ce05)
            BASE_TAG="$CE_BASE_TAG"
            JOB_ARGS=(--loss ce --ce_label_smoothing "$CE_SMOOTHING")
            EPOCHS_FOR_JOB="$CE_EPOCHS"
            SCHED_FOR_JOB="$CE_SCHED_EPOCHS"
            ;;
          *)
            echo "ERROR: unknown paired method key: $METHOD_KEY"
            exit 1
            ;;
        esac

        case "$WINDOW_KEY" in
          leader)
            WINDOW_TAG="$LEADER_WINDOW_TAG"
            make_common_args "$EPOCHS_FOR_JOB" "$SCHED_FOR_JOB" "$LEADER_ENS_MODE" "$LEADER_ENS_EPOCH_START" "$LEADER_ENS_EPOCH_END" "$LEADER_ENS_T" "$ENS_STRIDE"
            ;;
          ce)
            WINDOW_TAG="$CE_WINDOW_TAG"
            make_common_args "$EPOCHS_FOR_JOB" "$SCHED_FOR_JOB" "$CE_ENS_MODE" "$CE_ENS_EPOCH_START" "$CE_ENS_EPOCH_END" "$CE_ENS_T" "$ENS_STRIDE"
            ;;
          *)
            echo "ERROR: unknown paired window key: $WINDOW_KEY"
            exit 1
            ;;
        esac

        TAG="${BASE_TAG}-${WINDOW_TAG}"
        SAVE="${RESULTS_ROOT}/seed${SEED}/${TAG}/fold${FOLD}"

        mkdir -p "$SAVE"
        LOG="$SAVE/train.log"
        EXIT_CODE_FILE="$SAVE/.exit_code"

        if [[ "$RESUME" == "1" && -f "$EXIT_CODE_FILE" ]]; then
          prev_code="$(tr -d '[:space:]' < "$EXIT_CODE_FILE")"
          if [[ "$prev_code" == "0" ]]; then
            echo ">> [fold ${FOLD} seed ${SEED}] ${TAG} already completed successfully; skipping"
            continue
          fi
        fi

        echo ">> [fold ${FOLD} seed ${SEED}] ${TAG} -> $LOG"

        set +e
        torchrun \
          --nproc_per_node="$NPROC_PER_NODE" \
          --master_port="$MASTER_PORT" \
          "$SCRIPT_DIR/train.py" "${COMMON[@]}" \
            --wandb_run_name "${RUN_ID}-seed${SEED}-fold${FOLD}-${TAG}" \
            "${EXTRA_ARGS_ARR[@]}" \
            "${JOB_ARGS[@]}" \
            --seed "$SEED" \
            --split_file "${SPLITS_DIR}/fold${FOLD}.json" \
            --save_dir "$SAVE" \
            >"$LOG" 2>&1

        code=$?
        echo "$code" > "$EXIT_CODE_FILE"

        if [[ "$code" != 0 ]]; then
          FAIL_COUNT=$((FAIL_COUNT + 1))
          OVERALL_FAIL_COUNT=$((OVERALL_FAIL_COUNT + 1))
          printf '%s\t%s\t%s\n' "seed${SEED}/${TAG}" "$code" "$LOG" >> "$FAIL_SUMMARY"
          echo "!! [fold ${FOLD} seed ${SEED}] ${TAG} FAILED (code $code). See $LOG"
        fi
        set -e
      done
    done
  else
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
  fi

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
