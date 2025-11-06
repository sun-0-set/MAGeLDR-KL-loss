#!/usr/bin/env bash
set -euo pipefail

# ---------- config ----------
K=6
GPUS=(0 1 2 3 4 5 6 7)

SPLITS_DIR="splits/k6_promptcv"
TSV="../data/DREsS/DREsS_New_cleaned.tsv"
MODEL="../models/deberta-v3-large"

EPOCHS=35
T=7
BATCH=6
ACCUM=2
MAXLEN=808

COMMON=(--tsv "$TSV" --model_name "$MODEL" --max_length "$MAXLEN"
        --epochs "$EPOCHS" --batch_size "$BATCH" --grad_accum "$ACCUM"
        --use_fast_tokenizer 1 --num_workers 4 --prefetch_factor 4
        --ens_t "$T" --ens_stride 1)

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_SHOW_CPP_STACKTRACES=1

JOBS=(
  "ce"
  "jager --joint 0 --mixture 0 --conf_gating 0 --reassignment 0"
  "jager --joint 1 --mixture 0 --conf_gating 0 --reassignment 0"
  "jager --joint 0 --mixture 1 --conf_gating 0 --reassignment 0"
  "jager --joint 0 --mixture 1 --conf_gating 1 --reassignment 0"
  "jager --joint 0 --mixture 1 --conf_gating 0 --reassignment 1"
  "jager --joint 0 --mixture 1 --conf_gating 1 --reassignment 1"
  "jager --joint 1 --mixture 1 --conf_gating 0 --reassignment 0"
  "jager --joint 1 --mixture 1 --conf_gating 1 --reassignment 0"
  "jager --joint 1 --mixture 1 --conf_gating 0 --reassignment 1"
  "jager --joint 1 --mixture 1 --conf_gating 1 --reassignment 1"
)

# --- preflight ---
if [[ ! -d "$SPLITS_DIR" ]]; then
  echo "ERROR: SPLITS_DIR not found: $SPLITS_DIR"; exit 1
fi
count=$(ls -1 "$SPLITS_DIR"/fold*.json 2>/dev/null | wc -l | tr -d ' ')
if [[ "$count" != "$K" ]]; then
  echo "ERROR: expected K=$K split files in $SPLITS_DIR, found $count"
  ls -1 "$SPLITS_DIR"/fold*.json 2>/dev/null || true
  exit 1
fi

# ---------- build task arrays (no string packing) ----------
TASK_LOSS=()
TASK_ARGS=()
TASK_TAG=()
TASK_FOLD=()

for job in "${JOBS[@]}"; do
  read -r LOSS ARGS <<<"$job"        # ARGS = full remainder (may be empty)
  if [[ -n "${ARGS:-}" ]]; then
    TAG="${LOSS}-$(echo "$ARGS" | tr ' ' '-' | tr -s '-')"
  else
    TAG="${LOSS}"
  fi
  for f in $(seq 0 $((K-1))); do
    TASK_LOSS+=("$LOSS")
    TASK_ARGS+=("${ARGS:-}")
    TASK_TAG+=("$TAG")
    TASK_FOLD+=("$f")
  done
done

total=${#TASK_LOSS[@]}
if (( total == 0 )); then echo "No tasks."; exit 0; fi

# ---------- scheduler: keep EXACTLY 8 running when >=8 remain ----------
slots=${#GPUS[@]}
declare -a PIDS; PIDS=()
declare -a LOGS; LOGS=()
for ((s=0; s<slots; s++)); do PIDS[$s]=0; LOGS[$s]=""; done

task_i=0
fail=0

launch_on_slot () {
  local s="$1"
  local gpu="${GPUS[$s]}"

  local LOSS="${TASK_LOSS[$task_i]}"
  local ARGS="${TASK_ARGS[$task_i]}"
  local TAG="${TASK_TAG[$task_i]}"
  local FOLD="${TASK_FOLD[$task_i]}"
  ((task_i++))

  local SAVE="results/${TAG}/fold${FOLD}"
  mkdir -p "$SAVE"
  LOGS[$s]="${SAVE}/train.log"

  echo ">> [GPU $gpu] ${TAG} fold${FOLD}"
  (
    set +e
    ARGS_ARR=()
    if [[ -n "$ARGS" ]]; then
      # split ARGS (space-separated flags) into an array safely
      read -r -a ARGS_ARR <<<"$ARGS"
    fi
    CUDA_VISIBLE_DEVICES="$gpu" \
      python -u train.py "${COMMON[@]}" \
        --loss "$LOSS" "${ARGS_ARR[@]}" \
        --split_file "${SPLITS_DIR}/fold${FOLD}.json" \
        --save_dir "$SAVE"
    echo $? > "${SAVE}/.exit_code"
  ) > "${LOGS[$s]}" 2>&1 &

  PIDS[$s]=$!
}

# fill up to 8 (or fewer if total < 8)
to_start=$(( total < slots ? total : slots ))
for ((s=0; s<to_start; s++)); do launch_on_slot "$s"; done

# keep exactly 8 running until tasks < 8 remain
while (( task_i < total )); do
  # wait for one child to finish
  if ! wait -n; then true; fi
  # find finished slots and refill immediately
  for ((s=0; s<slots && task_i<total; s++)); do
    pid=${PIDS[$s]}
    if (( pid != 0 )) && ! kill -0 "$pid" 2>/dev/null; then
      # check exit
      save_dir=$(dirname "${LOGS[$s]}")
      code=$(cat "${save_dir}/.exit_code" 2>/dev/null || echo 1)
      if [[ "$code" != "0" ]]; then
        echo "!! [GPU ${GPUS[$s]}] FAILED (code $code). See ${LOGS[$s]}"
        fail=1
      fi
      PIDS[$s]=0
      launch_on_slot "$s"   # refill → keep concurrency at 8
    fi
  done
done

# finish remaining ≤8 jobs
for ((s=0; s<slots; s++)); do
  pid=${PIDS[$s]}
  if (( pid != 0 )); then
    wait "$pid" || true
    save_dir=$(dirname "${LOGS[$s]}")
    code=$(cat "${save_dir}/.exit_code" 2>/dev/null || echo 1)
    if [[ "$code" != "0" ]]; then
      echo "!! [GPU ${GPUS[$s]}] FAILED (code $code). See ${LOGS[$s]}"
      fail=1
    fi
  fi
done

if (( fail != 0 )); then
  echo "Some tasks failed. Inspect 'results/*/fold*/train.log' for details."
  exit 1
fi
echo "All tasks finished OK."
