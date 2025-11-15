#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"
echo "[info] PWD=$PWD"

K=6
GPUS=(0 1 2 3 4 5)

SPLITS_DIR="$SCRIPT_DIR/splits/k6_scorecv"

TSV="$SCRIPT_DIR/../data/DREsS/DREsS_New_cleaned.tsv"
MODEL="$SCRIPT_DIR/../models/deberta-v3-large"

EPOCHS=35
T=7
BATCH=4
ACCUM=8
MAXLEN=1024

COMMON=(--tsv "$TSV" --model_name "$MODEL" --max_length "$MAXLEN"
        --epochs "$EPOCHS" --batch_size "$BATCH" --grad_accum "$ACCUM"
        --use_fast_tokenizer 1 --num_workers 4 --prefetch_factor 4
        --ens_t "$T" --ens_stride 1 --log_epoch_stats)

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_SHOW_CPP_STACKTRACES=1

JOBS=(
  "ce"
  "ce --ce_label_smoothing 0.1"
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

echo "[check] SPLITS_DIR=$SPLITS_DIR"
ls -l "$SPLITS_DIR" || { echo "ERROR: cannot ls $SPLITS_DIR"; exit 1; }
count=$(ls -1 "$SPLITS_DIR"/fold*.json 2>/dev/null | wc -l | tr -d ' ')
[[ "$count" == "$K" ]] || { echo "ERROR: expected K=$K splits, found $count"; exit 1; }
[[ -f "$TSV" ]] || { echo "ERROR: TSV missing: $TSV"; exit 1; }
[[ -d "$MODEL" ]] || echo "[note] MODEL dir not found; HF might download."

splits_name="$(basename "$SPLITS_DIR")"
base_root="$SCRIPT_DIR/results/${splits_name}"

RESULTS_ROOT="$base_root"
run_idx=1
while [[ -d "$RESULTS_ROOT" ]]; do
  RESULTS_ROOT="${base_root}_run${run_idx}"
  run_idx=$((run_idx + 1))
done
mkdir -p "$RESULTS_ROOT"
echo "[info] RESULTS_ROOT=$RESULTS_ROOT"

TASK_LOSS=(); TASK_ARGS=(); TASK_TAG=(); TASK_FOLD=()
for job in "${JOBS[@]}"; do
  read -r LOSS ARGS <<<"$job"
  if [[ -n "$ARGS" ]]; then
    TAG="${LOSS}-$(echo "$ARGS" | tr ' ' '-' | tr -s '-')"
  else
    TAG="${LOSS}"
  fi
  for f in $(seq 0 $((K-1))); do
    TASK_LOSS+=("$LOSS")
    TASK_ARGS+=("$ARGS")
    TASK_TAG+=("$TAG")
    TASK_FOLD+=("$f")
  done
done
TOTAL=${#TASK_LOSS[@]}
SLOTS=${#GPUS[@]}
echo "[plan] total tasks: $TOTAL"

run_worker() {
  local slot="$1"
  local gpu="${GPUS[$slot]}"
  local i

  for (( i=slot; i<TOTAL; i+=SLOTS )); do
    local LOSS="${TASK_LOSS[$i]}"
    local ARGS="${TASK_ARGS[$i]}"
    local TAG="${TASK_TAG[$i]}"
    local FOLD="${TASK_FOLD[$i]}"

    local SAVE="${RESULTS_ROOT}/${TAG}/fold${FOLD}"
    mkdir -p "$SAVE"
    local LOG="$SAVE/train.log"

    echo ">> [GPU $gpu] ${TAG} fold${FOLD} → $LOG"

    (
      set +e
      ARGS_ARR=()
      if [[ -n "$ARGS" ]]; then
        read -r -a ARGS_ARR <<<"$ARGS"
      fi
      CUDA_VISIBLE_DEVICES="$gpu" \
        python -u "$SCRIPT_DIR/train.py" "${COMMON[@]}" \
          --loss "$LOSS" "${ARGS_ARR[@]}" \
          --split_file "${SPLITS_DIR}/fold${FOLD}.json" \
          --save_dir "$SAVE"
      echo $? > "$SAVE/.exit_code"
    ) >"$LOG" 2>&1

    local code
    code=$(cat "$SAVE/.exit_code" 2>/dev/null || echo 1)
    if [[ "$code" != "0" ]]; then
      echo "!! [GPU $gpu] ${TAG} fold${FOLD} FAILED (code $code). See $LOG"
    fi
  done
}

pids=()
for (( s=0; s<SLOTS; s++ )); do
  run_worker "$s" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

if (( status != 0 )); then
  echo "One or more tasks failed. Grep 'FAILED' in ${RESULTS_ROOT}/*/fold*/train.log"
  exit 1
fi

echo "All tasks finished OK. Results in: $RESULTS_ROOT"
