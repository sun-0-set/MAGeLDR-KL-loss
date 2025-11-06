#!/usr/bin/env bash
set -euo pipefail

# ── pin working dir to this script ──
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"
echo "[info] PWD=$PWD"

# ── config ──
K=6
GPUS=(0 1 2 3 4 5 6 7)

SPLITS_DIR="$SCRIPT_DIR/splits/k6_promptcv"
TSV="$SCRIPT_DIR/../data/DREsS/DREsS_New_cleaned.tsv"
MODEL="$SCRIPT_DIR/../models/deberta-v3-large"

EPOCHS=35
T=7
BATCH=38
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

# ── preflight ──
echo "[check] SPLITS_DIR=$SPLITS_DIR"
ls -l "$SPLITS_DIR" || { echo "ERROR: cannot ls $SPLITS_DIR"; exit 1; }
count=$(ls -1 "$SPLITS_DIR"/fold*.json 2>/dev/null | wc -l | tr -d ' ')
[[ "$count" == "$K" ]] || { echo "ERROR: expected K=$K splits, found $count"; exit 1; }
[[ -f "$TSV" ]] || { echo "ERROR: TSV missing: $TSV"; exit 1; }
[[ -d "$MODEL" ]] || echo "[note] MODEL dir not found; HF might download."

# ── build task arrays (no delimiters, no IFS tricks) ──
TASK_LOSS=(); TASK_ARGS=(); TASK_TAG=(); TASK_FOLD=()
for job in "${JOBS[@]}"; do
  read -r LOSS ARGS <<<"$job"          # ARGS may be empty; 'read' sets it to ""
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
echo "[plan] total tasks: $TOTAL ; workers/GPUs: $SLOTS (exactly one per GPU)"

# ── worker function: takes a slot index (0..7), runs i=slot, i+=SLOTS ──
run_worker() {
  local slot="$1"
  local gpu="${GPUS[$slot]}"
  local i
  for (( i=slot; i<TOTAL; i+=SLOTS )); do
    local LOSS="${TASK_LOSS[$i]}"
    local ARGS="${TASK_ARGS[$i]}"
    local TAG="${TASK_TAG[$i]}"
    local FOLD="${TASK_FOLD[$i]}"

    local SAVE="$SCRIPT_DIR/results/${TAG}/fold${FOLD}"
    mkdir -p "$SAVE"
    local LOG="$SAVE/train.log"

    echo ">> [GPU $gpu] ${TAG} fold${FOLD} → $LOG"
    (
      set +e
      # Split ARGS into array safely
      ARGS_ARR=()
      if [[ -n "$ARGS" ]]; then read -r -a ARGS_ARR <<<"$ARGS"; fi
      CUDA_VISIBLE_DEVICES="$gpu" \
        python -u "$SCRIPT_DIR/train.py" "${COMMON[@]}" \
          --loss "$LOSS" "${ARGS_ARR[@]}" \
          --split_file "${SPLITS_DIR}/fold${FOLD}.json" \
          --save_dir "$SAVE"
      echo $? > "$SAVE/.exit_code"
    ) >"$LOG" 2>&1

    code=$(cat "$SAVE/.exit_code" 2>/dev/null || echo 1)
    if [[ "$code" != "0" ]]; then
      echo "!! [GPU $gpu] ${TAG} fold${FOLD} FAILED (code $code). See $LOG"
      # continue to next assigned task; we report failures at the end
    fi
  done
}

# ── launch exactly one worker per GPU ──
pids=()
for (( s=0; s<SLOTS; s++ )); do
  run_worker "$s" &
  pids+=("$!")
done

# ── wait for all workers ──
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

if (( status != 0 )); then
  echo "One or more tasks failed. Grep 'FAILED' in results/*/fold*/train.log"
  exit 1
fi
echo "All tasks finished OK."
