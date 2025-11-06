#!/usr/bin/env bash
set -euo pipefail

# ---- constants ----
K=6
GPUS=(0 1 2 3 4 5 6 7)
SPLITS_DIR="splits/k6_promptcv"
TSV=../data/DREsS/DREsS_New_cleaned.tsv
MODEL=../models/deberta-v3-large

EPOCHS=35
T=7               # last-epoch ensemble window
BATCH=6
ACCUM=2
MAXLEN=808

COMMON="--tsv $TSV --model_name $MODEL --max_length $MAXLEN \
 --epochs $EPOCHS --batch_size $BATCH --grad_accum $ACCUM \
 --use_fast_tokenizer 1 --num_workers 4 --prefetch_factor 4 \
 --ens_t $T --ens_stride 1"

# Keep allocator friendly on H100
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_SHOW_CPP_STACKTRACES=1

# ---- configs (11) ----
declare -a JOBS=(
  "ce                   ''"
  "jager --joint 0 --mixture 0"
  "jager --joint 1 --mixture 0"
  "jager --joint 0 --mixture 1 --conf_gating 0 --reassignment 0"
  "jager --joint 0 --mixture 1 --conf_gating 1 --reassignment 0"
  "jager --joint 0 --mixture 1 --conf_gating 0 --reassignment 1"
  "jager --joint 0 --mixture 1 --conf_gating 1 --reassignment 1"
  "jager --joint 1 --mixture 1 --conf_gating 0 --reassignment 0"
  "jager --joint 1 --mixture 1 --conf_gating 1 --reassignment 0"
  "jager --joint 1 --mixture 1 --conf_gating 0 --reassignment 1"
  "jager --joint 1 --mixture 1 --conf_gating 1 --reassignment 1"
)

# Build a task queue: each item is "gpu=-1|loss|args|tag|fold"
tasks=()
for i in "${!JOBS[@]}"; do
  read -r LOSS ARGS <<<"${JOBS[$i]}"
  TAG="${LOSS}-$(echo "$ARGS" | tr ' ' '-' | tr -s '-')"
  for f in $(seq 0 $((K-1))); do
    tasks+=("-1|$LOSS|$ARGS|$TAG|$f")
  done
done

# Track running PIDs per GPU
declare -A PID
declare -A LOG

next=0
num_tasks=${#tasks[@]}

launch_one () {
  local gpu="$1" loss="$2" args="$3" tag="$4" fold="$5"
  local save="results/${tag}/fold${fold}"
  mkdir -p "$save"
  LOG[$gpu]="${save}/train.log"
  echo ">> [GPU $gpu] $tag fold${fold}"
  ( set +e
    CUDA_VISIBLE_DEVICES=$gpu \
      python -u train.py $COMMON --loss $loss $args \
        --split_file "${SPLITS_DIR}/fold${fold}.json" \
        --save_dir "$save"
    echo $? > "${save}/.exit_code"
  ) > "${LOG[$gpu]}" 2>&1 &
  PID[$gpu]=$!
}

# Live tails
for g in "${GPUS[@]}"; do
  touch /tmp/log.${g}.txt
done
{
  for g in "${GPUS[@]}"; do
    [ -z "${LOG[$g]:-}" ] && LOG[$g]="/tmp/log.${g}.txt"
    stdbuf -oL -eL awk -v p="[GPU ${g}]" '{print p, $0}' < <(tail -n +1 -F "${LOG[$g]}") &
  done
} >/dev/stderr

# Main scheduling loop
alive=0
while (( next < num_tasks )) || (( alive > 0 )); do
  # fill free GPUs
  for g in "${GPUS[@]}"; do
    if [[ -z "${PID[$g]:-}" ]] || ! kill -0 "${PID[$g]}" 2>/dev/null; then
      if (( next < num_tasks )); then
        IFS='|' read -r _ loss args tag fold <<< "${tasks[$next]}"
        launch_one "$g" "$loss" "$args" "$tag" "$fold"
        ((alive++))
        ((next++))
      fi
    fi
  done

  # check for exits
  for g in "${GPUS[@]}"; do
    if [[ -n "${PID[$g]:-}" ]] && ! kill -0 "${PID[$g]}" 2>/dev/null; then
      # job ended; verify exit code
      # infer save dir from latest LOG[g] path:
      save_dir=$(dirname "${LOG[$g]}")
      code=0
      [[ -f "${save_dir}/.exit_code" ]] && code=$(cat "${save_dir}/.exit_code")
      if (( code != 0 )); then
        echo "!! job on GPU $g failed with code $code. See ${LOG[$g]}"
        exit $code
      fi
      unset PID[$g]
      ((alive--))
    fi
  done
  sleep 2
done

echo "All tasks finished."
