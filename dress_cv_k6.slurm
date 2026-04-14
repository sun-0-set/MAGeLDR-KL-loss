#!/bin/bash -l
#SBATCH -p a100 
#SBATCH --job-name=jager_dress_cv_k6_40ep
## SBATCH --job-name=jager_dress_cv_k6_jager0100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=02:30:00
#SBATCH --array=0-5
#SBATCH --output=../slurm_logs/jager_dress_cv_k6_40ep.out
#SBATCH --error=../slurm_logs/jager_dress_cv_k6_40ep.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavlo.sultanskyi@adelaide.edu.au  	# Email to which notifications will be sent

# ---- Environment (conda) ---------------------------------------------------

# Make sure usual shell config is loaded (modules, etc.)
if [[ -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

# Initialise conda in this non-interactive shell
eval "$(conda shell.bash hook)"

# Activate the env
conda activate jager-cv

echo "[info] which python: $(which python)"
echo "[info] which torchrun: $(which torchrun)"
python -c "import torch; print('[info] torch', torch.__version__, 'cuda', torch.cuda.is_available())"


set -euo pipefail

# Prefer the directory where `sbatch` was run (project root on HPC)
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
  # Fallback for running locally`
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
fi

cd "$SCRIPT_DIR"
echo "[info] PWD=$PWD"
echo "[info] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-none}"
echo "[info] SLURM_JOB_ID=${SLURM_JOB_ID:-none}  TASK_ID=${SLURM_ARRAY_TASK_ID:-none}"

# ---- CV + data config -------------------------------------------------------

K=6

SPLITS_DIR="$SCRIPT_DIR/../splits/k6_promptcv"
DATA_PATH="$SCRIPT_DIR/../data/DREsS/DREsS_New_cleaned.tsv"

MODEL="$SCRIPT_DIR/../models/deberta-v3-large"

EPOCHS=40
BATCH=8
ACCUM=2
MAXLEN=808
JAGER_DEBUG=1 

COMMON=(--data_path "$DATA_PATH" --model_name "$MODEL" --max_length "$MAXLEN"
        --epochs "$EPOCHS" --batch_size "$BATCH" --grad_accum "$ACCUM"
        --num_workers 4 --prefetch_factor 4)

export OMP_NUM_THREADS=1
export JAGeR_DEBUG="$JAGER_DEBUG"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_SHOW_CPP_STACKTRACES=1

JOBS=(
  "ce"
  "ce --ce_label_smoothing 0.1"
  "jager --no-joint --no-mixture --no-conf_gating --no-reassignment --lambda0 1 --lambda_min .5 --C 1e-1"
  "jager --no-joint --mixture --no-conf_gating --no-reassignment --lambda0 1 --lambda_min .5 --C 1e-1"
  "jager --no-joint --mixture --conf_gating --no-reassignment --lambda0 1 --lambda_min .5 --C 1e-1"
  "jager --no-joint --mixture --conf_gating --reassignment --lambda0 1 --lambda_min .5 --C 1e-1"
  "jager --joint --no-mixture --no-conf_gating --no-reassignment --lambda0 1 --lambda_min .5 --C 1e-1"
  "jager --joint --mixture --no-conf_gating --no-reassignment --lambda0 1 --lambda_min .5 --C 1e-1"
  "jager --joint --mixture --conf_gating --no-reassignment --lambda0 1 --lambda_min .5 --C 1e-1"
  "jager --joint --mixture --conf_gating --reassignment --lambda0 1 --lambda_min .5 --C 1e-1"
)

# ---- Fold selection (one fold per array task) -------------------------------

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Run with sbatch (not srun)."
  exit 1
fi

FOLD="${SLURM_ARRAY_TASK_ID}"

if (( FOLD < 0 || FOLD >= K )); then
  echo "ERROR: FOLD=$FOLD is outside [0, $((K-1))]"
  exit 1
fi

echo "[info] running fold ${FOLD} on host $(hostname)"

# ---- Basic sanity checks ----------------------------------------------------

echo "[check] SPLITS_DIR=$SPLITS_DIR"
ls -l "$SPLITS_DIR" || { echo "ERROR: cannot ls $SPLITS_DIR"; exit 1; }

count=$(ls -1 "$SPLITS_DIR"/fold*.json 2>/dev/null | wc -l | tr -d ' ')
[[ "$count" == "$K" ]] || { echo "ERROR: expected K=$K splits, found $count"; exit 1; }

[[ -f "$DATA_PATH" ]] || { echo "ERROR: data file missing: $DATA_PATH"; exit 1; }
[[ -d "$MODEL" ]] || echo "[note] MODEL dir not found; HF might download."
[[ -f "$MODEL/tokenizer.json" ]] || { echo "ERROR: saved fast tokenizer missing: $MODEL/tokenizer.json"; exit 1; }

splits_name="$(basename "$SPLITS_DIR")"
RESULTS_ROOT="$SCRIPT_DIR/../results/${splits_name}/${SLURM_ARRAY_JOB_ID}"
mkdir -p "$RESULTS_ROOT"
echo "[info] RESULTS_ROOT=$RESULTS_ROOT"
echo "[check] tokenizer.json present at $MODEL/tokenizer.json"

MASTER_PORT=$((29500 + FOLD))
FAIL_COUNT=0
FAIL_SUMMARY="$RESULTS_ROOT/fold${FOLD}.failed_jobs.txt"
rm -f "$FAIL_SUMMARY"

# ---- Loop over loss configs for this fold -----------------------------------

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
    --nproc_per_node=4 \
    --master_port="${MASTER_PORT}" \
    "$SCRIPT_DIR/train.py" "${COMMON[@]}" \
      --loss "$LOSS" "${ARGS_ARR[@]}" \
      --split_file "${SPLITS_DIR}/fold${FOLD}.json" \
      --save_dir "$SAVE" \
      >"$LOG" 2>&1

  code=$?
  echo "$code" > "$SAVE/.exit_code"

  if [[ "$code" != 0 ]]; then
    FAIL_COUNT=$((FAIL_COUNT + 1))
    printf '%s\t%s\t%s\n' "$TAG" "$code" "$LOG" >> "$FAIL_SUMMARY"
    echo "!! [fold ${FOLD}] ${TAG} FAILED (code $code). See $LOG"
  fi
  set -e
done

if (( FAIL_COUNT > 0 )); then
  echo "[summary] Fold ${FOLD} finished with ${FAIL_COUNT} failed run(s). See $FAIL_SUMMARY"
  exit 1
fi

echo "[done] Fold ${FOLD} finished. Results in: $RESULTS_ROOT"
