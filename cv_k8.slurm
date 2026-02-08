#!/bin/bash -l
#SBATCH -p a100 
#SBATCH --job-name=jager_ela_cv_k8_alpha3_lam1.25
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=28:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm_logs/jager_ela_cv_k8_alpha3_lam1.25.out
#SBATCH --error=slurm_logs/jager_ela_cv_k8_alpha3_lam1.25.err
#SBATCH --mail-type=END					    	# Send a notification email when the job is done (=END)
#SBATCH --mail-type=FAIL   					# Send a notification email when the job fails (=FAIL)
#SBATCH --mail-user=pavlo.sultanskyi@adelaide.edu.au  	# Email to which notifications will be sent

# ---- Environment (conda) ---------------------------------------------------

# Make sure usual shell config is loaded (modules, etc.)
if [[ -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

# Initialise conda in this non-interactive shell
eval "$(conda shell.bash hook)"

# Activate the env
conda activate jager-cv-k8

echo "[info] which python: $(which python)"
echo "[info] which torchrun: $(which torchrun)"
python -c "import torch; print('[info] torch', torch.__version__, 'cuda', torch.cuda.is_available())"


set -euo pipefail

# Prefer the directory where `sbatch` was run (project root on HPC)
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
  # Fallback for running locally with `bash cv_k8.slurm`
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
fi

cd "$SCRIPT_DIR"
echo "[info] PWD=$PWD"
echo "[info] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-none}"
echo "[info] SLURM_JOB_ID=${SLURM_JOB_ID:-none}  TASK_ID=${SLURM_ARRAY_TASK_ID:-none}"

# ---- CV + data config -------------------------------------------------------

K=8

SPLITS_DIR="$SCRIPT_DIR/splits/k8_promptcv_ela"
DATA_PATH="$SCRIPT_DIR/../data/English-Language-Learners-Evaluation/ELA_Dataset_cleaned.csv"

MODEL="$SCRIPT_DIR/../models/deberta-v3-large"

EPOCHS=40
T=10
BATCH=4
ACCUM=8
MAXLEN=1024

COMMON=(--data_path "$DATA_PATH" --model_name "$MODEL" --max_length "$MAXLEN"
        --epochs "$EPOCHS" --batch_size "$BATCH" --grad_accum "$ACCUM"
        --use_fast_tokenizer 1 --num_workers 4 --prefetch_factor 4
        --ens_t "$T" --ens_stride 1 --log_epoch_stats)

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_SHOW_CPP_STACKTRACES=1

JOBS=(
  "ce"
  "ce --ce_label_smoothing 0.1"
  "jager --joint 0 --mixture 0 --conf_gating 0 --reassignment 0 --lambda0 3 --alpha 1.25"
  "jager --joint 1 --mixture 0 --conf_gating 0 --reassignment 0 --lambda0 3 --alpha 1.25"
  # "jager --joint 0 --mixture 1 --conf_gating 0 --reassignment 0 --lambda0 3 --alpha 1.25"
  # "jager --joint 0 --mixture 1 --conf_gating 1 --reassignment 0 --lambda0 3 --alpha 1.25"
  # "jager --joint 0 --mixture 1 --conf_gating 0 --reassignment 1 --lambda0 3 --alpha 1.25"
  # "jager --joint 0 --mixture 1 --conf_gating 1 --reassignment 1 --lambda0 3 --alpha 1.25"
  "jager --joint 1 --mixture 1 --conf_gating 0 --reassignment 0 --lambda0 3 --alpha 1.25"
  "jager --joint 1 --mixture 1 --conf_gating 1 --reassignment 0 --lambda0 3 --alpha 1.25"
  "jager --joint 1 --mixture 1 --conf_gating 0 --reassignment 1 --lambda0 3 --alpha 1.25"
  "jager --joint 1 --mixture 1 --conf_gating 1 --reassignment 1 --lambda0 3 --alpha 1.25"
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

splits_name="$(basename "$SPLITS_DIR")"
RESULTS_ROOT="$SCRIPT_DIR/results/${splits_name}/${SLURM_ARRAY_JOB_ID}"
mkdir -p "$RESULTS_ROOT"
echo "[info] RESULTS_ROOT=$RESULTS_ROOT"

MASTER_PORT=$((29500 + FOLD))

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
    echo "!! [fold ${FOLD}] ${TAG} FAILED (code $code). See $LOG"
  fi
  set -e
done

echo "[done] Fold ${FOLD} finished. Results in: $RESULTS_ROOT"
