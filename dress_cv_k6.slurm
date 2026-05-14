#!/bin/bash -l
#SBATCH -p a100
#SBATCH --job-name=dress_k6_jager_c0_s62_121_10min_timeouted
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:10:00
#SBATCH --array=348,350
#SBATCH --output=../slurm_logs/%x_%A_%a.out
#SBATCH --error=../slurm_logs/%x_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pavlo.sultanskyi@adelaide.edu.au
#SBATCH --exclude=p2-gpu-26

set -euo pipefail

# ---- Environment (conda) ---------------------------------------------------

if [[ -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

eval "$(conda shell.bash hook)"
conda activate jager-cv

echo "[info] which python: $(which python)"
echo "[info] which torchrun: $(which torchrun)"
python -c "import torch; print('[info] torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"

# ---- Project location ------------------------------------------------------

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
fi

cd "$SCRIPT_DIR"

echo "[info] PWD=$PWD"
echo "[info] SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-none}"
echo "[info] SLURM_JOB_ID=${SLURM_JOB_ID:-none} TASK_ID=${SLURM_ARRAY_TASK_ID:-none}"
echo "[info] host=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits || true

# ---- Seed/fold mapping -----------------------------------------------------

K=6
SEED_START=62
SEED_END=121
TASK_COUNT=$(((SEED_END - SEED_START + 1) * K))

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID is not set. Run with sbatch."
  exit 1
fi

TASK_ID="$SLURM_ARRAY_TASK_ID"

if (( TASK_ID < 0 || TASK_ID >= TASK_COUNT )); then
  echo "ERROR: TASK_ID=$TASK_ID is outside [0, $((TASK_COUNT - 1))]"
  exit 1
fi

SEED=$((SEED_START + TASK_ID / K))
FOLD=$((TASK_ID % K))

echo "[info] mapped TASK_ID=$TASK_ID to seed=$SEED fold=$FOLD"

# ---- DREsS C=0 JAGeR run config ------------------------------------------

ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
RUN_ID="${RUN_ID:-hpc_dress_k6_a100_jager_c0_s62_121_e18_b9_ga4_${ARRAY_JOB_ID}}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_DISABLE_ADDR2LINE="${TORCH_DISABLE_ADDR2LINE:-1}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export JOB_SET=leader_ce05_paired
export PAIR_INCLUDE_CE=0
export PAIR_WINDOW_MATRIX=0
export PAIR_SEEDS="$SEED"
export LEADER_C=0
export LEADER_LAMBDA0=3
export LEADER_EPOCHS=18
export LEADER_SCHED_EPOCHS=40
export LEADER_ENS_MODE=fixed_range
export LEADER_ENS_EPOCH_START=10
export LEADER_ENS_EPOCH_END=16
export SAVE_EPOCH_VAL_PREDS=1
export BATCH=8
export ACCUM=5
export MAXLEN=808
export NPROC_PER_NODE=4
export NUM_WORKERS=0
export RESUME=1
export HF_OFFLINE=1
export WANDB_ENABLED=0
export TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-$SCRIPT_DIR/../cache/tokens}"

mkdir -p "$TOKEN_CACHE_DIR"

echo "[info] RUN_ID=$RUN_ID"
echo "[info] BATCH=$BATCH ACCUM=$ACCUM NPROC_PER_NODE=$NPROC_PER_NODE"
echo "[info] LEADER_C=$LEADER_C epochs=$LEADER_EPOCHS sched=$LEADER_SCHED_EPOCHS OOF=E${LEADER_ENS_EPOCH_START}-${LEADER_ENS_EPOCH_END}"
echo "[info] TOKEN_CACHE_DIR=$TOKEN_CACHE_DIR"

./dress_cv_k6_vast.sh "$FOLD" "$RUN_ID"
