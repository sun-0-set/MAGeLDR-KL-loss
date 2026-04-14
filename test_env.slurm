#!/bin/bash
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=test_env_%j.out

if [[ -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi
eval "$(conda shell.bash hook)"
conda activate jager-cv

echo "[info] python: $(which python)"
echo "[info] torchrun: $(which torchrun)"
python - << 'EOF'
import torch
print("torch:", torch.__version__)
print("cuda.is_available():", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
EOF
torchrun --version
