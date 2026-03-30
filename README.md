# Multi-Trait Ordinal Classification Experiments

This repository is an active research workspace for multi-trait ordinal classification on essay-response data.

The current setup uses a shared Transformer encoder plus one prediction head per trait to score `(prompt, essay)` pairs.

In the DREsS experiments, those traits are typically `content`, `organization`, and `language`, each treated as an ordinal label in `1..5`.

## What This Repo Is Testing

The main question is how different loss constructions behave for multi-trait ordinal prediction, especially when the traits are modeled either independently or jointly.

Current experiments compare:

- `ce`: standard per-trait cross-entropy
- `ce` with label smoothing
- `jager`: the ordinal loss family implemented in [loss.py](/scratchdata1/users/a1841939/MAGeLDR-KL-loss/loss.py)
- joint vs per-trait modeling via `--joint` / `--no-joint`
- mixture-based modeling via `--mixture` / `--no-mixture`
- confidence-gated updates via `--conf_gating`
- competitor reassignment via `--reassignment`
- label-distribution-aware margins controlled by `--lambda0`, `--lambda_min`, `--alpha`, and `--C`

The launch scripts sweep combinations of these switches to study which parts help for multi-trait ordinal classification.

## Research Status

This is research-in-progress code, not a stable package.

Expect the following to change as experiments evolve:

- CLI flags
- default paths
- launch scripts
- result layouts
- naming

## Data Assumptions

The DREsS dataset is not included in the repository but us available publicly [here](https://haneul-yoo.github.io/dress/).

Training expects a CSV or TSV with:

- `prompt`
- `essay`
- one or more numeric ordinal target columns

By default, [train.py](/scratchdata1/users/a1841939/MAGeLDR-KL-loss/train.py) looks for:

`../data/DREsS/DREsS_New_cleaned.tsv`

[make_prompt_cv_splits.py](/scratchdata1/users/a1841939/MAGeLDR-KL-loss/make_prompt_cv_splits.py) can be used to generate prompt-grouped cross-validation splits so prompts stay isolated across train, validation, and test folds.

## Main Files

- [train.py](/scratchdata1/users/a1841939/MAGeLDR-KL-loss/train.py): train and evaluate a single experiment
- [loss.py](/scratchdata1/users/a1841939/MAGeLDR-KL-loss/loss.py): loss implementations and ablation logic
- [modeling_multitask.py](/scratchdata1/users/a1841939/MAGeLDR-KL-loss/modeling_multitask.py): shared encoder with one head per trait
- [make_prompt_cv_splits.py](/scratchdata1/users/a1841939/MAGeLDR-KL-loss/make_prompt_cv_splits.py): prompt-grouped CV split generation
- [launch_cv_k6.sh](/scratchdata1/users/a1841939/MAGeLDR-KL-loss/launch_cv_k6.sh) and [dress_cv_k6.slurm](/scratchdata1/users/a1841939/MAGeLDR-KL-loss/dress_cv_k6.slurm): example ablation sweeps
- [utils/aggregate_results.py](/scratchdata1/users/a1841939/MAGeLDR-KL-loss/utils/aggregate_results.py): collect per-run `metrics.json` files

## Minimal Setup

```bash
conda create -y -n jager-cv python=3.13
conda activate jager-cv
pip install -r requirements.txt
```

## Minimal Run

After preparing the data file and a split JSON:

```bash
python train.py \
  --data_path ../data/DREsS/DREsS_New_cleaned.tsv \
  --split_file splits/k6_promptcv/fold0.json \
  --model_name ../models/deberta-v3-large \
  --loss jager --joint --mixture --conf_gating --reassignment \
  --lambda0 3 --lambda_min 0.5 --C 1e-1 \
  --epochs 3 --batch_size 2 --grad_accum 8 \
  --save_dir runs/example
```

For the full ablation grid, adapt the provided launch scripts to your local machine or cluster.
