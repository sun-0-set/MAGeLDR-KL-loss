## MAGe‑LDR‑KL

3‑head DeBERTa-v3-large for Content/Organization/Language (5 classes each). Losses: MAGe‑LDR-KL (mixture/uniform), ALDR‑KL, CE.

Trained on DREsS dataset with `prompt`, `essay` and targets: `content`,`organization`,`language` (ints in 1..5). Not included in the repository - available [here](https://haneul-yoo.github.io/dress/)

Examples of runs:

### Test run (1 epoch)
```bash
python train.py \
  --loss mage --distribution mixture --lambda0 1 --alpha 2 --C 0.3 \
  --split_file splits/dress_seed42.json --eval_test \
  --epochs 1 --batch_size 2 --grad_accum 4 \
  --model_name tasksource/deberta-small-long-nli --max_length 512 \
  --save_dir sweeps/smoke
```

## 3 epochs with batch size 2, gradiaent accumulation at 8:
```bash
python train.py \
  --loss mage --distribution mixture --lambda0 1 --alpha 2 --C 0.3 \
  --split_file splits/dress_seed42.json --eval_test --inference_with_prior \
  --epochs 3 --batch_size 2 --grad_accum 8 \
  --model_name /path/to/local_models/deberta-v3-large --max_length 1024 \
  --hf_offline --use_fast_tokenizer 1 \
  --save_dir sweeps/run1
```

## Nested CV (joint‑stratified)
```bash
python sweep.py --phase nested \
  --tsv ../data/DREsS/DREsS_New_cleaned.tsv \
  --nested_k 5 --nested_j 1 --nested_seed 42 \
  --concurrency 2 --devices 0,1
```

## Outputs per run
- `best.pt`, `run_args.json`, `metrics.json` (val/test: acc, QWK, F1; overall micro‑F1).


*Active research code; interfaces may evolve.*

