#!/usr/bin/env python3
"""
Prompt-grouped K-fold CV (K=8) with simple stratification:

- Keep each prompt entirely in one fold (no leakage).
- Two-pass assignment:
  (1) Place prompts that carry scarce classes first, round-robin by current fold size
      (tie-break: fold with fewest scarce essays so far).
  (2) Place remaining prompts by pure size balance (always to the smallest fold).
- Optional minimal fix-up: if a fold has zero for a scarce class (and it's reasonably
  present globally), move one small prompt that contains it from a surplus fold,
  respecting a ±size_tolerance around the target fold size.

Outputs: fold0..fold7.json with {train,val,test}, plus meta.json.
"""

import argparse, os, json, math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd


# ---------- helpers ----------

def infer_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    raise SystemExit(f"[error] Could not find any of {candidates} in columns={list(cols)}")

def sorted_unique_non_nan(series: pd.Series):
    vals = pd.unique(series.dropna())
    try:
        return sorted(vals)
    except Exception:
        # Fallback if mixed types
        return sorted(vals, key=lambda x: str(x))

def fold_size_bounds(n_total: int, k: int, tol: float) -> Tuple[int, int]:
    target = n_total / k
    lo = math.floor((1.0 - tol) * target)
    hi = math.ceil((1.0 + tol) * target)
    return lo, hi

def human_counts(d: Dict[Any, int]) -> str:
    items = ", ".join(f"{k}:{int(v)}" for k, v in d.items())
    return "{" + items + "}"


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--outdir", default="splits/k8_promptcv")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt-col", default=None,
                    help="Column with prompt id/name; tries ['prompt_id','prompt','topic','Task','Prompt'] if omitted.")
    ap.add_argument("--id-col", default=None,
                    help="Optional stable ID column. If omitted, row index (0..N-1) is used.")
    ap.add_argument("--y-cols", default="content,organization,language",
                    help="Comma-separated label columns (per-head analytic scores).")
    ap.add_argument("--scarce-thresh", type=float, default=0.10,
                    help="Score share threshold to mark a class as scarce (per head). Default=0.10 (10%%).")
    ap.add_argument("--min-class-global", type=int, default=20,
                    help="Ignore scarce classes with very low absolute support (< this).")
    ap.add_argument("--size-tolerance", type=float, default=0.15,
                    help="Allowed ± deviation from N/K when checking fold sizes. Default=0.15 (±15%%).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load data
    df = pd.read_csv(args.tsv, sep="\t")
    if args.prompt_col is None:
        args.prompt_col = infer_col(df.columns, ["prompt_id", "prompt", "topic", "Task", "Prompt"])
    prompt_col = args.prompt_col

    ycols = [c.strip() for c in args.y_cols.split(",") if c.strip()]
    for c in ycols:
        if c not in df.columns:
            raise SystemExit(f"[error] Missing label column: {c}")

    if args.id_col is None:
        ids = np.arange(len(df), dtype=int)
    else:
        if args.id_col not in df.columns:
            raise SystemExit(f"[error] id-col '{args.id_col}' not found in data.")
        ids = df[args.id_col].to_numpy()

    # Global per-head unique classes & counts
    head_classes: Dict[str, List[Any]] = {}
    global_counts: Dict[str, Dict[Any, int]] = {}
    scarce_classes: Dict[str, List[Any]] = {}

    for h in ycols:
        head_classes[h] = sorted_unique_non_nan(df[h])
        cnt = Counter(df[h].dropna())
        global_counts[h] = {c: int(cnt.get(c, 0)) for c in head_classes[h]}
        total_h = sum(global_counts[h].values())
        scarce = []
        for c, v in global_counts[h].items():
            if total_h == 0:
                continue
            share = v / total_h
            if share < args.scarce_thresh and v >= args.min_class_global:
                scarce.append(c)
        scarce_classes[h] = scarce

    n_total = len(df)
    K = int(args.k)
    lo_bound, hi_bound = fold_size_bounds(n_total, K, args.size_tolerance)
    target = n_total / K

    print(f"[info] N={n_total}, K={K}, target per fold ≈ {target:.1f}, bounds: [{lo_bound}, {hi_bound}]")
    print(f"[info] prompt column: {prompt_col}")
    for h in ycols:
        print(f"[info] {h}: classes={head_classes[h]} global={human_counts(global_counts[h])} "
              f"scarce<{args.scarce_thresh:.0%}=> {scarce_classes[h]} (min abs {args.min_class_global})")

    # Build per-prompt summaries
    groups = []  # each: dict(prompt, ids, n, per_head_counts, scarce_weight)
    for pval, gdf in df.groupby(prompt_col, sort=True, dropna=False):
        gidx = ids[gdf.index.to_numpy()]
        n = len(gidx)
        per_head_counts = {}
        scarce_weight = 0
        for h in ycols:
            cnt = Counter(gdf[h].dropna())
            head_cnt = {c: int(cnt.get(c, 0)) for c in head_classes[h]}
            per_head_counts[h] = head_cnt
            # add counts of scarce classes for weight
            scarce_weight += sum(head_cnt.get(c, 0) for c in scarce_classes[h])
        groups.append({
            "prompt": pval,
            "ids": gidx.tolist(),
            "n": n,
            "per_head_counts": per_head_counts,
            "scarce_weight": int(scarce_weight),
        })

    # Sort prompts deterministically for assignment
    def prompt_key(g):
        # Deterministic tie-breaker
        return (str(g["prompt"]))

    groups.sort(key=lambda g: (-g["scarce_weight"], -g["n"], prompt_key(g)))
    carriers = [g for g in groups if g["scarce_weight"] > 0]
    others   = [g for g in groups if g["scarce_weight"] == 0]
    others.sort(key=lambda g: (-g["n"], prompt_key(g)))

    # Folds state
    folds = []
    for j in range(K):
        folds.append({
            "ids": [],
            "n": 0,
            "scarce": 0,
            "sum_counts": {h: {c: 0 for c in head_classes[h]} for h in ycols},
            "prompts": [],   # keep prompt objects for possible fix-up
        })

    def place_on_fold(g, j):
        f = folds[j]
        f["ids"].extend(g["ids"])
        f["n"] += g["n"]
        f["scarce"] += g["scarce_weight"]
        for h in ycols:
            for c, v in g["per_head_counts"][h].items():
                f["sum_counts"][h][c] += v
        f["prompts"].append(g)

    # --- Pass 1: place scarce-class carriers (size-first; tie-break scarce essays) ---
    for g in carriers:
        # smallest n; tie-break by smallest scarce
        ns = [f["n"] for f in folds]
        min_n = min(ns)
        cand = [j for j, f in enumerate(folds) if f["n"] == min_n]
        if len(cand) == 1:
            j = cand[0]
        else:
            scarce_vals = [(folds[j]["scarce"], j) for j in cand]
            scarce_vals.sort()
            j = scarce_vals[0][1]
        place_on_fold(g, j)

    # --- Pass 2: fill rest by pure size balance ---
    for g in others:
        j = int(np.argmin([f["n"] for f in folds]))
        place_on_fold(g, j)

    # ----- Sanity check & minimal fix-up -----
    def coverage_matrix() -> Dict[str, Dict[Any, List[int]]]:
        cov = {h: {c: [0]*K for c in head_classes[h]} for h in ycols}
        for j, f in enumerate(folds):
            for h in ycols:
                for c, v in f["sum_counts"][h].items():
                    cov[h][c][j] = int(v)
        return cov

    sizes = [f["n"] for f in folds]
    print(f"[after assign] fold sizes: {sizes} (min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f})")

    cov = coverage_matrix()
    # identify deficits for scarce classes
    deficits = []  # tuples of (h, c, fold_j)
    for h in ycols:
        for c in scarce_classes[h]:
            if global_counts[h][c] < args.min_class_global:
                continue
            for j in range(K):
                if cov[h][c][j] == 0:
                    deficits.append((h, c, j))

    # Try to fix each deficit with one small move (if feasible)
    moves_made = 0
    for (h, c, j_def) in deficits:
        if cov[h][c][j_def] > 0:
            continue  # might have been fixed by earlier move
        # candidate donors: folds with cov > 0 for (h,c)
        donors = [k for k in range(K) if cov[h][c][k] > 0 and k != j_def]
        if not donors:
            continue
        # choose donor fold with greatest surplus of (h,c), tie-break by largest size
        donors.sort(key=lambda k: (cov[h][c][k], folds[k]["n"]), reverse=True)

        moved = False
        for k in donors:
            # candidate prompts inside donor fold that contain (h,c); choose smallest n
            prompts_with_class = [g for g in folds[k]["prompts"]
                                  if g["per_head_counts"][h].get(c, 0) > 0]
            if not prompts_with_class:
                continue
            prompts_with_class.sort(key=lambda g: (g["n"], str(g["prompt"])))
            for g in prompts_with_class:
                # check size bounds if we move g from k -> j_def
                new_n_def = folds[j_def]["n"] + g["n"]
                new_n_k   = folds[k]["n"]     - g["n"]
                if (new_n_def <= hi_bound) and (new_n_k >= lo_bound):
                    # perform the move
                    folds[k]["prompts"].remove(g)
                    folds[j_def]["prompts"].append(g)
                    folds[k]["ids"] = [i for i in folds[k]["ids"] if i not in set(g["ids"])]
                    folds[j_def]["ids"].extend(g["ids"])
                    folds[k]["n"]  -= g["n"]
                    folds[j_def]["n"] += g["n"]
                    folds[k]["scarce"]  -= g["scarce_weight"]
                    folds[j_def]["scarce"] += g["scarce_weight"]
                    for hh in ycols:
                        for cc, vv in g["per_head_counts"][hh].items():
                            folds[k]["sum_counts"][hh][cc]     -= vv
                            folds[j_def]["sum_counts"][hh][cc] += vv
                    cov = coverage_matrix()
                    print(f"[fix] moved prompt={g['prompt']} (n={g['n']}) from fold{k}→fold{j_def} to cover scarce ({h}={c})")
                    moves_made += 1
                    moved = True
                    break
            if moved:
                break

    sizes = [f["n"] for f in folds]
    print(f"[final] fold sizes: {sizes} (min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}); moves={moves_made}")

    # Coverage report for scarce classes
    print("[coverage] scarce-class counts per fold (head:class => [f0..fK-1])")
    for h in ycols:
        for c in scarce_classes[h]:
            arr = [folds[j]["sum_counts"][h][c] for j in range(K)]
            print(f"  {h}:{c} => {arr}")

    # ----- write outputs -----
    os.makedirs(args.outdir, exist_ok=True)
    all_idx = set(int(x) for x in ids.tolist())
    for j in range(K):
        val_ids = sorted(int(x) for x in folds[j]["ids"])
        train_ids = sorted(all_idx - set(val_ids))
        out = {"train": train_ids, "val": val_ids, "test": val_ids}
        with open(os.path.join(args.outdir, f"fold{j}.json"), "w") as fh:
            json.dump(out, fh)
    meta = {
        "k": K,
        "seed": args.seed,
        "prompt_col": prompt_col,
        "y_cols": ycols,
        "scarce_thresh": args.scarce_thresh,
        "min_class_global": args.min_class_global,
        "size_tolerance": args.size_tolerance,
        "sizes": sizes,
        "scarce_classes": {h: list(map(lambda x: x if isinstance(x, (int, float, str)) else str(x),
                                       scarce_classes[h])) for h in ycols},
    }
    with open(os.path.join(args.outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[done] wrote {K} folds to {args.outdir}/fold*.json and meta.json")


if __name__ == "__main__":
    main()
