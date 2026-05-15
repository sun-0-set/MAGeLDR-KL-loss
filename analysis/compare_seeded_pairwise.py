#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp, wilcoxon

from aggregate_oof import discover_cfg_dirs, load_oof_csv_groups, oof_metrics_from_df


LOWER_IS_BETTER_METRICS = {"macroEMD"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare two configs across seed*/config/fold* OOF outputs."
    )
    p.add_argument("results_root")
    p.add_argument("config_a", help="Config basename or relative path; positive deltas favor A.")
    p.add_argument("config_b", help="Config basename or relative path.")
    p.add_argument("--window-a", default=None, help="OOF window for config A, e.g. T5.")
    p.add_argument("--window-b", default=None, help="OOF window for config B, e.g. E8-14.")
    p.add_argument("--metric", default="qwk_avg")
    p.add_argument(
        "--seed-range",
        default=None,
        help="Inclusive numeric seed range to compare, e.g. 55-64 or seed55-seed64.",
    )
    return p.parse_args()


def metric_direction(metric: str) -> int:
    return -1 if metric in LOWER_IS_BETTER_METRICS or metric.lower().endswith("emd") else 1


def seed_from_relpath(relpath: str) -> str:
    first = relpath.split(os.sep, 1)[0]
    return first if first.startswith("seed") else "."


def seed_number(seed: str) -> int | None:
    match = re.fullmatch(r"seed(\d+)", seed)
    return int(match.group(1)) if match else None


def seed_sort_key(seed: str):
    num = seed_number(seed)
    return (0, num) if num is not None else (1, seed)


def parse_seed_range(seed_range: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(?:seed)?(\d+)\s*[-:]\s*(?:seed)?(\d+)\s*", seed_range)
    if match is None:
        raise ValueError(
            f"Invalid --seed-range {seed_range!r}; expected START-END, e.g. 55-64."
        )
    lo, hi = int(match.group(1)), int(match.group(2))
    if lo > hi:
        raise ValueError(f"Invalid --seed-range {seed_range!r}; start must be <= end.")
    return lo, hi


def filter_seeds_by_range(seeds: list[str], seed_range: str | None) -> list[str]:
    if seed_range is None:
        return seeds
    lo, hi = parse_seed_range(seed_range)
    selected = []
    for seed in seeds:
        num = seed_number(seed)
        if num is not None and lo <= num <= hi:
            selected.append(seed)
    return selected


def ref_matches(relpath: str, ref: str) -> bool:
    ref = ref.rstrip(os.sep)
    return relpath == ref or os.path.basename(relpath) == ref or relpath.endswith(os.sep + ref)


def find_cfgs(results_root: str, ref: str):
    matches = []
    for cfg_dir in discover_cfg_dirs(results_root):
        rel = os.path.relpath(cfg_dir, results_root)
        if ref_matches(rel, ref):
            matches.append((seed_from_relpath(rel), rel, cfg_dir))
    if not matches:
        raise FileNotFoundError(f"No config directories matching {ref!r} under {results_root}")
    return matches


def load_window_dfs(cfg_dir: str, window: str | None):
    groups = load_oof_csv_groups(cfg_dir)
    if window is None:
        if len(groups) != 1:
            choices = ", ".join(g[0] for g in groups)
            raise ValueError(f"{cfg_dir} has multiple windows ({choices}); pass --window-a/--window-b")
        window = groups[0][0]

    for window_tag, dfs, fold_ids in groups:
        if window_tag == window:
            return {int(fold_idx): df for fold_idx, df in zip(fold_ids, dfs)}
    choices = ", ".join(g[0] for g in groups)
    raise KeyError(f"{cfg_dir} has no window {window}; choices: {choices}")


def metrics_from_fold_dfs(fold_dfs: dict[int, pd.DataFrame]):
    fold_metrics = {
        fold_idx: oof_metrics_from_df(df)
        for fold_idx, df in fold_dfs.items()
    }
    pooled_metrics = oof_metrics_from_df(
        pd.concat([fold_dfs[k] for k in sorted(fold_dfs)], ignore_index=True)
    )
    return fold_metrics, pooled_metrics


def mean_ci(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(x.mean())
    if n == 1:
        return mean, 0.0, mean, mean
    se = float(x.std(ddof=1) / math.sqrt(n))
    hw = float(student_t.ppf(0.975, n - 1) * se)
    return mean, se, mean - hw, mean + hw


def main():
    args = parse_args()
    results_root = os.path.abspath(args.results_root)
    direction = metric_direction(args.metric)

    cfgs_a = find_cfgs(results_root, args.config_a)
    cfgs_b = find_cfgs(results_root, args.config_b)
    by_seed_a = {seed: (rel, path) for seed, rel, path in cfgs_a}
    by_seed_b = {seed: (rel, path) for seed, rel, path in cfgs_b}
    all_common_seeds = sorted(set(by_seed_a) & set(by_seed_b), key=seed_sort_key)
    common_seeds = filter_seeds_by_range(all_common_seeds, args.seed_range)
    if not common_seeds:
        available = ", ".join(all_common_seeds) if all_common_seeds else "<none>"
        suffix = (
            f" in requested range {args.seed_range!r}"
            if args.seed_range is not None
            else ""
        )
        raise RuntimeError(
            f"No common seeds{suffix} between the matched config directories. "
            f"Available common seeds: {available}"
        )

    seed_fold_deltas = defaultdict(list)
    seed_pooled_deltas = {}
    rows = []
    for seed in common_seeds:
        rel_a, path_a = by_seed_a[seed]
        rel_b, path_b = by_seed_b[seed]
        fold_dfs_a = load_window_dfs(path_a, args.window_a)
        fold_dfs_b = load_window_dfs(path_b, args.window_b)
        common_folds = sorted(set(fold_dfs_a) & set(fold_dfs_b))
        if not common_folds:
            raise RuntimeError(f"No common folds for seed={seed}.")

        metrics_a, pooled_a = metrics_from_fold_dfs({f: fold_dfs_a[f] for f in common_folds})
        metrics_b, pooled_b = metrics_from_fold_dfs({f: fold_dfs_b[f] for f in common_folds})
        if args.metric not in pooled_a or args.metric not in pooled_b:
            raise KeyError(f"Metric {args.metric!r} not available for seed={seed} pooled OOF")
        seed_pooled_deltas[seed] = direction * float(pooled_a[args.metric] - pooled_b[args.metric])

        for fold in common_folds:
            if args.metric not in metrics_a[fold] or args.metric not in metrics_b[fold]:
                raise KeyError(f"Metric {args.metric!r} not available for seed={seed} fold={fold}")
            raw_delta = float(metrics_a[fold][args.metric] - metrics_b[fold][args.metric])
            signed_delta = direction * raw_delta
            seed_fold_deltas[seed].append(signed_delta)
            rows.append((seed, fold, raw_delta, signed_delta, rel_a, rel_b))

    all_deltas = np.array([row[3] for row in rows], dtype=float)
    seed_fold_means = np.array([np.mean(seed_fold_deltas[s]) for s in common_seeds], dtype=float)
    seed_pooled = np.array([seed_pooled_deltas[s] for s in common_seeds], dtype=float)
    all_mean, all_se, all_lo, all_hi = mean_ci(all_deltas)
    pooled_mean, pooled_se, pooled_lo, pooled_hi = mean_ci(seed_pooled)
    seed_mean, seed_se, seed_lo, seed_hi = mean_ci(seed_fold_means)

    better_note = "higher is better" if direction > 0 else "lower is better"
    print(f"Config A: {args.config_a}")
    print(f"Config B: {args.config_b}")
    print(f"Metric:   {args.metric} ({better_note}; positive signed delta favors A)")
    print(f"Seeds:    {', '.join(common_seeds)}")
    print(f"Pairs:    {len(rows)} seed-fold pairs")

    print("\nSeed Means")
    for seed in common_seeds:
        vals = np.array(seed_fold_deltas[seed], dtype=float)
        print(
            f"  {seed}: pooled_signed_delta={seed_pooled_deltas[seed]:+.6f} "
            f"fold_mean_signed_delta={vals.mean():+.6f} folds={vals.size}"
        )

    print("\nSummary")
    print(
        f"  seed_pooled_delta={pooled_mean:+.6f} se={pooled_se:.6f} "
        f"95%_CI=[{pooled_lo:+.6f}, {pooled_hi:+.6f}] n_seeds={seed_pooled.size}"
    )
    print(
        f"  seed_fold_mean_delta={seed_mean:+.6f} se={seed_se:.6f} "
        f"95%_CI=[{seed_lo:+.6f}, {seed_hi:+.6f}] n_seeds={seed_fold_means.size}"
    )
    print(
        f"  seed_fold_delta={all_mean:+.6f} se={all_se:.6f} "
        f"95%_CI=[{all_lo:+.6f}, {all_hi:+.6f}] n_pairs={all_deltas.size}"
    )
    if seed_pooled.size > 1:
        print(f"  seed_pooled_t_p={ttest_1samp(seed_pooled, 0.0).pvalue:.6g}")
        print(f"  seed_fold_mean_t_p={ttest_1samp(seed_fold_means, 0.0).pvalue:.6g}")
        try:
            print(f"  seed_pooled_wilcoxon_p={wilcoxon(seed_pooled).pvalue:.6g}")
        except ValueError:
            pass


if __name__ == "__main__":
    main()
