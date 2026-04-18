#!/usr/bin/env python3
import argparse
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from aggregate_oof import discover_cfg_dirs, load_oof_csv_groups, oof_metrics_from_df


LOWER_IS_BETTER_METRICS = {"macroEMD"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare two configs with matched-fold OOF deltas."
    )
    p.add_argument(
        "results_root",
        help="Root containing config folders, or run folders that contain config folders.",
    )
    p.add_argument(
        "config_a",
        help="Config A path relative to results_root, or an absolute config directory.",
    )
    p.add_argument(
        "config_b",
        help="Config B path relative to results_root, or an absolute config directory.",
    )
    p.add_argument(
        "--window",
        default="E8-14",
        help="Ensemble window tag to compare, e.g. E8-14 or T10.",
    )
    p.add_argument(
        "--metric",
        default="qwk_avg",
        help="Metric key from the OOF summaries, e.g. qwk_avg or macroEMD.",
    )
    return p.parse_args()


def resolve_cfg_dir(results_root: str, cfg_ref: str) -> str:
    if os.path.isabs(cfg_ref):
        if not os.path.isdir(cfg_ref):
            raise FileNotFoundError(f"Config directory does not exist: {cfg_ref}")
        return os.path.abspath(cfg_ref)

    direct = os.path.join(results_root, cfg_ref)
    if os.path.isdir(direct):
        return os.path.abspath(direct)

    matches = []
    for cfg_dir in discover_cfg_dirs(results_root):
        rel = os.path.relpath(cfg_dir, results_root)
        if rel == cfg_ref or os.path.basename(cfg_dir) == cfg_ref:
            matches.append(cfg_dir)

    if not matches:
        raise FileNotFoundError(
            f"Could not resolve config '{cfg_ref}' under {results_root}"
        )
    if len(matches) > 1:
        rels = ", ".join(sorted(os.path.relpath(m, results_root) for m in matches))
        raise ValueError(f"Config '{cfg_ref}' is ambiguous under {results_root}: {rels}")
    return os.path.abspath(matches[0])


def load_window_map(cfg_dir: str):
    groups = load_oof_csv_groups(cfg_dir)
    out = {}
    for window_tag, dfs, fold_ids in groups:
        out[window_tag] = {fold_idx: df for fold_idx, df in zip(fold_ids, dfs)}
    return out


def metric_direction(metric: str) -> int:
    return -1 if metric in LOWER_IS_BETTER_METRICS or metric.lower().endswith("emd") else 1


def t_interval(values: np.ndarray):
    n = int(values.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    mean = float(values.mean())
    if n == 1:
        return mean, 0.0, mean, mean

    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(n)
    tcrit = float(student_t.ppf(0.975, df=n - 1))
    half_width = tcrit * se
    return mean, se, mean - half_width, mean + half_width


def metric_value(df: pd.DataFrame, metric: str) -> float:
    metrics = oof_metrics_from_df(df)
    if metric not in metrics:
        available = ", ".join(sorted(metrics))
        raise KeyError(f"Metric '{metric}' not available. Choices: {available}")
    return float(metrics[metric])


def fmt_signed(x: float) -> str:
    return f"{x:+.6f}"


def main():
    args = parse_args()
    results_root = os.path.abspath(args.results_root)
    cfg_a_dir = resolve_cfg_dir(results_root, args.config_a)
    cfg_b_dir = resolve_cfg_dir(results_root, args.config_b)

    window_map_a = load_window_map(cfg_a_dir)
    window_map_b = load_window_map(cfg_b_dir)
    if args.window not in window_map_a:
        raise KeyError(
            f"{os.path.relpath(cfg_a_dir, results_root)} does not have window {args.window}. "
            f"Available: {', '.join(sorted(window_map_a))}"
        )
    if args.window not in window_map_b:
        raise KeyError(
            f"{os.path.relpath(cfg_b_dir, results_root)} does not have window {args.window}. "
            f"Available: {', '.join(sorted(window_map_b))}"
        )

    folds_a = window_map_a[args.window]
    folds_b = window_map_b[args.window]
    common_folds = sorted(set(folds_a) & set(folds_b))
    if not common_folds:
        raise RuntimeError(
            f"No overlapping folds for window {args.window} between the two configs."
        )

    missing_a = sorted(set(folds_b) - set(folds_a))
    missing_b = sorted(set(folds_a) - set(folds_b))
    direction = metric_direction(args.metric)

    fold_rows = []
    concat_a = []
    concat_b = []
    for fold_idx in common_folds:
        df_a = folds_a[fold_idx]
        df_b = folds_b[fold_idx]
        score_a = metric_value(df_a, args.metric)
        score_b = metric_value(df_b, args.metric)
        raw_delta = score_a - score_b
        signed_delta = direction * raw_delta
        fold_rows.append((fold_idx, score_a, score_b, raw_delta, signed_delta))
        concat_a.append(df_a)
        concat_b.append(df_b)

    fold_deltas = np.array([row[4] for row in fold_rows], dtype=float)
    mean_delta, se_delta, ci_lo, ci_hi = t_interval(fold_deltas)

    pooled_a = metric_value(pd.concat(concat_a, ignore_index=True), args.metric)
    pooled_b = metric_value(pd.concat(concat_b, ignore_index=True), args.metric)
    pooled_raw_delta = pooled_a - pooled_b
    pooled_signed_delta = direction * pooled_raw_delta

    wins_a = int(np.sum(fold_deltas > 0))
    wins_b = int(np.sum(fold_deltas < 0))
    ties = int(np.sum(np.isclose(fold_deltas, 0.0)))

    better_note = "higher is better" if direction > 0 else "lower is better"
    favored_note = "positive signed delta favors A"

    print(f"Config A: {os.path.relpath(cfg_a_dir, results_root)}")
    print(f"Config B: {os.path.relpath(cfg_b_dir, results_root)}")
    print(f"Window:   {args.window}")
    print(f"Metric:   {args.metric} ({better_note}; {favored_note})")
    print(f"Folds:    {','.join(str(f) for f in common_folds)}")
    if missing_a:
        print(f"Missing from A: {','.join(str(f) for f in missing_a)}")
    if missing_b:
        print(f"Missing from B: {','.join(str(f) for f in missing_b)}")

    print("\nPooled OOF")
    print(f"  A={pooled_a:.6f}  B={pooled_b:.6f}  raw_delta={fmt_signed(pooled_raw_delta)}  signed_delta={fmt_signed(pooled_signed_delta)}")

    print("\nPaired Fold Deltas")
    print("  fold  score_A    score_B    raw_delta  signed_delta")
    for fold_idx, score_a, score_b, raw_delta, signed_delta in fold_rows:
        print(
            f"  {fold_idx:>4}  {score_a:>8.6f}  {score_b:>8.6f}  "
            f"{fmt_signed(raw_delta):>10}  {fmt_signed(signed_delta):>12}"
        )

    print("\nSummary")
    print(f"  mean_signed_delta={fmt_signed(mean_delta)}")
    print(f"  se={se_delta:.6f}")
    print(f"  95%_CI=[{ci_lo:+.6f}, {ci_hi:+.6f}]")
    print(f"  wins_A={wins_a} wins_B={wins_b} ties={ties}")


if __name__ == "__main__":
    main()
