#!/usr/bin/env python3
import argparse
import glob
import os
import re
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.metrics import cohen_kappa_score


OOF_RE = re.compile(r"oof_val_((?:T\d+)|(?:E\d+-\d+))\.csv$")
FOLD_RE = re.compile(r"fold(\d+)$")


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate per-fold OOF CSVs into a leaderboard.")
    p.add_argument(
        "results_root",
        help="Root containing config folders, or run folders that contain config folders.",
    )
    p.add_argument(
        "--expected-folds",
        type=int,
        default=None,
        help="Warn when a config does not contain exactly this many fold OOF files.",
    )
    return p.parse_args()


def discover_cfg_dirs(results_root: str):
    """Discover config directories under either root/config/fold* or root/run/config/fold* layouts."""
    pattern = os.path.join(results_root, "**", "fold*", "oof_val_*.csv")
    cfg_dirs = {
        os.path.dirname(os.path.dirname(path))
        for path in glob.glob(pattern, recursive=True)
    }
    return sorted(cfg_dirs)


def load_oof_csv_groups(cfg_dir: str):
    """Load per-fold OOF CSV groups for one config, separated by ensemble window tag."""
    paths = sorted(glob.glob(os.path.join(cfg_dir, "fold*", "oof_val_*.csv")))
    if not paths:
        return []

    entries_by_window = {}
    for path in paths:
        fold_name = os.path.basename(os.path.dirname(path))
        fold_match = FOLD_RE.search(fold_name)
        oof_match = OOF_RE.search(os.path.basename(path))
        if fold_match is None or oof_match is None:
            raise ValueError(f"Could not parse fold/window from path: {path}")
        window_tag = oof_match.group(1)
        entries_by_window.setdefault(window_tag, []).append((int(fold_match.group(1)), path))

    groups = []
    for window_tag, entries in sorted(entries_by_window.items()):
        paths_by_fold = {}
        for fold_idx, path in entries:
            paths_by_fold.setdefault(fold_idx, []).append(path)
        duplicate_folds = {
            fold_idx: fold_paths
            for fold_idx, fold_paths in paths_by_fold.items()
            if len(fold_paths) > 1
        }
        if duplicate_folds:
            raise ValueError(
                f"Found multiple OOF CSVs for the same fold/window in {cfg_dir} "
                f"({window_tag}): {duplicate_folds}"
            )

        entries.sort(key=lambda x: x[0])
        dfs = [pd.read_csv(path) for _, path in entries]
        fold_ids = [fold_idx for fold_idx, _ in entries]
        groups.append((window_tag, dfs, fold_ids))

    return groups


def infer_heads(df: pd.DataFrame, y_suffix: str = "_y"):
    """Infer head names from columns ending with y_suffix (default: *_y)."""
    heads = sorted({c[:-len(y_suffix)] for c in df.columns if c.endswith(y_suffix) and len(c) > len(y_suffix)})
    return heads


def per_head_cols(df: pd.DataFrame, head: str):
    """
    Identify columns for one head:
      - y:    head_y
      - pred: head_pred
      - p_k:  head_p1..head_pK  (strictly numeric suffix)
    """
    prefix = f"{head}_p"
    pcols = []
    for c in df.columns:
        if c.startswith(prefix):
            suffix = c[len(prefix):]
            if suffix.isdigit():
                pcols.append(c)
    pcols.sort(key=lambda x: int(x[len(prefix):]))

    return {
        "y": f"{head}_y",
        "pred": f"{head}_pred",
        "pcols": pcols,
    }


def label_values_from_pcols(pcols):
    """Recover the actual ordinal support values from columns like head_p1, head_p2, ..."""
    vals = []
    for c in pcols:
        m = re.search(r"_p(\d+)$", c)
        if m is None:
            raise ValueError(f"Could not infer label value from probability column: {c}")
        vals.append(int(m.group(1)))
    return np.array(vals, dtype=int)


def emd_one_sample(p: np.ndarray, y: int, label_values: np.ndarray) -> float:
    """
    EMD between predicted ordinal distribution p over label_values
    and a one-hot at y, using scipy.stats.wasserstein_distance.
    Normalized to [0,1] by dividing by the support span.
    """
    xs = np.asarray(label_values, dtype=float)
    K = xs.shape[0]
    if K <= 1:
        return 0.0

    s = p.sum()
    if s > 0:
        p = p / s

    v_weights = np.zeros(K, dtype=float)
    matches = np.where(xs == float(y))[0]
    if not matches.size:
        raise ValueError(f"Label {y} is outside the probability support {label_values.tolist()}")
    v_weights[int(matches[0])] = 1.0

    wd = wasserstein_distance(xs, xs, p, v_weights)
    span = float(xs[-1] - xs[0])
    return wd / span if span > 0 else 0.0


def oof_metrics_from_df(df: pd.DataFrame, heads=None):
    """
    Compute per-head OOF metrics from a concatenated OOF dataframe:
      - QWK via sklearn (quadratic)
      - EMD via scipy (normalized), class-balanced like train.py
      - averages across heads
    """
    if heads is None:
        heads = infer_heads(df)
    out = {}
    qwk_vals = []
    emd_vals = []

    if not heads:
        raise ValueError("No prediction heads inferred from OOF dataframe.")

    for h in heads:
        cols = per_head_cols(df, h)
        y_col = cols["y"]
        pred_col = cols["pred"]

        missing = [c for c in (y_col, pred_col) if c not in df.columns]
        if missing:
            raise ValueError(f"Head '{h}' is missing required columns: {missing}")
        if not cols["pcols"]:
            raise ValueError(f"Head '{h}' is missing probability columns.")

        y_raw = df[y_col].to_numpy()
        mask = np.isfinite(y_raw)
        if not mask.any():
            raise ValueError(f"Head '{h}' has no finite labels in column {y_col}.")
        y = y_raw[mask].astype(int)

        label_values = label_values_from_pcols(cols["pcols"])
        y_pred = np.asarray(df.loc[mask, pred_col], dtype=int)
        P = np.asarray(df.loc[mask, cols["pcols"]], dtype=float)

        q = cohen_kappa_score(y, y_pred, weights="quadratic")
        qwk_vals.append(q)

        sample_emds = np.array([
            emd_one_sample(P[i], int(y[i]), label_values=label_values)
            for i in range(len(y))
        ], dtype=float)
        class_emd_means = [
            float(sample_emds[y == int(label)].mean())
            for label in label_values
            if np.any(y == int(label))
        ]
        emd_mean = (
            float(np.nanmean(np.asarray(class_emd_means, dtype=float)))
            if class_emd_means
            else float("nan")
        )

        emd_vals.append(emd_mean)
        out[f"qwk_{h}"] = float(q)
        out[f"emd_{h}"] = emd_mean

    out["qwk_avg"] = float(np.mean(qwk_vals)) if qwk_vals else float("nan")
    out["macroEMD"] = float(np.nanmean(emd_vals)) if np.isfinite(np.array(emd_vals, float)).any() else float("nan")
    out["n"] = int(len(df))
    return out


def fold_ci_from_fold_oofs(dfs, heads=None):
    """
    Compute fold-level qwk_avg and macroEMD mean ± 95% CI
    using metrics computed on each fold's OOF separately.
    """
    vals_q, vals_e = [], []
    for df in dfs:
        m = oof_metrics_from_df(df, heads=heads)
        vals_q.append(m["qwk_avg"])
        vals_e.append(m["macroEMD"])

    vals_q = np.array(vals_q, float)
    vals_e = np.array(vals_e, float)

    def mean_ci(x):
        x = x[np.isfinite(x)]
        n = int(x.size)
        if n == 0:
            return float("nan"), float("nan")
        mu = float(x.mean())
        if n > 1:
            sd = float(x.std(ddof=1))
            ci = 1.96 * sd / sqrt(n)
        else:
            ci = 0.0
        return mu, ci

    q_mu, q_ci = mean_ci(vals_q)
    e_mu, e_ci = mean_ci(vals_e)
    return q_mu, q_ci, e_mu, e_ci


def main():
    args = parse_args()
    results_root = os.path.abspath(args.results_root)
    rows = []

    cfg_dirs = discover_cfg_dirs(results_root)

    for cfg_dir in cfg_dirs:
        cfg = os.path.relpath(cfg_dir, results_root)

        try:
            oof_groups = load_oof_csv_groups(cfg_dir)
        except ValueError as exc:
            print(f"[warn] skipping {cfg}: {exc}")
            continue

        for window_tag, fold_dfs, fold_ids in oof_groups:
            if args.expected_folds is not None and len(fold_dfs) != args.expected_folds:
                print(
                    f"[warn] {cfg} ({window_tag}) has {len(fold_dfs)} fold OOF files; "
                    f"expected {args.expected_folds}. folds={fold_ids}"
                )

            all_oof = pd.concat(fold_dfs, ignore_index=True)

            out_oof = os.path.join(cfg_dir, f"oof_all_{window_tag}.csv")
            all_oof.to_csv(out_oof, index=False)

            heads = infer_heads(all_oof)
            try:
                oof_m = oof_metrics_from_df(all_oof, heads=heads)
                q_mu, q_ci, e_mu, e_ci = fold_ci_from_fold_oofs(fold_dfs, heads=heads)
            except ValueError as exc:
                print(f"[warn] skipping {cfg} ({window_tag}): {exc}")
                continue

            row = {
                "config": cfg,
                "ensemble_window": window_tag,
                "n_total": oof_m["n"],
                "n_folds": len(fold_dfs),
                "folds_present": ",".join(str(i) for i in fold_ids),
                "oof_qwk_avg": oof_m["qwk_avg"],
                "oof_macroEMD": oof_m["macroEMD"],
                "fold_qwk_avg_mean": q_mu,
                "fold_qwk_avg_ci95": q_ci,
                "fold_macroEMD_mean": e_mu,
                "fold_macroEMD_ci95": e_ci,
            }

            for h in heads:
                row[f"oof_qwk_{h}"] = oof_m.get(f"qwk_{h}", float("nan"))
                row[f"oof_emd_{h}"] = oof_m.get(f"emd_{h}", float("nan"))

            rows.append(row)

    if rows:
        tbl = pd.DataFrame(rows).sort_values("oof_qwk_avg", ascending=False)
        leaderboard = os.path.join(results_root, "_oof_leaderboard.csv")
        tbl.to_csv(leaderboard, index=False)
        print(tbl.to_string(index=False))
        print(f"\nWrote leaderboard -> {leaderboard}")
    else:
        print(f"No OOF files found under {results_root}")


if __name__ == "__main__":
    main()
