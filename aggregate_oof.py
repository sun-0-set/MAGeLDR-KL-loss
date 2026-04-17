#!/usr/bin/env python3
import os
import glob
import re
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import wasserstein_distance


def load_oof_csvs(cfg_dir: str):
    """Load all per-fold OOF CSVs for a config."""
    paths = sorted(glob.glob(os.path.join(cfg_dir, "fold*", "oof_val_T*.csv")))
    if not paths:
        return [], None, []
    # Infer T from the first file name (e.g. oof_val_T7.csv / oof_val_T10.csv)
    m = re.search(r"oof_val_T(\d+)\.csv$", os.path.basename(paths[0]))
    T = m.group(1) if m else "X"
    dfs = [pd.read_csv(p) for p in paths if os.path.exists(p)]
    return dfs, T, paths

def infer_heads(df: pd.DataFrame, y_suffix: str = "_y"):
    """Infer head names from columns ending with y_suffix (default: *_y)."""
    heads = sorted({c[:-len(y_suffix)] for c in df.columns if c.endswith(y_suffix) and len(c) > len(y_suffix)})
    return heads



def per_head_cols(df: pd.DataFrame, head: str):
    """
    Identify columns for one head:
      - y:    head_y
      - pred: head_pred
      - exp:  head_exp
      - p_k:  head_p1..head_pK  (strictly numeric suffix)
    """
    prefix = f"{head}_p"
    pcols = []
    for c in df.columns:
        if c.startswith(prefix):
            suffix = c[len(prefix):]
            if suffix.isdigit():        # only keep p1, p2, ...; skip 'pred'
                pcols.append(c)
    pcols.sort(key=lambda x: int(x[len(prefix):]))  # order by k

    return {
        "y": f"{head}_y",
        "pred": f"{head}_pred",
        "exp": f"{head}_exp",
        "pcols": pcols,
        "K": len(pcols),
    }


def emd_one_sample(p: np.ndarray, y: int, label_base: int = 1, normalize: bool = True) -> float:
    """
    EMD between predicted ordinal distribution p over {label_base..label_base+K-1}
    and a one-hot at y, using scipy.stats.wasserstein_distance.
    Normalized to [0,1] by dividing by (K-1).
    """
    K = p.shape[0]
    if K <= 1:
        return 0.0
    # ensure valid prob dist
    s = p.sum()
    if s > 0:
        p = p / s
    xs = np.arange(label_base, label_base + K, dtype=float)

    v_weights = np.zeros(K, dtype=float)
    idx = y - label_base
    if 0 <= idx < K:
        v_weights[idx] = 1.0

    wd = wasserstein_distance(xs, xs, p, v_weights)
    return wd / (K - 1) if normalize else wd


def oof_metrics_from_df(df: pd.DataFrame, heads=None):
    """
    Compute per-head OOF metrics from a concatenated OOF dataframe:
      - QWK via sklearn (quadratic)
      - EMD via scipy (normalized)
      - averages across heads
    """
    if heads is None:
        heads = infer_heads(df)
    out = {}
    qwk_vals = []
    emd_vals = []

    for h in heads:
        cols = per_head_cols(df, h)
        y_col = cols["y"]
        pred_col = cols["pred"]
        exp_col = cols["exp"]


        if y_col not in df.columns:
            # Head missing in this config
            out[f"qwk_{h}"] = float("nan")
            out[f"emd_{h}"] = float("nan")
            continue

        y_raw = df[y_col].to_numpy()
        mask = np.isfinite(y_raw)
        y = y_raw[mask].astype(int)

        # Infer whether labels are 0-based or 1-based (used for argmax + EMD grid).
        base = 0 if int(np.nanmin(y)) == 0 else 1

        if pred_col in df.columns:
            y_pred = np.asarray(df.loc[mask, pred_col], dtype=int)
        elif exp_col in df.columns:
            y_pred = np.rint(np.asarray(df.loc[mask, exp_col], dtype=float)).astype(int)
        elif cols["pcols"]:
            P_pred = np.asarray(df.loc[mask, cols["pcols"]], dtype=float)
            y_pred = P_pred.argmax(axis=1) + base
        else:
            out[f"qwk_{h}"] = float("nan")
            out[f"emd_{h}"] = float("nan")
            continue

        # Quadratic weighted kappa from sklearn
        q = cohen_kappa_score(y, y_pred, weights="quadratic")
        qwk_vals.append(q)

        # EMD: average per-sample normalized distance if probs available
        if cols["pcols"]:
            P = np.asarray(df.loc[mask, cols["pcols"]], dtype=float)  # (N, K)
            K = len(cols["pcols"])
            emds = [emd_one_sample(P[i], int(y[i]), label_base=base, normalize=True)
                    for i in range(len(y))]
            emd_mean = float(np.mean(emds))
        else:
            emd_mean = float("nan")

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
    K = len(dfs)

    def mean_ci(x):
        x = x[np.isfinite(x)]
        n = int(x.size)
        if n == 0:
            return float('nan'), float('nan')
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
    results_root = "../results/k6_promptcv_dress/lambda3-.5_C.1"
    rows = []

    cfg_dirs = [
        d for d in glob.glob(os.path.join(results_root, "*"))
        if os.path.isdir(d)
    ]

    for cfg_dir in sorted(cfg_dirs):
        cfg = os.path.basename(cfg_dir)

        fold_dfs, T, paths = load_oof_csvs(cfg_dir)
        if not fold_dfs:
            continue

        # Concatenate all folds for global OOF
        all_oof = pd.concat(fold_dfs, ignore_index=True)

        # Save concatenated OOF for convenience
        t_suffix = T if T is not None else "X"
        out_oof = os.path.join(cfg_dir, f"oof_all_T{t_suffix}.csv")
        all_oof.to_csv(out_oof, index=False)

        # OOF metrics on full concatenation
        heads = infer_heads(all_oof)

        # OOF metrics on full concatenation
        oof_m = oof_metrics_from_df(all_oof, heads=heads)

        # Fold mean ± CI (per-fold OOF metrics)
        q_mu, q_ci, e_mu, e_ci = fold_ci_from_fold_oofs(fold_dfs, heads=heads)

        row = {
            "config": cfg,
            "n_total": oof_m["n"],
            "oof_qwk_avg": oof_m["qwk_avg"],
            "oof_macroEMD": oof_m["macroEMD"],
            "fold_qwk_avg_mean": q_mu,
            "fold_qwk_avg_ci95": q_ci,
            "fold_macroEMD_mean": e_mu,
            "fold_macroEMD_ci95": e_ci,
        }

        # Per-head metrics (dynamic columns, consistent across configs if schema is consistent)
        for h in heads:
            row[f"oof_qwk_{h}"] = oof_m.get(f"qwk_{h}", float("nan"))
            row[f"oof_emd_{h}"] = oof_m.get(f"emd_{h}", float("nan"))

        rows.append(row)

    if rows:
        tbl = pd.DataFrame(rows).sort_values("oof_qwk_avg", ascending=False)
        leaderboard = os.path.join(results_root, "_oof_leaderboard.csv")
        tbl.to_csv(leaderboard, index=False)
        print(tbl.to_string(index=False))
        print(f"\nWrote leaderboard → {leaderboard}")
    else:
        print("No OOF files found under results/*/fold*/oof_val_T*.csv")


if __name__ == "__main__":
    main()
