#!/usr/bin/env python3
import argparse
import os
import math
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from aggregate_oof import discover_cfg_dirs, load_oof_csv_groups, oof_metrics_from_df


def ref_matches(relpath: str, ref: str) -> bool:
    ref = ref.rstrip(os.sep)
    return relpath == ref or os.path.basename(relpath) == ref or relpath.endswith(os.sep + ref)


def seed_from_relpath(relpath: str) -> str:
    first = relpath.split(os.sep, 1)[0]
    return first if first.startswith("seed") else "."


def find_cfgs(results_root: str, ref: str):
    out = {}
    for cfg_dir in discover_cfg_dirs(results_root):
        rel = os.path.relpath(cfg_dir, results_root)
        if ref_matches(rel, ref):
            out[seed_from_relpath(rel)] = cfg_dir
    if not out:
        raise FileNotFoundError(f"No config directories matching {ref!r} under {results_root}")
    return out


def load_window_metric(cfg_dir: str, window: str, metric: str):
    groups = load_oof_csv_groups(cfg_dir)
    for window_tag, dfs, fold_ids in groups:
        if window_tag == window:
            df = pd.concat(dfs, ignore_index=True)
            m = oof_metrics_from_df(df)
            if metric not in m:
                raise KeyError(f"{metric!r} not found for {cfg_dir} {window}; keys={sorted(m)}")
            return float(m[metric])
    choices = ", ".join(g[0] for g in groups)
    raise KeyError(f"{cfg_dir} has no window {window}; choices: {choices}")


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    mu = float(x.mean())
    if n == 1:
        return mu, 0.0, mu, mu
    se = float(x.std(ddof=1) / math.sqrt(n))
    hw = float(student_t.ppf(0.975, n - 1) * se)
    return mu, se, mu - hw, mu + hw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root")
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--epochs", default="1-18")
    ap.add_argument("--metric", default="qwk_avg")
    ap.add_argument("--out-prefix", default="peak_epoch_by_seed")
    args = ap.parse_args()

    root = os.path.abspath(args.results_root)
    e0, e1 = map(int, args.epochs.split("-"))
    epochs = list(range(e0, e1 + 1))

    all_epoch_rows = []
    peak_rows = []

    for cfg_ref in args.configs:
        cfgs = find_cfgs(root, cfg_ref)
        for seed, cfg_dir in sorted(cfgs.items()):
            vals = []
            for e in epochs:
                window = f"E{e}-{e}"
                val = load_window_metric(cfg_dir, window, args.metric)
                vals.append(val)
                all_epoch_rows.append({
                    "seed": seed,
                    "config": cfg_ref,
                    "epoch": e,
                    args.metric: val,
                })

            vals_np = np.asarray(vals, dtype=float)
            peak_i = int(np.nanargmax(vals_np))
            peak_epoch = epochs[peak_i]
            peak_val = float(vals_np[peak_i])
            final_val = float(vals_np[-1])
            first_val = float(vals_np[0])

            peak_rows.append({
                "seed": seed,
                "config": cfg_ref,
                "peak_epoch": peak_epoch,
                f"peak_{args.metric}": peak_val,
                f"final_epoch_{epochs[-1]}_{args.metric}": final_val,
                f"first_epoch_{epochs[0]}_{args.metric}": first_val,
                "drop_peak_to_final": peak_val - final_val,
                "gain_first_to_peak": peak_val - first_val,
            })

    epoch_df = pd.DataFrame(all_epoch_rows)
    peak_df = pd.DataFrame(peak_rows)

    epoch_path = f"{args.out_prefix}_curves.csv"
    peak_path = f"{args.out_prefix}_peaks.csv"
    summary_path = f"{args.out_prefix}_summary.csv"

    epoch_df.to_csv(epoch_path, index=False)
    peak_df.to_csv(peak_path, index=False)

    summary_rows = []
    for cfg, g in peak_df.groupby("config"):
        peak_epochs = g["peak_epoch"].to_numpy(float)
        drops = g["drop_peak_to_final"].to_numpy(float)
        peaks = g[f"peak_{args.metric}"].to_numpy(float)
        finals = g[f"final_epoch_{epochs[-1]}_{args.metric}"].to_numpy(float)

        peak_mu, peak_se, peak_lo, peak_hi = mean_ci(peaks)
        final_mu, final_se, final_lo, final_hi = mean_ci(finals)
        drop_mu, drop_se, drop_lo, drop_hi = mean_ci(drops)

        mode_epoch = int(pd.Series(g["peak_epoch"]).mode().iloc[0])

        summary_rows.append({
            "config": cfg,
            "n_seeds": len(g),
            "mean_peak_epoch": float(np.mean(peak_epochs)),
            "median_peak_epoch": float(np.median(peak_epochs)),
            "mode_peak_epoch": mode_epoch,
            "min_peak_epoch": int(np.min(peak_epochs)),
            "max_peak_epoch": int(np.max(peak_epochs)),
            f"mean_peak_{args.metric}": peak_mu,
            f"peak_{args.metric}_ci95_lo": peak_lo,
            f"peak_{args.metric}_ci95_hi": peak_hi,
            f"mean_final_{args.metric}": final_mu,
            f"final_{args.metric}_ci95_lo": final_lo,
            f"final_{args.metric}_ci95_hi": final_hi,
            "mean_drop_peak_to_final": drop_mu,
            "drop_ci95_lo": drop_lo,
            "drop_ci95_hi": drop_hi,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)

    print(f"Wrote {epoch_path}")
    print(f"Wrote {peak_path}")
    print(f"Wrote {summary_path}")

    print("\nPeaks by seed/config:")
    print(peak_df.sort_values(["config", "seed"]).to_string(index=False))

    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
