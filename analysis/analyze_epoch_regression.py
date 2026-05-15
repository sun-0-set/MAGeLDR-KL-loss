#!/usr/bin/env python3
import argparse
import os
import math
import numpy as np
import pandas as pd
from scipy.stats import t as student_t, ttest_1samp, wilcoxon

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


def get_window_group(cfg_dir, tag):
    groups = load_oof_csv_groups(cfg_dir)
    for window_tag, dfs, fold_ids in groups:
        if window_tag == tag:
            return dfs, fold_ids
    choices = ", ".join(g[0] for g in groups)
    raise KeyError(f"{cfg_dir} has no window {tag}; choices: {choices}")


def pooled_metric_for_epoch(cfg_dir, epoch, metric):
    tag = f"E{epoch}-{epoch}"
    dfs, fold_ids = get_window_group(cfg_dir, tag)
    all_df = pd.concat(dfs, ignore_index=True)
    m = oof_metrics_from_df(all_df)
    if metric not in m:
        raise KeyError(f"{metric!r} not in metrics for {cfg_dir} {tag}; keys={sorted(m)}")
    return float(m[metric])


def summarize_paired(name, values):
    values = np.asarray(values, dtype=float)
    mu, se, lo, hi = mean_ci(values)
    print(
        f"{name}: mean={mu:+.6f} se={se:.6f} "
        f"95%_CI=[{lo:+.6f}, {hi:+.6f}] n={values.size}"
    )
    if values.size > 1:
        print(f"  t_p={ttest_1samp(values, 0.0).pvalue:.6g}")
        try:
            print(f"  wilcoxon_p={wilcoxon(values).pvalue:.6g}")
        except ValueError:
            pass
        print(f"  positive={int((values > 0).sum())}/{values.size}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root")
    ap.add_argument("--jager", default="jager-leader-bare-joint-lambda0-3-C-5e-2-E4-8")
    ap.add_argument("--ce", default="ce-label-smoothing-0.05-E4-8")
    ap.add_argument("--epochs", default="1-18")
    ap.add_argument("--late-start", type=int, default=8)
    ap.add_argument("--tail-start", type=int, default=16)
    ap.add_argument("--metric", default="qwk_avg")
    ap.add_argument("--out", default="epoch_regression_analysis")
    args = ap.parse_args()

    root = os.path.abspath(args.results_root)
    a, b = map(int, args.epochs.split("-"))
    epochs = np.arange(a, b + 1, dtype=int)

    j_cfgs = find_cfgs(root, args.jager)
    c_cfgs = find_cfgs(root, args.ce)
    seeds = sorted(set(j_cfgs) & set(c_cfgs))
    if not seeds:
        raise RuntimeError("No common seeds")

    rows = []
    for seed in seeds:
        for method, cfgs in [("jager", j_cfgs), ("ce", c_cfgs)]:
            cfg_dir = cfgs[seed]
            for e in epochs:
                rows.append({
                    "seed": seed,
                    "method": method,
                    "epoch": int(e),
                    args.metric: pooled_metric_for_epoch(cfg_dir, int(e), args.metric),
                })

    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    curve_path = os.path.join(args.out, "epoch_curves.csv")
    df.to_csv(curve_path, index=False)

    # Per-epoch JAGeR - CE deltas.
    wide = df.pivot_table(index=["seed", "epoch"], columns="method", values=args.metric).reset_index()
    wide["jager_minus_ce"] = wide["jager"] - wide["ce"]

    per_epoch = []
    for e, g in wide.groupby("epoch"):
        vals = g["jager_minus_ce"].to_numpy(float)
        mu, se, lo, hi = mean_ci(vals)
        p = ttest_1samp(vals, 0.0).pvalue if len(vals) > 1 else np.nan
        per_epoch.append({
            "epoch": int(e),
            "delta_mean": mu,
            "delta_se": se,
            "delta_ci95_lo": lo,
            "delta_ci95_hi": hi,
            "t_p": p,
            "positive_seeds": int((vals > 0).sum()),
            "n_seeds": int(vals.size),
        })
    per_epoch_df = pd.DataFrame(per_epoch)
    per_epoch_path = os.path.join(args.out, "per_epoch_jager_minus_ce.csv")
    per_epoch_df.to_csv(per_epoch_path, index=False)

    # Curve-shape diagnostics per seed/method.
    diag_rows = []
    for (seed, method), g in df.groupby(["seed", "method"]):
        g = g.sort_values("epoch")
        es = g["epoch"].to_numpy(float)
        ys = g[args.metric].to_numpy(float)

        peak_i = int(np.nanargmax(ys))
        peak_epoch = int(es[peak_i])
        peak_val = float(ys[peak_i])
        final_val = float(ys[-1])

        late_mask = es >= args.late_start
        late_slope = float(np.polyfit(es[late_mask], ys[late_mask], 1)[0])

        tail_mask = es >= args.tail_start
        tail_mean = float(np.nanmean(ys[tail_mask]))

        diag_rows.append({
            "seed": seed,
            "method": method,
            "peak_epoch": peak_epoch,
            "peak_qwk": peak_val,
            "epoch_final": int(es[-1]),
            "final_qwk": final_val,
            "tail_mean_qwk": tail_mean,
            "drop_peak_to_final": peak_val - final_val,
            "drop_peak_to_tail": peak_val - tail_mean,
            f"slope_E{args.late_start}_{int(es[-1])}": late_slope,
        })

    diag = pd.DataFrame(diag_rows)
    diag_path = os.path.join(args.out, "curve_diagnostics_by_seed.csv")
    diag.to_csv(diag_path, index=False)

    dw = diag.pivot(index="seed", columns="method")
    slope_col = f"slope_E{args.late_start}_{b}"

    paired = pd.DataFrame({
        "seed": seeds,
        "jager_peak_epoch": [dw.loc[s, ("peak_epoch", "jager")] for s in seeds],
        "ce_peak_epoch": [dw.loc[s, ("peak_epoch", "ce")] for s in seeds],
        "peak_epoch_jager_minus_ce": [
            dw.loc[s, ("peak_epoch", "jager")] - dw.loc[s, ("peak_epoch", "ce")]
            for s in seeds
        ],
        "jager_late_slope": [dw.loc[s, (slope_col, "jager")] for s in seeds],
        "ce_late_slope": [dw.loc[s, (slope_col, "ce")] for s in seeds],
        "late_slope_jager_minus_ce": [
            dw.loc[s, (slope_col, "jager")] - dw.loc[s, (slope_col, "ce")]
            for s in seeds
        ],
        "jager_drop_peak_to_final": [dw.loc[s, ("drop_peak_to_final", "jager")] for s in seeds],
        "ce_drop_peak_to_final": [dw.loc[s, ("drop_peak_to_final", "ce")] for s in seeds],
        "ce_minus_jager_drop_peak_to_final": [
            dw.loc[s, ("drop_peak_to_final", "ce")] - dw.loc[s, ("drop_peak_to_final", "jager")]
            for s in seeds
        ],
        "jager_drop_peak_to_tail": [dw.loc[s, ("drop_peak_to_tail", "jager")] for s in seeds],
        "ce_drop_peak_to_tail": [dw.loc[s, ("drop_peak_to_tail", "ce")] for s in seeds],
        "ce_minus_jager_drop_peak_to_tail": [
            dw.loc[s, ("drop_peak_to_tail", "ce")] - dw.loc[s, ("drop_peak_to_tail", "jager")]
            for s in seeds
        ],
    })

    paired_path = os.path.join(args.out, "paired_regression_diagnostics.csv")
    paired.to_csv(paired_path, index=False)

    print(f"Wrote {curve_path}")
    print(f"Wrote {per_epoch_path}")
    print(f"Wrote {diag_path}")
    print(f"Wrote {paired_path}")

    print("\nPer-epoch JAGeR - CE:")
    print(per_epoch_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nRegression diagnostics, positive means supports 'CE regresses earlier/more':")
    summarize_paired(
        "peak_epoch_jager_minus_ce",
        paired["peak_epoch_jager_minus_ce"].to_numpy(float),
    )
    summarize_paired(
        "late_slope_jager_minus_ce",
        paired["late_slope_jager_minus_ce"].to_numpy(float),
    )
    summarize_paired(
        "ce_minus_jager_drop_peak_to_final",
        paired["ce_minus_jager_drop_peak_to_final"].to_numpy(float),
    )
    summarize_paired(
        "ce_minus_jager_drop_peak_to_tail",
        paired["ce_minus_jager_drop_peak_to_tail"].to_numpy(float),
    )


if __name__ == "__main__":
    main()
