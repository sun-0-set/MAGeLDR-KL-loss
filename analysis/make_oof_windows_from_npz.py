#!/usr/bin/env python3
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd


def parse_windows(s: str):
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"E?(\d+)-(\d+)", part)
        if not m:
            raise ValueError(f"Bad window: {part!r}; expected like 8-12 or E8-12")
        a, b = int(m.group(1)), int(m.group(2))
        if b < a:
            raise ValueError(f"Bad window {part!r}: end < start")
        out.append((a, b))
    return out


def as_str_list(x):
    out = []
    for v in np.asarray(x).tolist():
        if isinstance(v, bytes):
            v = v.decode()
        out.append(str(v))
    return out


def write_oof(npz_path: str, windows, *, overwrite: bool = False):
    fold_dir = os.path.dirname(npz_path)
    fold_name = os.path.basename(os.path.abspath(fold_dir))

    z = np.load(npz_path, allow_pickle=True)

    epochs = np.asarray(z["epochs"]).astype(int)      # (E,)
    ids = np.asarray(z["ids"]).astype(np.int64)       # (N,)
    y = np.asarray(z["y"])                            # (H, N)
    p = np.asarray(z["p"], dtype=np.float64)          # (E, H, N, C)
    classes = np.asarray(z["classes"]).astype(int)    # (C,)
    head_names = as_str_list(z["head_names"])         # H names

    if p.ndim != 4:
        raise ValueError(f"{npz_path}: expected p shape (E,H,N,C), got {p.shape}")
    if y.ndim != 2:
        raise ValueError(f"{npz_path}: expected y shape (H,N), got {y.shape}")

    E, H, N, C = p.shape
    if epochs.shape != (E,):
        raise ValueError(f"{npz_path}: epochs shape {epochs.shape} incompatible with p {p.shape}")
    if y.shape != (H, N):
        raise ValueError(f"{npz_path}: y shape {y.shape} incompatible with p {p.shape}")
    if ids.shape != (N,):
        raise ValueError(f"{npz_path}: ids shape {ids.shape} incompatible with p {p.shape}")
    if classes.shape != (C,):
        raise ValueError(f"{npz_path}: classes shape {classes.shape} incompatible with p {p.shape}")
    if len(head_names) != H:
        raise ValueError(f"{npz_path}: {len(head_names)} head_names but H={H}")

    for a, b in windows:
        tag = f"E{a}-{b}"
        out_path = os.path.join(fold_dir, f"oof_val_{tag}.csv")
        if os.path.exists(out_path) and not overwrite:
            print(f"skip existing {out_path}")
            continue

        mask = (epochs >= a) & (epochs <= b)
        selected = epochs[mask].tolist()
        expected = list(range(a, b + 1))
        if selected != expected:
            raise RuntimeError(
                f"{npz_path}: requested {tag}, but selected epochs={selected}; "
                f"expected contiguous {expected}; available={epochs.tolist()}"
            )

        avg_p = p[mask].mean(axis=0)  # (H, N, C)

        # renormalise defensively after averaging
        denom = avg_p.sum(axis=-1, keepdims=True)
        avg_p = np.divide(avg_p, denom, out=np.zeros_like(avg_p), where=denom != 0)

        rows = []
        for i in range(N):
            row = {"id": int(ids[i]), "fold": fold_name}
            for h, name in enumerate(head_names):
                probs = avg_p[h, i]  # (C,)
                pred = int(classes[int(np.argmax(probs))])
                exp = float((probs * classes).sum())
                row[f"{name}_y"] = int(y[h, i])
                row[f"{name}_exp"] = exp
                row[f"{name}_pred"] = pred
                for j, c in enumerate(classes):
                    row[f"{name}_p{int(c)}"] = float(probs[j])
            rows.append(row)

        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"wrote {out_path} from epochs={selected}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root")
    ap.add_argument(
        "--windows",
        required=True,
        help="Comma-separated windows, e.g. 1-5,4-8,8-12,8-18",
    )
    ap.add_argument(
        "--config-regex",
        default=None,
        help="Only process config dirs whose relative path matches this regex.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.results_root)
    windows = parse_windows(args.windows)
    cfg_re = re.compile(args.config_regex) if args.config_regex else None

    paths = sorted(glob.glob(os.path.join(root, "**", "fold*", "epoch_val_preds.npz"), recursive=True))
    if not paths:
        raise SystemExit(f"No epoch_val_preds.npz found under {root}")

    n = 0
    for npz_path in paths:
        cfg_dir = os.path.dirname(os.path.dirname(npz_path))
        rel_cfg = os.path.relpath(cfg_dir, root)
        if cfg_re is not None and not cfg_re.search(rel_cfg):
            continue
        write_oof(npz_path, windows, overwrite=args.overwrite)
        n += 1

    print(f"processed {n} epoch_val_preds.npz files")


if __name__ == "__main__":
    main()
