#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.aggregate_oof import load_oof_csv_groups, oof_metrics_from_df
from analysis.make_oof_windows_from_npz import write_oof
from loss import JAGeRLoss, MultiHeadCELoss


def smoke_losses() -> None:
    torch.manual_seed(0)
    labels = torch.tensor([[1, 2], [2, 3], [3, 4], [4, 1]], dtype=torch.long)
    ids = torch.arange(labels.shape[0])

    ce_logits = torch.randn(4, 2, 4, requires_grad=True)
    ce_loss = MultiHeadCELoss(labels, K=4, level_offset=1)(ce_logits, ids)
    assert torch.isfinite(ce_loss)
    ce_loss.backward()

    jager_logits = torch.randn(4, 2, 4, dtype=torch.float64, requires_grad=True)
    jager = JAGeRLoss(
        labels,
        K=4,
        def_batch_size=2,
        joint=True,
        mixture=False,
        conf_gating=False,
        reassignment=False,
        level_offset=1,
    )
    jager_loss = jager(jager_logits, ids, update_state=False)
    assert torch.isfinite(jager_loss)
    jager_loss.backward()


def smoke_oof_windows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fold_dir = Path(tmp) / "run" / "config" / "fold0"
        fold_dir.mkdir(parents=True)
        classes = np.array([1, 2, 3, 4], dtype=np.int64)
        probs = np.zeros((2, 2, 3, 4), dtype=np.float64)
        probs[..., :] = 0.05
        probs[0, 0, :, [0, 1, 2]] = 0.85
        probs[0, 1, :, [1, 2, 3]] = 0.85
        probs[1, 0, :, [0, 1, 2]] = 0.75
        probs[1, 1, :, [1, 2, 3]] = 0.75
        probs /= probs.sum(axis=-1, keepdims=True)

        np.savez(
            fold_dir / "epoch_val_preds.npz",
            epochs=np.array([1, 2], dtype=np.int64),
            ids=np.array([10, 11, 12], dtype=np.int64),
            y=np.array([[1, 2, 3], [2, 3, 4]], dtype=np.int64),
            p=probs,
            classes=classes,
            head_names=np.array(["content", "language"]),
        )

        write_oof(str(fold_dir / "epoch_val_preds.npz"), [(1, 2)])
        groups = load_oof_csv_groups(str(fold_dir.parent))
        assert len(groups) == 1
        tag, dfs, fold_ids = groups[0]
        assert tag == "E1-2"
        assert fold_ids == [0]
        metrics = oof_metrics_from_df(pd.concat(dfs, ignore_index=True))
        assert metrics["n"] == 3
        assert np.isfinite(metrics["qwk_avg"])
        assert np.isfinite(metrics["macroEMD"])


def main() -> None:
    smoke_losses()
    smoke_oof_windows()
    print("smoke_core ok")


if __name__ == "__main__":
    main()
