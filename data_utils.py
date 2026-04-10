import fcntl
import hashlib
import json
import os
import shutil
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

DEFAULT_PROMPT_COL = "prompt"
DEFAULT_ESSAY_COL = "essay"
TOKEN_CACHE_VERSION = 1
TOKEN_CACHE_BATCH_SIZE = 256


def _json_dump(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _source_signature(path: str) -> dict:
    st = os.stat(path)
    return {
        "path": os.path.realpath(path),
        "size": int(st.st_size),
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
    }


def _normalize_existing_ref(ref) -> str:
    text = str(ref)
    expanded = os.path.expanduser(text)
    if os.path.exists(expanded):
        return os.path.realpath(expanded)
    return text


def _tokenizer_signature(tokenizer_name: str, tokenizer) -> dict:
    tokenizer_name_ref = _normalize_existing_ref(tokenizer_name)
    name_or_path_ref = _normalize_existing_ref(getattr(tokenizer, "name_or_path", tokenizer_name))
    sig = {
        "tokenizer_name": tokenizer_name_ref,
        "name_or_path": name_or_path_ref,
        "class": tokenizer.__class__.__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "model_input_names": list(getattr(tokenizer, "model_input_names", [])),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
    }
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        sig["vocab_size"] = int(vocab_size)

    tok_path = name_or_path_ref
    if os.path.isdir(tok_path):
        file_sigs = []
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
            "spm.model",
            "sentencepiece.bpe.model",
        ):
            fp = os.path.join(tok_path, name)
            if os.path.exists(fp):
                st = os.stat(fp)
                file_sigs.append({
                    "name": name,
                    "size": int(st.st_size),
                    "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
                })
        if file_sigs:
            sig["tokenizer_files"] = file_sigs
    return sig


def _cache_spec(
    *,
    path: str,
    tokenizer_name: str,
    tokenizer,
    max_length: int,
    prompt_col: str,
    essay_col: str,
) -> tuple[str, dict]:
    spec = {
        "version": TOKEN_CACHE_VERSION,
        "source": _source_signature(path),
        "tokenizer": _tokenizer_signature(tokenizer_name, tokenizer),
        "max_length": int(max_length),
        "prompt_col": str(prompt_col),
        "essay_col": str(essay_col),
    }
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24], spec


def _default_token_cache_root(path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(path)), ".essay_token_cache")


def _cache_metadata_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "metadata.json")


def _load_clean_frame(path: str, prompt_col: str, essay_col: str):
    sep = "\t" if path.lower().endswith(".tsv") else ","
    df = pd.read_csv(path, sep=sep)

    if prompt_col not in df.columns or essay_col not in df.columns:
        raise ValueError(f"Expected columns '{prompt_col}' and '{essay_col}' in {path}")

    label_cols = [
        c for c in df.columns
        if c not in (prompt_col, essay_col)
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not label_cols:
        raise ValueError(
            f"No numeric label columns found in {path}. "
            f"All columns except '{prompt_col}'/'{essay_col}' should be label columns."
        )

    df = df.dropna(subset=[prompt_col, essay_col] + label_cols).reset_index(drop=True)
    df["row_id"] = df.index.astype(np.int64)
    return df, label_cols


def _save_array(path: str, arr: np.ndarray) -> None:
    np.save(path, arr, allow_pickle=False)


def _build_cache_into_dir(
    cache_dir: str,
    *,
    df: pd.DataFrame,
    label_cols: list[str],
    tokenizer,
    max_length: int,
    prompt_col: str,
    essay_col: str,
    spec: dict,
) -> None:
    prompts = df[prompt_col].astype(str).tolist()
    essays = df[essay_col].astype(str).tolist()
    labels = df[label_cols].to_numpy(dtype=np.int64, copy=True)
    row_ids = df["row_id"].to_numpy(dtype=np.int64, copy=True)

    feature_chunks = {}
    feature_keys = None
    lengths = []
    total = len(row_ids)

    for start in range(0, total, TOKEN_CACHE_BATCH_SIZE):
        stop = min(total, start + TOKEN_CACHE_BATCH_SIZE)
        enc = tokenizer(
            prompts[start:stop],
            essays[start:stop],
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        if "input_ids" not in enc:
            raise RuntimeError("Tokenizer output is missing input_ids; cannot build token cache.")
        if "attention_mask" not in enc:
            enc["attention_mask"] = [[1] * len(seq) for seq in enc["input_ids"]]

        batch_keys = sorted(enc.keys())
        if feature_keys is None:
            feature_keys = batch_keys
        elif batch_keys != feature_keys:
            raise RuntimeError(
                f"Inconsistent tokenizer keys while building cache: {batch_keys} vs {feature_keys}"
            )

        batch_lengths = np.asarray([len(seq) for seq in enc["input_ids"]], dtype=np.int64)
        lengths.append(batch_lengths)

        for key in feature_keys:
            seqs = enc[key]
            if len(seqs) != (stop - start):
                raise RuntimeError(
                    f"Tokenizer key {key!r} returned {len(seqs)} sequences for batch size {stop - start}."
                )
            for seq, exp_len in zip(seqs, batch_lengths.tolist()):
                if len(seq) != exp_len:
                    raise RuntimeError(
                        f"Tokenizer key {key!r} produced length {len(seq)} but input_ids length is {exp_len}."
                    )
            flat = (
                np.concatenate([np.asarray(seq, dtype=np.int32) for seq in seqs], axis=0)
                if seqs else np.empty((0,), dtype=np.int32)
            )
            feature_chunks.setdefault(key, []).append(flat)

    if feature_keys is None:
        feature_keys = ["input_ids", "attention_mask"]

    lengths_arr = np.concatenate(lengths, axis=0) if lengths else np.empty((0,), dtype=np.int64)
    offsets = np.empty((len(lengths_arr) + 1,), dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths_arr, out=offsets[1:])

    os.makedirs(cache_dir, exist_ok=True)
    _save_array(os.path.join(cache_dir, "offsets.npy"), offsets)
    _save_array(os.path.join(cache_dir, "labels.npy"), labels)
    _save_array(os.path.join(cache_dir, "row_ids.npy"), row_ids)
    for key in feature_keys:
        parts = feature_chunks.get(key, [])
        flat = np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=np.int32)
        _save_array(os.path.join(cache_dir, f"{key}.npy"), flat)

    _json_dump(
        _cache_metadata_path(cache_dir),
        {
            "version": TOKEN_CACHE_VERSION,
            "cache_spec": spec,
            "feature_keys": feature_keys,
            "target_cols": list(label_cols),
            "num_rows": int(len(row_ids)),
            "max_length": int(max_length),
            "prompt_col": prompt_col,
            "essay_col": essay_col,
        },
    )


def _ensure_token_cache(
    *,
    path: str,
    tokenizer_name: str,
    tokenizer,
    max_length: int,
    prompt_col: str,
    essay_col: str,
    token_cache_dir: Optional[str],
):
    key, spec = _cache_spec(
        path=path,
        tokenizer_name=tokenizer_name,
        tokenizer=tokenizer,
        max_length=max_length,
        prompt_col=prompt_col,
        essay_col=essay_col,
    )
    root = os.path.abspath(token_cache_dir or _default_token_cache_root(path))
    os.makedirs(root, exist_ok=True)
    cache_dir = os.path.join(root, key)
    meta_path = _cache_metadata_path(cache_dir)
    lock_path = cache_dir + ".lock"
    built_here = False

    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            if not os.path.exists(meta_path):
                tmp_dir = cache_dir + f".tmp.{os.getpid()}"
                if os.path.isdir(tmp_dir):
                    shutil.rmtree(tmp_dir)
                if os.path.isdir(cache_dir):
                    shutil.rmtree(cache_dir)
                df, label_cols = _load_clean_frame(path, prompt_col, essay_col)
                try:
                    _build_cache_into_dir(
                        tmp_dir,
                        df=df,
                        label_cols=label_cols,
                        tokenizer=tokenizer,
                        max_length=max_length,
                        prompt_col=prompt_col,
                        essay_col=essay_col,
                        spec=spec,
                    )
                    os.rename(tmp_dir, cache_dir)
                    built_here = True
                finally:
                    if os.path.isdir(tmp_dir):
                        shutil.rmtree(tmp_dir)

            with open(meta_path) as f:
                metadata = json.load(f)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    return cache_dir, metadata, built_here


class EssayDataset(Dataset):
    """
    Generic essay dataset with an on-disk token cache.

    - Expects a CSV/TSV with at least ['prompt', 'essay'].
    - All other numeric columns are treated as label/response columns.
    - The first access builds a reusable cache keyed by data file, tokenizer, and max_length.
    """

    def __init__(
        self,
        path: str,
        tokenizer_name: str = "microsoft/deberta-v3-large",
        max_length: int = 1600,
        tokenizer=None,
        prompt_col: str = DEFAULT_PROMPT_COL,
        essay_col: str = DEFAULT_ESSAY_COL,
        token_cache_dir: Optional[str] = None,
    ):
        tokenizer_obj = tokenizer or AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        cache_dir, metadata, cache_built = _ensure_token_cache(
            path=path,
            tokenizer_name=tokenizer_name,
            tokenizer=tokenizer_obj,
            max_length=max_length,
            prompt_col=prompt_col,
            essay_col=essay_col,
            token_cache_dir=token_cache_dir,
        )

        self.path = os.path.abspath(path)
        self.prompt_col = prompt_col
        self.essay_col = essay_col
        self.label_cols = list(metadata["target_cols"])
        self.max_length = int(max_length)
        self.cache_dir = cache_dir
        self.cache_built = bool(cache_built)
        self.feature_keys = list(metadata["feature_keys"])
        self.offsets = np.load(os.path.join(cache_dir, "offsets.npy"), mmap_mode="r")
        self.labels = np.load(os.path.join(cache_dir, "labels.npy"), mmap_mode="r")
        self.row_ids = np.load(os.path.join(cache_dir, "row_ids.npy"), mmap_mode="r")
        self.features = {
            key: np.load(os.path.join(cache_dir, f"{key}.npy"), mmap_mode="r")
            for key in self.feature_keys
        }

        if int(self.labels.shape[0]) != int(self.row_ids.shape[0]):
            raise RuntimeError(
                f"Token cache is inconsistent: labels={self.labels.shape[0]} row_ids={self.row_ids.shape[0]}"
            )
        if int(self.offsets.shape[0]) != int(self.row_ids.shape[0]) + 1:
            raise RuntimeError(
                f"Token cache is inconsistent: offsets={self.offsets.shape[0]} rows={self.row_ids.shape[0]}"
            )

    def __len__(self):
        return int(self.row_ids.shape[0])

    def __getitem__(self, i: int):
        start = int(self.offsets[i])
        stop = int(self.offsets[i + 1])
        item = {
            key: torch.tensor(self.features[key][start:stop], dtype=torch.long)
            for key in self.feature_keys
        }
        if "attention_mask" not in item:
            item["attention_mask"] = torch.ones((stop - start,), dtype=torch.long)
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        item["ids"] = torch.tensor(int(self.row_ids[i]), dtype=torch.long)
        return item

    def get_all_targets_tensor(self) -> torch.Tensor:
        return torch.tensor(self.labels, dtype=torch.long)

    @property
    def target_cols(self) -> list[str]:
        return list(self.label_cols)


# Backwards-compatible aliases for old code paths
TARGET_COLS = None


class DRESSDataset(EssayDataset):
    """Alias kept so old imports still work."""

    def __init__(
        self,
        tsv_path: str,
        tokenizer_name: str = "microsoft/deberta-v3-large",
        max_length: int = 1600,
        tokenizer=None,
        token_cache_dir: Optional[str] = None,
    ):
        super().__init__(
            path=tsv_path,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            tokenizer=tokenizer,
            prompt_col=DEFAULT_PROMPT_COL,
            essay_col=DEFAULT_ESSAY_COL,
            token_cache_dir=token_cache_dir,
        )
