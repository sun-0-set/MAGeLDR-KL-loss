import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

TARGET_COLS = ["content", "organization", "language"]

class DRESSDataset(Dataset):
    def __init__(self, tsv_path: str, tokenizer_name: str = "tasksource/deberta-small-long-nli",
                 max_length: int = 1600, tokenizer=None):
        self.df = pd.read_csv(tsv_path, sep="\t").dropna(subset=["prompt", "essay"] + TARGET_COLS).reset_index(drop=True)
        self.df["row_id"] = self.df.index.astype(int)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_length = max_length

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        enc = self.tokenizer(
            str(row["prompt"]), str(row["essay"]),
            truncation=True, padding=False,
            max_length=self.max_length, return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        y = row[TARGET_COLS].values.astype("int64")
        item["labels"] = torch.tensor(y) 
        item["ids"] = torch.tensor(row["row_id"]) 
        # TODO load from existing pt
        return item

    def get_all_targets_tensor(self):
        return torch.tensor(
            self.df[TARGET_COLS].to_numpy(copy=True),
            dtype=torch.long
        )