import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

DEFAULT_PROMPT_COL = "prompt"
DEFAULT_ESSAY_COL = "essay"


class EssayDataset(Dataset):
    """
    Generic essay dataset:

    - Expects a CSV/TSV with at least ['prompt', 'essay'].
    - All other *numeric* columns are treated as label/response columns.
      (So keep ids/folds non-numeric if you don't want them as targets.)
    - Labels are assumed to be ordinal integers (e.g. 1..K).
    """
    def __init__(
        self,
        path: str,
        tokenizer_name: str = "microsoft/deberta-v3-large",
        max_length: int = 1600,
        tokenizer=None,
        prompt_col: str = DEFAULT_PROMPT_COL,
        essay_col: str = DEFAULT_ESSAY_COL,
    ):
        sep = "\t" if path.lower().endswith(".tsv") else ","
        df = pd.read_csv(path, sep=sep)

        if prompt_col not in df.columns or essay_col not in df.columns:
            raise ValueError(f"Expected columns '{prompt_col}' and '{essay_col}' in {path}")

        # Treat all other numeric columns as label columns.
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
        df["row_id"] = df.index.astype(int)

        self.df = df
        self.prompt_col = prompt_col
        self.essay_col = essay_col
        self.label_cols = label_cols
        self.max_length = max_length
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        enc = self.tokenizer(
            str(row[self.prompt_col]),
            str(row[self.essay_col]),
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}

        y = row[self.label_cols].to_numpy(dtype="int64")
        item["labels"] = torch.tensor(y, dtype=torch.long)
        item["ids"] = torch.tensor(row["row_id"], dtype=torch.long)
        return item

    def get_all_targets_tensor(self) -> torch.Tensor:
        return torch.tensor(
            self.df[self.label_cols].to_numpy(copy=True),
            dtype=torch.long,
        )

    @property
    def target_cols(self) -> list[str]:
        return list(self.label_cols)


# Backwards-compatible aliases for old code paths
TARGET_COLS = None


class DRESSDataset(EssayDataset):
    """Alias kept so old imports still work."""
    def __init__(self, tsv_path: str,
                 tokenizer_name: str = "microsoft/deberta-v3-large",
                 max_length: int = 1600,
                 tokenizer=None):
        super().__init__(
            path=tsv_path,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            tokenizer=tokenizer,
            prompt_col=DEFAULT_PROMPT_COL,
            essay_col=DEFAULT_ESSAY_COL,
        )
