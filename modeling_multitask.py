from transformers import AutoConfig, AutoModel
import torch
import torch.nn as nn

class MultiHeadDeberta(nn.Module):
    def __init__(
        self,
        model_name: str = "tasksource/deberta-small-long-nli",
        num_heads: int = 3,
        num_classes: int = 5,
        *,
        freeze_encoder: bool = False,
        dropout: float = 0.0,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        torch_dtype: torch.dtype | None = None,
        enable_grad_ckpt: bool = False,
        **kwargs,
    ):
        super().__init__()
        # Allow alias kwargs for convenience
        if "gradient_checkpointing" in kwargs:
            enable_grad_ckpt = bool(kwargs.pop("gradient_checkpointing"))
        if "grad_ckpt" in kwargs:
            enable_grad_ckpt = bool(kwargs.pop("grad_ckpt"))
        # Load config/model strictly from local if requested
        self.config = AutoConfig.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config=self.config,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        if enable_grad_ckpt:
            # Disable KV cache (required with checkpointing)
            if hasattr(self.encoder.config, "use_cache"):
                self.encoder.config.use_cache = False
            # Prefer NON-REENTRANT checkpointing to avoid “backward through graph a second time”
            if hasattr(self.encoder, "gradient_checkpointing_enable"):
                try:
                    self.encoder.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                except TypeError:
                    # Older Transformers: kwarg not supported — fall back to default
                    self.encoder.gradient_checkpointing_enable()


        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        hidden = self.config.hidden_size
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.heads = nn.ModuleList([nn.Linear(hidden, num_classes) for _ in range(num_heads)])

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # CLS pooling
        pooled = out.last_hidden_state[:, 0]              # (B, hidden)
        pooled = self.dropout(pooled)
        logits = torch.stack([head(pooled) for head in self.heads], dim=1)  # (B, 3, K)
        return {"logits": logits}
