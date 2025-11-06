# train.py
import math, os, random, argparse, csv, json
import contextlib
import torch
from torch import nn, amp
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
try:
    from torch.amp import GradScaler as AmpGradScaler
except Exception:
    AmpGradScaler = None
torch.set_float32_matmul_precision("medium")

from datetime import timedelta
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerFast
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, cohen_kappa_score, recall_score
import numpy as np
from modeling_multitask import MultiHeadDeberta
from data_utils import DRESSDataset, TARGET_COLS
from loss import JAGeRLoss, MultiHeadCELoss

try:
    import wandb
    _WANDB_OK = True
except Exception:
    wandb = None
    _WANDB_OK = False
    

def _gather_stats_from_buffer(buf: "torch.Tensor", idx: list[int], per_head: bool):
    import torch
    x = buf.detach()
    if x.is_cuda:
        x = x.cpu()
    sel = x[idx]
    if per_head:
        if sel.dim() == 1:
            sel = sel[:, None]
        means = sel.float().mean(dim=0)
        mins  = sel.float().min(dim=0).values
        maxs  = sel.float().max(dim=0).values
        return dict(mean=means.numpy().tolist(),
                    min=mins.numpy().tolist(),
                    max=maxs.numpy().tolist())
    else:
        return dict(mean=float(sel.float().mean().item()),
                    min=float(sel.float().min().item()),
                    max=float(sel.float().max().item()))



def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_main_process():
    return (not is_dist()) or dist.get_rank() == 0

def setup_distributed():
    # Initialize DDP from torchrun env if present
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank != -1 and not is_dist():
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=1800))
        torch.cuda.set_device(local_rank)
    return local_rank

def parse_args():
    p = argparse.ArgumentParser()

    # --- Data / general -------------------------------------------------
    p.add_argument("--tsv", default="../data/DREsS/DREsS_New_cleaned.tsv")
    p.add_argument("--split_file", type=str, default="splits/dress_seed42.json",
                   help="JSON with {train,val,test} id lists; if missing, do random split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--stratify_joint", type=int, default=1,
                   help="Use joint stratification over (content, organization, language) if no --split_file.")
    p.add_argument("--limit_train", type=int, default=0, help="use only N training examples (0=all)")
    p.add_argument("--limit_val", type=int, default=0, help="use only N validation examples (0=all)")

    # --- Model / Tokenizer ---------------------------------------------
    p.add_argument("--model_name", default="tasksource/deberta-small-long-nli")
    p.add_argument("--hf_offline", action="store_true",
                   help="Force offline mode and local files only for HF")
    p.add_argument("--use_fast_tokenizer", type=int, default=1,
                   help="1=fast tokenizer, 0=slow (SentencePiece)")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--pad_to_multiple_of", type=int, default=8, help="dynamic padding grid")

    # --- DataLoader / perf ------------------------------------------------
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--prefetch_factor", type=int, default=4,
                   help="Batches to prefetch per worker (effective only when num_workers>0)")

    # --- Training / optimization --------------------------------------
    p.add_argument("--batch_size", type=int, default=2)         # per-device
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    p.add_argument("--grad_ckpt", action="store_true",
                   help="Enable gradient checkpointing for the encoder")
    p.add_argument("--freeze_encoder", action="store_true", help="train heads only (faster on CPU)")
    p.add_argument("--unfreeze_at_epoch", type=int, default=-1, help="-1 = never unfreeze in this run")
    p.add_argument("--log_every", type=int, default=50, help="Log training stats every N optimizer steps")
    p.add_argument("--save_model", action="store_true", help="Save best.pt (OFF by default; turn on for finalist runs).")

    # --- Loss family / hyperparameters --------------------------------
    p.add_argument("--loss", choices=["jager", "ce"], default="jager",
                   help="jager = JAGeR, ce = cross-entropy")
    p.add_argument("--joint", type=int, default=1,
                   help="JAGeR: 1 = use joint K^H space; 0 = per-head")
    p.add_argument("--mixture", type=int, default=1,
                   help="JAGeR: enable mixture prior / ρ estimation")
    p.add_argument("--conf_gating", type=int, default=1,
                   help="JAGeR: enable confidence-gated ρ update")
    p.add_argument("--reassignment", type=int, default=1,
                   help="JAGeR: enable y_pred_max reassignment term")
    p.add_argument("--lambda0", type=float, default=1.0, help="initial λ (JAGeR)")
    p.add_argument("--alpha", type=float, default=2.0, help="α (JAGeR)")
    p.add_argument("--C", type=float, default=1e-1, help="margin C (JAGeR)")
    p.add_argument("--ce_label_smoothing", type=float, default=0.0, help="label smoothing for CE baseline")
    p.add_argument("--log_epoch_stats", action="store_true",
                   help="Write per-epoch min/mean/max for λ (and ρ if available) to save_dir/epoch_stats.csv")

    # --- Evaluation / run modes ---------------------------------------
    p.add_argument("--eval_test", action="store_true",
                   help="After training, reload best.pt and evaluate on test")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training: load best.pt from --save_dir and run eval (val/test)")

    # --- W&B / logging -------------------------------------------------
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", type=str, default="jager", help="W&B project name")
    p.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team) or None")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online","offline","disabled"], help="W&B mode")

    return p.parse_args()


def evaluate(model, dl, loss_fn, device, args=None):
    model.eval()
    use_amp = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    tot_loss, n = 0.0, 0
    all_true = [[], [], []]  # per-head true labels (1..K)
    all_pred = [[], [], []]  # per-head preds (1..K)
    K = 5
    sum_emd_by_class = torch.zeros(3, K, dtype=torch.float64)
    count_by_class   = torch.zeros(3, K, dtype=torch.long)

    with torch.inference_mode():
        for batch in dl:
            pin = (device.type == "cuda")
            ids = batch["ids"].to(device, non_blocking=pin)
            input_ids = batch["input_ids"].to(device, non_blocking=pin)
            attention_mask = batch["attention_mask"].to(device, non_blocking=pin)
            labels = batch["labels"]  # CPU (B,3), values 1..K

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(input_ids=input_ids, attention_mask=attention_mask)

            y_pred = out["logits"].to(torch.float64)  # (B, 3, 5)

            # don't mutate JAGeR state on validation
            try:
                loss = loss_fn(y_pred, ids, update_state=False)
            except TypeError:
                loss = loss_fn(y_pred, ids)

            bsz = input_ids.size(0)
            tot_loss += loss.item() * bsz
            n += bsz

            # plain argmax decoding – no prior, no λ scaling
            pred_idx = y_pred.argmax(-1).cpu()  # (B,3)

            # --- probabilistic EMD/W1 per-sample (uses probabilities) ---
            probs  = torch.softmax(y_pred, dim=-1)                         # (B,3,K) on device
            levels = torch.arange(1, K+1, device=probs.device, dtype=probs.dtype)  # 1..K
            # absolute distances to the TRUE label (broadcast)
            abs_dist = (levels.view(1,1,-1) - labels.to(probs).unsqueeze(-1).float()).abs()  # (B,3,K)
            emd_batch = (probs * abs_dist).sum(dim=-1).cpu()                # (B,3) on CPU
            # aggregate per true-class for macro over classes
            one_hot = torch.nn.functional.one_hot(labels - 1, num_classes=K).to(dtype=torch.long)  # (B,3,K) on CPU
            sum_emd_by_class += (emd_batch.unsqueeze(-1) * one_hot).sum(dim=0).to(torch.float64)   # (3,K)
            count_by_class   += one_hot.sum(dim=0)

            # collect for CM/QWK/MAE
            for h in range(3):
                all_true[h].extend(labels[:, h].tolist())
                all_pred[h].extend((pred_idx[:, h] + 1).tolist())

    # ---- Confusion matrices + QWK + EMD ----
    cms, qwks, emds, tail_recalls = [], [], [], []
    label_list = list(range(1, K + 1))
    for h in range(3):
        y_t = np.array(all_true[h], dtype=int)
        y_p = np.array(all_pred[h], dtype=int)

        cms.append(confusion_matrix(y_t, y_p, labels=label_list))
        qwks.append(float(cohen_kappa_score(y_t, y_p, weights="quadratic")))

        # per-class EMD averages (ignore empty classes via nan)
        sums   = sum_emd_by_class[h].numpy()
        counts = count_by_class[h].numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            per_class_emd = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        emds.append(float(np.nanmean(per_class_emd)))

        # ---- Tail Recall@0 (strict exact-match), macro over minority classes ----
        if args is not None and hasattr(args, "minority_classes") and args.minority_classes and len(args.minority_classes) == 3:
            minority = list(args.minority_classes[h])
        else:
            # fallback: pick 2 least frequent classes in this split
            counts_h = np.bincount(np.clip(y_t, 1, K), minlength=K+1)[1:]
            minority = list(np.argsort(counts_h)[:2] + 1)  # to 1..K
        # per-class recall (exact match); zero_division=0 handles empty support gracefully
        per_class_rec = recall_score(y_t, y_p, labels=label_list, average=None, zero_division=0)
        mr = [per_class_rec[c-1] for c in minority if 1 <= c <= K]
        tail_recalls.append(float(np.mean(mr)) if len(mr) else float("nan"))

    macro_emd = float(sum(emds) / len(emds))  # mean over 3 heads
    tail_recall_avg = float(sum(tail_recalls) / len(tail_recalls)) if len(tail_recalls) else float("nan")

    return (
        tot_loss / max(n, 1),
        cms,
        qwks,
        emds,
        macro_emd,
        tail_recalls,
        tail_recall_avg
    )



def joint_stratified_indices(ds, val_frac: float, seed: int, min_count: int = 2):
    """
    Returns train_idx, val_idx using *joint* stratification over (content, organization, language).
    Graceful fallback to (content, organization) then (content) then random if some strata are too small.
    Assumes labels in ds.df[TARGET_COLS] are 1..K.
    """
    K = 5
    y = ds.df[TARGET_COLS].to_numpy(copy=False)  # shape (N,3), values 1..K
    N = y.shape[0]
    idx = np.arange(N)

    def try_split(strata):
        # Ensure every class has at least 2 samples for stratify
        _, counts = np.unique(strata, return_counts=True)
        if (counts < min_count).any():
            return None
        return train_test_split(
            idx, test_size=val_frac, random_state=seed, stratify=strata
        )

    # 3D joint: (c-1)*K^2 + (o-1)*K + (l-1)
    joint_3 = (y[:,0]-1)*K*K + (y[:,1]-1)*K + (y[:,2]-1)
    out = try_split(joint_3)
    if out is not None:
        return out

    # 2D fallback: (c-1)*K + (o-1)
    joint_2 = (y[:,0]-1)*K + (y[:,1]-1)
    out = try_split(joint_2)
    if out is not None:
        return out

    # 1D fallback: content only
    out = try_split(y[:,0]-1)
    if out is not None:
        return out

    # Final fallback: random split
    train_idx, val_idx = train_test_split(idx, test_size=val_frac, random_state=seed, shuffle=True, stratify=None)
    return train_idx, val_idx


def _maybe_init_wandb(args):
    """Initialize W&B if requested and available. Returns a run or None."""
    if not getattr(args, "wandb", False) or args.wandb_mode == "disabled":
        return None
    if not _WANDB_OK:
        print("[W&B] wandb not installed; proceeding without logging.")
        return None
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        mode=args.wandb_mode,
        name=f"{args.loss}-j{args.joint}-m{args.mixture}-cg{args.conf_gating}-ra{args.reassignment}"
             f"-bs{args.batch_size}-ga{args.grad_accum}",
        tags=[args.loss,
              f"joint={args.joint}", f"mixture={args.mixture}",
              f"conf_gating={args.conf_gating}", f"reassign={args.reassignment}"],
        config=vars(args),
    )
    return run


def _log_cms_as_wandb_images(cms, split: str, head_names=("content", "organization", "language"), step: int | None = None):
    """
    Render a list[3] of confusion matrices as small heatmaps and log to W&B.
    Safe no-op if W&B is unavailable.
    """
    if wandb is None or wandb.run is None:
        return
    import matplotlib.pyplot as plt
    if not cms:
        return
    K = int(np.asarray(cms[0]).shape[0])
    vmax = max(int(np.asarray(cm).max()) for cm in cms)
    for h, name in enumerate(head_names):
        cm = np.asarray(cms[h], dtype=np.int32)
        fig = plt.figure(figsize=(3.2, 3.2), dpi=150)
        ax = fig.add_subplot(111)
        ax.imshow(cm, vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{split.upper()} CM – {name}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(list(range(1, K+1))); ax.set_yticklabels(list(range(1, K+1)))
        for i in range(K):
            for j in range(K):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
        fig.tight_layout()
        wandb.log({f"{split}/cm_{name}": wandb.Image(fig)}, step=step)
        plt.close(fig)


def main():
    args = parse_args()

    if os.path.sep in args.model_name or args.model_name.startswith("."):
        args.model_name = os.path.abspath(os.path.expanduser(args.model_name))

    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # avoid noisy warnings
    
    run_wandb = _maybe_init_wandb(args) if is_main_process() else None
    
    np.random.seed(args.seed + (dist.get_rank() if is_dist() else 0))
    
    os.makedirs(args.save_dir, exist_ok=True)
    local_rank = setup_distributed()
    set_seed(args.seed + (dist.get_rank() if is_dist() else 0))

    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() and local_rank != -1 \
             else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    if device.type == "cuda":
        # A100: enable TF32; bf16 autocast below
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Build dataset 
    tok_path = os.path.join(args.model_name, "tokenizer.json")
    if os.path.isdir(args.model_name) and os.path.exists(tok_path):
        # Bypass AutoTokenizer entirely > no warning
        tok = PreTrainedTokenizerFast(tokenizer_file=tok_path)
        # Merge special tokens map manually (pad/cls/sep/etc.)
        spec_path = os.path.join(args.model_name, "special_tokens_map.json")
        added = 0
        if os.path.exists(spec_path):
            import json as _json
            with open(spec_path) as _f:
                spec = _json.load(_f)
            to_add = {k: v for k, v in spec.items() if isinstance(v, str)}
            if to_add:
                added = tok.add_special_tokens(to_add)
        if tok.pad_token is None:
            if tok.sep_token is not None:
                tok.pad_token = tok.sep_token
            elif tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            else:
                added += tok.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tok = AutoTokenizer.from_pretrained(
            args.model_name,
            use_fast=bool(args.use_fast_tokenizer),
            local_files_only=bool(args.hf_offline),
            trust_remote_code=False,
        )
    if is_main_process(): print(f"[tok] class={tok.__class__.__name__} fast={getattr(tok,'is_fast',None)} src={'tokenizer.json' if os.path.exists(tok_path) else args.model_name}")

    # Build dataset
    ds = DRESSDataset(args.tsv, tokenizer_name=args.model_name, max_length=args.max_length, tokenizer=tok)
    # Use frozen split if present
    if args.split_file and os.path.exists(args.split_file):
        import json
        with open(args.split_file) as f: split = json.load(f)
        train_ids = split["train"]; val_ids = split["val"]; test_ids = split["test"]
        ds_train = Subset(ds, train_ids)
        ds_val   = Subset(ds, val_ids)
        ds_test  = Subset(ds, test_ids)
    else:
        # fallback: build joint-stratified split over (content, organization, language)
        if int(args.stratify_joint) == 1:
            tr_idx, va_idx = joint_stratified_indices(ds, val_frac=args.val_frac, seed=args.seed)
            ds_train = Subset(ds, tr_idx)
            ds_val   = Subset(ds, va_idx)
            train_ids = list(map(int, tr_idx))
            val_ids   = list(map(int, va_idx))
        else:
            ds_train, ds_val = ds.random_split(val_frac=args.val_frac, seed=args.seed)
            # random_split returns Subsets; grab their indices
            train_ids = list(map(int, getattr(ds_train, "indices", range(len(ds_train)))))
            val_ids   = list(map(int, getattr(ds_val,   "indices", range(len(ds_val)))))
        ds_test = None

    eval_test_flag = bool(args.eval_test and ds_test is not None)

    # Optional subsetting for faster local iterations
    if args.limit_train and args.limit_train < len(ds_train):
        ds_train = Subset(ds_train, list(range(args.limit_train)))
    if args.limit_val and args.limit_val < len(ds_val):
        ds_val = Subset(ds_val, list(range(args.limit_val)))

    # (ds already has `tok`; keep a no-op assignment for clarity)
    if hasattr(ds, "tokenizer"): ds.tokenizer = tok

    # Dynamic padding collator (pads each batch to the batch max)
    collate = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=(args.pad_to_multiple_of if args.pad_to_multiple_of > 0 else None))
    # safety: ensure pad token is set (should already be via special_tokens_map)
    if tok.pad_token is None:
        # prefer BERT-like sep as pad, else fall back to adding a PAD token
        if tok.sep_token is not None:
            tok.pad_token = tok.sep_token
        elif tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({'pad_token': '[PAD]'})

    # ---- Samplers (DDP) ----
    train_sampler = None
    if is_dist():
        # DistributedSampler handles shuffling
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(ds_train, shuffle=True, drop_last=False)

    pin = (device.type == "cuda")
    
    _pf = (args.prefetch_factor if args.num_workers > 0 else None)
    dl_train = DataLoader(
        ds_train, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers, persistent_workers=(args.num_workers > 0),
        pin_memory=pin, collate_fn=collate, prefetch_factor=_pf,
    )
    
    dl_val = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=(args.num_workers > 0),
        pin_memory=pin, collate_fn=collate, prefetch_factor=_pf,
    )

    # ---- Determine minority classes from TRAIN distribution (per head) ----
    def _compute_minority_classes(dl, K=5, threshold=0.10, bottom_m=2):
        counts = np.zeros((3, K), dtype=np.int64)
        total = np.zeros(3, dtype=np.int64)
        for batch in dl:
            lab = batch["labels"].cpu().numpy()  # (B,3), values 1..K
            for h in range(3):
                vh = lab[:, h]
                total[h] += vh.size
                binc = np.bincount(np.clip(vh, 1, K), minlength=K+1)[1:]
                counts[h] += binc
        minority = []
        for h in range(3):
            prev = counts[h] / max(total[h], 1)
            idx = [c+1 for c in range(K) if prev[c] < threshold]
            if not idx:
                idx = list(np.argsort(prev)[:bottom_m] + 1)
            minority.append(idx)
        return minority

    minority_classes = _compute_minority_classes(dl_train, K=5)
    setattr(args, "minority_classes", minority_classes)
    if is_main_process():
        print(f"[info] minority classes per head (from TRAIN): {minority_classes}")
    
    model = MultiHeadDeberta(
        args.model_name,
        num_heads=3,
        num_classes=5,
        freeze_encoder=args.freeze_encoder,
        dropout=args.dropout,
        local_files_only=bool(args.hf_offline),
        trust_remote_code=False,
        torch_dtype=None, 
        enable_grad_ckpt=bool(getattr(args, "grad_ckpt", False)), 
    ).to(device)

    try:
        if 'added' in locals() and added:
            (model.module if is_dist() else model).resize_token_embeddings(len(tok))
    except Exception:
        pass

    # Print gradient checkpointing status (safe on DDP/non-DDP)
    def unwrap(m): return m.module if hasattr(m, "module") else m
    if is_main_process():
        base = unwrap(model)
        enc = base.encoder
        flag = getattr(enc, "is_gradient_checkpointing", None)
        if flag is None:
            flag = getattr(getattr(enc, "config", object()), "gradient_checkpointing", False)
        print(f">> Grad checkpointing: {flag}")

    
    # Wrap with DDP if launched via torchrun
    if is_dist():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False
        )
    
    # Optional head-only warmup
    if args.freeze_encoder:
        enc = model.module.encoder if is_dist() else model.encoder
        for p in enc.parameters():
            p.requires_grad = False
        if is_main_process():
            print(">> Encoder frozen (training heads only).")
        

    # Stateful loss (init ONCE with full Y)
    Y_all = ds.get_all_targets_tensor().to(device)
    if args.loss == "jager":
        loss_fn = JAGeRLoss(
            Y=Y_all, K=5,
            joint=bool(args.joint),
            mixture=bool(args.mixture),
            conf_gating=bool(args.conf_gating),
            reassignment=bool(args.reassignment),
            level_offset=1,
            softplus=True,
            λ0=args.lambda0,
            α=args.alpha,
            C=args.C,
            log_to_wandb=bool(getattr(args, "wandb", False)),
            def_batch_size=args.batch_size,
            steps_per_epoch=len(dl_train)  # per-rank micro-batch steps/epoch (DDP-safe)
        ).to(device)
    else:
        loss_fn = MultiHeadCELoss(Y=Y_all, K=5, label_smoothing=args.ce_label_smoothing).to(device)


    # ---- Static buffer stats (λ, ρ) captured once per run/split ----
    idx_all = sorted(set(train_ids) | set(val_ids) | (set(test_ids) if eval_test_flag else set()))
    views = {
        "train": train_ids,
        "val":   val_ids,
        **({"test": test_ids} if eval_test_flag else {}),
        "all":   idx_all,
    }
    # Pick up buffers from the loss object (not all losses expose these)
    λ = getattr(loss_fn, "λ", None)
    ρ = getattr(loss_fn, "ρ", None)
    static_stats = {}
    # λ and ρ stats: treat as per-head if tensor rank ≥ 2
    for view_name, idx in views.items():
        if not idx:
            continue
        prefix = f"static/{view_name}/"
        if isinstance(λ, torch.Tensor):
            per_head = λ.dim() >= 2
            static_stats[prefix + "lambda"] = _gather_stats_from_buffer(λ, idx, per_head=per_head)
        if isinstance(ρ, torch.Tensor):
            per_head = ρ.dim() >= 2
            static_stats[prefix + "rho"] = _gather_stats_from_buffer(ρ, idx, per_head=per_head)
    if is_main_process() and run_wandb is not None:
        # flatten for W&B
        flat = {}
        for k, v in static_stats.items():
            if isinstance(v, dict) and isinstance(v.get("mean"), list):
                for h, m in enumerate(v["mean"]): flat[f"{k}/mean/h{h}"] = m
                for h, m in enumerate(v.get("min", [])): flat[f"{k}/min/h{h}"] = m
                for h, m in enumerate(v.get("max", [])): flat[f"{k}/max/h{h}"] = m
            else:
                flat[f"{k}/mean"] = v["mean"]; flat[f"{k}/min"] = v["min"]; flat[f"{k}/max"] = v["max"]
        wandb.log(flat, step=0)



    # -------- Eval-only fast path (no training) --------
    if args.eval_only:
        import json
        best_path = os.path.join(args.save_dir, "best.pt")
        assert os.path.exists(best_path), f"Missing {best_path}"
        ckpt = torch.load(best_path, map_location="cpu")
        (model.module if is_dist() else model).load_state_dict(ckpt["model"])

        # Build test loader if we have a split
        test_loader = None
        if ds_test is not None:
            test_loader = DataLoader(
                ds_test, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, persistent_workers=(args.num_workers > 0),
                pin_memory=pin, collate_fn=collate, prefetch_factor=_pf
            )

        # Evaluate (read-only)
        val_loss, vcms, vqwks, v_emds, v_macro_emd, v_tails, v_tails_avg = evaluate(
            model.module if is_dist() else model,
            dl_val, loss_fn, device, args=args
        )
        v_avg_qwk = float(sum(vqwks)/len(vqwks)) if len(vqwks) else float("nan")
        payload = {
            "val": {
                "loss": val_loss,
                "qwk": vqwks,
                "qwk_average": v_avg_qwk,
                "cm": [cm.tolist() for cm in vcms],
                "emd": v_emds,
                "macroEMD": v_macro_emd,
                "tail_recall0": v_tails,
                "tail_recall0_average": v_tails_avg,
            }
        }
        if args.eval_test and test_loader is not None:
            test_loss, tcms, tqwks, t_emds, t_macro_emd, t_tails, t_tails_avg = evaluate(
                model.module if is_dist() else model,
                test_loader, loss_fn, device, args=args
            )
            t_avg_qwk = float(sum(tqwks)/len(tqwks)) if len(tqwks) else float("nan")
            payload["test"] = {
                "loss": test_loss,
                "qwk": tqwks,
                "qwk_average": t_avg_qwk,
                "cm": [cm.tolist() for cm in tcms],
                "emd": t_emds,
                "macroEMD": t_macro_emd,
                "tail_recall0": t_tails,
                "tail_recall0_average": t_tails_avg,
            }

        payload.setdefault("extra", {})["static_stats"] = static_stats
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[EVAL-ONLY] wrote {os.path.join(args.save_dir, 'metrics.json')}")
        return


    optim = torch.optim.AdamW(model.parameters(), fused=(device.type == "cuda"), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler: 10% warmup
    total_steps = math.ceil(len(dl_train) / args.grad_accum) * args.epochs
    warmup = max(1, int(0.1 * total_steps))
    sched = get_linear_schedule_with_warmup(optim, warmup, total_steps)

    # AMP dtype: prefer bf16 on A100
    use_amp   = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = None
    if AmpGradScaler is not None:
        scaler = AmpGradScaler(enabled=use_amp and (amp_dtype is torch.float16))

    best_qwk = float("-inf")
    improved = False
    global_step = 0
    for epoch in range(1, args.epochs+1):
        # Ensure different shuffles across epochs in DDP
        if is_dist() and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        # Optional unfreeze at a chosen epoch (for GPU runs)
        if args.unfreeze_at_epoch == epoch:
            enc = model.module.encoder if is_dist() else model.encoder
            for p in enc.parameters():
                p.requires_grad = True
            if is_main_process():
                print(f">> Unfroze encoder at epoch {epoch}.")
        model.train()
        running, step_in_accum = 0.0, 0
        for step, batch in enumerate(dl_train, start=1):
            ids = batch["ids"].to(device, non_blocking=pin)
            input_ids = batch["input_ids"].to(device, non_blocking=pin)
            attention_mask = batch["attention_mask"].to(device, non_blocking=pin)

            # Decide sync scope for DDP: skip gradient all-reduce on non-boundary steps
            next_accum   = step_in_accum + 1
            ddp          = (is_dist() and hasattr(model, "no_sync"))
            sync_needed  = (next_accum % args.grad_accum == 0)
            sync_ctx     = contextlib.nullcontext() if (not ddp or sync_needed) else model.no_sync()

            with sync_ctx:
                # Autocast forward (bf16 on A100)
                with amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                y_pred = out["logits"].to(torch.float64)  # cast up for JAGeR
                # per-forward state updates in loss; pass 0-based micro-step consistent across ranks
                micro_step = (epoch - 1) * len(dl_train) + (step - 1)
                try:
                    loss = loss_fn(y_pred, ids, update_state=True, global_step=micro_step) / args.grad_accum
                except TypeError:
                    # CE baseline signature (no global_step)
                    loss = loss_fn(y_pred, ids) / args.grad_accum
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                running += loss.item()
                
                step_in_accum = next_accum

            if step_in_accum % args.grad_accum == 0:
                print(f"[epoch {epoch}] step {step}/{len(dl_train)}: loss={running/step_in_accum:.4f} ")    
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optim); scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True); sched.step()
                
                if run_wandb is not None:
                    # Per-optimizer-step logging (use last computed loss)
                    log_dict = {"train/loss_step": float(loss.detach().cpu()), "epoch": epoch}
                    # λ stats (only for JAGeR loss)
                    λ = getattr(loss_fn, "λ", None)
                    if isinstance(λ, torch.Tensor):
                        lam_det = λ.detach()
                        log_dict["train/lambda_min"] = float(lam_det.min().cpu())
                        log_dict["train/lambda_max"] = float(lam_det.max().cpu())
                    if (global_step % max(1, args.log_every)) == 0:
                        wandb.log(log_dict, step=global_step)
                    global_step += 1


        # finish partial accumulation only if we didn't just step above
        if (step_in_accum % args.grad_accum) != 0:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler is not None and scaler.is_enabled():
                scaler.step(optim); scaler.update()
            else:
                optim.step()
            optim.zero_grad(set_to_none=True); sched.step()

        # Monitor λ range
        train_loss_avg = running / max(1, math.ceil(len(dl_train)/args.grad_accum))
        if hasattr(loss_fn, "λ") and isinstance(getattr(loss_fn, "λ", None), torch.Tensor):
            with torch.no_grad():
                λ = loss_fn.λ
                lam_min = λ.min().item(); lam_max = λ.max().item()
            if is_main_process():
                print(f"[epoch {epoch}] train_loss(avg per step)={running/max(1, math.ceil(len(dl_train)/args.grad_accum)):.4f} "
                        f"lambda[min,max]=[{lam_min:.6f},{lam_max:.6f}]")

        # Evaluate on rank 0 only
        if is_main_process():
            val_loss, cms, qwks, emds, macro_emd, tail_recalls, tail_recall_avg = evaluate(
                model.module if is_dist() else model,
                dl_val, loss_fn, device, args
            )
            # average QWK across heads
            avg_qwk = float(sum(qwks) / len(qwks)) if len(qwks) else float("nan")
            print(
                f"[epoch {epoch}] val_loss={val_loss:.4f} "
                f"qwk={tuple(f'{x:.4f}' for x in qwks)} "
                f"averageQWK={avg_qwk:.4f} "
                f"macroEMD={macro_emd:.4f} "
                f"tailR0={tuple(f'{x:.4f}' for x in tail_recalls)} "
                f"tailR0avg={tail_recall_avg:.4f}"
            )
            if run_wandb is not None:
                wandb.log({
                    "train/loss_epoch": float(running / max(1, math.ceil(len(dl_train)/args.grad_accum))),
                    "val/loss": float(val_loss),
                    "val/qwk_content": float(qwks[0]),
                    "val/qwk_organization": float(qwks[1]),
                    "val/qwk_language": float(qwks[2]),
                    "val/qwk_average": float(avg_qwk),
                    "val/emd_content": float(emds[0]),
                    "val/emd_organization": float(emds[1]),
                    "val/emd_language": float(emds[2]),
                    "val/macroEMD": float(macro_emd),
                    "val/tail_recall0_content": float(tail_recalls[0]),
                    "val/tail_recall0_organization": float(tail_recalls[1]),
                    "val/tail_recall0_language": float(tail_recalls[2]),
                    "val/tail_recall0_average": float(tail_recall_avg),
                }, step=global_step)

            head_names = ["content", "organization", "language"]
            for h, name in enumerate(head_names):
                print(f"[{name}] confusion matrix (rows=true 1..5, cols=pred 1..5):")
                cm = cms[h]
                print("\n".join("  " + " ".join(f"{v:4d}" for v in row) for row in cm))
            # W&B heatmaps for VAL
            if run_wandb is not None:
                _log_cms_as_wandb_images(cms, split="val", head_names=head_names, step=global_step)


        # ----- determine improvement by avg QWK (maximize), regardless of saving -----
        improved = False
        if is_main_process() and (avg_qwk > best_qwk):
            best_qwk = avg_qwk
            improved = True

        # ----- save best.pt only if requested -----
        if improved and is_main_process() and args.save_model:
            path = os.path.join(args.save_dir, "best.pt")
            to_save = unwrap(model).state_dict()
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
                "model": to_save,
                "config": {"model_name": args.model_name, "K": 5, "heads": 3},
                "best_avg_qwk": float(best_qwk),
                "epoch": int(epoch),
            }, path)
            with open(os.path.join(args.save_dir, "run_args.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
            print(f"  ↑ saved new best (avgQWK={best_qwk:.4f}) to {path}")
            if run_wandb is not None:
                try:
                    art = wandb.Artifact(
                        f"best-{args.loss}-j{args.joint}-m{args.mixture}"
                        f"-cg{args.conf_gating}-ra{args.reassignment}",
                        type="model",
                        metadata={"epoch": int(epoch), "best_avg_qwk": float(best_qwk)},
                    )
                    art.add_file(os.path.join(args.save_dir, "best.pt"))
                    wandb.log_artifact(art)
                except Exception as e:
                    print("[W&B] artifact log skipped:", e)

            # --- append per-epoch stats (refit only; enabled by flag) ---
            if args.log_epoch_stats:
                # union of train/val/(test if present) was built earlier
                ids = idx_all
                row = {"epoch": int(epoch)}
                # λ: JAGeR shape depends on joint/per-head implementation
                λ = getattr(loss_fn, "λ", None)
                if isinstance(λ, torch.Tensor):
                    sel = λ.detach().cpu()[ids]
                    if sel.dim() == 1:
                        row.update({
                            "λ_min": float(sel.min().item()),
                            "λ_mean": float(sel.mean().item()),
                            "λ_max": float(sel.max().item()),
                        })
                    else:
                        means = sel.float().mean(dim=0); mins = sel.min(dim=0).values; maxs = sel.max(dim=0).values
                        for h in range(means.numel()):
                            row[f"λ_min_h{h}"]  = float(mins[h].item())
                            row[f"λ_mean_h{h}"] = float(means[h].item())
                            row[f"λ_max_h{h}"]  = float(maxs[h].item())
                # ρ for JAGeR-mixture (1 per head per sample), if present
                ρ = getattr(loss_fn, "ρ", None)
                if isinstance(ρ, torch.Tensor):
                    sel = ρ.detach().cpu()[ids]
                    means = sel.float().mean(dim=0); mins = sel.min(dim=0).values; maxs = sel.max(dim=0).values
                    for h in range(means.numel()):
                        row[f"ρ_min_h{h}"]  = float(mins[h].item())
                        row[f"ρ_mean_h{h}"] = float(means[h].item())
                        row[f"ρ_max_h{h}"]  = float(maxs[h].item())
                # append
                path = os.path.join(args.save_dir, "epoch_stats.csv")
                is_new = not os.path.exists(path)
                cols = ["epoch"] + [k for k in row.keys() if k != "epoch"]
                with open(path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=cols)
                    if is_new: w.writeheader()
                    w.writerow(row)

               # Log epoch_stats.csv as a separate artifact (avoid duplicating best.pt)
                if run_wandb is not None:
                    try:
                        stats_path = os.path.join(args.save_dir, "epoch_stats.csv")
                        art = wandb.Artifact(
                           f"epoch-stats-{args.loss}-j{args.joint}-m{args.mixture}"
                            f"-cg{args.conf_gating}-ra{args.reassignment}",
                            type="metrics",
                            metadata={"epoch": epoch}
                        )
                        art.add_file(stats_path)
                        wandb.log_artifact(art)
                    except Exception as e:
                        print("[W&B] epoch-stats artifact log skipped:", e)


        # TEST evaluation on improve (even if we didn't save): reload best.pt iff it exists
        if args.eval_test and improved and ds_test is not None and is_main_process():
            # reload best
            best_path = os.path.join(args.save_dir, "best.pt")
            if args.save_model and os.path.exists(best_path):
                ckpt = torch.load(best_path, map_location="cpu")
                unwrap(model).load_state_dict(ckpt["model"])
            else:
                print("[TEST] evaluating current in-memory weights (unsaved best)")

            test_loader = DataLoader(
                ds_test, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers,
                persistent_workers=(args.num_workers > 0),
                pin_memory=pin,
                collate_fn=collate,
                prefetch_factor=_pf,
            )

            # evaluate returns: loss, cms, qwks, emds, macro_emd, tail_recalls, tail_recall_avg
            test_loss, tcms, tqwks, temds, tmacro_emd, ttails, ttails_avg = evaluate(
                unwrap(model), test_loader, loss_fn, device, args
            )
            # average QWK across heads (test)
            tavg_qwk = float(sum(tqwks) / len(tqwks)) if len(tqwks) else float("nan")

            print(
                f"[TEST] loss={test_loss:.4f} "
                f"qwk={tuple(f'{x:.4f}' for x in tqwks)} "
                f"averageQWK={tavg_qwk:.4f} "
                f"emd={tuple(f'{x:.4f}' for x in temds)} "
                f"macroEMD={tmacro_emd:.4f} "
                f"tailR0={tuple(f'{x:.4f}' for x in ttails)} "
                f"tailR0avg={ttails_avg:.4f}"
            )

            if run_wandb is not None:
                wandb.log({
                    "test/loss": float(test_loss),
                    "test/qwk_content": float(tqwks[0]),
                    "test/qwk_organization": float(tqwks[1]),
                    "test/qwk_language": float(tqwks[2]),
                    "test/qwk_average": float(tavg_qwk),
                    "test/emd_content": float(temds[0]),
                    "test/emd_organization": float(temds[1]),
                    "test/emd_language": float(temds[2]),
                    "test/macroEMD": float(tmacro_emd),
                    "test/tail_recall0_content": float(ttails[0]),
                    "test/tail_recall0_organization": float(ttails[1]),
                    "test/tail_recall0_language": float(ttails[2]),
                    "test/tail_recall0_average": float(ttails_avg)
                }, step=global_step)
                # W&B heatmaps for TEST
                _log_cms_as_wandb_images(tcms, split="test", head_names=("content","organization","language"), step=global_step)


            # write compact metrics.json
            import json
            run_args = vars(args).copy()
            metrics = {
                "val": {
                    "loss": val_loss,
                    "qwk": qwks,
                    "qwk_average": avg_qwk,
                    "cm": [cm.tolist() for cm in cms],
                    "emd": emds,
                    "macroEMD": macro_emd,
                    "tail_recall0": tail_recalls,
                    "tail_recall0_average": tail_recall_avg,
                },
                "test": {
                    "loss": test_loss,
                    "qwk": tqwks,
                    "qwk_average": tavg_qwk,
                    "cm": [cm.tolist() for cm in tcms],
                    "emd": temds,
                    "macroEMD": tmacro_emd,
                    "tail_recall0": ttails,
                    "tail_recall0_average": ttails_avg,
                },
                "args": {
                    k: run_args[k] for k in [
                        "loss", "joint", "mixture", "conf_gating", "reassignment",
                        "lambda0", "alpha", "C",
                        "ce_label_smoothing", "seed",
                        "model_name", "max_length", "batch_size",
                        "grad_accum", "epochs"
                    ]
                }
            }
            metrics.setdefault("extra", {})["static_stats"] = static_stats
            with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"[TEST] wrote {os.path.join(args.save_dir, 'metrics.json')}")


        if is_dist(): 
            dist.barrier()

    # --- Always ensure VAL is present in metrics.json (inner CV needs this) ---
    if is_main_process() and dl_val is not None:
        try:
            v_loss, v_cms, v_qwks, v_emds, v_macro_emd, v_tails, v_tails_avg = evaluate(
                model.module if is_dist() else model,
                dl_val, loss_fn, device, args
            )
            import json
            p = os.path.join(args.save_dir, "metrics.json")
            obj = {}
            if os.path.exists(p):
                with open(p) as f:
                    obj = json.load(f)
            v_avg_qwk = float(sum(v_qwks) / len(v_qwks)) if len(v_qwks) else float("nan")
            obj["val"] = {
                "loss": v_loss,
                "qwk": v_qwks,
                "qwk_average": v_avg_qwk,
                "cm": [cm.tolist() for cm in v_cms],
                "emd": v_emds,
                "macroEMD": v_macro_emd,
                "tail_recall0": v_tails,
                "tail_recall0_average": v_tails_avg,
}

            obj.setdefault("extra", {})["static_stats"] = obj.get("extra", {}).get("static_stats", static_stats)
            with open(p, "w") as f:
                json.dump(obj, f, indent=2)
            print(f"[VAL] updated {p}")
        except Exception as e:
            print(f"[warn] could not append VAL metrics: {e}")



    if is_dist(): dist.destroy_process_group()
    if run_wandb is not None: run_wandb.finish()

    print("Done.")

if __name__ == "__main__":
    main()
