import math, os, random, argparse, csv, json, re, shutil, subprocess
import contextlib
from typing import Any, TypedDict
import torch
from torch import nn, amp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, Subset
import torch.distributed as dist
try:
    from torch.amp.grad_scaler import GradScaler as AmpGradScaler
except Exception:
    AmpGradScaler = None
torch.set_float32_matmul_precision("medium")

from datetime import timedelta
from transformers import get_linear_schedule_with_warmup, DataCollatorWithPadding, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, cohen_kappa_score, recall_score
import numpy as np
from modeling_multitask import MultiHeadDeberta
from data_utils import EssayDataset
from loss import JAGeRLoss, MultiHeadCELoss
try:
    import wandb
    _WANDB_OK = True
except Exception:
    wandb = None
    _WANDB_OK = False
    

class _EvalPayload(TypedDict):
    ids: torch.Tensor
    labels: torch.Tensor
    values: torch.Tensor


def _gather_stats_from_buffer(buf: "torch.Tensor", idx: list[int], per_head: bool):
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


def _collect_static_stats(loss_fn, views: dict[str, list[int]]):
    λ = getattr(loss_fn, "λ", None)
    ρ = getattr(loss_fn, "ρ", None)
    static_stats = {}
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
    return static_stats


def _log_static_stats_to_wandb(static_stats):
    if wandb is None or wandb.run is None or not static_stats:
        return
    flat = {}
    for k, v in static_stats.items():
        if isinstance(v, dict) and isinstance(v.get("mean"), list):
            for h, m in enumerate(v["mean"]):
                flat[f"{k}/mean/h{h}"] = m
            for h, m in enumerate(v.get("min", [])):
                flat[f"{k}/min/h{h}"] = m
            for h, m in enumerate(v.get("max", [])):
                flat[f"{k}/max/h{h}"] = m
        else:
            flat[f"{k}/mean"] = v["mean"]
            flat[f"{k}/min"] = v["min"]
            flat[f"{k}/max"] = v["max"]
    wandb.log(flat, step=-1)  # type: ignore[union-attr]


def _configure_wandb_metrics():
    if wandb is None or wandb.run is None:
        return

    wandb.define_metric("epoch")  # type: ignore[union-attr]
    epoch_metrics = (
        "train/loss_epoch",
        "val/loss",
        "val/qwk_average",
        "val/macroEMD",
        "val/mEMD",
        "val/tail_recall0_average",
        "val/qwk_*",
        "val/emd_*",
        "val/tail_recall0_*",
        "test/loss",
        "test/qwk_average",
        "test/macroEMD",
        "test/mEMD",
        "test/tail_recall0_average",
        "test/qwk_*",
        "test/emd_*",
        "test/tail_recall0_*",
    )
    for metric_name in epoch_metrics:
        wandb.define_metric(metric_name, step_metric="epoch")  # type: ignore[union-attr]


def _dedupe_by_id(ids: "torch.Tensor", *tensors: "torch.Tensor"):
    if ids.numel() == 0:
        return (ids, *tensors)
    order = ids.argsort()
    ids_sorted = ids[order]
    tensors_sorted = [t[order] for t in tensors]
    keep = torch.ones(ids_sorted.shape[0], dtype=torch.bool)
    keep[1:] = ids_sorted[1:] != ids_sorted[:-1]
    return (ids_sorted[keep], *(t[keep] for t in tensors_sorted))


def _gather_eval_payload(ids: "torch.Tensor", labels: "torch.Tensor", values: "torch.Tensor"):
    ids_cpu = ids.detach().cpu()
    labels_cpu = labels.detach().cpu()
    values_cpu = values.detach().cpu()
    if is_dist():
        payload: _EvalPayload = {"ids": ids_cpu, "labels": labels_cpu, "values": values_cpu}
        if is_main_process():
            gathered: list[_EvalPayload | None] = [None for _ in range(dist.get_world_size())]
            dist.gather_object(payload, gathered, dst=0)
            parts = [part for part in gathered if part is not None]
            ids_cpu = torch.cat([part["ids"] for part in parts], dim=0)
            labels_cpu = torch.cat([part["labels"] for part in parts], dim=0)
            values_cpu = torch.cat([part["values"] for part in parts], dim=0)
        else:
            dist.gather_object(payload, None, dst=0)
            return ids_cpu, labels_cpu, values_cpu
    return _dedupe_by_id(ids_cpu, labels_cpu, values_cpu)


@contextlib.contextmanager
def _override_attrs(obj, **updates):
    old = {name: getattr(obj, name) for name in updates}
    for name, value in updates.items():
        setattr(obj, name, value)
    try:
        yield
    finally:
        for name, value in old.items():
            setattr(obj, name, value)


def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def _dataset_row_ids(ds) -> np.ndarray:
    if isinstance(ds, Subset):
        base = _dataset_row_ids(ds.dataset)
        idx = np.asarray(ds.indices, dtype=np.int64)
        return base[idx]
    return np.arange(len(ds), dtype=np.int64)


def _get_nvidia_driver_version() -> tuple[str | None, str]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            out = subprocess.check_output(
                [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=10,
            )
            versions = sorted({line.strip() for line in out.splitlines() if line.strip()})
            if versions:
                return ",".join(versions), nvidia_smi
        except Exception as exc:
            smi_err = f"{type(exc).__name__}: {exc}"
    else:
        smi_err = "nvidia-smi not found"

    proc_path = "/proc/driver/nvidia/version"
    try:
        with open(proc_path) as f:
            text = f.read()
        match = re.search(r"Kernel Module\s+([0-9.]+)", text)
        if match:
            return match.group(1), proc_path
    except OSError as exc:
        proc_err = f"{type(exc).__name__}: {exc}"
    else:
        proc_err = f"could not parse {proc_path}"

    return None, f"{smi_err}; {proc_err}"


class ShardedEvalSampler(Sampler[int]):
    def __init__(self, dataset, *, num_replicas: int | None = None, rank: int | None = None):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if is_dist() else 1
        if rank is None:
            rank = dist.get_rank() if is_dist() else 0
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        n = len(self.dataset)
        if self.rank >= n:
            return 0
        return (n - self.rank + self.num_replicas - 1) // self.num_replicas


def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_main_process():
    return (not is_dist()) or dist.get_rank() == 0

def setup_distributed():
    # Initialize DDP from torchrun env if present
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    dist_uses_cuda = False
    if local_rank != -1 and not is_dist():
        dist_uses_cuda = torch.cuda.is_available() and dist.is_nccl_available()
        backend = "nccl" if dist_uses_cuda else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=1800))
        if dist_uses_cuda:
            torch.cuda.set_device(local_rank)
    elif is_dist():
        dist_uses_cuda = dist.get_backend() == "nccl"
    return local_rank, dist_uses_cuda

def parse_args():
    p = argparse.ArgumentParser()

    # --- Data / general -------------------------------------------------
    p.add_argument(
        "--data_path", "--tsv",
        dest="data_path",
        default="../data/DREsS/DREsS_New_cleaned.tsv",
        help="Path to CSV/TSV with columns ['prompt','essay', <label1>, <label2>, ...].",
    )
    p.add_argument(
        "--split_file", type=str, default="splits/dress_seed42.json",
        help="JSON with {train,val,test} id lists; ids are row indices into the data file."
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument(
        "--ens_mode",
        choices=["fixed_range", "tail"],
        default="fixed_range",
        help="How to choose validation epochs for OOF ensembling.",
    )
    p.add_argument(
        "--ens_epoch_start",
        type=int,
        default=8,
        help="First epoch to include when --ens_mode fixed_range.",
    )
    p.add_argument(
        "--ens_epoch_end",
        type=int,
        default=14,
        help="Last epoch to include when --ens_mode fixed_range.",
    )
    p.add_argument(
        "--ens_t",
        type=int,
        default=10,
        help="Number of trailing epochs to ensemble when --ens_mode tail.",
    )
    p.add_argument(
        "--ens_stride",
        type=int,
        default=1,
        help="Stride applied after selecting OOF ensemble epochs.",
    )
    p.add_argument("--val_frac", type=float, default=0.1,
                   help="If no --split_file, fraction used for validation.")
    p.add_argument("--limit_train", type=int, default=0, help="use only N training examples (0=all)")
    p.add_argument("--limit_val", type=int, default=0, help="use only N validation examples (0=all)")

    # --- Label support (ordinal) ---------------------------------------
    p.add_argument("--level_offset", type=int, default=1,
                   help="Lowest ordinal label value (inclusive).")
    p.add_argument("--max_level", type=int, default=None,
                   help="Highest ordinal label value (inclusive). If not set, inferred from data.")

    # --- Model / Tokenizer ---------------------------------------------
    p.add_argument("--model_name", default="microsoft/deberta-v3-large")
    p.add_argument("--hf_offline", action="store_true",
                   help="Force offline mode and local files only for HF")
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--pad_to_multiple_of", type=int, default=8, help="dynamic padding grid")
    p.add_argument(
        "--token_cache_dir",
        type=str,
        default=None,
        help="Reusable on-disk token cache directory (default: alongside the data file).",
    )

    # --- DataLoader / perf ------------------------------------------------
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--prefetch_factor", type=int, default=4,
                   help="Batches to prefetch per worker (effective only when num_workers>0)")

    # --- Training / optimization --------------------------------------
    p.add_argument("--batch_size", type=int, default=2)         # per-device
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument(
        "--sched_epochs",
        type=int,
        default=None,
        help="Scheduler horizon in epochs. Defaults to --epochs; set larger to stop early without compressing the LR schedule.",
    )
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    p.add_argument("--grad_ckpt", action="store_true",
                   help="Enable gradient checkpointing for the encoder")
    p.add_argument("--freeze_encoder", action="store_true", help="train heads only")
    p.add_argument("--unfreeze_at_epoch", type=int, default=-1, help="-1 = never unfreeze in this run")
    p.add_argument("--log_every", type=int, default=50, help="Log training stats every N optimizer steps")
    p.add_argument("--save_model", action="store_true", help="Save best.pt (OFF by default; turn on for finalist runs).")
    p.add_argument(
        "--init_model_from",
        type=str,
        default=None,
        help="Warm-start model weights from a checkpoint file, or from best.pt inside a checkpoint directory.",
    )

    # --- Loss family / hyperparameters --------------------------------
    p.add_argument("--loss", choices=["jager", "ce"], default="jager",
                   help="jager = JAGeR, ce = cross-entropy")
    p.add_argument("--joint", action=argparse.BooleanOptionalAction, default=True,
                   help="JAGeR: use joint K^H space (--no-joint for per-head)")
    p.add_argument("--mixture", action=argparse.BooleanOptionalAction, default=True,
                   help="JAGeR: enable mixture prior / ρ estimation")
    p.add_argument("--conf_gating", action=argparse.BooleanOptionalAction, default=True,
                   help="JAGeR: enable confidence-gated ρ update")
    p.add_argument("--reassignment", action=argparse.BooleanOptionalAction, default=True,
                   help="JAGeR: enable y_pred_max reassignment term")
    p.add_argument("--lambda0", type=float, default=1.0, help="initial λ (JAGeR)")
    p.add_argument("--lambda_min", type=float, help="minimum λ (JAGeR); overrides alpha if set")
    p.add_argument("--alpha", type=float, default=2.0, help="α (JAGeR)")
    p.add_argument("--C", type=float, default=1e-1, help="margin C (JAGeR)")
    p.add_argument("--ce_label_smoothing", type=float, default=0.0, help="label smoothing for CE baseline")

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
    p.add_argument("--wandb_group", type=str, default=None, help="Optional W&B run group")
    p.add_argument("--wandb_job_type", type=str, default=None, help="Optional W&B job type")
    p.add_argument("--wandb_run_name", type=str, default=None, help="Optional explicit W&B run name")
    p.add_argument("--wandb_run_id", type=str, default=None, help="Optional explicit W&B run id for resume")
    p.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated extra W&B tags")

    args = p.parse_args()
    if args.sched_epochs is None:
        args.sched_epochs = args.epochs
    if args.sched_epochs < args.epochs:
        p.error("--sched_epochs must be greater than or equal to --epochs.")
    if args.ens_mode == "fixed_range":
        if args.ens_epoch_end < args.ens_epoch_start:
            p.error("--ens_epoch_end must be greater than or equal to --ens_epoch_start.")
        if args.epochs < args.ens_epoch_end:
            p.error("--epochs must be greater than or equal to --ens_epoch_end when --ens_mode fixed_range.")
    return args


def evaluate(model, dl, loss_fn, device, args=None, return_payload: bool = False):
    assert args is not None, "evaluate requires args with num_heads/K/level_offset/target_cols/minority_classes"
    assert hasattr(args, "num_heads") and hasattr(args, "K") and hasattr(args, "level_offset")
    assert hasattr(args, "target_cols") and hasattr(args, "minority_classes")

    model.eval()
    use_amp = (device.type == "cuda")
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    if is_main_process():
        print(f"Evaluating with amp_dtype={amp_dtype} on device={device} (use_amp={use_amp})")

    num_heads = int(args.num_heads)
    K = int(args.K)
    level_offset = int(args.level_offset)
    head_names = list(args.target_cols)
    base_model = model.module if hasattr(model, "module") else model

    all_ids = []
    all_labels = []
    all_logits = []
    all_true = [[] for _ in range(num_heads)]
    all_pred = [[] for _ in range(num_heads)]
    sum_emd_by_class = torch.zeros(num_heads, K, dtype=torch.float64)
    count_by_class   = torch.zeros(num_heads, K, dtype=torch.long)
    loss_sum = 0.0
    loss_weight = 0
    logits_dtype = None
    override_ctx = (
        _override_attrs(loss_fn, conf_gating=False, log_to_wandb=False)
        if isinstance(loss_fn, JAGeRLoss)
        else contextlib.nullcontext()
    )

    with torch.inference_mode(), override_ctx:
        for batch in dl:
            pin = (device.type == "cuda")
            ids = batch["ids"].to(device, non_blocking=pin)
            input_ids = batch["input_ids"].to(device, non_blocking=pin)
            attention_mask = batch["attention_mask"].to(device, non_blocking=pin)
            labels = batch["labels"].detach().cpu()

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):  # type: ignore[attr-defined]
                out = model(input_ids=input_ids, attention_mask=attention_mask)

            y_pred = (out["logits"] if isinstance(out, dict) and "logits" in out else out)  # type: ignore[union-attr]
            if args.loss == "jager": y_pred = y_pred.to(torch.float64)
            logits_dtype = y_pred.dtype
            if isinstance(loss_fn, JAGeRLoss):
                loss = loss_fn(y_pred, ids, update_state=False)
                weight = int(input_ids.size(0)) if loss_fn.joint else int(labels.numel())
            elif isinstance(loss_fn, MultiHeadCELoss):
                loss = loss_fn(y_pred, ids)
                weight = int(labels.numel())
            else:
                labels_dev = (labels - level_offset).to(device, non_blocking=pin)
                flat_logits = y_pred.reshape(-1, y_pred.shape[-1])
                flat_labels = labels_dev.reshape(-1)
                loss = F.cross_entropy(flat_logits, flat_labels, reduction="mean")
                weight = int(flat_labels.numel())
            loss_sum += float(loss.item()) * weight
            loss_weight += weight

            logits = y_pred.detach().cpu()
            all_ids.append(ids.detach().cpu())
            all_labels.append(labels)
            all_logits.append(logits)

    loss_stats = torch.tensor([loss_sum, float(loss_weight)], dtype=torch.float64, device=device)
    if is_dist():
        dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
    tot_loss = float(loss_stats[0].item() / max(loss_stats[1].item(), 1.0))

    if logits_dtype is None:
        if args.loss == "jager":
            logits_dtype = torch.float64
        elif use_amp:
            logits_dtype = amp_dtype
        else:
            logits_dtype = next(base_model.heads.parameters()).dtype  # type: ignore[union-attr]

    empty_ids = torch.empty((0,), dtype=torch.long)
    empty_labels = torch.empty((0, num_heads), dtype=torch.long)
    empty_logits = torch.empty((0, num_heads, K), dtype=logits_dtype)
    ids_cat = torch.cat(all_ids, dim=0) if all_ids else empty_ids
    labels_cat = torch.cat(all_labels, dim=0) if all_labels else empty_labels
    logits_cat = torch.cat(all_logits, dim=0) if all_logits else empty_logits
    ids_all, labels_all, logits_all = _gather_eval_payload(ids_cat, labels_cat, logits_cat)

    if is_dist() and not is_main_process():
        if return_payload:
            return (
                float("nan"), [], [], [], float("nan"), [], float("nan"),
                np.empty((0,), dtype=np.int64), {}, {}
            )
        return (float("nan"), [], [], [], float("nan"), [], float("nan"))
    pred_idx = logits_all.argmax(-1)
    probs = torch.softmax(logits_all.float(), dim=-1)

    levels = torch.arange(level_offset, level_offset + K, device=probs.device, dtype=probs.dtype)
    abs_dist = (levels.view(1, 1, -1) - labels_all.to(probs).unsqueeze(-1).float()).abs() / max(K - 1, 1)
    emd_batch = (probs * abs_dist).sum(dim=-1)

    idx = (labels_all - level_offset).clamp(0, K - 1)
    one_hot = torch.nn.functional.one_hot(idx, num_classes=K).to(dtype=torch.long)
    sum_emd_by_class += (emd_batch.unsqueeze(-1) * one_hot).sum(dim=0).to(torch.float64)
    count_by_class   += one_hot.sum(dim=0)

    for h in range(num_heads):
        all_true[h].extend(labels_all[:, h].tolist())
        all_pred[h].extend((pred_idx[:, h] + level_offset).tolist())

    cms, qwks, emds, tail_recalls = [], [], [], []
    label_list = list(range(level_offset, level_offset + K))
    for h in range(num_heads):
        y_t = np.array(all_true[h], dtype=int)
        y_p = np.array(all_pred[h], dtype=int)

        cms.append(confusion_matrix(y_t, y_p, labels=label_list))
        qwks.append(float(cohen_kappa_score(y_t, y_p, weights="quadratic")))

        sums   = sum_emd_by_class[h].numpy()
        counts = count_by_class[h].numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            per_class_emd = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        emds.append(float(np.nanmean(per_class_emd)))

        minority = list(args.minority_classes[h])  # must be present
        per_class_rec = recall_score(y_t, y_p, labels=label_list, average=None, zero_division=0)
        mr = [per_class_rec[c - level_offset] for c in minority]  # type: ignore[index]
        tail_recalls.append(float(np.mean(mr)) if mr else float("nan"))

    macro_emd = float(sum(emds) / len(emds)) if emds else float("nan")
    tail_recall_avg = float(sum(tail_recalls) / len(tail_recalls)) if tail_recalls else float("nan")

    if return_payload:
        ids_np = ids_all.numpy()
        y_payload = {name: labels_all[:, h].numpy() for h, name in enumerate(head_names)}
        p_payload = {name: probs[:, h, :].numpy() for h, name in enumerate(head_names)}
        return (
            tot_loss,
            cms,
            qwks,
            emds,
            macro_emd,
            tail_recalls,
            tail_recall_avg,
            ids_np,
            y_payload,
            p_payload,
        )

    return (
        tot_loss,
        cms,
        qwks,
        emds,
        macro_emd,
        tail_recalls,
        tail_recall_avg
    )

def _maybe_init_wandb(args):
    if not getattr(args, "wandb", False) or args.wandb_mode == "disabled":
        return None
    if not _WANDB_OK:
        print("[W&B] wandb not installed; proceeding without logging.")
        return None
    default_name = (
        f"{args.loss}-j{int(args.joint)}-m{int(args.mixture)}-cg{int(args.conf_gating)}-ra{int(args.reassignment)}"
        f"-bs{args.batch_size}-ga{args.grad_accum}"
    )
    tags = [
        args.loss,
        f"joint={int(args.joint)}",
        f"mixture={int(args.mixture)}",
        f"conf_gating={int(args.conf_gating)}",
        f"reassign={int(args.reassignment)}",
    ]
    if args.wandb_tags:
        tags.extend([tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()])

    init_kwargs = dict(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        mode=args.wandb_mode,
        name=args.wandb_run_name or default_name,
        group=args.wandb_group or None,
        job_type=args.wandb_job_type or None,
        tags=tags,
        config=vars(args),
    )
    if args.wandb_run_id:
        init_kwargs["id"] = args.wandb_run_id
        init_kwargs["resume"] = "allow"

    run = wandb.init(  # type: ignore[union-attr]
        **init_kwargs,
    )
    _configure_wandb_metrics()
    return run


def _log_cms_as_wandb_images(cms, split: str, head_names, level_offset: int, step: int | None = None):
        
    if wandb is None or wandb.run is None:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        print("[W&B] matplotlib not installed; skipping confusion-matrix images.")
        return
    if not cms:
        return
    if head_names is None:
        head_names = [f"head{h}" for h in range(len(cms))]
    K = int(np.asarray(cms[0]).shape[0])
    tick_labels = list(range(level_offset, level_offset + K))
    vmax = max(int(np.asarray(cm).max()) for cm in cms)
    for h, name in enumerate(head_names):
        if h >= len(cms):
            break
        cm = np.asarray(cms[h], dtype=np.int32)
        fig = plt.figure(figsize=(3.2, 3.2), dpi=150)
        ax = fig.add_subplot(111)
        ax.imshow(cm, vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{split.upper()} CM – {name}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(tick_labels); ax.set_yticklabels(tick_labels)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)  # type: ignore[arg-type]
        fig.tight_layout()
        wandb.log({f"{split}/cm_{name}": wandb.Image(fig)}, step=step)
        plt.close(fig)


def _load_tokenizer_strict(model_name: str):
    tok_path = os.path.join(model_name, "tokenizer.json")
    added = 0

    if not os.path.isdir(model_name):
        raise RuntimeError(
            f"Refusing to train without a local model directory: {model_name}. "
            "Point --model_name to the saved directory that contains tokenizer.json."
        )
    if not os.path.exists(tok_path):
        raise RuntimeError(
            f"Refusing to train without a saved fast tokenizer: missing {tok_path}. "
            "Create/save tokenizer.json first, then rerun."
        )

    tok = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )

    bad_fast_wrappers = {"Tokenizer", "TokenizersBackend"}
    if (not getattr(tok, "is_fast", False)) or (tok.__class__.__name__ in bad_fast_wrappers):
        raise RuntimeError(
            f"Refusing to train with tokenizer class={tok.__class__.__name__!r} from local model dir {model_name}. "
            "Expected a usable fast tokenizer loaded from the saved local model bundle."
        )

    tok_name_or_path = getattr(tok, "name_or_path", "")
    if tok_name_or_path:
        want = os.path.realpath(model_name)
        got = os.path.realpath(tok_name_or_path)
        if want != got:
            raise RuntimeError(
                f"Refusing to train because tokenizer resolved to {tok_name_or_path!r}, not the requested local model dir {model_name!r}."
            )

    return tok, added, "local_saved_model"


def main():
    args = parse_args()
    
    # Force-safe flags when mixture=0
    if args.loss == "jager" and not args.mixture:
        args.conf_gating = False
        args.reassignment = False

    if os.path.sep in args.model_name or args.model_name.startswith("."):
        args.model_name = os.path.abspath(os.path.expanduser(args.model_name))

    if args.hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # avoid noisy warnings
    
    run_wandb = None

    try:
    
        os.makedirs(args.save_dir, exist_ok=True)
        local_rank, dist_uses_cuda = setup_distributed()
        run_wandb = _maybe_init_wandb(args) if is_main_process() else None
        np.random.seed(args.seed + (dist.get_rank() if is_dist() else 0))
        set_seed(args.seed + (dist.get_rank() if is_dist() else 0))

        if local_rank != -1:
            device = torch.device(f"cuda:{local_rank}") if dist_uses_cuda else torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if is_main_process():
            driver_version, driver_source = _get_nvidia_driver_version()
            if driver_version is not None:
                print(f"[env] nvidia_driver_version={driver_version} source={driver_source}")
            else:
                print(f"[env] nvidia_driver_version=unavailable detail={driver_source}")

        # Build dataset 
        tok, added, tok_src = _load_tokenizer_strict(
            args.model_name,
        )
        if is_main_process():
            print(
                f"[tok] class={tok.__class__.__name__} fast={getattr(tok, 'is_fast', None)} "
                f"src={tok_src}"
            )
        # ---- Generic dataset + splits ----
        ds = EssayDataset(
            path=args.data_path,
            tokenizer_name=args.model_name,
            max_length=args.max_length,
            tokenizer=tok,
            token_cache_dir=args.token_cache_dir,
        )
        if is_main_process():
            cache_state = "built" if ds.cache_built else "loaded"
            print(f"[data] token_cache={cache_state} path={ds.cache_dir} rows={len(ds)}")
        target_cols = ds.target_cols
        num_heads = len(target_cols)
        Y_all_cpu = ds.get_all_targets_tensor()

        if args.split_file and os.path.exists(args.split_file):
            with open(args.split_file) as f:
                split = json.load(f)
            train_ids = [int(i) for i in split["train"]]
            val_ids   = [int(i) for i in split["val"]]
            test_ids  = [int(i) for i in split.get("test", [])]
            ds_train = Subset(ds, train_ids)
            ds_val   = Subset(ds, val_ids)
            ds_test  = Subset(ds, test_ids) if test_ids else None
        else:
            idx_all = np.arange(len(ds))
            tr_idx, va_idx = train_test_split(
                idx_all,
                test_size=args.val_frac,
                random_state=args.seed,
                shuffle=True,
                stratify=None,
            )
            train_ids = [int(i) for i in tr_idx]
            val_ids   = [int(i) for i in va_idx]
            ds_train = Subset(ds, train_ids)
            ds_val   = Subset(ds, val_ids)
            ds_test  = None
            
        # Whether a test set exists AND the user asked to evaluate it
        eval_test_flag = bool(args.eval_test and ds_test is not None)

        level_offset = int(args.level_offset)
        if args.max_level is not None:
            max_level = int(args.max_level)
        else:
            max_level = int(Y_all_cpu.max().item())
        if max_level < level_offset:
            raise ValueError(f"max_level ({max_level}) must be >= level_offset ({level_offset}).")
        if ((Y_all_cpu < level_offset) | (Y_all_cpu > max_level)).any():
            y_min = int(Y_all_cpu.min().item()); y_max = int(Y_all_cpu.max().item())
            raise ValueError(
                f"Label values outside declared support [{level_offset}, {max_level}]: min={y_min}, max={y_max}."
            )
        K = max_level - level_offset + 1
        args.target_cols = target_cols
        args.num_heads = num_heads
        args.level_offset = level_offset
        args.max_level = max_level
        args.K = K
        head_names = list(args.target_cols)

        if args.limit_train and args.limit_train < len(ds_train):
            ds_train = Subset(ds_train, list(range(args.limit_train)))
        if args.limit_val and args.limit_val < len(ds_val):
            ds_val = Subset(ds_val, list(range(args.limit_val)))
        train_view_ids = _dataset_row_ids(ds_train).tolist()
        val_view_ids = _dataset_row_ids(ds_val).tolist()
        test_view_ids = _dataset_row_ids(ds_test).tolist() if ds_test is not None else []

        collate = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=(args.pad_to_multiple_of if args.pad_to_multiple_of > 0 else None))
        if tok.pad_token is None:
            if tok.sep_token is not None:
                tok.pad_token = tok.sep_token
            elif tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            else:
                # track newly added pad token so embeddings can be resized once
                n_added = tok.add_special_tokens({'pad_token': '[PAD]'})
                added += n_added

        # ---- Samplers (DDP) ----
        train_sampler = None
        val_sampler = None
        test_sampler = None
        if is_dist():
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(ds_train, shuffle=True, drop_last=False)
            val_sampler = ShardedEvalSampler(ds_val)
            if ds_test is not None:
                test_sampler = ShardedEvalSampler(ds_test)

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
            sampler=val_sampler,
            num_workers=args.num_workers, persistent_workers=(args.num_workers > 0),
            pin_memory=pin, collate_fn=collate, prefetch_factor=_pf,
        )
        dl_test = None
        if args.eval_test and ds_test is not None:
            dl_test = DataLoader(
                ds_test, batch_size=args.batch_size, shuffle=False,
                sampler=test_sampler,
                num_workers=args.num_workers, persistent_workers=(args.num_workers > 0),
                pin_memory=pin, collate_fn=collate, prefetch_factor=_pf,
            )

        # ---- Determine minority classes from TRAIN distribution (per head) ----
        def _compute_minority_classes(row_ids, K, num_heads, level_offset, threshold=0.10, bottom_m=2):
            counts = np.zeros((num_heads, K), dtype=np.int64)
            lab = Y_all_cpu[row_ids].cpu().numpy()
            if lab.shape[1] != num_heads:
                raise ValueError(f"Expected {num_heads} heads, got {lab.shape[1]}")
            idx = np.clip(lab - level_offset, 0, K - 1)
            total = np.full(num_heads, idx.shape[0], dtype=np.int64)
            for h in range(num_heads):
                counts[h] += np.bincount(idx[:, h], minlength=K)
            minority = []
            label_vals = np.arange(level_offset, level_offset + K)
            for h in range(num_heads):
                prev = counts[h] / max(total[h], 1)
                chosen = label_vals[prev < threshold]
                if chosen.size == 0:
                    chosen = label_vals[np.argsort(prev)[:bottom_m]]
                minority.append(chosen.tolist())
            return minority

        minority_classes = _compute_minority_classes(train_view_ids, K=K, num_heads=num_heads, level_offset=level_offset)
        setattr(args, "minority_classes", minority_classes)
        if is_main_process():
            print(f"[info] minority classes per head (from TRAIN): {minority_classes}")
        
        model = MultiHeadDeberta(
            args.model_name,
            num_heads=num_heads,
            num_classes=K,
            freeze_encoder=args.freeze_encoder,
            dropout=args.dropout,
            local_files_only=bool(args.hf_offline),
            trust_remote_code=False,
            torch_dtype=None, 
            enable_grad_ckpt=bool(getattr(args, "grad_ckpt", False)), 
        ).to(device)

        if added:
            model.resize_token_embeddings(len(tok))  # type: ignore[union-attr]

        def unwrap(m): return m.module if hasattr(m, "module") else m

        def _resolve_checkpoint_path(path: str) -> str:
            ckpt_path = os.path.abspath(os.path.expanduser(path))
            if os.path.isdir(ckpt_path):
                ckpt_path = os.path.join(ckpt_path, "best.pt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Missing checkpoint for --init_model_from: {ckpt_path}")
            return ckpt_path

        if args.init_model_from:
            init_path = _resolve_checkpoint_path(args.init_model_from)
            ckpt = torch.load(init_path, map_location="cpu")
            if not isinstance(ckpt, dict) or "model" not in ckpt:
                raise KeyError(f"{init_path} does not contain a 'model' state dict.")
            incompat = unwrap(model).load_state_dict(ckpt["model"], strict=True)  # type: ignore[union-attr]
            if is_main_process():
                print(f"[init] loaded model weights from {init_path}")
                missing = list(getattr(incompat, "missing_keys", []))
                unexpected = list(getattr(incompat, "unexpected_keys", []))
                if missing or unexpected:
                    print(f"[init] load_state_dict mismatch: missing={missing}, unexpected={unexpected}")

        if is_main_process():
            enc_dtype = next(model.encoder.parameters()).dtype  # type: ignore[union-attr]
            head_dtype = next(model.heads.parameters()).dtype  # type: ignore[union-attr]
            print(f"[model] encoder_dtype={enc_dtype} heads_dtype={head_dtype}")

        # Print gradient checkpointing status
        if is_main_process():
            base = unwrap(model)
            enc = base.encoder  # type: ignore[union-attr]
            flag = getattr(enc, "is_gradient_checkpointing", None)
            if flag is None:
                flag = getattr(getattr(enc, "config", object()), "gradient_checkpointing", False)
            print(f">> Grad checkpointing: {flag}")

        
        # Wrap with DDP if launched via torchrun
        if is_dist():
            ddp_kwargs: dict[str, Any] = {"find_unused_parameters": False}
            if device.type == "cuda":
                assert device.index is not None
                ddp_kwargs["device_ids"] = [device.index]
                ddp_kwargs["output_device"] = device.index
            model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
        
        # MultiHeadDeberta already applies freeze_encoder during construction.
        if args.freeze_encoder:
            if is_main_process():
                print(">> Encoder frozen (training heads only).")
            

        # Stateful loss
        Y_all = Y_all_cpu.to(device)
        if args.loss == "jager":
            effective_micro_batch = args.batch_size * (dist.get_world_size() if is_dist() else 1)
            train_state_ids = torch.as_tensor(train_view_ids, dtype=torch.long, device=device)
            loss_fn = JAGeRLoss(
                Y=Y_all,
                stats_ids=train_state_ids,
                K=K,
                joint=bool(args.joint),
                mixture=bool(args.mixture),
                conf_gating=bool(args.conf_gating),
                reassignment=bool(args.reassignment),
                level_offset=args.level_offset,
                λ0=args.lambda0,
                λmin=getattr(args, "lambda_min", None),
                α=args.alpha,
                C=args.C,
                log_to_wandb=bool(getattr(args, "wandb", False)),
                def_batch_size=effective_micro_batch,
                steps_per_epoch=len(dl_train)
            ).to(device)
        else:
            loss_fn = MultiHeadCELoss(
                Y = Y_all,
                level_offset = args.level_offset,
                label_smoothing=args.ce_label_smoothing
            ).to(device)

        def _loss_state_for_ckpt(loss_obj):
            if not isinstance(loss_obj, JAGeRLoss):
                return None
            return {k: v.detach().cpu() for k, v in loss_obj.state_dict().items()}

        def _restore_loss_state_from_ckpt(loss_obj, ckpt, ckpt_path: str):
            if not isinstance(loss_obj, JAGeRLoss):
                return
            state = ckpt.get("loss_state")
            if state is None:
                if is_main_process():
                    print(f"[warn] {ckpt_path} has no JAGeR loss_state; eval loss will use reinitialized buffers.")
                return
            if isinstance(state, dict) and "Kπ" in state:
                state = {k: v for k, v in state.items() if k != "Kπ"}
            incompat = loss_obj.load_state_dict(state, strict=False)
            missing = list(getattr(incompat, "missing_keys", []))
            unexpected = list(getattr(incompat, "unexpected_keys", []))
            if (missing or unexpected) and is_main_process():
                print(f"[warn] JAGeR loss_state mismatch for {ckpt_path}: missing={missing}, unexpected={unexpected}")

        def _metrics_args_payload():
            run_args = vars(args).copy()
            keys = [
                "loss", "joint", "mixture", "conf_gating", "reassignment",
                "lambda0", "lambda_min", "alpha", "C",
                "ce_label_smoothing", "seed",
                "model_name", "init_model_from", "max_length", "batch_size",
                "grad_accum", "epochs", "sched_epochs",
                "ens_mode", "ens_epoch_start", "ens_epoch_end", "ens_t", "ens_stride",
            ]
            return {k: run_args.get(k) for k in keys}


        # ---- Static buffer stats (λ, ρ) captured once per run/split ----
        idx_all = sorted(set(train_view_ids) | set(val_view_ids) | (set(test_view_ids) if eval_test_flag else set()))
        views = {
            "train": train_view_ids,
            "val":   val_view_ids,
            **({"test": test_view_ids} if eval_test_flag else {}),
            "all":   idx_all,
        }
        def _current_static_stats():
            return _collect_static_stats(loss_fn, views)

        static_stats = _current_static_stats()
        if is_main_process() and run_wandb is not None and not args.eval_only:
            _log_static_stats_to_wandb(static_stats)



        # -------- Eval-only fast path (no training) --------
        if args.eval_only:
            best_path = os.path.join(args.save_dir, "best.pt")
            assert os.path.exists(best_path), f"Missing {best_path}"
            ckpt = torch.load(best_path, map_location="cpu")
            unwrap(model).load_state_dict(ckpt["model"])  # type: ignore[union-attr]
            _restore_loss_state_from_ckpt(loss_fn, ckpt, best_path)
            static_stats = _current_static_stats()
            if is_main_process() and run_wandb is not None:
                _log_static_stats_to_wandb(static_stats)

            payload = None
            val_loss, vcms, vqwks, v_emds, v_macro_emd, v_tails, v_tails_avg = evaluate(
                model.module if is_dist() else model,
                dl_val, loss_fn, device, args=args
            )
            if is_main_process():
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
            if args.eval_test and dl_test is not None:
                test_loss, tcms, tqwks, t_emds, t_macro_emd, t_tails, t_tails_avg = evaluate(
                    model.module if is_dist() else model,
                    dl_test, loss_fn, device, args=args
                )
                if is_main_process() and payload is not None:
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

            if is_main_process() and payload is not None:
                payload["args"] = _metrics_args_payload()
                payload.setdefault("extra", {})["static_stats"] = static_stats
                os.makedirs(args.save_dir, exist_ok=True)
                with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
                    json.dump(payload, f, indent=2)
                print(f"[EVAL-ONLY] wrote {os.path.join(args.save_dir, 'metrics.json')}")
            return


        optim = torch.optim.AdamW(model.parameters(), fused=(device.type == "cuda"), lr=args.lr, weight_decay=args.weight_decay)

        # scheduler: 10% warmup over the chosen scheduler horizon
        total_steps = math.ceil(len(dl_train) / args.grad_accum) * args.sched_epochs
        warmup = max(1, int(0.1 * total_steps))
        sched = get_linear_schedule_with_warmup(optim, warmup, total_steps)
        if is_main_process():
            print(
                f"[sched] epochs={args.epochs} sched_epochs={args.sched_epochs} "
                f"total_steps={total_steps} warmup_steps={warmup}"
            )

        use_amp   = (device.type == "cuda")
        amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        scaler = None
        if AmpGradScaler is not None:
            scaler = AmpGradScaler(enabled=use_amp and (amp_dtype is torch.float16))

        best_qwk = float("-inf")
        improved = False
        global_step = 0
        val_pred_epochs = []  # list of {"epoch": int, "ids": np.ndarray, "y": dict, "p": dict}
        last_val_metrics = None

        def _scalar_float(x: float | torch.Tensor) -> float:
            if isinstance(x, torch.Tensor):
                return float(x.detach().cpu())
            return float(x)

        def _log_train_step(
            step_loss: float | torch.Tensor,
            epoch_num: int,
            *,
            step_loss_value: float | None = None,
            should_log: bool | None = None,
        ):
            nonlocal global_step
            if should_log is None:
                should_log = (global_step % max(1, args.log_every)) == 0
            if run_wandb is not None and should_log:
                if step_loss_value is None:
                    step_loss_value = _scalar_float(step_loss)
                log_dict = {"train/loss_step": step_loss_value, "epoch": epoch_num}
                λ = getattr(loss_fn, "λ", None)
                if isinstance(λ, torch.Tensor):
                    lam_det = λ.detach()
                    lam_min = float(lam_det.min().cpu())
                    lam_max = float(lam_det.max().cpu())
                    log_dict["train/lambda_min"] = lam_min
                    log_dict["train/lambda_max"] = lam_max
                    log_dict["jager/lambda_min"] = lam_min
                    log_dict["jager/lambda_max"] = lam_max
                lam_reg = getattr(loss_fn, "_last_lambda_reg", None)
                if isinstance(lam_reg, torch.Tensor):
                    log_dict["jager/lambda_reg"] = float(lam_reg.detach().cpu())
                elif lam_reg is not None:
                    log_dict["jager/lambda_reg"] = float(lam_reg)
                ρ = getattr(loss_fn, "ρ", None)
                if isinstance(ρ, torch.Tensor):
                    log_dict["loss/ρ"] = float(ρ.detach().mean().cpu())
                wandb.log(log_dict, step=global_step)  # type: ignore[union-attr]
            global_step += 1

        for epoch in range(1, args.epochs+1):
            if is_dist() and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            # Unfreeze at a chosen epoch (for GPU runs)
            if args.unfreeze_at_epoch == epoch:
                enc = model.module.encoder if is_dist() else model.encoder  # type: ignore[union-attr]
                for p in enc.parameters():  # type: ignore[union-attr]
                    p.requires_grad = True
                if is_main_process():
                    print(f">> Unfroze encoder at epoch {epoch}.")
            model.train()
            running = torch.zeros((), dtype=torch.float64, device=device)
            window_loss = torch.zeros_like(running)
            num_train_batches = len(dl_train)
            for step, batch in enumerate(dl_train, start=1):
                ids = batch["ids"].to(device, non_blocking=pin)
                input_ids = batch["input_ids"].to(device, non_blocking=pin)
                attention_mask = batch["attention_mask"].to(device, non_blocking=pin)

                # Decide sync scope for DDP: skip gradient all-reduce on non-boundary steps
                window_start = ((step - 1) // args.grad_accum) * args.grad_accum + 1
                accum_target = min(args.grad_accum, num_train_batches - window_start + 1)
                accum_index  = step - window_start + 1
                ddp          = (is_dist() and hasattr(model, "no_sync"))
                sync_needed  = (accum_index == accum_target)
                sync_ctx     = contextlib.nullcontext() if (not ddp or sync_needed) else model.no_sync()  # type: ignore[union-attr]

                with sync_ctx:
                    # Autocast forward (bf16 on A100)
                    with amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):  # type: ignore[attr-defined]
                        out = model(input_ids=input_ids, attention_mask=attention_mask)
                    y_pred = out["logits"].to(torch.float64) if args.loss == "jager" else out["logits"]  # cast up for JAGeR
                    # per-forward state updates in loss; pass 0-based micro-step consistent across ranks
                    micro_step = (epoch - 1) * len(dl_train) + (step - 1)
                    try:
                        loss = loss_fn(y_pred, ids, update_state=True, global_step=micro_step) / accum_target
                    except TypeError:
                        # CE baseline signature (no global_step)
                        loss = loss_fn(y_pred, ids) / accum_target
                    if scaler is not None and scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    loss_for_log = loss.detach()
                    if loss_for_log.dtype != torch.float64:
                        loss_for_log = loss_for_log.to(torch.float64)
                    running.add_(loss_for_log)
                    window_loss.add_(loss_for_log)

                if sync_needed:
                    if scaler is not None and scaler.is_enabled():
                        scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if scaler is not None and scaler.is_enabled():
                        scaler.step(optim); scaler.update()
                    else:
                        optim.step()
                    optim.zero_grad(set_to_none=True); sched.step()
                    should_log_step = (global_step % max(1, args.log_every)) == 0 or step == len(dl_train)
                    window_loss_value = None
                    if should_log_step and is_main_process():
                        window_loss_value = _scalar_float(window_loss)
                        print(f"[epoch {epoch}] step {step}/{len(dl_train)}: loss={window_loss_value:.4f} ")
                    _log_train_step(
                        window_loss,
                        epoch,
                        step_loss_value=window_loss_value,
                        should_log=should_log_step,
                    )
                    window_loss.zero_()

            train_loss_avg = (
                _scalar_float(running / max(1, math.ceil(len(dl_train)/args.grad_accum)))
                if is_main_process()
                else float("nan")
            )

            # Monitor λ range
            if is_main_process() and hasattr(loss_fn, "λ") and isinstance(getattr(loss_fn, "λ", None), torch.Tensor):
                with torch.no_grad():
                    λ = loss_fn.λ
                    lam_min = λ.min().item(); lam_max = λ.max().item()  # type: ignore[operator]
                lam_reg = getattr(loss_fn, '_last_lambda_reg', None)
                lam_reg_str = f" lambda_reg={_scalar_float(lam_reg):.6f}" if lam_reg is not None else ""
                print(f"[epoch {epoch}] train_loss(avg per step)={train_loss_avg:.4f} "
                        f"lambda[min,max]=[{lam_min:.6f},{lam_max:.6f}]{lam_reg_str}")

            val_loss, cms, qwks, emds, macro_emd, tail_recalls, tail_recall_avg, ids_e, y_e, p_e = evaluate(
                model.module if is_dist() else model,
                dl_val, loss_fn, device, args, return_payload=True
            )
            avg_qwk = float("nan")
            if is_main_process():
                val_pred_epochs.append({"epoch": int(epoch), "ids": ids_e, "y": y_e, "p": p_e})
                last_val_metrics = {
                    "loss": float(val_loss),
                    "qwk": [float(x) for x in qwks],
                    "qwk_average": float(sum(qwks) / len(qwks)) if len(qwks) else float("nan"),
                    "cm": [cm.tolist() for cm in cms],
                    "emd": [float(x) for x in emds],
                    "macroEMD": float(macro_emd),
                    "tail_recall0": [float(x) for x in tail_recalls],
                    "tail_recall0_average": float(tail_recall_avg),
                }

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
                    log = {
                        "epoch": int(epoch),
                        "train/loss_epoch": train_loss_avg,
                        "val/loss": float(val_loss),
                        "val/qwk_average": float(avg_qwk),
                        "val/macroEMD": float(macro_emd),
                        "val/mEMD": float(macro_emd),
                        "val/tail_recall0_average": float(tail_recall_avg),
                    }
                    for h, name in enumerate(head_names):
                        log[f"val/qwk_{name}"] = float(qwks[h])
                        log[f"val/emd_{name}"] = float(emds[h])
                        log[f"val/tail_recall0_{name}"] = float(tail_recalls[h])
                    wandb.log(log, step=global_step)  # type: ignore[union-attr]

                for h, name in enumerate(head_names):
                    if h >= len(cms):
                        break
                    lo = int(args.level_offset); hi = lo + K - 1
                    print(f"[{name}] confusion matrix (rows=true {lo}..{hi}, cols=pred {lo}..{hi}):")
                    cm = cms[h]
                    print("\n".join("  " + " ".join(f"{v:4d}" for v in row) for row in cm))
                # W&B heatmaps for VAL
                if run_wandb is not None:
                    _log_cms_as_wandb_images(cms, split="val", head_names=head_names, level_offset=level_offset, step=global_step)
                # --- append per-epoch metrics CSV (one row per epoch) ---
                try:
                    os.makedirs(args.save_dir, exist_ok=True)
                    metrics_path = os.path.join(args.save_dir, "epoch_metrics.csv")
                    row = {
                        "epoch": int(epoch),
                        "train_loss_avg": train_loss_avg,
                        "val_loss": float(val_loss),
                        "qwk_avg": float(avg_qwk),
                        "macroEMD": float(macro_emd),
                        "tailR0_avg": float(tail_recall_avg),
                    }
                    for h, name in enumerate(head_names):
                        row[f"qwk_{name}"] = float(qwks[h])
                        row[f"emd_{name}"] = float(emds[h])
                        row[f"tailR0_{name}"] = float(tail_recalls[h])
                    is_new = not os.path.exists(metrics_path)
                    with open(metrics_path, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(row.keys()))
                        if is_new:
                            w.writeheader()
                        w.writerow(row)
                except Exception as e:
                    print(f"[warn] failed to write epoch_metrics.csv: {e}")


            # ----- determine improvement by avg QWK (maximize), regardless of saving -----
            improved = False
            if is_main_process():
                if math.isfinite(avg_qwk):
                    if avg_qwk > best_qwk:
                        best_qwk = avg_qwk
                        improved = True
                else:
                    print(f"[warn] epoch {epoch} averageQWK is non-finite ({avg_qwk}); skipping best-model update")
            if is_dist():
                improved_t = torch.tensor(int(improved), device=device)
                dist.broadcast(improved_t, src=0)
                improved = bool(improved_t.item())

            # ----- save best.pt only if requested -----
            if improved and is_main_process() and args.save_model:
                path = os.path.join(args.save_dir, "best.pt")
                to_save = unwrap(model).state_dict()  # type: ignore[union-attr]
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save({
                    "model": to_save,
                    "loss_state": _loss_state_for_ckpt(loss_fn),
                    "config": {
                        "model_name": args.model_name,
                        "K": K,
                        "heads": num_heads,
                        "target_cols": target_cols,
                    },
                    "best_avg_qwk": float(best_qwk),
                    "epoch": int(epoch),
                }, path)
                with open(os.path.join(args.save_dir, "run_args.json"), "w") as f:
                    json.dump(vars(args), f, indent=2)
                print(f"  ↑ saved new best (avgQWK={best_qwk:.4f}) to {path}")
                if run_wandb is not None:
                    try:
                        art = wandb.Artifact(  # type: ignore[union-attr]
                            f"best-{args.loss}-j{int(args.joint)}-m{int(args.mixture)}"
                            f"-cg{int(args.conf_gating)}-ra{int(args.reassignment)}",
                            type="model",
                            metadata={"epoch": int(epoch), "best_avg_qwk": float(best_qwk)},
                        )
                        art.add_file(os.path.join(args.save_dir, "best.pt"))
                        wandb.log_artifact(art)  # type: ignore[union-attr]
                    except Exception as e:
                        print("[W&B] artifact log skipped:", e)


            # TEST evaluation on improve (even if we didn't save): reload best.pt iff it exists
            if args.eval_test and improved and dl_test is not None:
                best_path = os.path.join(args.save_dir, "best.pt")
                if is_dist():
                    dist.barrier()
                if args.save_model:
                    if os.path.exists(best_path):
                        ckpt = torch.load(best_path, map_location="cpu")
                        unwrap(model).load_state_dict(ckpt["model"])  # type: ignore[union-attr]
                        _restore_loss_state_from_ckpt(loss_fn, ckpt, best_path)
                    elif is_main_process():
                        print(f"[warn] missing {best_path}; evaluating current in-memory weights")
                elif is_main_process():
                    print("[TEST] evaluating current in-memory weights (unsaved best)")

                # evaluate returns: loss, cms, qwks, emds, macro_emd, tail_recalls, tail_recall_avg
                test_loss, tcms, tqwks, temds, tmacro_emd, ttails, ttails_avg = evaluate(
                    unwrap(model), dl_test, loss_fn, device, args
                )
                if is_main_process():
                    current_static_stats = _current_static_stats()
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
                        log = {
                            "epoch": int(epoch),
                            "test/loss": float(test_loss),
                            "test/qwk_average": float(tavg_qwk),
                            "test/macroEMD": float(tmacro_emd),
                            "test/mEMD": float(tmacro_emd),
                            "test/tail_recall0_average": float(ttails_avg),
                        }
                        for h, name in enumerate(head_names):
                            log[f"test/qwk_{name}"] = float(tqwks[h])
                            log[f"test/emd_{name}"] = float(temds[h])
                            log[f"test/tail_recall0_{name}"] = float(ttails[h])
                        wandb.log(log, step=global_step)  # type: ignore[union-attr]
                        # W&B heatmaps for TEST
                        _log_cms_as_wandb_images(tcms, split="test", head_names=head_names, level_offset=level_offset, step=global_step)

                    # write compact metrics.json
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
                        "args": _metrics_args_payload()
                    }
                    metrics.setdefault("extra", {})["static_stats"] = current_static_stats
                    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
                        json.dump(metrics, f, indent=2)
                    print(f"[TEST] wrote {os.path.join(args.save_dir, 'metrics.json')}")


            if is_dist(): 
                dist.barrier()

        # ---------- Probability ensembling on VAL (OOF for this fold) ----------
        if is_main_process() and len(val_pred_epochs) > 0:
            ens_mode = str(getattr(args, "ens_mode", "fixed_range"))
            S = max(1, int(getattr(args, "ens_stride", 1)))
            if ens_mode == "fixed_range":
                ens_epoch_start = max(1, int(getattr(args, "ens_epoch_start", 8)))
                ens_epoch_end = int(getattr(args, "ens_epoch_end", 14))
                if ens_epoch_end < ens_epoch_start:
                    raise ValueError(
                        f"Invalid fixed OOF range: start={ens_epoch_start}, end={ens_epoch_end}"
                    )
                keep = [
                    rec
                    for rec in val_pred_epochs
                    if ens_epoch_start <= int(rec["epoch"]) <= ens_epoch_end
                ]
                selected_epoch_set = [int(rec["epoch"]) for rec in keep]
                expected_epoch_set = list(range(ens_epoch_start, ens_epoch_end + 1))
                if selected_epoch_set != expected_epoch_set:
                    seen_epochs = [int(rec["epoch"]) for rec in val_pred_epochs]
                    raise RuntimeError(
                        "Fixed OOF range requires the full contiguous epoch set "
                        f"{expected_epoch_set}, but found {selected_epoch_set}. Seen epochs={seen_epochs}"
                    )
                keep = keep[::S]
                window_tag = f"E{ens_epoch_start}-{ens_epoch_end}"
            elif ens_mode == "tail":
                T = max(1, int(getattr(args, "ens_t", 10)))
                keep = [rec for rec in val_pred_epochs[-T:]][::S]
                window_tag = f"T{T}"
            else:
                raise ValueError(f"Unsupported ens_mode={ens_mode!r}")

            selected_epochs = [int(rec["epoch"]) for rec in keep]
            print(f"[oof] ensembling val predictions from epochs={selected_epochs} mode={ens_mode}")
            base_ids = keep[0]["ids"]
            for rec in keep[1:]:
                if not np.array_equal(base_ids, rec["ids"]):
                    raise RuntimeError("Validation ids changed across epochs; cannot ensemble. Check val shuffle.")
            head_names = list(keep[0]["p"].keys())
            avg_p = {}
            for h in head_names:
                stack = np.stack([rec["p"][h] for rec in keep], axis=0)  # (Ekeep, N, C)
                avg_p[h] = stack.mean(axis=0)                            # (N, C)
            y_true = keep[0]["y"]
            ids    = base_ids
            rows = []
            fold_name = os.path.basename(os.path.abspath(args.save_dir))
            for i in range(len(ids)):
                row = {"id": int(ids[i]), "fold": fold_name}
                for h in head_names:
                    p = avg_p[h][i]                 # (C,)
                    C = p.shape[-1]
                    classes = np.arange(int(args.level_offset), int(args.level_offset) + C, dtype=float)
                    exp = float((p * classes).sum())
                    pred = int(classes[int(np.argmax(p))])
                    yt = int(y_true[h][i])
                    row[f"{h}_y"]   = yt
                    row[f"{h}_exp"] = exp
                    row[f"{h}_pred"]= pred
                    for j, c in enumerate(classes):
                        row[f"{h}_p{int(c)}"] = float(p[j])
                rows.append(row)
            os.makedirs(args.save_dir, exist_ok=True)
            oof_path = os.path.join(args.save_dir, f"oof_val_{window_tag}.csv")
            with open(oof_path, "w", newline="") as f:
                fieldnames = list(rows[0].keys()) if rows else ["id"]
                w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
                for r in rows: w.writerow(r)
            print(f"[oof] wrote ensembled OOF-val predictions: {oof_path}")


        if is_main_process() and last_val_metrics is not None:
            try:
                final_static_stats = _current_static_stats()
                p = os.path.join(args.save_dir, "metrics.json")
                obj = {}
                if os.path.exists(p):
                    with open(p) as f:
                        try:
                            obj = json.load(f)
                        except Exception:
                            obj = {}
                # If best-epoch VAL metrics were already written (from eval_test on improve),
                # keep them as-is and store final-epoch VAL metrics separately.
                if "val" in obj:
                    obj.setdefault("extra", {})["val_final"] = last_val_metrics
                else:
                    obj["val"] = last_val_metrics

                obj["args"] = _metrics_args_payload()
                obj.setdefault("extra", {})["static_stats"] = obj.get("extra", {}).get("static_stats", final_static_stats)
                with open(p, "w") as f:
                    json.dump(obj, f, indent=2)
                print(f"[VAL] updated {p}")
            except Exception as e:
                print(f"[warn] could not append VAL metrics: {e}")


        print("Done.")
    

    finally:
        if run_wandb is not None:
            try:
                run_wandb.finish()
            except Exception:
                pass

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
