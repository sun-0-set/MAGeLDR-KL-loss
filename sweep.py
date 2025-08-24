# sweep.py
import math, os, csv, subprocess, itertools, json, shlex, sys, time, argparse
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from collections import Counter
import numpy as np

try:
    import torch
    NPROC = max(1, torch.cuda.device_count())
    CUDA_DEVS = list(range(torch.cuda.device_count()))
except Exception:
    # Fallback (or override with env var)
    NPROC = int(os.environ.get("NPROC", "1"))
    CUDA_DEVS = [int(x) for x in os.environ.get("DEVICES", "0").split(",") if x.strip().isdigit()]

EPOCHS = int(os.environ.get("INNER_EPOCHS", "7"))
INNER_EPOCHS = EPOCHS  # will be overridden by --inner_epochs at parse time


# ===== module-level defaults (overridden in main from CLI args) =====
BS = 2                # --batch_size
GRAD_ACCUM = 8        # --grad_accum
LR = None             # --lr (only appended if not None)
NUM_WORKERS = 4       # --num_workers
PREFETCH_FACTOR = 4   # --prefetch_factor

# === grids (Phase 1) ===
K = 5
# MAGe
LAMBDA0S        = [0.3, 3.0]    
ALPHAS          = [1.25, 2.0]    
C_VALUES_MAGER  = [0.3, 1]    
# ALDR-KL 
C_VALUES_ALDR   = [0.3, 1]     
# CE 
CE_LABEL_SMOOTH = [0.0, 0.1] 



# If a run's tag/save_dir basename is in FINALISTS, append --save_model (phase 2).
FINALISTS = set()
FINALISTS_FILE = Path("sweeps/finalists.txt")
if FINALISTS_FILE.exists():
    for ln in FINALISTS_FILE.read_text().splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            FINALISTS.add(ln)
if os.environ.get("FINALISTS"):
    FINALISTS.update(x.strip() for x in os.environ["FINALISTS"].split(",") if x.strip())


SAVE_ROOT = Path("sweeps")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
SPLIT_FILE = "splits/dress_seed42.json"   # frozen split you generated
DEFAULT_TSV = "../data/DREsS/DREsS_New_cleaned.tsv"

# ---- helpers ----
def run(cmd, save_dir, env: Dict[str, str] | None = None):
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(x) for x in cmd]  # <- make every arg a string (handles ints/Paths)
    print(">>", " ".join(shlex.quote(x) for x in cmd))
    env2 = os.environ.copy()
    if env: env2.update(env)
    subprocess.run(cmd, check=True, env=env2)

def _run_pool(jobs: List[dict], concurrency: int, device_ids: List[int]):
    """
    Run many 1-GPU jobs concurrently by pinning CUDA_VISIBLE_DEVICES.
    Each `job` is a dict: {"cmd": [...], "save_dir": Path}
    """
    if concurrency <= 1 or len(device_ids) <= 1:
        # sequential fallback
        for j in jobs:
            run(j["cmd"], j["save_dir"])
        return
    # Only safe when each job uses nproc==1
    active = []
    free = device_ids[:]
    i = 0
    def launch(j, dev):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(dev)
        # keep CPU thread count modest and give each torchrun a unique rendezvous port
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MASTER_ADDR", "127.0.0.1")
        env["MASTER_PORT"] = str(29400 + int(dev))
        cmd = [str(x) for x in j["cmd"]]
        print(f">> [GPU {dev}] " + " ".join(shlex.quote(x) for x in cmd))
        p = subprocess.Popen(cmd, env=env)  # non-blocking
        return {"proc": p, "gpu": dev, "save_dir": j["save_dir"], "cmd": cmd}
    while i < len(jobs) or active:
        # launch up to concurrency
        while i < len(jobs) and free and len(active) < concurrency:
            dev = free.pop(0)
            active.append(launch(jobs[i], dev)); i += 1
        # poll
        time.sleep(0.5)
        still = []
        for a in active:
            rc = a["proc"].poll()
            if rc is None:
                still.append(a); continue
            # finished
            free.append(a["gpu"])
            if rc != 0:
                print(f"[WARN] job failed (rc={rc}) in {a['save_dir']}")
        active = still

def mager_cmd(distribution, lambda0, alpha, C, save_dir, nproc: int = NPROC, eval_test: bool = True):
    return [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={nproc}",
        "train.py",
        "--loss", "mager",
        "--distribution", distribution,
        "--lambda0", str(lambda0),
        "--alpha", str(alpha),
        "--C", str(C),
        "--split_file", SPLIT_FILE,
    ] + (["--eval_test"] if eval_test else []) + [
        "--save_dir", str(save_dir),
        "--epochs", INNER_EPOCHS, "--batch_size", BS, "--grad_accum", GRAD_ACCUM,
        "--model_name","../models/deberta-v3-large",
        "--max_length","1024",
        "--num_workers", str(NUM_WORKERS),
        "--prefetch_factor", str(PREFETCH_FACTOR),
    ] + (["--lr", str(LR)] if LR is not None else [])

def aldrkl_cmd(lambda0, alpha, C, save_dir, nproc: int = NPROC, eval_test: bool = True):
    return [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={nproc}",
        "train.py",
        "--loss", "aldrkl",
        "--lambda0", str(lambda0),
        "--alpha", str(alpha),
        "--C", str(C),
        "--split_file", SPLIT_FILE,
    ] + (["--eval_test"] if eval_test else []) + [
        "--save_dir", str(save_dir),
        "--epochs", INNER_EPOCHS, "--batch_size", BS, "--grad_accum", GRAD_ACCUM,
        "--model_name", "../models/deberta-v3-large",
        "--max_length", "1024",
        "--num_workers", str(NUM_WORKERS),
        "--prefetch_factor", str(PREFETCH_FACTOR),
    ] + (["--lr", str(LR)] if LR is not None else [])

def ce_cmd(label_smoothing, save_dir, nproc: int = NPROC, eval_test: bool = True):
    return [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={nproc}",
        "train.py",
        "--loss", "ce",
        "--ce_label_smoothing", str(label_smoothing),
        "--split_file", SPLIT_FILE,
    ] + (["--eval_test"] if eval_test else []) + [
        "--save_dir", str(save_dir),
        "--epochs", INNER_EPOCHS, "--batch_size", BS, "--grad_accum", GRAD_ACCUM,
        "--model_name", "../models/deberta-v3-large",
        "--max_length", "1024",
        "--num_workers", str(NUM_WORKERS),
        "--prefetch_factor", str(PREFETCH_FACTOR),
    ] + (["--lr", str(LR)] if LR is not None else [])

# -----------------------------
# Joint labels & nested splits
# -----------------------------
def _load_joint(tsv_path: str):
    import pandas as pd
    df = pd.read_csv(tsv_path, sep="\t")
    c = df["content"].astype(int).to_numpy()
    o = df["organization"].astype(int).to_numpy()
    l = df["language"].astype(int).to_numpy()
    K = 5
    joint3 = (c-1)*K*K + (o-1)*K + (l-1)          # content+org+lang (most specific)
    joint2 = (c-1)*K + (o-1)                      # content+org
    joint1 = (c-1)                                 # content
    return len(df), {"joint3": joint3, "joint2": joint2, "joint1": joint1}

def _labels_ok(y: list[int] | np.ndarray, n_splits: int) -> bool:
    """For stratification to work: need ≥2 classes, and at least:
       - 2 per class if we'll do a single stratified shuffle split (J==1),
       - n_splits per class for KFold otherwise."""
    counts = Counter(map(int, y))
    if len(counts) < 2:
        return False
    min_needed = 2 if n_splits == 1 else n_splits
    return min(counts.values()) >= min_needed

def _pick_labels_for_outer(labels: dict, n_splits: int) -> tuple[np.ndarray, str, bool]:
    """Choose best available labels for outer StratifiedKFold, else signal unstratified."""
    for key in ("joint3", "joint2", "joint1"):
        y = np.asarray(labels[key])
        if _labels_ok(y, n_splits):
            return y, key, True
    # fallback: unstratified
    return np.arange(len(next(iter(labels.values())))), "kfold", False

def _pick_labels_for_subset(indices: list[int], labels: dict, n_splits: int) -> tuple[np.ndarray, str, bool]:
    """Choose best labels for inner StratifiedKFold on a subset of rows."""
    idx = np.asarray(indices, dtype=int)
    for key in ("joint3", "joint2", "joint1"):
        y = np.asarray(labels[key])[idx]
        if _labels_ok(y, n_splits):
            return y, key, True
    # fallback: unstratified
    return np.arange(len(idx)), "kfold", False

@dataclass
class NestedSpec:
    K: int
    J: int
    seed: int
    val_frac_refit: float = 0.1

def _build_nested_indices(tsv: str, spec: NestedSpec):
    """Return a dict with outer folds and inner folds, all using joint stratification."""
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, KFold, ShuffleSplit
    N, labels = _load_joint(tsv)
    idx_all = list(range(N))
    outer = []
    # ---- OUTER split (with fallback) ----
    y_outer, name_outer, strat_ok = _pick_labels_for_outer(labels, spec.K)
    if strat_ok:
        splitter_outer = StratifiedKFold(n_splits=spec.K, shuffle=True, random_state=spec.seed)
        split_iter = splitter_outer.split(idx_all, y_outer)
    else:
        splitter_outer = KFold(n_splits=spec.K, shuffle=True, random_state=spec.seed)
        split_iter = splitter_outer.split(idx_all)
        print(f"[nested] OUTER using fallback '{name_outer}' (unstratified KFold).")
    for f, (trainval_idx, test_idx) in enumerate(split_iter):
        trainval_idx = list(map(int, trainval_idx))
        test_idx     = list(map(int, test_idx))
        # ---- INNER split (J==1 handled via ShuffleSplit) ----
        y_inner, name_inner, inner_ok = _pick_labels_for_subset(trainval_idx, labels, spec.J)
        inner = []
        if spec.J == 1:
            val_frac = max(0.001, float(spec.val_frac_refit))
            if inner_ok:
                try:
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=spec.seed+f)
                    inner_iter = [next(sss.split(trainval_idx, y_inner))]
                except ValueError as e:
                    # e.g., least-populated class has only 1 member → can’t stratify
                    ss = ShuffleSplit(n_splits=1, test_size=val_frac, random_state=spec.seed+f)
                    inner_iter = [next(ss.split(np.arange(len(trainval_idx))))]
                    print(f"[nested] INNER(o={f}) warn: stratified J=1 failed ({e}); using unstratified ShuffleSplit.")
            else:
                ss = ShuffleSplit(n_splits=1, test_size=val_frac, random_state=spec.seed+f)
                inner_iter = [next(ss.split(np.arange(len(trainval_idx))))]
                print(f"[nested] INNER(o={f}) fallback: unstratified ShuffleSplit (J=1).")
        else:
            if inner_ok:
                # Proper K-fold for J>1
                skf_inner = StratifiedKFold(n_splits=spec.J, shuffle=True, random_state=spec.seed+f)
                inner_iter = skf_inner.split(trainval_idx, y_inner)
            else:
                kf_inner = KFold(n_splits=spec.J, shuffle=True, random_state=spec.seed+f)
                inner_iter = kf_inner.split(trainval_idx)
                print(f"[nested] INNER(o={f}) using fallback 'kfold' (unstratified).")
        for j, (tr, va) in enumerate(inner_iter):
            inner_train = [int(trainval_idx[i]) for i in tr]
            inner_val   = [int(trainval_idx[i]) for i in va]
            inner.append({"train": inner_train, "val": inner_val})
        # ---- Refit split: small stratified val slice IF feasible, else random shuffle split ----
        try:
            # Try to stratify using the same label choice* as inner
            lab_for_refit, _, ok_refit = _pick_labels_for_subset(trainval_idx, labels, int(1/spec.val_frac_refit) if spec.val_frac_refit>0 else 10)
            if ok_refit:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=spec.val_frac_refit, random_state=spec.seed+100+f)
                tr_refit, va_refit = next(sss.split(trainval_idx, lab_for_refit))
            else:
                raise ValueError("refit strat not feasible")
                print("refit strat not feasible")
        except Exception:
            rng = np.random.default_rng(spec.seed+100+f)
            perm = rng.permutation(len(trainval_idx))
            n_val = max(1, int(round(spec.val_frac_refit * len(trainval_idx))))
            va_refit = perm[:n_val]; tr_refit = perm[n_val:]
            print(f"[nested] REFIT(o={f}) fallback to random val slice (size={n_val}).")
        refit = {
            "train": [int(trainval_idx[i]) for i in tr_refit],
            "val":   [int(trainval_idx[i]) for i in va_refit],
            "test":  test_idx
        }
        outer.append({
            "test": test_idx,
            "trainval": trainval_idx,
            "inner": inner,
            "refit": refit,
            "outer_strat": name_outer,
            "inner_strat": name_inner
        })
    return {"outer": outer, "seed": spec.seed, "K": spec.K, "J": spec.J}

def _write_split_json(path: Path, split: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f)
    return str(path)

def _score_from_val_json(p: Path) -> float:
    try:
        m = json.loads(p.read_text())
        val = m.get("val", {})
        if "qwk" in val and isinstance(val["qwk"], (list, tuple)) and val["qwk"]:
            return float(sum(val["qwk"]) / len(val["qwk"]))
        if "micro_overall" in val:
            return float(val["micro_overall"])
        if "loss" in val:
            return -float(val["loss"])
    except Exception:
        pass
    return float("-inf")


# Phase helpers (selection/rerun)
# -----------------------------
def _read_metrics(save_dir: Path):
    p = save_dir / "metrics.json"
    if not p.exists():
        return None, None
    try:
        m = json.loads(p.read_text())
        return m.get("args", {}), m.get("val", {})
    except Exception:
        return None, None

def _score_from_val(val: dict) -> float:
    """
    Single scalar score for ranking finalists.
    Prefers mean QWK across heads; falls back to micro_overall; else negative loss.
    """
    if not val:
        return float("-inf")
    qwk = val.get("qwk")
    if isinstance(qwk, (list, tuple)) and qwk:
        return float(sum(qwk) / len(qwk))
    if "micro_overall" in val:
        return float(val["micro_overall"])
    if "loss" in val:
        return -float(val["loss"])
    return float("-inf")

def _family_from(args: dict, tag: str) -> str:
    loss = (args or {}).get("loss")
    if not loss:
        if "mager" in tag: loss = "mager"
        elif "aldrkl" in tag: loss = "aldrkl"
        else: loss = "ce"
    if loss == "mager":
        dist = (args or {}).get("distribution")
        if not dist:
            dist = "mixture" if "mixture" in tag else "uniform"
        return f"mager-{dist}"
    return loss

def _build_leaderboard_and_finalists(save_root: Path, topk_global: int, topk_family: int) -> List[str]:
    """Scan sweeps/*/metrics.json, write leaderboard.json & finalists.txt, return finalists list."""
    candidates = []
    for sd in sorted(save_root.iterdir()):
        if not sd.is_dir(): continue
        args, val = _read_metrics(sd)
        if val is None:   # missing or unreadable
            continue
        score = _score_from_val(val)
        family = _family_from(args, sd.name)
        candidates.append({"tag": sd.name, "family": family, "score": score})
    # Global ranking
    g = sorted(candidates, key=lambda x: x["score"], reverse=True)
    # Per-family ranking
    fam_map: Dict[str, List[dict]] = {}
    for c in candidates:
        fam_map.setdefault(c["family"], []).append(c)
    for k in fam_map:
        fam_map[k].sort(key=lambda x: x["score"], reverse=True)
    finalists = set(x["tag"] for x in g[:topk_global])
    for fam, lst in fam_map.items():
        finalists.update(x["tag"] for x in lst[:topk_family])
    # Persist
    (save_root / "leaderboard.json").write_text(json.dumps({"global": g, "by_family": fam_map}, indent=2))
    (save_root / "finalists.txt").write_text("\n".join(sorted(finalists)) + "\n")
    print(f"[select] finalists → {save_root/'finalists.txt'}  (global={topk_global}, per-family={topk_family})")
    return sorted(finalists)

def _read_index_map(save_root: Path) -> Dict[str, dict]:
    idx = {}
    p = save_root / "index.csv"
    if not p.exists():
        return idx
    with p.open() as f:
        r = csv.DictReader(f)
        for row in r:
            idx[row["tag"]] = row
    return idx

def _reconstruct_cmd_from_index(row: dict, save_dir: Path, nproc: int) -> List[str]:
    loss = row["loss"]
    if loss == "mager":
        return mager_cmd(row["distribution"], row["lambda0"], row["alpha"], row["C"], save_dir, nproc=nproc)
    if loss == "aldrkl":
        return aldrkl_cmd(row["lambda0"], row["alpha"], row["C"], save_dir, nproc=nproc)
    if loss == "ce":
        return ce_cmd(row["label_smoothing"], save_dir, nproc=nproc)
    raise ValueError(f"Unknown loss in index row: {loss}")

# Treat presence of metrics.json as "done" for resuming.
def _already_done(save_dir: Path) -> bool:
    return (save_dir / "metrics.json").exists()

def _phase_nested(tsv: str, args):
    """
    Nested CV:
      - Build joint-stratified outer K and inner J splits
      - For each outer fold, score each config by mean(inner-val) across J
      - Refit best config on outer trainval (with small val) and evaluate on outer test
    """
    spec = NestedSpec(K=int(args.nested_k), J=int(args.nested_j), seed=int(args.nested_seed),
                      val_frac_refit=float(args.nested_val_frac))
    nested = _build_nested_indices(tsv, spec)
    nest_dir = SAVE_ROOT / f"nested_K{spec.K}_J{spec.J}_s{spec.seed}"
    nest_dir.mkdir(parents=True, exist_ok=True)
    (nest_dir / "splits.json").write_text(json.dumps(nested, indent=2))

    # Build the full Phase-1 grid once
    grid = []
    for distribution in ("mixture", "uniform"):
        for l0, a, cval in itertools.product(LAMBDA0S, ALPHAS, C_VALUES_MAGER):
            grid.append(("mager", {"distribution": distribution, "lambda0": l0, "alpha": a, "C": cval}))
    for l0, a, cval in itertools.product(LAMBDA0S, ALPHAS, C_VALUES_ALDR):
        grid.append(("aldrkl", {"lambda0": l0, "alpha": a, "C": cval}))
    for ls in CE_LABEL_SMOOTH:
        grid.append(("ce", {"label_smoothing": ls}))

    summary_rows = []
    # new: keep detailed inner scores per config to build a leaderboard
    from collections import defaultdict
    inner_rows = []      # rows: {"outer": o, "config": key, "inner": j, "val_score": sc}
    select_counts = defaultdict(int)
    selected_per_fold = []  # rows: {"outer": o, "config": tag, "mean_inner": best_score}
    for f, outer in enumerate(nested["outer"]):
        fold_dir = nest_dir / f"outer{f}"
        fold_dir.mkdir(exist_ok=True)
        # prepare inner split files
        inner_split_paths = []
        for j, inner in enumerate(outer["inner"]):
            sp = {"train": inner["train"], "val": inner["val"], "test": []}
            pth = _write_split_json(fold_dir / "splits" / f"inner{j}.json", sp)
            inner_split_paths.append(pth)

        # build all inner jobs for this outer fold, then run them concurrently (1 GPU/job)
        cfg2tags = {}
        jobs = []
        for (loss, hp) in grid:
            key = None
            for j, spath in enumerate(inner_split_paths):
                if loss == "mager":
                    dist = hp["distribution"]; l0 = hp["lambda0"]; a = hp["alpha"]; C = hp["C"]
                    tag = f"mager-{dist}_l{l0}_a{a}_C{C}_o{f}_j{j}"
                    sd = fold_dir / "inner" / tag
                    cmd = mager_cmd(dist, str(l0), str(a), str(C), sd, nproc=1, eval_test=False)
                    key = ("mager", dist, l0, a, C)
                elif loss == "aldrkl":
                    l0 = hp["lambda0"]; a = hp["alpha"]; C = hp["C"]
                    tag = f"aldrkl_l{l0}_a{a}_C{C}_o{f}_j{j}"
                    sd = fold_dir / "inner" / tag
                    cmd = aldrkl_cmd(str(l0), str(a), str(C), sd, nproc=1, eval_test=False)
                    key = ("aldrkl", l0, a, C)
                else:
                    ls = hp["label_smoothing"]
                    tag = f"ce_ls{ls}_o{f}_j{j}"
                    sd = fold_dir / "inner" / tag
                    cmd = ce_cmd(str(ls), sd, nproc=1, eval_test=False)
                    key = ("ce", ls)
                # Inner-CV: keep metrics.json strictly VAL for scoring
                cmd += ["--split_file", spath]
                if args.wandb_inner:
                    cmd.append("--wandb")
                cfg2tags.setdefault(key, []).append(sd)
                # Resume: only launch if this inner job hasn't produced metrics yet
                if not _already_done(sd):
                    jobs.append({"cmd": cmd, "save_dir": sd})
                else:
                    print(f"[resume] skip inner job (exists): {sd}")
        if not args.dry_run:
            _run_pool(jobs, concurrency=min(args.concurrency, len(CUDA_DEVS)), device_ids=CUDA_DEVS or [0])
        else:
            for j in jobs:
                print(">>", " ".join(shlex.quote(str(x)) for x in j["cmd"]))
        # score each config by mean(inner) after all jobs finish
        best_score, best_cfg = float("-inf"), None
        for key, sds in cfg2tags.items():
            inner_scores = [ _score_from_val_json(sd / "metrics.json") for sd in sds ]
            # log raw inner scores per config (one row per inner fold)
            for j_idx, sc in enumerate(inner_scores):
                inner_rows.append({"outer": f, "config": key, "inner": j_idx, "val_score": sc})
            mean_sc = sum(inner_scores)/max(1,len(inner_scores))
            if mean_sc > best_score:
                best_score, best_cfg = mean_sc, key
        if best_cfg is not None:
            select_counts[best_cfg] += 1
            selected_per_fold.append({"outer": f, "config": best_cfg, "mean_inner": round(best_score,6)})
        else:
            print(f"[nested] o{f}: no inner metrics found (dry_run or missing metrics); skipping refit.")
            continue

        # ---- Refit best on train+val with small val; evaluate on test ----
        refit_split = outer["refit"]
        sp_refit = {"train": refit_split["train"], "val": refit_split["val"], "test": refit_split["test"]}
        refit_path = _write_split_json(fold_dir / "splits" / "refit.json", sp_refit)
        # best_cfg keys reconstructed for refit
        if best_cfg[0] == "mager":
            _, dist, l0, a, C = best_cfg
            tag = f"best_o{f}_mager-{dist}_l{l0}_a{a}_C{C}"
            sd = fold_dir / "best" / tag
            cmd = mager_cmd(dist, str(l0), str(a), str(C), sd, nproc=max(1, args.nproc))
        elif best_cfg[0] == "aldrkl":
            _, l0, a, C = best_cfg
            tag = f"best_o{f}_aldrkl_l{l0}_a{a}_C{C}"
            sd = fold_dir / "best" / tag
            cmd = aldrkl_cmd(str(l0), str(a), str(C), sd, nproc=max(1, args.nproc))
        else:
            _, ls = best_cfg
            tag = f"best_o{f}_ce_ls{ls}"
            sd = fold_dir / "best" / tag
            cmd = ce_cmd(str(ls), sd, nproc=max(1, args.nproc))
        # drive checkpointing on small val, then run TEST; also log per-epoch stats (refit only)
        cmd += ["--split_file", refit_path, "--save_model", "--epochs", str(args.final_epochs), "--log_epoch_stats"]
        if args.wandb:
            cmd.append("--wandb")
        # Resume: if refit/test already wrote metrics.json, don't rerun
        if _already_done(sd):
            print(f"[resume] skip refit/test (exists): {sd}")
            # still proceed to collect summary below
            pass
        elif args.dry_run:
            print(">>", " ".join(shlex.quote(str(x)) for x in cmd))
            # no files written in dry_run → record placeholder row and continue
            summary_rows.append({
                "outer_fold": f, "tag": tag, "score_inner_mean": round(best_score, 6),
                "test_micro_overall": None, "test_loss": None, "test_qwk_mean": None,
            })
            continue
        else:
            run(cmd, sd)
        # pick headline numbers for a tiny summary CSV (robust read)
        try:
            m = json.loads((sd / "metrics.json").read_text())
            test = m.get("test", {}) or {}
        except Exception as e:
            print(f"[nested] warn: cannot read metrics for outer {f}: {e}")
            test = {}
        summary_rows.append({
            "outer_fold": f,
            "tag": tag,
            "score_inner_mean": round(best_score, 6),
            "test_micro_overall": test.get("micro_overall", None),
            "test_loss": test.get("loss", None),
            "test_qwk_mean": (sum(test.get("qwk", []) )/3.0 if test.get("qwk") else None),
        })
        # continue outer loop

    # write nested summary
    import csv, statistics as stats
    with open(nest_dir / "nested_summary.csv", "w", newline="") as f:
        cols = ["outer_fold","tag","score_inner_mean","test_micro_overall","test_loss","test_qwk_mean"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(summary_rows)
    print(f"[nested] wrote {nest_dir/'nested_summary.csv'}")
    
    # 1) raw inner scores (one row per (outer fold, config, inner fold))
    with open(nest_dir / "nested_inner_scores.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["outer", "config", "inner", "val_score"])
        w.writeheader()
        w.writerows(inner_rows)

    # 2) aggregate by config across all outer×inner folds
    from collections import defaultdict
    by_cfg = defaultdict(list)
    for r in inner_rows:
        if r["val_score"] is not None:
            by_cfg[r["config"]].append(float(r["val_score"]))

    agg_rows = []
    for cfg, vals in by_cfg.items():
        vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
        mu = stats.fmean(vals) if vals else float("nan")
        sd = stats.pstdev(vals) if len(vals) > 1 else (0.0 if vals else float("nan"))
        agg_rows.append({
            "config": cfg,
            "inner_mean": mu,
            "inner_std": sd,
            "n": len(vals),
            "selected_count": select_counts.get(cfg, 0),
        })
    agg_rows.sort(key=lambda r: r["inner_mean"], reverse=True)

    with open(nest_dir / "nested_inner_leaderboard.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["config", "inner_mean", "inner_std", "n", "selected_count"])
        w.writeheader()
        w.writerows(agg_rows)

    # 3) how often each config was selected as the fold winner
    with open(nest_dir / "nested_selection_counts.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["config", "selected_count"])
        w.writeheader()
        for cfg, cnt in sorted(select_counts.items(), key=lambda x: (-x[1], x[0])):
            w.writerow({"config": cfg, "selected_count": cnt})

    # 4) which config won each outer fold (and its mean inner score)
    with open(nest_dir / "nested_selected_per_fold.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["outer", "config", "mean_inner"])
        w.writeheader()
        w.writerows(selected_per_fold)
    


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["1","2","sweep","rerun","nested"], default="1",
                    help="1/sweep: run grid & write finalists; 2/rerun: re-run finalists with --save_model")
    ap.add_argument("--nproc", type=int, default=NPROC, help="Processes per job (DDP). Use 1 for sharded 1-GPU jobs.")
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("CONCURRENCY", "1")),
                    help="Max number of jobs to run concurrently (only for nproc=1).")
    ap.add_argument("--devices", type=str, default=os.environ.get("DEVICES", ""),
                    help='Comma list of visible GPU ids to shard across, e.g. "0,1". Default = all.')

    ap.add_argument("--topk_global", type=int, default=int(os.environ.get("TOPK_GLOBAL", "5")))
    ap.add_argument("--topk_family", type=int, default=int(os.environ.get("TOPK_FAMILY", "2")))
    ap.add_argument("--final_epochs", type=int, default=10,
                    help="Epochs for finalist re-runs (phase 2). Defaults to same as EPOCHS.")
    ap.add_argument("--inner_epochs", type=int, default=5,
                    help="Epochs for inner CV / ranking jobs.")
    ap.add_argument("--dry_run", action="store_true", help="Print what would run, don’t execute.")
    # W&B control: by default we only log refit. Use --wandb_inner to also log inner folds.
    ap.add_argument("--wandb", action="store_true", help="Enable W&B logging for refit runs")
    ap.add_argument("--wandb_inner", action="store_true", help="Also enable W&B logging for inner-CV runs")
    # nested params
    ap.add_argument("--tsv", type=str, default=DEFAULT_TSV, help="TSV path for nested joint labels")
    ap.add_argument("--nested_k", type=int, default=5, help="Outer folds")
    ap.add_argument("--nested_j", type=int, default=3, help="Inner folds")
    ap.add_argument("--nested_seed", type=int, default=42, help="Seed for K/J splits")
    ap.add_argument("--nested_val_frac", type=float, default=0.1, help="Val frac for outer refit checkpointing")
    ap.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    ap.add_argument("--prefetch_factor", type=int, default=4, help="Number of batches to prefetch")
    ap.add_argument('--batch_size', type=int, default=2, help="Batch size for training")
    ap.add_argument('--grad_accum', type=int, default=8, help="Gradient accumulation steps")
    ap.add_argument('--lr', type=float, default=None, help='Override learning rate for all runs')

    args = ap.parse_args()
    # make inner-epoch choice visible to command builders
    global INNER_EPOCHS
    INNER_EPOCHS = args.inner_epochs
    # Safety: the pool only supports 1-GPU jobs; if user asks for DDP + concurrency, run sequentially.
    if args.nproc > 1 and args.concurrency > 1:
        print("[warn] concurrency>1 is only supported with --nproc=1; forcing concurrency=1.")
        args.concurrency = 1
    # Optional device override for the pool
    global CUDA_DEVS
    if args.devices.strip():
        CUDA_DEVS = [int(x) for x in args.devices.split(",") if x.strip().isdigit()]

    global BS
    if args.batch_size is not None:
        BS = args.batch_size

    global GRAD_ACCUM
    if args.grad_accum is not None:
        GRAD_ACCUM = args.grad_accum
        
    global LR
    if args.lr is not None:
        LR = args.lr

    global NUM_WORKERS
    if args.num_workers is not None:
        NUM_WORKERS = args.num_workers

    global PREFETCH_FACTOR
    if args.prefetch_factor is not None:
        PREFETCH_FACTOR = args.prefetch_factor

    if args.phase == "nested":
        return _phase_nested(args.tsv, args)


    # phase 2: re-run finalists (micro-grid for MAGe/ALDR; exact for CE)
    if args.phase in ("2", "rerun"):
        if not FINALISTS and FINALISTS_FILE.exists():
            # load from file we wrote in phase 1
            for ln in FINALISTS_FILE.read_text().splitlines():
                ln = ln.strip()
                if ln and not ln.startswith("#"):
                    FINALISTS.add(ln)
        # If still empty, compute now from whatever metrics exist
        if not FINALISTS:
            _ = _build_leaderboard_and_finalists(SAVE_ROOT, args.topk_global, args.topk_family)
            for ln in FINALISTS_FILE.read_text().splitlines():
                ln = ln.strip()
                if ln and not ln.startswith("#"):
                    FINALISTS.add(ln)
        if not FINALISTS:
            print("[rerun] No finalists found.")
            return
        index_map = _read_index_map(SAVE_ROOT)
        # helpers for micro-grid
        def _clip(x, lo, hi): return max(lo, min(hi, x))
        def _uniq_sorted(vals):
            out = sorted({round(float(v), 6) for v in vals})
            return out
        for tag in sorted(FINALISTS):
            base_sd = SAVE_ROOT / tag
            row = index_map.get(tag)
            if not row:
                print(f"[rerun] missing in index.csv: {tag}; skipping"); continue
            loss = row["loss"]
            if loss == "ce":
                # Just re-run exact config to save checkpoint
                if (base_sd / "best.pt").exists():
                    print(f"[rerun] skip (exists): {base_sd}")
                    continue
                cmd = _reconstruct_cmd_from_index(row, base_sd, args.nproc)
                if "--save_model" not in cmd: cmd.append("--save_model")
                for i,c in enumerate(cmd):
                    if c == "--epochs" and i+1 < len(cmd):
                        cmd[i+1] = str(args.final_epochs)
                print(f"[rerun] CE finalist → {tag}")
                if args.dry_run:
                    print(">>", " ".join(shlex.quote(x) for x in map(str, cmd)))
                else:
                    run(cmd, base_sd)
                continue
            # MAGe / ALDR-KL micro-grid around the finalist
            l0 = float(row["lambda0"]); a = float(row["alpha"]); C = float(row["C"])
            l0s = _uniq_sorted([_clip(l0 / math.sqrt(3), 0.1, 30.0), l0, _clip(l0 * math.sqrt(3), 0.1, 30.0)])
            alphas = _uniq_sorted([_clip(a - 0.1, 1.1, 2.5), a, _clip(a + 0.1, 1.1, 2.5)])
            Cs = _uniq_sorted([_clip(C / 3.0, 0.01, 3.0), C, _clip(C * 3.0, 0.01, 3.0)])
            for l0p, ap, Cp in itertools.product(l0s, alphas, Cs):
                if loss == "mager":
                    dist = row["distribution"]
                    new_tag = f"mager-{dist}_l{l0p}_a{ap}_C{Cp}_p2"
                    sd = SAVE_ROOT / new_tag
                    cmd = mager_cmd(dist, l0p, ap, Cp, sd, nproc=args.nproc)
                if loss == "aldrkl":
                    new_tag = f"aldrkl_l{l0p}_a{ap}_C{Cp}_p2"
                    sd = SAVE_ROOT / new_tag
                    cmd = aldrkl_cmd(l0p, ap, Cp, sd, nproc=args.nproc)
                else:
                    continue
                # force save + finalist epochs
                if (sd / "best.pt").exists():
                    print(f"[rerun] skip (exists): {sd}")
                    continue
                if "--save_model" not in cmd: cmd.append("--save_model")
                for i,c in enumerate(cmd):
                    if c == "--epochs" and i+1 < len(cmd):
                        cmd[i+1] = str(args.final_epochs)
                print(f"[rerun] {loss} finalist micro-grid → {sd.name}")
                if args.dry_run:
                    print(">>", " ".join(shlex.quote(x) for x in map(str, cmd)))
                else:
                    run(cmd, sd)
        return

    # phase 1: full sweep (ranking-only, no checkpoints); also writes finalists.txt at the end
    rows = []
    jobs = []

    # --- MAGe: mixture & uniform (Phase 1 grid) ---
    for distribution in ("mixture", "uniform"):
        for l0, a, cval in itertools.product(LAMBDA0S, ALPHAS, C_VALUES_MAGER):
            tag = f"mager-{distribution}_l{l0}_a{a}_C{cval}_dec"
            sd = SAVE_ROOT / tag
            cmd = mager_cmd(distribution, l0, a, cval, sd, nproc=args.nproc)
            if args.nproc == 1 and not args.dry_run:
                jobs.append({"cmd": cmd, "save_dir": sd})
            else:
                if args.dry_run: print(">>", " ".join(shlex.quote(str(x)) for x in cmd))
                else: run(cmd, sd)
            rows.append({"loss":"mager","distribution":distribution,"lambda0":l0,"alpha":a,"C":cval,"tag":tag})

    # --- ALDR-KL ---
    for l0, a, cval in itertools.product(LAMBDA0S, ALPHAS, C_VALUES_ALDR):
        tag = f"aldrkl_l{l0}_a{a}_C{cval}"
        sd = SAVE_ROOT / tag
        cmd = aldrkl_cmd(l0, a, cval, sd, nproc=args.nproc)
        if args.nproc == 1 and not args.dry_run:
            jobs.append({"cmd": cmd, "save_dir": sd})
        else:
            if args.dry_run: print(">>", " ".join(shlex.quote(str(x)) for x in cmd))
            else: run(cmd, sd)
        rows.append({"loss":"aldrkl","lambda0":l0,"alpha":a,"C":cval,"tag":tag})

    # # --- CE baseline ---
    for ls in CE_LABEL_SMOOTH:
        tag = f"ce_ls{ls}"
        sd = SAVE_ROOT / tag
        cmd = ce_cmd(ls, sd, nproc=args.nproc)
        if args.nproc == 1 and not args.dry_run:
            jobs.append({"cmd": cmd, "save_dir": sd})
        else:
            if args.dry_run: print(">>", " ".join(shlex.quote(str(x)) for x in cmd))
            else: run(cmd, sd)
        rows.append({"tag": tag, "loss": "ce", "label_smoothing": ls})

    # If there are queued jobs, run them (1 GPU/job)
    if jobs:
        _run_pool(jobs, concurrency=min(args.concurrency, len(CUDA_DEVS)), device_ids=CUDA_DEVS or [0])

    # index csv of run arguments
    with open(SAVE_ROOT/"index.csv","w", newline="") as f:
        # standardized columns for easy phase-2 reconstruction
        cols = ["tag","loss","distribution","lambda0","alpha","C","label_smoothing"]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    # build leaderboard & finalists list after completion
    _build_leaderboard_and_finalists(SAVE_ROOT, args.topk_global, args.topk_family)

if __name__ == "__main__":
    main()
