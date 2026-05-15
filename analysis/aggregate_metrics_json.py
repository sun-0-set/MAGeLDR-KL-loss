# aggregate_results.py
import json, csv
from pathlib import Path

def flatten(head_stats):
    keys = ["content","organization","language"]
    return {k:v for k,v in zip(keys, head_stats)}

def main(root="sweeps", out="sweeps/results.csv"):
    rows=[]
    for p in Path(root).glob("**/metrics.json"):
        with open(p) as f: m = json.load(f)
        args = m["args"]
        val  = m["val"]; test = m.get("test", {})
        row = {
            "save_dir": str(p.parent),
            "loss": args.get("loss"),
            "distribution": args.get("distribution"),
            "lambda0": args.get("lambda0"),
            "alpha": args.get("alpha"),
            "C": args.get("C"),
            "ce_label_smoothing": args.get("ce_label_smoothing"),
            "seed": args.get("seed"),
            "val_loss": val["loss"],
            **{f"val_acc_{k}":v for k,v in flatten(val["acc"]).items()},
            **{f"val_qwk_{k}":v for k,v in flatten(val["qwk"]).items()},
            **{f"val_f1m_{k}":v for k,v in flatten(val["f1_macro"]).items()},
            **{f"val_f1w_{k}":v for k,v in flatten(val["f1_weighted"]).items()},
            "val_micro_overall": val["micro_overall"],
        }
        if test:
            row.update({
                "test_loss": test["loss"],
                **{f"test_acc_{k}":v for k,v in flatten(test["acc"]).items()},
                **{f"test_qwk_{k}":v for k,v in flatten(test["qwk"]).items()},
                **{f"test_f1m_{k}":v for k,v in flatten(test["f1_macro"]).items()},
                **{f"test_f1w_{k}":v for k,v in flatten(test["f1_weighted"]).items()},
                "test_micro_overall": test["micro_overall"],
            })
        rows.append(row)
    # Write CSV
    if rows:
        cols = sorted(rows[0].keys())
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        print(f"Wrote {out} ({len(rows)} runs)")
    else:
        print("No metrics.json found under", root)

if __name__ == "__main__":
    main()
