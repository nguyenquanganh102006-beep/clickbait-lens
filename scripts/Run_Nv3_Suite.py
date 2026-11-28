from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from scripts.train_consistency_ic import run as run_experiment

SUITE: list[dict] = [
    {
        "label": "C3.1 Truncated + Weighted CE",
        "experiment": "c3_1",
        "model": {"name": "ic-c3-1-truncated", "hidden_dim": 64, "dropout": 0.5},
        "imbalance": {
            "loss_type": "weighted_ce",
            "weight_strategy": "inverse_sqrt",
            "use_balanced_sampler": False,
            "hard_negative_ratio": 0.0,
        },
        "training": {"learning_rate": 0.0001, "weight_decay": 0.05, "epochs": 15},
    },
    {
        "label": "C3.2 Summary + Weighted CE",
        "experiment": "c3_2",
        "model": {"name": "ic-c3-2-summary", "hidden_dim": 64, "dropout": 0.5},
        "imbalance": {
            "loss_type": "weighted_ce",
            "weight_strategy": "inverse_sqrt",
            "use_balanced_sampler": False,
            "hard_negative_ratio": 0.0,
        },
        "training": {"learning_rate": 0.0001, "weight_decay": 0.05, "epochs": 15},
    },
    {
        "label": "C3.3 Image-Chunk Similarity",
        "experiment": "c3_3",
        "model": {
            "name": "ic-c3-3-similarity",
            "aggregation": "similarity",
            "hidden_dim": 64,
            "dropout": 0.3,
        },
        "imbalance": {
            "loss_type": "weighted_ce",
            "weight_strategy": "inverse_sqrt",
            "use_balanced_sampler": False,
            "hard_negative_ratio": 0.0,
        },
        "training": {"learning_rate": 0.0003, "weight_decay": 0.02, "epochs": 20},
    },
    {
        "label": "C3.4 Top-K Evidence + MLP",
        "experiment": "c3_4",
        "model": {
            "name": "ic-c3-4-topk-mlp",
            "aggregation": "topk_mlp",
            "top_k": 3,
            "hidden_dim": 64,
            "dropout": 0.5,
        },
        "imbalance": {
            "loss_type": "weighted_ce",
            "weight_strategy": "inverse_sqrt",
            "use_balanced_sampler": False,
            "hard_negative_ratio": 0.0,
        },
        "training": {"learning_rate": 0.0001, "weight_decay": 0.05, "epochs": 15},
    },
    {
        "label": "C3.5 Max Pooling",
        "experiment": "c3_5",
        "model": {
            "name": "ic-c3-5-max",
            "aggregation": "max",
            "hidden_dim": 64,
            "dropout": 0.5,
        },
        "imbalance": {
            "loss_type": "weighted_ce",
            "weight_strategy": "inverse_sqrt",
            "use_balanced_sampler": False,
            "hard_negative_ratio": 0.0,
        },
        "training": {"learning_rate": 0.0001, "weight_decay": 0.05, "epochs": 15},
    },
    {
        "label": "C3.5 Mean Pooling",
        "experiment": "c3_5",
        "model": {
            "name": "ic-c3-5-mean",
            "aggregation": "mean",
            "hidden_dim": 64,
            "dropout": 0.5,
        },
        "imbalance": {
            "loss_type": "weighted_ce",
            "weight_strategy": "inverse_sqrt",
            "use_balanced_sampler": False,
            "hard_negative_ratio": 0.0,
        },
        "training": {"learning_rate": 0.0001, "weight_decay": 0.05, "epochs": 15},
    },
    {
        "label": "C3.5 Attention + Weighted CE (Reg)",
        "experiment": "c3_5",
        "model": {
            "name": "ic-c3-5-attn-wce",
            "aggregation": "attention",
            "hidden_dim": 64,
            "dropout": 0.5,
        },
        "imbalance": {
            "loss_type": "weighted_ce",
            "weight_strategy": "inverse_sqrt",
            "use_balanced_sampler": False,
            "hard_negative_ratio": 0.0,
        },
        "training": {"learning_rate": 0.0001, "weight_decay": 0.05, "epochs": 15},
    },
    {
        "label": "C3.5 Attention + Focal Loss (Reg)",
        "experiment": "c3_5",
        "model": {
            "name": "ic-c3-5-attn-focal",
            "aggregation": "attention",
            "hidden_dim": 64,
            "dropout": 0.5,
        },
        "imbalance": {
            "loss_type": "focal",
            "weight_strategy": "inverse_sqrt",
            "focal_gamma": 2.0,
            "use_balanced_sampler": False,
            "hard_negative_ratio": 0.0,
        },
        "training": {"learning_rate": 0.0001, "weight_decay": 0.05, "epochs": 15},
    },
    {
        "label": "C3.5 Attention + Hard Negative 5%",
        "experiment": "c3_5",
        "model": {
            "name": "ic-c3-5-attn-hn05",
            "aggregation": "attention",
            "hidden_dim": 64,
            "dropout": 0.5,
        },
        "imbalance": {
            "loss_type": "weighted_ce",
            "weight_strategy": "inverse_sqrt",
            "use_balanced_sampler": False,
            "hard_negative_ratio": 0.05,
        },
        "training": {"learning_rate": 0.0001, "weight_decay": 0.05, "epochs": 15},
    },
]


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if k == "label":
            continue
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k].update(v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def _write_summary(rows: list[dict], results_root: Path) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: r["ic_macro_f1"], reverse=True)

    fields = [
        "label",
        "experiment",
        "aggregation",
        "loss_type",
        "hard_negative_ratio",
        "accuracy",
        "ic_macro_f1",
        "f1_consistent",
        "f1_inconsistent",
        "f1_insufficient",
        "run_id",
    ]
    csv_path = results_root / "nv3_comparison.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)

    lines = [
        "# Bảng so sánh kết quả NV3 — Image–Content Consistency (Validation, 4.206 bài)",
        "",
        "| Thí nghiệm | Exp | Pooling | Loss | HN | Accuracy | **Macro-F1** | F1-Consistent | F1-Inconsistent | F1-Insufficient | Run ID |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in ordered:
        lines.append(
            f"| {r['label']} | `{r['experiment']}` | `{r['aggregation']}` | `{r['loss_type']}` | "
            f"{r['hard_negative_ratio']:.2f} | {r['accuracy']:.4f} | **{r['ic_macro_f1']:.4f}** | "
            f"{r['f1_consistent']:.4f} | {r['f1_inconsistent']:.4f} | "
            f"{r['f1_insufficient']:.4f} | `{r['run_id']}` |"
        )
    md_path = results_root / "nv3_comparison.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full NV3 C3.1-C3.5 comparison suite.")
    parser.add_argument("--base-config", default=str(ROOT / "configs/consistency_ic.yaml"))
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--skip-checksum", action="store_true")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of experiments, e.g. --only c3_3 c3_4 c3_5",
    )
    args = parser.parse_args()

    base_cfg = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    selected = [s for s in SUITE if not args.only or s["experiment"] in set(args.only)]
    rows: list[dict] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, spec in enumerate(selected, start=1):
            print(f"\n=== [{idx}/{len(selected)}] {spec['label']} ===", flush=True)
            cfg = _deep_merge(base_cfg, spec)
            cfg_path = Path(tmpdir) / f"cfg_{idx}_{cfg['model']['name']}.yaml"
            cfg_path.write_text(
                yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            run_path = run_experiment(
                cfg_path,
                args.dataset_root,
                synthetic=args.synthetic,
                skip_checksum=args.skip_checksum,
            )
            run_id = run_path.name
            metrics_data = json.loads(
                (ROOT / "results" / run_id / "metrics.json").read_text(encoding="utf-8")
            )
            ic = metrics_data["metrics"]["consistency"]["ic"]
            per_cls = ic["per_class"]
            rows.append(
                {
                    "label": spec["label"],
                    "experiment": cfg["experiment"],
                    "aggregation": cfg["model"].get("aggregation", "global"),
                    "loss_type": cfg["imbalance"]["loss_type"],
                    "hard_negative_ratio": float(cfg["imbalance"]["hard_negative_ratio"]),
                    "accuracy": float(ic["accuracy"] or 0.0),
                    "ic_macro_f1": float(ic["macro_f1"] or 0.0),
                    "f1_consistent": float(per_cls["0"]["f1"]),
                    "f1_inconsistent": float(per_cls["1"]["f1"]),
                    "f1_insufficient": float(per_cls["2"]["f1"]),
                    "run_id": run_id,
                }
            )
            _write_summary(rows, ROOT / "results")

    print("\n=== SUITE COMPLETE ===")
    print(f"CSV summary: {ROOT / 'results' / 'nv3_comparison.csv'}")
    print(f"MD  summary: {ROOT / 'results' / 'nv3_comparison.md'}")


if __name__ == "__main__":
    main()