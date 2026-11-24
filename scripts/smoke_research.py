"""Offline integration test: train heads, save/load checkpoint, log predictions.

Not an experiment baseline. Never selects models using the test split.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.common.dataset import ClickbaitDataset, collate_samples, file_sha256
from src.common.contracts import compute_loss, prediction_record, write_predictions
from src.common.metrics import compute_metrics
from src.common.model import SmokeModel
from src.common.runtime import (
    load_config,
    seed_everything,
    new_run,
    log_event,
    write_json,
    save_checkpoint,
    load_checkpoint,
)
from scripts.validate_research_dataset import validate


def smoke_features(sample):
    f = sample["features"]
    pixels = np.asarray(f["image"].resize((16, 16)), dtype=np.float32) / 255
    return [
        len(f["title"].split()) / 100,
        len(f["content"].split()) / 2000,
        len(f["title"]) / 1000,
        len(f["content"]) / 10000,
        *pixels.mean(axis=(0, 1)).tolist(),
        f["image"].width / f["image"].height / 4,
    ]


def run(config_path, dataset_root=None, synthetic=False, output_root=ROOT):
    config = load_config(config_path)
    seed_everything(config["seed"])
    n = config["training"]["batch_size"]
    root = Path(dataset_root) if dataset_root else None
    if synthetic:
        x = torch.randn(n, 8)
        y = torch.randint(0, 2, (n, 5)).float()
        targets = {
            "detection": y.max(1).values,
            "types": y,
            "consistency": torch.randint(0, 3, (n, 3)),
        }
        masks = {k: torch.ones_like(v, dtype=torch.float32) for k, v in targets.items()}
        batch = {
            "targets": targets,
            "masks": masks,
            "article_id": [f"synthetic_{i}" for i in range(n)],
        }
        val_x = x.clone()
        val_batch = batch
        fingerprint = "synthetic-fixture-not-a-dataset-release"
        origin = "synthetic"
        split_hash = schema_hash = fingerprint
        manifest_hash = fingerprint
    else:
        if root is None:
            raise ValueError("--dataset-root required unless --synthetic")
        validate(root)
        train = ClickbaitDataset(root, "train", image_root=config["data"].get("image_root"))
        validation = ClickbaitDataset(
            root, "validation", verify_checksums=False, image_root=config["data"].get("image_root")
        )

        def get(ds):
            samples = [ds[i] for i in range(min(n, len(ds)))]
            b = collate_samples(samples, as_torch=True)
            return torch.tensor([smoke_features(s) for s in samples], dtype=torch.float32), b

        x, batch = get(train)
        val_x, val_batch = get(validation)
        origin = "pseudo"
        manifest_hash = file_sha256(root / "manifest.json")
        split_hash = file_sha256(root / "split_assignments.csv")
        schema_hash = file_sha256(root / "label_schema.json")
    run_id, run_path, metadata = new_run(config, output_root)
    metadata.update(
        {
            "dataset_version": "synthetic" if synthetic else "1.0",
            "dataset_manifest_sha256": manifest_hash,
            "split_assignments_sha256": split_hash,
            "label_schema_sha256": schema_hash,
            "label_origin": origin,
            "official_benchmark": False,
            "validation_overlap_train": bool(synthetic),
        }
    )
    write_json(run_path / "run.json", metadata)
    events = run_path / "events.jsonl"
    log_event(events, event="start", run_id=run_id, synthetic=synthetic)
    model = SmokeModel(hidden_dim=config["model"]["hidden_dim"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    for step in range(1, config["training"]["steps"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x)
        loss, parts = compute_loss(outputs, batch["targets"], batch["masks"], config["task"])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        optimizer.step()
        log_event(
            events,
            event="train_step",
            step=step,
            loss=float(loss.detach()),
            losses={k: float(v.detach()) for k, v in parts.items()},
        )
    model.eval()
    with torch.no_grad():
        pred = model(val_x)
    metrics = compute_metrics(pred, val_batch["targets"], val_batch["masks"], config["task"])
    resultdir = Path(output_root) / "results" / run_id
    checkpointdir = Path(output_root) / "checkpoints" / run_id
    checkpoint = checkpointdir / "last.pt"
    save_checkpoint(checkpoint, model, optimizer, metadata=metadata, step=step, config=config)
    reloaded = SmokeModel(hidden_dim=config["model"]["hidden_dim"])
    load_checkpoint(
        checkpoint, reloaded, expected_split_sha256=split_hash, expected_schema_sha256=schema_hash
    )
    reloaded.eval()
    with torch.no_grad():
        roundtrip = reloaded(val_x)
    if not all(torch.equal(pred[k], roundtrip[k]) for k in pred):
        raise AssertionError("Checkpoint roundtrip changed predictions")
    records = [
        prediction_record(
            aid,
            pred,
            i,
            run_id=run_id,
            dataset_version=metadata["dataset_version"],
            split="validation",
            task=config["task"],
            label_origin=origin,
        )
        for i, aid in enumerate(val_batch["article_id"])
    ]
    write_predictions(resultdir / "predictions.jsonl", records, val_batch["article_id"])
    write_json(
        resultdir / "metrics.json",
        {
            "format_version": "1.0",
            "run_id": run_id,
            "split": "validation",
            "label_origin": origin,
            "official_benchmark": False,
            "n": len(records),
            "metrics": metrics,
        },
    )
    log_event(
        events, event="complete", step=step, checkpoint_roundtrip=True, n_predictions=len(records)
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "checkpoint": str(checkpoint),
                "predictions": len(records),
                "official_benchmark": False,
            }
        )
    )
    return run_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/smoke.yaml"))
    parser.add_argument("--dataset-root")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    run(args.config, args.dataset_root, args.synthetic)
