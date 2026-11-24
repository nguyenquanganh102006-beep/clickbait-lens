import json
import math
import torch
from torch.nn import functional as F

TYPES = (
    "curiosity_gap",
    "exaggeration",
    "emotional_bait",
    "question_based",
    "misleading_presentation",
)
RELATIONS = ("tc", "ti", "ic")
CONSISTENCY = ("Consistent", "Inconsistent", "Insufficient Evidence")
TASKS = {
    "detection": ("detection",),
    "type_classification": ("types",),
    "consistency": ("consistency",),
    "multitask": ("detection", "types", "consistency"),
}


def active_tasks(task):
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    return TASKS[task]


def validate_thresholds(detection_threshold, type_thresholds=None):
    thresholds = [0.5] * 5 if type_thresholds is None else list(type_thresholds)
    if len(thresholds) != 5 or not all(
        math.isfinite(v) and 0 <= v <= 1 for v in [detection_threshold, *thresholds]
    ):
        raise ValueError("Invalid thresholds")
    return thresholds


def compute_loss(outputs, targets, masks, task="multitask"):
    """Mean over valid labels only; all-masked tasks give differentiable zero."""
    losses = {}
    for name in active_tasks(task):
        logits = outputs[name]
        target = targets[name]
        mask = masks[name].to(logits.dtype)
        if name == "consistency":
            if logits.shape != (*target.shape, 3):
                raise ValueError("Expected consistency logits [B,3,3]")
            valid = mask * (target != -100)
            loss = F.cross_entropy(
                logits.reshape(-1, 3), target.reshape(-1), ignore_index=-100, reduction="none"
            ).reshape(target.shape)
        else:
            if logits.shape != target.shape:
                raise ValueError(f"{name} target shape mismatch")
            valid = mask
            loss = F.binary_cross_entropy_with_logits(
                logits, target.to(logits.dtype), reduction="none"
            )
        losses[name] = (loss * valid).sum() / valid.sum().clamp_min(1)
    return sum(losses.values()), losses


def prediction_record(
    article_id,
    outputs,
    index,
    *,
    run_id,
    dataset_version,
    split,
    task,
    label_origin,
    detection_threshold=0.5,
    type_thresholds=None,
):
    thresholds = validate_thresholds(detection_threshold, type_thresholds)
    record = {
        "format_version": "1.0",
        "article_id": article_id,
        "run_id": run_id,
        "dataset_version": dataset_version,
        "split": split,
        "task": task,
        "label_origin": label_origin,
        "outputs": {},
    }
    for name in active_tasks(task):
        raw = outputs[name][index].detach().cpu()
        if not torch.isfinite(raw).all():
            raise ValueError("Nonfinite logits")
        if name == "detection":
            p = float(raw.sigmoid())
            record["outputs"][name] = {
                "probability": p,
                "label": int(p >= detection_threshold),
                "threshold": detection_threshold,
            }
        elif name == "types":
            p = raw.sigmoid().tolist()
            if len(p) != 5:
                raise ValueError("Expected five type scores")
            record["outputs"][name] = {
                t: {
                    "probability": p[i],
                    "label": int(p[i] >= thresholds[i]),
                    "threshold": thresholds[i],
                }
                for i, t in enumerate(TYPES)
            }
        else:
            if tuple(raw.shape) != (3, 3):
                raise ValueError("Expected three relations/classes")
            p = raw.softmax(-1)
            record["outputs"][name] = {
                rel: {"probabilities": p[i].tolist(), "label": CONSISTENCY[int(p[i].argmax())]}
                for i, rel in enumerate(RELATIONS)
            }
    if "types" in record["outputs"] and "detection" in record["outputs"]:
        derived = int(any(v["label"] for v in record["outputs"]["types"].values()))
        record["detection_type_conflict"] = derived != record["outputs"]["detection"]["label"]
    return record


def write_predictions(path, records, expected_ids):
    records = list(records)
    ids = [r["article_id"] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids) or len(ids) != len(expected_ids):
        raise ValueError("Prediction coverage must match requested IDs exactly")
    from pathlib import Path

    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
