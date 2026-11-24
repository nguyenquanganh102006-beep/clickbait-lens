"""Fixed-class macro F1; never hide absent classes or masked relation targets."""

import numpy as np
from src.common.contracts import TYPES, RELATIONS, active_tasks, validate_thresholds


def classification_metrics(y, pred, classes):
    y = np.asarray(y).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    if len(y) != len(pred):
        raise ValueError("Metric input length mismatch")
    if len(y) == 0:
        return {"n": 0, "accuracy": None, "macro_f1": None, "per_class": {}}
    rows = {}
    for c in classes:
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows[str(c)] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "support": int((y == c).sum()),
        }
    return {
        "n": len(y),
        "accuracy": float((y == pred).mean()),
        "macro_f1": float(np.mean([r["f1"] for r in rows.values()])),
        "per_class": rows,
    }


def compute_metrics(
    outputs, targets, masks, task="multitask", detection_threshold=0.5, type_thresholds=None
):
    thresholds = validate_thresholds(detection_threshold, type_thresholds)
    result = {}
    for key in active_tasks(task):
        p = outputs[key].detach().cpu()
        y = targets[key].detach().cpu().numpy()
        valid = masks[key].detach().cpu().numpy().astype(bool)
        if key == "detection":
            result[key] = classification_metrics(
                y[valid], (p.sigmoid().numpy() >= detection_threshold)[valid], [0, 1]
            )
        elif key == "types":
            predicted = p.sigmoid().numpy() >= np.asarray(thresholds)
            per_label = {
                t: classification_metrics(
                    y[:, i][valid[:, i]], predicted[:, i][valid[:, i]], [0, 1]
                )
                for i, t in enumerate(TYPES)
            }
            micro = classification_metrics(y[valid], predicted[valid], [0, 1])
            positive_f1 = [v["per_class"]["1"]["f1"] for v in per_label.values() if v["n"]]
            result[key] = {
                "micro_f1": micro["per_class"]["1"]["f1"] if micro["n"] else None,
                "macro_f1": float(np.mean(positive_f1)) if positive_f1 else None,
                "per_label": per_label,
            }
        else:
            predicted = p.argmax(-1).numpy()
            valid &= y != -100
            result[key] = {
                r: classification_metrics(
                    y[:, i][valid[:, i]], predicted[:, i][valid[:, i]], [0, 1, 2]
                )
                for i, r in enumerate(RELATIONS)
            }
    return result
