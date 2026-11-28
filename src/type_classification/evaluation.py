"""Thresholds, metrics and uncertainty for multi-label Type predictions (NV1).

Protocol: thresholds are fit on TRAIN out-of-fold probabilities (grouped CV), then applied
unchanged to validation, so validation numbers are an honest held-out estimate.
"""

import numpy as np
import torch

from src.common.contracts import TYPES
from src.common.metrics import compute_metrics

GRID = np.round(np.arange(0.02, 0.981, 0.01), 2)


def f1(y, pred):
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def macro_f1(P, Y, th):
    pred = P >= np.asarray(th)
    return float(np.mean([f1(Y[:, k], pred[:, k]) for k in range(Y.shape[1])]))


def fit_thresholds(P, Y, mode):
    """mode: fixed (0.5) | global (one t, max macro-F1) | per_label (t_k, max F1_k).
    Ties go to the threshold closest to 0.5 (deterministic, least extreme)."""
    order = sorted(GRID, key=lambda t: (abs(t - 0.5), t))
    k = Y.shape[1]
    if mode == "fixed":
        return [0.5] * k
    if mode == "global":
        return [float(max(order, key=lambda t: macro_f1(P, Y, [t] * k)))] * k
    if mode == "per_label":
        return [float(max(order, key=lambda t: f1(Y[:, j], P[:, j] >= t))) for j in range(k)]
    raise ValueError(mode)


def auprc(y, p):
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, p)) if y.any() else float("nan")


def to_logits(P):
    P = np.clip(np.asarray(P, dtype=np.float64), 1e-7, 1 - 1e-7)
    return torch.from_numpy(np.log(P / (1 - P))).float()


def evaluate(P, Y, thresholds, detection=None):
    """Contract metrics (src.common.metrics) + Hamming, subset accuracy, AUPRC, logic check."""
    Y = np.asarray(Y, dtype=np.float32)
    res = compute_metrics(
        {"types": to_logits(P)},
        {"types": torch.from_numpy(Y)},
        {"types": torch.ones(Y.shape)},
        "type_classification",
        type_thresholds=thresholds,
    )["types"]
    pred = P >= np.asarray(thresholds)
    gold = Y.astype(bool)
    res.update(
        {
            "thresholds": dict(zip(TYPES, map(float, thresholds))),
            "hamming_loss": float((pred != gold).mean()),
            "subset_accuracy": float((pred == gold).all(1).mean()),
            "macro_auprc": float(np.nanmean([auprc(Y[:, k], P[:, k]) for k in range(5)])),
            "per_label_auprc": {t: auprc(Y[:, k], P[:, k]) for k, t in enumerate(TYPES)},
            "label_cardinality_pred": float(pred.sum(1).mean()),
            "label_cardinality_gold": float(gold.sum(1).mean()),
        }
    )
    if detection is not None:
        det = np.asarray(detection).astype(int)
        any_type = pred.any(1)
        res["logic"] = {
            # Detection-Type rule of the dataset: clickbait == 1 <=> at least one Type
            "types_on_gold_nonclickbait_rate": float(any_type[det == 0].mean()),
            "no_type_on_gold_clickbait_rate": float((~any_type[det == 1]).mean()),
            "derived_detection_f1": f1(det, any_type),
        }
    return res


def bootstrap_ci(P, Y, thresholds, groups, n=1000, seed=20261124, metric=macro_f1):
    """95% CI of a metric by resampling validation leakage groups (articles of one group move
    together, so near-duplicates do not shrink the interval artificially)."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    stats = []
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        stats.append(metric(P[idx], Y[idx], thresholds))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def paired_bootstrap(Pa, tha, Pb, thb, Y, groups, n=2000, seed=20261124, label=None):
    """Macro-F1(a) - Macro-F1(b) on the same resampled groups: mean, 95% CI, one-sided p
    (share of resamples where a is not better). label=k compares F1 of label k only."""
    if label is not None:
        k = label
        Pa, Pb, Y = Pa[:, [k]], Pb[:, [k]], Y[:, [k]]
        tha, thb = [np.asarray(tha)[k]], [np.asarray(thb)[k]]
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    diffs = []
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        diffs.append(macro_f1(Pa[idx], Y[idx], tha) - macro_f1(Pb[idx], Y[idx], thb))
    diffs = np.array(diffs)
    return {
        "mean_diff": float(diffs.mean()),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p_not_better": float((diffs <= 0).mean()),
    }
