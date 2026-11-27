"""NV1 Type Classification with classical multi-label ML (T1.0-T1.5 + improvements).

  python scripts/train_type_nv1.py --config configs/type_nv1/experiments.yaml \
      --dataset-root ../data_pr/dataset_v1.0 --cache ../data_pr/nv1_cache [--only key ...]

Protocol per experiment:
  1. 5-fold GroupKFold on TRAIN (leakage groups) -> out-of-fold (OOF) probabilities per grid point;
     the grid point with the best OOF macro AUPRC is kept (threshold-free selection).
  2. Thresholds (fixed / global / per-label) are fit on the OOF probabilities of that point.
  3. Refit on all of train, predict VALIDATION once, score with the OOF thresholds.
The test split is never read. Artifacts follow docs/RESEARCH_CONTRACTS.md.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.contracts import TYPES, prediction_record, write_predictions  # noqa: E402
from src.common.dataset import ClickbaitDataset, file_sha256, read_csv  # noqa: E402
from src.common.runtime import log_event, new_run, write_json  # noqa: E402
from src.type_classification.evaluation import (  # noqa: E402
    bootstrap_ci,
    evaluate,
    fit_thresholds,
    macro_f1,
    to_logits,
)
from src.type_classification.multilabel import (  # noqa: E402
    CLASSIFIERS,
    CONTENT_CHARS,
    SEED,
    Features,
    StackingMeta,
)

MANIFEST = ROOT / "data/manifests/dataset_v1.0"


def check_snapshot(root):
    """Split/schema must be byte-identical to the published release; data.csv is recorded."""
    pub = json.loads((MANIFEST / "manifest.json").read_text(encoding="utf-8"))
    for rel in [
        "split_assignments.csv",
        "label_schema.json",
        "splits/train.csv",
        "splits/validation.csv",
    ]:
        if file_sha256(Path(root) / rel) != pub["protected_sha256"][rel]:
            raise ValueError(f"Snapshot differs from the shared release: {rel}")
    return {
        "dataset_version": "1.0",
        "dataset_manifest_sha256": file_sha256(MANIFEST / "manifest.json"),
        "split_assignments_sha256": pub["protected_sha256"]["split_assignments.csv"],
        "label_schema_sha256": pub["protected_sha256"]["label_schema.json"],
        "local_data_csv_sha256": file_sha256(Path(root) / "data.csv"),
        "published_data_csv_sha256": pub["protected_sha256"]["data.csv"],
        "label_origin": "pseudo",
    }


class Split:
    """Rows, labels and feature views of one split (aligned with the feature cache)."""

    def __init__(self, root, split, cache, groups_by_id):
        rows = ClickbaitDataset(root, split, verify_checksums=False).rows
        schema = json.loads((Path(root) / "label_schema.json").read_text(encoding="utf-8"))
        self.ids = [r["article_id"] for r in rows]
        self.Y = np.array([[int(r[c]) for c in schema["type_columns"]] for r in rows], float)
        self.detection = np.array([int(r["clickbait_label"]) for r in rows])
        self.groups = np.array([groups_by_id[a] for a in self.ids])
        self.source = [r["source"] for r in rows]
        self.titles = [r["title"] for r in rows]
        pos = {a: i for i, a in enumerate(cache["ids"])}
        idx = np.array([pos[a] for a in self.ids])
        sem = None
        if "title_emb" in cache:
            t, l, c = (cache[k][idx] for k in ("title_emb", "lead_emb", "content_emb"))
            cos = np.column_stack([(t * c).sum(1), (t * l).sum(1), (l * c).sum(1)])
            sem = np.hstack([t, l, c, cos]).astype(np.float32)
        self.view = {
            "title": self.titles,
            "content": [(r["content"] or "")[:CONTENT_CHARS] for r in rows],
            "hand": cache["hand"][idx],
            "sem": sem,
            "source": self.source,
        }

    def subset(self, idx):
        return {
            k: (
                None
                if v is None
                else (v[idx] if isinstance(v, np.ndarray) else [v[i] for i in idx])
            )
            for k, v in self.view.items()
        }


def load_cache(path):
    path = Path(path)
    cache = {
        "ids": json.loads((path / "ids.json").read_text(encoding="utf-8")),
        "hand": np.load(path / "handcrafted.npy"),
        "hand_names": json.loads((path / "handcrafted_names.json").read_text(encoding="utf-8")),
        "meta": json.loads((path / "meta.json").read_text(encoding="utf-8")),
    }
    for k in ("title_emb", "lead_emb", "content_emb"):
        if (path / f"{k}.npy").exists():
            cache[k] = np.load(path / f"{k}.npy")
    return cache


def folds(groups, n):
    from sklearn.model_selection import GroupKFold

    return list(
        GroupKFold(n_splits=n, shuffle=True, random_state=SEED).split(groups, groups=groups)
    )


def grid_points(grid):
    from itertools import product

    keys = sorted(grid)
    return [dict(zip(keys, vals)) for vals in product(*(grid[k] for k in keys))] or [{}]


def macro_auprc(P, Y):
    from sklearn.metrics import average_precision_score

    return float(np.mean([average_precision_score(Y[:, k], P[:, k]) for k in range(Y.shape[1])]))


def run_base(exp, train, val, cache, cv, events):
    """Grid search on OOF, refit, predict validation. Returns OOF/val probs + fitted model."""
    make_clf = CLASSIFIERS[exp["classifier"]]
    points = grid_points(exp.get("grid", {}))
    oof = {i: np.zeros_like(train.Y) for i in range(len(points))}
    for f, (tr, te) in enumerate(cv):
        feats = Features(exp["blocks"], cache["hand_names"])
        Xtr = feats.fit_transform(train.subset(tr))
        Xte = feats.transform(train.subset(te))
        for i, params in enumerate(points):
            oof[i][te] = make_clf(**params).fit(Xtr, train.Y[tr]).predict_proba(Xte)
        log_event(events, event="cv_fold", step=f + 1)
    scores = [macro_auprc(oof[i], train.Y) for i in range(len(points))]
    best = int(np.argmax(scores))
    log_event(
        events, event="grid", step=0, points=points, oof_macro_auprc=scores, best=points[best]
    )
    feats = Features(exp["blocks"], cache["hand_names"])
    Xtr = feats.fit_transform(train.view)
    clf = make_clf(**points[best]).fit(Xtr, train.Y)
    Pval = clf.predict_proba(feats.transform(val.view))
    return oof[best], Pval, {"features": feats, "classifier": clf}, points[best], scores


def run_stack(exp, train, val, cv, base_results):
    names = exp["stack"]
    Ptr = np.hstack([base_results[n]["oof"] for n in names])
    Pva = np.hstack([base_results[n]["val"] for n in names])
    oof = np.zeros_like(train.Y)
    for tr, te in cv:  # meta OOF on the same folds -> honest thresholds for the stack too
        oof[te] = (
            StackingMeta(**exp.get("meta", {})).fit(Ptr[tr], train.Y[tr]).predict_proba(Ptr[te])
        )
    meta = StackingMeta(**exp.get("meta", {})).fit(Ptr, train.Y)
    return oof, meta.predict_proba(Pva), {"meta": meta, "bases": names}, exp.get("meta", {}), []


def interpret(model, exp, cache):
    """Top weighted features per label (linear) or importances (CatBoost)."""
    clf, feats = model.get("classifier"), model.get("features")
    if clf is None or feats is None:
        return None
    names = np.array(feats.names())
    out = {}
    if hasattr(clf, "coefficients"):
        W = clf.coefficients()
        for k, t in enumerate(TYPES):
            order = np.argsort(W[k])
            out[t] = {
                "positive": [[names[i], float(W[k, i])] for i in order[::-1][:25]],
                "negative": [[names[i], float(W[k, i])] for i in order[:10]],
            }
    elif exp["classifier"] == "ovr_catboost":
        for k, t in enumerate(TYPES):
            imp = clf.models[k].get_feature_importance()
            order = np.argsort(imp)[::-1][:15]
            out[t] = [[names[i], float(imp[i])] for i in order]
    return out


def per_source(P, Y, th, sources, min_n=100):
    sources = np.asarray(sources)
    res = {}
    for s in sorted(set(sources)):
        m = sources == s
        if m.sum() >= min_n:
            res[s] = {
                "n": int(m.sum()),
                "macro_f1": macro_f1(P[m], Y[m], th),
                "label_rate": float(Y[m].mean()),
            }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if cfg["task"] != "type_classification" or cfg["seed"] != SEED:
        raise ValueError("NV1 config: task type_classification, seed 20261124")
    data_meta = check_snapshot(args.dataset_root)
    cache = load_cache(args.cache)
    if cache["meta"]["data_csv_sha256"] != data_meta["local_data_csv_sha256"]:
        raise ValueError("Feature cache was built from another data.csv")
    groups = {
        r["article_id"]: r["leakage_group_id"]
        for r in read_csv(Path(args.dataset_root) / "split_assignments.csv")
    }
    train = Split(args.dataset_root, "train", cache, groups)
    val = Split(args.dataset_root, "validation", cache, groups)
    cv = folds(train.groups, cfg["cv"]["folds"])
    results = {}
    for key, exp in cfg["experiments"].items():
        if (
            args.only
            and key not in args.only
            and key
            not in sum(
                [
                    cfg["experiments"][o].get("stack", [])
                    for o in args.only
                    if o in cfg["experiments"]
                ],
                [],
            )
        ):
            continue
        t0 = time.time()
        run_cfg = {
            "format_version": "1.0",
            "run_mode": "research",
            "task": "type_classification",
            "seed": SEED,
            "experiment": exp["id"],
            "experiment_key": key,
            "data": cfg["data"],
            "model": {"name": key.replace("_", "-"), **{k: v for k, v in exp.items() if k != "id"}},
            "cv": cfg["cv"],
            "threshold": cfg["threshold"],
        }
        run_id, run_dir, meta = new_run(run_cfg, ROOT)
        import importlib.metadata as md

        for pkg in ("scikit-learn", "scipy", "catboost", "transformers"):
            try:
                meta["versions"][pkg] = md.version(pkg)
            except md.PackageNotFoundError:
                meta["versions"][pkg] = "unavailable"
        meta.update(data_meta, feature_cache=cache["meta"], experiment=exp["id"], device="cpu")
        write_json(run_dir / "run.json", meta)
        events = run_dir / "events.jsonl"
        log_event(events, event="start", step=0, train=len(train.ids), validation=len(val.ids))
        if "stack" in exp:
            oof, Pval, model, best, scores = run_stack(exp, train, val, cv, results)
        else:
            oof, Pval, model, best, scores = run_base(exp, train, val, cache, cv, events)
        results[key] = {"oof": oof, "val": Pval}
        by_mode = {}
        for mode in cfg["threshold"]["modes"]:
            th = fit_thresholds(oof, train.Y, mode)
            r = evaluate(Pval, val.Y, th, val.detection)
            r["train_oof_macro_f1"] = macro_f1(oof, train.Y, th)
            r["macro_f1_ci95"] = bootstrap_ci(Pval, val.Y, th, val.groups)
            by_mode[mode] = r
        th = [by_mode[cfg["threshold"]["export"]]["thresholds"][t] for t in TYPES]
        res_dir = ROOT / "results" / run_id
        np.save(res_dir / "oof_probs.npy", oof)
        np.save(res_dir / "val_probs.npy", Pval)
        write_json(
            res_dir / "metrics.json",
            {
                "run_id": run_id,
                "experiment": exp["id"],
                "experiment_key": key,
                "diagnostic": bool(exp.get("diagnostic")),
                "split": "validation",
                "label_origin": "pseudo",
                "n": len(val.ids),
                "official_benchmark": False,
                "selected_params": best,
                "grid_oof_macro_auprc": scores,
                "oof_macro_auprc": macro_auprc(oof, train.Y),
                "threshold_fit_on": "train_oof",
                "exported_threshold_mode": cfg["threshold"]["export"],
                "metrics_by_threshold_mode": by_mode,
                "per_source_macro_f1": per_source(Pval, val.Y, th, val.source),
                "seconds": round(time.time() - t0, 1),
                "note": "Thresholds and hyperparameters chosen on train out-of-fold predictions; "
                "validation scored once. Test not used (gold pending).",
            },
        )
        top = interpret(model, exp, cache)
        if top:
            write_json(res_dir / "interpretation.json", top)
        logits = to_logits(Pval)
        records = [
            prediction_record(
                a,
                {"types": logits},
                i,
                run_id=run_id,
                dataset_version="1.0",
                split="validation",
                task="type_classification",
                label_origin="pseudo",
                type_thresholds=th,
            )
            for i, a in enumerate(val.ids)
        ]
        write_predictions(res_dir / "predictions.jsonl", records, val.ids)
        ck = ROOT / "checkpoints" / run_id
        ck.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, ck / "model.joblib")  # sklearn objects: load only trusted files
        write_json(
            ck / "provenance.json",
            {
                **data_meta,
                "run_id": run_id,
                "git_commit": meta["git_commit"],
                "thresholds": dict(zip(TYPES, th)),
            },
        )
        log_event(events, event="end", step=0, seconds=round(time.time() - t0, 1))
        pl = by_mode["per_label"]
        print(
            f"[{key}] {exp['id']} best={best} OOF-AUPRC={macro_auprc(oof, train.Y):.4f} "
            f"VAL macroF1={pl['macro_f1']:.4f} microF1={pl['micro_f1']:.4f} "
            f"AUPRC={pl['macro_auprc']:.4f} ({time.time() - t0:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
