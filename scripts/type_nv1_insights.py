"""NV1 data-science insights beyond model scores (run after train_type_nv1.py).

  python scripts/type_nv1_insights.py --dataset-root ../data_pr/dataset_v1.0 \
      --cache ../data_pr/nv1_cache --queue ../data_pr/nv1_review_queue.csv

A. Label reliability (TRAIN): Cohen kappa between the two independent AI annotation passes
   (primary vs blind_review) per Type; label prevalence by annotation pipeline/provider
   (chi-square). Metadata is analysed, never used as a model feature.
B. Confident Learning on train out-of-fold probabilities of the best model: estimated label-noise
   rate per Type, validated against pass disagreements; a ranked review queue (outside Git).
C. Leave-one-source-out: retrain without one outlet, score that outlet's validation articles;
   the drop vs the in-distribution model measures reliance on outlet style.
D. Learning curve: validation macro AUPRC vs fraction of training groups.
Test split is never read.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.contracts import TYPES  # noqa: E402
from src.common.dataset import read_csv  # noqa: E402
from src.type_classification.analysis import cohen_kappa, confident_learning  # noqa: E402
from src.type_classification.evaluation import auprc, macro_f1  # noqa: E402
from src.type_classification.multilabel import CLASSIFIERS, SEED, Features  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from report_type_nv1 import latest_runs  # noqa: E402
from train_type_nv1 import Split, load_cache  # noqa: E402

OUT = ROOT / "docs/type_classification/nv1"
SHORT = ["Curiosity", "Exaggeration", "Emotional", "Question", "Misleading"]
LINEAR_KEY = "t1_1_tfidf_title_content_br"  # transparent linear model reused for C and D
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def macro_ap(P, Y):
    return float(np.nanmean([auprc(Y[:, k], P[:, k]) for k in range(Y.shape[1])]))


def pass_labels(passes, phase):
    for p in passes:
        if p.get("phase") == phase:
            types = (p.get("output") or {}).get("types") or {}
            try:
                return [int(types[t]["label"]) for t in TYPES]
            except (KeyError, TypeError, ValueError):
                return None
    return None


def reliability(rows, Y):
    pairs, ids, adjudicated = [], [], 0
    for i, r in enumerate(rows):
        try:
            passes = json.loads(r.get("annotation_passes_json") or "[]")
        except json.JSONDecodeError:
            continue
        a, b = pass_labels(passes, "primary"), pass_labels(passes, "blind_review")
        if a is not None and b is not None:
            pairs.append((a, b))
            ids.append(i)
            adjudicated += any(p.get("phase") == "adjudication" for p in passes)
    A = np.array([p[0] for p in pairs])
    B = np.array([p[1] for p in pairs])
    per_label = {
        t: {
            "kappa": cohen_kappa(A[:, k], B[:, k]),
            "raw_agreement": float((A[:, k] == B[:, k]).mean()),
            "prevalence_primary": float(A[:, k].mean()),
            "prevalence_review": float(B[:, k].mean()),
        }
        for k, t in enumerate(TYPES)
    }
    bias = {}
    for col in ("annotation_quality_mode", "label_provider"):
        vals = np.array([str(r[col] or "(trống)") for r in rows])
        keep = [v for v in sorted(set(vals)) if (vals == v).sum() >= 200]
        bias[col] = {}
        for k, t in enumerate(TYPES):
            table = [
                [
                    int(((vals == v) & (Y[:, k] == 1)).sum()),
                    int(((vals == v) & (Y[:, k] == 0)).sum()),
                ]
                for v in keep
            ]
            chi2, p, _, _ = stats.chi2_contingency(table)
            bias[col][t] = {
                "p": float(p),
                "chi2": float(chi2),
                "prevalence": {str(v): float(Y[vals == v, k].mean()) for v in keep},
                "n": {str(v): int((vals == v).sum()) for v in keep},
            }
    # is the pipeline effect separable from the outlet effect? Cramer's V source x pipeline
    src = np.array([r["source"] for r in rows])
    mode = np.array([r["annotation_quality_mode"] for r in rows])
    s_u, m_u = sorted(set(src)), sorted(set(mode))
    ct = np.array([[int(((src == s) & (mode == m)).sum()) for m in m_u] for s in s_u])
    chi2 = stats.chi2_contingency(ct)[0]
    confound = {
        "cramers_v_source_pipeline": float(np.sqrt(chi2 / (ct.sum() * (min(ct.shape) - 1)))),
        "sources_with_single_pipeline": int(sum((row >= 50).sum() <= 1 for row in ct)),
        "sources": len(s_u),
        "pipeline_by_source": {
            s: {m: int(c) for m, c in zip(m_u, row) if c} for s, row in zip(s_u, ct)
        },
    }
    disagree = np.zeros_like(Y, dtype=bool)
    disagree[ids] = A != B
    double = np.zeros(len(rows), dtype=bool)
    double[ids] = True
    return (
        {
            "n_double_annotated": len(ids),
            "n_adjudicated": adjudicated,
            "per_label": per_label,
            "prevalence_bias": bias,
            "confound": confound,
        },
        disagree,
        double,
    )


def noise_audit(oof, Y, rows, disagree, double, queue_path):
    suspect, margin, th = confident_learning(oof, Y)
    modes = np.array([r["annotation_quality_mode"] for r in rows])
    out = {}
    for k, t in enumerate(TYPES):
        s = suspect[:, k]
        d_s, d_n = disagree[double & s, k], disagree[double & ~s, k]
        table = [[int(d_s.sum()), int((~d_s).sum())], [int(d_n.sum()), int((~d_n).sum())]]
        odds, p = stats.fisher_exact(table) if min(map(sum, table)) else (float("nan"), 1.0)
        out[t] = {
            "suspect_rate": float(s.mean()),
            "suspect_missing_positive": int((s & (Y[:, k] == 0)).sum()),
            "suspect_false_positive": int((s & (Y[:, k] == 1)).sum()),
            "thresholds": th[k],
            "pass_disagreement_if_suspect": float(d_s.mean()) if len(d_s) else None,
            "pass_disagreement_if_not": float(d_n.mean()) if len(d_n) else None,
            "enrichment_odds_ratio": float(odds),
            "enrichment_p": float(p),
            "suspect_rate_by_mode": {
                m: float(s[modes == m].mean())
                for m in sorted(set(modes))
                if (modes == m).sum() >= 200
            },
        }
    items = [(margin[i, k], i, k) for i, k in zip(*np.where(suspect))]
    items.sort(reverse=True)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with open(queue_path, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(
            [
                "rank",
                "article_id",
                "type",
                "given_label",
                "oof_probability",
                "margin",
                "annotation_quality_mode",
                "title",
            ]
        )
        for rank, (m, i, k) in enumerate(items, 1):
            wr.writerow(
                [
                    rank,
                    rows[i]["article_id"],
                    TYPES[k],
                    int(Y[i, k]),
                    round(float(oof[i, k]), 4),
                    round(float(m), 4),
                    rows[i]["annotation_quality_mode"],
                    rows[i]["title"],
                ]
            )
    examples = [
        {"type": TYPES[k], "given": int(Y[i, k]), "p": float(oof[i, k]), "title": rows[i]["title"]}
        for _, i, k in items[:15]
    ]
    return {"per_label": out, "total_flags": len(items), "examples": examples}


def fit_predict(params, blocks, train_view, Ytr, test_view, hand_names):
    feats = Features(blocks, hand_names)
    X = feats.fit_transform(train_view)
    clf = CLASSIFIERS["br_lr"](**params).fit(X, Ytr)
    return clf.predict_proba(feats.transform(test_view))


def loso(train, val, run, hand_names, min_train=300, min_val=60):
    params = run["selected_params"]
    blocks = ["tfidf_title", "tfidf_content"]
    th = [run["metrics_by_threshold_mode"]["per_label"]["thresholds"][t] for t in TYPES]
    P_in = np.load(run["dir"] / "val_probs.npy")
    src_tr, src_va = np.array(train.source), np.array(val.source)
    res = {}
    for s in sorted(set(src_va)):
        mtr, mva = src_tr == s, src_va == s
        if mtr.sum() < min_train or mva.sum() < min_val:
            continue
        keep = np.where(~mtr)[0]
        P_out = fit_predict(
            params,
            blocks,
            train.subset(keep),
            train.Y[keep],
            val.subset(np.where(mva)[0]),
            hand_names,
        )
        Yv = val.Y[mva]
        res[s] = {
            "n_train_removed": int(mtr.sum()),
            "n_val": int(mva.sum()),
            "in_dist_macro_auprc": macro_ap(P_in[mva], Yv),
            "loso_macro_auprc": macro_ap(P_out, Yv),
            "in_dist_macro_f1": macro_f1(P_in[mva], Yv, th),
            "loso_macro_f1": macro_f1(P_out, Yv, th),
        }
        print("LOSO", s, json.dumps(res[s]), flush=True)
    return res


def learning_curve(train, val, run, hand_names, fracs=(0.1, 0.25, 0.5, 0.75, 1.0)):
    params = run["selected_params"]
    groups = np.unique(train.groups)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(groups)
    out = []
    for f in fracs:
        chosen = set(perm[: max(1, int(round(f * len(groups))))])
        idx = np.array([i for i, g in enumerate(train.groups) if g in chosen])
        P = fit_predict(
            params,
            ["tfidf_title", "tfidf_content"],
            train.subset(idx),
            train.Y[idx],
            val.view,
            hand_names,
        )
        out.append(
            {
                "fraction": f,
                "n_train": int(len(idx)),
                "macro_auprc": macro_ap(P, val.Y),
                "per_label_auprc": {t: auprc(val.Y[:, k], P[:, k]) for k, t in enumerate(TYPES)},
            }
        )
        print("LC", f, out[-1]["macro_auprc"], flush=True)
    return out


def figures(ins, fig_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": SURF,
            "savefig.facecolor": SURF,
            "axes.edgecolor": GRID,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.labelcolor": INK2,
        }
    )

    def clean(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)

    # 10. reliability (kappa) next to CL suspect rate: two panels, one axis each
    rel = ins["reliability"]["per_label"]
    cl = ins["noise_audit"]["per_label"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    kap = [rel[t]["kappa"] for t in TYPES][::-1]
    axes[0].barh(SHORT[::-1], kap, color=BLUE, height=0.6)
    for i, v in enumerate(kap):
        axes[0].text(v + 0.01, i, f"{v:.2f}".replace(".", ","), va="center", fontsize=8, color=INK)
    axes[0].set_xlim(0, 1)
    axes[0].set_title(
        f"κ giữa 2 lượt gán AI độc lập\n(n={ins['reliability']['n_double_annotated']:,} bài)".replace(
            ",", "."
        ),
        loc="left",
        fontsize=10,
        color=INK,
    )
    sus = [cl[t]["suspect_rate"] * 100 for t in TYPES][::-1]
    axes[1].barh(SHORT[::-1], sus, color=ORANGE, height=0.6)
    for i, v in enumerate(sus):
        axes[1].text(v + 0.1, i, f"{v:.1f}%".replace(".", ","), va="center", fontsize=8, color=INK)
    axes[1].set_title(
        "% bài train nghi nhãn sai\n(Confident Learning)", loc="left", fontsize=10, color=INK
    )
    for ax in axes:
        clean(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "10_chat_luong_nhan.png", dpi=150)
    plt.close(fig)
    # 11. LOSO dumbbell
    lo = ins["leave_one_source_out"]
    names = sorted(lo, key=lambda s: lo[s]["in_dist_macro_auprc"] - lo[s]["loso_macro_auprc"])
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1.3))
    for i, s in enumerate(names):
        a, b = lo[s]["in_dist_macro_auprc"], lo[s]["loso_macro_auprc"]
        ax.plot([b, a], [i, i], color=GRID, lw=2, zorder=1)
        ax.plot(a, i, "o", color=BLUE, ms=7, label="có báo này trong train" if i == 0 else None)
        ax.plot(b, i, "o", color=ORANGE, ms=7, label="bỏ báo này khỏi train" if i == 0 else None)
        ax.text(
            max(a, b) + 0.01,
            i,
            f"−{a - b:.3f}".replace(".", ","),
            va="center",
            fontsize=8,
            color=INK,
        )
    ax.set_yticks(range(len(names)), [f"{s} (val {lo[s]['n_val']})" for s in names], fontsize=8)
    ax.set_xlabel("Macro AUPRC trên bài validation của báo đó")
    ax.set_title(
        "Leave-one-source-out: mô hình có phụ thuộc phong cách nguồn?", loc="left", color=INK
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    clean(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "11_leave_one_source_out.png", dpi=150)
    plt.close(fig)
    # 12. learning curve
    lc = ins["learning_curve"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = [p["n_train"] for p in lc]
    for k, (t, s) in enumerate(zip(TYPES, SHORT)):
        y = [p["per_label_auprc"][t] for p in lc]
        ax.plot(x, y, "-o", color=CAT[k], lw=1.5, ms=4)
        ax.text(x[-1] * 1.02, y[-1], s, va="center", fontsize=8, color=INK)
    ax.plot(x, [p["macro_auprc"] for p in lc], "-o", color=INK, lw=2.5, ms=5)
    ax.text(x[-1] * 1.02, lc[-1]["macro_auprc"], "Macro", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Số bài train (lấy mẫu theo nhóm leakage)")
    ax.set_ylabel("AUPRC trên validation")
    ax.set_title("Learning curve: thêm dữ liệu gán nhãn còn giúp bao nhiêu?", loc="left", color=INK)
    ax.set_xlim(0, x[-1] * 1.18)
    ax.grid(color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "12_learning_curve.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument(
        "--queue", required=True, help="review queue CSV (contains titles; keep outside Git)"
    )
    args = ap.parse_args()
    runs = latest_runs()
    best = json.loads((OUT / "results.json").read_text(encoding="utf-8"))["best_overall"]
    cache = load_cache(args.cache)
    groups = {
        r["article_id"]: r["leakage_group_id"]
        for r in read_csv(Path(args.dataset_root) / "split_assignments.csv")
    }
    train = Split(args.dataset_root, "train", cache, groups)
    val = Split(args.dataset_root, "validation", cache, groups)
    from src.common.dataset import ClickbaitDataset

    rows = ClickbaitDataset(args.dataset_root, "train", verify_checksums=False).rows
    rel, disagree, double = reliability(rows, train.Y)
    oof = np.load(runs[best]["dir"] / "oof_probs.npy")
    ins = {
        "best_model_for_noise_audit": best,
        "reliability": rel,
        "noise_audit": noise_audit(oof, train.Y, rows, disagree, double, Path(args.queue)),
        "leave_one_source_out": loso(train, val, runs[LINEAR_KEY], cache["hand_names"]),
        "learning_curve": learning_curve(train, val, runs[LINEAR_KEY], cache["hand_names"]),
    }
    (OUT / "insights.json").write_text(
        json.dumps(ins, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    figures(ins, OUT / "figures")
    print(
        json.dumps(
            {
                "kappa": {t: round(v["kappa"], 3) for t, v in rel["per_label"].items()},
                "suspect_rate": {
                    t: round(v["suspect_rate"], 4)
                    for t, v in ins["noise_audit"]["per_label"].items()
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
