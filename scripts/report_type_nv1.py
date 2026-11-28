"""Aggregate NV1 runs into report tables, significance tests, error analysis and figures.

  python scripts/report_type_nv1.py --dataset-root ../data_pr/dataset_v1.0

Reads the latest results/<run_id>/ per experiment_key (validation probabilities saved by
train_type_nv1.py), writes docs/type_classification/nv1/results.json, results_tables.md and
figures/6_*.png .. 9_*.png. Only validation predictions are used; test is never read.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.contracts import TYPES  # noqa: E402
from src.common.dataset import ClickbaitDataset, read_csv  # noqa: E402
from src.type_classification.evaluation import f1, macro_f1, paired_bootstrap  # noqa: E402

OUT = ROOT / "docs/type_classification/nv1"
SHORT = ["Curiosity", "Exaggeration", "Emotional", "Question", "Misleading"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
ORANGE = "#eb6834"


def latest_runs():
    runs = {}
    for p in sorted((ROOT / "results").glob("type_classification__*/metrics.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        if "experiment_key" in m and (p.parent / "val_probs.npy").exists():
            m["dir"] = p.parent
            runs[m["experiment_key"]] = m  # sorted by UTC stamp: last wins
    return runs


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}".replace(".", ",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--baseline", default="t1_1_tfidf_title_br")
    args = ap.parse_args()
    runs = latest_runs()
    ds = ClickbaitDataset(args.dataset_root, "validation", verify_checksums=False)
    groups = {
        r["article_id"]: r["leakage_group_id"]
        for r in read_csv(Path(args.dataset_root) / "split_assignments.csv")
    }
    rows = ds.rows
    Y = np.array([[int(r[c]) for c in ds.schema["type_columns"]] for r in rows], float)
    G = np.array([groups[r["article_id"]] for r in rows])
    P = {k: np.load(m["dir"] / "val_probs.npy") for k, m in runs.items()}
    TH = {
        k: [m["metrics_by_threshold_mode"]["per_label"]["thresholds"][t] for t in TYPES]
        for k, m in runs.items()
    }
    main_keys = [k for k, m in runs.items() if not m.get("diagnostic")]
    best = max(
        main_keys, key=lambda k: runs[k]["metrics_by_threshold_mode"]["per_label"]["macro_f1"]
    )
    lines = []
    w = lines.append
    # ---- table 1: all models (per-label thresholds fit on train OOF)
    w(
        "### Bảng 1. So sánh mô hình trên validation (threshold từng nhãn, chọn trên OOF của train)\n"
    )
    w(
        "| Mã | Mô hình | Macro-F1 [95% CI] | Micro-F1 | Hamming ↓ | Subset acc | Macro AUPRC | "
        "OOF Macro-F1 (train) |"
    )
    w("|---|---|---|---:|---:|---:|---:|---:|")
    for k in sorted(runs, key=lambda k: (runs[k]["experiment"], k)):
        r = runs[k]["metrics_by_threshold_mode"]["per_label"]
        lo, hi = r["macro_f1_ci95"]
        tag = " *(chẩn đoán)*" if runs[k].get("diagnostic") else ""
        w(
            f"| {runs[k]['experiment']} | `{k}`{tag} | **{fmt(r['macro_f1'])}** [{fmt(lo)}–{fmt(hi)}] | "
            f"{fmt(r['micro_f1'])} | {fmt(r['hamming_loss'])} | {fmt(r['subset_accuracy'])} | "
            f"{fmt(r['macro_auprc'])} | {fmt(r['train_oof_macro_f1'])} |"
        )
    # ---- table 2: per-label F1
    w("\n### Bảng 2. F1 từng nhãn (validation)\n")
    w("| Mô hình | " + " | ".join(SHORT) + " |")
    w("|---|" + "---:|" * 5)
    per_label = {}
    for k in sorted(main_keys, key=lambda k: (runs[k]["experiment"], k)):
        r = runs[k]["metrics_by_threshold_mode"]["per_label"]["per_label"]
        per_label[k] = [r[t]["per_class"]["1"]["f1"] for t in TYPES]
        w(f"| `{k}` | " + " | ".join(fmt(v) for v in per_label[k]) + " |")
    best_per_label = {t: max(main_keys, key=lambda k: per_label[k][i]) for i, t in enumerate(TYPES)}
    # ---- table 3: threshold strategy effect (T2.5-like analysis inside NV1)
    w("\n### Bảng 3. Ảnh hưởng của chiến lược threshold (Macro-F1 validation)\n")
    w("| Mô hình | 0,5 cố định | 1 ngưỡng chung | Ngưỡng từng nhãn |")
    w("|---|---:|---:|---:|")
    for k in sorted(main_keys, key=lambda k: (runs[k]["experiment"], k)):
        mm = runs[k]["metrics_by_threshold_mode"]
        w(
            f"| `{k}` | {fmt(mm['fixed']['macro_f1'])} | {fmt(mm['global']['macro_f1'])} | "
            f"{fmt(mm['per_label']['macro_f1'])} |"
        )
    # ---- table 4: paired bootstrap tests
    w("\n### Bảng 4. Kiểm định bootstrap ghép cặp (theo nhóm leakage, 2.000 lần)\n")
    w("| So sánh (A − B) | ΔMacro-F1 | 95% CI | p (A không tốt hơn) |")
    w("|---|---:|---|---:|")
    pairs = [
        (best, args.baseline),
        ("t1_1_tfidf_title_content_br", "t1_1_tfidf_title_br"),
        ("t1_2_tfidf_chains", "t1_1_tfidf_title_content_br"),
        ("t1_5_combined_lr", "t1_1_tfidf_title_content_br"),
        ("t1_5_stacking", "t1_5_combined_lr"),
        ("t1_5_stacking", "t1_5_stacking_own_label"),
        ("t1_3_hand_catboost", "t1_0_prior"),
    ]
    tests = []
    for a, b in pairs:
        if a in P and b in P and a != b:
            d = paired_bootstrap(P[a], TH[a], P[b], TH[b], Y, G)
            tests.append({"a": a, "b": b, **d})
            w(
                f"| `{a}` − `{b}` | {fmt(d['mean_diff'])} | [{fmt(d['ci95'][0])}; {fmt(d['ci95'][1])}] | "
                f"{fmt(d['p_not_better'])} |"
            )
    # ---- table 4b: per-label ablations (research questions 2 and 3 of the branch)
    w("\n### Bảng 4b. Ablation theo từng nhãn (ΔF1, bootstrap ghép cặp theo nhóm leakage)\n")
    w("| Ablation (A − B) | " + " | ".join(SHORT) + " |")
    w("|---|" + "---:|" * 5)
    ablations = [
        ("Thêm nội dung", "t1_1_tfidf_title_content_br", "t1_1_tfidf_title_br"),
        ("Logit mọi nhãn vs cùng nhãn", "t1_5_stacking", "t1_5_stacking_own_label"),
    ]
    label_tests = []
    for name, a, b in ablations:
        if a not in P or b not in P:
            continue
        cells = []
        for i, t in enumerate(TYPES):
            d = paired_bootstrap(P[a], TH[a], P[b], TH[b], Y, G, label=i)
            label_tests.append({"ablation": name, "a": a, "b": b, "label": t, **d})
            star = "*" if d["ci95"][0] > 0 or d["ci95"][1] < 0 else ""
            v = f"{d['mean_diff']:+.3f}".replace("-0.000", "0.000").replace("+0.000", "0.000")
            cells.append(f"{v}{star}".replace(".", ",").replace("-", "−"))
        w(f"| {name} (`{a}` − `{b}`) | " + " | ".join(cells) + " |")
    w("\n\\* khoảng tin cậy 95% không chứa 0.")
    # ---- table 5: per-label best method + logic
    w("\n### Bảng 5. Mô hình tốt nhất theo từng nhãn\n")
    w("| Nhãn | Mô hình tốt nhất | F1 | Mô hình chung tốt nhất (`" + best + "`) |")
    w("|---|---|---:|---:|")
    for i, t in enumerate(TYPES):
        k = best_per_label[t]
        w(f"| {SHORT[i]} | `{k}` | {fmt(per_label[k][i])} | {fmt(per_label[best][i])} |")
    lg = runs[best]["metrics_by_threshold_mode"]["per_label"]["logic"]
    w(
        f"\nRàng buộc Detection–Type của `{best}`: dự đoán có Type trên bài gold non-clickbait "
        f"{fmt(lg['types_on_gold_nonclickbait_rate'])}; không dự đoán Type nào trên bài gold clickbait "
        f"{fmt(lg['no_type_on_gold_clickbait_rate'])}; F1 của Detection suy ra từ Type "
        f"{fmt(lg['derived_detection_f1'])}."
    )
    # ---- error analysis on best model
    Pb, thb = P[best], np.asarray(TH[best])
    pred = Pb >= thb
    errors = {}
    for i, t in enumerate(TYPES):
        fp = np.where(pred[:, i] & (Y[:, i] == 0))[0]
        fn = np.where(~pred[:, i] & (Y[:, i] == 1))[0]
        errors[t] = {
            "false_positives": [
                [rows[j]["title"], float(Pb[j, i])] for j in fp[np.argsort(-Pb[fp, i])][:8]
            ],
            "false_negatives": [
                [rows[j]["title"], float(Pb[j, i])] for j in fn[np.argsort(Pb[fn, i])][:8]
            ],
        }
    # confusion between labels: when gold has i, how often is j predicted (co-prediction)
    conf = np.array([[float(pred[Y[:, i] == 1, j].mean()) for j in range(5)] for i in range(5)])
    result = {
        "best_overall": best,
        "best_per_label": best_per_label,
        "paired_tests": tests,
        "per_label_ablations": label_tests,
        "errors_best": errors,
        "gold_i_predicted_j": conf.round(3).tolist(),
        "per_source_best": runs[best]["per_source_macro_f1"],
        "source_only": runs.get("t1_0_source_only", {})
        .get("metrics_by_threshold_mode", {})
        .get("per_label", {})
        .get("macro_f1"),
    }
    (OUT / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (OUT / "results_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figures(runs, main_keys, per_label, best, conf, OUT / "figures")
    print("\n".join(lines))


def figures(runs, keys, per_label, best, conf, fig_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

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
    order = sorted(
        keys, key=lambda k: runs[k]["metrics_by_threshold_mode"]["per_label"]["macro_f1"]
    )
    # 6. model comparison with CI
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(order) + 1.2))
    for i, k in enumerate(order):
        r = runs[k]["metrics_by_threshold_mode"]["per_label"]
        lo, hi = r["macro_f1_ci95"]
        c = ORANGE if k == best else BLUE[4]
        ax.plot([lo, hi], [i, i], color=c, lw=2)
        ax.plot(r["macro_f1"], i, "o", color=c, ms=7)
        ax.text(
            hi + 0.005,
            i,
            f"{r['macro_f1']:.3f}".replace(".", ","),
            va="center",
            fontsize=8,
            color=INK,
        )
    ax.set_yticks(range(len(order)), [f"{runs[k]['experiment']}  {k}" for k in order], fontsize=8)
    ax.set_xlabel("Macro-F1 validation (điểm) và 95% CI bootstrap theo nhóm (thanh)")
    ax.set_title("So sánh mô hình NV1 (cam: tốt nhất)", loc="left", color=INK)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "6_so_sanh_mo_hinh.png", dpi=150)
    plt.close(fig)
    # 7. per-label F1 heatmap
    seq = LinearSegmentedColormap.from_list("seq", BLUE)
    M = np.array([per_label[k] for k in order])
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(order) + 1.4))
    im = ax.imshow(M, cmap=seq, vmin=0, vmax=max(0.01, M.max()), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(5):
            ax.text(
                j,
                i,
                f"{M[i, j]:.2f}".replace(".", ","),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if M[i, j] > M.max() * 0.6 else INK,
            )
    ax.set_xticks(range(5), SHORT, rotation=25, ha="right")
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.set_title("F1 từng nhãn trên validation", loc="left", color=INK)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "7_f1_tung_nhan.png", dpi=150)
    plt.close(fig)
    # 8. threshold strategy effect
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(order) + 1.2))
    for i, k in enumerate(order):
        mm = runs[k]["metrics_by_threshold_mode"]
        xs = [mm[m]["macro_f1"] for m in ("fixed", "global", "per_label")]
        ax.plot([xs[0], xs[2]], [i, i], color=GRID, lw=2, zorder=1)
        for x, c, lab in zip(
            xs, [INK2, BLUE[3], BLUE[6]], ["0,5 cố định", "1 ngưỡng chung", "ngưỡng từng nhãn"]
        ):
            ax.plot(x, i, "o", color=c, ms=6, label=lab if i == 0 else None, zorder=2)
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.set_xlabel("Macro-F1 validation")
    ax.set_title("Hiệu quả tối ưu threshold (chọn trên OOF train)", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "8_threshold.png", dpi=150)
    plt.close(fig)
    # 9. label confusion (best model)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(conf, cmap=seq, vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(
                j,
                i,
                f"{conf[i, j]:.2f}".replace(".", ","),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if conf[i, j] > 0.6 else INK,
            )
    ax.set_xticks(range(5), SHORT, rotation=25, ha="right")
    ax.set_yticks(range(5), SHORT)
    ax.set_xlabel("nhãn được dự đoán j")
    ax.set_ylabel("bài gold có nhãn i")
    ax.set_title(f"Tỉ lệ dự đoán j khi gold có i\n({best})", loc="left", color=INK, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(fig_dir / "9_nham_lan_nhan.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
