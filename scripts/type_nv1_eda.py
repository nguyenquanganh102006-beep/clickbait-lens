"""NV1 exploratory + statistical analysis on TRAIN only (validation/test untouched).

  python scripts/type_nv1_eda.py --dataset-root ../data_pr/dataset_v1.0 --cache ../data_pr/nv1_cache

Writes docs/type_classification/nv1/eda.json and figures/*.png (summary artifacts for the report).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.contracts import TYPES  # noqa: E402
from src.common.dataset import ClickbaitDataset  # noqa: E402
from src.type_classification.analysis import (  # noqa: E402
    feature_tests,
    fightin_words,
    label_structure,
    lexicon_tests,
)
from src.type_classification.features import lexicon_counts, tokens  # noqa: E402
from src.type_classification.lexicons import LEXICONS  # noqa: E402

OUT = ROOT / "docs/type_classification/nv1"
SHORT = ["Curiosity", "Exaggeration", "Emotional", "Question", "Misleading"]
# design tokens (validated default palette): ink, secondary ink, grid, surface, blue ramp
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
NEG, MID, POS = "#d95926", "#f0efec", "#1c5cab"


def _fig_style(plt):
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


def figures(eda, fig_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    _fig_style(plt)
    fig_dir.mkdir(parents=True, exist_ok=True)
    seq = LinearSegmentedColormap.from_list("seq", BLUE)
    div = LinearSegmentedColormap.from_list("div", [NEG, MID, POS])
    # 1. prevalence
    prev = [eda["labels"]["prevalence"][t] * 100 for t in TYPES]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(SHORT[::-1], prev[::-1], color=BLUE[4], height=0.6)
    for i, v in enumerate(prev[::-1]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_xlabel("% bài trong train mang nhãn")
    ax.set_title("Tỉ lệ dương từng loại clickbait (train, n=19.629)", loc="left", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(fig_dir / "1_ti_le_nhan.png", dpi=150)
    plt.close(fig)
    # 2. conditional probability heatmap with lift annotation
    cond = np.array(eda["labels"]["conditional_p_j_given_i"])
    lift = np.array(eda["labels"]["lift"])
    fig, ax = plt.subplots(figsize=(6.2, 5))
    im = ax.imshow(cond, cmap=seq, vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(
                j,
                i,
                f"{cond[i, j]:.2f}\nlift {lift[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cond[i, j] > 0.55 else INK,
            )
    ax.set_xticks(range(5), SHORT, rotation=30, ha="right")
    ax.set_yticks(range(5), SHORT)
    ax.set_xlabel("nhãn j")
    ax.set_ylabel("điều kiện: bài có nhãn i")
    ax.set_title("P(j | i) và lift giữa các loại (train)", loc="left", color=INK)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(fig_dir / "2_dong_xuat_hien.png", dpi=150)
    plt.close(fig)
    # 3. feature effect heatmap (rank-biserial), only q<0.001 cells colored
    tests = eda["feature_tests"]
    feats = list(dict.fromkeys(r["feature"] for r in tests))
    M = np.full((len(feats), 5), np.nan)
    for r in tests:
        if r["q_bh"] < 1e-3:
            M[feats.index(r["feature"]), TYPES.index(r["label"])] = r["rank_biserial"]
    fig, ax = plt.subplots(figsize=(6.4, 0.26 * len(feats) + 1.5))
    lim = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1
    im = ax.imshow(M, cmap=div, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(len(feats)):
        for j in range(5):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7, color=INK)
    ax.set_xticks(range(5), SHORT, rotation=30, ha="right")
    ax.set_yticks(range(len(feats)), feats, fontsize=8)
    ax.set_title(
        "Hiệu ứng đặc trưng (rank-biserial; ô trống: q ≥ 0,001)", loc="left", color=INK, fontsize=10
    )
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "3_hieu_ung_dac_trung.png", dpi=150)
    plt.close(fig)
    # 4. key terms per label (small multiples)
    fig, axes = plt.subplots(1, 5, figsize=(15, 4.2))
    for ax, t, s in zip(axes, TYPES, SHORT):
        rows = eda["key_terms"][t][:12][::-1]
        ax.barh([r["term"] for r in rows], [r["z"] for r in rows], color=BLUE[5], height=0.6)
        ax.set_title(s, loc="left", color=INK, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("z log-odds")
    fig.suptitle(
        "Từ/cụm từ đặc trưng nhất của từng loại (Fightin' Words, tiêu đề train)",
        x=0.01,
        ha="left",
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "4_tu_khoa_dac_trung.png", dpi=150)
    plt.close(fig)
    # 5. prevalence by source (style confound)
    src = eda["by_source"]
    names = [s for s in src if src[s]["n"] >= 150]
    M = np.array([[src[s]["prevalence"][t] for t in TYPES] for s in names]) * 100
    fig, ax = plt.subplots(figsize=(6.6, 0.32 * len(names) + 1.4))
    im = ax.imshow(M, cmap=seq, aspect="auto", vmin=0)
    for i in range(len(names)):
        for j in range(5):
            ax.text(
                j,
                i,
                f"{M[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if M[i, j] > M.max() * 0.6 else INK,
            )
    ax.set_xticks(range(5), SHORT, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), [f"{s} ({src[s]['n']})" for s in names], fontsize=8)
    ax.set_title(
        "% bài mang nhãn theo nguồn báo (train, nguồn ≥150 bài)", loc="left", color=INK, fontsize=10
    )
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "5_nhan_theo_nguon.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()
    ds = ClickbaitDataset(args.dataset_root, "train", verify_checksums=False)
    schema = ds.schema
    rows = ds.rows
    Y = np.array([[int(r[c]) for c in schema["type_columns"]] for r in rows])
    cache = Path(args.cache)
    ids = json.loads((cache / "ids.json").read_text(encoding="utf-8"))
    pos = {a: i for i, a in enumerate(ids)}
    X = np.load(cache / "handcrafted.npy")[[pos[r["article_id"]] for r in rows]]
    names = json.loads((cache / "handcrafted_names.json").read_text(encoding="utf-8"))
    hits = np.array([[lexicon_counts(r["title"])[k] for k in LEXICONS] for r in rows])
    toks = []
    for r in rows:
        t = tokens(r["title"])
        toks.append(t + [f"{a} {b}" for a, b in zip(t, t[1:])])
    by_source = {}
    for s in sorted({r["source"] for r in rows}):
        m = np.array([r["source"] == s for r in rows])
        by_source[s] = {"n": int(m.sum()), "prevalence": dict(zip(TYPES, Y[m].mean(0).tolist()))}
    eda = {
        "split": "train",
        "labels": label_structure(Y, list(TYPES)),
        "feature_tests": feature_tests(X, Y, names, list(TYPES)),
        "lexicon_tests": lexicon_tests(hits, Y, list(LEXICONS), list(TYPES)),
        "key_terms": {t: fightin_words(toks, Y[:, k], top=25) for k, t in enumerate(TYPES)},
        "by_source": by_source,
        "detection_type_rule": {
            "clickbait_without_type": int(
                sum(int(r["clickbait_label"]) == 1 and not y.any() for r, y in zip(rows, Y))
            ),
            "type_without_clickbait": int(
                sum(int(r["clickbait_label"]) == 0 and y.any() for r, y in zip(rows, Y))
            ),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eda.json").write_text(json.dumps(eda, ensure_ascii=False, indent=1), encoding="utf-8")
    figures(eda, OUT / "figures")
    sig = [r for r in eda["feature_tests"] if r["q_bh"] < 1e-3]
    print(
        json.dumps(
            {
                "labels": eda["labels"]["prevalence"],
                "significant_feature_tests": len(sig),
                "of": len(eda["feature_tests"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
