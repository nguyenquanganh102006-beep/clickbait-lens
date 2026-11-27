"""NV1 (classical multi-label Type classification): features, analysis, models, evaluation."""

import numpy as np
import pytest

from src.type_classification import analysis, evaluation, features, multilabel


def test_handcrafted_features_are_deterministic_and_named():
    row = {
        "title": "Vì sao 5 con giáp này giàu nhất? Sự thật khiến bạn sốc!",
        "lead": "Năm 2026 có 5 con giáp...",
        "content": "Năm 2026 có 5 con giáp may mắn. " * 20,
    }
    f = features.row_features(row)
    assert list(f) == features.FEATURE_NAMES
    assert f["t_question_mark"] == 1 and f["t_exclaim"] == 1 and f["t_starts_number"] == 0
    assert f["lex_question"] >= 1 and f["lex_emotional"] >= 1 and f["lex_exaggeration"] >= 1
    assert f["tc_numbers_missing"] == 0  # "5" appears in the body
    assert features.row_features(row) == f
    X = features.feature_matrix([row, {"title": "", "lead": "", "content": ""}])
    assert X.shape == (2, len(features.FEATURE_NAMES)) and np.isfinite(X).all()


def test_lexicon_matches_whole_words_only():
    counts = features.lexicon_counts("Cực phẩm nhất định")  # 'nhất' is a word here, 'cực' too
    assert counts["exaggeration"] == 2
    assert features.lexicon_counts("Sao Việt")["question"] == 1  # known ambiguity, documented
    assert features.lexicon_counts("chiếc ghế")["emotional"] == 0  # no substring hits


def test_bh_fdr_monotone_and_bounded():
    q = analysis.bh_fdr([0.001, 0.04, 0.03, 0.5])
    assert np.all(q <= 1) and q[0] <= q[2] <= q[1] <= q[3]


def test_label_structure_and_fightin_words():
    Y = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    s = analysis.label_structure(Y, ["a", "b"])
    assert s["cooccurrence"] == [[2, 1], [1, 2]] and s["conditional_p_j_given_i"][0][1] == 0.5
    docs = [["sốc", "x"]] * 30 + [["y", "x"]] * 30
    top = analysis.fightin_words(docs, [1] * 30 + [0] * 30, top=1, min_count=5)
    assert top[0]["term"] == "sốc" and top[0]["z"] > 3


def _toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    Y = (X[:, :5] + rng.normal(scale=0.5, size=(n, 5)) > 0.6).astype(float)
    return X, Y


@pytest.mark.parametrize("name", ["prior", "br_lr", "ecc_lr", "lp_lr", "br_svm"])
def test_classifiers_return_probabilities(name):
    X, Y = _toy()
    clf = multilabel.CLASSIFIERS[name]().fit(X[:90], Y[:90])
    P = clf.predict_proba(X[90:])
    assert P.shape == (30, 5) and np.all((P >= 0) & (P <= 1))


def test_stacking_meta_shapes():
    X, Y = _toy()
    P = multilabel.BinaryRelevance().fit(X, Y).predict_proba(X)
    meta = multilabel.StackingMeta().fit(np.hstack([P, P]), Y)
    assert meta.predict_proba(np.hstack([P, P])).shape == Y.shape


def test_features_fit_only_on_given_rows():
    titles = ["giá vàng hôm nay", "giá xăng hôm nay", "vì sao giá vàng tăng"] * 4
    view = {
        "title": titles,
        "content": ["x y"] * 12,
        "hand": np.arange(24, dtype=float).reshape(12, 2),
        "sem": None,
        "source": list("pq") * 6,
    }
    feats = multilabel.Features(["tfidf_title", "hand", "source"], ["h1", "h2"])
    Xtr = feats.fit_transform({k: (v[:9] if v is not None else None) for k, v in view.items()})
    Xte = feats.transform({k: (v[9:] if v is not None else None) for k, v in view.items()})
    assert Xtr.shape[1] == Xte.shape[1] == len(feats.names())


def test_thresholds_and_metrics():
    rng = np.random.default_rng(1)
    Y = (rng.random((300, 5)) < [0.3, 0.2, 0.15, 0.1, 0.05]).astype(float)
    P = np.clip(Y * 0.4 + rng.random((300, 5)) * 0.5, 0, 1)
    per = evaluation.fit_thresholds(P, Y, "per_label")
    glob = evaluation.fit_thresholds(P, Y, "global")
    assert evaluation.macro_f1(P, Y, per) >= evaluation.macro_f1(P, Y, glob) - 1e-12
    det = Y.any(1).astype(int)
    r = evaluation.evaluate(P, Y, per, det)
    assert {"micro_f1", "macro_f1", "hamming_loss", "macro_auprc", "logic"} <= set(r)
    assert abs(r["macro_f1"] - evaluation.macro_f1(P, Y, per)) < 1e-9  # contract metric agrees
    groups = np.repeat(np.arange(150), 2)
    lo, hi = evaluation.bootstrap_ci(P, Y, per, groups, n=200)
    assert lo <= r["macro_f1"] <= hi
    d = evaluation.paired_bootstrap(P, per, P, per, Y, groups, n=50)
    assert d["mean_diff"] == 0 and d["p_not_better"] == 1.0


def test_cohen_kappa_and_confident_learning():
    assert analysis.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert abs(analysis.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-12
    Y = np.array([[1], [1], [1], [0], [0], [0], [0], [0]])
    oof = np.array([[0.9], [0.8], [0.1], [0.1], [0.2], [0.1], [0.95], [0.05]])
    suspect, margin, th = analysis.confident_learning(oof, Y)
    # row 2: labeled 1 but confidently negative; row 6: labeled 0 but above t1
    assert suspect[:, 0].tolist() == [False, False, True, False, False, False, True, False]
    assert margin[6, 0] > 0 and margin[2, 0] > 0 and 0 < th[0]["t1"] < 1
