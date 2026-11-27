"""Statistical analyses for NV1 (run on TRAIN only): label structure, feature tests, key terms.

- label_structure: prevalence, co-occurrence, P(j|i), lift = P(i,j) / (P(i) P(j))
- feature_tests: Mann-Whitney U per (feature, label) + rank-biserial effect size + BH-FDR
- lexicon_tests: Fisher exact test of "title contains a lexicon term" vs label, odds ratio
- fightin_words: log-odds ratio with informative Dirichlet prior (Monroe, Colaresi & Quinn,
  2008) -> z-scores of terms that distinguish positive vs negative titles of a label
"""

from collections import Counter

import numpy as np
from scipy import stats


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        running = min(running, p[i] * n / rank)
        adj[i] = running
    return adj


def label_structure(Y, names):
    Y = np.asarray(Y, int)
    n = len(Y)
    co = Y.T @ Y
    prev = Y.mean(0)
    cond = co / np.maximum(Y.sum(0)[:, None], 1)  # row i: P(j | i)
    lift = (co / n) / np.outer(prev, prev)
    combos = Counter(map(tuple, Y))
    return {
        "n": n,
        "prevalence": dict(zip(names, prev.round(4).tolist())),
        "label_cardinality": float(Y.sum(1).mean()),
        "label_density": float(Y.mean()),
        "distinct_combinations": len(combos),
        "top_combinations": [[[int(x) for x in c], int(k)] for c, k in combos.most_common(10)],
        "cooccurrence": co.tolist(),
        "conditional_p_j_given_i": cond.round(4).tolist(),
        "lift": lift.round(3).tolist(),
    }


def feature_tests(X, Y, feature_names, label_names):
    """For every feature x label: medians, rank-biserial r = 2*AUC - 1 (sign: + means larger in
    positives), Mann-Whitney p, BH-FDR across all tests."""
    rows = []
    for k, lab in enumerate(label_names):
        pos, neg = X[Y[:, k] == 1], X[Y[:, k] == 0]
        for j, f in enumerate(feature_names):
            a, b = pos[:, j], neg[:, j]
            if np.all(a == a[0]) and np.all(b == a[0]):
                u, p = len(a) * len(b) / 2, 1.0
            else:
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            r = 2 * u / (len(a) * len(b)) - 1
            rows.append(
                {
                    "label": lab,
                    "feature": f,
                    "mean_pos": float(a.mean()),
                    "mean_neg": float(b.mean()),
                    "rank_biserial": float(r),
                    "p": float(p),
                }
            )
    adj = bh_fdr([r["p"] for r in rows])
    for r, q in zip(rows, adj):
        r["q_bh"] = float(q)
    return rows


def lexicon_tests(hits, Y, lexicon_names, label_names):
    """hits [N, L] counts -> presence; Fisher exact per (lexicon, label), odds ratio, BH-FDR."""
    rows = []
    for k, lab in enumerate(label_names):
        for j, lex in enumerate(lexicon_names):
            h = hits[:, j] > 0
            y = Y[:, k] == 1
            table = [
                [int((h & y).sum()), int((h & ~y).sum())],
                [int((~h & y).sum()), int((~h & ~y).sum())],
            ]
            odds, p = stats.fisher_exact(table)
            rows.append(
                {
                    "label": lab,
                    "lexicon": lex,
                    "rate_pos": float(h[y].mean()),
                    "rate_neg": float(h[~y].mean()),
                    "odds_ratio": float(odds),
                    "p": float(p),
                }
            )
    adj = bh_fdr([r["p"] for r in rows])
    for r, q in zip(rows, adj):
        r["q_bh"] = float(q)
    return rows


def fightin_words(token_lists, y, top=20, min_count=10, alpha0=None):
    """Monroe et al. (2008) log-odds with informative Dirichlet prior (corpus frequencies).
    Returns top positive-class terms with z-score and counts."""
    y = np.asarray(y).astype(bool)
    pos, neg, allc = Counter(), Counter(), Counter()
    for toks, lab in zip(token_lists, y):
        (pos if lab else neg).update(toks)
        allc.update(toks)
    vocab = [w for w, c in allc.items() if c >= min_count]
    a_total = alpha0 or float(sum(allc[w] for w in vocab)) / 10  # prior strength
    tot = sum(allc[w] for w in vocab)
    n_pos, n_neg = sum(pos[w] for w in vocab), sum(neg[w] for w in vocab)
    out = []
    for w in vocab:
        a = a_total * allc[w] / tot
        lo_p = np.log((pos[w] + a) / (n_pos + a_total - pos[w] - a))
        lo_n = np.log((neg[w] + a) / (n_neg + a_total - neg[w] - a))
        var = 1 / (pos[w] + a) + 1 / (neg[w] + a)
        out.append((w, (lo_p - lo_n) / np.sqrt(var), pos[w], neg[w]))
    out.sort(key=lambda x: -x[1])
    return [
        {"term": w, "z": float(z), "count_pos": int(cp), "count_neg": int(cn)}
        for w, z, cp, cn in out[:top]
    ]
