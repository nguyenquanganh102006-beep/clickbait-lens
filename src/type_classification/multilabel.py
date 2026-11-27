"""Feature blocks and multi-label classifiers for NV1 (Type classification with classical ML).

Every block/classifier is fit on the rows it is given (train folds), then applied to held-out
rows. Classifiers return probabilities [N,5] for the 5 Type labels in contract order.
"""

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 20261124
N_JOBS = int(__import__("os").environ.get("NV1_JOBS", "5"))  # parallel label/chain fits


def _fit(model, X, y):
    return model.fit(X, y)


def _parallel_fit(models, X, ys):
    """Fit independent models on all cores (results identical to a sequential loop)."""
    from joblib import Parallel, delayed

    return Parallel(n_jobs=N_JOBS)(delayed(_fit)(m, X, y) for m, y in zip(models, ys))


CONTENT_CHARS = 2000  # lead region of the body; whole articles (median ~655 words) add noise


# ----------------------------------------------------------------------------- feature blocks
class Block:
    """fit(view) / transform(view) -> sparse [n, d]. view: dict of aligned arrays/lists."""

    def fit(self, view):
        return self

    def transform(self, view):
        raise NotImplementedError


class TfidfTitle(Block):
    """Syllable 1-2 grams + char_wb 2-5 grams of the title (Vietnamese needs no segmenter here:
    syllable bigrams cover most two-syllable words, char grams cover spelling variants)."""

    def fit(self, view):
        self.word = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=2, sublinear_tf=True
        ).fit(view["title"])
        self.char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=3,
            sublinear_tf=True,
            max_features=200_000,
        ).fit(view["title"])
        return self

    def transform(self, view):
        return sparse.hstack(
            [self.word.transform(view["title"]), self.char.transform(view["title"])]
        )

    def names(self):
        return [f"title_w:{t}" for t in self.word.get_feature_names_out()] + [
            f"title_c:{t}" for t in self.char.get_feature_names_out()
        ]


class TfidfContent(Block):
    def fit(self, view):
        self.vec = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.5,
            sublinear_tf=True,
            max_features=200_000,
        ).fit(view["content"])
        return self

    def transform(self, view):
        return self.vec.transform(view["content"])

    def names(self):
        return [f"content_w:{t}" for t in self.vec.get_feature_names_out()]


class Dense(Block):
    """Standardized dense block (handcrafted or semantic); scaler fit on train folds only."""

    def __init__(self, key, names=None):
        self.key, self._names = key, names

    def fit(self, view):
        self.scaler = StandardScaler().fit(view[self.key])
        return self

    def transform(self, view):
        return sparse.csr_matrix(self.scaler.transform(view[self.key]))

    def names(self):
        return self._names or [f"{self.key}_{i}" for i in range(self.scaler.n_features_in_)]


class Source(Block):
    """Diagnostic only: one-hot news source (metadata, never a model feature by contract)."""

    def fit(self, view):
        self.enc = OneHotEncoder(handle_unknown="ignore").fit(np.asarray(view["source"])[:, None])
        return self

    def transform(self, view):
        return self.enc.transform(np.asarray(view["source"])[:, None])

    def names(self):
        return [f"source:{s}" for s in self.enc.categories_[0]]


def make_blocks(names, hand_names=None):
    table = {
        "tfidf_title": TfidfTitle,
        "tfidf_content": TfidfContent,
        "hand": lambda: Dense("hand", hand_names),
        "sem": lambda: Dense("sem"),
        "source": Source,
    }
    return [table[n]() for n in names]


class Features:
    def __init__(self, block_names, hand_names=None):
        self.blocks = make_blocks(block_names, hand_names)

    def fit_transform(self, view):
        return sparse.hstack([b.fit(view).transform(view) for b in self.blocks]).tocsr()

    def transform(self, view):
        return sparse.hstack([b.transform(view) for b in self.blocks]).tocsr()

    def names(self):
        return [n for b in self.blocks for n in b.names()]


# ----------------------------------------------------------------------------- classifiers
def _lr(C, balanced=True):
    return LogisticRegression(
        C=C,
        class_weight="balanced" if balanced else None,
        solver="liblinear",
        max_iter=2000,
        random_state=SEED,
    )


class BinaryRelevance:
    """One independent binary classifier per label (base: logistic regression)."""

    def __init__(self, C=1.0, base="lr"):
        self.C, self.base = C, base

    def _make(self):
        if self.base == "lr":
            return _lr(self.C)
        if self.base == "svm":
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.svm import LinearSVC

            # LinearSVC has no probabilities: Platt scaling with inner 3-fold CV
            return CalibratedClassifierCV(
                LinearSVC(C=self.C, class_weight="balanced", random_state=SEED),
                method="sigmoid",
                cv=3,
            )
        raise ValueError(self.base)

    def fit(self, X, Y):
        k = Y.shape[1]
        self.models = _parallel_fit(
            [self._make() for _ in range(k)], X, [Y[:, j] for j in range(k)]
        )
        return self

    def predict_proba(self, X):
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.models])

    def coefficients(self):
        """[5, d] weights for linear LR bases (interpretability)."""
        return np.vstack([m.coef_[0] for m in self.models])


class EnsembleChains:
    """Ensemble of Classifier Chains (Read et al., 2011): each chain feeds the previous labels'
    predictions to the next classifier; random label orders averaged to remove order bias."""

    def __init__(self, C=1.0, n_chains=5):
        self.C, self.n_chains = C, n_chains

    def fit(self, X, Y):
        from sklearn.multioutput import ClassifierChain

        chains = [
            ClassifierChain(_lr(self.C), order="random", random_state=SEED + i)
            for i in range(self.n_chains)
        ]
        self.chains = _parallel_fit(chains, X, [Y] * self.n_chains)
        return self

    def predict_proba(self, X):
        return np.mean([c.predict_proba(X) for c in self.chains], axis=0)


class LabelPowerset:
    """Each observed label combination is one class (31 in train); label probability =
    sum of probabilities of the combinations containing it."""

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, Y):
        combos, cls = np.unique(Y.astype(int), axis=0, return_inverse=True)
        self.combos = combos
        self.model = LogisticRegression(
            C=self.C, class_weight="balanced", max_iter=3000, random_state=SEED
        ).fit(X, cls.ravel())
        return self

    def predict_proba(self, X):
        p = self.model.predict_proba(X)  # [N, n_combos] in self.model.classes_ order
        return p @ self.combos[self.model.classes_].astype(float)


class OvRCatBoost:
    """One CatBoost per label on dense features (non-linear interactions, robust to scale)."""

    def __init__(self, iterations=600, depth=6, learning_rate=0.05, threads=12):
        self.params = dict(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            auto_class_weights="Balanced",
            random_seed=SEED,
            thread_count=threads,
            verbose=0,
            allow_writing_files=False,
        )

    def fit(self, X, Y):
        from catboost import CatBoostClassifier

        X = X.toarray() if sparse.issparse(X) else X
        self.models = [CatBoostClassifier(**self.params).fit(X, Y[:, k]) for k in range(Y.shape[1])]
        return self

    def predict_proba(self, X):
        X = X.toarray() if sparse.issparse(X) else X
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.models])


class PriorBaseline:
    """T1.0: predicts each label's train prevalence for every article (no text used)."""

    def fit(self, X, Y):
        self.p = Y.mean(0)
        return self

    def predict_proba(self, X):
        return np.tile(self.p, (X.shape[0], 1))


CLASSIFIERS = {
    "prior": PriorBaseline,
    "br_lr": lambda **kw: BinaryRelevance(base="lr", **kw),
    "br_svm": lambda **kw: BinaryRelevance(base="svm", **kw),
    "ecc_lr": EnsembleChains,
    "lp_lr": LabelPowerset,
    "ovr_catboost": OvRCatBoost,
}


class StackingMeta:
    """Level-2 model: for each label, logistic regression on ALL base models' probabilities of
    ALL labels -> learns cross-label dependencies (e.g. P(emotional) informs curiosity)."""

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, P, Y):
        Z = self._z(P)
        self.models = [_lr(self.C).fit(Z, Y[:, k]) for k in range(Y.shape[1])]
        return self

    @staticmethod
    def _z(P):
        P = np.clip(P, 1e-4, 1 - 1e-4)
        return np.log(P / (1 - P))  # logits are better-scaled inputs than probabilities

    def predict_proba(self, P):
        Z = self._z(P)
        return np.column_stack([m.predict_proba(Z)[:, 1] for m in self.models])
