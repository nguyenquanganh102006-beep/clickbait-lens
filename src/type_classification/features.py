"""Handcrafted, interpretable features for multi-label Type classification (NV1).

Groups (prefix in the feature name):
  t_   title statistics and punctuation
  lex_ title lexicon hits per tactic (src.type_classification.lexicons)
  c_   content-head statistics and lexicon density
  tc_  title-content relation (targets Misleading presentation: does the body back the title?)
All functions are deterministic, label-free and use only the row's own text, so they can be
computed for every split without leakage. Anything fit on data (TF-IDF vocabulary, scaling)
lives in the model pipelines and is fit on train folds only.
"""

import math
import re
import unicodedata

import numpy as np

from src.type_classification.lexicons import LEXICONS

CONTENT_HEAD_TOKENS = 400  # body slice compared with the title (lead region of the article)
_TOKEN = re.compile(r"\w+", re.UNICODE)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_QUOTES = "\"“”‘’'«»"


def normalize(text):
    return unicodedata.normalize("NFC", text or "").lower()


def tokens(text):
    return _TOKEN.findall(normalize(text))


def _phrase_regex(phrases):
    alts = sorted({re.escape(p) for p in phrases}, key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)", re.UNICODE)


LEXICON_RE = {name: _phrase_regex(words) for name, words in LEXICONS.items()}


def lexicon_counts(text):
    t = normalize(text)
    return {name: len(rx.findall(t)) for name, rx in LEXICON_RE.items()}


def _numbers(text):
    return {n.replace(",", ".") for n in _NUMBER.findall(text or "")}


def row_features(row):
    """One article -> dict of float features (order fixed by FEATURE_NAMES)."""
    title = unicodedata.normalize("NFC", row["title"] or "")
    lead = row.get("lead", "") or ""
    content = row.get("content", "") or ""
    t_tok = tokens(title)
    c_tok = tokens(content)
    head = c_tok[:CONTENT_HEAD_TOKENS]
    letters = [ch for ch in title if ch.isalpha()]
    stripped = title.strip()
    f = {
        "t_chars": len(title),
        "t_words": len(t_tok),
        "t_avg_word_len": float(np.mean([len(w) for w in t_tok])) if t_tok else 0.0,
        "t_has_number": float(bool(_NUMBER.search(title))),
        "t_n_numbers": len(_NUMBER.findall(title)),
        "t_starts_number": float(bool(re.match(r"^\W*\d", title))),
        "t_question_mark": title.count("?"),
        "t_ends_question": float(stripped.endswith("?")),
        "t_exclaim": title.count("!"),
        "t_ellipsis": title.count("...") + title.count("…"),
        "t_ends_ellipsis": float(stripped.endswith("...") or stripped.endswith("…")),
        "t_quotes": sum(title.count(q) for q in _QUOTES),
        "t_colon": title.count(":"),
        "t_dash": title.count(" - ") + title.count("–") + title.count("—"),
        "t_comma": title.count(","),
        "t_upper_ratio": sum(ch.isupper() for ch in letters) / len(letters) if letters else 0.0,
        "t_allcaps_words": sum(1 for w in title.split() if len(w) > 1 and w.isupper()),
    }
    for name, n in lexicon_counts(title).items():
        f[f"lex_{name}"] = n
    c_lex = lexicon_counts(" ".join(head))
    per_k = 1000.0 / max(1, len(head))
    f.update(
        {
            "c_log_words": math.log1p(len(c_tok)),
            "c_lead_words": len(tokens(lead)),
            "c_emotional_per_1k": c_lex["emotional"] * per_k,
            "c_exaggeration_per_1k": c_lex["exaggeration"] * per_k,
            "c_question_marks": content.count("?"),
        }
    )
    t_set = set(t_tok)
    head_set, all_set, lead_set = set(head), set(c_tok), set(tokens(lead))
    t_nums = _numbers(title)
    body_nums = _numbers(content) | _numbers(lead)
    f.update(
        {
            "tc_title_in_head": len(t_set & head_set) / len(t_set) if t_set else 0.0,
            "tc_title_in_body": len(t_set & all_set) / len(t_set) if t_set else 0.0,
            "tc_title_in_lead": len(t_set & lead_set) / len(t_set) if t_set else 0.0,
            "tc_numbers_missing": len(t_nums - body_nums),
            "tc_numbers_missing_ratio": len(t_nums - body_nums) / len(t_nums) if t_nums else 0.0,
            "tc_title_lead_len_ratio": len(t_tok) / max(1, len(tokens(lead))),
        }
    )
    return f


FEATURE_NAMES = list(row_features({"title": "a", "lead": "", "content": ""}).keys())


def feature_matrix(rows):
    """rows -> float32 array [N, len(FEATURE_NAMES)]."""
    out = np.zeros((len(rows), len(FEATURE_NAMES)), dtype=np.float32)
    for i, r in enumerate(rows):
        f = row_features(r)
        out[i] = [f[k] for k in FEATURE_NAMES]
    return out
