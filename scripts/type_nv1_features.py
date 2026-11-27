"""Build the NV1 feature cache once (outside Git): handcrafted features + sentence embeddings.

  python scripts/type_nv1_features.py --dataset-root ../data_pr/dataset_v1.0 --out ../data_pr/nv1_cache

Writes ids.json (row order), handcrafted.npy [N,F] + handcrafted_names.json, and with
--embeddings: title_emb.npy / lead_emb.npy / content_emb.npy [N,384] (L2-normalized,
multilingual-e5-small, "query: " prefix, mean pooling). Nothing here reads labels or fits on data,
so every split is encoded the same way; models are fit on train folds only.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.dataset import file_sha256, read_csv  # noqa: E402
from src.type_classification.features import FEATURE_NAMES, feature_matrix  # noqa: E402

EMBEDDER = ("intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3")


def embed(texts, max_length, batch_size, threads):
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(EMBEDDER[0], revision=EMBEDDER[1])
    model = AutoModel.from_pretrained(EMBEDDER[0], revision=EMBEDDER[1]).eval()
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    order = np.argsort([len(t) for t in texts])  # length bucketing: less padding, same values
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            idx = order[s : s + batch_size]
            enc = tok(
                ["query: " + texts[i] for i in idx],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
            out[idx] = torch.nn.functional.normalize(v, dim=-1).numpy()
            if s // batch_size % 100 == 0:
                print(f"  {s + len(idx)}/{len(texts)}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--embeddings", action="store_true")
    p.add_argument("--threads", type=int, default=12)
    args = p.parse_args()
    root, out = Path(args.dataset_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(read_csv(root / "data.csv"), key=lambda r: r["article_id"])
    ids = [r["article_id"] for r in rows]
    t0 = time.time()
    (out / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
    np.save(out / "handcrafted.npy", feature_matrix(rows))
    (out / "handcrafted_names.json").write_text(json.dumps(FEATURE_NAMES), encoding="utf-8")
    meta = {
        "data_csv_sha256": file_sha256(root / "data.csv"),
        "n": len(ids),
        "handcrafted_features": len(FEATURE_NAMES),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    print("handcrafted done", round(time.time() - t0, 1), "s", flush=True)
    if args.embeddings:
        for name, texts, max_len in (
            ("title", [r["title"] for r in rows], 64),
            ("lead", [r.get("lead", "") or "" for r in rows], 128),
            ("content", [(r["content"] or "")[:2000] for r in rows], 256),
        ):
            print("embed", name, flush=True)
            np.save(out / f"{name}_emb.npy", embed(texts, max_len, 64, args.threads))
        meta["embedder"] = {
            "model_id": EMBEDDER[0],
            "revision": EMBEDDER[1],
            "pooling": "mean",
            "prefix": "query: ",
            "max_length": {"title": 64, "lead": 128, "content": 256},
        }
    meta["seconds"] = round(time.time() - t0, 1)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
