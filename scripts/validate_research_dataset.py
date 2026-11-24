"""Validate local snapshot and the versioned team split; never create a new split."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.dataset import file_sha256, read_csv, verify_release


def validate(root):
    root = Path(root)
    manifest = verify_release(root)
    published = json.loads(
        (ROOT / "data/manifests/dataset_v1.0/manifest.json").read_text(encoding="utf-8")
    )
    for rel in ["data.csv", "split_assignments.csv", "label_schema.json"]:
        if file_sha256(root / rel) != published["protected_sha256"][rel]:
            raise ValueError(f"Not the shared dataset: {rel}")
    records = read_csv(root / "split_assignments.csv")
    ids = set()
    groups = {}
    counts = {}
    for row in records:
        if row["article_id"] in ids:
            raise ValueError("Duplicate article ID")
        ids.add(row["article_id"])
        groups.setdefault(row["leakage_group_id"], set()).add(row["split"])
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    if counts != {"train": 19629, "validation": 4206, "test": 4207}:
        raise ValueError("Wrong split counts")
    if any(len(v) > 1 for v in groups.values()):
        raise ValueError("Group leakage")
    return {
        "rows": len(ids),
        "groups": len(groups),
        "counts": counts,
        "dataset_version": manifest["dataset_version"],
        "gold_status": manifest["gold_test_status"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.dataset_root)))
