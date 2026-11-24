import json
from pathlib import Path
import pytest
from src.common.dataset import file_sha256, read_csv
from src.common.runtime import load_config

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/manifests/dataset_v1.0"


def test_committed_manifest_bytes_and_group_disjointness():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for rel in [
        "label_schema.json",
        "image_manifest.csv",
        "split_distribution.csv",
        "validation_report.json",
        "split_assignments.csv",
        "splits/train.csv",
        "splits/validation.csv",
        "splits/test.csv",
    ]:
        assert file_sha256(DATA / rel) == manifest["protected_sha256"][rel], rel
    records = read_csv(DATA / "split_assignments.csv")
    byid = {r["article_id"]: r for r in records}
    assert len(byid) == len(records) == 28042
    groups = {}
    counts = {}
    for row in records:
        groups.setdefault(row["leakage_group_id"], set()).add(row["split"])
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    assert all(len(v) == 1 for v in groups.values())
    assert counts == {"train": 19629, "validation": 4206, "test": 4207}
    for split in counts:
        membership = read_csv(DATA / f"splits/{split}.csv")
        assert len(membership) == counts[split]
        assert all(byid[r["article_id"]] == r for r in membership)
    images = read_csv(DATA / "image_manifest.csv")
    identity = {}
    assert len(images) == 28042
    for row in images:
        identity.setdefault(row["file_sha256"], set()).add(byid[row["article_id"]]["split"])
    assert all(len(v) == 1 for v in identity.values())


def test_schema_and_config_taxonomy():
    schema = json.loads((DATA / "label_schema.json").read_text(encoding="utf-8"))
    assert len(schema["types"]) == 5
    assert schema["consistency_classes"] == {
        "Consistent": 0,
        "Inconsistent": 1,
        "Insufficient Evidence": 2,
    }
    for name in ["smoke", "detection", "type_classification", "consistency"]:
        c = load_config(ROOT / f"configs/{name}.yaml")
        assert c["seed"] == 20261124


def test_no_training_text_or_images_in_manifest_folder():
    assert not (DATA / "data.csv").exists()
    assert not (DATA / "images").exists()
    assert all(p.suffix not in {".jpg", ".png", ".pt", ".zip"} for p in DATA.rglob("*"))
