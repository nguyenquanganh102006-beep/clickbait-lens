import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_contributor_batch as batch
from src.providers.base import ProviderError


class FakeProvider:
    provider_name = "gemini"
    model = "fake-model"
    api_key = "never-print-this"
    requests_per_minute = 10


def record(aid):
    return {"article_id": aid, "title": "title", "content": "body", "lead": "", "image_path": "", "contributor": "Lê Văn Duẩn"}


def result(aid):
    masks = {k: True for k in ["detection", "curiosity_gap", "exaggeration", "emotional_bait", "question_based", "misleading_presentation", "consistency_tc", "consistency_ti", "consistency_ic"]}
    row = {**record(aid), "annotation_quality_mode": "multimodal-v4.1-reference-consensus", "annotation_input_hash": "", "annotation_requested_model": "fake-model", "annotation_status": "pseudo_labeled", "label_mask_json": json.dumps(masks), "annotation_total_tokens": 12, "annotation_model_calls": 2}
    row["annotation_result_hash"] = batch.result_hash(row)
    return row


def setup_files(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    with source.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(record("x"))); w.writeheader(); w.writerows([record("duan_1"), {**record("other"), "contributor": "Khác"}, record("duan_2")])
    args = SimpleNamespace(csv=source, contributor="Lê Văn Duẩn", output=tmp_path / "out", image_root=tmp_path, seed_csv=None, model="fake-model", timeout=90, rpm=10, max_records=None, rate_retries=0)
    monkeypatch.setattr(batch, "quality_input_hash", lambda rec, root, provider: "expected")
    return args


def test_filter_and_seed_skip_api(tmp_path, monkeypatch):
    args = setup_files(tmp_path, monkeypatch); provider = FakeProvider()
    seeded = result("duan_1"); seeded["annotation_input_hash"] = "expected"; seeded["annotation_result_hash"] = batch.result_hash(seeded)
    seed = tmp_path / "seed.csv"
    # Bootstrap CSV is ordinary DictReader input and must match the same resume integrity contract.
    with seed.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=seeded); w.writeheader(); w.writerow(seeded)
    args.seed_csv = seed
    seen = []
    def fake_review(rec, *_):
        seen.append(rec["article_id"]); r = result(rec["article_id"]); r["annotation_input_hash"] = "expected"; r["annotation_result_hash"] = batch.result_hash(r); return r
    monkeypatch.setattr(batch, "reviewed_record", fake_review)
    assert batch.run(args, provider) == 0
    assert seen == ["duan_2"]
    assert (args.output / "checkpoint.jsonl").exists()
    assert len(list(csv.DictReader((args.output / "completed.csv").open(encoding="utf-8-sig")))) == 2


def test_quota_stops_and_checkpoints(tmp_path, monkeypatch):
    args = setup_files(tmp_path, monkeypatch)
    def quota(*_): raise ProviderError("exhausted", kind="rate_limit", status_code=429, quota_exhausted=True)
    monkeypatch.setattr(batch, "reviewed_record", quota)
    assert batch.run(args, FakeProvider()) == 2
    progress = json.loads((args.output / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "quota_exhausted" and progress["last_id"] == "duan_1"
    assert progress["remaining"] == 2


def test_resume_validates_hashes_and_no_recall(tmp_path, monkeypatch):
    args = setup_files(tmp_path, monkeypatch); args.output.mkdir()
    r = result("duan_1"); r["annotation_input_hash"] = "expected"; r["annotation_result_hash"] = batch.result_hash(r)
    batch.append_checkpoint(args.output / "checkpoint.jsonl", r)
    seen=[]
    monkeypatch.setattr(batch, "reviewed_record", lambda rec, *_: seen.append(rec["article_id"]) or (_ for _ in ()).throw(AssertionError()))
    # only ensure bad tampered result is rejected before any model invocation.
    tampered = r.copy(); tampered["title"] = "tampered"
    (args.output / "checkpoint.jsonl").write_text(json.dumps({"result": tampered}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="result hash"):
        batch.run(args, FakeProvider())
    assert not seen


def test_partial_rows_route_to_review_queue():
    assert not batch.is_full({"annotation_status": "needs_review", "label_mask_json": "{}"})
