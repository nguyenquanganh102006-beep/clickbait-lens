import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from scripts import run_contributor_batch as batch
from scripts import run_parallel_contributor as parallel
from src.providers.base import ProviderError


class FakeProvider:
    provider_name = "gemini"
    model = "fake-model"
    api_key = "never-print-this"

    def __init__(self, fn=None):
        self.fn = fn
        self.calls = 0
        self.lock = threading.Lock()

    def generate_multimodal(self, prompt, schema, image, mime_type="image/jpeg"):
        with self.lock:
            self.calls += 1
        if self.fn:
            return self.fn(prompt, schema, image, mime_type)
        return (prompt, schema, image, mime_type)


def record(aid, contributor="Nguyen Minh Tuan"):
    return {"article_id": aid, "title": "title " + aid, "content": "body " + aid,
            "lead": "", "image_path": "", "contributor": contributor}


def full_result(rec, input_hash="expected"):
    masks = {k: True for k in ["detection", "curiosity_gap", "exaggeration", "emotional_bait",
                                "question_based", "misleading_presentation", "consistency_tc",
                                "consistency_ti", "consistency_ic"]}
    row = {**rec, "annotation_quality_mode": parallel.VERSION,
           "annotation_input_hash": input_hash, "annotation_requested_model": "fake-model",
           "annotation_status": "pseudo_labeled", "label_mask_json": json.dumps(masks),
           "annotation_total_tokens": 7, "annotation_model_calls": 2}
    row["annotation_result_hash"] = batch.result_hash(row)
    return row


def test_global_rate_limit_spaces_calls_across_threads():
    starts = []
    start_lock = threading.Lock()

    def call(*_):
        with start_lock:
            starts.append(time.monotonic())
        return "ok"

    provider = FakeProvider(call)
    wrapped = parallel.SharedRateLimitedProvider(lambda: provider, model="fake-model", rpm=1200)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: wrapped.generate_multimodal("p", object, None), range(4)))
    starts.sort()
    assert len(starts) == 4
    assert all(b - a >= 0.04 for a, b in zip(starts, starts[1:]))
    assert wrapped.physical_calls == provider.calls == 4


def test_quota_sets_global_stop_and_blocks_later_requests():
    def quota(*_):
        raise ProviderError("quota", kind="rate_limit", status_code=429, quota_exhausted=True)

    provider = FakeProvider(quota)
    wrapped = parallel.SharedRateLimitedProvider(lambda: provider, model="fake-model", rpm=600)
    try:
        wrapped.generate_multimodal("p", object, None)
    except ProviderError as exc:
        assert exc.quota_exhausted
    else:
        raise AssertionError("quota error expected")
    assert wrapped.stop_event.is_set()
    try:
        wrapped.generate_multimodal("p", object, None)
    except ProviderError as exc:
        assert exc.kind == "stopped" and not exc.quota_exhausted
    else:
        raise AssertionError("stopped request expected")
    assert provider.calls == 1


def test_slow_client_creation_does_not_reserve_an_early_dispatch_slot():
    entered = threading.Event()
    release = threading.Event()
    factory_lock = threading.Lock()
    factory_calls = 0
    starts = []

    def factory():
        nonlocal factory_calls
        with factory_lock:
            factory_calls += 1
            first = factory_calls == 1
        if first:
            entered.set()
            assert release.wait(5)
        return FakeProvider(lambda *_: starts.append(time.monotonic()))

    wrapped = parallel.SharedRateLimitedProvider(factory, model='fake-model', rpm=600)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(wrapped.generate_multimodal, 'p', object, None)
        assert entered.wait(5)
        second = pool.submit(wrapped.generate_multimodal, 'p', object, None)
        try:
            second.result(timeout=5)
        finally:
            release.set()
        first.result(timeout=5)
    starts.sort()
    assert len(starts) == 2
    assert starts[1] - starts[0] >= .08
    assert wrapped.physical_calls == 2


def test_resume_and_published_skip_avoid_provider_work(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    rows = [record("tuan_1"), record("tuan_2"), record("tuan_3"), record("other", "Other")]
    with source.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    output = tmp_path / "out"; output.mkdir()
    provider = FakeProvider()
    monkeypatch.setattr(parallel, "quality_input_hash", lambda rec, root, p: "expected-" + rec["article_id"])

    cached = full_result(rows[0], "expected-tuan_1")
    batch.append_checkpoint(output / "checkpoint.jsonl", cached)
    published_row = full_result(rows[1], "old-model-hash")
    published = tmp_path / "published.csv"
    with published.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(published_row)); writer.writeheader(); writer.writerow(published_row)

    args = SimpleNamespace(csv=source, contributor="Nguyen Minh Tuan", output=output, image_root=tmp_path,
                           model="fake-model", timeout=90, rpm=600, workers=4, record_attempts=3,
                           skip_published=published, seed_csv=None, key_stdin=False)
    seen = []

    def review(rec, *_):
        seen.append(rec["article_id"])
        return full_result(rec, "expected-" + rec["article_id"])

    assert parallel.run(args, provider=provider, review_fn=review) == 0
    assert seen == ["tuan_3"]
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    assert progress["processed_articles"] == 2
    assert progress["external_completed_skipped"] == 1
    assert progress["remaining"] == 0
    checkpoint = batch.load_jsonl(output / "checkpoint.jsonl")
    assert set(checkpoint) == {"tuan_1", "tuan_3"}
    assert (output / "completed.csv").exists()
