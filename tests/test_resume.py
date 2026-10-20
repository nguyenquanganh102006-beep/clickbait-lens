from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.jobs.checkpoint import build_cache_key
from src.jobs.runner import JobRunner
from src.jobs.storage import JobStore
from src.labeling.schemas import DetectionOutput
from src.providers.base import ProviderError, ProviderResponse
from src.providers.mock import MockProvider


def _create_job(store: JobStore, records: list[dict[str, Any]], *, settings: dict[str, Any] | None = None) -> str:
    return store.create_job(
        records,
        ["detection"],
        provider="mock",
        model="mock-v1",
        mode="independent",
        prompt_version="v1",
        taxonomy_version="v1",
        schema_version="v1",
        settings=settings or {"requests_per_minute": 600, "max_retries": 0},
    )


class CountingMockProvider(MockProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def generate_structured(self, prompt, output_schema):
        self.calls += 1
        return super().generate_structured(prompt, output_schema)


def _records(count: int = 2) -> list[dict[str, Any]]:
    return [
        {
            "record_id": f"r{index}",
            "title": f"Tiêu đề {index}",
            "content": None,
            "_original": {"headline": f"Tiêu đề {index}", "gold": index},
        }
        for index in range(count)
    ]


def test_resume_keeps_checkpoint_and_processes_only_pending_items(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    job_id = _create_job(store, _records())

    first_claim = store.claim_next_work(job_id, "independent")
    first = first_claim[0]
    output = DetectionOutput(status="success", label=1, confidence=0.9, reason="Dấu hiệu rõ.")
    store.complete_items(
        job_id,
        [
            {
                "record_id": first["record_id"],
                "task": first["task"],
                "output": output.model_dump(mode="json"),
                "cache_key": first["cache_key"],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            }
        ],
    )
    interrupted_claim = store.claim_next_work(job_id, "independent")
    store.resume(job_id)

    provider = CountingMockProvider()
    JobRunner(store, provider, {"requests_per_minute": 600, "max_retries": 0}).run(job_id)

    items = store.get_items(job_id)
    assert store.get_job(job_id)["status"] == "completed"
    assert [item["status"] for item in items] == ["success", "success"]
    assert provider.calls == 1
    assert items[0]["attempts"] == 0  # reused from the prediction cache
    assert items[1]["attempts"] == 1  # claim recovery did not count as an API attempt


def test_runner_retries_temporary_provider_errors_and_counts_attempts(tmp_path: Path, monkeypatch):
    class FailOnceProvider(CountingMockProvider):
        def generate_structured(self, prompt, output_schema):
            self.calls += 1
            if self.calls == 1:
                raise ProviderError("Thử lại", kind="rate_limit", retryable=True, status_code=429)
            return MockProvider.generate_structured(self, prompt, output_schema)

    monkeypatch.setattr(JobRunner, "_retry_wait", lambda self, job_id, attempt: True)
    store = JobStore(tmp_path / "jobs.sqlite3")
    job_id = _create_job(store, _records(1), settings={"requests_per_minute": 600, "max_retries": 1})
    provider = FailOnceProvider()

    JobRunner(store, provider, {"requests_per_minute": 600, "max_retries": 1}).run(job_id)

    item = store.get_items(job_id)[0]
    assert store.get_job(job_id)["status"] == "completed"
    assert item["status"] == "success"
    assert item["attempts"] == 2
    assert provider.calls == 2


def test_error_output_is_not_cached_and_retry_failed_calls_provider_again(tmp_path: Path):
    class ErrorThenSuccessProvider(CountingMockProvider):
        def generate_structured(self, prompt, output_schema):
            self.calls += 1
            if self.calls == 1:
                output = DetectionOutput(status="error", label=None, confidence=None, reason="Cần thử lại.")
                return ProviderResponse(output, output.model_dump_json(), self.model)
            return MockProvider.generate_structured(self, prompt, output_schema)

    store = JobStore(tmp_path / "jobs.sqlite3")
    job_id = _create_job(store, _records(1), settings={"requests_per_minute": 600, "max_retries": 0})
    item_before = store.get_items(job_id)[0]
    provider = ErrorThenSuccessProvider()
    runner = JobRunner(store, provider, {"requests_per_minute": 600, "max_retries": 0})

    runner.run(job_id)
    assert store.get_job(job_id)["status"] == "completed_with_errors"
    assert store.get_cached_result(item_before["cache_key"]) is None
    assert store.retry_failed(job_id) == 1
    runner.run(job_id)

    item_after = store.get_items(job_id)[0]
    assert store.get_job(job_id)["status"] == "completed"
    assert item_after["status"] == "success"
    assert provider.calls == 2


def test_independent_cache_key_does_not_depend_on_unselected_tasks():
    args = {
        "provider": "mock",
        "model": "mock-v1",
        "prompt_version": "v1",
        "taxonomy_version": "v1",
        "schema_version": "v1",
        "settings": {"max_content_chars": 20000},
    }
    record = {"record_id": "r1", "title": "Tiêu đề", "content": None}
    one_task = build_cache_key(record, "detection", mode="independent", selected_tasks=["detection"], **args)
    more_tasks = build_cache_key(record, "detection", mode="independent", selected_tasks=["detection", "type"], **args)
    combined_one = build_cache_key(record, "detection", mode="combined", selected_tasks=["detection"], **args)
    combined_more = build_cache_key(record, "detection", mode="combined", selected_tasks=["detection", "type"], **args)

    assert one_task == more_tasks
    assert combined_one != combined_more
    changed_content = {**record, "content": "Toàn văn thay đổi"}
    assert combined_more != build_cache_key(changed_content, "detection", mode="combined", selected_tasks=["detection", "type"], **args)
    assert one_task == build_cache_key(changed_content, "detection", mode="independent", selected_tasks=["detection"], **args)


def test_job_store_never_persists_api_key(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    job_id = _create_job(store, _records(1), settings={"api_key": "secret-value", "max_retries": 1})

    assert "api_key" not in store.get_job(job_id)["settings"]
    assert "secret-value" not in (tmp_path / "jobs.sqlite3").read_bytes().decode("latin-1")


def test_budget_guard_checkpoints_then_resumes_after_user_raises_limit(tmp_path: Path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    settings = {
        "requests_per_minute": 600,
        "concurrency": 1,
        "max_retries": 0,
        "budget_usd": 0.00000001,
        "input_price_per_million": 100.0,
        "output_price_per_million": 100.0,
    }
    job_id = _create_job(store, _records(3), settings=settings)
    provider = CountingMockProvider()
    runner = JobRunner(store, provider, settings)

    runner.run(job_id)

    assert store.get_job(job_id)["status"] == "budget_paused"
    assert store.get_progress(job_id)["success"] == 1
    assert store.get_progress(job_id)["pending"] == 2
    assert store.spent_cost(job_id) > settings["budget_usd"]

    updated = {**settings, "budget_usd": 1.0}
    store.update_job_settings(job_id, updated)
    store.resume(job_id)
    JobRunner(store, provider, updated).run(job_id)

    assert store.get_job(job_id)["status"] == "completed"
    assert store.get_progress(job_id)["success"] == 3


@pytest.mark.parametrize("error,expected_status", [
    (ProviderError("quota exhausted", kind="rate_limit", status_code=429, quota_exhausted=True), "quota_paused"),
    (ProviderError("missing model", kind="model_not_found", status_code=404), "model_required"),
    (ProviderError("unsupported schema", kind="request_error", status_code=400), "model_required"),
    (ProviderError("bad key", kind="authentication", status_code=403), "auth_required"),
    (ProviderError("server unavailable", kind="server_error", status_code=503), "service_paused"),
    (ProviderError("network timed out", kind="timeout"), "service_paused"),
])
def test_runner_stops_global_failures_without_failing_remaining_dataset(tmp_path, error, expected_status):
    class FailingProvider(CountingMockProvider):
        def generate_structured(self, prompt, schema):
            self.calls += 1
            raise error
    store = JobStore(tmp_path / "jobs.sqlite3")
    job_id = _create_job(store, _records(10))
    provider = FailingProvider()
    JobRunner(store, provider, {"requests_per_minute": 600, "max_retries": 0}).run(job_id)
    assert store.get_job(job_id)["status"] == expected_status
    assert provider.calls == 1
    assert store.get_progress(job_id)["error"] == 1
    assert store.get_progress(job_id)["pending"] == 9
    store.retry_failed(job_id)
    JobRunner(store, MockProvider(), {"requests_per_minute": 600, "max_retries": 0}).run(job_id)
    assert store.get_progress(job_id)["success"] == 10


@pytest.mark.parametrize("stop_status", ["paused", "stopped"])
def test_pause_or_stop_finishes_inflight_only_and_resume_keeps_checkpoint(tmp_path, stop_status):
    import threading
    store = JobStore(tmp_path / "jobs.sqlite3")
    job_id = _create_job(store, _records(3))
    entered, release = threading.Event(), threading.Event()

    class BlockingProvider(CountingMockProvider):
        def generate_structured(self, prompt, schema):
            entered.set()
            assert release.wait(3)
            return super().generate_structured(prompt, schema)

    provider = BlockingProvider()
    worker = threading.Thread(target=JobRunner(store, provider, {"requests_per_minute": 600}).run, args=(job_id,))
    worker.start()
    assert entered.wait(3)
    store.set_job_status(job_id, stop_status)
    release.set()
    worker.join(3)
    assert not worker.is_alive()
    assert store.get_progress(job_id)["success"] == 1
    assert store.get_progress(job_id)["pending"] == 2
    store.resume(job_id)
    JobRunner(store, MockProvider(), {"requests_per_minute": 600}).run(job_id)
    assert store.get_progress(job_id)["success"] == 3
