"""Background-capable job execution with checkpoints, retries and RPM limits."""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Mapping, Sequence

from src.labeling.engine import LabelExecution, LabelingEngine
from src.labeling.prompts import PROMPT_VERSION, SCHEMA_VERSION, load_taxonomy
from src.labeling.validation import LabelValidationError
from src.providers.base import LLMProvider, ProviderError
from src.jobs.storage import JobStore


class _RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.limit = max(1, int(requests_per_minute))
        self.times: deque[float] = deque()
        self.lock = threading.Lock()
        self.next_request = 0.0

    def wait_for_slot(self, should_continue) -> bool:
        while should_continue():
            with self.lock:
                now = time.monotonic()
                if now >= self.next_request:
                    self.next_request = now + 60.0 / self.limit
                    return True
                delay = self.next_request - now
            deadline = time.monotonic() + min(delay, 0.25)
            while time.monotonic() < deadline:
                if not should_continue():
                    return False
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        return False


class JobRunner:
    def __init__(self, store: JobStore, provider: LLMProvider, runtime_settings: Mapping[str, Any] | None = None) -> None:
        self.store = store
        self.provider = provider
        self.runtime_settings = dict(runtime_settings or {})
        self._rate_limiter = _RateLimiter(int(self.runtime_settings.get("requests_per_minute", 10)))
        self._retry_state = threading.local()

    def run(self, job_id: str) -> None:
        job = self.store.get_job(job_id)
        if job is None:
            return
        status = job["status"]
        if status not in {"queued", "running"}:
            return
        self.store.set_job_status(job_id, "running")

        settings = dict(job.get("settings", {}))
        settings.update(self.runtime_settings)
        self._rate_limiter.limit = max(1, int(settings.get("requests_per_minute", 10)))
        taxonomy = load_taxonomy()
        engine = LabelingEngine(
            self.provider,
            taxonomy,
            prompt_version=job["prompt_version"],
            max_content_chars=int(settings.get("max_content_chars", 20000)),
            min_consistency_content_chars=int(settings.get("min_consistency_content_chars", 80)),
        )
        workers = max(1, min(int(settings.get("concurrency", 1)), 8))
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"label-{job_id[:6]}")
        active: dict[Future[None], list[dict[str, Any]]] = {}

        try:
            while True:
                job_state = self.store.get_job(job_id)
                if job_state is None:
                    return
                may_claim = job_state["status"] == "running"

                while may_claim and len(active) < workers:
                    if self._budget_reached(job_id, settings):
                        self.store.set_job_status(job_id, "budget_paused", "Đã đạt ngưỡng ngân sách đã cấu hình.")
                        may_claim = False
                        break
                    work = self.store.claim_next_work(job_id, job["mode"])
                    if not work:
                        break
                    future = executor.submit(self._process_work, job, engine, work, settings)
                    active[future] = work

                if not active:
                    job_state = self.store.get_job(job_id)
                    progress = self.store.get_progress(job_id)
                    if job_state and job_state["status"] == "running" and progress.get("pending", 0) == 0:
                        final_status = "completed_with_errors" if progress.get("error", 0) else "completed"
                        self.store.set_job_status(job_id, final_status)
                    return

                done, _ = wait(active, timeout=0.25, return_when=FIRST_COMPLETED)
                for future in done:
                    finished_work = active.pop(future, [])
                    try:
                        future.result()
                    except Exception as exc:
                        # Item-level failures are handled in _process_work; this is a last-resort guard.
                        self.store.fail_items(job_id, finished_work, "runner_error", f"Lỗi job runner: {type(exc).__name__}")

                job_state = self.store.get_job(job_id)
                if job_state and job_state["status"] in {"paused", "stopped", "budget_paused", "auth_required"}:
                    if active:
                        continue
                    return
        finally:
            executor.shutdown(wait=True, cancel_futures=False)

    def _process_work(
        self,
        job: Mapping[str, Any],
        engine: LabelingEngine,
        work: Sequence[Mapping[str, Any]],
        settings: Mapping[str, Any],
    ) -> None:
        job_id = str(job["job_id"])
        pending: list[Mapping[str, Any]] = []
        for item in work:
            cached = self.store.get_cached_result(item["cache_key"])
            if cached is None:
                pending.append(item)
            else:
                self.store.complete_from_cache(
                    job_id, item["record_id"], item["task"], item["cache_key"], cached
                )
        if not pending:
            return

        max_retries = max(0, int(settings.get("max_retries", 3)))
        attempt = 0
        while True:
            if not self._job_is_running(job_id):
                self.store.release_items(job_id, pending)
                return
            try:
                record = dict(pending[0]["input"])
                tasks = [str(item["task"]) for item in pending]
                requires_request = engine.requires_request(record, tasks)
                if requires_request and job["mode"] == "combined":
                    # Keep the original request context when only some items
                    # were cached or retried; never cache a smaller prompt
                    # under the full combined request's cache key.
                    tasks = list(job["tasks"])
                if requires_request and self.provider.provider_name != "mock" and not self._rate_limiter.wait_for_slot(lambda: self._job_is_running(job_id)):
                    self.store.release_items(job_id, pending)
                    return
                if not self._job_is_running(job_id):
                    self.store.release_items(job_id, pending)
                    return
                if requires_request:
                    self.store.increment_attempts(job_id, pending)
                execution = engine.label_record(record, tasks, mode=str(job["mode"]))
                results = self._make_results(job, pending, execution, settings)
                self.store.complete_items(job_id, results)
                budget = settings.get("budget_usd")
                if budget and self.store.spent_cost(job_id) >= float(budget):
                    self.store.set_job_status(job_id, "budget_paused", "Đã đạt ngưỡng ngân sách sau khi lưu checkpoint.")
                return
            except ProviderError as exc:
                retryable = (exc.retryable or exc.kind == "schema_validation") and not exc.quota_exhausted
                if retryable and attempt < max_retries and self._job_is_running(job_id):
                    self._retry_state.minimum = exc.retry_after_seconds or (60 if exc.status_code == 429 else 0)
                    if not self._retry_wait(job_id, attempt):
                        self.store.release_items(job_id, pending)
                        return
                    attempt += 1
                    continue
                self.store.fail_items(job_id, pending, exc.kind, str(exc))
                if exc.kind == "authentication":
                    self.store.set_job_status(job_id, "auth_required", "Cần kiểm tra lại API key hoặc quyền truy cập.")
                elif exc.status_code == 429:
                    self.store.set_job_status(job_id, "quota_paused", "API giới hạn quota/tốc độ. Đã tạm dừng để tránh gọi dồn; kiểm tra quota rồi tiếp tục.")
                elif exc.status_code in {400, 404}:
                    self.store.set_job_status(job_id, "model_required", "Model từ chối yêu cầu. Hãy kiểm tra lại model ở bước 1 hoặc tạo job mới với model khác.")
                elif (exc.status_code is not None and exc.status_code >= 500) or exc.kind == "timeout":
                    self.store.set_job_status(job_id, "service_paused", "Dịch vụ API hoặc mạng chưa phản hồi sau các lần thử. Đã tạm dừng và giữ phần chưa xử lý; kiểm tra model rồi thử lại.")
                return
            except LabelValidationError as exc:
                self._retry_state.minimum = 0
                if attempt < max_retries and self._job_is_running(job_id):
                    if not self._retry_wait(job_id, attempt):
                        self.store.release_items(job_id, pending)
                        return
                    attempt += 1
                    continue
                self.store.fail_items(job_id, pending, "label_validation", str(exc))
                return
            except Exception as exc:
                self.store.fail_items(job_id, pending, "processing_error", f"Lỗi xử lý: {type(exc).__name__}")
                return

    def _make_results(
        self,
        job: Mapping[str, Any],
        pending: Sequence[Mapping[str, Any]],
        execution: LabelExecution,
        settings: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        usage_owner = next(iter(execution.usage_by_task), None)
        combined_usage = next(iter(execution.usage_by_task.values()), {})
        if job["mode"] == "combined" and pending:
            usage_owner = pending[0]["task"]
        for item in pending:
            task = str(item["task"])
            output = execution.outputs[task]
            usage = execution.usage_by_task.get(task, {})
            if job["mode"] == "combined":
                usage = combined_usage if task == usage_owner else {}
            cost = self._cost(usage, settings) if task == usage_owner else None
            results.append(
                {
                    "record_id": item["record_id"],
                    "task": task,
                    "output": output.model_dump(mode="json"),
                    "input_truncated": execution.input_truncated.get(task, False),
                    "usage": usage,
                    "estimated_cost_usd": cost,
                    "cache_key": item["cache_key"],
                }
            )
        return results

    @staticmethod
    def _cost(usage: Mapping[str, Any], settings: Mapping[str, Any]) -> float | None:
        input_rate = settings.get("input_price_per_million")
        output_rate = settings.get("output_price_per_million")
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if input_rate is None or output_rate is None or input_tokens is None or output_tokens is None:
            return None
        return (int(input_tokens) * float(input_rate) + int(output_tokens) * float(output_rate)) / 1_000_000

    def _budget_reached(self, job_id: str, settings: Mapping[str, Any]) -> bool:
        budget = settings.get("budget_usd")
        has_prices = settings.get("input_price_per_million") is not None and settings.get("output_price_per_million") is not None
        return bool(budget and has_prices and self.store.spent_cost(job_id) >= float(budget))

    def _job_is_running(self, job_id: str) -> bool:
        job = self.store.get_job(job_id)
        return bool(job and job["status"] == "running")

    def _retry_wait(self, job_id: str, attempt: int) -> bool:
        delay = max(float(getattr(self._retry_state, "minimum", 0)), min(30.0, (2**attempt) + random.uniform(0, 0.75)))
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            if not self._job_is_running(job_id):
                return False
            time.sleep(min(0.2, max(0, deadline - time.monotonic())))
        return self._job_is_running(job_id)

