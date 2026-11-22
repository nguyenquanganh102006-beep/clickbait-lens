"""Concurrent, resumable Gemini runner with one process-wide RPM limiter."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import getpass
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.labeling.quality import VERSION, quality_input_hash, result_hash, reviewed_record
from src.providers.base import ProviderError, sanitize_error_message
from scripts.run_contributor_batch import (
    DEFAULT_CSV, DEFAULT_OUT, append_checkpoint, atomic_csv, atomic_json,
    is_full, load_jsonl, load_leader_audit, read_contributor,
)

DEFAULT_SKIP_PUBLISHED = ROOT.parent / "data_pr" / "clean_final" / "data_tuan_labeled.csv"
OBSOLETE = {"type_misleading_framing", "type_image_text_mismatch", "type_context_manipulation"}


class SharedRateLimitedProvider:
    """Thread-local provider clients, globally spaced physical generation calls."""

    def __init__(self, provider_factory, *, model: str, rpm: int, stop_event=None, provider_name="gemini", api_key=""):
        if rpm < 1:
            raise ValueError("rpm must be positive")
        self._factory = provider_factory
        self._local = threading.local()
        self._lock = threading.Lock()
        self._next_at = 0.0
        self._interval = 60.0 / rpm
        self._stop = stop_event or threading.Event()
        self._stop_reason = None
        self._calls = 0
        self._meta_lock = threading.Lock()
        self.model = model
        self.provider_name = provider_name
        self.api_key = api_key
        self.requests_per_minute = None  # quality._call must not add a second limiter.

    @property
    def stop_event(self):
        return self._stop

    @property
    def stop_reason(self):
        with self._meta_lock:
            return self._stop_reason

    @property
    def physical_calls(self):
        with self._meta_lock:
            return self._calls

    def request_stop(self, reason: str):
        with self._meta_lock:
            if self._stop_reason is None or (self._stop_reason == "infrastructure_error" and reason in {"quota_exhausted", "authentication_error"}):
                self._stop_reason = reason
        self._stop.set()

    def _client(self):
        client = getattr(self._local, "client", None)
        if client is None:
            client = self._factory()
            self._local.client = client
        return client

    def generate_multimodal(self, prompt, output_schema, image, mime_type="image/jpeg"):
        if self._stop.is_set():
            raise ProviderError("Runner stopped before a new provider request.", kind="stopped")
        # Client construction may block (SDK imports/connections). Do it before
        # reserving a dispatch slot so late initialization cannot bunch calls.
        client = self._client()
        with self._lock:
            while True:
                if self._stop.is_set():
                    raise ProviderError("Runner stopped before a new provider request.", kind="stopped")
                delay = self._next_at-time.monotonic()
                if delay <= 0:break
                if self._stop.wait(delay):
                    raise ProviderError("Runner stopped before a new provider request.", kind="stopped")
            if self._stop.is_set():
                raise ProviderError("Runner stopped before a new provider request.", kind="stopped")
            with self._meta_lock:
                self._calls += 1
            self._next_at=time.monotonic()+self._interval
        try:
            return client.generate_multimodal(prompt, output_schema, image, mime_type)
        except ProviderError as exc:
            if exc.quota_exhausted:
                self.request_stop("quota_exhausted")
            elif exc.kind == "authentication":
                self.request_stop("authentication_error")
            raise


def _acquire_output_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    lock_file = (output / ".runner.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            lock_file.seek(0)
            if lock_file.tell() == 0:
                lock_file.write(b"0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock_file.close()
        raise RuntimeError("Another batch runner holds this output directory") from exc
    return lock_file


def _read_csv_retry(path: Path, tries=5):
    last = None
    for attempt in range(tries):
        try:
            with path.open(encoding="utf-8-sig", newline="") as f:
                return list(csv.DictReader(f))
        except (PermissionError, OSError) as exc:
            last = exc
            if attempt + 1 < tries:
                time.sleep(.05 * (attempt + 1))
    raise last


def _published_skip_ids(path: Path | None, records_by_id: dict[str, dict]) -> set[str]:
    if not path or not path.exists():
        return set()
    skipped = set()
    for row in _read_csv_retry(path):
        aid = row.get("article_id")
        source = records_by_id.get(aid)
        if not source or not is_full(row) or row.get("annotation_result_hash") != result_hash(row):
            continue
        if any(str(row.get(k, "") or "") != str(source.get(k, "") or "") for k in ("title", "content", "image_path", "contributor")):
            continue
        skipped.add(aid)
    return skipped


def _validate_result(row, record, image_root, provider, version=VERSION, hash_fn=quality_input_hash):
    if row.get("annotation_quality_mode") != version:
        raise ValueError("checkpoint schema/quality version mismatch")
    if row.get("annotation_input_hash") != hash_fn(record, image_root, provider):
        raise ValueError("checkpoint input/model/prompt/schema/image hash mismatch")
    if row.get("annotation_requested_model") != getattr(provider, "model", ""):
        raise ValueError("checkpoint model mismatch")
    if row.get("annotation_result_hash") != result_hash(row):
        raise ValueError("checkpoint result hash mismatch")


def _export_preserving_review_sample(output, ordered, progress, source_fields):
    """Reuse atomic CSV/JSON helpers while leaving the root's audited 5% sample alone."""
    slim = [{k: v for k, v in row.items() if k not in OBSOLETE} for row in ordered]
    fields = list(dict.fromkeys([k for k in source_fields if k not in OBSOLETE] + [k for row in slim for k in row]))
    leader = load_leader_audit(output)
    held = {aid for aid, status in leader.items() if status in {"disputed", "quarantined"}}
    atomic_csv(output / "completed.csv", [r for r in slim if is_full(r) and r["article_id"] not in held], fields)
    atomic_csv(output / "progress.csv", slim, fields)
    atomic_csv(output / "review_queue.csv", [r for r in slim if not is_full(r) or r["article_id"] in held], fields)
    atomic_json(output / "progress.json", progress)


def run(args, provider=None, provider_factory=None, review_fn=None):
    records = read_contributor(args.csv, args.contributor)
    mode=getattr(args,'mode','robust');version=VERSION;hash_fn=quality_input_hash
    if mode=='fast':
        from src.labeling.fast import VERSION as fast_version,input_hash as fast_hash,annotate_record as fast_review
        version=fast_version;hash_fn=fast_hash
        if review_fn is None:review_fn=fast_review
    canonical_path=getattr(args,'canonical_titles',None)
    input_holds=set()
    if canonical_path and canonical_path.exists():
        candidates={r['article_id']:r for r in json.loads(canonical_path.read_text(encoding='utf-8')).get('rows',[])}
        for record in records:
            c=candidates.get(record['article_id'])
            if not c:continue
            if c['og_title'].rstrip().endswith((',',':',';','-','–','—','…','...')):
                input_holds.add(record['article_id']);continue
            record.update(source_original_title=record['title'],title=c['og_title'],
                          title_repair_method='og_title_when_title_equals_category',annotation_input_variant='og_title_v1')
    reserve_path=getattr(args,'reserved_ids',None)
    reserved=set(json.loads(reserve_path.read_text(encoding='utf-8')).get('ids',[])) if reserve_path and reserve_path.exists() else set()
    records_by_id = {r["article_id"]: r for r in records}
    if len(records_by_id) != len(records):
        raise ValueError("Duplicate article_id in selected contributor")
    with args.csv.open(encoding="utf-8-sig", newline="") as f:
        source_fields = list(csv.DictReader(f).fieldnames or [])

    output_lock = _acquire_output_lock(args.output)
    stop_event = threading.Event()
    try:
        if provider is None and provider_factory is None:
            if not args.key_stdin:
                raise ValueError("Use --key-stdin")
            print("Reading ephemeral Gemini key (hidden; never saved)", flush=True)
            key = getpass.getpass("Gemini API key: ") if sys.stdin.isatty() else sys.stdin.readline().strip()
            if not key:
                raise ValueError("Empty API key")
            from src.providers.gemini import GeminiProvider
            provider_factory = lambda: GeminiProvider(key, args.model, args.timeout)
            metadata = SimpleNamespace(model=args.model, provider_name="gemini", api_key=key)
        elif provider_factory is not None:
            metadata = SimpleNamespace(model=args.model, provider_name="gemini", api_key="")
        else:
            provider_factory = lambda: provider
            metadata = provider
        shared = SharedRateLimitedProvider(provider_factory, model=getattr(metadata, "model", args.model), rpm=args.rpm,
                                           stop_event=stop_event, provider_name=getattr(metadata, "provider_name", "gemini"),
                                           api_key=getattr(metadata, "api_key", ""))
        checkpoint = args.output / "checkpoint.jsonl"
        results = load_jsonl(checkpoint)
        if args.seed_csv:
            for seed in _read_csv_retry(args.seed_csv):
                aid = seed.get("article_id")
                if aid not in records_by_id:
                    raise ValueError(f"Seed ID not in contributor input: {aid}")
                _validate_result(seed, records_by_id[aid], args.image_root, metadata,version,hash_fn)
                if aid in results and results[aid] != seed:
                    raise ValueError(f"Seed conflicts with checkpoint: {aid}")
                if aid not in results:
                    append_checkpoint(checkpoint, seed)
                    results[aid] = seed
        for aid, row in results.items():
            if aid not in records_by_id:
                raise ValueError(f"Checkpoint ID not in selected input: {aid}")
            _validate_result(row, records_by_id[aid], args.image_root, metadata,version,hash_fn)

        published = _published_skip_ids(args.skip_published, records_by_id)
        external_skipped = len(published - results.keys())
        pending = [r for r in records if r["article_id"] not in results and r["article_id"] not in published and r['article_id'] not in reserved and r['article_id'] not in input_holds]
        attempted = errors = 0
        reported_tokens = sum(int(r.get("annotation_total_tokens", 0) or 0) for r in results.values())
        logical_passes = sum(int(r.get("annotation_model_calls", 0) or 0) for r in results.values())
        prior_attempts = sum(int(r.get("annotation_attempts", r.get("annotation_model_calls", 0)) or 0) for r in results.values())
        prior_progress = args.output / "progress.json"
        if prior_progress.exists():
            try:
                previous = json.loads(prior_progress.read_text(encoding="utf-8"))
                prior_attempts = max(prior_attempts, int(previous.get("reported_model_calls", previous.get("calls", 0)) or 0))
            except (OSError, ValueError, TypeError):
                pass
        physical_start = shared.physical_calls
        last_id = last_safe_error = None
        status = "running"
        since_export = 0
        consecutive_errors = 0
        max_attempts = max(1, int(args.record_attempts))
        review = review_fn or reviewed_record

        def progress_snapshot():
            ordered = [results[r["article_id"]] for r in records if r["article_id"] in results]
            leader = load_leader_audit(args.output)
            held = {aid for aid, value in leader.items() if value in {"disputed", "quarantined"}}
            return ordered, dict(
                attempted_articles=attempted, processed_articles=len(ordered),
                completed=sum(is_full(r) and r["article_id"] not in held for r in ordered),
                needs_review=sum(not is_full(r) or r["article_id"] in held for r in ordered),
                errors=errors, total_rows=len(records), remaining=max(0, len(records)-len(ordered)-external_skipped),
                external_completed_skipped=external_skipped, reported_tokens=reported_tokens,
                reserved_for_agents=len(reserved),source_input_holds=len(input_holds),quality_mode=mode,
                logical_passes=logical_passes, reported_attempts=prior_attempts + shared.physical_calls-physical_start,
                calls=prior_attempts + shared.physical_calls-physical_start,
                reported_model_calls=prior_attempts + shared.physical_calls-physical_start,
                physical_api_calls_this_run=shared.physical_calls-physical_start,
                workers=args.workers, rpm=args.rpm, status=status,
                last_id=last_id, last_safe_error=last_safe_error,
                heartbeat=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

        def save(force=False):
            nonlocal since_export
            ordered, progress = progress_snapshot()
            from scripts.run_contributor_batch import atomic_json
            atomic_json(args.output / "progress.json", progress)
            if force or since_export >= 10:
                _export_preserving_review_sample(args.output, ordered, progress, source_fields)
                since_export = 0

        save(True)

        def do_record(rec):
            aid = rec["article_id"]
            for attempt in range(1, max_attempts + 1):
                try:
                    row = review(rec, shared, args.image_root)
                    return aid, row, None, attempt
                except ProviderError as exc:
                    if exc.kind == "stopped":
                        return aid, None, exc, attempt
                    if exc.quota_exhausted:
                        shared.request_stop("quota_exhausted")
                        return aid, None, exc, attempt
                    if exc.kind == "authentication":
                        shared.request_stop("authentication_error")
                        return aid, None, exc, attempt
                    if attempt < max_attempts and exc.retryable:
                        delay = exc.retry_after_seconds
                        if exc.kind == "rate_limit" and delay is None:
                            delay = min(2 ** attempt, 30)
                        if delay and stop_event.wait(max(0, delay)):
                            return aid, None, ProviderError("Runner stopped during retry wait.", kind="stopped"), attempt
                        continue
                    return aid, None, exc, attempt
                except Exception as exc:
                    return aid, None, exc, attempt
            raise AssertionError("unreachable")

        pending_ids = {r["article_id"] for r in pending}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="label") as pool:
            futures = {}
            next_index = 0
            while next_index < len(pending) or futures:
                # Refresh publisher output immediately before each new submission.
                while (not stop_event.is_set() and len(futures) < args.workers and next_index < len(pending)):
                    if args.skip_published and args.skip_published.exists():
                        fresh = _published_skip_ids(args.skip_published, records_by_id)
                        external_skipped = len((published | fresh) - results.keys())
                        published.update(fresh)
                        while next_index < len(pending) and pending[next_index]["article_id"] in published:
                            next_index += 1
                    if next_index >= len(pending):
                        break
                    rec = pending[next_index]
                    next_index += 1
                    attempted += 1
                    last_id = rec["article_id"]
                    futures[pool.submit(do_record, rec)] = rec
                if not futures:
                    break
                try:
                    done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                except KeyboardInterrupt:
                    shared.request_stop('interrupted')
                    continue  # Drain submitted work and persist successes before exiting.
                for future in done:
                    rec = futures.pop(future)
                    aid, row, exc, attempts = future.result()
                    if row is not None:
                        append_checkpoint(checkpoint, row)
                        results[aid] = row
                        logical_passes += int(row.get("annotation_model_calls", 0) or 0)
                        reported_tokens += int(row.get("annotation_total_tokens", 0) or 0)
                        consecutive_errors = 0
                        since_export += 1
                        print(json.dumps({"article_id": aid, "status": row.get("annotation_status"),
                                          "logical_passes": row.get("annotation_model_calls"),
                                          "tokens": row.get("annotation_total_tokens")}, ensure_ascii=True), flush=True)
                    elif exc is not None and getattr(exc, "kind", None) == "stopped":
                        pass
                    else:
                        errors += 1
                        consecutive_errors += 1
                        last_safe_error = sanitize_error_message(exc, getattr(metadata, "api_key", ""))
                        print(json.dumps({"article_id": aid, "error": last_safe_error, "attempts": attempts}, ensure_ascii=True), flush=True)
                        if getattr(exc, "quota_exhausted", False):
                            shared.request_stop("quota_exhausted")
                        elif getattr(exc, "kind", None) == "authentication":
                            shared.request_stop("authentication_error")
                        elif consecutive_errors >= 3:
                            shared.request_stop("infrastructure_error")
                    save()
                if shared.stop_event.is_set():
                    next_index = len(pending)
            # ThreadPool context waits for every already-submitted task, preserving successes.

        stop_reason = shared.stop_reason
        if stop_reason:
            status = stop_reason
        elif len(results) + external_skipped >= len(records):
            status = "completed"
        elif errors:
            status = "completed_with_errors"
        else:
            status = "stopped"
        save(True)
        return 2 if status in {"quota_exhausted", "authentication_error"} else 3 if status == "infrastructure_error" else 0
    finally:
        output_lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--image-root", type=Path, default=ROOT.parent / "data_pr" / "clean_final")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--skip-published", type=Path, default=DEFAULT_SKIP_PUBLISHED)
    parser.add_argument("--contributor", default="Nguyễn Minh Tuấn")
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--key-stdin", action="store_true")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--rpm", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--record-attempts", type=int, default=3)
    parser.add_argument("--seed-csv", type=Path)
    parser.add_argument('--mode',choices=['robust','fast'],default='robust')
    parser.add_argument('--canonical-titles',type=Path)
    parser.add_argument('--reserved-ids',type=Path)
    args = parser.parse_args(argv)
    if min(args.timeout, args.rpm, args.workers, args.record_attempts) < 1:
        parser.error("timeout, rpm, workers, and record-attempts must be positive")
    if args.workers > 4:
        parser.error("workers must be at most 4")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
