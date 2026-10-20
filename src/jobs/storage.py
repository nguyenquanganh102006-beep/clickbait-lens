"""SQLite-backed job queue, task checkpoints and prediction cache."""

from __future__ import annotations

import json
import hashlib
import math
import sqlite3
import uuid
import pandas as pd
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from zoneinfo import ZoneInfo

from src.jobs.checkpoint import build_cache_key
from src.paths import STATE_DIR


DEFAULT_DB_PATH = STATE_DIR / "jobs.sqlite3"
TERMINAL_ITEM_STATUSES = {"success", "insufficient_data", "not_applicable", "abstain", "error"}


def _json_clean(value: Any) -> Any:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _json_clean(value.item())
        except (ValueError, TypeError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def json_dumps(value: Any) -> str:
    return json.dumps(_json_clean(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _json_loads(value: str | None, fallback: Any = None) -> Any:
    if value is None:
        return fallback
    return json.loads(value)


class JobStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    tasks_json TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    taxonomy_version TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    settings_json TEXT NOT NULL,
                    input_fingerprint TEXT NOT NULL,
                    total_items INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error_summary TEXT
                );
                CREATE TABLE IF NOT EXISTS job_items (
                    job_id TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    original_json TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    result_json TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error_kind TEXT,
                    error_message TEXT,
                    input_truncated INTEGER NOT NULL DEFAULT 0,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    estimated_cost_usd REAL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, record_id, task),
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_job_items_status ON job_items(job_id, status);
                CREATE INDEX IF NOT EXISTS idx_job_items_record ON job_items(job_id, record_id);
                CREATE TABLE IF NOT EXISTS prediction_cache (
                    cache_key TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def create_job(
        self,
        records: Sequence[Mapping[str, Any]],
        tasks: Sequence[str],
        *,
        provider: str,
        model: str,
        mode: str,
        prompt_version: str,
        taxonomy_version: str,
        schema_version: str,
        settings: Mapping[str, Any],
    ) -> str:
        if not records or not tasks:
            raise ValueError("Job cần ít nhất một bản ghi và một task.")
        if mode not in {"independent", "combined"}:
            raise ValueError("Chế độ chạy không hợp lệ.")
        if len(tasks) != len(set(tasks)) or set(tasks) - {"detection", "type", "consistency"}:
            raise ValueError("Danh sách tác vụ không hợp lệ hoặc bị lặp.")
        ids = [str(record.get("record_id") or "") for record in records]
        duplicates = sorted(value for value, count in Counter(ids).items() if value and count > 1)
        if duplicates:
            raise ValueError("ID bản ghi bị trùng; hãy xử lý trước khi tạo job: " + ", ".join(duplicates[:10]))
        if any(not value for value in ids):
            raise ValueError("Mỗi bản ghi cần có record_id trước khi tạo job.")

        job_id = uuid.uuid4().hex
        now = datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds")
        safe_settings = {
            key: value
            for key, value in settings.items()
            if not any(secret_word in key.casefold() for secret_word in ("key", "secret", "token", "password"))
        }
        fingerprint_data = [
            {"record_id": record.get("record_id"), "title": record.get("title"), "content": record.get("content")}
            for record in records
        ]
        fingerprint = hashlib.sha256(json_dumps(fingerprint_data).encode("utf-8")).hexdigest()

        with self._connect() as conn:
            conn.execute(
                """INSERT INTO jobs (
                    job_id, status, provider, model, mode, tasks_json, prompt_version,
                    taxonomy_version, schema_version, settings_json, input_fingerprint,
                    total_items, created_at, updated_at
                ) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id, provider, model, mode, json_dumps(list(tasks)), prompt_version,
                    taxonomy_version, schema_version, json_dumps(safe_settings), fingerprint,
                    len(records) * len(tasks), now, now,
                ),
            )
            for record in records:
                canonical_record = {key: value for key, value in record.items() if key != "_original"}
                original = record.get("_original", {})
                for task in tasks:
                    cache_key = build_cache_key(
                        record,
                        task,
                        provider=provider,
                        model=model,
                        prompt_version=prompt_version,
                        taxonomy_version=taxonomy_version,
                        schema_version=schema_version,
                        mode=mode,
                        selected_tasks=tasks,
                        settings=settings,
                    )
                    conn.execute(
                        """INSERT INTO job_items (
                            job_id, record_id, task, input_json, original_json, cache_key,
                            status, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
                        (
                            job_id, str(record["record_id"]), task, json_dumps(canonical_record),
                            json_dumps(original), cache_key, now,
                        ),
                    )
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        job = dict(row)
        job["tasks"] = _json_loads(job.pop("tasks_json"), [])
        job["settings"] = _json_loads(job.pop("settings_json"), {})
        return job

    def list_jobs(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as conn:
            ids = conn.execute(
                "SELECT job_id FROM jobs ORDER BY created_at DESC LIMIT ?", (max(1, limit),)
            ).fetchall()
        return [job for row in ids if (job := self.get_job(row["job_id"])) is not None]

    def set_job_status(self, job_id: str, status: str, error_summary: str | None = None) -> None:
        now = datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error_summary = ?, updated_at = ? WHERE job_id = ?",
                (status, error_summary, now, job_id),
            )

    def update_job_settings(self, job_id: str, updates: Mapping[str, Any]) -> None:
        """Persist user-approved runtime setting changes without credentials."""
        job = self.get_job(job_id)
        if job is None:
            raise ValueError("Không tìm thấy job cần cập nhật cấu hình.")
        settings = dict(job["settings"])
        settings.update(updates)
        safe_settings = {
            key: value
            for key, value in settings.items()
            if not any(secret_word in key.casefold() for secret_word in ("key", "secret", "token", "password"))
        }
        now = datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET settings_json = ?, updated_at = ? WHERE job_id = ?",
                (json_dumps(safe_settings), now, job_id),
            )

    def get_progress(self, job_id: str) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM job_items WHERE job_id = ? GROUP BY status",
                (job_id,),
            ).fetchall()
        counts = {row["status"]: int(row["count"]) for row in rows}
        total = sum(counts.values())
        finished = sum(counts.get(status, 0) for status in TERMINAL_ITEM_STATUSES)
        counts.update({"total": total, "completed": finished, "pending": counts.get("pending", 0), "running": counts.get("running", 0), "error": counts.get("error", 0)})
        return counts

    def get_items(
        self,
        job_id: str,
        *,
        task: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM job_items WHERE job_id = ?"
        params: list[Any] = [job_id]
        if task is not None:
            query += " AND task = ?"
            params.append(task)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY rowid"
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, limit))
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._decode_item(row) for row in rows]

    def get_task_summary(self, job_id: str) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
        """Aggregate progress without loading article bodies into the UI."""
        with self._connect() as conn:
            rows = conn.execute("SELECT task,status,COUNT(*) AS count FROM job_items WHERE job_id=? GROUP BY task,status", (job_id,)).fetchall()
            tokens = conn.execute("SELECT SUM(input_tokens) AS input_tokens,SUM(output_tokens) AS output_tokens,SUM(total_tokens) AS total_tokens FROM job_items WHERE job_id=?", (job_id,)).fetchone()
        tasks: dict[str, dict[str, int]] = {}
        for row in rows:
            tasks.setdefault(row["task"], {})[row["status"]] = row["count"]
        return tasks, {key: int(value) for key, value in dict(tokens).items() if value is not None}

    def get_result_signature(self, job_id: str) -> str:
        with self._connect() as conn:
            rows = conn.execute("SELECT status,COUNT(*) AS count,SUM(attempts) AS attempts,MAX(updated_at) AS updated FROM job_items WHERE job_id=? GROUP BY status ORDER BY status", (job_id,)).fetchall()
        return json_dumps([dict(row) for row in rows])

    @staticmethod
    def _decode_item(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["input"] = _json_loads(item.pop("input_json"), {})
        item["original"] = _json_loads(item.pop("original_json"), {})
        item["result"] = _json_loads(item.pop("result_json"), None)
        return item

    def claim_next_work(self, job_id: str, mode: str) -> list[dict[str, Any]]:
        """Atomically claim one independent task or one record's combined tasks."""
        now = datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            first = conn.execute(
                "SELECT * FROM job_items WHERE job_id = ? AND status = 'pending' ORDER BY rowid LIMIT 1",
                (job_id,),
            ).fetchone()
            if first is None:
                return []
            if mode == "combined":
                rows = conn.execute(
                    "SELECT * FROM job_items WHERE job_id = ? AND record_id = ? AND status = 'pending' ORDER BY rowid",
                    (job_id, first["record_id"]),
                ).fetchall()
            else:
                rows = [first]
            for row in rows:
                conn.execute(
                    "UPDATE job_items SET status = 'running', updated_at = ? WHERE job_id = ? AND record_id = ? AND task = ?",
                    (now, job_id, row["record_id"], row["task"]),
                )
            return [self._decode_item(row) for row in rows]

    def get_cached_result(self, cache_key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT result_json FROM prediction_cache WHERE cache_key = ?", (cache_key,)).fetchone()
        return None if row is None else _json_loads(row["result_json"], None)

    def complete_items(self, job_id: str, results: Sequence[Mapping[str, Any]]) -> None:
        now = datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for result in results:
                output = dict(result["output"])
                status = str(output.get("status", "error"))
                if status not in TERMINAL_ITEM_STATUSES:
                    status = "error"
                usage = result.get("usage") or {}
                error_kind = "model_error" if status == "error" else None
                error_message = str(output.get("reason") or "Model trả về status error.")[:500] if status == "error" else None
                item_params = (
                    status,
                    json_dumps({
                        "output": output,
                        "input_truncated": bool(result.get("input_truncated", False)),
                        "usage": usage,
                    }),
                    int(bool(result.get("input_truncated", False))),
                    usage.get("input_tokens"),
                    usage.get("output_tokens"),
                    usage.get("total_tokens"),
                    result.get("estimated_cost_usd"),
                    error_kind,
                    error_message,
                    now,
                    job_id,
                    str(result["record_id"]),
                    str(result["task"]),
                )
                conn.execute(
                    """UPDATE job_items SET status = ?, result_json = ?, input_truncated = ?,
                        input_tokens = ?, output_tokens = ?, total_tokens = ?, estimated_cost_usd = ?,
                        error_kind = ?, error_message = ?, updated_at = ?
                       WHERE job_id = ? AND record_id = ? AND task = ?""",
                    item_params,
                )
                cache_key = result.get("cache_key")
                if cache_key and status != "error":
                    conn.execute(
                        "INSERT OR REPLACE INTO prediction_cache(cache_key, result_json, created_at) VALUES (?, ?, ?)",
                        (cache_key, json_dumps({"output": output, "input_truncated": bool(result.get("input_truncated", False)), "usage": usage}), now),
                    )

    def complete_from_cache(
        self,
        job_id: str,
        record_id: str,
        task: str,
        cache_key: str,
        cached: Mapping[str, Any],
    ) -> None:
        now = datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds")
        output = dict(cached.get("output", {}))
        status = str(output.get("status", "error"))
        if status not in TERMINAL_ITEM_STATUSES:
            status = "error"
        with self._connect() as conn:
            conn.execute(
                """UPDATE job_items SET status = ?, result_json = ?, input_truncated = ?,
                    input_tokens = ?, output_tokens = ?, total_tokens = ?, estimated_cost_usd = 0,
                    updated_at = ? WHERE job_id = ? AND record_id = ? AND task = ?""",
                (
                    status,
                    json_dumps({"output": output, "input_truncated": cached.get("input_truncated", False), "usage": {}, "cached": True}),
                    int(bool(cached.get("input_truncated", False))),
                    None, None, None,
                    now, job_id, record_id, task,
                ),
            )

    def fail_items(self, job_id: str, items: Sequence[Mapping[str, Any]], kind: str, message: str) -> None:
        now = datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds")
        with self._connect() as conn:
            for item in items:
                conn.execute(
                    """UPDATE job_items SET status = 'error', error_kind = ?, error_message = ?, updated_at = ?
                       WHERE job_id = ? AND record_id = ? AND task = ?""",
                    (kind, message[:500], now, job_id, item["record_id"], item["task"]),
                )

    def release_items(self, job_id: str, items: Sequence[Mapping[str, Any]]) -> None:
        """Return claimed work to pending when a pause arrives before a request."""
        now = datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds")
        with self._connect() as conn:
            for item in items:
                conn.execute(
                    "UPDATE job_items SET status = 'pending', updated_at = ? WHERE job_id = ? AND record_id = ? AND task = ? AND status = 'running'",
                    (now, job_id, item["record_id"], item["task"]),
                )

    def increment_attempts(self, job_id: str, items: Sequence[Mapping[str, Any]]) -> None:
        now = datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds")
        with self._connect() as conn:
            for item in items:
                conn.execute(
                    "UPDATE job_items SET attempts = attempts + 1, updated_at = ? WHERE job_id = ? AND record_id = ? AND task = ? AND status = 'running'",
                    (now, job_id, item["record_id"], item["task"]),
                )

    def retry_failed(self, job_id: str) -> int:
        now = datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds")
        with self._connect() as conn:
            result = conn.execute(
                "UPDATE job_items SET status = 'pending', error_kind = NULL, error_message = NULL, updated_at = ? WHERE job_id = ? AND status = 'error'",
                (now, job_id),
            )
            count = result.rowcount
            conn.execute(
                "UPDATE jobs SET status = 'queued', error_summary = NULL, updated_at = ? WHERE job_id = ?",
                (now, job_id),
            )
        return count

    def resume(self, job_id: str) -> None:
        now = datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE job_items SET status = 'pending', updated_at = ? WHERE job_id = ? AND status = 'running'",
                (now, job_id),
            )
            conn.execute(
                "UPDATE jobs SET status = 'queued', error_summary = NULL, updated_at = ? WHERE job_id = ?",
                (now, job_id),
            )

    def spent_cost(self, job_id: str) -> float:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(estimated_cost_usd), 0) AS total FROM job_items WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return float(row["total"] or 0)

    def get_record_results(self, job_id: str) -> list[dict[str, Any]]:
        items = self.get_items(job_id)
        grouped: dict[str, dict[str, Any]] = {}
        for item in items:
            row = grouped.setdefault(
                item["record_id"],
                {
                    "record_id": item["record_id"],
                    "input": item["input"],
                    "original": item["original"],
                    "tasks": {},
                },
            )
            result = item.get("result") or {}
            row["tasks"][item["task"]] = {
                "status": item["status"],
                "output": result.get("output"),
                "input_truncated": bool(item.get("input_truncated")),
                "error_kind": item.get("error_kind"),
                "error_message": item.get("error_message"),
                "attempts": item.get("attempts", 0),
                "input_tokens": item.get("input_tokens"),
                "output_tokens": item.get("output_tokens"),
                "total_tokens": item.get("total_tokens"),
                "estimated_cost_usd": item.get("estimated_cost_usd"),
            }
        return list(grouped.values())
