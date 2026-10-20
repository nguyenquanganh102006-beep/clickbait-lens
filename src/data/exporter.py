"""Build reviewable, provenance-rich exports from completed job results."""

from __future__ import annotations

from collections import Counter
from io import BytesIO
from typing import Any, Mapping, Sequence

import pandas as pd

from src.jobs.storage import json_dumps


TASK_FIELDS = {
    "detection": ("label", "reason"),
    "type": ("labels", "reason"),
    "consistency": ("label", "evidence", "reason"),
}


def _json_text(value: Any) -> str:
    return json_dumps(value)


def _public_settings(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _public_settings(item)
            for key, item in value.items()
            if not any(
                word in str(key).casefold()
                for word in ("api_key", "secret", "password", "credential", "budget", "price", "cost")
            )
        }
    if isinstance(value, (list, tuple)):
        return [_public_settings(item) for item in value]
    return value


def review_reasons(
    tasks: Mapping[str, Mapping[str, Any]], confidence_threshold: float = 0.6
) -> list[dict[str, Any]]:
    """Return task-level reasons for adding a record to the human review queue."""
    reasons: list[dict[str, Any]] = []
    for task, result in tasks.items():
        status = str(result.get("status") or "pending")
        output = result.get("output") or {}
        confidence = output.get("confidence")
        reason = None
        if status in {"insufficient_data", "abstain", "error"}:
            reason = status
        elif status == "not_applicable":
            continue
        elif status == "success" and confidence is None:
            reason = "missing_confidence"
        elif status == "success" and float(confidence) < confidence_threshold:
            reason = "low_confidence"
        if result.get("input_truncated"):
            reason = "input_truncated" if reason is None else f"{reason};input_truncated"
        if reason:
            reasons.append(
                {
                    "task": task,
                    "status": status,
                    "review_reason": reason,
                    "confidence": confidence,
                    "output": output,
                    "error_kind": result.get("error_kind"),
                    "error_message": result.get("error_message"),
                    "attempts": result.get("attempts", 0),
                    "input_truncated": bool(result.get("input_truncated")),
                }
            )
    return reasons


def _metadata(job: Mapping[str, Any], review_status: str) -> dict[str, Any]:
    return {
        "job_id": job.get("job_id"),
        "provider": job.get("provider"),
        "model": job.get("model"),
        "mode": job.get("mode"),
        "prompt_version": job.get("prompt_version"),
        "taxonomy_version": job.get("taxonomy_version"),
        "schema_version": job.get("schema_version"),
        "label_source": "mock" if job.get("provider") == "mock" else "llm",
        "human_verified": False,
        "review_status": review_status,
    }


def _labeled_rows(
    job: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_columns = {str(key) for record in records for key in (record.get("original") or {})}
    generated_prefix = "pseudo"
    suffix = 1
    while any(column.startswith(generated_prefix + "_") for column in source_columns):
        suffix += 1
        generated_prefix = f"pseudo{suffix}"
    for record in records:
        original = dict(record.get("original") or {})
        if "record_id" not in original:
            original["record_id"] = record.get("record_id")
        tasks = record.get("tasks") or {}
        reasons = review_reasons(tasks, confidence_threshold)
        review_status = "needs_review" if reasons else "pending"
        row = dict(original)
        for key, value in _metadata(job, review_status).items():
            row[f"{generated_prefix}_{key}"] = value
        row[f"{generated_prefix}_review_reasons"] = _json_text([item["review_reason"] for item in reasons])
        for task in job.get("tasks", tasks.keys()):
            result = tasks.get(task, {})
            output = result.get("output") or {}
            prefix = f"{generated_prefix}_{task}"
            row[f"{prefix}_status"] = result.get("status", "pending")
            for field in TASK_FIELDS.get(task, ()):
                row[f"{prefix}_{field}"] = _json_text(output.get(field)) if field == "labels" else output.get(field)
            row[f"{prefix}_confidence"] = output.get("confidence")
            row[f"{prefix}_input_truncated"] = bool(result.get("input_truncated", False))
            row[f"{prefix}_attempts"] = result.get("attempts", 0)
            row[f"{prefix}_error_kind"] = result.get("error_kind")
            row[f"{prefix}_error_message"] = result.get("error_message")
            row[f"{prefix}_input_tokens"] = result.get("input_tokens")
            row[f"{prefix}_output_tokens"] = result.get("output_tokens")
        rows.append(row)
    return rows


def _prediction_rows(
    job: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for record in records:
        tasks = record.get("tasks") or {}
        reasons = review_reasons(tasks, confidence_threshold)
        review_status = "needs_review" if reasons else "pending"
        labels: dict[str, Any] = {}
        for task, result in tasks.items():
            labels[task] = {
                **(result.get("output") or {}),
                "status": result.get("status", "pending"),
                "input_truncated": bool(result.get("input_truncated", False)),
                "attempts": result.get("attempts", 0),
                "error_kind": result.get("error_kind"),
                "error_message": result.get("error_message"),
                "usage": {
                    "input_tokens": result.get("input_tokens"),
                    "output_tokens": result.get("output_tokens"),
                    "total_tokens": result.get("total_tokens"),
                },
            }
        predictions.append(
            {
                "record_id": record.get("record_id"),
                "input": record.get("input", {}),
                "original": record.get("original", {}),
                "labels": labels,
                "metadata": _metadata(job, review_status),
                "review_reasons": reasons,
            }
        )
    return predictions


def _summary(
    job: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    confidence_threshold: float,
) -> dict[str, Any]:
    task_statuses: dict[str, Counter[str]] = {}
    label_counts: dict[str, Counter[str]] = {}
    total_tokens = Counter()
    review_count = 0
    error_count = 0

    for record in records:
        tasks = record.get("tasks") or {}
        if review_reasons(tasks, confidence_threshold):
            review_count += 1
        for task, result in tasks.items():
            task_statuses.setdefault(task, Counter())[str(result.get("status") or "pending")] += 1
            if result.get("status") == "error":
                error_count += 1
            output = result.get("output") or {}
            if task == "detection" and output.get("label") is not None:
                label_counts.setdefault(task, Counter())[str(output["label"])] += 1
            elif task == "consistency" and output.get("label"):
                label_counts.setdefault(task, Counter())[str(output["label"])] += 1
            elif task == "type":
                for label in output.get("labels", []):
                    label_counts.setdefault(task, Counter())[str(label)] += 1
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                value = result.get(key)
                if value is not None:
                    total_tokens[key] += int(value)

    return {
        "job": {
            key: job.get(key)
            for key in (
                "job_id", "status", "provider", "model", "mode", "tasks",
                "prompt_version", "taxonomy_version", "schema_version",
                "input_fingerprint", "total_items", "created_at", "updated_at", "error_summary",
            )
        },
        "settings": _public_settings(job.get("settings", {})),
        "records_exported": len(records),
        "review_threshold": confidence_threshold,
        "records_needing_review": review_count,
        "task_status_counts": {task: dict(counts) for task, counts in task_statuses.items()},
        "label_counts": {task: dict(counts) for task, counts in label_counts.items()},
        "error_count": error_count,
        "token_totals": dict(total_tokens),
    }


def build_export_files(
    job: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    confidence_threshold: float = 0.6,
) -> dict[str, bytes]:
    """Return all planned export artifacts as UTF-8 bytes for download or save."""
    confidence_threshold = min(1.0, max(0.0, float(confidence_threshold)))
    labeled = pd.DataFrame(_labeled_rows(job, records, confidence_threshold))
    labeled_buffer = BytesIO()
    labeled.to_csv(labeled_buffer, index=False, encoding="utf-8-sig")

    predictions = _prediction_rows(job, records, confidence_threshold)
    predictions_jsonl = "".join(_json_text(item) + "\n" for item in predictions).encode("utf-8")

    review_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    for record in records:
        canonical_input = record.get("input") or {}
        original = record.get("original") or {}
        for item in review_reasons(record.get("tasks") or {}, confidence_threshold):
            output = item["output"]
            review_rows.append(
                {
                    "record_id": record.get("record_id"),
                    "task": item["task"],
                    "status": item["status"],
                    "review_reason": item["review_reason"],
                    "confidence": item["confidence"],
                    "reason": output.get("reason"),
                    "evidence": output.get("evidence"),
                    "error_kind": item["error_kind"],
                    "error_message": item["error_message"],
                    "attempts": item["attempts"],
                    "input_truncated": item["input_truncated"],
                    "source": canonical_input.get("source", original.get("source")),
                    "url": canonical_input.get("url", original.get("url")),
                    "title": canonical_input.get("title", original.get("title")),
                }
            )
        for task, result in (record.get("tasks") or {}).items():
            if result.get("status") == "error":
                error_rows.append(
                    {
                        "record_id": record.get("record_id"),
                        "task": task,
                        "error_kind": result.get("error_kind"),
                        "error_message": result.get("error_message"),
                        "attempts": result.get("attempts", 0),
                    }
                )

    review_csv = pd.DataFrame(
        review_rows,
        columns=("record_id", "task", "status", "review_reason", "confidence", "reason", "evidence", "error_kind", "error_message", "attempts", "input_truncated", "source", "url", "title"),
    )
    review_buffer = BytesIO()
    review_csv.to_csv(review_buffer, index=False, encoding="utf-8-sig")

    errors_csv = pd.DataFrame(error_rows, columns=("record_id", "task", "error_kind", "error_message", "attempts"))
    errors_buffer = BytesIO()
    errors_csv.to_csv(errors_buffer, index=False, encoding="utf-8-sig")

    summary_json = (_json_text(_summary(job, records, confidence_threshold)) + "\n").encode("utf-8")
    return {
        "labeled_dataset.csv": labeled_buffer.getvalue(),
        "predictions.jsonl": predictions_jsonl,
        "review_queue.csv": review_buffer.getvalue(),
        "summary.json": summary_json,
        "errors.csv": errors_buffer.getvalue(),
    }
