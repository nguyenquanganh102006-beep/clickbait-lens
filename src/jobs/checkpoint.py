"""Cache-key generation for durable, reproducible task checkpoints."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


def _task_input(record: Mapping[str, Any], task: str) -> dict[str, Any]:
    if task == "detection":
        fields = ("title",)
    elif task == "type":
        fields = ("title", "content")
    elif task == "consistency":
        fields = ("title", "content")
    else:
        fields = tuple(sorted(key for key in record if not key.startswith("_")))
    return {field: record.get(field) for field in fields}


def build_cache_key(
    record: Mapping[str, Any],
    task: str,
    *,
    provider: str,
    model: str,
    prompt_version: str,
    taxonomy_version: str,
    schema_version: str,
    mode: str,
    selected_tasks: Sequence[str],
    settings: Mapping[str, Any] | None = None,
) -> str:
    request_input = _task_input(record, task)
    if mode == "combined":
        for selected_task in selected_tasks:
            request_input.update(_task_input(record, selected_task))
    input_json = json.dumps(request_input, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    input_hash = hashlib.sha256(input_json.encode("utf-8")).hexdigest()
    key_material = {
        "cache_format": 2,
        "input_hash": input_hash,
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "taxonomy_version": taxonomy_version,
        "schema_version": schema_version,
        "task": task,
        "mode": mode,
        "selected_tasks": sorted(selected_tasks) if mode == "combined" else [task],
        "settings": {
            key: value
            for key, value in (settings or {}).items()
            if key in {"max_content_chars", "min_consistency_content_chars"}
        },
    }
    serialized = json.dumps(key_material, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
