"""Canonical column mapping and task-level dataset validation."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

import pandas as pd


CANONICAL_FIELDS = (
    "record_id",
    "title",
    "content",
    "lead",
    "source",
    "url",
    "published_at",
)

_ALIASES: dict[str, tuple[str, ...]] = {
    "record_id": ("record_id", "recordid", "id", "article_id", "news_id"),
    "title": ("title", "headline", "news_title", "article_title", "tieu_de", "tieude"),
    "content": ("content", "body", "body_text", "full_text", "article_text", "text"),
    "lead": ("lead", "sapo", "summary", "lead_paragraph", "description", "abstract"),
    "source": ("source", "publisher", "outlet", "newspaper", "publication"),
    "url": ("url", "link", "article_url", "news_url"),
    "published_at": ("published_at", "publish_datetime", "published", "date", "datetime"),
}


def _normalized_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def suggest_column_mapping(columns: Sequence[Any]) -> dict[str, str | None]:
    """Suggest source columns based on common Vietnamese and English names."""
    normalized_columns: dict[str, list[str]] = {}
    for column in columns:
        normalized_columns.setdefault(_normalized_name(column), []).append(str(column))

    mapping: dict[str, str | None] = {}
    used: set[str] = set()
    for field in CANONICAL_FIELDS:
        match = None
        for alias in _ALIASES[field]:
            candidates = normalized_columns.get(_normalized_name(alias), [])
            available = [candidate for candidate in candidates if candidate not in used]
            if available:
                match = available[0]
                break
        mapping[field] = match
        if match is not None:
            used.add(match)
    return mapping


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text or None


def _normalize_url(value: str) -> str:
    try:
        parts = urlsplit(value.strip())
    except ValueError:
        return value.strip().split("#", 1)[0].rstrip("/")
    if not parts.netloc:
        return value.strip().split("#", 1)[0].rstrip("/")
    netloc = parts.netloc.casefold()
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.casefold(), netloc, path, parts.query, ""))


def _stable_record_id(row: Mapping[str, Any], mapped: Mapping[str, str | None]) -> str:
    url_value = _as_text(row.get(mapped.get("url"))) if mapped.get("url") else None
    if url_value:
        identity = "url:" + _normalize_url(url_value)
    else:
        fields = ("title", "content", "lead", "source")
        values = {
            field: _as_text(row.get(mapped.get(field))) if mapped.get(field) else None
            for field in fields
        }
        if not any(values.values()):
            # Include the full source row when no canonical text is mapped.
            values = {str(key): _as_text(value) for key, value in row.items()}
        identity = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"rec_{digest}"


def normalize_records(
    frame: pd.DataFrame,
    mapping: Mapping[str, str | None],
) -> list[dict[str, Any]]:
    """Create canonical records while retaining every original source column."""
    records: list[dict[str, Any]] = []
    generated_counts: Counter[str] = Counter()
    for raw_row in frame.to_dict(orient="records"):
        raw = dict(raw_row)
        lookup = {str(key): value for key, value in raw.items()}
        record = dict(raw)
        for field in CANONICAL_FIELDS:
            source_column = mapping.get(field)
            value = lookup.get(source_column) if source_column else None
            if field == "record_id":
                value = _as_text(value)
            elif field in {"title", "content", "lead", "source", "url", "published_at"}:
                value = _as_text(value)
            record[field] = value

        if not record["record_id"]:
            base_id = _stable_record_id(lookup, mapping)
            generated_counts[base_id] += 1
            occurrence = generated_counts[base_id]
            record["record_id"] = base_id if occurrence == 1 else f"{base_id}_{occurrence}"
        record["_original"] = raw
        records.append(record)
    return records


def diagnose_dataset(frame: pd.DataFrame, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize empty columns, duplicate identifiers/URLs and missing titles."""
    empty_columns = [str(column) for column in frame.columns if frame[column].isna().all()]
    ids = [str(record.get("record_id")) for record in records if record.get("record_id")]
    urls = [str(record.get("url")).strip() for record in records if record.get("url")]
    duplicate_ids = sorted(value for value, count in Counter(ids).items() if count > 1)
    duplicate_urls = sorted(value for value, count in Counter(urls).items() if count > 1)
    missing_titles = sum(not _as_text(record.get("title")) for record in records)
    return {
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "empty_columns": empty_columns,
        "duplicate_ids": duplicate_ids,
        "duplicate_urls": duplicate_urls,
        "missing_title_count": missing_titles,
    }


def task_eligibility(
    records: Sequence[Mapping[str, Any]],
    *,
    min_consistency_content_chars: int = 80,
    max_content_chars: int = 20000,
) -> dict[str, dict[str, int]]:
    """Count task-ready records and known missing/length limitations."""
    result = {
        "detection": {"eligible": 0, "missing_title": 0},
        "type": {"eligible": 0, "missing_title": 0},
        "consistency": {
            "eligible": 0,
            "missing_title": 0,
            "insufficient_content": 0,
            "truncated_content": 0,
        },
    }
    for record in records:
        has_title = bool(_as_text(record.get("title")))
        content = _as_text(record.get("content"))
        has_content = bool(content)
        for task in ("detection", "type"):
            result[task]["eligible" if has_title else "missing_title"] += 1
        if not has_title:
            result["consistency"]["missing_title"] += 1
        elif not has_content:
            result["consistency"]["insufficient_content"] += 1
        elif len(content) < max(1, int(min_consistency_content_chars)):
            result["consistency"]["insufficient_content"] += 1
        elif len(content) > max(1, int(max_content_chars)):
            result["consistency"]["truncated_content"] += 1
        else:
            result["consistency"]["eligible"] += 1
    return result
