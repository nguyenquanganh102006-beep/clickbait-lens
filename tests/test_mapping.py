import pandas as pd

from src.data.mapping import (
    diagnose_dataset,
    normalize_records,
    suggest_column_mapping,
    task_eligibility,
)


def test_suggest_mapping_for_common_khdl_columns():
    mapping = suggest_column_mapping(["id", "title", "body_text", "sapo", "url"])

    assert mapping["record_id"] == "id"
    assert mapping["title"] == "title"
    assert mapping["content"] == "body_text"
    assert mapping["lead"] == "sapo"
    assert mapping["url"] == "url"


def test_normalization_preserves_source_fields_and_generates_stable_ids():
    frame = pd.DataFrame(
        {
            "headline": ["  Tiêu đề  "],
            "body_text": ["Toàn văn"],
            "legacy_label": [1],
            "url": ["HTTPS://Example.com/article/#section"],
        }
    )
    mapping = {
        "record_id": None,
        "title": "headline",
        "content": "body_text",
        "lead": None,
        "source": None,
        "url": "url",
        "published_at": None,
    }

    first = normalize_records(frame, mapping)[0]
    second = normalize_records(frame, mapping)[0]

    assert first["record_id"] == second["record_id"]
    assert first["title"] == "Tiêu đề"
    assert first["content"] == "Toàn văn"
    assert first["legacy_label"] == 1
    assert first["_original"]["headline"] == "  Tiêu đề  "


def test_task_eligibility_keeps_title_only_rows_for_detection_and_type():
    frame = pd.DataFrame({"title": ["Có tiêu đề", ""], "body_text": [None, None]})
    records = normalize_records(
        frame,
        {
            "record_id": None,
            "title": "title",
            "content": "body_text",
            "lead": None,
            "source": None,
            "url": None,
            "published_at": None,
        },
    )

    eligibility = task_eligibility(records)
    diagnostics = diagnose_dataset(frame, records)
    assert eligibility["detection"] == {"eligible": 1, "missing_title": 1}
    assert eligibility["type"] == {"eligible": 1, "missing_title": 1}
    assert eligibility["consistency"]["insufficient_content"] == 1
    assert diagnostics["missing_title_count"] == 1


def test_consistency_eligibility_observes_minimum_and_truncation_limits():
    records = [
        {"record_id": "short", "title": "Có title", "content": "ngắn"},
        {"record_id": "long", "title": "Có title", "content": "x" * 101},
        {"record_id": "ready", "title": "Có title", "content": "x" * 80},
    ]

    eligibility = task_eligibility(records, min_consistency_content_chars=80, max_content_chars=100)

    assert eligibility["consistency"] == {
        "eligible": 1,
        "missing_title": 0,
        "insufficient_content": 1,
        "truncated_content": 1,
    }
def test_numeric_excel_headers_and_malformed_url_are_safe():
    import pandas as pd
    from src.data.mapping import normalize_records
    frame = pd.DataFrame({1: ["Tiêu đề"], 2: ["http://[invalid"]})
    records = normalize_records(frame, {"title": "1", "url": "2"})
    assert records[0]["title"] == "Tiêu đề"
    assert records[0]["record_id"].startswith("rec_")
    assert records[0]["_original"][1] == "Tiêu đề"

