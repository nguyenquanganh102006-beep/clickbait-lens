import json
import math
from io import BytesIO

import pandas as pd

from src.data.exporter import build_export_files


def test_export_preserves_gold_label_and_adds_pseudo_label_without_secrets():
    job = {
        "job_id": "job-123",
        "status": "completed",
        "provider": "mock",
        "model": "mock-v1",
        "mode": "independent",
        "tasks": ["detection", "type"],
        "prompt_version": "v1",
        "taxonomy_version": "v1",
        "schema_version": "v1",
        "settings": {
            "api_key": "secret-value",
            "max_retries": 2,
            "budget_usd": 20.0,
            "input_price_per_million": 3.0,
            "output_price_per_million": 8.0,
        },
    }
    records = [
        {
            "record_id": "r1",
            "input": {"record_id": "r1", "title": "Tiêu đề", "source": "Báo A"},
            "original": {"headline": "Tiêu đề", "label": 0, "source": "Báo A", "optional": math.nan},
            "tasks": {
                "detection": {
                    "status": "success",
                    "output": {"status": "success", "label": 1, "confidence": 0.92, "reason": "Dấu hiệu rõ."},
                    "attempts": 1,
                    "input_tokens": 20,
                    "output_tokens": 8,
                    "total_tokens": 28,
                    "estimated_cost_usd": 0.001,
                },
                "type": {
                    "status": "abstain",
                    "output": {"status": "abstain", "labels": [], "confidence": None, "reason": "Cần kiểm tra."},
                    "attempts": 1,
                    "input_truncated": True,
                },
            },
        }
    ]

    exports = build_export_files(job, records)
    labeled = pd.read_csv(BytesIO(exports["labeled_dataset.csv"]))
    prediction = json.loads(exports["predictions.jsonl"].decode("utf-8").splitlines()[0])
    review = pd.read_csv(BytesIO(exports["review_queue.csv"]))
    summary = json.loads(exports["summary.json"].decode("utf-8"))

    assert labeled.loc[0, "label"] == 0
    assert labeled.loc[0, "pseudo_detection_label"] == 1
    assert labeled.loc[0, "pseudo_type_labels"] == "[]"
    assert prediction["metadata"]["human_verified"] is False
    assert prediction["labels"]["detection"]["label"] == 1
    assert "estimated_cost_usd" not in prediction["labels"]["detection"]
    assert prediction["original"]["optional"] is None
    assert set(review["task"]) == {"type"}
    assert summary["records_needing_review"] == 1
    assert not any("cost" in key.casefold() or "budget" in key.casefold() or "price" in key.casefold() for key in summary)
    assert "budget_usd" not in summary["settings"]
    assert "input_price_per_million" not in summary["settings"]
    assert "pseudo_detection_estimated_cost_usd" not in labeled.columns
    assert "secret-value" not in "".join(content.decode("utf-8", errors="replace") for content in exports.values())
    assert set(exports) == {
        "labeled_dataset.csv",
        "predictions.jsonl",
        "review_queue.csv",
        "summary.json",
        "errors.csv",
    }


def test_existing_pseudo_columns_are_preserved_when_labeling_again():
    job = {"job_id": "demo", "provider": "mock", "tasks": ["detection"]}
    records = [{"record_id": "r1", "original": {"title": "Tin thử", "pseudo_detection_label": 0},
                "tasks": {"detection": {"status": "success", "output": {"label": 1, "confidence": 0.9}}}}]
    frame = pd.read_csv(BytesIO(build_export_files(job, records)["labeled_dataset.csv"]))
    assert frame.loc[0, "pseudo_detection_label"] == 0
    assert frame.loc[0, "pseudo2_detection_label"] == 1
    assert frame.loc[0, "pseudo2_label_source"] == "mock"
