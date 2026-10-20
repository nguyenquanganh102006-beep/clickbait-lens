import pandas as pd

from src.labeling.engine import LabelingEngine
from src.labeling.schemas import ConsistencyOutput, DetectionOutput, TypeOutput
from src.providers.mock import MockProvider


def _record(title="Tiêu đề thử", content=None):
    return {"record_id": "r1", "title": title, "content": content, "lead": "Một đoạn sapo"}


def test_title_only_data_runs_detection_and_type_but_marks_consistency_insufficient():
    engine = LabelingEngine(MockProvider(), min_consistency_content_chars=30)

    result = engine.label_record(_record(content=None), ["detection", "type", "consistency"])

    assert isinstance(result.outputs["detection"], DetectionOutput)
    assert isinstance(result.outputs["type"], TypeOutput)
    assert result.outputs["detection"].status == "success"
    assert result.outputs["type"].status == "success"
    assert result.outputs["consistency"].status == "insufficient_data"


def test_short_content_is_not_used_as_full_article_consistency():
    engine = LabelingEngine(MockProvider(), min_consistency_content_chars=30)

    result = engine.label_record(
        _record(content="Bài viết rất ngắn."),
        ["consistency"],
    )

    assert isinstance(result.outputs["consistency"], ConsistencyOutput)
    assert result.outputs["consistency"].status == "insufficient_data"


def test_truncated_content_abstains_for_consistency_without_api_call():
    class CountingMock(MockProvider):
        calls = 0

        def generate_structured(self, prompt, output_schema):
            self.calls += 1
            return super().generate_structured(prompt, output_schema)

    provider = CountingMock()
    engine = LabelingEngine(provider, max_content_chars=10, min_consistency_content_chars=5)
    result = engine.label_record(_record(content="x" * 100), ["consistency"])

    assert result.outputs["consistency"].status == "abstain"
    assert result.input_truncated["consistency"] is True
    assert provider.calls == 0


def test_combined_mode_handles_nested_schema_and_prechecks_missing_content():
    engine = LabelingEngine(MockProvider())

    result = engine.label_record(
        _record(content=None),
        ["detection", "type", "consistency"],
        mode="combined",
    )

    assert result.outputs["detection"].status == "success"
    assert result.outputs["type"].status == "success"
    assert result.outputs["consistency"].status == "insufficient_data"
