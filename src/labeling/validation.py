"""Post-provider checks against the selected taxonomy and task inputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.labeling.schemas import ConsistencyOutput, DetectionOutput, TypeOutput


class LabelValidationError(ValueError):
    """Raised when a structured result is valid JSON but invalid for this task."""


def validate_task_output(
    task: str,
    output: BaseModel,
    taxonomy: dict[str, Any],
    *,
    has_content: bool,
) -> BaseModel:
    if task == "detection" and not isinstance(output, DetectionOutput):
        raise LabelValidationError("Schema Detection không hợp lệ.")
    if task == "type":
        if not isinstance(output, TypeOutput):
            raise LabelValidationError("Schema Type không hợp lệ.")
        allowed = set(taxonomy.get("clickbait_types", []))
        invalid = sorted(set(output.labels) - allowed)
        if invalid:
            raise LabelValidationError("Nhãn Type không thuộc taxonomy: " + ", ".join(invalid))
        if not has_content and "misleading" in output.labels:
            raise LabelValidationError("Không thể gán misleading khi thiếu toàn văn content.")
    if task == "consistency" and not isinstance(output, ConsistencyOutput):
        raise LabelValidationError("Schema Consistency không hợp lệ.")
    if task == "consistency" and not has_content and output.status == "success":
        raise LabelValidationError("Không thể đánh giá consistency khi thiếu toàn văn content.")
    return output
