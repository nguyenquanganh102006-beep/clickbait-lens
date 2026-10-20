"""Versioned task outputs for structured pseudo-labels."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TaskStatus = Literal["success", "insufficient_data", "not_applicable", "error", "abstain"]


class TaskOutputBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: TaskStatus = Field(description="Trạng thái của tác vụ.")
    confidence: float | None = Field(ge=0, le=1, description="Mức tự tin do model tự đánh giá, không phải xác suất đã hiệu chỉnh.")
    reason: str | None = Field(max_length=500, description="Giải thích ngắn dựa trên dữ liệu đầu vào, không gồm suy luận ẩn.")


class DetectionOutput(TaskOutputBase):
    label: Literal[0, 1] | None = Field(description="0 là Non-clickbait; 1 là Clickbait.")

    @model_validator(mode="after")
    def validate_label_status(self) -> "DetectionOutput":
        if self.status == "success" and self.label is None:
            raise ValueError("Nhãn Detection thành công phải có label 0 hoặc 1.")
        if self.status != "success" and self.label is not None:
            raise ValueError("Detection chưa thành công không được chứa label.")
        return self


class TypeOutput(TaskOutputBase):
    labels: list[str] = Field(description="Danh sách nhãn clickbait type; có thể rỗng.")

    @model_validator(mode="after")
    def validate_labels_status(self) -> "TypeOutput":
        if self.status != "success" and self.labels:
            raise ValueError("Type chưa thành công không được chứa nhãn.")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Danh sách nhãn Type không được có phần tử trùng.")
        return self


class ConsistencyOutput(TaskOutputBase):
    label: Literal["consistent", "partial", "inconsistent"] | None = Field(
        description="Mức độ nhất quán giữa tiêu đề và toàn văn bài báo."
    )
    evidence: str | None = Field(max_length=500, description="Bằng chứng ngắn lấy từ nội dung được cung cấp.")

    @model_validator(mode="after")
    def validate_label_status(self) -> "ConsistencyOutput":
        if self.status == "success" and self.label is None:
            raise ValueError("Consistency thành công phải có một nhãn hợp lệ.")
        if self.status != "success" and self.label is not None:
            raise ValueError("Consistency chưa thành công không được chứa label.")
        return self


class CombinedOutput(BaseModel):
    """One-request output; unselected tasks must be marked not_applicable."""

    model_config = ConfigDict(extra="forbid")

    detection: DetectionOutput = Field(description="Kết quả Clickbait Detection.")
    type: TypeOutput = Field(description="Kết quả Clickbait Type Classification.")
    consistency: ConsistencyOutput = Field(description="Kết quả Title–Content Consistency.")
