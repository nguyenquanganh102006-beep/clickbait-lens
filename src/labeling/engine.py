"""Independent and combined task execution with missing-data safeguards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from src.labeling.prompts import (
    PROMPT_VERSION,
    build_combined_prompt,
    build_task_prompt,
    load_taxonomy,
)
from src.labeling.schemas import (
    CombinedOutput,
    ConsistencyOutput,
    DetectionOutput,
    TypeOutput,
    TaskOutputBase,
)
from src.labeling.validation import LabelValidationError, validate_task_output
from src.providers.base import LLMProvider, ProviderResponse


TASK_SCHEMAS: dict[str, type[BaseModel]] = {
    "detection": DetectionOutput,
    "type": TypeOutput,
    "consistency": ConsistencyOutput,
}


class TypeAssessment(BaseModel):
    """Keep title style independent from whether the article answers its tease."""
    model_config = ConfigDict(extra="forbid")
    title_style: TypeOutput
    content_check: TypeOutput


class ConsistencyDraft(TaskOutputBase):
    # Coverage is adjudicated locally before enforcing public label/status rules.
    label: Literal["consistent", "partial", "inconsistent"] | None
    evidence: str | None = Field(max_length=500)


class ConsistencyAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    body_coverage: Literal["substantive", "context_only", "incomplete", "out_of_sync"]
    body_evidence: str | None = Field(max_length=500)
    output: ConsistencyDraft


@dataclass
class LabelExecution:
    outputs: dict[str, BaseModel]
    input_truncated: dict[str, bool] = field(default_factory=dict)
    usage_by_task: dict[str, dict[str, int | None]] = field(default_factory=dict)


class LabelingEngine:
    def __init__(
        self,
        provider: LLMProvider,
        taxonomy: dict[str, Any] | None = None,
        *,
        prompt_version: str = PROMPT_VERSION,
        max_content_chars: int = 20000,
        min_consistency_content_chars: int = 80,
    ) -> None:
        self.provider = provider
        self.taxonomy = taxonomy or load_taxonomy()
        self.prompt_version = prompt_version
        self.max_content_chars = max(1, int(max_content_chars))
        self.min_consistency_content_chars = max(1, int(min_consistency_content_chars))

    def label_record(
        self,
        record: dict[str, Any],
        tasks: Iterable[str],
        *,
        mode: str = "independent",
    ) -> LabelExecution:
        selected = list(dict.fromkeys(tasks))
        invalid_tasks = set(selected) - set(TASK_SCHEMAS)
        if invalid_tasks:
            raise ValueError("Task không được hỗ trợ: " + ", ".join(sorted(invalid_tasks)))
        if not selected:
            raise ValueError("Phải chọn ít nhất một task.")
        if mode not in {"independent", "combined"}:
            raise ValueError("Mode phải là independent hoặc combined.")
        if mode == "combined":
            return self._label_combined(record, selected)
        return self._label_independent(record, selected)

    def requires_request(self, record: dict[str, Any], tasks: Iterable[str]) -> bool:
        """Return whether at least one selected task can usefully call the provider."""
        return bool(self.requestable_tasks(record, tasks))

    def requestable_tasks(self, record: dict[str, Any], tasks: Iterable[str]) -> list[str]:
        """Return selected tasks that pass local missing/length checks."""
        ready: list[str] = []
        for task in tasks:
            if task not in TASK_SCHEMAS:
                raise ValueError(f"Task không được hỗ trợ: {task}")
            truncated = self._content_was_truncated(record) and task in {"type", "consistency"}
            if self._precheck(task, record, truncated) is None:
                ready.append(task)
        return ready

    def _label_independent(self, record: dict[str, Any], tasks: list[str]) -> LabelExecution:
        outputs: dict[str, BaseModel] = {}
        truncated: dict[str, bool] = {}
        usage: dict[str, dict[str, int | None]] = {}
        complete_content = self._has_complete_content(record)

        for task in tasks:
            was_truncated = self._content_was_truncated(record) and task in {"type", "consistency"}
            truncated[task] = was_truncated
            prechecked = self._precheck(task, record, was_truncated)
            if prechecked is not None:
                outputs[task] = prechecked
                continue

            prompt = build_task_prompt(
                task,
                record,
                self.taxonomy,
                self.prompt_version,
                content_is_complete=complete_content,
            )
            structured_type = task == "type" and self.prompt_version == "v2"
            structured_consistency = task == "consistency" and self.prompt_version == "v2"
            if structured_type:
                prompt += self._type_assessment_instruction()
            if structured_consistency:
                prompt += self._consistency_assessment_instruction()
            schema = TypeAssessment if structured_type else ConsistencyAssessment if structured_consistency else TASK_SCHEMAS[task]
            response = self.provider.generate_structured(prompt, schema)
            data = self._merge_type_assessment(response.data, complete_content) if structured_type else self._merge_consistency_assessment(response.data, record) if structured_consistency else response.data
            outputs[task] = validate_task_output(
                task,
                data,
                self.taxonomy,
                has_content=complete_content,
            )
            usage[task] = self._usage(response)
            self._validate_explanation(task, outputs[task])
        return LabelExecution(outputs, truncated, usage)

    def _label_combined(self, record: dict[str, Any], tasks: list[str]) -> LabelExecution:
        complete_content = self._has_complete_content(record)
        outputs: dict[str, BaseModel] = {}
        truncated = {
            task: self._content_was_truncated(record) and task in {"type", "consistency"}
            for task in tasks
        }
        ready = self.requestable_tasks(record, tasks)
        for task in tasks:
            prechecked = self._precheck(task, record, truncated[task])
            if prechecked is not None:
                outputs[task] = prechecked

        usage: dict[str, dict[str, int | None]] = {}
        if ready:
            effective_record = dict(record)
            if not complete_content:
                effective_record["content"] = None
            prompt = build_combined_prompt(
                effective_record,
                ready,
                self.taxonomy,
                self.prompt_version,
            )
            structured_type = "type" in ready and self.prompt_version == "v2"
            structured_consistency = "consistency" in ready and self.prompt_version == "v2"
            if structured_type:
                prompt += self._type_assessment_instruction()
            if structured_consistency:
                prompt += self._consistency_assessment_instruction()
            schema = create_model("CombinedAssessment", __config__=ConfigDict(extra="forbid"), detection=(DetectionOutput, ...),
                type=(TypeAssessment if structured_type else TypeOutput, ...),
                consistency=(ConsistencyAssessment if structured_consistency else ConsistencyOutput, ...)
            ) if structured_type or structured_consistency else CombinedOutput
            response: ProviderResponse = self.provider.generate_structured(prompt, schema)
            combined = response.data
            if not isinstance(combined, schema):
                raise ValueError("Kết quả Combined không khớp schema.")
            for task in ready:
                task_output = getattr(combined, task)
                if task == "type" and structured_type:
                    task_output = self._merge_type_assessment(task_output, complete_content)
                if task == "consistency" and structured_consistency:
                    task_output = self._merge_consistency_assessment(task_output, record)
                outputs[task] = validate_task_output(
                    task,
                    task_output,
                    self.taxonomy,
                    has_content=complete_content,
                )
                self._validate_explanation(task, outputs[task])
            # One combined API request has one usage record. Attribute it once.
            usage[ready[0]] = self._usage(response)
        return LabelExecution(outputs, truncated, usage)

    @staticmethod
    def _type_assessment_instruction() -> str:
        return (
            "\n\nĐỊNH DẠNG ĐÁNH GIÁ TYPE HAI PHẦN (cùng một yêu cầu API): "
            "title_style chỉ xét cách viết TIÊU ĐỀ, chọn trong curiosity_gap/exaggeration/emotional/question_based; "
            "content_check chỉ bổ sung misleading hoặc exaggeration khi có đối chiếu khẳng định cụ thể với body/lead. Mỗi phần dùng status/labels/confidence/reason theo schema. "
            "Khi gọi gộp, hai phần này nằm trong trường type. Không dùng việc body trả lời điều bị giấu để xóa nhãn title_style. "
            "Nếu tiêu đề thực sự dùng Động thái kỳ lạ/Bước ngoặt bất ngờ mà không nói động thái/kết cục, "
            "body tiết lộ câu trả lời không miễn trừ curiosity_gap. Không yêu cầu clickbait phải sai sự thật hay nghiêm trọng. "
            "Không có body: content_check là not_applicable với labels=[]. "
            "Content khớp: content_check success với labels=[], nhưng không sửa title_style. "
            "Mã chương trình sẽ hợp các nhãn của hai phần, không cho phần đối chiếu xóa thủ pháp tiêu đề."
        )

    @staticmethod
    def _consistency_assessment_instruction() -> str:
        return (
            "\n\nĐỊNH DẠNG CONSISTENCY CÓ KIỂM TRA THÂN BÀI: "
            "Trước output, điền body_coverage theo chính CONTENT (không phải lead): "
            "substantive nếu có thông tin thực chất về sự kiện/chủ thể chính, context_only nếu chỉ giải thích chung, "
            "incomplete nếu có dấu hiệu thiếu phần chính, out_of_sync nếu content và tiêu đề thuộc các phiên bản khác nhau. "
            "body_evidence phải là một trích đoạn liên tục NGUYÊN VĂN từ content, không thêm dấu ba chấm hoặc lời dẫn; "
            "nếu không có bằng chứng thân bài thì null. output là ConsistencyOutput. Khi gọi gộp, các trường này nằm trong consistency. "
            "Không lấy lead làm body_evidence. Lead khớp nhưng content chỉ có định nghĩa chung không đủ để xác nhận consistent. "
            "Mã chương trình sẽ đưa về abstain nếu body_coverage không phải substantive; không đoán toàn văn từ sapo."
        )

    @staticmethod
    def _merge_consistency_assessment(value: BaseModel, record: dict[str, Any]) -> ConsistencyOutput:
        if not isinstance(value, ConsistencyAssessment):
            raise LabelValidationError("Thiếu đánh giá độ đầy đủ của thân bài.")
        if value.body_coverage != "substantive":
            detail = {"context_only": "chỉ có ngữ cảnh chung", "incomplete": "có dấu hiệu bị thiếu",
                      "out_of_sync": "không đồng bộ với tiêu đề"}[value.body_coverage]
            return ConsistencyOutput(status="abstain", label=None, confidence=None,
                reason=(f"Thân bài {detail}; chưa đủ dữ liệu để kết luận. {value.output.reason or ''}")[:500],
                evidence=value.body_evidence)
        if value.output.status == "success":
            evidence = " ".join((value.body_evidence or "").split())
            content = " ".join(str(record.get("content") or "").split())
            if not evidence or evidence not in content:
                raise LabelValidationError("Bằng chứng thân bài phải là đoạn nguyên văn có trong content.")
        try:
            return ConsistencyOutput.model_validate(value.output.model_dump())
        except ValueError as exc:
            raise LabelValidationError("Trạng thái/nhãn Consistency không hợp lệ.") from exc

    def _merge_type_assessment(self, value: BaseModel, has_content: bool) -> TypeOutput:
        if not isinstance(value, TypeAssessment):
            raise LabelValidationError("Thiếu hai phần đánh giá Type.")
        style, checked = value.title_style, value.content_check
        if "misleading" in style.labels or set(checked.labels) - {"misleading", "exaggeration"}:
            raise LabelValidationError("Type phải tách thủ pháp tiêu đề khỏi misleading.")
        for part in (style, checked):
            validate_task_output("type", part, self.taxonomy, has_content=has_content)
            self._validate_explanation("type", part)
        labels = list(dict.fromkeys(style.labels + checked.labels))
        successful = style.status == "success" and checked.status in {"success", "not_applicable"}
        if labels:
            successful = True
        confidence = [p.confidence for p in (style, checked) if p.status == "success" and p.confidence is not None]
        reason = f"Tiêu đề: {style.reason or style.status} Đối chiếu: {checked.reason or checked.status}"[:500]
        return TypeOutput(status="success" if successful else "abstain", labels=labels,
                          confidence=min(confidence) if confidence and successful else None, reason=reason)

    def _precheck(self, task: str, record: dict[str, Any], was_truncated: bool) -> BaseModel | None:
        title = str(record.get("title") or "").strip()
        content = str(record.get("content") or "").strip()
        if task in {"detection", "type", "consistency"} and not title:
            return self._empty_output(task, "insufficient_data", "Thiếu tiêu đề.")
        if task == "consistency":
            if not content:
                return self._empty_output(task, "insufficient_data", "Thiếu toàn văn content; không dùng lead/sapo thay thế.")
            if len(content) < self.min_consistency_content_chars:
                return self._empty_output(task, "insufficient_data", "Toàn văn quá ngắn để đối chiếu đáng tin cậy.")
            if was_truncated:
                return self._empty_output(task, "abstain", "Toàn văn vượt giới hạn độ dài và sẽ bị cắt; không đánh giá như đã đọc đủ bài.")
            if self.prompt_version != "v1" and (
                record.get("content_is_complete") is False or record.get("content_truncated") is True
            ):
                return self._empty_output(task, "abstain", "Dữ liệu đánh dấu content chưa đầy đủ; cần kiểm tra lại nguồn trước khi kết luận lệch tiêu đề.")
        return None

    def _validate_explanation(self, task: str, output: BaseModel) -> None:
        if self.prompt_version == "v1" or output.status != "success":
            return
        if not str(output.reason or "").strip():
            raise LabelValidationError("Kết quả thành công phải có lý do dựa trên dữ liệu.")
        if task == "consistency" and not str(output.evidence or "").strip():
            raise LabelValidationError("Consistency thành công phải có bằng chứng đối chiếu.")

    def _empty_output(self, task: str, status: str, reason: str) -> BaseModel:
        common = {"status": status, "confidence": None, "reason": reason}
        if task == "detection":
            return DetectionOutput(**common, label=None)
        if task == "type":
            return TypeOutput(**common, labels=[])
        return ConsistencyOutput(**common, label=None, evidence=None)

    def _content_was_truncated(self, record: dict[str, Any]) -> bool:
        content = str(record.get("content") or "")
        return len(content) > self.max_content_chars

    def _has_complete_content(self, record: dict[str, Any]) -> bool:
        content = str(record.get("content") or "").strip()
        return (bool(content) and len(content) <= self.max_content_chars
                and (self.prompt_version == "v1" or not (
                    record.get("content_is_complete") is False or record.get("content_truncated") is True
                )))

    @staticmethod
    def _usage(response: ProviderResponse) -> dict[str, int | None]:
        return {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
        }
