"""Independent and combined task execution with missing-data safeguards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from pydantic import BaseModel

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
)
from src.labeling.validation import validate_task_output
from src.providers.base import LLMProvider, ProviderResponse


TASK_SCHEMAS: dict[str, type[BaseModel]] = {
    "detection": DetectionOutput,
    "type": TypeOutput,
    "consistency": ConsistencyOutput,
}


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
            response = self.provider.generate_structured(prompt, TASK_SCHEMAS[task])
            outputs[task] = validate_task_output(
                task,
                response.data,
                self.taxonomy,
                has_content=complete_content,
            )
            usage[task] = self._usage(response)
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
            response: ProviderResponse = self.provider.generate_structured(prompt, CombinedOutput)
            combined = response.data
            if not isinstance(combined, CombinedOutput):
                raise ValueError("Kết quả Combined không khớp schema.")
            for task in ready:
                task_output = getattr(combined, task)
                outputs[task] = validate_task_output(
                    task,
                    task_output,
                    self.taxonomy,
                    has_content=complete_content,
                )
            # One combined API request has one usage record. Attribute it once.
            usage[ready[0]] = self._usage(response)
        return LabelExecution(outputs, truncated, usage)

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
        return None

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
        return bool(content) and len(content) <= self.max_content_chars

    @staticmethod
    def _usage(response: ProviderResponse) -> dict[str, int | None]:
        return {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
        }
