"""Deterministic offline provider for UI demos and tests."""

from __future__ import annotations

import hashlib
import json
from typing import Any, TypeVar

from pydantic import BaseModel

from src.providers.base import LLMProvider, ProviderError, ProviderResponse


OutputModel = TypeVar("OutputModel", bound=BaseModel)


class MockProvider(LLMProvider):
    provider_name = "mock"

    def __init__(self, api_key: str = "", model: str = "mock-v1", timeout_seconds: int = 60) -> None:
        super().__init__(api_key, model, timeout_seconds)

    def validate_key(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return ["mock-v1"]

    def generate_structured(
        self,
        prompt: str,
        output_schema: type[OutputModel],
    ) -> ProviderResponse:
        try:
            value = _example_from_schema(output_schema.model_json_schema(), prompt)
            parsed = output_schema.model_validate(value)
        except Exception as exc:
            raise ProviderError("Không tạo được phản hồi mô phỏng theo schema.", kind="schema_validation") from exc
        raw_text = json.dumps(parsed.model_dump(mode="json"), ensure_ascii=False)
        return ProviderResponse(
            data=parsed,
            raw_text=raw_text,
            model=self.model,
            input_tokens=self.estimate_tokens(prompt),
            output_tokens=self.estimate_tokens(raw_text),
            total_tokens=self.estimate_tokens(prompt) + self.estimate_tokens(raw_text),
        )


def _example_from_schema(schema: dict[str, Any], prompt: str) -> Any:
    """Produce a repeatable schema-valid example without calling an API."""
    definitions = schema.get("$defs", {})
    return _example_value(schema, prompt, definitions)


def _example_value(schema: dict[str, Any], prompt: str, definitions: dict[str, Any]) -> Any:
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        enum_values = schema["enum"]
        return "success" if "success" in enum_values else enum_values[0]
    if "anyOf" in schema:
        options = [option for option in schema["anyOf"] if option.get("type") != "null"]
        return None if not options else _example_value(options[0], prompt, definitions)
    if "$ref" in schema:
        ref_name = str(schema["$ref"]).rsplit("/", 1)[-1]
        target = definitions.get(ref_name)
        return None if target is None else _example_value(target, prompt, definitions)
    value_type = schema.get("type")
    if value_type == "object":
        return {
            key: _example_value(child, prompt, definitions)
            for key, child in schema.get("properties", {}).items()
        }
    if value_type == "array":
        return []
    if value_type == "string":
        if schema.get("format") == "date-time":
            return "2025-01-01T00:00:00Z"
        return "Mẫu minh họa ngoại tuyến."
    if value_type == "integer":
        return 0
    if value_type == "number":
        return 0.5
    if value_type == "boolean":
        return False
    if value_type == "null":
        return None
    seed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return seed[:8]
