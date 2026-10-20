"""Shared provider interface and sanitized provider errors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeVar
import re

from pydantic import BaseModel


OutputModel = TypeVar("OutputModel", bound=BaseModel)


@dataclass(frozen=True)
class ProviderResponse:
    data: BaseModel
    raw_text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class ProviderError(RuntimeError):
    """A provider failure with a retry policy safe for the job runner."""

    def __init__(
        self,
        message: str,
        *,
        kind: str = "provider_error",
        retryable: bool = False,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
        quota_exhausted: bool = False,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        self.quota_exhausted = quota_exhausted


class LLMProvider(ABC):
    """Common interface for remote providers and the offline mock."""

    provider_name = "base"

    def __init__(self, api_key: str, model: str, timeout_seconds: int = 60) -> None:
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    def validate_key(self) -> bool:
        """Check the credentials with a low-impact provider request."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """Return model identifiers visible to the configured account."""

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        output_schema: type[OutputModel],
    ) -> ProviderResponse:
        """Generate and validate a response against a Pydantic schema."""

    def estimate_tokens(self, text: str) -> int:
        """Return a rough text-only estimate; this is not provider usage."""
        return max(1, (len(text) + 3) // 4)


def sanitize_error_message(message: object, api_key: str = "") -> str:
    """Strip credential text and truncate provider details before displaying."""
    safe = str(message).replace("\r", " ").replace("\n", " ")
    if api_key:
        safe = safe.replace(api_key, "[đã ẩn]")
    safe = re.sub(r"AIza[0-9A-Za-z_-]{30,}|sk-[0-9A-Za-z_-]{16,}", "[đã ẩn]", safe)
    return safe[:2000]


def quota_details(exc: Exception) -> dict[str, Any]:
    """Read retry hints before display truncation; distinguish exhausted quotas."""
    text = str(exc)
    headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
    retry = headers.get("retry-after")
    match = re.search(r"(?:retry in\s*|retrydelay[^0-9]*)(\d+(?:\.\d+)?)", text, re.I)
    try:
        seconds = max(0.0, float(retry if retry is not None else match[1])) if retry is not None or match else None
    except (ValueError, TypeError):
        seconds = None
    exhausted = bool(re.search(r"insufficient_quota|(?:limit\s*[:=]\s*0\b)|perday|per_day|daily", text, re.I))
    return {"retry_after_seconds": seconds, "quota_exhausted": exhausted}


def usage_value(usage: Any, *names: str) -> int | None:
    """Read a token count from SDK response objects or dictionaries."""
    for name in names:
        if isinstance(usage, dict):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        if isinstance(value, int):
            return value
    return None
