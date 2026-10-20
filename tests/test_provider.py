import json
from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict

from src.providers.gemini import GeminiProvider
from src.providers.mock import MockProvider
from src.providers.openai import OpenAIProvider


class ExampleOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    label: int
    confidence: float
    reason: str


def test_gemini_missing_model_has_actionable_error():
    provider = object.__new__(GeminiProvider)
    provider.api_key = "test-key"
    provider.model = "unavailable-model"
    error = RuntimeError("Model unavailable")
    error.code = 404
    mapped = provider._map_exception(error)
    assert mapped.kind == "model_not_found"
    assert not mapped.retryable
    assert "404" in str(mapped)
    assert "bước 1" in str(mapped)


def test_quota_retry_hints_and_secret_redaction():
    from src.providers.base import quota_details, sanitize_error_message
    error = RuntimeError("Please retry in 73.5s, check plan and billing details")
    assert quota_details(error) == {"retry_after_seconds": 73.5, "quota_exhausted": False}
    assert quota_details(RuntimeError("quota limit: 0"))["quota_exhausted"]
    assert "fake-secret" not in sanitize_error_message("request fake-secret rejected", "fake-secret")


def test_mock_provider_returns_schema_valid_offline_response():
    provider = MockProvider()

    response = provider.generate_structured("Mẫu tiêu đề", ExampleOutput)

    assert isinstance(response.data, ExampleOutput)
    assert response.model == "mock-v1"
    assert response.total_tokens is not None
    assert "Mẫu minh họa ngoại tuyến" in response.raw_text


def test_mock_provider_model_listing_and_connection():
    provider = MockProvider()

    assert provider.validate_key()
    assert provider.list_models() == ["mock-v1"]


def test_gemini_timeout_is_applied_and_sdk_retries_are_disabled(monkeypatch):
    from google import genai

    received = {}

    def fake_client(**kwargs):
        received.update(kwargs)
        return object()

    monkeypatch.setattr(genai, "Client", fake_client)
    GeminiProvider("test-key", "test-model", timeout_seconds=17)

    options = received["http_options"]
    assert options.timeout == 17_000
    assert options.retry_options.attempts == 1


def test_openai_adapter_requests_strict_json_schema_and_parses_usage(monkeypatch):
    import openai

    received = {}
    content = json.dumps({"status": "success", "label": 1, "confidence": 0.8, "reason": "Dấu hiệu."})

    def create(**kwargs):
        received.update(kwargs)
        message = SimpleNamespace(refusal=None, content=content)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=7, total_tokens=19),
            model="test-model",
        )

    def fake_client(**kwargs):
        received["client_options"] = kwargs
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(openai, "OpenAI", fake_client)
    provider = OpenAIProvider("test-key", "test-model", timeout_seconds=23)
    response = provider.generate_structured("Tiêu đề kiểm thử", ExampleOutput)

    assert response.data.label == 1
    assert response.total_tokens == 19
    assert received["client_options"]["timeout"] == 23
    assert received["client_options"]["max_retries"] == 0
    assert received["response_format"]["json_schema"]["strict"] is True
    assert received["response_format"]["json_schema"]["schema"]["additionalProperties"] is False


def test_gemini_adapter_uses_structured_schema_and_parses_usage(monkeypatch):
    from google import genai

    received = {}
    content = '{"status":"success","label":1,"confidence":0.8,"reason":"Dấu hiệu."}'
    parsed = ExampleOutput(status="success", label=1, confidence=0.8, reason="Dấu hiệu.")

    class FakeModels:
        def generate_content(self, **kwargs):
            received.update(kwargs)
            return SimpleNamespace(
                parsed=parsed,
                text=content,
                usage_metadata=SimpleNamespace(
                    prompt_token_count=11,
                    candidates_token_count=6,
                    total_token_count=17,
                ),
                model_version="test-model",
            )

    def fake_client(**kwargs):
        received["client_options"] = kwargs
        return SimpleNamespace(models=FakeModels())

    monkeypatch.setattr(genai, "Client", fake_client)
    provider = GeminiProvider("test-key", "test-model", timeout_seconds=31)
    response = provider.generate_structured("Tiêu đề kiểm thử", ExampleOutput)

    assert response.data == parsed
    assert response.total_tokens == 17
    assert received["config"].response_json_schema == ExampleOutput.model_json_schema()
    assert received["config"].response_mime_type == "application/json"
    assert received["client_options"]["http_options"].timeout == 31_000


def test_gemini_detection_numeric_enum_and_raw_json(monkeypatch):
    from google import genai
    from src.labeling.schemas import DetectionOutput

    received = {}

    def generate_content(**kwargs):
        received.update(kwargs)
        return SimpleNamespace(
            text='{"status":"success","label":1,"confidence":0.8,"reason":"Dấu hiệu."}',
            parsed=None,
        )

    monkeypatch.setattr(genai, "Client", lambda **kwargs: SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    ))
    response = GeminiProvider("test-key", "test-model").generate_structured("Tin", DetectionOutput)
    config = received["config"]
    assert config.response_schema is None
    assert config.response_json_schema["properties"]["label"]["anyOf"][0]["enum"] == [0, 1]
    assert isinstance(response.data, DetectionOutput)
    assert response.data.label == 1
