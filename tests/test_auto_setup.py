from types import SimpleNamespace

import pytest

from src.providers.base import ProviderError
from src.providers.mock import MockProvider
from src.ui.auto_setup import auto_find, rank_models


def factory_for(names, failures, calls):
    def factory(provider, key, model, timeout):
        def generate(prompt, schema):
            calls.append(model)
            failure = failures.get(model)
            if isinstance(failure, list):
                failure = failure.pop(0) if failure else None
            if failure:
                raise failure
            return MockProvider().generate_structured(prompt, schema)
        return SimpleNamespace(list_models=lambda: names, generate_structured=generate)
    return factory


def test_auto_skips_400_404_then_stops_at_first_valid_model():
    calls = []
    factory = factory_for(["a", "b", "c", "d"], {
        "a": ProviderError("unsupported schema", status_code=400),
        "b": ProviderError("unavailable", status_code=404),
    }, calls)
    chosen, results = auto_find("gemini", "fake", 5, ["detection"], "independent", factory, lambda _: None)
    assert chosen == "c"
    assert calls == ["a", "b", "c"]
    assert not results["a"]["ok"] and not results["b"]["ok"]
    assert results["c"]["ok"]


def test_auto_waits_once_for_429_and_recovers():
    calls, waits = [], []
    factory = factory_for(["a", "b"], {"a": [ProviderError("retry in 1s", status_code=429)]}, calls)
    chosen, _ = auto_find("gemini", "fake", 5, ["detection"], "independent", factory, lambda _: None, sleep=waits.append)
    assert chosen == "a" and calls == ["a", "a"]
    assert len(waits) == 10


@pytest.mark.parametrize("error", [
    ProviderError("bad key", kind="authentication", status_code=403),
])
def test_auto_stops_on_quota_or_auth_without_calling_other_models(error):
    calls, waits = [], []
    factory = factory_for(["a", "b"], {"a": error}, calls)
    with pytest.raises(ProviderError):
        auto_find("gemini", "fake", 5, ["detection"], "independent", factory, lambda _: None, sleep=waits.append)
    assert calls == ["a"] and waits == []


def test_auto_bounds_rate_limit_retry_and_number_of_candidates():
    calls, waits = [], []
    factory = factory_for(["a", "b"], {"a": ProviderError("rate limit", status_code=429)}, calls)
    chosen, _ = auto_find("gemini", "fake", 5, ["detection"], "independent", factory, lambda _: None, sleep=waits.append)
    assert chosen == "b"
    assert calls == ["a", "a", "b"] and len(waits) == 60
    calls.clear()
    names = list("abcdefghij")
    factory = factory_for(names, {name: ProviderError("bad model", status_code=400) for name in names}, calls)
    with pytest.raises(ProviderError):
        auto_find("gemini", "fake", 5, ["detection"], "independent", factory, lambda _: None)
    assert len(calls) == 6


def test_auto_checks_all_selected_tasks_and_filters_other_modalities():
    assert rank_models(["gemini-3-flash", "gemini-3-image", "gemini-3-tts"]) == ["gemini-3-flash"]
    calls = []
    factory = factory_for(["a"], {}, calls)
    auto_find("gemini", "fake", 5, ["detection", "type", "consistency"], "independent", factory, lambda _: None)
    assert calls == ["a", "a", "a"]


def test_auto_prioritizes_saved_successes_and_retries_503():
    calls, waits = [], []
    factory = factory_for(["argon", "good-a", "good-b"], {
        "good-a": [ProviderError("temporary", status_code=503), ProviderError("temporary", status_code=503)],
    }, calls)
    chosen, _ = auto_find("gemini", "fake", 5, ["detection"], "independent", factory, lambda _: None,
                         preferred="good-a", known_results={"good-a": {"ok": True}, "good-b": {"ok": True}, "argon": {"ok": False}}, sleep=waits.append)
    assert chosen == "good-b"
    assert calls == ["good-a", "good-a", "good-b"] and waits == [3]


def test_model_specific_zero_quota_does_not_block_other_models():
    calls = []
    factory = factory_for(["a", "b"], {"a": ProviderError("daily limit: 0", status_code=429)}, calls)
    chosen, _ = auto_find("gemini", "fake", 5, ["detection"], "independent", factory, lambda _: None)
    assert chosen == "b" and calls == ["a", "b"]
