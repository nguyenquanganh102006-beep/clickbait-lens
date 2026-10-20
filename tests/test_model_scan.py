from types import SimpleNamespace

from src.providers.base import ProviderError
from src.providers.mock import MockProvider
from src.ui.model_scan import scan_models


def test_full_scan_retries_429_after_delay_and_saves_each_result():
    now = [0.0]
    calls, snapshots = [], []

    def sleep(seconds):
        now[0] += seconds

    def factory(provider, key, model, timeout):
        def generate(prompt, schema):
            attempts = sum(name == model for name, _ in calls)
            calls.append((model, now[0]))
            if model == "a" and not attempts:
                raise ProviderError("retry in 70s", status_code=429)
            if model == "b":
                raise ProviderError("unsupported", status_code=400)
            return MockProvider().generate_structured(prompt, schema)
        return SimpleNamespace(list_models=lambda: ["a", "b", "c"], generate_structured=generate)

    results = scan_models("gemini", "fake", 5, ["detection"], "independent", factory,
                          lambda results, message, total: snapshots.append((len(results), message)),
                          sleep=sleep, clock=lambda: now[0])
    assert [name for name, _ in calls] == ["a", "b", "c", "a"]
    assert calls[-1][1] - calls[0][1] >= 71
    assert results["a"]["ok"] and results["a"]["attempts"] == 2
    assert results["c"]["ok"] and not results["b"]["ok"]
    assert not results["a"]["retry_pending"]
    assert any(size == 1 for size, _ in snapshots)


def test_full_scan_auth_failure_stops_and_reports():
    import pytest
    calls = []

    def factory(provider, key, model, timeout):
        def generate(prompt, schema):
            calls.append(model)
            raise ProviderError("invalid fake-secret", status_code=403)
        return SimpleNamespace(list_models=lambda: ["a", "b"], generate_structured=generate)

    with pytest.raises(ValueError) as error:
        scan_models("gemini", "fake-secret", 5, ["detection"], "independent", factory,
                    lambda *args: None, sleep=lambda _: None)
    assert calls == ["a"]
    assert "fake-secret" not in str(error.value)


def test_resume_scan_skips_finished_models_and_retries_only_pending_429():
    from datetime import datetime, timedelta, timezone
    calls = []
    initial = {
        "a": {"ok": True, "attempts": 1},
        "b": {"ok": False, "attempts": 1, "status_code": 429, "retry_pending": True,
              "checked_at": (datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat(), "retry_delay": 60},
    }

    def factory(provider, key, model, timeout):
        def generate(prompt, schema):
            calls.append(model)
            return MockProvider().generate_structured(prompt, schema)
        return SimpleNamespace(list_models=lambda: ["a", "b", "c"], generate_structured=generate)

    results = scan_models("gemini", "fake", 5, ["detection"], "independent", factory,
                          lambda *args: None, sleep=lambda _: None, interval=0, initial_results=initial)
    assert calls == ["c", "b"]
    assert all(result["ok"] for result in results.values())
    assert results["b"]["attempts"] == 2


def test_background_scan_continues_on_checkpoint_permission_error(monkeypatch, tmp_path):
    import time
    from src.ui import model_scan, local_memory
    monkeypatch.setattr(local_memory, "STATE_DIR", tmp_path)
    monkeypatch.setattr(model_scan, "_SCANS", {})
    original = model_scan.scan_models
    monkeypatch.setattr(model_scan, "scan_models", lambda *args: original(*args, sleep=lambda _: None, interval=0))
    saves = []
    actual_save = local_memory.save_model_scan

    def fail_once(*args):
        saves.append(1)
        if len(saves) == 1:
            raise PermissionError(5, "Access is denied")
        return actual_save(*args)

    monkeypatch.setattr(local_memory, "save_model_scan", fail_once)
    factory = lambda *args: SimpleNamespace(list_models=lambda: ["a", "b"], generate_structured=MockProvider().generate_structured)
    context = "b" * 64
    model_scan.start_scan(context, "gemini", "fake", 5, ["detection"], "independent", factory)
    deadline = time.monotonic() + 3
    while model_scan.scan_snapshot(context)["running"] and time.monotonic() < deadline:
        time.sleep(0.01)
    snapshot = model_scan.scan_snapshot(context)
    assert not snapshot["running"] and snapshot["complete"]
    assert not snapshot["save_error"] and not snapshot["error"]
    assert len(local_memory.load_model_results(context)) == 2
    assert local_memory.load_model_scan(context)["complete"]


def test_full_scan_retries_temporary_server_error_once_without_skipping_other_models():
    from src.providers.base import ProviderError
    now = [0.0]
    calls = []
    def factory(provider, key, model, timeout):
        def generate(prompt, schema):
            attempts = calls.count(model)
            calls.append(model)
            if model == "a" and attempts == 0:
                raise ProviderError("server unavailable", status_code=503)
            return MockProvider().generate_structured(prompt, schema)
        return SimpleNamespace(list_models=lambda: ["a", "b"], generate_structured=generate)
    result = scan_models("gemini", "fake", 5, ["detection"], "independent", factory,
        lambda *args: None, clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds))
    assert calls == ["a", "b", "a"]
    assert result["a"]["ok"] and result["a"]["attempts"] == 2
