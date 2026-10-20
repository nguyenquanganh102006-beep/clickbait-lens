from types import SimpleNamespace
from src.jobs import runner


def test_limiter_spaces_requests_without_initial_burst_and_supports_cancel(monkeypatch):
    now = [0.0]
    def sleep(seconds):
        now[0] += seconds
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=sleep))
    limiter = runner._RateLimiter(10)
    assert limiter.wait_for_slot(lambda: True)
    assert now[0] == 0
    assert limiter.wait_for_slot(lambda: True)
    assert now[0] >= 6
    assert limiter.wait_for_slot(lambda: True)
    assert now[0] >= 12
    assert not limiter.wait_for_slot(lambda: False)
