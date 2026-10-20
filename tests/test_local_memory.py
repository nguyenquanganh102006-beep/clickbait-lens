import json

import pytest

from src.ui import local_memory


def test_atomic_save_retries_temporary_windows_file_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(local_memory, "STATE_DIR", tmp_path)
    replace = local_memory.os.replace
    calls = []

    def locked(source, target):
        calls.append(target)
        if len(calls) < 3:
            raise PermissionError(5, "Access is denied")
        replace(source, target)

    monkeypatch.setattr(local_memory.os, "replace", locked)
    monkeypatch.setattr(local_memory.time, "sleep", lambda _: None)
    local_memory.save_model_results("a" * 64, {"model": {"ok": True}})
    assert len(calls) == 3
    assert local_memory.load_model_results("a" * 64)["model"]["ok"]
    assert len(list(tmp_path.iterdir())) == 1


def test_permanent_lock_preserves_previous_results(monkeypatch, tmp_path):
    monkeypatch.setattr(local_memory, "STATE_DIR", tmp_path)
    local_memory.save_model_results("a" * 64, {"old": {"ok": True}})

    def locked(*args):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(local_memory.os, "replace", locked)
    monkeypatch.setattr(local_memory.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        local_memory.save_model_results("a" * 64, {"new": {"ok": True}})
    assert "old" in local_memory.load_model_results("a" * 64)
    assert len(list(tmp_path.iterdir())) == 1


def test_checkpoint_metadata_and_legacy_model_files_are_compatible(monkeypatch, tmp_path):
    monkeypatch.setattr(local_memory, "STATE_DIR", tmp_path)
    context = "a" * 64
    legacy = tmp_path / f"models_{context}.json"
    legacy.write_text(json.dumps({"old": {"ok": True}}), encoding="utf-8")
    assert local_memory.load_model_results(context)["old"]["ok"]
    local_memory.save_model_scan(context, {"new": {"ok": True}}, {"running": True, "complete": False})
    local_memory.save_model_results(context, {"new": {"ok": False}})
    assert local_memory.load_model_scan(context)["running"]
    assert not local_memory.load_model_results(context)["new"]["ok"]


def test_mixed_jsonl_columns_round_trip_without_losing_source_values(monkeypatch, tmp_path):
    import pandas as pd
    monkeypatch.setattr(local_memory, "STATE_DIR", tmp_path)
    frame = pd.DataFrame({"title": ["Tin 1", "Tin 2", "Tin 3"], "mixed": [1, "001", None],
                          "nested": [{"key": 1}, ["a", "b"], None]})
    local_memory.save_dataset(frame, "mixed.jsonl", {"title": "title"})
    restored, metadata = local_memory.load_dataset()
    assert restored.loc[0, "mixed"] == 1
    assert restored.loc[1, "mixed"] == "001"
    assert restored.loc[0, "nested"] == {"key": 1}
    assert restored.loc[1, "nested"] == ["a", "b"]
    assert metadata["filename"] == "mixed.jsonl"


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows sharing violation regression")
def test_real_windows_file_lock_preserves_checkpoint_then_recovers(monkeypatch, tmp_path):
    import ctypes
    from ctypes import wintypes
    monkeypatch.setattr(local_memory, "STATE_DIR", tmp_path)
    context = "a" * 64
    local_memory.save_model_results(context, {"old": {"ok": True}})
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                   wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(tmp_path / f"models_{context}.json"), 0x80000000, 1, None, 3, 0, None)
    assert handle != ctypes.c_void_p(-1).value
    try:
        with pytest.raises(PermissionError):
            local_memory.save_model_results(context, {"new": {"ok": True}})
        assert "old" in local_memory.load_model_results(context)
    finally:
        kernel.CloseHandle(handle)
    local_memory.save_model_results(context, {"new": {"ok": True}})
    assert local_memory.load_model_results(context)["new"]["ok"]
