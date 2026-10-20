"""Local reusable settings and datasets; secrets are protected by Windows DPAPI."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from zipfile import ZipFile

import pandas as pd
from src.paths import STATE_DIR as APP_STATE_DIR
from src.jobs.storage import _json_clean

STATE_DIR = APP_STATE_DIR / "remembered"
_FILE_LOCK = threading.RLock()


class _Blob(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def _crypt(payload: bytes, *, decrypt: bool = False) -> bytes:
    if os.name != "nt":
        raise OSError("Lưu API key được mã hóa hiện chỉ hỗ trợ Windows.")
    buffer = ctypes.create_string_buffer(payload)
    source = _Blob(len(payload), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    target = _Blob()
    api = ctypes.WinDLL("crypt32", use_last_error=True)
    function = api.CryptUnprotectData if decrypt else api.CryptProtectData
    function.argtypes = [ctypes.POINTER(_Blob), ctypes.c_void_p, ctypes.c_void_p,
                         ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_Blob)]
    function.restype = wintypes.BOOL
    if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)):
        raise ctypes.WinError(ctypes.get_last_error())
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        kernel.LocalFree(target.data)


def _write(name: str, payload: bytes) -> None:
    with _FILE_LOCK:
        _write_locked(name, payload)


def _write_locked(name: str, payload: bytes) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=STATE_DIR)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        for attempt in range(8):
            try:
                os.replace(temporary, STATE_DIR / name)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(0.05 * 2 ** attempt, 0.5))
    finally:
        try:
            Path(temporary).unlink(missing_ok=True)
        except PermissionError:
            pass


def _read_optional(name: str) -> bytes | None:
    with _FILE_LOCK:
        try:
            return (STATE_DIR / name).read_bytes()
        except FileNotFoundError:
            return None


def save_provider(config: dict) -> None:
    _write("provider.dpapi", _crypt(json.dumps(config, ensure_ascii=False).encode("utf-8")))


def load_provider() -> dict | None:
    payload = _read_optional("provider.dpapi")
    return json.loads(_crypt(payload, decrypt=True)) if payload is not None else None


def forget_provider() -> None:
    (STATE_DIR / "provider.dpapi").unlink(missing_ok=True)


def save_dataset(frame: pd.DataFrame, filename: str, mapping: dict) -> None:
    parquet = BytesIO()
    try:
        frame.to_parquet(parquet, index=False)
        data_name, data_payload = "data.parquet", parquet.getvalue()
    except (ValueError, TypeError):
        # Mixed scalar/nested JSONL columns cannot always be represented by
        # Arrow. Preserve their values as ordinary JSON, never executable pickle.
        value = _json_clean(frame.to_dict(orient="split"))
        value["dtypes"] = [str(dtype) for dtype in frame.dtypes]
        data_name, data_payload = "data.json", json.dumps(value, ensure_ascii=False).encode("utf-8")
    bundle = BytesIO()
    with ZipFile(bundle, "w") as archive:
        archive.writestr(data_name, data_payload)
        archive.writestr("metadata.json", json.dumps({"filename": filename, "mapping": mapping}, ensure_ascii=False))
    _write("dataset.zip", bundle.getvalue())


def load_dataset() -> tuple[pd.DataFrame, dict] | None:
    payload = _read_optional("dataset.zip")
    if payload is None:
        return None
    with ZipFile(BytesIO(payload)) as archive:
        if "data.parquet" in archive.namelist():
            frame = pd.read_parquet(BytesIO(archive.read("data.parquet")))
        else:
            value = json.loads(archive.read("data.json"))
            frame = pd.DataFrame(value["data"], columns=value["columns"])
            for index, dtype in enumerate(value.get("dtypes", [])):
                try:
                    frame[frame.columns[index]] = frame.iloc[:, index].astype(dtype)
                except (TypeError, ValueError):
                    pass
        metadata = json.loads(archive.read("metadata.json"))
    return frame, metadata


def forget_dataset() -> None:
    (STATE_DIR / "dataset.zip").unlink(missing_ok=True)


def load_model_results(context: str) -> dict:
    if len(context) != 64 or any(char not in "0123456789abcdef" for char in context):
        raise ValueError("Invalid model context")
    payload = _read_optional(f"models_{context}.json")
    saved = json.loads(payload) if payload is not None else {}
    return saved["results"] if saved.get("format_version") == 1 else saved


def save_model_results(context: str, results: dict) -> None:
    if len(context) != 64 or any(char not in "0123456789abcdef" for char in context):
        raise ValueError("Invalid model context")
    with _FILE_LOCK:
        try:
            metadata = load_model_scan(context)
        except ValueError:
            metadata = None
        save_model_scan(context, results, metadata)


def load_model_scan(context: str) -> dict | None:
    if len(context) != 64 or any(char not in "0123456789abcdef" for char in context):
        raise ValueError("Invalid model context")
    payload = _read_optional(f"models_{context}.json")
    if payload is None:
        return None
    saved = json.loads(payload)
    return saved.get("scan") if saved.get("format_version") == 1 else None


def save_model_scan(context: str, results: dict, metadata: dict | None) -> None:
    if len(context) != 64 or any(char not in "0123456789abcdef" for char in context):
        raise ValueError("Invalid model context")
    payload = {"format_version": 1, "results": results, "scan": metadata}
    _write(f"models_{context}.json", json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def save_workflow(tasks: list[str], mode: str) -> None:
    _write("workflow.json", json.dumps({"tasks": tasks, "mode": mode}).encode("utf-8"))


def load_workflow() -> dict:
    payload = _read_optional("workflow.json")
    saved = json.loads(payload) if payload is not None else {}
    tasks = saved.get("tasks", ["detection"])
    mode = saved.get("mode", "independent")
    if not isinstance(tasks, list) or set(tasks) - {"detection", "type", "consistency"} or mode not in {"independent", "combined"}:
        raise ValueError("Cấu hình tác vụ đã lưu không hợp lệ.")
    return {"tasks": list(dict.fromkeys(tasks)), "mode": mode}
