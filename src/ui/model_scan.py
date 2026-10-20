"""Background account-wide model checks with incremental local checkpoints."""
from copy import deepcopy
from datetime import datetime, timezone
import re
import threading
import time

from src.labeling.probe import probe_labeling
from src.ui import local_memory
from src.runtime import API_ACTIVITY_LOCK

_SCANS = {}
_LOCK = API_ACTIVITY_LOCK


def scan_models(provider, key, timeout, tasks, mode, factory, update, *, sleep=time.sleep, clock=time.monotonic, interval=6, initial_results=None):
    """Probe every listed model and retry each 429 once after its delay."""
    if not tasks:
        raise ValueError("Chưa chọn tác vụ.")
    names = sorted(set(factory(provider, key, "model-check", timeout).list_models()))
    results = {name: deepcopy(value) for name, value in (initial_results or {}).items() if name in names}
    retries = []
    last_request = clock() - interval

    def probe(name, attempt):
        nonlocal last_request
        sleep(max(0, interval - (clock() - last_request)))
        last_request = clock()
        update(results, f"Đang gọi thử {name} · lần {attempt}", len(names))
        try:
            probe_labeling(factory(provider, key, name, timeout), tasks, mode)
            result = {"ok": True, "detail": "Đã gọi API và xác nhận kết quả hợp lệ.", "status_code": None}
        except Exception as exc:
            detail = str(exc.__cause__ or exc)
            detail = detail.replace(key, "[REDACTED]") if key else detail
            result = {"ok": False, "detail": detail[:6000], "status_code": getattr(exc, "status_code", None), "error_kind": getattr(exc, "kind", None), "retry_after_seconds": getattr(exc, "retry_after_seconds", None)}
        result.update(checked_at=datetime.now(timezone.utc).isoformat(), attempts=attempt, retry_pending=False)
        results[name] = result
        return result

    for name in names:
        if name in results:
            previous = results[name]
            if previous.get("retry_pending") and previous.get("attempts", 1) < 2:
                try:
                    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(previous["checked_at"])).total_seconds()
                except (KeyError, ValueError):
                    elapsed = 0
                retries.append((clock() + max(0, previous.get("retry_delay", 60) - elapsed), name))
            continue
        result = probe(name, 1)
        if result["status_code"] in {429, 500, 502, 503, 504} or result.get("error_kind") == "timeout":
            match = re.search(r"(?:retry in\s*|retrydelay[^0-9]*)(\d+(?:\.\d+)?)", result["detail"], re.I)
            hint = result.get("retry_after_seconds")
            requested_delay = hint if hint is not None else (float(match[1]) if match else 59)
            delay = max(60, requested_delay + 1) if result["status_code"] == 429 else 3
            result["retry_pending"] = True
            result["retry_delay"] = delay
            retries.append((clock() + delay, name))
        update(results, f"Đã thử {len(results)}/{len(names)} model", len(names))
        if result["status_code"] in {401, 403}:
            raise ValueError("Dừng vì API key hoặc quyền project bị từ chối. " + result["detail"])
    for index, (ready_at, name) in enumerate(retries):
        while clock() < ready_at:
            remaining = ready_at - clock()
            update(results, f"Chờ {int(remaining)+1} giây trước khi thử lại {name}", len(names))
            sleep(min(remaining, 1))
        probe(name, 2)
        update(results, f"Đã thử lại {index+1}/{len(retries)} model gặp lỗi tạm thời", len(names))
    return results


def scan_snapshot(context):
    with _LOCK:
        return deepcopy(_SCANS.get(context))


def any_scan_running():
    with _LOCK:
        return any(scan["running"] for scan in _SCANS.values())


def retry_save_scan(context):
    with _LOCK:
        snapshot = _SCANS[context]
        local_memory.save_model_scan(context, snapshot["results"], {key: value for key, value in snapshot.items() if key not in {"results", "save_error"}})
        _SCANS[context]["save_error"] = None


def start_scan(context, provider, key, timeout, tasks, mode, factory, *, resume=False):
    previous = local_memory.load_model_results(context)
    initial = previous if resume else {}
    with _LOCK:
        from src.ui.jobs import _has_active_worker
        if _has_active_worker():
            raise ValueError("Đang có job gán nhãn chạy. Hãy tạm dừng hoặc đợi job kết thúc trước khi kiểm tra model.")
        if any(scan["running"] for scan in _SCANS.values()):
            raise ValueError("Đang có một lượt kiểm tra model. Hãy đợi lượt đó hoàn tất.")
        _SCANS[context] = {"run_id": time.time_ns(), "running": True, "complete": False, "results": deepcopy(previous), "message": "Đang lấy danh sách model…", "total": 0, "error": None, "save_error": None}

    last_saved = None
    def update(results, message, total):
        nonlocal last_saved
        with _LOCK:
            _SCANS[context].update(results=deepcopy(results), message=message, total=total)
        if results == last_saved:
            return
        try:
            with _LOCK:
                metadata = {key: value for key, value in _SCANS[context].items() if key not in {"results", "save_error"}}
            local_memory.save_model_scan(context, results, metadata)
            last_saved = deepcopy(results)
            with _LOCK:
                _SCANS[context]["save_error"] = None
        except OSError as exc:
            with _LOCK:
                _SCANS[context]["save_error"] = str(exc).replace(key, "[REDACTED]")[:1000]

    def work():
        try:
            if initial:
                scan_models(provider, key, timeout, tasks, mode, factory, update, initial_results=initial)
            else:
                scan_models(provider, key, timeout, tasks, mode, factory, update)
            with _LOCK:
                _SCANS[context]["message"] = "Đã kiểm tra xong toàn bộ model."
                _SCANS[context]["complete"] = True
        except Exception as exc:
            with _LOCK:
                _SCANS[context]["error"] = str(exc).replace(key, "[REDACTED]")[:6000]
        finally:
            with _LOCK:
                _SCANS[context]["running"] = False
            try:
                retry_save_scan(context)
            except OSError as exc:
                with _LOCK:
                    _SCANS[context]["save_error"] = str(exc).replace(key, "[REDACTED]")[:1000]
    threading.Thread(target=work, name="model-scan", daemon=True).start()
