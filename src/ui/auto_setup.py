"""Bounded model discovery and labeling probes with quota-aware recovery."""
import re
import time

from src.labeling.engine import LabelingEngine
from src.providers.base import ProviderError, sanitize_error_message
from src.labeling.probe import SAMPLE, probe_labeling

def rank_models(names, preferred=""):
    excluded = ("image", "audio", "tts", "embedding", "live", "robotics", "computer-use", "research", "video", "veo", "imagen")
    names = [name for name in set(names) if not any(part in name.lower() for part in excluded)]
    def rank(name):
        version = re.search(r"(?:gemini-|gpt-)(\d+(?:\.\d+)?)", name)
        return (name != preferred, "preview" in name or "exp" in name,
                "flash" not in name and "mini" not in name, -float(version[1]) if version else 0, name)
    return sorted(names, key=rank)


def auto_find(provider, key, timeout, tasks, mode, factory, report, *, preferred="", known_results=None, sleep=time.sleep):
    """Return the first verified model; try at most six candidates, one quota retry."""
    if not tasks:
        raise ValueError("Chọn ít nhất một tác vụ ở bước 3.")
    report("Đang tải danh sách model…")
    names = factory(provider, key, "model-check", timeout).list_models()
    known_results = known_results or {}
    good = [name for name, result in known_results.items() if result.get("ok")]
    failed = {name for name, result in known_results.items() if not result.get("ok")}
    candidates = (rank_models(good, preferred) + rank_models(
        [name for name in names if name not in good and name not in failed], preferred
    ))[:6]
    results = {}
    if not candidates:
        raise ValueError("API không trả về model văn bản phù hợp. Kiểm tra quyền truy cập của project.")
    retry_used = False
    for index, name in enumerate(candidates):
        server_retry_used = False
        while True:
            report(f"Đang gọi thử {index + 1}/{len(candidates)}: {name}")
            try:
                probe_labeling(factory(provider, key, name, timeout), tasks, mode)
                results[name] = {"ok": True, "detail": "Gọi thử đúng luồng gán nhãn thành công."}
                report(f"Đã tìm được model chạy được: {name}")
                return name, results
            except Exception as exc:
                detail = sanitize_error_message(exc, key)
                results[name] = {"ok": False, "detail": detail}
                code = getattr(exc, "status_code", None)
                if code in {401, 403} or getattr(exc, "kind", None) == "authentication":
                    raise ProviderError("API key hoặc quyền project bị từ chối. " + detail, kind="authentication", status_code=code) from exc
                if code in {500, 502, 503, 504} and not server_retry_used:
                    server_retry_used = True
                    report(f"{name}: máy chủ lỗi tạm thời {code}. Chờ 3 giây rồi thử lại một lần…")
                    sleep(3)
                    continue
                if code == 429 or getattr(exc, "kind", None) == "rate_limit":
                    daily = any(part in detail.lower() for part in ("perday", "per_day", "daily", "limit: 0", "limit:0", "insufficient_quota"))
                    match = re.search(r"(?:retry in\s*|retrydelay[^0-9]*)(\d+(?:\.\d+)?)", detail, re.I)
                    wait = max(10, int(float(match[1])) + 1) if match else 60
                    if daily or retry_used or wait > 60:
                        report(f"{name}: quota chưa cho phép chạy. Chuyển sang model khác, không gọi lại model này.")
                        break
                    retry_used = True
                    for remaining in range(wait, 0, -1):
                        report(f"API giới hạn tốc độ (429). Chờ {remaining} giây rồi tự thử lại một lần…")
                        sleep(1)
                    continue
                report(f"Bỏ qua {name}: {detail}")
                # Unsupported model/schema (400/404), transient errors and invalid
                # outputs all move to the next candidate; no retry storm.
                break
    raise ProviderError("Chưa tìm được model chạy được sau tối đa 6 model. " + " | ".join(f"{name}: {result['detail']}" for name, result in results.items()), kind="configuration")
