"""Verify models against the exact labeling workflow, scoped to this session."""
import hashlib
import json
import time

import streamlit as st

from src.labeling.engine import LabelingEngine
from src.labeling.probe import probe_labeling
from src.labeling.prompts import PROMPT_VERSION, SCHEMA_VERSION
from src.providers.base import sanitize_error_message
from src.ui.activity import activity
from src.ui.auto_setup import rank_models
from src.ui import local_memory
from src.ui.model_scan import start_scan, scan_snapshot, retry_save_scan
from src.ui.auto_setup import SAMPLE
from datetime import datetime, timezone


def context_results(context):
    all_results = st.session_state.setdefault("model_verifications", {})
    if context not in all_results:
        try:
            all_results[context] = local_memory.load_model_results(context)
        except (OSError, ValueError):
            all_results[context] = {}
            st.warning("Không đọc được danh sách model đã lưu. Hãy kiểm tra lại model.")
    return all_results[context]


@st.fragment(run_every=2)
def render_scan_progress(context):
    snapshot = scan_snapshot(context)
    if not snapshot:
        return
    passed = sum(result.get("ok", False) for result in snapshot["results"].values())
    pending = sum(result.get("retry_pending", False) for result in snapshot["results"].values())
    st.info(snapshot["message"])
    st.caption(f"Đã thử {len(snapshot['results'])}/{snapshot['total']} model · Qua {passed} · Chờ thử lại: {pending}. Có thể chuyển bước trong khi chạy nền.")
    if snapshot["error"]:
        st.error(snapshot["error"])
    if snapshot.get("save_error"):
        st.warning("Windows đang chặn lưu tệp. Lượt quét vẫn tiếp tục; kết quả hiện vẫn giữ trong bộ nhớ. Đừng đóng ứng dụng trước khi lưu thành công.")
        if st.button("Thử lưu lại kết quả hiện có", key=f"retry_save_{context}"):
            try:
                retry_save_scan(context)
                st.success("Đã lưu kết quả hiện có trên máy.")
            except OSError:
                st.error("Tệp vẫn đang bị khóa hoặc không có quyền ghi. Đóng ứng dụng đang mở tệp và thử lưu lại.")
    if not snapshot["running"] and st.session_state.get("finished_model_scan") != snapshot["run_id"]:
        st.session_state.setdefault("model_verifications", {})[context] = snapshot["results"]
        usable = [name for name, result in snapshot["results"].items() if result.get("ok")]
        config = st.session_state.get(f"scan_config_{context}")
        if usable and config:
            preferred = st.session_state.get("provider_config", {}).get("model", "")
            ranked = rank_models(usable, preferred)
            chosen = preferred if preferred in usable else (ranked or sorted(usable))[0]
            config = {**config, "model": chosen}
            st.session_state["provider_config"] = config
            st.session_state["selected_model"] = chosen
            if st.session_state.get("remember_provider", True):
                try:
                    local_memory.save_provider(config)
                except OSError:
                    st.warning("Đã chọn model trong phiên nhưng không lưu được kết nối trên máy.")
        st.session_state["finished_model_scan"] = snapshot["run_id"]
        st.rerun(scope="app")


def check_current_model(config):
    """Explicit fresh API call from the jobs page; update the persisted verdict."""
    tasks = list(st.session_state.get("selected_tasks", ["detection"]))
    mode = st.session_state.get("labeling_mode", "independent")
    context = check_context(config["provider"], config.get("api_key", ""), tasks, mode)
    results = context_results(context)
    from src.ui.provider import create_provider
    try:
        with activity(f"Kiểm tra trực tiếp model đang chọn: {config['model']}"):
            probe_labeling(create_provider(config["provider"], config.get("api_key", ""), config["model"], int(config.get("timeout_seconds", 60))), tasks, mode)
        result = {"ok": True, "detail": "Model vừa gọi thử thành công với tác vụ hiện tại."}
    except Exception as exc:
        result = {"ok": False, "detail": sanitize_error_message(exc, config.get("api_key", ""))}
    result["checked_at"] = datetime.now(timezone.utc).isoformat()
    results[config["model"]] = result
    try:
        local_memory.save_model_results(context, results)
    except OSError:
        st.warning("Đã kiểm tra nhưng chưa lưu được kết quả trên máy.")
    return result


def check_context(provider, key, tasks, mode):
    return hashlib.sha256(json.dumps(
        [provider, key.strip(), sorted(tasks), mode, PROMPT_VERSION, SCHEMA_VERSION]
    ).encode()).hexdigest()


def verified_for_config(config, tasks=None, mode=None):
    if config.get("provider") == "mock":
        return True
    context = check_context(config.get("provider"), config.get("api_key", ""),
                            tasks if tasks is not None else st.session_state.get("selected_tasks", ["detection"]),
                            mode if mode is not None else st.session_state.get("labeling_mode", "independent"))
    result = context_results(context).get(config.get("model"), {})
    return result.get("ok", False)


def render_model_checks(provider, key, timeout, candidates, create_provider):
    from src.ui.jobs import _has_active_worker
    job_running = _has_active_worker()
    tasks = list(st.session_state.get("selected_tasks", ["detection"]))
    mode = st.session_state.get("labeling_mode", "independent")
    context = check_context(provider, key, tasks, mode)
    all_results = st.session_state.setdefault("model_verifications", {})
    results = context_results(context)
    st.markdown("#### Kiểm tra model cho phiên gán nhãn")
    snapshot = scan_snapshot(context)
    scanning = bool(snapshot and snapshot["running"])
    try:
        scan_metadata = local_memory.load_model_scan(context)
    except (OSError, ValueError):
        scan_metadata = None
    interrupted = bool(scan_metadata and scan_metadata.get("total", 0) > 0 and not scan_metadata.get("complete") and not scanning)
    st.caption("Kết quả được lưu trên máy theo API key, tác vụ và chế độ chạy. Lần mở sau dùng lại, không cần gọi API để kiểm tra lại.")
    if interrupted:
        st.warning("Lượt kiểm tra trước chưa hoàn tất. Nút bên dưới sẽ tiếp tục các model còn lại và lượt thử lại 429; không gọi lại những model đã xong.")
    if job_running:
        st.info("Đang có job gán nhãn chạy. Kiểm tra model sẽ được bật lại sau khi job kết thúc hoặc tạm dừng.")
    if st.button("Tiếp tục kiểm tra model còn lại" if interrupted else "Tự kiểm tra toàn bộ model và lưu kết quả", type="primary", width="stretch",
                 key="auto_setup_provider", disabled=job_running or scanning or not bool(key.strip() and tasks)):
        try:
            st.session_state[f"scan_config_{context}"] = {
                "provider": provider, "api_key": key, "timeout_seconds": timeout,
                "requests_per_minute": int(st.session_state.get("provider_rpm", 10)),
                "concurrency": int(st.session_state.get("provider_concurrency", 1)),
                "max_retries": int(st.session_state.get("provider_retries", 3)),
            }
            start_scan(context, provider, key, timeout, tasks, mode, create_provider, resume=interrupted)
            scanning = True
        except ValueError as exc:
            st.warning(str(exc))
    st.caption("Tự chạy hết danh sách trong nền, không dừng ở model đầu tiên thành công. Cách nhau 6 giây; gặp 429 thì chờ ít nhất 60 giây rồi thử lại một lần. Kết quả được lưu sau từng model.")
    render_scan_progress(context)
    with st.expander("Kiểm tra thủ công / thử model khác", expanded=False):
        button_area = st.container()
        st.caption(f"Tác vụ: {', '.join(tasks) or 'chưa chọn'} · Chế độ: {mode}. Đổi tác vụ ở bước 3 sẽ cần kiểm tra lại.")
        st.info("Mỗi model được chọn sẽ gọi API bằng bài mẫu có sẵn, theo đúng prompt và schema gán nhãn. Không gửi dữ liệu của bạn. Các lệnh gọi có thể dùng quota và phát sinh phí API.")
        candidates = sorted(set(candidates))
        candidate_version = hashlib.sha256(json.dumps(candidates).encode()).hexdigest()[:12]
        selected = st.multiselect("Model cần kiểm tra (tối đa 3 model mỗi lượt)", candidates,
                                  default=candidates[:1], max_selections=3,
                                  key=f"check_candidates_{context}_{candidate_version}")
        quota_key = hashlib.sha256(f"{provider}:{key.strip()}".encode()).hexdigest()
        cooldowns = st.session_state.setdefault("model_check_cooldowns", {})
        remaining = max(0, int(cooldowns.get(quota_key, 0) - time.time()) + 1)
        with button_area:
            check_clicked = st.button("Kiểm tra khả năng gán nhãn", type="primary", width="stretch",
                                      disabled=job_running or scanning or remaining > 0 or not bool(key.strip() and selected and tasks), key="verify_label_models")
            if remaining:
                st.warning(f"API vừa trả 429. Tạm khóa kiểm tra còn khoảng {remaining} giây để tránh gọi dồn. Sau đó tải lại trang để thử; quota ngày hoặc billing có thể cần xử lý riêng.")
            if not key.strip():
                st.caption("Nhập API key để bật kiểm tra.")
            elif not selected:
                st.caption("Chưa có model để kiểm tra. Tải lại danh sách hoặc nhập model ID ở phần bên dưới.")
            elif not tasks:
                st.caption("Chọn ít nhất một tác vụ ở bước 3 để bật kiểm tra.")
        if check_clicked:
            quota_blocked = False
            progress = st.progress(0, text="Chuẩn bị kiểm tra model…")
            record = {
                "record_id": "model-check", "title": "Thư viện thành phố mở cửa trở lại sau sửa chữa",
                "content": "Thư viện thành phố đã mở cửa trở lại vào sáng thứ Hai sau hai tháng sửa chữa. "
                           "Bạn đọc có thể đến mượn sách từ 8 giờ đến 17 giờ hằng ngày. Khu đọc sách được bổ sung bàn ghế và hệ thống chiếu sáng.",
            }
            for index, name in enumerate(selected):
                progress.progress(index / len(selected), text=f"Đang kiểm tra {index + 1}/{len(selected)}: {name}")
                try:
                    with activity(f"Gọi thử gán nhãn: {name}"):
                        probe_labeling(create_provider(provider, key, name, timeout), tasks, mode)
                    results[name] = {"ok": True, "detail": "Đã gọi API và kiểm tra kết quả đúng schema."}
                except Exception as exc:
                    results[name] = {"ok": False, "detail": sanitize_error_message(exc, key)}
                    if getattr(exc, "status_code", None) == 429 or getattr(exc, "kind", None) == "rate_limit":
                        cooldowns[quota_key] = time.time() + 60
                        st.session_state["model_check_cooldowns"] = cooldowns
                        quota_blocked = True
                        st.warning("Dừng kiểm tra vì API trả 429: giới hạn tốc độ hoặc quota. Các model còn lại chưa được gọi. Chờ ít nhất 60 giây; nếu vẫn lỗi, kiểm tra quota/billing của project.")
                st.session_state["model_verifications"] = all_results
                local_memory.save_model_results(context, results)
                if quota_blocked:
                    break
            progress.progress((index + 1) / len(selected), text="Đã dừng do giới hạn API (429)." if quota_blocked else "Đã kiểm tra xong các model đã chọn.")
    usable = sorted(name for name, result in results.items() if result["ok"])
    failed = [{"Model": name, "Lý do": result["detail"]} for name, result in results.items() if not result["ok"]]
    if failed:
        with st.expander(f"Model chưa vượt qua kiểm tra ({len(failed)})", expanded=True):
            st.dataframe(failed, hide_index=True, width="stretch")
    if not usable:
        st.info("Bấm Tự tìm model chạy được và lưu kết nối để thiết lập. Chỉ model vượt qua gọi thử mới được dùng.")
        return ""
    st.success(f"{len(usable)} model đã vượt qua kiểm tra và được nhớ cho cấu hình gán nhãn này.")
    if st.session_state.get("selected_model") not in usable:
        saved = st.session_state.get("provider_config", {}).get("model")
        st.session_state["selected_model"] = saved if saved in usable else usable[0]
    return st.selectbox("Model đã kiểm tra — dùng để gán nhãn", usable, key="selected_model")
