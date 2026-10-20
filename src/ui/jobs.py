"""Job creation, dry-run, progress and lifecycle controls for Streamlit."""

from __future__ import annotations

import threading
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import streamlit as st
import yaml

from src.jobs.runner import JobRunner
from src.jobs.storage import DEFAULT_DB_PATH, JobStore
from src.labeling.prompts import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    load_taxonomy,
)
from src.providers.base import LLMProvider, ProviderError
from src.ui.provider import create_provider
from src.ui.model_checks import verified_for_config, check_current_model
from src.ui.selection import sync_job_selector, select_job_from_widget
from src.ui.widgets import number_input, selectbox
from src.ui.model_scan import any_scan_running
from src.runtime import API_ACTIVITY_LOCK


TASK_LABELS = {
    "detection": "Phát hiện clickbait (Clickbait Detection)",
    "type": "Phân loại loại clickbait (Clickbait Type)",
    "consistency": "Nhất quán tiêu đề–toàn văn (Title–Content Consistency)",
}
JOB_LABELS = {"dry_run": "Chạy thử", "batch": "Chạy toàn bộ"}
STATUS_LABELS = {
    "queued": "Đang chờ",
    "running": "Đang chạy",
    "paused": "Đã tạm dừng",
    "stopped": "Đã dừng",
    "budget_paused": "Đã tạm dừng",
    "auth_required": "Cần kiểm tra API key/quyền truy cập",
    "quota_paused": "Tạm dừng do giới hạn quota API",
    "model_required": "Cần kiểm tra lại model",
    "service_paused": "Tạm dừng do lỗi dịch vụ hoặc mạng",
    "completed": "Hoàn tất",
    "completed_with_errors": "Hoàn tất, có lỗi",
    "runner_failed": "Bộ chạy job gặp lỗi",
}

_ACTIVE_WORKERS: dict[str, threading.Thread] = {}
_ACTIVE_LOCK = API_ACTIVITY_LOCK


def _get_store() -> JobStore:
    return JobStore(DEFAULT_DB_PATH)


def _load_runtime_defaults() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
    with path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    return dict(values.get("jobs") or {})


def _runtime_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    settings = _load_runtime_defaults()
    settings.update(
        {
            key: config.get(key)
            for key in ("requests_per_minute", "concurrency", "max_retries")
        }
    )
    # Ignore cost settings from old sessions and persisted jobs.
    settings.update(
        {
            "budget_usd": None,
            "input_price_per_million": None,
            "output_price_per_million": None,
        }
    )
    settings["timeout_seconds"] = int(config.get("timeout_seconds", 60))
    return settings


def _configured_provider(config: Mapping[str, Any], tasks=None, mode=None) -> LLMProvider:
    if not verified_for_config(config, tasks, mode):
        raise ProviderError("Vào bước 1 và kiểm tra model với tác vụ hiện tại trước khi chạy. Kết quả kiểm tra chỉ có hiệu lực trong phiên này.", kind="configuration")
    provider = str(config.get("provider") or "")
    model = str(config.get("model") or "")
    if not provider or not model:
        raise ProviderError("Hãy lưu cấu hình nhà cung cấp và model trước khi tạo job.", kind="configuration")
    if provider != "mock" and not str(config.get("api_key") or "").strip():
        raise ProviderError("Chưa có API key cho kết nối này. Hãy nhập và lưu lại ở bước 1 trước khi chạy job.", kind="authentication")
    return create_provider(provider, str(config.get("api_key") or ""), model, int(config.get("timeout_seconds", 60)))


def _start_worker(store: JobStore, job_id: str, config: Mapping[str, Any]) -> None:
    job = store.get_job(job_id)
    if not job:
        raise ValueError("Không tìm thấy job.")
    provider = _configured_provider(config, job["tasks"], job["mode"])
    with _ACTIVE_LOCK:
        if any_scan_running():
            raise RuntimeError("Đang kiểm tra model trong nền. Hãy đợi hoàn tất trước khi chạy job.")
        active_ids = [key for key, thread in _ACTIVE_WORKERS.items() if thread.is_alive()]
        if active_ids:
            raise RuntimeError("Một job khác đang chạy trong ứng dụng này. Hãy đợi hoặc tạm dừng job đó.")

        def work() -> None:
            try:
                JobRunner(store, provider, _runtime_settings(config)).run(job_id)
            except Exception as exc:
                store.set_job_status(job_id, "runner_failed", f"Lỗi runner: {type(exc).__name__}")
            finally:
                with _ACTIVE_LOCK:
                    _ACTIVE_WORKERS.pop(job_id, None)

        thread = threading.Thread(target=work, name=f"job-{job_id[:8]}", daemon=True)
        _ACTIVE_WORKERS[job_id] = thread
        thread.start()


def _has_active_worker() -> bool:
    with _ACTIVE_LOCK:
        return any(thread.is_alive() for thread in _ACTIVE_WORKERS.values())


def _create_job(
    store: JobStore, records: Sequence[Mapping[str, Any]], tasks: Sequence[str],
    mode: str, config: Mapping[str, Any], *, job_kind: str,
) -> str:
    with _ACTIVE_LOCK:
        if _has_active_worker():
            raise RuntimeError("Đang có job chạy. Hãy đợi hoặc tạm dừng trước khi tạo job khác.")
        return _create_job_locked(store, records, tasks, mode, config, job_kind=job_kind)


def _create_job_locked(
    store: JobStore,
    records: Sequence[Mapping[str, Any]],
    tasks: Sequence[str],
    mode: str,
    config: Mapping[str, Any],
    *,
    job_kind: str,
) -> str:
    provider = _configured_provider(config, tasks, mode)
    taxonomy = load_taxonomy()
    runtime = _runtime_settings(config)
    settings = {
        **runtime,
        "job_kind": job_kind,
        "source_dataset_rows": len(st.session_state.get("dataset_records", [])),
        "dry_run_samples": len(records) if job_kind == "dry_run" else None,
    }
    job_id = store.create_job(
        records,
        tasks,
        provider=provider.provider_name,
        model=provider.model,
        mode=mode,
        prompt_version=PROMPT_VERSION,
        taxonomy_version=str(taxonomy["version"]),
        schema_version=SCHEMA_VERSION,
        settings=settings,
    )
    store.set_job_status(job_id, "queued")
    try:
        _start_worker(store, job_id, config)
    except (ProviderError, RuntimeError) as exc:
        store.set_job_status(job_id, "runner_failed", str(exc))
        raise
    return job_id


def _job_title(job: Mapping[str, Any]) -> str:
    created = str(job.get("created_at", ""))[:19].replace("T", " ")
    kind = JOB_LABELS.get((job.get("settings") or {}).get("job_kind"), "Job")
    status = STATUS_LABELS.get(str(job.get("status")), str(job.get("status")))
    return f"{created} · {kind} · {job.get('provider')}/{job.get('model')} · {status} · {str(job.get('job_id'))[:8]}"


@st.fragment(run_every=3)
def _live_job_progress(store: JobStore, job_id: str, rendered_status: str, worker_active: bool) -> None:
    latest = store.get_job(job_id)
    if (latest and latest["status"] != rendered_status) or _has_active_worker() != worker_active:
        st.rerun(scope="app")
    progress = store.get_progress(job_id)
    total = max(1, int(progress.get("total", 0)))
    completed = int(progress.get("completed", 0))
    st.progress(min(1.0, completed / total), text=f"{completed:,} / {progress.get('total', 0):,} tác vụ đã hoàn tất")

    cols = st.columns(5)
    cols[0].metric("Thành công", progress.get("success", 0))
    cols[1].metric("Thiếu dữ liệu", progress.get("insufficient_data", 0))
    cols[2].metric("Tạm không kết luận (abstain)", progress.get("abstain", 0))
    cols[3].metric("Lỗi", progress.get("error", 0))
    cols[4].metric("Đang chờ/chạy", progress.get("pending", 0) + progress.get("running", 0))
    if rendered_status in {"queued", "running"}:
        st.caption(
            f"Hiện tại: {progress.get('running', 0):,} tác vụ đang xử lý · "
            f"{progress.get('pending', 0):,} tác vụ chờ đến lượt. "
            "Nếu chưa có kết quả mới, hệ thống có thể đang chờ API hoặc giới hạn tốc độ."
        )


def _show_job_progress(store: JobStore, job: Mapping[str, Any]) -> None:
    job_id = str(job["job_id"])
    _live_job_progress(store, job_id, str(job.get("status")), _has_active_worker())
    errors = store.get_items(job_id, status="error", limit=5)
    error_count = store.get_progress(job_id).get("error", 0)
    if errors:
        with st.expander(f"Xem lỗi và cách xử lý ({error_count:,} tác vụ)", expanded=True):
            st.caption("Sửa kết nối hoặc cấu hình trước, sau đó bấm Thử lại các tác vụ lỗi. Các tác vụ đã thành công được giữ lại.")
            for item in errors[:5]:
                st.error(str(item.get("error_message") or item.get("error_kind") or "Không có chi tiết lỗi."))
            if error_count > 5:
                st.caption("Đang hiển thị 5 lỗi đầu. Xem các lỗi còn lại trong bước 5 hoặc tệp errors.csv.")
    task_counts, token_totals = store.get_task_summary(job_id)
    if task_counts:
        rows = [
            {"Tác vụ": TASK_LABELS.get(task, task), **dict(counts)}
            for task, counts in task_counts.items()
        ]
        st.dataframe(rows, width="stretch", hide_index=True)
    if token_totals:
        st.caption(
            "Token API trả về: "
            + " · ".join(f"{key.replace('_tokens', '')}: {value:,}" for key, value in token_totals.items())
        )


def _resume_job(store: JobStore, job: Mapping[str, Any], config: Mapping[str, Any], *, retry_failed=False) -> None:
    if not config or config.get("provider") != job.get("provider") or config.get("model") != job.get("model"):
        raise ProviderError("Để tiếp tục, hãy lưu đúng nhà cung cấp và model đã dùng cho job này.", kind="configuration")
    _configured_provider(config, job["tasks"], job["mode"])
    with _ACTIVE_LOCK:
        if _has_active_worker():
            raise RuntimeError("Hãy đợi job đang chạy kết thúc trước khi tiếp tục.")
        settings = _runtime_settings(config)
        store.update_job_settings(job["job_id"], settings)
        store.resume(job["job_id"])
        if retry_failed:
            store.retry_failed(job["job_id"])
        _start_worker(store, str(job["job_id"]), config)


def render_jobs_tab() -> None:
    st.subheader("Chạy, tạm dừng và tiếp tục job")
    store = _get_store()
    records = st.session_state.get("dataset_records", [])
    tasks = list(st.session_state.get("selected_tasks", []))
    mode = str(st.session_state.get("labeling_mode", "independent"))
    config = st.session_state.get("provider_config", {})

    with st.expander("Trước khi chạy", expanded=not bool(records)):
        st.write("Chọn nhà cung cấp, nạp dữ liệu, ánh xạ cột và chọn tác vụ ở ba tab đầu.")
        if records:
            st.success(f"Đã sẵn sàng {len(records):,} bản ghi; tác vụ: {', '.join(TASK_LABELS.get(t, t) for t in tasks) or 'chưa chọn'}.")
        if config:
            label = "Mô phỏng ngoại tuyến" if config.get("provider") == "mock" else f"{config.get('provider')} / {config.get('model')}"
            st.caption(f"Cấu hình đã lưu trong phiên: {label}.")
        else:
            st.caption("Chưa lưu hoặc khôi phục kết nối AI.")

    st.markdown("#### Kiểm tra model trước khi chạy")
    st.caption("Gọi trực tiếp model đang lưu bằng bài mẫu và đúng tác vụ hiện tại. Không gửi bộ dữ liệu của bạn; có dùng quota API.")
    if st.button("Kiểm tra model đang chọn ngay bây giờ", key="check_current_model", disabled=not bool(config and tasks) or _has_active_worker() or any_scan_running()):
        result = check_current_model(config)
        if result["ok"]:
            st.success(result["detail"])
        else:
            st.error(result["detail"])
    missing = []
    if not config:
        missing.append("Bước 1: lưu kết nối AI và model.")
    elif not verified_for_config(config):
        missing.append("Bước 1: kiểm tra lại model cho key và tác vụ của phiên này, rồi lưu cấu hình.")
    if not records:
        missing.append("Bước 2: nạp dữ liệu và kiểm tra ánh xạ cột.")
    elif not any(record.get("title") for record in records):
        missing.append("Bước 2: chọn cột Tiêu đề có dữ liệu.")
    if st.session_state.get("dataset_diagnostics", {}).get("duplicate_ids"):
        missing.append("Bước 2: cột ID có giá trị trùng. Bỏ ánh xạ ID để ứng dụng tự tạo ID duy nhất hoặc sửa cột nguồn.")
    if not tasks:
        missing.append("Bước 3: chọn ít nhất một tác vụ.")
    if any_scan_running():
        missing.append("Đang kiểm tra model trong nền. Đợi lượt quét hoàn tất trước khi chạy để tránh gọi API đồng thời.")
    for message in missing:
        st.warning(message)
    can_create = not missing and not _has_active_worker()
    if can_create:
        st.caption(f"Sẽ xử lý {len(records):,} bản ghi × {len(tasks)} tác vụ. Nên chạy thử và kiểm tra kết quả trước khi chạy toàn bộ.")
    if config.get("provider") in {"gemini", "openai"}:
        st.warning("Khi chạy thử hoặc xử lý toàn bộ, nội dung đã ánh xạ sẽ được gửi tới API đã chọn.")
    elif config.get("provider") == "mock":
        st.info("Chế độ mô phỏng chạy ngoại tuyến, không gửi dữ liệu đến API.")

    if _has_active_worker():
        st.info("Đang có một job chạy trong ứng dụng này. Tạm dừng hoặc đợi job đó trước khi khởi tạo job khác.")

    sample_max = min(10, len(records)) if records else 1
    sample_default = min(5, sample_max)
    if st.session_state.get("dry_run_sample_size", sample_default) > sample_max:
        st.session_state["dry_run_sample_size"] = sample_default
    sample_size = number_input(
        "Số bản ghi cho chạy thử (5–10 nếu bộ dữ liệu có đủ)",
        min_value=1,
        max_value=sample_max,
        value=sample_default,
        step=1,
        key="dry_run_sample_size",
        disabled=not bool(records),
    )
    create_cols = st.columns(2)
    with create_cols[0]:
        if st.button("Tạo và chạy thử", type="primary", disabled=not can_create, key="create_dry_run"):
            try:
                job_id = _create_job(store, records[: int(sample_size)], tasks, mode, config, job_kind="dry_run")
                st.session_state["selected_job_id"] = job_id
                st.session_state["job_selector"] = job_id
                st.session_state["results_job_selector"] = job_id
                st.success(f"Đã bắt đầu chạy thử {job_id[:8]}.")
                st.rerun()
            except (ProviderError, ValueError, RuntimeError) as exc:
                st.error(str(exc))
    with create_cols[1]:
        if st.button("Tạo và chạy toàn bộ", disabled=not can_create, key="create_full_job"):
            try:
                job_id = _create_job(store, records, tasks, mode, config, job_kind="batch")
                st.session_state["selected_job_id"] = job_id
                st.session_state["job_selector"] = job_id
                st.session_state["results_job_selector"] = job_id
                st.success(f"Đã bắt đầu job {job_id[:8]}.")
                st.rerun()
            except (ProviderError, ValueError, RuntimeError) as exc:
                st.error(str(exc))

    st.divider()
    st.markdown("#### Lịch sử job")
    refresh_col, selector_col = st.columns([1, 5])
    with refresh_col:
        if st.button("Làm mới", key="refresh_jobs"):
            st.rerun()
    jobs = store.list_jobs(limit=50)
    if not jobs:
        st.caption("Chưa có job nào.")
        return
    by_id = {str(job["job_id"]): job for job in jobs}
    ids = list(by_id)
    current_id = st.session_state.get("selected_job_id")
    selected_id = current_id if current_id in by_id else ids[0]
    selected_id = sync_job_selector("job_selector", ids)
    with selector_col:
        selected_id = selectbox(
            "Chọn job",
            ids,
            index=ids.index(selected_id),
            format_func=lambda item_id: _job_title(by_id[item_id]),
            key="job_selector",
            on_change=select_job_from_widget, args=("job_selector",),
        )
    st.session_state["selected_job_id"] = selected_id
    job = by_id[selected_id]
    status = str(job.get("status"))
    st.markdown(f"**Trạng thái:** {STATUS_LABELS.get(status, status)} · **Job ID:** `{selected_id}`")
    st.caption(
        f"{JOB_LABELS.get((job.get('settings') or {}).get('job_kind'), 'Job')} · "
        f"{job.get('provider')} / {job.get('model')} · {job.get('mode')} · "
        f"Tác vụ: {', '.join(TASK_LABELS.get(task, task) for task in job.get('tasks', []))}"
    )
    if job.get("error_summary"):
        st.warning(str(job["error_summary"]))
    if status in {"queued", "running"}:
        st.info("Tiến độ và trạng thái tự cập nhật mỗi 3 giây. Bạn có thể chuyển bước trong lúc chờ.")
    elif status == "completed":
        st.success("Lượt chạy đã kết thúc. Chọn bước 5 để rà soát và tải kết quả.")
    elif status == "completed_with_errors":
        st.warning("Lượt chạy đã kết thúc nhưng có lỗi. Xem nguyên nhân bên dưới trước khi thử lại.")
    _show_job_progress(store, job)

    action_cols = st.columns(4)
    if status in {"queued", "running"}:
        with action_cols[0]:
            if st.button("Tạm dừng", key="pause_job"):
                store.set_job_status(selected_id, "paused")
                st.rerun()
        with action_cols[1]:
            if st.button("Dừng job", key="stop_job"):
                store.set_job_status(selected_id, "stopped")
                st.rerun()

    current_config_matches = bool(
        config
        and config.get("provider") == job.get("provider")
        and config.get("model") == job.get("model")
    )
    worker_active = _has_active_worker() or any_scan_running()
    can_resume = not worker_active and (
        status in {"queued", "paused", "stopped", "budget_paused", "auth_required", "runner_failed", "quota_paused", "model_required", "service_paused"}
        or status == "running"
    )
    if can_resume:
        with action_cols[2]:
            if st.button("Tiếp tục phần còn lại", disabled=not current_config_matches, key="resume_job"):
                try:
                    _resume_job(store, job, config)
                    st.rerun()
                except (ProviderError, ValueError, RuntimeError) as exc:
                    st.error(str(exc))
        if not current_config_matches:
            st.caption("Để tiếp tục, hãy lưu đúng nhà cung cấp và model của job ở bước 1. Nếu đổi model, tạo lượt chạy mới.")

    progress = store.get_progress(selected_id)
    with action_cols[3]:
        if st.button(
            "Thử lại các tác vụ lỗi",
            disabled=not bool(progress.get("error")) or not current_config_matches or _has_active_worker() or any_scan_running(),
            key="retry_failed_job",
        ):
            try:
                _resume_job(store, job, config, retry_failed=True)
                st.rerun()
            except (ProviderError, RuntimeError) as exc:
                st.error(str(exc))
