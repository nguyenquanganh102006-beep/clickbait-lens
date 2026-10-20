"""Results review, summaries and local export controls."""

from __future__ import annotations

import json
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st

from src.data.exporter import build_export_files, review_reasons
from src.jobs.storage import DEFAULT_DB_PATH, JobStore
from src.ui.activity import activity
from src.ui.selection import sync_job_selector, select_job_from_widget
from src.ui.widgets import selectbox
from src.paths import OUTPUT_DIR


TASK_LABELS = {
    "detection": "Phát hiện clickbait (Clickbait Detection)",
    "type": "Phân loại loại clickbait (Clickbait Type)",
    "consistency": "Nhất quán tiêu đề–toàn văn (Title–Content Consistency)",
}


def _store() -> JobStore:
    return JobStore(DEFAULT_DB_PATH)


def _display_label(task: str, output: Mapping[str, Any]) -> str:
    if task == "type":
        labels = output.get("labels") or []
        return ", ".join(str(label) for label in labels) if labels else "(không có nhãn)"
    label = output.get("label")
    return "" if label is None else str(label)


def _review_rows(
    job: Mapping[str, Any], records: Sequence[Mapping[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        canonical = record.get("input") or {}
        original = record.get("original") or {}
        tasks = record.get("tasks") or {}
        reasons_by_task = {item["task"]: item for item in review_reasons(tasks, threshold)}
        gold_fields = {
            str(key): value
            for key, value in original.items()
            if str(key).casefold() in {"label", "final_label", "gold_label", "ground_truth"}
        }
        for task, result in tasks.items():
            output = result.get("output") or {}
            review = reasons_by_task.get(task, {})
            rows.append(
                {
                    "record_id": record.get("record_id"),
                    "task": task,
                    "task_name": TASK_LABELS.get(task, task),
                    "status": result.get("status", "pending"),
                    "label": _display_label(task, output),
                    "confidence": output.get("confidence"),
                    "reason": output.get("reason"),
                    "evidence": output.get("evidence"),
                    "review_reason": review.get("review_reason"),
                    "input_truncated": bool(result.get("input_truncated")),
                    "attempts": result.get("attempts", 0),
                    "error_kind": result.get("error_kind"),
                    "error_message": result.get("error_message"),
                    "source": canonical.get("source", original.get("source")),
                    "url": canonical.get("url", original.get("url")),
                    "title": canonical.get("title", original.get("title")),
                    "gold_labels": json.dumps(gold_fields, ensure_ascii=False) if gold_fields else "",
                }
            )
    return rows


def _save_exports(job_id: str, exports: Mapping[str, bytes]) -> Path:
    target = OUTPUT_DIR / job_id
    target.mkdir(parents=True, exist_ok=True)
    for filename, content in exports.items():
        (target / filename).write_bytes(content)
    return target


def render_results_tab() -> None:
    st.subheader("Rà soát và xuất kết quả")
    st.caption("Chọn lượt chạy, kiểm tra nhãn và lỗi, rồi tải bộ kết quả. Bộ tải xuống gồm toàn bộ kết quả của lượt chạy, không phụ thuộc bộ lọc bên dưới.")
    store = _store()
    jobs = store.list_jobs(limit=50)
    if not jobs:
        st.info("Chưa có job. Hãy tạo lượt chạy thử hoặc job toàn bộ trong tab Chạy job trước.")
        return

    by_id = {str(job["job_id"]): job for job in jobs}
    selected_id = st.session_state.get("selected_job_id")
    if selected_id not in by_id:
        selected_id = next(iter(by_id))
    selected_id = sync_job_selector("results_job_selector", list(by_id))
    job_id = selectbox(
        "Job cần xem",
        list(by_id),
        index=list(by_id).index(selected_id),
        format_func=lambda value: (
            f"{str(by_id[value].get('created_at', ''))[:19].replace('T', ' ')} · "
            f"{by_id[value].get('provider')}/{by_id[value].get('model')} · {value[:8]}"
        ),
        key="results_job_selector",
        on_change=select_job_from_widget, args=("results_job_selector",),
    )
    st.session_state["selected_job_id"] = job_id
    job = by_id[job_id]
    if st.session_state.get("results_filter_job_id") != job_id:
        for key in list(st.session_state):
            if key.startswith("result_"):
                st.session_state.pop(key, None)
        st.session_state["results_filter_job_id"] = job_id
    if st.button("Làm mới kết quả", key="refresh_results"):
        st.rerun()
    signature = store.get_result_signature(job_id)
    snapshot_key = (str(store.db_path), job_id, signature)
    snapshot = st.session_state.get("results_snapshot")
    if not snapshot or snapshot["key"] != snapshot_key:
        with st.spinner("Đang đọc kết quả đã lưu…"):
            snapshot = {"key": snapshot_key, "records": store.get_record_results(job_id)}
        st.session_state["results_snapshot"] = snapshot
    records = snapshot["records"]
    if job.get("status") not in {"completed", "completed_with_errors"}:
        st.info("Đây là kết quả tạm thời của job chưa hoàn tất. Bấm Làm mới kết quả để lấy tiến độ và dữ liệu mới nhất.")
    threshold = st.slider(
        "Ngưỡng confidence cần đưa vào hàng đợi rà soát",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        key="review_confidence_threshold",
        help="Confidence do model tự đánh giá, chưa được hiệu chỉnh thành xác suất đúng.",
    )
    progress = store.get_progress(job_id)
    review_count = sum(bool(review_reasons(record.get("tasks") or {}, threshold)) for record in records)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Bản ghi", len(records))
    metric_cols[1].metric("Tác vụ hoàn tất", progress.get("completed", 0))
    metric_cols[2].metric("Tác vụ lỗi", progress.get("error", 0))
    metric_cols[3].metric("Bản ghi cần review", review_count)
    st.caption(
        f"Provider: {job.get('provider')} · Model: {job.get('model')} · Mode: {job.get('mode')} · "
        f"Prompt {job.get('prompt_version')} · Taxonomy {job.get('taxonomy_version')} · Schema {job.get('schema_version')}"
    )

    rows = _review_rows(job, records, threshold)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        filter_cols = st.columns(4)
        with filter_cols[0]:
            task_values = ["Tất cả", *[task for task in job.get("tasks", []) if task in TASK_LABELS]]
            task_filter = st.selectbox("Tác vụ", task_values, format_func=lambda value: TASK_LABELS.get(value, value), key="result_task_filter")
        with filter_cols[1]:
            all_statuses = sorted(str(value) for value in frame["status"].dropna().unique())
            status_filter = st.multiselect("Trạng thái", all_statuses, default=all_statuses, key="result_status_filter")
        with filter_cols[2]:
            sources = sorted(str(value) for value in frame["source"].dropna().unique())
            source_filter = st.selectbox("Nguồn", ["Tất cả", *sources], key="result_source_filter")
        with filter_cols[3]:
            confidence_filter = st.selectbox(
                "Mức tự tin (confidence)",
                ["Tất cả", "Thấp hơn ngưỡng", "Từ ngưỡng trở lên", "Chưa có"],
                key="result_confidence_filter",
            )

        label_options = sorted(value for value in frame["label"].dropna().unique() if value)
        selected_labels = st.multiselect("Lọc theo nhãn", label_options, key="result_label_filter")
        search = st.text_input("Tìm trong tiêu đề hoặc URL", key="result_text_filter").strip().casefold()

        filtered = frame.copy()
        if task_filter != "Tất cả":
            filtered = filtered[filtered["task"] == task_filter]
        if status_filter:
            filtered = filtered[filtered["status"].isin(status_filter)]
        else:
            filtered = filtered.iloc[0:0]
        if source_filter != "Tất cả":
            filtered = filtered[filtered["source"].astype(str) == source_filter]
        if confidence_filter == "Thấp hơn ngưỡng":
            filtered = filtered[filtered["confidence"].notna() & (filtered["confidence"] < threshold)]
        elif confidence_filter == "Từ ngưỡng trở lên":
            filtered = filtered[filtered["confidence"].notna() & (filtered["confidence"] >= threshold)]
        elif confidence_filter == "Chưa có":
            filtered = filtered[filtered["confidence"].isna()]
        if selected_labels:
            filtered = filtered[filtered["label"].isin(selected_labels)]
        if search:
            filtered = filtered[
                filtered["title"].fillna("").astype(str).str.casefold().str.contains(search, regex=False)
                | filtered["url"].fillna("").astype(str).str.casefold().str.contains(search, regex=False)
            ]

        display_columns = [
            "record_id", "task_name", "status", "label", "confidence", "review_reason",
            "source", "title", "url", "gold_labels", "reason", "evidence", "input_truncated",
            "attempts", "error_kind", "error_message",
        ]
        st.caption(f"Hiển thị {len(filtered):,} / {len(frame):,} dòng tác vụ.")
        st.dataframe(filtered[display_columns], width="stretch", hide_index=True)
        if frame["gold_labels"].astype(bool).any():
            st.caption("Nhãn gốc được hiển thị để đối chiếu và vẫn giữ nguyên; ứng dụng không ghi đè hoặc tự kết luận accuracy.")

    export_key = (snapshot_key, threshold, job.get("status"), job.get("updated_at"))
    export_cache = st.session_state.get("results_export_cache")
    if not export_cache or export_cache["key"] != export_key:
        with activity("Chuẩn bị các tệp kết quả"):
            exports = build_export_files(job, records, confidence_threshold=threshold)
            archive = BytesIO()
            with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
                for filename, content in exports.items():
                    bundle.writestr(filename, content)
            export_cache = {"key": export_key, "exports": exports, "archive": archive.getvalue()}
        st.session_state["results_export_cache"] = export_cache
    exports = export_cache["exports"]
    st.markdown("#### Tải tệp xuất")
    st.download_button(
        "Tải toàn bộ kết quả (.zip)", export_cache["archive"],
        file_name=f"ket_qua_{job_id[:8]}.zip", mime="application/zip",
        type="primary", key=f"download_bundle_{job_id}",
    )
    st.caption("labeled_dataset.csv: dữ liệu và nhãn · review_queue.csv: cần kiểm tra · errors.csv: lỗi · predictions.jsonl và summary.json: chi tiết.")
    download_cols = st.columns(len(exports))
    for column, (filename, content) in zip(download_cols, exports.items()):
        mime = "application/json" if filename.endswith(".json") or filename.endswith(".jsonl") else "text/csv"
        column.download_button(
            filename,
            data=content,
            file_name=filename,
            mime=mime,
            key=f"download_{filename}_{job_id}",
            width="stretch",
        )
    if st.button("Lưu toàn bộ tệp vào data/output", key="save_exports"):
        try:
            target = _save_exports(job_id, exports)
            st.success(f"Đã lưu 5 tệp tại `{target}`.")
        except OSError as exc:
            st.error(f"Không ghi được tệp xuất: {exc}")

