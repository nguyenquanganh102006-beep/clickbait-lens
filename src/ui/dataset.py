"""Dataset import and column mapping tab."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from src.ui import local_memory
from src.paths import INPUT_DIR
from src.ui.activity import activity

from src.data.loader import DatasetLoadError, list_workbook_sheets, load_dataset
from src.data.mapping import (
    CANONICAL_FIELDS,
    diagnose_dataset,
    normalize_records,
    suggest_column_mapping,
    task_eligibility,
)


FIELD_LABELS = {
    "record_id": "ID bản ghi",
    "title": "Tiêu đề (title)",
    "content": "Toàn văn (content)",
    "lead": "Sapo/lead",
    "source": "Nguồn",
    "url": "URL",
    "published_at": "Thời điểm xuất bản",
}


def _clear_mapping_state() -> None:
    for field in CANONICAL_FIELDS:
        st.session_state.pop(f"dataset_map_{field}", None)


def _clear_dataset_state() -> None:
    for key in (
        "dataset_frame",
        "dataset_filename",
        "dataset_mapping",
        "dataset_records",
        "dataset_diagnostics",
    ):
        st.session_state.pop(key, None)
    _clear_mapping_state()


def _load_into_session(source: object, filename: str, sheet_name: str | None = None) -> None:
    try:
        with activity(f"Đọc tệp {filename}") as status:
            status.write(f"Sheet: {sheet_name or 'mặc định'}. Excel lớn có thể mất vài phút; hãy chờ đọc xong.")
            frame = load_dataset(source, filename=filename, sheet_name=sheet_name)
            status.write(f"Đã đọc {len(frame):,} dòng và {len(frame.columns):,} cột.")
    except DatasetLoadError as exc:
        st.error(str(exc))
        st.caption("Dữ liệu đã đọc trước đó được giữ lại; tệp mới chưa được áp dụng.")
        return
    st.session_state["dataset_frame"] = frame
    st.session_state["dataset_filename"] = filename
    for key in ("dataset_mapping", "dataset_records", "dataset_diagnostics"):
        st.session_state.pop(key, None)
    _clear_mapping_state()


@st.cache_data(show_spinner=False, max_entries=4)
def _cached_workbook_sheets(content: bytes, filename: str) -> list[str]:
    return list_workbook_sheets(content, filename)


def render_dataset_tab() -> None:
    st.subheader("Nạp dữ liệu")
    st.write("Chọn tệp CSV, XLSX hoặc JSONL để xem trước và ánh xạ các cột.")
    st.caption("Chọn tệp → bấm Đọc tệp → kiểm tra cột Tiêu đề và Toàn văn bên dưới.")
    source_mode = st.radio(
        "Nguồn dữ liệu",
        ["Tải tệp lên", "Chọn trong data/input"],
        horizontal=True,
        key="dataset_source_mode",
    )

    source = None
    filename = ""
    if source_mode == "Tải tệp lên":
        uploaded = st.file_uploader(
            "Chọn tệp dữ liệu",
            type=["csv", "xlsx", "jsonl"],
            key="dataset_upload",
        )
        if uploaded is not None:
            source, filename = uploaded, uploaded.name
            if uploaded.size > 20 * 1024 * 1024:
                st.info("Tệp lớn: đọc Excel có thể mất vài phút. CSV thường nạp nhanh hơn. Hãy chờ thông báo đã đọc xong.")
    else:
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        local_files = sorted(
            path for path in INPUT_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".jsonl"}
        )
        if not local_files:
            st.info("Chưa có tệp được hỗ trợ trong data/input/.")
        else:
            selected_path = st.selectbox(
                "Tệp trong data/input/",
                local_files,
                format_func=lambda path: path.name,
                key="dataset_local_path",
            )
            source, filename = selected_path, selected_path.name

    sheet_name: str | None = None
    if source is not None and Path(filename).suffix.lower() == ".xlsx":
        try:
            with activity("Đọc danh sách sheet Excel"):
                content = source.read_bytes() if isinstance(source, Path) else source.getvalue()
                sheets = _cached_workbook_sheets(content, filename)
        except DatasetLoadError as exc:
            st.error(str(exc))
            sheets = []
        if sheets:
            sheet_name = st.selectbox("Sheet Excel", sheets, key="dataset_sheet")

    if source is not None and st.button("Đọc tệp", type="primary", key="load_dataset"):
        _load_into_session(source, filename, sheet_name)

    frame = st.session_state.get("dataset_frame")
    if not isinstance(frame, pd.DataFrame):
        return

    st.divider()
    st.markdown(f"**Tệp đang xem:** `{st.session_state.get('dataset_filename', '—')}`")
    st.success(f"Đã đọc tệp: {len(frame):,} dòng · {len(frame.columns):,} cột")
    st.dataframe(frame.head(10), width="stretch", hide_index=True)

    if frame.empty:
        st.warning("Tệp không có bản ghi.")
    if len(frame.columns) == 0:
        st.warning("Không tìm thấy cột dữ liệu trong tệp.")
        return

    st.markdown("#### Ánh xạ cột chuẩn")
    st.caption("Cột nguồn được giữ nguyên. Các trường chuẩn chỉ bổ sung dữ liệu đã ánh xạ.")
    suggestions = suggest_column_mapping(list(frame.columns))
    columns: list[str | None] = [None, *[str(column) for column in frame.columns]]
    mapping: dict[str, str | None] = {}
    column_groups = st.columns(3)
    for index, field in enumerate(CANONICAL_FIELDS):
        suggested = suggestions[field]
        existing = st.session_state.get(f"dataset_map_{field}", suggested)
        selected_index = columns.index(existing) if existing in columns else 0
        with column_groups[index % len(column_groups)]:
            selected = st.selectbox(
                FIELD_LABELS[field],
                options=columns,
                index=selected_index,
                format_func=lambda value: "— Không ánh xạ —" if value is None else value,
                key=f"dataset_map_{field}",
            )
        mapping[field] = selected

    if mapping == st.session_state.get("dataset_mapping") and "dataset_records" in st.session_state:
        records = st.session_state["dataset_records"]
        diagnostics = st.session_state["dataset_diagnostics"]
    else:
        with activity("Kiểm tra ánh xạ và chuẩn hóa bản ghi"):
            records = normalize_records(frame, mapping)
            diagnostics = diagnose_dataset(frame, records)
    eligibility = task_eligibility(records, min_consistency_content_chars=80, max_content_chars=20000)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Bản ghi", diagnostics["row_count"])
    metric_cols[1].metric("Thiếu tiêu đề", diagnostics["missing_title_count"])
    metric_cols[2].metric("ID trùng", len(diagnostics["duplicate_ids"]))
    metric_cols[3].metric("URL trùng", len(diagnostics["duplicate_urls"]))
    if diagnostics["empty_columns"]:
        st.warning("Cột hoàn toàn trống: " + ", ".join(diagnostics["empty_columns"]))
    if diagnostics["duplicate_ids"]:
        st.warning("Có ID trùng: " + ", ".join(diagnostics["duplicate_ids"][:10]))
    if diagnostics["duplicate_urls"]:
        st.info(f"Phát hiện {len(diagnostics['duplicate_urls'])} URL bị lặp.")

    st.markdown("#### Số bản ghi đủ dữ liệu theo tác vụ")
    eligibility_cols = st.columns(3)
    for column, task, label in zip(
        eligibility_cols,
        ("detection", "type", "consistency"),
        (
            "Phát hiện clickbait",
            "Phân loại loại clickbait",
            "Nhất quán tiêu đề–toàn văn",
        ),
    ):
        with column:
            st.metric(label, eligibility[task]["eligible"])
            if task == "consistency":
                st.caption(
                    f"Thiếu tiêu đề: {eligibility[task]['missing_title']} · "
                    f"Toàn văn thiếu/quá ngắn: {eligibility[task]['insufficient_content']} · "
                    f"Vượt giới hạn {20_000:,} ký tự: {eligibility[task]['truncated_content']}"
                )
            else:
                st.caption(f"Thiếu tiêu đề: {eligibility[task]['missing_title']}")

    st.session_state["dataset_mapping"] = mapping
    st.session_state["dataset_records"] = records
    st.session_state["dataset_diagnostics"] = diagnostics
    st.markdown("#### Dùng lại cho những lần sau")
    st.caption("Lưu bộ dữ liệu đang xem cùng ánh xạ cột. Lần mở ứng dụng sau sẽ tự khôi phục; lưu bộ mới sẽ thay thế bản nhớ trước.")
    save_col, forget_col = st.columns(2)
    if save_col.button("Lưu dữ liệu và ánh xạ trên máy", key="remember_dataset"):
        try:
            with activity("Lưu dữ liệu và ánh xạ trên máy"):
                local_memory.save_dataset(frame, st.session_state.get("dataset_filename", "dataset"), mapping)
            st.success("Đã nhớ bộ dữ liệu. Lần mở sau không cần tải tệp lên lại.")
        except Exception:
            st.error("Không lưu được bộ dữ liệu. Bản đang xem vẫn dùng được; kiểm tra quyền ghi và định dạng các cột.")
    if forget_col.button("Xóa dữ liệu đã nhớ trên máy", key="forget_dataset"):
        try:
            local_memory.forget_dataset()
            st.success("Đã xóa bản nhớ. Dữ liệu đang mở và kết quả job vẫn được giữ.")
        except OSError:
            st.error("Không xóa được bản nhớ. Kiểm tra quyền ghi thư mục state.")
    if not mapping.get("title"):
        st.warning("Hãy chọn cột Tiêu đề ở phần ánh xạ để có thể gán nhãn.")
    elif diagnostics["missing_title_count"] == len(records):
        st.warning("Cột Tiêu đề đang chọn không có dữ liệu. Hãy kiểm tra lại ánh xạ.")
    else:
        st.success("Dữ liệu đã sẵn sàng. Chọn bước 3 ở thanh bên để chọn tác vụ.")
