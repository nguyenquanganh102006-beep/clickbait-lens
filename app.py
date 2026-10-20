"""Streamlit entry point for the KHDL Auto Labeler."""

from __future__ import annotations

import streamlit as st
from src import __version__

from src.ui.dataset import render_dataset_tab
from src.ui.jobs import render_jobs_tab
from src.ui.provider import render_provider_tab
from src.ui.results import render_results_tab
from src.ui.tasks import render_tasks_tab
from src.ui import local_memory
from src.ui.provider import PROVIDER_LABELS
from src.data.mapping import normalize_records, diagnose_dataset
from src.ui.activity import render_activity_history


st.set_page_config(
    page_title="KHDL Auto Labeler",
    page_icon="📰",
    layout="wide",
)

st.title("KHDL Auto Labeler")
st.caption(f"Công cụ gán nhãn giả clickbait chạy cục bộ cho nhóm KHDL · v{__version__}")
if not st.session_state.get("local_memory_loaded"):
    st.session_state["local_memory_loaded"] = True
    try:
        workflow = local_memory.load_workflow()
        st.session_state["selected_tasks"] = workflow["tasks"]
        st.session_state["labeling_mode"] = workflow["mode"]
        st.session_state["labeling_mode_label"] = "Độc lập (khuyến nghị)" if workflow["mode"] == "independent" else "Gộp trong một yêu cầu"
        st.session_state["workflow_last_saved"] = workflow
    except (OSError, ValueError):
        st.warning("Không đọc được cấu hình tác vụ đã lưu. Tạm dùng tác vụ Phát hiện clickbait; kiểm tra lại ở bước 3.")
    try:
        if not st.session_state.get("provider_config"):
            remembered = local_memory.load_provider()
            if remembered:
                st.session_state["provider_config"] = remembered
                st.session_state["provider_label"] = next(label for label, value in PROVIDER_LABELS.items() if value == remembered["provider"])
                for widget, field in {
                    "provider_api_key": "api_key", "provider_model_before_connection": "model",
                    "provider_timeout": "timeout_seconds", "provider_rpm": "requests_per_minute",
                    "provider_concurrency": "concurrency", "provider_retries": "max_retries",
                }.items():
                    st.session_state[widget] = remembered[field]
    except Exception:
        st.warning("Không khôi phục được kết nối AI đã nhớ. Hãy nhập lại ở bước 1 và lưu để thay thế bản cũ.")
    try:
        if "dataset_frame" not in st.session_state:
            with st.spinner("Đang khôi phục dữ liệu đã lưu trên máy…"):
                saved_dataset = local_memory.load_dataset()
                if saved_dataset is not None:
                    frame, metadata = saved_dataset
                    mapping = metadata["mapping"]
                    records = normalize_records(frame, mapping)
                    st.session_state.update({
                        "dataset_frame": frame, "dataset_filename": metadata["filename"],
                        "dataset_mapping": mapping, "dataset_records": records,
                        "dataset_diagnostics": diagnose_dataset(frame, records),
                    })
                    for field, column in mapping.items():
                        st.session_state[f"dataset_map_{field}"] = column
    except Exception:
        st.warning("Không khôi phục được dữ liệu đã nhớ. Hãy nạp và lưu lại ở bước 2.")
st.session_state.setdefault("selected_tasks", ["detection"])
st.session_state.setdefault("labeling_mode", "independent")
pages = {
    "1. Kết nối AI": render_provider_tab,
    "2. Nạp dữ liệu": render_dataset_tab,
    "3. Chọn tác vụ": render_tasks_tab,
    "4. Chạy gán nhãn": render_jobs_tab,
    "5. Xem & tải kết quả": render_results_tab,
}
with st.sidebar:
    st.header("Các bước xử lý")
    page = st.radio("Đi đến bước", list(pages), key="workflow_page")
    st.caption("Thiết lập bước 1–3, chạy thử ở bước 4 rồi tải kết quả ở bước 5.")

# Preserve only widgets on hidden pages. Writing the state of a visible
# input can send its previous value back to the browser while the user types.
for key in list(st.session_state):
    hidden_provider = page != "1. Kết nối AI" and (
        key.startswith("provider_") or key in {"selected_model", "remember_provider"}
    )
    hidden_dataset = page != "2. Nạp dữ liệu" and (
        key.startswith("dataset_map_") or key in {
            "dataset_source_mode", "dataset_local_path", "dataset_sheet",
        }
    )
    hidden_tasks = page != "3. Chọn tác vụ" and key in {
        "selected_tasks", "labeling_mode_label",
    }
    if hidden_provider or hidden_dataset or hidden_tasks:
        st.session_state[key] = st.session_state[key]

pages[page]()

with st.sidebar:
    st.divider()
    st.markdown("**Trạng thái phiên hiện tại**")
    config = st.session_state.get("provider_config", {})
    if config:
        st.success(f"AI: {config['provider']} / {config['model']}")
    else:
        st.info("Bước 1: Chưa lưu kết nối AI")
    records = st.session_state.get("dataset_records", [])
    if records:
        st.success(f"Dữ liệu: {len(records):,} bản ghi")
    else:
        st.info("Bước 2: Chưa có dữ liệu để chạy")
    st.caption("Có thể nhớ kết nối AI ở bước 1 và lưu dữ liệu kèm ánh xạ ở bước 2 để dùng cho lần mở sau.")
    render_activity_history()
