"""Task selection, mode and immutable prompt preview."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml

from src.labeling.engine import LabelingEngine
from src.labeling.prompts import (
    PROMPT_VERSION,
    build_combined_prompt,
    build_task_prompt,
    load_task_prompt,
    load_taxonomy,
)
from src.providers.mock import MockProvider
from src.ui import local_memory


TASK_LABELS = {
    "detection": "Phát hiện clickbait (Clickbait Detection)",
    "type": "Phân loại loại clickbait (Clickbait Type)",
    "consistency": "Nhất quán tiêu đề–toàn văn (Title–Content Consistency)",
}


def render_tasks_tab() -> None:
    st.subheader("Cấu hình tác vụ và prompt")
    st.caption("Lựa chọn được lưu tự động trong phiên. Nếu chỉ cần nhãn 0/1, giữ tác vụ Phát hiện clickbait.")
    taxonomy = load_taxonomy()
    selected = st.multiselect(
        "Chọn một hoặc nhiều tác vụ",
        options=list(TASK_LABELS),
        **({} if "selected_tasks" in st.session_state else {"default": ["detection"]}),
        format_func=lambda task: TASK_LABELS[task],
        key="selected_tasks",
    )
    mode_label = st.radio(
        "Cách gửi yêu cầu",
        ["Độc lập (khuyến nghị)", "Gộp trong một yêu cầu"],
        horizontal=True,
        help="Độc lập giúp kiểm tra từng rubric riêng. Chế độ gộp dùng một yêu cầu cho các tác vụ đã chọn.",
        key="labeling_mode_label",
    )
    mode = "independent" if mode_label.startswith("Độc lập") else "combined"
    st.session_state["labeling_mode"] = mode
    workflow = {"tasks": list(selected), "mode": mode}
    if st.session_state.get("workflow_last_saved") != workflow:
        try:
            local_memory.save_workflow(list(selected), mode)
            st.session_state["workflow_last_saved"] = workflow
        except OSError:
            st.warning("Tác vụ đã áp dụng cho phiên này nhưng chưa lưu được trên máy.")
    st.caption("Chế độ đang áp dụng: " + ("Độc lập" if mode == "independent" else "Gộp"))

    if not selected:
        st.warning("Chọn ít nhất một tác vụ trước khi chạy.")
    if "consistency" in selected:
        mapping = st.session_state.get("dataset_mapping", {})
        if not mapping.get("content"):
            st.warning("Title–Content Consistency cần ánh xạ trường toàn văn `content`; lead/sapo không thay thế toàn văn.")
    if "type" in selected and not st.session_state.get("dataset_mapping", {}).get("content"):
        st.info("Clickbait Type vẫn có thể chạy từ tiêu đề; nhãn `misleading` cần đối chiếu toàn văn và sẽ không được tự đoán.")

    with st.expander("Danh sách loại clickbait và phiên bản cấu hình"):
        st.code("\n".join(f"- {label}" for label in taxonomy["clickbait_types"]), language="text")
        st.caption(f"Taxonomy {taxonomy['version']} · Prompt {PROMPT_VERSION} · Schema v1")
    if selected:
        st.success("Đã chọn tác vụ. Tiếp theo: chọn bước 4 để chạy thử.")

    preview_task = st.selectbox(
        "Xem rubric",
        options=list(TASK_LABELS),
        format_func=lambda task: TASK_LABELS[task],
        key="prompt_preview_task",
    )
    with st.expander("Xem nội dung prompt và dữ liệu mẫu", expanded=False):
        try:
            rubric = load_task_prompt(preview_task)
            st.markdown(rubric)
            records = st.session_state.get("dataset_records", [])
            if records:
                sample_record = records[0]
                settings_path = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
                with settings_path.open("r", encoding="utf-8") as handle:
                    job_settings = (yaml.safe_load(handle) or {}).get("jobs", {})
                engine = LabelingEngine(
                    MockProvider(),
                    taxonomy,
                    max_content_chars=int(job_settings.get("max_content_chars", 20000)),
                    min_consistency_content_chars=int(job_settings.get("min_consistency_content_chars", 80)),
                )
                mode = st.session_state.get("labeling_mode", "independent")
                preview_tasks = list(st.session_state.get("selected_tasks", [])) if mode == "combined" else [preview_task]
                ready_tasks = engine.requestable_tasks(sample_record, preview_tasks)
                content = str(sample_record.get("content") or "")
                content_is_complete = bool(content.strip()) and len(content) <= engine.max_content_chars
                if not ready_tasks:
                    st.info("Bản ghi mẫu sẽ được xử lý cục bộ với trạng thái thiếu dữ liệu/abstain; không gửi yêu cầu API cho tác vụ đang xem.")
                else:
                    if mode == "combined":
                        effective_record = dict(sample_record)
                        if not content_is_complete:
                            effective_record["content"] = None
                        prompt = build_combined_prompt(effective_record, ready_tasks, taxonomy)
                    else:
                        prompt = build_task_prompt(
                            preview_task,
                            sample_record,
                            taxonomy,
                            content_is_complete=content_is_complete,
                        )
                    st.caption("Prompt mẫu đã ghép theo mode và giới hạn nội dung; xem trước không gửi API.")
                    st.code(prompt, language="text")
            else:
                st.caption("Nạp dữ liệu để xem prompt đã ghép với bản ghi thực tế.")
        except ValueError as exc:
            st.error(str(exc))

    st.session_state["taxonomy_version"] = str(taxonomy["version"])
