"""Main annotation workflow with real local JPEG attachments."""
from pathlib import Path
import json
import streamlit as st
from src.labeling.multimodal import read_records, export_csv, TYPES
from src.providers.base import sanitize_error_message
from src.runtime import API_ACTIVITY_LOCK
from src.labeling.quality import reviewed_record,VERSION as QUALITY_VERSION

def render_multimodal_tab():
    st.subheader("Gán nhãn theo quy chuẩn đa phương thức")
    st.caption("Title + toàn văn + thumbnail thực tế · 5 Type · TC/TI/IC · Unknown giữ null")
    default = Path(__file__).resolve().parents[3] / "data_pr" / "clean_final"
    path = st.text_input("CSV cần gán nhãn", str(default / "sample_1000.csv"), key="annotation_csv")
    image_root = st.text_input("Thư mục gốc để giải đường dẫn image_path", str(default), key="annotation_image_root")
    st.caption("Ví dụ image_path=images/a_1.jpg thì gốc là clean_final, không phải clean_final/images.")
    count = st.number_input("Số bài chạy thử", min_value=1, max_value=100, value=3, key="annotation_count")
    mode = st.radio("Chế độ", ["Mô phỏng ngoại tuyến", "API thực tế"], key="annotation_mode")
    st.caption('Chế độ rà soát dùng 2 lượt đánh giá/bài, tối đa thêm 1 lượt phân xử khi cần; chi phí cao hơn lượt đơn. Bất đồng chưa đủ căn cứ giữ null, không ép nhãn.')
    st.info("Ảnh chỉ nối qua ID cần audit; confidence chưa hiệu chỉnh. Gold validation/test phải do người gán nhãn độc lập.")
    if st.button("Chạy gán nhãn", key="annotation_run"):
        try:
            records = read_records(Path(path), int(count))
            provider = None
            if mode == "API thực tế":
                from src.ui.provider import get_active_provider
                provider = get_active_provider()
                provider.requests_per_minute=max(1,int(st.session_state.get('provider_config',{}).get('requests_per_minute',10)))
                if provider.provider_name == "mock": raise ValueError("Chọn Gemini/OpenAI ở bước Kết nối AI trước khi chạy API thực tế.")
            results = []; st.session_state["annotation_results"] = []; progress = st.progress(0)
            with API_ACTIVITY_LOCK:
                for i, record in enumerate(records):
                    try:
                        engine=reviewed_record
                        with st.spinner(f"Đang đánh giá bài {i+1}/{len(records)}…"):
                            result = engine(record, provider, Path(image_root), mock=mode == "Mô phỏng ngoại tuyến")
                    except Exception as exc:
                        st.error(f"Bài {record['article_id']}: " + sanitize_error_message(exc, getattr(provider,"api_key","")))
                        break
                    results.append(result)
                    # Preserve completed rows even if a later request fails or quota is exhausted.
                    st.session_state["annotation_results"] = results
                    progress.progress((i+1)/max(1,len(records)))
            incomplete=sum(any(str(x).startswith('semantic_review_unavailable:') for x in json.loads(r.get('annotation_review_flags','[]'))) for r in results)
            if incomplete:
                st.warning(f"Đã lưu {len(results)} bài, nhưng {incomplete} bài chưa hoàn tất rà soát do lỗi API. Không dùng các mask=false làm nhãn train.")
            elif len(results) == len(records):
                st.success(f"Hoàn tất {len(results)}/{len(records)} bài. Mô phỏng không tạo nhãn train.")
            else:
                st.warning(f"Đã dừng; giữ được {len(results)}/{len(records)} kết quả. Chưa hoàn tất lượt chạy.")
        except Exception as exc:
            st.error(sanitize_error_message(exc))
    results = st.session_state.get("annotation_results", [])
    if results:
        st.dataframe([{k:r.get(k) for k in ("article_id","title","clickbait_label", *("type_"+t for t in TYPES),"consistency_tc","consistency_ti","consistency_ic","annotation_status")} for r in results])
        st.download_button("Tải CSV kết quả", export_csv(results), "annotations.csv", "text/csv", key="annotation_download")
        with st.expander("Bằng chứng và hàng chờ review"):
            for r in results:
                st.write(r["article_id"], r["annotation_review_flags"], r["annotation_input_warnings"],r.get('label_mask_json',''))
                st.json(json.loads(r.get("annotation_grounded_json") or r["annotation_raw_json"]))
    st.caption(f"Phiên bản: {QUALITY_VERSION}. Chỉ các target có mask=true được dùng cho train.")
