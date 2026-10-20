"""Provider settings tab. Credentials stay in the current Streamlit session."""

from __future__ import annotations

import hashlib

import streamlit as st

from src.providers import GeminiProvider, MockProvider, OpenAIProvider, ProviderError
from src.ui import local_memory
from src.ui.activity import activity
from src.ui.model_checks import render_model_checks, verified_for_config, check_context, context_results
from src.ui.widgets import number_input


PROVIDER_LABELS = {
    "Gemini": "gemini",
    "OpenAI": "openai",
    "Mô phỏng ngoại tuyến": "mock",
}


def create_provider(provider_name: str, api_key: str, model: str, timeout_seconds: int = 60):
    if provider_name == "gemini":
        return GeminiProvider(api_key, model, timeout_seconds)
    if provider_name == "openai":
        return OpenAIProvider(api_key, model, timeout_seconds)
    if provider_name == "mock":
        return MockProvider(model=model or "mock-v1", timeout_seconds=timeout_seconds)
    raise ProviderError("Nhà cung cấp chưa được hỗ trợ.", kind="configuration")


def get_active_provider():
    """Build an in-memory provider from current UI session settings."""
    config = st.session_state.get("provider_config", {})
    name = config.get("provider")
    model = config.get("model")
    if not name or not model:
        raise ProviderError("Hãy chọn nhà cung cấp và model trước.", kind="configuration")
    return create_provider(
        name,
        config.get("api_key", ""),
        model,
        int(config.get("timeout_seconds", 60)),
    )


def render_provider_tab() -> None:
    st.subheader("Cấu hình nhà cung cấp")
    st.caption("Có thể nhớ kết nối trên máy. API key được mã hóa bằng tài khoản Windows hiện tại.")
    st.info("Nhập API key rồi nhấn Enter hoặc bấm ra ngoài ô nhập để tự tải danh sách model. Chọn model rồi bấm Lưu cấu hình.")

    provider_label = st.selectbox("Nhà cung cấp", list(PROVIDER_LABELS), key="provider_label")
    provider_name = PROVIDER_LABELS[provider_label]
    key_owner = st.session_state.get("provider_key_owner", provider_name)
    if key_owner != provider_name:
        keys = dict(st.session_state.get("provider_keys_in_session", {}))
        keys[key_owner] = st.session_state.get("provider_api_key", "")
        st.session_state["provider_keys_in_session"] = keys
        st.session_state["provider_api_key"] = keys.get(provider_name, "")
    st.session_state["provider_key_owner"] = provider_name
    api_key = ""
    if provider_name in {"gemini", "openai"}:
        api_key = st.text_input(
            "API key",
            type="password",
            key="provider_api_key",
            help="Không đưa API key vào tệp dữ liệu, log hoặc tài liệu chia sẻ.",
        )

    timeout_seconds = number_input(
        "Thời gian chờ tối đa (giây)", min_value=5, max_value=600, value=60, step=5,
        key="provider_timeout",
    )
    if provider_name == "mock":
        model = "mock-v1"
        st.success("Chế độ mô phỏng hoạt động ngoại tuyến, không gửi dữ liệu đến API.")
    else:
        # Reserve space before discovery messages so the check controls stay
        # near the API key instead of below a long list of status messages.
        model_checks_area = st.container(border=True)
        fingerprint = hashlib.sha256(f"{provider_name}:{api_key.strip()}".encode()).hexdigest()
        if st.session_state.get("model_discovery_fingerprint") != fingerprint:
            st.session_state["model_discovery_fingerprint"] = fingerprint
            st.session_state.pop("available_models", None)
            st.session_state.pop("model_check_result", None)
            st.session_state.pop("selected_model", None)
            cached = context_results(check_context(provider_name, api_key.strip(),
                st.session_state.get("selected_tasks", ["detection"]),
                st.session_state.get("labeling_mode", "independent"))) if api_key.strip() else {}
            if cached:
                st.session_state["discovered_provider"] = provider_name
                st.session_state["available_models"] = sorted(cached)
                st.session_state["model_check_result"] = {
                    "provider": provider_name, "level": "success",
                    "message": "Đã khôi phục danh sách model đã kiểm tra trên máy. Không gọi lại API. Có thể quét lại toàn bộ để cập nhật.",
                }
            elif api_key.strip():
                try:
                    with activity(f"Tự tải danh sách model {provider_label}") as status:
                        status.write(f"Đang chờ API. Thời gian chờ tối đa mỗi yêu cầu: {int(timeout_seconds)} giây.")
                        client = create_provider(provider_name, api_key.strip(), "model-check", int(timeout_seconds))
                        discovered = sorted(set(client.list_models()))
                    st.session_state["discovered_provider"] = provider_name
                    st.session_state["available_models"] = discovered
                    st.session_state["model_check_result"] = {
                        "provider": provider_name, "level": "success" if discovered else "warning",
                        "message": f"Đã tự tải {len(discovered)} model từ API. Chọn model rồi bấm Lưu cấu hình."
                        if discovered else "API không trả về model. Thử tải lại hoặc nhập model ID thủ công.",
                    }
                except Exception:
                    st.session_state["model_check_result"] = {
                        "provider": provider_name, "level": "error",
                        "message": st.session_state["activity_history"][-1]["detail"],
                    }
        discovered_provider = st.session_state.get("discovered_provider")
        models = st.session_state.get("available_models", []) if discovered_provider == provider_name else []
        manual_model = st.checkbox("Nhập model ID thủ công", key="provider_manual_model")
        if manual_model:
            model = st.text_input("Model ID", key="provider_manual_model_id").strip()
        else:
            model = st.text_input(
                "Thêm model ID để kiểm tra (tùy chọn)",
                key="provider_model_before_connection",
                placeholder="Nhập model ID nếu muốn dùng thủ công",
            ).strip()

        if st.button("Kiểm tra kết nối và tải model", key="test_provider"):
            if not api_key.strip():
                st.error("Nhập API key trước khi kiểm tra kết nối.")
            else:
                try:
                    with activity(f"Kết nối {provider_label} và lấy danh sách model") as status:
                        status.write("Đang khởi tạo kết nối…")
                        provider = create_provider(provider_name, api_key, model or "model-check", int(timeout_seconds))
                        status.write(f"Đang chờ API trả danh sách model. Thời gian chờ tối đa mỗi yêu cầu: {int(timeout_seconds)} giây.")
                        models = provider.list_models()
                        status.write(f"API đã trả về {len(models)} model.")
                    st.session_state["discovered_provider"] = provider_name
                    st.session_state["available_models"] = models
                    st.session_state["model_check_result"] = {
                        "provider": provider_name,
                        "level": "success" if models else "warning",
                        "message": f"Đã tải xong: API trả về {len(models)} model. Chọn model ở ô Model khả dụng phía trên rồi bấm Lưu cấu hình."
                        if models else "Kết nối được nhưng API không trả về model có thể chọn. Bạn có thể nhập model ID thủ công.",
                    }
                    st.rerun()
                except Exception as exc:
                    st.session_state.pop("available_models", None)
                    st.session_state["model_check_result"] = {
                        "provider": provider_name, "level": "error", "message": st.session_state["activity_history"][-1]["detail"],
                    }
                    st.rerun()

        check_result = st.session_state.get("model_check_result", {})
        if check_result.get("provider") == provider_name:
            getattr(st, check_result["level"])(check_result["message"])
            if models and check_result["level"] == "success":
                st.caption("Danh sách API chưa xác nhận quyền chạy. Bấm Kiểm tra khả năng gán nhãn bên dưới để lọc model dùng được.")
                with st.expander(f"Xem toàn bộ {len(models)} model API trả về"):
                    st.code("\n".join(models), language="text")
                if len(models) == 1:
                    st.info("API trả về một model. Cần gọi thử gán nhãn để xác nhận tài khoản sử dụng được.")

        candidates = ([model] if manual_model else [*models, *([model] if model else [])])
        with model_checks_area:
            model = render_model_checks(provider_name, api_key.strip(), int(timeout_seconds), candidates, create_provider)

    st.markdown("#### Giới hạn chạy")
    limit_cols = st.columns(3)
    with limit_cols[0]:
        rpm = number_input("Yêu cầu/phút", min_value=1, max_value=600, value=10, key="provider_rpm")
    with limit_cols[1]:
        concurrency = number_input("Đồng thời", min_value=1, max_value=8, value=1, key="provider_concurrency")
    with limit_cols[2]:
        max_retries = number_input("Số lần thử lại tối đa", min_value=0, max_value=10, value=3, key="provider_retries")

    if provider_name != "mock":
        st.info("Khi chạy gán nhãn, tiêu đề và nội dung đã chọn sẽ được gửi tới nhà cung cấp API.")

    draft = {
        "provider": provider_name,
        "model": model,
        "api_key": api_key.strip(),
        "timeout_seconds": int(timeout_seconds),
        "requests_per_minute": int(rpm),
        "concurrency": int(concurrency),
        "max_retries": int(max_retries),
    }

    remember = st.checkbox("Nhớ kết nối AI trên máy cho những lần sau", value=True, key="remember_provider")
    if st.button("Lưu cấu hình", type="primary", key="save_provider_config"):
        if provider_name != "mock" and not api_key.strip():
            st.error("Nhập API key trước khi lưu cấu hình.")
        elif not model:
            st.error("Kiểm tra khả năng gán nhãn rồi chọn model đã vượt qua kiểm tra trước khi lưu.")
        elif not verified_for_config(draft):
            st.error("Model chưa được kiểm tra với key và tác vụ hiện tại.")
        else:
            st.session_state["provider_config"] = draft
            try:
                if remember:
                    with activity("Mã hóa và lưu kết nối AI trên máy"):
                        local_memory.save_provider(draft)
                else:
                    with activity("Xóa bản nhớ kết nối AI"):
                        local_memory.forget_provider()
                st.success("Đã lưu kết nối trên máy." if remember else "Đã lưu cho phiên này; bản nhớ trên máy đã được xóa.")
            except OSError:
                st.error("Cấu hình dùng được trong phiên này nhưng chưa lưu được trên máy. Kiểm tra quyền ghi thư mục state.")

    if st.button("Xóa kết nối AI đã nhớ trên máy", key="forget_provider"):
        try:
            local_memory.forget_provider()
            st.success("Đã xóa bản nhớ trên máy. Kết nối hiện tại vẫn dùng được đến khi đóng phiên.")
        except OSError:
            st.error("Không xóa được bản nhớ. Kiểm tra quyền ghi thư mục state.")

    saved = st.session_state.get("provider_config")
    if saved and saved != draft:
        st.warning("Có thay đổi chưa lưu. Job mới vẫn dùng cấu hình đã lưu trước đó; bấm Lưu cấu hình để áp dụng thay đổi.")
    elif saved:
        st.caption("Cấu hình đã lưu. Tiếp theo: chọn bước 2 để nạp dữ liệu.")
