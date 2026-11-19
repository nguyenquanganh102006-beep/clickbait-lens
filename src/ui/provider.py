"""Connection settings for the current UI and CLI."""
import streamlit as st
from src.providers import GeminiProvider, OpenAIProvider, ProviderError
from src.ui import local_memory
from src.providers.base import sanitize_error_message

PROVIDER_LABELS = {'Gemini':'gemini', 'OpenAI':'openai'}

def create_provider(provider_name, api_key, model, timeout_seconds=60):
    adapter = {'gemini': GeminiProvider, 'openai': OpenAIProvider}.get(provider_name)
    if adapter is None:
        raise ProviderError('Chọn Gemini hoặc OpenAI.', kind='configuration')
    return adapter(api_key, model, timeout_seconds)

def get_active_provider():
    config = st.session_state.get('provider_config', {})
    if not config.get('model'):
        raise ProviderError('Lưu kết nối AI trước khi chạy.', kind='configuration')
    provider = create_provider(config['provider'], config.get('api_key',''), config['model'], int(config.get('timeout_seconds',90)))
    provider.requests_per_minute = max(1,int(config.get('requests_per_minute',10)))
    return provider

def render_provider_tab():
    st.subheader('Kết nối AI')
    label = st.selectbox('Nhà cung cấp', list(PROVIDER_LABELS), key='provider_label')
    name = PROVIDER_LABELS[label]
    key = st.text_input('API key', type='password', key='provider_api_key')
    model = st.text_input('Model có hỗ trợ ảnh', value='gemini-3.1-flash-lite' if name=='gemini' else '', key='provider_model')
    timeout = st.number_input('Thời gian chờ mỗi yêu cầu (giây)', min_value=5, max_value=120, value=90, key='provider_timeout')
    rpm = st.number_input('Số yêu cầu tối đa mỗi phút', min_value=1, max_value=1000, value=10, key='provider_rpm')
    remember = st.checkbox('Nhớ kết nối trên máy bằng Windows DPAPI', value=False, key='remember_provider')
    st.caption('Lưu cấu hình không gọi API. Mỗi bài thực tế dùng 2–3 lượt đánh giá, có thể phát sinh phí.')
    if st.button('Lưu cấu hình', key='save_provider'):
        if not key.strip() or not model.strip():
            st.error('Cần API key và tên model.')
        else:
            config = dict(provider=name, api_key=key.strip(), model=model.strip(), timeout_seconds=int(timeout),
                          requests_per_minute=int(rpm), concurrency=1, max_retries=2)
            st.session_state['provider_config'] = config
            if remember:
                try:
                    local_memory.save_provider(config)
                    st.success('Đã lưu trong phiên và nhớ kết nối trên máy.')
                except Exception as exc:
                    st.error('Đã lưu trong phiên, chưa nhớ được trên máy: '+sanitize_error_message(exc,key))
            else:
                st.success('Đã lưu cấu hình trong phiên hiện tại.')
    if st.button('Xóa kết nối đã nhớ trên máy', key='forget_provider'):
        local_memory.forget_provider()
        st.success('Đã xóa bản nhớ trên máy. Kết nối trong phiên hiện tại vẫn dùng được.')
