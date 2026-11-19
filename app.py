"""Main entry point for the multimodal labeler."""
import streamlit as st
from src import __version__
from src.ui import local_memory
from src.ui.provider import render_provider_tab, PROVIDER_LABELS
from src.ui.multimodal import render_multimodal_tab

st.set_page_config(page_title="KHDL Auto Labeler", page_icon="📰", layout="wide")
st.title("KHDL Auto Labeler")
st.caption(f"Gán nhãn đa phương thức · v{__version__}")
if not st.session_state.get('local_memory_loaded'):
    st.session_state['local_memory_loaded'] = True
    try:
        config = local_memory.load_provider()
        if config and not st.session_state.get('provider_config'):
            st.session_state['provider_config'] = config
            st.session_state['provider_label'] = next(k for k,v in PROVIDER_LABELS.items() if v == config['provider'])
            for widget, field in {'provider_api_key':'api_key', 'provider_model':'model',
                                  'provider_timeout':'timeout_seconds', 'provider_rpm':'requests_per_minute'}.items():
                if field in config: st.session_state[widget] = config[field]
    except Exception:
        st.warning('Không đọc được kết nối đã nhớ. Hãy nhập lại ở trang Kết nối AI.')
pages = {'1. Kết nối AI': render_provider_tab, '2. Gán nhãn đa phương thức': render_multimodal_tab}
if st.session_state.get('workflow_page') not in pages:
    st.session_state['workflow_page'] = '2. Gán nhãn đa phương thức'
with st.sidebar:
    page = st.radio('Đi đến bước', list(pages), key='workflow_page')
    st.caption('Quy trình chính: 5 Type, Detection và TC/TI/IC; rà soát mù theo từng nhãn.')
for key in list(st.session_state):
    if (page != '1. Kết nối AI' and key.startswith('provider_')) or (
        page != '2. Gán nhãn đa phương thức' and key in {'annotation_csv','annotation_image_root','annotation_count','annotation_mode'}
    ):
        st.session_state[key] = st.session_state[key]
pages[page]()
