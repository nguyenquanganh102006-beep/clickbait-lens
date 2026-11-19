from pathlib import Path
from streamlit.testing.v1 import AppTest
from src.ui import local_memory

APP = str(Path(__file__).resolve().parents[1] / 'app.py')

def test_only_current_workflow_is_available(monkeypatch, tmp_path):
    monkeypatch.setattr(local_memory,'STATE_DIR',tmp_path/'remembered')
    app=AppTest.from_file(APP).run()
    assert app.radio(key='workflow_page').options == ['1. Kết nối AI','2. Gán nhãn đa phương thức']
    assert not any(x.key=='v4_robust' for x in app.checkbox)
    assert not app.exception

def test_connection_saved_without_api_and_preserved_on_navigation(monkeypatch,tmp_path):
    monkeypatch.setattr(local_memory,'STATE_DIR',tmp_path/'remembered')
    app=AppTest.from_file(APP).run()
    app.radio(key='workflow_page').set_value('1. Kết nối AI').run()
    app.text_input(key='provider_api_key').set_value('test-only-placeholder')
    app.text_input(key='provider_model').set_value('vision-test-model')
    app.button(key='save_provider').click().run()
    assert app.session_state['provider_config']['model']=='vision-test-model'
    app.radio(key='workflow_page').set_value('2. Gán nhãn đa phương thức').run()
    app.radio(key='workflow_page').set_value('1. Kết nối AI').run()
    assert app.text_input(key='provider_model').value=='vision-test-model'
    assert app.text_input(key='provider_api_key').value=='test-only-placeholder'
    assert not app.exception
