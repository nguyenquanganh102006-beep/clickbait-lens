from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.ui import jobs, results
from src.jobs.storage import JobStore
from src.ui import local_memory, provider
from src.providers.mock import MockProvider


APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def isolated_memory(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from src.ui import model_scan
    monkeypatch.setattr(model_scan, "_SCANS", {})
    monkeypatch.setattr(jobs, "_ACTIVE_WORKERS", {})
    monkeypatch.setattr(local_memory, "STATE_DIR", tmp_path / "remembered")
    monkeypatch.setattr(provider, "create_provider", lambda *args: SimpleNamespace(list_models=lambda: [], generate_structured=MockProvider().generate_structured))


def _widget(elements, key):
    return next(element for element in elements if element.key == key)


def test_api_key_can_be_entered_replaced_and_preserved():
    app = AppTest.from_file(str(APP)).run()
    key_input = _widget(app.text_input, "provider_api_key")
    assert not key_input.disabled
    key_input.set_value("fake-key-first").run()
    assert _widget(app.text_input, "provider_api_key").value == "fake-key-first"
    _widget(app.text_input, "provider_api_key").set_value("fake-key-replaced").run()
    assert _widget(app.text_input, "provider_api_key").value == "fake-key-replaced"
    _widget(app.radio, "workflow_page").set_value("3. Chọn tác vụ").run()
    _widget(app.radio, "workflow_page").set_value("1. Kết nối AI").run()
    assert _widget(app.text_input, "provider_api_key").value == "fake-key-replaced"
    _widget(app.text_input, "provider_api_key").set_value("").run()
    assert _widget(app.text_input, "provider_api_key").value == ""
    assert not app.exception


def test_model_discovery_feedback_survives_rerun(monkeypatch):
    from types import SimpleNamespace
    from src.providers.base import ProviderError
    monkeypatch.setattr(provider, "create_provider", lambda *args: SimpleNamespace(
        list_models=lambda: ["test-model-a", "test-model-b"]
    ))
    app = AppTest.from_file(str(APP)).run()
    _widget(app.text_input, "provider_api_key").set_value("fake-key").run()
    _widget(app.button, "test_provider").click().run()
    assert not app.exception
    assert any("2 model" in message.value for message in app.success)
    assert app.session_state["activity_history"][-1]["state"] == "Hoàn tất"
    assert next(w for w in app.multiselect if w.key.startswith("check_candidates_")).options == ["test-model-a", "test-model-b"]
    _widget(app.button, "test_provider").click().run()
    assert any("2 model" in message.value for message in app.success)

    def fail():
        raise ProviderError("Kết nối thất bại thử nghiệm", kind="authentication")
    monkeypatch.setattr(provider, "create_provider", lambda *args: SimpleNamespace(list_models=fail))
    _widget(app.button, "test_provider").click().run()
    assert not app.exception
    assert any("thất bại" in message.value for message in app.error)
    assert app.session_state["activity_history"][-1]["state"] == "Lỗi"
    assert "thất bại" in app.session_state["activity_history"][-1]["detail"]
    assert not any(element.key == "selected_model" for element in app.selectbox)


def test_models_load_automatically_once_and_refresh_when_key_changes(monkeypatch):
    from types import SimpleNamespace
    calls = []

    def create(name, key, model, timeout):
        calls.append((name, key))
        return SimpleNamespace(list_models=lambda: [f"model-{len(calls)}"])

    monkeypatch.setattr(provider, "create_provider", create)
    app = AppTest.from_file(str(APP)).run()
    assert not calls
    _widget(app.text_input, "provider_api_key").set_value("fake-one").run()
    assert next(w for w in app.multiselect if w.key.startswith("check_candidates_")).options == ["model-1"]
    _widget(app.number_input, "provider_rpm").set_value(15).run()
    assert len(calls) == 1
    _widget(app.text_input, "provider_api_key").set_value("fake-two").run()
    assert len(calls) == 2
    assert next(w for w in app.multiselect if w.key.startswith("check_candidates_")).options == ["model-2"]
    assert not app.exception


def test_only_models_that_pass_labeling_are_selectable(monkeypatch):
    from types import SimpleNamespace
    from src.providers.base import ProviderError
    from src.ui.model_checks import verified_for_config

    def create(name, key, model, timeout):
        def generate(prompt, schema):
            if model == "blocked-model":
                raise ProviderError("Model unavailable (404)", kind="model_not_found", status_code=404)
            return MockProvider().generate_structured(prompt, schema)
        return SimpleNamespace(list_models=lambda: ["good-model", "blocked-model"], generate_structured=generate)

    monkeypatch.setattr(provider, "create_provider", create)
    app = AppTest.from_file(str(APP)).run()
    _widget(app.text_input, "provider_api_key").set_value("fake-key").run()
    assert not any(w.key == "selected_model" for w in app.selectbox)

    next(w for w in app.multiselect if w.key.startswith("check_candidates_")).set_value(["blocked-model", "good-model"]).run()
    _widget(app.button, "verify_label_models").click().run()
    assert not app.exception
    assert _widget(app.selectbox, "selected_model").options == ["good-model"]
    assert any("404" in str(table.value) for table in app.dataframe)
    _widget(app.button, "save_provider_config").click().run()
    assert app.session_state["provider_config"]["model"] == "good-model"
    _widget(app.radio, "workflow_page").set_value("3. Chọn tác vụ").run()
    _widget(app.multiselect, "selected_tasks").set_value(["detection", "type"]).run()
    _widget(app.radio, "workflow_page").set_value("1. Kết nối AI").run()
    assert not any(w.key == "selected_model" for w in app.selectbox)
    next(w for w in app.multiselect if w.key.startswith("check_candidates_")).set_value(["good-model"]).run()
    _widget(app.button, "verify_label_models").click().run()
    assert _widget(app.selectbox, "selected_model").options == ["good-model"]
    _widget(app.text_input, "provider_api_key").set_value("another-fake-key").run()
    assert not any(w.key == "selected_model" for w in app.selectbox)


def test_one_click_auto_setup_saves_verified_connection(monkeypatch):
    import time
    from src.ui import model_scan
    scan_function = model_scan.scan_models
    monkeypatch.setattr(model_scan, "scan_models", lambda *args: scan_function(*args, sleep=lambda _: None, interval=0))
    from types import SimpleNamespace
    from src.providers.base import ProviderError
    calls = []

    def factory(name, key, model, timeout):
        def generate(prompt, schema):
            calls.append(model)
            if model == "a-unavailable":
                raise ProviderError("404 model unavailable", status_code=404)
            return MockProvider().generate_structured(prompt, schema)
        return SimpleNamespace(list_models=lambda: ["a-unavailable", "b-working", "c-unused"], generate_structured=generate)

    monkeypatch.setattr(provider, "create_provider", factory)
    app = AppTest.from_file(str(APP)).run()
    _widget(app.text_input, "provider_api_key").set_value("fake-key").run()
    _widget(app.button, "auto_setup_provider").click().run()
    from src.ui.model_checks import check_context
    context = check_context("gemini", "fake-key", ["detection"], "independent")
    deadline = time.monotonic() + 3
    while model_scan.scan_snapshot(context)["running"] and time.monotonic() < deadline:
        time.sleep(0.01)
    app.run()
    assert not app.exception
    assert calls == ["a-unavailable", "b-working", "c-unused"]
    assert app.session_state["provider_config"]["model"] == "b-working"
    assert _widget(app.selectbox, "selected_model").options == ["b-working", "c-unused"]
    assert local_memory.load_provider()["model"] == "b-working"


def test_navigation_preserves_provider_draft_and_saved_config(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "DEFAULT_DB_PATH", tmp_path / "jobs.sqlite")
    monkeypatch.setattr(results, "DEFAULT_DB_PATH", tmp_path / "jobs.sqlite")
    app = AppTest.from_file(str(APP)).run()
    _widget(app.selectbox, "provider_label").select("Mô phỏng ngoại tuyến").run()
    _widget(app.button, "save_provider_config").click().run()
    assert app.session_state["provider_config"]["provider"] == "mock"
    _widget(app.number_input, "provider_rpm").set_value(15).run()
    assert any("chưa lưu" in warning.value for warning in app.warning)
    for page in ("2. Nạp dữ liệu", "3. Chọn tác vụ", "4. Chạy gán nhãn", "5. Xem & tải kết quả", "1. Kết nối AI"):
        _widget(app.radio, "workflow_page").set_value(page).run()
        assert not app.exception
    assert _widget(app.number_input, "provider_rpm").value == 15
    assert app.session_state["provider_config"]["requests_per_minute"] == 10


def test_dataset_mapping_survives_navigation():
    app = AppTest.from_file(str(APP)).run()
    app.session_state["dataset_frame"] = pd.DataFrame({"custom_title": ["Tiêu đề thử"], "body": ["Nội dung"]})
    app.session_state["dataset_filename"] = "test.csv"
    _widget(app.radio, "workflow_page").set_value("2. Nạp dữ liệu").run()
    _widget(app.selectbox, "dataset_map_title").select("custom_title").run()
    assert app.session_state["dataset_records"][0]["title"] == "Tiêu đề thử"
    _widget(app.radio, "workflow_page").set_value("3. Chọn tác vụ").run()
    _widget(app.radio, "workflow_page").set_value("2. Nạp dữ liệu").run()
    assert not app.exception
    assert _widget(app.selectbox, "dataset_map_title").value == "custom_title"
    assert app.session_state["dataset_records"][0]["title"] == "Tiêu đề thử"


def test_existing_job_progress_and_export_render(monkeypatch, tmp_path):
    db = tmp_path / "jobs.sqlite"
    monkeypatch.setattr(jobs, "DEFAULT_DB_PATH", db)
    monkeypatch.setattr(results, "DEFAULT_DB_PATH", db)
    store = JobStore(db)
    job_id = store.create_job(
        [{"record_id": "r1", "title": "Tin thử", "_original": {"title": "Tin thử"}}],
        ["detection"], provider="mock", model="mock-v1", mode="independent",
        prompt_version="v1", taxonomy_version="v1", schema_version="v1", settings={},
    )
    app = AppTest.from_file(str(APP)).run()
    _widget(app.radio, "workflow_page").set_value("4. Chạy gán nhãn").run()
    assert not app.exception
    assert app.session_state["selected_job_id"] == job_id
    assert any(metric.label == "Thành công" for metric in app.metric)
    _widget(app.radio, "workflow_page").set_value("5. Xem & tải kết quả").run()
    assert not app.exception
    assert any(button.label == "Tải toàn bộ kết quả (.zip)" for button in app.get("download_button"))


def test_new_session_restores_saved_provider_and_dataset():
    app = AppTest.from_file(str(APP)).run()
    _widget(app.text_input, "provider_api_key").set_value("fake-secret-for-test").run()
    _widget(app.text_input, "provider_model_before_connection").set_value("test-model").run()
    _widget(app.button, "verify_label_models").click().run()
    _widget(app.button, "save_provider_config").click().run()
    assert not app.exception
    encrypted = (local_memory.STATE_DIR / "provider.dpapi").read_bytes()
    assert b"fake-secret-for-test" not in encrypted
    app.session_state["dataset_frame"] = pd.DataFrame({"custom": ["Tiêu đề tiếng Việt"], "label": [0]})
    app.session_state["dataset_filename"] = "news.xlsx"
    _widget(app.radio, "workflow_page").set_value("2. Nạp dữ liệu").run()
    _widget(app.selectbox, "dataset_map_title").select("custom").run()
    _widget(app.button, "remember_dataset").click().run()
    restored = AppTest.from_file(str(APP)).run()
    assert not restored.exception
    assert _widget(restored.text_input, "provider_api_key").value == "fake-secret-for-test"
    assert restored.session_state["provider_config"]["model"] == "test-model"
    assert restored.session_state["dataset_records"][0]["title"] == "Tiêu đề tiếng Việt"
    assert restored.session_state["dataset_frame"]["label"].tolist() == [0]
    _widget(restored.button, "forget_provider").click().run()
    _widget(restored.radio, "workflow_page").set_value("2. Nạp dữ liệu").run()
    _widget(restored.button, "forget_dataset").click().run()
    fresh = AppTest.from_file(str(APP)).run()
    assert not fresh.exception
    assert "provider_config" not in fresh.session_state
    assert "dataset_frame" not in fresh.session_state


def test_saved_model_verdict_restores_and_step4_fresh_check_invalidates(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from src.providers.base import ProviderError
    from src.ui.model_checks import check_context
    monkeypatch.setattr(jobs, "DEFAULT_DB_PATH", tmp_path / "jobs.sqlite")
    config = {"provider": "gemini", "model": "saved-model", "api_key": "fake-test",
              "timeout_seconds": 5, "requests_per_minute": 10, "concurrency": 1, "max_retries": 0}
    local_memory.save_provider(config)
    context = check_context("gemini", "fake-test", ["detection"], "independent")
    local_memory.save_model_results(context, {"saved-model": {"ok": True, "detail": "previous check"}})
    app = AppTest.from_file(str(APP)).run()
    assert _widget(app.selectbox, "selected_model").options == ["saved-model"]
    _widget(app.radio, "workflow_page").set_value("4. Chạy gán nhãn").run()

    def fail(prompt, schema):
        raise ProviderError("model unavailable now", status_code=404)
    monkeypatch.setattr(provider, "create_provider", lambda *args: SimpleNamespace(generate_structured=fail))
    _widget(app.button, "check_current_model").click().run()
    assert not app.exception
    assert any("unavailable now" in error.value for error in app.error)
    assert not local_memory.load_model_results(context)["saved-model"]["ok"]
    fresh = AppTest.from_file(str(APP)).run()
    assert not any(widget.key == "selected_model" for widget in fresh.selectbox)
    assert not fresh.exception


def test_job_selection_is_shared_between_history_and_results(monkeypatch, tmp_path):
    db = tmp_path / "jobs.sqlite3"
    monkeypatch.setattr(jobs, "DEFAULT_DB_PATH", db)
    monkeypatch.setattr(results, "DEFAULT_DB_PATH", db)
    store = JobStore(db)
    ids = [store.create_job([{"record_id": f"r{i}", "title": f"News {i}"}], ["detection"],
             provider="mock", model="mock-v1", mode="independent", prompt_version="v1", taxonomy_version="v1",
             schema_version="v1", settings={}) for i in range(2)]
    app = AppTest.from_file(str(APP)).run()
    _widget(app.radio, "workflow_page").set_value("4. Chạy gán nhãn").run()
    _widget(app.selectbox, "job_selector").set_value(ids[0]).run()
    _widget(app.radio, "workflow_page").set_value("5. Xem & tải kết quả").run()
    assert _widget(app.selectbox, "results_job_selector").value == ids[0]
    _widget(app.selectbox, "results_job_selector").set_value(ids[1]).run()
    _widget(app.radio, "workflow_page").set_value("4. Chạy gán nhãn").run()
    assert _widget(app.selectbox, "job_selector").value == ids[1]
    assert not app.exception


def test_tasks_and_mode_restore_in_new_session():
    app = AppTest.from_file(str(APP)).run()
    _widget(app.radio, "workflow_page").set_value("3. Chọn tác vụ").run()
    _widget(app.multiselect, "selected_tasks").set_value(["detection", "type"]).run()
    _widget(app.radio, "labeling_mode_label").set_value("Gộp trong một yêu cầu").run()
    fresh = AppTest.from_file(str(APP)).run()
    assert fresh.session_state["selected_tasks"] == ["detection", "type"]
    assert fresh.session_state["labeling_mode"] == "combined"
    assert not fresh.exception


def test_provider_switch_does_not_send_previous_providers_key(monkeypatch):
    from types import SimpleNamespace
    calls = []
    def create(name, key, model, timeout):
        calls.append((name, key))
        return SimpleNamespace(list_models=lambda: [])
    monkeypatch.setattr(provider, "create_provider", create)
    app = AppTest.from_file(str(APP)).run()
    _widget(app.text_input, "provider_api_key").set_value("fake-gemini-only").run()
    _widget(app.selectbox, "provider_label").select("OpenAI").run()
    assert _widget(app.text_input, "provider_api_key").value == ""
    assert calls == [("gemini", "fake-gemini-only")]
    _widget(app.text_input, "provider_api_key").set_value("fake-openai-only").run()
    _widget(app.selectbox, "provider_label").select("Gemini").run()
    assert _widget(app.text_input, "provider_api_key").value == "fake-gemini-only"
    assert ("openai", "fake-gemini-only") not in calls
