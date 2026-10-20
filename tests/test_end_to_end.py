"""Exercise actual loaders, normalization, engine, SQLite runner and exports."""
from io import BytesIO
import json

import pandas as pd
import pytest

from src.data.loader import load_dataset
from src.data.mapping import normalize_records, suggest_column_mapping
from src.data.exporter import build_export_files
from src.jobs.runner import JobRunner
from src.jobs.storage import JobStore
from src.labeling.prompts import PROMPT_VERSION, SCHEMA_VERSION, load_taxonomy
from src.providers.mock import MockProvider


@pytest.mark.parametrize("extension", ["csv", "xlsx", "jsonl"])
@pytest.mark.parametrize("mode", ["independent", "combined"])
def test_import_all_tasks_run_and_export_preserving_source(extension, mode, tmp_path):
    source = pd.DataFrame({
        "id": ["r1", "r2", "r3"],
        "title": ["Thư viện mở cửa", None, "Nội dung gây bất ngờ?"],
        "content": ["Thư viện mở cửa từ 8 giờ sáng đến 17 giờ chiều. " * 5, None, None],
        "lead": [None, None, "Sapo không phải toàn văn"],
        "label": [0, 1, 1],
        "source": ["Báo thử", "Báo thử", "Nguồn khác"],
    })
    if extension == "csv":
        payload = source.to_csv(index=False).encode("utf-8-sig")
    elif extension == "xlsx":
        buffer = BytesIO()
        source.to_excel(buffer, index=False)
        payload = buffer.getvalue()
    else:
        payload = source.to_json(orient="records", lines=True, force_ascii=False).encode("utf-8")
    frame = load_dataset(payload, filename=f"test.{extension}")
    records = normalize_records(frame, suggest_column_mapping(list(frame.columns)))
    store = JobStore(tmp_path / "jobs.sqlite3")
    settings = {"concurrency": 2, "requests_per_minute": 600, "max_retries": 0}
    job_id = store.create_job(records, ["detection", "type", "consistency"], provider="mock", model="mock-v1",
                              mode=mode, prompt_version=PROMPT_VERSION, schema_version=SCHEMA_VERSION,
                              taxonomy_version=load_taxonomy()["version"], settings=settings)
    JobRunner(store, MockProvider(), settings).run(job_id)
    job = store.get_job(job_id)
    assert job["status"] == "completed"
    assert store.get_progress(job_id)["completed"] == 9
    results = store.get_record_results(job_id)
    by_id = {row["record_id"]: row for row in results}
    assert by_id["r2"]["tasks"]["detection"]["status"] == "insufficient_data"
    assert by_id["r3"]["tasks"]["consistency"]["status"] == "insufficient_data"
    assert by_id["r1"]["tasks"]["consistency"]["status"] == "success"
    exports = build_export_files(job, results)
    assert len(exports) == 5
    output = pd.read_csv(BytesIO(exports["labeled_dataset.csv"]))
    assert output["label"].tolist() == [0, 1, 1]
    assert output["id"].tolist() == ["r1", "r2", "r3"]
    assert len([json.loads(line) for line in exports["predictions.jsonl"].decode().splitlines()]) == 3
    assert json.loads(exports["summary.json"])["records_exported"] == 3
    assert pd.read_csv(BytesIO(exports["errors.csv"])).empty
    assert not pd.read_csv(BytesIO(exports["review_queue.csv"])).empty
