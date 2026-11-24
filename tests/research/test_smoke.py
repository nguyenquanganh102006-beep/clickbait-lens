import json
from pathlib import Path
from scripts.smoke_research import run

ROOT = Path(__file__).resolve().parents[2]


def test_synthetic_smoke_end_to_end(tmp_path):
    path = run(ROOT / "configs/smoke.yaml", synthetic=True, output_root=tmp_path)
    metadata = json.loads((path / "run.json").read_text())
    assert metadata["dataset_version"] == "synthetic" and not metadata["official_benchmark"]
    result = tmp_path / "results" / path.name
    metrics = json.loads((result / "metrics.json").read_text())
    assert metrics["label_origin"] == "synthetic" and metrics["n"] == 8
    assert len((result / "predictions.jsonl").read_text().splitlines()) == 8
    events = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "complete" and events[-1]["checkpoint_roundtrip"]
