"""Explicit small, paid-API integration check using the locally saved provider."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ui.local_memory import load_provider
from src.labeling.probe import SAMPLE
from src.labeling.prompts import PROMPT_VERSION, SCHEMA_VERSION, load_taxonomy
from src.ui.provider import create_provider
from src.jobs.runner import JobRunner
from src.jobs.storage import JobStore
from src.data.exporter import build_export_files


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    config = load_provider()
    if not config or config["provider"] == "mock" or not config.get("api_key"):
        raise RuntimeError("Save a real provider key before explicitly running this integration check")
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "provider": config["provider"],
              "model": config["model"], "data": "one built-in synthetic article", "checks": []}
    with tempfile.TemporaryDirectory(prefix="khdl-live-") as directory:
        store = JobStore(Path(directory) / "jobs.sqlite3")
        for mode in ["independent", "combined"]:
            print(f"TEST mode={mode} model={config['model']} tasks=detection,type,consistency", flush=True)
            runtime = {"requests_per_minute": 10, "concurrency": 1, "max_retries": 1}
            record = {**SAMPLE, "_original": {"title": SAMPLE["title"], "content": SAMPLE["content"], "label": 0}}
            job_id = store.create_job([record], ["detection", "type", "consistency"], mode=mode,
                provider=config["provider"], model=config["model"], prompt_version=PROMPT_VERSION,
                taxonomy_version=load_taxonomy()["version"], schema_version=SCHEMA_VERSION, settings=runtime)
            provider = create_provider(config["provider"], config["api_key"], config["model"], 45)
            JobRunner(store, provider, runtime).run(job_id)
            progress = store.get_progress(job_id)
            results = store.get_record_results(job_id)
            exports = build_export_files(store.get_job(job_id), results)
            report["checks"].append({"mode": mode, "status": store.get_job(job_id)["status"],
                "passed": progress.get("success") == 3, "progress": progress,
                "attempts": sum(item["attempts"] for item in store.get_items(job_id)),
                "outputs": {task: result["output"] for task, result in results[0]["tasks"].items()},
                "errors": [item.get("error_message") for item in store.get_items(job_id) if item.get("error_message")],
                "export_files": list(exports)})
            print(f"RESULT mode={mode} passed={report['checks'][-1]['passed']} progress={progress}", flush=True)
    report["passed"] = all(check["passed"] for check in report["checks"])
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    target = ROOT / "state" / "live_smoke_report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"FINISHED passed={report['passed']} report={target}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
