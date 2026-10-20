"""Explicit full account probe: every listed model, one delayed retry for 429."""
import json
from pathlib import Path
import re
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ui.local_memory import load_provider
from src.ui.provider import create_provider
from src.ui.auto_setup import SAMPLE
from src.labeling.engine import LabelingEngine


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    config = load_provider()
    key, provider = config["api_key"], config["provider"]
    names = sorted(set(create_provider(provider, key, "model-check", 20).list_models()))
    target = Path(__file__).resolve().parents[1] / "state" / "model_probe_all.json"
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "provider": provider,
              "tasks": ["detection"], "listed_count": len(names), "complete": False, "results": []}
    last_call = 0
    retry_queue = []

    def save():
        report["passed_count"] = sum(item.get("passed", False) for item in report["results"])
        report["tested_count"] = len(report["results"])
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    def probe(name, attempt):
        nonlocal last_call
        time.sleep(max(0, 6 - (time.monotonic() - last_call)))
        last_call = time.monotonic()
        print(f"TEST attempt={attempt} model={name}", flush=True)
        try:
            execution = LabelingEngine(create_provider(provider, key, name, 20)).label_record(SAMPLE, ["detection"])
            result = {"passed": True, "output": execution.outputs["detection"].model_dump(), "status_code": None}
        except Exception as exc:
            raw = exc.__cause__ or exc
            detail = str(raw).replace(key, "[REDACTED]")[:6000]
            result = {"passed": False, "status_code": getattr(exc, "status_code", None), "detail": detail}
        result["at"] = datetime.now(timezone.utc).isoformat()
        print(f"RESULT model={name} attempt={attempt} passed={result['passed']} code={result['status_code']}", flush=True)
        return result

    save()
    print(f"START total={len(names)} interval=6s retry429=once_after_60s_or_server_delay", flush=True)
    for index, name in enumerate(names):
        result = probe(name, 1)
        entry = {"model": name, "attempts": [result], "passed": result["passed"]}
        report["results"].append(entry)
        if result["status_code"] == 429:
            match = re.search(r"(?:retry in\s*|retrydelay[^0-9]*)(\d+(?:\.\d+)?)", result["detail"], re.I)
            delay = max(60, float(match[1]) + 1) if match else 60
            entry["retry_after_seconds"] = delay
            retry_queue.append((time.monotonic() + delay, entry))
        save()
        print(f"PROGRESS first_pass={index+1}/{len(names)} passed={report['passed_count']} queued_retry={len(retry_queue)}", flush=True)
    for index, (ready_at, entry) in enumerate(retry_queue):
        while time.monotonic() < ready_at:
            remaining = ready_at - time.monotonic()
            print(f"WAIT model={entry['model']} seconds={int(remaining)+1}", flush=True)
            time.sleep(min(remaining, 30))
        result = probe(entry["model"], 2)
        entry["attempts"].append(result)
        entry["passed"] = result["passed"]
        save()
        print(f"PROGRESS retries={index+1}/{len(retry_queue)} passed={report['passed_count']}", flush=True)
    report["complete"] = True
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    save()
    print(f"DONE tested={report['tested_count']} passed={report['passed_count']} report={target}", flush=True)


if __name__ == "__main__":
    main()
