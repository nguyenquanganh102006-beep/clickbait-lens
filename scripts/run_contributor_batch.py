"""Resumable, checkpointed Gemini batch for one CSV contributor."""
from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import os
import sys
import tempfile
import time
import contextlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.labeling.quality import VERSION, quality_input_hash, result_hash, reviewed_record
from src.providers.base import ProviderError, sanitize_error_message

BASE = ROOT.parent / "data_pr" / "clean_final"
DEFAULT_CSV = BASE / "data.csv"
DEFAULT_OUT = ROOT / "data" / "output" / "tuan_run"
OBSOLETE = {"type_misleading_framing", "type_image_text_mismatch", "type_context_manipulation"}


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        for attempt in range(8):
            try:
                os.replace(name, path)
                break
            except PermissionError:
                if attempt == 7: raise
                time.sleep(min(.05 * 2**attempt, .5))
    finally:
        if os.path.exists(name): os.unlink(name)


def atomic_json(path: Path, obj: dict) -> None:
    atomic_bytes(path, (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def atomic_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    import io
    b = io.StringIO(newline="")
    w = csv.DictWriter(b, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    atomic_bytes(path, b"\xef\xbb\xbf" + b.getvalue().encode("utf-8"))


def read_contributor(path: Path, contributor: str) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "contributor" not in reader.fieldnames or "article_id" not in reader.fieldnames:
            raise ValueError("Input CSV must include contributor and article_id columns")
        return [dict(r) for r in reader if r.get("contributor", "").strip() == contributor]


def load_jsonl(path: Path) -> dict[str, dict]:
    found = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip(): continue
                obj = json.loads(line); row = obj.get("result")
                if not isinstance(row, dict) or not row.get("article_id"): raise ValueError(f"Invalid checkpoint line {n}")
                if row["article_id"] in found: raise ValueError("Duplicate article_id in checkpoint")
                found[row["article_id"]] = row
    return found


def append_checkpoint(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"article_id": row["article_id"], "result": row}, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())


def validate_result(row: dict, record: dict, image_root: Path, provider) -> None:
    if row.get("annotation_quality_mode") != VERSION: raise ValueError("checkpoint schema/quality version mismatch")
    if row.get("annotation_input_hash") != quality_input_hash(record, image_root, provider): raise ValueError("checkpoint input/model/prompt/schema/image hash mismatch")
    if row.get("annotation_requested_model") != getattr(provider, "model", ""): raise ValueError("checkpoint model mismatch")
    if row.get("annotation_result_hash") != result_hash(row): raise ValueError("checkpoint result hash mismatch")


def is_full(row: dict) -> bool:
    try: masks = json.loads(row.get("label_mask_json", "{}"))
    except (TypeError, json.JSONDecodeError): return False
    required = {"detection", "curiosity_gap", "exaggeration", "emotional_bait", "question_based", "misleading_presentation", "consistency_tc", "consistency_ti", "consistency_ic"}
    return required.issubset(masks) and all(masks[k] is True for k in required) and row.get("annotation_status") == "pseudo_labeled"


def load_leader_audit(path: Path) -> dict[str, str]:
    audit = path / "leader_audit.jsonl"
    statuses = {}
    if audit.exists():
        with audit.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip(): continue
                item = json.loads(line)
                aid = item.get("article_id")
                status = item.get("review_status")
                if not aid or status not in {"accepted", "disputed", "quarantined"}:
                    raise ValueError(f"Invalid leader audit receipt at line {line_no}")
                statuses[aid] = status
    return statuses


def export_all(out: Path, ordered: list[dict], progress: dict, source_fields: list[str]) -> None:
    slim = [{k: v for k, v in r.items() if k not in OBSOLETE} for r in ordered]
    fields = list(dict.fromkeys([k for k in source_fields if k not in OBSOLETE] + [k for r in slim for k in r]))
    leader = load_leader_audit(out)
    held = {aid for aid, s in leader.items() if s in {"disputed", "quarantined"}}
    atomic_csv(out / "completed.csv", [r for r in slim if is_full(r) and r["article_id"] not in held], fields)
    atomic_csv(out / "progress.csv", slim, fields)
    atomic_csv(out / "review_queue.csv", [r for r in slim if not is_full(r) or r["article_id"] in held], fields)
    sample = [r for r in slim if int(hashlib.sha256(r["article_id"].encode()).hexdigest(), 16) % 20 == 0]
    atomic_bytes(out / "review_sample.jsonl", b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in sample))
    atomic_json(out / "progress.json", progress)


def run(args, provider=None) -> int:
    records = read_contributor(args.csv, args.contributor)
    by_id = {r["article_id"]: r for r in records}
    if len(by_id) != len(records): raise ValueError("Duplicate article_id in selected contributor")
    with args.csv.open(encoding="utf-8-sig", newline="") as f:
        source_fields = list(csv.DictReader(f).fieldnames or [])
    if args.seed_csv:
        with args.seed_csv.open(encoding="utf-8-sig", newline="") as f: seeds = list(csv.DictReader(f))
    else: seeds = []
    if provider is None:
        from src.providers.gemini import GeminiProvider
        if args.key_stdin:
            print("Reading ephemeral Gemini key (hidden; never saved)", flush=True)
            key = getpass.getpass("Gemini API key: ") if sys.stdin.isatty() else sys.stdin.readline().strip()
        else: raise ValueError("Use --key-stdin")
        if not key: raise ValueError("Empty API key")
        provider = GeminiProvider(key, args.model, args.timeout)
    provider.requests_per_minute = args.rpm
    args.output.mkdir(parents=True, exist_ok=True)
    lock_path = args.output / ".runner.lock"
    lock_file = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            lock_file.seek(0); lock_file.write(b"0"); lock_file.flush(); lock_file.seek(0)
            try: msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc: raise RuntimeError("Another batch runner holds this output directory") from exc
        else:
            import fcntl
            try: fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc: raise RuntimeError("Another batch runner holds this output directory") from exc
    except BaseException:
        lock_file.close(); raise
    checkpoint = args.output / "checkpoint.jsonl"
    results = load_jsonl(checkpoint)
    for row in seeds:
        aid = row.get("article_id")
        if aid not in by_id: raise ValueError(f"Seed ID not in contributor input: {aid}")
        validate_result(row, by_id[aid], args.image_root, provider)
        if aid in results and results[aid] != row: raise ValueError(f"Seed conflicts with checkpoint: {aid}")
        if aid not in results: append_checkpoint(checkpoint, row); results[aid] = row
    for aid, row in results.items():
        if aid not in by_id: raise ValueError(f"Checkpoint ID not in selected input: {aid}")
        validate_result(row, by_id[aid], args.image_root, provider)
    errors = 0; attempted_articles = 0; calls = sum(int(r.get("annotation_model_calls", 0) or 0) for r in results.values())
    reported_tokens = sum(int(r.get("annotation_total_tokens", 0) or 0) for r in results.values())
    status = "running"; last_safe_error = None; last_id = None; processed_since_export = 0
    ordered = [results[r["article_id"]] for r in records if r["article_id"] in results]
    progress = {}
    def save(force=False):
        nonlocal processed_since_export, ordered, progress
        ordered = [results[r["article_id"]] for r in records if r["article_id"] in results]
        leader = load_leader_audit(args.output)
        held = {aid for aid, s in leader.items() if s in {"disputed", "quarantined"}}
        progress = dict(attempted=attempted_articles, attempted_articles=attempted_articles, processed_articles=len(ordered), completed=sum(is_full(r) and r["article_id"] not in held for r in ordered), needs_review=sum(not is_full(r) or r["article_id"] in held for r in ordered), errors=errors,
                total_rows=len(records), remaining=len(records)-len(results), reported_tokens=reported_tokens, calls=calls, reported_model_calls=calls,
                leader_reviewed=sum(aid in {r["article_id"] for r in ordered} for aid in leader), disputed=sum(1 for aid, s in leader.items() if s == "disputed" and aid in {r["article_id"] for r in ordered}), quarantined=sum(1 for aid, s in leader.items() if s == "quarantined" and aid in {r["article_id"] for r in ordered}),
                status=status, last_id=last_id, last_safe_error=last_safe_error, heartbeat=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        atomic_json(args.output / "progress.json", progress)
        if force or processed_since_export >= 10:
            export_all(args.output, ordered, progress, source_fields); processed_since_export = 0
    save(True)
    transient = 0
    traversed = 0
    for rec in records:
        aid = rec["article_id"]
        if aid in results: continue
        if args.max_records is not None and traversed >= args.max_records: break
        traversed += 1
        attempted_articles += 1; last_id = aid
        retry = 0
        while True:
          try:
            row = reviewed_record(rec, provider, args.image_root)
            append_checkpoint(checkpoint, row); results[aid] = row
            calls += int(row.get("annotation_model_calls", 0) or 0)
            reported_tokens += int(row.get("annotation_total_tokens", 0) or 0); transient = 0
            print(json.dumps({"article_id": aid, "status": row.get("annotation_status"), "calls": row.get("annotation_model_calls"), "tokens": row.get("annotation_total_tokens")}, ensure_ascii=False), flush=True)
            break
          except ProviderError as exc:
            msg = sanitize_error_message(exc, getattr(provider, "api_key", "")); last_safe_error = msg; errors += 1
            calls += int(getattr(exc, "attempt_count", 1) or 1)
            if exc.quota_exhausted or exc.kind == "authentication": status = "quota_exhausted" if exc.quota_exhausted else "authentication_error"; save(True); return 2
            if exc.kind == "rate_limit":
                if retry < args.rate_retries:
                    retry += 1
                    delay = exc.retry_after_seconds if exc.retry_after_seconds is not None else min(2 ** retry, 30)
                    time.sleep(max(0, delay)); continue
            transient += 1
            print(json.dumps({"article_id": aid, "error": msg}, ensure_ascii=False), flush=True)
            if transient >= 3:
                status = "infrastructure_error"; save(True); return 3
            break
          except Exception as exc:
            last_safe_error = sanitize_error_message(exc, getattr(provider, "api_key", "")); errors += 1; transient += 1
            print(json.dumps({"article_id": aid, "error": last_safe_error}, ensure_ascii=False), flush=True)
            if transient >= 3: status = "infrastructure_error"; save(True); return 3
            break
        processed_since_export += 1; save()
    status = "completed" if len(results) == len(records) else "pilot_limit" if args.max_records is not None and traversed >= args.max_records else "completed_with_errors"
    save(True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV); p.add_argument("--image-root", type=Path, default=BASE)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT); p.add_argument("--contributor", default="Nguyễn Minh Tuấn")
    p.add_argument("--model", default="gemini-3.1-flash-lite"); p.add_argument("--key-stdin", action="store_true")
    p.add_argument("--timeout", type=int, default=90); p.add_argument("--rpm", type=int, default=10)
    p.add_argument("--seed-csv", type=Path); p.add_argument("--max-records", type=int); p.add_argument("--rate-retries", type=int, default=3)
    args = p.parse_args(argv)
    if args.timeout < 1 or args.rpm < 1 or args.rate_retries < 0: p.error("timeout/rpm must be positive and retries nonnegative")
    if args.max_records is not None and args.max_records < 0: p.error("max-records must be nonnegative")
    return run(args)


if __name__ == "__main__": raise SystemExit(main())
