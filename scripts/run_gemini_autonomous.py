"""One command: whole-contributor queue, Gemini worker, live CSV publication."""
import argparse
import getpass
import hashlib
import json
import re
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_parallel_contributor as worker
from scripts.publish_contributor_labels import publish
from scripts.assemble_fast_agents import assemble
from scripts.run_contributor_batch import atomic_json

def timestamp():
    return datetime.now(timezone(timedelta(hours=7))).isoformat()

def model_worker_dir(args):
    if args.model=='gemini-3.5-flash-lite':return args.run_dir/'fast_run'
    slug=re.sub(r'[^a-zA-Z0-9_-]+','_',args.model)[:80]
    suffix=hashlib.sha256(args.model.encode()).hexdigest()[:8]
    return args.run_dir/'model_runs'/f'{slug}-{suffix}'

def execute(args,provider_factory,*,worker_run=None,publish_fn=None,assemble_fn=None,poll_seconds=3):
    """Key stays inside the provider factory; it is never passed to a child process."""
    worker_run=worker_run or worker.run
    publish_fn=publish_fn or publish
    assemble_fn=assemble_fn or assemble
    run_dir=args.run_dir;run_dir.mkdir(parents=True,exist_ok=True)
    # No packet, slice, article limit, or manually reserved ID list.
    settings=SimpleNamespace(csv=args.csv,contributor=args.contributor,image_root=args.image_root,
        output=model_worker_dir(args),skip_published=args.destination,model=args.model,
        timeout=args.timeout,rpm=args.rpm,workers=args.workers,record_attempts=3,
        seed_csv=None,key_stdin=False,mode='fast',canonical_titles=run_dir/'source_title_review_holds.json',reserved_ids=None)
    stop=threading.Event();failures=[];last_signature=None;last_votes=None;last_report={}

    def snapshot():
        nonlocal last_signature,last_votes,last_report
        vote_paths=sorted(run_dir.glob('luna_fast_*.jsonl'))
        vote_signature=tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in vote_paths)
        if vote_signature!=last_votes:
            assemble_fn(run_dir,args.csv,args.image_root,args.contributor)
            last_votes=vote_signature
        checkpoints=[run_dir/'fast_run/checkpoint.jsonl',*sorted((run_dir/'model_runs').glob('*/checkpoint.jsonl'))]
        paths=[*run_dir.glob('*.jsonl'),*checkpoints,settings.output/'progress.json']
        signature=tuple((str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths if p.exists())
        if signature==last_signature:return
        if not any(p.exists() and p.stat().st_size for p in [run_dir/'checkpoint.jsonl',*checkpoints,run_dir/'fast_agents_checkpoint.jsonl']):return
        last_report=publish_fn(run_dir,args.destination)
        progress_path=settings.output/'progress.json'
        progress=json.loads(progress_path.read_text(encoding='utf8')) if progress_path.exists() else {}
        status=dict(timestamp=timestamp(),model=args.model,worker=progress,publication=last_report,
                    input=str(args.csv),output=str(args.destination),worker_directory=str(settings.output),manual_batch_assignment_required=False)
        atomic_json(run_dir/'autonomous_status.json',status)
        event=dict(timestamp=status['timestamp'],model=args.model,status=progress.get('status','starting'),
                   processed=progress.get('processed_articles',0),published=last_report.get('published',0),
                   needs_review=last_report.get('needs_review',0))
        with (run_dir/'autonomous_events.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(event)+'\n')
        # Select a deterministic ~5% sample; this does not claim it was reviewed.
        checkpoint=settings.output/'checkpoint.jsonl'
        if checkpoint.exists():
            from scripts.publish_contributor_labels import read_lines
            rows=[x['result'] for x in read_lines(checkpoint)]
            atomic_json(run_dir/'autonomous_review_sample.json',dict(reviewed=False,
                article_ids=[r['article_id'] for r in rows[19::20]],sampling='every twentieth checkpoint; leader review pending'))
        last_signature=signature

    def monitor():
        while not stop.is_set():
            try:snapshot()
            except Exception as exc:
                # Preserve worker checkpoints; retry publication without another API call.
                failures.append(type(exc).__name__)
                print(json.dumps({'publisher_retry':type(exc).__name__}),flush=True)
            stop.wait(poll_seconds)

    thread=threading.Thread(target=monitor,name='csv-publisher',daemon=True)
    thread.start()
    try:
        code=worker_run(settings,provider_factory=provider_factory)
    finally:
        stop.set();thread.join()
        snapshot()  # A final publication must succeed before reporting completion.
    progress=json.loads((settings.output/'progress.json').read_text(encoding='utf8'))
    summary=dict(timestamp=timestamp(),status=progress['status'],exit_code=code,model=args.model,
        processed=progress.get('processed_articles',0),published=last_report.get('published',0),
        needs_review=last_report.get('needs_review',0),output=str(args.destination),publisher_retries=len(failures))
    atomic_json(run_dir/'autonomous_final_report.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return code

def main():
    p=argparse.ArgumentParser(description=__doc__)
    base=ROOT.parent/'data_pr/clean_final'
    p.add_argument('--csv',type=Path,default=base/'data.csv')
    p.add_argument('--destination',type=Path,default=base/'data_tuan_labeled.csv')
    p.add_argument('--image-root',type=Path,default=base)
    p.add_argument('--run-dir',type=Path,default=ROOT/'data/output/tuan_run')
    p.add_argument('--contributor',default='Nguyễn Minh Tuấn')
    p.add_argument('--model',default='gemini-3.5-flash-lite')
    p.add_argument('--workers',type=int,choices=range(1,5),default=4)
    p.add_argument('--rpm',type=int,default=10)
    p.add_argument('--timeout',type=int,default=90)
    args=p.parse_args()
    if min(args.rpm,args.timeout)<1:p.error('rpm and timeout must be positive')
    key=getpass.getpass('Gemini API key (hidden, never saved): ')
    if not key:p.error('API key is required')
    from src.providers.gemini import GeminiProvider
    # One owner prevents two autonomous publishers from writing the same CSV.
    lock=worker._acquire_output_lock(args.run_dir/'autonomous_owner')
    try:return execute(args,lambda:GeminiProvider(key,args.model,args.timeout))
    finally:lock.close()

if __name__=='__main__':raise SystemExit(main())
