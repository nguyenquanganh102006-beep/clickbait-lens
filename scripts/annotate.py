"""Small, bounded pilot; existing clean CSVs are never modified."""
import argparse
import json
import sys
import os
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.labeling.multimodal import read_records, export_csv
from src.providers.base import sanitize_error_message
from src.labeling.quality import reviewed_record,quality_input_hash,result_hash,VERSION as QUALITY_VERSION

def save_results(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(export_csv(rows))
        for attempt in range(8):
            try:
                os.replace(name,path)
                break
            except PermissionError:
                if attempt==7:raise
                time.sleep(min(.05*2**attempt,.5))
    finally:
        if os.path.exists(name):os.unlink(name)

def main():
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[2] / "data_pr" / "clean_final"
    parser.add_argument("--csv", type=Path, default=base/"sample_1000.csv")
    parser.add_argument("--image-root", type=Path, default=base)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--offset", type=int, default=0, help="Skip already checked pilot rows")
    parser.add_argument("--live", action="store_true", help="Use saved encrypted provider settings; sends article + image to API")
    parser.add_argument("--model", help="Override model for this pilot only; does not change saved credentials/config")
    parser.add_argument("--key-stdin", action="store_true", help="Read ephemeral API key from stdin, never save it")
    parser.add_argument("--provider", choices=["gemini","openai"], default="gemini")
    parser.add_argument("--resume",action='store_true',help='Reuse completed matching robust results; never overwrite source CSV')
    parser.add_argument('--retry-incomplete',action='store_true',help='With --resume, retry only rows whose semantic review failed due to API errors')
    args = parser.parse_args()
    if not 1 <= args.limit <= 10: parser.error("pilot limit must be 1..10")
    if args.offset < 0: parser.error("offset must be nonnegative")
    if args.output.resolve()==args.csv.resolve():parser.error('output must not overwrite input CSV')
    if args.output.exists() and not args.resume: parser.error("output already exists; use --resume or choose a new path")
    if args.resume and not args.live:parser.error('--resume is only needed for live API results')
    if args.retry_incomplete and not args.resume:parser.error('--retry-incomplete requires --resume')
    if args.key_stdin and not args.live: parser.error("--key-stdin requires --live")
    records = read_records(args.csv,args.limit+args.offset)[args.offset:];provider=None
    if args.live:
        from src.ui.local_memory import load_provider
        from src.ui.provider import create_provider
        if args.key_stdin:
            print('Waiting for ephemeral API key on stdin (not echoed or saved)',flush=True)
            if sys.stdin.isatty():
                import getpass
                key=getpass.getpass('API key (hidden): ').strip()
            else:
                key=sys.stdin.readline().strip()
            if not key: parser.error('Empty API key')
            config={'provider':args.provider,'api_key':key,'model':args.model or 'gemini-3.1-flash-lite','timeout_seconds':90}
        else:
            config=load_provider()
        if not config or config.get("provider") not in {"gemini","openai"}:
            parser.error("No saved Gemini/OpenAI configuration; use UI step 1")
        provider=create_provider(config["provider"],config["api_key"],args.model or config["model"],min(120,int(config.get("timeout_seconds",60))))
        provider.requests_per_minute=max(1,int(config.get('requests_per_minute',10)))
    results=[];cached={}
    if args.resume and args.output.exists():
        import csv
        with args.output.open(encoding='utf-8-sig',newline='') as f:prior=list(csv.DictReader(f))
        if len({r['article_id'] for r in prior})!=len(prior):parser.error('resume CSV has duplicate IDs')
        cached={r['article_id']:r for r in prior}
        if set(cached)-{r['article_id'] for r in records}:parser.error('resume output contains IDs outside selected input')
        if [r['article_id'] for r in prior]!=[r['article_id'] for r in records[:len(prior)]]:parser.error('resume output must be an ordered prefix of the selected input')
    def checkpoint():
        completed=dict(cached);completed.update({x['article_id']:x for x in results})
        save_results(args.output,[completed[r['article_id']] for r in records if r['article_id'] in completed])
    for r in records:
        try:
            if r['article_id'] in cached:
                result=cached[r['article_id']]
                if result.get('annotation_quality_mode')!=QUALITY_VERSION or result.get('annotation_input_hash')!=quality_input_hash(r,args.image_root,provider):
                    parser.error('resume result does not match current model/prompt/schema/input/image')
                if result.get('annotation_result_hash')!=result_hash(result):parser.error('resume CSV row integrity check failed')
                incomplete=any(str(x).startswith('semantic_review_unavailable:') for x in json.loads(result.get('annotation_review_flags','[]')))
                if args.retry_incomplete and incomplete:result=reviewed_record(r,provider,args.image_root)
                else:print(json.dumps({'article_id':r['article_id'],'cache':'reused'},ensure_ascii=False),flush=True)
            else:result=reviewed_record(r,provider,args.image_root,mock=not args.live)
        except Exception as exc:
            print(json.dumps({"article_id":r['article_id'],"error":sanitize_error_message(exc,getattr(provider,'api_key',''))},ensure_ascii=False))
            if results or cached:checkpoint()
            return 1
        results.append(result);checkpoint()
        print(json.dumps({k:result.get(k) for k in ['article_id','annotation_status','clickbait_label','consistency_tc','consistency_ti','consistency_ic','annotation_review_flags','annotation_input_warnings','annotation_total_tokens','annotation_model_calls']},ensure_ascii=False),flush=True)
    incomplete=any(any(str(x).startswith('semantic_review_unavailable:') for x in json.loads(r.get('annotation_review_flags','[]'))) for r in results)
    return 2 if incomplete else 0

if __name__ == '__main__':raise SystemExit(main())
