"""Build full target labels from two blind Luna assignments, without Gemini calls."""
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.publish_contributor_labels import read_lines,adjudicate,validate_vote_text
from scripts.run_contributor_batch import atomic_bytes,OBSOLETE
from src.labeling.quality import quality_input_hash,result_hash,VERSION,ReferenceOutput
from src.labeling.multimodal import load_image

_SOURCE_CACHE={}

def source_records(source,contributor):
    stamp=(str(source.resolve()),source.stat().st_size,source.stat().st_mtime_ns,contributor)
    if stamp not in _SOURCE_CACHE:
        with source.open(encoding='utf-8-sig',newline='') as f:
            rows=[r for r in csv.DictReader(f) if contributor is None or r['contributor']==contributor]
        if len(_SOURCE_CACHE)>=2:_SOURCE_CACHE.clear()
        _SOURCE_CACHE[stamp]=rows
    return _SOURCE_CACHE[stamp]


def assemble(run_dir,source,image_root,contributor=None):
    primary={}
    for path in sorted(run_dir.glob('luna_independent_*.jsonl')):
        for item in read_lines(path):primary[item['article_id']]=item
    secondary={}
    for path in sorted(run_dir.glob('luna_secondary_*.jsonl')):
        for item in read_lines(path):secondary.setdefault(item['article_id'],[]).append(item)
    eligible=set(primary)&set(secondary)
    rows=[];pending=[]
    provider=SimpleNamespace(provider_name='luna-agents',model='gpt-6-luna')
    if not eligible:
        atomic_bytes(run_dir/'luna_consensus_checkpoint.jsonl',b'')
        return dict(ready=0,pending=[],paired=0)
    for record in source_records(source,contributor):
            aid=record['article_id']
            if aid not in eligible:continue
            if record.get('annotation_input_variant')=='og_title_v1':
                actual=[primary[aid],*secondary[aid]]
                actual=[v for v in actual if v.get('annotated_input_title')==record['title'] and v.get('annotation_input_variant')=='og_title_v1']
                if not actual or actual[0] is not primary[aid] or len(actual)<2:pending.append(aid);continue
                second=[v for v in actual if v is not primary[aid]]
            else:second=secondary[aid]
            if contributor is not None and record['contributor']!=contributor:raise ValueError('Contributor mismatch')
            image,warning=load_image(record,image_root)
            if image is None:pending.append(aid);continue
            first=dict(annotator=primary[aid]['annotator']+':primary',wire_output=primary[aid]['wire_output'])
            votes=[first]+[dict(annotator=r['annotator'],wire_output=r['wire_output']) for r in second][:2]
            try:
                for v in votes:validate_vote_text(ReferenceOutput.model_validate(v['wire_output']))
            except ValueError:pending.append(aid);continue
            base={k:v for k,v in record.items() if k not in OBSOLETE}
            base.update(annotation_input_hash=quality_input_hash(record,image_root,provider),annotation_model_calls=0,
                        annotation_total_tokens=None,annotation_attempts=0,annotation_usage_incomplete=True,
                        annotation_passes_json='[]',annotation_schema_version=VERSION,annotation_prompt_version=VERSION,
                        annotation_input_warnings=json.dumps([warning] if warning else []),annotation_image_status='usable')
            base['annotation_result_hash']=result_hash(base)
            try: row=adjudicate(base,votes)
            except ValueError:pending.append(aid);continue
            row.update(annotation_source_kind='raw_unlabeled_dataset',annotation_agent_evaluations=len(votes),
                       annotation_requested_model='gpt-6-luna',annotation_notes='Full consensus of independent Luna agents using actual source text and images; no Gemini API calls for this result; Luna token usage unavailable.')
            row['annotation_result_hash']=result_hash(row);rows.append(row)
    atomic_bytes(run_dir/'luna_consensus_checkpoint.jsonl',b''.join((json.dumps(dict(result=r),ensure_ascii=False)+'\n').encode() for r in rows))
    return dict(ready=len(rows),pending=pending,paired=len(eligible))


if __name__=='__main__':
    import argparse
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,default=root/'data/output/tuan_run')
    parser.add_argument('--contributor',default='Nguyễn Minh Tuấn')
    args=parser.parse_args()
    print(json.dumps(assemble(args.run_dir,root.parent/'data_pr/clean_final/data.csv',root.parent/'data_pr/clean_final',args.contributor),ensure_ascii=False))
