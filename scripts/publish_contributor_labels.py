"""Publish full labels, retaining batch originals and explicit leader adjudications."""
import csv
import hashlib
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.labeling.quality import ReferenceOutput, normalized_output, consensus, result_hash, TYPES, RELATIONS
from src.labeling.multimodal import grounded_output
from scripts.run_contributor_batch import atomic_bytes, atomic_json, is_full, load_jsonl, load_leader_audit


def read_lines(path):
    if not path.exists(): return []
    # An agent may be appending the last line while the publisher snapshots it.
    data=path.read_bytes(); lines=data.splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith(b'\n') and line.strip()]


def validate_vote_text(wire):
    texts=[wire.annotation_notes]
    for decision in list(wire.types.__dict__.values())+[getattr(wire,t) for t in RELATIONS]:
        texts.extend([decision.reason,decision.evidence.image_observation or ''])
    if any('\ufffd' in s or (s.count('?')>=4 and len(re.findall(r'\w\?\w',s))>=2) for s in texts):
        raise ValueError('Corrupted free-text encoding in agent vote')


def adjudicate(original, votes):
    if original['annotation_result_hash']!=result_hash(original):raise ValueError('Invalid original checksum')
    outputs=[normalized_output(ReferenceOutput.model_validate(v['wire_output'])) for v in votes]
    if len(votes)<2 or len({v['annotator'] for v in votes})<2:raise ValueError('Need two distinct voting passes')
    agreed,masks,flags,_=consensus(outputs,original,True)
    if flags or not all(masks.values()): raise ValueError('Leader correction lacks full valid agreement')
    leader_vote=any(v['annotator']=='leader_ai' for v in votes)
    gemini_vote=any(v['annotator'].startswith('gemini:') for v in votes)
    row=dict(original)
    row.update(clickbait_label=agreed.clickbait_label,
               clickbait_types=json.dumps([t for t in TYPES if getattr(agreed.types,t).label==1]),
               misleading_subtype=json.dumps(agreed.misleading_subtype),
               annotation_status='pseudo_labeled',annotation_quality_mode='multimodal-v4.1-cross-model-consensus' if gemini_vote else 'multimodal-v4.1-leader-adjudicated' if leader_vote else 'multimodal-v4.1-luna-peer-consensus',
               label_source='AI:Gemini+Luna consensus' if gemini_vote else 'AI:leader+Luna adjudication' if leader_vote else 'AI:Luna peer consensus',annotation_model='+'.join(dict.fromkeys(v.get('model',v['annotator']) for v in votes)),
               annotation_source_result_hash=original['annotation_result_hash'],
               annotation_source_input_hash=original['annotation_input_hash'],
               annotation_adjudication_votes_json=json.dumps(votes,ensure_ascii=False),
               annotation_raw_json=agreed.model_dump_json(),
               annotation_grounded_json=grounded_output(agreed,original).model_dump_json(),
               annotation_raw_is_consensus=True,annotation_review_flags='[]',
               annotation_notes='Per-target plurality with at least two valid agreeing votes; ties are unresolved. Original Gemini passes retained. Not human gold.' if gemini_vote else 'Leader-reviewed independent AI adjudication; original Gemini passes retained.' if leader_vote else 'Independent Luna peer consensus on full input; original Gemini passes retained. Not a leader visual audit or human gold.',
               annotated_at=datetime.now(timezone.utc).isoformat(),label_mask_json=json.dumps(masks),human_review_status='pending')
    grounded=grounded_output(agreed,original)
    row['type_evidence']=json.dumps({t:getattr(grounded.types,t).evidence.model_dump() for t in TYPES},ensure_ascii=False)
    for t in TYPES: row['type_'+t]=getattr(agreed.types,t).label
    for t in RELATIONS: row[t]=getattr(agreed,t).label
    row['label_status']=json.dumps({k:'pseudo_labeled' for k in ['detection','types',*RELATIONS,*('type_'+t for t in TYPES)]})
    row['annotation_input_hash']=hashlib.sha256((row['annotation_source_input_hash']+row['annotation_adjudication_votes_json']).encode()).hexdigest()
    row['annotation_result_hash']=result_hash(row)
    return row


def publish(run_dir, destination):
    results={x['result']['article_id']:x['result'] for x in read_lines(run_dir/'checkpoint.jsonl')}
    gemini_processed=len(results)
    checkpoints=[run_dir/'fast_run/checkpoint.jsonl',*sorted((run_dir/'model_runs').glob('*/checkpoint.jsonl'))]
    for checkpoint in checkpoints:
        for item in read_lines(checkpoint):
            row=item['result'];old=results.get(row['article_id'])
            if old and is_full(old) and not is_full(row):continue
            results[row['article_id']]=row
    gemini_processed=len(results)
    hold_path=run_dir/'source_title_review_holds.json'
    title_holds={r['article_id']:r for r in json.loads(hold_path.read_text(encoding='utf-8')).get('rows',[])} if hold_path.exists() else {}
    corrections={x['article_id']:x for x in read_lines(run_dir/'leader_adjudications.jsonl')}
    for item in read_lines(run_dir/'luna_consensus_checkpoint.jsonl'):
        agent=dict(item['result']);aid=agent['article_id']
        current=results.get(aid,{})
        if current.get('annotation_quality_mode')=='multimodal-fast-v1-single-pass' and is_full(current):
            if not is_full(agent) or (current.get('annotation_input_variant')=='og_title_v1' and agent.get('annotation_input_variant')!='og_title_v1'):
                continue
        if aid in results:
            if aid in corrections:continue
            original=results[aid]
            agent.update(annotation_initial_gemini_result_hash=original['annotation_result_hash'],
                         annotation_initial_gemini_passes_json=original['annotation_passes_json'],
                         annotation_initial_gemini_tokens=original.get('annotation_total_tokens'),
                         annotation_initial_gemini_calls=original.get('annotation_model_calls'))
            agent['annotation_result_hash']=result_hash(agent)
        results[aid]=agent
    for item in read_lines(run_dir/'title_repairs/luna_consensus_checkpoint.jsonl'):
        repaired=dict(item['result']);aid=repaired['article_id'];hold=title_holds.get(aid)
        if not hold or repaired.get('annotation_input_variant')!='og_title_v1' or repaired.get('title')!=hold['og_title'] or repaired.get('source_original_title')!=hold['title']:
            raise ValueError('Repaired title does not match original source metadata')
        if repaired['annotation_result_hash']!=result_hash(repaired):raise ValueError('Repaired checksum mismatch')
        if aid in results:
            old=results[aid];repaired.update(annotation_initial_gemini_result_hash=old['annotation_result_hash'],annotation_initial_gemini_passes_json=old.get('annotation_passes_json','[]'))
        repaired['annotation_result_hash']=result_hash(repaired);results[aid]=repaired
    for item in read_lines(run_dir/'fast_agents_checkpoint.jsonl'):
        agent=item['result'];aid=agent['article_id']
        if aid not in results or not is_full(results[aid]):results[aid]=agent
    if not results: raise ValueError('No checkpointed results to publish')
    leader=load_leader_audit(run_dir)
    leader_expectations={r['article_id']:r.get('expected_labels') for r in read_lines(run_dir/'leader_audit.jsonl')}
    independent={};invalid_votes=[]
    for path in sorted(run_dir.glob('luna_independent_*.jsonl')):
        for vote in read_lines(path):
            try:validate_vote_text(ReferenceOutput.model_validate(vote['wire_output']))
            except ValueError as exc:
                invalid_votes.append(dict(article_id=vote.get('article_id'),error=str(exc)));continue
            independent[vote['article_id']]=vote
    full=[]; partial=[]; disagreements=[]; corrected=0;peer_corrected=0
    for aid,original in results.items():
        if original['annotation_result_hash']!=result_hash(original): raise ValueError('Bad checksum: '+aid)
        row=dict(original)
        if aid in title_holds and row.get('annotation_input_variant')!='og_title_v1':
            row.update(annotation_status='needs_review',input_review_status='category_heading_in_title',
                       label_mask_json=json.dumps({t:False for t in (*TYPES,'detection',*RELATIONS)}))
            for key in ['clickbait_label',*('type_'+t for t in TYPES),*RELATIONS]:row[key]=''
            for key in ['clickbait_types','type_evidence','misleading_subtype']:row[key]='null'
            partial.append(row);continue
        if aid in corrections:
            row=adjudicate(original,corrections[aid]['votes'])
            if any(v['annotator']=='leader_ai' for v in corrections[aid]['votes']):corrected+=1
            else:peer_corrected+=1
        elif leader.get(aid) in {'disputed','quarantined'}:
            partial.append(row);continue
        if aid in title_holds and row.get('annotation_input_variant')=='og_title_v1':
            hold=title_holds[aid]
            if row.get('title')!=hold['og_title'] or row.get('source_original_title')!=hold['title']:
                raise ValueError('Canonical title does not match source metadata')
        vote=None if row.get('annotation_input_variant')=='og_title_v1' or row.get('annotation_quality_mode')=='multimodal-fast-v1-single-pass' else independent.get(aid)
        if vote:
            wire=ReferenceOutput.model_validate(vote['wire_output'])
            actual={'detection':wire.clickbait_label,**{t:getattr(wire.types,t).label for t in TYPES},**{t:getattr(wire,t).label for t in RELATIONS}}
            mismatches=[]
            for target,value in actual.items():
                column='clickbait_label' if target=='detection' else 'type_'+target if target in TYPES else target
                if value is not None and str(row.get(column,'')) not in {'',str(value)}:mismatches.append(target)
            row.update(independent_annotator=vote['annotator'],independent_vote_json=json.dumps(vote['wire_output'],ensure_ascii=False),
                       independent_disagreements_json=json.dumps(mismatches))
            if mismatches:
                disagreements.append(dict(article_id=aid,targets=mismatches))
            # Resolve using valid votes, not unanimous agreement between systems.
            # A purely visual mismatch alone remains a review candidate for clickbait.
            suspicious_visual=(str(row.get('type_misleading_presentation'))=='1' and
                all(str(row.get('type_'+t))=='0' for t in TYPES if t!='misleading_presentation') and
                row.get('consistency_tc')=='Consistent' and
                any(row.get(t)=='Inconsistent' for t in ['consistency_ti','consistency_ic']))
            if suspicious_visual and mismatches:
                partial.append(row);continue
            if (mismatches or not is_full(row)) and row.get('annotation_passes_json') and aid not in corrections and not row.get('annotation_source_kind'):
                passes=json.loads(row['annotation_passes_json'])
                votes=[dict(annotator='gemini:'+p['phase']+':'+p['model'],model=p['model'],wire_output=p['wire_output']) for p in passes]
                votes.append(dict(annotator=vote['annotator'],model='gpt-6-luna',wire_output=vote['wire_output']))
                try:
                    combined=adjudicate(original,votes)
                except ValueError:
                    partial.append(row);continue
                combined.update(independent_annotator=row['independent_annotator'],independent_vote_json=row['independent_vote_json'],independent_disagreements_json=row['independent_disagreements_json'])
                row=combined
        row['leader_review_status']=('corrected' if any(v['annotator']=='leader_ai' for v in corrections[aid]['votes']) else 'peer_adjudicated') if aid in corrections else leader.get(aid,'not_sampled')
        expected=leader_expectations.get(aid) or {}
        drift=[k for k,v in expected.items() if str(row.get(k,''))!=str(v)]
        if drift:
            row['leader_review_status']='label_drift_needs_review'
            row['leader_disagreements_json']=json.dumps(drift)
            partial.append(row);continue
        row['annotation_result_hash']=result_hash(row)
        (full if is_full(row) else partial).append(row)
    fields=list(dict.fromkeys(k for r in list(results.values())+full for k in r))
    def save_csv(path,rows):
        normalized=[{k:r.get(k,'') for k in fields} for r in rows]
        for row in normalized: row['annotation_result_hash']=result_hash(row)
        s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(normalized)
        atomic_bytes(path,b'\xef\xbb\xbf'+s.getvalue().encode('utf-8'))
    save_csv(destination,full);save_csv(run_dir/'publication_review_queue.csv',partial)
    report=dict(published=len(full),processed=len(results),gemini_processed=gemini_processed,luna_only_processed=len(results)-gemini_processed,needs_review=len(partial),leader_corrected=corrected,peer_corrected=peer_corrected,
                source_title_review_candidates=len(title_holds),repaired_title_rows=sum(r.get('annotation_input_variant')=='og_title_v1' for r in full),
                independent_luna_rows=len(independent),invalid_agent_votes=invalid_votes,luna_gemini_disagreements=disagreements,
                leader_reviewed=sum(aid in results for aid in leader),human_gold=False,
                output=str(destination),timestamp=datetime.now(timezone.utc).isoformat())
    atomic_json(run_dir/'publication_status.json',report)
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true')
    root=Path(__file__).resolve().parents[1]
    parser.add_argument('--run-dir',type=Path,default=root/'data/output/tuan_run')
    parser.add_argument('--destination',type=Path,default=root.parent/'data_pr/clean_final/data_tuan_labeled.csv')
    parser.add_argument('--source',type=Path,default=root.parent/'data_pr/clean_final/data.csv')
    parser.add_argument('--worker-progress',type=Path)
    parser.add_argument('--fast',action='store_true')
    parser.add_argument('--image-root',type=Path,default=root.parent/'data_pr/clean_final')
    parser.add_argument('--contributor',default='Nguyễn Minh Tuấn')
    args=parser.parse_args()
    run_dir=args.run_dir;last=None;previous_signature=None;previous_votes=None
    while True:
        repair_dir=run_dir/'title_repairs'
        vote_paths=sorted([*run_dir.glob('luna_fast_*.jsonl')]) if args.fast else sorted([*run_dir.glob('luna_independent_*.jsonl'),*run_dir.glob('luna_secondary_*.jsonl'),*repair_dir.glob('luna_independent_*.jsonl'),*repair_dir.glob('luna_secondary_*.jsonl')])
        votes_signature=tuple((p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in vote_paths)
        if args.watch and votes_signature!=previous_votes:
            from scripts.assemble_luna_consensus import assemble
            if args.fast:
                from scripts.assemble_fast_agents import assemble as fast_assemble
                result=fast_assemble(run_dir,args.source,args.image_root,args.contributor)
            else:result=assemble(run_dir,args.source,args.image_root,args.contributor)
            print(json.dumps({'luna_assembly':{'ready':result['ready'],'pending_count':len(result['pending']),'paired':result['paired']}},ensure_ascii=False),flush=True)
            if not args.fast and (repair_dir/'source.csv').exists():
                fixed=assemble(repair_dir,repair_dir/'source.csv',args.image_root,args.contributor)
                print(json.dumps({'title_repair_assembly':{'ready':fixed['ready'],'paired':fixed['paired']}},ensure_ascii=False),flush=True)
            previous_votes=votes_signature
        signature=tuple((p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(run_dir.glob('*.jsonl')))
        progress_path=args.worker_progress or run_dir/'progress.json'
        checkpoint_paths=[run_dir/'fast_run/checkpoint.jsonl',*sorted((run_dir/'model_runs').glob('*/checkpoint.jsonl'))]
        for checkpoint_path in checkpoint_paths:
            if checkpoint_path.exists():signature+=((str(checkpoint_path),checkpoint_path.stat().st_size,checkpoint_path.stat().st_mtime_ns),)
        if progress_path.exists():signature+=(('progress',progress_path.stat().st_mtime_ns),)
        if args.watch and signature==previous_signature:
            time.sleep(5);continue
        try:
            report=publish(run_dir,args.destination)
        except PermissionError:
            if not args.watch:raise
            # Windows readers/OneDrive may briefly lock an atomic CSV replace.
            # Retry publication from saved votes; never repeat labeling calls.
            print(json.dumps({'publisher_retry':'PermissionError'}),flush=True)
            time.sleep(5)
            continue
        previous_signature=signature
        state=(report['processed'],report['published'],report['independent_luna_rows'],report['leader_corrected'])
        if state!=last:print(json.dumps(report,ensure_ascii=False),flush=True);last=state
        if not args.watch:break
        progress=json.loads(progress_path.read_text(encoding='utf-8'))
        if progress.get('status')!='running' and not args.fast:break
        time.sleep(5)
