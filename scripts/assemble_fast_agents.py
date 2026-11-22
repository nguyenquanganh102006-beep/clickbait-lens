"""Export actual single-agent decisions without claiming consensus."""
import json
from scripts.assemble_luna_consensus import source_records
from scripts.publish_contributor_labels import read_lines,validate_vote_text
from scripts.run_contributor_batch import atomic_bytes,is_full
from src.labeling.fast import export_vote
from src.labeling.quality import ReferenceOutput

def assemble(run_dir,source,image_root,contributor):
    votes={}
    for path in sorted(run_dir.glob('luna_fast_*.jsonl')):
        for vote in read_lines(path):votes[vote['article_id']]=vote
    records={r['article_id']:r for r in source_records(source,contributor)} if votes else {}
    hold_path=run_dir/'source_title_review_holds.json'
    holds={r['article_id']:r for r in json.loads(hold_path.read_text(encoding='utf8'))['rows']} if hold_path.exists() else {}
    output=[];pending=[]
    for aid,vote in votes.items():
        record=dict(records[aid]);hold=holds.get(aid)
        if hold:
            if vote.get('annotated_input_title')!=hold['og_title'] or vote.get('annotation_input_variant')!='og_title_v1':
                pending.append(aid);continue
            record.update(source_original_title=record['title'],title=hold['og_title'],annotation_input_variant='og_title_v1',title_repair_method='og_title_when_title_equals_category')
        wire=ReferenceOutput.model_validate(vote['wire_output']);validate_vote_text(wire)
        row=export_vote(record,wire,image_root,model='gpt-6-luna',requested_model='gpt-6-luna',source='AI:Luna fast single pass',provider_name='luna-agents',agent_vote=vote)
        output.append({'result':row})
        if not is_full(row):pending.append(aid)
    atomic_bytes(run_dir/'fast_agents_checkpoint.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output).encode('utf8'))
    return {'ready':len(output)-len(pending),'pending':pending,'paired':len(votes)}
