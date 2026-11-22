"""Format explicit agent decisions; never infer labels from article text."""
import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.labeling.quality import ReferenceOutput,TYPES,RELATIONS
from scripts.publish_contributor_labels import read_lines,validate_vote_text

LABELS={'C':'Consistent','I':'Inconsistent','IE':'Insufficient Evidence','NA':'Not Available'}


def make_wire(record,decision):
    labels=decision['types']
    if not isinstance(labels,list) or len(labels)!=5 or any(x is not None and (type(x) is not int or x not in (0,1)) for x in labels):
        raise ValueError('Agent must explicitly supply all five nullable type decisions')
    image_status=decision['image_assessment'];observation=decision.get('image_observation')
    count=len(record['paragraphs'])
    def evidence(ref=None,paragraph=None,image=None):
        if paragraph is not None and (type(paragraph) is not int or not 1<=paragraph<=count):
            raise ValueError('Invalid source paragraph reference')
        return dict(text_reference=ref,content_paragraph_id=paragraph,image_observation=image)
    types={}
    for key,label in zip(TYPES,labels):
        support=decision.get('support',{}).get(key,{})
        if label==1 and not support.get('reason'):raise ValueError('Positive type requires agent rationale: '+key)
        reason=support.get('reason') or ('The provided title and thumbnail do not show separate evidence of this tactic after comparison with the full article.' if label==0 else 'The provided input is insufficient for this type decision.')
        ev=evidence(support.get('text_ref','title') if label==1 else None,support.get('paragraph'),observation if support.get('use_image') else None)
        types[key]=dict(label=label,evidence=ev,reason=reason)
    data=dict(image_assessment=image_status,types=types,clickbait_label=1 if 1 in labels else 0 if all(v==0 for v in labels) else None,
              misleading_subtype=decision.get('misleading_subtype'),confidence=None,
              annotation_notes='Independent agent assessment of the full article and actual local image. Explicit decisions were serialized without automatic label inference.')
    relations=decision['relations']
    if len(relations)!=3:raise ValueError('Agent must supply TC, TI and IC')
    for name,value in zip(RELATIONS,relations):
        label=LABELS.get(value['label'],value['label'])
        reason=value.get('reason')
        if not reason:raise ValueError('Relation requires an agent rationale')
        if label in {'Consistent','Inconsistent'}:
            if name!='consistency_ti' and value.get('paragraph') is None:raise ValueError('TC/IC need a supporting body paragraph')
            if name!='consistency_tc' and not observation:raise ValueError('TI/IC need an actual image observation')
        data[name]=dict(label=label,reason=reason,evidence=evidence('title' if name!='consistency_ic' else None,
            value.get('paragraph'),observation if name!='consistency_tc' else None))
    wire=ReferenceOutput.model_validate(data);validate_vote_text(wire)
    return wire


def emit(packet,decisions,output,annotator):
    records={r['article_id']:r for r in read_lines(packet)}
    text=decisions.read_text(encoding='utf-8')
    items=json.loads(text) if text.lstrip().startswith('[') else read_lines(decisions)
    if not items:
        value=json.loads(decisions.read_text(encoding='utf-8'));items=value if isinstance(value,list) else [value]
    if len({r['article_id'] for r in items})!=len(items):raise ValueError('Duplicate decision IDs')
    existing={r['article_id'] for r in read_lines(output)}
    new=[]
    for d in items:
        aid=d['article_id']
        if aid not in records:raise ValueError('ID outside assigned source packet')
        wire=make_wire(records[aid],d)
        if aid not in existing:new.append(dict(article_id=aid,annotator=annotator,wire_output=wire.model_dump(),notes=d.get('notes',''),
            annotated_input_title=records[aid]['title'],annotation_input_variant=records[aid].get('annotation_input_variant','original')))
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() and output.stat().st_size and not output.read_bytes().endswith(b'\n'):
        raise ValueError('Incomplete last output line; inspect before appending')
    with output.open('a',encoding='utf-8',newline='\n') as f:
        for row in new:f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    return len(new)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--decisions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--annotator',required=True);a=p.parse_args()
    print('Appended validated agent votes:',emit(a.packet,a.decisions,a.output,a.annotator))
