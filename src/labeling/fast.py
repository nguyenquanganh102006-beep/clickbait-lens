"""User-authorized single-pass labeling, retaining structural evidence checks."""
import hashlib
import json
from datetime import datetime,timezone
from src.labeling.multimodal import load_image,build_prompt,grounded_output,TYPES,RELATIONS
from src.labeling.quality import ReferenceOutput,normalized_output,blocked_targets,_call,result_hash

VERSION='multimodal-fast-v1-single-pass'
PREFIX='REFERENCE-ONLY SINGLE PASS. Use the full article and attached image. Give concise evidence-based reasons (at most 20 words each). Select text_reference and paragraph IDs; do not invent quotes. Return all schema fields.\n'

def fast_prompt(record,has_image):
    return PREFIX+build_prompt(record,has_image)

def input_hash(record,image_root,provider):
    image,_=load_image(record,image_root)
    scope=json.dumps([VERSION,provider.provider_name,provider.model,ReferenceOutput.model_json_schema()],sort_keys=True)
    prompt=fast_prompt(record,image is not None) if provider.provider_name=='gemini' else build_prompt(record,image is not None)
    return hashlib.sha256((scope+prompt).encode()+(image or b'')).hexdigest()

def export_vote(record,wire,image_root,*,model,requested_model,source,provider_name,attempts=0,tokens=None,prompt_sha=None,agent_vote=None):
    from types import SimpleNamespace
    image,warning=load_image(record,image_root)
    wire=ReferenceOutput.model_validate(wire.model_dump() if hasattr(wire,'model_dump') else wire)
    output=normalized_output(wire)
    blocked,flags=blocked_targets(output,record,image is not None)
    grounded=grounded_output(output,record)
    row={k:v for k,v in record.items() if k not in {'type_misleading_framing','type_image_text_mismatch','type_context_manipulation'}}
    masks={t:t not in blocked and getattr(output.types,t).label is not None for t in TYPES}
    masks['detection']='detection' not in blocked and output.clickbait_label is not None
    for t in RELATIONS:masks[t]=t not in blocked and getattr(output,t).label!='Not Available'
    row['clickbait_label']=output.clickbait_label if masks['detection'] else ''
    for t in TYPES:row['type_'+t]=getattr(output.types,t).label if masks[t] else ''
    for t in RELATIONS:row[t]=getattr(output,t).label if masks[t] else ''
    row['clickbait_types']=json.dumps([t for t in TYPES if getattr(output.types,t).label==1] if all(masks[t] for t in TYPES) else None)
    row['misleading_subtype']=json.dumps(output.misleading_subtype)
    row['type_evidence']=json.dumps({t:getattr(grounded.types,t).evidence.model_dump() if masks[t] else None for t in TYPES},ensure_ascii=False)
    if not all(masks.values()):flags.extend('unresolved:'+t for t,v in masks.items() if not v)
    passes=[] if agent_vote is not None else [dict(phase='single_pass',model=model,wire_output=wire.model_dump(),tokens=tokens,attempts=attempts,prompt_sha256=prompt_sha)]
    row.update(annotation_status='pseudo_labeled' if not flags and all(masks.values()) else 'needs_review',
               label_status=json.dumps({t:'pseudo_labeled' if v else 'needs_review' for t,v in masks.items()}),
               label_source=source,human_review_status='pending',annotation_model=model,annotation_requested_model=requested_model,
               annotation_quality_mode=VERSION,annotation_schema_version=VERSION,annotation_prompt_version=VERSION,
               annotation_passes_json=json.dumps(passes,ensure_ascii=False),annotation_raw_json=output.model_dump_json(),
               annotation_grounded_json=grounded.model_dump_json(),annotation_raw_is_consensus=False,
               annotation_review_flags=json.dumps(list(dict.fromkeys(flags))),annotation_input_warnings=json.dumps([warning] if warning else []),
               annotation_image_status=output.image_assessment,annotation_model_calls=0 if agent_vote is not None else 1,
               annotation_agent_evaluations=1 if agent_vote is not None else 0,annotation_attempts=attempts,
               annotation_total_tokens=tokens,annotation_usage_incomplete=agent_vote is not None or attempts>1,
               label_mask_json=json.dumps(masks),annotated_at=datetime.now(timezone.utc).isoformat(),
               annotation_notes='Single-pass pseudo-labels under the user-selected throughput policy. No blind consensus or human verification is claimed.')
    if agent_vote is not None:row['annotation_source_vote_json']=json.dumps(agent_vote,ensure_ascii=False)
    row['annotation_input_hash']=input_hash(record,image_root,SimpleNamespace(provider_name=provider_name,model=requested_model))
    row['annotation_result_hash']=result_hash(row)
    return row

def annotate_record(record,provider,image_root):
    image,_=load_image(record,image_root);prompt=fast_prompt(record,image is not None)
    _,response,attempts=_call(provider,prompt,image)
    return export_vote(record,response.data,image_root,model=response.model,requested_model=provider.model,
                       source='AI:Gemini fast single pass',provider_name=provider.provider_name,attempts=attempts,
                       tokens=response.total_tokens,prompt_sha=hashlib.sha256(prompt.encode()).hexdigest())
