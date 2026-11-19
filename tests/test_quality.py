import csv
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image
from src.labeling.multimodal import blank_mock,TYPES,RELATIONS,export_csv
from src.labeling.quality import reviewed_record,consensus,quality_input_hash,result_hash,TARGETS,ReferenceOutput
from src.providers.base import ProviderError

def record(tmp_path,title='Tin thông báo hoạt động cộng đồng'):
    (tmp_path/'images').mkdir(exist_ok=True)
    Image.new('RGB',(240,240),'blue').save(tmp_path/'images/a_1.jpg')
    return {'article_id':'a','title':title,'content':'Thông tin chính của hoạt động cộng đồng.\nThông tin bổ sung.',
            'lead':'','image_caption':'','image_path':'images/a_1.jpg','image_valid':'True'}

def negative(has_image=True):
    o=blank_mock(has_image,True)
    for t in TYPES:getattr(o.types,t).label=0
    o.clickbait_label=0
    for t in RELATIONS:getattr(o,t).label='Consistent' if has_image or t=='consistency_tc' else 'Not Available'
    o.consistency_tc.evidence.content_paragraph_id=1
    if has_image:
        o.consistency_ti.evidence.image_observation='Ảnh màu xanh.'
        o.consistency_ic.evidence.image_observation='Ảnh màu xanh.'
        o.consistency_ic.evidence.content_paragraph_id=1
    return o

def positive(r):
    o=negative();o.types.curiosity_gap.label=1;o.types.curiosity_gap.evidence.text_reference='title'
    o.clickbait_label=1
    return o

class Fake:
    provider_name='fake';model='fake-model'
    def __init__(self,outputs):self.outputs=list(outputs);self.prompts=[]
    def generate_multimodal(self,prompt,schema,image):
        self.prompts.append(prompt)
        o=self.outputs.pop(0)
        if isinstance(o,Exception):raise o
        data=o.model_dump()
        if schema is ReferenceOutput:
            for d in list(data['types'].values())+[data[k] for k in RELATIONS]:
                d['evidence'].pop('text_span',None);d['evidence'].pop('content_span',None)
        return SimpleNamespace(data=schema.model_validate(data),model=self.model,total_tokens=10)

def test_blind_agreement_publishes_masks_without_leaking_primary(tmp_path):
    r=record(tmp_path);a=negative();a.types.curiosity_gap.reason='PRIVATE_PRIMARY_REASON_MARKER'
    p=Fake([a,negative()]);result=reviewed_record(r,p,tmp_path)
    assert len(p.prompts)==2 and 'PRIVATE_PRIMARY_REASON_MARKER' not in p.prompts[1]
    assert result['annotation_status']=='pseudo_labeled' and result['clickbait_label']==0
    assert all(json.loads(result['label_mask_json']).values())
    assert result['annotation_total_tokens']==20
    assert len(json.loads(result['annotation_passes_json']))==2

def test_adjudication_can_resolve_bait_with_two_valid_evidence_votes(tmp_path):
    r=record(tmp_path,'Góp quà rồi người ấy lại nói với tôi thế này')
    p=Fake([negative(),positive(r),positive(r)])
    result=reviewed_record(r,p,tmp_path)
    assert len(p.prompts)==3 and result['clickbait_label']==1
    assert result['type_curiosity_gap']==1
    assert result['annotation_model_calls']==3
    assert result['annotation_status']=='pseudo_labeled'

def test_no_majority_never_forces_unknown_to_zero(tmp_path):
    r=record(tmp_path);a=negative();b=positive(r);c=negative();c.types.curiosity_gap.label=None;c.clickbait_label=None
    result,masks,notes,_=consensus([a,b,c],r,True)
    assert result.types.curiosity_gap.label is None and not masks['curiosity_gap']
    assert 'Mô phỏng' not in result.types.curiosity_gap.reason
    assert result.clickbait_label is None and not masks['detection']
    assert masks['consistency_tc']

def test_placeholder_preserves_text_relation_and_masks_visual_targets(tmp_path):
    r=record(tmp_path);a=negative();a.image_assessment='placeholder'
    a.consistency_ti.label='Not Available';a.consistency_ic.label='Not Available'
    p=Fake([a,a.model_copy(deep=True)]);result=reviewed_record(r,p,tmp_path)
    masks=json.loads(result['label_mask_json'])
    assert result['consistency_tc']=='Consistent' and masks['consistency_tc']
    assert result['consistency_ti']=='Not Available' and not masks['consistency_ti']
    assert result['clickbait_label']=='' and result['annotation_status']=='needs_review'
    assert len(p.prompts)==2

def test_invalid_evidence_blocks_only_affected_target(tmp_path):
    r=record(tmp_path);a=negative();a.consistency_ti.evidence.content_span='Citation không có thật'
    result,masks,_,_=consensus([a,a.model_copy(deep=True)],r,True)
    assert not masks['consistency_ti'] and masks['consistency_tc'] and masks['detection']

def test_review_outage_cannot_fall_back_to_single_pass(tmp_path):
    r=record(tmp_path);p=Fake([negative(),ProviderError('unavailable',retryable=False)])
    result=reviewed_record(r,p,tmp_path)
    assert result['annotation_status']=='needs_review' and result['clickbait_label']==''
    assert not any(json.loads(result['label_mask_json']).values())

def test_exhausted_quota_stops_without_more_requests(tmp_path):
    r=record(tmp_path);p=Fake([negative(),ProviderError('quota',quota_exhausted=True,retryable=True)])
    with pytest.raises(ProviderError):reviewed_record(r,p,tmp_path)
    assert len(p.prompts)==2

def test_hash_changes_when_image_or_prompt_context_changes(tmp_path):
    r=record(tmp_path);p=Fake([]);a=quality_input_hash(r,tmp_path,p)
    r['content']+=' Nội dung mới.';b=quality_input_hash(r,tmp_path,p)
    assert a!=b
    p.model='other-model';assert quality_input_hash(r,tmp_path,p)!=b
    before=quality_input_hash(r,tmp_path,p)
    Image.new('RGB',(240,240),'red').save(tmp_path/'images/a_1.jpg')
    assert quality_input_hash(r,tmp_path,p)!=before

def test_mock_stays_unlabeled_and_makes_no_requests(tmp_path):
    r=record(tmp_path);p=Fake([]);o=reviewed_record(r,p,tmp_path,mock=True)
    assert not p.prompts and o['annotation_status']=='mock'
    assert not any(json.loads(o['label_mask_json']).values())

def test_source_numeric_conflict_only_masks_related_targets(tmp_path):
    r=record(tmp_path);r['content']='Trước đây phí là 500 đồng, nay là 20 triệu đồng, gấp 40 lần.'
    o,masks,_,_=consensus([negative(),negative()],r,True)
    assert not masks['exaggeration'] and not masks['misleading_presentation']
    assert not masks['consistency_tc'] and not masks['consistency_ic']
    assert masks['consistency_ti'] and masks['question_based']
    assert o.clickbait_label is None

def test_result_integrity_roundtrip_and_edit_detection(tmp_path):
    r=record(tmp_path);o=reviewed_record(r,Fake([negative(),negative()]),tmp_path)
    import io
    row=next(csv.DictReader(io.StringIO(export_csv([o]).decode('utf-8-sig'))))
    assert row['annotation_result_hash']==result_hash(row)
    row['clickbait_label']='1';assert row['annotation_result_hash']!=result_hash(row)

def test_cli_resume_reuses_matching_row_without_api(monkeypatch,tmp_path):
    import io,sys
    from scripts import annotate as cli
    from src.ui import provider
    r=record(tmp_path);p=Fake([negative(),negative()]);p.provider_name='gemini';p.model='offline-cache-model'
    o=reviewed_record(r,p,tmp_path);src=tmp_path/'input.csv';dest=tmp_path/'output.csv'
    src.write_bytes(export_csv([r]));dest.write_bytes(export_csv([o]))
    never=Fake([]);never.provider_name=p.provider_name;never.model=p.model
    monkeypatch.setattr(provider,'create_provider',lambda *args:never)
    monkeypatch.setattr(sys,'stdin',io.StringIO('fake-test-key\n'))
    monkeypatch.setattr(sys,'argv',['cli','--csv',str(src),'--image-root',str(tmp_path),'--output',str(dest),
        '--live','--key-stdin','--model',p.model,'--resume','--limit','1'])
    assert cli.main()==0 and not never.prompts

def test_reference_wire_schema_forbids_generated_text_quotes(tmp_path):
    r=record(tmp_path);o=negative().model_dump()
    with pytest.raises(ValueError):ReferenceOutput.model_validate(o)

def test_consensus_requires_body_and_visual_evidence_for_relations(tmp_path):
    r=record(tmp_path);a=negative();a.consistency_tc.evidence.content_paragraph_id=None
    a.consistency_ti.evidence.image_observation=None
    _,masks,_,_=consensus([a,a.model_copy(deep=True)],r,True)
    assert not masks['consistency_tc'] and not masks['consistency_ti']
    assert masks['consistency_ic']

def test_placeholder_policy_exports_not_available_even_if_models_mislabel_relation(tmp_path):
    r=record(tmp_path);a=negative();a.image_assessment='placeholder'
    o,masks,_,_=consensus([a,a.model_copy(deep=True)],r,True)
    assert o.consistency_ti.label=='Not Available' and o.consistency_ic.label=='Not Available'
    assert not masks['consistency_ti'] and masks['consistency_tc']

def test_atomic_export_retries_transient_windows_file_lock(monkeypatch,tmp_path):
    from scripts import annotate as cli
    original=cli.os.replace;calls=[]
    def locked(a,b):
        calls.append(1)
        if len(calls)<3:raise PermissionError('temporary lock')
        return original(a,b)
    monkeypatch.setattr(cli.os,'replace',locked);monkeypatch.setattr(cli.time,'sleep',lambda _:None)
    path=tmp_path/'out.csv';cli.save_results(path,[{'article_id':'a'}])
    assert len(calls)==3 and 'article_id' in path.read_text(encoding='utf-8-sig')

def test_no_extra_adjudication_only_to_resolve_placeholder_availability(tmp_path):
    r=record(tmp_path);a=negative();b=negative();a.image_assessment=b.image_assessment='placeholder'
    a.consistency_ti.label='Not Available';a.consistency_ic.label='Not Available'
    p=Fake([a,b]);result=reviewed_record(r,p,tmp_path)
    assert len(p.prompts)==2
    assert result['consistency_ti']=='Not Available' and result['consistency_tc']=='Consistent'

def test_ui_preserves_path_when_switching_pages(monkeypatch,tmp_path):
    from streamlit.testing.v1 import AppTest
    from src.ui import local_memory
    monkeypatch.setattr(local_memory,'STATE_DIR',tmp_path/'remembered')
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run()
    app.radio(key='workflow_page').set_value('2. Gán nhãn đa phương thức').run()
    app.text_input(key='annotation_csv').set_value('custom_sample.csv').run()
    app.radio(key='workflow_page').set_value('1. Kết nối AI').run()
    app.radio(key='workflow_page').set_value('2. Gán nhãn đa phương thức').run()
    assert app.text_input(key='annotation_csv').value=='custom_sample.csv'
    assert not app.exception

def test_failed_adjudication_preserves_already_agreed_targets(tmp_path):
    r=record(tmp_path);p=Fake([negative(),positive(r),ProviderError('busy',retryable=False,kind='server_error')])
    result=reviewed_record(r,p,tmp_path);mask=json.loads(result['label_mask_json'])
    assert result['clickbait_label']=='' and not mask['curiosity_gap']
    assert result['consistency_tc']=='Consistent' and mask['consistency_tc']
    assert result['annotation_usage_incomplete'] and result['annotation_model_calls']==3

def test_unresolved_relation_is_null_not_fabricated_insufficient_evidence(tmp_path):
    r=record(tmp_path);a=negative();b=negative();c=negative()
    b.consistency_tc.label='Inconsistent';c.consistency_tc.label='Insufficient Evidence'
    result=reviewed_record(r,Fake([a,b,c]),tmp_path)
    assert result['consistency_tc']==''
    assert json.loads(result['annotation_raw_json'])['consistency_tc']['label'] is None
    assert json.loads(result['annotation_grounded_json'])['consistency_tc']['label'] is None

def test_quality_respects_provider_request_spacing(tmp_path,monkeypatch):
    from src.labeling import quality
    sleeps=[];monkeypatch.setattr(quality.time,'monotonic',lambda:100.)
    monkeypatch.setattr(quality.time,'sleep',lambda value:sleeps.append(value))
    r=record(tmp_path);p=Fake([negative(),negative()]);p.requests_per_minute=60
    reviewed_record(r,p,tmp_path)
    assert sleeps==[1.0]
