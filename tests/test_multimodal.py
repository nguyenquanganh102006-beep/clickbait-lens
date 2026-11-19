import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image
from src.labeling.multimodal import (
    MultimodalOutput, blank_mock, review_flags, mock_record, build_prompt,
    load_image, TYPES, export_csv, grounded_output, exact_quote, numeric_ratio_conflicts,
)

def record(tmp_path):
    (tmp_path/'images').mkdir(exist_ok=True)
    Image.new('RGB',(240,240),'white').save(tmp_path/'images/a_1.jpg')
    return dict(article_id='a',title='Tại sao gọi là bún thang?',content='Bài viết giải thích tên gọi của món bún thang.',
                lead='',image_caption='',image_path='images/a_1.jpg',image_valid='True',source_level='3',
                type_misleading_framing='',type_image_text_mismatch='',type_context_manipulation='')

def output_all_negative():
    result=blank_mock(True,True)
    for t in TYPES:getattr(result.types,t).label=0
    result.clickbait_label=0
    for k in ('consistency_tc','consistency_ti','consistency_ic'):getattr(result,k).label='Consistent'
    return result

def test_detection_follows_types_without_overriding_conflict(tmp_path):
    r=record(tmp_path);o=output_all_negative();o.types.curiosity_gap.label=1
    o.types.curiosity_gap.evidence.text_span=r['title']
    assert 'detection_type_logic_conflict' in review_flags(o,r,True)
    assert o.clickbait_label==0

def test_unknown_type_is_not_negative(tmp_path):
    r=record(tmp_path);o=output_all_negative();o.types.exaggeration.label=None
    assert 'detection_type_logic_conflict' in review_flags(o,r,True)
    o.clickbait_label=None
    assert not review_flags(o,r,True)

def test_clickbait_can_be_consistent_and_mismatch_is_not_automatic_type(tmp_path):
    r=record(tmp_path);o=output_all_negative();o.clickbait_label=1
    o.types.curiosity_gap.label=1;o.types.curiosity_gap.evidence.text_span=r['title']
    assert review_flags(o,r,True)==[]
    o=output_all_negative();o.consistency_tc.label='Inconsistent';o.consistency_tc.evidence.content_span=r['content']
    assert review_flags(o,r,True)==[]

def test_no_image_means_not_available_not_invented_observation(tmp_path):
    r=record(tmp_path);o=blank_mock(False,True)
    assert not review_flags(o,r,False)
    o.consistency_ti.label='Consistent';o.consistency_ti.evidence.image_observation='Một bát bún'
    flags=review_flags(o,r,False)
    assert 'consistency_ti:availability_conflict' in flags
    assert 'consistency_ti:image_not_observed' in flags

def test_placeholder_is_unavailable_and_quarantined(tmp_path):
    r=record(tmp_path);o=output_all_negative();o.image_assessment='placeholder'
    o.consistency_ti.label='Not Available';o.consistency_ic.label='Not Available'
    flags=review_flags(o,r,True)
    assert 'image_requires_human_review:placeholder' in flags
    assert 'consistency_ti:availability_conflict' not in flags
    assert 'negative_detection_without_visual_evidence' in flags

def test_evidence_accepts_layout_whitespace_but_not_ellipsis_or_changed_words():
    assert exact_quote('Câu một. Câu hai.','Câu một.\nCâu hai.')=='Câu một.\nCâu hai.'
    assert exact_quote('Câu một... Câu hai.','Câu một.\nCâu hai.') is None
    assert exact_quote('Câu sai.','Câu đúng.') is None

def test_reference_grounding_uses_source_without_changing_labels(tmp_path):
    r=record(tmp_path);r['content']='Đoạn thứ nhất.\nĐoạn thứ hai.';o=output_all_negative()
    o.consistency_tc.evidence.content_paragraph_id=2
    o.consistency_tc.evidence.text_reference='title'
    assert review_flags(o,r,True)==[]
    grounded=grounded_output(o,r)
    assert grounded.consistency_tc.evidence.content_span=='Đoạn thứ hai.'
    assert grounded.consistency_tc.evidence.text_span==r['title']
    assert grounded.clickbait_label==o.clickbait_label
    assert o.consistency_tc.evidence.content_span is None

def test_reference_ids_and_source_mismatch_do_not_bypass_validation(tmp_path):
    r=record(tmp_path);r['content']='Đoạn một.\nĐoạn hai.';o=output_all_negative()
    o.consistency_tc.evidence.content_paragraph_id=99
    assert 'consistency_tc:invalid_paragraph_reference' in review_flags(o,r,True)
    o.consistency_tc.evidence.content_paragraph_id=1;o.consistency_tc.evidence.content_span='Đoạn hai.'
    assert 'consistency_tc:paragraph_reference_mismatch' in review_flags(o,r,True)


def test_teaser_candidate_only_triggers_review_never_forces_label(tmp_path):
    r=record(tmp_path);r['title']='Tặng một món quà, người ấy lại nói với tôi thế này';o=output_all_negative()
    assert 'negative_detection_with_unresolved_teaser_candidate' in review_flags(o,r,True)
    assert o.clickbait_label==0

def test_narrow_numeric_screen_preserves_source_and_does_not_assign_type(tmp_path):
    bad={'content':'Trước đây phí là 500 đồng. Nay mức phí là 20 triệu đồng, gấp 40 lần.'}
    assert numeric_ratio_conflicts(bad)==['paragraph_1:money_ratio_claim_conflict']
    assert numeric_ratio_conflicts({'content':bad['content'].replace('500 đồng','500 nghìn đồng')})==[]
    assert numeric_ratio_conflicts({'content':'Doanh thu 20 triệu đồng, gấp 40 lần.'})==[]

def test_subtype_and_evidence_are_checked(tmp_path):
    r=record(tmp_path);o=output_all_negative();o.misleading_subtype=['framing']
    o.types.emotional_bait.evidence.content_span='Bằng chứng bịa không có trong bài'
    assert 'subtype_without_positive_misleading' in review_flags(o,r,True)
    assert 'emotional_bait:invented_content_evidence' in review_flags(o,r,True)

def test_legacy_type_and_partial_rejected():
    o=blank_mock(True,True).model_dump();o['types']['misleading']={}
    with pytest.raises(ValueError):MultimodalOutput.model_validate(o)
    o=blank_mock(True,True).model_dump();o['consistency_tc']['label']='partial'
    with pytest.raises(ValueError):MultimodalOutput.model_validate(o)



def test_input_boundaries_and_mock_never_publishes_train_labels(tmp_path):
    r=record(tmp_path)
    result=mock_record(r,tmp_path)
    assert result['annotation_status']=='mock' and result['clickbait_label']==''
    r['image_path']='../escape.jpg'
    with pytest.raises(ValueError):load_image(r,tmp_path)
    with pytest.raises(ValueError):build_prompt({'content':'x'*101},False,max_content_chars=100)

def test_openai_image_transport(monkeypatch,tmp_path):
    import openai
    from src.providers.openai import OpenAIProvider
    got={};o=output_all_negative()
    def create(**kwargs):
        got.update(kwargs);return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=o.model_dump_json(),refusal=None))],usage=None,model='vision')
    monkeypatch.setattr(openai,'OpenAI',lambda **kwargs:SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    p=OpenAIProvider('test-key','vision');p.generate_multimodal('prompt',MultimodalOutput,b'jpeg')
    assert got['messages'][1]['content'][1]['image_url']['url'].startswith('data:image/jpeg;base64,')

def test_gemini_image_transport(monkeypatch):
    from google import genai
    from src.providers.gemini import GeminiProvider
    got={};o=output_all_negative()
    def generate_content(**kwargs):got.update(kwargs);return SimpleNamespace(parsed=o,text=o.model_dump_json())
    monkeypatch.setattr(genai,'Client',lambda **kwargs:SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    GeminiProvider('test-key','vision').generate_multimodal('prompt',MultimodalOutput,b'jpeg')
    assert got['contents'][1].inline_data.data==b'jpeg'

def test_main_ui_runs_sample_and_retains_results(monkeypatch,tmp_path):
    from streamlit.testing.v1 import AppTest
    from src.ui import local_memory
    monkeypatch.setattr(local_memory,'STATE_DIR',tmp_path/'remembered')
    r=record(tmp_path);sample=tmp_path/'sample.csv'
    with sample.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(r));w.writeheader();w.writerow(r)
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run()
    app.radio(key='workflow_page').set_value('2. Gán nhãn đa phương thức').run()
    app.text_input(key='annotation_csv').set_value(str(sample))
    app.text_input(key='annotation_image_root').set_value(str(tmp_path))
    app.number_input(key='annotation_count').set_value(1)
    app.button(key='annotation_run').click().run()
    assert not app.exception
    assert app.session_state['annotation_results'][0]['annotation_status']=='mock'
