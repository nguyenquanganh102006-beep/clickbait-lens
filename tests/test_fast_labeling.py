import json
from types import SimpleNamespace
from PIL import Image
from scripts.emit_agent_votes import make_wire
from src.providers.base import ProviderResponse
from src.labeling.fast import annotate_record,export_vote

def setup(tmp_path):
    Image.new('RGB',(40,40),'white').save(tmp_path/'photo.jpg')
    record=dict(article_id='sample',title='A classroom opens',lead='',content='A classroom opens today.',image_path='photo.jpg')
    decision=dict(types=[0]*5,image_assessment='usable',image_observation='A classroom.',relations=[
        dict(label='C',paragraph=1,reason='The body reports the classroom opening.'),
        dict(label='C',reason='The classroom illustrates the title.'),
        dict(label='C',paragraph=1,reason='The classroom illustrates the body.')])
    return record,make_wire({'paragraphs':[{}]},decision)

def test_fast_provider_uses_one_call_with_actual_image_and_honest_provenance(tmp_path):
    record,wire=setup(tmp_path);calls=[]
    def generate(prompt,schema,image):
        calls.append(image)
        return ProviderResponse(wire,'','fake',total_tokens=12)
    provider=SimpleNamespace(provider_name='gemini',model='fake',generate_multimodal=generate)
    row=annotate_record(record,provider,tmp_path)
    assert len(calls)==1 and calls[0]==(tmp_path/'photo.jpg').read_bytes()
    assert row['annotation_model_calls']==1 and row['annotation_raw_is_consensus'] is False
    assert len(json.loads(row['annotation_passes_json']))==1
    assert row['human_review_status']=='pending'

def test_single_agent_does_not_invent_quorum_or_turn_unknown_into_zero(tmp_path):
    record,wire=setup(tmp_path)
    wire.types.curiosity_gap.label=None;wire.clickbait_label=None
    row=export_vote(record,wire.model_dump(),tmp_path,model='gpt-6-luna',requested_model='gpt-6-luna',source='AI:Luna fast single pass',provider_name='luna-agents',agent_vote={'annotator':'one-agent'})
    assert row['clickbait_label']=='' and row['type_curiosity_gap']==''
    assert row['annotation_status']=='needs_review'
    assert row['annotation_model_calls']==0 and row['annotation_agent_evaluations']==1
    assert json.loads(row['annotation_passes_json'])==[]
