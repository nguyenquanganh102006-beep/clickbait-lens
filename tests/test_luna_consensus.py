import csv
import json
from PIL import Image
from scripts.assemble_luna_consensus import assemble
from src.labeling.multimodal import blank_mock,TYPES,RELATIONS
from src.labeling.quality import result_hash


def test_two_blind_agent_votes_create_labels_without_gemini_calls(tmp_path):
    Image.new('RGB',(32,32),'white').save(tmp_path/'image.jpg')
    record=dict(article_id='duan_1',contributor='Lê Văn Duẩn',title='Music rehearsal',content='People rehearse on stage.',image_path='image.jpg')
    source=tmp_path/'source.csv'
    with source.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=record);w.writeheader();w.writerow(record)
    wire=blank_mock(True,True).model_dump();wire['clickbait_label']=0
    for t in TYPES:wire['types'][t]['label']=0
    for t in RELATIONS:
        wire[t]['label']='Consistent';wire[t]['evidence'].update(content_paragraph_id=1,image_observation='A person on a stage.')
    for d in list(wire['types'].values())+[wire[t] for t in RELATIONS]:
        d['evidence'].pop('text_span');d['evidence'].pop('content_span')
    run=tmp_path/'run';run.mkdir()
    for filename,annotator in [('luna_independent_one.jsonl','luna-thread-one'),('luna_secondary_two.jsonl','luna-thread-two')]:
        (run/filename).write_text(json.dumps(dict(article_id='duan_1',annotator=annotator,wire_output=wire))+'\n',encoding='utf-8')
    report=assemble(run,source,tmp_path)
    assert report['ready']==1 and not report['pending']
    row=json.loads((run/'luna_consensus_checkpoint.jsonl').read_text(encoding='utf-8'))['result']
    assert row['annotation_model_calls']==0 and row['annotation_agent_evaluations']==2
    assert row['annotation_total_tokens'] is None and row['annotation_usage_incomplete']
    assert row['label_source']=='AI:Luna peer consensus' and row['human_review_status']=='pending'
    assert row['annotation_result_hash']==result_hash(row)
