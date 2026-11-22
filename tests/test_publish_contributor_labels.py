import json
import csv
import pytest
from scripts import publish_contributor_labels as publisher
from src.labeling.multimodal import blank_mock,TYPES,RELATIONS
from src.labeling.quality import result_hash


def sample():
    data=blank_mock(True,True).model_dump()
    for t in TYPES:data['types'][t]['label']=0
    data['clickbait_label']=0
    for t in RELATIONS:
        data[t]['label']='Consistent'
        data[t]['evidence'].update(content_paragraph_id=1,image_observation='A person on a stage.')
    for d in list(data['types'].values())+[data[t] for t in RELATIONS]:
        d['evidence'].pop('text_span');d['evidence'].pop('content_span')
    row=dict(article_id='duan_test',title='Music rehearsal',content='People rehearse on stage.',
             annotation_input_hash='original-input',annotation_passes_json='[]',annotation_model_calls=2,
             annotation_status='pseudo_labeled',clickbait_label=0,
             label_mask_json=json.dumps({t:True for t in (*TYPES,'detection',*RELATIONS)}))
    for t in TYPES:row['type_'+t]=0
    for t in RELATIONS:row[t]='Consistent'
    row['annotation_result_hash']=result_hash(row)
    return row,data


def test_agent_adjudication_preserves_original_api_responses_and_requires_two_annotators():
    original,wire=sample()
    votes=[dict(annotator='luna',wire_output=wire),dict(annotator='leader_ai',wire_output=wire)]
    corrected=publisher.adjudicate(original,votes)
    assert corrected['annotation_source_result_hash']==original['annotation_result_hash']
    assert corrected['annotation_passes_json']==original['annotation_passes_json']
    assert corrected['annotation_model_calls']==2
    assert corrected['human_review_status']=='pending'
    assert corrected['annotation_result_hash']==result_hash(corrected)
    with pytest.raises(ValueError):publisher.adjudicate(original,[votes[0],votes[0]])


def test_luna_disagreement_without_two_valid_votes_is_not_published(tmp_path):
    original,wire=sample();wire['consistency_ti']['label']='Inconsistent'
    run=tmp_path/'run';run.mkdir()
    (run/'checkpoint.jsonl').write_text(json.dumps(dict(result=original))+'\n',encoding='utf-8')
    (run/'luna_independent_test.jsonl').write_text(json.dumps(dict(article_id=original['article_id'],annotator='luna',wire_output=wire))+'\n',encoding='utf-8')
    report=publisher.publish(run,tmp_path/'final.csv')
    assert report['published']==0 and report['needs_review']==1
    assert report['luna_gemini_disagreements'][0]['targets']==['consistency_ti']


def test_live_snapshot_ignores_incomplete_last_jsonl_line(tmp_path):
    path=tmp_path/'live.jsonl';path.write_bytes(b'{"complete":1}\n{"part')
    assert publisher.read_lines(path)==[{'complete':1}]


def test_canonical_fast_title_is_not_overwritten_by_old_category_vote(tmp_path):
    original,_=sample();aid=original['article_id']
    old=dict(original,title='Category',annotation_status='needs_review')
    old['annotation_result_hash']=result_hash(old)
    fixed=dict(original,title='Actual article headline',source_original_title='Category',annotation_input_variant='og_title_v1',annotation_quality_mode='multimodal-fast-v1-single-pass')
    fixed['annotation_result_hash']=result_hash(fixed)
    run=tmp_path/'run';(run/'fast_run').mkdir(parents=True)
    (run/'checkpoint.jsonl').write_text(json.dumps({'result':original})+'\n')
    (run/'fast_run/checkpoint.jsonl').write_text(json.dumps({'result':fixed})+'\n')
    (run/'luna_consensus_checkpoint.jsonl').write_text(json.dumps({'result':old})+'\n')
    (run/'source_title_review_holds.json').write_text(json.dumps({'rows':[dict(article_id=aid,title='Category',og_title='Actual article headline')]}))
    report=publisher.publish(run,tmp_path/'final.csv')
    rows=list(csv.DictReader((tmp_path/'final.csv').open(encoding='utf-8-sig')))
    assert report['published']==1 and rows[0]['title']=='Actual article headline'


def test_published_checksums_include_empty_columns_from_other_rows(tmp_path):
    original,wire=sample();other=dict(original,article_id='duan_other')
    other['annotation_result_hash']=result_hash(other)
    run=tmp_path/'run';run.mkdir()
    (run/'checkpoint.jsonl').write_text(''.join(json.dumps(dict(result=r))+'\n' for r in [original,other]),encoding='utf-8')
    (run/'luna_independent_test.jsonl').write_text(json.dumps(dict(article_id=original['article_id'],annotator='luna',wire_output=wire))+'\n',encoding='utf-8')
    publisher.publish(run,tmp_path/'final.csv')
    rows=list(csv.DictReader((tmp_path/'final.csv').open(encoding='utf-8-sig')))
    assert len(rows)==2
    assert all(result_hash(r)==r['annotation_result_hash'] for r in rows)


def test_two_valid_gemini_votes_can_outvote_one_luna_vote(tmp_path):
    original,wire=sample()
    original['annotation_passes_json']=json.dumps([dict(phase=phase,model='gemini-test',wire_output=wire) for phase in ['primary','blind_review']])
    original['annotation_result_hash']=result_hash(original)
    wire=json.loads(json.dumps(wire));wire['consistency_ti']['label']='Inconsistent'
    run=tmp_path/'run';run.mkdir()
    (run/'checkpoint.jsonl').write_text(json.dumps(dict(result=original))+'\n',encoding='utf-8')
    (run/'luna_independent_test.jsonl').write_text(json.dumps(dict(article_id=original['article_id'],annotator='luna',wire_output=wire))+'\n',encoding='utf-8')
    report=publisher.publish(run,tmp_path/'final.csv')
    rows=list(csv.DictReader((tmp_path/'final.csv').open(encoding='utf-8-sig')))
    assert report['published']==1 and rows[0]['consistency_ti']=='Consistent'
    assert rows[0]['label_source']=='AI:Gemini+Luna consensus'


def test_tied_two_against_two_votes_remain_unresolved():
    from src.labeling.quality import ReferenceOutput,normalized_output,consensus
    original,wire=sample();a=normalized_output(ReferenceOutput.model_validate(wire))
    b=a.model_copy(deep=True);b.consistency_ti.label='Inconsistent'
    _,masks,_,_=consensus([a,a.model_copy(deep=True),b,b.model_copy(deep=True)],original,True)
    assert not masks['consistency_ti']


def test_corrupted_agent_free_text_is_rejected():
    from src.labeling.quality import ReferenceOutput
    _,wire=sample()
    wire['consistency_ti']['evidence']['image_observation']='?nh nh?n vi?n y t? ?ang ti?m cho m?t ng??i.'
    with pytest.raises(ValueError,match='encoding'):
        publisher.validate_vote_text(ReferenceOutput.model_validate(wire))


def test_published_labels_cannot_drift_from_leader_review_without_recheck(tmp_path):
    original,_=sample();run=tmp_path/'run';run.mkdir()
    (run/'checkpoint.jsonl').write_text(json.dumps(dict(result=original))+'\n',encoding='utf-8')
    receipt=dict(article_id=original['article_id'],review_status='accepted',expected_labels={'clickbait_label':1})
    (run/'leader_audit.jsonl').write_text(json.dumps(receipt)+'\n',encoding='utf-8')
    report=publisher.publish(run,tmp_path/'final.csv')
    assert report['published']==0 and report['needs_review']==1
