import csv
import json
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from scripts.emit_agent_votes import make_wire
from scripts.run_gemini_autonomous import execute,model_worker_dir
from src.providers.base import ProviderError,ProviderResponse

class Provider:
    model='fake-model'
    provider_name='gemini'
    api_key='never-save-this-key'
    def __init__(self,quota_after=None):self.calls=0;self.quota_after=quota_after
    def generate_multimodal(self,prompt,schema,image,mime_type='image/jpeg'):
        self.calls+=1
        if self.quota_after is not None and self.calls>self.quota_after:
            raise ProviderError('Daily quota exhausted',kind='rate_limit',quota_exhausted=True,status_code=429)
        assert image
        wire=make_wire({'paragraphs':[{}]},dict(types=[0]*5,image_assessment='usable',
            image_observation='A classroom.',relations=[dict(label='C',paragraph=1,reason='The classroom opened.'),
            dict(label='C',reason='A classroom illustrates the title.'),dict(label='C',paragraph=1,reason='The image illustrates the opening.')]))
        return ProviderResponse(wire,'','fake-model',total_tokens=10)

def test_whole_queue_stops_at_quota_publishes_then_resumes_without_manual_batches(tmp_path):
    Image.new('RGB',(40,40),'white').save(tmp_path/'photo.jpg')
    source=tmp_path/'source.csv'
    records=[dict(article_id=f'tuan_{i}',contributor='Tuan',title='A classroom opens',lead='',content='A classroom opens today.',image_path='photo.jpg') for i in range(3)]
    with source.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    args=SimpleNamespace(run_dir=tmp_path/'run',csv=source,contributor='Tuan',image_root=tmp_path,
        destination=tmp_path/'final.csv',model='fake-model',timeout=30,rpm=6000,workers=1)
    provider=Provider(quota_after=2)
    assert execute(args,lambda:provider,poll_seconds=.01)==2
    assert provider.calls==3
    report=json.loads((args.run_dir/'autonomous_final_report.json').read_text())
    assert report['status']=='quota_exhausted' and report['published']==2
    assert len(list(csv.DictReader(args.destination.open(encoding='utf-8-sig'))))==2
    resumed=Provider()
    assert execute(args,lambda:resumed,poll_seconds=.01)==0
    assert resumed.calls==1
    rows=list(csv.DictReader(args.destination.open(encoding='utf-8-sig')))
    assert {r['article_id'] for r in rows}=={'tuan_0','tuan_1','tuan_2'}
    assert all(r['clickbait_label']=='0' and r['consistency_ti']=='Consistent' for r in rows)
    assert json.loads((args.run_dir/'autonomous_status.json').read_text())['manual_batch_assignment_required'] is False
    # Changing model must preserve earlier checkpoints and their published rows.
    old_dir=model_worker_dir(args);before=(old_dir/'checkpoint.jsonl').read_bytes()
    records.append({**records[0],'article_id':'tuan_3'})
    with source.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    args.model='second-model';second=Provider()
    assert model_worker_dir(args)!=old_dir
    assert execute(args,lambda:second,poll_seconds=.01)==0
    assert second.calls==1
    assert (old_dir/'checkpoint.jsonl').read_bytes()==before
    rows=list(csv.DictReader(args.destination.open(encoding='utf-8-sig')))
    assert {r['article_id'] for r in rows}=={'tuan_0','tuan_1','tuan_2','tuan_3'}
    assert next(r for r in rows if r['article_id']=='tuan_3')['annotation_requested_model']=='second-model'
    for path in args.run_dir.rglob('*'):
        if path.is_file():assert provider.api_key not in path.read_text(encoding='utf8')
