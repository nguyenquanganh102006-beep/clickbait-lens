import pytest
from scripts.emit_agent_votes import make_wire

def decision():
    return dict(types=[0,0,0,0,0],image_assessment='usable',image_observation='A person in a classroom.',
                relations=[dict(label='C',paragraph=1,reason='The body supports the title.'),dict(label='C',reason='The class image matches the topic.'),dict(label='C',paragraph=1,reason='The image illustrates the body.')])

def test_formatter_preserves_explicit_decisions_and_references():
    d=decision();d['types'][0]=1;d['support']={'curiosity_gap':dict(reason='The title hides the central answer.',paragraph=2)}
    w=make_wire({'paragraphs':[{},{}]},d)
    assert w.clickbait_label==1 and w.types.curiosity_gap.label==1
    assert w.types.curiosity_gap.evidence.content_paragraph_id==2
    assert w.consistency_ti.evidence.image_observation==d['image_observation']

def test_missing_agent_decisions_or_source_references_are_not_invented():
    d=decision();d['types']=[0]
    with pytest.raises(ValueError):make_wire({'paragraphs':[{}]},d)
    d=decision();d['relations'][0]['paragraph']=99
    with pytest.raises(ValueError):make_wire({'paragraphs':[{}]},d)
