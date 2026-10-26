import json

import pytest

from src.labeling.engine import LabelingEngine, TypeAssessment, ConsistencyAssessment
from src.labeling.prompts import build_task_prompt, build_combined_prompt, load_taxonomy, load_task_prompt
from src.labeling.schemas import ConsistencyOutput, TypeOutput
from src.labeling.validation import LabelValidationError
from src.providers.mock import MockProvider
from src.providers.base import ProviderResponse
from src.jobs.checkpoint import build_cache_key
from src.data.exporter import review_reasons


def payload(prompt):
    return json.loads(prompt.rsplit('\n', 1)[1])


def test_context_separated_and_unknown_completeness_not_asserted():
    record = {'title': 'Title', 'lead': 'LEAD_CONTEXT', 'content': 'BODY_TEXT'}
    for prompt in [build_task_prompt('consistency', record, load_taxonomy()),
                   build_combined_prompt(record, ['type', 'consistency'], load_taxonomy())]:
        data = payload(prompt)
        assert data['lead'] == 'LEAD_CONTEXT'
        assert data['content'] == 'BODY_TEXT'
        assert data['content_is_complete'] is None
    assert 'LEAD_CONTEXT' not in build_task_prompt('detection', record, load_taxonomy())
    record['content'] = None
    assert payload(build_task_prompt('consistency', record, load_taxonomy(), content_is_complete=False))['lead'] is None


@pytest.mark.parametrize('mode', ['independent', 'combined'])
def test_explicit_incomplete_content_abstains_even_when_long(mode):
    class NoRequest(MockProvider):
        def generate_structured(self, *args):
            pytest.fail('Incomplete content must not trigger consistency API call')
    result = LabelingEngine(NoRequest()).label_record(
        {'title': 'Title', 'lead': 'valid looking lead', 'content': 'text ' * 100,
         'content_is_complete': False}, ['consistency'], mode=mode)
    assert result.outputs['consistency'].status == 'abstain'


def test_missing_evidence_rejected_in_v2_but_legacy_supported():
    class NoEvidence(MockProvider):
        def generate_structured(self, prompt, schema):
            data = ConsistencyOutput(status='success', label='consistent', confidence=.9, reason='Supported', evidence=None)
            if schema is ConsistencyAssessment:
                data = ConsistencyAssessment(body_coverage='substantive', body_evidence='body', output=data.model_dump())
            return ProviderResponse(data, data.model_dump_json(), 'test')
    record = {'title': 'Title', 'content': 'body ' * 40}
    with pytest.raises(LabelValidationError, match='bằng chứng'):
        LabelingEngine(NoEvidence()).label_record(record, ['consistency'])
    assert LabelingEngine(NoEvidence(), prompt_version='v1').label_record(record, ['consistency']).outputs['consistency'].status == 'success'
    assert '— v1' in load_task_prompt('detection', 'v1')


def test_context_changes_invalidate_v2_cache_but_not_title_only_or_legacy():
    args = dict(provider='gemini', model='test', taxonomy_version='v1', schema_version='v1', mode='independent', selected_tasks=['type'])
    a = {'title': 'Title', 'content': 'body', 'lead': 'first'}
    b = {**a, 'lead': 'second'}
    assert build_cache_key(a, 'type', prompt_version='v2', **args) != build_cache_key(b, 'type', prompt_version='v2', **args)
    assert build_cache_key(a, 'type', prompt_version='v1', **args) == build_cache_key(b, 'type', prompt_version='v1', **args)
    assert build_cache_key(a, 'detection', prompt_version='v2', **args) == build_cache_key(b, 'detection', prompt_version='v2', **args)


def test_disagreement_goes_to_review_even_with_perfect_confidence():
    def result(**output):
        return {'status': 'success', 'output': {'confidence': 1., **output}}
    tasks = {'detection': result(label=1), 'type': result(labels=[])}
    assert review_reasons(tasks)[0]['review_reason'] == 'detection_positive_type_empty'
    tasks['type'] = result(labels=['misleading'])
    tasks['consistency'] = result(label='consistent')
    assert review_reasons(tasks)[0]['review_reason'] == 'misleading_but_consistent'
    assert tasks['consistency']['output']['label'] == 'consistent'
    tasks = {'detection': result(label=0), 'type': result(labels=['emotional'])}
    assert review_reasons(tasks)[0]['review_reason'] == 'detection_negative_with_type_labels'
    tasks['type'] = result(labels=['misleading'])
    assert review_reasons(tasks) == []  # body-only misleading need not contradict title-only detection


def test_body_answer_cannot_erase_title_curiosity_gap():
    style = TypeOutput(status='success', labels=['curiosity_gap'], confidence=.9, reason='Hides outcome')
    check = TypeOutput(status='success', labels=[], confidence=1., reason='Article gives outcome')
    result = LabelingEngine(MockProvider())._merge_type_assessment(TypeAssessment(title_style=style, content_check=check), True)
    assert result.labels == ['curiosity_gap']
    check.labels = ['exaggeration', 'misleading']
    result = LabelingEngine(MockProvider())._merge_type_assessment(TypeAssessment(title_style=style, content_check=check), True)
    assert result.labels == ['curiosity_gap', 'exaggeration', 'misleading']


def test_lead_only_support_demotes_to_abstain_and_invented_quote_rejected():
    engine = LabelingEngine(MockProvider())
    output = ConsistencyOutput(status='success', label='consistent', confidence=1., reason='Lead matches', evidence='lead event')
    data = ConsistencyAssessment(body_coverage='context_only', body_evidence='general context', output=output.model_dump())
    assert engine._merge_consistency_assessment(data, {'content':'general context', 'lead':'lead event'}).status == 'abstain'
    data.output.status = 'abstain'
    data.output.label = 'inconsistent'
    assert engine._merge_consistency_assessment(data, {'content':'general context'}).label is None
    data.output.status = 'success'
    data.body_coverage = 'substantive'
    data.body_evidence = 'invented quotation'
    with pytest.raises(LabelValidationError, match='nguyên văn'):
        engine._merge_consistency_assessment(data, {'content':'general context'})
