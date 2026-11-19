"""Audit stored robust pilot artifacts without making API requests."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.labeling.quality import ReferenceOutput, normalized_output, consensus, result_hash, TYPES, RELATIONS


def audit(source, output):
    inputs = list(csv.DictReader(Path(source).open(encoding='utf-8-sig', newline='')))
    rows = list(csv.DictReader(Path(output).open(encoding='utf-8-sig', newline='')))
    assert [r['article_id'] for r in rows] == [r['article_id'] for r in inputs]
    assert len({r['article_id'] for r in rows}) == len(rows)
    details = []
    for original, row in zip(inputs, rows):
        for key in ('title', 'content', 'image_path'):
            assert row.get(key) == original.get(key), (row['article_id'], key)
        assert result_hash(row) == row['annotation_result_hash']
        assert row['human_review_status'] == 'pending'
        passes = json.loads(row['annotation_passes_json'])
        outputs = [normalized_output(ReferenceOutput.model_validate(p['wire_output'])) for p in passes]
        result, masks, _, _ = consensus(outputs, original, True)
        assert masks == json.loads(row['label_mask_json'])
        for target, valid in masks.items():
            column = 'clickbait_label' if target == 'detection' else 'type_' + target if target in TYPES else target
            value = result.clickbait_label if target == 'detection' else getattr(result.types, target).label if target in TYPES else getattr(result, target).label
            expected = str(value) if valid or value == 'Not Available' else ''
            assert row[column] == expected, (row['article_id'], column)
        details.append(dict(article_id=row['article_id'], status=row['annotation_status'],
                            detection=row['clickbait_label'], usable_targets=sum(masks.values()),
                            flags=json.loads(row['annotation_review_flags']),
                            reported_tokens=int(row['annotation_total_tokens'])))
    return dict(rows=len(rows), status=dict(Counter(r['status'] for r in details)),
                usable_targets=sum(r['usable_targets'] for r in details), possible_targets=len(rows)*9,
                reported_tokens=sum(r['reported_tokens'] for r in details), articles=details,
                checks=['unique ordered IDs', 'unchanged title/content/image path', 'CSV checksums',
                        'reference-only wire schema', 'recomputed valid-vote consensus and target masks',
                        'exported labels match consensus', 'human review remains pending'],
                limitation='Consensus and structural checks are not gold-label accuracy.')


if __name__ == '__main__':
    result = audit(sys.argv[1], sys.argv[2])
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(text + '\n', encoding='utf-8')
    print(text)
