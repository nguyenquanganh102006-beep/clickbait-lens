from pathlib import Path

import yaml

from src.labeling.prompts import build_combined_prompt, build_task_prompt, load_taxonomy


def test_detection_prompt_does_not_send_body_or_record_identifier():
    record = {
        "record_id": "private-record-id",
        "title": "Một tiêu đề",
        "content": "Nội dung toàn văn riêng tư",
    }

    prompt = build_task_prompt("detection", record, load_taxonomy())

    assert "Một tiêu đề" in prompt
    assert "Nội dung toàn văn riêng tư" not in prompt
    assert "private-record-id" not in prompt
    assert "content_is_complete" not in prompt


def test_combined_prompt_only_sends_fields_needed_by_selected_tasks():
    record = {
        "record_id": "private-record-id",
        "title": "Một tiêu đề",
        "content": "Nội dung toàn văn riêng tư",
    }

    detection_prompt = build_combined_prompt(record, ["detection"], load_taxonomy())
    consistency_prompt = build_combined_prompt(record, ["consistency"], load_taxonomy())

    assert "Nội dung toàn văn riêng tư" not in detection_prompt
    assert "Nội dung toàn văn riêng tư" in consistency_prompt
    assert "private-record-id" not in detection_prompt
    assert "private-record-id" not in consistency_prompt


def test_taxonomy_can_move_to_an_explicit_new_version(tmp_path: Path):
    path = tmp_path / "taxonomy.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": "v2",
                "clickbait_types": ["curiosity_gap", "listicle"],
                "consistency_labels": ["consistent", "partial", "inconsistent"],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    taxonomy = load_taxonomy(path)

    assert taxonomy["version"] == "v2"
    assert taxonomy["clickbait_types"] == ["curiosity_gap", "listicle"]
