import pytest
from src.labeling.probe import probe_labeling
from src.labeling.schemas import DetectionOutput
from src.providers.base import ProviderResponse
from src.providers.mock import MockProvider


def test_schema_valid_error_output_does_not_verify_model():
    class ErrorProvider(MockProvider):
        def generate_structured(self, prompt, schema):
            data = DetectionOutput(status="error", label=None, reason="Cannot label", confidence=None)
            return ProviderResponse(data, data.model_dump_json(), "fake")
    with pytest.raises(ValueError, match="detection"):
        probe_labeling(ErrorProvider(), ["detection"])


@pytest.mark.parametrize("mode", ["independent", "combined"])
def test_probe_verifies_all_selected_tasks(mode):
    assert len(probe_labeling(MockProvider(), ["detection", "type", "consistency"], mode).outputs) == 3
