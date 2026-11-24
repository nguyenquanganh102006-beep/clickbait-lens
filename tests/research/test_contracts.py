import json
from pathlib import Path
import pytest
import torch
from src.common.contracts import compute_loss, prediction_record, write_predictions
from src.common.model import MultiTaskHeads, SmokeModel
from src.common.runtime import save_checkpoint, load_checkpoint, seed_everything
from src.common.metrics import classification_metrics, compute_metrics


def targets(n=2):
    y = {
        "detection": torch.tensor([0.0, 1.0]),
        "types": torch.tensor([[0.0] * 5, [1.0, 0.0, 0.0, 0.0, 0.0]]),
        "consistency": torch.tensor([[0, -100, 2], [1, 0, 0]]),
    }
    masks = {k: torch.ones_like(v, dtype=torch.float32) for k, v in y.items()}
    masks["consistency"][0, 1] = 0
    return y, masks


def test_shapes_masked_loss_and_backward():
    model = MultiTaskHeads(8)
    out = model(torch.randn(2, 8))
    y, m = targets()
    assert {k: tuple(v.shape) for k, v in out.items()} == {
        "detection": (2,),
        "types": (2, 5),
        "consistency": (2, 3, 3),
    }
    loss, _ = compute_loss(out, y, m)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    changed = {k: v.detach().clone().requires_grad_() for k, v in out.items()}
    with torch.no_grad():
        changed["consistency"][0, 1] = torch.tensor([100.0, -100.0, 50.0])
    other, _ = compute_loss(changed, y, m)
    assert torch.allclose(loss, other)


def test_all_masked_loss_is_zero():
    out = MultiTaskHeads(8)(torch.randn(2, 8))
    y, m = targets()
    zero = {k: torch.zeros_like(v) for k, v in m.items()}
    loss, _ = compute_loss(out, y, zero)
    assert loss.item() == 0
    loss.backward()


def test_prediction_coverage_and_probabilities(tmp_path):
    out = MultiTaskHeads(8)(torch.randn(2, 8))
    record = prediction_record(
        "a",
        out,
        0,
        run_id="r",
        dataset_version="1.0",
        split="validation",
        task="multitask",
        label_origin="pseudo",
    )
    assert len(record["outputs"]["types"]) == 5
    assert abs(sum(record["outputs"]["consistency"]["tc"]["probabilities"]) - 1) < 1e-6
    path = tmp_path / "predictions.jsonl"
    with pytest.raises(ValueError):
        write_predictions(path, [record, record], ["a", "b"])
    assert not path.exists()
    write_predictions(path, [record], ["a"])
    assert json.loads(path.read_text())["article_id"] == "a"
    with pytest.raises(FileExistsError):
        write_predictions(path, [record], ["a"])


def test_checkpoint_roundtrip_and_identity_guard(tmp_path):
    seed_everything(20261124)
    model = SmokeModel()
    optimizer = torch.optim.AdamW(model.parameters())
    meta = {
        "run_id": "fixture",
        "dataset_version": "synthetic",
        "dataset_manifest_sha256": "m",
        "split_assignments_sha256": "s",
        "label_schema_sha256": "l",
        "git_commit": "test",
    }
    file = tmp_path / "last.pt"
    save_checkpoint(file, model, optimizer, metadata=meta, step=1, config={})
    copy = SmokeModel()
    state = load_checkpoint(
        file, copy, expected_split_sha256="s", expected_schema_sha256="l", restore_rng=True
    )
    assert state["step"] == 1
    x = torch.randn(2, 8)
    assert all(torch.equal(model(x)[k], copy(x)[k]) for k in model(x))
    with pytest.raises(ValueError):
        load_checkpoint(file, copy, expected_split_sha256="wrong", expected_schema_sha256="l")


def test_fixed_class_macro_f1():
    metric = classification_metrics([0, 0], [0, 0], [0, 1, 2])
    assert metric["accuracy"] == 1 and metric["macro_f1"] == pytest.approx(1 / 3)
    assert metric["per_class"]["1"]["support"] == 0


def test_masked_metrics_and_shared_thresholds():
    out = MultiTaskHeads(8)(torch.randn(2, 8))
    y, m = targets()
    empty = compute_metrics(out, y, {k: torch.zeros_like(v) for k, v in m.items()})
    assert empty["types"]["macro_f1"] is None and empty["consistency"]["tc"]["n"] == 0
    scores = {"detection": torch.tensor([0.0, 0.0])}
    metric = compute_metrics(
        scores,
        {"detection": torch.tensor([0.0, 0.0])},
        {"detection": torch.ones(2)},
        task="detection",
        detection_threshold=0.6,
    )
    assert metric["detection"]["accuracy"] == 1
    with pytest.raises(ValueError):
        compute_metrics(out, y, m, type_thresholds=[0.5])
