from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts.train_consistency_ic import run
from src.consistency.ic_models import (
    CaptionNLIModel,
    GlobalICModel,
    PassageICModel,
    chunk_content,
    extractive_summarize_content,
    truncate_content,
)
from src.consistency.imbalance import (
    ConsistencyImbalanceLoss,
    HardNegativeICDataset,
    compute_class_weights,
)

ROOT = Path(__file__).resolve().parents[2]


def test_long_text_chunking_and_summary():
    words = [f"tu{i}" for i in range(500)]
    content = " ".join(words)
    assert len(truncate_content(content, max_words=120).split()) == 120
    chunks = chunk_content(content, chunk_words=100, overlap_words=20, max_chunks=5)
    assert len(chunks) == 5
    assert len(chunks[0].split()) == 100
    summary = extractive_summarize_content(
        "Gia vang tang manh",
        "Gia vang hom nay tang manh. Nhieu nha dau tu quan tam. Thi truong on dinh.",
        max_sentences=2,
    )
    assert "Gia vang" in summary


@pytest.mark.parametrize("agg", ["similarity", "topk_mlp", "max", "mean", "topk", "attention"])
def test_passage_ic_model_contract_shapes(agg):
    model = PassageICModel(embed_dim=32, aggregation=agg, top_k=2, hidden_dim=64)
    img = torch.randn(4, 32)
    chunks = torch.randn(4, 5, 32)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 1, 0, 0, 0]])
    out = model(img, chunks, mask)
    assert tuple(out["consistency"].shape) == (4, 3, 3)
    assert torch.isfinite(out["consistency"]).all()


def test_global_and_caption_nli_models():
    global_model = GlobalICModel(embed_dim=32, hidden_dim=64)
    out_g = global_model(torch.randn(3, 32), torch.randn(3, 32))
    assert tuple(out_g["consistency"].shape) == (3, 3, 3)

    nli_model = CaptionNLIModel(calibrate=True)
    out_n = nli_model(torch.randn(3, 3))
    assert tuple(out_n["consistency"].shape) == (3, 3, 3)


def test_imbalance_loss_respects_masks_and_all_masked_zero():
    targets = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, -100]])
    masks = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    weights = compute_class_weights(targets, masks, strategy="inverse_sqrt")
    assert weights.shape == (3,)

    logits = torch.randn(3, 3, 3, requires_grad=True)
    for loss_type in ("cross_entropy", "weighted_ce", "focal"):
        loss_fn = ConsistencyImbalanceLoss(loss_type=loss_type, class_weights=weights)
        loss = loss_fn(logits, targets, masks)
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None
        logits.grad.zero_()

    zero_loss = ConsistencyImbalanceLoss(loss_type="focal")(
        logits, targets, torch.zeros_like(masks)
    )
    assert zero_loss.item() == 0.0
    zero_loss.backward()


def test_hard_negative_rejects_validation_split():
    class DummyValDataset:
        split = "validation"

        def __len__(self):
            return 4

    with pytest.raises(ValueError, match="strictly train-only"):
        HardNegativeICDataset(DummyValDataset(), negative_ratio=0.25)


@pytest.mark.parametrize("exp", ["c3_1", "c3_3", "c3_5", "c3_6"])
def test_synthetic_end_to_end_nv3(tmp_path, exp):
    cfg = yaml.safe_load((ROOT / "configs/consistency_ic.yaml").read_text(encoding="utf-8"))
    cfg["experiment"] = exp
    cfg["training"]["epochs"] = 2
    cfg["training"]["batch_size"] = 8
    cfg_path = tmp_path / f"{exp}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    run_path = run(cfg_path, synthetic=True, output_root=tmp_path)
    result_dir = tmp_path / "results" / run_path.name
    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["split"] == "validation"
    assert metrics["metrics"]["consistency"]["ic"]["macro_f1"] is not None
    assert (tmp_path / "checkpoints" / run_path.name / "best.pt").exists()
    assert len((result_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 12