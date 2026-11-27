from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from scripts.validate_research_dataset import validate
from src.common.contracts import prediction_record, write_predictions
from src.common.dataset import ClickbaitDataset, file_sha256
from src.common.metrics import compute_metrics
from src.common.runtime import (
    load_checkpoint,
    log_event,
    new_run,
    save_checkpoint,
    seed_everything,
    write_json,
)
from src.consistency.ic_models import (
    CaptionNLIModel,
    GlobalICModel,
    HuggingFaceVLMEncoder,
    IC_RELATION_INDEX,
    OfflineHashEncoder,
    PassageICModel,
    chunk_content,
    extractive_summarize_content,
    truncate_content,
)
from src.consistency.imbalance import (
    ConsistencyImbalanceLoss,
    HardNegativeICDataset,
    compute_class_weights,
    make_balanced_sampler,
)


def load_experiment_config(path: str | Path) -> dict:
    """Load and validate an NV3 experiment YAML config."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if cfg.get("task") != "consistency":
        raise ValueError("NV3 config must set task: consistency")
    if cfg.get("seed") != 20261124:
        raise ValueError("Shared dataset seed is fixed at 20261124")
    if cfg.get("data", {}).get("dataset_version") != "1.0":
        raise ValueError("Unsupported dataset version")
    if not re.fullmatch(r"[a-z0-9-]+", cfg["model"]["name"]):
        raise ValueError("model.name must match [a-z0-9-]+")
    if cfg.get("experiment") not in ("c3_1", "c3_2", "c3_3", "c3_4", "c3_5", "c3_6"):
        raise ValueError(f"Unknown experiment: {cfg.get('experiment')}")
    return cfg


def build_encoder(cfg: dict, device: str):
    """Instantiate vision-language encoder from config."""
    enc_cfg = cfg["encoder"]
    backend = enc_cfg.get("backend", "offline-hash")
    if backend == "offline-hash":
        return OfflineHashEncoder(embed_dim=int(enc_cfg.get("embed_dim", 64)))
    if backend == "hf-vlm":
        return HuggingFaceVLMEncoder(
            model_id=enc_cfg["model_id"],
            revision=enc_cfg.get("revision"),
            device=device,
            max_length=int(enc_cfg.get("max_length", 64)),
        )
    raise ValueError(f"Unknown encoder backend: {backend}")


def build_model(cfg: dict, embed_dim: int) -> torch.nn.Module:
    """Instantiate the NV3 model corresponding to C3.1-C3.6."""
    exp = cfg["experiment"]
    m_cfg = cfg["model"]
    hidden_dim = int(m_cfg.get("hidden_dim", 256))
    dropout = float(m_cfg.get("dropout", 0.2))
    if exp in ("c3_1", "c3_2"):
        return GlobalICModel(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout)
    if exp in ("c3_3", "c3_4", "c3_5"):
        default_agg = {"c3_3": "similarity", "c3_4": "topk_mlp", "c3_5": "attention"}[exp]
        agg = m_cfg.get("aggregation", default_agg)
        return PassageICModel(
            embed_dim=embed_dim,
            aggregation=agg,
            top_k=int(m_cfg.get("top_k", 3)),
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    return CaptionNLIModel(calibrate=True)


def _extract_split_features(dataset, encoder, cfg: dict, split_name: str) -> dict:
    """Encode a dataset split into compact tensors for fast, reproducible training."""
    exp = cfg["experiment"]
    d_cfg = cfg["data"]
    max_chunks = int(d_cfg.get("max_chunks", 8))

    if exp == "c3_6":
        nli_dir = d_cfg.get("nli_logits_dir")
        if not nli_dir:
            raise ValueError(
                "Experiment c3_6 requires data.nli_logits_dir containing "
                "train.pt and validation.pt ({article_id: tensor[3]}) from Caption + NLI."
            )
        nli_map = torch.load(
            Path(nli_dir) / f"{split_name}.pt", map_location="cpu", weights_only=True
        )
        article_ids, targets_list, masks_list, logits_list = [], [], [], []
        for i in range(len(dataset)):
            sample = dataset[i]
            aid = sample["article_id"]
            if aid not in nli_map:
                raise KeyError(f"Missing precomputed NLI logits for article_id={aid}")
            article_ids.append(aid)
            targets_list.append(sample["targets"]["consistency"])
            masks_list.append(sample["masks"]["consistency"])
            logits_list.append(torch.as_tensor(nli_map[aid], dtype=torch.float32))
        return {
            "article_id": article_ids,
            "nli_logits": torch.stack(logits_list),
            "targets": torch.tensor(np.stack(targets_list), dtype=torch.int64),
            "masks": torch.tensor(np.stack(masks_list), dtype=torch.float32),
        }

    embed_dim = encoder.embed_dim
    article_ids = []
    targets_list = []
    masks_list = []
    img_embs: list[torch.Tensor] = []
    txt_embs: list[torch.Tensor] = []
    chunk_embs_all: list[torch.Tensor] = []
    chunk_masks_all: list[torch.Tensor] = []

    for i in range(len(dataset)):
        sample = dataset[i]
        feats = sample["features"]
        article_ids.append(sample["article_id"])
        targets_list.append(sample["targets"]["consistency"])
        masks_list.append(sample["masks"]["consistency"])

        img_vec = encoder.encode_images([feats["image"]])[0]
        img_embs.append(img_vec)

        if exp == "c3_1":
            text = truncate_content(feats["content"], max_words=int(d_cfg["truncate_words"]))
            txt_embs.append(encoder.encode_texts([text])[0])
        elif exp == "c3_2":
            text = extractive_summarize_content(
                feats["title"],
                feats["content"],
                feats.get("lead", ""),
                max_sentences=int(d_cfg["summary_sentences"]),
                max_words=int(d_cfg["summary_words"]),
            )
            txt_embs.append(encoder.encode_texts([text])[0])
        else:
            chunks = chunk_content(
                feats["content"],
                chunk_words=int(d_cfg["chunk_words"]),
                overlap_words=int(d_cfg["overlap_words"]),
                max_chunks=max_chunks,
            )
            c_vecs = encoder.encode_texts(chunks)
            padded = torch.zeros((max_chunks, embed_dim), dtype=torch.float32)
            mask = torch.zeros(max_chunks, dtype=torch.float32)
            k = min(len(chunks), max_chunks)
            padded[:k] = c_vecs[:k]
            mask[:k] = 1.0
            chunk_embs_all.append(padded)
            chunk_masks_all.append(mask)

    bundle = {
        "article_id": article_ids,
        "image_emb": torch.stack(img_embs),
        "targets": torch.tensor(np.stack(targets_list), dtype=torch.int64),
        "masks": torch.tensor(np.stack(masks_list), dtype=torch.float32),
    }
    if exp in ("c3_1", "c3_2"):
        bundle["text_emb"] = torch.stack(txt_embs)
    else:
        bundle["chunk_embs"] = torch.stack(chunk_embs_all)
        bundle["chunk_mask"] = torch.stack(chunk_masks_all)
    return bundle


def _make_synthetic_split(split: str, n: int, cfg: dict, seed_offset: int) -> dict:
    """Create a deterministic offline synthetic split for integration verification."""
    rng = np.random.default_rng(cfg["seed"] + seed_offset)
    embed_dim = int(cfg["encoder"].get("embed_dim", 64))
    max_chunks = int(cfg["data"].get("max_chunks", 8))

    ic_labels = np.array([0] * max(n - 6, 1) + [1, 1, 1, 2, 2, 2])[:n]
    rng.shuffle(ic_labels)
    targets = np.full((n, 3), -100, dtype=np.int64)
    targets[:, IC_RELATION_INDEX] = ic_labels
    masks = np.zeros((n, 3), dtype=np.float32)
    masks[:, IC_RELATION_INDEX] = 1.0

    img = torch.tensor(rng.normal(size=(n, embed_dim)), dtype=torch.float32)
    bundle = {
        "article_id": [f"synthetic_{split}_{i}" for i in range(n)],
        "image_emb": img,
        "targets": torch.from_numpy(targets),
        "masks": torch.from_numpy(masks),
    }
    if cfg["experiment"] in ("c3_1", "c3_2"):
        bundle["text_emb"] = torch.tensor(rng.normal(size=(n, embed_dim)), dtype=torch.float32)
    elif cfg["experiment"] == "c3_6":
        bundle["nli_logits"] = torch.tensor(rng.normal(size=(n, 3)), dtype=torch.float32)
    else:
        bundle["chunk_embs"] = torch.tensor(
            rng.normal(size=(n, max_chunks, embed_dim)), dtype=torch.float32
        )
        bundle["chunk_mask"] = torch.ones((n, max_chunks), dtype=torch.float32)
    return bundle


def _forward_batch(model: torch.nn.Module, exp: str, batch: tuple, device: torch.device) -> dict:
    if exp in ("c3_1", "c3_2"):
        img, txt, _, _ = batch
        return model(img.to(device), txt.to(device))
    if exp == "c3_6":
        nli_logits, _, _ = batch
        return model(nli_logits.to(device))
    img, chunks, cmask, _, _ = batch
    return model(img.to(device), chunks.to(device), cmask.to(device))


def _make_tensor_dataset(bundle: dict, exp: str) -> TensorDataset:
    if exp in ("c3_1", "c3_2"):
        return TensorDataset(
            bundle["image_emb"], bundle["text_emb"], bundle["targets"], bundle["masks"]
        )
    if exp == "c3_6":
        return TensorDataset(bundle["nli_logits"], bundle["targets"], bundle["masks"])
    return TensorDataset(
        bundle["image_emb"],
        bundle["chunk_embs"],
        bundle["chunk_mask"],
        bundle["targets"],
        bundle["masks"],
    )


def evaluate(
    model: torch.nn.Module,
    val_bundle: dict,
    exp: str,
    batch_size: int,
    device: torch.device,
) -> tuple[dict, dict]:
    """Run validation forward pass and compute fixed-class Macro-F1 metrics."""
    model.eval()
    loader = DataLoader(_make_tensor_dataset(val_bundle, exp), batch_size=batch_size, shuffle=False)
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            out = _forward_batch(model, exp, batch, device)
            all_logits.append(out["consistency"].cpu())
    outputs = {"consistency": torch.cat(all_logits, dim=0)}
    targets = {"consistency": val_bundle["targets"]}
    masks = {"consistency": val_bundle["masks"]}
    metrics = compute_metrics(outputs, targets, masks, task="consistency")
    return outputs, metrics


def run(
    config_path: str | Path,
    dataset_root: str | Path | None = None,
    *,
    synthetic: bool = False,
    output_root: str | Path = ROOT,
) -> Path:
    """Execute a full NV3 train + validation experiment and emit contract artifacts."""
    cfg = load_experiment_config(config_path)
    seed_everything(cfg["seed"])

    dev_pref = cfg["training"].get("device", "auto")
    if dev_pref == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = dev_pref
    device = torch.device(device_str)
    exp = cfg["experiment"]

    if synthetic:
        n_syn = max(int(cfg["training"]["batch_size"]), 12)
        train_bundle = _make_synthetic_split("train", n_syn, cfg, seed_offset=1)
        val_bundle = _make_synthetic_split("val", n_syn, cfg, seed_offset=2)
        embed_dim = int(cfg["encoder"].get("embed_dim", 64))
        origin = "synthetic"
        fingerprint = "synthetic-fixture-not-a-dataset-release"
        manifest_hash = split_hash = schema_hash = fingerprint
    else:
        if dataset_root is None:
            raise ValueError("--dataset-root is required unless --synthetic is passed")
        root = Path(dataset_root)
        validate(root)
        encoder = None if exp == "c3_6" else build_encoder(cfg, device_str)
        embed_dim = (
            int(cfg["encoder"].get("embed_dim", 64)) if encoder is None else encoder.embed_dim
        )

        train_ds = ClickbaitDataset(root, "train", image_root=cfg["data"].get("image_root"))
        val_ds = ClickbaitDataset(
            root, "validation", verify_checksums=False, image_root=cfg["data"].get("image_root")
        )
        hn_ratio = float(cfg["imbalance"].get("hard_negative_ratio", 0.0))
        if hn_ratio > 0 and exp != "c3_6":
            train_ds = HardNegativeICDataset(
                train_ds, negative_ratio=hn_ratio, seed=cfg["seed"]
            )

        train_bundle = _extract_split_features(train_ds, encoder, cfg, "train")
        val_bundle = _extract_split_features(val_ds, encoder, cfg, "validation")
        origin = "pseudo"
        manifest_hash = file_sha256(root / "manifest.json")
        split_hash = file_sha256(root / "split_assignments.csv")
        schema_hash = file_sha256(root / "label_schema.json")

    run_id, run_path, metadata = new_run(cfg, output_root)
    metadata.update(
        {
            "dataset_version": "synthetic" if synthetic else "1.0",
            "dataset_manifest_sha256": manifest_hash,
            "split_assignments_sha256": split_hash,
            "label_schema_sha256": schema_hash,
            "label_origin": origin,
            "official_benchmark": False,
            "device": device_str,
            "experiment": exp,
        }
    )
    write_json(run_path / "run.json", metadata)
    events_path = run_path / "events.jsonl"
    log_event(events_path, event="start", run_id=run_id, experiment=exp, synthetic=synthetic)

    imb_cfg = cfg["imbalance"]
    class_weights = compute_class_weights(
        train_bundle["targets"],
        train_bundle["masks"],
        relation_idx=IC_RELATION_INDEX,
        strategy=imb_cfg.get("weight_strategy", "inverse_sqrt"),
    )
    criterion = ConsistencyImbalanceLoss(
        loss_type=imb_cfg.get("loss_type", "focal"),
        class_weights=class_weights,
        focal_gamma=float(imb_cfg.get("focal_gamma", 2.0)),
        relation_idx=IC_RELATION_INDEX,
    ).to(device)

    train_tds = _make_tensor_dataset(train_bundle, exp)
    sampler = None
    if imb_cfg.get("use_balanced_sampler", True):
        ic_train_labels = train_bundle["targets"][:, IC_RELATION_INDEX].numpy()
        sampler = make_balanced_sampler(
            ic_train_labels,
            power=float(imb_cfg.get("sampler_power", 0.5)),
            seed=cfg["seed"],
        )

    batch_size = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(
        train_tds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=int(cfg["training"].get("num_workers", 0)),
    )

    model = build_model(cfg, embed_dim=embed_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
    )

    checkpoint_dir = Path(output_root) / "checkpoints" / run_id
    result_dir = Path(output_root) / "results" / run_id
    best_score = -1.0
    best_outputs = None
    best_metrics = None
    global_step = 0

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            out = _forward_batch(model, exp, batch, device)
            y_cons = batch[-2].to(device)
            m_cons = batch[-1].to(device)
            loss = criterion(out["consistency"], y_cons, m_cons)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss at step {global_step}")
            loss.backward()
            grad_clip = float(cfg["training"].get("grad_clip", 1.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        val_outputs, val_metrics = evaluate(model, val_bundle, exp, batch_size, device)
        ic_macro_f1 = val_metrics["consistency"]["ic"]["macro_f1"]
        score = -1.0 if ic_macro_f1 is None else float(ic_macro_f1)
        log_event(
            events_path,
            event="epoch_end",
            step=global_step,
            epoch=epoch,
            train_loss=float(np.mean(epoch_losses)),
            val_ic_macro_f1=score,
        )

        if score >= best_score:
            best_score = score
            best_outputs = val_outputs
            best_metrics = val_metrics
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                metadata=metadata,
                step=global_step,
                config=cfg,
            )

    save_checkpoint(
        checkpoint_dir / "last.pt",
        model,
        optimizer,
        metadata=metadata,
        step=global_step,
        config=cfg,
    )

    reloaded = build_model(cfg, embed_dim=embed_dim).to(device)
    load_checkpoint(
        checkpoint_dir / "best.pt",
        reloaded,
        expected_split_sha256=split_hash,
        expected_schema_sha256=schema_hash,
    )
    roundtrip_outputs, _ = evaluate(reloaded, val_bundle, exp, batch_size, device)
    if not torch.allclose(best_outputs["consistency"], roundtrip_outputs["consistency"], atol=1e-5):
        raise AssertionError("best.pt roundtrip changed validation predictions")

    records = [
        prediction_record(
            aid,
            best_outputs,
            i,
            run_id=run_id,
            dataset_version=metadata["dataset_version"],
            split="validation",
            task="consistency",
            label_origin=origin,
        )
        for i, aid in enumerate(val_bundle["article_id"])
    ]
    write_predictions(result_dir / "predictions.jsonl", records, val_bundle["article_id"])
    write_json(
        result_dir / "metrics.json",
        {
            "format_version": "1.0",
            "run_id": run_id,
            "split": "validation",
            "label_origin": origin,
            "official_benchmark": False,
            "n": len(records),
            "selected_metric": "ic.macro_f1",
            "best_ic_macro_f1": best_score,
            "metrics": best_metrics,
        },
    )
    log_event(
        events_path,
        event="complete",
        step=global_step,
        best_ic_macro_f1=best_score,
        checkpoint_roundtrip=True,
        n_predictions=len(records),
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "experiment": exp,
                "best_ic_macro_f1": round(best_score, 4),
                "predictions": len(records),
            }
        )
    )
    return run_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NV3 Image-Content Consistency model.")
    parser.add_argument("--config", default=str(ROOT / "configs/consistency_ic.yaml"))
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    run(args.config, args.dataset_root, synthetic=args.synthetic)