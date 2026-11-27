from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

IC_RELATION_INDEX = 2
NUM_CONSISTENCY_CLASSES = 3


def compute_class_weights(
    targets: np.ndarray | torch.Tensor,
    masks: np.ndarray | torch.Tensor | None = None,
    *,
    relation_idx: int = IC_RELATION_INDEX,
    strategy: str = "inverse_sqrt",
    beta: float = 0.999,
) -> torch.Tensor:
    """Compute 3-class weights strictly from training labels.

    Args:
        targets: Array/Tensor of shape [N, 3] with values in {0, 1, 2, -100}.
        masks: Optional validity mask of shape [N, 3] (1=valid, 0=NA).
        relation_idx: 0 for tc, 1 for ti, 2 for ic.
        strategy: One of "uniform", "inverse", "inverse_sqrt", "effective_num".
        beta: Effective-number hyperparameter (used when strategy="effective_num").

    Returns:
        Float32 tensor of shape [3], normalized so mean(weights) == 1.0.
    """
    if strategy == "uniform":
        return torch.ones(NUM_CONSISTENCY_CLASSES, dtype=torch.float32)

    y = np.asarray(targets)[:, relation_idx]
    valid = y != -100
    if masks is not None:
        valid &= np.asarray(masks)[:, relation_idx].astype(bool)
    y_valid = y[valid]

    counts = np.array(
        [max(int((y_valid == c).sum()), 1) for c in range(NUM_CONSISTENCY_CLASSES)],
        dtype=np.float64,
    )

    if strategy == "inverse":
        raw = counts.sum() / (NUM_CONSISTENCY_CLASSES * counts)
    elif strategy == "inverse_sqrt":
        raw = np.sqrt(counts.sum() / counts)
    elif strategy == "effective_num":
        effective = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        raw = 1.0 / np.maximum(effective, 1e-12)
    else:
        raise ValueError(f"Unknown class weight strategy: {strategy}")

    normalized = raw / raw.mean()
    return torch.tensor(normalized, dtype=torch.float32)


class ConsistencyImbalanceLoss(nn.Module):
    """Masked Cross Entropy / Weighted CE / Focal Loss for consistency logits [B, 3, 3].

    Preserves the repo contract: Not Available (-100) or mask=0 entries are ignored,
    and an all-masked batch returns a differentiable zero tensor.
    """

    def __init__(
        self,
        *,
        loss_type: str = "weighted_ce",
        class_weights: Sequence[float] | torch.Tensor | None = None,
        focal_gamma: float = 2.0,
        relation_idx: int | None = IC_RELATION_INDEX,
    ) -> None:
        super().__init__()
        if loss_type not in ("cross_entropy", "weighted_ce", "focal"):
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        self.loss_type = loss_type
        self.focal_gamma = float(focal_gamma)
        self.relation_idx = relation_idx
        if class_weights is not None:
            weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
            if weight_tensor.shape != (NUM_CONSISTENCY_CLASSES,):
                raise ValueError("class_weights must have shape [3]")
            self.register_buffer("class_weights", weight_tensor)
        else:
            self.register_buffer("class_weights", None)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar masked loss from consistency logits [B, 3, 3]."""
        if logits.ndim != 3 or logits.shape[1:] != (3, 3):
            raise ValueError(f"Expected consistency logits [B, 3, 3], got {tuple(logits.shape)}")

        if self.relation_idx is not None:
            rel_logits = logits[:, self.relation_idx, :]
            rel_targets = targets[:, self.relation_idx].long()
            rel_masks = masks[:, self.relation_idx].to(logits.dtype)
        else:
            rel_logits = logits.reshape(-1, NUM_CONSISTENCY_CLASSES)
            rel_targets = targets.reshape(-1).long()
            rel_masks = masks.reshape(-1).to(logits.dtype)

        valid = rel_masks * (rel_targets != -100).to(logits.dtype)
        weight = self.class_weights if self.loss_type in ("weighted_ce", "focal") else None

        if self.loss_type == "focal":
            ce_unweighted = F.cross_entropy(
                rel_logits,
                rel_targets,
                ignore_index=-100,
                reduction="none",
            )
            pt = torch.exp(-ce_unweighted)
            focal_factor = (1.0 - pt).pow(self.focal_gamma)
            if weight is not None:
                safe_targets = rel_targets.clamp(min=0)
                alpha = weight[safe_targets]
                per_item = alpha * focal_factor * ce_unweighted
            else:
                per_item = focal_factor * ce_unweighted
        else:
            per_item = F.cross_entropy(
                rel_logits,
                rel_targets,
                weight=weight,
                ignore_index=-100,
                reduction="none",
            )

        return (per_item * valid).sum() / valid.sum().clamp_min(1.0)


def make_balanced_sampler(
    labels: np.ndarray | Sequence[int],
    *,
    power: float = 0.5,
    seed: int = 20261124,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler for train split only.

    Args:
        labels: 1D integer array of class labels in {0, 1, 2, -100} on train split.
        power: 0.0 -> uniform, 0.5 -> square-root oversampling (recommended), 1.0 -> full balance.
        seed: Deterministic generator seed.
    """
    y = np.asarray(labels, dtype=np.int64)
    counts = {
        c: max(int((y == c).sum()), 1) for c in range(NUM_CONSISTENCY_CLASSES)
    }
    sample_weights = np.zeros(len(y), dtype=np.float64)
    for i, cls in enumerate(y):
        if cls in counts:
            sample_weights[i] = 1.0 / (counts[int(cls)] ** power)
        else:
            sample_weights[i] = 0.0

    if sample_weights.sum() <= 0:
        raise ValueError("No valid consistency labels available for balanced sampling")

    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(y),
        replacement=True,
        generator=generator,
    )


class HardNegativeICDataset(Dataset):
    """Train-only wrapper that injects controlled mismatched (image, content) negatives.

    Strictly forbids wrapping validation or test splits to prevent evaluation leakage.
    Swaps the image with another article's image on a small fraction of Consistent samples
    and sets consistency_ic = 1 (Inconsistent), while masking tc and ti as -100 so
    synthetic swaps never corrupt tc/ti targets.
    """

    def __init__(
        self,
        base_dataset,
        *,
        negative_ratio: float = 0.05,
        seed: int = 20261124,
    ) -> None:
        split = getattr(base_dataset, "split", None)
        if split != "train":
            raise ValueError(
                f"HardNegativeICDataset is strictly train-only; refused split={split!r}"
            )
        if not 0.0 <= negative_ratio <= 0.5:
            raise ValueError("negative_ratio must be within [0.0, 0.5]")

        self.base = base_dataset
        self.split = "train"
        self.schema = getattr(base_dataset, "schema", None)
        self.negative_ratio = float(negative_ratio)
        self.seed = int(seed)

        rng = random.Random(seed)
        n = len(base_dataset)
        num_neg = int(round(n * self.negative_ratio))
        self._pairs: list[tuple[int, int]] = []
        if num_neg > 0 and n > 1:
            anchors = rng.sample(range(n), k=min(num_neg, n))
            for anchor_idx in anchors:
                donor_idx = rng.randrange(n - 1)
                if donor_idx >= anchor_idx:
                    donor_idx += 1
                self._pairs.append((anchor_idx, donor_idx))

    def __len__(self) -> int:
        return len(self.base) + len(self._pairs)

    def __getitem__(self, index: int) -> dict:
        if index < len(self.base):
            return self.base[index]

        anchor_idx, donor_idx = self._pairs[index - len(self.base)]
        anchor = self.base[anchor_idx]
        donor = self.base[donor_idx]

        features = dict(anchor["features"])
        features["image"] = donor["features"]["image"]

        consistency_target = np.array([-100, -100, 1], dtype=np.int64)
        consistency_mask = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        targets = dict(anchor["targets"])
        targets["consistency"] = consistency_target
        masks = dict(anchor["masks"])
        masks["consistency"] = consistency_mask

        return {
            "article_id": f"{anchor['article_id']}__hardneg_{donor_idx}",
            "features": features,
            "targets": targets,
            "masks": masks,
            "target_origin": anchor["target_origin"],
        }