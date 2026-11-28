from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

IC_RELATION_INDEX = 2
NUM_CLASSES = 3


def truncate_content(content: str, max_words: int = 180) -> str:
    """Keep the opening max_words tokens for C3.1 truncated baseline."""
    words = content.split()
    return " ".join(words[:max_words])


def extractive_summarize_content(
    title: str,
    content: str,
    lead: str = "",
    *,
    max_sentences: int = 5,
    max_words: int = 200,
) -> str:
    """Deterministic extractive summary for C3.2 without external API calls.

    Scores sentences by lexical overlap with title + lead plus positional prior
    (opening sentences carry strong journalistic summary signal).
    """
    cleaned = re.sub(r"\s+", " ", content).strip()
    if not cleaned:
        return lead.strip() or title.strip()

    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", cleaned) if s.strip()]
    if len(sentences) <= max_sentences:
        return truncate_content(" ".join(sentences), max_words=max_words)

    anchor_tokens = set(re.findall(r"\w+", f"{title} {lead}".lower()))
    scored: list[tuple[float, int, str]] = []
    for idx, sent in enumerate(sentences):
        tokens = set(re.findall(r"\w+", sent.lower()))
        overlap = len(tokens & anchor_tokens) / max(math.sqrt(len(tokens) + 1), 1.0)
        pos_bonus = 0.35 / (idx + 1)
        scored.append((overlap + pos_bonus, idx, sent))

    top = sorted(scored, key=lambda item: item[0], reverse=True)[:max_sentences]
    ordered = [sent for _, _, sent in sorted(top, key=lambda item: item[1])]
    prefix = f"{lead.strip()} " if lead.strip() else ""
    return truncate_content(f"{prefix}{' '.join(ordered)}".strip(), max_words=max_words)


def chunk_content(
    content: str,
    *,
    chunk_words: int = 120,
    overlap_words: int = 30,
    max_chunks: int = 8,
) -> list[str]:
    """Split long article content (median 658, max 7,007 words) into overlapping passages."""
    words = content.split()
    if not words:
        return [""]
    if len(words) <= chunk_words:
        return [" ".join(words)]

    stride = max(chunk_words - overlap_words, 1)
    chunks: list[str] = []
    for start in range(0, len(words), stride):
        chunk = " ".join(words[start : start + chunk_words])
        chunks.append(chunk)
        if start + chunk_words >= len(words) or len(chunks) >= max_chunks:
            break
    return chunks


def _pack_ic_logits(ic_logits: torch.Tensor) -> dict[str, torch.Tensor]:
    """Embed [B, 3] IC logits into contract-compliant [B, 3, 3] consistency tensor."""
    if ic_logits.ndim != 2 or ic_logits.shape[1] != NUM_CLASSES:
        raise ValueError(f"Expected ic_logits [B, 3], got {tuple(ic_logits.shape)}")
    full = ic_logits.new_zeros((ic_logits.shape[0], 3, NUM_CLASSES))
    full[:, IC_RELATION_INDEX, :] = ic_logits
    return {"consistency": full}


class OfflineHashEncoder:
    """Fast deterministic offline encoder for smoke tests and unit tests (no downloads)."""

    def __init__(self, embed_dim: int = 64) -> None:
        self.embed_dim = int(embed_dim)

    def encode_images(self, images: Sequence, batch_size: int = 32) -> torch.Tensor:
        del batch_size
        vectors = []
        for img in images:
            arr = np.asarray(img.resize((16, 16)), dtype=np.float32) / 255.0
            flat = arr.reshape(-1)
            idx = np.linspace(0, len(flat) - 1, self.embed_dim).astype(int)
            vec = flat[idx]
            norm = np.linalg.norm(vec) + 1e-8
            vectors.append(vec / norm)
        return torch.tensor(np.stack(vectors), dtype=torch.float32)

    def encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        del batch_size
        vectors = []
        for text in texts:
            vec = np.zeros(self.embed_dim, dtype=np.float32)
            for token in re.findall(r"\w+", text.lower()):
                digest = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)
                vec[digest % self.embed_dim] += 1.0
            norm = np.linalg.norm(vec) + 1e-8
            vectors.append(vec / norm)
        return torch.tensor(np.stack(vectors), dtype=torch.float32)


class HuggingFaceVLMEncoder:
    """Batched FP16 wrapper for SigLIP 2 or Multilingual CLIP vision-language encoders.

    Requires `transformers` installed in the user's training environment.
    Pin exact `model_id` and `revision` in configs/consistency_ic.yaml.
    """

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        device: str = "cpu",
        max_length: int = 64,
    ) -> None:
        from transformers import AutoModel, AutoProcessor

        self.device = torch.device(device)
        self.use_amp = self.device.type == "cuda"
        self.max_length = int(max_length)
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModel.from_pretrained(model_id, revision=revision).to(self.device)
        self.model.eval()
        cfg = self.model.config
        if hasattr(cfg, "projection_dim"):
            self.embed_dim = int(cfg.projection_dim)
        elif hasattr(cfg, "vision_config") and hasattr(cfg.vision_config, "hidden_size"):
            self.embed_dim = int(cfg.vision_config.hidden_size)
        else:
            raise ValueError(f"Cannot infer embedding dim for {model_id}")

    @staticmethod
    def _to_tensor(feats) -> torch.Tensor:
        if isinstance(feats, torch.Tensor):
            return feats
        for attr in ("pooler_output", "image_embeds", "text_embeds"):
            val = getattr(feats, attr, None)
            if isinstance(val, torch.Tensor):
                return val
        if hasattr(feats, "last_hidden_state"):
            return feats.last_hidden_state[:, 0]
        raise TypeError(f"Unsupported feature output type: {type(feats)}")

    @torch.no_grad()
    def encode_images(self, images: Sequence, batch_size: int = 32) -> torch.Tensor:
        chunks = []
        for start in range(0, len(images), batch_size):
            batch = list(images[start : start + batch_size])
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                feats = self._to_tensor(self.model.get_image_features(**inputs))
            chunks.append(F.normalize(feats.float(), dim=-1).cpu())
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        chunks = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            inputs = self.processor(
                text=batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
                feats = self._to_tensor(self.model.get_text_features(**inputs))
            chunks.append(F.normalize(feats.float(), dim=-1).cpu())
        return torch.cat(chunks, dim=0)


class GlobalICModel(nn.Module):
    """C3.1 (Truncated) & C3.2 (Summarized) global image-text consistency classifier.

    Inputs:
        image_emb: [B, D] L2-normalized image embedding.
        text_emb:  [B, D] L2-normalized content embedding.
    Output:
        {"consistency": logits [B, 3, 3]}
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        in_dim = embed_dim * 4 + 1
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_CLASSES),
        )

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> dict[str, torch.Tensor]:
        v = F.normalize(image_emb, dim=-1)
        t = F.normalize(text_emb, dim=-1)
        cos = (v * t).sum(dim=-1, keepdim=True)
        z = torch.cat([v, t, torch.abs(v - t), v * t, cos], dim=-1)
        return _pack_ic_logits(self.mlp(z))


class PassageICModel(nn.Module):
    """C3.3 / C3.4 / C3.5 passage-level image-content consistency classifier.

    Supports pooling/aggregation strategies required by Plan train.docx Section 3.4:
      - "similarity" (C3.3): summary stats [max, mean, min, std, top2_mean] of image-chunk cosine.
      - "topk_mlp"   (C3.4): Top-K chunk evidence mean-pooled + interaction features -> MLP.
      - "max" / "mean" / "topk" / "attention" (C3.5): evidence aggregation over chunks.

    Inputs:
        image_emb:  [B, D] image embedding.
        chunk_embs: [B, M, D] chunk embeddings (padded to M).
        chunk_mask: [B, M] float/bool mask (1=valid chunk, 0=padding).
    Output:
        {"consistency": logits [B, 3, 3]}
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        aggregation: str = "attention",
        top_k: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        valid_aggs = ("similarity", "topk_mlp", "max", "mean", "topk", "attention")
        if aggregation not in valid_aggs:
            raise ValueError(f"Unsupported aggregation={aggregation!r}; choose from {valid_aggs}")
        self.aggregation = aggregation
        self.top_k = int(top_k)

        if aggregation == "similarity":
            self.classifier = nn.Sequential(
                nn.Linear(5, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, NUM_CLASSES),
            )
        else:
            if aggregation == "attention":
                self.attn_proj = nn.Sequential(
                    nn.Linear(embed_dim * 3 + 1, hidden_dim // 2),
                    nn.Tanh(),
                    nn.Linear(hidden_dim // 2, 1),
                )
            in_dim = embed_dim * 4 + 3
            self.classifier = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, NUM_CLASSES),
            )

    def _similarity_stats(self, sims: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute [max, mean, min, std, top2_mean] similarity summary [B, 5]."""
        valid = mask.bool()
        neg_inf = sims.masked_fill(~valid, -1e4)
        pos_inf = sims.masked_fill(~valid, 1e4)

        sim_max = neg_inf.max(dim=1).values
        sim_min = pos_inf.min(dim=1).values
        denom = valid.sum(dim=1).clamp_min(1).to(sims.dtype)
        sim_mean = (sims * valid.to(sims.dtype)).sum(dim=1) / denom

        diff = (sims - sim_mean.unsqueeze(1)) * valid.to(sims.dtype)
        sim_std = torch.sqrt((diff.pow(2).sum(dim=1) / denom).clamp_min(1e-8))

        k2 = min(2, sims.shape[1])
        top2 = neg_inf.topk(k=k2, dim=1).values
        top2_valid = (top2 > -1e3).to(sims.dtype)
        top2_mean = (top2 * top2_valid).sum(dim=1) / top2_valid.sum(dim=1).clamp_min(1.0)
        return torch.stack([sim_max, sim_mean, sim_min, sim_std, top2_mean], dim=-1)

    def forward(
        self,
        image_emb: torch.Tensor,
        chunk_embs: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        v = F.normalize(image_emb, dim=-1)
        c = F.normalize(chunk_embs, dim=-1)
        mask_f = chunk_mask.to(v.dtype)
        sims = torch.einsum("bd,bmd->bm", v, c)

        if self.aggregation == "similarity":
            stats = self._similarity_stats(sims, chunk_mask)
            return _pack_ic_logits(self.classifier(stats))

        neg_inf = sims.masked_fill(~chunk_mask.bool(), -1e4)
        if self.aggregation == "max":
            best_idx = neg_inf.argmax(dim=1)
            agg = c[torch.arange(c.shape[0], device=c.device), best_idx]
        elif self.aggregation == "mean":
            agg = (c * mask_f.unsqueeze(-1)).sum(dim=1) / mask_f.sum(dim=1, keepdim=True).clamp_min(
                1.0
            )
        elif self.aggregation in ("topk", "topk_mlp"):
            k = min(self.top_k, c.shape[1])
            top_vals, top_idx = neg_inf.topk(k=k, dim=1)
            gathered = torch.gather(c, 1, top_idx.unsqueeze(-1).expand(-1, -1, c.shape[-1]))
            top_valid = (top_vals > -1e3).to(v.dtype).unsqueeze(-1)
            agg = (gathered * top_valid).sum(dim=1) / top_valid.sum(dim=1).clamp_min(1.0)
        else:
            v_exp = v.unsqueeze(1).expand_as(c)
            attn_in = torch.cat([v_exp, c, v_exp * c, sims.unsqueeze(-1)], dim=-1)
            scores = self.attn_proj(attn_in).squeeze(-1)
            scores = scores.masked_fill(~chunk_mask.bool(), -1e4)
            weights = F.softmax(scores, dim=-1).unsqueeze(-1)
            agg = (c * weights).sum(dim=1)

        agg = F.normalize(agg, dim=-1)
        stats = self._similarity_stats(sims, chunk_mask)
        z = torch.cat([v, agg, torch.abs(v - agg), v * agg, stats[:, :3]], dim=-1)
        return _pack_ic_logits(self.classifier(z))


class CaptionNLIModel(nn.Module):
    """C3.6: Image Captioning + Textual NLI Verification model.

    Maps NLI 3-class logits (entailment, contradiction, neutral) from a pretrained
    multilingual NLI backbone (or Task 1's fine-tuned NLI model) into the repo's
    consistency classes (0=Consistent, 1=Inconsistent, 2=Insufficient Evidence)
    via an optional learnable calibration layer.
    """

    def __init__(
        self,
        *,
        entailment_idx: int = 0,
        contradiction_idx: int = 2,
        neutral_idx: int = 1,
        calibrate: bool = True,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "perm",
            torch.tensor([entailment_idx, contradiction_idx, neutral_idx], dtype=torch.long),
        )
        self.calibration = nn.Linear(NUM_CLASSES, NUM_CLASSES) if calibrate else nn.Identity()
        if calibrate:
            nn.init.eye_(self.calibration.weight)
            nn.init.zeros_(self.calibration.bias)

    def forward(self, nli_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert NLI logits [B, 3] into contract-compliant [B, 3, 3] consistency logits."""
        if nli_logits.ndim != 2 or nli_logits.shape[1] != NUM_CLASSES:
            raise ValueError(f"Expected nli_logits [B, 3], got {tuple(nli_logits.shape)}")
        reordered = nli_logits[:, self.perm]
        calibrated = self.calibration(reordered)
        return _pack_ic_logits(calibrated)