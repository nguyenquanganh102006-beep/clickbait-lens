"""Shared dataset interface: fixed split, model-independent text and image transforms."""

import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def file_sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        # A long article can exceed the platform csv default field size.
        csv.field_size_limit(10_000_000)
        return list(csv.DictReader(f))


def verify_release(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["protected_sha256"].items():
        if file_sha256(root / rel) != expected:
            raise ValueError(f"Release checksum mismatch: {rel}")
    return manifest


class ClickbaitDataset:
    """PyTorch DataLoader-compatible map dataset; torch is optional until batching.

    image_root is the directory containing actual image files, NOT its parent.
    Features contain only title/content/image (+ optional lead/caption).
    No tokenizer, random split, fit, resizing or normalization is applied implicitly.
    """

    def __init__(
        self,
        root,
        split="train",
        *,
        image_root=None,
        image_transform=None,
        text_transform=None,
        allow_pseudo_test=False,
        gold_csv=None,
        verify_checksums=True,
    ):
        self.root = Path(root)
        self.split = split
        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown split")
        if verify_checksums:
            verify_release(self.root)
        self.schema = json.loads((self.root / "label_schema.json").read_text(encoding="utf-8"))
        members = read_csv(self.root / f"splits/{split}.csv")
        ids = [r["article_id"] for r in members]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate split membership")
        records = {r["article_id"]: r for r in read_csv(self.root / "data.csv")}
        self.rows = [records[aid] for aid in ids]
        if any(r["split"] != split for r in self.rows):
            raise ValueError("Split mismatch")
        self.target_origin = "pseudo"
        if gold_csv is not None:
            if split != "test":
                raise ValueError("Gold overlay is reserved for test")
            gold_csv = Path(gold_csv)
            receipt = json.loads(gold_csv.with_suffix(".receipt.json").read_text(encoding="utf-8"))
            if not receipt.get("complete") or receipt["gold_sha256"] != file_sha256(gold_csv):
                raise ValueError("Gold release missing or incomplete")
            if receipt["split_assignments_sha256"] != file_sha256(
                self.root / "split_assignments.csv"
            ):
                raise ValueError("Gold belongs to another split")
            gold = read_csv(gold_csv)
            byid = {r["article_id"]: r for r in gold}
            if len(gold) != len(ids) or set(byid) != set(ids):
                raise ValueError("Gold ID coverage mismatch")
            for row in self.rows:
                g = byid[row["article_id"]]
                for col in [
                    "clickbait_label",
                    *self.schema["type_columns"],
                    *self.schema["consistency_columns"],
                ]:
                    row[col] = g[col]
            self.target_origin = "human_gold"
        elif split == "test" and not allow_pseudo_test:
            raise ValueError(
                "Test is not human gold yet. Supply gold_csv, or explicitly allow_pseudo_test=True for diagnostics only."
            )
        self.image_root = Path(image_root) if image_root is not None else self.root / "images"
        self.image_transform = image_transform
        self.text_transform = text_transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        relative = Path(row["image_path"].replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "images":
            raise ValueError("Unsafe image path")
        with Image.open(self.image_root / Path(*relative.parts[1:])) as im:
            image = im.convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)
        features = {
            "title": row["title"],
            "content": row["content"],
            "image": image,
            "lead": row.get("lead", ""),
            "image_caption": row.get("image_caption", ""),
        }
        if self.text_transform is not None:
            transformed = self.text_transform({k: v for k, v in features.items() if k != "image"})
            features = {"text": transformed, "image": image}
        mapping = self.schema["consistency_classes"]
        consistency = np.array(
            [mapping.get(row[c], -100) for c in self.schema["consistency_columns"]], dtype=np.int64
        )
        if any(
            row[c] not in (*mapping, "Not Available") for c in self.schema["consistency_columns"]
        ):
            raise ValueError("Invalid consistency target")
        targets = {
            "detection": np.float32(row["clickbait_label"]),
            "types": np.array([int(row[c]) for c in self.schema["type_columns"]], dtype=np.float32),
            "consistency": consistency,
        }
        masks = {
            "detection": np.float32(1),
            "types": np.ones(5, dtype=np.float32),
            "consistency": (consistency != -100).astype(np.float32),
        }
        return {
            "article_id": row["article_id"],
            "features": features,
            "targets": targets,
            "masks": masks,
            "target_origin": self.target_origin,
        }


def collate_samples(samples, *, as_torch=False):
    """Preserve raw variable-length text/images; stack only numeric targets/masks.

    Supply a deterministic eval image_transform and tokenizer to build tensors.
    PyTorch default_collate cannot batch raw PIL images; use this function.
    """
    if not samples:
        raise ValueError("Empty batch")
    result = {
        "article_id": [s["article_id"] for s in samples],
        "features": [s["features"] for s in samples],
        "target_origin": [s["target_origin"] for s in samples],
    }
    for section in ("targets", "masks"):
        result[section] = {
            k: np.stack([s[section][k] for s in samples]) for k in samples[0][section]
        }
    if as_torch:
        import torch

        for section in ("targets", "masks"):
            result[section] = {k: torch.from_numpy(v) for k, v in result[section].items()}
    return result


def _seed_worker(worker_id):
    import random
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader(dataset, batch_size=16, num_workers=0):
    """Fixed seed and train-only shuffling. Requires separately installed torch."""
    from functools import partial
    import torch

    generator = torch.Generator().manual_seed(dataset.schema["seed"])
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataset.split == "train",
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=generator,
        collate_fn=partial(collate_samples, as_torch=True),
    )
