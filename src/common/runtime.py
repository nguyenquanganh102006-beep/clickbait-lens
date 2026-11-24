import hashlib
import importlib.metadata
import json
import os
import random
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
import yaml


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    from src.common.contracts import active_tasks

    active_tasks(config["task"])
    if config["seed"] != 20261124:
        raise ValueError("Shared dataset seed is fixed at 20261124")
    if config["data"]["dataset_version"] != "1.0":
        raise ValueError("Unsupported dataset version")
    if config["run_mode"] != "smoke":
        raise ValueError("This runner is an integration smoke test, not a production trainer")
    if config["training"]["batch_size"] < 1 or config["training"]["steps"] < 1:
        raise ValueError("Invalid training size")
    if not re.fullmatch("[a-z0-9-]+", config["model"]["name"]):
        raise ValueError("Invalid model name")
    return config


def new_run(config, repo_root):
    root = Path(repo_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{config['task']}__{config['model']['name']}__v1-0__s{config['seed']}__{stamp}__{canonical_hash(config)[:8]}"
    path = root / "experiments" / run_id
    path.mkdir(parents=True, exist_ok=False)
    for folder in ["checkpoints", "results"]:
        (root / folder / run_id).mkdir(parents=True, exist_ok=False)
    (path / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
        dirty = True
    versions = {}
    for name in ["torch", "numpy", "Pillow", "PyYAML"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    metadata = {
        "run_id": run_id,
        "format_version": "1.0",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "config_sha256": canonical_hash(config),
        "versions": versions,
        "seed": config["seed"],
        "run_mode": config["run_mode"],
    }
    return run_id, path, metadata


def log_event(path, *, event, step=0, **fields):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "step": int(step),
        **fields,
    }
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def save_checkpoint(path, model, optimizer, *, metadata, step, config):
    required = {
        "run_id",
        "dataset_version",
        "dataset_manifest_sha256",
        "split_assignments_sha256",
        "label_schema_sha256",
        "git_commit",
    }
    if not required <= metadata.keys():
        raise ValueError("Checkpoint lacks provenance")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": "1.0",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "step": int(step),
        "config": config,
        "metadata": metadata,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng_state": random.getstate(),
        "numpy_rng_state": (lambda s: [s[0], s[1].tolist(), int(s[2]), int(s[3]), float(s[4])])(
            np.random.get_state()
        ),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path, model, *, expected_split_sha256, expected_schema_sha256, optimizer=None, restore_rng=False
):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != "1.0":
        raise ValueError("Unsupported checkpoint")
    meta = state["metadata"]
    if (
        meta["split_assignments_sha256"] != expected_split_sha256
        or meta["label_schema_sha256"] != expected_schema_sha256
    ):
        raise ValueError("Checkpoint dataset split/schema mismatch")
    model.load_state_dict(state["model_state_dict"], strict=True)
    if optimizer is not None and state["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if restore_rng:
        torch.set_rng_state(state["torch_rng_state"])
        random.setstate(state["python_rng_state"])
        n = state["numpy_rng_state"]
        np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
        if torch.cuda.is_available() and state["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(state["cuda_rng_states"])
    return state
