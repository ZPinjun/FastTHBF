"""Shared paths, configuration, and CPU artifact loading."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def configured_path(variable, default):
    path = Path(os.environ.get(variable, default)).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


DATA_DIR = configured_path("FTHBF_DATA_DIR", "data_Probs")
CHECKPOINT_DIR = configured_path("FTHBF_CHECKPOINT_DIR", "checkpoints")
SCENARIO_DIR = configured_path("FTHBF_SCENARIO_DIR", "deepmimo_scenarios")
RESULT_ROOT = configured_path("FTHBF_RESULT_DIR", "results")


def read_config(name):
    path = Path(name)
    if not path.is_absolute():
        path = ROOT / "configs" / path
    return json.loads(path.read_text())


def result_dir(name):
    path = RESULT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def checkpoint_path(layers):
    return CHECKPOINT_DIR / f"unfolded_wmmse_model_{layers}Iter.pt"


def load_artifact(path, *, map_location="cpu"):
    """Load tensor-based release artifacts without arbitrary pickle objects."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.name} in {path.parent}. See docs/DATA.md or "
            "checkpoints/README.md for preparation and installation instructions."
        )
    return torch.load(path, map_location=map_location, weights_only=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment():
    versions = {}
    for name in ("torch", "numpy", "scipy", "matplotlib", "deepmimo", "pymanopt", "autograd"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "torch_threads": torch.get_num_threads(),
            "packages": versions}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def record_run(name, config):
    write_json(Path(result_dir(name)) / "run.json", {
        "config": config, "environment": environment(), "device": "cpu"
    })
