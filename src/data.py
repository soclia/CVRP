"""Validated tensor datasets with content identities independent of filenames."""
import hashlib
import json
from pathlib import Path

import torch

KEYS = ("depot_xy", "node_xy", "node_demand")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_id(data):
    digest = hashlib.sha256()
    for key in KEYS:
        tensor = data[key].detach().cpu().float().contiguous()
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_data(data, problem_size=None):
    if not isinstance(data, dict) or any(k not in data for k in KEYS):
        raise ValueError("Dataset must contain depot_xy, node_xy, node_demand tensors")
    if any(not isinstance(data[k], torch.Tensor) for k in KEYS):
        raise ValueError("Dataset values must be tensors")
    depot, nodes, demand = (data[k] for k in KEYS)
    if nodes.ndim != 3 or nodes.shape[-1] != 2:
        raise ValueError("node_xy shape must be (instances, customers, 2)")
    b, n, _ = nodes.shape
    if b < 1 or n < 1 or depot.shape != (b, 1, 2) or demand.shape != (b, n):
        raise ValueError("Dataset dimensions are inconsistent or empty")
    if problem_size is not None and n != problem_size:
        raise ValueError(f"Dataset has {n} customers; expected {problem_size}")
    for key in KEYS:
        if not data[key].is_floating_point() or not torch.isfinite(data[key]).all():
            raise ValueError(f"{key} must contain finite floating point values")
    if (depot < 0).any() or (depot > 1).any() or (nodes < 0).any() or (nodes > 1).any():
        raise ValueError("Coordinates must be normalized to [0, 1]")
    if (demand <= 0).any() or (demand > 1).any():
        raise ValueError("Normalized customer demand must be in (0, 1]")
    return b, n


def load_dataset(path, problem_size=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Fixed dataset not found: {path}; no random fallback is performed")
    data = torch.load(path, map_location="cpu", weights_only=True)
    validate_data(data, problem_size)
    return data


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_reference(path, data):
    reference = json.loads(Path(path).read_text(encoding="utf-8"))
    if reference.get("dataset_id") != dataset_id(data):
        raise ValueError("Reference cost vector is not bound to this exact dataset")
    costs = torch.tensor(reference["costs"], dtype=torch.float64)
    if costs.shape != (data["node_xy"].shape[0],) or not torch.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError("Reference must contain one positive finite cost per instance")
    if not reference.get("source"):
        raise ValueError("Reference must identify its solver/source and budget")
    return costs, reference["source"]
