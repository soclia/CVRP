"""Explicit, strict, safe-by-default checkpoint loading."""
from pathlib import Path

import torch

from . import model_params, variant_name
from .CVRPModel import CVRPModel, _with_hacd_defaults
from .data import sha256_file


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_checkpoint(path, trusted_legacy=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=not trusted_legacy)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a checkpoint dictionary containing model_state_dict")
    return checkpoint


def checkpoint_params(checkpoint):
    saved = checkpoint.get("model_params")
    if not isinstance(saved, dict):
        raise ValueError("Checkpoint requires explicit model_params from the actual training configuration; tensor shapes cannot determine attention heads")
    required = {"embedding_dim", "encoder_layer_num", "head_num", "qkv_dim", "ff_hidden_dim", "logit_clipping"}
    missing = required - saved.keys()
    if missing:
        raise ValueError(f"Checkpoint model_params missing required architecture fields: {sorted(missing)}")
    if saved:
        # Use neutral POMO defaults before applying saved flags, never full-HACD flags.
        p = model_params("pomo")
        for key in ("hacd_hilo_enabled", "hacd_anchor_enabled", "hacd_capacity_residual_enabled"):
            p.pop(key, None)
        p.update(saved)
        p["sqrt_embedding_dim"] = saved.get("sqrt_embedding_dim", p["embedding_dim"] ** 0.5)
        p = _with_hacd_defaults(p)
        if "hacd_anchor_method" not in saved and p["hacd_anchor_enabled"]:
            raise ValueError("Anchor checkpoint requires explicit hacd_anchor_method from the actual training configuration")
    p["eval_type"] = "argmax"
    return p


def initialize_model(model, path):
    """Transfer matching tensors into a new run, without inferring a source policy."""
    checkpoint = read_checkpoint(path)
    saved = checkpoint.get("model_params")
    if not isinstance(saved, dict):
        raise ValueError("Initialization requires source model_params")
    for key in ("embedding_dim", "encoder_layer_num", "head_num", "qkv_dim", "ff_hidden_dim"):
        if saved.get(key) != model.model_params[key]:
            raise ValueError(f"Initialization architecture differs: {key}")
    state = checkpoint["model_state_dict"]
    target = model.state_dict()
    if state.keys() != target.keys() or any(state[k].shape != target[k].shape for k in target):
        raise ValueError("Initialization requires the same parameterized modules; partial POMO-to-HACD loading is unsupported")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Initialization contains nonfinite tensors")
    model.load_state_dict(state, strict=True)
    return {"source": str(Path(path).resolve()), "source_sha256": sha256_file(path),
            "source_epoch": checkpoint.get("epoch"), "source_model_params": saved,
            "source_metadata": checkpoint.get("metadata", {}),
            "mode": "parameter initialization with target configuration; new optimizer and random streams"}


def load_model(path, device, expected_variant=None, trusted_legacy=False):
    checkpoint = read_checkpoint(path, trusted_legacy)
    params = checkpoint_params(checkpoint)
    variant = variant_name(params)
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(f"Checkpoint variant is {variant}, expected {expected_variant}")
    model = CVRPModel(**params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    metadata = dict(checkpoint.get("metadata", {}))
    metadata.update(variant=variant, epoch=checkpoint.get("epoch"),
                    parameters=sum(p.numel() for p in model.parameters()), model_params=params,
                    training_seed=checkpoint.get("training", {}).get("seed", metadata.get("training_seed")),
                    training_problem_size=checkpoint.get("training", {}).get("problem_size", metadata.get("training_problem_size")))
    return model, checkpoint, metadata
