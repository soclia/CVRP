"""One source of truth for architecture, ablations and runtime settings."""
import json
import math
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "precision": "fp32", "tf32": False,
    "model": {
        "embedding_dim": 128, "encoder_layer_num": 6, "head_num": 8,
        "qkv_dim": 16, "ff_hidden_dim": 512, "logit_clipping": 10,
        "hacd_local_k": 16, "hacd_anchor_demand_ratio": 0.2,
        "hacd_anchor_peripheral_ratio": 0.2, "hacd_anchor_kmeans_iters": 3,
        "hacd_anchor_method": "kmeans", "hacd_anchor_temperature": 0.1,
        "hacd_branch_hidden_dim": 64, "hacd_residual_beta": 2.0,
        "hacd_depot_beta": 2.0, "hacd_fusion_init_bias": -2.0,
    },
    "optimizer": {"lr": 1e-4, "weight_decay": 1e-6},
    "training": {
        "20": {"epochs": 1000, "batch_size": 32, "micro_batch_size": 16,
               "milestones": [800, 900], "decoder_checkpoint": False},
        "50": {"epochs": 1000, "batch_size": 32, "micro_batch_size": 8,
               "milestones": [800, 900], "decoder_checkpoint": True},
        "100": {"epochs": 8100, "batch_size": 64, "micro_batch_size": 4,
                "milestones": [8000, 8050], "decoder_checkpoint": True},
        "episodes_per_epoch": 10000, "validation_size": 1000,
        "validation_aug": 8, "seeds": [1234, 1235, 1236],
    },
    "evaluation": {"batch_size": 32, "augmentation": 8,
                   "repeats": 3, "warmup_batches": 3},
}

VARIANTS = {
    "pomo": {"hacd_enabled": False},
    "hilo-anchor": {"hacd_enabled": True, "hacd_capacity_residual_enabled": False},
    "residual-only": {"hacd_enabled": True, "hacd_hilo_enabled": False, "hacd_anchor_enabled": False},
    "no-anchors": {"hacd_enabled": True, "hacd_anchor_enabled": False},
    "no-capacity-risk": {"hacd_enabled": True, "hacd_without_capacity_feature": True},
    "no-depot-residual": {"hacd_enabled": True, "hacd_without_depot_residual": True},
    "no-fusion-gate": {"hacd_enabled": True, "hacd_without_fusion_gate": True},
    "param-matched": {"hacd_enabled": False, "ff_hidden_dim": 976},
    "full": {"hacd_enabled": True},
}


def load_profile(path=None):
    if path is None:
        return deepcopy(DEFAULTS)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def model_params(variant="full", profile=None):
    p = dict((profile or load_profile())["model"])
    # Named variants own structural switches; profiles supply numeric settings.
    for flag in ("hacd_enabled", "hacd_hilo_enabled", "hacd_anchor_enabled",
                 "hacd_capacity_residual_enabled", "hacd_without_capacity_feature",
                 "hacd_without_depot_residual", "hacd_without_fusion_gate",
                 "hacd_depot_residual_enabled", "hacd_gate_enabled",
                 "hacd_capacity_feature_enabled"):
        p.pop(flag, None)
    p.update(VARIANTS[variant])
    for flag in ("hacd_hilo_enabled", "hacd_anchor_enabled", "hacd_capacity_residual_enabled"):
        p.setdefault(flag, p["hacd_enabled"])
    for flag in ("hacd_without_capacity_feature", "hacd_without_depot_residual", "hacd_without_fusion_gate"):
        p.setdefault(flag, False)
    p.update(sqrt_embedding_dim=math.sqrt(p["embedding_dim"]), eval_type="argmax")
    # None selects K=4 for <=50 customers, otherwise K=8, also at zero-shot inference.
    p.setdefault("hacd_cluster_anchor_num", None)
    return p


def variant_name(p):
    for name in VARIANTS:
        reference = model_params(name)
        keys = ("hacd_hilo_enabled", "hacd_anchor_enabled", "hacd_capacity_residual_enabled",
                "hacd_without_capacity_feature", "hacd_without_depot_residual",
                "hacd_without_fusion_gate", "ff_hidden_dim")
        if all(p.get(k, reference[k]) == reference[k] for k in keys):
            return name
    return "custom"


def setup_runtime(device="auto", seed=1234, tf32=False):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    result = torch.device(device)
    if result.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA PyTorch is unavailable. Install a CUDA build or use --device cpu for a smoke test.")
        torch.cuda.set_device(result)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    return result


def hardware_info(device):
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return {"device": str(device), "gpu": gpu, "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda, "precision": "fp32",
            "tf32": torch.backends.cuda.matmul.allow_tf32,
            "target_gpu_verified": bool(gpu and "4090" in gpu)}


# HACD for CVRP package
from .CVRPModel import CVRPModel
from .CVRPEnv import CVRPEnv
from .CVRProblemDef import get_random_problems, augment_xy_data_by_8_fold
from .CVRPTrainer import CVRPTrainer
