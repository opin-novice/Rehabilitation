"""Config + provenance (Layer 4). Reproducibility is the primary Q1 lever."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class PretrainCfg:
    model_type: str = "tcn"
    pretext: str = "contrastive"          # contrastive | masked
    pool: str = "irds_only"               # irds_only | all_corpora (scale ablation)
    d_model: int = 128
    dropout: float = 0.3
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 300
    temperature: float = 0.07             # contrastive only
    proj_dim: int = 32                    # contrastive only
    mask_ratio: float = 0.30              # masked only
    mask_type: str = "joint"              # masked only: joint | temporal
    aug_exclude: List[str] = field(default_factory=list)
    seq_len: int = 100
    num_joints: int = 25
    num_channels: int = 3
    seed: int = 145


@dataclass
class FinetuneCfg:
    model_type: str = "tcn"
    condition: str = "scratch"            # scratch|contrastive_lp|contrastive_ft|masked_lp|masked_ft
    init_ckpt: str = ""
    freeze_encoder: bool = False
    seed: int = 145


def config_hash(cfg) -> str:
    """Deterministic short hash of a dataclass config."""
    payload = json.dumps(asdict(cfg), sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()[:12]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def provenance(cfg, extra: Optional[dict] = None) -> dict:
    """Sidecar written next to every artifact for reproducibility."""
    prov = {
        "git_sha": git_sha(),
        "config_hash": config_hash(cfg),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": asdict(cfg),
    }
    if extra:
        prov.update(extra)
    return prov
