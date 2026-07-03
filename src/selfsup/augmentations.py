"""Skeleton augmentations for contrastive pretraining (Phase 1).

Each augmentation maps a tensor (T, J, C) -> (T, J, C). The AUG_REGISTRY encodes
the *clinical-validity* taxonomy: whether an augmentation is safe during
pretraining vs. supervised fine-tuning. Speed/limb-scale are pretrain-only
because movement duration and limb geometry carry clinical scoring signal.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F


def resample_to_fixed_length(x: torch.Tensor, length: int) -> torch.Tensor:
    """Linear-resample (T, J, C) along time to (length, J, C)."""
    T, J, C = x.shape
    if T == length:
        return x
    # (1, J*C, T) for interpolate over the last (time) axis
    xt = x.reshape(T, J * C).permute(1, 0).unsqueeze(0)     # (1, J*C, T)
    xt = F.interpolate(xt, size=length, mode="linear", align_corners=False)
    return xt.squeeze(0).permute(1, 0).reshape(length, J, C)


def temporal_crop(x: torch.Tensor, min_ratio: float = 0.80) -> torch.Tensor:
    T = x.shape[0]
    crop_len = max(2, int(T * random.uniform(min_ratio, 1.0)))
    start = random.randint(0, T - crop_len)
    return resample_to_fixed_length(x[start:start + crop_len], T)


def joint_masking(x: torch.Tensor, n_joints: int = 4) -> torch.Tensor:
    J = x.shape[1]
    idx = random.sample(range(J), min(n_joints, J))
    out = x.clone()
    out[:, idx, :] = 0.0
    return out


def speed_perturbation(x: torch.Tensor, min_rate: float = 0.8, max_rate: float = 1.2) -> torch.Tensor:
    T = x.shape[0]
    rate = random.uniform(min_rate, max_rate)
    new_len = max(2, int(T * rate))
    stretched = resample_to_fixed_length(x, new_len)
    return resample_to_fixed_length(stretched, T)


def gaussian_joint_noise(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    return x + torch.randn_like(x) * sigma


def _rotation_matrix_y(angle_rad: float, device, dtype) -> torch.Tensor:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
                        device=device, dtype=dtype)


def rotation_3d(x: torch.Tensor, max_angle_deg: float = 15.0) -> torch.Tensor:
    angle = math.radians(random.uniform(-max_angle_deg, max_angle_deg))
    R = _rotation_matrix_y(angle, x.device, x.dtype)
    return x @ R.T


def limb_scaling(x: torch.Tensor, min_scale: float = 0.9, max_scale: float = 1.1) -> torch.Tensor:
    centroid = x.mean(dim=(0, 1), keepdim=True)
    scale = random.uniform(min_scale, max_scale)
    return centroid + (x - centroid) * scale


# name -> (fn, pretrain_ok, finetune_ok)
AUG_REGISTRY: Dict[str, Tuple[Callable, bool, bool]] = {
    "temporal_crop":  (temporal_crop,       True, True),
    "joint_mask":     (joint_masking,       True, True),
    "gaussian_noise": (gaussian_joint_noise, True, True),
    "rotation_y":     (rotation_3d,         True, True),
    "speed_perturb":  (speed_perturbation,  True, False),   # duration is clinical
    "limb_scale":     (limb_scaling,        True, False),   # geometry normalized upstream
}

PRETRAIN_AUGS: List[str] = [k for k, v in AUG_REGISTRY.items() if v[1]]
FINETUNE_AUGS: List[str] = [k for k, v in AUG_REGISTRY.items() if v[2]]


def apply_augs(x: torch.Tensor, names: List[str]) -> torch.Tensor:
    for n in names:
        x = AUG_REGISTRY[n][0](x)
    return x


def two_views(
    x: torch.Tensor,
    pool: List[str] | None = None,
    k: int = 2,
    exclude: List[str] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return two independently augmented views of a single sequence (T, J, C).

    `exclude` drops augmentations for ablation runs (leave-one-out).
    """
    pool = list(pool or PRETRAIN_AUGS)
    if exclude:
        pool = [p for p in pool if p not in exclude]
    a1 = random.sample(pool, min(k, len(pool)))
    a2 = random.sample(pool, min(k, len(pool)))
    return apply_augs(x.clone(), a1), apply_augs(x.clone(), a2)


def batch_two_views(xb: torch.Tensor, **kw) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized-ish two-view generation for a batch (B, T, J, C)."""
    v1, v2 = [], []
    for i in range(xb.shape[0]):
        a, b = two_views(xb[i], **kw)
        v1.append(a); v2.append(b)
    return torch.stack(v1), torch.stack(v2)
