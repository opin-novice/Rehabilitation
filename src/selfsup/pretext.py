"""Pretext tasks (Layer 2). One ABC, many SSL objectives.

Add a new SSL paradigm by subclassing PretextTask; the generic trainer
(pretrain.py) needs no changes. This is the primary scalability seam.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .augmentations import batch_two_views
from .heads import ProjectionHead, ReconstructionDecoder


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """SimCLR NT-Xent. z1,z2: (B, D); (z1[i], z2[i]) are positive pairs."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                     # (2B, D)
    sim = torch.mm(z, z.T) / temperature              # (2B, 2B)
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float("-inf"))
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels + B, labels])          # positive index for each row
    return F.cross_entropy(sim, labels)


class PretextTask(ABC):
    name: str = "pretext"

    @abstractmethod
    def build_head(self, encoder: nn.Module) -> nn.Module: ...

    @abstractmethod
    def loss(self, encoder: nn.Module, head: nn.Module, xb: torch.Tensor) -> torch.Tensor:
        """xb: (B, T, J, C) unlabeled batch on the correct device."""


class ContrastiveNTXent(PretextTask):
    name = "contrastive"

    def __init__(self, temperature: float = 0.07, proj_dim: int = 32,
                 aug_exclude: List[str] | None = None) -> None:
        self.temperature = temperature
        self.proj_dim = proj_dim
        self.aug_exclude = aug_exclude or []

    def build_head(self, encoder: nn.Module) -> nn.Module:
        return ProjectionHead(encoder.out_dim, hidden=64, proj_dim=self.proj_dim)

    def loss(self, encoder, head, xb):
        v1, v2 = batch_two_views(xb, exclude=self.aug_exclude)
        v1 = v1.to(xb.device); v2 = v2.to(xb.device)
        z1 = head(encoder.forward_features(v1))
        z2 = head(encoder.forward_features(v2))
        return nt_xent_loss(z1, z2, self.temperature)


class MaskedMotion(PretextTask):
    name = "masked"

    def __init__(self, mask_ratio: float = 0.30, mask_type: str = "joint",
                 seq_len: int = 100, num_joints: int = 25, num_channels: int = 3) -> None:
        assert mask_type in {"joint", "temporal"}
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.seq_len, self.num_joints, self.num_channels = seq_len, num_joints, num_channels

    def build_head(self, encoder: nn.Module) -> nn.Module:
        return ReconstructionDecoder(
            encoder.out_dim, self.seq_len, self.num_joints, self.num_channels)

    def _mask(self, xb: torch.Tensor):
        B, T, J, C = xb.shape
        x_masked = xb.clone()
        mask = torch.zeros(B, T, J, C, dtype=torch.bool, device=xb.device)
        if self.mask_type == "joint":
            n = max(1, int(J * self.mask_ratio))
            for b in range(B):
                idx = torch.randperm(J, device=xb.device)[:n]
                x_masked[b, :, idx, :] = 0.0
                mask[b, :, idx, :] = True
        else:  # temporal
            n = max(1, int(T * self.mask_ratio))
            for b in range(B):
                idx = torch.randperm(T, device=xb.device)[:n]
                x_masked[b, idx, :, :] = 0.0
                mask[b, idx, :, :] = True
        return x_masked, mask

    def loss(self, encoder, head, xb):
        x_masked, mask = self._mask(xb)
        pred = head(encoder.forward_features(x_masked))
        return F.mse_loss(pred[mask], xb[mask])


def build_pretext(kind: str, **kw) -> PretextTask:
    kind = kind.lower()
    if kind in {"contrastive", "simclr", "ntxent"}:
        return ContrastiveNTXent(**kw)
    if kind in {"masked", "mae", "masked_motion"}:
        return MaskedMotion(**kw)
    raise ValueError(f"Unknown pretext task: {kind}")
