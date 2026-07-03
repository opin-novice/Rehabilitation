"""SSL heads that attach to an encoder embedding (Layer 1).

All heads consume the encoder's pooled embedding (B, out_dim). They are used
only during pretraining and discarded before fine-tuning.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """SimCLR projection MLP: out_dim -> hidden -> proj_dim."""

    def __init__(self, in_dim: int, hidden: int = 64, proj_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ReconstructionDecoder(nn.Module):
    """Masked-motion decoder: embedding -> flattened (T*J*C) reconstruction."""

    def __init__(self, in_dim: int, seq_len: int, num_joints: int,
                 num_channels: int, hidden: int = 256) -> None:
        super().__init__()
        self.seq_len, self.num_joints, self.num_channels = seq_len, num_joints, num_channels
        out = seq_len * num_joints * num_channels
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        return self.net(z).view(B, self.seq_len, self.num_joints, self.num_channels)
