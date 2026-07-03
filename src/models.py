"""
Rehabilitation exercise quality assessment models.

Architecture hierarchy (increasing complexity):
  E0_BaselineMLP              — flatten all joints+time, MLP regression.
  RehabTransformerRegressor   — dual Transformer: JointMLP → SpatialAttn → TemporalAttn.
  GraphAwareTransformerRegressor — same dual Transformer but spatial attention uses a
                                   learnable bone-topology bias (ALiBi-style structure).

Shape contract throughout: [B, T, J, C]
  B = batch size
  T = seq_len (default 100)
  J = num_joints (25)
  C = num_channels (3)
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from rehab_dataset import NUM_CHANNELS, NUM_JOINTS
from constants import SKELETON_EDGES as _SKELETON_EDGES


def _graph_distance_matrix(num_joints: int = NUM_JOINTS) -> np.ndarray:
    """BFS shortest-path distance between every joint pair on the skeleton graph."""
    import collections
    adj: dict[int, list[int]] = {i: [] for i in range(num_joints)}
    for i, j in _SKELETON_EDGES:
        adj[i].append(j)
        adj[j].append(i)
    dist = np.full((num_joints, num_joints), num_joints, dtype=np.float32)
    for src in range(num_joints):
        dist[src, src] = 0
        q: collections.deque[int] = collections.deque([src])
        visited = {src}
        while q:
            node = q.popleft()
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    dist[src, nb] = dist[src, node] + 1
                    q.append(nb)
    return dist


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class SinCosPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding. Adds to sequence dimension."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        assert x.ndim == 3, f"PositionalEncoding expects [B, L, D], got {x.shape}"
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# E0 Baseline — no temporal modeling, validates labels are learnable
# ---------------------------------------------------------------------------

class E0_BaselineMLP(nn.Module):
    """Flatten [B, T, J, C] → mean over T → MLP → scalar.

    This is the simplest possible pipeline. Run this first to confirm the
    data and label signal are correct before adding Transformer complexity.
    """

    def __init__(
        self,
        seq_len: int = 100,
        num_joints: int = NUM_JOINTS,
        num_channels: int = NUM_CHANNELS,
        hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.num_joints = num_joints
        self.num_channels = num_channels
        flat_dim = num_joints * num_channels  # 75

        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, J, C = x.shape
        assert T == self.seq_len, f"Expected seq_len={self.seq_len}, got T={T}"
        assert J == self.num_joints, f"Expected {self.num_joints} joints, got J={J}"
        assert C == self.num_channels, f"Expected {self.num_channels} channels, got C={C}"

        x = x.reshape(B, T, J * C)          # [B, T, J*C]
        assert x.shape == (B, T, J * C)

        x = x.mean(dim=1)                    # [B, J*C]  — mean over time
        assert x.shape == (B, J * C)

        out = self.net(x)                    # [B, 1]
        assert out.shape == (B, 1), f"Head output shape error: {out.shape}"
        return out


# ---------------------------------------------------------------------------
# Joint encoder (shared across time steps)
# ---------------------------------------------------------------------------

class JointMLP(nn.Module):
    """Per-joint feature extractor applied independently to each (frame, joint).

    Flatten joints per frame so spatial structure is preserved (unlike max-pool).
    After the MLP, we have a feature vector per joint per frame.
    """

    def __init__(
        self,
        in_channels: int = NUM_CHANNELS,
        hidden: int = 64,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, J, C]
        assert x.ndim == 4, f"JointMLP expects [B, T, J, C], got {x.shape}"
        B, T, J, C = x.shape
        out = self.net(x)    # [B, T, J, out_dim]
        assert out.shape == (B, T, J, self.out_dim), (
            f"JointMLP output shape error: {out.shape}"
        )
        return out


# ---------------------------------------------------------------------------
# Spatial Transformer — attends over the joint dimension per frame
# ---------------------------------------------------------------------------

class SpatialTransformerEncoder(nn.Module):
    """Self-attention over the J joint dimension for each frame independently.

    Input:  [B, T, J, D]
    Output: [B, T, D]  (mean-pooled over joints after attention)
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        dim_feedforward: int | None = None,
    ) -> None:
        super().__init__()
        ff = dim_feedforward or d_model * 4
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, J, D]
        assert x.ndim == 4, f"SpatialTransformerEncoder expects [B, T, J, D], got {x.shape}"
        B, T, J, D = x.shape
        assert D == self.d_model, f"Expected d_model={self.d_model}, got D={D}"

        x_flat = x.reshape(B * T, J, D)                    # [B*T, J, D]
        attended = self.encoder(x_flat)                     # [B*T, J, D]
        assert attended.shape == (B * T, J, D)

        pooled = attended.mean(dim=1)                       # [B*T, D]
        assert pooled.shape == (B * T, D)

        out = pooled.reshape(B, T, D)                       # [B, T, D]
        assert out.shape == (B, T, D)
        return out


# ---------------------------------------------------------------------------
# Temporal Transformer — attends over the T frame dimension
# ---------------------------------------------------------------------------

class TemporalTransformerEncoder(nn.Module):
    """Self-attention over the T temporal dimension.

    Input:  [B, T, D]
    Output: [B, D]  (mean-pooled over time after attention)
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        dim_feedforward: int | None = None,
    ) -> None:
        super().__init__()
        ff = dim_feedforward or d_model * 4
        self.pos = SinCosPositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        assert x.ndim == 3, f"TemporalTransformerEncoder expects [B, T, D], got {x.shape}"
        B, T, D = x.shape
        assert D == self.d_model, f"Expected d_model={self.d_model}, got D={D}"

        x = self.pos(x)                    # [B, T, D]
        attended = self.encoder(x)         # [B, T, D]
        assert attended.shape == (B, T, D)

        out = attended.mean(dim=1)         # [B, D]
        assert out.shape == (B, D)
        return out


# ---------------------------------------------------------------------------
# Full dual-Transformer model (E2 / E3)
# ---------------------------------------------------------------------------

class RehabTransformerRegressor(nn.Module):
    """Full rehabilitation quality scoring model.

    Pipeline:
      [B, T, J, C]  +  optional exercise_id [B]
        → JointMLP           [B, T, J, joint_dim]
        → projection         [B, T, J, d_model]
        → SpatialTransformer [B, T, d_model]    (attends over joints)
        → + exercise_emb     [B, T, d_model]    (added if num_exercises > 0)
        → TemporalTransformer [B, d_model]       (attends over frames)
        → regression head    [B, 1]
    """

    def __init__(
        self,
        seq_len: int = 100,
        num_joints: int = NUM_JOINTS,
        num_channels: int = NUM_CHANNELS,
        joint_dim: int = 128,
        d_model: int = 256,
        spatial_heads: int = 4,
        spatial_layers: int = 2,
        temporal_heads: int = 4,
        temporal_layers: int = 3,
        dropout: float = 0.1,
        num_exercises: int = 0,     # 0 = no exercise embedding; 5 = pool mode
        multitask: bool = False,    # True = also predict PO and CF sub-scores
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.num_joints = num_joints
        self.num_channels = num_channels
        self.d_model = d_model
        self.num_exercises = num_exercises
        self.multitask = multitask

        self.joint_encoder = JointMLP(in_channels=num_channels, hidden=64, out_dim=joint_dim)
        self.proj = nn.Linear(joint_dim, d_model)

        # Learned exercise type embedding — added to every temporal position
        if num_exercises > 0:
            self.exercise_emb = nn.Embedding(num_exercises, d_model)
        else:
            self.exercise_emb = None

        self.spatial = SpatialTransformerEncoder(
            d_model=d_model,
            nhead=spatial_heads,
            num_layers=spatial_layers,
            dropout=dropout,
        )
        self.temporal = TemporalTransformerEncoder(
            d_model=d_model,
            nhead=temporal_heads,
            num_layers=temporal_layers,
            dropout=dropout,
            max_seq_len=seq_len + 10,
        )

        def _make_head(d: int, dropout: float) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d // 2, 1),
            )

        self.head    = _make_head(d_model, dropout)   # total score (TS)
        if multitask:
            self.head_po = _make_head(d_model, dropout)   # position offset (PO)
            self.head_cf = _make_head(d_model, dropout)   # correctness/fluency (CF)
        else:
            self.head_po = None
            self.head_cf = None

    def forward(
        self,
        x: torch.Tensor,
        exercise_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, J, C = x.shape
        assert T == self.seq_len, f"Expected T={self.seq_len}, got {T}"
        assert J == self.num_joints, f"Expected J={self.num_joints}, got {J}"
        assert C == self.num_channels, f"Expected C={self.num_channels}, got {C}"

        x = self.joint_encoder(x)        # [B, T, J, joint_dim]
        x = self.proj(x)                 # [B, T, J, d_model]
        assert x.shape == (B, T, J, self.d_model)

        x = self.spatial(x)              # [B, T, d_model]
        assert x.shape == (B, T, self.d_model)

        # Add exercise embedding broadcast over all T positions
        if self.exercise_emb is not None and exercise_id is not None:
            assert exercise_id.shape == (B,), (
                f"exercise_id must be [B], got {exercise_id.shape}"
            )
            emb = self.exercise_emb(exercise_id)   # [B, d_model]
            x = x + emb.unsqueeze(1)               # [B, T, d_model]

        feat = self.temporal(x)          # [B, d_model]
        assert feat.shape == (B, self.d_model)

        ts = self.head(feat)             # [B, 1]
        assert ts.shape == (B, 1), f"Head output shape error: {ts.shape}"

        if self.multitask and self.head_po is not None:
            po = self.head_po(feat)      # [B, 1]
            cf = self.head_cf(feat)      # [B, 1]
            return ts, po, cf

        return ts


# ---------------------------------------------------------------------------
# Graph-Aware Spatial Encoder
# ---------------------------------------------------------------------------

class GraphAwareSpatialEncoder(nn.Module):
    """Spatial Transformer whose attention is biased by skeleton graph distance.

    Each attention head learns a scalar slope that scales the pre-computed
    graph-distance matrix (ALiBi-style). Joints that are anatomically close
    (small graph distance) receive smaller negative bias → stronger attention.
    Joints far apart (e.g., left hand ↔ right foot) are suppressed.

    Input:  [B, T, J, D]
    Output: [B, T, D]  (mean-pooled over joints after attention)
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_joints: int = NUM_JOINTS,
        dim_feedforward: int | None = None,
        use_graph_bias: bool = True,
        graph_bias_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        self.d_model        = d_model
        self.nhead          = nhead
        self.num_joints     = num_joints
        self.use_graph_bias = use_graph_bias
        # Continuous strength of the structural prior (N4 transferability sweep):
        #   1.0 = full bone-distance prior (default, original behavior)
        #   0.0 = no prior  (equivalent to free/learned spatial attention)
        # Interpolating lets us trace the accuracy-vs-transferability frontier
        # without toggling architectures or invalidating existing checkpoints.
        self.graph_bias_lambda = float(graph_bias_lambda)
        ff = dim_feedforward or d_model * 4

        # Pre-compute graph distance, registered as a buffer (not trained)
        dist = _graph_distance_matrix(num_joints)           # [J, J]
        self.register_buffer("dist", torch.from_numpy(dist))

        # One learnable log-slope per attention head (positive → negative bias)
        self.log_slopes = nn.Parameter(torch.zeros(nhead))  # [H]

        # Standard TransformerEncoder layers (we inject bias via attn_mask)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ff,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            for _ in range(num_layers)
        ])

    def _build_attn_bias(self) -> torch.Tensor:
        """Returns [H, J, J] additive bias; dist buffer already lives on correct device."""
        slopes = torch.exp(self.log_slopes)                 # [H] all positive
        bias = -self.graph_bias_lambda * slopes[:, None, None] * self.dist[None]  # [H, J, J]
        return bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, J, D]
        B, T, J, D = x.shape
        assert D == self.d_model
        x_flat = x.reshape(B * T, J, D)                    # [B*T, J, D]

        # Build additive attn_mask: [H, J, J] → [B*T*H, J, J] for MHA
        # PyTorch MHA with batch_first=True accepts attn_mask as [J, J] or
        # [B*nhead, J, J]. We broadcast over B*T by repeating.
        bias = self._build_attn_bias()                      # [H, J, J]
        BT   = B * T
        # Expand: [H, J, J] → [BT, H, J, J] → [BT*H, J, J]
        attn_mask = bias.unsqueeze(0).expand(BT, -1, -1, -1).reshape(BT * self.nhead, J, J)

        out = x_flat
        mask = attn_mask if self.use_graph_bias else None
        for layer in self.layers:
            # TransformerEncoderLayer.forward accepts attn_mask=[S,S] or [BT*H,S,S]
            out = layer(out, src_mask=mask)

        pooled = out.mean(dim=1)                            # [B*T, D]
        return pooled.reshape(B, T, D)                      # [B, T, D]


# ---------------------------------------------------------------------------
# Graph-Aware Transformer Regressor
# ---------------------------------------------------------------------------

class GraphAwareTransformerRegressor(nn.Module):
    """Dual Transformer with bone-topology-biased spatial attention.

    Identical to RehabTransformerRegressor except SpatialTransformerEncoder
    is replaced with GraphAwareSpatialEncoder, which biases spatial attention
    by skeleton graph distance — anatomically adjacent joints attend more.
    """

    def __init__(
        self,
        seq_len: int = 100,
        num_joints: int = NUM_JOINTS,
        num_channels: int = NUM_CHANNELS,
        joint_dim: int = 128,
        d_model: int = 128,
        spatial_heads: int = 4,
        spatial_layers: int = 2,
        temporal_heads: int = 4,
        temporal_layers: int = 3,
        dropout: float = 0.1,
        num_exercises: int = 0,
        multitask: bool = False,
        use_graph_bias: bool = True,
        graph_bias_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        self.seq_len      = seq_len
        self.num_joints   = num_joints
        self.num_channels = num_channels
        self.d_model      = d_model
        self.multitask    = multitask

        self.joint_encoder = JointMLP(in_channels=num_channels, hidden=64, out_dim=joint_dim)
        self.proj          = nn.Linear(joint_dim, d_model)

        self.exercise_emb = nn.Embedding(num_exercises, d_model) if num_exercises > 0 else None

        self.spatial = GraphAwareSpatialEncoder(
            d_model=d_model,
            nhead=spatial_heads,
            num_layers=spatial_layers,
            dropout=dropout,
            num_joints=num_joints,
            use_graph_bias=use_graph_bias,
            graph_bias_lambda=graph_bias_lambda,
        )
        self.temporal = TemporalTransformerEncoder(
            d_model=d_model,
            nhead=temporal_heads,
            num_layers=temporal_layers,
            dropout=dropout,
            max_seq_len=seq_len + 10,
        )

        def _head(d: int, p: float) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d // 2),
                nn.GELU(),
                nn.Dropout(p),
                nn.Linear(d // 2, 1),
            )

        self.head    = _head(d_model, dropout)
        self.head_po = _head(d_model, dropout) if multitask else None
        self.head_cf = _head(d_model, dropout) if multitask else None

    def forward(
        self,
        x: torch.Tensor,
        exercise_id: torch.Tensor | None = None,
    ):
        B, T, J, C = x.shape
        x = self.joint_encoder(x)           # [B, T, J, joint_dim]
        x = self.proj(x)                    # [B, T, J, d_model]
        x = self.spatial(x)                 # [B, T, d_model]

        if self.exercise_emb is not None and exercise_id is not None:
            x = x + self.exercise_emb(exercise_id).unsqueeze(1)

        feat = self.temporal(x)             # [B, d_model]
        ts   = self.head(feat)

        if self.multitask and self.head_po is not None:
            return ts, self.head_po(feat), self.head_cf(feat)
        return ts


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    model_type: Literal["e0_mlp", "transformer", "graph_transformer"] = "transformer",
    seq_len: int = 100,
    dropout: float = 0.1,
    joint_dim: int = 128,
    d_model: int = 128,
    spatial_heads: int = 4,
    spatial_layers: int = 2,
    temporal_heads: int = 4,
    temporal_layers: int = 3,
    num_exercises: int = 0,
    multitask: bool = False,
    use_graph_bias: bool = True,
    graph_bias_lambda: float = 1.0,
) -> nn.Module:
    if model_type == "e0_mlp":
        return E0_BaselineMLP(seq_len=seq_len, dropout=dropout)
    if model_type == "graph_transformer":
        return GraphAwareTransformerRegressor(
            seq_len=seq_len,
            joint_dim=joint_dim,
            d_model=d_model,
            spatial_heads=spatial_heads,
            spatial_layers=spatial_layers,
            temporal_heads=temporal_heads,
            temporal_layers=temporal_layers,
            dropout=dropout,
            num_exercises=num_exercises,
            multitask=multitask,
            use_graph_bias=use_graph_bias,
            graph_bias_lambda=graph_bias_lambda,
        )
    # default: "transformer"
    return RehabTransformerRegressor(
        seq_len=seq_len,
        joint_dim=joint_dim,
        d_model=d_model,
        spatial_heads=spatial_heads,
        spatial_layers=spatial_layers,
        temporal_heads=temporal_heads,
        temporal_layers=temporal_layers,
        dropout=dropout,
        num_exercises=num_exercises,
        multitask=multitask,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
