"""Baseline models for comparison against the dual-Transformer.

Models implemented:
  LSTM_Regressor     — Bidirectional LSTM over flattened joint features per frame.
                       Standard deep sequence baseline.
  STGCNRegressor     — Spatial-Temporal Graph CNN (Yan et al., 2018 - simplified).
                       Exploits Kinect v2 bone topology via graph convolution.

Both accept the same [B, T, J, C] input contract as RehabTransformerRegressor and
support optional exercise_id embedding for pooled multi-exercise training.

Kinect v2 joint topology (25 joints, 0-indexed):
  Spine: 0(hip)→1(spine_mid)→20(spine_shoulder)→2(neck)→3(head)
  L-arm: 20→4(shoulder_L)→5(elbow_L)→6(wrist_L)→7(hand_L); 6→22(thumb_L); 7→21(tip_L)
  R-arm: 20→8(shoulder_R)→9(elbow_R)→10(wrist_R)→11(hand_R); 10→24(thumb_R); 11→23(tip_R)
  L-leg: 0→12(hip_L)→13(knee_L)→14(ankle_L)→15(foot_L)
  R-leg: 0→16(hip_R)→17(knee_R)→18(ankle_R)→19(foot_R)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rehab_dataset import NUM_CHANNELS, NUM_JOINTS
from constants import SKELETON_EDGES


def build_normalised_adjacency(num_joints: int = NUM_JOINTS) -> torch.Tensor:
    """Symmetric normalised adjacency A_hat = D^{-1/2}(A+I)D^{-1/2}."""
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in SKELETON_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A = A + np.eye(num_joints, dtype=np.float32)          # self-loops
    deg = A.sum(axis=1)
    D_inv_sqrt = np.diag(np.where(deg > 0, deg ** -0.5, 0.0))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.from_numpy(A_hat)                         # [J, J]


# ---------------------------------------------------------------------------
# ST-GCN building block
# ---------------------------------------------------------------------------

class STGCNBlock(nn.Module):
    """One spatial-graph conv + temporal conv residual block.

    Input/output shape: [B, C_in, T, J] → [B, C_out, T, J]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel: int = 9,
        dropout: float = 0.2,
        stride: int = 1,
    ) -> None:
        super().__init__()
        pad = temporal_kernel // 2

        self.gcn_weight = nn.Parameter(
            torch.empty(out_channels, in_channels).normal_(0, 0.01)
        )                                                  # [C_out, C_in]

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(temporal_kernel, 1),
                      padding=(pad, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )

        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T, J]
        # Graph convolution: aggregate neighbour features via A, then channel-mix
        # x_agg[b, c, t, v] = sum_u x[b, c, t, u] * A[u, v]
        x_agg = torch.einsum("bctj,jk->bctk", x, A)      # [B, C_in, T, J]
        # Channel transform: equivalent to 1×1 conv over C dim
        B, C, T, J = x_agg.shape
        x_gcn = torch.einsum("oc,bctj->botj", self.gcn_weight, x_agg)  # [B, C_out, T, J]

        x_tcn = self.tcn(x_gcn)                           # [B, C_out, T, J]
        res   = self.residual(x)                           # [B, C_out, T, J]
        return self.act(x_tcn + res)


# ---------------------------------------------------------------------------
# ST-GCN Regressor
# ---------------------------------------------------------------------------

class STGCNRegressor(nn.Module):
    """Spatial-Temporal Graph CNN for rehabilitation quality regression.

    Three ST-GCN blocks (channels: 3→32→64→128) + global pool + MLP head.
    Deliberately small to avoid overfitting on 300-sample folds.
    Optionally injects exercise embedding before the regression head.
    """

    def __init__(
        self,
        seq_len: int = 100,
        num_joints: int = NUM_JOINTS,
        num_channels: int = NUM_CHANNELS,
        base_channels: int = 32,
        dropout: float = 0.3,
        num_exercises: int = 0,
        multitask: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len     = seq_len
        self.num_joints  = num_joints
        self.num_channels = num_channels
        self.multitask   = multitask

        A = build_normalised_adjacency(num_joints)
        self.register_buffer("A", A)

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        # Input BN to normalise raw xyz
        self.bn_in = nn.BatchNorm2d(num_channels)

        self.block1 = STGCNBlock(num_channels, c1, dropout=dropout)
        self.block2 = STGCNBlock(c1, c2, dropout=dropout)
        self.block3 = STGCNBlock(c2, c3, dropout=dropout)

        feat_dim = c3

        if num_exercises > 0:
            self.exercise_emb = nn.Embedding(num_exercises, feat_dim)
        else:
            self.exercise_emb = None

        def _head(d: int, p: float) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d // 2),
                nn.GELU(),
                nn.Dropout(p),
                nn.Linear(d // 2, 1),
            )

        self.head    = _head(feat_dim, dropout)
        self.head_po = _head(feat_dim, dropout) if multitask else None
        self.head_cf = _head(feat_dim, dropout) if multitask else None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled ST-GCN embedding [B, C3] (used by the sensor-ID probe)."""
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.bn_in(x)
        x = self.block1(x, self.A)
        x = self.block2(x, self.A)
        x = self.block3(x, self.A)
        return x.mean(dim=(2, 3))                          # [B, C3]

    def forward(
        self,
        x: torch.Tensor,
        exercise_id: torch.Tensor | None = None,
    ):
        # x: [B, T, J, C]
        B, T, J, C = x.shape
        # Permute to [B, C, T, J] for Conv2d
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.bn_in(x)
        x = self.block1(x, self.A)
        x = self.block2(x, self.A)
        x = self.block3(x, self.A)

        # Global average pool over T and J
        feat = x.mean(dim=(2, 3))                          # [B, C3]

        if self.exercise_emb is not None and exercise_id is not None:
            feat = feat + self.exercise_emb(exercise_id)  # [B, C3]

        ts = self.head(feat)

        if self.multitask and self.head_po is not None:
            return ts, self.head_po(feat), self.head_cf(feat)
        return ts


# ---------------------------------------------------------------------------
# Bidirectional LSTM Regressor
# ---------------------------------------------------------------------------

class LSTMRegressor(nn.Module):
    """Bidirectional LSTM over per-frame joint features.

    Pipeline:
      [B, T, J, C] → flatten J×C → [B, T, J*C]
      → 2-layer BiLSTM → mean-pool over T → dropout → MLP head

    Classic deep sequence baseline; no spatial structure modelling.
    """

    def __init__(
        self,
        seq_len: int = 100,
        num_joints: int = NUM_JOINTS,
        num_channels: int = NUM_CHANNELS,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_exercises: int = 0,
        multitask: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len      = seq_len
        self.num_joints   = num_joints
        self.num_channels = num_channels
        self.multitask    = multitask
        flat_dim = num_joints * num_channels            # 75

        self.out_dim = hidden_size * 2                   # embedding size (bidirectional)

        self.lstm = nn.LSTM(
            input_size=flat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        feat_dim = hidden_size * 2                     # bidirectional

        if num_exercises > 0:
            self.exercise_emb = nn.Embedding(num_exercises, feat_dim)
        else:
            self.exercise_emb = None

        def _head(d: int, p: float) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d // 2),
                nn.GELU(),
                nn.Dropout(p),
                nn.Linear(d // 2, 1),
            )

        self.drop    = nn.Dropout(dropout)
        self.head    = _head(feat_dim, dropout)
        self.head_po = _head(feat_dim, dropout) if multitask else None
        self.head_cf = _head(feat_dim, dropout) if multitask else None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled BiLSTM embedding [B, 2*H] (pre-dropout, pre-head)."""
        B, T, J, C = x.shape
        x = x.reshape(B, T, J * C)                    # [B, T, J*C]
        out, _ = self.lstm(x)                          # [B, T, 2*H]
        return out.mean(dim=1)                         # [B, 2*H]

    def forward(
        self,
        x: torch.Tensor,
        exercise_id: torch.Tensor | None = None,
    ):
        feat = self.drop(self.forward_features(x))     # [B, 2*H]

        if self.exercise_emb is not None and exercise_id is not None:
            feat = feat + self.exercise_emb(exercise_id)

        ts = self.head(feat)

        if self.multitask and self.head_po is not None:
            return ts, self.head_po(feat), self.head_cf(feat)
        return ts


# ---------------------------------------------------------------------------
# TCN Baseline — Temporal Convolutional Network
# ---------------------------------------------------------------------------

class TCNBlock(nn.Module):
    """One residual dilated causal conv block.

    Input/output: [B, C, T]  (channels-first for Conv1d)
    Dilation doubles each block, giving exponentially growing receptive field.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation  # causal: pad left only
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=pad, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.drop  = nn.Dropout(dropout)
        self.act   = nn.GELU()
        self.pad   = pad
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def _trim(self, x: torch.Tensor) -> torch.Tensor:
        """Remove future-looking padding to enforce causality."""
        return x[:, :, : -self.pad] if self.pad > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        out = self.act(self.bn1(self._trim(self.conv1(x))))
        out = self.drop(out)
        out = self.bn2(self._trim(self.conv2(out)))
        return self.act(out + res)


class TCNRegressor(nn.Module):
    """Temporal Convolutional Network for rehabilitation quality regression.

    Pipeline:
      [B, T, J, C] -> flatten J*C -> [B, J*C, T] (channels-first)
      -> input projection -> 4 TCN blocks (doubling dilation) -> mean pool -> head

    Dilated causal convolutions over time; no spatial structure modelling.
    Well-cited time-series regression baseline (Bai et al., 2018).
    """

    def __init__(
        self,
        seq_len: int = 100,
        num_joints: int = NUM_JOINTS,
        num_channels: int = NUM_CHANNELS,
        d_model: int = 128,
        kernel_size: int = 3,
        num_blocks: int = 4,
        dropout: float = 0.3,
        num_exercises: int = 0,
        multitask: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len      = seq_len
        self.num_joints   = num_joints
        self.num_channels = num_channels
        self.multitask    = multitask
        flat_dim = num_joints * num_channels   # 75
        self.out_dim = d_model                 # embedding size exposed for SSL heads

        # Project flat input to d_model channels
        self.input_proj = nn.Sequential(
            nn.Conv1d(flat_dim, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # 4 TCN blocks with exponentially growing dilation: 1, 2, 4, 8
        self.blocks = nn.ModuleList([
            TCNBlock(d_model, d_model, kernel_size=kernel_size,
                     dilation=2 ** i, dropout=dropout)
            for i in range(num_blocks)
        ])

        if num_exercises > 0:
            self.exercise_emb = nn.Embedding(num_exercises, d_model)
        else:
            self.exercise_emb = None

        def _head(d: int, p: float) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d // 2),
                nn.GELU(),
                nn.Dropout(p),
                nn.Linear(d // 2, 1),
            )

        self.drop    = nn.Dropout(dropout)
        self.head    = _head(d_model, dropout)
        self.head_po = _head(d_model, dropout) if multitask else None
        self.head_cf = _head(d_model, dropout) if multitask else None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled embedding [B, d_model] (pre-dropout, pre-head).

        This is the SSL encoder entry point: contrastive/masked heads and the
        regression head all attach to this same representation.
        """
        B, T, J, C = x.shape
        # Flatten joints and put channels first: [B, J*C, T]
        x = x.reshape(B, T, J * C).permute(0, 2, 1)
        x = self.input_proj(x)                 # [B, d_model, T]
        for block in self.blocks:
            x = block(x)                       # [B, d_model, T]
        return x.mean(dim=2)                    # [B, d_model] — mean pool over T

    def forward(
        self,
        x: torch.Tensor,
        exercise_id: torch.Tensor | None = None,
    ):
        feat = self.drop(self.forward_features(x))   # [B, d_model]

        if self.exercise_emb is not None and exercise_id is not None:
            feat = feat + self.exercise_emb(exercise_id)

        ts = self.head(feat)

        if self.multitask and self.head_po is not None:
            return ts, self.head_po(feat), self.head_cf(feat)
        return ts


# ---------------------------------------------------------------------------
# SCT — Spatial-Channel Transformer (unified joint+time attention)
# ---------------------------------------------------------------------------

class SCTRegressor(nn.Module):
    """Spatial-Channel Transformer: joint and time attended in a single block.

    Unlike the dual-stage Dual-Transformer (spatial then temporal separately),
    SCT flattens the T*J token sequence and applies one unified Transformer,
    letting attention span both temporal and spatial dimensions simultaneously.
    This directly ablates the dual-stage design choice.

    Pipeline:
      [B, T, J, C] -> JointMLP per (t,j) -> [B, T*J, d_model]
      -> sinusoidal pos encoding over T*J tokens
      -> L-layer Transformer -> mean pool -> head

    Token count: T*J = 100*25 = 2500  (manageable with d_model=64)
    """

    def __init__(
        self,
        seq_len: int = 100,
        num_joints: int = NUM_JOINTS,
        num_channels: int = NUM_CHANNELS,
        joint_dim: int = 64,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_exercises: int = 0,
        multitask: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len      = seq_len
        self.num_joints   = num_joints
        self.num_channels = num_channels
        self.d_model      = d_model
        self.multitask    = multitask
        self.n_tokens     = seq_len * num_joints   # T*J

        # Per-(joint,frame) feature extractor
        self.joint_mlp = nn.Sequential(
            nn.Linear(num_channels, joint_dim),
            nn.GELU(),
            nn.Linear(joint_dim, d_model),
        )

        # Learnable position embedding over T*J tokens
        self.pos_emb = nn.Embedding(self.n_tokens, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        if num_exercises > 0:
            self.exercise_emb = nn.Embedding(num_exercises, d_model)
        else:
            self.exercise_emb = None

        def _head(d: int, p: float) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d // 2),
                nn.GELU(),
                nn.Dropout(p),
                nn.Linear(d // 2, 1),
            )

        self.drop    = nn.Dropout(dropout)
        self.head    = _head(d_model, dropout)
        self.head_po = _head(d_model, dropout) if multitask else None
        self.head_cf = _head(d_model, dropout) if multitask else None

    def forward(
        self,
        x: torch.Tensor,
        exercise_id: torch.Tensor | None = None,
    ):
        B, T, J, C = x.shape
        # Token sequence: [B, T*J, C] -> JointMLP -> [B, T*J, d_model]
        x_tok = x.reshape(B, T * J, C)
        x_tok = self.joint_mlp(x_tok)           # [B, T*J, d_model]

        # Add learned positional embeddings
        pos = torch.arange(T * J, device=x.device)
        x_tok = x_tok + self.pos_emb(pos)       # [B, T*J, d_model]

        x_tok = self.encoder(x_tok)             # [B, T*J, d_model]
        feat  = self.drop(x_tok.mean(dim=1))    # [B, d_model]

        if self.exercise_emb is not None and exercise_id is not None:
            feat = feat + self.exercise_emb(exercise_id)

        ts = self.head(feat)

        if self.multitask and self.head_po is not None:
            return ts, self.head_po(feat), self.head_cf(feat)
        return ts


# ---------------------------------------------------------------------------
# Parameter counter (same interface as models.py)
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Encoder factory for self-supervised pretraining (Paper 2 / src/ssl)
# ---------------------------------------------------------------------------
# The backbone regressors double as SSL encoders: `forward_features(x) -> (B, D)`
# returns the pooled embedding, and `.out_dim` exposes D. SSL heads
# (projection / reconstruction) attach to this embedding; the regression head is
# ignored during pretraining. Building the encoder with the SAME class keeps
# parameter names identical, so a pretrained checkpoint loads cleanly into the
# fine-tuning regressor via load_state_dict(..., strict=False).

_ENCODER_BUILDERS = {
    "tcn": lambda **kw: TCNRegressor(
        seq_len=kw.get("seq_len", 100),
        d_model=kw.get("d_model", 128),
        num_blocks=kw.get("tcn_blocks", 4),
        dropout=kw.get("dropout", 0.3),
    ),
    "lstm": lambda **kw: LSTMRegressor(
        seq_len=kw.get("seq_len", 100),
        hidden_size=kw.get("lstm_hidden", 128),
        num_layers=kw.get("lstm_layers", 2),
        dropout=kw.get("dropout", 0.3),
    ),
}


def build_encoder(model_type: str = "tcn", **kw) -> nn.Module:
    """Return a backbone exposing `forward_features(x) -> (B, out_dim)`.

    Supported: "tcn", "lstm" (the plan's two SSL arms). Other backbones can be
    added by implementing `forward_features`/`out_dim` and registering here.
    """
    key = model_type.lower()
    if key not in _ENCODER_BUILDERS:
        raise NotImplementedError(
            f"build_encoder: '{model_type}' has no SSL encoder. "
            f"Available: {sorted(_ENCODER_BUILDERS)}. "
            "Add forward_features()/out_dim to the backbone and register it here."
        )
    enc = _ENCODER_BUILDERS[key](**kw)
    if not hasattr(enc, "forward_features") or not hasattr(enc, "out_dim"):
        raise TypeError(f"Encoder for '{model_type}' lacks forward_features/out_dim.")
    return enc


def encoder_state_dict(encoder: nn.Module) -> dict:
    """State dict with regression-head params stripped (encoder weights only)."""
    return {
        k: v for k, v in encoder.state_dict().items()
        if not (k.startswith("head") or k.startswith("exercise_emb"))
    }
