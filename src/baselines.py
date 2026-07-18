#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
baselines.py
============
NTU X-View model zoo with ONE interface, so the driver treats every model identically:

    logits = model(x, length=..., present=..., mask=...)      # x: (B, M, T, V, 3) -> (B, C)

Two models:
  * STGCN                -- the structural baseline. Real spatial-graph + temporal conv on the
                            25-joint Kinect topology. NOT viewpoint-invariant: it consumes raw
                            coordinates, so a camera change is a distribution shift it must LEARN
                            to absorb. This is the arm the viewpoint theorem is meant to beat.
  * EGRUActionClassifier -- OURS. Reuses the exact equivariant encoder + invariant projection from
                            equivariant_gru (so the viewpoint theorem and the G4 dead-node masking
                            hook come along unchanged), with a GRU + linear head for 60-way action
                            classification. Viewpoint invariance is a THEOREM here, not learned.

CTR-GCN is a documented extension point (build_model('ctrgcn') raises with a pointer) rather than a
stub pretending to be the real thing -- a fake baseline is worse than an honest TODO.

Multi-body: NTU actions have up to 2 skeletons. Each present body is scored independently and the
LOGITS are mean-pooled over present bodies (the standard ST-GCN treatment), so an absent (all-zero)
body contributes nothing.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cde_model_mp import KINECT_BONES                                    # noqa: E402
from equivariant_gru import EquivariantSkeletonEncoder, InvariantProjection  # noqa: E402

N_JOINTS = 25


# =============================================================================
# shared 25-joint adjacency (symmetric-normalised) for the GCN baseline
# =============================================================================
def normalized_adjacency():
    """A_hat = D^-1/2 (A + I) D^-1/2 over the Kinect skeleton graph. (V,V) buffer."""
    A = torch.eye(N_JOINTS)
    for a, b in KINECT_BONES:
        A[a, b] = 1.0
        A[b, a] = 1.0
    deg = A.sum(1)
    dinv = torch.diag(deg.clamp(min=1.0).pow(-0.5))
    return dinv @ A @ dinv


# =============================================================================
# ST-GCN spatial-configuration partitioning (the PUBLISHED graph, K=3)
# =============================================================================
CENTER_JOINT = 20                                                # SpineShoulder (ST-GCN's centre)


def _bfs_hops(num_node, bones, center):
    """True hop distance from every joint to `center` (full BFS, not truncated at max_hop=1)."""
    adj = {i: [] for i in range(num_node)}
    for a, b in bones:
        adj[a].append(b)
        adj[b].append(a)
    dist, frontier = {center: 0}, [center]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    if len(dist) != num_node:
        raise ValueError("KINECT_BONES does not connect all 25 joints -- partitioning is undefined")
    return dist


def _normalize_digraph(A):
    Dl = A.sum(0)
    Dn = np.zeros_like(A)
    for i in range(A.shape[0]):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** -1
    return A @ Dn


def _edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1.0
    return A


def stgcn_partition_adjacency(center=CENTER_JOINT):
    """ST-GCN 'spatial configuration partitioning' (Yan et al., AAAI'18 §3.3) as implemented in
    pyskl: the neighbourhood splits into K=3 subsets -- root (self-loop), CENTRIPETAL (inward, the
    edge points toward the body centre) and CENTRIFUGAL (outward) -- each carrying its OWN weights.

    Orientation is derived by full BFS from the centre joint, so it is a property of the skeleton
    tree rather than of a truncated hop matrix.  (The original release computes hop distances with
    max_hop=1, which leaves d(j, centre)=inf for most joints and makes the centripetal/centrifugal
    test degenerate; pyskl -- the implementation that actually reaches the published X-View number
    -- defines the split by inward/outward edge lists instead, which is what we follow.)

    This is exactly what `normalized_adjacency()` above collapses into a single A_hat, which is why
    the compact baseline is a vanilla GCN wearing an ST-GCN label.  Returns (3, V, V).
    """
    V = N_JOINTS
    dist = _bfs_hops(V, KINECT_BONES, center)
    inward = []
    for a, b in KINECT_BONES:
        if dist[a] > dist[b]:
            inward.append((a, b))                                # a -> b moves toward the centre
        elif dist[b] > dist[a]:
            inward.append((b, a))
        else:
            raise ValueError(f"bone {(a, b)} is equidistant from the centre -- orientation ambiguous")
    outward = [(j, i) for i, j in inward]
    iden = _edge2mat([(i, i) for i in range(V)], V)              # root partition (self-loops)
    inm = _normalize_digraph(_edge2mat(inward, V))               # centripetal
    outm = _normalize_digraph(_edge2mat(outward, V))             # centrifugal
    return torch.tensor(np.stack([iden, inm, outm]), dtype=torch.float32)     # (K=3, V, V)


# =============================================================================
# ST-GCN baseline
# =============================================================================
class STGCNBlock(nn.Module):
    """Spatial graph conv (fixed adjacency) + temporal conv, residual, BN, ReLU."""

    def __init__(self, c_in, c_out, stride=1, t_kernel=9):
        super().__init__()
        self.gc = nn.Conv2d(c_in, c_out, kernel_size=1)                 # spatial: 1x1 then A-mix
        self.tc = nn.Sequential(
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, (t_kernel, 1), (stride, 1), ((t_kernel - 1) // 2, 0)),
            nn.BatchNorm2d(c_out))
        self.res = (nn.Identity() if (c_in == c_out and stride == 1)
                    else nn.Sequential(nn.Conv2d(c_in, c_out, 1, (stride, 1)), nn.BatchNorm2d(c_out)))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        # x: (N, C, T, V).  spatial mix: sum_v' x[..,v'] A[v',v]
        r = self.res(x)
        x = self.gc(x)
        x = torch.einsum("nctv,vw->nctw", x, A)                         # graph convolution
        x = self.tc(x)
        return self.relu(x + r)


class STGCN(nn.Module):
    """Compact ST-GCN. forward(x (B,M,T,V,3), ...) -> logits (B, C)."""

    def __init__(self, n_classes=60, base=64, **_):
        super().__init__()
        self.register_buffer("A", normalized_adjacency(), persistent=False)
        self.data_bn = nn.BatchNorm1d(3 * N_JOINTS)
        self.blocks = nn.ModuleList([
            STGCNBlock(3, base), STGCNBlock(base, base),
            STGCNBlock(base, base * 2, stride=2), STGCNBlock(base * 2, base * 2),
            STGCNBlock(base * 2, base * 4, stride=2), STGCNBlock(base * 4, base * 4)])
        self.fc = nn.Linear(base * 4, n_classes)
        self.uses_mask = False                                          # baseline: no G4 hook

    def _one_body(self, x):                                            # x: (B,T,V,3)
        B, T, V, _ = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, V * 3, T)
        x = self.data_bn(x).reshape(B, V, 3, T).permute(0, 2, 3, 1)    # (B,3,T,V)
        for blk in self.blocks:
            x = blk(x, self.A)
        return self.fc(x.mean((2, 3)))                                 # global pool -> (B,C)

    def forward(self, x, length=None, present=None, mask=None):
        B, M, T, V, _ = x.shape
        logits = torch.stack([self._one_body(x[:, m]) for m in range(M)], dim=1)   # (B,M,C)
        if present is None:
            return logits.mean(1)
        w = present.unsqueeze(-1)                                       # (B,M,1)
        return (logits * w).sum(1) / w.sum(1).clamp(min=1.0)           # pool over present bodies


# =============================================================================
# ST-GCN, FAITHFUL to the published architecture (the control anchor)
# =============================================================================
class STGCNFullBlock(nn.Module):
    """Published st_gcn block: partitioned graph conv (K weight sets) + temporal conv, residual."""

    def __init__(self, c_in, c_out, K, stride=1, t_kernel=9, residual=True, dropout=0.0):
        super().__init__()
        self.K = K
        self.gc = nn.Conv2d(c_in, c_out * K, kernel_size=1)               # K weight sets at once
        self.tc = nn.Sequential(
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, (t_kernel, 1), (stride, 1), ((t_kernel - 1) // 2, 0)),
            nn.BatchNorm2d(c_out), nn.Dropout(dropout, inplace=True))
        if not residual:
            self.res = None                                               # first block: no residual
        elif c_in == c_out and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(nn.Conv2d(c_in, c_out, 1, (stride, 1)), nn.BatchNorm2d(c_out))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):                                              # x (N,C,T,V), A (K,V,V)
        r = 0.0 if self.res is None else self.res(x)
        x = self.gc(x)
        n, kc, t, v = x.shape
        x = x.view(n, self.K, kc // self.K, t, v)
        x = torch.einsum("nkctv,kvw->nctw", x, A)                         # per-partition mix
        return self.relu(self.tc(x) + r)


class STGCNFull(nn.Module):
    """ST-GCN exactly as published (Yan et al., AAAI'18): K=3 spatial-configuration partitioning,
    10 blocks (64x4 / 128x3 / 256x3, stride-2 at blocks 5 and 8), learnable edge importance, and
    feature-level pooling over bodies before the classifier.  ~3.1M params.

    This exists because `STGCN` above is a COMPACT approximation -- single adjacency, 6 blocks, no
    edge importance, 1.74M params -- and cannot reach the published X-View number on any recipe.
    A baseline that cannot reproduce its own paper is not a control; per this file's own rule,
    'a fake baseline is worse than an honest gap'.
    """

    def __init__(self, n_classes=60, edge_importance=True, dropout=0.0, **_):
        super().__init__()
        A = stgcn_partition_adjacency()
        self.register_buffer("A", A, persistent=False)
        K = A.shape[0]
        self.data_bn = nn.BatchNorm1d(N_JOINTS * 3)
        cfg = [(3, 64, 1, False), (64, 64, 1, True), (64, 64, 1, True), (64, 64, 1, True),
               (64, 128, 2, True), (128, 128, 1, True), (128, 128, 1, True),
               (128, 256, 2, True), (256, 256, 1, True), (256, 256, 1, True)]
        self.blocks = nn.ModuleList([
            STGCNFullBlock(ci, co, K, stride=s, residual=r, dropout=dropout) for ci, co, s, r in cfg])
        # learnable per-edge mask on each block's adjacency (published: 'edge importance weighting')
        self.edge_imp = (nn.ParameterList([nn.Parameter(torch.ones_like(A)) for _ in cfg])
                         if edge_importance else None)
        self.fc = nn.Linear(256, n_classes)
        self.uses_mask = False                                            # baseline: no G4 hook

    def _one_body(self, x):                                               # (B,T,V,3) -> (B,256)
        B, T, V, _ = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, V * 3, T)
        x = self.data_bn(x).reshape(B, V, 3, T).permute(0, 2, 3, 1)       # (B,3,T,V)
        for i, blk in enumerate(self.blocks):
            A = self.A if self.edge_imp is None else self.A * self.edge_imp[i]
            x = blk(x, A)
        return x.mean((2, 3))                                             # global pool -> (B,256)

    def forward(self, x, length=None, present=None, mask=None):
        B, M, T, V, _ = x.shape
        feats = torch.stack([self._one_body(x[:, m]) for m in range(M)], dim=1)      # (B,M,256)
        if present is None:                                               # published: mean over M
            pooled = feats.mean(1)
        else:                                                             # ...but skip absent bodies
            w = present.unsqueeze(-1)
            pooled = (feats * w).sum(1) / w.sum(1).clamp(min=1.0)
        return self.fc(pooled)                                            # pool FEATURES, then fc


# =============================================================================
# OUR model: equivariant encoder -> invariant projection -> GRU -> classifier
# =============================================================================
class EGRUActionClassifier(nn.Module):
    """SE(3)-invariant action classifier. Viewpoint invariance is a theorem (the encoder is
    equivariant, the projection is invariant, everything after sees only invariants), and the G4
    dead-node masking hook threads straight through via use_mask.

    forward(x (B,M,T,V,3), length (B,M), present (B,M), mask (B,M,T,V)) -> logits (B,C).
    """

    def __init__(self, n_classes=60, n_scalar=32, n_vec=8, n_layers=2, lmax=2,
                 gru_hidden=128, gru_layers=1, head_hidden=128, dropout=0.2,
                 use_mask=False, use_chiral=False, streaming=False, **_):
        super().__init__()
        self.encoder = EquivariantSkeletonEncoder(n_scalar, n_vec, n_layers, lmax, N_JOINTS)
        self.proj = InvariantProjection(n_scalar, n_vec, N_JOINTS, use_chiral=use_chiral)
        self.use_mask = use_mask
        self.uses_mask = use_mask
        # STREAMING (F8, Time-to-First-Score). The bidirectional GRU + mean-over-ALL-frames pool of
        # the published classifier cannot emit a score until the whole clip has arrived, so it has no
        # honest TTFS. streaming=True makes the recurrence UNIDIRECTIONAL (causal: frame k depends
        # only on frames <=k) and reads out a CAUSAL RUNNING MEAN. Nothing here touches the encoder
        # or projection -- both are per-frame and already causal -- so the viewpoint theorem is
        # UNCHANGED (certify still passes). Note the training pool below (mean over valid frames) is
        # exactly the causal running mean evaluated at the LAST frame, so the training path is
        # identical; only the GRU direction and the head width change. The per-frame stream is
        # produced by _one_body_stream / forward_stream for the live benchmark.
        self.streaming = streaming
        in_dim = self.proj.dim + (N_JOINTS if use_mask else 0)         # + liveness vector
        self.gru = nn.GRU(in_dim, gru_hidden, gru_layers, batch_first=True,
                          bidirectional=not streaming)
        head_in = (1 if streaming else 2) * gru_hidden
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes))

    def _one_body(self, x, length, mask):                             # x: (B,T,V,3)
        B, T, V, _ = x.shape
        mflat = mask.reshape(B * T, V) if (self.use_mask and mask is not None) else \
            (torch.ones(B * T, V, dtype=x.dtype, device=x.device) if self.use_mask else None)
        h = self.encoder(x.reshape(B * T, V, 3), mflat)
        inv = self.proj(h, x.reshape(B * T, V, 3), mflat).reshape(B, T, -1)
        feats = [inv]
        if self.use_mask:
            feats.append((mflat.reshape(B, T, V) if mask is not None
                          else torch.ones(B, T, V, dtype=x.dtype, device=x.device)))
        seq, _ = self.gru(torch.cat(feats, dim=-1))
        if length is None:
            pooled = seq.mean(1)
        else:                                                          # ignore the zero-padded tail
            idx = torch.arange(T, device=x.device).unsqueeze(0)
            m = (idx < length.unsqueeze(1)).to(seq.dtype).unsqueeze(-1)
            pooled = (seq * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(pooled)                                       # (B,C)

    def forward(self, x, length=None, present=None, mask=None):
        B, M, T, V, _ = x.shape
        outs = []
        for m in range(M):
            lm = length[:, m] if length is not None else None
            mk = mask[:, m] if mask is not None else None
            outs.append(self._one_body(x[:, m], lm, mk))
        logits = torch.stack(outs, dim=1)                             # (B,M,C)
        if present is None:
            return logits.mean(1)
        w = present.unsqueeze(-1)
        return (logits * w).sum(1) / w.sum(1).clamp(min=1.0)

    # -- STREAMING readout (F8): the live, frame-by-frame prediction the deployed model emits ------
    def _one_body_stream(self, x, mask):                              # x: (B,T,V,3) -> (B,T,C)
        """Per-frame logits from the CAUSAL running mean of the unidirectional GRU. The value at
        frame k is exactly what a live deployment would output after seeing frames [0..k], with the
        running mean a single O(1) sum+count update per step (here vectorised via cumsum). At k=T-1
        it equals _one_body's full-sequence pool, so training and streaming share one objective."""
        assert self.streaming, "streaming readout requires streaming=True (unidirectional GRU)"
        B, T, V, _ = x.shape
        mflat = mask.reshape(B * T, V) if (self.use_mask and mask is not None) else \
            (torch.ones(B * T, V, dtype=x.dtype, device=x.device) if self.use_mask else None)
        h = self.encoder(x.reshape(B * T, V, 3), mflat)
        inv = self.proj(h, x.reshape(B * T, V, 3), mflat).reshape(B, T, -1)
        feats = [inv]
        if self.use_mask:
            feats.append((mflat.reshape(B, T, V) if mask is not None
                          else torch.ones(B, T, V, dtype=x.dtype, device=x.device)))
        seq, _ = self.gru(torch.cat(feats, dim=-1))                    # (B,T,H) causal
        run_mean = seq.cumsum(1) / torch.arange(
            1, T + 1, device=x.device, dtype=seq.dtype).view(1, T, 1)  # causal prefix mean
        return self.head(run_mean)                                     # (B,T,C)

    def forward_stream(self, x, present=None, mask=None):
        """(B,M,T,V,3) -> (B,T,C): the per-frame logit stream, pooled over present bodies at each
        frame. present-weighted exactly as forward() pools the final logits."""
        B, M, T, V, _ = x.shape
        outs = [self._one_body_stream(x[:, m], mask[:, m] if mask is not None else None)
                for m in range(M)]
        logits = torch.stack(outs, dim=1)                             # (B,M,T,C)
        if present is None:
            return logits.mean(1)
        w = present.view(B, M, 1, 1)
        return (logits * w).sum(1) / w.sum(1).clamp(min=1.0)          # (B,T,C)


# =============================================================================
def build_model(name, n_classes=60, **kw):
    name = name.lower()
    if name in ("egru", "ours"):
        return EGRUActionClassifier(n_classes=n_classes, **kw)
    if name in ("stgcn", "st-gcn"):
        return STGCN(n_classes=n_classes, **kw)
    if name in ("stgcn_full", "stgcn-full", "stgcnfull"):
        return STGCNFull(n_classes=n_classes, **kw)
    if name in ("ctrgcn", "ctr-gcn"):
        raise NotImplementedError(
            "CTR-GCN is a documented extension point, not scaffolded here. Drop the reference impl "
            "(github.com/Uate99/CTR-GCN) into a CTRGCN class exposing the same "
            "forward(x (B,M,T,V,3), length, present, mask) -> logits (B,C) interface, then register "
            "it here. A faked baseline is worse than an honest gap.")
    raise ValueError(f"unknown model '{name}'")


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
