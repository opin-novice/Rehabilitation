#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
egnn_encoder.py  --  RESEARCH SANDBOX (NOT for the paper)
=========================================================
A from-scratch E(n)-equivariant (Satorras-style) skeleton encoder that honours the SAME output
contract as the paper's steerable e3nn encoder, so it can be dropped behind the EXISTING invariant
cut (equivariant_gru.InvariantProjection) with no change to that code.

Contract (see equivariant_gru.py: encoder call at forward, and InvariantProjection.forward):
    forward(x, mask)  with  x:(N=B*T, J, 3)  root-relative coords,  mask:(N,J) or None
    returns h:(N, J, n_scalar + 3*n_vec)  laid out [ n_scalar scalars | n_vec x (x,y,z) ],
    where the scalars are rotation-INVARIANT and each vector channel is a genuine type-1 vector.

Because the scalars are functions of invariants (pairwise squared distances) and the vector channels
are equivariant (sums of scaled RELATIVE position vectors), EGNN + the same cut is ALSO exactly
SE(3)-invariant -- which is the point: it makes the viewpoint axis a tie by construction, so the
informative comparison against the steerable encoder is ACCURACY and NODE-FAILURE, not viewpoint.

No torch_geometric / torch_scatter (absent in this env); pure torch over the fixed Kinect graph,
whose edge list is reused READ-ONLY from cde_model_mp.skeleton_edges().
"""

import os
import sys

import torch
import torch.nn as nn

# read-only import of the existing fixed skeleton graph
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from cde_model_mp import skeleton_edges                       # noqa: E402


class EGNNLayer(nn.Module):
    """One Satorras EGNN message-passing step with n_vec equivariant vector channels."""

    def __init__(self, n_scalar, n_vec, hidden, coord_clamp=None):
        super().__init__()
        self.n_vec = n_vec
        # coord_clamp: optional symmetric bound on the coordinate-update coefficient. It is a
        # FORWARD-TIME op (no parameter), so a clamp=None model is bit-identical to the original and
        # existing checkpoints load unchanged. Hypothesis: under a dead joint the stale relative
        # vectors inject large spurious coeff*rel contributions into the vector channels; bounding
        # coeff damps that feature-loss cliff.
        self.coord_clamp = coord_clamp
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * n_scalar + 1 + 8, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU())
        self.node_mlp = nn.Sequential(
            nn.Linear(n_scalar + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_scalar))
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_vec))                          # one coefficient per vector channel

    def forward(self, h, V, x, src, dst, bemb):
        # h:(N,J,ns)  V:(N,J,nv,3)  x:(N,J,3)  src/dst:(E,)  bemb:(E,8)
        N, J, ns = h.shape
        hi, hj = h[:, src], h[:, dst]                          # (N,E,ns)
        rel = x[:, src] - x[:, dst]                            # (N,E,3)  equivariant
        d2 = (rel * rel).sum(-1, keepdim=True)                 # (N,E,1)  invariant
        be = bemb.unsqueeze(0).expand(N, -1, -1)               # (N,E,8)
        m = self.edge_mlp(torch.cat([hi, hj, d2, be], dim=-1))  # (N,E,hidden)  invariant messages

        agg = torch.zeros(N, J, m.shape[-1], device=h.device, dtype=h.dtype)
        agg.index_add_(1, dst, m)                              # sum incoming messages
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))     # scalar update (invariant)

        coeff = self.coord_mlp(m)                              # (N,E,nv)  invariant weights
        if self.coord_clamp is not None:
            coeff = coeff.clamp(-self.coord_clamp, self.coord_clamp)   # damp spatial updates
        rel_n = rel / (rel.norm(dim=-1, keepdim=True) + 1.0)   # (N,E,3)   equivariant direction
        dV = coeff.unsqueeze(-1) * rel_n.unsqueeze(2)          # (N,E,nv,3) equivariant
        aggV = torch.zeros(N, J, self.n_vec, 3, device=h.device, dtype=h.dtype)
        aggV.index_add_(1, dst, dV)
        V = V + aggV                                           # vector update (equivariant)
        return h, V


class EGNNEncoder(nn.Module):
    """E(n)-equivariant encoder matching the paper encoder's (n_scalar + 3*n_vec) output contract."""

    def __init__(self, n_scalar=32, n_vec=8, n_layers=4, n_joints=25, hidden=64, coord_clamp=None):
        super().__init__()
        self.n_scalar, self.n_vec, self.n_joints = n_scalar, n_vec, n_joints
        self.coord_clamp = coord_clamp
        src, dst, etype = skeleton_edges()                    # directed edges (E=48), read-only
        self.register_buffer("src", src)
        self.register_buffer("dst", dst)
        self.register_buffer("etype", etype)
        self.register_buffer("jidx", torch.arange(n_joints))
        n_bones = int(etype.max().item()) + 1                 # 24
        self.joint_emb = nn.Embedding(n_joints, 16)
        self.bone_emb = nn.Embedding(n_bones, 8)
        self.init_scalar = nn.Sequential(
            nn.Linear(16 + 1, hidden), nn.SiLU(), nn.Linear(hidden, n_scalar))
        self.init_gain = nn.Parameter(torch.randn(n_vec) * 0.1)
        self.layers = nn.ModuleList(
            [EGNNLayer(n_scalar, n_vec, hidden, coord_clamp=coord_clamp) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        # x:(N,J,3) root-relative. mask is accepted for signature parity but this sandbox EGNN is
        # trained/evaluated with use_mask=False (like the paper's headline model), so mask is unused.
        N, J, _ = x.shape
        r = x.norm(dim=-1, keepdim=True)                      # (N,J,1) invariant radius
        je = self.joint_emb(self.jidx).unsqueeze(0).expand(N, -1, -1)   # (N,J,16)
        h = self.init_scalar(torch.cat([je, r], dim=-1))     # (N,J,n_scalar) invariant
        V = x.unsqueeze(2) * self.init_gain.view(1, 1, -1, 1)  # (N,J,n_vec,3) equivariant
        bemb = self.bone_emb(self.etype)                     # (E,8)
        for layer in self.layers:
            h, V = layer(h, V, x, self.src, self.dst, bemb)
        # layout [ scalars | vec0(x,y,z) | vec1(x,y,z) | ... ] == what InvariantProjection reshapes
        return torch.cat([h, V.reshape(N, J, self.n_vec * 3)], dim=-1)
