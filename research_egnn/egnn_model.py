#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
egnn_model.py  --  RESEARCH SANDBOX (NOT for the paper)
=======================================================
The EGNN recurrence: the paper's SE3EquivariantGRU with ONLY its spatial encoder swapped for the
from-scratch EGNN. Everything right of the invariant cut -- InvariantProjection, the dt-aware
bidirectional GRU, the speed/dt channels, the head, and forward() -- is inherited UNCHANGED by
subclassing. No edit is made to src/equivariant_gru.py.
"""

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from equivariant_gru import SE3EquivariantGRU                 # noqa: E402

from egnn_encoder import EGNNEncoder                          # noqa: E402


class EGNNRecurrence(SE3EquivariantGRU):
    """SE3EquivariantGRU with the steerable e3nn encoder replaced by an E(n)-equivariant EGNN.

    super().__init__ builds (and we then discard) the e3nn encoder; the GRU in_dim, proj, and head
    are sized from (n_scalar, n_vec, use_chiral) only, so keeping n_scalar=32, n_vec=8 leaves the
    whole downstream pipeline valid. forward() touches self.encoder exactly once, so the swap is
    complete.
    """

    def __init__(self, n_scalar=32, n_vec=8, n_layers=4, egnn_hidden=64,
                 dropout=0.2, n_exercises=5, use_speed=True, use_chiral=False, coord_clamp=None):
        super().__init__(n_scalar=n_scalar, n_vec=n_vec, dropout=dropout,
                         n_exercises=n_exercises, use_speed=use_speed, use_chiral=use_chiral)
        # replace the steerable encoder; the old e3nn module is dereferenced and its params leave
        # the module tree, so model.parameters() carries the EGNN encoder only.
        self.encoder = EGNNEncoder(n_scalar=n_scalar, n_vec=n_vec, n_layers=n_layers,
                                   n_joints=self.n_joints, hidden=egnn_hidden, coord_clamp=coord_clamp)
