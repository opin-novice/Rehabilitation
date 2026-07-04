"""Joint-angle / relative-joint-vector input representation (reviewer round-3, C2, Q4).

The reviewer asks for a bone-length-preserving / joint-angle representation as the
*main* model input (not a post-hoc canonicalization) to remove padding and
global-scale/translation artifacts at the source. We use per-joint vectors relative
to the skeleton parent (constants.SKELETON_EDGES):

    v[joint] = pos[joint] - pos[parent(joint)]      (root -> 0)

This is translation-invariant (global position cancels) and preserves bone length
(anthropometry), while keeping the canonical (N, T, 25, 3) tensor shape so the
existing TCN/ST-GCN backbones consume it unchanged.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from constants import SKELETON_EDGES, NUM_JOINTS, NUM_CHANNELS, SEQ_LEN  # noqa: E402

# parent[j] = parent joint index of j, or -1 for the root.
_PARENT = [-1] * NUM_JOINTS
for _p, _c in SKELETON_EDGES:
    _PARENT[_c] = _p


def to_relative_joint_vectors(X: np.ndarray) -> np.ndarray:
    """(N, T, 25, 3) coords -> (N, T, 25, 3) parent-relative bone vectors."""
    assert X.shape[-2:] == (NUM_JOINTS, NUM_CHANNELS), X.shape
    out = np.zeros_like(X)
    for j, p in enumerate(_PARENT):
        if p >= 0:
            out[:, :, j, :] = X[:, :, j, :] - X[:, :, p, :]
    return out.astype(np.float32)


def per_sequence_zscore(X: np.ndarray) -> np.ndarray:
    """(N, T, 25, 3) -> strict per-sequence, per-coordinate z-score over time.

    Unlike the pipeline's train-fit global StandardScaler, this removes each
    sequence's own scale/translation (the normalization the manuscript ablates
    for Q10).
    """
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-6
    return ((X - mu) / sd).astype(np.float32)


def xyz4d_to_pooled_cols(X: np.ndarray) -> np.ndarray:
    """(N, T, 25, 3) -> (N*T, 100) in the KIMORE 25-joint x 4-col CSV layout.

    Columns 0-2 of each 4-col joint block hold x,y,z; column 3 is a zero
    orientation placeholder (ignored by rehab_dataset._select_xyz).
    """
    N, T = X.shape[:2]
    cols = np.zeros((N * T, NUM_JOINTS * 4), dtype=np.float32)
    flat = X.reshape(N * T, NUM_JOINTS, NUM_CHANNELS)
    for j in range(NUM_JOINTS):
        cols[:, j * 4:j * 4 + 3] = flat[:, j, :]
    return cols
