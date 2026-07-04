"""Naive-feature baseline (PRD R5): total joint path length + mean speed.

Printed in every zero-shot table so a null SSL result cannot be dismissed --
replicates the Paper-1 kinematic baseline that beat trained models cross-corpus.
"""
from __future__ import annotations

import numpy as np


def compute_naive_features(X: np.ndarray) -> np.ndarray:
    """(N, T, J, C) -> (N, 2): [total_path_length, mean_speed]."""
    diffs = np.diff(X, axis=1)                       # (N, T-1, J, C)
    step = np.linalg.norm(diffs, axis=3).sum(axis=2)  # (N, T-1) summed over joints
    path_length = step.sum(axis=1)                    # (N,)
    mean_speed = path_length / max(X.shape[1], 1)
    return np.stack([path_length, mean_speed], axis=1)


def compute_naive_features_masked(X: np.ndarray, joint_mask=None) -> np.ndarray:
    """Naive features summed over a subset of joints (Q2 sensitivity).

    joint_mask: boolean array of length J selecting which canonical joints to
    include. None -> all joints (identical to compute_naive_features). Use this
    to exclude zero-padded joints (UI-PRMD 22->25) or duplicated permutation
    targets (REHAB246 thumb=wrist) so the baseline uses only genuinely distinct
    joints -- proving padding/duplication cannot inflate the naive baseline.
    """
    if joint_mask is not None:
        X = X[:, :, np.asarray(joint_mask, dtype=bool), :]
    return compute_naive_features(X)


def _zscore_per_sequence(X: np.ndarray) -> np.ndarray:
    """Per-sequence, per-joint-coordinate z-score (matches the main pipeline)."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-6
    return (X - mu) / sd


def naive_auroc(X: np.ndarray, labels: np.ndarray, joint_mask=None,
                zscore: bool = False) -> float:
    """Best unsupervised AUROC over the two naive features (direction-agnostic).

    joint_mask / zscore expose the Q2 sensitivity variants (shared-joints-only,
    per-sequence z-scored coordinates) while preserving the original default.
    """
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels)
    if labels is None or len(np.unique(labels)) < 2:
        return float("nan")
    Xf = _zscore_per_sequence(X) if zscore else X
    feats = compute_naive_features_masked(Xf, joint_mask)
    best = 0.5
    for j in range(feats.shape[1]):
        try:
            a = roc_auc_score(labels, feats[:, j])
        except ValueError:
            continue
        best = max(best, a, 1.0 - a)
    return float(best)
