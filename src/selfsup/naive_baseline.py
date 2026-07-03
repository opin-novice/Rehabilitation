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


def naive_auroc(X: np.ndarray, labels: np.ndarray) -> float:
    """Best unsupervised AUROC over the two naive features (direction-agnostic)."""
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels)
    if labels is None or len(np.unique(labels)) < 2:
        return float("nan")
    feats = compute_naive_features(X)
    best = 0.5
    for j in range(feats.shape[1]):
        try:
            a = roc_auc_score(labels, feats[:, j])
        except ValueError:
            continue
        best = max(best, a, 1.0 - a)
    return float(best)
