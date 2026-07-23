#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
canonicalize.py  --  RESEARCH SANDBOX (NOT for the paper)
=========================================================
Q3 baseline: an UNSUPERVISED canonicalization front-end (PCA principal-axis frame) feeding a
NON-equivariant backbone (the paper's PointCloudTransformerRegressor). This is the "estimate a
canonical frame, then run a plain model" route to viewpoint invariance.

The point of the experiment: PCA gives a rotation-covariant frame, so AGGREGATE accuracy should be
roughly flat under a test-camera rotation -- but the eigenvector sign/order convention is
discontinuous (near-degenerate covariance, symmetric poses), so individual sequences' scores should
still MOVE under rotation. That residual PER-SEQUENCE degradation is exactly what the paper argues an
exact invariant cut avoids and a frame-estimator does not.
"""

import numpy as np


def pca_canonicalize(sample):
    """Rotate every frame's skeleton into its own PCA principal-axis frame (translation removed).

    Deterministic sign convention: each principal axis is oriented so its largest-magnitude joint
    projection is positive; the frame is then forced right-handed. Rotation-covariant in principle,
    but the sign rule is what injects the per-sequence discontinuity under camera rotation.
    """
    x = np.asarray(sample["x"], dtype=np.float64)             # (L, J, 3)
    out = np.empty_like(x)
    for f in range(x.shape[0]):
        p = x[f]
        q = p - p.mean(0, keepdims=True)                      # centre
        cov = q.T @ q
        w, V = np.linalg.eigh(cov)                            # ascending eigenvalues
        V = V[:, ::-1].copy()                                 # descending: principal axes as columns
        for a in range(3):
            proj = q @ V[:, a]
            if proj[np.argmax(np.abs(proj))] < 0:             # fix sign by dominant joint
                V[:, a] = -V[:, a]
        if np.linalg.det(V) < 0:                              # keep a proper rotation
            V[:, 2] = -V[:, 2]
        out[f] = q @ V                                        # express joints in the canonical frame
    s = dict(sample)
    s["x"] = out.astype(np.float32)
    return s
