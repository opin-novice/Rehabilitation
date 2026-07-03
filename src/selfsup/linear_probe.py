"""Linear probe (Layer 2): monitor representation quality + pretraining sanity.

Used both as a checkpoint-selection signal during pretraining and as the
'the encoder DID learn something' sanity check that makes a downstream null
credible (see ARCHITECTURE_PAPER2.md, requirement 5).
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def extract_features(encoder: torch.nn.Module, x: np.ndarray,
                     device: str = "cpu", batch_size: int = 256) -> np.ndarray:
    encoder.eval()
    feats = []
    xt = torch.from_numpy(x.astype(np.float32))
    for i in range(0, len(xt), batch_size):
        fb = encoder.forward_features(xt[i:i + batch_size].to(device))
        feats.append(fb.cpu().numpy())
    return np.concatenate(feats, axis=0)


def linear_probe_score(encoder, x: np.ndarray, y: np.ndarray, device: str = "cpu") -> dict:
    """Ridge probe on frozen features -> R^2 + Spearman. y can be score or label."""
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict

    feats = extract_features(encoder, x, device=device)
    n = len(feats)
    if n < 5:
        return {"r2": float("nan"), "spearman": float("nan"), "n": n}
    cv = min(5, n)
    pred = cross_val_predict(Ridge(alpha=1.0), feats, y, cv=cv)
    rho = spearmanr(pred, y).correlation
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1e-9
    return {"r2": 1.0 - ss_res / ss_tot, "spearman": float(rho), "n": n}
