"""Zero-shot cross-sensor evaluation (PRD R5).

Applies each fine-tuned fold model (no retraining) to a target corpus and reports
AUROC, rank-transfer (Spearman of prediction vs label order), the pred_SD
degeneracy gate, and the naive-feature baseline. Rank metric + naive baseline
directly address the metric-conflation critique.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from models_stgcn import TCNRegressor, LSTMRegressor, STGCNRegressor  # noqa: E402
from selfsup.data import load_corpus_with_labels  # noqa: E402
from selfsup.naive_baseline import naive_auroc  # noqa: E402

DEGENERACY_PRED_SD = 0.10


def _rebuild_model(ckpt: dict) -> torch.nn.Module:
    a = ckpt.get("args", {})
    mt = a.get("model_type", "tcn")
    if mt == "stgcn":
        m = STGCNRegressor(seq_len=100, base_channels=a.get("base_channels", 32),
                           dropout=a.get("dropout", 0.3))
    elif mt == "lstm":
        m = LSTMRegressor(seq_len=100, hidden_size=a.get("lstm_hidden", 128),
                          num_layers=a.get("lstm_layers", 2), dropout=a.get("dropout", 0.3))
    else:
        m = TCNRegressor(seq_len=100, d_model=a.get("d_model", 128),
                         num_blocks=a.get("tcn_blocks", 4), dropout=a.get("dropout", 0.3))
    m.load_state_dict(ckpt["model_state"], strict=False)
    m.eval()
    return m


@torch.no_grad()
def _predict(model, X: np.ndarray, batch: int = 256) -> np.ndarray:
    xt = torch.from_numpy(X.astype(np.float32))
    out = []
    for i in range(0, len(xt), batch):
        y = model(xt[i:i + batch])
        if isinstance(y, tuple):
            y = y[0]
        out.append(y.squeeze(-1).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def evaluate_zeroshot(condition_dir: str, corpus: str, dummy: bool = False) -> Optional[dict]:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    X, labels, _ = load_corpus_with_labels(corpus, dummy=dummy)
    if X.shape[0] == 0:
        print(f"[SKIP] zeroshot {corpus}: no data")
        return None
    ckpts = sorted(glob.glob(os.path.join(condition_dir, "fold_*", "best_model.pt")))
    if not ckpts:
        print(f"[SKIP] zeroshot: no fold checkpoints in {condition_dir}")
        return None

    aurocs, rhos, sds = [], [], []
    for cp in ckpts:
        model = _rebuild_model(torch.load(cp, map_location="cpu"))
        preds = _predict(model, X)
        sds.append(float(np.std(preds)))
        if labels is not None and len(np.unique(labels)) > 1:
            try:
                a = roc_auc_score(labels, preds)
                aurocs.append(float(max(a, 1.0 - a)))
            except ValueError:
                pass
            rhos.append(float(spearmanr(preds, labels).correlation))

    mean_sd = float(np.mean(sds)) if sds else float("nan")
    result = {
        "corpus": corpus,
        "n": int(X.shape[0]),
        "n_folds": len(ckpts),
        "mean_auroc": float(np.mean(aurocs)) if aurocs else None,
        "mean_rank_spearman": float(np.nanmean(rhos)) if rhos else None,
        "mean_pred_sd": mean_sd,
        "degenerate": bool(mean_sd < DEGENERACY_PRED_SD),
        "naive_auroc": naive_auroc(X, labels) if labels is not None else None,
    }
    out = Path(condition_dir) / f"zeroshot_{corpus}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[zeroshot] {corpus} @ {os.path.basename(condition_dir)}: "
          f"AUROC={result['mean_auroc']} naive={result['naive_auroc']} "
          f"pred_SD={mean_sd:.3f} degenerate={result['degenerate']}")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition_dir", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--dummy", action="store_true")
    a = ap.parse_args()
    evaluate_zeroshot(a.condition_dir, a.corpus, dummy=a.dummy)
