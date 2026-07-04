"""Reviewer round-3, C1 (Q6): does a spatial-prior backbone (ST-GCN) change the null?

Evaluates the 77-fold LOSO ST-GCN models (trained by
`train_loso.py --model_type stgcn --loso --out_dir results/kimore_loso_78fold_stgcn`)
in the identical zero-shot protocol used for the TCN: per-fold direction-agnostic
AUROC on REHAB246 + UI-PRMD, mean rank-Spearman, pred_SD degeneracy gate, the naive
kinematic baseline, and a 3-way sensor-identity probe on the ST-GCN embeddings.

If ST-GCN is also at chance AND its sensor-ID probe is also near-perfect, the null is
architecture-independent -- the single most direct answer to the ``single-backbone''
weakness.

Run (after training completes):  python src/reviewer_round3_c1_stgcn_eval.py
Out: outputs/reviewer_round3/c1_stgcn.{json,md}
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/
from selfsup.zeroshot_eval import _rebuild_model, _predict, DEGENERACY_PRED_SD  # noqa: E402
from selfsup.data import load_corpus_with_labels  # noqa: E402
from selfsup.naive_baseline import naive_auroc  # noqa: E402

STGCN_DIR = "results/kimore_loso_78fold_stgcn"
OUT_DIR = "outputs/reviewer_round3"


def _auroc(labels, preds):
    a = roc_auc_score(labels, preds)
    return float(max(a, 1.0 - a))


@torch.no_grad()
def _features(model, X, batch=256):
    xt = torch.from_numpy(X.astype(np.float32))
    out = []
    for i in range(0, len(xt), batch):
        out.append(model.forward_features(xt[i:i + batch]).cpu().numpy())
    return np.concatenate(out)


def zeroshot(ckpts, X, labels):
    aurocs, rhos, sds = [], [], []
    for cp in ckpts:
        m = _rebuild_model(torch.load(cp, map_location="cpu"))
        p = _predict(m, X)
        sds.append(float(np.std(p)))
        if labels is not None and len(np.unique(labels)) > 1:
            aurocs.append(_auroc(labels, p))
            rhos.append(float(spearmanr(p, labels).correlation))
    a = np.array(aurocs)
    return {
        "n_folds": len(ckpts),
        "mean_auroc": float(a.mean()) if len(a) else None,
        "std_auroc": float(a.std()) if len(a) else None,
        "ci95_auroc": float(1.96 * a.std() / np.sqrt(len(a))) if len(a) else None,
        "mean_rank_spearman": float(np.nanmean(rhos)) if rhos else None,
        "mean_pred_sd": float(np.mean(sds)) if sds else None,
        "degenerate": bool(np.mean(sds) < DEGENERACY_PRED_SD) if sds else None,
        "naive_auroc": naive_auroc(X, labels),
    }


def _adabn_predict(model, X, batch=256):
    """AdaBN (Li et al. 2018): recompute BatchNorm running stats on the target
    domain, then predict. A parameter-free domain-adaptation baseline."""
    import torch.nn as nn
    # reset + re-estimate BN running statistics on the target batch stream
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()
            m.momentum = None          # cumulative moving average over the passes
            m.train()
    xt = torch.from_numpy(X.astype(np.float32))
    with torch.no_grad():
        for i in range(0, len(xt), batch):
            model(xt[i:i + batch])     # forward-only: updates BN running stats
    model.eval()
    return _predict(model, X, batch)


def zeroshot_adabn(ckpts, X, labels):
    aurocs, sds = [], []
    for cp in ckpts:
        m = _rebuild_model(torch.load(cp, map_location="cpu"))
        p = _adabn_predict(m, X)
        sds.append(float(np.std(p)))
        if labels is not None and len(np.unique(labels)) > 1:
            aurocs.append(_auroc(labels, p))
    a = np.array(aurocs)
    return {"n_folds": len(ckpts),
            "mean_auroc": float(a.mean()) if len(a) else None,
            "std_auroc": float(a.std()) if len(a) else None,
            "ci95_auroc": float(1.96 * a.std() / np.sqrt(len(a))) if len(a) else None,
            "mean_pred_sd": float(np.mean(sds)) if sds else None}


def sensor_probe(ckpts, corpora, n_models=5):
    """3-way (and pairwise) balanced-accuracy probe on ST-GCN embeddings."""
    lm = {name: i for i, name in enumerate(corpora)}
    a3 = []
    for cp in ckpts[:n_models]:
        m = _rebuild_model(torch.load(cp, map_location="cpu"))
        Fs, ys = [], []
        for name, X in corpora.items():
            F = _features(m, X)
            Fs.append(F)
            ys.append(np.full(len(F), lm[name]))
        F = np.concatenate(Fs)
        y = np.concatenate(ys)
        a3.append(float(np.mean(cross_val_score(
            LogisticRegression(max_iter=1000, C=1), F, y, cv=3,
            scoring="balanced_accuracy"))))
    return {"mean_3way_balanced_acc": float(np.mean(a3)) if a3 else None,
            "std_3way_balanced_acc": float(np.std(a3)) if a3 else None,
            "chance_3way": 1.0 / len(corpora), "n_models": min(n_models, len(ckpts))}


def run() -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpts = sorted(glob.glob(os.path.join(STGCN_DIR, "fold_*", "best_model.pt")))
    if not ckpts:
        print(f"[SKIP] no ST-GCN checkpoints in {STGCN_DIR} (training not finished?)")
        return {}
    print(f"[c1] {len(ckpts)} ST-GCN fold checkpoints")

    Xr, yr, _ = load_corpus_with_labels("REHAB246")
    Xu, yu, _ = load_corpus_with_labels("UIPRMD")
    Xk, _, _ = load_corpus_with_labels("KIMORE")

    results = {
        "backbone": "stgcn",
        "REHAB246": zeroshot(ckpts, Xr, np.asarray(yr)),
        "UIPRMD": zeroshot(ckpts, Xu, np.asarray(yu)),
        "REHAB246_adabn": zeroshot_adabn(ckpts, Xr, np.asarray(yr)),
        "UIPRMD_adabn": zeroshot_adabn(ckpts, Xu, np.asarray(yu)),
        "sensor_id_probe": sensor_probe(
            ckpts, {"KIMORE": Xk, "REHAB246": Xr, "UIPRMD": Xu}),
    }
    with open(os.path.join(OUT_DIR, "c1_stgcn.json"), "w") as f:
        json.dump(results, f, indent=2)

    r246, ru = results["REHAB246"], results["UIPRMD"]
    pr = results["sensor_id_probe"]
    md = (
        "### C1: ST-GCN zero-shot (77-fold LOSO)\n\n"
        "| Backbone | REHAB246 AUROC | UI-PRMD AUROC | naive (R/U) | sensor-ID 3-way acc |\n"
        "|---|---|---|---|---|\n"
        f"| ST-GCN | {r246['mean_auroc']:.3f} ± {r246['std_auroc']:.3f} "
        f"(95% CI ±{r246['ci95_auroc']:.3f}) | "
        f"{ru['mean_auroc']:.3f} ± {ru['std_auroc']:.3f} (95% CI ±{ru['ci95_auroc']:.3f}) | "
        f"{r246['naive_auroc']:.3f} / {ru['naive_auroc']:.3f} | "
        f"{pr['mean_3way_balanced_acc']:.3f} (chance {pr['chance_3way']:.2f}) |\n"
    )
    with open(os.path.join(OUT_DIR, "c1_stgcn.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)
    return results


if __name__ == "__main__":
    run()
