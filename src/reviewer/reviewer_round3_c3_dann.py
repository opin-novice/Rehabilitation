"""Reviewer round-3, C3 (Q5): DANN domain-adversarial baseline (stronger than CORAL).

Trains a single domain-adversarial network per target corpus: a shared TCN encoder
with (a) a regression head supervised on labeled KIMORE scores and (b) a domain
classifier (KIMORE vs unlabeled target) behind a gradient-reversal layer, so the
encoder is pushed toward sensor-invariant features. Reported zero-shot, exactly like
the CORAL and all-corpora single-model baselines in Table II.

Inputs are standardized with a scaler fit on KIMORE (matching train_loso). The
sensor-ID probe predicts DANN *should* help domain confusion; the scientific
question is whether it lifts transfer AUROC above chance / the naive baseline.

Usage:  python src/reviewer/reviewer_round3_c3_dann.py [--smoke] [--epochs 60]
Out:    outputs/reviewer_round3/c3_dann.{json,md}
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ root (file now in src/reviewer/)
from constants import SEQ_LEN, NUM_JOINTS, NUM_CHANNELS  # noqa: E402
from models_stgcn import TCNRegressor  # noqa: E402
from selfsup.data import load_corpus_with_labels  # noqa: E402
from selfsup.naive_baseline import naive_auroc  # noqa: E402

OUT_DIR = "outputs/reviewer_round3"


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lamb * grad, None


def grad_reverse(x, lamb):
    return _GradReverse.apply(x, lamb)


class DANN(nn.Module):
    def __init__(self, d_model=128, dropout=0.3):
        super().__init__()
        self.enc = TCNRegressor(seq_len=SEQ_LEN, d_model=d_model, num_blocks=4, dropout=dropout)
        self.domain = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(64, 2))

    def features(self, x):
        return self.enc.forward_features(x)

    def regress(self, x):
        y = self.enc(x)
        return y[0] if isinstance(y, tuple) else y

    def domain_logits(self, x, lamb):
        return self.domain(grad_reverse(self.features(x), lamb))


def _standardize(train_xyz, *others):
    n, T = train_xyz.shape[:2]
    sc = StandardScaler().fit(train_xyz.reshape(n * T, -1))
    def apply(A):
        m = A.shape[0]
        return sc.transform(A.reshape(m * T, -1)).reshape(m, T, NUM_JOINTS, NUM_CHANNELS).astype(np.float32)
    return (apply(train_xyz), *[apply(o) for o in others])


def train_dann(X_src, y_src, X_tgt, epochs=60, bs=32, lr=1e-3, device="cpu"):
    Xs, Xt = _standardize(X_src, X_tgt)
    ys = ((y_src - y_src.mean()) / (y_src.std() + 1e-6)).astype(np.float32)
    m = DANN().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xs_t = torch.tensor(Xs, device=device); ys_t = torch.tensor(ys, device=device).unsqueeze(1)
    Xt_t = torch.tensor(Xt, device=device)
    ns, nt = len(Xs_t), len(Xt_t)
    for ep in range(epochs):
        m.train()
        lamb = 2.0 / (1.0 + np.exp(-10 * ep / max(epochs, 1))) - 1.0   # DANN schedule
        perm = torch.randperm(ns, device=device)
        for i in range(0, ns, bs):
            idx = perm[i:i + bs]
            xs = Xs_t[idx]
            xt = Xt_t[torch.randint(0, nt, (len(idx),), device=device)]
            opt.zero_grad()
            reg_loss = nn.functional.mse_loss(m.regress(xs), ys_t[idx])
            dl = m.domain_logits(torch.cat([xs, xt], 0), lamb)
            dy = torch.cat([torch.zeros(len(xs), dtype=torch.long, device=device),
                            torch.ones(len(xt), dtype=torch.long, device=device)])
            dom_loss = nn.functional.cross_entropy(dl, dy)
            (reg_loss + dom_loss).backward()
            opt.step()
    m.eval()
    return m


@torch.no_grad()
def _auroc(m, X_tgt, labels, X_src_ref, device="cpu"):
    # standardize target with a scaler fit on source (same as training)
    _, Xt = _standardize(X_src_ref, X_tgt)
    p = m.regress(torch.tensor(Xt, device=device)).squeeze(-1).cpu().numpy()
    a = roc_auc_score(labels, p)
    return float(max(a, 1.0 - a)), float(np.std(p))


def run(epochs=60, smoke=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xk, yk, _ = load_corpus_with_labels("KIMORE")
    yk = np.asarray(yk, dtype=np.float32)
    results = {"method": "DANN", "epochs": epochs, "device": device}
    for corpus in ("REHAB246", "UIPRMD"):
        Xt, lt, _ = load_corpus_with_labels(corpus)
        lt = np.asarray(lt)
        if smoke:
            Xk_, yk_, Xt_ = Xk[:40], yk[:40], Xt[:40]
        else:
            Xk_, yk_, Xt_ = Xk, yk, Xt
        m = train_dann(Xk_, yk_, Xt_, epochs=epochs, device=device)
        auroc, sd = _auroc(m, Xt, lt, Xk_, device=device)
        results[corpus] = {"dann_auroc": auroc, "pred_sd": sd,
                           "naive_auroc": naive_auroc(Xt, lt)}
        print(f"{corpus}: DANN AUROC={auroc:.3f} (pred_sd={sd:.3f}) naive={results[corpus]['naive_auroc']:.3f}")
    with open(os.path.join(OUT_DIR, "c3_dann.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(epochs=2 if a.smoke else a.epochs, smoke=a.smoke)
