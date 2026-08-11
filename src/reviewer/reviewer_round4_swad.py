"""Reviewer round-4, T6 follow-up (concern #8): SWAD domain-generalization baseline.

The reviewer asked for deeper DA/DG treatment. IRM needs >=2 *labeled* training
domains (we have one labeled source, KIMORE) and TENT minimizes classification entropy
(our head is regression; AdaBN is the regression-appropriate TTA, already tested). SWAD
(Cha et al., NeurIPS 2021) is the one cited DG method that genuinely applies to a single
labeled source: it seeks *flat minima* by densely averaging weights over the tail of
training, which provably improves out-of-domain generalization. We train it on labeled
KIMORE and evaluate zero-shot on REHAB246 + UI-PRMD, exactly like the CORAL / DANN rows
of Table `tab:robustness-c`.

If SWAD is *also* at chance and below the naive baseline, concern #8 turns from an
argument into evidence: even an explicit flat-minima DG method does not bridge the
compound cross-sensor shift.

Run:  python src/reviewer/reviewer_round4_swad.py [--epochs 90] [--smoke]
Out:  outputs/reviewer_round4/swad.{json,md}
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
from torch.optim.swa_utils import AveragedModel, update_bn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ root (file now in src/reviewer/)
from constants import SEQ_LEN, NUM_JOINTS, NUM_CHANNELS  # noqa: E402
from models_stgcn import TCNRegressor  # noqa: E402
from selfsup.data import load_corpus_with_labels  # noqa: E402
from selfsup.naive_baseline import naive_auroc  # noqa: E402

OUT_DIR = "outputs/reviewer_round4"
SEED = 0


def _subjects(uids):
    return np.array([str(u).split("::")[1] for u in uids])


def _score(model, x):
    """Regression scalar [B] from a (possibly averaged) TCNRegressor."""
    out = model(x)
    out = out[0] if isinstance(out, tuple) else out
    return out.squeeze(-1)


def _make_scaler(X_train):
    n, T = X_train.shape[:2]
    sc = StandardScaler().fit(X_train.reshape(n * T, -1))
    return sc


def _apply(sc, A):
    m, T = A.shape[:2]
    return sc.transform(A.reshape(m * T, -1)).reshape(
        m, T, NUM_JOINTS, NUM_CHANNELS).astype(np.float32)


def train_swad(X_src, y_src, subs, epochs=90, bs=32, lr=1e-3, val_frac=0.2,
               warmup_frac=0.4, patience=5, device="cpu"):
    """Train a KIMORE regressor with SWAD-style dense weight averaging.

    Window: dense per-step averaging begins once validation loss plateaus (best
    epoch + `patience`), and always covers at least the final (1-warmup_frac) of
    training so the average is taken in the flat region. BN stats are recomputed on
    the training stream after averaging (required for SWA correctness).
    """
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    uniq = np.unique(subs)
    n_val = max(1, int(round(val_frac * len(uniq))))
    val_subj = set(rng.choice(uniq, size=n_val, replace=False).tolist())
    is_val = np.array([s in val_subj for s in subs])

    sc = _make_scaler(X_src[~is_val])
    Xtr = _apply(sc, X_src[~is_val]); Xva = _apply(sc, X_src[is_val])
    mu, sd = float(y_src[~is_val].mean()), float(y_src[~is_val].std() + 1e-6)
    ytr = ((y_src[~is_val] - mu) / sd).astype(np.float32)
    yva = ((y_src[is_val] - mu) / sd).astype(np.float32)

    Xtr_t = torch.tensor(Xtr, device=device); ytr_t = torch.tensor(ytr, device=device)
    Xva_t = torch.tensor(Xva, device=device); yva_t = torch.tensor(yva, device=device)

    model = TCNRegressor(seq_len=SEQ_LEN, d_model=128, num_blocks=4, dropout=0.3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    avg = AveragedModel(model)

    ntr = len(Xtr_t)
    warmup_ep = int(warmup_frac * epochs)
    best_val, best_ep = float("inf"), 0
    collecting = False
    n_collected = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(ntr, device=device)
        for i in range(0, ntr, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = nn.functional.mse_loss(_score(model, Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            if collecting:
                avg.update_parameters(model)
                n_collected += 1
        model.eval()
        with torch.no_grad():
            vloss = float(nn.functional.mse_loss(_score(model, Xva_t), yva_t))
        if vloss < best_val:
            best_val, best_ep = vloss, ep
        # start dense collection once val plateaus, and no later than the flat tail
        if not collecting and (ep >= best_ep + patience or ep >= warmup_ep):
            collecting = True
    # ensure the averaged model has collected at least once
    if n_collected == 0:
        avg.update_parameters(model)
    # recompute BN running stats for the averaged weights
    bn_loader = [(Xtr_t[i:i + bs],) for i in range(0, ntr, bs)]
    update_bn(bn_loader, avg, device=device)
    avg.eval()
    return avg, sc, {"n_swa_updates": int(n_collected), "val_best": best_val,
                     "val_best_epoch": int(best_ep), "epochs": epochs}


@torch.no_grad()
def eval_zeroshot(model, sc, X_tgt, labels, device="cpu"):
    Xt = _apply(sc, X_tgt)
    p = _score(model, torch.tensor(Xt, device=device)).cpu().numpy()
    a = roc_auc_score(labels, p)
    return float(max(a, 1.0 - a)), float(np.std(p))


def run(epochs=90, smoke=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xk, yk, uk = load_corpus_with_labels("KIMORE")
    yk = np.asarray(yk, dtype=np.float32)
    subs = _subjects(uk)
    if smoke:
        Xk, yk, subs = Xk[:60], yk[:60], subs[:60]
        epochs = 3
    avg, sc, meta = train_swad(Xk, yk, subs, epochs=epochs, device=device)

    results = {"method": "SWAD", "device": device, **meta}
    for corpus in ("REHAB246", "UIPRMD"):
        Xt, lt, _ = load_corpus_with_labels(corpus)
        lt = np.asarray(lt)
        auroc, psd = eval_zeroshot(avg, sc, Xt, lt, device=device)
        results[corpus] = {"swad_auroc": auroc, "pred_sd": psd,
                           "naive_auroc": naive_auroc(Xt, lt)}
        print(f"{corpus}: SWAD AUROC={auroc:.3f} (pred_sd={psd:.3f}) "
              f"naive={results[corpus]['naive_auroc']:.3f}")

    with open(os.path.join(OUT_DIR, "swad.json"), "w") as f:
        json.dump(results, f, indent=2)
    r, u = results["REHAB246"], results["UIPRMD"]
    md = ("### SWAD (flat-minima domain generalization) zero-shot\n\n"
          "| Method | REHAB246 AUROC | UI-PRMD AUROC | naive (R/U) |\n"
          "|---|---|---|---|\n"
          f"| SWAD | {r['swad_auroc']:.3f} (SD {r['pred_sd']:.3f}) | "
          f"{u['swad_auroc']:.3f} (SD {u['pred_sd']:.3f}) | "
          f"{r['naive_auroc']:.3f} / {u['naive_auroc']:.3f} |\n")
    with open(os.path.join(OUT_DIR, "swad.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(epochs=a.epochs, smoke=a.smoke)
