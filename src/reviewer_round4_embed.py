"""Reviewer round-4, T4 (concern #4): embedding visualization of the sensor-entanglement.

Produces a 2x2 t-SNE figure of penultimate features on pooled KIMORE+REHAB246+UI-PRMD:
  rows = backbone {TCN scratch, ST-GCN}
  cols = coloring {by sensor/corpus, by correctness label (targets only)}
The left column should show clean sensor separation (visual proof of the 1.00 probe);
the right column should show NO label separation (movement quality is not encoded).

Run:  python src/reviewer_round4_embed.py
Out:  figures/embedding_tsne.png
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/
from selfsup.zeroshot_eval import _rebuild_model  # noqa: E402
from selfsup.data import load_corpus_with_labels  # noqa: E402

FIG_DIR = "figures"
BACKBONES = {
    "TCN (scratch)": "archive/legacy_results/kimore_loso_78fold/A_scratch",
    "ST-GCN": "archive/legacy_results/kimore_loso_78fold_stgcn",
}
SENSOR_COLORS = {"KIMORE": "#4477AA", "REHAB246": "#EE6677", "UIPRMD": "#228833"}
MAX_PER_CORPUS = 400  # subsample for legible t-SNE


@torch.no_grad()
def _features(model, X, batch=256):
    xt = torch.from_numpy(X.astype(np.float32))
    out = []
    for i in range(0, len(xt), batch):
        out.append(model.forward_features(xt[i:i + batch]).cpu().numpy())
    return np.concatenate(out)


def _load_pooled(seed=0):
    rng = np.random.default_rng(seed)
    Xs, sensor, label = [], [], []
    for corpus in ("KIMORE", "REHAB246", "UIPRMD"):
        X, y, _ = load_corpus_with_labels(corpus)
        idx = rng.choice(len(X), size=min(MAX_PER_CORPUS, len(X)), replace=False)
        Xs.append(X[idx]); sensor += [corpus] * len(idx)
        if corpus == "KIMORE":
            label += [np.nan] * len(idx)          # continuous score -> not a binary label
        else:
            label += list(np.asarray(y)[idx])
    return np.concatenate(Xs), np.array(sensor), np.array(label, dtype=float)


def _first_ckpt(d):
    c = sorted(glob.glob(os.path.join(d, "fold_*", "best_model.pt")))
    return c[0] if c else None


def run():
    os.makedirs(FIG_DIR, exist_ok=True)
    X, sensor, label = _load_pooled()
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for r, (bb, d) in enumerate(BACKBONES.items()):
        cp = _first_ckpt(d)
        if cp is None:
            print(f"[SKIP] {bb}: no checkpoint in {d}")
            continue
        F = _features(_rebuild_model(torch.load(cp, map_location="cpu")), X)
        emb = TSNE(n_components=2, perplexity=30, init="pca",
                   random_state=0).fit_transform(F)
        # col 0: by sensor
        ax = axes[r][0]
        for s, c in SENSOR_COLORS.items():
            m = sensor == s
            ax.scatter(emb[m, 0], emb[m, 1], s=6, c=c, label=s, alpha=0.6, linewidths=0)
        ax.set_title(f"{bb} — colored by sensor")
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.legend(markerscale=2, fontsize=8, loc="best")
        # col 1: by correctness (targets only)
        ax = axes[r][1]
        tgt = np.isin(sensor, ["REHAB246", "UIPRMD"]) & ~np.isnan(label)
        for lv, c, nm in ((1.0, "#4477AA", "correct"), (0.0, "#EE6677", "incorrect")):
            m = tgt & (label == lv)
            ax.scatter(emb[m, 0], emb[m, 1], s=6, c=c, label=nm, alpha=0.6, linewidths=0)
        ax.scatter(emb[~tgt, 0], emb[~tgt, 1], s=5, c="#CCCCCC", alpha=0.3,
                   label="KIMORE (no binary label)", linewidths=0)
        ax.set_title(f"{bb} — target reps colored by correctness")
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.legend(markerscale=2, fontsize=8, loc="best")
    fig.suptitle("t-SNE of penultimate features: sensor separates cleanly, "
                 "correctness does not", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIG_DIR, "embedding_tsne.png")
    fig.savefig(out, dpi=200)
    print(f"[embed] wrote {out}")
    return out


if __name__ == "__main__":
    run()
