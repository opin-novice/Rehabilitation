"""Reviewer round-3, Q10: does strict per-sequence z-scoring change the zero-shot null?

The pipeline standardizes inputs with a train-fit global StandardScaler (which
preserves per-sequence scale). This script ablates the reviewer's suggested
alternative -- strict per-sequence z-scoring, which removes per-sequence scale --
by (build) writing a per-sequence z-scored copy of KIMORE_pooled, (train, external)
running the same 77-fold LOSO, and (eval) applying the identical transform to the
target corpora and evaluating zero-shot.

Usage:
  python src/reviewer_round3_q10_norm.py build
  #   python src/train_loso.py --model_type tcn --loso --resume \
  #       --pooled_dir KIMORE_pooled_perseq --out_dir archive/legacy_results/kimore_loso_78fold_perseq \
  #       --epochs 100 --batch_size 16 --patience 100 --d_model 128
  python src/reviewer_round3_q10_norm.py eval
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/
from constants import SEQ_LEN, NUM_JOINTS, NUM_CHANNELS  # noqa: E402
from rehab_dataset import _select_xyz_columns  # noqa: E402
from selfsup.features import per_sequence_zscore, xyz4d_to_pooled_cols  # noqa: E402
from selfsup.zeroshot_eval import _rebuild_model, _predict, DEGENERACY_PRED_SD  # noqa: E402
from selfsup.data import load_corpus_with_labels  # noqa: E402
from selfsup.naive_baseline import naive_auroc  # noqa: E402

SRC_POOLED = "KIMORE_pooled"
DST_POOLED = "KIMORE_pooled_perseq"
PERSEQ_DIR = "archive/legacy_results/kimore_loso_78fold_perseq"
OUT_DIR = "outputs/reviewer_round3"


def build() -> None:
    os.makedirs(DST_POOLED, exist_ok=True)
    x_raw = pd.read_csv(os.path.join(SRC_POOLED, "Train_X.csv"), header=None).values.astype(np.float32)
    n = x_raw.shape[0] // SEQ_LEN
    xyz = _select_xyz_columns(x_raw).reshape(n, SEQ_LEN, NUM_JOINTS, NUM_CHANNELS)
    cols = xyz4d_to_pooled_cols(per_sequence_zscore(xyz))
    pd.DataFrame(cols).to_csv(os.path.join(DST_POOLED, "Train_X.csv"), header=False, index=False)
    for fn in ("Train_Y.csv", "subject_ids.csv", "exercise_ids.csv", "meta.csv",
               "Train_PO.csv", "Train_CF.csv"):
        src = os.path.join(SRC_POOLED, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(DST_POOLED, fn))
    print(f"[q10-build] wrote {DST_POOLED} ({n} samples, per-sequence z-scored input).")


def _auroc(labels, preds):
    a = roc_auc_score(labels, preds)
    return float(max(a, 1.0 - a))


def evaluate() -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpts = sorted(glob.glob(os.path.join(PERSEQ_DIR, "fold_*", "best_model.pt")))
    if not ckpts:
        print(f"[SKIP] no per-seq checkpoints in {PERSEQ_DIR} (train step not done?)")
        return {}
    out = {"normalization": "per_sequence_zscore", "n_folds": len(ckpts)}
    for corpus in ("REHAB246", "UIPRMD"):
        X, y, _ = load_corpus_with_labels(corpus)
        Xn = per_sequence_zscore(X)
        y = np.asarray(y)
        aurocs, rhos, sds = [], [], []
        for cp in ckpts:
            m = _rebuild_model(torch.load(cp, map_location="cpu"))
            p = _predict(m, Xn)
            sds.append(float(np.std(p)))
            if len(np.unique(y)) > 1:
                aurocs.append(_auroc(y, p))
                rhos.append(float(spearmanr(p, y).correlation))
        a = np.array(aurocs)
        out[corpus] = {
            "mean_auroc": float(a.mean()), "std_auroc": float(a.std()),
            "ci95_auroc": float(1.96 * a.std() / np.sqrt(len(a))),
            "mean_rank_spearman": float(np.nanmean(rhos)),
            "mean_pred_sd": float(np.mean(sds)),
            "degenerate": bool(np.mean(sds) < DEGENERACY_PRED_SD),
            "naive_auroc_perseq": naive_auroc(Xn, y),
        }
        print(f"{corpus}: per-seq-norm AUROC={out[corpus]['mean_auroc']:.3f} "
              f"+/- {out[corpus]['std_auroc']:.3f}")
    with open(os.path.join(OUT_DIR, "q10_perseq_norm.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    (build if cmd == "build" else evaluate if cmd == "eval" else lambda: print("usage: build|eval"))()
