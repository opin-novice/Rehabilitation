"""Protocol inflation demonstration (COARSE, SINGLE-SEED — SUPERSEDED).

.. deprecated::
    This 3-point, single-seed sanity check (Protocols A/B/C) is NOT the canonical
    source for the manuscript's protocol-inflation numbers. Its Protocol A uses
    StratifiedGroupKFold(shuffle=False) on a single seed, which can land on an
    unlucky low baseline (rho~0.446 vs the 20-seed average ~0.489) and conflates
    de-stratification with subject leakage. The resulting inflation_B_vs_A (~0.05)
    and inflation_C_vs_A (~0.06) OVERSTATE the effect.

    CANONICAL SOURCE: src/novelty/protocol_decomposition.py -> 
    outputs/novelty/protocol_decomposition.json (20-seed 2x2 factorial). It reports
    subject-leakage main effect = +0.026 rho (95% CI [0.0001, 0.053]) and a
    clinical-stratification main effect ~0. Cite THOSE numbers in the manuscript.
    This script is retained only as a quick qualitative demo.

Shows that evaluating with random sample-level CV (where different exercises
from the same subject can appear in both train and test) inflates Spearman rho
versus strict Leave-One-Subject-Out (LOSO) protocol.

Approach: Ridge regression under 3 evaluation protocols using identical features.
  Protocol A: Stratified LOSO (ours) — all 5 exercises of a subject held out together
  Protocol B: GroupKFold (random subject splits, no stratification)
  Protocol C: Random KFold (sample-level split — subjects can leak across folds)

The difference B-A estimates stratification inflation.
The difference C-A estimates subject-leakage inflation.

Usage:
  python src/protocol_inflation.py
  python src/protocol_inflation.py --pooled_dir KIMORE_pooled --out_dir outputs/protocol_inflation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from constants import SEQ_LEN, NUM_JOINTS, NUM_CHANNELS, JOINT_STARTS

EXERCISE_NAMES = {0: "k01", 1: "k02", 2: "k03", 3: "k04", 4: "k05"}
ALPHAS = np.logspace(-2, 4, 20)

# Suppress ill-conditioned matrix warnings from RidgeCV's internal SVD
warnings.filterwarnings("ignore", message=".*ill-conditioned.*")
warnings.filterwarnings("ignore", message=".*rcond.*")


def extract_features(X_raw: np.ndarray, exercise_ids: np.ndarray) -> np.ndarray:
    """Statistical skeleton features: 455-dim vector per sample."""
    NUM_CH = NUM_CHANNELS
    n = len(exercise_ids)
    xyz_cols = []
    for s in JOINT_STARTS:
        xyz_cols.extend([s, s + 1, s + 2])

    X_3d = X_raw.reshape(n, SEQ_LEN, 100)
    feats = []
    for i in range(n):
        seq = X_3d[i][:, xyz_cols].reshape(SEQ_LEN, NUM_JOINTS, NUM_CH)
        mn  = seq.mean(axis=0)
        sd  = seq.std(axis=0)
        lo  = seq.min(axis=0)
        hi  = seq.max(axis=0)
        vel = np.abs(np.diff(seq, axis=0)).mean(axis=0)
        onehot = np.zeros(5, dtype=np.float32)
        onehot[int(exercise_ids[i])] = 1.0
        feats.append(np.concatenate([mn.ravel(), sd.ravel(), lo.ravel(), hi.ravel(),
                                       (hi - lo).ravel(), vel.ravel(), onehot]).astype(np.float32))
    return np.stack(feats, axis=0)


def eval_protocol(X: np.ndarray, y: np.ndarray,
                  exercise_ids: np.ndarray, cv_splitter,
                  split_kwargs: dict) -> dict:
    """Run one protocol; return per-exercise pooled Spearman and fold-level means."""
    oof_true, oof_pred, oof_eid = [], [], []
    fold_rhos = []

    for train_idx, val_idx in cv_splitter.split(X, **split_kwargs):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        mdl = RidgeCV(alphas=ALPHAS, cv=5)
        mdl.fit(X_tr_s, y_tr)
        y_hat = mdl.predict(X_val_s)

        oof_true.extend(y_val.tolist())
        oof_pred.extend(y_hat.tolist())
        oof_eid.extend(exercise_ids[val_idx].tolist())

        # Fold-level mean rho across exercises
        fold_ex_rhos = []
        for eid in np.unique(exercise_ids[val_idx]):
            m = exercise_ids[val_idx] == eid
            if m.sum() < 3:
                continue
            rho, _ = spearmanr(y_val[m], y_hat[m])
            fold_ex_rhos.append(float(rho))
        if fold_ex_rhos:
            fold_rhos.append(float(np.mean(fold_ex_rhos)))

    oof_true = np.array(oof_true)
    oof_pred = np.array(oof_pred)
    oof_eid  = np.array(oof_eid)

    per_ex = {}
    for eid in range(5):
        m = oof_eid == eid
        if m.sum() < 3:
            continue
        rho, _ = spearmanr(oof_true[m], oof_pred[m])
        per_ex[f"k0{eid+1}"] = float(rho)

    return {
        "mean_rho_pooled": float(np.mean(list(per_ex.values()))) if per_ex else float("nan"),
        "mean_rho_fold":   float(np.nanmean(fold_rhos)) if fold_rhos else float("nan"),
        "std_rho_fold":    float(np.nanstd(fold_rhos))  if fold_rhos else float("nan"),
        "per_exercise":    per_ex,
        "n_folds":         len(fold_rhos),
    }


def main() -> None:
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooled_dir", default="KIMORE_pooled")
    parser.add_argument("--out_dir",    default="outputs/protocol_inflation")
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X_raw        = pd.read_csv(os.path.join(args.pooled_dir, "Train_X.csv"), header=None).values
    y            = pd.read_csv(os.path.join(args.pooled_dir, "Train_Y.csv"), header=None).values.ravel()
    exercise_ids = pd.read_csv(os.path.join(args.pooled_dir, "exercise_ids.csv"), header=None).values.ravel()
    subject_ids  = pd.read_csv(os.path.join(args.pooled_dir, "subject_ids.csv"), header=None).values.ravel()
    meta         = pd.read_csv(os.path.join(args.pooled_dir, "meta.csv"))
    group_labels = meta["group"].values
    print(f"  N={len(y)}, subjects={len(np.unique(subject_ids))}")

    print("Extracting features...")
    X = extract_features(X_raw, exercise_ids)

    # ── Protocol A: Stratified LOSO (our paper) ──────────────────────────────
    print("\nProtocol A — Stratified LOSO (ours)...")
    cv_a = StratifiedGroupKFold(n_splits=5, shuffle=False)
    res_a = eval_protocol(X, y, exercise_ids, cv_a,
                          {"y": group_labels, "groups": subject_ids})
    print(f"  Mean rho (pooled): {res_a['mean_rho_pooled']:.4f}")
    print(f"  Mean rho (fold-avg): {res_a['mean_rho_fold']:.4f} ± {res_a['std_rho_fold']:.4f}")

    # ── Protocol B: GroupKFold (subject-level, no stratification) ────────────
    print("\nProtocol B — GroupKFold (subject-level, unstratified)...")
    cv_b = GroupKFold(n_splits=5)
    res_b = eval_protocol(X, y, exercise_ids, cv_b,
                          {"y": y, "groups": subject_ids})
    print(f"  Mean rho (pooled): {res_b['mean_rho_pooled']:.4f}")
    print(f"  Mean rho (fold-avg): {res_b['mean_rho_fold']:.4f} ± {res_b['std_rho_fold']:.4f}")

    # ── Protocol C: Random KFold (sample-level — subjects leak) ──────────────
    print("\nProtocol C — Random KFold (sample-level; subjects leak across folds)...")
    np.random.seed(42)
    cv_c = KFold(n_splits=5, shuffle=True, random_state=42)
    res_c = eval_protocol(X, y, exercise_ids, cv_c, {"y": y})
    print(f"  Mean rho (pooled): {res_c['mean_rho_pooled']:.4f}")
    print(f"  Mean rho (fold-avg): {res_c['mean_rho_fold']:.4f} ± {res_c['std_rho_fold']:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PROTOCOL INFLATION SUMMARY")
    print("=" * 60)
    for label, res in [("A (Stratified LOSO)", res_a),
                        ("B (GroupKFold, unstratified)", res_b),
                        ("C (Random KFold, sample-level)", res_c)]:
        inflation = res["mean_rho_pooled"] - res_a["mean_rho_pooled"]
        print(f"\nProtocol {label}:")
        print(f"  Mean rho (pooled): {res['mean_rho_pooled']:.4f}  "
              f"({'base' if inflation == 0 else f'+{inflation:.4f} vs A' if inflation > 0 else f'{inflation:.4f} vs A'})")
        print(f"  Per-exercise: " + "  ".join(
            f"{k}={v:.3f}" for k, v in sorted(res["per_exercise"].items())))

    results = {
        "A_stratified_loso": res_a,
        "B_group_kfold":     res_b,
        "C_random_kfold":    res_c,
        "inflation_B_vs_A":  res_b["mean_rho_pooled"] - res_a["mean_rho_pooled"],
        "inflation_C_vs_A":  res_c["mean_rho_pooled"] - res_a["mean_rho_pooled"],
    }
    out_path = os.path.join(args.out_dir, "protocol_inflation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {out_path}")


if __name__ == "__main__":
    main()
