"""Ridge Regression baseline with 5-fold Stratified LOSO.

Extracts per-joint statistical features from skeleton sequences (mean, std,
min, max, mean-abs-velocity per joint per axis) then fits Ridge regression
using the same cross-validation protocol as the DL models.

Comparable to Guo & Khan 2021 (mean rho=0.522 with handcrafted features).

Usage:
  python src/ridge_baseline.py
  python src/ridge_baseline.py --pooled_dir KIMORE_pooled --out_dir outputs/ridge_baseline
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from constants import SEQ_LEN, NUM_JOINTS, NUM_CHANNELS, JOINT_STARTS

EXERCISE_NAMES = {
    0: "Trunk Lateral Flex.",
    1: "Trunk Forward Flex.",
    2: "Trunk Rotation",
    3: "Hip Abduction",
    4: "Hip Circumduction",
}


# ── Feature extraction ───────────────────────────────────────────────────────

def _select_xyz(raw_seq: np.ndarray) -> np.ndarray:
    """raw_seq: (SEQ_LEN, 100) → (SEQ_LEN, 25, 3)."""
    cols = []
    for start in JOINT_STARTS:
        cols.extend([start, start + 1, start + 2])
    return raw_seq[:, cols].reshape(SEQ_LEN, NUM_JOINTS, NUM_CHANNELS)


def extract_features(X_raw: np.ndarray, exercise_ids: np.ndarray) -> np.ndarray:
    """
    X_raw: (N*SEQ_LEN, 100)  — raw CSV layout
    exercise_ids: (N,)

    Returns: (N, n_features) feature matrix.
    Feature groups:
      1. Per-joint per-axis statistics: mean, std, min, max, range  → 25*3*5 = 375
      2. Per-joint per-axis mean-abs-velocity                       → 25*3*1 = 75
      3. Exercise one-hot                                            →       5
    Total: 455 features
    """
    n = len(exercise_ids)
    X_3d = X_raw.reshape(n, SEQ_LEN, 100)   # (N, T, raw_cols)

    feats = []
    for i in range(n):
        seq = _select_xyz(X_3d[i])           # (T, 25, 3)

        # 1. Statistical moments per joint per axis
        mn   = seq.mean(axis=0)              # (25, 3)
        sd   = seq.std(axis=0)               # (25, 3)
        lo   = seq.min(axis=0)               # (25, 3)
        hi   = seq.max(axis=0)               # (25, 3)
        rng  = hi - lo                       # (25, 3)
        stat_feats = np.stack([mn, sd, lo, hi, rng], axis=-1).ravel()  # 375

        # 2. Mean absolute velocity per joint per axis
        vel  = np.abs(np.diff(seq, axis=0)).mean(axis=0)  # (25, 3)
        vel_feats = vel.ravel()              # 75

        # 3. Exercise one-hot
        onehot = np.zeros(5, dtype=np.float32)
        onehot[int(exercise_ids[i])] = 1.0  # 5

        feats.append(np.concatenate([stat_feats, vel_feats, onehot]).astype(np.float32))

    return np.stack(feats, axis=0)           # (N, 455)


# ── LOSO cross-validation ─────────────────────────────────────────────────────

def run_loso(X_feat: np.ndarray, y: np.ndarray,
             exercise_ids: np.ndarray, subject_ids: np.ndarray,
             group_labels: np.ndarray, out_dir: str) -> dict:
    """5-fold Stratified LOSO using the same StratifiedGroupKFold as DL models."""
    kfold = StratifiedGroupKFold(n_splits=5, shuffle=False)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    all_oof: list[dict] = []
    fold_rhos: list[float] = []

    # alphas search: same as RidgeCV default logarithmic range
    alphas = np.logspace(-2, 4, 40)

    for fold_idx, (train_idx, val_idx) in enumerate(
        kfold.split(X_feat, group_labels, groups=subject_ids)
    ):
        X_tr, X_val = X_feat[train_idx], X_feat[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Standardise features on train; apply to val
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        # Select alpha by inner 5-fold CV on the training set
        ridge_cv = RidgeCV(alphas=alphas, cv=5)
        ridge_cv.fit(X_tr_s, y_tr)
        best_alpha = float(ridge_cv.alpha_)

        # Refit on full training set with best alpha
        ridge = Ridge(alpha=best_alpha)
        ridge.fit(X_tr_s, y_tr)
        y_hat = ridge.predict(X_val_s)

        # Compute fold Spearman per exercise
        fold_rhos_ex = {}
        for eid in np.unique(exercise_ids[val_idx]):
            mask = exercise_ids[val_idx] == eid
            if mask.sum() < 3:
                continue
            rho, _ = spearmanr(y_val[mask], y_hat[mask])
            fold_rhos_ex[int(eid)] = float(rho)

        fold_rho = float(np.mean(list(fold_rhos_ex.values()))) if fold_rhos_ex else float("nan")
        fold_rhos.append(fold_rho)
        print(f"  Fold {fold_idx}: alpha={best_alpha:.2f}  mean_rho={fold_rho:.4f}  "
              + "  ".join(f"k0{e+1}={v:.3f}" for e, v in sorted(fold_rhos_ex.items())))

        # Collect OOF predictions
        for local_i, global_i in enumerate(val_idx):
            all_oof.append({
                "subject_id":  int(subject_ids[global_i]),
                "exercise_id": int(exercise_ids[global_i]),
                "y_true":      float(y[global_i]),
                "y_pred":      float(y_hat[local_i]),
                "abs_error":   float(abs(y[global_i] - y_hat[local_i])),
                "fold":        fold_idx,
            })

    oof_df = pd.DataFrame(all_oof)
    oof_path = os.path.join(out_dir, "oof_predictions_all.csv")
    oof_df.to_csv(oof_path, index=False)

    # Pooled per-exercise Spearman (matches primary metric of DL models)
    print("\nPooled per-exercise Spearman (N=all OOF):")
    per_ex_rhos = {}
    for eid in range(5):
        sub = oof_df[oof_df["exercise_id"] == eid]
        if len(sub) < 5:
            continue
        rho, p = spearmanr(sub["y_true"].values, sub["y_pred"].values)
        per_ex_rhos[eid] = {"n": len(sub), "spearman": float(rho), "p": float(p)}
        print(f"  k0{eid+1} {EXERCISE_NAMES[eid]:<28}: rho={rho:.4f}  (n={len(sub)})")

    mean_rho = float(np.mean([v["spearman"] for v in per_ex_rhos.values()]))
    print(f"\n  Mean rho (pooled OOF): {mean_rho:.4f}")
    print(f"  Mean rho (fold-avg):  {float(np.nanmean(fold_rhos)):.4f} ± "
          f"{float(np.nanstd(fold_rhos)):.4f}")

    results = {
        "model": "Ridge Regression (statistical features, LOSO)",
        "n_features": int(X_feat.shape[1]),
        "mean_rho_pooled": mean_rho,
        "mean_rho_fold_avg": float(np.nanmean(fold_rhos)),
        "std_rho_fold":  float(np.nanstd(fold_rhos)),
        "per_exercise": {
            f"k0{eid+1}": v["spearman"] for eid, v in sorted(per_ex_rhos.items())
        },
        "fold_rhos": fold_rhos,
    }
    with open(os.path.join(out_dir, "ridge_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {out_dir}/ridge_results.json")
    print(f"OOF     -> {oof_path}")
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    np.random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--pooled_dir", default="KIMORE_pooled")
    parser.add_argument("--out_dir",    default="outputs/ridge_baseline")
    args = parser.parse_args()

    print("Loading KIMORE_pooled data...")
    X_raw       = pd.read_csv(os.path.join(args.pooled_dir, "Train_X.csv"), header=None).values
    y           = pd.read_csv(os.path.join(args.pooled_dir, "Train_Y.csv"), header=None).values.ravel()
    exercise_ids = pd.read_csv(os.path.join(args.pooled_dir, "exercise_ids.csv"), header=None).values.ravel()
    subject_ids  = pd.read_csv(os.path.join(args.pooled_dir, "subject_ids.csv"), header=None).values.ravel()
    meta         = pd.read_csv(os.path.join(args.pooled_dir, "meta.csv"))

    # Stratification label: clinical group (same as DL models)
    group_labels = meta["group"].values

    print(f"  N={len(y)}  exercises={np.unique(exercise_ids).tolist()}  "
          f"subjects={len(np.unique(subject_ids))}  groups={len(np.unique(group_labels))}")

    print("\nExtracting statistical features...")
    X_feat = extract_features(X_raw, exercise_ids)
    print(f"  Feature matrix: {X_feat.shape}")

    print("\nRunning 5-fold Stratified LOSO...")
    run_loso(X_feat, y, exercise_ids, subject_ids, group_labels, args.out_dir)


if __name__ == "__main__":
    main()
