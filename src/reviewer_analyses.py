"""Comprehensive reviewer-requested analyses.

Runs in one pass:
  1. Extended zero-shot eval: per-fold AUROC + std, per-exercise AUROC
  2. CORAL domain adaptation baseline
  3. Canonicalization (pelvis-center + bone-length norm) baseline

Output: results/reviewer_analyses.json
"""
from __future__ import annotations
import json, os, sys, glob, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models_stgcn import TCNRegressor
from selfsup.data import load_corpus_with_labels
from selfsup.naive_baseline import naive_auroc, compute_naive_features
from selfsup.zeroshot_eval import DEGENERACY_PRED_SD

RESULTS_DIR = "results/kimore_loso_78fold"
OUT_PATH = "results/reviewer_analyses.json"
DEVICE = torch.device("cpu")

CONDITIONS = [
    ("A_scratch", "A. Scratch"),
    ("B_contrastive_lp", "B. Contrastive LP"),
    ("C_contrastive_ft", "C. Contrastive FT"),
    ("D_masked_lp", "D. Masked LP"),
    ("E_masked_ft", "E. Masked FT"),
]

TORCH_SEQ_LEN = 100

def _rebuild_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    a = ckpt.get("args", {})
    m = TCNRegressor(seq_len=TORCH_SEQ_LEN, d_model=a.get("d_model", 128),
                     num_blocks=a.get("tcn_blocks", 4), dropout=a.get("dropout", 0.3))
    m.load_state_dict(ckpt["model_state"], strict=False)
    m.eval()
    return m

@torch.no_grad()
def _predict(model, X, batch=256):
    xt = torch.from_numpy(X.astype(np.float32))
    out = []
    for i in range(0, len(xt), batch):
        y = model(xt[i:i+batch])
        if isinstance(y, tuple): y = y[0]
        out.append(y.squeeze(-1).cpu().numpy())
    return np.concatenate(out) if out else np.array([])

def _extract_features(model, X, batch=256):
    """Extract penultimate-layer features from TCN via forward_features()."""
    xt = torch.from_numpy(X.astype(np.float32))
    all_feats = []
    for i in range(0, len(xt), batch):
        xb = xt[i:i+batch]
        feat = model.forward_features(xb)
        all_feats.append(feat.detach().cpu().numpy())
    return np.concatenate(all_feats, axis=0) if all_feats else np.array([])

# ============================================================
# 1. Extended zero-shot evaluation (per-fold + per-exercise)
# ============================================================
def extended_zeroshot(condition_dir, corpus, X, labels, manifest):
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    ckpts = sorted(glob.glob(os.path.join(condition_dir, "fold_*", "best_model.pt")))
    if not ckpts:
        return None

    # Per-fold AUROCs
    fold_aurocs = []
    fold_rhos = []
    fold_sds = []
    # Per-exercise AUROCs (if manifest has exercise_id)
    has_exercise = manifest is not None and "exercise_id" in manifest.columns
    per_ex_aurocs = {}  # ex_id -> list of AUROC per fold
    per_ex_naive = {}   # ex_id -> naive AUROC

    for cp in ckpts:
        model = _rebuild_model(cp)
        preds = _predict(model, X)
        sd = float(np.std(preds))
        fold_sds.append(sd)

        if labels is not None and len(np.unique(labels)) > 1:
            try:
                a = roc_auc_score(labels, preds)
                a = float(max(a, 1.0 - a))
                fold_aurocs.append(a)
            except ValueError:
                pass
            fold_rhos.append(float(spearmanr(preds, labels).correlation))

        # Per-exercise within this fold
        if has_exercise:
            for ex_id in manifest["exercise_id"].unique():
                mask = manifest["exercise_id"].values == ex_id
                if mask.sum() < 2:
                    continue
                ex_labels = labels[mask]
                ex_preds = preds[mask]
                if len(np.unique(ex_labels)) < 2:
                    continue
                try:
                    ex_a = roc_auc_score(ex_labels, ex_preds)
                    ex_a = float(max(ex_a, 1.0 - ex_a))
                    per_ex_aurocs.setdefault(int(ex_id), []).append(ex_a)
                except ValueError:
                    pass

    # Per-exercise naive AUROC
    if has_exercise:
        for ex_id in manifest["exercise_id"].unique():
            mask = manifest["exercise_id"].values == ex_id
            if mask.sum() < 2:
                continue
            ex_X = X[mask]
            ex_labels = labels[mask]
            per_ex_naive[int(ex_id)] = naive_auroc(ex_X, ex_labels)

    mean_sd = float(np.mean(fold_sds)) if fold_sds else float("nan")
    result = {
        "n_folds": len(ckpts),
        "mean_auroc": float(np.mean(fold_aurocs)) if fold_aurocs else None,
        "std_auroc": float(np.std(fold_aurocs)) if len(fold_aurocs) > 1 else None,
        "ci95_auroc": (1.96 * float(np.std(fold_aurocs)) / np.sqrt(len(fold_aurocs))
                       if len(fold_aurocs) > 1 else None),
        "min_auroc": float(np.min(fold_aurocs)) if fold_aurocs else None,
        "max_auroc": float(np.max(fold_aurocs)) if fold_aurocs else None,
        "fold_aurocs": [round(x, 4) for x in fold_aurocs[:10]] + ["..."],
        "mean_rank_spearman": float(np.nanmean(fold_rhos)) if fold_rhos else None,
        "mean_pred_sd": mean_sd,
        "degenerate": bool(mean_sd < DEGENERACY_PRED_SD),
        "naive_auroc": naive_auroc(X, labels) if labels is not None else None,
        "per_exercise_mean_auroc": {
            str(k): float(np.mean(v)) for k, v in sorted(per_ex_aurocs.items())
        },
        "per_exercise_naive_auroc": {str(k): v for k, v in sorted(per_ex_naive.items())},
    }
    return result

# ============================================================
# 2. CORAL domain adaptation baseline
# ============================================================
def compute_coral_baseline(condition_dir, X_source, y_source,
                           X_target, labels_target, manifest):
    """Apply CORAL alignment between source (KIMORE) and target features.

    Uses scratch TCN encoder features. Aligns target feature distribution
    to source via whitening + re-coloring. Then trains linear probe on
    source aligned features and evaluates on target.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    ckpts = sorted(glob.glob(os.path.join(condition_dir, "fold_*", "best_model.pt")))
    if not ckpts or y_source is None or labels_target is None:
        return None

    fold_aurocs = []
    for cp in ckpts[:10]:  # Use first 10 folds (balanced sample)
        model = _rebuild_model(cp)

        # Extract features
        F_source = _extract_features(model, X_source)
        F_target = _extract_features(model, X_target)

        if F_source.shape[0] == 0 or F_target.shape[0] == 0:
            continue

        # Standardize
        scaler = StandardScaler()
        Fs = scaler.fit_transform(F_source)
        Ft = scaler.transform(F_target)

        # CORAL: align target covariance to source covariance
        # Source whitening
        cov_s = np.cov(Fs, rowvar=False) + 1e-6 * np.eye(Fs.shape[1])
        L_s = np.linalg.cholesky(cov_s)

        # Target whitening  
        cov_t = np.cov(Ft, rowvar=False) + 1e-6 * np.eye(Ft.shape[1])
        L_t = np.linalg.cholesky(cov_t)

        # Align: Ft_aligned = Ft @ inv(L_t).T @ L_s.T
        Ft_aligned = Ft @ np.linalg.inv(L_t).T @ L_s.T

        # Train logistic regression on source, eval on aligned target
        # Use binary labels for source (threshold at mean for KIMORE continuous scores)
        y_source_bin = (y_source > y_source.mean()).astype(int)

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(Fs, y_source_bin)
        preds = clf.predict_proba(Ft_aligned)[:, 1]

        if len(np.unique(labels_target)) > 1:
            try:
                a = roc_auc_score(labels_target, preds)
                fold_aurocs.append(float(max(a, 1.0 - a)))
            except ValueError:
                pass

    if not fold_aurocs:
        return None
    return {
        "mean_auroc": float(np.mean(fold_aurocs)),
        "std_auroc": float(np.std(fold_aurocs)) if len(fold_aurocs) > 1 else None,
        "n_folds_evaluated": len(fold_aurocs),
    }

# ============================================================
# 3. Canonicalization: pelvis-center + bone-length normalize
# ============================================================
NUM_JOINTS = 25
NUM_CHANNELS = 3

def pelvis_center(X):
    """Subtract hip-center (joint 0) from all joints: (N,T,J,C) -> (N,T,J,C)."""
    hip = X[:, :, 0:1, :]  # (N, T, 1, C)
    return X - hip

def bone_length_normalize(X):
    """Divide by mean bone length across all frames and joints.

    Bone length = L2 distance between adjacent joints in skeleton.
    Uses a simple skeleton chain: hip->spine->neck->head left/right chains.
    """
    # Approximate skeleton edges for Kinect v2 25-joint
    edges = [(0,1),(1,20),(20,2),(2,3),   # spine
             (20,4),(4,5),(5,6),(6,7),    # L-arm
             (7,21),(6,22),               # L-hand
             (20,8),(8,9),(9,10),(10,11), # R-arm
             (11,23),(10,24),             # R-hand
             (0,12),(12,13),(13,14),(14,15), # L-leg
             (0,16),(16,17),(17,18),(18,19)] # R-leg
    bone_lengths = []
    for i, j in edges:
        bl = np.linalg.norm(X[:, :, i, :] - X[:, :, j, :], axis=-1)  # (N, T)
        bone_lengths.append(bl)
    mean_bl = np.mean(bone_lengths, axis=0, keepdims=True)  # (1, T)
    # Mean over time
    global_mean = float(np.mean(mean_bl))
    if global_mean < 1e-8:
        return X
    return X / global_mean

def canonicalize(X):
    """Apply pelvis centering then bone-length normalization."""
    return bone_length_normalize(pelvis_center(X))

def compute_canonical_baseline(condition_dir, corpus, X_raw, labels, manifest):
    """Evaluate existing fold models on CANONICALIZED inputs."""
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    # Apply canonicalization to raw z-score-normalized inputs
    X_canon = canonicalize(X_raw)

    ckpts = sorted(glob.glob(os.path.join(condition_dir, "fold_*", "best_model.pt")))
    if not ckpts:
        return None

    aurocs, sds = [], []
    for cp in ckpts:
        model = _rebuild_model(cp)
        preds = _predict(model, X_canon)
        sds.append(float(np.std(preds)))
        if labels is not None and len(np.unique(labels)) > 1:
            try:
                a = roc_auc_score(labels, preds)
                aurocs.append(float(max(a, 1.0 - a)))
            except ValueError:
                pass

    mean_sd = float(np.mean(sds)) if sds else float("nan")
    return {
        "mean_auroc": float(np.mean(aurocs)) if aurocs else None,
        "std_auroc": float(np.std(aurocs)) if len(aurocs) > 1 else None,
        "mean_pred_sd": mean_sd,
        "degenerate": bool(mean_sd < DEGENERACY_PRED_SD),
        "n_folds": len(ckpts),
    }

# ============================================================
# 4. All-corpora pool zero-shot evaluation
# ============================================================
PRETRAIN_DIR = "outputs/ssl_pretrain"

def _train_kimore(model, X_train, y_train, epochs=100, lr=1e-3):
    """Train a TCN on KIMORE continuous scores with MSE loss."""
    from torch.utils.data import DataLoader, TensorDataset
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.from_numpy(X_train.astype(np.float32))
    yt = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1)
    ds = TensorDataset(xt, yt)
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    best_loss = float("inf")
    patience = 100
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            if isinstance(pred, tuple): pred = pred[0]
            loss = torch.nn.functional.mse_loss(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(ds)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            stale = 0
        else:
            stale += 1
        if stale >= patience // 2:
            break
    model.eval()
    return model

def evaluate_all_corpora_zeroshot(X_kimore, y_kimore, X_rehab, labels_rehab,
                                   X_uiprmd, labels_uiprmd):
    """Evaluate zero-shot with all-corpora pretrained encoders."""
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    pool = "all_corpora"
    results = {}
    ssl_configs = [
        ("contrastive", "contrastive"),
        ("masked", "masked"),
    ]
    training_modes = [
        ("lp", False, "Linear Probe"),
        ("ft", True, "Fine-tune"),
    ]

    for ssl_name, ssl_file in ssl_configs:
        ckpt_path = os.path.join(PRETRAIN_DIR, pool, f"{ssl_file}_encoder.pt")
        if not os.path.isfile(ckpt_path):
            print(f"  [SKIP] SSL checkpoint not found: {ckpt_path}")
            continue
        ssl_ckpt = torch.load(ckpt_path, map_location="cpu")
        enc_state = ssl_ckpt.get("encoder_state", ssl_ckpt.get("model_state", {}))

        for mode_name, freeze, mode_label in training_modes:
            cond_key = f"all_corpora_{ssl_name}_{mode_name}"
            print(f"\n  --- {cond_key} ({mode_label}, {pool}) ---")

            for corpus_name, X_c, labels_c in [
                ("REHAB246", X_rehab, labels_rehab),
                ("UIPRMD", X_uiprmd, labels_uiprmd),
            ]:
                t0 = time.time()
                model = TCNRegressor(seq_len=TORCH_SEQ_LEN, d_model=128,
                                     num_blocks=4, dropout=0.3)
                model.load_state_dict(enc_state, strict=False)
                model.eval()

                if freeze:
                    for p in model.parameters():
                        p.requires_grad = False
                    # Replace + train head only
                    model.head = torch.nn.Sequential(
                        torch.nn.LayerNorm(128),
                        torch.nn.Linear(128, 64),
                        torch.nn.GELU(),
                        torch.nn.Dropout(0.3),
                        torch.nn.Linear(64, 1),
                    )
                    model = _train_kimore(model, X_kimore, y_kimore, epochs=50, lr=1e-2)
                else:
                    model = _train_kimore(model, X_kimore, y_kimore)

                preds = _predict(model, X_c)
                sd_val = float(np.std(preds))
                auroc = None
                if labels_c is not None and len(np.unique(labels_c)) > 1:
                    try:
                        a = roc_auc_score(labels_c, preds)
                        auroc = float(max(a, 1.0 - a))
                    except ValueError:
                        pass
                t = time.time() - t0
                key = f"all_corpora/{ssl_name}_{mode_name}/{corpus_name}"
                results[key] = {
                    "mean_auroc": auroc,
                    "pred_sd": sd_val,
                    "degenerate": bool(sd_val < DEGENERACY_PRED_SD),
                    "pool": pool,
                }
                print(f"    {corpus_name}: AUROC={auroc:.4f} pred_SD={sd_val:.3f} t={t:.0f}s")

    # Also evaluate scratch (no SSL)
    print(f"\n  --- all_corpora_scratch (from scratch, all KIMORE) ---")
    for corpus_name, X_c, labels_c in [
        ("REHAB246", X_rehab, labels_rehab),
        ("UIPRMD", X_uiprmd, labels_uiprmd),
    ]:
        t0 = time.time()
        model = TCNRegressor(seq_len=TORCH_SEQ_LEN, d_model=128, num_blocks=4, dropout=0.3)
        model = _train_kimore(model, X_kimore, y_kimore)
        preds = _predict(model, X_c)
        sd_val = float(np.std(preds))
        auroc = None
        if labels_c is not None and len(np.unique(labels_c)) > 1:
            try:
                a = roc_auc_score(labels_c, preds)
                auroc = float(max(a, 1.0 - a))
            except ValueError:
                pass
        t = time.time() - t0
        key = f"all_corpora/scratch/{corpus_name}"
        results[key] = {
            "mean_auroc": auroc,
            "pred_sd": sd_val,
            "degenerate": bool(sd_val < DEGENERACY_PRED_SD),
            "pool": pool,
        }
        print(f"    {corpus_name}: AUROC={auroc:.4f} pred_SD={sd_val:.3f} t={t:.0f}s")

    return results

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Reviewer analyses: loading data...")
    print("=" * 60)

    # Load all data once
    X_rehab, labels_rehab, _ = load_corpus_with_labels("REHAB246")
    X_uiprmd, labels_uiprmd, _ = load_corpus_with_labels("UIPRMD")
    # Load manifests for per-exercise info
    import pandas as pd
    rehab_man = pd.read_csv("outputs/validity/rehab246_manifest.csv")
    uiprmd_man = pd.read_csv("outputs/validity_uiprmd/uiprmd_manifest.csv")

    # Load KIMORE source data for CORAL
    X_kimore, y_kimore, uids_kimore = load_corpus_with_labels("KIMORE")

    print(f"  REHAB246: {X_rehab.shape}, labels {np.unique(labels_rehab) if labels_rehab is not None else None}")
    print(f"  UI-PRMD:  {X_uiprmd.shape}, labels {np.unique(labels_uiprmd) if labels_uiprmd is not None else None}")
    print(f"  KIMORE:   {X_kimore.shape}, labels range {y_kimore.min():.1f}-{y_kimore.max():.1f}" if y_kimore is not None else "  KIMORE: no labels")

    all_results = {}

    # -------------------------------------------------------
    # 1. Extended zero-shot for each condition
    # -------------------------------------------------------
    for cond_dir, cond_label in CONDITIONS:
        cond_path = os.path.join(RESULTS_DIR, cond_dir)
        print(f"\n--- Extended zeroshot: {cond_label} ---")

        for corpus_name, X_c, labels_c, man in [
            ("REHAB246", X_rehab, labels_rehab, rehab_man),
            ("UIPRMD", X_uiprmd, labels_uiprmd, uiprmd_man),
        ]:
            t0 = time.time()
            r = extended_zeroshot(cond_path, corpus_name, X_c, labels_c, man)
            t = time.time() - t0
            if r:
                key = f"{cond_dir}/{corpus_name}"
                all_results[key] = r
                print(f"  {corpus_name}: AUROC={r['mean_auroc']:.4f}±{r['std_auroc']:.4f} "
                      f"CI95=[{r['mean_auroc']-1.96*r['std_auroc']:.4f},{r['mean_auroc']+1.96*r['std_auroc']:.4f}]"
                      f" pred_SD={r['mean_pred_sd']:.3f} t={t:.0f}s")
                if r.get("per_exercise_mean_auroc"):
                    for ex, v in r["per_exercise_mean_auroc"].items():
                        print(f"    Ex{ex}: AUROC={v:.4f} (naive={r['per_exercise_naive_auroc'].get(ex, 'N/A')})")

    # -------------------------------------------------------
    # 2. CORAL baseline
    # -------------------------------------------------------
    print("\n--- CORAL domain adaptation baseline ---")
    # Use scratch condition as feature extractor
    scratch_path = os.path.join(RESULTS_DIR, "A_scratch")
    for corpus_name, X_c, labels_c in [
        ("REHAB246", X_rehab, labels_rehab),
        ("UIPRMD", X_uiprmd, labels_uiprmd),
    ]:
        if y_kimore is None or labels_c is None:
            print(f"  [SKIP] {corpus_name}: missing labels for CORAL")
            continue
        t0 = time.time()
        r = compute_coral_baseline(scratch_path, X_kimore, y_kimore, X_c, labels_c, None)
        t = time.time() - t0
        if r:
            key = f"coral_baseline/{corpus_name}"
            all_results[key] = r
            print(f"  CORAL -> {corpus_name}: AUROC={r['mean_auroc']:.4f}±{r['std_auroc']:.4f} "
                  f"over {r['n_folds_evaluated']} folds (t={t:.0f}s)")

    # -------------------------------------------------------
    # 3. Canonicalization baseline
    # -------------------------------------------------------
    print("\n--- Canonicalization baseline (pelvis-center + bone-length norm) ---")
    for cond_dir, cond_label in CONDITIONS:
        cond_path = os.path.join(RESULTS_DIR, cond_dir)
        for corpus_name, X_c, labels_c in [
            ("REHAB246", X_rehab, labels_rehab),
            ("UIPRMD", X_uiprmd, labels_uiprmd),
        ]:
            t0 = time.time()
            r = compute_canonical_baseline(cond_path, corpus_name, X_c, labels_c, None)
            t = time.time() - t0
            if r:
                key = f"canonicalized/{cond_dir}/{corpus_name}"
                all_results[key] = r
                print(f"  {cond_label} on {corpus_name}: AUROC={r['mean_auroc']:.4f}±{r['std_auroc']:.4f} "
                      f"pred_SD={r['mean_pred_sd']:.3f} deg={r['degenerate']} (t={t:.0f}s)")

    # -------------------------------------------------------
    # 4. All-corpora pool zero-shot
    # -------------------------------------------------------
    print("\n--- All-corpora pool zero-shot (transductive upper bound) ---")
    all_corpora_results = evaluate_all_corpora_zeroshot(
        X_kimore, y_kimore, X_rehab, labels_rehab, X_uiprmd, labels_uiprmd)
    all_results.update(all_corpora_results)

    # Save
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
