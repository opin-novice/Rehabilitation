"""Does the sensor-identity probe survive the UI-PRMD geometry repair?

The paper's central mechanistic claim is that encoders represent *sensor identity* rather
than movement quality: a 3-way probe on penultimate TCN features separates KIMORE /
REHAB246 / UI-PRMD at balanced accuracy 1.00 against 0.33 chance. It pre-empts the obvious
confound by zeroing padded/duplicated joints {7,11,22,23,24}.

That control is insufficient. UI-PRMD's sequences were built by reading parent-relative
bone offsets as if they were world coordinates, so 57.3% of their coordinates never change
(REHAB246: 0.000). Zeroing padded joints removes only the 0.120 attributable to padding.
UI-PRMD was therefore separable by a single scalar -- "what fraction of your coordinates
are constant" -- and a perfect probe involving it is not evidence of learned entanglement.

This script re-runs the probe with UI-PRMD loaded from the raw (buggy) cache and from the
forward-kinematics cache, everything else held fixed, and additionally reports a
*featureless* control: how well the constant-coordinate fraction ALONE separates the
corpora. If the probe stays at 1.00 under FK, the mechanism claim survives and the paper
needs a table patch. If it collapses, the mechanism was partly an artifact.

Usage:
    python src/sensor_probe_fk_check.py [--n-models 5]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/
from selfsup.zeroshot_eval import _rebuild_model
from selfsup.data import load_corpus_with_labels

RESULTS_DIR = "archive/legacy_results/kimore_loso_78fold"
CONDITION = "A_scratch"
SEQ_LEN = 100

# The paper reports the probe at 1.00 on both TCN and ST-GCN features, arguing sensor
# entanglement is architecture-independent. Both are checked here so the FK result closes
# that claim rather than half of it.
BACKBONES = {
    "TCN (A_scratch)": os.path.join(RESULTS_DIR, CONDITION),
    "ST-GCN": "archive/legacy_results/kimore_loso_78fold_stgcn",
}
PADDED_JOINTS = [7, 11, 22, 23, 24]  # the paper's stated control
OUT = "outputs/novelty/sensor_probe_fk_check.json"

UIPRMD_CACHES = {
    "raw (paper)": "outputs/validity_uiprmd_raw",
    "fk reference": "outputs/validity_uiprmd",
    "fk corrected": "outputs/validity_uiprmd_corrected",
}


def _rebuild(ckpt_path):
    """Dispatches on ckpt["args"]["model_type"], so TCN and ST-GCN both load."""
    return _rebuild_model(torch.load(ckpt_path, map_location="cpu", weights_only=False))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _feats(model, X, batch=256):
    model = model.to(DEVICE)
    xt = torch.from_numpy(X.astype(np.float32))
    out = [model.forward_features(xt[i:i + batch].to(DEVICE)).cpu().numpy()
           for i in range(0, len(xt), batch)]
    return np.concatenate(out, axis=0)


def _zero_padded(X: np.ndarray) -> np.ndarray:
    """Apply the paper's stated control: zero padded/duplicated joints in every corpus."""
    Z = X.copy()
    Z[:, :, PADDED_JOINTS, :] = 0.0
    return Z


def _balanced_cv(F, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    return float(np.mean(cross_val_score(
        LogisticRegression(max_iter=1000, C=1.0), F, y, cv=3,
        scoring="balanced_accuracy")))


def const_fraction_per_sequence(X: np.ndarray) -> np.ndarray:
    """(N,T,J,C) -> (N,1) fraction of that sequence's coordinates with zero temporal sd."""
    sd = X.std(axis=1)                       # (N, J, C)
    return (sd <= 1e-9).mean(axis=(1, 2)).reshape(-1, 1)


def load_uiprmd(cache_dir: str) -> np.ndarray:
    return np.load(os.path.join(cache_dir, "uiprmd_sequences.npy"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-models", type=int, default=5)
    args = ap.parse_args()

    X_kimore, _, _ = load_corpus_with_labels("KIMORE")
    X_rehab, _, _ = load_corpus_with_labels("REHAB246")
    print(f"KIMORE {X_kimore.shape}  REHAB246 {X_rehab.shape}")

    rows = []
    for backbone, bdir in BACKBONES.items():
      ckpts = sorted(glob.glob(os.path.join(bdir, "fold_*", "best_model.pt")))[:args.n_models]
      if not ckpts:
        print(f"[skip] {backbone}: no checkpoints under {bdir}")
        continue
      print(f"\n### {backbone}: {len(ckpts)} checkpoints")

      for label, cache in UIPRMD_CACHES.items():
        if not os.path.isdir(cache):
            print(f"[skip] {label}: {cache} missing")
            continue
        X_ui = load_uiprmd(cache)

        cf = {"KIMORE": const_fraction_per_sequence(X_kimore).mean(),
              "REHAB246": const_fraction_per_sequence(X_rehab).mean(),
              "UIPRMD": const_fraction_per_sequence(X_ui).mean()}

        # Featureless control: can the constant-coordinate fraction alone identify the corpus?
        Fc = np.vstack([const_fraction_per_sequence(X_kimore),
                        const_fraction_per_sequence(X_rehab),
                        const_fraction_per_sequence(X_ui)])
        yc = np.concatenate([np.zeros(len(X_kimore)), np.ones(len(X_rehab)),
                             np.full(len(X_ui), 2)])
        acc_scalar = _balanced_cv(Fc, yc)

        for control in (False, True):
            Xk, Xr, Xu = ((_zero_padded(X_kimore), _zero_padded(X_rehab), _zero_padded(X_ui))
                          if control else (X_kimore, X_rehab, X_ui))
            a3, aku, akr = [], [], []
            for cp in ckpts:
                m = _rebuild(cp)
                fk_, fr_, fu_ = _feats(m, Xk), _feats(m, Xr), _feats(m, Xu)
                F = np.concatenate([fk_, fr_, fu_])
                y = np.concatenate([np.zeros(len(fk_)), np.ones(len(fr_)),
                                    np.full(len(fu_), 2)])
                a3.append(_balanced_cv(F, y))
                aku.append(_balanced_cv(np.concatenate([fk_, fu_]),
                                        np.concatenate([np.zeros(len(fk_)), np.ones(len(fu_))])))
                akr.append(_balanced_cv(np.concatenate([fk_, fr_]),
                                        np.concatenate([np.zeros(len(fk_)), np.ones(len(fr_))])))
            rows.append({
                "backbone": backbone,
                "uiprmd_geometry": label,
                "padded_joints_zeroed": control,
                "const_frac_UIPRMD": round(float(cf["UIPRMD"]), 4),
                "const_frac_REHAB246": round(float(cf["REHAB246"]), 4),
                "const_frac_KIMORE": round(float(cf["KIMORE"]), 4),
                "scalar_only_3way_acc": round(acc_scalar, 4),
                "probe_3way": round(float(np.mean(a3)), 4),
                "probe_3way_sd": round(float(np.std(a3)), 4),
                "probe_KIMORE_vs_UIPRMD": round(float(np.mean(aku)), 4),
                "probe_KIMORE_vs_REHAB246": round(float(np.mean(akr)), 4),
            })
            print(f"  {label:14s} zeroed={str(control):5s} | 3-way {np.mean(a3):.4f} "
                  f"| K-vs-U {np.mean(aku):.4f} | K-vs-R {np.mean(akr):.4f} "
                  f"| const-frac-only {acc_scalar:.4f}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"backbones": list(BACKBONES), "n_models": args.n_models,
                   "chance_3way": 1 / 3, "chance_2way": 0.5,
                   "rows": rows}, f, indent=2)
    print(f"\n-> {OUT}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
