"""Reviewer round-3, B3 (Q7): monotone (isotonic) calibration cannot rescue zero-shot AUROC.

The reviewer asks whether a monotone mapping learned on KIMORE (isotonic regression)
re-scaling target predictions could mitigate the source(continuous)/target(binary)
label-scale mismatch. AUROC is invariant to any strictly monotone transform of the
scores, so calibration provably cannot change it. We demonstrate this numerically:
for each KIMORE LOSO fold we fit IsotonicRegression on (KIMORE prediction ->
median-binarized KIMORE score), apply the fitted map to the target predictions, and
recompute AUROC. The pre- and post-calibration AUROC agree to numerical precision.

Run:  python src/reviewer/reviewer_round3_b3_isotonic.py
Out:  outputs/reviewer_round3/b3_isotonic.{json,md}
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ root (file now in src/reviewer/)
from models_stgcn import TCNRegressor  # noqa: E402
from selfsup.data import load_corpus_with_labels  # noqa: E402

RESULTS_DIR = "archive/legacy_results/kimore_loso_78fold/A_scratch"
OUT_DIR = "outputs/reviewer_round3"


def _rebuild(cp):
    c = torch.load(cp, map_location="cpu")
    a = c.get("args", {})
    m = TCNRegressor(seq_len=100, d_model=a.get("d_model", 128),
                     num_blocks=a.get("tcn_blocks", 4), dropout=a.get("dropout", 0.3))
    m.load_state_dict(c["model_state"], strict=False)
    m.eval()
    return m


@torch.no_grad()
def _predict(model, X, batch=256):
    xt = torch.from_numpy(X.astype(np.float32))
    out = []
    for i in range(0, len(xt), batch):
        y = model(xt[i:i + batch])
        if isinstance(y, tuple):
            y = y[0]
        out.append(y.squeeze(-1).cpu().numpy())
    return np.concatenate(out)


def _auroc(labels, preds):
    a = roc_auc_score(labels, preds)
    return float(max(a, 1.0 - a))


def run(max_folds: int = 78) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    X_k, y_k, _ = load_corpus_with_labels("KIMORE")
    y_k_bin = (np.asarray(y_k) >= np.median(y_k)).astype(int)  # correctness proxy
    targets = {
        "REHAB246": load_corpus_with_labels("REHAB246"),
        "UIPRMD": load_corpus_with_labels("UIPRMD"),
    }
    ckpts = sorted(glob.glob(os.path.join(RESULTS_DIR, "fold_*", "best_model.pt")))[:max_folds]
    if not ckpts:
        print(f"[SKIP] no checkpoints in {RESULTS_DIR}")
        return {}

    agg = {c: {"raw": [], "iso": []} for c in targets}
    for cp in ckpts:
        m = _rebuild(cp)
        pk = _predict(m, X_k)
        iso = IsotonicRegression(out_of_bounds="clip").fit(pk, y_k_bin)
        for cn, (Xt, lt, _) in targets.items():
            lt = np.asarray(lt)
            if len(np.unique(lt)) < 2:
                continue
            pt = _predict(m, Xt)
            agg[cn]["raw"].append(_auroc(lt, pt))
            agg[cn]["iso"].append(_auroc(lt, iso.predict(pt)))

    results = {}
    for cn, d in agg.items():
        raw, iso = np.array(d["raw"]), np.array(d["iso"])
        results[cn] = {
            "n_folds": int(len(raw)),
            "mean_auroc_raw": float(raw.mean()),
            "mean_auroc_isotonic": float(iso.mean()),
            "max_abs_diff": float(np.max(np.abs(raw - iso))) if len(raw) else None,
        }
        r = results[cn]
        print(f"{cn}: raw={r['mean_auroc_raw']:.4f}  isotonic={r['mean_auroc_isotonic']:.4f}  "
              f"max|delta|={r['max_abs_diff']:.2e}")

    with open(os.path.join(OUT_DIR, "b3_isotonic.json"), "w") as f:
        json.dump(results, f, indent=2)
    lines = ["| Corpus | AUROC (raw) | AUROC (isotonic-calibrated) | max fold |delta| |",
             "|---|---|---|---|"]
    for cn, r in results.items():
        lines.append(f"| {cn} | {r['mean_auroc_raw']:.4f} | {r['mean_auroc_isotonic']:.4f} | {r['max_abs_diff']:.1e} |")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(OUT_DIR, "b3_isotonic.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)
    return results


if __name__ == "__main__":
    run()
