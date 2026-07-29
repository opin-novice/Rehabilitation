"""Re-run the entire UI-PRMD zero-shot column under corrected (FK) geometry.

WHY
---
paper/archive/manuscript.tex's UI-PRMD numbers were computed from a cache built by reading
UI-PRMD's parent-relative bone offsets as world coordinates: 57.3% of coordinates never
changed (REHAB246: 0.000). Verified by reproduction -- the raw cache returns the published
naive baseline 0.538 exactly, while the forward-kinematics cache returns 0.527.

Everything downstream of that cache therefore needs recomputing, not editing. This script
re-runs, for both geometries so every delta is auditable:

  * zero-shot AUROC / rank-Spearman / pred_SD for conditions A-E (77 LOSO folds)
  * the same for the all-corpora pretraining pool (UI-PRMD is *inside* that pool)
  * robustness rows: ST-GCN backbone, relative-joint-vector input, per-sequence z-scoring
  * naive kinematic baseline, incl. the shared-joints and z-scored sensitivity variants
  * per-exercise AUROC breakdown

Model weights are NOT retrained here: every checkpoint is frozen and only inference is
re-run. The one thing this cannot fix by inference alone is the all-corpora *pretraining*,
whose encoder saw the degenerate UI-PRMD sequences as unlabeled data; that row is reported
with an explicit caveat (see --note in the output JSON).

Usage:
    python src/rerun_uiprmd_column.py [--geometries fk raw] [--corpus UIPRMD]
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
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/
from selfsup.zeroshot_eval import _rebuild_model, _predict, DEGENERACY_PRED_SD  # noqa: E402
from selfsup.naive_baseline import naive_auroc  # noqa: E402
from selfsup.features import to_relative_joint_vectors, per_sequence_zscore  # noqa: E402

GEOM_CACHE = {
    "fk": "outputs/validity_uiprmd",
    "raw": "outputs/validity_uiprmd_raw",
    "fk_corrected": "outputs/validity_uiprmd_corrected",
}

MAIN_DIR = "archive/legacy_results/kimore_loso_78fold"
ALLCORP_DIR = "outputs/ssl_results_allcorpora"
CONDITIONS = [
    ("A_scratch", "Scratch"), ("B_contrastive_lp", "Contrastive LP"),
    ("C_contrastive_ft", "Contrastive FT"), ("D_masked_lp", "Masked LP"),
    ("E_masked_ft", "Masked FT"),
]
# Robustness rows. The bonevec and perseq models were TRAINED on transformed inputs, so
# their evaluation must apply the same transform to the target corpus -- exactly as
# reviewer_round3_c2.evaluate and reviewer_round3_q10_norm.evaluate do. Feeding them raw
# coordinates silently scores a model on a distribution it never saw.
ROBUSTNESS = [
    ("archive/legacy_results/kimore_loso_78fold_stgcn", "ST-GCN backbone", None),
    ("archive/legacy_results/kimore_loso_78fold_bonevec", "Rel.-joint input", "bonevec"),
    ("archive/legacy_results/kimore_loso_78fold_perseq", "Per-seq. z-score", "perseq"),
]

INPUT_TRANSFORMS = {
    "bonevec": lambda X: to_relative_joint_vectors(X),
    "perseq": lambda X: per_sequence_zscore(X),
}
# Naive-baseline sensitivity row. The paper excludes each corpus's OWN artefact joints:
# UI-PRMD's three zero-padded slots (22,23,24), and REHAB246's four duplicated permutation
# targets (thumb=wrist). Joints 7 and 11 are duplicated in REHAB246 but are genuine joints
# in UI-PRMD, so excluding them here would delete real signal rather than an artefact.
UIPRMD_PADDED_JOINTS = [22, 23, 24]
OUT_JSON = "outputs/novelty/uiprmd_column_rerun.json"
OUT_MD = "outputs/novelty/uiprmd_column_rerun.md"


def load_cache(geom: str):
    base = GEOM_CACHE[geom]
    X = np.load(os.path.join(base, "uiprmd_sequences.npy")).astype(np.float32)
    man = pd.read_csv(os.path.join(base, "uiprmd_manifest.csv"))
    return X, man["correct_label"].values, man


DEVICE = torch.device("cpu")


@torch.no_grad()
def _predict_dev(model, X: np.ndarray, batch: int = 512) -> np.ndarray:
    """Same maths as selfsup.zeroshot_eval._predict, but honours DEVICE."""
    model = model.to(DEVICE)
    xt = torch.from_numpy(X.astype(np.float32))
    out = []
    for i in range(0, len(xt), batch):
        y = model(xt[i:i + batch].to(DEVICE))
        if isinstance(y, tuple):
            y = y[0]
        out.append(y.squeeze(-1).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def eval_dir(cond_dir: str, X: np.ndarray, labels: np.ndarray, max_folds: int | None = None):
    """Frozen-checkpoint inference over every fold in cond_dir."""
    cps = sorted(glob.glob(os.path.join(cond_dir, "fold_*", "best_model.pt")))
    if max_folds:
        cps = cps[:max_folds]
    if not cps:
        return None
    aurocs, rhos, sds, per_fold_preds = [], [], [], []
    for cp in cps:
        preds = _predict_dev(_rebuild_model(torch.load(cp, map_location="cpu", weights_only=False)), X)
        per_fold_preds.append(preds)
        sds.append(float(np.std(preds)))
        a = roc_auc_score(labels, preds)
        aurocs.append(float(max(a, 1.0 - a)))
        rhos.append(float(spearmanr(preds, labels).correlation))
    mean_sd = float(np.mean(sds))
    return {
        "n_folds": len(cps),
        "mean_auroc": float(np.mean(aurocs)),
        "sd_auroc": float(np.std(aurocs)),
        "mean_rank_spearman": float(np.nanmean(rhos)),
        "mean_pred_sd": mean_sd,
        "degenerate": bool(mean_sd < DEGENERACY_PRED_SD),
        "_mean_preds": np.mean(per_fold_preds, axis=0),
    }


def per_exercise(preds: np.ndarray, labels: np.ndarray, man: pd.DataFrame):
    out = {}
    for ex, idx in man.groupby("exercise_id").groups.items():
        i = np.asarray(list(idx))
        if len(np.unique(labels[i])) < 2:
            continue
        a = roc_auc_score(labels[i], preds[i])
        out[int(ex)] = round(float(max(a, 1.0 - a)), 4)
    return out


def naive_block(X: np.ndarray, labels: np.ndarray):
    mask = np.ones(X.shape[2], dtype=bool)
    mask[UIPRMD_PADDED_JOINTS] = False
    return {
        "naive_auroc": round(float(naive_auroc(X, labels)), 4),
        "naive_shared_joints": round(float(naive_auroc(X, labels, joint_mask=mask)), 4),
        "naive_zscored": round(float(naive_auroc(X, labels, zscore=True)), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometries", nargs="+", default=["raw", "fk"],
                    choices=list(GEOM_CACHE))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    global DEVICE
    DEVICE = torch.device(args.device)
    print(f"device={DEVICE}")

    results = {}
    for geom in args.geometries:
        if not os.path.isdir(GEOM_CACHE[geom]):
            print(f"[skip] {geom}: {GEOM_CACHE[geom]} missing")
            continue
        X, labels, man = load_cache(geom)
        const = float((X.std(axis=1) <= 1e-9).mean())
        print(f"\n===== geometry={geom}  X={X.shape}  const-coord frac={const:.4f} =====")
        g = {"cache": GEOM_CACHE[geom], "const_coord_fraction": round(const, 4),
             "naive": naive_block(X, labels), "conditions": {}, "allcorpora": {},
             "robustness": {}, "per_exercise": {}}
        print(f"  naive: {g['naive']}")

        for cond, label in CONDITIONS:
            r = eval_dir(os.path.join(MAIN_DIR, cond), X, labels)
            if r is None:
                continue
            g["per_exercise"][label] = per_exercise(r.pop("_mean_preds"), labels, man)
            g["conditions"][label] = r
            print(f"  [main]  {label:16s} AUROC={r['mean_auroc']:.4f}"
                  f" (sd {r['sd_auroc']:.3f}) rho={r['mean_rank_spearman']:+.3f}"
                  f" predSD={r['mean_pred_sd']:.3f}{' DEGENERATE' if r['degenerate'] else ''}")

        for cond, label in CONDITIONS:
            r = eval_dir(os.path.join(ALLCORP_DIR, cond), X, labels)
            if r is None:
                continue
            r.pop("_mean_preds", None)
            g["allcorpora"][label] = r
            print(f"  [allc]  {label:16s} AUROC={r['mean_auroc']:.4f}"
                  f" predSD={r['mean_pred_sd']:.3f}")

        for d, label, tf in ROBUSTNESS:
            Xr = INPUT_TRANSFORMS[tf](X) if tf else X
            r = eval_dir(d, Xr, labels)
            if r is None:
                print(f"  [robu]  {label:16s} MISSING {d}")
                continue
            r.pop("_mean_preds", None)
            r["input_transform"] = tf or "raw_coords"
            g["robustness"][label] = r
            print(f"  [robu]  {label:16s} AUROC={r['mean_auroc']:.4f}"
                  f" (sd {r['sd_auroc']:.3f}) predSD={r['mean_pred_sd']:.3f}"
                  f"{' DEGENERATE' if r['degenerate'] else ''}")

        results[geom] = g

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "note": ("Frozen-checkpoint inference only. The all-corpora rows re-score models "
                     "whose PRETRAINING pool contained the degenerate UI-PRMD sequences; "
                     "re-pretraining is required to fully clear that row."),
            "results": results,
        }, f, indent=2)

    # markdown delta table
    lines = ["# UI-PRMD column: raw (published) vs FK (repaired)", "",
             "| row | raw | fk | delta |", "|---|---|---|---|"]
    if "raw" in results and "fk" in results:
        r, k = results["raw"], results["fk"]
        lines.append(f"| const-coord fraction | {r['const_coord_fraction']} | "
                     f"{k['const_coord_fraction']} | "
                     f"{k['const_coord_fraction'] - r['const_coord_fraction']:+.4f} |")
        for nk in r["naive"]:
            lines.append(f"| naive: {nk} | {r['naive'][nk]:.4f} | {k['naive'][nk]:.4f} | "
                         f"{k['naive'][nk] - r['naive'][nk]:+.4f} |")
        for sect in ("conditions", "allcorpora", "robustness"):
            for name in r.get(sect, {}):
                if name not in k.get(sect, {}):
                    continue
                a, b = r[sect][name]["mean_auroc"], k[sect][name]["mean_auroc"]
                lines.append(f"| {sect}: {name} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n-> {OUT_JSON}\n-> {OUT_MD}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
