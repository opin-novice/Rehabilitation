#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pareto_k2_eval.py  --  RESEARCH SANDBOX (isolated; writes only research_egnn/outputs/)
=======================================================================================
Fills the ONE missing, un-inventable number for tab:pareto: the k=2 ("2 dead") node-failure MAD
for the two sandbox arms (tuned EGNN and Canon-PCA+PCT). final_tables.json already has k=2 for the
paper models (EGRU 7.55, InvGRU 11.64, PCT 7.10); the sandbox never persisted it.

EVAL ONLY over existing checkpoints -- no training. Protocol is byte-identical to
sandbox_nodefail_sweep.per_seed: mode='hold', draws=3, seed=dd*977+i, subject_folds(k=5, seed),
split(test_fold=f, val_fold=(f+1)%5). Corruption is applied to the RAW skeleton, then each model's
own preprocessing runs (collate for EGNN; pca_canonicalize+batch_fixed_grid for canon).

  EGNN : seeds 0,1,2 x 5 folds  (egnn_s{seed}_f{fold}.pt)
  Canon: seed 0 x 5 folds       (canon_pct_s0_f{fold}.pt)  -- only seed 0 trained; labelled as such.

Writes result under key "pareto_k2" in research_egnn/outputs/sandbox_results.json.
"""

import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
for p in (_HERE, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import kimore_cde_data as kd                                  # noqa: E402
import block2_transforms as bt                               # noqa: E402
from train_cde import metrics                                # noqa: E402
from joint_failure import fail_joints                        # noqa: E402
from egnn_model import EGNNRecurrence                        # noqa: E402
from models_curvenet import PointCloudTransformerRegressor   # noqa: E402
from canonicalize import pca_canonicalize                    # noqa: E402

OUT = os.path.join(_HERE, "outputs")
assert os.path.abspath(OUT).replace("\\", "/").endswith("research_egnn/outputs"), \
    f"refusing to run: output dir {OUT} is not the sandbox"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCORE_MAX = kd.SCORE_MAX
N_FRAMES = 100          # canon arm trained with n_frames=100 (train_canon_s0.json)
K_EVAL = 2
DRAWS = 3


# ---------------- EGNN ----------------
@torch.no_grad()
def _pred_egnn(m, samples, bs=8):
    out = []
    for i in range(0, len(samples), bs):
        t, x, y, e, n = kd.collate(samples[i: i + bs], device=DEVICE)
        out.append(m(t, x, e, n).float().cpu().numpy())
    return np.concatenate(out)


def _load_egnn(seed, fold):
    m = EGNNRecurrence(n_scalar=32, n_vec=8, n_layers=4, egnn_hidden=64,
                       use_chiral=False, coord_clamp=None).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(OUT, f"egnn_s{seed}_f{fold}.pt"), map_location=DEVICE))
    return m.eval()


# ---------------- Canon ----------------
@torch.no_grad()
def _pred_canon(m, samples, bs=8):
    canon = [pca_canonicalize(s) for s in samples]
    out = []
    for i in range(0, len(canon), bs):
        x, y, e = bt.batch_fixed_grid(canon[i: i + bs], N_FRAMES, "linear", device=DEVICE)
        out.append(m(x, exercise_id=e).squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)


def _load_canon(seed, fold):
    m = PointCloudTransformerRegressor(
        seq_len=N_FRAMES, num_joints=kd.N_JOINTS, num_channels=3, dim=256,
        spatial_depth=6, temporal_depth=3, heads=4, dropout=0.1, k=10, num_exercises=5).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(OUT, f"canon_pct_s{seed}_f{fold}.pt"), map_location=DEVICE))
    return m.eval()


def eval_arm(name, loader, predictor, seeds):
    """Return per-seed [clean_mad, k2_mad] averaged over 5 folds (draws averaged at k=2)."""
    S = kd.load_all_exercises(max_len=150, verbose=False)
    per_seed_clean, per_seed_k2 = [], []
    for seed in seeds:
        folds = kd.subject_folds(S, k=5, seed=seed)
        cleans, k2s = [], []
        for f in range(5):
            _, _, te = kd.split(S, folds, test_fold=f, val_fold=(f + 1) % 5)
            y = np.array([s["y"] for s in te])
            m = loader(seed, f)
            cleans.append(metrics(predictor(m, te), y)["MAD"])
            draw_mads = []
            for dd in range(DRAWS):
                failed = [fail_joints(s, K_EVAL, "hold", seed=dd * 977 + i) for i, s in enumerate(te)]
                draw_mads.append(metrics(predictor(m, failed), y)["MAD"])
            k2s.append(float(np.mean(draw_mads)))
            del m
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        per_seed_clean.append(float(np.mean(cleans)))
        per_seed_k2.append(float(np.mean(k2s)))
        print(f"  {name} seed {seed}: clean {per_seed_clean[-1]:.3f}  k=2 {per_seed_k2[-1]:.3f}")
    return {
        "seeds": list(seeds),
        "clean_mad_mean": float(np.mean(per_seed_clean)),
        "k2_mad_mean": float(np.mean(per_seed_k2)),
        "k2_mad_std": float(np.std(per_seed_k2, ddof=1)) if len(seeds) > 1 else 0.0,
        "per_seed_clean": per_seed_clean,
        "per_seed_k2": per_seed_k2,
    }


def main():
    print(f"[pareto-k2] device={DEVICE}  k={K_EVAL} hold draws={DRAWS}")
    res = {}
    print("[pareto-k2] EGNN (seeds 0,1,2 x 5 folds) ...")
    res["EGNN"] = eval_arm("EGNN", _load_egnn, _pred_egnn, seeds=(0, 1, 2))
    print("[pareto-k2] Canon-PCA+PCT (seed 0 x 5 folds) ...")
    res["Canon-PCA+PCT"] = eval_arm("Canon", _load_canon, _pred_canon, seeds=(0,))

    path = os.path.join(OUT, "sandbox_results.json")
    blob = json.load(open(path)) if os.path.exists(path) else {}
    blob["pareto_k2"] = {"k": K_EVAL, "mode": "hold", "draws": DRAWS, "arms": res,
                         "paper_reference_k2": {"egru": 7.550, "invgru": 11.642, "pct": 7.097}}
    with open(path, "w") as fh:
        json.dump(blob, fh, indent=2)
    print(f"[pareto-k2] wrote key 'pareto_k2' -> {path}")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
