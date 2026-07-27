#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sandbox_eval.py  --  RESEARCH SANDBOX (NOT for the paper)
=========================================================
Evaluates the isolated arms on the three axes, using imported pure functions (no edits to the
hard-coded harnesses). Reads only from research_egnn/outputs/ (own checkpoints) and, READ-ONLY, the
paper's reported aggregates for a side-by-side. Writes research_egnn/outputs/sandbox_results.json.

Axes:
  accuracy    pooled clean MAD.
  viewpoint   rotate test skeletons about gravity (bt.rotate_sample) over the paper's angle sweep;
              report aggregate MAD and worst PER-SEQUENCE degradation (mean|pred-clean|*50).
  node-fail   joint_failure.fail_joints ('hold'), MAD lost over k dead joints.
  cert        (EGNN only) rotate the input, confirm the score moves only at machine precision.

    python research_egnn/sandbox_eval.py
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
from models_curvenet import build_pct_for_checkpoint         # noqa: E402
from joint_failure import fail_joints, hash_samples          # noqa: E402
from egnn_model import EGNNRecurrence                        # noqa: E402
from canonicalize import pca_canonicalize                    # noqa: E402

OUT = os.path.join(_HERE, "outputs")
SCORE_MAX = kd.SCORE_MAX
ANGLES = [0, 15, 30, 45, 60, 90, 120, 150, 180]
NF_LEVELS = [0, 1, 2, 4, 8]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paper's reported 3-seed aggregates (from results.tex), for a labelled side-by-side ONLY.
PAPER = {
    "EGRU (e3nn, ours)": {"clean": 6.73, "view_degr": 9e-6, "nf_lost": 3.76, "params": "0.66M"},
    "InvariantGRU":      {"clean": 6.31, "view_degr": 0.0,   "nf_lost": 12.05, "params": "0.21M"},
    "PCT (baseline)":    {"clean": 6.47, "view_degr": 9.42,  "nf_lost": 2.27, "params": "4.91M"},
}


def load_egnn(f):
    m = EGNNRecurrence(n_scalar=32, n_vec=8, n_layers=4, use_chiral=False).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(OUT, f"egnn_s0_f{f}.pt"), map_location=DEVICE))
    return m.eval()


def load_canon(f):
    sd = torch.load(os.path.join(OUT, f"canon_pct_s0_f{f}.pt"), map_location=DEVICE)
    m = build_pct_for_checkpoint(
        sd, seq_len=100, num_joints=kd.N_JOINTS, num_channels=3, dim=256, spatial_depth=6,
        temporal_depth=3, heads=4, dropout=0.1, k=10).to(DEVICE)
    m.load_state_dict(sd)
    return m.eval()


@torch.no_grad()
def pred_egnn(m, samples, bs=8):
    out = []
    for i in range(0, len(samples), bs):
        t, x, y, e, n = kd.collate(samples[i: i + bs], device=DEVICE)
        out.append(m(t, x, e, n).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def pred_canon(m, samples, bs=8):
    canon = [pca_canonicalize(s) for s in samples]
    out = []
    for i in range(0, len(canon), bs):
        x, y, e = bt.batch_fixed_grid(canon[i: i + bs], 100, "linear", device=DEVICE)
        out.append(m(x, exercise_id=e).squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)


def eval_arm(name, load, pred, S, folds, args_folds=5):
    """Returns aggregate dict for one arm over all folds."""
    accs, view_worst_degr, view_worst_mad, nf_lost, cert = [], [], [], [], []
    for f in range(args_folds):
        tr, va, te = kd.split(S, folds, test_fold=f, val_fold=(f + 1) % args_folds)
        y = np.array([s["y"] for s in te])
        m = load(f)
        clean = pred(m, te)
        accs.append(metrics(clean, y)["MAD"])

        # viewpoint sweep
        degrs, mads = [], []
        for deg in ANGLES:
            rot = [bt.rotate_sample(s, deg) for s in te]
            p = pred(m, rot)
            mads.append(metrics(p, y)["MAD"])
            degrs.append(float(np.mean(np.abs(p - clean)) * SCORE_MAX))
        view_worst_degr.append(max(degrs))
        view_worst_mad.append(max(mads))
        # invariance certificate: 90deg move, max per-sample score shift (in 0-50 units)
        rot90 = [bt.rotate_sample(s, 90) for s in te]
        cert.append(float(np.max(np.abs(pred(m, rot90) - clean)) * SCORE_MAX))

        # node-failure (hold), MAD lost 0->8
        nf = {}
        for k in NF_LEVELS:
            if k == 0:
                nf[k] = metrics(clean, y)["MAD"]
                continue
            draws = []
            for d in range(3):
                cor = [fail_joints(s, k, "hold", seed=d * 977 + i) for i, s in enumerate(te)]
                draws.append(metrics(pred(m, cor), y)["MAD"])
            nf[k] = float(np.mean(draws))
        nf_lost.append(nf[max(NF_LEVELS)] - nf[0])
        del m
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return {
        "clean_mad": float(np.mean(accs)),
        "view_worst_degr": float(np.mean(view_worst_degr)),
        "view_worst_mad": float(np.mean(view_worst_mad)),
        "invariance_cert_maxshift": float(np.mean(cert)),
        "nf_lost_0to8": float(np.mean(nf_lost)),
    }


def main():
    S = kd.load_all_exercises(max_len=150, verbose=False)
    folds = kd.subject_folds(S, k=5, seed=0)

    have_egnn = os.path.exists(os.path.join(OUT, "egnn_s0_f0.pt"))
    have_canon = os.path.exists(os.path.join(OUT, "canon_pct_s0_f0.pt"))
    res = {}
    if have_egnn:
        print("evaluating EGNN ...")
        res["EGNN (E(n)-equiv, sandbox)"] = eval_arm("egnn", load_egnn, pred_egnn, S, folds)
    if have_canon:
        print("evaluating canonicalizer+PCT ...")
        res["Canon-PCA + PCT (sandbox)"] = eval_arm("canon", load_canon, pred_canon, S, folds)

    print(f"\n{'='*90}\nSANDBOX RESULTS  (1 seed x 5 folds; NOT for the paper)\n{'='*90}")
    hdr = f"  {'arm':<28s} {'clean':>7s} {'view MAD':>9s} {'view degr':>10s} {'cert':>9s} {'nf lost':>8s}"
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    for k, r in res.items():
        print(f"  {k:<28s} {r['clean_mad']:7.3f} {r['view_worst_mad']:9.3f} "
              f"{r['view_worst_degr']:10.4f} {r['invariance_cert_maxshift']:9.2e} {r['nf_lost_0to8']:+8.2f}")
    print("  " + "-" * (len(hdr) - 2))
    for k, r in PAPER.items():
        print(f"  {k+' [paper 3-seed]':<28s} {r['clean']:7.3f} {'--':>9s} "
              f"{r['view_degr']:10.4f} {'--':>9s} {r['nf_lost']:+8.2f}")
    print(f"{'='*90}")
    print("  view MAD/degr: worst over the 0-180 azimuth sweep. cert: max per-sample score shift")
    print("  under a 90 rotation (0-50 units). nf lost: MAD gained over 8 dead joints (hold).")

    with open(os.path.join(OUT, "sandbox_results.json"), "w") as fh:
        json.dump({"sandbox": res, "paper_reference": PAPER,
                   "angles": ANGLES, "nf_levels": NF_LEVELS}, fh, indent=2)
    print(f"\nwrote {os.path.join(OUT, 'sandbox_results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
