#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
block3_baselines.py
===================
Block-3 (cross-viewpoint transfer) tests for TCN, ST-GCN, and Ridge baselines.

Rotates test skeletons about the vertical axis (0° → 180° azimuth) and measures:
  - Degradation: |s_baseline(rotated) - s_baseline(clean)| per model (own clean baseline)
  - Accuracy:   MAD(s_baseline(rotated), y) vs clinical labels

Both metrics required: degradation alone is insufficient (a deaf model has zero degradation).

Run:
  python src/block3_baselines.py --model tcn --fold 0
  python src/block3_baselines.py --model stgcn --cv
  python src/block3_baselines.py --model ridge --cv
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimore_cde_data as kd
import block2_transforms as bt
import invariant_controls as ic
from models_stgcn import TCNRegressor, STGCNRegressor
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.preprocessing import StandardScaler
from train_cde import metrics

ANGLES = [0, 15, 30, 45, 60, 90, 120, 150, 180]
SCORE_MAX = kd.SCORE_MAX


def extract_ridge_features(samples):
    """Extract per-joint statistics for Ridge from KIMORE samples."""
    features = []
    for s in samples:
        # invariant_series returns (L, n_feat) of invariant features
        inv_feats = ic.invariant_series(s)  # (L, n_feat)
        # Aggregate: mean over time
        agg_feats = inv_feats.mean(axis=0)  # (n_feat,)
        features.append(agg_feats)
    return np.array(features)  # (N, n_feat)


@torch.no_grad()
def predict_tcn(model, samples, device, bs=8):
    """Predict using TCN model."""
    model.eval()
    out = []
    for i in range(0, len(samples), bs):
        chunk = samples[i: i + bs]
        t, x, y, e, n = kd.collate(chunk, device=device, dtype=torch.float32)
        p = model(x, exercise_id=e).squeeze(-1)
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict_stgcn(model, samples, device, bs=8):
    """Predict using ST-GCN model."""
    model.eval()
    out = []
    for i in range(0, len(samples), bs):
        chunk = samples[i: i + bs]
        t, x, y, e, n = kd.collate(chunk, device=device, dtype=torch.float32)
        p = model(x, exercise_id=e).squeeze(-1)
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)


def predict_ridge(ridge_model, scaler, samples, device):
    """Predict using Ridge regression on hand-crafted features."""
    X = extract_ridge_features(samples)
    X_scaled = scaler.transform(X)
    return ridge_model.predict(X_scaled)


def run_fold(f, samples, folds, args, device):
    """Run Block-3 rotation test for a single fold."""
    val_f = (f + 1) % args.folds
    tr, va, te = kd.split(samples, folds, test_fold=f, val_fold=val_f)
    y = np.array([s["y"] for s in te])
    floor = kd.exercise_mean_floor(tr, te)
    floor_mad = metrics(floor, y)["MAD"]

    rows = []

    if args.model in ["tcn", "stgcn"]:
        # Load torch model
        model_cls = TCNRegressor if args.model == "tcn" else STGCNRegressor
        model = model_cls(
            seq_len=args.seq_len,
            num_joints=kd.N_JOINTS,
            num_channels=3,
            num_exercises=5,
        ).to(device)

        ckpt_name = f"{args.model}_wacv_s{args.model_seed}_f{f}.pt"
        ckpt_path = os.path.join(args.ckpt, ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        predict_fn = predict_tcn if args.model == "tcn" else predict_stgcn

        # Clean reference
        s0 = predict_fn(model, te, device)
        clean_mad = metrics(s0, y)["MAD"]
        has_signal = bool(clean_mad < floor_mad)

        if not has_signal:
            print(f"  [WARN] fold {f} {args.model.upper()}: clean MAD {clean_mad:.3f} >= floor "
                  f"{floor_mad:.3f} (NO-SIGNAL)")

        # Rotation sweep
        for deg in ANGLES:
            rot = [bt.rotate_sample(s, deg) for s in te]
            s_rot = predict_fn(model, rot, device)
            rec = {
                "fold": f, "angle": deg, "floor_mad": floor_mad,
                "has_signal": has_signal, "clean_mad": clean_mad,
                "mad": metrics(s_rot, y)["MAD"],
                "degradation": float(np.mean(np.abs(s_rot - s0)) * SCORE_MAX),
            }
            rows.append(rec)
            print(f"  fold {f} angle {deg:3d}deg   MAD {rec['mad']:6.3f} "
                  f"(degradation {rec['degradation']:6.3f})   floor {floor_mad:6.3f}")

    elif args.model == "ridge":
        # Ridge regression on hand-crafted features (CPU only)
        X_tr = extract_ridge_features(tr)
        y_tr = np.array([s["y"] for s in tr])

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)

        ridge = SklearnRidge(alpha=1.0)
        ridge.fit(X_tr_scaled, y_tr)

        # Clean reference
        s0 = predict_ridge(ridge, scaler, te, device)
        clean_mad = metrics(s0, y)["MAD"]
        has_signal = bool(clean_mad < floor_mad)

        if not has_signal:
            print(f"  [WARN] fold {f} Ridge: clean MAD {clean_mad:.3f} >= floor "
                  f"{floor_mad:.3f} (NO-SIGNAL)")

        # Rotation sweep
        for deg in ANGLES:
            rot = [bt.rotate_sample(s, deg) for s in te]
            s_rot = predict_ridge(ridge, scaler, rot, device)
            rec = {
                "fold": f, "angle": deg, "floor_mad": floor_mad,
                "has_signal": has_signal, "clean_mad": clean_mad,
                "mad": metrics(s_rot, y)["MAD"],
                "degradation": float(np.mean(np.abs(s_rot - s0)) * SCORE_MAX),
            }
            rows.append(rec)
            print(f"  fold {f} angle {deg:3d}deg   MAD {rec['mad']:6.3f} "
                  f"(degradation {rec['degradation']:6.3f})   floor {floor_mad:6.3f}")

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tcn", "stgcn", "ridge"], required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cv", action="store_true", help="run all 5 folds")
    ap.add_argument("--model-seed", type=int, default=0,
                    help="which training seed's checkpoints to use")
    ap.add_argument("--seq-len", type=int, default=100)
    ap.add_argument("--ckpt", type=str, default="outputs/baselines")
    ap.add_argument("--out", type=str, default="outputs/cde_block2")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = kd.load_all_exercises(max_len=args.seq_len, verbose=False)
    folds = kd.subject_folds(samples, k=args.folds, seed=args.model_seed)

    fold_ids = range(args.folds) if args.cv else [args.fold]

    print(f"\n{'='*78}")
    print(f"BLOCK 3 — {args.model.upper()} BASELINE (cross-viewpoint transfer)")
    print(f"{'='*78}")

    rows = []
    for f in fold_ids:
        print(f"\nFold {f}:")
        rows += run_fold(f, samples, folds, args, device)

    # Aggregate
    print(f"\n{'-'*78}\nAGGREGATE over {len(fold_ids)} folds\n{'-'*78}")
    hdr = f"  {'angle':>7s} | {'MAD':>9s} {'degradation':>14s} | {'floor':>7s}"
    print(hdr)
    for angle in ANGLES:
        sel = [r for r in rows if r["angle"] == angle]
        if not sel:
            continue
        m = np.mean([r["mad"] for r in sel])
        d = np.mean([r["degradation"] for r in sel])
        fl = np.mean([r["floor_mad"] for r in sel])
        print(f"  {angle:7d} | {m:9.3f} {d:14.3f} | {fl:7.3f}")

    # Signal gate
    ns = sorted({r["fold"] for r in rows if not r["has_signal"]})
    if ns:
        print(f"\n  [WARN] SIGNAL GATE FAILED on folds {ns}")
    else:
        print(f"\n  signal gate: PASS on all {len(fold_ids)} folds")

    # Save results
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"block3_{args.model}_s{args.model_seed}_results.json")
    with open(out_path, "w") as fh:
        json.dump({"args": vars(args), "rows": rows}, fh, indent=2)
    print(f"\nSaved results: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
