#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_stgcn_wacv.py
===================
Spatial-Temporal Graph CNN (Yan et al., 2018) trained on the EXACT SAME subject-disjoint folds
as EGRU/PCT.

ST-GCN exploits the Kinect v2 skeleton graph topology via graph convolutions but has no rotation
invariance by construction. It's a domain-standard baseline for skeleton-action recognition.

Expected to fail Block 3 similarly to non-spatial models like TCN (degradation ~10 MAD),
demonstrating that graph structure alone doesn't confer viewpoint robustness.

Run:
  python src/train_stgcn_wacv.py --fold 0
  python src/train_stgcn_wacv.py --cv
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimore_cde_data as kd
from models_stgcn import STGCNRegressor
from train_cde import metrics, fmt, floor_metrics


@torch.no_grad()
def evaluate(model, samples, device, batch_size=8):
    """Evaluate ST-GCN on KIMORE samples."""
    model.eval()
    preds, trues = [], []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i: i + batch_size]
        # ST-GCN input contract: [B, T, J, C]
        t, x, y, e, n = kd.collate(chunk, device=device, dtype=torch.float32)
        p = model(x, exercise_id=e).squeeze(-1)
        preds.append(p.float().cpu().numpy())
        trues.append(y.float().cpu().numpy())
    return metrics(np.concatenate(preds), np.concatenate(trues))


def train_fold(train_s, val_s, test_s, args, device, verbose=True):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = STGCNRegressor(
        seq_len=args.seq_len,
        num_joints=kd.N_JOINTS,
        num_channels=3,
        base_channels=args.base_channels,
        dropout=args.dropout,
        num_exercises=(5 if args.pooled else 0),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    huber = nn.HuberLoss(delta=args.huber_delta)

    floor_val = floor_metrics(train_s, val_s, args.pooled)
    floor_test = floor_metrics(train_s, test_s, args.pooled)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"  ST-GCN baseline: {n_par:,} params | seq_len={args.seq_len} "
              f"base_channels={args.base_channels}")
        print(f"  train {len(train_s)} / val {len(val_s)} / test {len(test_s)} subjects")
        print(f"  mean-predictor floor:  val {fmt(floor_val)}")

    best_val, best_state, best_ep = math.inf, None, -1
    train_times = []

    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        ep_start = time.time()

        # Shuffle and batch
        perm = torch.randperm(len(train_s))
        for i in range(0, len(train_s), args.batch_size):
            chunk = [train_s[j] for j in perm[i: i + args.batch_size]]
            t, x, y, e, n_real = kd.collate(chunk, device=device, dtype=torch.float32)

            opt.zero_grad()
            logits = model(x, exercise_id=e).squeeze(-1)
            loss = huber(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot += loss.item() * len(chunk)
            n += len(chunk)

        sched.step()
        ep_time = time.time() - ep_start
        train_times.append(ep_time)

        # Validation
        with torch.no_grad():
            val_metrics = evaluate(model, val_s, device)
            val_loss = val_metrics["MAD"]

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_ep = ep

        if (ep + 1) % args.log_interval == 0 and verbose:
            print(f"  ep {ep:3d}  train_loss {tot / n:7.4f}  val_MAD {val_loss:7.4f}  "
                  f"best {best_val:7.4f} (ep {best_ep})")

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        test_metrics = evaluate(model, test_s, device)

    if verbose:
        print(f"  best_ep {best_ep} / {args.epochs} validation MAD = {best_val:.4f}")
        print(f"  test metrics: {fmt(test_metrics)}")
        print(f"  mean epoch time: {np.mean(train_times):.2f}s")

    return model, best_state, test_metrics, floor_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0, help="which fold to train (0-4)")
    ap.add_argument("--cv", action="store_true", help="cross-validation: run all 5 folds")
    ap.add_argument("--pooled", action="store_true", default=True,
                    help="pool all exercises (default True)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--seq-len", type=int, default=100,
                    help="max sequence length (via uniform index subsampling)")
    ap.add_argument("--base-channels", type=int, default=32,
                    help="ST-GCN base channel count")
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="training seed (for reproducibility)")
    ap.add_argument("--model-seed", type=int, default=0,
                    help="which seed's fold partition to use (must match Block-3 eval seed)")
    ap.add_argument("--ckpt", type=str, default="outputs/baselines",
                    help="checkpoint save directory")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load all KIMORE exercises, build subject-disjoint folds
    samples = kd.load_all_exercises(max_len=args.seq_len, verbose=False)
    # CRITICAL: use model_seed for fold partition consistency with Block-3 evals
    folds = kd.subject_folds(samples, k=args.folds, seed=args.model_seed)

    fold_ids = range(args.folds) if args.cv else [args.fold]
    os.makedirs(args.ckpt, exist_ok=True)

    print(f"\n{'='*78}")
    print(f"ST-GCN (Spatial-Temporal Graph CNN) — WACV baseline training")
    print(f"{'='*78}")
    print(f"Fold assignment seed: {args.model_seed}")
    print(f"Training seed: {args.seed}")
    print(f"Folds: {fold_ids}")

    all_results = {}

    for f in fold_ids:
        print(f"\n{'-'*78}\nFOLD {f}\n{'-'*78}")

        val_f = (f + 1) % args.folds
        tr, va, te = kd.split(samples, folds, test_fold=f, val_fold=val_f)

        model, best_state, test_metrics, floor_test = train_fold(
            tr, va, te, args, device, verbose=True
        )

        # Save checkpoint
        ckpt_path = os.path.join(args.ckpt, f"stgcn_wacv_s{args.seed}_f{f}.pt")
        torch.save(best_state, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

        all_results[f"fold_{f}"] = {
            "test_metrics": test_metrics,
            "floor_mad": float(floor_test["MAD"]),
            "checkpoint": ckpt_path,
        }

    # Summary
    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    fold_mads = [all_results[f"fold_{f}"]["test_metrics"]["MAD"] for f in fold_ids]
    print(f"Test MAD per fold: {[f'{m:.3f}' for m in fold_mads]}")
    print(f"Mean test MAD: {np.mean(fold_mads):.4f} ± {np.std(fold_mads):.4f}")

    # Save training results
    results_path = os.path.join(args.ckpt, f"stgcn_wacv_s{args.seed}_results.json")
    with open(results_path, "w") as fh:
        json.dump({
            "args": vars(args),
            "fold_results": all_results,
            "mean_mad": float(np.mean(fold_mads)),
            "std_mad": float(np.std(fold_mads)),
        }, fh, indent=2)
    print(f"Saved results: {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
