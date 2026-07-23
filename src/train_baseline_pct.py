#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_baseline_pct.py
=====================
The TARGET PAPER's architecture (CurveNet point-cloud encoder + axial self-attention), trained
and evaluated on the EXACT SAME subject-disjoint folds as our SE(3)-equivariant Neural CDE.

Why this script exists
----------------------
`src/train_reproduce.py` reports MAD 4.85 on Kimore_ex1. That number is not comparable to a
subject-disjoint one -- but NOT for the reason one first assumes. In KIMORE Exercise 1 each
subject contributes exactly ONE recording, so its `train_test_split` is already subject-disjoint;
there is no subject to leak. The inflation comes from OPTIMISTIC EPOCH SELECTION: it evaluates on
the test set every epoch and reports the minimum, with no validation split (train_reproduce.py
lines 334-350). Measured with the same model, data and split, that bias is worth +1.111 MAD
(16.8%) -- see `protocol_null.py`.

This script is the apples-to-apples control. Identical folds, identical labels, identical floor,
and the reported epoch is chosen on a held-out VALIDATION fold. The only differences that remain
are the architecture and the input pathway:

  * ours     : natural cubic spline on the ACTUAL Kinect arrival stamps. No resampling.
  * baseline : forced through the resampling operator R onto a uniform T-frame grid, because
               its architecture requires a fixed frame axis. We give it its BEST CASE and report
               whichever of {linear, cubic, forward-fill} is strongest (PROJECT_BRIEF 8.4).

Reading the result
------------------
The mean-predictor floor is the thing that matters. If the baseline ALSO fails to clear it on
subject-disjoint splits, then subject-level score regression on KIMORE Ex1 is near-null for
everyone, published MADs are a protocol artifact, and Block 1 ("clean-data parity") is not a
row we can win OR lose -- it is a row that means nothing, and the paper's weight must rest
entirely on the structural axes (Blocks 2-4). That would be a finding, not a failure.

Run:
  python src/train_baseline_pct.py --fold 0
  python src/train_baseline_pct.py --cv --operator linear
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
import kimore_cde_data as kd                                    # noqa: E402
import block2_transforms as bt                                  # noqa: E402
from models_curvenet import PointCloudTransformerRegressor      # noqa: E402
from train_cde import metrics, fmt, floor_metrics                              # noqa: E402


@torch.no_grad()
def evaluate(model, samples, n_frames, operator, device, batch_size=8):
    model.eval()
    preds, trues = [], []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i: i + batch_size]
        x, y, e = bt.batch_fixed_grid(chunk, n_frames, operator, device=device)
        p = model(x, exercise_id=e).squeeze(-1)
        preds.append(p.float().cpu().numpy())
        trues.append(y.float().cpu().numpy())
    return metrics(np.concatenate(preds), np.concatenate(trues))


def train_fold(train_s, val_s, test_s, args, device, verbose=True):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = PointCloudTransformerRegressor(
        seq_len=args.n_frames, num_joints=kd.N_JOINTS, num_channels=3,
        dim=args.dim, spatial_depth=args.spatial_depth,
        temporal_depth=args.temporal_depth, heads=4, dropout=args.dropout, k=args.k,
        num_exercises=(5 if args.pooled else 0),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    huber = nn.HuberLoss(delta=args.huber_delta)

    floor_val = floor_metrics(train_s, val_s, args.pooled)
    floor_test = floor_metrics(train_s, test_s, args.pooled)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"  PCT baseline: {n_par:,} params | resampling R = {args.operator} "
              f"-> {args.n_frames} uniform frames")
        print(f"  train {len(train_s)} / val {len(val_s)} / test {len(test_s)} subjects")
        print(f"  mean-predictor floor:  val {fmt(floor_val)}")

    # ---- rotation augmentation (the Block-3 fairness arm) --------------------
    # Our own Block-2 doctrine says an augmentation budget must be given to BOTH arms or to
    # neither. Block 3 previously withheld rotation augmentation from the baseline, which is that
    # doctrine violated. This arm supplies it: the baseline gets the cheapest, most standard
    # defence against viewpoint change and we let it use it. Our claim survives only if it is the
    # RIGHT claim -- that our invariance is exact and free, while theirs must be bought and is
    # approximate. Rotation acts on coordinates and resampling acts on time, so they commute: we
    # pre-resample once and rotate the grid on GPU, at negligible cost.
    def _rot_batch(xb):
        B = xb.shape[0]
        th = torch.rand(B, device=xb.device) * (2 * math.pi)
        c, s = torch.cos(th), torch.sin(th)
        R = torch.zeros(B, 3, 3, device=xb.device, dtype=xb.dtype)
        R[:, 0, 0] = c
        R[:, 0, 2] = s
        R[:, 1, 1] = 1.0
        R[:, 2, 0] = -s
        R[:, 2, 2] = c                            # about the vertical (gravity) axis
        return torch.einsum("btjc,bdc->btjd", xb, R)

    if args.aug_drop > 0:
        # Same augmentation budget as ours. The baseline consumes the BYTE-IDENTICAL corrupted
        # stream (same order, same drop levels, same jitter -- assert the digests match), and then
        # does the one thing its architecture forces it to do: push it through R. That is the whole
        # comparison. Cannot pre-resample: the corruption is redrawn every epoch.
        if verbose:
            print(f"  aug: burst drops ~ U[0, {args.aug_drop:.0%}] + jitter x{args.aug_jitter:.1f}"
                  f"  -> R = {args.operator}   (test sweeps to 70%: EXTRAPOLATION)")
            print(f"  aug stream hash (ep0): "
                  f"{bt.aug_stream_hash(train_s, args.batch_size, args.aug_drop, args.aug_jitter, args.seed, 0)[:16]}")
        xtr = None
    else:
        # Pre-resample once: R is deterministic and does not change across epochs.
        xtr, ytr, etr = bt.batch_fixed_grid(train_s, args.n_frames, args.operator, device=device)

    best_val, best_state, best_ep = math.inf, None, -1
    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0

        if args.aug_drop > 0:
            batches = (bt.batch_fixed_grid(c, args.n_frames, args.operator, device=device)
                       for c in bt.aug_epoch(train_s, args.batch_size, args.aug_drop,
                                             args.aug_jitter, args.seed, ep))
        else:
            perm = torch.randperm(len(train_s), device=device)
            batches = ((xtr[perm[i: i + args.batch_size]],
                        ytr[perm[i: i + args.batch_size]],
                        etr[perm[i: i + args.batch_size]])
                       for i in range(0, len(train_s), args.batch_size))

        for xb, yb, eb in batches:
            if args.aug_rot:
                xb = _rot_batch(xb)
            opt.zero_grad(set_to_none=True)
            pred = model(xb, exercise_id=eb).squeeze(-1)
            loss = huber(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        sched.step()

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            m_val = evaluate(model, val_s, args.n_frames, args.operator, device, args.batch_size)
            if m_val["MAD"] < best_val:
                best_val, best_ep = m_val["MAD"], ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"    ep {ep+1:3d}  train_loss {tot/n:.5f}   val {fmt(m_val)}"
                      f"{'  *' if best_ep == ep else ''}")

    if best_state is not None:
        model.load_state_dict(best_state)
    m_test = evaluate(model, test_s, args.n_frames, args.operator, device, args.batch_size)
    if verbose:
        print(f"  best epoch {best_ep+1}")
        print(f"  TEST   PCT   {fmt(m_test)}")
        print(f"  TEST   floor {fmt(floor_test)}")
        beat = m_test["MAD"] < floor_test["MAD"]
        print(f"  -> {'beats' if beat else 'DOES NOT BEAT'} the mean-predictor floor "
              f"({m_test['MAD']:.3f} vs {floor_test['MAD']:.3f})")
    return model, m_test, floor_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exercise", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=150)
    ap.add_argument("--n-frames", type=int, default=100)
    ap.add_argument("--operator", type=str, default="linear",
                    choices=["linear", "cubic", "zero_fill"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cv", action="store_true")

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--huber-delta", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--spatial-depth", type=int, default=6)
    ap.add_argument("--temporal-depth", type=int, default=3)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pooled", action="store_true")
    # -- shared Block-2 augmentation (IDENTICAL budget to ours; see block2_transforms.aug_epoch)
    ap.add_argument("--aug-drop", type=float, default=0.0,
                    help="train on burst drops ~ U[0, aug_drop], then resample. Test goes to 0.7")
    ap.add_argument("--aug-jitter", type=float, default=1.0)
    ap.add_argument("--aug-rot", action="store_true",
                    help="Block-3 fairness arm: train with random azimuth rotations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs/cde_block2")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    samples = (kd.load_all_exercises(max_len=args.max_len) if args.pooled
               else kd.load_exercise(args.exercise, max_len=args.max_len))
    folds = kd.subject_folds(samples, k=args.folds, seed=args.seed)   # SAME folds as train_cde
    fold_ids = range(args.folds) if args.cv else [args.fold]

    os.makedirs(args.out, exist_ok=True)
    results = []
    for f in fold_ids:
        val_f = (f + 1) % args.folds
        tr, va, te = kd.split(samples, folds, test_fold=f, val_fold=val_f)
        assert not ({s["subject"] for s in tr} & {s["subject"] for s in te}), "subject leak"
        print(f"\n{'='*70}\nPCT BASELINE -- FOLD {f}  (val = fold {val_f})\n{'='*70}")
        t0 = time.time()
        model, m_test, floor = train_fold(tr, va, te, args, device)
        print(f"  ({time.time()-t0:.0f}s)")
        results.append({"fold": f, "test": m_test, "floor": floor})
        # Augmented runs are tagged separately: the clean pct_pooled_f*.pt are the Block-3
        # baselines and must not be clobbered.
        tag = "pooled" if args.pooled else f"ex{args.exercise}"
        if args.aug_drop > 0:
            tag += "aug"
        if args.aug_rot:
            tag += "rot"
        # Seed is part of the checkpoint identity: training is nondeterministic.
        torch.save(model.state_dict(),
                   os.path.join(args.out, f"pct_{tag}_s{args.seed}_f{f}.pt"))

    if len(results) > 1:
        print(f"\n{'='*70}\nPCT, SUBJECT-DISJOINT {args.folds}-FOLD CV (Exercise {args.exercise})\n{'='*70}")
        for key in ("MAD", "RMSE", "MAPE"):
            o = np.array([r["test"][key] for r in results])
            b = np.array([r["floor"][key] for r in results])
            print(f"  {key:5s}  PCT {o.mean():7.3f} +/- {o.std():5.3f}    "
                  f"floor {b.mean():7.3f} +/- {b.std():5.3f}")
        w = sum(r["test"]["MAD"] < r["floor"]["MAD"] for r in results)
        print(f"\n  folds where PCT beats the floor: {w}/{len(results)}")

    suffix = (("_aug" if args.aug_drop > 0 else "") + ("_rot" if args.aug_rot else "")
              + f"_s{args.seed}")
    with open(os.path.join(args.out, f"pct_results_ex{args.exercise}{suffix}.json"), "w") as fh:
        json.dump({"args": vars(args), "results": results}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
