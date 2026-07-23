#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_egru.py
=============
Train THE METHOD: SE3EquivariantGRU -- SE(3)-invariant, irregular-sampling-native, no flow.

Protocol is identical to every other model in this repo, so the numbers are comparable:
subject-disjoint pooled 5-fold CV, val-selected epoch (never test-selected), per-exercise mean
floor reported alongside, subject-cluster bootstrap on the difference.

Bars to clear, all measured on these exact folds:

    per-exercise mean floor            8.25     <- must beat, or nothing else matters
    SE(3)-equivariant Neural CDE       8.43     <- the flow we dropped
    ridge on invariants                7.68
    GRU on HAND-CRAFTED invariants     6.54     <- must match: we learn the features instead
    PCT (target architecture)          6.07     <- PARITY here is the paper

Run:  python src/train_egru.py --cv
"""

import argparse
import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimore_cde_data as kd                                     # noqa: E402
import block2_transforms as bt                                   # noqa: E402
from equivariant_gru import SE3EquivariantGRU, count_parameters  # noqa: E402
from train_cde import metrics, fmt, floor_metrics                # noqa: E402
from protocol_null import bootstrap_delta                        # noqa: E402

SCORE_MAX = kd.SCORE_MAX


def variant_tag(args):
    """Checkpoint identity = (variant, seed, fold). Anything less has already cost us a model."""
    return ("egru" + ("aug" if args.aug_drop > 0 else "")
            + ("nospeed" if args.no_speed else "")
            + ("chi" if args.chiral else ""))


@torch.no_grad()
def predict(model, samples, device, bs=8):
    model.eval()
    out = []
    for i in range(0, len(samples), bs):
        t, x, y, e, n = kd.collate(samples[i: i + bs], device=device)
        out.append(model(t, x, e, n).float().cpu().numpy())
    return np.concatenate(out)


def train_fold(tr, va, te, args, device, verbose=True):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = SE3EquivariantGRU(
        n_scalar=args.n_scalar, n_vec=args.n_vec, n_layers=args.layers,
        lmax=args.lmax, gru_hidden=args.gru_hidden, dropout=args.dropout,
        n_exercises=5, use_speed=not args.no_speed, use_chiral=args.chiral,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    huber = nn.HuberLoss(delta=args.huber_delta)

    floor_val = floor_metrics(tr, va, True)
    floor_test = floor_metrics(tr, te, True)
    if verbose:
        print(f"  model: {count_parameters(model):,} params | "
              f"train {len(tr)} seqs / {len({s['subject'] for s in tr})} subjects")
        print(f"  floor (per-exercise mean):  val {fmt(floor_val)}")

    if verbose and args.aug_drop > 0:
        print(f"  aug: burst drops ~ U[0, {args.aug_drop:.0%}] + jitter x{args.aug_jitter:.1f}"
              f"  (test sweeps to 70% -- the top levels are EXTRAPOLATION)")
        print(f"  aug stream hash (ep0): "
              f"{bt.aug_stream_hash(tr, args.batch_size, args.aug_drop, args.aug_jitter, args.seed, 0)[:16]}")

    best, best_state, best_ep = math.inf, None, -1
    for ep in range(args.epochs):
        model.train()
        tot, n_seen = 0.0, 0
        if args.aug_drop > 0:
            # The SHARED corrupted stream -- byte-identical to the one the PCT baseline consumes.
            stream = (kd.collate(c, device=device) for c in
                      bt.aug_epoch(tr, args.batch_size, args.aug_drop, args.aug_jitter,
                                   args.seed, ep))
        else:
            stream = kd.Batcher(tr, args.batch_size, shuffle=True, device=device)
        for t, x, y, e, n in stream:
            opt.zero_grad(set_to_none=True)
            pred = model(t, x, e, n)
            loss = huber(pred, y)

            # L_consistency (PROJECT_BRIEF 3.4.4): the SAME motion, sub-sampled differently, must
            # score the same. This is the irregular-sampling claim as a TRAINING signal, not just
            # an evaluation-time hope -- and it is only expressible because the model reads dt
            # rather than a fixed frame grid.
            if args.lambda_cons > 0:
                keep = torch.rand(x.shape[1], device=device) > args.cons_drop
                keep[0] = True
                keep[-1] = True
                pred2 = model(t[:, keep], x[:, keep], e,
                              torch.clamp(n, max=int(keep.sum().item())))
                loss = loss + args.lambda_cons * ((pred - pred2) ** 2).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            tot += loss.item() * len(y)
            n_seen += len(y)
        sched.step()

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            m_val = metrics(predict(model, va, device), [s["y"] for s in va])
            if m_val["MAD"] < best:
                best, best_ep = m_val["MAD"], ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"    ep {ep+1:3d}  train_loss {tot/n_seen:.5f}   val {fmt(m_val)}"
                      f"{'  *' if best_ep == ep else ''}")

    if best_state is not None:
        model.load_state_dict(best_state)
    pred = predict(model, te, device)
    y = np.array([s["y"] for s in te])
    m_test = metrics(pred, y)
    if verbose:
        print(f"  best epoch {best_ep+1}")
        print(f"  TEST   ours  {fmt(m_test)}")
        print(f"  TEST   floor {fmt(floor_test)}")
        print(f"  -> {'beats' if m_test['MAD'] < floor_test['MAD'] else 'DOES NOT BEAT'} "
              f"the floor ({m_test['MAD']:.3f} vs {floor_test['MAD']:.3f})")
    return model, m_test, floor_test, pred, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--max-len", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--huber-delta", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--n-scalar", type=int, default=32)
    ap.add_argument("--n-vec", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lmax", type=int, default=2)
    ap.add_argument("--gru-hidden", type=int, default=128)
    ap.add_argument("--no-speed", action="store_true",
                    help="ablation: drop the invariant speed channels")
    ap.add_argument("--chiral", action="store_true",
                    help="admit parity-ODD pseudo-scalars: invariance group O(3) -> SO(3), so the "
                         "model can see trajectory handedness. Rotation invariance is unaffected "
                         "(det=+1 on proper rotations); see chirality.py and certify_egru E6-E8.")
    ap.add_argument("--lambda-cons", type=float, default=0.0)
    ap.add_argument("--cons-drop", type=float, default=0.3)
    # -- shared Block-2 augmentation (the SAME budget is given to the PCT baseline)
    ap.add_argument("--aug-drop", type=float, default=0.0,
                    help="train on burst drops ~ U[0, aug_drop]. Test sweeps to 0.7 (extrapolation)")
    ap.add_argument("--aug-jitter", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs/cde_block2")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    S = kd.load_all_exercises(max_len=args.max_len, verbose=True)
    folds = kd.subject_folds(S, k=args.folds, seed=args.seed)
    fold_ids = range(args.folds) if args.cv else [args.fold]

    os.makedirs(args.out, exist_ok=True)
    rows, eh, ef, subj = [], [], [], []
    for f in fold_ids:
        tr, va, te = kd.split(S, folds, test_fold=f, val_fold=(f + 1) % args.folds)
        assert not ({s["subject"] for s in tr} & {s["subject"] for s in te}), "subject leak"
        print(f"\n{'='*70}\nFOLD {f}\n{'='*70}")
        t0 = time.time()
        model, m_test, floor, pred, y = train_fold(tr, va, te, args, device)
        print(f"  ({time.time()-t0:.0f}s)")
        rows.append({"fold": f, "test": m_test, "floor": floor})
        fl = kd.exercise_mean_floor(tr, te)
        eh += list((pred - y) * SCORE_MAX)
        ef += list((fl - y) * SCORE_MAX)
        subj += [s["subject"] for s in te]
        # Every variant AND every seed gets its own path. Training is nondeterministic (cuDNN GRU
        # backward + e3nn index_add_ use atomics), so a checkpoint is identified by (variant, seed,
        # fold) or it is not identified at all. An earlier tagging scheme omitted both and an
        # ablation silently overwrote the headline model.
        torch.save(model.state_dict(),
                   os.path.join(args.out, f"{variant_tag(args)}_s{args.seed}_pooled_f{f}.pt"))
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    if len(rows) > 1:
        print(f"\n{'='*70}\nSE(3)-EQUIVARIANT GRU -- POOLED, SUBJECT-DISJOINT {args.folds}-FOLD\n{'='*70}")
        for k in ("MAD", "RMSE", "MAPE"):
            o = np.array([r["test"][k] for r in rows])
            b = np.array([r["floor"][k] for r in rows])
            print(f"  {k:5s}  ours {o.mean():7.3f} +/- {o.std():5.3f}    "
                  f"per-exercise mean {b.mean():7.3f} +/- {b.std():5.3f}")
        w = sum(r["test"]["MAD"] < r["floor"]["MAD"] for r in rows)
        d, lo, hi, n = bootstrap_delta(eh, ef, subj)
        print(f"\n  folds beating the floor: {w}/{len(rows)}")
        print(f"  MAD(ours) - MAD(floor) = {d:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]"
              f"   (subject-cluster bootstrap, n={n})")
        print("  -> " + ("GENUINELY beats the mean predictor." if hi < 0
                         else "indistinguishable from the mean predictor."))
        print(f"\n  reference on these folds:  PCT 6.065 | GRU-handcrafted 6.539 | "
              f"ridge 7.678 | floor 8.249 | SE(3)-CDE 8.427")

    with open(os.path.join(args.out, f"{variant_tag(args)}_s{args.seed}_results.json"), "w") as fh:
        json.dump({"args": vars(args), "results": rows}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
