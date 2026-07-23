#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_cde.py
============
Block-2 phase: train the SE(3)-equivariant Neural CDE on KIMORE clinical scores.

Protocol
--------
  * Subject-disjoint 5-fold CV. Every reported number is on held-out SUBJECTS. (A prior study
    in this repo turned on exactly this: a within-subject split can look strong while the
    subject-level effect is null. We do not repeat that mistake.)
  * Inner validation fold for early stopping / model selection; the test fold is touched once.
  * Fixed-step RK4 (PROJECT_BRIEF 5.1: exactly equivariance-preserving, step-size independent).
  * Huber loss on y = score/50, reported as MAD / RMSE / MAPE on the original 0-50 scale so the
    numbers are directly comparable to the target paper.

The MEAN-PREDICTOR FLOOR is reported next to every result and is not optional. KIMORE scores
are tightly clustered (median y = 0.86), so a model that has learned nothing can still post a
respectable-looking MAD. Any claim of accuracy that does not clear this floor is not a result.

Run:
  python src/train_cde.py --overfit          # sanity: can it memorise 8 samples?
  python src/train_cde.py --fold 0           # one subject-disjoint fold
  python src/train_cde.py --cv               # all 5 folds
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
from cde_model import SE3NeuralCDE, count_parameters          # noqa: E402
from cde_model_mp import SE3MessagePassingCDE                 # noqa: E402
import kimore_cde_data as kd                                  # noqa: E402

SCORE_MAX = kd.SCORE_MAX


# =============================================================================
# Metrics (on the original 0-50 clinical scale)
# =============================================================================
def metrics(pred, true):
    """pred/true: normalised (0-1) tensors -> dict on the 0-50 clinical scale."""
    p = np.asarray(pred, dtype=np.float64) * SCORE_MAX
    y = np.asarray(true, dtype=np.float64) * SCORE_MAX
    err = p - y
    return {
        "MAD": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-6)) * 100.0),
    }


def fmt(m):
    return f"MAD {m['MAD']:6.3f}   RMSE {m['RMSE']:6.3f}   MAPE {m['MAPE']:6.2f}%"


# =============================================================================
# Evaluation
# =============================================================================
@torch.no_grad()
def evaluate(model, samples, device, batch_size=8, n_steps=None):
    model.eval()
    preds, trues = [], []
    for t, x, y, e, _ in kd.Batcher(samples, batch_size, shuffle=False, device=device):
        preds.append(model(t, x, n_steps=n_steps, ex_id=e).float().cpu().numpy())
        trues.append(y.float().cpu().numpy())
    return metrics(np.concatenate(preds), np.concatenate(trues))


def floor_metrics(train_s, eval_s, pooled):
    """Mean-predictor floor. Pooled => PER-EXERCISE mean (the model is given the exercise ID,
    so a predictor that uses ONLY that ID is the baseline it must beat; see
    kimore_cde_data.exercise_mean_floor)."""
    if pooled:
        pred = kd.exercise_mean_floor(train_s, eval_s)
    else:
        pred = np.full(len(eval_s), float(np.mean([s["y"] for s in train_s])))
    return metrics(pred, [s["y"] for s in eval_s])


# =============================================================================
# Training
# =============================================================================
def train_fold(train_s, val_s, test_s, args, device, verbose=True):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.arch == "mp":
        model = SE3MessagePassingCDE(
            n_joints=kd.N_JOINTS, n_scalar=args.n_scalar, n_vec=args.n_vec,
            lmax=args.lmax, hidden=args.hidden, gain=args.gain, n_steps=args.n_steps,
            dropout=args.dropout, n_exercises=(5 if args.pooled else 0),
            n_readout=args.n_readout,
        ).to(device)
    else:
        model = SE3NeuralCDE(
            n_joints=kd.N_JOINTS, n_scalar=args.n_scalar, n_vec=args.n_vec,
            hidden=args.hidden, gain=args.gain, n_steps=args.n_steps, dropout=args.dropout,
            n_exercises=(5 if args.pooled else 0),
            n_readout=args.n_readout, use_dots=not args.no_dots,
        ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    huber = nn.HuberLoss(delta=args.huber_delta)

    floor_val = floor_metrics(train_s, val_s, args.pooled)
    floor_test = floor_metrics(train_s, test_s, args.pooled)

    if verbose:
        n_sub = len({s["subject"] for s in train_s})
        print(f"  model: {count_parameters(model):,} params | "
              f"train {len(train_s)} seqs / {n_sub} subjects, "
              f"val {len(val_s)} / test {len(test_s)} seqs")
        print(f"  floor ({'per-exercise' if args.pooled else 'global'} mean):"
              f"  val {fmt(floor_val)}")

    best_val, best_state, best_ep = math.inf, None, -1
    for ep in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        for t, x, y, e, _ in kd.Batcher(train_s, args.batch_size, shuffle=True, device=device):
            opt.zero_grad(set_to_none=True)
            pred = model(t, x, ex_id=e)
            loss = huber(pred, y)

            # L_consistency (PROJECT_BRIEF 3.4.4): the same motion, sub-sampled differently,
            # must score the same. This is what OPERATIONALISES the irregular-sampling claim --
            # it is a training signal, not just an evaluation-time hope.
            if args.lambda_cons > 0:
                keep = torch.rand(x.shape[1], device=device) > args.cons_drop
                keep[0] = keep[-1] = True                  # keep the endpoints: same [t_0, t_N]
                pred2 = model(t[:, keep], x[:, keep], ex_id=e)
                loss = loss + args.lambda_cons * ((pred - pred2) ** 2).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        sched.step()

        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            m_val = evaluate(model, val_s, device, args.batch_size)
            if m_val["MAD"] < best_val:
                best_val, best_ep = m_val["MAD"], ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"    ep {ep+1:3d}  train_loss {tot/n:.5f}   val {fmt(m_val)}"
                      f"{'  *' if best_ep == ep else ''}")

    if best_state is not None:
        model.load_state_dict(best_state)

    m_test = evaluate(model, test_s, device, args.batch_size)
    m_val = evaluate(model, val_s, device, args.batch_size)
    if verbose:
        print(f"  best epoch {best_ep+1}")
        print(f"  TEST   ours  {fmt(m_test)}")
        print(f"  TEST   floor {fmt(floor_test)}")
        beat = m_test["MAD"] < floor_test["MAD"]
        print(f"  -> {'beats' if beat else 'DOES NOT BEAT'} the mean-predictor floor "
              f"({m_test['MAD']:.3f} vs {floor_test['MAD']:.3f})")
    return model, m_val, m_test, floor_test


def load_samples(args):
    if args.pooled:
        return kd.load_all_exercises(max_len=args.max_len, verbose=True)
    return kd.load_exercise(args.exercise, max_len=args.max_len)


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exercise", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=150)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cv", action="store_true", help="run all folds")
    ap.add_argument("--overfit", action="store_true", help="sanity: memorise 8 samples")

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--huber-delta", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=2)

    ap.add_argument("--arch", choices=["global", "mp"], default="mp",
                    help="mp = skeleton message-passing field (PROJECT_BRIEF 3.3); "
                         "global = the earlier global-latent field (ablation)")
    ap.add_argument("--lmax", type=int, default=2)
    ap.add_argument("--n-scalar", type=int, default=32)
    ap.add_argument("--n-vec", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--gain", type=float, default=0.2)
    ap.add_argument("--n-steps", type=int, default=48)
    ap.add_argument("--n-readout", type=int, default=4,
                    help="K equally spaced trajectory checkpoints for the invariant head")
    ap.add_argument("--no-dots", action="store_true",
                    help="drop pairwise <v_i,v_j> invariants from the head (ablation)")

    ap.add_argument("--lambda-cons", type=float, default=0.0)
    ap.add_argument("--cons-drop", type=float, default=0.3)

    ap.add_argument("--pooled", action="store_true",
                    help="pool all 5 exercises (~380 seqs); exercise ID enters as a 0e scalar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs/cde_block2")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    samples = load_samples(args)

    # ---- sanity: can the architecture memorise a handful of sequences? -------
    if args.overfit:
        print("\n[SANITY] overfit 8 samples (no dropout, no weight decay).")
        print("If this does NOT reach ~0 MAD, the bug is in the architecture or the optimiser,")
        print("not in the data -- and no amount of tuning on the real split will tell us that.")
        # The subset MUST span the label range. Taking samples[:8] in file order gives 8
        # CG/Expert subjects whose scores are nearly identical (label std 0.040): on such a set
        # the mean predictor is already near-perfect, so "memorising" and "predicting the mean"
        # are indistinguishable and the test certifies nothing. Spread across the score range.
        ys = np.array([s["y"] for s in samples])
        idx = np.argsort(ys)[np.linspace(0, len(ys) - 1, 8).astype(int)]
        sub = [samples[i] for i in idx]
        print(f"  label-diverse subset: std {ys[idx].std():.4f} "
              f"(vs {ys[:8].std():.4f} for the first 8 in file order), "
              f"range [{ys[idx].min():.2f}, {ys[idx].max():.2f}]")
        args.dropout, args.wd, args.eval_every = 0.0, 0.0, 5
        train_fold(sub, sub, sub, args, device)
        return 0

    folds = kd.subject_folds(samples, k=args.folds, seed=args.seed)
    fold_ids = range(args.folds) if args.cv else [args.fold]

    os.makedirs(args.out, exist_ok=True)
    results = []
    for f in fold_ids:
        val_f = (f + 1) % args.folds
        tr, va, te = kd.split(samples, folds, test_fold=f, val_fold=val_f)
        assert not ({s["subject"] for s in tr} & {s["subject"] for s in te}), "subject leak"
        print(f"\n{'='*70}\nFOLD {f}  (val = fold {val_f})\n{'='*70}")
        t0 = time.time()
        model, m_val, m_test, floor = train_fold(tr, va, te, args, device)
        print(f"  ({time.time()-t0:.0f}s)")
        results.append({"fold": f, "test": m_test, "floor": floor, "val": m_val})
        tag = ("pooled" if args.pooled else f"ex{args.exercise}")
        tag = f"{tag}_{args.arch}" if args.arch != "mp" else tag
        torch.save(model.state_dict(), os.path.join(args.out, f"cde_{tag}_f{f}.pt"))
        # Backprop-through-solver retains a deep graph (n_steps x 4 stages of tensor products).
        # Without an explicit release the folds accumulate and the run dies with a bad_alloc
        # partway through the CV -- which is exactly what happened the first time.
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    if len(results) > 1:
        tag = "POOLED (Es1-5)" if args.pooled else f"Exercise {args.exercise}"
        floor_name = "per-exercise mean" if args.pooled else "global mean"
        print(f"\n{'='*70}\nSUBJECT-DISJOINT {args.folds}-FOLD CV  --  {tag}\n{'='*70}")
        for key in ("MAD", "RMSE", "MAPE"):
            ours = np.array([r["test"][key] for r in results])
            base = np.array([r["floor"][key] for r in results])
            print(f"  {key:5s}  ours {ours.mean():7.3f} +/- {ours.std():5.3f}    "
                  f"{floor_name} {base.mean():7.3f} +/- {base.std():5.3f}")
        w = sum(r["test"]["MAD"] < r["floor"]["MAD"] for r in results)
        print(f"\n  folds where we beat the floor: {w}/{len(results)}")
        if w < len(results):
            print("  ^ NOT a clean win. Do not write this up as an accuracy result.")

    with open(os.path.join(args.out, f"results_ex{args.exercise}.json"), "w") as fh:
        json.dump({"args": vars(args), "results": results}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
