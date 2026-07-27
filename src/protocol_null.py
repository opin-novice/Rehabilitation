#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
protocol_null.py
================
Nails down WHY the target architecture reports MAD ~4.85 on KIMORE Ex1 while scoring at the
mean-predictor floor under an honest protocol.

The mechanism is NOT subject leakage
------------------------------------
Our first hypothesis was subject leakage, and it was WRONG -- worth recording so nobody
re-derives it. In KIMORE Exercise 1 each subject contributes exactly ONE recording, so
`train_test_split` over samples is ALREADY subject-disjoint. There is no subject to leak.

The actual mechanism is OPTIMISTIC EPOCH SELECTION. `src/train_reproduce.py` (lines 334-350)
evaluates on the TEST set every epoch and keeps the minimum:

    test_mad, ... = evaluate_mad(model, test_loader, ...)
    if test_mad < best_mad:
        best_mad = test_mad          # <- the reported number
        best_epoch = epoch

There is no validation split. With a noisy 16-sample test metric tracked over 60+ epochs, the
MINIMUM is a biased estimate of generalisation: you are selecting on the thing you report. A
secondary leak sits at line 132-134, where the input StandardScaler is fit on train+test ("Scale
X on ALL data (matches original repo)").

This script measures both, with the SAME model, data and split, so the comparison is exact:

    honest    : test MAD at the epoch chosen by a held-out VALIDATION fold
    inflated  : min over epochs of the test MAD          (the train_reproduce protocol)
    floor     : predict the training mean

and puts a paired bootstrap CI on (model - floor) so "beats the floor" is a claim with an
interval attached rather than a coin flip on 16 subjects.

Run:  python src/protocol_null.py --model pct --cv
      python src/protocol_null.py --model cde --cv
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimore_cde_data as kd                                    # noqa: E402
import block2_transforms as bt                                  # noqa: E402
from models_curvenet import PointCloudTransformerRegressor      # noqa: E402
from cde_model import SE3NeuralCDE                              # noqa: E402
from train_cde import metrics, fmt                              # noqa: E402

SCORE_MAX = kd.SCORE_MAX


# =============================================================================
# Paired bootstrap on the per-subject absolute errors
# =============================================================================
def bootstrap_delta(err_model, err_floor, subjects, n_boot=10000, seed=0):
    """95% CI on mean|err_model| - mean|err_floor|, paired, CLUSTER-bootstrapped by SUBJECT.

    The resampling unit MUST be the subject, not the sequence. In the pooled setting each
    subject contributes five recordings whose errors are strongly correlated (same person, same
    impairment, same body). Resampling sequences would treat those five as five independent
    draws, understate the variance, and hand us a CI that is too narrow -- manufacturing
    significance out of within-subject correlation. We resample subjects WITH REPLACEMENT and
    take each drawn subject's whole block of recordings.

    Negative interval entirely below 0  => the model genuinely beats the floor.
    Interval straddling 0               => indistinguishable from predicting the mean.
    """
    rng = np.random.default_rng(seed)
    a = np.abs(np.asarray(err_model, dtype=np.float64))
    b = np.abs(np.asarray(err_floor, dtype=np.float64))
    subjects = np.asarray(subjects)

    uniq = np.unique(subjects)
    blocks = [np.where(subjects == u)[0] for u in uniq]
    n_sub = len(uniq)

    d = np.empty(n_boot)
    for k in range(n_boot):
        pick = rng.integers(0, n_sub, size=n_sub)
        idx = np.concatenate([blocks[i] for i in pick])
        d[k] = a[idx].mean() - b[idx].mean()
    return (float(a.mean() - b.mean()),
            float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), n_sub)


# =============================================================================
# Per-epoch test tracking (this is what exposes the optimistic-selection bias)
# =============================================================================
def run_fold(model_name, tr, va, te, args, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if model_name == "pct":
        # 0, NOT 5. This script produces the epoch-selection audit the paper quotes
        # (6.42+/-0.44 honest vs 5.21+/-0.19 test-selected). Those numbers were produced when
        # num_exercises was accepted and IGNORED, i.e. by an unconditioned head. Now that the
        # argument is live, passing 5 would silently train a different model and the audit would
        # no longer describe the run it is cited for. Conditioning is a separate, measured arm
        # (train_baseline_pct.py --exercise-cond); it does not belong in the protocol audit.
        nex = 0
        model = PointCloudTransformerRegressor(
            seq_len=args.n_frames, num_joints=kd.N_JOINTS, num_channels=3,
            dim=256, spatial_depth=6, temporal_depth=3, heads=4, dropout=0.1, k=10,
            num_exercises=nex,
        ).to(device)
        xtr, ytr, etr = bt.batch_fixed_grid(tr, args.n_frames, "linear", device=device)
        xva, yva, eva = bt.batch_fixed_grid(va, args.n_frames, "linear", device=device)
        xte, yte, ete = bt.batch_fixed_grid(te, args.n_frames, "linear", device=device)

        fwd_tr = lambda idx: model(xtr[idx], exercise_id=etr[idx]).squeeze(-1)   # noqa: E731
        eval_sets = {"val": (xva, yva, eva), "test": (xte, yte, ete)}
        lr = 1e-3
    else:
        model = SE3NeuralCDE(n_joints=kd.N_JOINTS, n_scalar=32, n_vec=16,
                             gain=1.0, n_steps=32, dropout=0.1,
                             n_exercises=(5 if args.pooled else 0)).to(device)
        ttr, xtr_, ytr, etr, _ = kd.collate(tr, device=device)
        tva, xva_, yva, eva, _ = kd.collate(va, device=device)
        tte, xte_, yte, ete, _ = kd.collate(te, device=device)
        fwd_tr = lambda idx: model(ttr[idx], xtr_[idx], ex_id=etr[idx])   # noqa: E731
        eval_sets = {"val": ((tva, xva_), yva, eva), "test": ((tte, xte_), yte, ete)}
        lr = 3e-3

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    huber = nn.HuberLoss(delta=0.1)
    n_tr = len(tr)

    @torch.no_grad()
    def evaluate(key):
        model.eval()
        xs, ys, es = eval_sets[key]
        if model_name == "pct":
            p = model(xs, exercise_id=es).squeeze(-1)
        else:
            p = model(xs[0], xs[1], ex_id=es)
        return p.float().cpu().numpy(), ys.float().cpu().numpy()

    hist = []                                    # (epoch, val_mad, test_mad, test_preds)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, args.batch_size):
            idx = perm[i: i + args.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = huber(fwd_tr(idx), ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        pv, yv = evaluate("val")
        pt, yt = evaluate("test")
        hist.append({
            "epoch": ep,
            "val_mad": metrics(pv, yv)["MAD"],
            "test_mad": metrics(pt, yt)["MAD"],
            "test_pred": pt.copy(),
        })

    y_test = np.array([s["y"] for s in te], dtype=np.float64)

    # --- honest: pick the epoch by VALIDATION, then read TEST once ------------
    best_val_ep = int(np.argmin([h["val_mad"] for h in hist]))
    honest_pred = hist[best_val_ep]["test_pred"]

    # --- inflated: the train_reproduce protocol (min over epochs ON TEST) -----
    inflated_ep = int(np.argmin([h["test_mad"] for h in hist]))
    inflated_pred = hist[inflated_ep]["test_pred"]

    floor_pred = (kd.exercise_mean_floor(tr, te) if args.pooled
                  else np.full(len(te), float(np.mean([s["y"] for s in tr]))))

    return {
        "honest": metrics(honest_pred, y_test),
        "inflated": metrics(inflated_pred, y_test),
        "floor": metrics(floor_pred, y_test),
        "best_val_epoch": best_val_ep,
        "inflated_epoch": inflated_ep,
        "err_honest": ((honest_pred - y_test) * SCORE_MAX).tolist(),
        "err_floor": ((floor_pred - y_test) * SCORE_MAX).tolist(),
        "subjects": [s["subject"] for s in te],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["pct", "cde"], default="pct")
    ap.add_argument("--exercise", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--pooled", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-frames", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs/cde_block2")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = (kd.load_all_exercises(verbose=False) if args.pooled
               else kd.load_exercise(args.exercise, verbose=False))
    folds = kd.subject_folds(samples, k=args.folds, seed=args.seed)
    fold_ids = range(args.folds) if args.cv else [0]

    print(f"\n{'='*78}")
    print(f"PROTOCOL AUDIT -- {args.model.upper()}, KIMORE Exercise {args.exercise}, "
          f"subject-disjoint {args.folds}-fold")
    print(f"{'='*78}")

    rows, eh, ef, subj = [], [], [], []
    for f in fold_ids:
        tr, va, te = kd.split(samples, folds, test_fold=f, val_fold=(f + 1) % args.folds)
        r = run_fold(args.model, tr, va, te, args, device)
        rows.append(r)
        eh += r["err_honest"]
        ef += r["err_floor"]
        subj += r["subjects"]
        print(f"\n  fold {f}:  honest(val-selected, ep {r['best_val_epoch']+1:2d})  "
              f"{fmt(r['honest'])}")
        print(f"           inflated(min-on-test, ep {r['inflated_epoch']+1:2d})  "
              f"{fmt(r['inflated'])}")
        print(f"           floor                        {fmt(r['floor'])}")

    n_sub_total = len(set(subj))
    print(f"\n{'-'*78}\nPOOLED OVER {len(rows)} FOLDS "
          f"({len(eh)} held-out sequences from {n_sub_total} held-out subjects)\n{'-'*78}")
    for key in ("MAD", "RMSE", "MAPE"):
        h = np.array([r["honest"][key] for r in rows])
        i = np.array([r["inflated"][key] for r in rows])
        fl = np.array([r["floor"][key] for r in rows])
        print(f"  {key:5s}  honest {h.mean():7.3f} +/- {h.std():5.3f}   "
              f"inflated {i.mean():7.3f} +/- {i.std():5.3f}   "
              f"floor {fl.mean():7.3f} +/- {fl.std():5.3f}")

    d, lo, hi, n_sub = bootstrap_delta(eh, ef, subj)
    print(f"\n  PAIRED BOOTSTRAP, MAD(model) - MAD(floor), CLUSTER-resampled by SUBJECT "
          f"(n={n_sub}):")
    print(f"     delta = {d:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
    if hi < 0:
        print(f"     -> CI entirely below 0: the model GENUINELY beats the mean predictor.")
    else:
        print(f"     -> CI STRADDLES 0: indistinguishable from predicting the training mean.")
        print(f"        Subject-level score regression has NO signal here. Any downstream")
        print(f"        robustness curve built on this model would be measuring noise.")

    infl = np.mean([r["inflated"]["MAD"] for r in rows])
    hon = np.mean([r["honest"]["MAD"] for r in rows])
    print(f"\n  OPTIMISTIC-SELECTION BIAS (the train_reproduce protocol):")
    print(f"     min-over-epochs on test = {infl:.3f}   vs   val-selected = {hon:.3f}")
    print(f"     inflation = {hon - infl:+.3f} MAD ({100*(hon-infl)/hon:+.1f}%), from selecting")
    print(f"     the reported epoch on the reported test set. Same model, same data, same split.")

    os.makedirs(args.out, exist_ok=True)
    # Distinct filename per slice so a later pooled run does not clobber the single-exercise
    # artifact (and vice versa). Pooled keeps the historical name for backward compatibility.
    suffix = "" if args.pooled else f"_ex{args.exercise}_seed{args.seed}"
    with open(os.path.join(args.out, f"protocol_audit_{args.model}{suffix}.json"), "w") as fh:
        json.dump({"args": vars(args), "folds": rows,
                   "bootstrap": {"delta": d, "lo": lo, "hi": hi,
                                 "n_subjects_resampled": n_sub,
                                 "unit": "subject-cluster"}}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
