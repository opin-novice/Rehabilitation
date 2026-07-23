#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_dt_ablation.py
==================
PHASE 0 GATE -- is the irregular-sampling claim a MECHANISM or a PASSENGER?

The claim under test: "the recurrence consumes the sensor's real inter-arrival gaps dt_k, so the
model handles irregular sampling natively." If that is load-bearing, denying the model dt must
hurt it. If MAD is unchanged, dt is inert at the recurrence and the pillar is empty as stated --
which sends us to the F1b pivot (native-rate invariant band-power), not to defensive prose.

THREE conditions, because `dt := ones` conflates two things and a reviewer would catch it:

    real     true dt                          the claim as stated (control)
    seqmean  each seq's MEAN real dt           kills ONLY the frame-to-frame variation
                                               = irregular sampling; TEMPO/duration preserved
    ones     dt := 1 everywhere                kills the variation AND the mean level

dt carries (a) its variation = the irregular-sampling signal, and (b) its mean level = TEMPO
(durations here span 2.3-240 s, and the loader preserves them). So:
    real  vs seqmean  -> the CLEAN test of the irregular-sampling claim.
    real  vs ones     -> "does dt matter in ANY form" (also deletes the duration cue).
    seqmean vs ones   -> the size of the duration cue by itself.

LEAKAGE IS SEALED IN THE MODEL (equivariant_gru: dt_mode transforms dt once, before BOTH the
input channel and the speed = d/dt division). There is exactly one real-dt source and displacement
d never touches dt, so the ablated model cannot reconstruct the real grid.

Two ways to run it:
    --mode test    test-time ablation on the EXISTING seed-0 checkpoints. Seconds. Answers
                   "does our REPORTED model depend on dt?". Uses seqmean as the fair probe (feeding
                   `ones`, scale ~1.0, to a model trained on real dt ~0.007 is a 150x scale shock
                   that breaks it for reasons unrelated to the claim -- reported, but flagged).
    --mode train   from-scratch 5-fold retrain under each dt_mode, seed 42, fully deterministic.
                   Answers the STRONGER question "can a model ALLOWED to use dt beat one denied
                   it?". This is the gate.

Because the noise floor is now bitwise zero (Phase 1), any delta here is pure dt dependence.

Run:  python src/run_dt_ablation.py --mode test
      python src/run_dt_ablation.py --mode train --epochs 80
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
import determinism                                                # noqa: E402 (CUBLAS env first)
import kimore_cde_data as kd                                      # noqa: E402
from equivariant_gru import SE3EquivariantGRU                     # noqa: E402
from train_cde import metrics, floor_metrics                      # noqa: E402

MODES = ["real", "seqmean", "ones"]
SCORE_MAX = kd.SCORE_MAX


@torch.no_grad()
def predict(model, samples, device, bs=8):
    model.eval()
    out = []
    for i in range(0, len(samples), bs):
        t, x, y, e, n = kd.collate(samples[i: i + bs], device=device)
        out.append(model(t, x, e, n).float().cpu().numpy())
    return np.concatenate(out)


# =============================================================================
# TEST-TIME ABLATION on existing checkpoints (fast directional read)
# =============================================================================
def run_test(args, device):
    print("=" * 78)
    print("PHASE 0 -- TEST-TIME dt ABLATION on the EXISTING seed-0 checkpoints")
    print("  does our ALREADY-REPORTED model change its predictions when dt is ablated?")
    print("=" * 78)
    S = kd.load_all_exercises(max_len=args.max_len, verbose=False)
    folds = kd.subject_folds(S, k=args.folds, seed=0)             # MUST match the ckpt seed (0)

    per = {m: [] for m in MODES}
    floors = []
    for f in range(args.folds):
        tr, va, te = kd.split(S, folds, test_fold=f, val_fold=(f + 1) % args.folds)
        y = np.array([s["y"] for s in te])
        floors.append(floor_metrics(tr, te, True)["MAD"])

        ckpt = os.path.join(args.ckpt, f"egru_s0_pooled_f{f}.pt")
        model = SE3EquivariantGRU(n_exercises=5, dropout=0.0).to(device)
        # strict=False: the checkpoint predates the dead-node embedding (dead_scalar), which is
        # unused when use_mask=False. G1 proved the dense-incidence forward is numerically identical
        # to the old index_add_, so dt_mode="real" must reproduce this checkpoint's reported MAD.
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)

        row = {}
        for m in MODES:
            model.dt_mode = m
            row[m] = metrics(predict(model, te, device), y)["MAD"]
            per[m].append(row[m])
        print(f"  fold {f}:  real {row['real']:.3f}   seqmean {row['seqmean']:.3f}   "
              f"ones {row['ones']:.3f}   floor {floors[f]:.3f}")

    _report(per, floors, mode_label="TEST-TIME (existing seed-0 checkpoints)",
            fair_pair=("real", "seqmean"), args=args)
    return per


# =============================================================================
# FROM-SCRATCH RETRAIN under each dt_mode (the gate)
# =============================================================================
def train_fold(tr, va, te, dt_mode, args, device):
    torch.manual_seed(args.seed)
    model = SE3EquivariantGRU(n_exercises=5, dropout=args.dropout, dt_mode=dt_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    huber = nn.HuberLoss(delta=0.1)

    best, best_pred = math.inf, None
    yte = np.array([s["y"] for s in te])
    for ep in range(args.epochs):
        model.train()
        for t, x, y, e, n in kd.Batcher(tr, args.batch_size, shuffle=True, device=device):
            opt.zero_grad(set_to_none=True)
            huber(model(t, x, e, n), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            mv = metrics(predict(model, va, device), [s["y"] for s in va])["MAD"]
            if mv < best:                                        # val-selected, never test-selected
                best = mv
                best_pred = predict(model, te, device)
    return metrics(best_pred, yte)["MAD"]


def run_train(args, device):
    print("=" * 78)
    print(f"PHASE 0 -- FROM-SCRATCH RETRAIN under each dt_mode  (seed {args.seed}, "
          f"{args.folds}-fold, {args.epochs} ep, DETERMINISTIC)")
    print("  can a model ALLOWED to use real dt beat one denied it? If not, the claim is dead.")
    print("=" * 78)
    S = kd.load_all_exercises(max_len=args.max_len, verbose=False)
    folds = kd.subject_folds(S, k=args.folds, seed=args.seed)

    per = {m: [] for m in MODES}
    floors = []
    for f in range(args.folds):
        tr, va, te = kd.split(S, folds, test_fold=f, val_fold=(f + 1) % args.folds)
        assert not ({s["subject"] for s in tr} & {s["subject"] for s in te}), "subject leak"
        floors.append(floor_metrics(tr, te, True)["MAD"])
        line = f"  fold {f}:"
        for m in MODES:
            t0 = time.time()
            mad = train_fold(tr, va, te, m, args, device)
            per[m].append(mad)
            line += f"   {m} {mad:.3f} ({time.time()-t0:.0f}s)"
        line += f"   floor {floors[f]:.3f}"
        print(line)

    _report(per, floors, mode_label=f"FROM-SCRATCH RETRAIN (seed {args.seed})",
            fair_pair=("real", "seqmean"), args=args)
    return per


# =============================================================================
def _report(per, floors, mode_label, fair_pair, args):
    print(f"\n{'-'*78}\n  {mode_label}\n  {'condition':<28s} {'MAD':>8s} {'+/-':>7s}")
    print(f"  {'-'*28} {'-'*8} {'-'*7}")
    stat = {}
    for m in MODES:
        a = np.array(per[m])
        stat[m] = (a.mean(), a.std())
        desc = {"real": "real dt (control)",
                "seqmean": "seq-mean dt (no variation)",
                "ones": "dt := 1 (no variation, no tempo)"}[m]
        print(f"  {desc:<28s} {a.mean():8.3f} {a.std():7.3f}")
    fl = np.array(floors)
    print(f"  {'per-exercise mean floor':<28s} {fl.mean():8.3f} {fl.std():7.3f}")

    a, b = fair_pair
    d_clean = stat[b][0] - stat[a][0]                # seqmean - real: the irregular-sampling effect
    d_tempo = stat["ones"][0] - stat["seqmean"][0]   # ones - seqmean: the duration cue by itself

    # PAIRED, fold-variance-aware. Same seed => same held-out subjects per fold, so the delta is
    # paired and its FOLD SCATTER is what decides significance -- not the point estimate alone. A
    # single wild fold can drag the mean past a fixed threshold while the effect is indistinguishable
    # from zero. Gate the verdict on whether the ~95% interval (mean +/- 2*SEM over folds) clears 0.
    pair = np.array(per[b]) - np.array(per[a])       # per-fold (seqmean - real)
    k = len(pair)
    sem = pair.std(ddof=1) / math.sqrt(k) if k > 1 else float("inf")
    lo, hi = d_clean - 2 * sem, d_clean + 2 * sem
    print(f"\n  irregular-sampling effect  (seqmean - real)   = {d_clean:+.3f} MAD"
          f"   [~95% {lo:+.3f}, {hi:+.3f}], per-fold SEM {sem:.3f}   <- THE CLAIM")
    print(f"  duration/tempo cue         (ones   - seqmean) = {d_tempo:+.3f} MAD")
    print(f"  both together              (ones   - real)    = "
          f"{stat['ones'][0] - stat['real'][0]:+.3f} MAD")

    print()
    if lo <= 0.0 <= hi:                               # interval spans 0 -> cannot distinguish
        print(f"  VERDICT -- NULL: the seqmean-real interval spans 0 ({lo:+.3f}, {hi:+.3f}).")
        print("  Giving the recurrence the real inter-arrival VARIATION does not measurably change")
        print("  MAD. The irregular-sampling claim is a PASSENGER at the recurrence, not a")
        print("  mechanism -- regardless of the point estimate's sign.")
        if d_clean < -args.null_thresh:
            print(f"  (Point estimate is {d_clean:+.3f}, i.e. real dt looks slightly WORSE, but the")
            print("   fold scatter makes that indistinguishable from zero -- likely one noisy fold,")
            print("   NOT evidence that dt actively hurts. Do not report it as an inversion.)")
        print("  => PIVOT TO F1b: move the grid-free advantage into a NATIVE-RATE INVARIANT")
        print("     band-power channel (kimore_cde_data.load_sample, compute_bandpower=True),")
        print("     which does NOT rely on the recurrence reading dt. Then re-run this gate.")
    elif hi < 0.0:
        print(f"  VERDICT -- MECHANISM: removing dt variation costs {d_clean:+.3f} MAD, interval")
        print(f"  ({lo:+.3f}, {hi:+.3f}) entirely below 0. The recurrence genuinely uses irregular")
        print("  timing; the claim stands on its own legs and F1b becomes a strengthening.")
    else:
        print(f"  VERDICT -- INVERTED (real dt worse by {-d_clean:.3f}, interval {lo:+.3f}..{hi:+.3f}")
        print("  entirely above 0): the dt variation is acting as NOISE the model overfits. This is")
        print("  a REAL effect only because the interval clears 0 -- investigate before claiming it.")
    print("-" * 78)

    os.makedirs(args.out, exist_ok=True)
    fname = f"phase0_dt_ablation_{'test' if args.mode == 'test' else f'train_s{args.seed}'}.json"
    with open(os.path.join(args.out, fname), "w") as fh:
        json.dump({"mode": args.mode, "per_fold": per, "floors": floors,
                   "mean": {m: stat[m][0] for m in MODES},
                   "std": {m: stat[m][1] for m in MODES},
                   "irregular_effect_seqmean_minus_real": d_clean,
                   "tempo_cue_ones_minus_seqmean": d_tempo}, fh, indent=2)
    print(f"  wrote {os.path.join(args.out, fname)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["test", "train"], default="test")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--null-thresh", type=float, default=0.05,
                    help="|seqmean - real| below this MAD => NULL => pivot to F1b")
    ap.add_argument("--ckpt", type=str, default="outputs/cde_block2")
    ap.add_argument("--out", type=str, default="outputs/cde_block2")
    args = ap.parse_args()

    cfg = determinism.enable(seed=args.seed, strict=True)
    print(determinism.report(cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}\n")

    if args.mode == "test":
        run_test(args, device)
    else:
        run_train(args, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
