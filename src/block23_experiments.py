#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
block23_experiments.py
======================
The two structural experiments, run on TRAINED models over subject-disjoint folds.

  BLOCK 2 -- irregular-sampling stress test.
      Sweep burst-drop rate 0 -> 70% (Gilbert-Elliott) + timestamp jitter at TEST time.
      Ours     : natural cubic spline on the ACTUAL corrupted stamps. No resampling.
      Baseline : forced through the resampling operator R onto a uniform grid. We report its
                 BEST case over {linear, cubic, forward-fill}, so "you sabotaged the baseline"
                 is not available as an objection.
      Both receive the BYTE-IDENTICAL corrupted timeline (t~_k, x_k); a SHA-256 input hash is
      asserted equal across pathways at every level and seed (PROJECT_BRIEF 8.4).

  BLOCK 3 -- cross-viewpoint transfer.
      Rotate the TEST skeletons about the vertical (gravity) axis: camera azimuth 0 -> 180 deg.
      Never rotate at training time -- otherwise a flat curve would only prove we augmented.
      Ours is flat BY THEOREM (certify_trainable G3: read-out invariance at 1e-15).
      The baseline learned its invariance from data and must decay.

THE TWO METRICS, AND WHY BOTH
-----------------------------
  Degradation(m, level) = | s_m(corrupt) - s_m(clean) |    -- each method against its OWN clean
                                                              score. No cross-method scaling.
  Accuracy(m, level)    = MAD( s_m(corrupt), y )           -- against the CLINICAL LABEL.

Degradation alone is NOT sufficient and we refuse to report it alone: a model that ignores its
input has degradation identically zero and would "win" every robustness plot. Accuracy-vs-level
is what makes the claim mean anything, and it is only interpretable because the pooled models
provably clear the mean-predictor floor (protocol_null.py). The floor is drawn on every plot.

THE SIGNAL GATE (--method)
--------------------------
"Ours" is the SE3EquivariantGRU (--method egru, default): SE(3)-invariant by construction and
irregular-sampling-native (the recurrence consumes the actual inter-arrival dt_k; no frame grid
exists anywhere in it). It clears the mean-predictor floor on 5/5 folds (6.43 vs 8.25 MAD) and
sits inside the baseline's own fold spread (PCT 6.15 +/- 0.48) -- so its flat curves are the flat
curves of a model that HAS signal.

The Neural CDE (--method cde) is retained as a control, NOT as the headline. Its pooled
checkpoints score 8.43 MAD against a 8.25 floor -- it does not clear the floor, so a flat
degradation curve from it would be indistinguishable from a model ignoring its input. Any run
whose clean score fails to beat the floor is marked NO-SIGNAL and its degradation columns are
declared uninterpretable (--strict makes this a hard abort).

Run:  python src/block23_experiments.py --block 3 --cv
      python src/block23_experiments.py --block 2 --cv
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimore_cde_data as kd                                    # noqa: E402
import block2_transforms as bt                                  # noqa: E402
import invariant_controls as ic                                 # noqa: E402
from cde_model import SE3NeuralCDE                              # noqa: E402
from equivariant_gru import SE3EquivariantGRU                   # noqa: E402
from models_curvenet import PointCloudTransformerRegressor      # noqa: E402
from train_cde import metrics                                   # noqa: E402

SCORE_MAX = kd.SCORE_MAX


def degr_stats(s_rot, s_clean, base):
    """Per-sequence score shift under rotation, as a DISTRIBUTION, not just its mean.

    `base` is the mean over test sequences -- the quantity previously reported alone, and
    mislabelled downstream as a worst case. A clinical claim of the form "an individual patient's
    score moves by up to X" is a statement about the tail, so we also record the median, the 95th
    percentile and the max. The max is the literal "up to"; p95 is the robust version, since a max
    over ~15 held-out sequences per fold rides on a single subject.
    """
    d = np.abs(np.asarray(s_rot) - np.asarray(s_clean)) * SCORE_MAX
    return {base: float(d.mean()),
            f"{base}_p50": float(np.percentile(d, 50)),
            f"{base}_p95": float(np.percentile(d, 95)),
            f"{base}_max": float(d.max())}


DROP_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
ANGLES = [0, 15, 30, 45, 60, 90, 120, 150, 180]
OPERATORS = ["linear", "cubic", "zero_fill"]


def hash_samples(samples):
    """SHA-256 over the (t~, x) actually handed to a method. Both pathways must match."""
    h = hashlib.sha256()
    for s in samples:
        h.update(np.ascontiguousarray(np.asarray(s["t"], dtype=np.float64)).tobytes())
        h.update(np.ascontiguousarray(np.asarray(s["x"], dtype=np.float64)).tobytes())
    return h.hexdigest()


# =============================================================================
# Prediction pathways
# =============================================================================
def build_ours(f, args, device):
    """The invariant, grid-free model under test. R = identity for every variant.

    egru   -- the proposed model: steerable e3nn encoder + recurrence on actual dt.
    invgru -- HAND-CRAFTED invariants (bone lengths, radii, key distances, speeds) + GRU. No
              e3nn, no tensor products. It is exactly as SO(3)-invariant as we are (every feature
              is a function of inner products of joint coordinates), so its rotation curve is also
              flat by theorem. It is the control that asks what the steerable machinery BUYS, and
              it earns a first-class row rather than a footnote.
    cde    -- the continuous-flow control. Fails the mean-predictor floor; kept, not headlined.
    """
    s = args.model_seed
    chi = "chi" if args.chiral else ""
    if args.method == "egru":
        m = SE3EquivariantGRU(
            n_scalar=args.n_scalar, n_vec=args.n_vec, n_layers=args.layers,
            lmax=args.lmax, gru_hidden=args.gru_hidden, dropout=0.0,
            n_exercises=5, use_speed=not args.no_speed, use_chiral=args.chiral).to(device)
        ck = f"{'egruaug' if args.aug else 'egru'}{chi}_s{s}_pooled_f{f}.pt"
        sd = torch.load(os.path.join(args.ckpt, ck), map_location=device)
        # Checkpoints banked before `encoder.dead_scalar` was added still load here. That parameter
        # is only read inside `if mask is not None` (equivariant_gru.py:181-183), and Blocks 2/3
        # never mask joints -- node failure lives in joint_failure.py. Verified empirically: two
        # different random dead_scalar draws give bit-identical predictions with mask=None.
        # The allow-list is deliberately exact: any OTHER missing key is a real mismatch and must
        # not be silently filled with random init.
        missing, unexpected = m.load_state_dict(sd, strict=False)
        stale = set(missing) - {"encoder.dead_scalar"}
        if stale or unexpected:
            raise RuntimeError(
                f"checkpoint {ck} does not match the current model: "
                f"missing={sorted(stale)} unexpected={sorted(unexpected)}")
        return m

    if args.method == "invgru":
        ck = torch.load(os.path.join(args.ckpt, f"invgru{chi}_s{s}_pooled_f{f}.pt"),
                        map_location=device)
        m = ic.InvariantGRU(ck["n_feat"]).to(device)
        m.load_state_dict(ck["state"])
        # The train-split scaler travels WITH the weights. Re-fitting it on corrupted test data
        # would leak and would silently change the input distribution.
        m._mu = ck["mu"].to(device)
        m._sd = ck["sd"].to(device)
        return m

    m = SE3NeuralCDE(n_joints=kd.N_JOINTS, n_scalar=args.n_scalar, n_vec=args.n_vec,
                     gain=args.gain, n_steps=args.n_steps, dropout=0.0,
                     n_exercises=5, n_readout=args.n_readout).to(device)
    m.load_state_dict(torch.load(os.path.join(args.ckpt, f"cde_pooled_f{f}.pt"),
                                 map_location=device))
    return m


@torch.no_grad()
def predict_ours(model, samples, args, device, bs=8):
    """OUR pathway: consume the ACTUAL stamps. No resampling operator anywhere."""
    model.eval()

    if args.method == "invgru":
        # InvariantGRU mean-pools over the PADDED sequence with no length mask, so a batched
        # prediction would depend on the longest sample in the batch -- and under Block-2 drops the
        # lengths change. We predict over the whole eval set in one call, exactly as it was
        # trained, which removes the batch-composition dependence entirely.
        X, _, e = ic.pad_series(samples, device, chiral=args.chiral)
        X = (X - model._mu) / model._sd
        return model(X, e).float().cpu().numpy()

    out = []
    for i in range(0, len(samples), bs):
        t, x, y, e, n = kd.collate(samples[i: i + bs], device=device)
        p = model(t, x, e, n) if args.method == "egru" else model(t, x, ex_id=e)
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def predict_pct(model, samples, n_frames, operator, device, bs=8):
    """BASELINE pathway: resample onto a uniform grid first (its architecture demands it)."""
    model.eval()
    out = []
    for i in range(0, len(samples), bs):
        x, y, e = bt.batch_fixed_grid(samples[i: i + bs], n_frames, operator, device=device)
        out.append(model(x, exercise_id=e).squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)


# =============================================================================
def run_fold(f, samples, folds, args, device):
    val_f = (f + 1) % args.folds
    tr, va, te = kd.split(samples, folds, test_fold=f, val_fold=val_f)
    y = np.array([s["y"] for s in te])
    floor = kd.exercise_mean_floor(tr, te)
    floor_mad = metrics(floor, y)["MAD"]

    ours = build_ours(f, args, device)

    ptag = "pooled" + ("aug" if args.aug else "") + ("rot" if args.pct_rot else "")
    pct_sd = torch.load(
        os.path.join(args.ckpt, f"pct_{ptag}_s{args.model_seed}_f{f}.pt"), map_location=device)
    # Infer exercise conditioning from the checkpoint rather than hardcoding it. The head's input
    # width is dim + num_exercises, so an unconditioned checkpoint (every banked one) gives 0 and a
    # --exercise-cond checkpoint gives 5. Hardcoding 5 silently worked only while the model ignored
    # the argument entirely; now that it uses it, the wrong value is a load-time shape error.
    n_ex_ckpt = int(pct_sd["head.1.weight"].shape[1]) - 256
    pct = PointCloudTransformerRegressor(
        seq_len=args.n_frames, num_joints=kd.N_JOINTS, num_channels=3,
        dim=256, spatial_depth=6, temporal_depth=3, heads=4, dropout=0.1, k=10,
        num_exercises=n_ex_ckpt).to(device)
    pct.load_state_dict(pct_sd)

    rows = []

    # ---- clean reference; both blocks share it ---------------------------
    s_ours0 = predict_ours(ours, te, args, device)
    s_pct0 = {op: predict_pct(pct, te, args.n_frames, op, device) for op in OPERATORS}

    # ---- SIGNAL GATE -----------------------------------------------------
    # A flat degradation curve is only evidence if the clean model beats the mean-predictor.
    # Otherwise "robust" and "deaf to its input" are the same picture.
    clean_mad = metrics(s_ours0, y)["MAD"]
    has_signal = bool(clean_mad < floor_mad)
    if not has_signal:
        msg = (f"NO-SIGNAL: fold {f} clean MAD {clean_mad:.3f} >= floor {floor_mad:.3f}. "
               f"Degradation is uninterpretable for this model.")
        if args.strict:
            raise SystemExit(f"  [ABORT] {msg}")
        print(f"  [WARN] {msg}")

    if args.block == 3:
        for deg in ANGLES:
            rot = [bt.rotate_sample(s, deg) for s in te]
            s_o = predict_ours(ours, rot, args, device)
            rec = {"fold": f, "angle": deg, "floor_mad": floor_mad,
                   "has_signal": has_signal, "clean_mad": clean_mad,
                   "ours_mad": metrics(s_o, y)["MAD"],
                   **degr_stats(s_o, s_ours0, "ours_degr")}
            for op in OPERATORS:
                s_p = predict_pct(pct, rot, args.n_frames, op, device)
                rec[f"pct_{op}_mad"] = metrics(s_p, y)["MAD"]
                rec.update(degr_stats(s_p, s_pct0[op], f"pct_{op}_degr"))
            rows.append(rec)
            print(f"  fold {f} angle {deg:3d}deg   ours MAD {rec['ours_mad']:6.3f} "
                  f"(degr {rec['ours_degr']:5.3f})   "
                  f"PCT-linear MAD {rec['pct_linear_mad']:6.3f} "
                  f"(degr {rec['pct_linear_degr']:5.3f})   floor {floor_mad:6.3f}")

    else:
        # ---- Block 2: corruption sweep -----------------------------------
        for lvl in DROP_LEVELS:
            per_seed = []
            for seed in range(args.seeds):
                cor = [bt.corrupt_sample(s, lvl, jitter_scale=1.0, seed=seed * 1000 + i)
                       for i, s in enumerate(te)]
                # FAIRNESS LOCK: both pathways must consume the identical corrupted bytes.
                hh = hash_samples(cor)
                s_o = predict_ours(ours, cor, args, device)
                rec = {"fold": f, "level": lvl, "seed": seed, "hash": hh,
                       "floor_mad": floor_mad,
                       "has_signal": has_signal, "clean_mad": clean_mad,
                       "ours_mad": metrics(s_o, y)["MAD"],
                       "ours_degr": float(np.mean(np.abs(s_o - s_ours0)) * SCORE_MAX)}
                for op in OPERATORS:
                    s_p = predict_pct(pct, cor, args.n_frames, op, device)
                    assert hash_samples(cor) == hh, "input diverged between pathways"
                    rec[f"pct_{op}_mad"] = metrics(s_p, y)["MAD"]
                    rec[f"pct_{op}_degr"] = float(np.mean(np.abs(s_p - s_pct0[op])) * SCORE_MAX)
                per_seed.append(rec)
            rows += per_seed
            m = lambda k: float(np.mean([r[k] for r in per_seed]))       # noqa: E731
            print(f"  fold {f} drop {lvl:4.0%}   ours MAD {m('ours_mad'):6.3f} "
                  f"(degr {m('ours_degr'):5.3f})   "
                  f"PCT-best MAD {min(m(f'pct_{o}_mad') for o in OPERATORS):6.3f}   "
                  f"floor {floor_mad:6.3f}")

    del ours, pct
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, choices=[2, 3], required=True)
    ap.add_argument("--method", choices=["egru", "invgru", "cde"], default="egru",
                    help="egru (proposed) | invgru (hand-crafted invariants: what does e3nn buy?) "
                         "| cde (continuous-flow control, fails the floor)")
    ap.add_argument("--pct-rot", action="store_true",
                    help="Block-3 fairness arm: load the ROTATION-AUGMENTED baseline")
    ap.add_argument("--model-seed", type=int, default=0,
                    help="which training seed's checkpoints to sweep (training is nondeterministic)")
    ap.add_argument("--strict", action="store_true",
                    help="abort instead of warn if the clean model fails the floor")
    ap.add_argument("--aug", action="store_true",
                    help="use the drop-augmented checkpoints (BOTH arms; identical aug budget)")
    ap.add_argument("--chiral", action="store_true",
                    help="sweep the SO(3) (parity-odd) checkpoints instead of the O(3) ones. "
                         "Block-3 degradation must stay 0.000: rotations have det=+1, so admitting "
                         "pseudo-scalars cannot cost the viewpoint theorem (certify_egru E8).")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--seeds", type=int, default=3, help="corruption seeds per level (block 2)")
    ap.add_argument("--max-len", type=int, default=150)
    ap.add_argument("--n-frames", type=int, default=100)
    ap.add_argument("--n-scalar", type=int, default=32)
    ap.add_argument("--n-vec", type=int, default=8)
    # -- egru
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lmax", type=int, default=2)
    ap.add_argument("--gru-hidden", type=int, default=128)
    ap.add_argument("--no-speed", action="store_true")
    # -- cde (control)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--n-steps", type=int, default=32)
    ap.add_argument("--n-readout", type=int, default=4)
    ap.add_argument("--ckpt", type=str, default="outputs/cde_block2")
    ap.add_argument("--out", type=str, default="outputs/cde_block2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = kd.load_all_exercises(max_len=args.max_len, verbose=False)
    # CRITICAL: the fold partition MUST use the same seed the checkpoints were trained with.
    # train_*.py builds folds with subject_folds(seed=training_seed); a checkpoint's held-out test
    # fold is defined by THAT partition. Evaluating a seed-N checkpoint on the seed-0 partition puts
    # subjects the model trained on into its "test" set -- a leak that silently deflates MAD (we saw
    # 6.6 -> 4.4). The partition follows the model, not a separate --seed knob.
    folds = kd.subject_folds(samples, k=args.folds, seed=args.model_seed)
    fold_ids = range(args.folds) if args.cv else [args.fold]

    name = {2: "IRREGULAR-SAMPLING STRESS TEST", 3: "CROSS-VIEWPOINT TRANSFER"}[args.block]
    print(f"\n{'='*78}\nBLOCK {args.block} -- {name}  (pooled, subject-disjoint)\n"
          f"ours = {args.method.upper()}   vs   PCT baseline\n{'='*78}")

    rows = []
    for f in fold_ids:
        rows += run_fold(f, samples, folds, args, device)

    # ---- aggregate -------------------------------------------------------
    key = "angle" if args.block == 3 else "level"
    levels = ANGLES if args.block == 3 else DROP_LEVELS
    print(f"\n{'-'*78}\nAGGREGATE over {len(fold_ids)} folds\n{'-'*78}")
    hdr = f"  {key:>7s} | {'ours MAD':>9s} {'ours degr':>10s} | " \
          f"{'PCT-best MAD':>13s} {'PCT-best degr':>14s} | {'floor':>7s}"
    print(hdr)
    for lv in levels:
        sel = [r for r in rows if abs(r[key] - lv) < 1e-9]
        if not sel:
            continue
        cm = np.mean([r["ours_mad"] for r in sel])
        cd = np.mean([r["ours_degr"] for r in sel])
        pm = min(np.mean([r[f"pct_{o}_mad"] for r in sel]) for o in OPERATORS)
        pd = min(np.mean([r[f"pct_{o}_degr"] for r in sel]) for o in OPERATORS)
        fl = np.mean([r["floor_mad"] for r in sel])
        print(f"  {lv:7.2f} | {cm:9.3f} {cd:10.3f} | {pm:13.3f} {pd:14.3f} | {fl:7.3f}")

    # ---- signal gate verdict --------------------------------------------
    ns = sorted({r["fold"] for r in rows if not r["has_signal"]})
    if ns:
        print(f"\n  [WARN] SIGNAL GATE FAILED on folds {ns}: clean MAD >= mean-predictor floor.")
        print(f"         The '{args.method}' degradation columns above are NOT interpretable --")
        print("         a model deaf to its input also has flat degradation. Do not plot these.")
    else:
        print(f"\n  signal gate: PASS on all {len(fold_ids)} folds "
              f"(clean MAD < mean-predictor floor). Flat degradation is evidence, not an artifact.")

    if args.block == 2:
        n_h = len({r["hash"] for r in rows})
        print(f"\n  fairness lock: {n_h} distinct corrupted-input hashes across "
              f"{len(rows)} (fold, level, seed) cells; both pathways asserted identical.")

    os.makedirs(args.out, exist_ok=True)
    sfx = ("_aug" if args.aug else "") + ("_rot" if args.pct_rot else "")
    if args.method != "egru":
        sfx += f"_{args.method}"
    if args.chiral:
        sfx += "_chi"           # MUST be in the filename: the driver scripts use the results file
    sfx += f"_s{args.model_seed}"   # as their idempotency marker, and a collision SKIPS the real run
    with open(os.path.join(args.out, f"block{args.block}{sfx}_results.json"), "w") as fh:
        json.dump({"args": vars(args), "rows": rows}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
