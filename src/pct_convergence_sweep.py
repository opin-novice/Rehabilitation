#!/usr/bin/env python
"""Block-3 cross-viewpoint sweep for the PCT arms ONLY.

Why this exists rather than `block23_experiments.py --block 3 --pct-rot`: that
harness builds and loads the EGRU arm unconditionally, and these PCT-only
convergence runs (60/180/540-epoch budget sweep) retrain no EGRU at all. Running
the full harness would re-measure an arm that did not change, against
checkpoints from a different directory. This script sweeps the PCT arms and
nothing else.

(The banked EGRU checkpoints DO load in the main harness -- they predate
`encoder.dead_scalar`, but that parameter is only read when a joint mask is
supplied, which Blocks 2/3 never do. See build_ours() in block23_experiments.py
for the guarded load and its exact allow-list.)

Degradation uses the shared `block23_experiments.degr_stats`, so the mean and the
p50/p95/max tail are recorded identically here and in the main harness. The
baseline is granted all three resampling operators; aggregation takes the best.

Usage:
  python src/pct_convergence_sweep.py --ckpt outputs/cde_block2_conv180 --rot
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimore_cde_data as kd                                     # noqa: E402
import block2_transforms as bt                                   # noqa: E402
from models_curvenet import PointCloudTransformerRegressor       # noqa: E402
from block23_experiments import predict_pct, degr_stats, ANGLES, OPERATORS  # noqa: E402
from train_cde import metrics                                     # noqa: E402

SCORE_MAX = kd.SCORE_MAX


def load_pct(path, device, k):
    """Build a PCT matching the checkpoint's head width, then load it.

    `k` (the kNN neighbourhood the CurveNet encoder gathers over) MUST be passed in and MUST match
    the value the checkpoint was trained with. Unlike num_exercises it changes no weight shape, so
    a mismatch loads cleanly through strict load_state_dict and silently returns wrong predictions:
    evaluating k=20 weights at k=10 moved this script's angle-0 clean MAD from 6.41 to 8.63 while
    raising no error at all. It cannot be inferred from the state_dict -- it is a runtime gather
    width, not a parameter -- so it is an explicit argument with no default.
    """
    sd = torch.load(path, map_location=device)
    n_ex = int(sd["head.1.weight"].shape[1]) - 256      # dim + num_exercises
    m = PointCloudTransformerRegressor(
        seq_len=100, num_joints=kd.N_JOINTS, num_channels=3, dim=256,
        spatial_depth=6, temporal_depth=3, heads=4, dropout=0.1, k=k,
        num_exercises=n_ex).to(device)
    m.load_state_dict(sd)
    return m, n_ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/cde_block2")
    ap.add_argument("--out", default=None, help="defaults to --ckpt")
    ap.add_argument("--rot", action="store_true", help="sweep the rotation-augmented arm")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-frames", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=150)
    ap.add_argument("--tag", default=None, help="output filename stem")
    ap.add_argument("--k", type=int, default=10,
                    help="kNN neighbourhood the encoder gathers over. MUST match what the "
                         "checkpoints were trained with -- it changes no weight shape, so a "
                         "mismatch loads silently and returns wrong numbers. 10 for the banked "
                         "arms, 20 for the steelman arm and for the reference implementation.")
    args = ap.parse_args()
    out_dir = args.out or args.ckpt
    device = "cuda" if torch.cuda.is_available() else "cpu"

    samples = kd.load_all_exercises(max_len=args.max_len, verbose=False)
    ptag = "pooledrot" if args.rot else "pooled"
    rows, n_ex_seen = [], set()

    for s in args.seeds:
        folds = kd.subject_folds(samples, args.folds, seed=s)
        for f in range(args.folds):
            val_f = (f + 1) % args.folds
            tr, _, te = kd.split(samples, folds, test_fold=f, val_fold=val_f)
            y = np.array([x["y"] for x in te])
            floor_mad = metrics(kd.exercise_mean_floor(tr, te), y)["MAD"]

            ck = os.path.join(args.ckpt, f"pct_{ptag}_s{s}_f{f}.pt")
            pct, n_ex = load_pct(ck, device, args.k)
            n_ex_seen.add(n_ex)

            base = {op: predict_pct(pct, te, args.n_frames, op, device) for op in OPERATORS}
            clean = min(metrics(base[op], y)["MAD"] for op in OPERATORS)
            for deg in ANGLES:
                rot = [bt.rotate_sample(x, deg) for x in te]
                rec = {"seed": s, "fold": f, "angle": deg, "floor_mad": float(floor_mad),
                       "clean_mad": float(clean), "has_signal": bool(clean < floor_mad)}
                for op in OPERATORS:
                    p = predict_pct(pct, rot, args.n_frames, op, device)
                    rec[f"pct_{op}_mad"] = float(metrics(p, y)["MAD"])
                    rec.update(degr_stats(p, base[op], f"pct_{op}_degr"))
                rows.append(rec)
            print(f"  seed {s} fold {f}: clean {clean:6.3f}  floor {floor_mad:6.3f}  "
                  f"180deg degr {rows[-1]['pct_linear_degr']:6.3f}")

    stem = args.tag or f"block3_pctonly_{'rot' if args.rot else 'clean'}"
    path = os.path.join(out_dir, f"{stem}.json")
    with open(path, "w") as fh:
        json.dump({"args": vars(args), "num_exercises_in_ckpt": sorted(n_ex_seen),
                   "rows": rows}, fh, indent=2)

    print(f"\nwrote {path}   (checkpoints were exercise-conditioned: {sorted(n_ex_seen) != [0]})")
    print(f"\n  {'angle':>5s}  {'best MAD':>9s}  {'best degr':>10s}")
    for a in ANGLES:
        sel = [r for r in rows if r["angle"] == a]
        mad = min(float(np.mean([r[f"pct_{op}_mad"] for r in sel])) for op in OPERATORS)
        dg = min(float(np.mean([r[f"pct_{op}_degr"] for r in sel])) for op in OPERATORS)
        print(f"  {a:5d}  {mad:9.3f}  {dg:10.3f}")


if __name__ == "__main__":
    main()
