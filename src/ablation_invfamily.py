#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ablation_invfamily.py
=====================
Reviewer Q2: the marginal contribution of each invariant FAMILY to accuracy and robustness.

We measure it the safe, honest way -- INFERENCE-ONLY, on the EXISTING trained chiral checkpoints,
by ZEROING one family at a time while preserving the projection dimension (so no retraining and no
architecture change; the zeroed slice is fed to the learned input weights as zeros). This mirrors
run_dt_ablation.py --mode test.

Families (equivariant_gru.InvariantProjection cut), the reviewer's five + bone lengths:
    scalars   tanh(s)                       parity-even
    norms     log1p||v||                    parity-even
    cosines   <vhat_i, vhat_j>              parity-even
    triples   det[vhat_a,vhat_b,vhat_c]     parity-ODD (learned pseudo-scalars)
    volumes   anatomical signed volumes     parity-ODD
    bones     inter-joint bone lengths      parity-even

Two axes are reported per family:
    ACCURACY    pooled clean MAD (and delta vs the full cut).
    ROBUSTNESS  node-failure MAD lost over k frozen joints (mode 'hold', hash-locked), the Sec IV-E
                axis restricted to the EGRU cut -- which families make the read-out brittle to a
                dead sensor node.

CAVEAT stated in the paper: zero-masking measures family importance on a model TRAINED WITH the
family (the weights may have distributed information), which is the standard read for an inference
ablation. It is NOT a retrain-without-family capacity ablation.

Run:  python src/ablation_invfamily.py --cv --chiral
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimore_cde_data as kd                                    # noqa: E402
from equivariant_gru import SE3EquivariantGRU                   # noqa: E402
from joint_failure import fail_joints, hash_samples, K_LEVELS   # noqa: E402
from train_cde import metrics                                   # noqa: E402

SCORE_MAX = kd.SCORE_MAX
FAMILIES = ["scalars", "norms", "cosines", "triples", "volumes", "bones"]
# node-failure levels reused from joint_failure but trimmed for the ablation grid
NF_LEVELS = [0, 1, 2, 4, 8]


@torch.no_grad()
def predict(m, samples, device, ablate, bs=8):
    m.eval()
    out = []
    for i in range(0, len(samples), bs):
        t, x, y, e, n = kd.collate(samples[i: i + bs], device=device)
        out.append(m(t, x, e, n, ablate=ablate).float().cpu().numpy())
    return np.concatenate(out)


def load_egru_chiral(ckpt_dir, seed, fold, device):
    m = SE3EquivariantGRU(dropout=0.0, n_exercises=5, use_chiral=True).to(device)
    # Same drift guard as joint_failure.load_arms: these checkpoints predate the (unused when
    # use_mask=False) encoder.dead_scalar parameter. Assert that is the ONLY missing key.
    inc = m.load_state_dict(torch.load(
        os.path.join(ckpt_dir, f"egruchi_s{seed}_pooled_f{fold}.pt"), map_location=device),
        strict=False)
    assert set(inc.missing_keys) <= {"encoder.dead_scalar"} and not inc.unexpected_keys, \
        f"egruchi checkpoint drift beyond the unused dead_scalar: {inc}"
    return m


def run_fold(seed, fold, S, folds, args, device):
    tr, va, te = kd.split(S, folds, test_fold=fold, val_fold=(fold + 1) % args.folds)
    y = np.array([s["y"] for s in te])
    floor = metrics(kd.exercise_mean_floor(tr, te), y)["MAD"]
    m = load_egru_chiral(args.ckpt, seed, fold, device)

    variants = [None] + FAMILIES        # None == full cut (anchor)
    rows = []

    # ---- clean accuracy per variant ----
    clean_pred = {}
    for fam in variants:
        ab = None if fam is None else {fam}
        p = predict(m, te, device, ab)
        clean_pred[fam] = p
        rows.append({"seed": seed, "fold": fold, "variant": fam or "full",
                     "k": 0, "draw": 0, "mad": metrics(p, y)["MAD"], "floor": floor})

    # ANCHOR: the full cut must reproduce the trained test MAD for this (seed, fold).
    full_k0 = rows[0]["mad"]

    # ---- node-failure robustness per variant (identical hash-locked corruption to all variants) ----
    for k in NF_LEVELS:
        if k == 0:
            continue
        for draw in range(args.draws):
            cor = [fail_joints(s, k, "hold", seed=draw * 977 + i) for i, s in enumerate(te)]
            hh = hash_samples(cor)
            for fam in variants:
                ab = None if fam is None else {fam}
                p = predict(m, cor, device, ab)
                assert hash_samples(cor) == hh, "input diverged between variants"
                rows.append({"seed": seed, "fold": fold, "variant": fam or "full",
                             "k": k, "draw": draw, "mad": metrics(p, y)["MAD"], "floor": floor})

    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"  seed {seed} fold {fold}: full-cut clean MAD {full_k0:6.3f}  (floor {floor:6.3f})")
    return rows


def summarise(rows):
    """Per variant: clean MAD (k=0), node-fail MAD at max k, MAD lost, all mean over (seed,fold,draw)."""
    variants = ["full"] + FAMILIES
    kmax = max(NF_LEVELS)
    out = []
    for fam in variants:
        clean = np.mean([r["mad"] for r in rows if r["variant"] == fam and r["k"] == 0])
        nf = np.mean([r["mad"] for r in rows if r["variant"] == fam and r["k"] == kmax])
        out.append({"variant": fam, "clean_mad": float(clean),
                    "nodefail_kmax_mad": float(nf), "mad_lost": float(nf - clean)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3, help="model seeds (checkpoints) to average")
    ap.add_argument("--draws", type=int, default=3, help="failed-joint draws per node-fail level")
    ap.add_argument("--chiral", action="store_true", help="use the chiral (SO(3)) headline model")
    ap.add_argument("--max-len", type=int, default=150)
    ap.add_argument("--ckpt", type=str, default="outputs/cde_block2")
    ap.add_argument("--out", type=str, default="outputs/cde_block2")
    args = ap.parse_args()
    assert args.chiral, "the parity-odd families exist only in the chiral checkpoints; pass --chiral"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = kd.load_all_exercises(max_len=args.max_len, verbose=False)

    print(f"\n{'='*78}\nREVIEWER Q2 -- per-invariant-family ablation (inference-only, zero-mask)")
    print(f"chiral EGRU; {args.seeds} seeds x {args.folds} folds; families: {', '.join(FAMILIES)}")
    print(f"{'='*78}")

    rows = []
    fold_ids = range(args.folds) if args.cv else [args.fold]
    for seed in range(args.seeds):
        folds = kd.subject_folds(S, k=args.folds, seed=seed)   # MUST match the ckpt's seed
        for fold in fold_ids:
            rows += run_fold(seed, fold, S, folds, args, device)

    summary = summarise(rows)
    full = next(r for r in summary if r["variant"] == "full")

    # ANCHOR check (aggregate): the full cut must land at the Table I EGRU SO(3) value 6.73
    # within the 0.33 MAD seed floor. A gross mismatch means wrong folds/checkpoints.
    assert abs(full["clean_mad"] - 6.73) < 0.5, \
        f"ANCHOR FAIL: full-cut clean MAD {full['clean_mad']:.3f} != Table I 6.73"

    print(f"\n{'-'*78}\n  {'family removed':<16s}  {'clean MAD':>10s}  {'d vs full':>10s}  "
          f"{'nodefail k'+str(max(NF_LEVELS)):>12s}  {'MAD lost':>10s}")
    print(f"  {'-'*16}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}")
    for r in summary:
        tag = "(full cut)" if r["variant"] == "full" else r["variant"]
        dcl = r["clean_mad"] - full["clean_mad"]
        print(f"  {tag:<16s}  {r['clean_mad']:10.3f}  {dcl:+10.3f}  "
              f"{r['nodefail_kmax_mad']:12.3f}  {r['mad_lost']:+10.3f}")
    print(f"{'-'*78}")
    print(f"  floor {full['clean_mad']:.2f} anchor OK (Table I EGRU SO(3) = 6.73).")
    print("  clean-MAD delta = family's marginal accuracy contribution (zero-mask, model trained")
    print("  WITH the family). MAD lost = node-failure brittleness over "
          f"{max(NF_LEVELS)} dead joints.")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ablation_invfamily.json"), "w") as fh:
        json.dump({"args": vars(args), "families": FAMILIES, "nf_levels": NF_LEVELS,
                   "rows": rows, "summary": summary}, fh, indent=2)
    print(f"\nwrote {os.path.join(args.out, 'ablation_invfamily.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
