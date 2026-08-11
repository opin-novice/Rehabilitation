#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_uiprmd.py  --  THIRD-CORPUS REPLICATION (WACV fix 1.4: widen the empirical base)
=====================================================================================
The reviewer objection 1.4 is "narrow empirical base": the positive story rests on KIMORE plus a
10-subject REHAB24-6 replication. This adds a THIRD, fully independent corpus -- UI-PRMD (Vakanski
et al.), Kinect, correct/incorrect segmented movements -- and asks the SAME three structural questions
train_rehab246.py asks, with the SAME code, so a reviewer sees the properties are not a two-corpus
coincidence:

  (a) accuracy TIE           -- EGRU vs a PCT baseline, in AUROC (subject-disjoint CV);
  (b) node-failure GRACE     -- AUROC lost over k dead joints (the steerable-encoder differentiator);
  (c) viewpoint INVARIANCE   -- logits unchanged under a 90 deg camera rotation (the theorem, re-certified).

DRY BY DESIGN. Every non-trivial routine -- the Mann-Whitney AUROC, the KIMORE-recipe preprocessing,
subject-disjoint folds, both training loops, and the clean/node-fail/viewpoint evaluation -- is IMPORTED
from train_rehab246.py, not re-implemented. This file overrides only the data source (UI-PRMD manifest
+ sequences, identical schema to REHAB24-6) and the output sink. If the two corpora ever diverge in
handling, that is a bug: they must be processed identically for the replication to mean anything.

ISOLATION: imports src/ modules READ-ONLY; writes ONLY under outputs/uiprmd/. Touches no KIMORE or
REHAB24-6 artifact. (Importing train_rehab246 creates an empty outputs/rehab246/ dir as a benign
side effect of that module's isolation guard; nothing is written there.)

    python src/train_uiprmd.py --model egru --cv                  # EGRU arm, 5 subject-disjoint folds
    python src/train_uiprmd.py --model pct  --cv                  # PCT baseline arm (for the tie)
    python src/train_uiprmd.py --model egru --cv --chiral --seed 1

Data prerequisite -- build the UI-PRMD binary testbed first (needs the INCORRECT set downloaded;
load_uiprmd_validity.py prints the exact command if it is missing):
    python src/load_uiprmd_validity.py --build
  expected: outputs/validity_uiprmd/uiprmd_sequences.npy   (N, SEQ_LEN, 25, 3)
            outputs/validity_uiprmd/uiprmd_manifest.csv    (rep_uid, exercise_id, subject_id, correct_label)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../src
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import kimore_cde_data as kd                                # noqa: E402
from equivariant_gru import count_parameters                # noqa: E402
# Reuse the SECOND-corpus machinery verbatim -- identical processing is the whole point of a replication.
from train_rehab246 import (                                # noqa: E402
    auroc, _preprocess, subject_folds_rehab246,
    train_fold_egru, train_fold_pct, evaluate_fold,
    NF_LEVELS,  # noqa: F401  (kept in scope so evaluate_fold's level set is identical)
)

# ----- ISOLATION GUARD: every write goes here and nowhere else -----------------------------------
OUT = os.path.join(_ROOT, "outputs", "uiprmd")
assert os.path.abspath(OUT).replace("\\", "/").endswith("outputs/uiprmd"), \
    f"refusing to run: output dir {OUT} is not the uiprmd sink"
os.makedirs(OUT, exist_ok=True)

SEQ_PATH = os.path.join(_ROOT, "outputs", "validity_uiprmd", "uiprmd_sequences.npy")
MAN_PATH = os.path.join(_ROOT, "outputs", "validity_uiprmd", "uiprmd_manifest.csv")
FPS = 30.0                                                  # UI-PRMD Kinect native rate (nominal, uniform)


def load_uiprmd_samples(verbose=True):
    """UI-PRMD binary testbed -> KIMORE-format sample dicts. Mirrors train_rehab246.load_rehab246_samples
    exactly (same schema, same preprocessing) -- only the paths differ."""
    if not (os.path.exists(SEQ_PATH) and os.path.exists(MAN_PATH)):
        raise FileNotFoundError(
            f"Missing UI-PRMD testbed. Build it first (needs the INCORRECT set downloaded):\n"
            f"    python src/load_uiprmd_validity.py --build\n"
            f"  expected: {SEQ_PATH}\n            {MAN_PATH}")
    seqs = np.load(SEQ_PATH)                                # (N, SEQ_LEN, 25, 3)
    man = pd.read_csv(MAN_PATH)
    assert len(seqs) == len(man), f"seq/manifest length mismatch: {len(seqs)} vs {len(man)}"
    T = seqs.shape[1]
    t_uniform = (np.arange(T, dtype=np.float64) / FPS) / kd.TIME_SCALE

    ex_ids = sorted(man["exercise_id"].unique())
    ex_remap = {e: i + 1 for i, e in enumerate(ex_ids)}    # -> 1..K (collate subtracts 1)
    n_exercises = len(ex_ids)

    samples = []
    for k, row in man.iterrows():
        x = _preprocess(seqs[k])                           # KIMORE recipe, imported
        label = int(row["correct_label"])
        samples.append({
            "t": t_uniform.copy(),
            "x": x,
            "n_frames": T,
            "y": float(label),
            "exercise": ex_remap[int(row["exercise_id"])],
            "subject": int(row["subject_id"]),
            "cohort": "correct" if label == 1 else "incorrect",
        })
    if verbose:
        pos = int(sum(s["y"] for s in samples))
        subs = sorted({s["subject"] for s in samples})
        print(f"[uiprmd] {len(samples)} reps  (correct={pos} / incorrect={len(samples)-pos})  "
              f"subjects={len(subs)}  exercises={n_exercises}")
    return samples, n_exercises


def main():
    ap = argparse.ArgumentParser(description="UI-PRMD third-corpus replication (WACV fix 1.4)")
    ap.add_argument("--model", choices=["egru", "pct"], default="egru")
    ap.add_argument("--cv", action="store_true", help="run all K subject-disjoint folds")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--draws", type=int, default=3, help="failed-joint draws per node-failure level")
    ap.add_argument("--n-scalar", type=int, default=32)
    ap.add_argument("--n-vec", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lmax", type=int, default=2)
    ap.add_argument("--gru-hidden", type=int, default=128)
    ap.add_argument("--no-speed", action="store_true")
    ap.add_argument("--chiral", action="store_true", help="admit parity-odd channels (deployed model)")
    ap.add_argument("--n-frames", type=int, default=100)
    ap.add_argument("--smoke", action="store_true", help="1 fold x few epochs for a wiring check")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.cv = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples, n_exercises = load_uiprmd_samples()
    folds = subject_folds_rehab246(samples, k=args.folds, seed=args.seed)  # generic subject-disjoint folds
    fold_ids = range(args.folds) if args.cv else [args.fold]

    print(f"\n{'='*76}\nUI-PRMD replication  model={args.model}  chiral={args.chiral}  "
          f"seed={args.seed}  device={device}\n  (writes to {OUT})\n{'='*76}")
    rows = []
    for f in fold_ids:
        tr, va, te = kd.split(samples, folds, test_fold=f, val_fold=(f + 1) % args.folds)
        if args.model == "egru":
            model = train_fold_egru(tr, va, args, n_exercises, device)
            n_params = count_parameters(model)
        else:
            model = train_fold_pct(tr, va, args, n_exercises, device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        res = evaluate_fold(model, te, args, device, draws=args.draws)
        res["fold"] = f
        res["n_test"] = len(te)
        rows.append(res)
        print(f"  fold {f}: clean AUROC {res['clean_auroc']:.3f} | "
              f"nf-lost 0->8 {res['nf_lost_0to8']:+.3f} | "
              f"view drift {res['view_logit_drift']:.2e} (AUROC@90 {res['view_auroc_90deg']:.3f})")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    def _m(key):
        v = [r[key] for r in rows if not np.isnan(r[key])]
        return (float(np.mean(v)), float(np.std(v, ddof=1) if len(v) > 1 else 0.0)) if v else (float("nan"), 0.0)

    summary = {
        "corpus": "uiprmd", "model": args.model, "chiral": args.chiral, "seed": args.seed,
        "n_exercises": n_exercises, "n_params": int(n_params), "folds": [r["fold"] for r in rows],
        "clean_auroc_mean_std": _m("clean_auroc"),
        "nf_lost_0to8_mean_std": _m("nf_lost_0to8"),
        "view_logit_drift_max": float(np.nanmax([r["view_logit_drift"] for r in rows])),
        "rows": rows,
    }
    ca, cs = summary["clean_auroc_mean_std"]
    la, ls = summary["nf_lost_0to8_mean_std"]
    print(f"{'-'*76}")
    print(f"  {args.model}: clean AUROC = {ca:.3f} +- {cs:.3f} | "
          f"node-fail AUROC lost 0->8 = {la:+.3f} +- {ls:.3f} | "
          f"max viewpoint logit drift = {summary['view_logit_drift_max']:.2e}")
    tag = "_chiral" if args.chiral else ""
    out_path = os.path.join(OUT, f"uiprmd_{args.model}{tag}_s{args.seed}.json")
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
