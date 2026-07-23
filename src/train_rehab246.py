#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_rehab246.py  --  SECOND-CORPUS REPLICATION (WACV Workstream C)
====================================================================
Replicates the paper's structural story on a SECOND rehab corpus, REHAB24-6 (Zenodo 13305826),
to de-risk the "heavy KIMORE reliance" weakness. REHAB24-6 labels are BINARY (correct / incorrect
repetition), so this is a CLASSIFICATION task measured in AUROC -- NOT a cross-corpus transfer of the
KIMORE regressor (which is a known null; see the validity study). It trains a fresh model on
REHAB24-6 and asks whether the three properties replicate:

  (a) accuracy TIE           -- EGRU vs a PCT baseline, in AUROC (subject-disjoint CV);
  (b) node-failure GRACE     -- AUROC lost over k dead joints (the steerable-encoder differentiator);
  (c) viewpoint INVARIANCE   -- logits unchanged under camera rotation (the theorem, re-certified).

The paper's SE3EquivariantGRU head is a single unbounded scalar, so it is used directly as a binary
LOGIT with BCEWithLogitsLoss -- ZERO architecture change. Preprocessing (root-relative, median-radius
scale norm) and batching (kd.collate) are reused verbatim from the KIMORE pipeline so the two corpora
are handled identically.

ISOLATION: imports the paper's src/ modules READ-ONLY; writes ONLY under outputs/rehab246/. It does not
touch outputs/cde_block2/ or any KIMORE artifact.

    python src/train_rehab246.py --model egru --cv                 # EGRU arm, 5 subject-disjoint folds
    python src/train_rehab246.py --model pct  --cv                 # PCT baseline arm (for the tie)
    python src/train_rehab246.py --model egru --cv --seed 1        # another seed

Data prerequisite (already built by src/load_rehab246.py --build):
    outputs/validity/rehab246_sequences.npy      (N, 150, 25, 3)
    outputs/validity/rehab246_manifest.csv       (rep_uid, exercise_id, subject_id, correct_label, ...)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../src
_ROOT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import kimore_cde_data as kd                                # noqa: E402  (constants, collate, folds)
import block2_transforms as bt                              # noqa: E402  (rotate_sample, batch_fixed_grid)
from equivariant_gru import SE3EquivariantGRU, count_parameters  # noqa: E402
from joint_failure import fail_joints                       # noqa: E402

# ----- ISOLATION GUARD: every write goes here and nowhere else -----------------------------------
OUT = os.path.join(_ROOT, "outputs", "rehab246")
assert os.path.abspath(OUT).replace("\\", "/").endswith("outputs/rehab246"), \
    f"refusing to run: output dir {OUT} is not the rehab246 sink"
os.makedirs(OUT, exist_ok=True)

SEQ_PATH = os.path.join(_ROOT, "outputs", "validity", "rehab246_sequences.npy")
MAN_PATH = os.path.join(_ROOT, "outputs", "validity", "rehab246_manifest.csv")
FPS = 30.0                                                  # REHAB24-6 joints are exported at 30 fps
NF_LEVELS = [0, 1, 2, 4, 8]


# =============================================================================
# Metric: AUROC via the Mann-Whitney U statistic (no sklearn dependency)
# =============================================================================
def auroc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")                                # undefined for a single-class split
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # average ranks for ties
    s = y_score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (ranks[order[i]] + ranks[order[j]])
        i = j + 1
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# =============================================================================
# Data: REHAB24-6 -> KIMORE-format sample dicts (identical preprocessing)
# =============================================================================
def _preprocess(x):
    """Root-relative + median-radius scale norm -- byte-for-byte the KIMORE recipe
    (kimore_cde_data.load_sample lines 109-114). x: (T, 25, 3)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x[:, kd.ROOT_JOINT:kd.ROOT_JOINT + 1, :]        # translation invariance
    radius = np.linalg.norm(x, axis=-1)                    # (T, 25)
    scale = float(np.median(radius[radius > 0])) if np.any(radius > 0) else 1.0
    return x / max(scale, 1e-6)


def load_rehab246_samples(verbose=True):
    if not (os.path.exists(SEQ_PATH) and os.path.exists(MAN_PATH)):
        raise FileNotFoundError(
            f"Missing REHAB24-6 testbed. Build it first:\n"
            f"    python src/load_rehab246.py --build\n"
            f"  expected: {SEQ_PATH}\n            {MAN_PATH}")
    seqs = np.load(SEQ_PATH)                                # (N, 150, 25, 3)
    man = pd.read_csv(MAN_PATH)
    assert len(seqs) == len(man), f"seq/manifest length mismatch: {len(seqs)} vs {len(man)}"
    T = seqs.shape[1]
    t_uniform = (np.arange(T, dtype=np.float64) / FPS) / kd.TIME_SCALE   # start at 0, O(1) units

    ex_ids = sorted(man["exercise_id"].unique())
    ex_remap = {e: i + 1 for i, e in enumerate(ex_ids)}    # -> 1..K (collate subtracts 1)
    n_exercises = len(ex_ids)

    samples = []
    for k, row in man.iterrows():
        x = _preprocess(seqs[k])                           # (T, 25, 3)
        label = int(row["correct_label"])
        samples.append({
            "t": t_uniform.copy(),
            "x": x,
            "n_frames": T,
            "y": float(label),                             # BINARY target (0/1) used as BCE logit tgt
            "exercise": ex_remap[int(row["exercise_id"])], # 1-based; collate -> 0-based index
            "subject": int(row["subject_id"]),
            "cohort": "correct" if label == 1 else "incorrect",  # stratifies subject_folds
        })
    if verbose:
        pos = sum(s["y"] for s in samples)
        subs = sorted({s["subject"] for s in samples})
        print(f"[rehab246] {len(samples)} reps  (correct={int(pos)} / incorrect={len(samples)-int(pos)})  "
              f"subjects={len(subs)}  exercises={n_exercises}")
    return samples, n_exercises


def subject_folds_rehab246(samples, k=5, seed=0):
    """K subject-disjoint folds for REHAB24-6. Unlike KIMORE's kd.subject_folds (which stratifies
    by a control/patient cohort), every REHAB24-6 subject contributes BOTH correct and incorrect
    repetitions, so each subject-disjoint fold already holds both classes -- no cohort stratification
    (and kd.subject_folds' hard-coded control/patient cohorts do not apply here)."""
    rng = np.random.RandomState(seed)
    subs = sorted({s["subject"] for s in samples})
    rng.shuffle(subs)
    subj_fold = {u: j % k for j, u in enumerate(subs)}
    folds = [[] for _ in range(k)]
    for i, s in enumerate(samples):
        folds[subj_fold[s["subject"]]].append(i)
    return [sorted(f) for f in folds]


# =============================================================================
# EGRU arm -- the paper's SE(3)-equivariant recurrence as a binary classifier
# =============================================================================
@torch.no_grad()
def _logits_egru(model, samples, device, bs=8):
    model.eval()
    out = []
    for i in range(0, len(samples), bs):
        t, x, y, e, n = kd.collate(samples[i:i + bs], device=device)
        out.append(model(t, x, e, n).float().cpu().numpy())   # unbounded scalar == logit
    return np.concatenate(out) if out else np.array([])


def _build_egru(args, n_exercises, device):
    return SE3EquivariantGRU(
        n_scalar=args.n_scalar, n_vec=args.n_vec, n_layers=args.layers, lmax=args.lmax,
        n_joints=kd.N_JOINTS, gru_hidden=args.gru_hidden, dropout=args.dropout,
        n_exercises=n_exercises, use_speed=not args.no_speed, use_chiral=args.chiral).to(device)


def train_fold_egru(tr, va, args, n_exercises, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = _build_egru(args, n_exercises, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.BCEWithLogitsLoss()
    yv = np.array([s["y"] for s in va])
    best, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        idx = np.random.permutation(len(tr))
        for i in range(0, len(tr), args.batch_size):
            batch = [tr[j] for j in idx[i:i + args.batch_size]]
            t, x, y, e, n = kd.collate(batch, device=device)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(t, x, e, n), y)             # y in {0,1}; logits vs BCE
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
        sched.step()
        if ep % args.eval_every == 0 or ep == args.epochs - 1:
            a = auroc(yv, _logits_egru(model, va, device))
            if not np.isnan(a) and a > best:
                best = a
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# =============================================================================
# PCT baseline arm -- non-equivariant control for the accuracy tie
# =============================================================================
@torch.no_grad()
def _logits_pct(model, samples, n_frames, device, bs=8):
    model.eval()
    out = []
    for i in range(0, len(samples), bs):
        x, y, e = bt.batch_fixed_grid(samples[i:i + bs], n_frames, "linear", device=device)
        out.append(model(x, exercise_id=e).squeeze(-1).float().cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def train_fold_pct(tr, va, args, n_exercises, device):
    from models_curvenet import PointCloudTransformerRegressor
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = PointCloudTransformerRegressor(
        seq_len=args.n_frames, num_joints=kd.N_JOINTS, num_channels=3, dim=256,
        spatial_depth=6, temporal_depth=3, heads=4, dropout=0.1, k=10,
        num_exercises=n_exercises).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.BCEWithLogitsLoss()
    yv = np.array([s["y"] for s in va])
    best, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        idx = np.random.permutation(len(tr))
        for i in range(0, len(tr), args.batch_size):
            batch = [tr[j] for j in idx[i:i + args.batch_size]]
            x, y, e = bt.batch_fixed_grid(batch, args.n_frames, "linear", device=device)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(x, exercise_id=e).squeeze(-1), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
        sched.step()
        if ep % args.eval_every == 0 or ep == args.epochs - 1:
            a = auroc(yv, _logits_pct(model, va, args.n_frames, device))
            if not np.isnan(a) and a > best:
                best = a
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# =============================================================================
# Evaluation: clean AUROC, node-failure grace, viewpoint certificate
# =============================================================================
def _predict(model, samples, args, device):
    if args.model == "egru":
        return _logits_egru(model, samples, device)
    return _logits_pct(model, samples, args.n_frames, device)


def evaluate_fold(model, te, args, device, draws=3):
    y = np.array([s["y"] for s in te])
    clean_logits = _predict(model, te, args, device)
    clean_auroc = auroc(y, clean_logits)

    # (b) node-failure: AUROC vs number of dead joints ('hold' fault), averaged over draws
    nf = {0: clean_auroc}
    for k in NF_LEVELS[1:]:
        accs = []
        for dd in range(draws):
            corrupted = [fail_joints(s, k, "hold", seed=dd * 977 + i) for i, s in enumerate(te)]
            accs.append(auroc(y, _predict(model, corrupted, args, device)))
        nf[k] = float(np.nanmean(accs))
    nf_lost = nf[0] - nf[max(NF_LEVELS)]                    # AUROC drop 0->8 (smaller = more graceful)

    # (c) viewpoint certificate: rotate the camera 90 deg about gravity; logits must be invariant
    rot = [bt.rotate_sample(s, 90.0) for s in te]
    rot_logits = _predict(model, rot, args, device)
    view_drift = float(np.max(np.abs(rot_logits - clean_logits))) if len(clean_logits) else float("nan")
    view_auroc = auroc(y, rot_logits)

    return {
        "clean_auroc": clean_auroc,
        "nf_auroc": nf,
        "nf_lost_0to8": nf_lost,
        "view_logit_drift": view_drift,       # ~0 for EGRU (theorem); large for PCT
        "view_auroc_90deg": view_auroc,
    }


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="REHAB24-6 second-corpus replication (WACV Workstream C)")
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
    # EGRU arch (mirrors src/train_egru.py defaults)
    ap.add_argument("--n-scalar", type=int, default=32)
    ap.add_argument("--n-vec", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lmax", type=int, default=2)
    ap.add_argument("--gru-hidden", type=int, default=128)
    ap.add_argument("--no-speed", action="store_true")
    ap.add_argument("--chiral", action="store_true", help="admit parity-odd channels (deployed model)")
    # PCT fixed-grid length
    ap.add_argument("--n-frames", type=int, default=100)
    ap.add_argument("--smoke", action="store_true", help="1 fold x few epochs for a wiring check")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.cv = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples, n_exercises = load_rehab246_samples()
    folds = subject_folds_rehab246(samples, k=args.folds, seed=args.seed)
    fold_ids = range(args.folds) if args.cv else [args.fold]

    print(f"\n{'='*76}\nREHAB24-6 replication  model={args.model}  chiral={args.chiral}  "
          f"seed={args.seed}  device={device}\n  (writes to {OUT})\n{'='*76}")
    rows = []
    for f in fold_ids:
        # kd.split returns (train, val, test); its val fold drives early stopping
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
        "model": args.model, "chiral": args.chiral, "seed": args.seed, "n_exercises": n_exercises,
        "n_params": int(n_params), "folds": [r["fold"] for r in rows],
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
    out_path = os.path.join(OUT, f"rehab246_{args.model}{tag}_s{args.seed}.json")
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
