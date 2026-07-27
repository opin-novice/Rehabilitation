#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sandbox_train.py  --  RESEARCH SANDBOX (NOT for the paper)
==========================================================
Trains the isolated baselines (EGNN recurrence, or canonicalizer+PCT) with the SAME recipe as the
paper's trainers, but writing checkpoints ONLY under research_egnn/outputs/. It never touches src/
or outputs/cde_block2/.

    python research_egnn/sandbox_train.py --model egnn   --cv        # E(n)-equivariant arm
    python research_egnn/sandbox_train.py --model canon  --cv        # PCA-canonicalize -> PCT arm

Budget: 1 seed (0) x 5 folds (triage). Data loaders, collate, splits, floor, and metrics are
imported READ-ONLY from the existing code.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
for p in (_HERE, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import kimore_cde_data as kd                                  # noqa: E402
import block2_transforms as bt                               # noqa: E402
from train_cde import metrics                                # noqa: E402
from models_curvenet import PointCloudTransformerRegressor   # noqa: E402
from egnn_model import EGNNRecurrence                        # noqa: E402
from canonicalize import pca_canonicalize                    # noqa: E402

# ----- ISOLATION GUARD: every write goes here and nowhere else -----------------------------------
OUT = os.path.join(_HERE, "outputs")
assert os.path.abspath(OUT).replace("\\", "/").endswith("research_egnn/outputs"), \
    f"refusing to run: output dir {OUT} is not the sandbox"
os.makedirs(OUT, exist_ok=True)


def config_tag(layers=4, hidden=64, lr=1e-3, clamp=None):
    """Checkpoint/JSON tag for an EGNN config. Backward-compatible: the deployed defaults
    (L4, h64, lr1e-3, no clamp) give "" so existing checkpoints/the done clamp sweep still resolve;
    clamp-only arms keep their historical `_c{clamp}` names. Only non-default axes append a token."""
    t = ""
    if layers != 4:
        t += f"_L{layers}"
    if hidden != 64:
        t += f"_h{hidden}"
    if lr != 1e-3:
        t += f"_lr{lr:g}"
    if clamp is not None:
        t += f"_c{clamp:g}"
    return t


# =============================================================================
# EGNN arm -- the recurrence recipe (mirrors src/train_egru.train_fold)
# =============================================================================
@torch.no_grad()
def _pred_egnn(model, samples, device, bs=8):
    model.eval()
    out = []
    for i in range(0, len(samples), bs):
        t, x, y, e, n = kd.collate(samples[i: i + bs], device=device)
        out.append(model(t, x, e, n).float().cpu().numpy())
    return np.concatenate(out)


def train_egnn_fold(tr, va, te, args, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = EGNNRecurrence(n_scalar=32, n_vec=8, n_layers=args.egnn_layers, egnn_hidden=args.egnn_hidden,
                           dropout=0.2, n_exercises=5, use_speed=True, use_chiral=False,
                           coord_clamp=args.coord_clamp).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.HuberLoss(delta=0.1)
    yv = np.array([s["y"] for s in va])
    best, best_state = 1e9, None
    for ep in range(args.epochs):
        model.train()
        idx = np.random.permutation(len(tr))
        for i in range(0, len(tr), args.bs):
            batch = [tr[j] for j in idx[i: i + args.bs]]
            t, x, y, e, n = kd.collate(batch, device=device)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(t, x, e, n), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep % 4 == 0 or ep == args.epochs - 1:
            mval = metrics(_pred_egnn(model, va, device), yv)["MAD"]
            if mval < best:
                best = mval
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


# =============================================================================
# Canonicalization arm -- PCA frame -> PCT (mirrors src/train_baseline_pct)
# =============================================================================
@torch.no_grad()
def _pred_canon(model, samples, n_frames, device, bs=8):
    model.eval()
    canon = [pca_canonicalize(s) for s in samples]
    out = []
    for i in range(0, len(canon), bs):
        x, y, e = bt.batch_fixed_grid(canon[i: i + bs], n_frames, "linear", device=device)
        out.append(model(x, exercise_id=e).squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)


def train_canon_fold(tr, va, te, args, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    trc = [pca_canonicalize(s) for s in tr]                   # canonicalize the training data once
    # 0, NOT 5: the banked canon_pct_*.pt this trains are loaded by sandbox_eval / pareto_k2_eval,
    # and were produced when num_exercises was accepted and IGNORED. Training conditioned here
    # would silently fork the canonicalization arm from the checkpoints it is compared against.
    model = PointCloudTransformerRegressor(
        seq_len=args.n_frames, num_joints=kd.N_JOINTS, num_channels=3, dim=256,
        spatial_depth=6, temporal_depth=3, heads=4, dropout=0.1, k=10, num_exercises=0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.canon_epochs)
    lossf = nn.HuberLoss(delta=0.1)
    yv = np.array([s["y"] for s in va])
    best, best_state = 1e9, None
    for ep in range(args.canon_epochs):
        model.train()
        idx = np.random.permutation(len(trc))
        for i in range(0, len(trc), args.bs):
            batch = [trc[j] for j in idx[i: i + args.bs]]
            x, y, e = bt.batch_fixed_grid(batch, args.n_frames, "linear", device=device)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(x, exercise_id=e).squeeze(-1), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if ep % 4 == 0 or ep == args.canon_epochs - 1:
            mval = metrics(_pred_canon(model, va, args.n_frames, device), yv)["MAD"]
            if mval < best:
                best = mval
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["egnn", "canon"], required=True)
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=80, help="EGNN epochs")
    ap.add_argument("--canon-epochs", type=int, default=60, help="PCT epochs")
    ap.add_argument("--egnn-layers", type=int, default=4)
    ap.add_argument("--egnn-hidden", type=int, default=64, help="EGNN message MLP width (sweep 64/128)")
    ap.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate (sweep e.g. 1e-3/5e-4)")
    ap.add_argument("--coord-clamp", type=float, default=None,
                    help="EGNN coordinate-update coefficient clamp (None=off; sweep 0.1/0.5/1.0)")
    ap.add_argument("--n-frames", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=150)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = kd.load_all_exercises(max_len=args.max_len, verbose=False)
    folds = kd.subject_folds(S, k=args.folds, seed=args.seed)
    fold_ids = range(args.folds) if args.cv else [args.fold]

    print(f"\n{'='*72}\nSANDBOX TRAIN  model={args.model}  seed={args.seed}  (writes to {OUT})\n{'='*72}")
    rows = []
    for f in fold_ids:
        tr, va, te = kd.split(S, folds, test_fold=f, val_fold=(f + 1) % args.folds)
        y = np.array([s["y"] for s in te])
        floor = metrics(kd.exercise_mean_floor(tr, te), y)["MAD"]
        if args.model == "egnn":
            model = train_egnn_fold(tr, va, te, args, device)
            pred = _pred_egnn(model, te, device)
            ctag = config_tag(args.egnn_layers, args.egnn_hidden, args.lr, args.coord_clamp)
            ck = f"egnn{ctag}_s{args.seed}_f{f}.pt"
        else:
            model = train_canon_fold(tr, va, te, args, device)
            pred = _pred_canon(model, te, args.n_frames, device)
            ck = f"canon_pct_s{args.seed}_f{f}.pt"
        mad = metrics(pred, y)["MAD"]
        torch.save(model.state_dict(), os.path.join(OUT, ck))
        rows.append({"fold": f, "mad": float(mad), "floor": float(floor), "ckpt": ck})
        print(f"  fold {f}: MAD {mad:6.3f}   floor {floor:6.3f}   -> {ck}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    mean_mad = float(np.mean([r["mad"] for r in rows]))
    ctag = config_tag(args.egnn_layers, args.egnn_hidden, args.lr, args.coord_clamp) if args.model == "egnn" else ""
    print(f"{'-'*72}\n  {args.model}{ctag} pooled clean MAD (mean over {len(rows)} folds) = {mean_mad:.3f}")
    with open(os.path.join(OUT, f"train_{args.model}{ctag}_s{args.seed}.json"), "w") as fh:
        json.dump({"args": vars(args), "rows": rows, "mean_mad": mean_mad}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
