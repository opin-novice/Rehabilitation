#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
invariant_controls.py
=====================
THE CELL THAT ISOLATES THE CAUSE.

Established, all on the same subject-disjoint pooled folds:

    model                        MAD    deep?  invariant?  continuous-time flow?
    ------------------------- ------- ------ ----------- ----------------------
    PCT (target architecture)    6.07    yes         NO                      no
    ridge on SO(3) invariants    7.68     no        yes                      no
    per-exercise mean floor      8.25      -          -                       -
    SE(3)-CDE (message-passing)  8.43    yes        yes                     YES
    SE(3)-CDE (global latent)    8.19    yes        yes                     YES

So invariant FEATURES carry signal (ridge beats the floor) but our deep invariant FLOW does not
(CI [-0.560, +0.864] -- indistinguishable from the mean). Three explanations remain, and they
imply very different papers:

  (a) EQUIVARIANCE is the problem -- deep models cannot exploit invariant structure here.
  (b) The CONTINUOUS-TIME CDE FORMULATION is the problem at this sample size.
  (c) Deep models of any kind overfit 45 subjects, and the ridge wins by being linear.

This script supplies the missing controls, both fully SO(3)-invariant and neither a flow:

  * InvariantMLP : deep net on the SAME aggregate invariant features the ridge used.
                   Deep + invariant + NO temporal model. Separates (c) from (a)/(b).
  * InvariantGRU : GRU over PER-FRAME invariant features (bone lengths, joint radii, key
                   inter-joint distances, speeds). Deep + invariant + temporal, but DISCRETE
                   time. This is the discrete-time twin of our CDE with identical information,
                   and it is the decisive control:

        GRU beats the floor  and  CDE does not   ->  (b): the CDE FLOW is what fails.
        GRU also fails,      ridge still wins    ->  (c): deep models overfit 45 subjects;
                                                     the story is sample size, not geometry.
        both fail badly                          ->  (a) is back on the table.

Every feature below is a function of inner products of joint coordinates only, so all of these
controls are exactly as SO(3)-invariant as our CDE. The comparison is clean.

Run:  python src/invariant_controls.py --cv
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
import chirality as ch                            # noqa: E402
import kimore_cde_data as kd                      # noqa: E402
from train_cde import metrics, fmt                # noqa: E402
from gravity_probe import so3_invariant_features  # noqa: E402
from cde_model_mp import KINECT_BONES             # noqa: E402
from protocol_null import bootstrap_delta         # noqa: E402

SCORE_MAX = kd.SCORE_MAX
KEY_PAIRS = [(6, 10), (6, 3), (10, 3), (5, 9), (14, 18), (7, 11)]   # wrists/head/elbows/ankles


# =============================================================================
# Per-frame SO(3)-invariant time series  (what the GRU sees)
# =============================================================================
def invariant_series(s, chiral=False, oracle_mask=False):
    """(L, F) time series of purely rotation-invariant quantities. Same information class the
    CDE's invariant read-out is entitled to -- inner products of joint coordinates only.

    Note every feature here is a NORM or a DISTANCE, hence parity-EVEN: like the EGRU's original
    invariant cut, this control was silently invariant under the full O(3) rather than the SO(3)
    it advertised. chiral=True appends the anatomical signed volumes (parity-ODD pseudo-scalars),
    which is the same information the chiral EGRU receives. Both arms must be upgraded together
    or the co-headline comparison stops being a comparison.

    oracle_mask: FAIRNESS CONTROL for the node-failure experiment (joint_failure.py), off by
    default and unused by the primary (headline) InvariantGRU pathway. When True and `s["dead"]`
    lists failed joint indices, any feature that reads a failed joint is zeroed at inference --
    the exact same principle the paper states for EGRU's liveness gate ("zeroes any generator
    touching a failed joint"), applied here so the hand-crafted arm gets the identical oracle
    knowledge of which node died. No retraining: same checkpoint, same standardisation, only the
    forward-pass features change.
    """
    x = np.asarray(s["x"], dtype=np.float64)                    # (L, 25, 3) root-relative
    L = x.shape[0]
    bones = np.stack([np.linalg.norm(x[:, a] - x[:, b], axis=-1)
                      for a, b in KINECT_BONES], axis=-1)       # (L, 24)
    radii = np.linalg.norm(x, axis=-1)                          # (L, 25)
    pairs = np.stack([np.linalg.norm(x[:, a] - x[:, b], axis=-1)
                      for a, b in KEY_PAIRS], axis=-1)          # (L, 6)
    speed = np.zeros((L, 25))
    speed[1:] = np.linalg.norm(np.diff(x, axis=0), axis=-1)     # (L, 25) invariant
    dt = np.diff(np.asarray(s["t"], dtype=np.float64), prepend=s["t"][0])[:, None]

    if oracle_mask and len(s.get("dead", [])):
        dead = np.asarray(s["dead"], dtype=int)
        live = np.ones(kd.N_JOINTS, dtype=np.float64)
        live[dead] = 0.0
        bones *= np.array([live[a] * live[b] for a, b in KINECT_BONES])[None, :]
        radii = radii * live[None, :]
        pairs *= np.array([live[a] * live[b] for a, b in KEY_PAIRS])[None, :]
        speed = speed * live[None, :]
        # dt is global (not joint-indexed) -- never zeroed.

    feats = [bones, radii, pairs, speed, dt]
    if chiral:
        chi = ch.anatomic_pseudoscalars(x)                      # (L, N_CHIRAL) parity-ODD
        if oracle_mask and len(s.get("dead", [])):
            chi = chi * ch.frame_liveness(live)[None, :]         # dead if ANY of its 6 joints died
        feats.append(chi)
    return np.nan_to_num(np.concatenate(feats, axis=-1))


class InvariantGRU(nn.Module):
    """Deep + invariant + temporal, but DISCRETE time. The control that indicts (or exonerates)
    the continuous-time CDE formulation."""

    def __init__(self, n_feat, hidden=128, layers=1, n_exercises=5, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(n_feat + n_exercises, hidden, layers,
                          batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, 128), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(128, 1))
        self.n_ex = n_exercises

    def forward(self, x, ex):
        oh = torch.nn.functional.one_hot(ex, self.n_ex).to(x.dtype)
        x = torch.cat([x, oh.unsqueeze(1).expand(-1, x.shape[1], -1)], dim=-1)
        h, _ = self.gru(x)
        return self.head(h.mean(1)).squeeze(-1)


class InvariantMLP(nn.Module):
    """Deep + invariant + NO temporal model: the ridge's features, nonlinearly."""

    def __init__(self, n_feat, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =============================================================================
def pad_series(samples, device, chiral=False, oracle_mask=False):
    seqs = [invariant_series(s, chiral=chiral, oracle_mask=oracle_mask) for s in samples]
    L = max(len(q) for q in seqs)
    F = seqs[0].shape[1]
    X = np.zeros((len(seqs), L, F))
    for i, q in enumerate(seqs):
        X[i, : len(q)] = q
    y = np.array([s["y"] for s in samples])
    e = np.array([s["exercise"] - 1 for s in samples])
    return (torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.float32, device=device),
            torch.tensor(e, dtype=torch.long, device=device))


def agg_features(samples, device):
    X = []
    for s in samples:
        inv = so3_invariant_features(s["x"])
        oh = np.zeros(5)
        oh[s["exercise"] - 1] = 1.0
        X.append(np.concatenate([inv, oh]))
    X = np.nan_to_num(np.stack(X))
    y = np.array([s["y"] for s in samples])
    return (torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.float32, device=device))


def train_one(kind, tr, va, te, device, epochs=120, lr=1e-3, seed=0, chiral=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if kind == "gru":
        Xtr, ytr, etr = pad_series(tr, device, chiral)
        Xva, yva, eva = pad_series(va, device, chiral)
        Xte, yte, ete = pad_series(te, device, chiral)
        mu, sd = Xtr.mean((0, 1), keepdim=True), Xtr.std((0, 1), keepdim=True) + 1e-6
        Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
        model = InvariantGRU(Xtr.shape[-1]).to(device)
        fwd = lambda X, i, e: model(X[i], e[i])                      # noqa: E731
        packs = ((Xtr, ytr, etr), (Xva, yva, eva), (Xte, yte, ete))
    else:
        Xtr, ytr = agg_features(tr, device)
        Xva, yva = agg_features(va, device)
        Xte, yte = agg_features(te, device)
        mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
        Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
        model = InvariantMLP(Xtr.shape[-1]).to(device)
        fwd = lambda X, i, e: model(X[i])                            # noqa: E731
        packs = ((Xtr, ytr, None), (Xva, yva, None), (Xte, yte, None))

    (Xtr, ytr, etr), (Xva, yva, eva), (Xte, yte, ete) = packs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    huber = nn.HuberLoss(delta=0.1)
    n = len(tr)

    best, best_pred, best_state = math.inf, None, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 16):
            idx = perm[i: i + 16]
            opt.zero_grad(set_to_none=True)
            huber(fwd(Xtr, idx, etr), ytr[idx]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            iv = torch.arange(len(va), device=device)
            it = torch.arange(len(te), device=device)
            mv = metrics(fwd(Xva, iv, eva).cpu().numpy(), yva.cpu().numpy())["MAD"]
            if mv < best:
                best = mv
                best_pred = fwd(Xte, it, ete).cpu().numpy()
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Persist weights AND the train-split feature standardisation. Block 2/3 reload this model,
    # and re-fitting mu/sd on corrupted test data would both leak and silently change the input
    # distribution -- the numbers would be meaningless. The scaler is part of the checkpoint.
    ckpt = {"state": best_state, "mu": mu.cpu(), "sd": sd.cpu(),
            "n_feat": int(Xtr.shape[-1]), "kind": kind, "chiral": bool(chiral)}
    return best_pred, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chiral", action="store_true",
                    help="append parity-ODD anatomical signed volumes: O(3) -> SO(3), matching "
                         "the chiral EGRU's information exactly")
    ap.add_argument("--out", type=str, default="outputs/cde_block2")
    args = ap.parse_args()

    tag = "invgruchi" if args.chiral else "invgru"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = kd.load_all_exercises(verbose=False)
    # Fold partition follows the seed, exactly as train_egru / train_baseline_pct do. Hardcoding
    # seed=0 here left invgru's s1/s2 checkpoints on the seed-0 partition while egru/pct used their
    # own -- so a multi-seed sweep would evaluate them on mismatched folds (a subject leak).
    folds = kd.subject_folds(S, k=args.folds, seed=args.seed)

    print("=" * 78)
    print("INVARIANT CONTROLS -- is it EQUIVARIANCE, the CDE FLOW, or the SAMPLE SIZE?")
    print("=" * 78)
    print("Both controls are exactly as SO(3)-invariant as our CDE. Neither is a flow.\n")

    err = {k: [] for k in ("mlp", "gru", "floor")}
    subj = []
    per_fold = {k: [] for k in ("mlp", "gru", "floor")}
    for f in range(args.folds):
        tr, va, te = kd.split(S, folds, test_fold=f, val_fold=(f + 1) % args.folds)
        y = np.array([s["y"] for s in te])
        fl = kd.exercise_mean_floor(tr, te)
        preds = {"floor": fl}
        for kind in ("mlp", "gru"):
            preds[kind], ckpt = train_one(kind, tr, va, te, device,
                                          epochs=args.epochs, seed=args.seed,
                                          chiral=args.chiral)
            if kind == "gru":
                os.makedirs(args.out, exist_ok=True)
                torch.save(ckpt, os.path.join(args.out, f"{tag}_s{args.seed}_pooled_f{f}.pt"))
        for k in ("mlp", "gru", "floor"):
            per_fold[k].append(metrics(preds[k], y)["MAD"])
            err[k] += list((preds[k] - y) * SCORE_MAX)
        subj += [s["subject"] for s in te]
        print(f"  fold {f}:  MLP-inv {per_fold['mlp'][-1]:6.3f}   "
              f"GRU-inv {per_fold['gru'][-1]:6.3f}   floor {per_fold['floor'][-1]:6.3f}")

    print(f"\n{'-'*78}")
    print(f"  {'model':<34s} {'MAD':>7s} {'+/-':>6s}   {'vs floor (95% CI, subj-cluster)':>34s}")
    print(f"  {'-'*34} {'-'*7} {'-'*6}   {'-'*34}")
    fl = np.array(per_fold["floor"])
    print(f"  {'per-exercise mean floor':<34s} {fl.mean():7.3f} {fl.std():6.3f}")
    for k, label in (("mlp", "MLP on invariants (deep, no flow)"),
                     ("gru", "GRU on invariants (deep, discrete)")):
        a = np.array(per_fold[k])
        d, lo, hi, n = bootstrap_delta(err[k], err["floor"], subj)
        verdict = "BEATS floor" if hi < 0 else ("WORSE" if lo > 0 else "indistinguishable")
        print(f"  {label:<34s} {a.mean():7.3f} {a.std():6.3f}   "
              f"{d:+.3f} [{lo:+.3f},{hi:+.3f}]  {verdict}")

    print("\n  reference (same folds):  ridge-inv 7.678   SE(3)-CDE 8.427   PCT 6.065")
    print("-" * 78)

    gru = np.array(per_fold["gru"]).mean()
    mlp = np.array(per_fold["mlp"]).mean()
    _, _, hi_g, _ = bootstrap_delta(err["gru"], err["floor"], subj)
    if hi_g < 0:
        print("VERDICT (b): a DISCRETE-time deep invariant model beats the floor while our")
        print("CONTINUOUS-TIME CDE does not. Equivariance is exonerated; the CDE FLOW is what")
        print("fails at this sample size. That is a precise, publishable negative result.")
    elif min(gru, mlp) > fl.mean() - 0.2:
        print("VERDICT (c): NO deep invariant model beats the floor -- only the linear ridge does.")
        print("The failure is DEEP MODELS ON 45 SUBJECTS, not equivariance and not the CDE.")
        print("The CDE is not uniquely at fault; the sample size is.")
    else:
        print("Mixed: see the per-model CIs above before drawing a conclusion.")
    print("-" * 78)

    with open(os.path.join(args.out, f"invariant_controls_{tag}_s{args.seed}.json"
                           if args.chiral else
                           f"invariant_controls_s{args.seed}.json"), "w") as fh:
        json.dump({"seed": args.seed, "chiral": bool(args.chiral), "per_fold": per_fold}, fh,
                  indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
