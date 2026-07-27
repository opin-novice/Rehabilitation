#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pooled_bootstrap_eval.py
=========================
WACV review W4: the pooled main-result table (tab:accuracy) reports EGRU / InvariantGRU / PCT
accuracy as mean +/- std over 3 seeds with NO confidence interval, while the single-exercise
protocol audit (protocol_null.py) reports a paired subject-cluster bootstrap 95% CI. The reviewer:
"Reporting the rigorous interval only where it looks unflattering, and a bare mean where it looks
fine, invites suspicion."

This closes the gap WITHOUT retraining. Every model's per-fold, per-seed checkpoint is already on
disk (outputs/cde_block2/{pct_pooled,egru,egruchi,invgru,invgruchi}_s{seed}_..._f{f}.pt), saved by
train_baseline_pct.py / train_egru.py / invariant_controls.py respectively -- all three call
kd.subject_folds(S, k=5, seed=seed) the SAME way, so a given (seed, fold) names an IDENTICAL
held-out subject set across all five model variants. We reload each checkpoint, run inference on
its own held-out fold, and pool per-sequence errors across all 3 seeds x 5 folds per model, then
hand them to protocol_null.bootstrap_delta -- the SAME function already used and cited for the
PCT-vs-floor pooled result in results.tex.

Run:  python src/pooled_bootstrap_eval.py
"""

import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimore_cde_data as kd                              # noqa: E402
import block2_transforms as bt                             # noqa: E402
from models_curvenet import build_pct_for_checkpoint        # noqa: E402
from equivariant_gru import SE3EquivariantGRU               # noqa: E402
from invariant_controls import InvariantGRU, invariant_series  # noqa: E402
from protocol_null import bootstrap_delta                   # noqa: E402

SCORE_MAX = kd.SCORE_MAX
OUT_DIR = "outputs/cde_block2"
SEEDS = (0, 1, 2)
FOLDS = 5
N_FRAMES = 100

VARIANTS = ("pct", "egru", "egruchi", "invgru", "invgruchi")


# =============================================================================
# Per-variant: reconstruct model from a bare checkpoint, run inference on `te`
# =============================================================================
def load_pct(seed, fold, device):
    path = os.path.join(OUT_DIR, f"pct_pooled_s{seed}_f{fold}.pt")
    sd = torch.load(path, map_location=device)
    # Conditioning is read off the checkpoint, never assumed: the banked pooled runs are
    # UNconditioned (head width 256) because num_exercises was ignored when they were trained.
    model = build_pct_for_checkpoint(
        sd, seq_len=N_FRAMES, num_joints=kd.N_JOINTS, num_channels=3,
        dim=256, spatial_depth=6, temporal_depth=3, heads=4, dropout=0.1, k=10,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


@torch.no_grad()
def predict_pct(model, samples, device):
    x, _, e = bt.batch_fixed_grid(samples, N_FRAMES, "linear", device=device)
    return model(x, exercise_id=e).squeeze(-1).float().cpu().numpy()


def load_egru(seed, fold, device, chiral):
    model = SE3EquivariantGRU(
        n_scalar=32, n_vec=8, n_layers=2, lmax=2, gru_hidden=128, dropout=0.2,
        n_exercises=5, use_speed=True, use_chiral=chiral,
    ).to(device)
    tag = "egruchi" if chiral else "egru"
    path = os.path.join(OUT_DIR, f"{tag}_s{seed}_pooled_f{fold}.pt")
    # strict=False: `encoder.dead_scalar` was added later for the occlusion-aware (use_mask=True)
    # variant and is never read when use_mask=False (the published/headline model, used here) --
    # see equivariant_gru.py forward(): the mask branch that consumes it is skipped when mask=None.
    missing, unexpected = model.load_state_dict(torch.load(path, map_location=device), strict=False)
    assert list(missing) == ["encoder.dead_scalar"] and not unexpected, (missing, unexpected)
    model.eval()
    return model


@torch.no_grad()
def predict_egru(model, samples, device, bs=8):
    out = []
    for i in range(0, len(samples), bs):
        t, x, _, e, n = kd.collate(samples[i: i + bs], device=device)
        out.append(model(t, x, e, n).float().cpu().numpy())
    return np.concatenate(out)


def load_invgru(seed, fold, device, chiral):
    tag = "invgruchi" if chiral else "invgru"
    path = os.path.join(OUT_DIR, f"{tag}_s{seed}_pooled_f{fold}.pt")
    ckpt = torch.load(path, map_location=device)
    model = InvariantGRU(ckpt["n_feat"]).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    # Older checkpoints predate the "chiral" key; the filename tag (invgru vs invgruchi) is
    # itself the ground truth for which arm was trained, so only check when the key is present.
    if "chiral" in ckpt:
        assert bool(ckpt["chiral"]) == chiral, "checkpoint/variant chirality mismatch"
    return model, ckpt["mu"].to(device), ckpt["sd"].to(device)


@torch.no_grad()
def predict_invgru(model, mu, sd, chiral, samples, device):
    seqs = [invariant_series(s, chiral=chiral) for s in samples]
    L = max(len(q) for q in seqs)
    F = seqs[0].shape[1]
    X = np.zeros((len(seqs), L, F))
    for i, q in enumerate(seqs):
        X[i, : len(q)] = q
    X = torch.tensor(X, dtype=torch.float32, device=device)
    X = (X - mu) / sd
    e = torch.tensor([s["exercise"] - 1 for s in samples], dtype=torch.long, device=device)
    return model(X, e).float().cpu().numpy()


# =============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    S = kd.load_all_exercises(max_len=150, verbose=False)

    err = {v: [] for v in VARIANTS}
    err_floor = {v: [] for v in VARIANTS}
    subj = {v: [] for v in VARIANTS}
    per_slice_mad = {v: [] for v in VARIANTS}
    # Raw out-of-fold predictions, banked so that any future rank/correlation statistic is a CPU
    # re-analysis instead of another GPU pass over 75 checkpoints. Keyed by seed because each seed
    # is a COMPLETE 5-fold OOF partition of the corpus: rho is computed once per seed on the full
    # vector, never per fold (a 15-sequence fold gives a rank correlation dominated by noise).
    oof = {s: {"y": [], "exercise": [], "subject": [], "floor": [],
               **{v: [] for v in VARIANTS}} for s in SEEDS}

    for seed in SEEDS:
        folds = kd.subject_folds(S, k=FOLDS, seed=seed)
        for f in range(FOLDS):
            tr, va, te = kd.split(S, folds, test_fold=f, val_fold=(f + 1) % FOLDS)
            y = np.array([s["y"] for s in te], dtype=np.float64)
            floor = kd.exercise_mean_floor(tr, te)
            subjects = [s["subject"] for s in te]
            floor_err = list((floor - y) * SCORE_MAX)

            preds = {}
            preds["pct"] = predict_pct(load_pct(seed, f, device), te, device)
            preds["egru"] = predict_egru(load_egru(seed, f, device, chiral=False), te, device)
            preds["egruchi"] = predict_egru(load_egru(seed, f, device, chiral=True), te, device)
            m, mu, sd = load_invgru(seed, f, device, chiral=False)
            preds["invgru"] = predict_invgru(m, mu, sd, False, te, device)
            m, mu, sd = load_invgru(seed, f, device, chiral=True)
            preds["invgruchi"] = predict_invgru(m, mu, sd, True, te, device)

            oof[seed]["y"] += list(y * SCORE_MAX)
            oof[seed]["exercise"] += [int(s["exercise"]) for s in te]
            oof[seed]["subject"] += list(subjects)
            oof[seed]["floor"] += list(np.asarray(floor, dtype=np.float64) * SCORE_MAX)

            for v in VARIANTS:
                p = np.asarray(preds[v], dtype=np.float64)
                mad = float(np.mean(np.abs(p - y)) * SCORE_MAX)
                per_slice_mad[v].append(mad)
                err[v] += list((p - y) * SCORE_MAX)
                err_floor[v] += floor_err
                subj[v] += subjects
                oof[seed][v] += list(p * SCORE_MAX)

            print(f"  seed {seed} fold {f}: " +
                  "  ".join(f"{v}={per_slice_mad[v][-1]:.3f}" for v in VARIANTS))

    print(f"\n{'-'*78}\nPOOLED OVER {len(SEEDS)} SEEDS x {FOLDS} FOLDS "
          f"(paired subject-cluster bootstrap, matching protocol_null.bootstrap_delta)\n{'-'*78}")

    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for v in VARIANTS:
        d, lo, hi, n_sub = bootstrap_delta(err[v], err_floor[v], subj[v])
        mad_mean = float(np.mean(per_slice_mad[v]))
        mad_std = float(np.std(per_slice_mad[v]))
        verdict = "CLEARS the floor" if hi < 0 else ("WORSE than floor" if lo > 0 else "STRADDLES 0")
        print(f"  {v:10s}  MAD {mad_mean:6.3f} +/- {mad_std:5.3f}   "
              f"delta {d:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  (n_subj={n_sub})  -> {verdict}")
        results[v] = {"mad_mean": mad_mean, "mad_std": mad_std,
                       "bootstrap": {"delta": d, "lo": lo, "hi": hi,
                                     "n_subjects_resampled": n_sub, "unit": "subject-cluster"}}
        with open(os.path.join(OUT_DIR, f"protocol_audit_{v}_pooled3seed.json"), "w") as fh:
            json.dump({"variant": v, "seeds": list(SEEDS), "folds": FOLDS,
                       **results[v]}, fh, indent=2)

    # ---- Spearman rho, reported against its own null -------------------------------------------
    # KIMORE's literature reports rho (0.74-0.965) and we report MAD, which reads as evasive even
    # though cross-paper rho is not protocol-comparable. So we report it -- with the mean-predictor
    # floor's rho beside it, exactly as MAD is reported against the floor. Without that null a rho
    # cannot be read: the floor is a model that ignores its input entirely.
    #
    # Two flavours, because they answer different questions:
    #   pooled  -- rho over all 5 exercises at once. Inflated by between-exercise score differences,
    #              though only mildly here (between-exercise variance is 4.8% of the total).
    #   within  -- rho inside each exercise, averaged over the five. This is the like-for-like
    #              comparison to the literature, whose numbers come from single-exercise models.
    # The floor's within-exercise rho is the diagnostic: it predicts one constant per exercise per
    # fold, so any departure from 0 is fold-to-fold drift in the training mean, not signal.
    def rho_pair(pred, y, ex):
        pooled = float(spearmanr(pred, y).statistic)
        per_ex = [float(spearmanr(pred[ex == e], y[ex == e]).statistic) for e in np.unique(ex)]
        return pooled, float(np.mean(per_ex))

    print(f"\n{'-'*78}\nSPEARMAN rho (per-seed OOF, mean +/- std over "
          f"{len(SEEDS)} seeds)\n{'-'*78}")
    print(f"  {'model':<12s} {'pooled rho':>18s} {'within-exercise rho':>22s}")
    rho_out = {}
    for v in list(VARIANTS) + ["floor"]:
        pooled, within = [], []
        for s in SEEDS:
            y_s = np.asarray(oof[s]["y"], dtype=np.float64)
            ex_s = np.asarray(oof[s]["exercise"], dtype=int)
            p_s = np.asarray(oof[s][v], dtype=np.float64)
            a, b = rho_pair(p_s, y_s, ex_s)
            pooled.append(a)
            within.append(b)
        rho_out[v] = {"pooled_mean": float(np.mean(pooled)), "pooled_std": float(np.std(pooled)),
                      "within_exercise_mean": float(np.mean(within)),
                      "within_exercise_std": float(np.std(within)),
                      "per_seed_pooled": pooled, "per_seed_within": within}
        print(f"  {v:<12s} {np.mean(pooled):9.4f} +/- {np.std(pooled):.4f}"
              f" {np.mean(within):13.4f} +/- {np.std(within):.4f}")
    print("\n  The floor row is the null: a model that ignores its input. Read every rho above it")
    print("  as an increment over that, not as an absolute. Cross-paper rho remains non-comparable")
    print("  (protocol, not architecture, sets it) -- reported so the comparison can be refused")
    print("  on the evidence rather than by omission.")

    with open(os.path.join(OUT_DIR, "spearman_pooled3seed.json"), "w") as fh:
        json.dump({"seeds": list(SEEDS), "folds": FOLDS, "rho": rho_out,
                   "note": "per-seed OOF; 'within' averages rho over the 5 exercises"}, fh, indent=2)
    # Bank the raw OOF predictions so no future rank statistic needs the GPU again.
    with open(os.path.join(OUT_DIR, "oof_predictions_pooled3seed.json"), "w") as fh:
        json.dump({"seeds": list(SEEDS), "folds": FOLDS, "score_max": SCORE_MAX,
                   "units": "clinical 0-50 scale", "oof": oof}, fh)

    with open(os.path.join(OUT_DIR, "pooled_bootstrap_eval_summary.json"), "w") as fh:
        json.dump({**results, "spearman": rho_out}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
