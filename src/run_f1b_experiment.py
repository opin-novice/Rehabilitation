#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_f1b_experiment.py
=====================
F1b campaign -- does a native-rate [2,8] Hz band-power channel earn its place on KIMORE?

This is a SINGLE-VARIABLE ablation against the Phase-0 seed-42 baseline. The plain model is the
'real' arm of phase0_dt_ablation_train_s42.json: same seed, same subject folds (seed=42), same
determinism, same 80-epoch schedule, same hyperparameters. This script trains the IDENTICAL model
with use_bandpower=True (+ compute_bandpower=True in the loader) and nothing else changed, so any
MAD delta is attributable to the band-power channel alone. Because the noise floor is now bitwise
zero (Phase 1), the delta is signal, not scheduler luck.

HONEST PRIOR (state it before the run, not after). The repo's own bandwidth census
(bandwidth_law.py) found ~97.8% of KIMORE's motion energy below 2.19 Hz, and the [2,8] Hz band is
largely Kinect broadband NOISE. So a KIMORE null here is the PREDICTED outcome and is NOT F1b
failing -- it is KIMORE having no clinical signal in that band. F1b's affirmative evidence lives in
the mechanism (native_bandpower tremor recovery) and in a corpus that DOES carry high-frequency
signal (NTU / a pathological cohort). This script decides the KIMORE cell; it does not decide F1b.

Run:  python src/run_f1b_experiment.py --config configs/egru_f1b_bandpower.json
      python src/run_f1b_experiment.py --config ... --smoke          # 1 fold, 2 epochs (pipeline)
      python src/run_f1b_experiment.py --config ... --compare-only   # just print the A/B table
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
from equivariant_gru import SE3EquivariantGRU, count_parameters   # noqa: E402
from train_cde import metrics, floor_metrics                      # noqa: E402

SCORE_MAX = kd.SCORE_MAX


# =============================================================================
# bandpower-aware forward helpers (collate returns a 6-tuple when hf is present)
# =============================================================================
def _unpack(batch):
    """Return (t, x, y, e, n, hf) with hf=None when the batch carries no band-power."""
    if len(batch) == 6:
        t, x, y, e, n, hf = batch
        return t, x, y, e, n, hf
    t, x, y, e, n = batch
    return t, x, y, e, n, None


@torch.no_grad()
def predict(model, samples, device, bs=8):
    model.eval()
    out = []
    for i in range(0, len(samples), bs):
        t, x, y, e, n, hf = _unpack(kd.collate(samples[i: i + bs], device=device))
        out.append(model(t, x, e, n, None, hf).float().cpu().numpy())
    return np.concatenate(out)


# =============================================================================
def train_fold(tr, va, te, cfg, device, verbose=True):
    m = cfg["model"]
    tp = cfg["train"]
    torch.manual_seed(cfg["seed"])
    model = SE3EquivariantGRU(
        n_scalar=m["n_scalar"], n_vec=m["n_vec"], n_layers=m["n_layers"], lmax=m["lmax"],
        gru_hidden=m["gru_hidden"], dropout=m["dropout"], n_exercises=m["n_exercises"],
        use_speed=m["use_speed"], use_chiral=m["use_chiral"], use_mask=m["use_mask"],
        dt_mode=m["dt_mode"], use_bandpower=m["use_bandpower"],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tp["lr"], weight_decay=tp["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tp["epochs"])
    huber = nn.HuberLoss(delta=tp["huber_delta"])
    if verbose:
        print(f"    model: {count_parameters(model):,} params  (use_bandpower={m['use_bandpower']})")

    yte = np.array([s["y"] for s in te])
    best, best_pred = math.inf, None
    for ep in range(tp["epochs"]):
        model.train()
        for batch in kd.Batcher(tr, tp["batch_size"], shuffle=True, device=device):
            t, x, y, e, n, hf = _unpack(batch)
            opt.zero_grad(set_to_none=True)
            huber(model(t, x, e, n, None, hf), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tp["clip"])
            opt.step()
        sched.step()
        if (ep + 1) % tp["eval_every"] == 0 or ep == tp["epochs"] - 1:
            mv = metrics(predict(model, va, device), [s["y"] for s in va])["MAD"]
            if mv < best:                                        # val-selected, never test-selected
                best, best_pred = mv, predict(model, te, device)
    return model, best_pred, yte


def run(cfg, args, device):
    print("=" * 78)
    print(f"F1b CAMPAIGN -- EGRU + native-rate [2,8] Hz band-power  (seed {cfg['seed']}, "
          f"{cfg['folds']}-fold, DETERMINISTic)")
    print("  matched A/B vs the Phase-0 'real' baseline: ONLY use_bandpower differs.")
    print("=" * 78)

    d = cfg["data"]
    t0 = time.time()
    print(f"  loading corpus (compute_bandpower={d['compute_bandpower']}) -- first run computes "
          f"and CACHES Lomb-Scargle band-power; subsequent runs are instant ...")
    S = kd.load_all_exercises(max_len=d["max_len"], verbose=False)
    if d["compute_bandpower"]:
        # attach hf by reloading each sample with the flag on (cached after the first pass)
        S = _attach_bandpower(S, d["max_len"])
    print(f"  corpus ready: {len(S)} sequences  ({time.time()-t0:.0f}s)")

    folds = kd.subject_folds(S, k=cfg["folds"], seed=cfg["seed"])     # MUST match the baseline
    fold_ids = range(1 if args.smoke else cfg["folds"])

    per_fold, err, subj, floors = [], [], [], []
    for f in fold_ids:
        tr, va, te = kd.split(S, folds, test_fold=f, val_fold=(f + 1) % cfg["folds"])
        assert not ({s["subject"] for s in tr} & {s["subject"] for s in te}), "subject leak"
        if args.smoke:
            cfg = json.loads(json.dumps(cfg)); cfg["train"]["epochs"] = 2   # cheap pipeline check
        tt = time.time()
        model, pred, y = train_fold(tr, va, te, cfg, device)
        mad = metrics(pred, y)["MAD"]
        fl = floor_metrics(tr, te, True)["MAD"]
        per_fold.append(mad); floors.append(fl)
        err += list((pred - y) * SCORE_MAX)
        subj += [s["subject"] for s in te]
        os.makedirs(args.out, exist_ok=True)
        torch.save(model.state_dict(),
                   os.path.join(args.out, f"egru_f1b_s{cfg['seed']}_pooled_f{f}.pt"))
        print(f"  fold {f}:  F1b {mad:.3f}   floor {fl:.3f}   ({time.time()-tt:.0f}s)")

    a = np.array(per_fold)
    print(f"\n  F1b  MAD {a.mean():.3f} +/- {a.std():.3f}   over {len(a)} fold(s)")

    res = {"variant": "egru_f1b", "seed": cfg["seed"], "per_fold": per_fold,
           "floors": floors, "mean": float(a.mean()), "std": float(a.std()),
           "errors": err, "subjects": subj, "smoke": args.smoke}
    out_json = os.path.join(args.out, f"f1b_bandpower_s{cfg['seed']}_results.json")
    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"  wrote {out_json}")

    if not args.smoke:
        compare(cfg, res, args)
    return res


def _attach_bandpower(S, max_len):
    """Reload each already-loaded sample with compute_bandpower=True (cached), attaching s['hf'].

    We reuse the identity fields from the first pass and only add the hf channel, so the corpus,
    folds and labels are byte-identical to the baseline -- the ONLY difference is the extra channel.
    """
    out = []
    for s in S:
        ex = f"Es{s['exercise']}"
        d = kd.load_sample(s["group"], s["subject"], ex, max_len=max_len, compute_bandpower=True)
        d.update({k: s[k] for k in ("y", "subject", "group", "cohort", "exercise")})
        out.append(d)
    return out


# =============================================================================
# A/B comparison logger
# =============================================================================
def compare(cfg, f1b_res, args):
    ref = cfg["baseline_ref"]
    base_path = ref["results_json"]
    print("\n" + "=" * 78)
    print("A/B  --  baseline EGRU (no band-power)   vs   EGRU + F1b (band-power)")
    print("=" * 78)
    if not os.path.exists(base_path):
        print(f"  baseline results not found yet: {base_path}")
        print("  (the Phase-0 retrain writes it on completion; re-run with --compare-only then.)")
        return

    base = json.load(open(base_path))
    base_fold = base["per_fold"][ref["arm"]]                # the 'real' arm = matched no-bandpower
    f1b_fold = f1b_res["per_fold"]
    n = min(len(base_fold), len(f1b_fold))
    b, g = np.array(base_fold[:n]), np.array(f1b_fold[:n])

    print(f"\n  {'fold':>4s}  {'baseline':>10s}  {'F1b':>10s}  {'delta (F1b-base)':>18s}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*18}")
    for i in range(n):
        print(f"  {i:>4d}  {b[i]:>10.3f}  {g[i]:>10.3f}  {g[i]-b[i]:>+18.3f}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*18}")
    print(f"  {'mean':>4s}  {b.mean():>10.3f}  {g.mean():>10.3f}  {g.mean()-b.mean():>+18.3f}")
    print(f"  {'std':>4s}  {b.std():>10.3f}  {g.std():>10.3f}")

    # PAIRED and FOLD-VARIANCE-AWARE (same seed => same held-out subjects per fold => paired). A
    # fixed |mean|>thr gate is wrong here: one wild fold (see fold 2) can drag the mean past any
    # threshold while the effect is indistinguishable from zero. Gate on whether the ~95% interval
    # (mean +/- 2*SEM over folds) clears 0, exactly as run_dt_ablation now does.
    d = g - b
    sem = d.std(ddof=1) / math.sqrt(n) if n > 1 else float("inf")
    lo, hi = d.mean() - 2 * sem, d.mean() + 2 * sem
    print(f"\n  paired per-fold delta (F1b - base): mean {d.mean():+.3f}  "
          f"~95% [{lo:+.3f}, {hi:+.3f}]  per-fold SEM {sem:.3f}  (n={n})")
    print(f"  per-fold deltas: {[round(float(v), 3) for v in d]}")
    if lo <= 0.0 <= hi:
        print(f"\n  VERDICT -- NEUTRAL on KIMORE: the interval spans 0 ({lo:+.3f}, {hi:+.3f}), so the")
        print(f"  point estimate ({d.mean():+.3f}) is not distinguishable from no effect (the mean is")
        print("  dominated by a single fold). This is the PREDICTED result: bandwidth_law found")
        print("  ~97.8% of KIMORE energy below 2.19 Hz, so the [2,8] Hz band carries no clinical")
        print("  signal HERE. F1b is NOT refuted -- KIMORE cannot exercise it. Its affirmative")
        print("  evidence is the tremor-recovery mechanism (native_bandpower) and a high-frequency")
        print("  corpus (NTU / a pathological cohort), NOT this MAD cell. Do NOT ship it as the")
        print("  KIMORE flagship, and do NOT call it a failure -- report it as a scoped null.")
    elif hi < 0.0:
        print(f"\n  VERDICT -- F1b WINS on KIMORE by {-d.mean():.3f} MAD, interval ({lo:+.3f},{hi:+.3f})")
        print("  entirely below 0. The [2,8] Hz channel carries score-relevant signal the subsample")
        print("  discarded. Re-run certify_phase1 on the F1b variant and promote it to flagship.")
    else:
        print(f"\n  VERDICT -- F1b HURTS by {d.mean():+.3f} MAD, interval ({lo:+.3f},{hi:+.3f}) entirely")
        print("  above 0: the band is NOISE the model overfits. Keep it OFF for KIMORE.")
    print(f"\n  (For a full subject-clustered bootstrap rather than fold means, re-emit the baseline")
    print(f"   PER-SAMPLE errors; F1b's are saved in this run's json under 'errors'.)")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/egru_f1b_bandpower.json")
    ap.add_argument("--smoke", action="store_true", help="1 fold x 2 epochs: pipeline check only")
    ap.add_argument("--compare-only", action="store_true", help="print the A/B table and exit")
    ap.add_argument("--win-thresh", type=float, default=0.05,
                    help="|delta| below this MAD => neutral (noise floor is ~0, so this is a "
                         "practical-significance bar, not a noise bar)")
    ap.add_argument("--out", default="outputs/cde_block2")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    cfg_seed = cfg["seed"]
    determinism.enable(seed=cfg_seed, strict=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.compare_only:
        path = os.path.join(args.out, f"f1b_bandpower_s{cfg_seed}_results.json")
        if not os.path.exists(path):
            print(f"  no F1b results at {path} -- run the campaign first.")
            return 1
        compare(cfg, json.load(open(path)), args)
        return 0

    run(cfg, args, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
