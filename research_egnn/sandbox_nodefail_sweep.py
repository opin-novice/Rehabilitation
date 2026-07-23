#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sandbox_nodefail_sweep.py  --  RESEARCH SANDBOX (NOT for the paper)
==================================================================
High-rigour confirmation of the EGNN node-failure signal: 3 seeds x 5 folds, for the baseline EGNN
and the coordinate-clamp arms {0.1, 0.5, 1.0}. Reports a mean +/- std matrix over the 3 seeds for
clean MAD and node-failure MAD-lost (0->8 dead joints, 'hold', 3 draws), plus a one-time invariance
certificate per arm (confirming the clamp does not break equivariance).

Reads only research_egnn/outputs/ checkpoints; writes the "nodefail_sweep_3seed" section into
research_egnn/outputs/sandbox_results.json. Touches nothing under src/ or outputs/cde_block2/.

    python research_egnn/sandbox_nodefail_sweep.py
"""

import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
for p in (_HERE, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import kimore_cde_data as kd                                  # noqa: E402
import block2_transforms as bt                               # noqa: E402
from train_cde import metrics                                # noqa: E402
from joint_failure import fail_joints                        # noqa: E402
from egnn_model import EGNNRecurrence                        # noqa: E402
from sandbox_train import config_tag                         # noqa: E402

OUT = os.path.join(_HERE, "outputs")
SCORE_MAX = kd.SCORE_MAX
NF_LEVELS = [0, 1, 2, 4, 8]
SEEDS = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# arm label -> config dict (layers, hidden, coord clamp). The checkpoint tag is derived via
# config_tag() so it always matches what sandbox_train.py wrote. The first four arms reproduce the
# original clamp sweep (tags "", _c0.1, _c0.5, _c1); the rest are the tuning grid. Arms whose seeds
# are not all trained yet are skipped, so this can be re-run as configs land.
ARMS = [
    ("EGNN base L4h64",  dict(layers=4, hidden=64,  clamp=None)),
    ("EGNN clamp 0.1",   dict(layers=4, hidden=64,  clamp=0.1)),
    ("EGNN clamp 0.5",   dict(layers=4, hidden=64,  clamp=0.5)),
    ("EGNN clamp 1.0",   dict(layers=4, hidden=64,  clamp=1.0)),
    ("EGNN deep L6h64",  dict(layers=6, hidden=64,  clamp=None)),
    ("EGNN wide L4h128", dict(layers=4, hidden=128, clamp=None)),
    ("EGNN L6h128",      dict(layers=6, hidden=128, clamp=None)),
]


def load(cfg, tag, seed, fold):
    m = EGNNRecurrence(n_scalar=32, n_vec=8, n_layers=cfg["layers"], egnn_hidden=cfg["hidden"],
                       use_chiral=False, coord_clamp=cfg["clamp"]).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(OUT, f"egnn{tag}_s{seed}_f{fold}.pt"), map_location=DEVICE))
    return m.eval()


@torch.no_grad()
def predict(m, samples, bs=8):
    out = []
    for i in range(0, len(samples), bs):
        t, x, y, e, n = kd.collate(samples[i: i + bs], device=DEVICE)
        out.append(m(t, x, e, n).float().cpu().numpy())
    return np.concatenate(out)


def per_seed(cfg, tag, seed, S, draws=3):
    """Return (clean_mad, nf_lost, cert) averaged over the 5 folds of this seed's partition."""
    folds = kd.subject_folds(S, k=5, seed=seed)
    cleans, losts, certs = [], [], []
    for f in range(5):
        tr, va, te = kd.split(S, folds, test_fold=f, val_fold=(f + 1) % 5)
        y = np.array([s["y"] for s in te])
        m = load(cfg, tag, seed, f)
        clean = predict(m, te)
        cleans.append(metrics(clean, y)["MAD"])
        # invariance cert (arbitrary proper rotation is over-kill; a 90deg azimuth suffices here)
        certs.append(float(np.max(np.abs(predict(m, [bt.rotate_sample(s, 90) for s in te]) - clean)) * SCORE_MAX))
        nf = {0: metrics(clean, y)["MAD"]}
        for k in NF_LEVELS[1:]:
            d = [metrics(predict(m, [fail_joints(s, k, "hold", seed=dd * 977 + i)
                                     for i, s in enumerate(te)]), y)["MAD"] for dd in range(draws)]
            nf[k] = float(np.mean(d))
        losts.append(nf[max(NF_LEVELS)] - nf[0])
        del m
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return float(np.mean(cleans)), float(np.mean(losts)), float(np.mean(certs))


def main():
    S = kd.load_all_exercises(max_len=150, verbose=False)
    matrix = {}
    print(f"\n{'='*82}\nHIGH-RIGOUR NODE-FAILURE SWEEP  (3 seeds x 5 folds; NOT for the paper)\n{'='*82}")
    hdr = f"  {'arm':<18s} {'clean MAD (mu+-sd)':>22s} {'nf lost 0->8 (mu+-sd)':>24s} {'cert':>9s}"
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    for label, cfg in ARMS:
        tag = config_tag(cfg["layers"], cfg["hidden"], 1e-3, cfg["clamp"])
        # require all seeds present
        if not all(os.path.exists(os.path.join(OUT, f"egnn{tag}_s{s}_f0.pt")) for s in SEEDS):
            print(f"  {label:<18s}  [skip: not all seeds trained yet]")
            continue
        cl, lo, ce = [], [], []
        for s in SEEDS:
            c, l, cert = per_seed(cfg, tag, s, S)
            cl.append(c); lo.append(l); ce.append(cert)
        rec = {"clean_mad_mean": float(np.mean(cl)), "clean_mad_std": float(np.std(cl, ddof=1)),
               "nf_lost_mean": float(np.mean(lo)), "nf_lost_std": float(np.std(lo, ddof=1)),
               "cert_mean": float(np.mean(ce)), "per_seed_clean": cl, "per_seed_nf_lost": lo}
        matrix[label] = rec
        print(f"  {label:<18s} {rec['clean_mad_mean']:9.3f} +- {rec['clean_mad_std']:5.3f}      "
              f"{rec['nf_lost_mean']:+9.3f} +- {rec['nf_lost_std']:5.3f}      {rec['cert_mean']:.1e}")
    print("  " + "-" * (len(hdr) - 2))
    print("  reference (paper 3-seed): EGRU nf lost +3.76 ; InvariantGRU +12.05 ; PCT +2.27")
    print(f"{'='*82}")

    # merge into sandbox_results.json without disturbing the earlier section
    path = os.path.join(OUT, "sandbox_results.json")
    blob = json.load(open(path)) if os.path.exists(path) else {}
    blob["nodefail_sweep_3seed"] = {"seeds": SEEDS, "nf_levels": NF_LEVELS, "matrix": matrix}
    with open(path, "w") as fh:
        json.dump(blob, fh, indent=2)
    print(f"\nupdated {path}  (key: nodefail_sweep_3seed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
