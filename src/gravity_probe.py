#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gravity_probe.py
================
THE DECISIVE DIAGNOSTIC: is full SO(3) invariance discarding the clinical signal?

The suspicion
-------------
Our SE(3)-equivariant CDE has an invariant read-out, so its score is unchanged by ANY rotation of
the body -- including rotations that tip the subject relative to GRAVITY. But KIMORE's clinical
scores are substantially about deviation from the vertical: trunk lean, arm elevation, how far
from upright the movement ends up. "Lift your arm vertically" is graded by how far from vertical
the arm actually gets. A model invariant to all of SO(3) cannot represent that quantity at all.

The target architecture (PCT) is NOT rotation-invariant. It sees raw camera-frame coordinates, so
the gravity direction is available to it for free. That would explain the whole result: PCT
extracts real subject-level signal (MAD 6.07) from the same 45 training subjects on which our
model, after its capacity bottleneck was removed, still cannot beat the mean predictor (8.19 vs
8.17) -- it now fits the TRAINING set easily (loss 0.003) and simply fails to generalise.

The test
--------
Two feature sets, one ridge regression, the SAME subject-disjoint folds, the SAME floor:

  SO(3)-INVARIANT   : pairwise joint distances + bone lengths + speeds.
                      Everything our architecture is allowed to see. Nothing about "up".
  GRAVITY-AWARE     : the invariant set PLUS angles-from-vertical (trunk lean, arm/leg elevation)
                      and the vertical coordinates of the joints. These exist ONLY because a
                      preferred axis exists.

If gravity-aware >> invariant, then SO(3) is the wrong group: it is a symmetry the DATA does not
have, and imposing it destroys signal. The physically correct group is rotation about the VERTICAL
axis (camera azimuth) -- which is also the only rotation a real deployed camera performs.

This is a linear probe on purpose: it measures what is LINEARLY DECODABLE from each feature set,
so a gap cannot be blamed on our optimiser, our architecture, or our hyperparameters.

Run:  python src/gravity_probe.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimore_cde_data as kd     # noqa: E402

from sklearn.linear_model import RidgeCV        # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

UP = np.array([0.0, 1.0, 0.0])          # Kinect v2 camera frame: +Y is up
SCORE_MAX = kd.SCORE_MAX

# Kinect v2 joint indices
SPINE_BASE, SPINE_MID, NECK, HEAD = 0, 1, 2, 3
SH_L, EL_L, WR_L = 4, 5, 6
SH_R, EL_R, WR_R = 8, 9, 10
HIP_L, KNEE_L = 12, 13
HIP_R, KNEE_R = 16, 17

SEGMENTS = [
    ("trunk", SPINE_BASE, NECK),
    ("upper_arm_L", SH_L, EL_L), ("forearm_L", EL_L, WR_L),
    ("upper_arm_R", SH_R, EL_R), ("forearm_R", EL_R, WR_R),
    ("thigh_L", HIP_L, KNEE_L), ("thigh_R", HIP_R, KNEE_R),
    ("head", NECK, HEAD),
]


def so3_invariant_features(x):
    """x: (L, 25, 3) root-relative. Features that survive ANY rotation -- what our model sees.

    Pairwise distances and speeds are functions of inner products only, so they are exactly the
    quantities an SO(3)-invariant read-out can access. Nothing here knows which way is up.
    """
    L = x.shape[0]
    f = []
    # segment lengths (mean + std over time)
    for _, a, b in SEGMENTS:
        d = np.linalg.norm(x[:, a] - x[:, b], axis=-1)
        f += [d.mean(), d.std()]
    # radial distance of each joint from the root (rotation-invariant)
    r = np.linalg.norm(x, axis=-1)                          # (L, 25)
    f += list(r.mean(0)) + list(r.std(0))
    # speed magnitudes (invariant)
    v = np.linalg.norm(np.diff(x, axis=0), axis=-1)         # (L-1, 25)
    f += list(v.mean(0)) + list(v.std(0))
    # a few inter-joint distances (wrist-wrist, wrist-head, ankle spread)
    for a, b in [(WR_L, WR_R), (WR_L, HEAD), (WR_R, HEAD), (EL_L, EL_R)]:
        d = np.linalg.norm(x[:, a] - x[:, b], axis=-1)
        f += [d.mean(), d.std()]
    return np.array(f, dtype=np.float64)


def gravity_features(x):
    """Features that EXIST ONLY because a preferred (vertical) axis exists.

    These are exactly what SO(3) invariance destroys and what a real clinician grades on:
    how far from vertical is the trunk, how high did the arm actually get.
    """
    f = []
    # angle of each body segment from the vertical
    for _, a, b in SEGMENTS:
        seg = x[:, b] - x[:, a]                                    # (L,3)
        n = np.linalg.norm(seg, axis=-1) + 1e-9
        cos = (seg @ UP) / n                                       # (L,) elevation cosine
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        f += [ang.mean(), ang.std(), ang.min(), ang.max()]
    # vertical coordinate of every joint (height above the root)
    h = x[:, :, 1]                                                 # (L,25)
    f += list(h.mean(0)) + list(h.std(0)) + list(h.max(0))
    return np.array(f, dtype=np.float64)


def build(samples, kind):
    X = []
    for s in samples:
        inv = so3_invariant_features(s["x"])
        if kind == "invariant":
            v = inv
        else:
            v = np.concatenate([inv, gravity_features(s["x"])])
        ex = np.zeros(5)
        ex[s["exercise"] - 1] = 1.0                                # exercise ID (both get it)
        X.append(np.concatenate([v, ex]))
    y = np.array([s["y"] for s in samples])
    return np.nan_to_num(np.stack(X)), y


def mad(p, y):
    return float(np.mean(np.abs(p - y)) * SCORE_MAX)


def main():
    samples = kd.load_all_exercises(verbose=False)
    folds = kd.subject_folds(samples, k=5, seed=0)
    print("=" * 78)
    print("GRAVITY PROBE -- does SO(3) invariance destroy the clinical signal?")
    print("=" * 78)
    print("Ridge regression, subject-disjoint 5-fold, pooled KIMORE (380 seqs / 77 subjects).")
    print("Both feature sets are given the exercise ID. Linear probe: a gap cannot be blamed")
    print("on our optimiser or architecture.\n")

    res = {k: [] for k in ("invariant", "gravity")}
    floors = []
    for f in range(5):
        tr, _, te = kd.split(samples, folds, test_fold=f, val_fold=(f + 1) % 5)
        y_te = np.array([s["y"] for s in te])
        floors.append(mad(kd.exercise_mean_floor(tr, te), y_te))
        for kind in ("invariant", "gravity"):
            Xtr, ytr = build(tr, kind)
            Xte, _ = build(te, kind)
            sc = StandardScaler().fit(Xtr)
            m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(Xtr), ytr)
            res[kind].append(mad(m.predict(sc.transform(Xte)), y_te))

    fl = np.array(floors)
    print(f"  {'feature set':<34s} {'MAD':>7s} {'+/-':>6s}")
    print(f"  {'-'*34} {'-'*7} {'-'*6}")
    print(f"  {'per-exercise mean (floor)':<34s} {fl.mean():7.3f} {fl.std():6.3f}")
    for kind, label in (("invariant", "SO(3)-invariant  (what WE see)"),
                        ("gravity", "+ gravity-aware  (what PCT sees)")):
        a = np.array(res[kind])
        print(f"  {label:<34s} {a.mean():7.3f} {a.std():6.3f}")

    inv = np.array(res["invariant"]).mean()
    grv = np.array(res["gravity"]).mean()
    print(f"\n  gravity features are worth {inv - grv:+.3f} MAD "
          f"({100*(inv-grv)/inv:+.1f}%) over the SO(3)-invariant set.")
    print(f"  invariant-only vs floor : {inv - fl.mean():+.3f} MAD")
    print(f"  gravity-aware vs floor  : {grv - fl.mean():+.3f} MAD")

    print("\n" + "-" * 78)
    if inv >= fl.mean() - 0.15 and grv < fl.mean() - 0.5:
        print("VERDICT: SO(3)-invariant features CANNOT beat the mean predictor; gravity-aware")
        print("features CAN. The clinical score lives in the body's orientation RELATIVE TO")
        print("GRAVITY -- precisely the quantity full SO(3) invariance annihilates.")
        print("SO(3) is a symmetry the DATA DOES NOT HAVE. Imposing it destroys the signal,")
        print("which is why our model cannot generalise while the non-equivariant PCT can.")
        print("\nFIX: quotient by the symmetry the deployment ACTUALLY has -- rotation about the")
        print("VERTICAL axis (camera azimuth), not all of SO(3). Gravity is observable and fixed;")
        print("a camera moves in azimuth. That keeps the viewpoint-invariance theorem (azimuth is")
        print("the realistic nuisance) while RETAINING the gravity direction that carries signal.")
    else:
        print("VERDICT: the gravity hypothesis is NOT supported by this probe. Look elsewhere.")
    print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
