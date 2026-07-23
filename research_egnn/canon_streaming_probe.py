#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
canon_streaming_probe.py  --  RESEARCH SANDBOX (isolated; writes only research_egnn/outputs/)
=============================================================================================
Workstream E: put a MEASURED number under the paper's confront-canonicalization argument, and
refuse to ship any invented statistic.

Context. The sandbox PCA-canonicalization baseline (canonicalize.pca_canonicalize) ties/beats
EGRU on every OFFLINE axis the sandbox tested: clean MAD 6.85, azimuth viewpoint degradation
0.0, node-fail loss 3.17. Two facts about that baseline decide whether the paper's differentiator
is honest:

  1. pca_canonicalize is PER-FRAME spatial PCA (covariance over the 25 joints WITHIN each frame).
     It needs no temporal lookahead -- it is already causal. So the paper must NOT claim canon
     "requires non-causal lookahead"; that is false. This probe verifies it and instead measures
     the REAL fragility.

  2. The canonical coordinates q@V are provably EXACTLY invariant to a rotation R of the input
     (cov -> R cov R^T, V -> R V, coords q@V unchanged) -- EXCEPT where the top eigenvalues are
     near-degenerate, where eigh's ordering (and the sign convention) can swap, producing a
     discontinuous frame. The sandbox only ever rotated about the vertical axis (azimuth), a
     1-parameter subgroup. This probe asks: does the swap that azimuth never triggered FIRE under
     arbitrary SO(3) -- the full group the paper's theorem covers (certified 9e-6, incl. NTU 45
     deg cross-view)?

Outputs (research_egnn/outputs/canon_streaming_probe.json):
  A) per-frame PCA-frame temporal jitter + eigen-gap degeneracy frequency (mechanism exists?)
  B) per-sequence canonical-coordinate shift under azimuth vs arbitrary SO(3) (does it fire?)

All numbers here are model-independent (input-space); a downstream continuous regressor's score
shift is bounded by these coordinate shifts. E2 (optional) turns B into a MAD via the canon PCT
checkpoints.
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
for p in (_HERE, _SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import kimore_cde_data as kd                    # noqa: E402  (read-only)
from canonicalize import pca_canonicalize       # noqa: E402  (the exact baseline under test)

# ----- ISOLATION GUARD: every write goes here and nowhere else -----------------------------------
OUT = os.path.join(_HERE, "outputs")
assert os.path.abspath(OUT).replace("\\", "/").endswith("research_egnn/outputs"), \
    f"refusing to run: output dir {OUT} is not the sandbox"
os.makedirs(OUT, exist_ok=True)


# =============================================================================
# PCA frame with the SAME convention as canonicalize.pca_canonicalize, but also
# returning the eigenvalues so we can measure degeneracy.
# =============================================================================
def pca_frame(p):
    """Return (V, w_desc) for one frame p=(J,3): principal axes as columns, descending eigenvalues.
    Sign/handedness convention is byte-identical to canonicalize.pca_canonicalize."""
    q = p - p.mean(0, keepdims=True)
    cov = q.T @ q
    w, V = np.linalg.eigh(cov)                    # ascending
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order].copy()
    for a in range(3):
        proj = q @ V[:, a]
        if proj[np.argmax(np.abs(proj))] < 0:
            V[:, a] = -V[:, a]
    if np.linalg.det(V) < 0:
        V[:, 2] = -V[:, 2]
    return V, w


def frame_angle(Va, Vb):
    """Geodesic angle (deg) between two rotation matrices Va, Vb (columns = axes)."""
    Rrel = Va.T @ Vb
    c = (np.trace(Rrel) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rand_so3(rng):
    """Haar-random proper rotation via QR (sign-fixed for uniqueness)."""
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))                   # make QR unique
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def canon_coords(x):
    """Canonical coordinates (L,J,3) for a raw skeleton x=(L,J,3), via the baseline itself."""
    return np.asarray(pca_canonicalize({"x": x})["x"], dtype=np.float64)


# =============================================================================
# Part A -- per-frame frame jitter + eigen-gap degeneracy
# =============================================================================
def part_a(samples, gap_thresholds=(0.02, 0.05, 0.10), flip_deg=45.0):
    per_frame_angles = []          # consecutive-frame frame rotation (deg)
    min_gaps = []                  # per-frame min relative eigen-gap (top-2, top-3)
    flips = 0
    n_transitions = 0
    for s in samples:
        x = np.asarray(s["x"], dtype=np.float64)
        Vs, ws = [], []
        for f in range(x.shape[0]):
            V, w = pca_frame(x[f])
            Vs.append(V)
            ws.append(w)
            w = np.maximum(w, 0.0)
            denom = w[0] if w[0] > 0 else 1.0
            gap12 = (w[0] - w[1]) / denom
            gap23 = (w[1] - w[2]) / denom
            min_gaps.append(float(min(gap12, gap23)))
        for f in range(len(Vs) - 1):
            ang = frame_angle(Vs[f], Vs[f + 1])
            per_frame_angles.append(ang)
            n_transitions += 1
            if ang > flip_deg:
                flips += 1
    per_frame_angles = np.array(per_frame_angles)
    min_gaps = np.array(min_gaps)
    return {
        "n_sequences": len(samples),
        "n_frame_transitions": int(n_transitions),
        "frame_jitter_deg": {
            "median": float(np.median(per_frame_angles)),
            "p95": float(np.percentile(per_frame_angles, 95)),
            "p99": float(np.percentile(per_frame_angles, 99)),
            "max": float(per_frame_angles.max()),
        },
        "flip_rate_gt_%gdeg" % flip_deg: float(flips / max(n_transitions, 1)),
        "degenerate_frame_fraction": {
            f"min_gap<{th}": float((min_gaps < th).mean()) for th in gap_thresholds
        },
    }


# =============================================================================
# Part B -- per-sequence canonical-coordinate shift: azimuth subgroup vs full SO(3)
# =============================================================================
def _coord_shift(x, R):
    """Max per-frame RMS joint displacement between canon(x) and canon(x rotated by R)."""
    c0 = canon_coords(x)
    cR = canon_coords(x @ R.T)                     # row-vector convention (matches rotate_sample)
    d = np.sqrt(((cR - c0) ** 2).sum(-1))          # (L,J) per-joint distance
    return float(d.mean(-1).max())                 # worst frame's mean-joint shift


def part_b(samples, n_rot=32, seed=0):
    from block2_transforms import azimuth_matrix
    rng = np.random.default_rng(seed)
    azimuths = [15, 30, 45, 60, 90, 120, 150, 180]
    az_R = [azimuth_matrix(a) for a in azimuths]
    so3_R = [rand_so3(rng) for _ in range(n_rot)]

    # body scale for normalization: mean joint distance from centroid on the clean frame
    scales = []
    for s in samples:
        x = np.asarray(s["x"], dtype=np.float64)
        q = x - x.mean(1, keepdims=True)
        scales.append(np.sqrt((q ** 2).sum(-1)).mean())
    body_scale = float(np.mean(scales))

    az_worst, so3_worst = [], []
    for s in samples:
        x = np.asarray(s["x"], dtype=np.float64)
        az_worst.append(max(_coord_shift(x, R) for R in az_R))
        so3_worst.append(max(_coord_shift(x, R) for R in so3_R))
    az_worst = np.array(az_worst)
    so3_worst = np.array(so3_worst)

    def summ(a):
        return {
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "max": float(a.max()),
            "frac_seq_shift_gt_1pct_bodyscale": float((a > 0.01 * body_scale).mean()),
        }

    return {
        "body_scale": body_scale,
        "n_so3_rotations": n_rot,
        "azimuth_subgroup": summ(az_worst),           # what the sandbox tested
        "arbitrary_SO3": summ(so3_worst),             # what the theorem covers
        "note": ("coord shift is EXACTLY 0 unless the eigenvalue order / sign convention swaps; "
                 "any nonzero value is a frame discontinuity, i.e. a per-sequence score shift a "
                 "certified-invariant model cannot have."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-so3", type=int, default=32, help="random SO(3) rotations per sequence")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=150)
    args = ap.parse_args()

    print("[E] loading KIMORE (pooled, read-only) ...")
    samples = kd.load_all_exercises(max_len=args.max_len, verbose=False)
    print(f"[E] {len(samples)} sequences")

    print("[E] Part A: per-frame PCA-frame jitter + eigen-gap degeneracy ...")
    a = part_a(samples)
    print("[E] Part B: canonical-coord shift, azimuth subgroup vs arbitrary SO(3) ...")
    b = part_b(samples, n_rot=args.n_so3, seed=args.seed)

    result = {
        "experiment": "canon_streaming_probe",
        "baseline_under_test": "canonicalize.pca_canonicalize (per-frame spatial PCA, causal)",
        "part_a_frame_jitter_and_degeneracy": a,
        "part_b_coord_shift_azimuth_vs_so3": b,
    }
    dst = os.path.join(OUT, "canon_streaming_probe.json")
    with open(dst, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[E] wrote {dst}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
