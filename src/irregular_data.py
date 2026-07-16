#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
irregular_data.py
=================
Task 8 (P0) -- The Empirical Reality Check.

Ingest at least one AUTHENTICALLY irregular real skeletal sequence to anchor the
irregular-sampling experiments, and prove it is numerically well-conditioned for the
SE(3)-equivariant Neural CDE vector field.

Why KIMORE Raw and not CMU MoCap / Vicon
----------------------------------------
Optical mocap (CMU, Vicon, OptiTrack/REHAB24-6, Human3.6M) is distributed gap-filled on a
UNIFORM clock -> any irregularity we add is synthetic, i.e. exactly the "engineered" critique
we must pre-empt. KIMORE is Kinect-v2 and ships the sensor's own per-frame RelativeTime in
`Raw/TimeStamp*.csv` (100-ns ticks). Those are REAL arrival timestamps: ~30 fps with genuine
jitter, and real multi-frame dropouts (USB bandwidth / CPU load) that appear as large gaps.
The *locations* of the gaps are decided by real sensor failure, not by us.

Corpus finding (survey_corpus): across 398 KIMORE recordings, 143 contain >=1 real
multi-frame dropout and 20 clear the full runsheet bar (CV>=0.2 AND >=1 burst gap >=3x mean).
Irregular sampling is systemic in the deployment sensor, not cherry-picked.

Runsheet Task 8 pass/fail (per sequence):
    CV = std(dt)/mean(dt) >= 0.2      AND      at least one burst gap  dt >= 3 * mean(dt).

This script:
  1. loads a KIMORE Raw sequence (timestamps + aligned 25-joint positions),
  2. audits irregularity against the thresholds (with strict asserts),
  3. surveys the whole corpus and writes a manifest of qualifying anchors,
  4. fits our natural cubic spline over the real irregular timeline and confirms dX/ds is
     well-conditioned and does NOT explode the real e3nn vector field (reuses Task 7).

Run:  python src/irregular_data.py            # survey + pick best anchor + audit + conditioning
      python src/irregular_data.py --raw <KIMORE/.../Raw>   # audit one specific sequence
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch

# import the certified building blocks from the equivariance suite (same dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import equivariance_suite as eqs  # noqa: E402

CV_MIN = 0.2                 # runsheet: coefficient of variation threshold
BURST_MULT = 3.0             # runsheet: burst gap = dt >= 3 * mean(dt)  (~3-frame dropout)
NOMINAL_MS = (28.0, 45.0)    # keep genuine ~30fps-nominal recordings (exclude paused/broken clocks)
MAXRATIO_CAP = 100.0         # exclude multi-second pauses (dt >> dropout regime)


# =============================================================================
# 1. KIMORE Raw loaders
# =============================================================================
def _read_nonempty_numeric_rows(path):
    """Read a KIMORE Raw csv, returning a list of token-lists for each non-empty line.
    KIMORE Raw files have leading blank lines and a trailing comma per row."""
    rows = []
    with open(path, "r") as fh:
        for line in fh:
            s = line.strip().strip(",").strip()
            if not s:
                continue
            toks = [t for t in s.split(",") if t != ""]
            rows.append(toks)
    return rows


def parse_timestamps(path):
    """KIMORE `TimeStamp*.csv` -> seconds since first frame (Kinect RelativeTime is 100-ns ticks)."""
    rows = _read_nonempty_numeric_rows(path)
    ticks = np.array([float(r[0]) for r in rows], dtype=np.float64)
    t = (ticks - ticks[0]) * 1e-7           # 100 ns -> s
    return t


def parse_positions(path):
    """KIMORE `JointPosition*.csv` -> (x (L,25,3), tracking (L,25)).
    Each row = 25 joints x [x, y, z, trackingState]."""
    rows = _read_nonempty_numeric_rows(path)
    L = len(rows)
    x = np.zeros((L, 25, 3), dtype=np.float64)
    trk = np.zeros((L, 25), dtype=np.float64)
    for i, r in enumerate(rows):
        v = np.array([float(t) for t in r], dtype=np.float64)
        n = min(len(v) // 4, 25)
        v = v[: n * 4].reshape(n, 4)
        x[i, :n] = v[:, :3]
        trk[i, :n] = v[:, 3]
    return x, trk


def load_kimore_raw(raw_dir):
    """Load an aligned (timestamps, positions) pair from a KIMORE `.../Raw` directory."""
    ts = glob.glob(os.path.join(raw_dir, "TimeStamp*.csv"))
    jp = glob.glob(os.path.join(raw_dir, "JointPosition*.csv"))
    if not ts or not jp:
        raise FileNotFoundError(f"missing TimeStamp/JointPosition in {raw_dir}")
    t = parse_timestamps(ts[0])
    x, trk = parse_positions(jp[0])
    n = min(len(t), len(x))
    if len(t) != len(x):
        print(f"  [warn] frame-count mismatch t={len(t)} x={len(x)} -> truncating to {n}")
    return {"t": t[:n], "x": x[:n], "tracking": trk[:n],
            "raw_dir": raw_dir, "n_frames": n}


# =============================================================================
# 2. Irregularity audit  (Runsheet Task 8)
# =============================================================================
def irregularity_audit(t, cv_min=CV_MIN, burst_mult=BURST_MULT):
    """Compute CV and burst-gap statistics for an arrival-time vector t (seconds)."""
    t = np.asarray(t, dtype=np.float64)
    dt = np.diff(t)
    dt = dt[dt > 0]                                   # guard against any non-monotone stamp
    mu, sd = float(dt.mean()), float(dt.std())
    cv = sd / mu
    burst_thresh = burst_mult * mu
    burst_idx = np.where(dt >= burst_thresh)[0]
    return {
        "n_frames": int(len(t)),
        "duration_s": float(t[-1] - t[0]),
        "mean_dt_ms": mu * 1e3,
        "nominal_fps": 1.0 / mu,
        "cv": cv,
        "n_burst": int(len(burst_idx)),
        "max_gap_ms": float(dt.max() * 1e3),
        "max_ratio": float(dt.max() / mu),
        "burst_frame_idx": burst_idx.tolist(),
        "cv_pass": cv >= cv_min,
        "burst_pass": len(burst_idx) >= 1,
        "pass": (cv >= cv_min) and (len(burst_idx) >= 1),
    }


def assert_irregular(audit):
    """Strict runsheet gate with a diagnostic on failure."""
    if not audit["cv_pass"]:
        raise AssertionError(
            f"Task 8 FAIL (CV): CV={audit['cv']:.3f} < {CV_MIN}. This recording is too uniform; "
            f"pick a sequence with more real dropouts (see survey_corpus ranking).")
    if not audit["burst_pass"]:
        raise AssertionError(
            f"Task 8 FAIL (burst): no gap >= {BURST_MULT}x mean ({BURST_MULT*audit['mean_dt_ms']:.0f} ms). "
            f"No multi-frame dropout present.")
    return True


# =============================================================================
# 3. Corpus survey  (find every authentic anchor we already have)
# =============================================================================
def survey_corpus(root="KIMORE"):
    """Scan every KIMORE `Raw/TimeStamp*.csv`; return per-sequence audit rows."""
    files = glob.glob(os.path.join(root, "**", "Raw", "TimeStamp*.csv"), recursive=True)
    rows = []
    for f in files:
        try:
            t = parse_timestamps(f)
        except Exception:
            continue
        if len(t) < 20:
            continue
        a = irregularity_audit(t)
        a["raw_dir"] = os.path.dirname(f)
        rows.append(a)
    return rows


CANONICAL_MAXR = 15.0        # "3-frame dropout" regime: gaps of a few dropped frames, not a pause


def qualifying_anchors(rows):
    """Sequences that are authentic anchors: pass thresholds AND look like a real ~30fps
    Kinect stream (nominal fps in range, no multi-second pause)."""
    out = []
    for a in rows:
        if not a["pass"]:
            continue
        if not (NOMINAL_MS[0] <= a["mean_dt_ms"] <= NOMINAL_MS[1]):
            continue
        if a["max_ratio"] > MAXRATIO_CAP:
            continue
        out.append(a)
    # prefer more real bursts, then higher CV, then more frames
    out.sort(key=lambda a: (a["n_burst"], a["cv"], a["n_frames"]), reverse=True)
    return out


def split_canonical_stress(anchors):
    """Canonical anchors = multi-frame dropouts in the runsheet's regime (max gap <= CANONICAL_MAXR
    x mean) -> best-conditioned, most defensible. Stress anchors = the big-gap outliers, kept for
    the corruption-pipeline stress sweep (Tasks 9-10)."""
    canonical = [a for a in anchors if a["max_ratio"] <= CANONICAL_MAXR]
    stress = [a for a in anchors if a["max_ratio"] > CANONICAL_MAXR]
    return canonical, stress


# =============================================================================
# 4. Spline conditioning + e3nn field explosion check   (ties to Task 7)
# =============================================================================
def _preprocess(seq, subsample=1):
    """Root-relative, unit-scaled positions on a [0,1]-normalized irregular timeline.

    Time-normalization to [0,1] is a pure rescale: CV and burst RATIOS are scale-invariant,
    so the audit is unchanged, but the CDE integrates on a well-scaled span. Standardizing the
    coordinate amplitude keeps the vector field in its benign regime (like Task 7 data)."""
    t = seq["t"].astype(np.float64)
    x = seq["x"].astype(np.float64)                       # (L,25,3)
    if subsample > 1:
        t = t[::subsample]
        x = x[::subsample]
    t01 = (t - t[0]) / (t[-1] - t[0])                     # [0,1], irregular pattern preserved
    xr = x - x[:, :1, :]                                  # root-relative (joint 0) -> translation-invariant
    scale = np.std(xr[xr != 0]) if np.any(xr != 0) else 1.0
    xr = xr / (scale + 1e-9)
    return t01, xr


def spline_conditioning_check(seq, layout=None, subsample=1):
    """Fit the natural cubic spline over the REAL irregular timeline and confirm dX/ds is
    bounded and does not explode the real e3nn vector field."""
    if layout is None:
        layout = eqs.IrrepLayout(n_scalar=4, n_vec=6)
    t01, xr = _preprocess(seq, subsample=subsample)
    L, J, _ = xr.shape
    t_t = torch.tensor(t01, dtype=torch.float64)
    x_t = torch.tensor(xr.reshape(1, L, J * 3), dtype=torch.float64)

    spline = eqs.NaturalCubicSpline(t_t, x_t)
    # dense sample of the control derivative U(s) = mean_j dX_j/ds across the whole span
    ss = torch.linspace(float(t_t[0]), float(t_t[-1]), 2000, dtype=torch.float64)[1:-1]
    Us = torch.stack([eqs._control_at(spline, s, J)[0] for s in ss])   # (S,3)
    max_dxds = float(Us.norm(dim=-1).max())
    med_dxds = float(Us.norm(dim=-1).median())

    result = {"n_frames": L, "n_joints": J, "max_dXds": max_dxds, "median_dXds": med_dxds,
              "spline_finite": bool(torch.isfinite(Us).all())}

    # --- feed through the REAL e3nn field + adaptive solver (reuse Task 7) ---
    if eqs.HAVE_E3NN:
        torch.set_grad_enabled(False)
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            torch.manual_seed(1)
            field = eqs.EquivariantCDEFunc_e3nn(layout, gain=0.2)
        finally:
            torch.set_default_dtype(prev)
        z0 = torch.zeros(1, layout.total, dtype=torch.float64)
        z0[:, : layout.n_scalar] = 0.5                        # mild non-zero scalar init
        zT, grid = eqs.integrate_dopri5(field, spline, z0, t_t[0], t_t[-1], J,
                                        layout=layout, atol=1e-7, rtol=1e-6, error_mode="neq")
        result.update({
            "e3nn_zT_finite": bool(torch.isfinite(zT).all()),
            "e3nn_zT_norm": float(zT.norm()),
            "dopri5_steps": len(grid),
            "exploded": (not bool(torch.isfinite(zT).all())) or len(grid) > 5000 or float(zT.norm()) > 1e6,
        })
    else:
        result["e3nn"] = "unavailable"
    return result


# =============================================================================
# 5. Main
# =============================================================================
def _print_audit(tag, a):
    print(f"  {tag}")
    print(f"    frames={a['n_frames']}  duration={a['duration_s']:.1f}s  "
          f"mean_dt={a['mean_dt_ms']:.2f} ms ({a['nominal_fps']:.1f} fps)")
    print(f"    CV = {a['cv']:.3f}  (>= {CV_MIN}? {a['cv_pass']})")
    print(f"    burst gaps (dt >= {BURST_MULT:.0f}x mean) = {a['n_burst']}  (>=1? {a['burst_pass']})  "
          f"| max gap = {a['max_gap_ms']:.0f} ms ({a['max_ratio']:.1f}x mean)")
    print(f"    -> Task 8 {'PASS' if a['pass'] else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(description="Task 8 -- authentic irregular skeletal anchor")
    ap.add_argument("--root", default="KIMORE", help="dataset root to survey")
    ap.add_argument("--raw", default=None, help="audit ONE specific KIMORE .../Raw dir")
    ap.add_argument("--out", default="outputs/irregular_anchor/manifest.json")
    ap.add_argument("--subsample", type=int, default=1)
    args = ap.parse_args()

    print("=" * 72)
    print("  TASK 8 -- EMPIRICAL REALITY CHECK (authentic irregular skeletal anchor)")
    print("=" * 72)

    if args.raw:
        seq = load_kimore_raw(args.raw)
        a = irregularity_audit(seq["t"])
        _print_audit(os.path.relpath(args.raw), a)
        assert_irregular(a)
        print("\n  [conditioning] real irregular timeline -> spline -> e3nn field")
        cond = spline_conditioning_check(seq, subsample=args.subsample)
        for k, v in cond.items():
            print(f"    {k}: {v}")
        assert not cond.get("exploded", False), "conditioning FAIL: real data explodes the field."
        print("\n  [TASK 8] PASS (single sequence).")
        return 0

    # --- corpus survey ---
    print(f"\n  surveying {args.root} ...")
    rows = survey_corpus(args.root)
    anchors = qualifying_anchors(rows)
    canonical, stress = split_canonical_stress(anchors)
    n_burst_any = sum(1 for a in rows if a["burst_pass"])
    print(f"  {len(rows)} recordings scanned")
    print(f"  {n_burst_any} contain >=1 real multi-frame dropout")
    print(f"  {sum(1 for a in rows if a['pass'])} clear CV>=0.2 AND >=1 burst")
    print(f"  {len(anchors)} qualify as clean ~30fps anchors "
          f"(mean_dt in {NOMINAL_MS} ms, max_ratio <= {MAXRATIO_CAP})")
    print(f"    -> {len(canonical)} canonical (max gap <= {CANONICAL_MAXR:.0f}x mean, ~few-frame dropout)")
    print(f"    -> {len(stress)} stress (bigger real gaps, for the Task 9-10 corruption sweep)")

    if not canonical:
        print("\n  [TASK 8] NO canonical anchor found -- keep sourcing (do NOT rely on synthetic).")
        return 1

    print("\n  === canonical qualifying anchors (primary tier) ===")
    print(f"  {'CV':>6} {'burst':>5} {'maxR':>6} {'frames':>6}  dir")
    for a in canonical[:8]:
        print(f"  {a['cv']:6.3f} {a['n_burst']:5d} {a['max_ratio']:6.1f} {a['n_frames']:6d}  "
              f"{os.path.relpath(a['raw_dir'])}")
    if stress:
        print("  --- stress anchors (big real gaps) ---")
        for a in stress[:4]:
            print(f"  {a['cv']:6.3f} {a['n_burst']:5d} {a['max_ratio']:6.1f} {a['n_frames']:6d}  "
                  f"{os.path.relpath(a['raw_dir'])}")

    best = canonical[0]
    print("\n  === CHOSEN ANCHOR (canonical, best-conditioned) ===")
    seq = load_kimore_raw(best["raw_dir"])
    a = irregularity_audit(seq["t"])
    _print_audit(os.path.relpath(best["raw_dir"]), a)
    assert_irregular(a)

    print("\n  [conditioning] real irregular timeline -> spline -> real e3nn field (Task 7 reuse)")
    cond = spline_conditioning_check(seq, subsample=args.subsample)
    for k, v in cond.items():
        print(f"    {k}: {v}")
    assert not cond.get("exploded", False), "conditioning FAIL: real data explodes the e3nn field."

    # --- write manifest (the P0 artifact) ---
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    manifest = {
        "criteria": {"cv_min": CV_MIN, "burst_mult": BURST_MULT,
                     "nominal_ms": NOMINAL_MS, "max_ratio_cap": MAXRATIO_CAP},
        "corpus": {"root": args.root, "n_scanned": len(rows),
                   "n_with_dropout": n_burst_any,
                   "n_pass_thresholds": sum(1 for a in rows if a["pass"]),
                   "n_qualifying_anchors": len(anchors),
                   "n_canonical": len(canonical), "n_stress": len(stress)},
        "chosen_anchor": {"raw_dir": best["raw_dir"], "audit": a},
        "chosen_conditioning": cond,
        "canonical_anchors": [
            {"raw_dir": x["raw_dir"], "cv": x["cv"], "n_burst": x["n_burst"],
             "max_ratio": x["max_ratio"], "n_frames": x["n_frames"]} for x in canonical],
        "stress_anchors": [
            {"raw_dir": x["raw_dir"], "cv": x["cv"], "n_burst": x["n_burst"],
             "max_ratio": x["max_ratio"], "n_frames": x["n_frames"]} for x in stress],
    }
    with open(args.out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\n  manifest written: {args.out}  ({len(anchors)} anchors)")

    print("\n" + "=" * 72)
    print("  [TASK 8] PASS -- authentic irregular anchor secured and well-conditioned.")
    print(f"    anchor: {os.path.relpath(best['raw_dir'])}  CV={a['cv']:.2f}  bursts={a['n_burst']}")
    print(f"    e3nn field: finite, ||zT||={cond.get('e3nn_zT_norm', float('nan')):.2f}, "
          f"{cond.get('dopri5_steps', '?')} solver steps (no explosion)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
