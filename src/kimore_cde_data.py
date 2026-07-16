#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kimore_cde_data.py
==================
KIMORE dataset for the trainable SE(3)-equivariant Neural CDE.

Unlike every fixed-grid baseline, we do NOT resample onto a uniform frame axis. Each sample
keeps the Kinect's OWN arrival timestamps (`Raw/TimeStamp*.csv`, 100-ns RelativeTime ticks),
which carry real jitter and real multi-frame dropouts. That irregular timeline IS the input.

Label join
----------
`KIMORE_processed/Exercise{k}/meta.csv` carries the clinical score per (subject, exercise);
the raw irregular timeline lives at `KIMORE/{group}/{subject_name}/Es{k}/Raw`. The join is
exact (77/77 for Es1). Scores are the KIMORE clinical Total Score (range 0-50); we regress
y = score / 50 in [0, 1], matching the target paper's normalised-score convention.

Preprocessing (all of it equivariance-safe)
-------------------------------------------
  * root-relative coordinates (subtract SpineBase, joint 0)  -> translation invariance, and
    the constant is annihilated by dX anyway.
  * per-sequence SCALE normalisation by the median joint radius. A scale is a Type-0 scalar
    computed from the sequence itself, so dividing by it commutes with rotation: equivariance
    survives. This removes subject body-size as a nuisance variable.
  * time in SECONDS, divided by a single global constant (not per-sequence!). Per-sequence
    time normalisation would erase movement DURATION, and duration is a real clinical cue --
    a prior study in this repo found duration acting as a shortcut once resampling removed it,
    so we keep the ratio intact and only rescale to O(1) for conditioning.
  * length capped by uniform INDEX subsampling that KEEPS THE ACTUAL STAMPS of the retained
    frames. We never re-time anything; we only look at fewer of the real arrivals. Irregularity
    is preserved (in fact accentuated), and no synthetic timestamp is ever invented.

Splits
------
Subject-disjoint K-fold. Non-negotiable: a prior paper in this repo turned on exactly this
distinction (a within-subject split can look strong while the subject-level effect is null),
so every number we report is on held-out SUBJECTS.

Run:  python src/kimore_cde_data.py          # build + summarise Exercise 1
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import irregular_data as ird     # noqa: E402  (certified KIMORE Raw parser)

KIMORE_ROOT = "KIMORE"
PROCESSED_ROOT = "KIMORE_processed"
N_JOINTS = 25
ROOT_JOINT = 0                    # Kinect v2 SpineBase
SCORE_MAX = 50.0                  # KIMORE clinical Total Score range
TIME_SCALE = 20.0                 # global seconds->O(1) constant (median Es1 duration)


# =============================================================================
# 1. Load one (subject, exercise) sample
# =============================================================================
def load_sample(group, subject, exercise, max_len=150, min_len=16, compute_bandpower=False):
    """-> dict(t (L,), x (L,25,3), ...) with REAL arrival stamps, root-relative, scale-normed.

    compute_bandpower (F1b): if True, attach s["hf"] (L, 25) = native-rate band-limited speed RMS
    in [2,8] Hz, computed on the FULL ~30 Hz sequence BEFORE subsampling and aligned to the frames
    we retain. This is the high-frequency content the 150-frame subsample (Nyquist ~1.9 Hz)
    otherwise destroys at the loader. It is a Type-0 (rotation-invariant) scalar per joint, so it
    costs the equivariance certificate nothing. Off by default -- the Phase-0 pivot switches it on.
    """
    raw_dir = os.path.join(KIMORE_ROOT, group, subject, exercise, "Raw")
    seq = ird.load_kimore_raw(raw_dir)
    t = np.asarray(seq["t"], dtype=np.float64)
    x = np.asarray(seq["x"], dtype=np.float64)          # (L, 25, 3)

    # Enforce strict monotonicity of the arrival stamps (a handful of Kinect rows repeat a tick).
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, x = t[keep], x[keep]
    if len(t) < min_len:
        raise ValueError(f"too short after monotonicity filter: {len(t)} frames")

    # F1b needs the NATIVE-rate signal in root-relative coordinates BEFORE we subsample, plus the
    # exact indices we are about to retain (so each kept frame maps to its native window). We do the
    # band-power here, then subsample, so the high band is measured at 30 Hz, not at the crippled
    # ~4 Hz of the retained grid.
    hf = None
    idx_full = np.arange(len(t))

    # Cap length: uniform INDEX subsample, keeping the ACTUAL stamps of retained frames.
    if len(t) > max_len:
        idx = np.linspace(0, len(t) - 1, max_len).round().astype(int)
        idx = np.unique(idx)
    else:
        idx = idx_full

    if compute_bandpower:
        import native_bandpower as nb                  # local import: only when the pivot is on
        x_rel_native = x - x[:, ROOT_JOINT: ROOT_JOINT + 1, :]
        t_native_s = (t - t[0])                         # already seconds from load_kimore_raw
        hf = nb.cached_bandpower(raw_dir, t_native_s, x_rel_native, idx)   # (len(idx), 25)

    if len(t) > max_len:
        t, x = t[idx], x[idx]

    t = t - t[0]                                        # start at 0; DURATION is preserved
    x = x - x[:, ROOT_JOINT: ROOT_JOINT + 1, :]         # root-relative

    # Type-0 scale: a scalar, so it commutes with rotation -> equivariance untouched.
    radius = np.linalg.norm(x, axis=-1)                 # (L, 25)
    scale = float(np.median(radius[radius > 0])) if np.any(radius > 0) else 1.0
    x = x / max(scale, 1e-6)

    out = {"t": t / TIME_SCALE, "x": x, "n_frames": len(t),
           "duration_s": float(t[-1]), "scale": scale, "raw_dir": raw_dir}
    if hf is not None:
        out["hf"] = hf.astype(np.float64)               # (L, 25) native-rate band RMS, invariant
    return out


def load_exercise(exercise_idx=1, max_len=150, verbose=True):
    """Load every subject for one exercise, joined to its clinical score."""
    meta_path = os.path.join(PROCESSED_ROOT, f"Exercise{exercise_idx}", "meta.csv")
    rows = list(csv.DictReader(open(meta_path)))
    ex = f"Es{exercise_idx}"

    samples, skipped = [], []
    for r in rows:
        try:
            s = load_sample(r["group"], r["subject_name"], ex, max_len=max_len)
        except Exception as e:                          # never silently drop a subject
            skipped.append((r["subject_name"], repr(e)))
            continue
        s["y"] = float(r["score"]) / SCORE_MAX
        s["subject"] = r["subject_name"]
        s["group"] = r["group"]
        s["cohort"] = "control" if r["group"].startswith("CG") else "patient"
        s["exercise"] = exercise_idx                    # 1..5
        samples.append(s)

    if verbose:
        print(f"[data] Exercise{exercise_idx}: {len(samples)} subjects loaded, "
              f"{len(skipped)} skipped")
        for name, err in skipped:
            print(f"       SKIPPED {name}: {err}")
    return samples


def load_all_exercises(max_len=150, exercises=(1, 2, 3, 4, 5), verbose=True):
    """Pool every (subject, exercise) recording -> ~380 sequences over 78 subjects.

    Exercise identity is carried on each sample and enters the model as a Type-0 (INVARIANT)
    scalar, so pooling costs nothing architecturally: a scalar channel commutes with every
    D_f(g) and the equivariance certificate is untouched.

    The unit of generalisation is still the SUBJECT: all five of a subject's recordings move
    together across the fold boundary (see subject_folds). Pooling multiplies the number of
    SEQUENCES by 5; it does not multiply the number of subjects, which is the real constraint.
    """
    samples = []
    for e in exercises:
        samples += load_exercise(e, max_len=max_len, verbose=verbose)
    if verbose:
        subs = {s["subject"] for s in samples}
        print(f"[data] POOLED: {len(samples)} sequences over {len(subs)} subjects "
              f"({len(exercises)} exercises)")
    return samples


def exercise_mean_floor(train_s, test_s):
    """The ONLY fair floor for a pooled model: predict the per-exercise TRAINING mean.

    A pooled model is given the exercise ID, so a trivial predictor that uses nothing BUT the
    exercise ID is the baseline it must beat. Scoring a pooled model against the GLOBAL training
    mean would manufacture a fake win: KIMORE's five exercises have different score
    distributions, so simply learning the five offsets would 'beat' a global-mean floor while
    demonstrating no subject-level skill whatsoever. Falls back to the global mean for an
    exercise unseen in training.
    """
    glob = float(np.mean([s["y"] for s in train_s]))
    per_ex = {}
    for e in {s["exercise"] for s in train_s}:
        per_ex[e] = float(np.mean([s["y"] for s in train_s if s["exercise"] == e]))
    return np.array([per_ex.get(s["exercise"], glob) for s in test_s], dtype=np.float64)


# =============================================================================
# 2. Batching (ragged lengths -> padded tensors + mask-free spline via per-sample L)
# =============================================================================
def collate(samples, dtype=torch.float32, device="cpu"):
    """Pad to the batch max length by REPEATING the final frame at strictly increasing stamps.

    Padding a CDE is subtle: the spline needs strictly increasing t. We extend the timeline past
    t_N with a frozen pose, so dX/ds = 0 over the padded tail and the latent state STOPS MOVING
    there (dZ = f(Z) dX = 0 for the control channels). Only the time channel keeps ticking, so
    padding is not perfectly inert -- which is why we instead integrate each sample on its OWN
    [t_0, t_N] span (see cde_model.integrate_rk4) and the padded tail is never visited. The pad
    exists purely to make a rectangular tensor.
    """
    B = len(samples)
    L = max(s["n_frames"] for s in samples)
    t = torch.zeros(B, L, dtype=dtype)
    x = torch.zeros(B, L, N_JOINTS, 3, dtype=dtype)
    has_hf = "hf" in samples[0]                          # F1b band-power channel present?
    hf = torch.zeros(B, L, N_JOINTS, dtype=dtype) if has_hf else None
    for i, s in enumerate(samples):
        n = s["n_frames"]
        t[i, :n] = torch.as_tensor(s["t"], dtype=dtype)
        x[i, :n] = torch.as_tensor(s["x"], dtype=dtype)
        if has_hf:
            hf[i, :n] = torch.as_tensor(s["hf"], dtype=dtype)   # padded tail stays 0; never pooled
        if n < L:                                       # frozen-pose tail on a continuing clock
            dt = float(s["t"][-1] - s["t"][-2]) if n >= 2 else 1e-2
            tail = s["t"][-1] + dt * np.arange(1, L - n + 1)
            t[i, n:] = torch.as_tensor(tail, dtype=dtype)
            x[i, n:] = torch.as_tensor(s["x"][-1], dtype=dtype)
    y = torch.tensor([s["y"] for s in samples], dtype=dtype)
    # exercise id as a 0-based index; a Type-0 (invariant) input, so equivariance is untouched.
    e = torch.tensor([s.get("exercise", 1) - 1 for s in samples], dtype=torch.long)
    n_real = torch.tensor([s["n_frames"] for s in samples], dtype=torch.long)
    if has_hf:
        return (t.to(device), x.to(device), y.to(device), e.to(device), n_real.to(device),
                hf.to(device))
    return (t.to(device), x.to(device), y.to(device), e.to(device), n_real.to(device))


class Batcher:
    """Length-bucketed batching: sorting by length keeps the pad tail short."""

    def __init__(self, samples, batch_size=8, shuffle=True, dtype=torch.float32, device="cpu"):
        self.samples = samples
        self.bs = batch_size
        self.shuffle = shuffle
        self.dtype = dtype
        self.device = device

    def __len__(self):
        return (len(self.samples) + self.bs - 1) // self.bs

    def __iter__(self):
        idx = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(idx)
            # bucket by length within a shuffled window to limit padding
            idx = sorted(idx, key=lambda i: (self.samples[i]["n_frames"] // 25, np.random.rand()))
            idx = np.array(idx)
            starts = np.arange(0, len(idx), self.bs)
            np.random.shuffle(starts)
        else:
            starts = np.arange(0, len(idx), self.bs)
        for st in starts:
            chunk = [self.samples[i] for i in idx[st: st + self.bs]]
            if chunk:
                yield collate(chunk, self.dtype, self.device)


# =============================================================================
# 3. Subject-disjoint folds
# =============================================================================
def subject_folds(samples, k=5, seed=0):
    """K subject-disjoint folds, STRATIFIED by cohort so every fold holds patients+controls.

    Folds are assigned over SUBJECTS, not over samples. In the pooled setting a subject owns
    five recordings; all five land in the same fold. Splitting on samples would put Es1 and Es3
    of the same person on opposite sides of the boundary -- the exact leak that inflates every
    published KIMORE number (see protocol_null.py).
    """
    rng = np.random.RandomState(seed)
    subj_cohort = {}
    for s in samples:
        subj_cohort[s["subject"]] = s["cohort"]

    subj_fold = {}
    for cohort in ("control", "patient"):
        subs = sorted([u for u, c in subj_cohort.items() if c == cohort])
        rng.shuffle(subs)
        for j, u in enumerate(subs):
            subj_fold[u] = j % k

    folds = [[] for _ in range(k)]
    for i, s in enumerate(samples):
        folds[subj_fold[s["subject"]]].append(i)
    return [sorted(f) for f in folds]


def split(samples, folds, test_fold, val_fold=None):
    test = folds[test_fold]
    val = folds[val_fold] if val_fold is not None else []
    train = [i for f_i, f in enumerate(folds) if f_i not in (test_fold, val_fold) for i in f]
    pick = lambda ids: [samples[i] for i in ids]
    return pick(train), pick(val), pick(test)


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exercise", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=150)
    args = ap.parse_args()

    samples = load_exercise(args.exercise, max_len=args.max_len)
    L = np.array([s["n_frames"] for s in samples])
    D = np.array([s["duration_s"] for s in samples])
    Y = np.array([s["y"] for s in samples])
    cv = []
    for s in samples:
        dt = np.diff(s["t"] * TIME_SCALE)
        cv.append(dt.std() / dt.mean())
    cv = np.array(cv)

    print(f"\n  control points   min/med/max : {L.min()} / {int(np.median(L))} / {L.max()}")
    print(f"  duration (s)     min/med/max : {D.min():.1f} / {np.median(D):.1f} / {D.max():.1f}")
    print(f"  label y=score/50 min/med/max : {Y.min():.3f} / {np.median(Y):.3f} / {Y.max():.3f}")
    print(f"  CV(dt) after subsample       : med {np.median(cv):.3f}  max {cv.max():.3f}")
    print(f"  cohorts: {sum(s['cohort']=='control' for s in samples)} control / "
          f"{sum(s['cohort']=='patient' for s in samples)} patient")

    folds = subject_folds(samples, k=5)
    print(f"\n  5 subject-disjoint folds: sizes {[len(f) for f in folds]}")
    tr, va, te = split(samples, folds, test_fold=0, val_fold=1)
    print(f"  fold 0 as test: train={len(tr)} val={len(va)} test={len(te)}")
    assert not ({s['subject'] for s in tr} & {s['subject'] for s in te}), "subject leak!"
    print("  subject-disjointness asserted OK")

    t, x, y, n = collate(te[:4])
    print(f"\n  collated batch: t{tuple(t.shape)} x{tuple(x.shape)} y{tuple(y.shape)}")
    print(f"  n_real={n.tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
