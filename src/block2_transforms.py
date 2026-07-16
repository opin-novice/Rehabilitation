#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
block2_transforms.py
====================
The two TEST-TIME transforms the trained Block-2/3 experiments sweep over, plus the resampling
operator R that every fixed-grid baseline is forced to apply.

Both transforms act on a LOADED sample dict from kimore_cde_data (real KIMORE arrival stamps).
Neither is ever applied at training time: the model trains on clean, canonical data and is
attacked only at test time. That is the whole point -- we are testing a STRUCTURAL property,
not a learned augmentation. If we trained on rotated data, a flat cross-viewpoint curve would
prove nothing.

FAIRNESS (PROJECT_BRIEF 8.4)
---------------------------
Both methods receive the byte-identical corrupted timeline (t~_k, x_k). The baseline is then
given its BEST CASE: we report it under linear, cubic and forward-fill resampling and let it
keep whichever is strongest. Ours reads the actual stamps. Beating their best workaround is
what defuses the "you sabotaged the baseline" objection.

  * corrupt_sample()  : Gilbert-Elliott burst drops + AR(1)/Gaussian timestamp jitter, applied
                        to the sample's OWN real timeline. Reuses the calibrated, hash-locked
                        operators from corruption_pipeline.py (Task 9).
  * rotate_sample()   : camera-azimuth rotation about the gravity (vertical) axis.
  * to_fixed_grid()   : the resampling operator R. Ours does NOT use this.
"""

import hashlib
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corruption_pipeline as cp        # noqa: E402  (calibrated GE drops, jitter, resample)
import kimore_cde_data as kd            # noqa: E402

# Kinect v2 camera frame: +Y is up (the gravity axis). A camera azimuth change is a rotation
# ABOUT that axis -- the horizontal sweep a clinician actually makes when repositioning a sensor.
UP_AXIS = np.array([0.0, 1.0, 0.0])


# =============================================================================
# 1. Cross-viewpoint (Block 3)
# =============================================================================
def azimuth_matrix(deg: float) -> np.ndarray:
    """Rotation about the vertical axis by `deg` degrees (camera azimuth change)."""
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]], dtype=np.float64)


def rotate_sample(sample: dict, deg: float) -> dict:
    """Rotate a sample's skeleton about the vertical axis. Timestamps untouched.

    HONESTY: for 3D skeleton input this is EXACTLY what moving the camera does -- the joint
    coordinates are expressed in the camera frame, so an azimuth change is a rigid rotation of
    that frame. What it does NOT simulate is the sensor's own pose-estimation degrading at
    oblique angles (Kinect's tracker is trained frontally and gets noisier in profile). So this
    isolates the GEOMETRIC effect of viewpoint, which is the effect our theorem is about, and
    it understates the difficulty a real camera move would present to BOTH methods equally.
    We state this in the paper rather than claiming a full sim-to-real viewpoint result.
    """
    out = dict(sample)
    R = azimuth_matrix(deg)
    out["x"] = sample["x"] @ R.T                     # (L, 25, 3), row-vector convention
    return out


# =============================================================================
# 2. Irregular-sampling corruption (Block 2)
# =============================================================================
def corrupt_sample(sample: dict, drop_rate: float, jitter_scale: float = 1.0,
                   mean_burst: float = 3.0, seed: int = 0) -> dict:
    """Apply burst drops + timestamp jitter to a sample's OWN real timeline.

    Frames are removed by a Gilbert-Elliott two-state Markov chain (drops CLUSTER, as they do in
    a real USB/CPU-starved Kinect -- independent Bernoulli thinning would be the easy case), and
    the surviving stamps are perturbed by Gaussian thermal noise plus an AR(1) oscillator drift.
    Endpoints are anchored so every method integrates over the SAME [t_0, t_N] span and the
    comparison is not confounded by a changed integration interval.

    Returns a sample with FEWER frames and jittered stamps. Nothing is interpolated: this is
    what the sensor actually delivered.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(sample["t"], dtype=np.float64) * kd.TIME_SCALE     # back to seconds
    x = np.asarray(sample["x"], dtype=np.float64)
    n = len(t)

    # --- burst drops -------------------------------------------------------
    p_gb, p_bg, eps_g, eps_b = cp.ge_params_from(drop_rate, mean_burst)
    keep = cp.gilbert_elliott_mask(n, p_gb, p_bg, eps_g, eps_b, rng)
    keep[0] = keep[-1] = True                                          # anchor the span
    if keep.sum() < 8:                                                 # spline needs control points
        keep[np.linspace(0, n - 1, 8).astype(int)] = True

    # --- timestamp jitter --------------------------------------------------
    base = float(np.median(np.diff(t)))
    tt = cp.jitter_timestamps(t, sigma_j=0.25 * base * jitter_scale, alpha=0.9,
                              sigma_d=0.10 * base * jitter_scale, rng=rng, base_period=base)

    t_c, x_c = tt[keep], x[keep]
    t_c = t_c - t_c[0]

    out = dict(sample)
    out["t"] = t_c / kd.TIME_SCALE
    out["x"] = x_c
    out["n_frames"] = len(t_c)
    out["drop_rate"] = drop_rate
    return out


# =============================================================================
# 3. The resampling operator R (baselines only)
# =============================================================================
def to_fixed_grid(sample: dict, n_frames: int, operator: str = "linear") -> np.ndarray:
    """R : (t~_k, x_k) -> uniform T x J x 3 grid. The baseline CANNOT avoid this step.

    Its architecture consumes a fixed frame axis, so irregular stamps must be projected onto a
    uniform grid first. That projection is a low-pass operator: it destroys spectral energy above
    the grid's Nyquist rate -- precisely the tremor/jerk band that is clinically decisive
    (quantified as I_loss by corruption_pipeline.spectral_loss). The information is gone before
    the baseline's first layer sees anything, which is the mechanism behind Block 2.
    """
    t = np.asarray(sample["t"], dtype=np.float64)
    x = np.asarray(sample["x"], dtype=np.float64)
    L, J, _ = x.shape
    grid = np.linspace(t[0], t[-1], n_frames)
    xr = cp.resample(operator, t, x.reshape(L, J * 3), grid)
    return xr.reshape(n_frames, J, 3)


def batch_fixed_grid(samples, n_frames: int, operator: str = "linear",
                     device="cpu", dtype=torch.float32):
    """Collate onto a shared uniform grid -> (x (B,T,J,3), y (B,), ex_id (B,)) for the baseline."""
    x = np.stack([to_fixed_grid(s, n_frames, operator) for s in samples])
    y = np.array([s["y"] for s in samples], dtype=np.float64)
    e = np.array([s.get("exercise", 1) - 1 for s in samples], dtype=np.int64)
    return (torch.as_tensor(x, dtype=dtype, device=device),
            torch.as_tensor(y, dtype=dtype, device=device),
            torch.as_tensor(e, dtype=torch.long, device=device))


# =============================================================================
# 4. Shared TRAINING-time augmentation  (the fair Block-2 re-run)
# =============================================================================
# Block 2 as first run was not a test of the architecture: the EGRU was trained on clean, near-
# uniform stamps (dt <= 0.026 s) and then attacked with gaps up to 0.22 s -- ~10x outside anything
# it had ever seen. It read dt natively but had never learned what a large gap MEANS. That is a
# training deficiency, not an architectural one, and it must be removed before the structural
# question can even be asked.
#
# So both arms now get the SAME augmentation budget. Unlike Block 3 -- where training on rotations
# would void a theorem and we therefore never do it -- the irregular-sampling claim was never a
# theorem, so augmenting is legitimate. What is NOT legitimate is augmenting only ours: that is the
# "you sabotaged the baseline" objection with the sign flipped. Hence one shared stream.
#
# LEVEL HOLD-OUT: train on drops ~ U[0, aug_drop] with aug_drop = 0.3, but TEST out to 0.7. The
# two hardest levels are therefore genuine extrapolation, not memorised attack. We are not training
# on the test corruption.
def aug_epoch(samples, batch_size: int, max_drop: float, jitter_scale: float,
              seed: int, epoch: int):
    """Yield one epoch of shuffled, corrupted sample-dict chunks.

    FAIRNESS LOCK: both trainers call this with the same (seed, epoch) and therefore consume the
    BYTE-IDENTICAL corrupted stream -- same order, same per-sample drop level, same jitter. The
    only thing that differs is what each does with a chunk: ours reads the actual stamps; the
    baseline must first push that same chunk through R. Assert with aug_stream_hash().
    """
    rng = np.random.default_rng([seed, epoch])
    order = rng.permutation(len(samples))
    for i in range(0, len(samples), batch_size):
        chunk = [samples[j] for j in order[i: i + batch_size]]
        if max_drop > 0:
            chunk = [corrupt_sample(s, float(rng.uniform(0.0, max_drop)),
                                    jitter_scale=jitter_scale,
                                    seed=int(rng.integers(1 << 31)))
                     for s in chunk]
        yield chunk


def aug_stream_hash(samples, batch_size: int, max_drop: float, jitter_scale: float,
                    seed: int, epoch: int) -> str:
    """SHA-256 over an epoch's corrupted stream. Both trainers must print the SAME digest."""
    h = hashlib.sha256()
    for chunk in aug_epoch(samples, batch_size, max_drop, jitter_scale, seed, epoch):
        for s in chunk:
            h.update(np.ascontiguousarray(np.asarray(s["t"], dtype=np.float64)).tobytes())
            h.update(np.ascontiguousarray(np.asarray(s["x"], dtype=np.float64)).tobytes())
    return h.hexdigest()
