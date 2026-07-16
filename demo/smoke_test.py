#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
smoke_test.py
=============
The whole demo thesis, in code, re-measured rather than asserted:

    ROTATE the input skeleton about the vertical axis  ->  EGRU score is IDENTICAL (~1e-6),
                                                           PCT score DECAYS.

If the EGRU invariance gate fails, the headline act is not real and we stop. It passes here on any
skeleton because the invariance is architectural (certify_egru), independent of whether the live
domain matches training -- which is exactly why Act 2 is bulletproof.

Also exercises the full live path: mediapipe_to_kinect25 -> preprocess_window -> DemoEngine.predict,
so a green run proves the remap, the preprocessing, and both checkpoints load and infer.

Run:  python demo/smoke_test.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mp_to_kinect import mediapipe_to_kinect25, preprocess_window, N_KINECT   # noqa: E402


def synthetic_mediapipe_buffer(T=64, fps=30.0, seed=0):
    """A plausible moving human as MediaPipe-style world landmarks (T, 33, 3), metres, +Y down.

    A rough standing skeleton with the arms doing a slow abduction sweep -- enough structure that
    the remap produces a sane skeleton and the models produce a non-degenerate score. Realism is
    irrelevant to the invariance gate (that holds for ANY input); it only makes the score plausible.
    """
    rng = np.random.default_rng(seed)
    lm = np.zeros((33, 3), dtype=np.float64)                 # base pose, +Y down (MediaPipe frame)
    # y grows downward; put hips at 0, head negative-y (up), feet positive-y (down).
    lm[0] = [0.00, -0.62, 0.00]                              # nose
    lm[7] = [-0.07, -0.60, 0.02]; lm[8] = [0.07, -0.60, 0.02]   # ears
    lm[11] = [-0.18, -0.50, 0.0]; lm[12] = [0.18, -0.50, 0.0]   # shoulders
    lm[13] = [-0.20, -0.25, 0.0]; lm[14] = [0.20, -0.25, 0.0]   # elbows
    lm[15] = [-0.22, 0.00, 0.0]; lm[16] = [0.22, 0.00, 0.0]     # wrists
    lm[17] = [-0.24, 0.05, 0.0]; lm[18] = [0.24, 0.05, 0.0]     # pinkies
    lm[19] = [-0.23, 0.05, 0.02]; lm[20] = [0.23, 0.05, 0.02]   # index
    lm[21] = [-0.20, 0.03, 0.02]; lm[22] = [0.20, 0.03, 0.02]   # thumbs
    lm[23] = [-0.10, 0.0, 0.0]; lm[24] = [0.10, 0.0, 0.0]       # hips
    lm[25] = [-0.11, 0.35, 0.0]; lm[26] = [0.11, 0.35, 0.0]     # knees
    lm[27] = [-0.11, 0.70, 0.0]; lm[28] = [0.11, 0.70, 0.0]     # ankles
    lm[31] = [-0.11, 0.75, -0.08]; lm[32] = [0.11, 0.75, -0.08] # foot index

    buf = np.repeat(lm[None], T, axis=0)
    # slow arm-abduction: raise wrists/elbows over the window (moves in the frontal plane)
    phase = np.linspace(0, np.pi, T)
    lift = 0.25 * np.sin(phase)
    for k, sgn in [(13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 1),
                   (19, 1), (20, 1), (21, 1), (22, 1)]:
        buf[:, k, 1] -= lift                                 # -y = upward
        buf[:, k, 0] += 0.15 * np.sin(phase) * np.sign(buf[0, k, 0])
    buf += rng.normal(0, 0.004, buf.shape)                   # mild sensor jitter
    t = np.arange(T) / fps
    return buf, t


def build_sample():
    """Full live path: synthetic MediaPipe frames -> Kinect-25 -> training-matched sample dict."""
    mp_buf, t = synthetic_mediapipe_buffer()
    kin = np.stack([mediapipe_to_kinect25(f) for f in mp_buf], axis=0)   # (T,25,3)
    assert kin.shape[1:] == (N_KINECT, 3)
    return preprocess_window(kin, t)


def main():
    sample = build_sample()
    print(f"[remap] built sample: {sample['n_frames']} frames, "
          f"duration {sample['duration_s']:.2f}s, scale {sample['scale']:.3f}")

    from engine import DemoEngine
    eng = DemoEngine()
    print(f"[engine] loaded EGRU x{len(eng.egru)} + PCT x{len(eng.pct)} on {eng.device}")

    angles = [0, 15, 30, 45, 60, 90, 120, 180]
    base = eng.predict(sample, exercise=1, angle=0)
    e0, p0 = base["egru"], base["pct"]

    print(f"\n{'angle':>6} | {'EGRU':>8} {'EGRU degr':>10} | {'PCT':>8} {'PCT degr':>9}")
    print("-" * 52)
    egru_max_degr = 0.0
    pct_max_degr = 0.0
    for a in angles:
        r = eng.predict(sample, exercise=1, angle=a)
        ed = abs(r["egru"] - e0)
        pd = abs(r["pct"] - p0)
        egru_max_degr = max(egru_max_degr, ed)
        pct_max_degr = max(pct_max_degr, pd)
        print(f"{a:>6} | {r['egru']:>8.3f} {ed:>10.2e} | {r['pct']:>8.3f} {pd:>9.3f}")

    print("-" * 52)
    print(f"EGRU max degradation over rotation : {egru_max_degr:.2e}  (clinical units, /50)")
    print(f"PCT  max degradation over rotation : {pct_max_degr:.3f}")

    # --- the gate ---
    TOL = 1e-3        # clinical units; architectural invariance is ~1e-5, this is generous slack
    assert egru_max_degr < TOL, (
        f"INVARIANCE GATE FAILED: EGRU moved {egru_max_degr:.2e} > {TOL} under rotation. "
        f"The headline act is not real -- stop and diagnose.")
    print(f"\n[PASS] EGRU invariance gate: degradation {egru_max_degr:.2e} < {TOL}.")
    if pct_max_degr > egru_max_degr * 100:
        print(f"[PASS] Contrast is live: PCT decays {pct_max_degr:.3f} while EGRU stays flat.")
    else:
        print(f"[WARN] PCT decay ({pct_max_degr:.3f}) is weak on this synthetic pose; "
              f"expected on real motion at larger angles.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
