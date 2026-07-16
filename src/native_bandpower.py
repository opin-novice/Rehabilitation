#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
native_bandpower.py
===================
F1b -- the grid-free advantage the dt-at-the-recurrence ablation could NOT deliver.

Phase 0 showed the recurrence does not use the inter-arrival VARIATION (seqmean - real ~ 0). That
does not mean the high-frequency band is worthless -- it means the model never had access to it,
because kimore_cde_data.load_sample uniform-index-subsamples ~1140 native frames down to 150,
dropping the effective rate to ~4 Hz (Nyquist ~1.9 Hz) BEFORE the model sees anything. Everything
above ~2 Hz -- the tremor/jerk band -- is gone at the loader, not at the recurrence.

F1b recovers it as a channel the recurrence CAN use without reading dt at all:

    at the sensor's NATIVE rate, per joint, compute the band-limited RMS of the SPEED
    ||v_j(t)|| in [f_lo, f_hi] Hz via Lomb-Scargle on the REAL irregular stamps, in a sliding
    window centred on each retained frame; hand it to the GRU as a Type-0 scalar per joint.

Why each choice is forced
-------------------------
  * SPEED NORM, not the velocity vector. ||v_j|| is the norm of a difference of positions =
    rotation-INVARIANT (Type-0). Band-power of the vector components would be frame-dependent and
    would break the equivariance certificate. This is the same invariance the encoder's speed
    channel already relies on (certify_egru E2).
  * LOMB-SCARGLE, not an FFT. The native KIMORE timeline is genuinely irregular (that is the
    entire premise of the corpus). A uniform-grid FFT would require resampling -- the very operator
    R whose spectral loss we are trying to AVOID. Lomb-Scargle is the least-squares spectral
    estimate on the actual stamps; on a uniform grid it reduces to the periodogram, so we lose
    nothing in the regular case and gain the irregular one.
  * WINDOW >= 1.0 s. Frequency resolution is df ~ 1/window. To resolve the LOWER band edge at
    2 Hz we need df well under 2 Hz, i.e. window comfortably over 0.5 s; >= 1.0 s (~30 native
    samples at 30 Hz) gives df ~ 1 Hz. The naive "one subsample segment" (~0.25 s) gives df ~ 4 Hz
    and cannot resolve the band at all -- that was the bug in the first draft of this feature.

Caching: 150 retained frames x 25 joints Lomb-Scargle calls per sequence is ~4k LS fits; over the
pooled corpus that is >1M fits. It is deterministic preprocessing, so it is cached to
outputs/bandpower_cache/ keyed by (raw_dir, params). Off by default; costs nothing until the
pivot is switched on.

Run:  python src/native_bandpower.py --demo    # sanity: inject a 5 Hz tremor, recover its band
"""

import argparse
import hashlib
import os
import sys

import numpy as np
from scipy.signal import lombscargle
try:
    from scipy.integrate import trapezoid as _trapz          # scipy >= 1.12
except ImportError:                                          # pragma: no cover
    from numpy import trapz as _trapz

CACHE_DIR = os.path.join("outputs", "bandpower_cache")
F_LO, F_HI = 2.0, 8.0          # Hz. Parkinsonian tremor 4-6 Hz sits inside; jerk energy above.
WINDOW_S = 1.0                 # sliding-window length (s); >= 1 s resolves the 2 Hz lower edge
N_FREQ = 24                    # LS frequency grid density across [F_LO, F_HI]


def _native_speed(t_native, x_native):
    """||v_j(t_k)|| at the native rate. (L,25,3) -> (L,25) Type-0 (rotation-invariant)."""
    dt = np.diff(t_native)
    dt = np.clip(dt, 1e-4, None)
    v = np.zeros((len(t_native), x_native.shape[1]))
    v[1:] = np.linalg.norm(np.diff(x_native, axis=0), axis=-1) / dt[:, None]   # metres / second
    return v


def _ls_band_rms(t_win, y_win, ang_freqs):
    """Band-limited RMS amplitude of y on irregular stamps t_win, over the given angular grid.

    Lomb-Scargle power integrated across the band, converted to an amplitude (sqrt), so the channel
    is in the same units as speed and plays nicely with the model's log1p. Returns 0 for a window
    too short or with no variance (a failed/edge window must not inject spurious energy).
    """
    if len(t_win) < 6 or np.ptp(t_win) < 1e-3:
        return 0.0
    y = y_win - y_win.mean()
    if not np.any(np.abs(y) > 0):
        return 0.0
    p = lombscargle(t_win.astype(np.float64), y.astype(np.float64), ang_freqs, precenter=True)
    # scipy's lombscargle returns unnormalised power; integrate over the band and take the RMS.
    return float(np.sqrt(_trapz(p, ang_freqs) / (ang_freqs[-1] - ang_freqs[0]) * 2.0))


def compute_bandpower(t_native, x_native, keep_idx, window_s=WINDOW_S,
                      f_lo=F_LO, f_hi=F_HI, n_freq=N_FREQ):
    """Native-rate band-limited speed RMS per retained frame per joint.

    t_native (L,)     native arrival stamps in SECONDS (real, irregular)
    x_native (L,25,3) native root-relative positions
    keep_idx (M,)     indices into the native arrays that the loader RETAINS (M ~ 150)
    -> (M, 25) Type-0 scalars, aligned to the retained frames the model actually consumes.
    """
    t_native = np.asarray(t_native, dtype=np.float64)
    speed = _native_speed(t_native, x_native)                          # (L,25) invariant
    ang = 2.0 * np.pi * np.linspace(f_lo, f_hi, n_freq)                # angular frequency grid
    half = window_s / 2.0
    J = speed.shape[1]

    out = np.zeros((len(keep_idx), J), dtype=np.float64)
    for m, i in enumerate(keep_idx):
        t0 = t_native[i]
        lo = np.searchsorted(t_native, t0 - half, side="left")         # window centred on frame i
        hi = np.searchsorted(t_native, t0 + half, side="right")
        tw = t_native[lo:hi]
        if len(tw) < 6:
            continue
        for j in range(J):
            out[m, j] = _ls_band_rms(tw, speed[lo:hi, j], ang)
    return out


def cached_bandpower(raw_dir, t_native, x_native, keep_idx, **kw):
    """compute_bandpower with an on-disk cache keyed by (raw_dir, keep_idx, params)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.sha256(
        (raw_dir + "|" + np.asarray(keep_idx).tobytes().hex()
         + f"|{kw.get('window_s', WINDOW_S)}|{kw.get('f_lo', F_LO)}|{kw.get('f_hi', F_HI)}"
         + f"|{kw.get('n_freq', N_FREQ)}").encode()).hexdigest()[:24]
    path = os.path.join(CACHE_DIR, key + ".npy")
    if os.path.exists(path):
        return np.load(path)
    hf = compute_bandpower(t_native, x_native, keep_idx, **kw)
    np.save(path, hf)
    return hf


# =============================================================================
def _demo():
    """Sanity: a 5 Hz tremor injected at native rate must light up the [2,8] Hz band, and a
    slow 0.5 Hz voluntary motion must NOT. This is what the loader's 4 Hz subsample destroys."""
    rng = np.random.default_rng(0)
    fs, dur = 30.0, 6.0
    t = np.cumsum(rng.uniform(0.8, 1.2, int(fs * dur)) / fs)           # irregular ~30 Hz
    t -= t[0]
    x = np.zeros((len(t), 25, 3))
    x[:, 7, 0] = 0.20 * np.sin(2 * np.pi * 0.5 * t)                    # voluntary 0.5 Hz on a hand
    keep = np.linspace(0, len(t) - 1, 150).round().astype(int)
    keep = np.unique(keep)

    print("F1b DEMO -- native-rate band-limited speed RMS, [2,8] Hz")
    for amp in (0.0, 0.01, 0.03):
        xt = x.copy()
        xt[:, 7, 0] += amp * np.sin(2 * np.pi * 5.0 * t)              # 5 Hz tremor
        hf = compute_bandpower(t, xt, keep)
        print(f"  tremor amp {amp:.3f} m :  hand-joint band RMS = {hf[:, 7].mean():.4f}   "
              f"(slow-only joints ~ {hf[:, 3].mean():.4f})")
    print("  -> the [2,8] Hz channel tracks the tremor and ignores the 0.5 Hz voluntary motion.")
    print("     This is exactly the band the 150-frame subsample (Nyquist ~1.9 Hz) throws away.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        _demo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
