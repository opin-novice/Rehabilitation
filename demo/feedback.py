#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
feedback.py
===========
Turns a buffered exercise session into an explainable, plain-language report.

Everything here is computed from the REAL skeleton motion (raw Kinect-25 metres per frame, with
real timestamps) using transparent biomechanics -- range of motion, smoothness, symmetry, tempo,
and rep-to-rep consistency. Each number is defensible to a judge ("your range of motion reached
84% of the target excursion"), which is exactly why the *headline* score is this composite and not
the raw AI number: the SE(3)-equivariant model was trained on Kinect depth and reads low on a
webcam (domain shift), so it rides along as a smaller calibrated "AI model score" badge instead.

The three small numeric helpers (_angle_between, _native_speed, _periodicity_score) and the LR_PAIRS
table are re-implemented inline (they are a few lines each) to keep this module dependent only on
numpy + scipy, mirroring:
    src/ntu_dataset.py:_angle_between, src/native_bandpower.py:_native_speed,
    src/novelty/periodicity.py:_periodicity_score, src/chirality.py:LR_PAIRS
"""

import json
import os

import numpy as np
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Kinect-25 joint indices (see demo/mp_to_kinect.py:KINECT_NAMES)
# ---------------------------------------------------------------------------
SPINE_BASE, SPINE_MID, NECK, HEAD, SPINE_SHOULDER = 0, 1, 2, 3, 20
SHO_L, SHO_R = 4, 8
ELB_L, ELB_R = 5, 9
WRI_L, WRI_R = 6, 10
HIP_L, HIP_R = 12, 16
KNEE_L, KNEE_R = 13, 17
ANK_L, ANK_R = 14, 18

# left/right pairs for symmetry (src/chirality.py:LR_PAIRS)
LR_PAIRS = [(4, 8), (5, 9), (6, 10), (7, 11), (12, 16), (13, 17), (14, 18), (15, 19)]

UP = np.array([0.0, 1.0, 0.0])          # +Y is up after mediapipe_to_kinect25's axis flip

EXERCISE_NAMES = {
    1: "Trunk Lateral Flexion",
    2: "Trunk Forward Flexion",
    3: "Trunk Rotation",
    4: "Hip Abduction",
    5: "Hip Circumduction",
}

HOWTO = {
    1: "Stand tall, bend your torso slowly to each side.",
    2: "Stand tall, bend forward from the hips and return.",
    3: "Keep hips forward, rotate your torso left and right.",
    4: "Stand on one leg, lift the other out to the side.",
    5: "Stand on one leg, draw slow circles with the other.",
}

# Per-exercise measurement profile.
#   angle   : how to build the primary angle series theta(t) in degrees
#   target  : "good" peak-to-peak excursion (deg) that maps to 100% range of motion
#   symmetry: 'bidir' (compare the two movement directions), 'lr' (left vs right limb),
#             or 'level' (shoulder/hip levelness during the movement)
#   cadence : (min, max) reps-per-minute that reads as a healthy, controlled pace
EXERCISE_PROFILES = {
    1: dict(angle=("trunk", "frontal"),   target=45.0, symmetry="bidir", cadence=(6, 25)),
    2: dict(angle=("trunk", "sagittal"),  target=60.0, symmetry="level", cadence=(5, 20)),
    3: dict(angle=("rotation", None),     target=70.0, symmetry="bidir", cadence=(6, 25)),
    4: dict(angle=("abduction", None),    target=40.0, symmetry="lr",    cadence=(6, 25)),
    5: dict(angle=("circumduction", None),target=40.0, symmetry="level", cadence=(6, 25)),
}

METRIC_ORDER = ["rom", "smoothness", "symmetry", "tempo", "consistency"]
METRIC_LABELS = {
    "rom": "Range of motion",
    "smoothness": "Smoothness",
    "symmetry": "Symmetry / balance",
    "tempo": "Tempo & rhythm",
    "consistency": "Consistency",
}

# Plain-language phrase bank: a 'good' line (strength) and a 'tip' line (improvement) per metric.
PHRASES = {
    "rom":        ("Excellent range of motion — you moved through the full target range.",
                   "Try to move a little further through the movement for a fuller range of motion."),
    "smoothness": ("Very smooth, controlled movement with no jerkiness.",
                   "Aim for slower, smoother motion — avoid sudden or shaky movements."),
    "symmetry":   ("Well balanced — both sides / directions matched nicely.",
                   "Work on balancing both sides evenly — one side moved more than the other."),
    "tempo":      ("Great steady rhythm and a healthy, controlled pace.",
                   "Keep a steadier, even rhythm — hold a consistent pace throughout."),
    "consistency":("Every repetition looked consistent — strong movement control.",
                   "Try to make each repetition look the same for more consistent control."),
}


# ---------------------------------------------------------------------------
# inlined numeric helpers
# ---------------------------------------------------------------------------
def _native_speed(t, x):
    """||v_j(t_k)||, (L,25,3)->(L,25) m/s. (src/native_bandpower.py:_native_speed)"""
    dt = np.clip(np.diff(t), 1e-4, None)
    v = np.zeros((len(t), x.shape[1]))
    v[1:] = np.linalg.norm(np.diff(x, axis=0), axis=-1) / dt[:, None]
    return v


def _periodicity_score(seq):
    """Spectral concentration of a (T,J,C) sequence in [0,1]. (src/novelty/periodicity.py)"""
    seqc = seq - seq.mean(axis=0, keepdims=True)
    s = (seqc ** 2).sum(axis=(1, 2))
    s = s - s.mean()
    if np.allclose(s, 0):
        return 0.0
    power = np.abs(np.fft.rfft(s)) ** 2
    power[0] = 0.0
    total = power.sum()
    return float(power.max() / total) if total > 0 else 0.0


def _lerp(v, x0, x1, y0, y1):
    """Clamped linear map of v from [x0,x1] onto [y0,y1]."""
    if x1 == x0:
        return y0
    f = (v - x0) / (x1 - x0)
    return float(np.clip(y0 + f * (y1 - y0), min(y0, y1), max(y0, y1)))


def _sparc(speed, fs, fc=10.0, amp_th=0.05):
    """Spectral arc length smoothness of a speed profile. More negative = less smooth.
    Balasubramanian et al. 2015. Returns a float ~[-1.5 smooth .. -6 jerky]."""
    if len(speed) < 4 or np.allclose(speed, 0):
        return -6.0
    n = int(2 ** (np.ceil(np.log2(len(speed))) + 2))       # zero-pad for resolution
    mag = np.abs(np.fft.rfft(speed, n))
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = mag / (mag.max() + 1e-12)
    keep = freq <= fc
    mag, freq = mag[keep], freq[keep]
    inband = mag >= amp_th
    if inband.sum() < 2:
        return -6.0
    mag, freq = mag[inband], freq[inband]
    df = np.diff(freq) / (freq[-1] - freq[0] + 1e-12)
    dm = np.diff(mag)
    return float(-np.sum(np.sqrt(df ** 2 + dm ** 2)))


# ---------------------------------------------------------------------------
# primary angle series per exercise
# ---------------------------------------------------------------------------
def _signed_plane_angle(v, plane):
    """Signed angle (deg) of vector v away from vertical, within a plane.
    frontal: tilt in X about +Y ; sagittal: tilt in Z about +Y."""
    if plane == "frontal":
        return np.degrees(np.arctan2(v[..., 0], v[..., 1]))
    return np.degrees(np.arctan2(v[..., 2], v[..., 1]))       # sagittal


def _azimuth(v):
    """Heading angle (deg) in the transverse (X-Z) plane."""
    return np.degrees(np.arctan2(v[..., 2], v[..., 0]))


def _angle_from_down(thigh):
    """Angle (deg) of a limb vector away from straight-down (0 = hanging, grows as it lifts)."""
    down = -UP
    n = np.linalg.norm(thigh, axis=-1)
    cs = np.clip((thigh @ down) / np.clip(n, 1e-8, None), -1.0, 1.0)
    return np.degrees(np.arccos(cs))


def primary_angle_series(x, profile):
    """Return theta(t) in degrees plus an optional second series for symmetry.

    x : (T,25,3) raw metres, +Y up. Returns (theta, aux) where aux is a dict of side series
    used by the symmetry metric (may be empty).
    """
    kind, plane = profile["angle"]
    if kind == "trunk":
        v = x[:, SPINE_SHOULDER] - x[:, SPINE_BASE]
        theta = _signed_plane_angle(v, plane)
        return theta, {}
    if kind == "rotation":
        sho = x[:, SHO_R] - x[:, SHO_L]
        hip = x[:, HIP_R] - x[:, HIP_L]
        theta = _azimuth(sho) - _azimuth(hip)
        theta = (theta + 180.0) % 360.0 - 180.0           # wrap to [-180,180]
        return theta, {}
    if kind == "abduction":
        left = _angle_from_down(x[:, KNEE_L] - x[:, HIP_L])
        right = _angle_from_down(x[:, KNEE_R] - x[:, HIP_R])
        # the "moving" leg is whichever has the larger excursion; theta drives it
        theta = left if (left.max() - left.min()) >= (right.max() - right.min()) else right
        return theta, {"left": left, "right": right}
    if kind == "circumduction":
        ank = x[:, ANK_R] - x[:, HIP_R]
        ank_l = x[:, ANK_L] - x[:, HIP_L]
        use = ank if np.std(ank) >= np.std(ank_l) else ank_l
        theta = _angle_from_down(use)
        return theta, {}
    raise ValueError(f"unknown angle kind {kind}")


# ---------------------------------------------------------------------------
# metrics (each returns a value in [0,100] plus a small detail dict)
# ---------------------------------------------------------------------------
def m_range_of_motion(theta, profile):
    excursion = float(np.percentile(theta, 97) - np.percentile(theta, 3))   # robust peak-to-peak
    pct = _lerp(excursion, 0.15 * profile["target"], profile["target"], 0, 100)
    return pct, {"excursion_deg": round(excursion, 1), "target_deg": profile["target"]}


def m_smoothness(t, x, moving_joints):
    fs = (len(t) - 1) / max(t[-1] - t[0], 1e-3)
    speed = _native_speed(t, x)[:, moving_joints].mean(axis=1)
    sparc = _sparc(speed, fs)
    pct = _lerp(sparc, -5.0, -1.6, 0, 100)
    return pct, {"sparc": round(sparc, 2)}


def m_symmetry(theta, aux, profile, x):
    mode = profile["symmetry"]
    if mode == "lr" and aux:
        l = aux["left"].max() - aux["left"].min()
        r = aux["right"].max() - aux["right"].min()
        bal = 1.0 - abs(l - r) / (l + r + 1e-6)
        return float(np.clip(bal, 0, 1) * 100), {"left_deg": round(l, 1), "right_deg": round(r, 1)}
    if mode == "bidir":
        rest = float(np.median(theta))
        pos = float(np.percentile(theta, 97) - rest)
        neg = float(rest - np.percentile(theta, 3))
        bal = 1.0 - abs(pos - neg) / (pos + neg + 1e-6)
        return float(np.clip(bal, 0, 1) * 100), {"pos_deg": round(pos, 1), "neg_deg": round(neg, 1)}
    # 'level': how level the shoulder line stays through the movement (low tilt std = symmetric)
    sho = x[:, SHO_R] - x[:, SHO_L]
    tilt = np.degrees(np.arctan2(sho[:, 1], np.linalg.norm(sho[:, [0, 2]], axis=1) + 1e-8))
    bal_pct = _lerp(float(np.std(tilt)), 12.0, 2.0, 0, 100)
    return bal_pct, {"tilt_std_deg": round(float(np.std(tilt)), 1)}


def _count_reps(theta, t):
    """Reps = peaks of the primary angle after light smoothing."""
    if len(theta) < 5:
        return 0, np.array([])
    sig = theta - np.median(theta)
    span = np.percentile(sig, 97) - np.percentile(sig, 3)
    if span < 1e-3:
        return 0, np.array([])
    fs = (len(t) - 1) / max(t[-1] - t[0], 1e-3)
    peaks, props = find_peaks(sig, prominence=0.25 * span, distance=max(int(0.4 * fs), 1))
    return len(peaks), peaks


def m_tempo(theta, t, x, profile):
    reps, peaks = _count_reps(theta, t)
    dur_min = max((t[-1] - t[0]) / 60.0, 1e-3)
    cadence = reps / dur_min
    lo, hi = profile["cadence"]
    if reps == 0:
        cad_pct = 0.0
    elif cadence < lo:
        cad_pct = _lerp(cadence, 0, lo, 30, 100)
    elif cadence > hi:
        cad_pct = _lerp(cadence, hi, 3 * hi, 100, 40)
    else:
        cad_pct = 100.0
    rhythm = _periodicity_score(x)
    pct = float(np.clip(0.6 * cad_pct + 0.4 * rhythm * 100, 0, 100))
    return pct, {"reps": reps, "cadence_per_min": round(cadence, 1), "rhythm": round(rhythm, 2),
                 "_peaks": peaks}


def m_consistency(theta, t):
    reps, peaks = _count_reps(theta, t)
    if len(peaks) < 2:
        return 55.0, {"note": "not enough reps to judge"}
    amps = theta[peaks]
    cv = float(np.std(amps) / (abs(np.mean(amps)) + 1e-6))
    pct = _lerp(cv, 0.5, 0.03, 0, 100)
    return pct, {"rep_cv": round(cv, 2)}


# ---------------------------------------------------------------------------
# calibration for the AI badge
# ---------------------------------------------------------------------------
def _load_calibration(exercise_id):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")
    a, b = 1.0, 0.0
    try:
        with open(path, "r", encoding="utf-8") as f:
            cal = json.load(f)
        pair = cal.get(str(exercise_id)) or cal.get("default") or [1.0, 0.0]
        a, b = float(pair[0]), float(pair[1])
    except Exception:
        pass
    return a, b


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
def compute_report(buf_x, buf_t, exercise_id, ai_score_raw=None, weights=None):
    """Assemble the full session report.

    buf_x : (T,25,3) raw metres per frame ; buf_t : (T,) seconds
    ai_score_raw : the EGRU clinical score in /50 (optional; drives the AI badge)
    Returns a dict consumed by report_card.render_report.
    """
    x = np.asarray(buf_x, dtype=np.float64)
    t = np.asarray(buf_t, dtype=np.float64)
    profile = EXERCISE_PROFILES[exercise_id]

    theta, aux = primary_angle_series(x, profile)
    moving = [WRI_L, WRI_R, SPINE_SHOULDER, ANK_L, ANK_R, KNEE_L, KNEE_R]

    metrics = {}
    rom_pct, rom_d = m_range_of_motion(theta, profile);          metrics["rom"] = (rom_pct, rom_d)
    sm_pct, sm_d = m_smoothness(t, x, moving);                   metrics["smoothness"] = (sm_pct, sm_d)
    sy_pct, sy_d = m_symmetry(theta, aux, profile, x);           metrics["symmetry"] = (sy_pct, sy_d)
    tp_pct, tp_d = m_tempo(theta, t, x, profile);                metrics["tempo"] = (tp_pct, tp_d)
    co_pct, co_d = m_consistency(theta, t);                      metrics["consistency"] = (co_pct, co_d)

    w = weights or {k: 1.0 for k in METRIC_ORDER}
    wsum = sum(w[k] for k in METRIC_ORDER)
    composite = sum(metrics[k][0] * w[k] for k in METRIC_ORDER) / wsum
    composite = float(np.clip(composite, 0, 100))
    stars = int(np.clip(round(composite / 20.0), 1, 5))

    # strengths (>=80) and improvements (<65), up to 2 each, disjoint by threshold
    ranked = sorted(METRIC_ORDER, key=lambda k: metrics[k][0], reverse=True)
    strengths = [PHRASES[k][0] for k in ranked if metrics[k][0] >= 80][:2]
    improvements = [PHRASES[k][1] for k in reversed(ranked) if metrics[k][0] < 65][:2]
    if not strengths:                                    # always give at least one positive
        strengths = [PHRASES[ranked[0]][0].replace("Excellent", "Good").replace("Very", "Fairly")]
    if not improvements:                                 # nothing weak -> a stretch goal
        improvements = ["Great session — keep this form and add a couple more repetitions."]

    ai_badge = None
    if ai_score_raw is not None:
        a, b = _load_calibration(exercise_id)
        ai_badge = float(np.clip(a * float(ai_score_raw) + b, 0, 50))

    band = ("Excellent" if composite >= 85 else "Good" if composite >= 70 else
            "Fair" if composite >= 50 else "Needs work")

    return {
        "exercise_id": exercise_id,
        "exercise_name": EXERCISE_NAMES[exercise_id],
        "composite": round(composite, 1),
        "stars": stars,
        "band": band,
        "ai_badge": None if ai_badge is None else round(ai_badge, 1),
        "ai_raw": None if ai_score_raw is None else round(float(ai_score_raw), 1),
        "reps": tp_d["reps"],
        "duration_s": round(float(t[-1] - t[0]), 1) if len(t) else 0.0,
        "metrics": [
            {"key": k, "label": METRIC_LABELS[k], "pct": round(metrics[k][0], 0),
             "detail": {kk: vv for kk, vv in metrics[k][1].items() if not kk.startswith("_")}}
            for k in METRIC_ORDER
        ],
        "strengths": strengths,
        "improvements": improvements,
    }
