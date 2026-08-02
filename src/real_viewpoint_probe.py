#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
real_viewpoint_probe.py
=======================
Score the seven real webcam takes in `demo/takes/` with the paper's OWN EGRU and PCT fold
ensembles, and report how much each model's score MOVES when the camera physically moves.

Why this exists. The paper's viewpoint experiment (fig:viewpoint) applies a rigid rotation to
ground-truth Kinect skeletons; its caption concedes that this "does not model per-view tracker
degradation". These takes do: one commodity webcam, MediaPipe as the pose estimator, and a camera
that was physically relocated between takes (straight_on / +-30 deg azimuth / tilt up / tilt down /
closer / farther). Running them converts "we rotated the skeletons" into "we moved the camera".

WHAT IS AND IS NOT CLAIMED.
  * n = 1 subject, n = 1 exercise (KIMORE Es1 arm lift). This is a DEPLOYMENT PROBE, not a
    benchmark. It is defensible at n = 1 for the same reason the paper argues for REHAB24-6:
    invariance is a per-prediction structural property, so it survives a single sequence where an
    accuracy claim would not.
  * Webcam + MediaPipe is not Kinect depth. Absolute scores read LOW because of that domain gap.
    We therefore report score SHIFT relative to each model's own `straight_on` score, never an
    absolute level and never an accuracy. The domain gap cancels in a within-model difference.

TWO ARMS, because the takes are not equal length.
  The seven captures were recorded independently and ran for different wall-clock durations
  (14.7 s / 14.8 s / 19.8 s), and `straight_on` additionally dropped frames (11.7 fps vs 30 fps).
  Duration is a real signal to the EGRU (it reads true arrival stamps) and is ERASED for the PCT
  (its architecture resamples onto a fixed 100-frame grid). An unequal-length comparison therefore
  penalises our model and flatters the baseline. We run both arms and report both:
     full    -- every take at its recorded length; duration differences left in.
     matched -- every take cropped to the shortest common duration, so length is held fixed and
                what remains between takes is (mostly) the camera pose.
  The `matched` arm is the one the paper should quote if the two disagree.

FIDELITY TO TRAINING. The raw takes run at ~30 fps for up to 596 frames; `load_sample` caps
training sequences at `max_len=150` control points by uniform index subsample. We apply the SAME
cap here (`_cap_len`, mirroring load_sample) before handing (t, x) to the demo's
`preprocess_window`, which is byte-identical to load_sample's root-relative + median-radius-scale +
t/TIME_SCALE normalisation. Nothing about the preprocessing is reimplemented in this file.

Usage:
    python src/real_viewpoint_probe.py
    python src/real_viewpoint_probe.py --out outputs/real_viewpoint/summary.json
"""

import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
_DEMO = os.path.join(_ROOT, "demo")
for _p in (_SRC, _DEMO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import kimore_cde_data as kd                      # noqa: E402
import mp_to_kinect as m2k                        # noqa: E402
from mp_to_kinect import preprocess_window        # noqa: E402  (the training-matched preprocess)
from engine import DemoEngine                     # noqa: E402  (the paper's checkpoints)
from invariant_controls import invariant_series   # noqa: E402  (rotation-invariant descriptor)

TAKES_DIR = os.path.join(_DEMO, "takes")
CKPT_DIR = os.path.join(_ROOT, "outputs", "cde_block2")
REFERENCE_VIEW = "straight_on"
MAX_LEN = 150                                     # kimore_cde_data.load_sample default
FOLDS = (0, 1, 2, 3, 4)
MODEL_SEED = 0

# Ordered for the figure: azimuth sweep, then tilt, then range.
VIEW_ORDER = ["minus30", "straight_on", "plus30", "tilt_up", "tilt_down", "closer", "farther"]

# Within-viewpoint control. One take per camera pose gives no way to separate a viewpoint effect
# from ordinary capture-to-capture variation, so we manufacture a same-pose null: score K
# equal-length windows at evenly spaced offsets INSIDE each take. The camera pose is fixed across
# those windows, so their spread is a noise floor for that model. A cross-viewpoint shift that does
# not clear its own model's floor is not evidence of anything and is reported as a null.
WINDOW_FRAC = 0.6              # window length, as a fraction of the shortest take
N_WINDOWS = 5

# Phase alignment. The runbook (demo/VARIANT_B1_RUNBOOK.md) replays ONE 15 s clip on a monitor and
# films it from each pose, so the underlying motion is identical across takes and only the playback
# phase differs. Comparing takes at whatever phase each recording happened to start confounds the
# camera move with "a different part of the exercise", which is what dominates the unaligned control
# below. We recover the per-take lag by cross-correlating a ROTATION-INVARIANT descriptor of the
# motion (invariant_controls.invariant_series -- norms and distances only), which is the one signal
# that cannot itself encode the camera pose we are trying to measure. Aligning on a
# viewpoint-dependent signal would beg the question; aligning on an invariant one cannot.
ALIGN_HZ = 30.0                # uniform grid for the cross-correlation
ALIGN_MIN_OVERLAP_S = 5.0      # shortest overlap a candidate lag may be scored on
ALIGN_MIN_R = 0.5              # below this the alignment is declared failed, not quietly used
N_PHASES = 8                   # paired phase offsets in the aligned arm

# The checkpoint families the paper trains and reports. If the demo engine ever silently loads
# something else, the entire claim collapses -- so we pin the names, not just the directory.
CKPT_PATTERNS = {
    "egru": re.compile(r"^egru_s\d+_pooled_f\d+\.pt$"),
    "pct": re.compile(r"^pct_pooled_f\d+\.pt$"),
}


# =============================================================================
# Provenance
# =============================================================================
def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def checkpoint_manifest():
    """The exact files DemoEngine(ckpt_dir=CKPT_DIR, model_seed=0, folds=0..4) will load.

    Asserts each one sits under outputs/cde_block2/ and matches its reported family, then hashes
    it, so every number in the artifact is traceable to a specific set of weights.
    """
    manifest = {}
    for model, names in (("egru", [f"egru_s{MODEL_SEED}_pooled_f{f}.pt" for f in FOLDS]),
                         ("pct", [f"pct_pooled_f{f}.pt" for f in FOLDS])):
        entries = []
        for name in names:
            path = os.path.join(CKPT_DIR, name)
            assert CKPT_PATTERNS[model].match(name), f"{name}: not a reported {model} checkpoint"
            assert os.path.isfile(path), f"missing checkpoint {path}"
            rel = os.path.relpath(path, _ROOT).replace("\\", "/")
            assert rel.startswith("outputs/cde_block2/"), f"{rel}: outside the paper's ckpt dir"
            entries.append({"path": rel,
                            "bytes": os.path.getsize(path),
                            "sha256": _sha256(path)})
        manifest[model] = entries
    return manifest


# =============================================================================
# Takes -> model samples
# =============================================================================
def load_take(path):
    """Read one recorded take -> (t (T,) seconds, x (T,25,3) raw Kinect-mapped metres, metadata)."""
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["_metadata"]))
    return (np.asarray(d["t"], dtype=np.float64),
            np.asarray(d["x"], dtype=np.float64),
            meta)


def _monotonic(t, x):
    """load_sample's strict-monotonicity filter (a webcam can repeat a tick under lag)."""
    keep = np.concatenate([[True], np.diff(t) > 0])
    return t[keep], x[keep]


def _cap_len(t, x, max_len=MAX_LEN):
    """load_sample's uniform INDEX subsample, keeping the ACTUAL stamps of retained frames.

    Mirrors kimore_cde_data.load_sample so the control-point count the models see here matches
    what they were trained on. Real inter-arrival gaps are preserved, not resampled away.
    """
    if len(t) <= max_len:
        return t, x
    idx = np.unique(np.linspace(0, len(t) - 1, max_len).round().astype(int))
    return t[idx], x[idx]


def build_sample(t, x, exercise, crop_s=None, start_s=0.0, max_len=MAX_LEN):
    """Raw take -> the dict DemoEngine.predict consumes, via the demo's own preprocess_window."""
    t, x = _monotonic(t, x)
    t = t - t[0]
    if crop_s is not None:
        keep = (t >= start_s - 1e-9) & (t <= start_s + crop_s + 1e-9)
        t, x = t[keep], x[keep]
    t, x = _cap_len(t, x, max_len)
    s = preprocess_window(x, t)
    s["exercise"] = int(exercise)
    s["y"] = 0.0                                   # collate needs a label; unused at inference
    return s


# =============================================================================
# Phase alignment (invariant descriptor cross-correlation)
# =============================================================================
def invariant_descriptor(t, x, hz=ALIGN_HZ):
    """Take -> (t_grid, Z (N,F)) rotation-invariant, z-scored motion descriptor on a uniform grid.

    Uses the project's own `invariant_series` (inner products of root-relative joint coordinates:
    norms and distances). Every feature is invariant to the camera pose, so the lag it recovers is
    a property of the MOTION, not of the viewpoint under test.
    """
    t, x = _monotonic(t, x)
    t = t - t[0]
    xr = x - x[:, 0:1, :]                                        # root-relative
    radius = np.linalg.norm(xr, axis=-1)
    scale = float(np.median(radius[radius > 0])) if np.any(radius > 0) else 1.0
    xr = xr / max(scale, 1e-6)

    F = invariant_series({"x": xr, "t": t / kd.TIME_SCALE})      # (L, F), invariant
    grid = np.arange(0.0, t[-1], 1.0 / hz)
    G = np.stack([np.interp(grid, t, F[:, k]) for k in range(F.shape[1])], axis=1)
    G = G - G.mean(axis=0, keepdims=True)
    G = G / (G.std(axis=0, keepdims=True) + 1e-9)
    return grid, G


def best_lag(ref_G, G, hz=ALIGN_HZ, min_overlap_s=ALIGN_MIN_OVERLAP_S):
    """Integer-sample lag maximising the correlation of `G` against `ref_G`, plus that correlation.

    Positive lag means this take's motion runs AHEAD of the reference: reference time u lines up
    with this take's time u + lag/hz. Correlation is computed on the overlap only and renormalised
    per lag, so long overlaps are not mechanically favoured over short ones.
    """
    n_ref, n = len(ref_G), len(G)
    min_ov = int(min_overlap_s * hz)
    best = (-np.inf, 0)
    for lag in range(-(n_ref - min_ov), n - min_ov + 1):
        a0, b0 = max(0, lag), max(0, -lag)
        m = min(n_ref - b0, n - a0)
        if m < min_ov:
            continue
        A, B = ref_G[b0:b0 + m], G[a0:a0 + m]
        A = A - A.mean(axis=0, keepdims=True)
        B = B - B.mean(axis=0, keepdims=True)
        denom = np.sqrt((A ** 2).sum(axis=0) * (B ** 2).sum(axis=0)) + 1e-12
        r = float(np.mean((A * B).sum(axis=0) / denom))           # mean per-feature Pearson r
        if r > best[0]:
            best = (r, lag)
    return best[1] / hz, best[0]


def align_takes(takes, reference=REFERENCE_VIEW):
    """-> {view: {'lag_s': float, 'r': float}} mapping reference time to each take's own clock."""
    desc = {v: invariant_descriptor(t, x) for v, (t, x, _) in takes.items()}
    _, ref_G = desc[reference]
    out = {}
    for v, (_, G) in desc.items():
        if v == reference:
            out[v] = {"lag_s": 0.0, "r": 1.0}
            continue
        lag, r = best_lag(ref_G, G)
        out[v] = {"lag_s": float(lag), "r": float(r)}
    return out


def run_aligned(engine, takes, align, window_s, n_phases=N_PHASES, lag_delta=0.0,
                max_len=MAX_LEN):
    """Paired comparison: every viewpoint scored on the SAME motion segment, at n_phases phases.

    With phase held fixed, the only thing separating two takes is the camera pose plus whatever the
    pose estimator did differently at that pose -- which is exactly the quantity of interest. The
    pairing also gives a real uncertainty: n_phases paired differences per viewpoint, so we can say
    whether a shift is distinguishable from zero instead of quoting a single number.
    """
    lag = {v: align[v]["lag_s"] + (0.0 if v == REFERENCE_VIEW else lag_delta) for v in takes}
    durs = {v: float(m["duration_s"]) for v, (_, _, m) in takes.items()}
    lo = max(-lag[v] for v in takes)
    hi = min(durs[v] - window_s - lag[v] for v in takes)
    if hi <= lo:
        return {"feasible": False, "reason": f"no common aligned window (lo={lo:.2f}, hi={hi:.2f})"}
    phases = np.linspace(lo, hi, n_phases)

    scores = {v: {"egru": [], "pct": []} for v in takes}
    for v, (t, x, meta) in takes.items():
        for u in phases:
            s = build_sample(t, x, meta["exercise"], crop_s=window_s,
                             start_s=float(u + lag[v]), max_len=max_len)
            out = engine.predict(s, exercise=meta["exercise"], angle=0.0)
            scores[v]["egru"].append(float(out["egru"]))
            scores[v]["pct"].append(float(out["pct"]))

    stats = {}
    for model in ("egru", "pct"):
        ref = np.array(scores[REFERENCE_VIEW][model])
        per_view = {}
        for v in takes:
            d = np.array(scores[v][model]) - ref                  # paired by motion phase
            sem = float(np.std(d, ddof=1) / np.sqrt(len(d)))
            per_view[v] = {
                "mean_shift": float(np.mean(d)),
                "sd_shift": float(np.std(d, ddof=1)),
                "sem_shift": sem,
                "abs_t": float(abs(np.mean(d)) / sem) if sem > 0 else float("inf"),
            }
        moved = [abs(per_view[v]["mean_shift"]) for v in per_view if v != REFERENCE_VIEW]
        # Paired-difference scatter at fixed phase: the residual once phase is controlled for.
        floor = float(np.sqrt(np.mean([per_view[v]["sd_shift"] ** 2
                                       for v in per_view if v != REFERENCE_VIEW])))
        stats[model] = {
            "reference_mean": float(np.mean(ref)),
            "per_view": per_view,
            "mean_abs_shift": float(np.mean(moved)),
            "max_abs_shift": float(np.max(moved)),
            "max_abs_shift_view": max((v for v in per_view if v != REFERENCE_VIEW),
                                      key=lambda v: abs(per_view[v]["mean_shift"])),
            "paired_sd": floor,
            "n_views_significant": int(sum(per_view[v]["abs_t"] > 2.0
                                           for v in per_view if v != REFERENCE_VIEW)),
        }
    return {"feasible": True, "window_s": float(window_s), "phases_s": [float(u) for u in phases],
            "lag_delta_s": float(lag_delta), "max_len": int(max_len),
            "alignment": align, "scores": scores, "stats": stats}


def run_robustness(engine, takes, align, window_s, ref_control_points):
    """Does the aligned headline survive the two things it could be an artefact of?

    (1) LAG. The recovered alignment is only as good as the cross-correlation; we re-run the whole
        paired arm with every non-reference lag shifted by +-0.5 s.
    (2) CONTROL-POINT COUNT. The straight_on reference was captured at 11.7 fps with real dropped
        frames while the other six ran at 30 fps, so after the 150-point cap the reference carries
        fewer control points than the takes it is compared against. We re-run with every take capped
        to the reference's own count, which removes that asymmetry. It should if anything help the
        PCT, which resamples onto a fixed grid and is indifferent to the count.

    A conclusion that flips under either of these is not a result.
    """
    out = {}
    for tag, kw in (("lag_minus_0.5s", dict(lag_delta=-0.5)),
                    ("lag_plus_0.5s", dict(lag_delta=+0.5)),
                    ("rate_matched", dict(max_len=ref_control_points))):
        r = run_aligned(engine, takes, align, window_s, **kw)
        if not r.get("feasible"):
            out[tag] = {"feasible": False, "reason": r.get("reason")}
            continue
        out[tag] = {"feasible": True,
                    **{m: {"mean_abs_shift": r["stats"][m]["mean_abs_shift"],
                          "max_abs_shift": r["stats"][m]["max_abs_shift"]}
                       for m in ("egru", "pct")}}
        out[tag]["pct_over_egru"] = (out[tag]["pct"]["mean_abs_shift"]
                                     / max(out[tag]["egru"]["mean_abs_shift"], 1e-9))
    return out


# =============================================================================
# The probe
# =============================================================================
def run_arm(engine, takes, crop_s=None):
    """Score every take under one arm -> {viewpoint: {model: score, ...}}."""
    rows = {}
    for view, (t, x, meta) in takes.items():
        s = build_sample(t, x, meta["exercise"], crop_s=crop_s)
        out = engine.predict(s, exercise=meta["exercise"], angle=0.0)
        rows[view] = {
            "egru": float(out["egru"]),
            "pct": float(out["pct"]),
            "n_control_points": int(s["n_frames"]),
            "duration_s": float(s["duration_s"]),
        }
    return rows


def summarise(rows):
    """Per-model shift from that model's own straight_on score, plus mean/max |shift|.

    Reported both in clinical score units (the paper's degradation convention) and as a percentage
    of that model's own straight_on score. The percentage matters because the two models sit at
    different levels under this domain shift, so an absolute-shift comparison on its own would
    partly reward whichever model scores lower. Both are in the artifact; neither is hidden.
    """
    assert REFERENCE_VIEW in rows, f"no {REFERENCE_VIEW} take to reference against"
    stats = {}
    for model in ("egru", "pct"):
        ref = rows[REFERENCE_VIEW][model]
        shifts = {v: rows[v][model] - ref for v in rows}
        moved = [abs(d) for v, d in shifts.items() if v != REFERENCE_VIEW]
        stats[model] = {
            "reference_score": float(ref),
            "shift": {v: float(d) for v, d in shifts.items()},
            "mean_abs_shift": float(np.mean(moved)),
            "max_abs_shift": float(np.max(moved)),
            "max_abs_shift_view": max((v for v in shifts if v != REFERENCE_VIEW),
                                      key=lambda v: abs(shifts[v])),
            "mean_abs_shift_pct_of_ref": float(100.0 * np.mean(moved) / abs(ref)),
            "max_abs_shift_pct_of_ref": float(100.0 * np.max(moved) / abs(ref)),
        }
    return stats


def run_window_control(engine, takes, window_s, n_windows=N_WINDOWS):
    """Same camera pose, different time window -> the noise floor each shift must clear.

    For every take we score `n_windows` windows of identical length at evenly spaced offsets. The
    camera did not move between them, so any score spread is capture/segment noise rather than a
    viewpoint effect. Cross-viewpoint means are computed from the SAME windows, so the effect and
    the floor are measured on identical footing.
    """
    per_view = {}
    for view, (t, x, meta) in takes.items():
        span = float(meta["duration_s"]) - window_s
        offs = np.linspace(0.0, max(span, 0.0), n_windows) if span > 0 else np.zeros(n_windows)
        scores = {"egru": [], "pct": []}
        for o in offs:
            s = build_sample(t, x, meta["exercise"], crop_s=window_s, start_s=float(o))
            out = engine.predict(s, exercise=meta["exercise"], angle=0.0)
            scores["egru"].append(float(out["egru"]))
            scores["pct"].append(float(out["pct"]))
        per_view[view] = {
            "offsets_s": [float(o) for o in offs],
            "scores": scores,
            "mean": {m: float(np.mean(v)) for m, v in scores.items()},
            "sd": {m: float(np.std(v, ddof=1)) for m, v in scores.items()},
            "range": {m: float(np.max(v) - np.min(v)) for m, v in scores.items()},
        }

    stats = {}
    for model in ("egru", "pct"):
        # Noise floor: RMS of the within-take SDs, pooled over all seven camera poses.
        floor = float(np.sqrt(np.mean([per_view[v]["sd"][model] ** 2 for v in per_view])))
        ref = per_view[REFERENCE_VIEW]["mean"][model]
        shifts = {v: per_view[v]["mean"][model] - ref for v in per_view}
        moved = [abs(d) for v, d in shifts.items() if v != REFERENCE_VIEW]
        stats[model] = {
            "reference_score": float(ref),
            "shift": {v: float(d) for v, d in shifts.items()},
            "mean_abs_shift": float(np.mean(moved)),
            "max_abs_shift": float(np.max(moved)),
            "within_view_sd": floor,
            "max_within_view_sd": float(max(per_view[v]["sd"][model] for v in per_view)),
            "effect_over_floor": float(np.mean(moved) / floor) if floor > 0 else float("inf"),
            "clears_floor": bool(np.max(moved) > 2.0 * floor),
        }
    return {"window_s": float(window_s), "n_windows": int(n_windows),
            "per_view": per_view, "stats": stats}


def main():
    ap = argparse.ArgumentParser(description="Real-camera viewpoint probe on demo/takes/.")
    ap.add_argument("--takes-dir", default=TAKES_DIR)
    ap.add_argument("--out", default=os.path.join(_ROOT, "outputs", "real_viewpoint", "summary.json"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # Verification gate 3: the preprocessing this probe uses must BE the training preprocessing.
    assert m2k.TIME_SCALE == kd.TIME_SCALE, (
        f"TIME_SCALE mismatch: demo {m2k.TIME_SCALE} vs training {kd.TIME_SCALE}")
    assert preprocess_window.__module__ == "mp_to_kinect", "preprocess_window was shadowed"

    # --- load takes ---
    takes, take_meta = {}, {}
    for name in sorted(os.listdir(args.takes_dir)):
        if not name.endswith(".npz"):
            continue
        t, x, meta = load_take(os.path.join(args.takes_dir, name))
        assert not meta.get("synthetic", False), f"{name}: synthetic take, not a real capture"
        takes[meta["viewpoint"]] = (t, x, meta)
        take_meta[meta["viewpoint"]] = meta
    assert takes, f"no takes found in {args.takes_dir}"
    clips = {m["clip_id"] for m in take_meta.values()}
    assert len(clips) == 1, f"takes span multiple clips {clips}; the probe assumes one"
    views = [v for v in VIEW_ORDER if v in takes] + [v for v in takes if v not in VIEW_ORDER]
    print(f"[probe] {len(takes)} real takes of {clips.pop()}: {', '.join(views)}")

    # --- provenance, before any inference ---
    manifest = checkpoint_manifest()
    print(f"[probe] checkpoints verified: {len(manifest['egru'])} EGRU + {len(manifest['pct'])} PCT "
          f"under outputs/cde_block2/")

    engine = DemoEngine(ckpt_dir=CKPT_DIR, device=args.device, model_seed=MODEL_SEED,
                        folds=FOLDS, load_pct=True)
    print(f"[probe] engine on {engine.device}")

    # --- the two arms ---
    crop_s = min(float(m["duration_s"]) for m in take_meta.values())
    arms = {}
    for arm, crop in (("full", None), ("matched", crop_s)):
        rows = run_arm(engine, takes, crop_s=crop)
        arms[arm] = {"crop_s": crop, "per_view": rows, "stats": summarise(rows)}
        e, p = arms[arm]["stats"]["egru"], arms[arm]["stats"]["pct"]
        print(f"\n[{arm}]  crop={'none' if crop is None else f'{crop:.2f}s'}")
        print(f"  {'viewpoint':<14}{'EGRU':>9}{'shift':>9}{'PCT':>10}{'shift':>9}   L")
        for v in views:
            r = rows[v]
            print(f"  {v:<14}{r['egru']:>9.3f}{e['shift'][v]:>+9.3f}"
                  f"{r['pct']:>10.3f}{p['shift'][v]:>+9.3f}   {r['n_control_points']}")
        print(f"  mean|shift|    EGRU {e['mean_abs_shift']:.3f}   PCT {p['mean_abs_shift']:.3f}")
        print(f"  max |shift|    EGRU {e['max_abs_shift']:.3f} ({e['max_abs_shift_view']})   "
              f"PCT {p['max_abs_shift']:.3f} ({p['max_abs_shift_view']})")
        print(f"  as % of own straight_on   EGRU {e['mean_abs_shift_pct_of_ref']:.1f}%   "
              f"PCT {p['mean_abs_shift_pct_of_ref']:.1f}%")

    # --- within-viewpoint control: the floor every shift above has to clear ---
    window_s = WINDOW_FRAC * crop_s
    ctrl = run_window_control(engine, takes, window_s)
    ce, cp = ctrl["stats"]["egru"], ctrl["stats"]["pct"]
    print(f"\n[control]  same pose, {N_WINDOWS} windows of {window_s:.2f}s per take")
    print(f"  {'viewpoint':<14}{'EGRU mean':>11}{'sd':>8}{'PCT mean':>11}{'sd':>8}")
    for v in views:
        r = ctrl["per_view"][v]
        print(f"  {v:<14}{r['mean']['egru']:>11.3f}{r['sd']['egru']:>8.3f}"
              f"{r['mean']['pct']:>11.3f}{r['sd']['pct']:>8.3f}")
    for tag, st in (("EGRU", ce), ("PCT", cp)):
        print(f"  {tag}: mean|shift| {st['mean_abs_shift']:.3f}  max {st['max_abs_shift']:.3f}  "
              f"within-view sd {st['within_view_sd']:.3f}  "
              f"effect/floor {st['effect_over_floor']:.2f}  "
              f"clears 2sd: {st['clears_floor']}")

    # --- aligned, paired arm: same motion segment seen from every pose ---
    align = align_takes(takes)
    worst = min(align[v]["r"] for v in align)
    print(f"\n[align]  invariant-descriptor lags vs {REFERENCE_VIEW} (worst r = {worst:.3f})")
    for v in views:
        print(f"  {v:<14}lag {align[v]['lag_s']:>+7.2f}s   r {align[v]['r']:.3f}")
    if worst < ALIGN_MIN_R:
        print(f"  [align] FAILED: worst r {worst:.3f} < {ALIGN_MIN_R}; aligned arm not reported.")
        aligned = {"feasible": False, "reason": f"alignment quality r={worst:.3f} below "
                                                f"{ALIGN_MIN_R}", "alignment": align}
    else:
        aligned = run_aligned(engine, takes, align, window_s)

    if aligned.get("feasible"):
        print(f"\n[aligned]  {N_PHASES} paired phases, {window_s:.2f}s window")
        ae, ap_ = aligned["stats"]["egru"], aligned["stats"]["pct"]
        print(f"  {'viewpoint':<14}{'EGRU d':>9}{'+-sem':>8}{'|t|':>7}"
              f"{'PCT d':>10}{'+-sem':>8}{'|t|':>7}")
        for v in views:
            if v == REFERENCE_VIEW:
                continue
            a, b = ae["per_view"][v], ap_["per_view"][v]
            print(f"  {v:<14}{a['mean_shift']:>+9.3f}{a['sem_shift']:>8.3f}{a['abs_t']:>7.1f}"
                  f"{b['mean_shift']:>+10.3f}{b['sem_shift']:>8.3f}{b['abs_t']:>7.1f}")
        for tag, st in (("EGRU", ae), ("PCT", ap_)):
            print(f"  {tag}: mean|shift| {st['mean_abs_shift']:.3f}  "
                  f"max {st['max_abs_shift']:.3f} ({st['max_abs_shift_view']})  "
                  f"paired sd {st['paired_sd']:.3f}  "
                  f"views significant {st['n_views_significant']}/{len(views) - 1}")
    else:
        print(f"\n[aligned]  not reported: {aligned.get('reason')}")

    # --- robustness: does the aligned headline survive lag error and the fps asymmetry? ---
    robust = {}
    if aligned.get("feasible"):
        rt, rx, rmeta = takes[REFERENCE_VIEW]
        ref_cp = int(build_sample(rt, rx, rmeta["exercise"],
                                  crop_s=window_s, start_s=0.0)["n_frames"])
        robust = run_robustness(engine, takes, align, window_s, ref_cp)
        base = aligned["stats"]["pct"]["mean_abs_shift"] / aligned["stats"]["egru"]["mean_abs_shift"]
        print(f"\n[robustness]  PCT/EGRU mean|shift| ratio, aligned baseline {base:.2f}x "
              f"(reference has {ref_cp} control points)")
        for tag, r in robust.items():
            if not r.get("feasible"):
                print(f"  {tag:<16}not feasible: {r.get('reason')}")
                continue
            print(f"  {tag:<16}EGRU {r['egru']['mean_abs_shift']:.3f}   "
                  f"PCT {r['pct']['mean_abs_shift']:.3f}   ratio {r['pct_over_egru']:.2f}x")

    payload = {
        "what": "Real-camera viewpoint probe: same exercise, seven physical camera poses, "
                "scored with the paper's own EGRU and PCT fold ensembles.",
        "scope": {
            "n_subjects": 1,
            "n_exercises": 1,
            "sensor": "commodity webcam + MediaPipe Pose (not Kinect depth)",
            "reports": "score SHIFT relative to each model's own straight_on score, in clinical "
                       "units (0-50); NOT absolute accuracy",
            "reference_view": REFERENCE_VIEW,
            "control": "one take per camera pose, so cross-viewpoint shifts are compared against a "
                       "same-pose noise floor measured over time windows within each take",
        },
        "config": {
            "ckpt_dir": "outputs/cde_block2",
            "model_seed": MODEL_SEED,
            "folds": list(FOLDS),
            "max_len": MAX_LEN,
            "time_scale": kd.TIME_SCALE,
            "score_max": kd.SCORE_MAX,
            "pct_n_frames": engine.n_frames,
            "pct_operator": engine.pct_operator,
            "device": str(engine.device),
            "preprocess": "demo/mp_to_kinect.py::preprocess_window (byte-identical to "
                          "src/kimore_cde_data.py::load_sample)",
        },
        "checkpoints": manifest,
        "takes": take_meta,
        "view_order": views,
        "arms": arms,
        "within_view_control": ctrl,
        "aligned": aligned,
        "robustness": robust,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[probe] wrote {os.path.relpath(args.out, _ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
