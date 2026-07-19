#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calibrate.py
============
Fit the per-exercise affine that maps the raw EGRU score (/50) to the "AI model score" badge in
the session demo, so it reads sensibly on THIS webcam.

Why: the SE(3)-equivariant model was trained on Kinect depth. On a webcam (via MediaPipe/Camo)
the input distribution shifts, so the raw score reads low (~11-15/50). Calibration is an HONEST
domain adjustment -- a fixed affine  display = clip(a*raw + b, 0, 50)  fitted from a few of YOUR
own reference performances -- not score inflation. The headline Movement Quality score never uses
this; only the small AI badge does.

Workflow (per exercise):
  1. Perform a few GOOD reference sessions   -> label them with the score they *should* read (e.g. 42).
  2. (optional) Perform a few WEAKER sessions -> label them lower (e.g. 22) to anchor the slope.
  3. This tool records each session's raw EGRU score, least-squares fits (a, b), and writes
     demo/calibration.json.

    python demo/calibrate.py --exercise 1 --target 42 --sessions 3           # good refs
    python demo/calibrate.py --exercise 1 --target 22 --sessions 2 --append  # weaker anchors, refit
    python demo/calibrate.py --show                                          # print current calib
    python demo/calibrate.py --reset                                         # restore identity

Recording controls: SPACE start a session, it auto-stops after --seconds; q to abort.
Pairs accumulate in demo/calibration_pairs.json; the fitted affine goes to demo/calibration.json.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mp_to_kinect import mediapipe_to_kinect25, preprocess_window     # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CAL_PATH = os.path.join(HERE, "calibration.json")
PAIRS_PATH = os.path.join(HERE, "calibration_pairs.json")


def _load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def fit_affine(raws, targets):
    """Least-squares (a,b) for target ~= a*raw + b. Falls back to pure offset when the raw values
    have no spread (can't estimate a slope from one cluster)."""
    raws = np.asarray(raws, float)
    targets = np.asarray(targets, float)
    if len(raws) >= 2 and raws.std() > 1e-3 and targets.std() > 1e-3:
        A = np.vstack([raws, np.ones_like(raws)]).T
        a, _ = np.linalg.lstsq(A, targets, rcond=None)[0]
        a = float(np.clip(a, 0.2, 5.0))                 # keep it a sane, monotone map
    else:
        a = 1.0                                         # not enough spread to estimate a slope
    b = float(targets.mean() - a * raws.mean())         # always centre the map on the data means
    return a, b


def refit_and_save(pairs):
    """pairs: {exercise_id(str): [[raw,target],...]}. Writes calibration.json."""
    cal = {"_comment": "Fitted by demo/calibrate.py. display = clip(a*raw+b,0,50).",
           "default": [1.0, 0.0]}
    for k, pr in pairs.items():
        if not pr:
            continue
        raws = [p[0] for p in pr]
        tgts = [p[1] for p in pr]
        cal[str(k)] = list(fit_affine(raws, tgts))
    for e in range(1, 6):                               # ensure all 5 keys exist
        cal.setdefault(str(e), [1.0, 0.0])
    _save_json(CAL_PATH, cal)
    return cal


def record_session_raw(eng, pose, cap, exercise, seconds):
    """Capture one timed session and return the raw EGRU score (/50), or None."""
    import cv2
    sess_x, sess_t, t0 = [], [], time.time()
    started = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        world = pose.world(rgb, int((time.time() - t0) * 1000))
        if started is None:
            cv2.putText(frame, f"CALIBRATE Ex{exercise}: SPACE to record {seconds}s, q abort",
                        (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        else:
            if world is not None:
                sess_t.append(time.time() - t0)
                sess_x.append(mediapipe_to_kinect25(world))
            remain = seconds - (time.time() - started)
            cv2.putText(frame, f"recording... {max(0, remain):4.1f}s", (24, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            if remain <= 0:
                break
        cv2.imshow("calibrate", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            return None
        if started is None and k == ord(' '):
            started = time.time()
    if len(sess_x) < 8:
        return None
    sample = preprocess_window(np.stack(sess_x), np.array(sess_t))
    sample["exercise"] = int(exercise)
    sample.setdefault("y", 0.0)
    return float(eng.predict_egru(sample))


def main():
    ap = argparse.ArgumentParser(description="Calibrate the AI-model-score badge for this webcam.")
    ap.add_argument("--exercise", type=int, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--target", type=float, help="score (/50) these sessions SHOULD read")
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--seconds", type=int, default=15)
    ap.add_argument("--cam", type=int, default=None)
    ap.add_argument("--append", action="store_true", help="keep existing pairs (default: replace this exercise's)")
    ap.add_argument("--show", action="store_true", help="print current calibration and exit")
    ap.add_argument("--reset", action="store_true", help="reset calibration to identity and exit")
    args = ap.parse_args()

    if args.show:
        print(json.dumps(_load_json(CAL_PATH, {}), indent=2))
        return 0
    if args.reset:
        _save_json(CAL_PATH, {"default": [1.0, 0.0], **{str(e): [1.0, 0.0] for e in range(1, 6)}})
        _save_json(PAIRS_PATH, {})
        print("[calibrate] reset to identity.")
        return 0
    if args.exercise is None or args.target is None:
        ap.error("provide --exercise and --target (or use --show/--reset)")

    import cv2
    from engine import DemoEngine
    from pose_backend import PoseBackend

    print("[calibrate] loading model...")
    eng = DemoEngine(load_pct=False)
    pose = PoseBackend()
    cam_idx = args.cam if args.cam is not None else 0
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise SystemExit(f"[calibrate] cannot open camera {cam_idx}. Try --cam N.")

    pairs = _load_json(PAIRS_PATH, {})
    key = str(args.exercise)
    if not args.append:
        pairs[key] = []
    pairs.setdefault(key, [])

    try:
        collected = 0
        while collected < args.sessions:
            print(f"[calibrate] Ex{args.exercise} session {collected+1}/{args.sessions} "
                  f"(target {args.target}/50) -- SPACE to start")
            raw = record_session_raw(eng, pose, cap, args.exercise, args.seconds)
            if raw is None:
                print("[calibrate] aborted / too few frames; skipping.")
                break
            pairs[key].append([raw, float(args.target)])
            collected += 1
            print(f"[calibrate]   raw EGRU = {raw:.2f}/50  -> paired with target {args.target}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

    _save_json(PAIRS_PATH, pairs)
    cal = refit_and_save(pairs)
    print(f"[calibrate] Ex{args.exercise} affine -> {cal[key]}  (display = a*raw + b)")
    print(f"[calibrate] wrote {CAL_PATH}")
    return 0


# ---- headless self-test of the fitting math (no camera) ----
def _selftest():
    a, b = fit_affine([10, 12, 30], [40, 42, 20])       # negatively-correlated toy set
    print(f"[selftest] fit_affine -> a={a:.3f}, b={b:.3f}")
    a2, b2 = fit_affine([11, 11.5, 10.5], [42, 42, 42]) # no spread -> pure offset
    assert abs((a2 * 11 + b2) - 42) < 2.0
    print(f"[selftest] no-spread offset -> a={a2:.3f}, b={b2:.3f} (maps ~11 -> ~42)  OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        sys.exit(main())
