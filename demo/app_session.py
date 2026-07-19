#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app_session.py
==============
The SINGLE-MODEL session demo for the Innovation Challenge (no comparison).

Story for a non-technical judge:
    1. Pick an exercise -> its NAME and a one-line "how to" show on screen.
    2. Press SPACE -> a 3-2-1 countdown, then a short timed session.
    3. Perform the exercise -> a live Movement Quality score updates.
    4. Session ends -> a clean report card appears (and is saved as a PNG):
       overall score, star rating, a small calibrated AI-model badge, a breakdown of
       range of motion / smoothness / symmetry / tempo / consistency, plus plain-language
       STRENGTHS and WHAT TO IMPROVE.

The headline Movement Quality score is computed from real, explainable biomechanics
(demo/feedback.py). The SE(3)-equivariant model rides along as the smaller calibrated
"AI model score" badge. Works with a normal webcam or a phone via Camo (auto-detected).

Controls
    1..5     select exercise (in SELECT / between sessions)
    SPACE    start the session (from SELECT) ; also dismiss the report
    e        end the current session early
    q / ESC  quit

Run:  python demo/app_session.py                 (auto-detect camera)
      python demo/app_session.py --cam 1         (force camera index)
      python demo/app_session.py --seconds 25    (session length)
      python demo/app_session.py --list          (list cameras and exit)
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mp_to_kinect import mediapipe_to_kinect25, preprocess_window     # noqa: E402
import feedback as fb                                                 # noqa: E402
import report_card as rc                                             # noqa: E402

# Kinect draw bones (same subset as app_camo.py)
DRAW_BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7),
    (20, 8), (8, 9), (9, 10), (10, 11),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]


def list_cameras():
    print("\n[cam] available cameras:")
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, fr = cap.read()
            if ok:
                print(f"  camera {i}: {fr.shape[1]}x{fr.shape[0]}")
            cap.release()
    print()


def find_camera():
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            cap.release()
            if ok:
                print(f"[cam] auto-detected camera {i}")
                return i
    return 0


def draw_skeleton(frame, kin_raw, color=(0, 230, 255)):
    h, w = frame.shape[:2]
    box = min(h, w) // 3
    ox, oy = w - box - 20, 20
    pts = kin_raw[:, :2].copy()
    pts[:, 1] = -pts[:, 1]
    lo, hi = pts.min(0), pts.max(0)
    span = np.maximum(hi - lo, 1e-3)
    px = ((pts - lo) / span * (box - 20) + [ox + 10, oy + 10]).astype(int)
    for a, b in DRAW_BONES:
        cv2.line(frame, tuple(px[a]), tuple(px[b]), color, 2)
    for p in px:
        cv2.circle(frame, tuple(p), 3, (255, 255, 255), -1)


def big_score_bar(frame, score):
    """Live Movement Quality bar shown during a session."""
    x, y, w = 30, 80, 360
    col = (0, 210, 90) if score >= 70 else (0, 190, 240) if score >= 50 else (60, 90, 231)
    cv2.putText(frame, "Movement Quality (live)", (x, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.rectangle(frame, (x, y), (x + w, y + 34), (60, 60, 60), -1)
    fill = int(np.clip(score / 100.0, 0, 1) * w)
    cv2.rectangle(frame, (x, y), (x + fill, y + 34), col, -1)
    cv2.putText(frame, f"{score:4.0f}/100", (x + w + 16, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def banner(frame, lines, sub=None):
    """Centered translucent banner used for SELECT / COUNTDOWN prompts."""
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, h // 2 - 120), (w, h // 2 + 120), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    y = h // 2 - 60
    for ln, scale, col in lines:
        sz = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)[0]
        cv2.putText(frame, ln, ((w - sz[0]) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 3)
        y += int(scale * 55) + 14
    if sub:
        sz = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(frame, sub, ((w - sz[0]) // 2, h // 2 + 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 2)


def main():
    ap = argparse.ArgumentParser(description="Single-model rehab session demo with report card.")
    ap.add_argument("--cam", type=int, default=None, help="camera index (default: auto-detect)")
    ap.add_argument("--list", action="store_true", help="list cameras and exit")
    ap.add_argument("--seconds", type=int, default=20, help="session length in seconds")
    ap.add_argument("--infer-every", type=int, default=6, help="score every N frames (live)")
    ap.add_argument("--min-frames", type=int, default=20, help="min frames before a live score")
    args = ap.parse_args()

    if args.list:
        list_cameras()
        return 0

    from engine import DemoEngine
    from pose_backend import PoseBackend

    print("[session] loading model (EGRU ensemble; PCT skipped)...")
    eng = DemoEngine(load_pct=False)                     # single model -> faster startup
    pose = PoseBackend()
    cam_idx = args.cam if args.cam is not None else find_camera()
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise SystemExit(f"[session] cannot open camera {cam_idx}. Try --cam N or --list.")
    print(f"[session] ready on {eng.device}. Pick an exercise (1-5) and press SPACE.")

    WIN = "Rehab Assessment  --  Innovation Challenge Demo"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    state = "SELECT"            # SELECT -> COUNTDOWN -> RECORDING -> REPORT
    exercise = 1
    t0 = time.time()            # pose clock (monotonic ms source)
    sess_x, sess_t = [], []     # full-session buffers (raw metres, seconds)
    sess_start = 0.0
    countdown_start = 0.0
    live_score = 0.0
    fno = 0
    report_img = None

    def reset_session():
        nonlocal sess_x, sess_t, live_score, fno
        sess_x, sess_t, live_score, fno = [], [], 0.0, 0

    def finish_session():
        nonlocal state, report_img
        if len(sess_x) >= 8:
            x = np.stack(sess_x)
            t = np.array(sess_t)
            sample = preprocess_window(x, t)
            sample["exercise"] = int(exercise)
            sample.setdefault("y", 0.0)
            ai_raw = eng.predict_egru(sample)                    # EGRU /50, no PCT
            rep = fb.compute_report(x, t, exercise, ai_score_raw=ai_raw)
            report_img, png = rc.render_report(rep, save=True)
            print(f"[session] report saved: {png}  (quality {rep['composite']:.0f}/100, "
                  f"{rep['reps']} reps, AI {rep['ai_badge']:.0f}/50)")
        else:
            report_img = None
            print("[session] too few frames captured -- no report.")
        state = "REPORT"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            world = pose.world(rgb, int((time.time() - t0) * 1000))
            kin_raw = mediapipe_to_kinect25(world) if world is not None else None

            # ---- REPORT: show the card full-window ----
            if state == "REPORT":
                if report_img is not None:
                    disp = cv2.resize(report_img, (frame.shape[1], frame.shape[0]))
                    cv2.putText(disp, "press SPACE / any key for a new session, q to quit",
                                (24, disp.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (150, 158, 170), 1)
                    cv2.imshow(WIN, disp)
                else:
                    banner(frame, [("No movement captured", 1.0, (80, 120, 255))],
                           "press SPACE to try again")
                    cv2.imshow(WIN, frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('q'), 27):
                    break
                if k != 255:
                    state = "SELECT"
                continue

            # ---- draw the tracked skeleton for all live states ----
            if kin_raw is not None:
                draw_skeleton(frame, kin_raw)

            if state == "SELECT":
                banner(frame,
                       [(f"Exercise {exercise}", 0.9, (0, 220, 255)),
                        (fb.EXERCISE_NAMES[exercise], 1.2, (255, 255, 255))],
                       fb.HOWTO[exercise])
                cv2.putText(frame, "keys: 1-5 choose exercise   SPACE start   q quit",
                            (24, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 200, 200), 2)

            elif state == "COUNTDOWN":
                remain = 3 - (time.time() - countdown_start)
                if remain <= 0:
                    state = "RECORDING"
                    reset_session()
                    sess_start = time.time()
                else:
                    banner(frame, [(fb.EXERCISE_NAMES[exercise], 0.9, (0, 220, 255)),
                                   (f"{int(np.ceil(remain))}", 3.0, (255, 255, 255))],
                           "get into position")

            elif state == "RECORDING":
                if kin_raw is not None:
                    sess_t.append(time.time() - t0)
                    sess_x.append(kin_raw)
                # live score every N frames
                fno += 1
                if len(sess_x) >= args.min_frames and fno % args.infer_every == 0:
                    try:
                        rep = fb.compute_report(np.stack(sess_x), np.array(sess_t), exercise)
                        live_score = rep["composite"]
                    except Exception:
                        pass
                remain = args.seconds - (time.time() - sess_start)
                big_score_bar(frame, live_score)
                cv2.putText(frame, fb.EXERCISE_NAMES[exercise], (30, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
                # timer ring text
                cv2.putText(frame, f"{max(0, remain):4.1f}s", (frame.shape[1] - 150, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 3)
                cv2.putText(frame, "perform the exercise   (e = end early)",
                            (30, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 200, 200), 2)
                if remain <= 0:
                    finish_session()

            cv2.imshow(WIN, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif state == "SELECT" and ord('1') <= k <= ord('5'):
                exercise = k - ord('0')
            elif state == "SELECT" and k == ord(' '):
                state, countdown_start = "COUNTDOWN", time.time()
            elif state == "RECORDING" and k == ord('e'):
                finish_session()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
