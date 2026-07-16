#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.py
======
The LIVE webcam demo (OpenCV backend -- the robust booth fallback; the polished web dashboard is
Task 6). Point a laptop webcam at a person doing a rehab exercise and watch two scores update in
real time:

    EGRU (ours)   -- flat under viewpoint rotation, BY THEOREM.
    PCT (target)  -- collapses as you rotate the viewpoint.

ACT CONTROLS (keyboard)
    q / ESC   quit
    1..5      select which KIMORE exercise the models score
    [ / ]     rotate the viewpoint -/+ 15 deg   (Act 2: the SAME skeleton is fed to both models)
    r         auto-sweep rotation 0 -> 90 -> 0 (hands-free Act 2)
    0         reset rotation to 0
    d         toggle Gilbert-Elliott frame drops (Act 3 flourish)
    SPACE     freeze/unfreeze the buffer (hold a pose for the judges)

WHAT TO VERIFY (needs your camera; I cannot run a webcam here):
    * the skeleton overlay tracks you and stands UPRIGHT (if upside-down, flip AXIS_SIGN[1] in
      mp_to_kinect.py -- but the smoke test already fixed Y-up, so it should be right);
    * as you press ] to rotate, the EGRU bar does NOT move and the PCT bar DOES. That contrast is
      the whole demo. It is guaranteed by construction, so if EGRU moves, something in the buffer
      wiring is wrong, not the model.

Run:  python demo/app.py            (default camera 0)
      python demo/app.py --cam 1    (external webcam)
"""

import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mp_to_kinect import mediapipe_to_kinect25, preprocess_window, KINECT_NAMES   # noqa: E402

# Kinect bone list for drawing (subset of KINECT_BONES, indices into the 25-joint array)
DRAW_BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7),
    (20, 8), (8, 9), (9, 10), (10, 11),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]


def draw_skeleton(frame, kin_raw, color=(0, 230, 255)):
    """Project the (unnormalized) Kinect-25 skeleton onto the frame for a visual overlay.

    A quick orthographic projection (x right, y up) scaled into a corner box -- purely cosmetic,
    so the judge sees the tracked body. The MODEL sees the preprocessed skeleton, not this.
    """
    h, w = frame.shape[:2]
    box = min(h, w) // 3
    ox, oy = w - box - 20, 20
    pts = kin_raw[:, :2].copy()
    pts[:, 1] = -pts[:, 1]                                # y up on screen
    lo, hi = pts.min(0), pts.max(0)
    span = np.maximum(hi - lo, 1e-3)
    px = ((pts - lo) / span * (box - 20) + [ox + 10, oy + 10]).astype(int)
    for a, b in DRAW_BONES:
        cv2.line(frame, tuple(px[a]), tuple(px[b]), color, 2)
    for p in px:
        cv2.circle(frame, tuple(p), 3, (255, 255, 255), -1)


def bar(frame, x, y, label, value, vmax=50.0, color=(0, 230, 0), drift=None):
    """Horizontal score bar in clinical units."""
    w = 260
    cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(frame, (x, y), (x + w, y + 26), (60, 60, 60), -1)
    fill = int(np.clip(value / vmax, 0, 1) * w)
    cv2.rectangle(frame, (x, y), (x + fill, y + 26), color, -1)
    txt = f"{value:5.1f}/50"
    if drift is not None:
        txt += f"   drift {drift:+5.1f}"
    cv2.putText(frame, txt, (x + w + 12, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--window", type=int, default=64, help="rolling buffer length (frames)")
    ap.add_argument("--infer-every", type=int, default=6, help="run models every N frames")
    ap.add_argument("--min-frames", type=int, default=24, help="min buffer before scoring")
    args = ap.parse_args()

    from engine import DemoEngine
    from pose_backend import PoseBackend

    print("[app] loading models (EGRU + PCT ensembles)...")
    eng = DemoEngine()
    pose = PoseBackend()
    print(f"[app] ready on {eng.device}. Point the camera at a person and press ] to rotate.")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"[app] cannot open camera {args.cam}. Try --cam 1.")

    buf_x = deque(maxlen=args.window)                    # Kinect-25 raw frames
    buf_t = deque(maxlen=args.window)                    # timestamps (s)
    t0 = time.time()

    exercise = 1
    angle = 0.0
    sweep = False
    sweep_dir = 1
    drop = False
    frozen = False
    last = {"egru": 0.0, "pct": 0.0}
    clean = {"egru": None, "pct": None}                  # scores at angle 0, for drift readout
    fno = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)                    # mirror = natural for the subject
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            world = pose.world(rgb, int((time.time() - t0) * 1000))

            kin_raw = None
            if world is not None and not frozen:
                kin_raw = mediapipe_to_kinect25(world)     # (33,3) -> (25,3)
                buf_t.append(time.time() - t0)
                buf_x.append(kin_raw)
                if drop and np.random.rand() < 0.4:       # Act 3: Gilbert-Elliott-ish drop
                    buf_t.pop(); buf_x.pop()

            if sweep:                                     # hands-free Act 2
                angle += sweep_dir * 2.0
                if angle >= 90:
                    angle, sweep_dir = 90, -1
                elif angle <= 0:
                    angle, sweep_dir = 0, 1

            # --- score every N frames ---
            fno += 1
            if len(buf_x) >= args.min_frames and fno % args.infer_every == 0:
                sample = preprocess_window(np.stack(buf_x), np.array(buf_t))
                r = eng.predict(sample, exercise=exercise, angle=angle)
                last = r
                if abs(angle) < 1e-6:
                    clean["egru"], clean["pct"] = r["egru"], r["pct"]

            # --- overlay ---
            if kin_raw is not None:
                draw_skeleton(frame, kin_raw)
            ed = None if clean["egru"] is None else last["egru"] - clean["egru"]
            pd = None if clean["pct"] is None else last["pct"] - clean["pct"]
            bar(frame, 20, 60, "EGRU (ours)", last["egru"], color=(0, 230, 0), drift=ed)
            bar(frame, 20, 120, "PCT (target)", last["pct"], color=(0, 140, 255), drift=pd)

            cv2.putText(frame, f"Exercise {exercise}   viewpoint {angle:+.0f} deg"
                        f"{'  [SWEEP]' if sweep else ''}{'  [DROP]' if drop else ''}"
                        f"{'  [FROZEN]' if frozen else ''}",
                        (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "keys: 1-5 exercise  [ ] rotate  r sweep  d drops  SPACE freeze  q quit",
                        (20, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("SE(3)-Equivariant Rehab Demo  --  EGRU vs PCT", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif ord('1') <= k <= ord('5'):
                exercise = k - ord('0')
            elif k == ord(']'):
                angle = min(180, angle + 15); sweep = False
            elif k == ord('['):
                angle = max(-180, angle - 15); sweep = False
            elif k == ord('0'):
                angle = 0.0; sweep = False
            elif k == ord('r'):
                sweep = not sweep
            elif k == ord('d'):
                drop = not drop
            elif k == ord(' '):
                frozen = not frozen
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
