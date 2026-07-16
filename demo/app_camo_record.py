#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app_camo_record.py
==================
Record a video of the Camo demo for booth backup / rehearsal review.

This is identical to app_camo.py but saves the output to an MP4 file while displaying it live.

Run:  python demo/app_camo_record.py            (auto-detect Camo, record to recordings/)
      python demo/app_camo_record.py --cam 1    (force camera 1)

Press q to quit. The video file is saved to demo/recordings/demo_TIMESTAMP.mp4.

Keyboard controls are identical to app_camo.py — perform the demo normally, and the video
captures everything: skeleton, scores, rotation, Act 2 contrast, etc.
"""

import argparse
import os
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mp_to_kinect import mediapipe_to_kinect25, preprocess_window, KINECT_NAMES   # noqa: E402

# Kinect bone list for drawing
DRAW_BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7),
    (20, 8), (8, 9), (9, 10), (10, 11),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]


def find_camo_camera():
    """Auto-detect the Camo camera device index."""
    print("[camo] searching for Camo virtual camera...")
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                cap = cv2.VideoCapture(i)
                ret2, _ = cap.read()
                cap.release()
                if ret2:
                    print(f"[camo] found camera {i}")
                    return i
    print("[camo] could not auto-detect Camo. Use --cam N to force it.")
    return -1


def setup_video_writer(output_path, frame_shape, fps=30):
    """Initialize video writer for MP4 output."""
    h, w = frame_shape[:2]
    # Use MP4V codec for broad compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"[record] WARNING: video writer failed to open. Trying MJPEG fallback...")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    return writer


def draw_skeleton(frame, kin_raw, color=(0, 230, 255)):
    """Project the Kinect-25 skeleton onto the frame."""
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
    ap = argparse.ArgumentParser(
        description="Record a video of the Camo demo for booth backup / rehearsal.")
    ap.add_argument("--cam", type=int, default=None,
                    help="force camera index (if not specified, auto-detect)")
    ap.add_argument("--window", type=int, default=64, help="rolling buffer length (frames)")
    ap.add_argument("--infer-every", type=int, default=6, help="run models every N frames")
    ap.add_argument("--min-frames", type=int, default=24, help="min buffer before scoring")
    ap.add_argument("--output", type=str, default=None,
                    help="output video file (default: demo/recordings/demo_TIMESTAMP.mp4)")
    args = ap.parse_args()

    from engine import DemoEngine
    from pose_backend import PoseBackend

    print("[record] loading models (EGRU + PCT ensembles)...")
    eng = DemoEngine()
    pose = PoseBackend()

    # Determine camera device
    if args.cam is not None:
        cam_idx = args.cam
        print(f"[camo] using camera {cam_idx} (forced)")
    else:
        cam_idx = find_camo_camera()
        if cam_idx < 0:
            print("[camo] falling back to camera 0")
            cam_idx = 0

    print(f"[record] ready on {eng.device}. Recording starts now.")

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise SystemExit(f"[record] cannot open camera {cam_idx}.")

    # Create output directory if needed
    rec_dir = os.path.join(os.path.dirname(__file__), "recordings")
    os.makedirs(rec_dir, exist_ok=True)

    # Determine output filename
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(rec_dir, f"demo_{timestamp}.mp4")
    else:
        output_file = args.output

    print(f"[record] video will be saved to: {output_file}")

    buf_x = deque(maxlen=args.window)
    buf_t = deque(maxlen=args.window)
    t0 = time.time()

    exercise = 1
    angle = 0.0
    sweep = False
    sweep_dir = 1
    drop = False
    frozen = False
    last = {"egru": 0.0, "pct": 0.0}
    clean = {"egru": None, "pct": None}
    fno = 0

    writer = None
    recording = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            world = pose.world(rgb, int((time.time() - t0) * 1000))

            # Initialize video writer on first frame
            if writer is None:
                writer = setup_video_writer(output_file, frame.shape, fps=30)
                recording = writer.isOpened()
                if recording:
                    print(f"[record] recording started at {output_file}")
                else:
                    print(f"[record] WARNING: recording failed to start")

            kin_raw = None
            if world is not None and not frozen:
                kin_raw = mediapipe_to_kinect25(world)
                buf_t.append(time.time() - t0)
                buf_x.append(kin_raw)
                if drop and np.random.rand() < 0.4:
                    buf_t.pop(); buf_x.pop()

            if sweep:
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
                        f"{'  [FROZEN]' if frozen else ''} [RECORDING]",
                        (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, "keys: 1-5 exercise  [ ] rotate  r sweep  d drops  SPACE freeze  q quit",
                        (20, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Record frame
            if recording and writer is not None:
                writer.write(frame)

            cv2.imshow("SE(3)-Equivariant Rehab Demo (Camo) -- RECORDING", frame)
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
        if writer is not None:
            writer.release()
            if recording:
                print(f"[record] saved: {output_file}")
        cv2.destroyAllWindows()
        pose.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
