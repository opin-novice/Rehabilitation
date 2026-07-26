#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
record_take.py
==============
Take-logger for the Variant B1 screen-to-webcam consistency experiment.

Captures one (clip_id, viewpoint) take: Camo video → MediaPipe → Kinect-25 → NPZ artifact.

Usage:
    # Synthetic dry-run (no camera needed)
    python demo/record_take.py --synthetic --clip-id kimore_es1_armlift --viewpoint straight_on

    # Real capture (Camo phone-as-webcam)
    python demo/record_take.py --clip-id kimore_es1_armlift --viewpoint straight_on --duration 30

    # Full argument list
    python demo/record_take.py --help

Artifact schema (NPZ with embedded JSON metadata):
    x           (T, 25, 3)  raw Kinect-mapped metres (not preprocessed)
    t           (T,)        arrival timestamps in seconds since take start
    clip_id     str         e.g. "kimore_es1_armlift"
    viewpoint   str         e.g. "straight_on", "plus30", "minus30", "tilt_up", ...
    exercise    int         KIMORE exercise id 1..5
    fps         float       measured frame rate
    duration_s  float       wall-clock duration of the take
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Synthetic dry-run generator (no camera, no MediaPipe)
# ---------------------------------------------------------------------------
from smoke_test import synthetic_mediapipe_buffer, mediapipe_to_kinect25  # noqa: E402


def generate_synthetic_take(T=300, fps=30.0, seed=0):
    """Generate a synthetic take mimicking a real capture.

    Returns (x_kinect, t_stamps) matching the NPZ schema.
    """
    mp_buf, t = synthetic_mediapipe_buffer(T=T, fps=fps, seed=seed)
    kin = np.stack([mediapipe_to_kinect25(f) for f in mp_buf], axis=0)
    return kin, t


# ---------------------------------------------------------------------------
# Real capture (Camo + MediaPipe)
# ---------------------------------------------------------------------------
def find_camo_camera():
    for i in range(10):
        import cv2
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return i
    return -1


def real_capture_loop(cam_idx, duration_s, window_s, min_frames):
    """Generator yielding (kinect25_frame, timestamp_s) tuples from Camo."""
    import cv2
    from pose_backend import PoseBackend

    pose = PoseBackend()
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise SystemExit(f"[record] cannot open camera {cam_idx}.")

    t0 = time.time()
    print(f"[record] capturing for {duration_s}s... (press Ctrl+C to stop early)")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.time() - t0
            world = pose.world(rgb, int(now * 1000))
            if world is not None:
                kin = mediapipe_to_kinect25(world)
                yield kin, now
            if now > duration_s:
                break
    except KeyboardInterrupt:
        print("\n[record] interrupted by user.")
    finally:
        cap.release()
        pose.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
VIEWPOINTS = [
    "straight_on", "plus30", "minus30",
    "tilt_up", "tilt_down",
    "closer", "farther",
]

CLIP_EXERCISE_MAP = {
    "kimore_es1_armlift": 1,
    "kimore_es3_trunkrot": 3,
    "kimore_es5_squat": 5,
}


def main():
    ap = argparse.ArgumentParser(
        description="Record one (clip_id, viewpoint) take for Variant B1.")
    ap.add_argument("--clip-id", type=str, default="kimore_es1_armlift",
                    choices=list(CLIP_EXERCISE_MAP.keys()),
                    help="Which exercise clip is being recorded")
    ap.add_argument("--viewpoint", type=str, default="straight_on",
                    choices=VIEWPOINTS,
                    help="Camera position for this take")
    ap.add_argument("--duration", type=float, default=15.0,
                    help="Capture duration in seconds (real capture only)")
    ap.add_argument("--cam", type=int, default=None,
                    help="Camera index (auto-detect if not specified)")
    ap.add_argument("--synthetic", action="store_true",
                    help="Dry-run mode: generate synthetic skeleton data, no camera")
    ap.add_argument("--output", type=str, default=None,
                    help="Output NPZ path (default: demo/takes/{clip_id}_{viewpoint}.npz)")
    ap.add_argument("--synth-frames", type=int, default=300,
                    help="Number of synthetic frames in dry-run mode")
    ap.add_argument("--synth-fps", type=float, default=30.0,
                    help="Synthetic frame rate")
    ap.add_argument("--synth-seed", type=int, default=42,
                    help="Random seed for synthetic data")
    args = ap.parse_args()

    exercise = CLIP_EXERCISE_MAP[args.clip_id]
    print(f"[record] take: clip={args.clip_id}  exercise={exercise}  viewpoint={args.viewpoint}")

    # --- Capture phase ---
    if args.synthetic:
        print("[record] SYNTHETIC dry-run mode (no camera)")
        x_kin, t_stamps = generate_synthetic_take(
            T=args.synth_frames, fps=args.synth_fps, seed=args.synth_seed)
        actual_duration = float(t_stamps[-1] - t_stamps[0])
        print(f"[record] generated {len(x_kin)} synthetic frames, "
              f"duration {actual_duration:.2f}s")
    else:
        cam_idx = args.cam if args.cam is not None else find_camo_camera()
        if cam_idx < 0:
            raise SystemExit(
                "[record] no camera found. Use --cam N to specify, or --synthetic for dry-run.")
        print(f"[record] using camera {cam_idx}")
        frames, stamps = [], []
        for kin, ts in real_capture_loop(cam_idx, args.duration, 64, 24):
            frames.append(kin)
            stamps.append(ts)
        if not frames or len(frames) < 3:
            raise SystemExit(f"[record] too few frames ({len(frames)}) captured — aborting.")
        x_kin = np.stack(frames, axis=0)
        t_stamps = np.array(stamps, dtype=np.float64)
        actual_duration = float(t_stamps[-1] - t_stamps[0])
        print(f"[record] captured {len(x_kin)} frames, "
              f"duration {actual_duration:.2f}s")

    # --- Build NPZ artifact ---
    t_rel = t_stamps - t_stamps[0]
    fps_measured = (len(t_rel) - 1) / actual_duration if actual_duration > 0 else 0.0

    metadata = {
        "clip_id": args.clip_id,
        "viewpoint": args.viewpoint,
        "exercise": int(exercise),
        "n_frames": int(len(x_kin)),
        "fps": float(round(fps_measured, 1)),
        "duration_s": float(round(actual_duration, 3)),
        "synthetic": bool(args.synthetic),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Determine output path
    if args.output is None:
        takes_dir = os.path.join(os.path.dirname(__file__), "takes")
        os.makedirs(takes_dir, exist_ok=True)
        out_path = os.path.join(takes_dir, f"{args.clip_id}_{args.viewpoint}.npz")
    else:
        out_path = args.output
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    np.savez_compressed(out_path,
                        x=x_kin.astype(np.float64),
                        t=t_rel.astype(np.float64),
                        _metadata=json.dumps(metadata))
    print(f"[record] saved {out_path}")
    print(f"[record] metadata: {json.dumps(metadata, indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
