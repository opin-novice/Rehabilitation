#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pose_backend.py
===============
Thin wrapper over MediaPipe's Tasks PoseLandmarker (the legacy `mp.solutions.pose` API is gone in
mediapipe >= 0.10.3x). Returns per-frame world landmarks as a plain (33, 3) numpy array in metres,
so app.py never touches the Tasks plumbing.

The .task model lives in demo/models/ (fetched once). If missing, we print the exact download URL.
"""

import os

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "pose_landmarker_full.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")


class PoseBackend:
    """Stateful VIDEO-mode pose tracker. Feed monotonically increasing timestamps (ms)."""

    def __init__(self, model_path=MODEL_PATH, num_poses=1):
        if not os.path.exists(model_path):
            raise SystemExit(
                f"[pose] model not found: {model_path}\n"
                f"       download it once with:\n"
                f"       curl -L -o \"{model_path}\" \"{MODEL_URL}\"")
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False)
        self._lm = mp_vision.PoseLandmarker.create_from_options(opts)

    def world(self, rgb_uint8: np.ndarray, timestamp_ms: int):
        """rgb_uint8: (H,W,3) uint8 RGB. Returns (33,3) world landmarks (metres) or None."""
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=np.ascontiguousarray(rgb_uint8))
        res = self._lm.detect_for_video(img, int(timestamp_ms))
        if not res.pose_world_landmarks:
            return None
        pts = res.pose_world_landmarks[0]
        return np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float64)

    def close(self):
        self._lm.close()
