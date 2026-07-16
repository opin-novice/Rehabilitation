#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mp_to_kinect.py
===============
Live capture bridge: MediaPipe Pose (33 landmarks) -> Kinect-v2 (25 joints), preprocessed
EXACTLY as kimore_cde_data.load_sample does, so a webcam frame lands in the same distribution
the EGRU/PCT checkpoints were trained on.

Two responsibilities, kept separate:
  1. GEOMETRY REMAP  -- map MediaPipe's 33 world landmarks onto the 25 Kinect joints the model
     expects, synthesizing the spine chain (SpineBase/SpineMid/SpineShoulder/Neck) and the hand
     tips/thumbs that MediaPipe reports differently. Pure function of one frame.
  2. TRAINING-MATCHED PREPROCESS -- root-relative on SpineBase (joint 0) + per-window median-radius
     scale norm + t/TIME_SCALE stamps. Byte-for-byte the operations in load_sample, so nothing the
     model sees at demo time differs in kind from training (only in domain -- that gap is Task 7's
     calibration job, NOT this file's).

AXIS CONVENTION. Kinect v2 is a right-handed camera frame with +Y UP (the gravity axis the whole
viewpoint theorem rotates about -- see block2_transforms.azimuth_matrix). MediaPipe world landmarks
are metric metres with +Y DOWN. We therefore negate Y. X and Z handedness only needs to be
CONSISTENT: the shipped (non-chiral) EGRU is O(3)-invariant, so a left/right mirror cannot change
its score (certify_egru E6). The signs live in AXIS_SIGN so Day-4 calibration can flip them without
touching logic.
"""

import numpy as np

TIME_SCALE = 20.0            # must match kimore_cde_data.TIME_SCALE (seconds -> O(1))
N_KINECT = 25

# --- MediaPipe Pose landmark indices (BlazePose 33) -------------------------------------------
MP = dict(
    nose=0, l_ear=7, r_ear=8,
    l_sho=11, r_sho=12, l_elb=13, r_elb=14, l_wri=15, r_wri=16,
    l_pinky=17, r_pinky=18, l_index=19, r_index=20, l_thumb=21, r_thumb=22,
    l_hip=23, r_hip=24, l_knee=25, r_knee=26, l_ank=27, r_ank=28,
    l_foot=31, r_foot=32,
)

# --- Kinect v2 joint order (indices 0..24), for reference / overlay labelling -----------------
KINECT_NAMES = [
    "SpineBase", "SpineMid", "Neck", "Head",
    "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft",
    "ShoulderRight", "ElbowRight", "WristRight", "HandRight",
    "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft",
    "HipRight", "KneeRight", "AnkleRight", "FootRight",
    "SpineShoulder", "HandTipLeft", "ThumbLeft", "HandTipRight", "ThumbRight",
]

# MediaPipe world axes -> Kinect axes.  (x, -y, z): flip Y so +Y is up.
AXIS_SIGN = np.array([1.0, -1.0, 1.0], dtype=np.float64)


def mediapipe_to_kinect25(world_xyz: np.ndarray) -> np.ndarray:
    """One frame of MediaPipe world landmarks (33, 3) -> Kinect-25 skeleton (25, 3), metres.

    Joints MediaPipe reports directly are copied; the spine chain and hand tips are synthesized
    from the anatomically adjacent landmarks. No normalization here -- that is preprocess_window's
    job, so this stays a pure geometric remap that is trivial to unit-test.
    """
    w = np.asarray(world_xyz, dtype=np.float64) * AXIS_SIGN           # (33,3) in Kinect axes
    j = np.zeros((N_KINECT, 3), dtype=np.float64)

    hip_mid = 0.5 * (w[MP["l_hip"]] + w[MP["r_hip"]])
    sho_mid = 0.5 * (w[MP["l_sho"]] + w[MP["r_sho"]])
    head = 0.5 * (w[MP["l_ear"]] + w[MP["r_ear"]])                    # head centre, not nose tip

    # --- spine chain (synthesized) ---
    j[0] = hip_mid                                                    # SpineBase  (the root)
    j[20] = sho_mid                                                   # SpineShoulder
    j[1] = 0.5 * (hip_mid + sho_mid)                                  # SpineMid
    j[2] = 0.5 * (sho_mid + head)                                     # Neck
    j[3] = head                                                       # Head

    # --- left arm ---
    j[4] = w[MP["l_sho"]]
    j[5] = w[MP["l_elb"]]
    j[6] = w[MP["l_wri"]]
    j[7] = 0.5 * (w[MP["l_pinky"]] + w[MP["l_index"]])               # HandLeft (palm centre)
    j[21] = w[MP["l_index"]]                                          # HandTipLeft
    j[22] = w[MP["l_thumb"]]                                         # ThumbLeft

    # --- right arm ---
    j[8] = w[MP["r_sho"]]
    j[9] = w[MP["r_elb"]]
    j[10] = w[MP["r_wri"]]
    j[11] = 0.5 * (w[MP["r_pinky"]] + w[MP["r_index"]])
    j[23] = w[MP["r_index"]]
    j[24] = w[MP["r_thumb"]]

    # --- left leg ---
    j[12] = w[MP["l_hip"]]
    j[13] = w[MP["l_knee"]]
    j[14] = w[MP["l_ank"]]
    j[15] = w[MP["l_foot"]]

    # --- right leg ---
    j[16] = w[MP["r_hip"]]
    j[17] = w[MP["r_knee"]]
    j[18] = w[MP["r_ank"]]
    j[19] = w[MP["r_foot"]]

    return j


def preprocess_window(x_seq: np.ndarray, t_seq: np.ndarray):
    """A rolling buffer -> a `sample` dict the model consumes, matching load_sample exactly.

    x_seq : (L, 25, 3) raw Kinect-mapped metres (from mediapipe_to_kinect25 per frame)
    t_seq : (L,)       arrival stamps in SECONDS (monotonic; real inter-arrival gaps preserved)

    Returns dict(t (L,), x (L,25,3), n_frames, ...) identical in form to kimore_cde_data.load_sample:
      * t := (t - t0) / TIME_SCALE            -- start at 0, duration preserved, O(1) scaled
      * x := x - x[:, 0:1, :]                 -- root-relative on SpineBase (translation invariance)
      * x := x / median(joint radius)         -- Type-0 scale (commutes with rotation)
    """
    x = np.asarray(x_seq, dtype=np.float64).copy()
    t = np.asarray(t_seq, dtype=np.float64).copy()

    # strict monotonic stamps (mirror load_sample's filter; a webcam can repeat a tick under lag)
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, x = t[keep], x[keep]

    t = t - t[0]
    x = x - x[:, 0:1, :]                                              # root-relative on SpineBase

    radius = np.linalg.norm(x, axis=-1)                              # (L,25)
    scale = float(np.median(radius[radius > 0])) if np.any(radius > 0) else 1.0
    x = x / max(scale, 1e-6)

    return {
        "t": t / TIME_SCALE,
        "x": x,
        "n_frames": len(t),
        "duration_s": float(t[-1]) if len(t) else 0.0,
        "scale": scale,
    }
