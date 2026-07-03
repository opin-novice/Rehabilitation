"""V1 - REHAB24-6 labeled validity testbed loader.

Produces a per-repetition manifest with BINARY correctness ground truth so the
zero-shot reliability diagnostic can be validated against real movement-quality
labels (reviewer weakness W5: reliability != validity).

Data layout (REHAB24-6, Zenodo 13305826):
  REHAB246/joints/Ex{1..6}/{video_id}-30fps.npy   ndarray (T, 26, 4)  [x,y,z,?]
  Segmentation.csv (sep=';'): video_id, repetition_number, exercise_id, person_id,
      first_frame, last_frame, mocap_erroneous, correctness (1=correct, 0=incorrect)
  Frame indices are defined on the 30-FPS stream -> we slice the -30fps arrays.

Label stats (verified): 1057 mocap-OK reps, 558 correct / 499 incorrect, 6 exercises,
10 subjects -- a balanced binary testbed.

Joint mapping: REHAB24-6 uses a 26-joint mocap skeleton; KIMORE/Kinect-v2 uses 25.
`map_26_to_25()` holds the anatomically-aligned index map, validated against the
official REHAB24-6 joints_names.txt (Zenodo 13305826: 26-joint OptiTrack Motive
skeleton). OptiTrack "*Arm/*ForeArm/*Hand" map to Kinect Shoulder/Elbow/Wrist; the
clavicle ("*Shoulder") and "Head_end" have no Kinect target and are dropped.

Usage:
  python src/load_rehab246.py --build      # parse + cache -> outputs/validity/
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/
from constants import SEQ_LEN, NUM_JOINTS, NUM_CHANNELS

SEG_CSV = "Segmentation.csv"
JOINTS_DIR = "REHAB246/joints"
OUT_DIR = "outputs/validity"


def _resample(seq: np.ndarray, target: int) -> np.ndarray:
    """Linear temporal resample (T, D) -> (target, D)."""
    T = seq.shape[0]
    if T == target:
        return seq
    if T < 2:
        return np.repeat(seq, target, axis=0)[:target]
    idx = np.linspace(0, T - 1, target)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    frac = (idx - lo)[:, None]
    return seq[lo] * (1 - frac) + seq[hi] * frac


# KIMORE Kinect-v2 (25 joints, target) <- REHAB24-6 OptiTrack Motive (26 joints).
# Source indices verified against the official REHAB24-6 joints_names.txt
# (Zenodo 13305826):
#    0 Hips  1 Spine  2 Spine1  3 Neck  4 Head  5 Head_end
#    6 LeftShoulder  7 LeftArm  8 LeftForeArm  9 LeftHand  10 LeftHand_end
#   11 RightShoulder 12 RightArm 13 RightForeArm 14 RightHand 15 RightHand_end
#   16 LeftUpLeg  17 LeftLeg  18 LeftFoot  19 LeftToeBase  20 LeftToeBase_end
#   21 RightUpLeg 22 RightLeg 23 RightFoot 24 RightToeBase 25 RightToeBase_end
# Target order is Kinect-v2 (constants.SKELETON_EDGES / KIMORE layout):
#    0 SpineBase 1 SpineMid 2 Neck 3 Head 4-7 L-arm 8-11 R-arm
#   12-15 L-leg 16-19 R-leg 20 SpineShoulder 21 HandTipL 22 ThumbL 23 HandTipR 24 ThumbR
# OptiTrack semantics: "*Arm" = glenohumeral (shoulder ball) -> Kinect Shoulder;
# "*ForeArm" starts at the elbow -> Kinect Elbow; "*Hand" is the wrist -> Kinect
# Wrist/Hand; "*Hand_end" is the fingertip -> Kinect HandTip. Kinect Thumb has no
# OptiTrack equivalent and is approximated by the wrist/hand joint. "*Shoulder"
# (clavicle, idx 6/11) and "Head_end" (idx 5) have no Kinect target and are dropped.
KINECT_FROM_REHAB26: list[int] = [
    0, 1, 3, 4,        #  0-3  SpineBase, SpineMid, Neck, Head
    7, 8, 9, 9,        #  4-7  Shoulder, Elbow, Wrist, Hand  (Left)
    12, 13, 14, 14,    #  8-11 Shoulder, Elbow, Wrist, Hand  (Right)
    16, 17, 18, 19,    # 12-15 Hip, Knee, Ankle, Foot        (Left)
    21, 22, 23, 24,    # 16-19 Hip, Knee, Ankle, Foot        (Right)
    2,                 # 20    SpineShoulder <- Spine1
    10, 9, 15, 14,     # 21-24 HandTipL, ThumbL, HandTipR, ThumbR
]
assert len(KINECT_FROM_REHAB26) == NUM_JOINTS


def map_26_to_25(seq26: np.ndarray) -> np.ndarray:
    """REHAB24-6 (26-joint OptiTrack) -> KIMORE Kinect-v2 (25-joint) layout.

    Anatomically-aligned permutation validated against the official joints_names.txt
    (Zenodo 13305826). Replaces the earlier first-25-joint placeholder.
    seq26: (T, 26, 3) -> (T, 25, 3).
    """
    assert seq26.shape[1] >= 26, f"expected >=26 source joints, got {seq26.shape[1]}"
    return seq26[:, KINECT_FROM_REHAB26, :]


def build() -> dict:
    if not os.path.exists(SEG_CSV) or not os.path.isdir(JOINTS_DIR):
        print(f"[SKIP] Need {SEG_CSV} and {JOINTS_DIR}/. Download REHAB24-6 (Zenodo 13305826):")
        print('  curl -L -o 3d_joints.zip "https://zenodo.org/records/13305826/files/3d_joints.zip?download=1"')
        print('  curl -L -o Segmentation.csv "https://zenodo.org/records/13305826/files/Segmentation.csv?download=1"')
        print('  curl -L -o REHAB246/joints_names.txt "https://zenodo.org/records/13305826/files/joints_names.txt?download=1"')
        return {}

    seg = pd.read_csv(SEG_CSV, sep=";")
    seg = seg[seg["mocap_erroneous"] == 0].reset_index(drop=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    manifest, seqs = [], []
    missing = 0
    for _, r in seg.iterrows():
        npy = os.path.join(JOINTS_DIR, f"Ex{int(r.exercise_id)}", f"{r.video_id}-30fps.npy")
        if not os.path.exists(npy):
            missing += 1
            continue
        arr = np.load(npy)  # (T, 26, 4)
        a, b = int(r.first_frame), int(r.last_frame)
        clip = arr[a:b + 1, :, :3]  # xyz only
        if clip.shape[0] < 2:
            continue
        s25 = map_26_to_25(clip)                       # (t, 25, 3)
        s25 = _resample(s25.reshape(s25.shape[0], -1), SEQ_LEN)
        s25 = s25.reshape(SEQ_LEN, NUM_JOINTS, NUM_CHANNELS).astype(np.float32)
        uid = f"{r.video_id}_ex{int(r.exercise_id)}_rep{int(r.repetition_number)}"
        manifest.append({
            "rep_uid": uid,
            "exercise_id": int(r.exercise_id),
            "subject_id": int(r.person_id),
            "correct_label": int(r.correctness),
            "n_frames": int(b - a + 1),
        })
        seqs.append(s25)

    man = pd.DataFrame(manifest)
    man_path = os.path.join(OUT_DIR, "rehab246_manifest.csv")
    man.to_csv(man_path, index=False)
    seq_path = os.path.join(OUT_DIR, "rehab246_sequences.npy")
    np.save(seq_path, np.stack(seqs, axis=0))

    bal = man["correct_label"].value_counts().to_dict()
    print(f"REHAB24-6 testbed: {len(man)} reps  (correct={bal.get(1,0)} / incorrect={bal.get(0,0)})")
    print(f"  exercises={sorted(man.exercise_id.unique())}  subjects={sorted(man.subject_id.unique())}")
    if missing:
        print(f"  [warn] {missing} segmentation rows had no matching -30fps.npy")
    print(f"  -> {man_path}")
    print(f"  -> {seq_path}  shape={np.stack(seqs).shape}")
    print("  joint map: anatomically-aligned 26->25 (validated vs joints_names.txt).")
    return {"n": len(man), "balance": bal}


def main() -> None:
    ap = argparse.ArgumentParser(description="V1 REHAB24-6 labeled testbed loader")
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    build()


if __name__ == "__main__":
    main()
