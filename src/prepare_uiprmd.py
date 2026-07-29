"""Convert UI-PRMD to KIMORE-compatible Train_X/Y CSVs.

UI-PRMD structure:
  Kinect/Positions/  m{01-10}_s{01-10}_e{01-10}_positions.txt      (correct,   label 1)
  Kinect/Angles/     m{01-10}_s{01-10}_e{01-10}_angles.txt
  ...and the same under "Incorrect Segmented Movements/" with an _inc suffix (label 0).
  10 movements x 10 subjects x 10 reps x 2 classes = 2000 samples.

GEOMETRY -- READ THIS BEFORE TRUSTING ANY OUTPUT
------------------------------------------------
This module used to document its input as "[T, 66] = 22 joints x 3 xyz (centimetres)"
and feed those numbers straight through as coordinates. That was wrong. The `Positions`
files are per-frame PARENT-RELATIVE BONE OFFSETS (joint 0 is the absolute root); read
literally, 19 of 66 slots are constant, so the skeleton is near-static and only the root
translates. Measured on the built tensor, 57.3% of all coordinates never changed.
See docs/worklog_2026-07-29.md §2 and §4.

World coordinates are recovered by forward kinematics against the matching `Angles/`
file -- src/uiprmd_fk.py, whose convention is transcribed from the authors' own
`Animation.m` and validated against Vicon ground truth.

TARGET -- UI-PRMD HAS NO CLINICAL SCORES
----------------------------------------
This module used to emit `Score = 35 + 15 * (1 - normalised L2 to the per-movement mean
trajectory)`, deliberately landing in KIMORE's 35-50 "CG/Expert" band. That number is
FABRICATED: it measures distance from the cohort mean, not movement quality, and nothing
clinical grounds it. Dressing it in KIMORE's units invites it to be consumed, and
reported, as though it were a clinician rating.

What UI-PRMD *does* provide is a genuine binary label: correct vs incorrect execution.
That is now the default target (`--target binary`). The synthetic score remains reachable
only via an explicit `--target synthetic`, which prints a warning and stamps
`target_is_synthetic=1` into meta.csv. Never report a regression result trained on it.

Compatibility with KIMORE pipeline:
  - Joints padded 22 -> 25 (3 dummy joints at end, all-zero, tracking_state=0)
  - Resampled to seq_len=100 (same as KIMORE)
  - Stored in KIMORE 100-col format (25 joints x 4 cols: x,y,z,tracking_state)
  - Subject IDs: s01=77 .. s10=86 (offset past KIMORE's 0-76)
  - Exercise IDs: m01=5 .. m10=14 (offset past KIMORE's 0-4)

Output:
  <out_dir>/Train_X.csv       (n*seq_len rows x 100 cols)
  <out_dir>/Train_Y.csv       (n rows: correctness label, or synthetic score if forced)
  <out_dir>/subject_ids.csv   (n rows, global subject IDs 77-86)
  <out_dir>/exercise_ids.csv  (n rows, global exercise IDs 5-14)
  <out_dir>/meta.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/
from uiprmd_fk import forward_kinematics

NUM_JOINTS_KIMORE = 25
COLS_PER_JOINT_KIMORE = 4          # x, y, z, tracking_state
TOTAL_KIMORE_COLS = NUM_JOINTS_KIMORE * COLS_PER_JOINT_KIMORE  # 100

NUM_JOINTS_UIPRMD = 22
SEQ_LEN = 100

SUBJECT_ID_OFFSET  = 77   # KIMORE subjects occupy 0-76
EXERCISE_ID_OFFSET = 5    # KIMORE exercises occupy 0-4

CORRECT_ROOT   = Path(r"D:\Rehabilation\Segmented Movements\Kinect")
INCORRECT_ROOT = Path(r"D:\Rehabilation\Incorrect Segmented Movements\Kinect")


def load_world_positions(pos_path: Path, ang_path: Path,
                         fk_variant: str = "reference") -> np.ndarray | None:
    """Load a UI-PRMD (offsets, angles) pair and return world coordinates [T, 66]."""
    offs = np.loadtxt(pos_path, delimiter=",", dtype=np.float64, ndmin=2)
    if offs.ndim != 2 or offs.shape[1] != NUM_JOINTS_UIPRMD * 3:
        return None
    if not ang_path.is_file():
        return None
    angs = np.loadtxt(ang_path, delimiter=",", dtype=np.float64, ndmin=2)
    if angs.shape[1] != NUM_JOINTS_UIPRMD * 3:
        return None
    T = min(len(offs), len(angs))
    if T < 2:
        return None
    world = forward_kinematics(offs[:T].reshape(T, NUM_JOINTS_UIPRMD, 3),
                               angs[:T].reshape(T, NUM_JOINTS_UIPRMD, 3),
                               variant=fk_variant)
    return world.reshape(T, NUM_JOINTS_UIPRMD * 3).astype(np.float32)


def resample(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly interpolate each column to target_len rows."""
    T = arr.shape[0]
    if T == target_len:
        return arr
    old_t = np.linspace(0.0, 1.0, T)
    new_t = np.linspace(0.0, 1.0, target_len)
    out = np.zeros((target_len, arr.shape[1]), dtype=np.float32)
    for c in range(arr.shape[1]):
        out[:, c] = np.interp(new_t, old_t, arr[:, c])
    return out


def pad_to_kimore_format(data: np.ndarray) -> np.ndarray:
    """Convert [T, 66] (22j x 3xyz) -> [T, 100] (25j x 4 KIMORE format).

    Real joints (0-21): tracking_state = 2.
    Padding joints (22-24): all zeros (tracking_state = 0).
    """
    T = data.shape[0]
    out = np.zeros((T, TOTAL_KIMORE_COLS), dtype=np.float32)
    for j in range(NUM_JOINTS_UIPRMD):
        out[:, j * 4]     = data[:, j * 3]       # x
        out[:, j * 4 + 1] = data[:, j * 3 + 1]   # y
        out[:, j * 4 + 2] = data[:, j * 3 + 2]   # z
        out[:, j * 4 + 3] = 2.0                   # tracking_state = tracked
    # joints 22, 23, 24 stay 0
    return out


def compute_synthetic_scores(
    sequences: dict[str, list[np.ndarray]],
    score_lo: float = 35.0,
    score_hi: float = 50.0,
) -> dict[str, list[float]]:
    """FABRICATED target: distance from the per-movement mean trajectory, mapped to 35-50.

    This is NOT a quality measure and NOT clinically grounded -- it says only how far a
    repetition sits from its cohort's average, which an atypical-but-correct execution
    fails just as readily as a genuinely poor one. Retained solely so the legacy
    behaviour is reproducible; see the module docstring.
    """
    scores: dict[str, list[float]] = {}
    for mov_id, reps in sequences.items():
        stack = np.stack(reps, axis=0)            # [N, seq_len, 66]
        mean_traj = stack.mean(axis=0)            # [seq_len, 66]
        dists = np.sqrt(((stack - mean_traj) ** 2).mean(axis=(1, 2)))  # [N]
        max_d = dists.max()
        normalised = np.zeros_like(dists) if max_d < 1e-8 else dists / max_d
        scores[mov_id] = [float(score_hi + (score_lo - score_hi) * n) for n in normalised]
    return scores


def _gather(root: Path, label: int, suffix: str) -> list[tuple]:
    """List (mov, subj, rep, label, positions_path, angles_path) under one class root."""
    pos_dir, ang_dir = root / "Positions", root / "Angles"
    if not pos_dir.is_dir():
        return []
    out = []
    for f in sorted(pos_dir.glob("*.txt")):
        parts = f.stem.split("_")               # ["m01","s01","e01","positions"(,"inc")]
        if len(parts) < 4:
            continue
        ang = ang_dir / f.name.replace("_positions", "_angles")
        out.append((parts[0], parts[1], parts[2], label, f, ang))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a KIMORE-format UI-PRMD tensor with true FK geometry.")
    parser.add_argument("--correct_root",   default=str(CORRECT_ROOT))
    parser.add_argument("--incorrect_root", default=str(INCORRECT_ROOT))
    parser.add_argument("--out_dir",        default=r"D:\Rehabilation\UIPRMD_processed")
    parser.add_argument("--seq_len",        type=int, default=SEQ_LEN)
    parser.add_argument("--fk-variant", choices=["reference", "corrected"], default="reference")
    parser.add_argument("--target", choices=["binary", "synthetic"], default="binary",
                        help="binary = real correct/incorrect label; "
                             "synthetic = FABRICATED legacy score (not clinically grounded)")
    args = parser.parse_args()

    records = _gather(Path(args.correct_root), 1, "")
    records += _gather(Path(args.incorrect_root), 0, "_inc")
    if not records:
        print(f"[FATAL] No positions files under {args.correct_root} / {args.incorrect_root}")
        raise SystemExit(1)
    print(f"Found {len(records)} repetitions "
          f"({sum(r[3] for r in records)} correct / {sum(1 - r[3] for r in records)} incorrect)")

    print(f"Running forward kinematics (variant={args.fk_variant}) and resampling ...")
    seq_by_mov: dict[str, list[np.ndarray]] = {}
    kept, skipped = [], []
    for mov_id, subj_id, rep_id, label, fpos, fang in records:
        world = load_world_positions(fpos, fang, args.fk_variant)
        if world is None:
            skipped.append(fpos.name)
            continue
        seq = resample(world, args.seq_len)          # [seq_len, 66]
        key = f"{mov_id}_c{label}"
        seq_by_mov.setdefault(key, []).append(seq)
        kept.append((mov_id, subj_id, rep_id, label, key, len(seq_by_mov[key]) - 1, fpos.name))
    if skipped:
        print(f"  skipped {len(skipped)} unusable: {skipped[:4]}{' ...' if len(skipped) > 4 else ''}")

    synthetic = compute_synthetic_scores(seq_by_mov) if args.target == "synthetic" else None
    if synthetic is not None:
        print("\n  *** WARNING: --target synthetic emits a FABRICATED score with no clinical\n"
              "      grounding (distance from the per-movement mean, mapped into KIMORE's\n"
              "      35-50 band). Do not report regression results trained on it. ***\n")

    all_x, all_y, all_sid, all_eid, meta_out = [], [], [], [], []
    for mov_id, subj_id, rep_id, label, key, idx, fname in kept:
        mov_int  = int(mov_id[1:]) - 1   # m01->0, m10->9
        subj_int = int(subj_id[1:]) - 1  # s01->0, s10->9
        seq = seq_by_mov[key][idx]
        target = float(label) if synthetic is None else synthetic[key][idx]

        all_x.append(pad_to_kimore_format(seq))
        all_y.append(target)
        all_sid.append(SUBJECT_ID_OFFSET + subj_int)
        all_eid.append(EXERCISE_ID_OFFSET + mov_int)
        meta_out.append({
            "mov_id": mov_id, "subj_id": subj_id, "rep_id": rep_id, "file": fname,
            "global_subject_id": SUBJECT_ID_OFFSET + subj_int,
            "exercise_id": EXERCISE_ID_OFFSET + mov_int,
            "correct_label": label,
            "target": target,
            "target_is_synthetic": int(synthetic is not None),
            "geometry": f"fk:{args.fk_variant}",
            "group": "UI-PRMD/Healthy",
        })

    X   = np.stack(all_x, axis=0).reshape(-1, TOTAL_KIMORE_COLS)   # [N*seq_len, 100]
    Y   = np.array(all_y, dtype=np.float32)
    SID = np.array(all_sid, dtype=np.int32)
    EID = np.array(all_eid, dtype=np.int32)
    META = pd.DataFrame(meta_out)

    xyz = np.stack(all_x, axis=0).reshape(len(all_y), args.seq_len, NUM_JOINTS_KIMORE, 4)[..., :3]
    const_frac = float((xyz.std(axis=1) <= 1e-9).mean())

    print(f"\nUI-PRMD processed:")
    print(f"  Samples: {len(Y)}  (X shape: {X.shape})")
    print(f"  Target: {args.target}  range [{Y.min():.2f}, {Y.max():.2f}]  mean={Y.mean():.3f}")
    print(f"  Constant-coordinate fraction: {const_frac:.3f} "
          f"(3/25 padded joints = 0.120 is the floor; the old offset build was 0.573)")
    print(f"  Subjects (global): {SID.min()} - {SID.max()}")
    print(f"  Exercises (global): {EID.min()} - {EID.max()}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X).to_csv(  os.path.join(args.out_dir, "Train_X.csv"),      header=False, index=False)
    pd.DataFrame(Y).to_csv(  os.path.join(args.out_dir, "Train_Y.csv"),      header=False, index=False)
    pd.DataFrame(SID).to_csv(os.path.join(args.out_dir, "subject_ids.csv"),  header=False, index=False)
    pd.DataFrame(EID).to_csv(os.path.join(args.out_dir, "exercise_ids.csv"), header=False, index=False)
    META.to_csv(             os.path.join(args.out_dir, "meta.csv"),          index=False)
    print(f"  Written to: {args.out_dir}")


if __name__ == "__main__":
    main()
