"""Convert UI-PRMD 'Segmented Movements' dataset to KIMORE-compatible Train_X/Y CSVs.

UI-PRMD structure:
  Kinect/Positions/  m{01-10}_s{01-10}_e{01-10}_positions.txt
  10 movements x 10 subjects x 10 reps = 1000 samples
  Each file: [T_variable, 66] — 22 joints x 3 xyz (centimetres)

Compatibility with KIMORE pipeline:
  - Joints padded 22 → 25 (3 dummy joints at end, all-zero, tracking_state=0)
  - Resampled to seq_len=100 (same as KIMORE)
  - Stored in KIMORE 100-col format (25 joints x 4 cols: x,y,z,tracking_state)
  - Subject IDs: s01=77 .. s10=86 (offset past KIMORE's 0-76)
  - Exercise IDs: m01=5 .. m10=14 (offset past KIMORE's 0-4)

Quality score:
  No clinical labels exist (all subjects healthy).
  Score = 35 + 15 * (1 - normalised_L2_distance_to_per-movement_mean_trajectory)
  → healthy-representative reps score ~50, outlier reps score ~35.
  Range [35, 50] sits in the KIMORE 'CG/Expert' band.

Output:
  <out_dir>/Train_X.csv       (n*seq_len rows x 100 cols)
  <out_dir>/Train_Y.csv       (n rows)
  <out_dir>/subject_ids.csv   (n rows, global subject IDs 77-86)
  <out_dir>/exercise_ids.csv  (n rows, global exercise IDs 5-14)
  <out_dir>/meta.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

NUM_JOINTS_KIMORE = 25
COLS_PER_JOINT_KIMORE = 4          # x, y, z, tracking_state
TOTAL_KIMORE_COLS = NUM_JOINTS_KIMORE * COLS_PER_JOINT_KIMORE  # 100

NUM_JOINTS_UIPRMD = 22
SEQ_LEN = 100

SUBJECT_ID_OFFSET  = 77   # KIMORE subjects occupy 0-76
EXERCISE_ID_OFFSET = 5    # KIMORE exercises occupy 0-4


def load_positions(path: str) -> np.ndarray:
    """Load a UI-PRMD positions file → float32 array [T, 66]."""
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    assert data.ndim == 2 and data.shape[1] == NUM_JOINTS_UIPRMD * 3, (
        f"{path}: expected {NUM_JOINTS_UIPRMD * 3} cols, got {data.shape}"
    )
    return data


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
    """Convert [T, 66] (22j x 3xyz) → [T, 100] (25j x 4 KIMORE format).

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


def compute_quality_scores(
    sequences: dict[str, list[np.ndarray]],
    score_lo: float = 35.0,
    score_hi: float = 50.0,
) -> dict[str, list[float]]:
    """For each movement type, score each rep by L2 proximity to the mean trajectory.

    Returns dict movement_id → list[float] of scores parallel to sequences[movement_id].
    """
    scores: dict[str, list[float]] = {}
    for mov_id, reps in sequences.items():
        stack = np.stack(reps, axis=0)            # [N, seq_len, 66]
        mean_traj = stack.mean(axis=0)            # [seq_len, 66]
        dists = np.sqrt(((stack - mean_traj) ** 2).mean(axis=(1, 2)))  # [N]
        max_d = dists.max()
        if max_d < 1e-8:
            normalised = np.zeros_like(dists)
        else:
            normalised = dists / max_d
        scores[mov_id] = [float(score_hi + (score_lo - score_hi) * n) for n in normalised]
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  default=r"D:\Rehabilation\Segmented Movements\Kinect\Positions")
    parser.add_argument("--out_dir",    default=r"D:\Rehabilation\UIPRMD_processed")
    parser.add_argument("--seq_len",    type=int, default=SEQ_LEN)
    args = parser.parse_args()

    root = Path(args.data_root)
    files = sorted(root.glob("*.txt"))
    print(f"Found {len(files)} position files in {root}")

    # Parse filenames: m{mov}_s{subj}_e{rep}_positions.txt
    records = []
    for f in files:
        parts = f.stem.split("_")   # ["m01","s01","e01","positions"]
        mov_id  = parts[0]          # "m01" … "m10"
        subj_id = parts[1]          # "s01" … "s10"
        rep_id  = parts[2]          # "e01" … "e10"
        records.append((mov_id, subj_id, rep_id, f))

    # Load + resample all sequences
    print("Loading and resampling sequences …")
    seq_by_mov: dict[str, list[np.ndarray]] = {}
    meta_rows = []
    for mov_id, subj_id, rep_id, fpath in records:
        raw = load_positions(str(fpath))
        seq = resample(raw, args.seq_len)          # [seq_len, 66]
        seq_by_mov.setdefault(mov_id, []).append(seq)
        meta_rows.append({"mov_id": mov_id, "subj_id": subj_id, "rep_id": rep_id, "file": fpath.name})

    # Compute quality scores per movement type
    print("Computing quality scores …")
    scores_by_mov = compute_quality_scores(seq_by_mov)

    # Build arrays
    all_x, all_y, all_sid, all_eid = [], [], [], []
    meta_out = []
    mov_ptr: dict[str, int] = {}   # pointer into scores_by_mov for ordering

    for mov_id, subj_id, rep_id, fpath in records:
        idx = mov_ptr.get(mov_id, 0)
        mov_ptr[mov_id] = idx + 1

        mov_int  = int(mov_id[1:]) - 1   # m01→0, m10→9
        subj_int = int(subj_id[1:]) - 1  # s01→0, s10→9

        seq = seq_by_mov[mov_id][idx]     # [seq_len, 66]
        kimore_seq = pad_to_kimore_format(seq)  # [seq_len, 100]
        score = scores_by_mov[mov_id][idx]

        all_x.append(kimore_seq)
        all_y.append(score)
        all_sid.append(SUBJECT_ID_OFFSET + subj_int)
        all_eid.append(EXERCISE_ID_OFFSET + mov_int)

        meta_out.append({
            "mov_id": mov_id, "subj_id": subj_id, "rep_id": rep_id,
            "global_subject_id": SUBJECT_ID_OFFSET + subj_int,
            "exercise_id": EXERCISE_ID_OFFSET + mov_int,
            "score": score,
            "group": "UI-PRMD/Healthy",
        })

    X   = np.concatenate([x.reshape(1, args.seq_len, TOTAL_KIMORE_COLS) for x in all_x], axis=0)
    X   = X.reshape(-1, TOTAL_KIMORE_COLS)               # [N*seq_len, 100]
    Y   = np.array(all_y, dtype=np.float32)
    SID = np.array(all_sid, dtype=np.int32)
    EID = np.array(all_eid, dtype=np.int32)
    META = pd.DataFrame(meta_out)

    n = Y.shape[0]
    print(f"\nUI-PRMD processed:")
    print(f"  Samples: {n}  (X shape: {X.shape})")
    print(f"  Score range: [{Y.min():.2f}, {Y.max():.2f}]  mean={Y.mean():.2f}")
    print(f"  Subjects (global): {SID.min()} – {SID.max()}")
    print(f"  Exercises (global): {EID.min()} – {EID.max()}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X).to_csv(  os.path.join(args.out_dir, "Train_X.csv"),      header=False, index=False)
    pd.DataFrame(Y).to_csv(  os.path.join(args.out_dir, "Train_Y.csv"),      header=False, index=False)
    pd.DataFrame(SID).to_csv(os.path.join(args.out_dir, "subject_ids.csv"),  header=False, index=False)
    pd.DataFrame(EID).to_csv(os.path.join(args.out_dir, "exercise_ids.csv"), header=False, index=False)
    META.to_csv(             os.path.join(args.out_dir, "meta.csv"),          index=False)
    print(f"  Written to: {args.out_dir}")

    # Per-movement score stats
    print("\nPer-movement quality score stats:")
    for mov_id in sorted(scores_by_mov.keys()):
        s = np.array(scores_by_mov[mov_id])
        print(f"  {mov_id}: [{s.min():.2f}, {s.max():.2f}]  mean={s.mean():.2f}  std={s.std():.2f}")


if __name__ == "__main__":
    main()
