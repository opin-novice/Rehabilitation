"""Convert raw KIMORE dataset folder structure into per-exercise Train_X/Y CSVs.

KIMORE folder layout expected:
  KIMORE/
    CG/
      Expert/     NE_ID1..E_ID17
      NotExpert/  NE_ID1..NE_ID27
    GPP/
      BackPain/   B_ID1..B_ID8
      Parkinson/  P_ID1..P_ID16
      Stroke/     S_ID1..S_ID10

Each subject folder contains Es1..Es5 subfolders, each with:
  Raw/JointPosition*.csv   — variable_len x 101 cols (100 real + trailing NaN)
  Label/ClinicalAssessment_*.xlsx — one row per subject, all 5 exercise scores

Output (one directory per exercise):
  <out_dir>/Exercise1/
    Train_X.csv       — (n_samples * seq_len) rows x 100 cols
    Train_Y.csv       — n_samples rows x 1 col  (TS score)
    subject_ids.csv   — n_samples rows x 1 col  (integer subject ID)
    meta.csv          — subject ID, group, exercise, raw_file, score
"""
from __future__ import annotations

import argparse
import glob
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

NUM_JOINTS = 25
COLS_PER_JOINT = 4          # x, y, z, tracking_state
TOTAL_RAW_COLS = NUM_JOINTS * COLS_PER_JOINT  # 100

# Map exercise folder name → column names in ClinicalAssessment Excel
EXERCISE_SCORE_COL = {
    "Es1": "clinical TS Ex#1",
    "Es2": "clinical TS Ex#2",
    "Es3": "clinical TS Ex#3",
    "Es4": "clinical TS Ex#4",
    "Es5": "clinical TS Ex#5",
}
EXERCISE_PO_COL = {
    "Es1": "clinical PO Ex#1",
    "Es2": "clinical PO Ex#2",
    "Es3": "clinical PO Ex#3",
    "Es4": "clinical PO Ex#4",
    "Es5": "clinical PO Ex#5",
}
EXERCISE_CF_COL = {
    "Es1": "clinical CF Ex#1",
    "Es2": "clinical CF Ex#2",
    "Es3": "clinical CF Ex#3",
    "Es4": "clinical CF Ex#4",
    "Es5": "clinical CF Ex#5",
}

# All subject groups in order: (subfolder_path, prefix_used_in_filenames)
GROUPS: list[tuple[str, str]] = [
    (r"CG\Expert",      "E"),
    (r"CG\NotExpert",   "NE"),
    (r"GPP\BackPain",   "B"),
    (r"GPP\Parkinson",  "P"),
    (r"GPP\Stroke",     "S"),
]


@dataclass
class SampleRecord:
    subject_idx: int       # 0-based integer ID across all subjects
    subject_name: str      # e.g. "E_ID1"
    group: str             # e.g. "CG/Expert"
    exercise: str          # e.g. "Es1"
    score: float           # TS = PO + CF
    po_score: float        # position offset component
    cf_score: float        # correctness/fluency component
    sequence: np.ndarray   # shape (seq_len, 100)


def resample_sequence(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """Linearly interpolate each column to exactly seq_len rows."""
    T_orig = arr.shape[0]
    if T_orig == seq_len:
        return arr.astype(np.float32)
    old_t = np.linspace(0.0, 1.0, T_orig)
    new_t = np.linspace(0.0, 1.0, seq_len)
    out = np.zeros((seq_len, arr.shape[1]), dtype=np.float32)
    for c in range(arr.shape[1]):
        out[:, c] = np.interp(new_t, old_t, arr[:, c])
    return out


def load_joint_position(csv_path: str, seq_len: int) -> np.ndarray:
    """Load JointPosition CSV, drop trailing NaN column, resample to seq_len.

    Returns float32 array of shape (seq_len, 100).

    Handles two known KIMORE raw-file quirks:
      - Leading blank lines (skip rows with wrong column count via on_bad_lines='skip')
      - Merged double-rows that produce 201 columns (also skipped)
    """
    df = pd.read_csv(csv_path, header=None, on_bad_lines="skip")

    # Drop trailing NaN column if present (KIMORE CSVs have 101 cols, last is NaN)
    if df.shape[1] == 101:
        df = df.iloc[:, :100]

    assert df.shape[1] == TOTAL_RAW_COLS, (
        f"{csv_path}: expected {TOTAL_RAW_COLS} cols, got {df.shape[1]}"
    )

    arr = df.values.astype(np.float32)
    # Replace any NaN/Inf with 0 (untracked joints have tracking_state != 2)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    resampled = resample_sequence(arr, seq_len)
    assert resampled.shape == (seq_len, TOTAL_RAW_COLS), (
        f"Resample shape error: {resampled.shape}"
    )
    return resampled


def load_clinical_score(xlsx_path: str, exercise_key: str) -> tuple[float, float, float] | None:
    """Extract TS, PO, CF scores for a given exercise from ClinicalAssessment Excel.

    Returns (ts, po, cf) or None if any value is missing.
    """
    ts_col = EXERCISE_SCORE_COL[exercise_key]
    po_col = EXERCISE_PO_COL[exercise_key]
    cf_col = EXERCISE_CF_COL[exercise_key]
    try:
        df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")
        for col in (ts_col, po_col, cf_col):
            if col not in df.columns:
                return None
        row = df.iloc[0]
        ts, po, cf = row[ts_col], row[po_col], row[cf_col]
        if any(pd.isna(v) for v in (ts, po, cf)):
            return None
        return float(ts), float(po), float(cf)
    except Exception as e:
        print(f"  [WARN] Could not read {xlsx_path}: {e}")
        return None


def find_joint_position_csv(raw_dir: str) -> str | None:
    """Find the single JointPosition CSV in a Raw/ directory."""
    matches = glob.glob(os.path.join(raw_dir, "JointPosition*.csv"))
    if not matches:
        return None
    return matches[0]


def find_clinical_assessment_xlsx(label_dir: str) -> str | None:
    matches = glob.glob(os.path.join(label_dir, "ClinicalAssessment_*.xlsx"))
    if not matches:
        return None
    return matches[0]


def collect_all_samples(
    kimore_root: str,
    exercises: list[str],
    seq_len: int,
) -> dict[str, list[SampleRecord]]:
    """Walk the full KIMORE tree and collect SampleRecord objects per exercise."""
    records: dict[str, list[SampleRecord]] = {ex: [] for ex in exercises}
    subject_idx = 0
    total_skipped = 0

    for group_rel, _ in GROUPS:
        group_path = os.path.join(kimore_root, group_rel)
        if not os.path.isdir(group_path):
            print(f"[WARN] Group folder not found: {group_path} — skipping.")
            continue

        subjects = sorted(os.listdir(group_path))
        for subj_name in subjects:
            subj_path = os.path.join(group_path, subj_name)
            if not os.path.isdir(subj_path):
                continue

            for ex in exercises:
                ex_path = os.path.join(subj_path, ex)
                raw_dir = os.path.join(ex_path, "Raw")
                label_dir = os.path.join(ex_path, "Label")

                if not os.path.isdir(raw_dir):
                    total_skipped += 1
                    continue

                jp_csv = find_joint_position_csv(raw_dir)
                if jp_csv is None:
                    print(f"  [WARN] No JointPosition CSV: {raw_dir}")
                    total_skipped += 1
                    continue

                ca_xlsx = find_clinical_assessment_xlsx(label_dir)
                if ca_xlsx is None:
                    print(f"  [WARN] No ClinicalAssessment XLSX: {label_dir}")
                    total_skipped += 1
                    continue

                scores = load_clinical_score(ca_xlsx, ex)
                if scores is None:
                    print(f"  [WARN] Missing score for {subj_name}/{ex} in {ca_xlsx}")
                    total_skipped += 1
                    continue
                ts, po, cf = scores

                try:
                    seq = load_joint_position(jp_csv, seq_len)
                except Exception as e:
                    print(f"  [ERROR] {jp_csv}: {e}")
                    total_skipped += 1
                    continue

                records[ex].append(SampleRecord(
                    subject_idx=subject_idx,
                    subject_name=subj_name,
                    group=group_rel.replace("\\", "/"),
                    exercise=ex,
                    score=ts,
                    po_score=po,
                    cf_score=cf,
                    sequence=seq,
                ))

            subject_idx += 1

    print(f"\nCollection complete. Skipped {total_skipped} subject/exercise pairs.")
    return records


def write_exercise_dataset(
    records: list[SampleRecord],
    out_dir: str,
    seq_len: int,
    exercise: str,
) -> None:
    if not records:
        print(f"  [WARN] No records for {exercise} — skipping.")
        return

    n = len(records)
    x_array = np.concatenate([r.sequence for r in records], axis=0)  # [n*seq_len, 100]
    y_array = np.array([r.score for r in records], dtype=np.float32)
    sid_array = np.array([r.subject_idx for r in records], dtype=np.int32)

    assert x_array.shape == (n * seq_len, TOTAL_RAW_COLS), (
        f"x_array shape error: {x_array.shape}"
    )
    assert y_array.shape == (n,)
    assert sid_array.shape == (n,)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    pd.DataFrame(x_array).to_csv(
        os.path.join(out_dir, "Train_X.csv"), header=False, index=False
    )
    pd.DataFrame(y_array).to_csv(
        os.path.join(out_dir, "Train_Y.csv"), header=False, index=False
    )
    pd.DataFrame(sid_array).to_csv(
        os.path.join(out_dir, "subject_ids.csv"), header=False, index=False
    )

    # Save metadata for inspection
    meta = pd.DataFrame([{
        "subject_idx": r.subject_idx,
        "subject_name": r.subject_name,
        "group": r.group,
        "exercise": r.exercise,
        "score": r.score,
        "po_score": r.po_score,
        "cf_score": r.cf_score,
    } for r in records])
    meta.to_csv(os.path.join(out_dir, "meta.csv"), index=False)

    print(
        f"  {exercise}: {n} samples | "
        f"score [{y_array.min():.2f}, {y_array.max():.2f}] mean={y_array.mean():.2f} | "
        f"Train_X shape={x_array.shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw KIMORE dataset to Train_X/Y CSVs."
    )
    parser.add_argument(
        "--kimore_root",
        default=r"D:\Rehabilation\KIMORE",
        help="Root folder containing CG/ and GPP/ subfolders.",
    )
    parser.add_argument(
        "--out_dir",
        default=r"D:\Rehabilation\KIMORE_processed",
        help="Output root; one subfolder per exercise will be created.",
    )
    parser.add_argument(
        "--seq_len", type=int, default=100,
        help="Target sequence length (resample all recordings to this).",
    )
    parser.add_argument(
        "--exercises", nargs="+",
        default=["Es1", "Es2", "Es3", "Es4", "Es5"],
        help="Which exercises to process.",
    )
    args = parser.parse_args()

    print(f"KIMORE root : {args.kimore_root}")
    print(f"Output dir  : {args.out_dir}")
    print(f"seq_len     : {args.seq_len}")
    print(f"Exercises   : {args.exercises}")
    print()

    records = collect_all_samples(args.kimore_root, args.exercises, args.seq_len)

    print("\nWriting per-exercise datasets:")
    for ex in args.exercises:
        ex_out = os.path.join(args.out_dir, f"Exercise{ex[2]}")  # Es1 -> Exercise1
        write_exercise_dataset(records[ex], ex_out, args.seq_len, ex)

    print(f"\nDone. Processed datasets in: {args.out_dir}")


if __name__ == "__main__":
    main()
