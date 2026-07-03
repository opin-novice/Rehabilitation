"""Generate synthetic KIMORE-style CSV data for pipeline smoke tests.

Produces:
  <out_dir>/Train_X.csv  — (n_samples * seq_len) rows × 100 columns
  <out_dir>/Train_Y.csv  — n_samples rows × 1 column
  <out_dir>/subject_ids.csv — n_samples rows × 1 column (integer IDs)

The learnable signal: the quality score is the mean absolute Z-displacement
of joint 0 (spine base) across the sequence, with added noise. This ensures
RMSE decreases provably as the model learns — if the loss stays flat, the
data pipeline is broken.

Column layout: 25 joints × 4 columns each = 100 columns.
  Columns 0,1,2 of each block = x, y, z.
  Column 3 = orientation (zero-filled, matching Kinect export convention).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Match JOINT_STARTS layout from rehab_dataset.py
NUM_JOINTS = 25
NUM_CHANNELS = 3
COLS_PER_JOINT = 4          # x, y, z, orientation
TOTAL_COLUMNS = NUM_JOINTS * COLS_PER_JOINT  # 100


def _make_one_sample(
    seq_len: int,
    rng: np.random.Generator,
    amplitude: float,
) -> tuple[np.ndarray, float]:
    """Generate one skeleton sequence and its quality score.

    Returns:
        raw: float32 array (seq_len, TOTAL_COLUMNS)
        score: float scalar in approximately [0, 1]
    """
    raw = np.zeros((seq_len, TOTAL_COLUMNS), dtype=np.float32)

    # Base skeleton: 25 joints with anatomically plausible offsets
    base_positions = rng.standard_normal((NUM_JOINTS, NUM_CHANNELS)).astype(np.float32)

    for t in range(seq_len):
        phase = 2 * np.pi * t / seq_len
        displacement = amplitude * np.sin(phase)
        for j in range(NUM_JOINTS):
            col_start = j * COLS_PER_JOINT
            raw[t, col_start]     = base_positions[j, 0] + displacement * 0.1 * (j % 3)
            raw[t, col_start + 1] = base_positions[j, 1] + displacement * 0.1 * ((j + 1) % 3)
            raw[t, col_start + 2] = base_positions[j, 2] + displacement       # primary z-motion
            # col col_start+3 stays zero (orientation)

    # Score = mean absolute z-displacement of joint 0 (spine base), normalized
    z_joint0 = raw[:, 2]  # column index 2 = z of joint 0
    raw_score = float(np.mean(np.abs(z_joint0 - z_joint0.mean())))
    noise = rng.normal(0, 0.02)
    score = float(np.clip(raw_score / (amplitude + 1e-6) + noise, 0.0, 1.0))
    return raw, score


def generate_dummy_dataset(
    out_dir: str,
    n_subjects: int = 10,
    reps_per_subject: int = 5,
    seq_len: int = 100,
    seed: int = 42,
) -> None:
    n_samples = n_subjects * reps_per_subject
    rng = np.random.default_rng(seed)

    all_x: list[np.ndarray] = []
    all_y: list[float] = []
    all_subject_ids: list[int] = []

    for subject_id in range(n_subjects):
        # Each subject has a characteristic amplitude (their "movement quality")
        subject_amplitude = rng.uniform(0.5, 2.0)

        for _ in range(reps_per_subject):
            seq, score = _make_one_sample(seq_len, rng, subject_amplitude)
            assert seq.shape == (seq_len, TOTAL_COLUMNS), (
                f"seq shape error: {seq.shape}"
            )
            all_x.append(seq)
            all_y.append(score)
            all_subject_ids.append(subject_id)

    # Stack and save
    x_array = np.concatenate(all_x, axis=0)  # [n_samples * seq_len, 100]
    y_array = np.array(all_y, dtype=np.float32).reshape(-1, 1)
    sid_array = np.array(all_subject_ids, dtype=np.int32).reshape(-1, 1)

    assert x_array.shape == (n_samples * seq_len, TOTAL_COLUMNS), (
        f"x_array shape error: {x_array.shape}"
    )
    assert y_array.shape == (n_samples, 1)
    assert sid_array.shape == (n_samples, 1)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    pd.DataFrame(x_array).to_csv(os.path.join(out_dir, "Train_X.csv"), header=False, index=False)
    pd.DataFrame(y_array).to_csv(os.path.join(out_dir, "Train_Y.csv"), header=False, index=False)
    pd.DataFrame(sid_array).to_csv(
        os.path.join(out_dir, "subject_ids.csv"), header=False, index=False
    )

    print(f"[dummy] Wrote {out_dir}/Train_X.csv  shape={x_array.shape}")
    print(f"[dummy] Wrote {out_dir}/Train_Y.csv  shape={y_array.shape}")
    print(f"[dummy] Wrote {out_dir}/subject_ids.csv  shape={sid_array.shape}")
    print(f"[dummy] Score range: [{y_array.min():.3f}, {y_array.max():.3f}]")
    print(f"[dummy] {n_subjects} subjects × {reps_per_subject} reps = {n_samples} samples")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic skeleton data for smoke tests.")
    parser.add_argument("--out_dir", default="dummy_data/Exercise1")
    parser.add_argument("--n_subjects", type=int, default=10,
                        help="Number of synthetic subjects.")
    parser.add_argument("--reps_per_subject", type=int, default=5,
                        help="Repetitions per subject (matches KIMORE convention).")
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dummy_dataset(
        out_dir=args.out_dir,
        n_subjects=args.n_subjects,
        reps_per_subject=args.reps_per_subject,
        seq_len=args.seq_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
