"""
Skeleton CSV regression dataset for KIMORE / UI-PRMD rehabilitation scoring.

Critical fix over original repo: split by subject group (GroupShuffleSplit),
not by random sample index, to prevent subject-level data leakage.

Expected CSV layout (original repo's Data_Proc/data_processing.py convention):
  Train_X.csv  —  N*seq_len rows × ≥100 columns.
                   Each joint occupies 4 columns (x, y, z, extra); 25 joints → 100 cols.
                   JOINT_STARTS picks column 0 of each joint block.
  Train_Y.csv  —  N rows × 1 column (one clinical/quality score per sample).

Optional:
  subject_ids.csv — N rows × 1 column (integer subject ID per sample).
                    If absent, IDs are inferred as sample_index // reps_per_subject.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset

# Original repo's joint stride: each joint occupies 4 raw columns (x, y, z, orientation).
# We keep only x, y, z (columns +0, +1, +2 of each block).
JOINT_STARTS: list[int] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36,
    40, 44, 48, 52, 56, 60, 64, 68, 72, 76,
    80, 84, 88, 92, 96,
]
NUM_JOINTS: int = len(JOINT_STARTS)   # 25
NUM_CHANNELS: int = 3                  # x, y, z


@dataclass
class ScalerBundle:
    """Carries fitted scalers so they can be serialized with the model checkpoint."""
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    po_scaler: Optional[StandardScaler] = None   # only set in multitask mode
    cf_scaler: Optional[StandardScaler] = None   # only set in multitask mode


def _select_xyz_columns(raw: np.ndarray) -> np.ndarray:
    """Pick x, y, z columns for each joint; raises if columns are missing."""
    cols: list[int] = []
    for start in JOINT_STARTS:
        cols.extend([start, start + 1, start + 2])

    required = max(cols)
    if raw.shape[1] <= required:
        raise ValueError(
            f"Train_X has {raw.shape[1]} columns but needs column index {required}. "
            "If your data has 25*3=75 columns without the extra orientation column, "
            "set JOINT_STARTS = list(range(0, 75, 3)) in rehab_dataset.py."
        )
    return raw[:, cols]


def load_csv_arrays(
    data_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw Train_X.csv and Train_Y.csv from *data_dir*."""
    x_path = os.path.join(data_dir, "Train_X.csv")
    y_path = os.path.join(data_dir, "Train_Y.csv")
    for p in (x_path, y_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    x = pd.read_csv(x_path, header=None).values.astype(np.float32)
    y = pd.read_csv(y_path, header=None).values.squeeze().astype(np.float32)
    return x, y


def infer_subject_ids(
    n_samples: int,
    reps_per_subject: int,
    subject_ids_csv: Optional[str] = None,
) -> np.ndarray:
    """Return integer subject IDs (length n_samples).

    If *subject_ids_csv* is provided, load it. Otherwise, assume that every
    *reps_per_subject* consecutive samples belong to the same subject — which
    matches KIMORE's data organisation (5 reps per subject, per exercise).
    """
    if subject_ids_csv and os.path.exists(subject_ids_csv):
        ids = pd.read_csv(subject_ids_csv, header=None).values.squeeze().astype(int)
        if len(ids) != n_samples:
            raise ValueError(
                f"subject_ids.csv has {len(ids)} rows but dataset has {n_samples} samples."
            )
        return ids
    return np.arange(n_samples) // reps_per_subject


class SkeletonRegressionDataset(Dataset):
    """Torch Dataset wrapping pre-split, pre-scaled skeleton arrays.

    Args:
        x: float32 array, shape (N, seq_len, NUM_JOINTS, NUM_CHANNELS)
        y: float32 array, shape (N,) or (N, 1)
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        assert x.ndim == 4, f"x must be 4-D (N, T, J, C), got shape {x.shape}"
        assert x.shape[2] == NUM_JOINTS, f"Expected {NUM_JOINTS} joints, got {x.shape[2]}"
        assert x.shape[3] == NUM_CHANNELS, f"Expected {NUM_CHANNELS} channels, got {x.shape[3]}"
        self._x = torch.from_numpy(x)              # [N, T, J, C]
        self._y = torch.from_numpy(y.reshape(-1, 1))  # [N, 1]

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._x[idx], self._y[idx]


def make_dataloaders(
    data_dir: str,
    seq_len: int = 100,
    batch_size: int = 8,
    test_size: float = 0.2,
    reps_per_subject: int = 5,
    subject_ids_csv: Optional[str] = None,
    seed: int = 145,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, ScalerBundle]:
    """Build train/val DataLoaders with subject-grouped splits.

    Returns:
        train_loader, val_loader, scaler_bundle
        The ScalerBundle must be saved with the model checkpoint for
        correct inverse-transform during evaluation.
    """
    raw_x, raw_y = load_csv_arrays(data_dir)

    # --- Validate dimensions ---
    if raw_x.shape[0] % seq_len != 0:
        raise ValueError(
            f"Train_X has {raw_x.shape[0]} rows, not divisible by seq_len={seq_len}. "
            "Check segmentation or adjust --seq_len."
        )
    n_samples = raw_x.shape[0] // seq_len
    if raw_y.shape[0] != n_samples:
        raise ValueError(
            f"Train_Y has {raw_y.shape[0]} rows but Train_X implies {n_samples} samples."
        )

    # --- Extract xyz columns: (N*seq_len, J*C) ---
    xyz = _select_xyz_columns(raw_x)
    assert xyz.shape == (n_samples * seq_len, NUM_JOINTS * NUM_CHANNELS), (
        f"xyz shape mismatch: expected {(n_samples * seq_len, NUM_JOINTS * NUM_CHANNELS)}, "
        f"got {xyz.shape}"
    )

    # --- Subject-aware split (KEY FIX: prevents leakage) ---
    sample_ids = np.arange(n_samples)
    subject_ids = infer_subject_ids(n_samples, reps_per_subject, subject_ids_csv)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(sample_ids, groups=subject_ids))

    n_train_subjects = len(np.unique(subject_ids[train_idx]))
    n_val_subjects = len(np.unique(subject_ids[val_idx]))

    def slice_rows(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rows = np.concatenate([np.arange(i * seq_len, (i + 1) * seq_len) for i in idx])
        return xyz[rows], raw_y[idx]

    train_xyz, train_y = slice_rows(train_idx)
    val_xyz, val_y = slice_rows(val_idx)

    # --- Fit scalers on TRAIN only ---
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    train_xyz_scaled = x_scaler.fit_transform(train_xyz).astype(np.float32)
    val_xyz_scaled = x_scaler.transform(val_xyz).astype(np.float32)

    train_y_scaled = y_scaler.fit_transform(train_y.reshape(-1, 1)).reshape(-1).astype(np.float32)
    val_y_scaled = y_scaler.transform(val_y.reshape(-1, 1)).reshape(-1).astype(np.float32)

    # --- Reshape to (N, T, J, C) ---
    n_train, n_val = len(train_idx), len(val_idx)

    train_x_4d = train_xyz_scaled.reshape(n_train, seq_len, NUM_JOINTS, NUM_CHANNELS)
    val_x_4d = val_xyz_scaled.reshape(n_val, seq_len, NUM_JOINTS, NUM_CHANNELS)

    assert train_x_4d.shape == (n_train, seq_len, NUM_JOINTS, NUM_CHANNELS), (
        f"train_x_4d shape error: {train_x_4d.shape}"
    )
    assert val_x_4d.shape == (n_val, seq_len, NUM_JOINTS, NUM_CHANNELS), (
        f"val_x_4d shape error: {val_x_4d.shape}"
    )

    train_ds = SkeletonRegressionDataset(train_x_4d, train_y_scaled)
    val_ds = SkeletonRegressionDataset(val_x_4d, val_y_scaled)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )

    bundle = ScalerBundle(x_scaler=x_scaler, y_scaler=y_scaler)

    print(
        f"[dataset] {n_samples} samples | "
        f"train={n_train} ({n_train_subjects} subjects) | "
        f"val={n_val} ({n_val_subjects} subjects)"
    )
    return train_loader, val_loader, bundle
