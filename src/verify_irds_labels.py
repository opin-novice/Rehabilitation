"""Empirical verification that IRDS filename field m{EE} = EXERCISE TYPE, not patient group.

Method: If m=exercise, then the SAME 10 subjects perform every exercise, so a fixed subject's
bone lengths must be near-constant across m-groups (variation = Kinect tracking noise ~5mm).
If m=patient-group, different people occupy each m-group, so bone-length variation across
m-groups would match between-subject variation (~15mm).

Run:  python src/verify_irds_labels.py --irds_dir "Segmented Movements/Kinect/Positions"
"""
from __future__ import annotations
import argparse, os
import numpy as np


def mean_bones(path: str) -> tuple[float, float, float]:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    data = data.reshape(-1, 22, 3)
    spine = np.linalg.norm(data[:, 1, :] - data[:, 0, :], axis=1).mean()
    ll    = np.linalg.norm(data[:, 9, :] - data[:, 8, :], axis=1).mean()
    rl    = np.linalg.norm(data[:, 13, :] - data[:, 12, :], axis=1).mean()
    return round(float(spine), 1), round(float(ll), 1), round(float(rl), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--irds_dir", default=r"Segmented Movements\Kinect\Positions")
    args = ap.parse_args()

    vals_s01 = []
    for m in range(1, 11):
        p = os.path.join(args.irds_dir, f"m{m:02d}_s01_e01_positions.txt")
        if os.path.exists(p):
            vals_s01.append(mean_bones(p))
    vals_diff = []
    for s in range(1, 11):
        p = os.path.join(args.irds_dir, f"m01_s{s:02d}_e01_positions.txt")
        if os.path.exists(p):
            vals_diff.append(mean_bones(p))

    sp_across_m = np.std([v[0] for v in vals_s01])
    sp_across_s = np.std([v[0] for v in vals_diff])
    print(f"Std(spine) across m-groups for s=01 : {sp_across_m:.1f} mm")
    print(f"Std(spine) across subjects in m=01  : {sp_across_s:.1f} mm")
    ratio = sp_across_m / sp_across_s if sp_across_s > 0 else float("nan")
    print(f"Ratio (m-group var / subject var): {ratio:.2f}")
    if ratio < 0.6:
        print("RESULT -> m = EXERCISE TYPE (same person across m-groups; variation is tracking noise)")
    elif ratio > 0.85:
        print("RESULT -> m = PATIENT GROUP (different people per m-group)")
    else:
        print("RESULT -> AMBIGUOUS")


if __name__ == "__main__":
    main()
