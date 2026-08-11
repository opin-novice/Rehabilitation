"""Reviewer round-3, B1 (Q2): naive-baseline sensitivity to joint padding + normalization.

The reviewer asks whether the naive kinematic baseline is affected by (a) zero-padded
missing joints and (b) per-sequence z-score normalization. We report three variants
per corpus:
  raw_all25   : path/speed over all 25 canonical joints, raw coordinates (the value
                printed in Table II).
  shared_only : same, but restricted to genuinely distinct joints -- excludes
                UI-PRMD's 3 zero-padded slots and REHAB246's 4 duplicated permutation
                targets (thumb=wrist).
  zscored     : per-sequence per-coordinate z-score first, then path/speed.

Key analytic facts this experiment confirms numerically:
  * UI-PRMD joints 22-24 are all-zero -> they contribute exactly 0 to a difference-
    based path/speed feature, so raw_all25 == shared_only for UI-PRMD.
  * REHAB246 duplicated joints double-count wrist motion; dropping them is the
    honest shared-joint comparison.

Run:  python src/reviewer/reviewer_round3_b1_naive.py
Out:  outputs/reviewer_round3/b1_naive_sensitivity.{json,md}
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ root (file now in src/reviewer/)
from constants import NUM_JOINTS  # noqa: E402
from selfsup.naive_baseline import naive_auroc  # noqa: E402
from load_rehab246 import KINECT_FROM_REHAB26  # noqa: E402

OUT_DIR = "outputs/reviewer_round3"


def _load(seq_path: str, man_path: str, label_col: str = "correct_label"):
    if not (os.path.exists(seq_path) and os.path.exists(man_path)):
        return None, None
    X = np.load(seq_path).astype(np.float32)
    man = pd.read_csv(man_path)
    y = man[label_col].values.astype(int)
    return X, y


def _rehab246_valid_mask() -> np.ndarray:
    """True for canonical joints whose source index is not a duplicate of an
    earlier canonical slot (drops the 4 thumb=wrist / hand=wrist duplicates)."""
    seen: set[int] = set()
    mask = np.zeros(NUM_JOINTS, dtype=bool)
    for canon_i, src_i in enumerate(KINECT_FROM_REHAB26):
        if src_i not in seen:
            mask[canon_i] = True
            seen.add(src_i)
    return mask


def _uiprmd_valid_mask() -> np.ndarray:
    """UI-PRMD is padded 22->25; canonical joints 22,23,24 are zero-filled."""
    mask = np.ones(NUM_JOINTS, dtype=bool)
    mask[22:25] = False
    return mask


def run() -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    corpora = {
        "REHAB246": (
            _load("outputs/validity/rehab246_sequences.npy",
                  "outputs/validity/rehab246_manifest.csv"),
            _rehab246_valid_mask(),
        ),
        "UI-PRMD": (
            _load("outputs/validity_uiprmd/uiprmd_sequences.npy",
                  "outputs/validity_uiprmd/uiprmd_manifest.csv"),
            _uiprmd_valid_mask(),
        ),
    }

    results = {}
    for name, ((X, y), mask) in corpora.items():
        if X is None:
            print(f"[SKIP] {name}: cached sequences/manifest not found")
            continue
        n_valid = int(mask.sum())
        results[name] = {
            "n": int(len(X)),
            "n_pos": int((y == 1).sum()),
            "n_valid_joints": n_valid,
            "raw_all25": naive_auroc(X, y),
            "shared_only": naive_auroc(X, y, joint_mask=mask),
            "zscored_all25": naive_auroc(X, y, zscore=True),
        }
        r = results[name]
        print(f"{name}: n={r['n']} valid_joints={n_valid}/25  "
              f"raw={r['raw_all25']:.3f}  shared={r['shared_only']:.3f}  "
              f"zscored={r['zscored_all25']:.3f}")

    with open(os.path.join(OUT_DIR, "b1_naive_sensitivity.json"), "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "| Corpus | N | Valid joints | Naive AUROC (raw, all 25) | Naive AUROC (shared joints only) | Naive AUROC (z-scored) |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['n']} | {r['n_valid_joints']}/25 | "
            f"{r['raw_all25']:.3f} | {r['shared_only']:.3f} | {r['zscored_all25']:.3f} |"
        )
    md = "\n".join(lines) + "\n"
    with open(os.path.join(OUT_DIR, "b1_naive_sensitivity.md"), "w") as f:
        f.write(md)
    print("\n" + md)
    return results


if __name__ == "__main__":
    run()
