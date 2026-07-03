"""V4 - UI-PRMD correct/incorrect labeled testbed (second external validity corpus).

Replicates the REHAB24-6 validity testbed (Task V1/V2) on a fully independent labeled
set so the reliability!=validity finding can be checked cross-dataset (Task V4 / W4).

UI-PRMD (Vakanski et al.) ships paired CORRECT and INCORRECT segmented movements:
  Segmented Movements/Kinect/Positions/m{MM}_s{SS}_e{RR}_positions.txt            -> correct (label 1)
  Incorrect Segmented Movements/Kinect/Positions/m{MM}_s{SS}_e{RR}_positions_inc.txt -> incorrect (label 0)
  Each file: [T, 66] = 22 Kinect joints x 3 xyz (cm). 10 movements x 10 subjects x 10 reps.

The repo bundles only the CORRECT set (all-healthy). The INCORRECT set must be downloaded
to build a binary testbed; until then this loader prints a [SKIP] with the exact command.
UI-PRMD's 22-joint Kinect skeleton is padded 22->25 exactly as src/irds_eval.load_irds_sequence,
so the KIMORE-trained models consume it with the identical zero-shot protocol.

Output (same schema as load_rehab246, so validity_eval consumes it unchanged):
  outputs/validity_uiprmd/uiprmd_manifest.csv   (rep_uid, exercise_id, subject_id, correct_label)
  outputs/validity_uiprmd/uiprmd_sequences.npy  (N, SEQ_LEN, 25, 3)

Then run the full V4 replication with:
  python src/validity_eval.py --infer --tag uiprmd \
      --manifest outputs/validity_uiprmd/uiprmd_manifest.csv \
      --seqs outputs/validity_uiprmd/uiprmd_sequences.npy

Usage:
  python src/load_uiprmd_validity.py --build
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/
from constants import SEQ_LEN, NUM_JOINTS, NUM_CHANNELS

UIPRMD_JOINTS = 22
CORRECT_DIR   = "Segmented Movements/Kinect/Positions"
INCORRECT_DIR = "Incorrect Segmented Movements/Kinect/Positions"
OUT_DIR       = "outputs/validity_uiprmd"


def _resample(seq: np.ndarray, target: int) -> np.ndarray:
    T = seq.shape[0]
    if T == target:
        return seq
    idx = np.linspace(0, T - 1, target)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    frac = (idx - lo)[:, None]
    return seq[lo] * (1 - frac) + seq[hi] * frac


def _load_seq(fpath: str) -> np.ndarray | None:
    """UI-PRMD positions file -> (SEQ_LEN, 25, 3); pads 22->25 joints (zeros), like IRDS."""
    raw = np.loadtxt(fpath, delimiter=",", dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != UIPRMD_JOINTS * 3:
        return None
    T = raw.shape[0]
    xyz25 = np.zeros((T, NUM_JOINTS, NUM_CHANNELS), dtype=np.float32)
    xyz25[:, :UIPRMD_JOINTS, :] = raw.reshape(T, UIPRMD_JOINTS, NUM_CHANNELS)
    return _resample(xyz25.reshape(T, -1), SEQ_LEN).reshape(SEQ_LEN, NUM_JOINTS, NUM_CHANNELS)


def _collect(dir_path: str, label: int) -> tuple[list[dict], list[np.ndarray]]:
    man, seqs = [], []
    for fname in sorted(os.listdir(dir_path)):
        # Correct set: m01_s01_e01_positions.txt ; incorrect set: m01_s01_e01_positions_inc.txt
        if not fname.endswith(".txt") or "_positions" not in fname:
            continue
        stem = fname.replace("_positions_inc.txt", "").replace("_positions.txt", "")
        parts = stem.split("_")
        if len(parts) != 3:
            continue
        try:
            mid, sid, rid = (int(p[1:]) for p in parts)
        except (IndexError, ValueError):
            continue
        s = _load_seq(os.path.join(dir_path, fname))
        if s is None:
            continue
        man.append({"rep_uid": f"uiprmd_m{mid:02d}_s{sid:02d}_e{rid:02d}_c{label}",
                    "exercise_id": mid, "subject_id": sid, "correct_label": label})
        seqs.append(s.astype(np.float32))
    return man, seqs


def build() -> dict:
    if not os.path.isdir(INCORRECT_DIR):
        print(f"[SKIP] Missing {INCORRECT_DIR}/ (UI-PRMD incorrect set not bundled).")
        print("       Download UI-PRMD and place the incorrect Kinect positions there:")
        print('         https://webpages.uidaho.edu/ui-prmd/  (Incorrect Segmented Movements)')
        print("       Then re-run: python src/load_uiprmd_validity.py --build")
        return {}
    if not os.path.isdir(CORRECT_DIR):
        print(f"[SKIP] Missing {CORRECT_DIR}/ (UI-PRMD correct set).")
        return {}

    man_c, seq_c = _collect(CORRECT_DIR, 1)
    man_i, seq_i = _collect(INCORRECT_DIR, 0)
    manifest = man_c + man_i
    seqs = seq_c + seq_i
    if not seqs:
        print("[SKIP] No UI-PRMD sequences parsed.")
        return {}

    os.makedirs(OUT_DIR, exist_ok=True)
    man = pd.DataFrame(manifest)
    man_path = os.path.join(OUT_DIR, "uiprmd_manifest.csv")
    seq_path = os.path.join(OUT_DIR, "uiprmd_sequences.npy")
    man.to_csv(man_path, index=False)
    np.save(seq_path, np.stack(seqs, axis=0))

    bal = man["correct_label"].value_counts().to_dict()
    print(f"UI-PRMD validity testbed: {len(man)} reps  (correct={bal.get(1,0)} / incorrect={bal.get(0,0)})")
    print(f"  exercises={sorted(man.exercise_id.unique())}  subjects={sorted(man.subject_id.unique())}")
    print(f"  -> {man_path}")
    print(f"  -> {seq_path}  shape={np.stack(seqs).shape}")
    print("  Next: python src/validity_eval.py --infer --tag uiprmd \\")
    print(f"          --manifest {man_path} --seqs {seq_path}")
    return {"n": len(man), "balance": bal}


def main() -> None:
    ap = argparse.ArgumentParser(description="V4 UI-PRMD correct/incorrect validity testbed")
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    build()


if __name__ == "__main__":
    main()
