"""V4 - UI-PRMD correct/incorrect labeled testbed (second external validity corpus).

Replicates the REHAB24-6 validity testbed (Task V1/V2) on a fully independent labeled
set so the reliability!=validity finding can be checked cross-dataset (Task V4 / W4).

UI-PRMD (Vakanski et al.) ships paired CORRECT and INCORRECT segmented movements:
  Segmented Movements/Kinect/Positions/m{MM}_s{SS}_e{RR}_positions.txt            -> correct (label 1)
  Incorrect Segmented Movements/Kinect/Positions/m{MM}_s{SS}_e{RR}_positions_inc.txt -> incorrect (label 0)
  Each file: [T, 66] = 22 Kinect joints x 3. 10 movements x 10 subjects x 10 reps.

GEOMETRY WARNING -- this header used to claim those 66 numbers were "xyz (cm)" world
coordinates. They are not. They are per-frame PARENT-RELATIVE BONE OFFSETS (joint 0 is the
absolute root); 19 of 66 slots are constant, so read literally the skeleton is near-static
and only the root translates (docs/worklog_2026-07-29.md §2). World positions require
forward kinematics against the matching Angles/ file -- see src/uiprmd_fk.py, which
transcribes the convention from the authors' own Animation.m.

Both geometries are buildable: --geometry fk (correct, default, -> outputs/validity_uiprmd)
and --geometry raw (the original bug, -> outputs/validity_uiprmd_raw) so the original V4
result can be reproduced and compared rather than merely discarded.
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
from uiprmd_fk import forward_kinematics

UIPRMD_JOINTS = 22
CORRECT_DIR   = "Segmented Movements/Kinect/Positions"
INCORRECT_DIR = "Incorrect Segmented Movements/Kinect/Positions"
CORRECT_ANG   = "Segmented Movements/Kinect/Angles"
INCORRECT_ANG = "Incorrect Segmented Movements/Kinect/Angles"
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


def _load_seq(fpath: str, ang_path: str | None = None,
              geometry: str = "fk", fk_variant: str = "reference") -> np.ndarray | None:
    """UI-PRMD sequence -> (SEQ_LEN, 25, 3); pads 22->25 joints (zeros), like IRDS.

    geometry="fk"  : run forward kinematics on (offsets, angles) to recover true world
                     coordinates. This is the correct reading of the files.
    geometry="raw" : treat the positions file as if it were world coordinates. This is
                     WRONG -- the files are parent-relative bone offsets, so 19/66 slots
                     are constant and the skeleton barely moves (worklog 2026-07-29 §2).
                     Retained only to reproduce the original V4 run for comparison.
    """
    raw = np.loadtxt(fpath, delimiter=",", dtype=np.float64, ndmin=2)
    if raw.ndim != 2 or raw.shape[1] != UIPRMD_JOINTS * 3:
        return None

    if geometry == "fk":
        if ang_path is None or not os.path.isfile(ang_path):
            return None
        ang = np.loadtxt(ang_path, delimiter=",", dtype=np.float64, ndmin=2)
        if ang.shape[1] != UIPRMD_JOINTS * 3:
            return None
        T = min(len(raw), len(ang))
        if T < 2:
            return None
        xyz = forward_kinematics(raw[:T].reshape(T, UIPRMD_JOINTS, 3),
                                 ang[:T].reshape(T, UIPRMD_JOINTS, 3),
                                 variant=fk_variant)
    elif geometry == "raw":
        T = raw.shape[0]
        xyz = raw.reshape(T, UIPRMD_JOINTS, 3)
    else:
        raise ValueError(f"unknown geometry {geometry!r}")

    xyz25 = np.zeros((T, NUM_JOINTS, NUM_CHANNELS), dtype=np.float32)
    xyz25[:, :UIPRMD_JOINTS, :] = xyz.astype(np.float32)
    return _resample(xyz25.reshape(T, -1), SEQ_LEN).reshape(SEQ_LEN, NUM_JOINTS, NUM_CHANNELS)


def _collect(dir_path: str, label: int, ang_dir: str | None = None,
             geometry: str = "fk", fk_variant: str = "reference"
             ) -> tuple[list[dict], list[np.ndarray]]:
    man, seqs, skipped = [], [], []
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
        ang_path = None
        if ang_dir is not None:
            # positions -> angles filename: m01_s01_e01_positions[_inc].txt -> ..._angles[_inc].txt
            ang_path = os.path.join(ang_dir, fname.replace("_positions", "_angles"))
        s = _load_seq(os.path.join(dir_path, fname), ang_path, geometry, fk_variant)
        if s is None:
            skipped.append(fname)
            continue
        man.append({"rep_uid": f"uiprmd_m{mid:02d}_s{sid:02d}_e{rid:02d}_c{label}",
                    "exercise_id": mid, "subject_id": sid, "correct_label": label})
        seqs.append(s.astype(np.float32))
    if skipped:
        print(f"  [{os.path.basename(dir_path)}] skipped {len(skipped)}: {skipped[:4]}"
              f"{' ...' if len(skipped) > 4 else ''}")
    return man, seqs


def build(geometry: str = "fk", fk_variant: str = "reference") -> dict:
    if not os.path.isdir(INCORRECT_DIR):
        print(f"[SKIP] Missing {INCORRECT_DIR}/ (UI-PRMD incorrect set not bundled).")
        print("       Download UI-PRMD and place the incorrect Kinect positions there:")
        print('         https://webpages.uidaho.edu/ui-prmd/  (Incorrect Segmented Movements)')
        print("       Then re-run: python src/load_uiprmd_validity.py --build")
        return {}
    if not os.path.isdir(CORRECT_DIR):
        print(f"[SKIP] Missing {CORRECT_DIR}/ (UI-PRMD correct set).")
        return {}

    man_c, seq_c = _collect(CORRECT_DIR, 1, CORRECT_ANG, geometry, fk_variant)
    man_i, seq_i = _collect(INCORRECT_DIR, 0, INCORRECT_ANG, geometry, fk_variant)
    manifest = man_c + man_i
    seqs = seq_c + seq_i
    if not seqs:
        print("[SKIP] No UI-PRMD sequences parsed.")
        return {}

    suffix = "" if (geometry == "fk" and fk_variant == "reference") else (
        f"_{geometry}" if geometry != "fk" else f"_{fk_variant}")
    out_dir = f"{OUT_DIR}{suffix}"
    os.makedirs(out_dir, exist_ok=True)
    man = pd.DataFrame(manifest)
    man_path = os.path.join(out_dir, "uiprmd_manifest.csv")
    seq_path = os.path.join(out_dir, "uiprmd_sequences.npy")
    man.to_csv(man_path, index=False)
    np.save(seq_path, np.stack(seqs, axis=0))

    arr = np.stack(seqs, axis=0)
    live = arr[:, :, :UIPRMD_JOINTS, :].std(axis=1)  # (N, 22, 3) temporal sd per slot
    print(f"  geometry={geometry}: dead coordinate slots "
          f"{int((live <= 1e-9).sum())}/{live.size}  median sd {float(np.median(live)):.3f}")

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
    ap.add_argument("--geometry", choices=["fk", "raw"], default="fk",
                    help="fk = forward kinematics (correct); raw = legacy bone-offset bug")
    ap.add_argument("--fk-variant", choices=["reference", "corrected"], default="reference",
                    help="reference = literal Animation.m; corrected = arms inherit torso chain")
    args = ap.parse_args()
    build(args.geometry, args.fk_variant)


if __name__ == "__main__":
    main()
