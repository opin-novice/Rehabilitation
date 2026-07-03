"""Pool all 5 KIMORE exercises into a single dataset with exercise ID column.

Output:
  <out_dir>/Train_X.csv        — (N_total * seq_len) rows x 100 cols (joint positions)
  <out_dir>/Train_Y.csv        — N_total rows x 1 col  (TS score)
  <out_dir>/subject_ids.csv    — N_total rows x 1 col  (integer subject ID, consistent across exercises)
  <out_dir>/exercise_ids.csv   — N_total rows x 1 col  (0-indexed exercise: 0..4)
  <out_dir>/meta.csv           — N_total rows, human-readable

Subjects keep the same integer ID across all exercises so LOSO splits correctly
hold out all 5 exercise recordings from a given subject simultaneously.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", default=r"D:\Rehabilation\KIMORE_processed")
    parser.add_argument("--out_dir",         default=r"D:\Rehabilation\KIMORE_pooled")
    parser.add_argument("--exercises",  nargs="+", default=["1","2","3","4","5"])
    args = parser.parse_args()

    all_x:    list[np.ndarray] = []
    all_y:    list[np.ndarray] = []
    all_po:   list[np.ndarray] = []
    all_cf:   list[np.ndarray] = []
    all_sid:  list[np.ndarray] = []
    all_eid:  list[np.ndarray] = []
    all_meta: list[pd.DataFrame] = []

    # Build a global subject→integer map from Exercise1's meta (all exercises share subjects)
    meta1 = pd.read_csv(os.path.join(args.processed_root, "Exercise1", "meta.csv"))
    # Map: subject_name → consistent integer (0-based across all exercises)
    unique_subjects = sorted(meta1["subject_name"].unique())
    subj_to_int = {s: i for i, s in enumerate(unique_subjects)}

    for ex_str in args.exercises:
        ex_dir = os.path.join(args.processed_root, f"Exercise{ex_str}")
        ex_idx = int(ex_str) - 1   # 0-based exercise id

        meta = pd.read_csv(os.path.join(ex_dir, "meta.csv"))
        x = pd.read_csv(os.path.join(ex_dir, "Train_X.csv"), header=None).values.astype(np.float32)
        y = pd.read_csv(os.path.join(ex_dir, "Train_Y.csv"), header=None).values.squeeze().astype(np.float32)

        n = len(meta)
        seq_len = x.shape[0] // n

        # Re-map subject IDs to the global consistent integer
        sid_global = np.array([subj_to_int[s] for s in meta["subject_name"]], dtype=np.int32)
        eid_arr    = np.full(n, ex_idx, dtype=np.int32)

        all_x.append(x)
        all_y.append(y)
        all_sid.append(sid_global)
        all_eid.append(eid_arr)

        # Propagate PO/CF sub-scores if present (added by updated prepare_kimore.py)
        if "po_score" in meta.columns and "cf_score" in meta.columns:
            all_po.append(meta["po_score"].values.astype(np.float32))
            all_cf.append(meta["cf_score"].values.astype(np.float32))

        meta = meta.copy()
        meta["global_subject_id"] = sid_global
        meta["exercise_id"] = ex_idx
        all_meta.append(meta)

        print(f"  Exercise {ex_str}: {n} samples, seq_len={seq_len}, "
              f"score [{y.min():.1f}, {y.max():.1f}]")

    X     = np.concatenate(all_x,   axis=0)
    Y     = np.concatenate(all_y,   axis=0)
    SID   = np.concatenate(all_sid, axis=0)
    EID   = np.concatenate(all_eid, axis=0)
    META  = pd.concat(all_meta, ignore_index=True)

    n_total = Y.shape[0]
    print(f"\nPooled: {n_total} samples total  "
          f"({len(unique_subjects)} unique subjects x ~5 exercises)")
    print(f"X shape: {X.shape}  Y shape: {Y.shape}")
    print(f"Score range: [{Y.min():.2f}, {Y.max():.2f}]  mean={Y.mean():.2f}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X).to_csv(   os.path.join(args.out_dir, "Train_X.csv"),      header=False, index=False)
    pd.DataFrame(Y).to_csv(   os.path.join(args.out_dir, "Train_Y.csv"),      header=False, index=False)
    pd.DataFrame(SID).to_csv( os.path.join(args.out_dir, "subject_ids.csv"),  header=False, index=False)
    pd.DataFrame(EID).to_csv( os.path.join(args.out_dir, "exercise_ids.csv"), header=False, index=False)
    META.to_csv(               os.path.join(args.out_dir, "meta.csv"),         index=False)

    if all_po and len(all_po) == len(args.exercises):
        PO = np.concatenate(all_po, axis=0)
        CF = np.concatenate(all_cf, axis=0)
        pd.DataFrame(PO).to_csv(os.path.join(args.out_dir, "Train_PO.csv"), header=False, index=False)
        pd.DataFrame(CF).to_csv(os.path.join(args.out_dir, "Train_CF.csv"), header=False, index=False)
        print(f"PO range: [{PO.min():.2f}, {PO.max():.2f}]  CF range: [{CF.min():.2f}, {CF.max():.2f}]")

    print(f"Written to: {args.out_dir}")


if __name__ == "__main__":
    main()
