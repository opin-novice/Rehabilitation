"""Robust loaders for the novelty analyses.

Every loader returns ``None`` (and prints a ``[SKIP]`` note) when its artifact is
missing, so each analysis degrades gracefully instead of crashing the run.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import numpy as np
import pandas as pd

from novelty import config


def load_pooled_kimore() -> dict | None:
    """Load the pooled KIMORE arrays. Returns a dict of arrays or None."""
    d = config.POOLED_DIR
    req = ["Train_X.csv", "Train_Y.csv", "exercise_ids.csv", "subject_ids.csv", "meta.csv"]
    if not all(os.path.exists(os.path.join(d, f)) for f in req):
        print(f"[SKIP] pooled KIMORE not found in {d}")
        return None
    X_raw = pd.read_csv(os.path.join(d, "Train_X.csv"), header=None).values
    y = pd.read_csv(os.path.join(d, "Train_Y.csv"), header=None).values.ravel()
    eid = pd.read_csv(os.path.join(d, "exercise_ids.csv"), header=None).values.ravel()
    sid = pd.read_csv(os.path.join(d, "subject_ids.csv"), header=None).values.ravel()
    meta = pd.read_csv(os.path.join(d, "meta.csv"))
    return {
        "X_raw": X_raw,
        "y": y,
        "exercise_ids": eid.astype(int),
        "subject_ids": sid,
        "group_labels": meta["group"].values,
        "meta": meta,
    }


def load_oof_table(exp_dir: str) -> pd.DataFrame | None:
    """Load combined OOF predictions for one experiment dir (relative or abs).

    Columns: subject_id, exercise_id, y_true, y_pred, abs_error, fold.
    Falls back to rebuilding from per-fold files. KIMORE exercises only (0-4).
    """
    base = config.resolve(exp_dir)
    combined = os.path.join(base, "oof_predictions_all.csv")
    if os.path.exists(combined):
        df = pd.read_csv(combined)
    else:
        fold_dfs = []
        for i in range(5):
            fp = os.path.join(base, f"fold_{i}", "oof_predictions.csv")
            if os.path.exists(fp):
                fold_dfs.append(pd.read_csv(fp))
        if not fold_dfs:
            return None
        df = pd.concat(fold_dfs, ignore_index=True)
    if "exercise_id" in df.columns:
        df = df[df["exercise_id"] < config.N_EXERCISES].reset_index(drop=True)
    return df


def load_all_oof() -> dict[str, pd.DataFrame]:
    """Return {model_name: oof_df} for every registered experiment with data."""
    out: dict[str, pd.DataFrame] = {}
    for name, d in config.EXPERIMENTS:
        df = load_oof_table(d)
        if df is not None:
            out[name] = df
        else:
            print(f"[SKIP] no OOF for '{name}' ({d}) - run src/generate_oof.py")
    return out


def load_irds_reliability() -> pd.DataFrame | None:
    """Load the IRDS reliability CSV produced by src/irds_eval.py."""
    p = config.IRDS_RELIABILITY_CSV
    if not os.path.exists(p):
        print(f"[SKIP] IRDS reliability CSV not found at {p} - run src/irds_eval.py")
        return None
    return pd.read_csv(p)
