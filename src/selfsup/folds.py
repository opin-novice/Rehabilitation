"""Stratified-LOSO fold assignment -- single source of truth (PRD R2).

Mirrors EXACTLY the split train_loso.py performs internally
(`StratifiedGroupKFold(n_splits=n_folds).split(arange(n), y=groups, groups=sids)`),
so outputs/folds.json faithfully documents what fine-tuning will do and the
pretraining leakage guard excludes the right KIMORE test subjects.

CLI:
  python src/selfsup/folds.py --pooled_dir KIMORE_pooled --n_folds 5 --out outputs/folds.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from selfsup.config import git_sha  # noqa: E402


def generate_folds(pooled_dir: str, n_folds: int = 5, out_path: str = "outputs/folds.json") -> dict:
    sids = pd.read_csv(os.path.join(pooled_dir, "subject_ids.csv"), header=None).values.squeeze().astype(int)
    groups = pd.read_csv(os.path.join(pooled_dir, "meta.csv"))["group"].values
    n = len(sids)

    # IMPORTANT: same call/params as train_loso.main() -> identical folds.
    sgkf = StratifiedGroupKFold(n_splits=n_folds)
    folds = []
    for fold_id, (train_idx, val_idx) in enumerate(sgkf.split(np.arange(n), y=groups, groups=sids)):
        test_subs = sorted({f"KIMORE::{s}" for s in sids[val_idx]})
        train_subs = sorted({f"KIMORE::{s}" for s in sids[train_idx]})
        folds.append({"fold_id": fold_id, "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
                      "train_subjects": train_subs, "test_subjects": test_subs})

    payload = {
        "n_folds": n_folds,
        "pooled_dir": pooled_dir,
        "n_samples": int(n),
        "n_subjects": int(len(set(sids))),
        "git_sha": git_sha(),
        "folds": folds,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f"[folds] wrote {n_folds}-fold LOSO split -> {out_path} "
          f"({n} samples, {len(set(sids))} subjects)")
    return payload


def excluded_subjects(folds_json: str = "outputs/folds.json") -> set:
    """Union of all KIMORE test subjects -> excluded from the pretraining pool."""
    data = json.loads(Path(folds_json).read_text())
    ex = set()
    for f in data["folds"]:
        ex.update(f["test_subjects"])
    return ex


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooled_dir", default="KIMORE_pooled")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--out", default="outputs/folds.json")
    a = ap.parse_args()
    generate_folds(a.pooled_dir, a.n_folds, a.out)
