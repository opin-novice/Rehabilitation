"""Real + dummy data adapters (Layer 0, PRD R1).

Produces canonical (N, 100, 25, 3) float32 arrays with namespaced subject UIDs
('CORPUS::sid') by reusing the existing corpus loaders and cached outputs. Every
real loader degrades gracefully to a [SKIP] + empty arrays when raw/cached data
is absent, so the whole pipeline runs in --dummy mode with no dataset present.
"""
from __future__ import annotations

import glob
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from constants import SEQ_LEN, NUM_JOINTS, NUM_CHANNELS  # noqa: E402

EMPTY = (np.empty((0, SEQ_LEN, NUM_JOINTS, NUM_CHANNELS), np.float32), np.array([], dtype=object), {})


# --------------------------------------------------------------------------- #
# Dummy data (smoke mode)                                                      #
# --------------------------------------------------------------------------- #
def synthesize_dummy_data(n_samples: int = 50, corpus: str = "DUMMY", seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, SEQ_LEN, NUM_JOINTS, NUM_CHANNELS)).astype(np.float32)
    uids = np.array([f"{corpus}::{i % 8}" for i in range(n_samples)], dtype=object)
    labels = rng.integers(0, 2, size=n_samples)          # binary correctness
    return X, uids, {"label_type": "binary", "labels": labels}


def write_dummy_pooled(out_dir: str, n_subjects: int = 8, n_ex: int = 5,
                       reps: int = 2, seed: int = 0) -> str:
    """Write a minimal KIMORE-pooled dataset train_loso.py can consume."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows_x, y, sids, eids, groups = [], [], [], [], []
    for s in range(n_subjects):
        grp = "H" if s < n_subjects // 2 else "P"       # 2 clinical groups
        for e in range(n_ex):
            for _ in range(reps):
                rows_x.append(rng.standard_normal((SEQ_LEN, 100)).astype(np.float32))
                y.append(rng.uniform(0, 50))
                sids.append(s); eids.append(e); groups.append(grp)
    X = np.concatenate(rows_x, axis=0)
    pd.DataFrame(X).to_csv(os.path.join(out_dir, "Train_X.csv"), header=False, index=False)
    pd.DataFrame(y).to_csv(os.path.join(out_dir, "Train_Y.csv"), header=False, index=False)
    pd.DataFrame(sids).to_csv(os.path.join(out_dir, "subject_ids.csv"), header=False, index=False)
    pd.DataFrame(eids).to_csv(os.path.join(out_dir, "exercise_ids.csv"), header=False, index=False)
    pd.DataFrame({"group": groups}).to_csv(os.path.join(out_dir, "meta.csv"), index=False)
    print(f"[dummy] wrote pooled dataset -> {out_dir} ({len(y)} samples, {n_subjects} subjects)")
    return out_dir


# --------------------------------------------------------------------------- #
# Real loaders (reuse existing pipeline outputs)                               #
# --------------------------------------------------------------------------- #
def load_kimore_unlabeled(pooled_dir: str = "KIMORE_pooled") -> Tuple[np.ndarray, np.ndarray, dict]:
    xp = os.path.join(pooled_dir, "Train_X.csv")
    if not os.path.exists(xp):
        print(f"[SKIP] KIMORE: {xp} not found")
        return EMPTY
    from rehab_dataset import _select_xyz_columns
    x_raw = pd.read_csv(xp, header=None).values.astype(np.float32)
    sids = pd.read_csv(os.path.join(pooled_dir, "subject_ids.csv"), header=None).values.squeeze().astype(int)
    y = pd.read_csv(os.path.join(pooled_dir, "Train_Y.csv"), header=None).values.squeeze().astype(np.float32)
    n = len(sids)
    xyz = _select_xyz_columns(x_raw).reshape(n, SEQ_LEN, NUM_JOINTS, NUM_CHANNELS).astype(np.float32)
    uids = np.array([f"KIMORE::{s}" for s in sids], dtype=object)
    return xyz, uids, {"label_type": "physician_score", "labels": y}


def _load_cached_corpus(seqs_glob: str, manifest_glob: str, corpus: str,
                        label_col: str = "correct_label", sid_col: str = "subject_id"):
    """Load a cached (N,SEQ_LEN,25,3) .npy + manifest produced by an existing loader."""
    seqs = glob.glob(seqs_glob)
    mans = glob.glob(manifest_glob)
    if not seqs or not mans:
        print(f"[SKIP] {corpus}: cached seqs/manifest not found "
              f"({seqs_glob} | {manifest_glob}). Run its loader first.")
        return EMPTY
    X = np.load(seqs[0]).astype(np.float32)
    man = pd.read_csv(mans[0])
    if X.shape[1:] != (SEQ_LEN, NUM_JOINTS, NUM_CHANNELS):
        print(f"[SKIP] {corpus}: cached shape {X.shape} != (*,{SEQ_LEN},{NUM_JOINTS},{NUM_CHANNELS})")
        return EMPTY
    sid = man[sid_col].values if sid_col in man else np.arange(len(X))
    uids = np.array([f"{corpus}::{s}" for s in sid], dtype=object)
    labels = man[label_col].values if label_col in man else None
    return X, uids, {"label_type": "binary", "labels": labels}


def load_rehab246_labeled():
    return _load_cached_corpus("outputs/validity/*seq*.npy", "outputs/validity/*manifest*.csv", "REHAB246")


def load_uiprmd_labeled():
    """UI-PRMD cache. Geometry is selectable so raw-vs-FK reruns are reproducible.

    The default cache (outputs/validity_uiprmd) is the forward-kinematics build. The
    original published numbers were produced from parent-relative bone offsets read as
    world coordinates (57.3% of coordinates constant); set UIPRMD_CACHE=
    outputs/validity_uiprmd_raw to reproduce them. See docs/worklog_2026-07-29.md §4-5.
    """
    base = os.environ.get("UIPRMD_CACHE", "outputs/validity_uiprmd")
    return _load_cached_corpus(f"{base}/*seq*.npy", f"{base}/*manifest*.csv", "UIPRMD")


def load_irds_unlabeled():
    # IRDS caches under outputs/irds_eval or outputs/validity_irds depending on run.
    for base in ("outputs/validity_irds", "outputs/irds_eval"):
        got = _load_cached_corpus(f"{base}/*seq*.npy", f"{base}/*manifest*.csv", "IRDS")
        if got[0].shape[0] > 0:
            return got
    print("[SKIP] IRDS: no cached sequences found (run src/irds_eval.py --build first).")
    return EMPTY


def load_all_corpora(dummy: bool = False, pooled_dir: str = "KIMORE_pooled",
                     n_dummy: int = 50) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return {corpus: (X, subject_uids)} for pool building. Labels kept separately."""
    if dummy:
        out = {}
        for i, name in enumerate(["IRDS", "KIMORE", "REHAB246", "UIPRMD"]):
            X, uids, _ = synthesize_dummy_data(n_dummy, corpus=name, seed=i)
            out[name] = (X, uids)
        return out
    loaders = {
        "IRDS": load_irds_unlabeled,
        "KIMORE": lambda: load_kimore_unlabeled(pooled_dir),
        "REHAB246": load_rehab246_labeled,
        "UIPRMD": load_uiprmd_labeled,
    }
    out = {}
    for name, fn in loaders.items():
        X, uids, _ = fn()
        if X.shape[0] > 0:
            out[name] = (X, uids)
    return out


def load_corpus_with_labels(corpus: str, dummy: bool = False):
    """Return (X, labels, uids) for a zero-shot test corpus."""
    if dummy:
        X, uids, meta = synthesize_dummy_data(40, corpus=corpus, seed=hash(corpus) % 100)
        return X, meta["labels"], uids
    fn = {"REHAB246": load_rehab246_labeled, "UIPRMD": load_uiprmd_labeled,
          "IRDS": load_irds_unlabeled, "KIMORE": lambda: load_kimore_unlabeled()}[corpus]
    X, uids, meta = fn()
    return X, meta.get("labels"), uids


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true")
    a = ap.parse_args()
    corpora = load_all_corpora(dummy=a.dummy)
    for name, (X, uids) in corpora.items():
        print(f"{name}: X={X.shape} uids={len(uids)} unique_subjects={len(set(uids))}")
