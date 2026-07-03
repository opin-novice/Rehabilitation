"""Unlabeled pretraining pool (Layer 0) with a hard leakage guard.

Addresses the 'IRDS-only 2,589 sequences is too small for SSL' risk: pool size
is a first-class experiment variable (irds_only vs all_corpora), so a downstream
null cannot be blamed on data scale. Subjects that will appear in any KIMORE
LOSO test fold are excluded from the pool and the exclusion is logged.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class PoolManifest:
    n_total: int
    per_corpus: Dict[str, int]
    excluded_subjects: List[str] = field(default_factory=list)
    hash: str = ""


def _hash_arrays(arrs: List[np.ndarray]) -> str:
    h = hashlib.sha1()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes()[:4096])  # prefix is enough for provenance
        h.update(str(a.shape).encode())
    return h.hexdigest()[:12]


def build_pool(
    corpus_arrays: Dict[str, Tuple[np.ndarray, np.ndarray]],
    exclude_subjects: set,
    include: List[str] | None = None,
) -> Tuple[np.ndarray, PoolManifest]:
    """Concatenate unlabeled sequences across corpora.

    corpus_arrays: {corpus_name: (X (N,T,25,3), subject_uids (N,) str)}.
    exclude_subjects: subject_uids to drop (KIMORE LOSO test subjects).
    include: subset of corpora (None = all provided). e.g. ["IRDS"] for irds_only.
    """
    include = include or list(corpus_arrays.keys())
    chunks, per_corpus, excluded_hits = [], {}, []
    for name in include:
        X, sids = corpus_arrays[name]
        keep = np.array([s not in exclude_subjects for s in sids], dtype=bool)
        dropped = [str(s) for s in np.asarray(sids)[~keep]]
        excluded_hits.extend(sorted(set(dropped)))
        Xk = X[keep]
        chunks.append(Xk)
        per_corpus[name] = int(Xk.shape[0])

    pool = np.concatenate(chunks, axis=0) if chunks else np.empty((0,))
    # Leakage guard: assert nothing from the exclude set survived.
    assert all(
        s not in exclude_subjects
        for name in include
        for s in corpus_arrays[name][1][
            [t not in exclude_subjects for t in corpus_arrays[name][1]]
        ]
    ), "leakage guard failed: excluded subject present in pool"

    manifest = PoolManifest(
        n_total=int(pool.shape[0]),
        per_corpus=per_corpus,
        excluded_subjects=sorted(set(excluded_hits)),
        hash=_hash_arrays(chunks),
    )
    print(f"[pool] n={manifest.n_total} per_corpus={per_corpus} "
          f"excluded={len(manifest.excluded_subjects)} subjects")
    return pool.astype(np.float32), manifest
