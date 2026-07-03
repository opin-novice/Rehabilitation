"""Cross-sensor harmonization (Layer 0).

Canonical schema = Kinect-v2 25-joint order (matches constants.SKELETON_EDGES).
Each corpus is mapped onto this schema so a single encoder can consume all of
them. This centralizes the joint mapping the plan left under-specified and makes
cross-sensor claims precise (record which capture/sensor each corpus uses).

Exact per-sensor index tables live with the existing corpus loaders
(load_rehab246.py, load_uiprmd_validity.py). Here we provide the canonical
schema, a generic remap utility, and a sensor registry; loaders supply indices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

CANONICAL_NUM_JOINTS = 25
CANONICAL_NUM_CHANNELS = 3

# sensor descriptor per corpus (label_type documents what supervision exists)
@dataclass
class CorpusSpec:
    name: str
    sensor: str          # kinect_v2 | kinect_one | optitrack | vicon
    native_joints: int
    label_type: str      # "physician_score" | "binary" | "none"
    cross_sensor: bool   # True if sensor != Kinect (true domain shift vs KIMORE)


CORPORA: Dict[str, CorpusSpec] = {
    "KIMORE":   CorpusSpec("KIMORE",   "kinect_v2",  25, "physician_score", False),
    "IRDS":     CorpusSpec("IRDS",     "kinect_one", 25, "binary",          False),
    "UIPRMD":   CorpusSpec("UIPRMD",   "kinect_v2",  22, "binary",          False),  # Kinect capture: NOT cross-sensor
    "REHAB246": CorpusSpec("REHAB246", "optitrack",  26, "binary",          True),   # true cross-sensor test
}


def remap_joints(x: np.ndarray, index_map: List[int]) -> np.ndarray:
    """Remap (N, T, J_src, C) -> (N, T, 25, C) via a length-25 index list.

    index_map[i] = source joint index for canonical joint i, or -1 to zero-fill
    (missing joint; downstream joint_mask augmentation makes the model robust).
    """
    assert x.ndim == 4, f"expected (N,T,J,C), got {x.shape}"
    assert len(index_map) == CANONICAL_NUM_JOINTS
    N, T, _, C = x.shape
    out = np.zeros((N, T, CANONICAL_NUM_JOINTS, C), dtype=x.dtype)
    for canon_i, src_i in enumerate(index_map):
        if src_i >= 0:
            out[:, :, canon_i, :] = x[:, :, src_i, :]
    return out


def harmonize(x: np.ndarray, corpus: str, index_map: Optional[List[int]] = None) -> np.ndarray:
    """Return canonical (N, T, 25, 3) arrays for a corpus.

    If the corpus is already 25-joint Kinect, pass it through; otherwise an
    index_map (from the corpus loader) is required.
    """
    spec = CORPORA[corpus]
    if x.shape[2] == CANONICAL_NUM_JOINTS and index_map is None:
        return x.astype(np.float32)
    if index_map is None:
        raise ValueError(
            f"{corpus} has {x.shape[2]} joints; supply index_map (len 25) from its loader."
        )
    return remap_joints(x, index_map).astype(np.float32)
