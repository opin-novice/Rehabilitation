#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
engine.py
=========
The inference core of the live demo. Wraps the TRAINED checkpoints behind one call so the capture
loop and the dashboard never touch model internals.

  * EGRU  (ours)      -- SE3EquivariantGRU. Invariant BY CONSTRUCTION: rotate the input skeleton and
                          the score is identical to ~1e-6 (this is the whole Act-2 thesis; smoke_test
                          re-measures it, we never assert it). 5-fold ensemble mean for a stable live
                          number.
  * PCT   (baseline)  -- PointCloudTransformerRegressor, the target paper (arXiv:2606.30309). Learned
                          its viewpoint tolerance from data, so it DECAYS under rotation. Resampled to
                          a fixed 100-frame grid (its architecture demands one).

Both models receive the BYTE-IDENTICAL sample -- clean, rotated, or corrupted -- exactly as
block23_experiments feeds them, so the side-by-side is honest.

Scores are returned in CLINICAL units (multiplied by SCORE_MAX = 50, the KIMORE Total Score range).
"""

import os
import sys

import numpy as np
import torch

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import kimore_cde_data as kd                                    # noqa: E402
import block2_transforms as bt                                  # noqa: E402
from equivariant_gru import SE3EquivariantGRU                   # noqa: E402
from models_curvenet import PointCloudTransformerRegressor      # noqa: E402

SCORE_MAX = kd.SCORE_MAX
N_JOINTS = kd.N_JOINTS
OPERATORS = ["linear", "cubic", "zero_fill"]


class DemoEngine:
    """Load once, predict many. Thread-agnostic; call predict() from the capture loop."""

    def __init__(self, ckpt_dir="outputs/cde_block2", device=None, model_seed=0,
                 folds=(0, 1, 2, 3, 4), n_frames=100, pct_operator="linear", load_pct=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_dir = ckpt_dir
        self.n_frames = n_frames
        self.pct_operator = pct_operator

        # --- EGRU ensemble (ours) ---
        self.egru = []
        for f in folds:
            m = SE3EquivariantGRU(
                n_scalar=32, n_vec=8, n_layers=2, lmax=2, gru_hidden=128,
                dropout=0.0, n_exercises=5, use_speed=True, use_chiral=False).to(self.device)
            ck = os.path.join(ckpt_dir, f"egru_s{model_seed}_pooled_f{f}.pt")
            # These checkpoints predate the occlusion-aware `dead_scalar` param, which is consulted
            # ONLY in masked mode (use_mask=True) -- never on the demo's unmasked path. Load
            # non-strict, but assert the ONLY gap is that mask-mode param so a real mismatch still
            # fails loudly.
            info = m.load_state_dict(torch.load(ck, map_location=self.device), strict=False)
            allowed_missing = {"encoder.dead_scalar"}
            unexpected = set(info.unexpected_keys)
            missing = set(info.missing_keys)
            assert not unexpected, f"{ck}: unexpected keys {unexpected}"
            assert missing <= allowed_missing, f"{ck}: unexpected missing keys {missing - allowed_missing}"
            m.eval()
            self.egru.append(m)

        # --- PCT ensemble (baseline) ---
        # The session demo (app_session.py) scores with EGRU only, so it passes load_pct=False to
        # skip this loop entirely -> faster startup, less memory. Default True keeps app.py /
        # app_camo.py (the comparison demos) byte-for-byte unchanged.
        self.pct = []
        if not load_pct:
            return
        for f in folds:
            m = PointCloudTransformerRegressor(
                seq_len=n_frames, num_joints=N_JOINTS, num_channels=3,
                dim=256, spatial_depth=6, temporal_depth=3, heads=4, dropout=0.1, k=10,
                num_exercises=5).to(self.device)
            ck = os.path.join(ckpt_dir, f"pct_pooled_f{f}.pt")
            m.load_state_dict(torch.load(ck, map_location=self.device))
            m.eval()
            self.pct.append(m)

    # -- individual pathways ------------------------------------------------
    @torch.no_grad()
    def _egru_one(self, model, sample):
        t, x, _, e, n = kd.collate([sample], device=self.device)
        return float(model(t, x, e, n).item())

    @torch.no_grad()
    def _pct_one(self, model, sample):
        x, _, e = bt.batch_fixed_grid([sample], self.n_frames, self.pct_operator,
                                      device=self.device)
        return float(model(x, exercise_id=e).squeeze(-1).item())

    def predict_egru(self, sample):
        """Ensemble mean EGRU score in clinical units."""
        s = np.mean([self._egru_one(m, sample) for m in self.egru])
        return float(s * SCORE_MAX)

    def predict_pct(self, sample):
        """Ensemble mean PCT score in clinical units."""
        s = np.mean([self._pct_one(m, sample) for m in self.pct])
        return float(s * SCORE_MAX)

    # -- the demo call ------------------------------------------------------
    def predict(self, sample, exercise=1, angle=0.0):
        """Score one buffered sample with BOTH models. `angle` rotates the skeleton about the
        vertical axis (Act 2) -- the SAME rotated bytes go to both models. Returns clinical scores.

        exercise: 1..5 (KIMORE exercise id; enters both models as a Type-0 invariant one-hot).
        """
        s = dict(sample)
        s["exercise"] = int(exercise)
        s.setdefault("y", 0.0)                            # collate needs a label; unused at inference
        if angle:
            s = bt.rotate_sample(s, float(angle))
            s["exercise"] = int(exercise)                # rotate_sample copies the dict; keep ex
        return {
            "egru": self.predict_egru(s),
            "pct": self.predict_pct(s),
            "angle": float(angle),
            "exercise": int(exercise),
            "n_frames": int(s["n_frames"]),
        }
