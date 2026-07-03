"""Paper-2 self-supervised pretraining package (`selfsup`).

Note: named `selfsup` (not `ssl`) to avoid colliding with Python's stdlib `ssl`.


Scalable SSL layer on top of the Paper-1 pipeline (see ARCHITECTURE_PAPER2.md).
Design: ONE encoder (models_stgcn.build_encoder) + pluggable pretext tasks
(PretextTask ABC) + generic trainer, feeding the existing train_loso.py LOSO
fine-tuning via --init_ckpt.

Layers
  0 data      : harmonize, pretrain_pool
  1 encoder   : models_stgcn.build_encoder + heads
  2 pretext   : augmentations, pretext, pretrain, linear_probe
  3 downstream: train_loso.py (--init_ckpt) + validity_eval.py (reused)
  4 orchestr. : config, registry, run_all
"""
from __future__ import annotations

__all__ = [
    "augmentations", "heads", "pretext", "pretrain", "linear_probe",
    "harmonize", "pretrain_pool", "config", "registry",
]
