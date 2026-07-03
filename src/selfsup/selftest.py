"""End-to-end self-test on synthetic data (no real datasets required).

Verifies the whole SSL architecture wires together:
  encoder.forward_features -> augmentations -> contrastive & masked pretrain
  -> encoder-only checkpoint -> loads into a fresh regressor (strict=False)
  -> pool leakage guard -> harmonize remap -> linear probe sanity.

Run:  python src/ssl/selftest.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models_stgcn import TCNRegressor, build_encoder, encoder_state_dict  # noqa: E402
from selfsup.augmentations import two_views, PRETRAIN_AUGS, FINETUNE_AUGS  # noqa: E402
from selfsup.config import PretrainCfg  # noqa: E402
from selfsup.harmonize import remap_joints  # noqa: E402
from selfsup.pretrain import pretrain  # noqa: E402
from selfsup.pretrain_pool import build_pool  # noqa: E402


def _synthetic(n=64, t=100, j=25, c=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, t, j, c)).astype(np.float32)


def main() -> int:
    torch.manual_seed(0)
    ok = True

    # 1. encoder + forward_features
    enc = build_encoder("tcn", d_model=64)
    x = torch.from_numpy(_synthetic(8))
    feat = enc.forward_features(x)
    assert feat.shape == (8, 64), feat.shape
    assert enc.out_dim == 64
    print("[1] encoder.forward_features OK", tuple(feat.shape))

    # 2. augmentations preserve shape; registry split is sane
    v1, v2 = two_views(x[0])
    assert v1.shape == x[0].shape == v2.shape
    assert set(FINETUNE_AUGS).issubset(set(PRETRAIN_AUGS))
    assert "speed_perturb" in PRETRAIN_AUGS and "speed_perturb" not in FINETUNE_AUGS
    print("[2] augmentations OK  pretrain=%d finetune=%d" % (len(PRETRAIN_AUGS), len(FINETUNE_AUGS)))

    # 3. pool leakage guard + harmonize
    X = _synthetic(20)
    sids = np.array([f"IRDS:{i%5}" for i in range(20)])
    pool, man = build_pool({"IRDS": (X, sids)}, exclude_subjects={"IRDS:0"})
    assert man.n_total == 16, man.n_total          # subject 0 (4 reps) excluded
    assert "IRDS:0" in man.excluded_subjects
    remapped = remap_joints(_synthetic(3, j=22), list(range(22)) + [-1, -1, -1])
    assert remapped.shape == (3, 100, 25, 3)
    print("[3] pool guard + harmonize OK  pool=%d" % man.n_total)

    # 4. contrastive + masked pretraining -> encoder-only checkpoints
    with tempfile.TemporaryDirectory() as td:
        xp = _synthetic(64)
        c_ckpt = pretrain(xp, PretrainCfg(pretext="contrastive", d_model=64,
                                          batch_size=16, epochs=2), td, device="cpu", log_every=1)
        m_ckpt = pretrain(xp, PretrainCfg(pretext="masked", d_model=64,
                                          batch_size=16, epochs=2), td, device="cpu", log_every=1)
        print("[4] pretraining OK")

        # 5. checkpoint loads into a FRESH regressor (encoder matches, head missing)
        for ck in (c_ckpt, m_ckpt):
            blob = torch.load(ck, map_location="cpu")
            fresh = TCNRegressor(seq_len=100, d_model=64)
            res = fresh.load_state_dict(blob["encoder_state"], strict=False)
            enc_keys = [k for k in blob["encoder_state"]]
            assert len(enc_keys) > 0
            assert all(not k.startswith("head") for k in enc_keys), "head leaked into encoder ckpt"
            assert all(k.startswith("head") for k in res.missing_keys), res.missing_keys
            assert res.unexpected_keys == [], res.unexpected_keys
        print("[5] encoder checkpoint loads into fresh regressor (strict=False) OK")

    # 6. linear probe sanity (optional deps)
    try:
        from selfsup.linear_probe import linear_probe_score
        y = np.random.default_rng(1).standard_normal(64).astype(np.float32)
        score = linear_probe_score(enc, _synthetic(64), y, device="cpu")
        assert "spearman" in score
        print("[6] linear probe OK  spearman=%.3f (random data -> ~0 expected)" % score["spearman"])
    except ImportError:
        print("[6] linear probe SKIPPED (scipy/sklearn not installed)")

    print("\nSELFTEST PASSED" if ok else "\nSELFTEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
