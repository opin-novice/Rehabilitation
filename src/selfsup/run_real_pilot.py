"""Real-data execution driver (PRD execution R1-R5).

Clean design given available data (IRDS sequences absent):
  - pretrain on UI-PRMD (Kinect, unlabeled)  -> outputs/ssl_pretrain/uiprmd_pool/
  - hold out REHAB24-6 (OptiTrack) as the PURE zero-shot cross-sensor test
  - UI-PRMD zero-shot is reported too, flagged transductive (in-pool)
  - KIMORE (380 real samples) used only for LOSO fine-tuning + eval

Bounded epoch budget for tractable wall-clock; full-scale settings in REPRODUCE_PAPER2.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from selfsup.config import PretrainCfg, provenance  # noqa: E402
from selfsup.data import load_uiprmd_labeled  # noqa: E402
from selfsup.folds import generate_folds, excluded_subjects  # noqa: E402
from selfsup.pretrain import pretrain  # noqa: E402
from selfsup.pretrain_pool import build_pool  # noqa: E402
from selfsup.run_all import run_loso_conditions, run_zeroshot  # noqa: E402
from selfsup.stats import compute_statistics  # noqa: E402
from selfsup.make_tables import make_tables  # noqa: E402
from selfsup.make_figures import make_figures  # noqa: E402

POOL_NAME = "uiprmd_pool"


def main(out_root: str, pooled_dir: str, n_folds: int,
         pretrain_epochs: int, finetune_epochs: int, d_model: int) -> int:
    folds_json = os.path.join(out_root, "folds.json")
    generate_folds(pooled_dir, n_folds=n_folds, out_path=folds_json)
    exclude = excluded_subjects(folds_json)

    # --- real unlabeled pretraining pool = UI-PRMD (REHAB24-6 held out) ---
    Xu, uids, _ = load_uiprmd_labeled()
    if Xu.shape[0] == 0:
        print("[FATAL] UI-PRMD cache missing; run src/load_uiprmd_validity.py --build")
        return 1
    x_pool, manifest = build_pool({"UIPRMD": (Xu, uids)}, exclude, include=["UIPRMD"])

    pre_dir = os.path.join(out_root, "ssl_pretrain", POOL_NAME)
    for pretext in ("contrastive", "masked"):
        cfg = PretrainCfg(pretext=pretext, pool=POOL_NAME, epochs=pretrain_epochs,
                          batch_size=128, d_model=d_model)
        ckpt = pretrain(x_pool, cfg, pre_dir, log_every=10)
        Path(ckpt + ".provenance.json").write_text(
            json.dumps(provenance(cfg, {"pool_manifest": manifest.__dict__}), indent=2, default=str))

    # --- 5 real LOSO conditions on KIMORE ---
    run_loso_conditions(out_root, pooled_dir, pool=POOL_NAME, n_folds=n_folds,
                        epochs=finetune_epochs, batch_size=16, d_model=d_model)

    # --- real zero-shot: REHAB24-6 (pure cross-sensor) + UI-PRMD (transductive) ---
    run_zeroshot(out_root, ["REHAB246", "UIPRMD"], dummy=False)

    # --- stats + artifacts ---
    compute_statistics(out_root, n_boot=10000)
    make_tables(out_root)
    make_figures(out_root)
    print(f"\n[REAL PILOT DONE] -> {out_root}/ssl_results")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="outputs")
    ap.add_argument("--pooled_dir", default="KIMORE_pooled")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--pretrain_epochs", type=int, default=60)
    ap.add_argument("--finetune_epochs", type=int, default=80)
    ap.add_argument("--d_model", type=int, default=128)
    a = ap.parse_args()
    raise SystemExit(main(a.out_root, a.pooled_dir, a.n_folds,
                          a.pretrain_epochs, a.finetune_epochs, a.d_model))
