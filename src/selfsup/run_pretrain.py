"""Pretraining driver on the real/dummy pool (PRD R3).

Builds two unlabeled pools and pretrains both SSL paradigms on each:
  irds_only   = IRDS                    (clean: REHAB24-6 / UI-PRMD stay pure zero-shot)
  all_corpora = IRDS + REHAB246 + UIPRMD (transductive upper-bound; KIMORE excluded
                because every KIMORE subject is a LOSO eval target -> leakage guard)
Each run saves an encoder-only checkpoint + provenance sidecar.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from selfsup.config import PretrainCfg, provenance  # noqa: E402
from selfsup.data import load_all_corpora  # noqa: E402
from selfsup.folds import excluded_subjects  # noqa: E402
from selfsup.pretrain import pretrain  # noqa: E402
from selfsup.pretrain_pool import build_pool  # noqa: E402

POOLS = {
    "irds_only": ["IRDS"],
    "all_corpora": ["IRDS", "REHAB246", "UIPRMD"],   # KIMORE excluded (all eval targets)
}


def run(dummy: bool = False, smoke: bool = False, folds_json: str = "outputs/folds.json",
        out_root: str = "outputs", device: str | None = None) -> Dict[str, str]:
    epochs = 2 if smoke else 300
    batch = 16 if smoke else 128
    exclude = excluded_subjects(folds_json) if os.path.exists(folds_json) else set()
    corpora = load_all_corpora(dummy=dummy, n_dummy=40 if smoke else 50)

    ckpts: Dict[str, str] = {}
    for pool_name, include in POOLS.items():
        include = [c for c in include if c in corpora]
        if not include:
            print(f"[SKIP] pool '{pool_name}': none of its corpora available")
            continue
        x_pool, manifest = build_pool(corpora, exclude, include=include)
        if x_pool.shape[0] < batch:
            batch = max(2, x_pool.shape[0] // 2)
        out_dir = os.path.join(out_root, "ssl_pretrain", pool_name)
        for pretext in ("contrastive", "masked"):
            cfg = PretrainCfg(pretext=pretext, pool=pool_name, epochs=epochs,
                              batch_size=batch, d_model=64 if smoke else 128)
            ckpt = pretrain(x_pool, cfg, out_dir, device=device, log_every=1 if smoke else 10)
            # provenance sidecar next to the checkpoint
            prov = provenance(cfg, {"pool_manifest": manifest.__dict__})
            Path(ckpt + ".provenance.json").write_text(json.dumps(prov, indent=2, default=str))
            ckpts[f"{pool_name}:{pretext}"] = ckpt
    return ckpts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--folds_json", default="outputs/folds.json")
    ap.add_argument("--out_root", default="outputs")
    a = ap.parse_args()
    got = run(dummy=a.dummy, smoke=a.smoke, folds_json=a.folds_json, out_root=a.out_root)
    print("[run_pretrain] checkpoints:")
    for k, v in got.items():
        print(f"  {k} -> {v}")
