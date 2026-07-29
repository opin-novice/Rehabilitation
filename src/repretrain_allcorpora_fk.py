"""Re-pretrain the all-corpora SSL pool on FK-corrected UI-PRMD, then re-fine-tune and re-score.

The all-corpora pool is IRDS (1000) + REHAB246 (1057) + UI-PRMD (2000) = 4057 sequences.
UI-PRMD is 49% of it, and in the published run those 2000 sequences were the degenerate
bone-offset build (57.3% of coordinates constant). Re-scoring the existing checkpoints on
corrected data is not enough for this row: the *encoder itself* was pretrained on that
degenerate half. This is the one part of the UI-PRMD column that inference cannot repair.

Everything is written under a separate out_root so the published artifacts are untouched
and the two runs can be diffed.

Stages:
  1. pretrain contrastive + masked on the FK pool          -> {out_root}/ssl_pretrain/all_corpora
  2. fine-tune the 5 LOSO conditions from those encoders   -> {out_root}/{condition}
  3. zero-shot score every condition on FK UI-PRMD + REHAB246

Usage:
    python src/repretrain_allcorpora_fk.py [--n-folds 5] [--epochs 100]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/
from selfsup.config import PretrainCfg, provenance  # noqa: E402
from selfsup.data import load_all_corpora  # noqa: E402
from selfsup.folds import excluded_subjects  # noqa: E402
from selfsup.pretrain import pretrain  # noqa: E402
from selfsup.pretrain_pool import build_pool  # noqa: E402
from selfsup.run_all import run_loso_conditions  # noqa: E402

POOL = "all_corpora"
INCLUDE = ["IRDS", "REHAB246", "UIPRMD"]   # KIMORE excluded: every subject is a LOSO target
OUT_ROOT = "outputs_fkrepair"
POOLED_DIR = "KIMORE_pooled"
FOLDS_JSON = "outputs/folds.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--pretrain-epochs", type=int, default=300)
    ap.add_argument("--skip-pretrain", action="store_true")
    args = ap.parse_args()

    # UI-PRMD must resolve to the FK cache; make that explicit rather than relying on default.
    os.environ.setdefault("UIPRMD_CACHE", "outputs/validity_uiprmd")
    print(f"UIPRMD_CACHE={os.environ['UIPRMD_CACHE']}")

    corpora = load_all_corpora(dummy=False, pooled_dir=POOLED_DIR)
    include = [c for c in INCLUDE if c in corpora]
    exclude = excluded_subjects(FOLDS_JSON) if os.path.exists(FOLDS_JSON) else set()
    x_pool, manifest = build_pool(corpora, exclude, include=include)
    print(f"pool: n_total={manifest.n_total} per_corpus={manifest.per_corpus} hash={manifest.hash}")

    # Sanity gate: the whole point is that UI-PRMD's half is no longer degenerate.
    ui = corpora["UIPRMD"][0]
    const = float((ui.std(axis=1) <= 1e-9).mean())
    print(f"UI-PRMD constant-coordinate fraction in pool: {const:.4f} "
          f"(0.120 = padding floor, 0.573 = the published degenerate build)")
    if const > 0.2:
        print("[ABORT] pool still contains the degenerate UI-PRMD build.")
        return 2

    out_dir = os.path.join(OUT_ROOT, "ssl_pretrain", POOL)
    if not args.skip_pretrain:
        for pretext in ("contrastive", "masked"):
            cfg = PretrainCfg(pretext=pretext, pool=POOL, epochs=args.pretrain_epochs,
                              batch_size=128, d_model=128)
            print(f"\n=== pretrain[{pretext}] {args.pretrain_epochs} epochs on {x_pool.shape} ===")
            ckpt = pretrain(x_pool, cfg, out_dir, log_every=25)
            prov = provenance(cfg, {"pool_manifest": manifest.__dict__,
                                    "uiprmd_geometry": "forward_kinematics",
                                    "uiprmd_const_frac": const})
            Path(ckpt + ".provenance.json").write_text(json.dumps(prov, indent=2, default=str))
            print(f"  -> {ckpt}")

    print(f"\n=== fine-tune {args.n_folds} folds x 5 conditions ===")
    run_loso_conditions(OUT_ROOT, POOLED_DIR, pool=POOL, n_folds=args.n_folds,
                        epochs=args.epochs, batch_size=16, force=True)

    print("\n=== zero-shot on FK caches ===")
    from selfsup.zeroshot_eval import evaluate_zeroshot
    from selfsup.registry import CONDITIONS
    summary = {}
    for name, cond in CONDITIONS.items():
        cdir = os.path.join(OUT_ROOT, cond.out_subdir)
        for corpus in ("UIPRMD", "REHAB246"):
            r = evaluate_zeroshot(cdir, corpus)
            if r:
                summary[f"{cond.out_subdir}/{corpus}"] = {
                    "auroc": r["mean_auroc"], "pred_sd": r["mean_pred_sd"],
                    "degenerate": r["degenerate"], "naive": r["naive_auroc"]}
    out = os.path.join(OUT_ROOT, "allcorpora_fk_zeroshot.json")
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"\n-> {out}")
    for k, v in summary.items():
        print(f"  {k:34s} AUROC={v['auroc']:.4f} predSD={v['pred_sd']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
