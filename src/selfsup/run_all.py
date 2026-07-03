"""Resumable DAG orchestration (Layer 4, PRD R4/R8/R9), mirroring src/novelty/run_all.py.

DAG:
  folds -> build_pool(scale) -> pretrain{contrastive,masked} -> linear_probe(sanity)
        -> LOSO x {scratch,contrastive_lp,contrastive_ft,masked_lp,masked_ft}
        -> zeroshot{REHAB246,UIPRMD,IRDS} -> stats -> tables/figures

Resumability: a node is skipped when its output artifact already exists (unless
--force). --smoke runs the whole DAG on dummy data in minutes (acceptance test).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from selfsup.config import PretrainCfg, config_hash  # noqa: E402
from selfsup.registry import CONDITIONS  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_LOSO = os.path.join(PROJECT_ROOT, "src", "train_loso.py")


def loso_commands(out_root: str, pretrain_ckpts: dict) -> list[str]:
    cmds = []
    for cond in CONDITIONS.values():
        ckpt = pretrain_ckpts.get(cond.init_ckpt_key, "") if cond.init_ckpt_key else ""
        flags = ["python src/train_loso.py --model_type tcn", f"--out_dir {out_root}/{cond.out_subdir}"]
        if ckpt:
            flags.append(f"--init_ckpt {ckpt}")
        if cond.freeze_encoder:
            flags.append("--freeze_encoder")
        cmds.append("  " + " ".join(flags))
    return cmds


def plan(out_root: str = "outputs") -> None:
    print("=== Paper-2 SSL pipeline plan ===")
    for pretext in ("contrastive", "masked"):
        for pool in ("irds_only", "all_corpora"):
            print(f"pretrain[{pretext},{pool}] -> hash {config_hash(PretrainCfg(pretext=pretext, pool=pool))}")
    print("\nLOSO conditions:")
    for c in loso_commands(out_root, {"contrastive": "<contrastive_encoder.pt>", "masked": "<masked_encoder.pt>"}):
        print(c)


def run_loso_conditions(out_root: str, pooled_dir: str, pool: str,
                        n_folds: int, epochs: int, batch_size: int,
                        d_model: int = 128, force: bool = False) -> None:
    """PRD R4: launch the 5 fine-tuning conditions via train_loso.py --init_ckpt."""
    ckpts = {k: os.path.join(out_root, "ssl_pretrain", pool, f"{k}_encoder.pt")
             for k in ("contrastive", "masked")}
    for name, cond in CONDITIONS.items():
        out_dir = os.path.join(out_root, cond.out_subdir)
        result = os.path.join(out_dir, "loso_results.json")
        if os.path.exists(result):
            mtime = datetime.fromtimestamp(os.path.getmtime(result)).isoformat(timespec="seconds")
            if not force:
                print(f"[loso] WARNING reusing CACHED result for {name}: {result} "
                      f"(mtime {mtime}). Pass --force to recompute — a stale cache can "
                      f"silently fine-tune from the wrong or absent encoder.")
                continue
            shutil.rmtree(out_dir)
            print(f"[loso] --force: deleted stale {out_dir} (was mtime {mtime})")
        cmd = [sys.executable, TRAIN_LOSO, "--model_type", "tcn",
               "--pooled_dir", pooled_dir, "--out_dir", out_dir,
               "--n_folds", str(n_folds), "--epochs", str(epochs),
               "--d_model", str(d_model),
               "--batch_size", str(batch_size), "--patience", str(max(2, epochs))]
        if cond.init_ckpt_key and os.path.exists(ckpts[cond.init_ckpt_key]):
            cmd += ["--init_ckpt", ckpts[cond.init_ckpt_key]]
        if cond.freeze_encoder:
            cmd += ["--freeze_encoder"]
        print(f"[loso] RUN {name}: {' '.join(os.path.basename(c) for c in cmd[:3])} ...")
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def run_zeroshot(out_root: str, corpora: list[str], dummy: bool) -> None:
    from selfsup.zeroshot_eval import evaluate_zeroshot
    for cond in CONDITIONS.values():
        cdir = os.path.join(out_root, cond.out_subdir)
        for corpus in corpora:
            evaluate_zeroshot(cdir, corpus, dummy=dummy)


def smoke(out_root: str = "outputs") -> int:
    """PRD R9: full pipeline on dummy data. Acceptance test."""
    from selfsup.data import write_dummy_pooled
    from selfsup.folds import generate_folds
    from selfsup.run_pretrain import run as run_pretrain
    from selfsup.stats import compute_statistics
    from selfsup.make_tables import make_tables
    from selfsup.make_figures import make_figures

    t0 = time.time()
    smoke_root = os.path.join(out_root, "_smoke")
    pooled = write_dummy_pooled(os.path.join(smoke_root, "pooled"), seed=1)
    folds_json = os.path.join(smoke_root, "folds.json")

    def check(path, label):
        ok = os.path.exists(path)
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {path}")
        return ok

    ok = True
    generate_folds(pooled, n_folds=2, out_path=folds_json)
    ok &= check(folds_json, "folds")

    run_pretrain(dummy=True, smoke=True, folds_json=folds_json, out_root=smoke_root, device="cpu")
    ok &= check(os.path.join(smoke_root, "ssl_pretrain", "irds_only", "contrastive_encoder.pt"), "pretrain(contrastive)")
    ok &= check(os.path.join(smoke_root, "ssl_pretrain", "irds_only", "masked_encoder.pt"), "pretrain(masked)")

    run_loso_conditions(smoke_root, pooled, pool="irds_only", n_folds=2, epochs=2, batch_size=16, d_model=64)
    ok &= check(os.path.join(smoke_root, CONDITIONS["scratch"].out_subdir, "loso_results.json"), "LOSO(scratch)")
    ok &= check(os.path.join(smoke_root, CONDITIONS["contrastive_ft"].out_subdir, "loso_results.json"), "LOSO(contrastive_ft)")

    run_zeroshot(smoke_root, ["REHAB246", "UIPRMD", "IRDS"], dummy=True)
    ok &= check(os.path.join(smoke_root, CONDITIONS["scratch"].out_subdir, "zeroshot_IRDS.json"), "zeroshot")

    compute_statistics(smoke_root, n_boot=100)
    ok &= check(os.path.join(smoke_root, "ssl_results", "stats.json"), "stats")

    make_tables(smoke_root)
    ok &= check(os.path.join(smoke_root, "ssl_results", "tables", "table1_kimore_rho.csv"), "tables")
    make_figures(smoke_root)

    dt = time.time() - t0
    print(f"\n[SMOKE TEST {'PASSED' if ok else 'FAILED'}]  {dt:.1f}s  -> {smoke_root}")
    return 0 if ok else 1


def full(out_root: str, pooled_dir: str, pool: str, n_folds: int, force: bool = False) -> int:
    """Full run on real data (long). Requires cached corpora + KIMORE_pooled.

    force=True re-runs the LOSO conditions even if loso_results.json exists. Use it
    whenever the encoders changed (e.g. a fresh pretrain), otherwise the skip-on-exists
    guard silently reuses stale LOSO results trained against different/absent encoders.
    """
    from selfsup.folds import generate_folds
    from selfsup.run_pretrain import run as run_pretrain
    from selfsup.stats import compute_statistics
    from selfsup.make_tables import make_tables
    from selfsup.make_figures import make_figures

    folds_json = os.path.join(out_root, "folds.json")
    generate_folds(pooled_dir, n_folds=n_folds, out_path=folds_json)
    run_pretrain(dummy=False, smoke=False, folds_json=folds_json, out_root=out_root)
    run_loso_conditions(out_root, pooled_dir, pool=pool, n_folds=n_folds, epochs=100,
                        batch_size=16, force=force)
    run_zeroshot(out_root, ["REHAB246", "UIPRMD", "IRDS"], dummy=False)
    compute_statistics(out_root)
    make_tables(out_root)
    make_figures(out_root)
    print(f"[full] done -> {out_root}/ssl_results")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="outputs")
    ap.add_argument("--pooled_dir", default="KIMORE_pooled")
    ap.add_argument("--pool", default="irds_only", choices=["irds_only", "all_corpora"])
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--plan", action="store_true", help="Print DAG plan + LOSO commands.")
    ap.add_argument("--smoke", action="store_true", help="Run full pipeline on dummy data (fast).")
    ap.add_argument("--force", action="store_true",
                    help="Re-run LOSO conditions even if loso_results.json exists. Set this after "
                         "a fresh pretrain so conditions actually use the new encoders (the "
                         "skip-on-exists guard would otherwise reuse stale results).")
    args = ap.parse_args()
    if args.plan:
        plan(args.out_root)
    elif args.smoke:
        raise SystemExit(smoke(args.out_root))
    else:
        raise SystemExit(full(args.out_root, args.pooled_dir, args.pool, args.n_folds, force=args.force))
