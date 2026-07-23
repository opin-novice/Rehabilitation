"""Orchestrate the heavy reviewer round-3 experiments (C1 eval, C2 train+eval, C3 DANN).

Waits for the ST-GCN LOSO training (launched separately) to finish, then runs each
GPU job sequentially. Resumable: every step is skipped when its output already exists.
Results land in outputs/reviewer_round3/.

Run (background):  python src/run_reviewer_round3_heavy.py
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
STGCN_DIR = "archive/legacy_results/kimore_loso_78fold_stgcn"
BONEVEC_DIR = "archive/legacy_results/kimore_loso_78fold_bonevec"
OUT = "outputs/reviewer_round3"


def sh(cmd):
    print(f"\n[heavy] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def wait_for_stgcn(timeout_s=7200, poll=60):
    """Wait until the ST-GCN LOSO run has written its final summary or >=77 folds."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        done = os.path.exists(os.path.join(STGCN_DIR, "loso_results.json"))
        folds = len(glob.glob(os.path.join(STGCN_DIR, "fold_*", "best_model.pt")))
        if done or folds >= 77:
            print(f"[heavy] ST-GCN ready (loso_results={done}, folds={folds})", flush=True)
            return True
        print(f"[heavy] waiting for ST-GCN... folds={folds}", flush=True)
        time.sleep(poll)
    print("[heavy] TIMEOUT waiting for ST-GCN; proceeding with whatever exists", flush=True)
    return False


def main():
    os.makedirs(OUT, exist_ok=True)

    # --- C1: ST-GCN zero-shot + AdaBN + sensor-ID probe ---
    wait_for_stgcn()
    if not os.path.exists(os.path.join(OUT, "c1_stgcn.json")):
        sh([PY, "src/reviewer_round3_c1_stgcn_eval.py"])

    # --- C2: bone-vector (joint-angle) input: train TCN LOSO, then eval ---
    if not os.path.exists("KIMORE_pooled_bonevec/Train_X.csv"):
        sh([PY, "src/reviewer_round3_c2.py", "build"])
    if len(glob.glob(os.path.join(BONEVEC_DIR, "fold_*", "best_model.pt"))) < 77:
        sh([PY, "src/train_loso.py", "--model_type", "tcn", "--loso", "--resume",
            "--pooled_dir", "KIMORE_pooled_bonevec", "--out_dir", BONEVEC_DIR,
            "--epochs", "100", "--batch_size", "16", "--patience", "100", "--d_model", "128"])
    if not os.path.exists(os.path.join(OUT, "c2_bonevec.json")):
        sh([PY, "src/reviewer_round3_c2.py", "eval"])

    # --- C3: DANN domain-adversarial baseline ---
    if not os.path.exists(os.path.join(OUT, "c3_dann.json")):
        sh([PY, "src/reviewer_round3_c3_dann.py", "--epochs", "60"])

    print("\n[heavy] ALL DONE -> outputs/reviewer_round3/", flush=True)


if __name__ == "__main__":
    main()
