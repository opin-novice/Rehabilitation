"""Background training launcher with crash-resilient resume.

Launches sequential training. If interrupted (power loss, etc.),
re-run with --resume to continue from where it left off.

Each exercise saves checkpoints every 100 epochs and per-rep results.
On resume, completed reps are skipped automatically.

Usage:
    python src/run_background.py              # fresh start
    python src/run_background.py --resume     # resume after interruption
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import os
import time

PYTHON = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".venv", "Scripts", "python.exe")
WORKDIR = os.path.dirname(os.path.dirname(__file__))
LOG_DIR = os.path.join(WORKDIR, "outputs", "reproduce", "logs")

# Training plan: (exercise, script, reps, epochs)
PHASES = [
    # Phase 1: Exercise 1 faithful reproduction (5 reps)
    ("faithful_e1", [
        ("Kimore_ex1", "train_reproduce.py", 5, 2000),
    ]),
    # Phase 2: Exercise 1 proper ML (5 reps)
    ("proper_e1", [
        ("Kimore_ex1", "train_proper.py", 5, 2000),
    ]),
    # Phase 3: Exercises 2-5 faithful (3 reps each)
    ("faithful_e25", [
        ("Kimore_ex2", "train_reproduce.py", 3, 2000),
        ("Kimore_ex3", "train_reproduce.py", 3, 2000),
        ("Kimore_ex4", "train_reproduce.py", 3, 2000),
        ("Kimore_ex5", "train_reproduce.py", 3, 2000),
    ]),
]


def run_task(ex, script, reps, epochs, resume):
    """Run one training task (exercise + script)."""
    log_file = os.path.join(LOG_DIR, f"{script.replace('.py','')}_{ex}.log")
    cmd = [
        PYTHON, os.path.join(WORKDIR, "src", script),
        "--ex", ex,
        "--reps", str(reps),
        "--epochs", str(epochs),
        "--batch_size", "1",
        "--lr", "0.0001",
        "--save_every", "100",
    ]
    if resume:
        cmd.append("--resume")

    print(f"  CMD: {' '.join(cmd)}")
    print(f"  Log: {log_file}")

    with open(log_file, "a") as log_f:
        log_f.write(f"\n{'='*60}\n")
        log_f.write(f"START: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write(f"{'='*60}\n")
        result = subprocess.run(cmd, cwd=WORKDIR, stdout=log_f, stderr=subprocess.STDOUT)

    print(f"  Done (rc={result.returncode})")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoints (skips completed reps)")
    parser.add_argument("--phase", type=int, default=0,
                        help="Start from this phase (0-indexed)")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Training Launcher")
    print(f"  Resume: {args.resume}")
    print(f"  Start phase: {args.phase}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    total_start = time.time()

    for pi, (phase_name, tasks) in enumerate(PHASES):
        if pi < args.phase:
            continue

        print(f"\n{'='*60}")
        print(f"PHASE {pi}: {phase_name}")
        print(f"{'='*60}")

        for ex, script, reps, epochs in tasks:
            print(f"\n  Task: {ex} ({script}) x{reps} reps, {epochs} epochs")
            rc = run_task(ex, script, reps, epochs, args.resume)
            if rc != 0:
                print(f"  WARNING: {ex} returned rc={rc}")

    elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Generate comparison
    print("\nGenerating comparison...")
    compare_script = os.path.join(WORKDIR, "src", "compare_results.py")
    if os.path.exists(compare_script):
        subprocess.run([PYTHON, compare_script], cwd=WORKDIR)


if __name__ == "__main__":
    main()
