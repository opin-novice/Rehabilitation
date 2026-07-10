"""Compare faithful reproduction vs proper ML results.

Loads summary.json from both experiments and generates a comparison table.

Usage:
    python src/compare_results.py --ex Kimore_ex1
    python src/compare_results.py --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Paper's reported results (Table 4, KIMORE)
PAPER_RESULTS = {
    "Kimore_ex1": {"mad": 0.071, "mape": 20.8, "rmse": 0.105},
    "Kimore_ex2": {"mad": 0.074, "mape": 19.2, "rmse": 0.108},
    "Kimore_ex3": {"mad": 0.069, "mape": 21.5, "rmse": 0.102},
    "Kimore_ex4": {"mad": 0.072, "mape": 20.1, "rmse": 0.106},
    "Kimore_ex5": {"mad": 0.070, "mape": 20.4, "rmse": 0.103},
}


def load_summary(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def print_comparison(ex: str, reproduce: dict | None, proper: dict | None):
    print(f"\n{'=' * 80}")
    print(f"  {ex}")
    print(f"{'=' * 80}")

    paper = PAPER_RESULTS.get(ex, {})

    # Header
    print(f"{'Method':<30} {'MAD':<15} {'MAPE':<15} {'RMSE':<15}")
    print(f"{'-' * 80}")

    # Paper
    if paper:
        print(f"{'Paper (reported)':<30} {paper['mad']:<15.4f} {paper['mape']:<15.1f}% {paper['rmse']:<15.4f}")

    # Faithful reproduction
    if reproduce:
        print(
            f"{'Faithful reproduction':<30} "
            f"{reproduce['mad_mean']:.4f} ± {reproduce['mad_std']:.4f}  "
            f"{reproduce['mape_mean']:.1f}% ± {reproduce['mape_std']:.1f}%  "
            f"{reproduce['rmse_mean']:.4f} ± {reproduce['rmse_std']:.4f}"
        )
    else:
        print(f"{'Faithful reproduction':<30} {'(not run)':<15}")

    # Proper ML
    if proper:
        print(
            f"{'Proper ML':<30} "
            f"{proper['mad_mean']:.4f} ± {proper['mad_std']:.4f}  "
            f"{proper['mape_mean']:.1f}% ± {proper['mape_std']:.1f}%  "
            f"{proper['rmse_mean']:.4f} ± {proper['rmse_std']:.4f}"
        )
    else:
        print(f"{'Proper ML':<30} {'(not run)':<15}")

    # Deltas
    if paper and reproduce:
        delta_mad = reproduce['mad_mean'] - paper['mad']
        delta_mape = reproduce['mape_mean'] - paper['mape']
        delta_rmse = reproduce['rmse_mean'] - paper['rmse']
        print(f"\n  Delta (reproduce - paper):")
        print(f"    MAD:  {delta_mad:+.4f}")
        print(f"    MAPE: {delta_mape:+.1f}%")
        print(f"    RMSE: {delta_rmse:+.4f}")

    if paper and proper:
        delta_mad = proper['mad_mean'] - paper['mad']
        delta_mape = proper['mape_mean'] - paper['mape']
        delta_rmse = proper['rmse_mean'] - paper['rmse']
        print(f"\n  Delta (proper - paper):")
        print(f"    MAD:  {delta_mad:+.4f}")
        print(f"    MAPE: {delta_mape:+.1f}%")
        print(f"    RMSE: {delta_rmse:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compare reproduction results.")
    parser.add_argument("--ex", default=None, help="Single exercise to compare")
    parser.add_argument("--all", action="store_true", help="Compare all exercises")
    parser.add_argument("--reproduce_dir", default="outputs/reproduce")
    parser.add_argument("--proper_dir", default="outputs/proper")
    args = parser.parse_args()

    exercises = ["Kimore_ex1", "Kimore_ex2", "Kimore_ex3", "Kimore_ex4", "Kimore_ex5"]

    if args.ex:
        exercises = [args.ex]
    elif not args.all:
        # Default: check which exercises have results
        exercises = [
            ex for ex in exercises
            if os.path.exists(os.path.join(args.reproduce_dir, ex, "summary.json"))
            or os.path.exists(os.path.join(args.proper_dir, ex, "summary.json"))
        ]
        if not exercises:
            print("No results found. Run train_reproduce.py and/or train_proper.py first.")
            sys.exit(1)

    print("=" * 80)
    print("  REPRODUCTION COMPARISON")
    print("=" * 80)

    for ex in exercises:
        reproduce = load_summary(os.path.join(args.reproduce_dir, ex, "summary.json"))
        proper = load_summary(os.path.join(args.proper_dir, ex, "summary.json"))
        print_comparison(ex, reproduce, proper)

    # Save comparison
    comparison = {}
    for ex in exercises:
        reproduce = load_summary(os.path.join(args.reproduce_dir, ex, "summary.json"))
        proper = load_summary(os.path.join(args.proper_dir, ex, "summary.json"))
        comparison[ex] = {
            "paper": PAPER_RESULTS.get(ex, {}),
            "reproduce": reproduce,
            "proper": proper,
        }

    out_path = os.path.join(args.reproduce_dir, "comparison.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved to {out_path}")


if __name__ == "__main__":
    main()
