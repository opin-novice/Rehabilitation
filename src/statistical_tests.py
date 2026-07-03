"""Pairwise statistical significance tests across LOSO experiments.

Reads loso_results.json from each experiment directory, extracts per-fold
RMSE and R² arrays, and runs:
  - Wilcoxon signed-rank test  (non-parametric, paired, n=5 folds)
  - Paired t-test              (parametric reference, n=5 folds)

Outputs:
  outputs/statistical_tests/pairwise_rmse.csv
  outputs/statistical_tests/pairwise_r2.csv
  outputs/statistical_tests/statistical_report.txt

Usage:
  python src/statistical_tests.py
  python src/statistical_tests.py --out_dir outputs/statistical_tests
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel

# ---------------------------------------------------------------------------
# Experiment registry — add new runs here
# ---------------------------------------------------------------------------

EXPERIMENTS: list[tuple[str, str]] = [
    ("Baseline (GroupKFold)",       "outputs/loso_pooled"),
    ("Exp A (Stratified)",          "outputs/loso_stratified_baseline"),
    ("Exp B (Aug d64 bs16)",        "outputs/loso_improved"),
    ("Exp C (Aug d64 bs32)",        "outputs/cfg3_batch32_reg"),
    ("Exp D (MT+UIPRMD d64)",       "outputs/loso_multitask_uiprmd"),
    ("Exp E (MT+UIPRMD d128)",      "outputs/loso_multitask_uiprmd_d128"),
    ("LSTM baseline",               "outputs/loso_lstm"),
    ("ST-GCN",                      "outputs/loso_stgcn"),
    ("GraphTransformer",            "outputs/loso_graph_transformer"),
]


def load_fold_metrics(directory: str) -> dict[str, list[float]] | None:
    path = os.path.join(directory, "loso_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    folds = data.get("folds", [])
    if not folds:
        return None
    return {
        "rmse": [f["rmse"] for f in folds],
        "r2":   [f["r2"]   for f in folds],
        "mae":  [f["mae"]  for f in folds],
    }


def safe_wilcoxon(a: list[float], b: list[float]) -> tuple[float, float]:
    diff = np.array(a) - np.array(b)
    if np.all(diff == 0):
        return float("nan"), 1.0
    try:
        stat, p = wilcoxon(a, b, alternative="two-sided")
        return float(stat), float(p)
    except ValueError:
        return float("nan"), float("nan")


def safe_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    try:
        stat, p = ttest_rel(a, b)
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


def build_pairwise_table(
    names: list[str],
    metrics: dict[str, list[float]],
    metric_key: str,
) -> pd.DataFrame:
    """Returns a DataFrame with one row per experiment pair."""
    rows = []
    for (i, name_a), (j, name_b) in itertools.combinations(enumerate(names), 2):
        a = metrics[name_a][metric_key]
        b = metrics[name_b][metric_key]
        mean_a = float(np.mean(a))
        mean_b = float(np.mean(b))
        w_stat, w_p = safe_wilcoxon(a, b)
        t_stat, t_p = safe_ttest(a, b)
        rows.append({
            "Exp A":     name_a,
            "Exp B":     name_b,
            f"Mean {metric_key.upper()} A": round(mean_a, 4),
            f"Mean {metric_key.upper()} B": round(mean_b, 4),
            "Delta (A-B)":  round(mean_a - mean_b, 4),
            "Wilcoxon W":   round(w_stat, 2) if not np.isnan(w_stat) else "n/a",
            "Wilcoxon p":   round(w_p,   4) if not np.isnan(w_p)   else "n/a",
            "t-stat":       round(t_stat, 3) if not np.isnan(t_stat) else "n/a",
            "t-test p":     round(t_p,   4) if not np.isnan(t_p)   else "n/a",
            "Sig (p<0.05)": "YES" if (not np.isnan(w_p) and w_p < 0.05) else "no",
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs/statistical_tests")
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Load available experiments
    available: dict[str, dict] = {}
    for name, directory in EXPERIMENTS:
        m = load_fold_metrics(directory)
        if m is not None:
            available[name] = m
        else:
            print(f"[SKIP] {name} — {directory}/loso_results.json not found")

    if len(available) < 2:
        print("Need at least 2 experiment results to compare. Run more experiments first.")
        return

    names = list(available.keys())
    print(f"\nLoaded {len(names)} experiments: {names}\n")

    # Summary table
    print(f"{'Experiment':<35} {'Mean RMSE':>10} {'Std RMSE':>9} {'Mean R2':>9} {'Std R2':>8} {'Mean Pearson':>13}")
    print("-" * 90)
    for name in names:
        m = available[name]
        raw_folds    = _load_raw_folds(name)
        pearson_vals = [f.get("pearson", float("nan")) for f in raw_folds]
        valid_pearsons = [v for v in pearson_vals if not np.isnan(v)]
        mean_pearson = float(np.mean(valid_pearsons)) if valid_pearsons else float("nan")
        print(f"  {name:<33} {np.mean(m['rmse']):>10.4f} {np.std(m['rmse']):>9.4f} "
              f"{np.mean(m['r2']):>9.4f} {np.std(m['r2']):>8.4f} {mean_pearson:>13.4f}")

    # Pairwise RMSE table
    rmse_df = build_pairwise_table(names, available, "rmse")
    rmse_path = os.path.join(args.out_dir, "pairwise_rmse.csv")
    rmse_df.to_csv(rmse_path, index=False)
    print(f"\nPairwise RMSE tests saved -> {rmse_path}")

    # Pairwise R² table
    r2_df = build_pairwise_table(names, available, "r2")
    r2_path = os.path.join(args.out_dir, "pairwise_r2.csv")
    r2_df.to_csv(r2_path, index=False)
    print(f"Pairwise R2 tests saved   -> {r2_path}")

    # Text report
    report_path = os.path.join(args.out_dir, "statistical_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Statistical Significance Report - KIMORE 5-fold LOSO CV\n")
        f.write("=" * 70 + "\n\n")
        f.write("Note: n=5 folds per experiment. Wilcoxon signed-rank (two-sided)\n")
        f.write("is the primary test (non-parametric, appropriate for n=5).\n")
        f.write("Paired t-test shown as a parametric reference only.\n\n")

        f.write("RMSE comparisons (lower RMSE = better):\n")
        f.write("-" * 70 + "\n")
        for _, row in rmse_df.iterrows():
            direction = "A<B (A better)" if row["Delta (A-B)"] < 0 else "A>B (B better)"
            f.write(f"  {row['Exp A']} vs {row['Exp B']}\n")
            f.write(f"    RMSE: {row['Mean RMSE A']:.4f} vs {row['Mean RMSE B']:.4f}  "
                    f"Delta={row['Delta (A-B)']:.4f}  ({direction})\n")
            f.write(f"    Wilcoxon p={row['Wilcoxon p']}  t-test p={row['t-test p']}  "
                    f"Significant: {row['Sig (p<0.05)']}\n\n")

        f.write("\nR2 comparisons (higher R2 = better):\n")
        f.write("-" * 70 + "\n")
        for _, row in r2_df.iterrows():
            direction = "A>B (A better)" if row["Delta (A-B)"] > 0 else "A<B (B better)"
            f.write(f"  {row['Exp A']} vs {row['Exp B']}\n")
            f.write(f"    R2: {row['Mean R2 A']:.4f} vs {row['Mean R2 B']:.4f}  "
                    f"Delta={row['Delta (A-B)']:.4f}  ({direction})\n")
            f.write(f"    Wilcoxon p={row['Wilcoxon p']}  t-test p={row['t-test p']}  "
                    f"Significant: {row['Sig (p<0.05)']}\n\n")

    print(f"Full text report saved    -> {report_path}")


def _load_raw_folds(name: str) -> list[dict]:
    for exp_name, directory in EXPERIMENTS:
        if exp_name == name:
            path = os.path.join(directory, "loso_results.json")
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f).get("folds", [])
    return []


if __name__ == "__main__":
    main()
