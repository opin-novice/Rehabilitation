"""Generate ablation bar chart comparing all LOSO experiments.

Usage:
  python src/plot_ablation.py --out_dir outputs/ablation_summary.png
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

# Ordered experiment list: (label, directory, color)
EXPERIMENTS = [
    ("Baseline\n(GroupKFold)",       "outputs/loso_pooled",               "#4C72B0"),
    ("Exp A\n(Stratified)",          "outputs/loso_stratified_baseline",  "#4C72B0"),
    ("Exp B\n(+Augment d64)",        "outputs/loso_improved",             "#55A868"),
    ("Exp C\n(+Batch32)",            "outputs/cfg3_batch32_reg",          "#55A868"),
    ("Exp D\n(+Multitask\n+UIPRMD d64)", "outputs/loso_multitask_uiprmd",   "#C44E52"),
    ("Exp E\n(+Multitask\n+UIPRMD d128)","outputs/loso_multitask_uiprmd_d128","#C44E52"),
]


def load_results(directory: str) -> dict | None:
    path = os.path.join(directory, "loso_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", default="outputs/ablation_summary.png")
    args = parser.parse_args()

    labels, rmse_means, rmse_stds, r2_means, r2_stds, colors = [], [], [], [], [], []

    for label, directory, color in EXPERIMENTS:
        results = load_results(directory)
        if results is None:
            print(f"[WARN] Missing {directory}/loso_results.json — skipping")
            continue
        labels.append(label)
        rmse_means.append(results["mean_rmse"])
        rmse_stds.append(results["std_rmse"])
        r2_means.append(results["mean_r2"])
        r2_stds.append(results["std_r2"])
        colors.append(color)

    x = np.arange(len(labels))
    width = 0.55

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # RMSE bar chart (lower is better)
    ax = axes[0]
    bars = ax.bar(x, rmse_means, width, yerr=rmse_stds, capsize=4,
                  color=colors, edgecolor="white", linewidth=0.8, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean RMSE (± std, 5-fold LOSO)", fontsize=10)
    ax.set_title("RMSE by Experiment (lower = better)", fontsize=11)
    ax.set_ylim(0, max(rmse_means) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, rmse_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # R² bar chart (higher is better)
    ax = axes[1]
    bars = ax.bar(x, r2_means, width, yerr=r2_stds, capsize=4,
                  color=colors, edgecolor="white", linewidth=0.8, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean R² (± std, 5-fold LOSO)", fontsize=10)
    ax.set_title("R² by Experiment (higher = better)", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, r2_means):
        offset = 0.005 if val >= 0 else -0.015
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", alpha=0.88, label="CV Split experiments"),
        Patch(facecolor="#55A868", alpha=0.88, label="Augmentation experiments"),
        Patch(facecolor="#C44E52", alpha=0.88, label="Multitask + UI-PRMD experiments"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Ablation Study: KIMORE 5-fold Stratified LOSO Cross-Validation", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_path) if os.path.dirname(args.out_path) else ".", exist_ok=True)
    fig.savefig(args.out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ablation chart -> {args.out_path}")


if __name__ == "__main__":
    main()
