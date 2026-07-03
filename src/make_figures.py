"""Generate all 5 paper figures from existing CSV outputs.

Usage:
    python src/make_figures.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300

# Journal-quality, consistent styling across all figures.
plt.rcParams.update({
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
})

# Single Okabe-Ito colorblind-friendly palette, applied consistently everywhere.
MODEL_COLORS = {
    "Ridge":                      "#0072B2",  # blue
    "Ridge (stat. features)":     "#0072B2",  # blue (CSV alias)
    "LSTM baseline":              "#009E73",  # bluish green
    "ST-GCN":                     "#D55E00",  # vermillion
    "GraphTransformer":           "#CC79A7",  # reddish purple
    "GraphTransformer (no bias)": "#E69F00",  # orange
    "TCN":                        "#F0E442",  # yellow
    "SCT":                        "#56B4E9",  # sky blue
    "Exp E (Transformer)":        "#000000",  # black
}

EXERCISE_LABELS = ["k01", "k02", "k03", "k04", "k05"]


def _load_spearman() -> pd.DataFrame:
    path = "outputs/sample_stats/per_exercise_spearman.csv"
    df = pd.read_csv(path)
    ours = df[df["Protocol"].str.contains("Stratified LOSO", na=False)].copy()
    ours = ours[~ours["Model"].str.contains("Mean-prediction", na=False)]
    return ours


def _parse_ci(ci_str: str) -> tuple[float, float] | None:
    if not isinstance(ci_str, str) or not ci_str.startswith("["):
        return None
    nums = re.findall(r"[-0-9.]+", ci_str)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None


# ---------------------------------------------------------------------------
# F1: Per-exercise Spearman rho — grouped bar chart
# ---------------------------------------------------------------------------
def fig1_kimore_per_exercise() -> str:
    df = _load_spearman()
    models = df["Model"].tolist()
    x = np.arange(len(EXERCISE_LABELS))
    n_models = len(models)
    bar_width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[e] for e in EXERCISE_LABELS]
        offset = (i - n_models / 2 + 0.5) * bar_width
        color = MODEL_COLORS.get(row["Model"], "#333333")
        ax.bar(x + offset, vals, bar_width, label=row["Model"], color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(EXERCISE_LABELS)
    ax.set_ylabel("Spearman rho")
    ax.set_title("Per-Exercise Spearman rho (pooled OOF, Stratified LOSO)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = str(OUT_DIR / "fig1_kimore_per_exercise.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  F1 -> {path}")
    return path


# ---------------------------------------------------------------------------
# F2: Mean rho with 95% CI — horizontal bar
# ---------------------------------------------------------------------------
def fig2_model_mean_rho() -> str:
    df = _load_spearman()
    df = df.sort_values("Mean rho", ascending=True)

    models = df["Model"].tolist()
    means = df["Mean rho"].tolist()

    err_lo, err_hi = [], []
    for i, (_, row) in enumerate(df.iterrows()):
        ci = _parse_ci(str(row.get("Mean rho 95% CI", "")))
        if ci:
            err_lo.append(means[i] - ci[0])
            err_hi.append(ci[1] - means[i])
        else:
            err_lo.append(0)
            err_hi.append(0)
    err_lo = np.array(err_lo)
    err_hi = np.array(err_hi)

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(models))
    colors = [MODEL_COLORS.get(m, "#333333") for m in models]
    bars = ax.barh(y_pos, means, xerr=[err_lo, err_hi], color=colors, height=0.6,
                   ecolor="black", capsize=3)

    # Ridge baseline reference line
    ridge_val = df.loc[df["Model"] == "Ridge (stat. features)", "Mean rho"].values
    if len(ridge_val):
        ax.axvline(ridge_val[0], color="#7570b3", linestyle="--", linewidth=1.2,
                   label=f"Ridge baseline ({ridge_val[0]:.3f})")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("Mean Spearman rho (pooled OOF)")
    ax.set_title("Model Comparison — Mean Spearman rho with 95% CI")
    ax.legend(fontsize=9, loc="lower right")
    ax.axvline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    path = str(OUT_DIR / "fig2_model_mean_rho.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  F2 -> {path}")
    return path


# ---------------------------------------------------------------------------
# F3: Protocol inflation bar chart
# ---------------------------------------------------------------------------
def fig3_protocol_inflation() -> str:
    path_json = "outputs/protocol_inflation/protocol_inflation.json"
    with open(path_json, encoding="utf-8") as f:
        data = json.load(f)

    labels = ["Stratified LOSO\n(ours)", "GroupKFold\n(random subjects)", "Random KFold\n(sample-level)"]
    values = [
        data["A_stratified_loso"]["mean_rho_pooled"],
        data["B_group_kfold"]["mean_rho_pooled"],
        data["C_random_kfold"]["mean_rho_pooled"],
    ]
    deltas = [0, data["inflation_B_vs_A"], data["inflation_C_vs_A"]]

    colors = ["#1b9e77", "#e6ab02", "#d95f02"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="black", linewidth=0.8)

    for i, (bar, val, delta) in enumerate(zip(bars, values, deltas)):
        label = f"{val:.3f}"
        if i > 0:
            label += f"\n(+{delta:.3f})"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Mean Spearman rho (pooled OOF)")
    ax.set_title("Protocol Inflation — Identical Ridge Features")
    ax.set_ylim(0, max(values) + 0.08)
    fig.tight_layout()
    path = str(OUT_DIR / "fig3_protocol_inflation.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  F3 -> {path}")
    return path


# ---------------------------------------------------------------------------
# F4: IRDS consistency — Kendall W + cross-exercise rho
# ---------------------------------------------------------------------------
def fig4_irds_consistency() -> str:
    df = pd.read_csv("outputs/irds_eval/irds_reliability.csv")
    df = df.sort_values("Kendall_W", ascending=True)

    models = df["model"].tolist()
    w_vals = df["Kendall_W"].tolist()
    rho_vals = df["mean_cross_ex_rho"].tolist()

    y_pos = np.arange(len(models))
    height = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_w = ax.barh(y_pos + height / 2, w_vals, height, label="Kendall W",
                     color="#1b9e77", edgecolor="black", linewidth=0.6)
    bars_r = ax.barh(y_pos - height / 2, rho_vals, height, label="Cross-exercise rho",
                     color="#d95f02", edgecolor="black", linewidth=0.6)

    for bar, val in zip(bars_w, w_vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    for bar, val in zip(bars_r, rho_vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel("Coefficient value")
    ax.set_title("IRDS Cross-Exercise Rank Consistency (zero-shot)")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    path = str(OUT_DIR / "fig4_irds_consistency.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  F4 -> {path}")
    return path


# ---------------------------------------------------------------------------
# F5: KIMORE rho vs IRDS Kendall W scatter
# ---------------------------------------------------------------------------
def fig5_kimore_vs_irds() -> str:
    spearman_df = _load_spearman()
    irds_df = pd.read_csv("outputs/irds_eval/irds_reliability.csv")

    merged = spearman_df.merge(irds_df, left_on="Model", right_on="model", how="inner")
    if merged.empty:
        print("  [SKIP F5] No overlapping models found between KIMORE and IRDS CSVs.")
        return ""

    x = merged["Mean rho"].values
    y = merged["Kendall_W"].values
    labels = merged["Model"].tolist()

    from scipy.stats import spearmanr as sr
    r, p = sr(x, y)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = [MODEL_COLORS.get(m, "#333333") for m in labels]
    ax.scatter(x, y, c=colors, s=80, edgecolors="black", linewidths=0.6, zorder=5)

    for i, label in enumerate(labels):
        ax.annotate(label, (x[i], y[i]), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)

    n_models = len(labels)
    ax.set_xlabel("KIMORE Mean Spearman rho")
    ax.set_ylabel("IRDS Kendall W")
    # Title states r and p explicitly; subtitle flags the result as exploratory.
    ax.set_title(
        "KIMORE vs IRDS: Exploratory Dissociation\n"
        f"Spearman r={r:.2f}, p={p:.2f}  —  "
        f"N={n_models} models; correlation NOT significant (exploratory)"
    )
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    # Caption: no significant rank-order relationship, but divergent per-model behaviour.
    caption = (
        f"No significant rank-order relationship (r={r:.2f}, p={p:.2f}) — "
        "individual models nonetheless show divergent KIMORE-vs-IRDS behaviour."
    )
    fig.text(0.5, 0.005, caption, ha="center", va="bottom",
             fontsize=8, style="italic", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = str(OUT_DIR / "fig5_kimore_vs_irds.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  F5 -> {path}")
    print(f"  F5 stats: Spearman r={r:.3f}, p={p:.3f}, N={n_models} models "
          "(NOT significant; exploratory)")
    return path


# ---------------------------------------------------------------------------
# F6: Pairwise effect-size heatmap (rank-biserial r) with FWER markers
# ---------------------------------------------------------------------------
def fig6_pairwise_effect_heatmap() -> str:
    path_csv = "outputs/sample_stats/pairwise_sample_level.csv"
    if not os.path.exists(path_csv):
        print(f"  [SKIP F6] {path_csv} not found.")
        return ""
    df = pd.read_csv(path_csv)

    # Ordered, de-duplicated list of models across both columns.
    models = list(dict.fromkeys(df["Model A"].tolist() + df["Model B"].tolist()))
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)

    mat = np.zeros((n, n), dtype=float)          # Effect r (diagonal = 0)
    fwer = np.zeros((n, n), dtype=bool)          # FWER-significant pairs
    for _, row in df.iterrows():
        a, b = idx[row["Model A"]], idx[row["Model B"]]
        r = float(row["Effect r"])
        mat[a, b] = mat[b, a] = r
        sig = str(row.get("Sig (FWER)", "no")).strip().lower() == "yes"
        fwer[a, b] = fwer[b, a] = sig

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="cividis", vmin=0.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Rank-biserial r (|effect size|)")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(models, fontsize=8)

    # Annotate each cell; append '*' where the pair survives FWER correction.
    thresh = mat.max() * 0.6 if mat.max() > 0 else 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            star = "*" if fwer[i, j] else ""
            txt = f"{mat[i, j]:.2f}{star}"
            color = "black" if mat[i, j] > thresh else "white"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_title("Pairwise effect size (rank-biserial r) — KIMORE abs-error\n"
                 "* = significant after Holm-Bonferroni (FWER)")
    fig.tight_layout()
    path = str(OUT_DIR / "fig6_pairwise_effect_heatmap.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  F6 -> {path}")
    sig_pairs = [(df.loc[k, "Model A"], df.loc[k, "Model B"])
                 for k in df.index
                 if str(df.loc[k, "Sig (FWER)"]).strip().lower() == "yes"]
    print(f"  F6 FWER-significant pairs: {sig_pairs}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures...")
    paths = []
    for fn in [fig1_kimore_per_exercise, fig2_model_mean_rho,
               fig3_protocol_inflation, fig4_irds_consistency,
               fig5_kimore_vs_irds, fig6_pairwise_effect_heatmap]:
        try:
            p = fn()
            paths.append(p)
        except Exception as e:
            print(f"  [ERROR] {fn.__name__}: {e}")
    print(f"\nDone. {len([p for p in paths if p])} figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
