"""Sample-level statistical significance analysis using pooled OOF predictions.

Computes the metrics the KIMORE literature actually reports (per-exercise
Spearman rho pooled across all subjects) and runs significance tests at N~305
rather than the underpowered N=5 fold-level Wilcoxon.

Three outputs:
  1. per_exercise_spearman.csv  — directly comparable to Karlov et al. 2024 table
  2. vs_mean_baseline.csv       — every model vs mean-prediction sanity check
  3. pairwise_sample_level.csv  — all model pairs, Wilcoxon + t-test at N~305
  4. sample_stats_report.txt    — full narrative text report

Usage:
  python src/sample_level_stats.py
  python src/sample_level_stats.py --out_dir outputs/sample_stats
"""
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon, ttest_rel

# ---------------------------------------------------------------------------
# Registry — only experiments with real KIMORE val predictions
# ---------------------------------------------------------------------------

EXPERIMENTS: list[tuple[str, str]] = [
    ("Ridge (stat. features)",   "outputs/ridge_baseline"),
    ("LSTM baseline",            "outputs/loso_lstm"),
    ("ST-GCN",                   "outputs/loso_stgcn"),
    ("GraphTransformer",         "outputs/loso_graph_transformer"),
    ("GraphTransformer (no bias)","outputs/loso_graph_transformer_no_bias"),
    ("TCN",                      "outputs/loso_tcn"),
    ("SCT",                      "outputs/loso_sct"),
    ("Exp E (Transformer)",      "outputs/loso_multitask_uiprmd_d128"),
]

# Published literature baselines (Spearman rho per exercise, k01..k05)
LITERATURE = [
    {"Model": "Karlov et al. 2024 (SOTA, contrastive+IRDS)",
     "k01": 0.79, "k02": 0.62, "k03": 0.77, "k04": 0.80, "k05": 0.74,
     "Mean rho": 0.744, "Protocol": "5-fold CV, transfer from IRDS"},
    {"Model": "Abedi et al. 2023 (cross-modal LSTM)",
     "k01": 0.76, "k02": 0.61, "k03": 0.73, "k04": 0.54, "k05": 0.67,
     "Mean rho": 0.662, "Protocol": "5-fold CV"},
    {"Model": "Guo & Khan 2021 (handcrafted + ML)",
     "k01": 0.55, "k02": 0.64, "k03": 0.63, "k04": 0.37, "k05": 0.42,
     "Mean rho": 0.522, "Protocol": "5-fold CV (implied)"},
    {"Model": "Capecci et al. 2019 (rule-based, original paper)",
     "k01": 0.44, "k02": 0.41, "k03": 0.46, "k04": 0.62, "k05": 0.30,
     "Mean rho": 0.446, "Protocol": "Dataset paper baseline"},
]

EXERCISE_NAMES = {
    0: "Trunk Lateral Flex.",
    1: "Trunk Forward Flex.",
    2: "Trunk Rotation",
    3: "Hip Abduction",
    4: "Hip Circumduction",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_oof(exp_dir: str) -> pd.DataFrame | None:
    """Load combined OOF CSV; rebuild from fold files if combined missing."""
    combined_path = os.path.join(exp_dir, "oof_predictions_all.csv")
    if os.path.exists(combined_path):
        return pd.read_csv(combined_path)

    fold_dfs = []
    for i in range(5):
        fp = os.path.join(exp_dir, f"fold_{i}", "oof_predictions.csv")
        if os.path.exists(fp):
            fold_dfs.append(pd.read_csv(fp))
    if not fold_dfs:
        return None
    df = pd.concat(fold_dfs, ignore_index=True)
    df.to_csv(combined_path, index=False)
    return df


def per_exercise_spearman(df: pd.DataFrame) -> dict[int, dict]:
    """Spearman rho per KIMORE exercise across all pooled OOF subjects."""
    results = {}
    for eid in range(5):   # KIMORE exercises only (IDs 0-4)
        sub = df[df["exercise_id"] == eid]
        if len(sub) < 5:
            continue
        rho, p = spearmanr(sub["y_true"].values, sub["y_pred"].values)
        results[eid] = {
            "n":        len(sub),
            "spearman": float(rho),
            "p_value":  float(p),
        }
    return results


def mean_prediction_baseline(df: pd.DataFrame) -> pd.Series:
    """LOO mean baseline: predict mean of all other folds' y_true per fold."""
    baseline = np.zeros(len(df), dtype=np.float64)
    df_reset = df.reset_index(drop=True)
    for fold in df_reset["fold"].unique():
        is_val   = df_reset["fold"] == fold
        is_train = ~is_val
        fold_mean = df_reset.loc[is_train, "y_true"].mean()
        baseline[is_val.values] = fold_mean
    return pd.Series(baseline, name="baseline_pred")


def safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = a - b
    if np.all(diff == 0):
        return float("nan"), 1.0
    try:
        stat, p = wilcoxon(a, b, alternative="two-sided")
        return float(stat), float(p)
    except ValueError:
        return float("nan"), float("nan")


def safe_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    try:
        stat, p = ttest_rel(a, b)
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


def rank_biserial_r(w_stat: float, n: int) -> float:
    """Effect size r for Wilcoxon signed-rank: r = 1 - 2W / (n*(n+1)/2)."""
    t_max = n * (n + 1) / 2
    if t_max == 0 or np.isnan(w_stat):
        return float("nan")
    return float(1.0 - 2.0 * w_stat / t_max)


def bootstrap_mean_rho_ci(
    df: pd.DataFrame, n_boot: int = 2000, seed: int = 42,
) -> tuple[float, float]:
    """95% CI for the mean-of-per-exercise Spearman rho (matches the point estimate).

    Each bootstrap iteration resamples rows *within each exercise* (stratified), recomputes
    per-exercise Spearman, then averages across exercises. This matches how the point estimate
    `Mean rho` is computed, unlike a single pooled all-exercise Spearman.
    """
    rng = np.random.default_rng(seed)
    eids = [e for e in range(5) if (df["exercise_id"] == e).sum() >= 5]
    sub = {e: df[df["exercise_id"] == e] for e in eids}
    means: list[float] = []
    for _ in range(n_boot):
        rhos = []
        for e in eids:
            s = sub[e]
            idx = rng.integers(0, len(s), len(s))
            yt = s["y_true"].values[idx]
            yp = s["y_pred"].values[idx]
            r, _ = spearmanr(yt, yp)
            if not np.isnan(r):
                rhos.append(r)
        if rhos:
            means.append(float(np.mean(rhos)))
    if not means:
        return float("nan"), float("nan")
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def bonferroni_holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values. Reject when adj_p <= 0.05."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adj = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        corrected = p * (m - rank)
        running_max = max(running_max, corrected)
        adj[orig_idx] = min(running_max, 1.0)
    return adj

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs/sample_stats")
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ── Load OOF data ────────────────────────────────────────────────────────
    oof: dict[str, pd.DataFrame] = {}
    for name, exp_dir in EXPERIMENTS:
        df = load_oof(exp_dir)
        if df is None:
            print(f"[SKIP] {name} — run generate_oof.py first")
        else:
            # Filter to KIMORE exercises only (IDs 0-4); UI-PRMD never in val set
            df = df[df["exercise_id"] < 5].reset_index(drop=True)
            oof[name] = df
            print(f"[OK]   {name}: {len(df)} KIMORE samples across {df['fold'].nunique()} folds")

    if not oof:
        print("\nNo OOF data available. Run: python src/generate_oof.py --all")
        return

    # ── 1. Per-exercise Spearman (matches published metric) ──────────────────
    print("\n--- Per-Exercise Spearman rho (pooled OOF, comparable to literature) ---")
    our_rows = []
    for name, df in oof.items():
        ex_stats = per_exercise_spearman(df)
        rho_vals = [v["spearman"] for v in ex_stats.values()]
        mean_rho = float(np.mean(rho_vals)) if rho_vals else float("nan")
        # Bootstrap 95% CI on the mean rho
        if not np.isnan(mean_rho) and len(df) > 10:
            ci_lo, ci_hi = bootstrap_mean_rho_ci(df)
            ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]"
        else:
            ci_str = ""
        row = {
            "Model":    name,
            "Protocol": "5-fold Stratified LOSO (ours)",
        }
        for eid in range(5):
            key = f"k0{eid+1}"
            row[key] = round(ex_stats[eid]["spearman"], 3) if eid in ex_stats else float("nan")
        row["Mean rho"] = round(mean_rho, 3)
        row["Mean rho 95% CI"] = ci_str
        our_rows.append(row)
        print(f"  {name}: " + "  ".join(
            f"k0{eid+1}={ex_stats.get(eid,{}).get('spearman', float('nan')):.3f}"
            for eid in range(5)
        ) + f"  mean={mean_rho:.3f}  CI={ci_str}")

    # Add mean-prediction baseline row (computed from first model's OOF)
    first_name = list(oof.keys())[0]
    first_df = oof[first_name]
    baseline_pred = mean_prediction_baseline(first_df)
    base_ex = {}
    for eid in range(5):
        mask = first_df["exercise_id"] == eid
        if mask.sum() >= 5:
            r, _ = spearmanr(first_df.loc[mask, "y_true"].values, baseline_pred[mask].values)
            base_ex[eid] = r if not np.isnan(r) else float("nan")
    base_rhos = [v for v in base_ex.values() if not np.isnan(v)]
    base_mean = float(np.mean(base_rhos)) if base_rhos else float("nan")
    base_row = {
        "Model": "Mean-prediction baseline",
        "Protocol": "5-fold Stratified LOSO (ours)",
    }
    for eid in range(5):
        base_row[f"k0{eid+1}"] = round(base_ex.get(eid, float("nan")), 3)
    base_row["Mean rho"] = round(base_mean, 3)
    base_row["Mean rho 95% CI"] = ""
    our_rows.append(base_row)
    print(f"  Mean-prediction baseline: mean rho={base_mean:.3f}")

    lit_rows = [
        {**r, "Protocol": r.get("Protocol", "")} for r in LITERATURE
    ]
    spearman_df = pd.DataFrame(our_rows + lit_rows)
    # Reorder columns
    col_order = ["Model", "Protocol", "k01", "k02", "k03", "k04", "k05", "Mean rho", "Mean rho 95% CI"]
    spearman_df = spearman_df.reindex(columns=col_order)
    sp_path = os.path.join(args.out_dir, "per_exercise_spearman.csv")
    spearman_df.to_csv(sp_path, index=False)
    print(f"\nSaved -> {sp_path}")

    # ── 2. vs Mean-Prediction Baseline ───────────────────────────────────────
    print("\n--- vs Mean-Prediction Baseline ---")
    baseline_rows = []
    for name, df in oof.items():
        baseline_pred = mean_prediction_baseline(df)
        tgts  = df["y_true"].values
        preds = df["y_pred"].values
        base  = baseline_pred.values

        rmse_m = float(np.sqrt(np.mean((tgts - preds) ** 2)))
        rmse_b = float(np.sqrt(np.mean((tgts - base)  ** 2)))
        ss_res = np.sum((tgts - preds) ** 2)
        ss_tot = np.sum((tgts - tgts.mean()) ** 2)
        r2_m   = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        abs_m = np.abs(tgts - preds)
        abs_b = np.abs(tgts - base)
        w_stat, w_p = safe_wilcoxon(abs_m, abs_b)

        baseline_rows.append({
            "Model":         name,
            "N":             len(df),
            "Model RMSE":    round(rmse_m, 4),
            "Baseline RMSE": round(rmse_b, 4),
            "Delta RMSE":    round(rmse_m - rmse_b, 4),
            "Model R2":      round(r2_m, 4),
            "Wilcoxon W":    round(w_stat, 1) if not np.isnan(w_stat) else "n/a",
            "Wilcoxon p":    f"{w_p:.2e}" if not np.isnan(w_p) else "n/a",
            "Sig (p<0.05)":  "YES" if (not np.isnan(w_p) and w_p < 0.05) else "no",
        })
        sig = "YES" if (not np.isnan(w_p) and w_p < 0.05) else "no"
        print(f"  {name}: RMSE {rmse_m:.4f} vs baseline {rmse_b:.4f}  "
              f"Wilcoxon p={w_p:.2e}  Sig={sig}")

    baseline_df = pd.DataFrame(baseline_rows)
    bl_path = os.path.join(args.out_dir, "vs_mean_baseline.csv")
    baseline_df.to_csv(bl_path, index=False)
    print(f"Saved -> {bl_path}")

    # ── 3. Pairwise sample-level tests ───────────────────────────────────────
    print("\n--- Pairwise Sample-Level Wilcoxon (matched on subject_id + exercise_id) ---")
    names = list(oof.keys())
    pairwise_rows = []
    raw_ps: list[float] = []   # parallel list for Bonferroni-Holm correction

    for name_a, name_b in itertools.combinations(names, 2):
        df_a = oof[name_a][["subject_id", "exercise_id", "abs_error"]].copy()
        df_b = oof[name_b][["subject_id", "exercise_id", "abs_error"]].copy()

        merged = df_a.merge(
            df_b.rename(columns={"abs_error": "abs_error_b"}),
            on=["subject_id", "exercise_id"],
            how="inner",
        )
        n = len(merged)
        if n < 10:
            print(f"  [WARN] {name_a} vs {name_b}: only {n} matched samples — skipping")
            continue

        ae_a = merged["abs_error"].values
        ae_b = merged["abs_error_b"].values
        w_stat, w_p = safe_wilcoxon(ae_a, ae_b)
        t_stat, t_p = safe_ttest(ae_a, ae_b)
        mean_a, mean_b = float(ae_a.mean()), float(ae_b.mean())
        winner = name_b if mean_b < mean_a else name_a
        r = rank_biserial_r(w_stat, n)

        raw_ps.append(w_p if not np.isnan(w_p) else 1.0)
        pairwise_rows.append({
            "Model A":      name_a,
            "Model B":      name_b,
            "N matched":    n,
            "Mean AE A":    round(mean_a, 4),
            "Mean AE B":    round(mean_b, 4),
            "Delta (A-B)":  round(mean_a - mean_b, 4),
            "Winner":       winner,
            "Wilcoxon W":   round(w_stat, 1) if not np.isnan(w_stat) else "n/a",
            "Wilcoxon p":   w_p,           # store float; formatted below
            "t-test p":     t_p,           # store float; formatted below
            "Effect r":     round(r, 3) if not np.isnan(r) else "n/a",
        })

    # Apply Bonferroni-Holm correction across all m pairwise tests
    adj_ps = bonferroni_holm(raw_ps)
    for row, adj_p in zip(pairwise_rows, adj_ps):
        raw_p = row["Wilcoxon p"]
        row["Wilcoxon p"]   = f"{raw_p:.4f}" if not np.isnan(raw_p) else "n/a"
        row["t-test p"]     = f"{row['t-test p']:.4f}" if not np.isnan(row["t-test p"]) else "n/a"
        row["Adj p (Holm)"] = f"{adj_p:.4f}"
        row["Sig (p<0.05)"] = "YES" if (not np.isnan(raw_p) and raw_p < 0.05) else "no"
        row["Sig (FWER)"]   = "YES" if adj_p < 0.05 else "no"

    for row in pairwise_rows:
        print(f"  {row['Model A']} vs {row['Model B']}:  N={row['N matched']}  "
              f"delta={row['Delta (A-B)']:+.4f}  p={row['Wilcoxon p']}  "
              f"adj_p={row['Adj p (Holm)']}  FWER={row['Sig (FWER)']}  r={row['Effect r']}")

    m_tests = len(pairwise_rows)
    bonf_threshold = 0.05 / m_tests if m_tests else 0.05
    print(f"\n  Bonferroni threshold (m={m_tests}): alpha={bonf_threshold:.4f}")
    fwer_sig = [r for r in pairwise_rows if r["Sig (FWER)"] == "YES"]
    print(f"  Pairs surviving FWER correction: {len(fwer_sig)} / {m_tests}")

    pairwise_df = pd.DataFrame(pairwise_rows)
    col_order_pw = ["Model A", "Model B", "N matched", "Mean AE A", "Mean AE B",
                    "Delta (A-B)", "Winner", "Wilcoxon W", "Wilcoxon p",
                    "Adj p (Holm)", "t-test p", "Effect r", "Sig (p<0.05)", "Sig (FWER)"]
    pairwise_df = pairwise_df.reindex(columns=col_order_pw)
    pw_path = os.path.join(args.out_dir, "pairwise_sample_level.csv")
    pairwise_df.to_csv(pw_path, index=False)
    print(f"Saved -> {pw_path}")

    # ── 4. Text report ───────────────────────────────────────────────────────
    report_path = os.path.join(args.out_dir, "sample_stats_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Sample-Level Statistical Analysis — KIMORE Rehabilitation Scoring\n")
        f.write("=" * 72 + "\n\n")
        n_ref = len(list(oof.values())[0]) if oof else 0
        f.write(f"N per model: ~{n_ref} matched OOF predictions (KIMORE subjects only)\n")
        f.write("Primary test: Paired Wilcoxon signed-rank (two-sided)\n")
        f.write("Secondary test: Paired t-test (parametric reference)\n\n")

        f.write("1. Per-Exercise Spearman Rho (pooled OOF)\n")
        f.write("   Method: Pool all out-of-fold predictions, compute Spearman per exercise.\n")
        f.write("   This matches the metric reported by all published KIMORE papers.\n")
        f.write("-" * 72 + "\n")
        f.write(spearman_df.to_string(index=False))
        f.write("\n\nNote: Our models use 5-fold Stratified LOSO; published papers use\n")
        f.write("5-fold CV without documented stratification, making direct comparison\n")
        f.write("indicative but not exact. LOSO is stricter (no subject leakage).\n\n")

        f.write("2. vs Mean-Prediction Baseline\n")
        f.write("-" * 72 + "\n")
        f.write(baseline_df.to_string(index=False))
        f.write("\n\n")

        f.write("3. Pairwise Sample-Level Tests\n")
        f.write(f"   Multiple comparisons: Holm-Bonferroni FWER correction over {m_tests} tests.\n")
        f.write(f"   Uncorrected alpha=0.05; nominal Bonferroni threshold={bonf_threshold:.4f}.\n")
        f.write(f"   Pairs surviving FWER correction: {len(fwer_sig)} / {m_tests}\n")
        f.write("   Effect size: rank-biserial r (|r|<0.1 trivial, 0.1-0.3 small, >0.3 medium)\n")
        f.write("-" * 72 + "\n")
        if not pairwise_df.empty:
            f.write(pairwise_df.to_string(index=False))
        f.write("\n")

    print(f"\nFull report -> {report_path}")
    print("\nSample-level analysis complete.")


if __name__ == "__main__":
    main()
