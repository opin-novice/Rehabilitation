"""N7 - Clinically-actionable deployment-readiness rubric.

A composite pass/fail screen for 'is this model safe to deploy on a new clinic's
data?', combining three orthogonal criteria from existing artifacts:

  1. ACCURACY    - beats chance at the sample level: mean KIMORE Spearman rho > 0
                   AND the lower bound of its bootstrap 95% CI > 0.
  2. CONSISTENCY - IRDS cross-exercise rank consistency Kendall W >= 0.50.
  3. INTEGRITY   - non-degenerate predictions under domain shift (pred_SD >= 0.10);
                   a variance-collapsed model gives spuriously high ICC/W (see N3).

A model is deployment-ready only if all three pass.

Usage:
  python src/novelty/deployment_rubric.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from novelty import config, io_utils


def _bootstrap_mean_rho_ci(df, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of per-exercise Spearman rho + 95% CI via stratified subject resampling."""
    rng = np.random.default_rng(seed)
    point_rhos = []
    for e in range(config.N_EXERCISES):
        sub = df[df["exercise_id"] == e]
        if len(sub) >= 5:
            r, _ = spearmanr(sub["y_true"], sub["y_pred"])
            point_rhos.append(r)
    point = float(np.mean(point_rhos)) if point_rhos else float("nan")

    boot = []
    by_ex = {e: df[df["exercise_id"] == e] for e in range(config.N_EXERCISES)}
    for _ in range(n_boot):
        rhos = []
        for e, sub in by_ex.items():
            if len(sub) < 5:
                continue
            idx = rng.integers(0, len(sub), len(sub))
            yt = sub["y_true"].values[idx]
            yp = sub["y_pred"].values[idx]
            if np.std(yt) > 0 and np.std(yp) > 0:
                r, _ = spearmanr(yt, yp)
                rhos.append(r)
        if rhos:
            boot.append(np.mean(rhos))
    lo = float(np.percentile(boot, 2.5)) if boot else float("nan")
    hi = float(np.percentile(boot, 97.5)) if boot else float("nan")
    return point, lo, hi


def run() -> dict:
    oof = io_utils.load_all_oof()
    rel = io_utils.load_irds_reliability()
    if not oof:
        return {"error": "OOF predictions unavailable"}
    rel_by_model = {}
    if rel is not None:
        rel_by_model = {row["model"]: row for _, row in rel.iterrows()}

    rows = []
    for name, df in oof.items():
        point, lo, hi = _bootstrap_mean_rho_ci(df)
        crit_accuracy = bool(point > 0 and lo > 0)

        r = rel_by_model.get(name)
        kendall_w = float(r["Kendall_W"]) if r is not None and "Kendall_W" in r else float("nan")
        pred_sd = float(r["pred_SD"]) if r is not None and "pred_SD" in r else float("nan")
        crit_consistency = bool(not np.isnan(kendall_w) and kendall_w >= config.KENDALL_W_MIN)
        crit_integrity = bool(not np.isnan(pred_sd) and pred_sd >= config.PRED_SD_MIN)

        deploy_ready = crit_accuracy and crit_consistency and crit_integrity
        reasons = []
        if not crit_accuracy:
            reasons.append("accuracy CI includes 0")
        if not crit_consistency:
            reasons.append(f"Kendall W < {config.KENDALL_W_MIN}")
        if not crit_integrity:
            reasons.append(f"degenerate (pred_SD < {config.PRED_SD_MIN})")

        rows.append({
            "model": name,
            "mean_rho": round(point, 4),
            "rho_ci95_low": round(lo, 4),
            "rho_ci95_high": round(hi, 4),
            "kendall_w": round(kendall_w, 4) if not np.isnan(kendall_w) else None,
            "pred_SD": round(pred_sd, 4) if not np.isnan(pred_sd) else None,
            "crit_accuracy": crit_accuracy,
            "crit_consistency": crit_consistency,
            "crit_integrity": crit_integrity,
            "deploy_ready": deploy_ready,
            "fail_reasons": "; ".join(reasons) if reasons else "",
        })

    df_out = pd.DataFrame(rows).sort_values("mean_rho", ascending=False)
    out = config.ensure_out()
    csv_path = os.path.join(out, "deployment_rubric.csv")
    df_out.to_csv(csv_path, index=False)

    results = {
        "thresholds": {"kendall_w_min": config.KENDALL_W_MIN,
                        "pred_sd_min": config.PRED_SD_MIN},
        "models": rows,
        "n_deploy_ready": int(df_out["deploy_ready"].sum()),
    }
    json_path = os.path.join(out, "deployment_rubric.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("DEPLOYMENT-READINESS RUBRIC")
    for r in rows:
        flag = "READY" if r["deploy_ready"] else "no   "
        print(f"  [{flag}] {r['model']:28s} rho={r['mean_rho']:.3f} "
              f"W={r['kendall_w']}  {r['fail_reasons']}")
    print(f"  -> {csv_path}")
    return results


def main() -> None:
    argparse.ArgumentParser(description="N7 deployment-readiness rubric").parse_args()
    run()


if __name__ == "__main__":
    main()
