"""N10 - Power analysis / design guideline for the dissociation claim.

The observed dissociation (KIMORE rank vs IRDS cross-exercise consistency) is
underpowered: r ~ -0.393 over N=7 models, IRDS N=10 subjects, p ~ 0.38. Rather
than overclaim, we estimate via Monte-Carlo HOW MANY models x subjects are needed
to detect an effect of the observed size at 80% power - a reusable design
guideline for the field.

Model: each model has a true (KIMORE-rho, IRDS-W) pair drawn from a bivariate
normal with correlation rho_target. The measured IRDS-W carries sampling noise
that shrinks with subjects: sd(W) = sd0 * sqrt(n0 / n_subjects). For each
(n_models, n_subjects) cell we simulate many studies and report the fraction in
which the correlation is significant (p < alpha) with the correct (negative)
sign.

Usage:
  python src/novelty/power_analysis.py --target -0.393 --sims 2000
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


def _observed_target() -> float:
    """Estimate the dissociation correlation from real artifacts if available."""
    rel = io_utils.load_irds_reliability()
    oof = io_utils.load_all_oof()
    if rel is None or not oof:
        return config.DISSOCIATION_R_OBS
    # KIMORE mean rho per model from OOF
    kim = {}
    for name, df in oof.items():
        rhos = []
        for e in range(config.N_EXERCISES):
            sub = df[df["exercise_id"] == e]
            if len(sub) >= 5:
                r, _ = spearmanr(sub["y_true"], sub["y_pred"])
                rhos.append(r)
        if rhos:
            kim[name] = float(np.mean(rhos))
    xs, ys = [], []
    for _, row in rel.iterrows():
        m = row.get("model")
        if m in kim and "Kendall_W" in row:
            xs.append(kim[m])
            ys.append(float(row["Kendall_W"]))
    if len(xs) >= 3:
        r, _ = spearmanr(xs, ys)
        return float(r)
    return config.DISSOCIATION_R_OBS


def _simulate_power(rho_target: float, n_models: int, n_subjects: int,
                    sd0: float, n0: int, alpha: float, sims: int,
                    rng: np.random.Generator) -> float:
    noise_sd = sd0 * np.sqrt(n0 / max(n_subjects, 1))
    hits = 0
    for _ in range(sims):
        z = rng.standard_normal(n_models)
        w = rng.standard_normal(n_models)
        x = z                                            # latent KIMORE rho
        y = rho_target * z + np.sqrt(max(1 - rho_target ** 2, 0)) * w  # latent IRDS W
        y_meas = y + rng.normal(0, noise_sd, n_models)   # measurement noise
        r, p = spearmanr(x, y_meas)
        if p < alpha and np.sign(r) == np.sign(rho_target):
            hits += 1
    return hits / sims


def run(target: float | None = None, sims: int = 2000) -> dict:
    rho_target = target if target is not None else _observed_target()
    rng = np.random.default_rng(7)
    models_grid = [7, 10, 15, 20, 30]
    subjects_grid = [10, 20, 30, 50]

    rows = []
    for nm in models_grid:
        for ns in subjects_grid:
            power = _simulate_power(
                rho_target, nm, ns,
                sd0=config.KENDALL_W_SD_AT_10, n0=config.IRDS_N_SUBJECTS,
                alpha=config.ALPHA, sims=sims, rng=rng,
            )
            rows.append({"n_models": nm, "n_subjects": ns, "power": round(power, 3)})

    df = pd.DataFrame(rows)
    # smallest (n_models, n_subjects) reaching 80% power (by total budget)
    ok = df[df["power"] >= 0.80].copy()
    if len(ok):
        ok["budget"] = ok["n_models"] * ok["n_subjects"]
        rec = ok.sort_values("budget").iloc[0]
        recommended = {"n_models": int(rec["n_models"]),
                       "n_subjects": int(rec["n_subjects"]),
                       "power": float(rec["power"])}
    else:
        recommended = None

    out = config.ensure_out()
    csv_path = os.path.join(out, "power_analysis.csv")
    df.to_csv(csv_path, index=False)
    results = {
        "rho_target": round(float(rho_target), 4),
        "alpha": config.ALPHA,
        "sims_per_cell": sims,
        "kendall_w_sd_at_10_subjects": config.KENDALL_W_SD_AT_10,
        "grid": rows,
        "recommended_for_80pct_power": recommended,
        "note": ("Current study (7 models, 10 subjects) power shown in grid; "
                 "use 'recommended' as the field design guideline."),
    }
    json_path = os.path.join(out, "power_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"POWER ANALYSIS (rho_target={rho_target:.3f}, alpha={config.ALPHA})")
    pivot = df.pivot(index="n_models", columns="n_subjects", values="power")
    print(pivot.to_string())
    print(f"  recommended for 80% power: {recommended}")
    print(f"  -> {csv_path}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="N10 power analysis")
    ap.add_argument("--target", type=float, default=None,
                    help="target dissociation correlation (default: estimate from data)")
    ap.add_argument("--sims", type=int, default=2000)
    args = ap.parse_args()
    run(target=args.target, sims=args.sims)


if __name__ == "__main__":
    main()
