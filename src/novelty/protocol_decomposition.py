"""N1 - Protocol-inflation DECOMPOSITION (2x2 factorial).

The existing src/protocol_inflation.py reports three points (LOSO, GroupKFold,
random KFold). This module completes the missing cell and runs the full 2x2
factorial design to separate the two inflation mechanisms as additive main
effects with bootstrap confidence intervals:

    factor 1: subject leakage   (subjects may appear in train AND test?)
    factor 2: clinical stratify (folds balanced across the 5 clinical groups?)

        +----------------------+-------------------------------+----------------------------------+
        |                      | unstratified                  | clinically stratified            |
        +----------------------+-------------------------------+----------------------------------+
        | subjects leak (Y)    | KFold(shuffle)                | StratifiedKFold(by group)        |
        | no leakage  (N)      | GroupKFold(by subject)        | StratifiedGroupKFold (our LOSO)  |
        +----------------------+-------------------------------+----------------------------------+

Main effect of leakage      = mean over stratify-levels of [rho(leak) - rho(noleak)]
Main effect of stratify     = mean over leakage-levels  of [rho(strat) - rho(unstrat)]
Interaction                 = [rho(Y,Y) - rho(Y,N)] - [rho(N,Y) - rho(N,N)]

Reuses extract_features + eval_protocol from protocol_inflation, so the
estimator (RidgeCV on identical features) is identical across cells - the only
thing that changes is the splitter. Falsifiable claim: a fixed, decomposable
delta-rho is attributable to evaluation protocol.

Usage:
  python src/novelty/protocol_decomposition.py --seeds 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from novelty import config, io_utils
from protocol_inflation import extract_features, eval_protocol

# 2x2 cell keys: (leakage, stratified)
CELLS = {
    ("leak", "unstrat"): "KFold (sample-level, subjects leak)",
    ("leak", "strat"):   "StratifiedKFold (by clinical group, subjects leak)",
    ("noleak", "unstrat"): "GroupKFold (subject-level)",
    ("noleak", "strat"):   "StratifiedGroupKFold (our Stratified LOSO)",
}


def _splitter(leakage: str, stratified: str, seed: int):
    """Return (splitter, split_kwargs) for a 2x2 cell."""
    if leakage == "leak" and stratified == "unstrat":
        return KFold(n_splits=5, shuffle=True, random_state=seed), {}
    if leakage == "leak" and stratified == "strat":
        # stratify by clinical group; NO grouping by subject => subjects leak
        return (StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
                {"y": "GROUP"})
    if leakage == "noleak" and stratified == "unstrat":
        return GroupKFold(n_splits=5), {"y": "Y", "groups": "SUBJECT"}
    # noleak + strat == our protocol
    return (StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed),
            {"y": "GROUP", "groups": "SUBJECT"})


def _resolve_kwargs(template: dict, y, group_labels, subject_ids) -> dict:
    """Map kwarg placeholders to concrete arrays for eval_protocol."""
    mapping = {"Y": y, "GROUP": group_labels, "SUBJECT": subject_ids}
    return {k: mapping[v] for k, v in template.items()}


def _has_shuffle(leakage: str, stratified: str) -> bool:
    """GroupKFold is deterministic (no shuffle); all others vary with seed."""
    return not (leakage == "noleak" and stratified == "unstrat")


def run(seeds: int = 20) -> dict:
    data = io_utils.load_pooled_kimore()
    if data is None:
        return {"error": "pooled KIMORE unavailable"}

    print("Extracting features...")
    X = extract_features(data["X_raw"], data["exercise_ids"])
    y = data["y"]
    eid = data["exercise_ids"]
    grp = data["group_labels"]
    sub = data["subject_ids"]
    print(f"  N={len(y)}, subjects={len(np.unique(sub))}, seeds={seeds}")

    # cell -> list of mean_rho_pooled across seeds
    cell_rhos: dict[tuple[str, str], list[float]] = {c: [] for c in CELLS}
    cell_per_ex: dict[tuple[str, str], dict] = {}

    for cell in CELLS:
        leakage, stratified = cell
        n_seeds = seeds if _has_shuffle(leakage, stratified) else 1
        for s in range(n_seeds):
            splitter, tmpl = _splitter(leakage, stratified, seed=42 + s)
            kwargs = _resolve_kwargs(tmpl, y, grp, sub)
            res = eval_protocol(X, y, eid, splitter, kwargs)
            cell_rhos[cell].append(res["mean_rho_pooled"])
            if s == 0:
                cell_per_ex[cell] = res["per_exercise"]
        print(f"  {CELLS[cell]:48s} rho={np.mean(cell_rhos[cell]):.4f} "
              f"(+/-{np.std(cell_rhos[cell]):.4f}, n_seeds={n_seeds})")

    def mean(cell):
        return float(np.mean(cell_rhos[cell]))

    r_NN = mean(("noleak", "unstrat"))
    r_NY = mean(("noleak", "strat"))
    r_YN = mean(("leak", "unstrat"))
    r_YY = mean(("leak", "strat"))

    main_leakage = 0.5 * ((r_YN - r_NN) + (r_YY - r_NY))
    main_stratify = 0.5 * ((r_NY - r_NN) + (r_YY - r_YN))
    interaction = (r_YY - r_YN) - (r_NY - r_NN)

    # Bootstrap CI for the leakage main effect across seeds (the shuffle-driven
    # uncertainty); GroupKFold cell is constant so its variance is 0.
    rng = np.random.default_rng(0)
    boot_leak, boot_strat = [], []
    n_boot = 2000
    leak_un = np.array(cell_rhos[("leak", "unstrat")])
    leak_st = np.array(cell_rhos[("leak", "strat")])
    noleak_st = np.array(cell_rhos[("noleak", "strat")])
    for _ in range(n_boot):
        a = rng.choice(leak_un)
        b = rng.choice(leak_st)
        c = rng.choice(noleak_st)
        boot_leak.append(0.5 * ((a - r_NN) + (b - c)))
        boot_strat.append(0.5 * ((c - r_NN) + (b - a)))

    def ci(arr):
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    grid = {f"{lk}|{st}": {"label": CELLS[(lk, st)],
                            "mean_rho": mean((lk, st)),
                            "sd_rho": float(np.std(cell_rhos[(lk, st)])),
                            "per_exercise": cell_per_ex.get((lk, st), {})}
            for (lk, st) in CELLS}

    results = {
        "design": "2x2 factorial: {subject_leakage} x {clinical_stratification}",
        "n_seeds": seeds,
        "grid": grid,
        "reference_cell": "noleak|strat (our Stratified LOSO)",
        "main_effect_subject_leakage": round(main_leakage, 4),
        "main_effect_subject_leakage_ci95": ci(boot_leak),
        "main_effect_clinical_stratification": round(main_stratify, 4),
        "main_effect_clinical_stratification_ci95": ci(boot_strat),
        "interaction": round(interaction, 4),
        "total_inflation_vs_loso": round(r_YN - r_NY, 4),  # naive 5-fold vs LOSO
        "interpretation": (
            "Positive leakage main effect => non-grouped CV inflates Spearman rho. "
            "Both mechanisms are additive if interaction ~ 0."
        ),
    }

    out = config.ensure_out()
    path = os.path.join(out, "protocol_decomposition.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n2x2 DECOMPOSITION")
    print(f"  main effect (subject leakage)      = {main_leakage:+.4f}  CI95={ci(boot_leak)}")
    print(f"  main effect (clinical stratify)    = {main_stratify:+.4f}  CI95={ci(boot_strat)}")
    print(f"  interaction                        = {interaction:+.4f}")
    print(f"  -> {path}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="N1 protocol-inflation 2x2 decomposition")
    ap.add_argument("--seeds", type=int, default=20,
                    help="shuffle repetitions for variance estimation")
    args = ap.parse_args()
    run(seeds=args.seeds)


if __name__ == "__main__":
    main()
