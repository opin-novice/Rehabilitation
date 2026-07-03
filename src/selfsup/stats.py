"""Statistics for the 5 conditions (PRD R6).

Bootstrap CIs over folds, Holm-Bonferroni across pairwise Wilcoxon tests, the
primary contrastive_ft-vs-masked_ft paired Wilcoxon, and the linear-probe sanity
gate (encoder must beat chance) that makes a downstream null credible.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from selfsup.registry import CONDITIONS, PRIMARY_CONTRAST  # noqa: E402


def _fold_spearmans(cond_dir: str) -> List[float]:
    p = os.path.join(cond_dir, "loso_results.json")
    if not os.path.exists(p):
        return []
    data = json.loads(Path(p).read_text())
    return [f.get("spearman", float("nan")) for f in data.get("folds", [])]


def bootstrap_ci(vals: List[float], n_boot: int = 10000, seed: int = 0):
    v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)]
    return {"mean": float(v.mean()), "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)), "n": int(len(v))}


def holm_bonferroni(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def _wilcoxon(a: List[float], b: List[float]):
    from scipy.stats import wilcoxon
    a = np.array(a); b = np.array(b)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 1 or np.allclose(a, b):
        return float("nan"), float(np.median(a - b) if len(a) else float("nan"))
    try:
        stat, p = wilcoxon(a, b)
    except ValueError:
        p = float("nan")
    return float(p), float(np.median(a - b))


def compute_statistics(results_root: str = "outputs", n_boot: int = 10000) -> dict:
    rho: Dict[str, List[float]] = {}
    for name, cond in CONDITIONS.items():
        rho[name] = _fold_spearmans(os.path.join(results_root, cond.out_subdir))

    cis = {name: bootstrap_ci(v, n_boot=n_boot) for name, v in rho.items()}
    base = cis.get("scratch", {}).get("mean", float("nan"))

    pairs, praw = [], []
    for a, b in itertools.combinations(CONDITIONS.keys(), 2):
        p, eff = _wilcoxon(rho[a], rho[b])
        pairs.append({"a": a, "b": b, "p_raw": p, "median_diff": eff})
        praw.append(p if np.isfinite(p) else 1.0)
    for pr, adj in zip(pairs, holm_bonferroni(praw)):
        pr["p_adjusted"] = adj
        pr["significant"] = bool(np.isfinite(adj) and adj < 0.05)

    pa, pb = PRIMARY_CONTRAST
    p_pri, eff_pri = _wilcoxon(rho.get(pa, []), rho.get(pb, []))

    probe = {}
    for lp in ("contrastive_lp", "masked_lp"):
        m = cis.get(lp, {}).get("mean", float("nan"))
        probe[lp] = {"mean_rho": m, "beats_chance": bool(np.isfinite(m) and m > 0.0)}

    out = {
        "bootstrap_cis": cis,
        "beats_scratch": {n: bool(np.isfinite(c["mean"]) and np.isfinite(base) and c["mean"] > base)
                          for n, c in cis.items()},
        "pairwise_wilcoxon": pairs,
        "primary_contrast": {"a": pa, "b": pb, "p": p_pri, "median_diff": eff_pri},
        "probe_sanity": probe,
    }
    dest = Path(results_root) / "ssl_results" / "stats.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"[stats] wrote {dest}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="outputs")
    ap.add_argument("--n_boot", type=int, default=10000)
    a = ap.parse_args()
    compute_statistics(a.results_root, a.n_boot)
