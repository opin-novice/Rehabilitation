"""Reviewer round-4, T1 (concerns #1, #2, #6): subject-clustered AUROC significance.

For each learned condition and the naive baseline, per target corpus, we test whether
zero-shot AUROC exceeds chance (0.50) and whether the naive baseline differs from each
learned model --- honoring the fact that each target corpus has only 10 subjects, by
bootstrapping over SUBJECT CLUSTERS (not repetitions). This simultaneously addresses:
  #1 a formal test that learned AUROCs differ from 0.50 (and naive vs each model),
  #2 subject-level uncertainty for the 10-subject target corpora,
  #6 whether the naive baseline itself is distinguishable from 0.50.

Orientation is fixed once on the full sample (so full-sample AUROC >= 0.5), then held
constant across bootstraps --- avoiding the upward bias of re-taking max(AUROC,1-AUROC)
per resample.

Run:  python src/reviewer/reviewer_round4_stats.py [--nboot 2000]
Out:  outputs/reviewer_round4/auroc_significance.{json,md} + cached preds .npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ root (file now in src/reviewer/)
from selfsup.zeroshot_eval import _rebuild_model, _predict  # noqa: E402
from selfsup.data import load_corpus_with_labels  # noqa: E402
from selfsup.naive_baseline import compute_naive_features  # noqa: E402

RESULTS_DIR = "archive/legacy_results/kimore_loso_78fold"
OUT_DIR = "outputs/reviewer_round4"
CONDITIONS = {
    "A_scratch": "Scratch", "B_contrastive_lp": "Contrastive LP",
    "C_contrastive_ft": "Contrastive FT", "D_masked_lp": "Masked LP",
    "E_masked_ft": "Masked FT",
}
CORPORA = {"REHAB246": "REHAB246", "UIPRMD": "UI-PRMD"}


def _subjects(uids):
    return np.array([str(u).split("::")[1] for u in uids])


def _mean_fold_preds(cond_dir, X):
    cps = sorted(glob.glob(os.path.join(cond_dir, "fold_*", "best_model.pt")))
    acc = np.zeros(len(X), dtype=np.float64)
    for cp in cps:
        acc += _predict(_rebuild_model(torch.load(cp, map_location="cpu")), X)
    return acc / max(len(cps), 1)


def _naive_score(X):
    """Best single naive feature as a 1-D score (path length or mean speed)."""
    return compute_naive_features(X)  # (N,2)


def _auroc_fixed(score, y):
    a = roc_auc_score(y, score)
    return a if a >= 0.5 else 1.0 - a


def _orient(score, y):
    """Return score oriented so full-sample AUROC >= 0.5 (sign fixed here, then frozen)."""
    return score if roc_auc_score(y, score) >= 0.5 else -score


def _boot_auroc(scores_by_name, y, subs, nboot, seed=0):
    """Subject-cluster bootstrap. Returns per-name arrays of bootstrap AUROCs and
    paired (naive - model) differences."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(subs)
    idx_by_sub = {s: np.where(subs == s)[0] for s in uniq}
    names = list(scores_by_name)
    boot = {n: [] for n in names}
    diff = {n: [] for n in names if n != "naive"}
    for _ in range(nboot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_sub[s] for s in pick])
        yy = y[rows]
        if len(np.unique(yy)) < 2:
            continue
        a = {}
        for n in names:
            try:
                a[n] = roc_auc_score(yy, scores_by_name[n][rows])
            except ValueError:
                a[n] = np.nan
            boot[n].append(a[n])
        for n in diff:
            diff[n].append(a["naive"] - a[n])
    return ({n: np.array(v) for n, v in boot.items()},
            {n: np.array(v) for n, v in diff.items()})


def _ci_p_gt(arr, null=0.5):
    arr = arr[~np.isnan(arr)]
    lo, hi = np.percentile(arr, [2.5, 97.5])
    # one-sided p that value > null; two-sided reported
    p_one = float(np.mean(arr <= null))
    p_two = float(min(1.0, 2 * min(p_one, 1 - p_one)))
    return float(lo), float(hi), p_two


def run(nboot=2000):
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for corpus in CORPORA:
        X, y, uids = load_corpus_with_labels(corpus)
        y = np.asarray(y).astype(int)
        subs = _subjects(uids)
        cache = os.path.join(OUT_DIR, f"preds_{corpus}.npz")
        scores = {}
        if os.path.exists(cache):
            d = np.load(cache, allow_pickle=True)
            for k in d.files:
                scores[k] = d[k]
        else:
            for cond in CONDITIONS:
                scores[cond] = _mean_fold_preds(os.path.join(RESULTS_DIR, cond), X)
            feats = _naive_score(X)
            # pick the better naive feature on full sample
            best_j = max(range(feats.shape[1]), key=lambda j: _auroc_fixed(feats[:, j], y))
            scores["naive"] = feats[:, best_j]
            np.savez(cache, **scores)
        # freeze orientation on full sample
        scores = {n: _orient(np.asarray(s, dtype=np.float64), y) for n, s in scores.items()}
        full = {n: float(roc_auc_score(y, s)) for n, s in scores.items()}
        boot, diff = _boot_auroc(scores, y, subs, nboot)

        rows = {}
        for n in scores:
            lo, hi, p = _ci_p_gt(boot[n], 0.5)
            rows[n] = {"auroc": full[n], "ci_lo": lo, "ci_hi": hi, "p_vs_0.5": p,
                       "sig_above_chance": bool(lo > 0.5)}
        for n in diff:
            d = diff[n][~np.isnan(diff[n])]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p_one = float(np.mean(d <= 0.0))
            rows[n]["naive_minus_model"] = float(np.mean(d))
            rows[n]["diff_ci"] = [float(lo), float(hi)]
            rows[n]["p_naive_gt_model"] = float(min(1.0, 2 * min(p_one, 1 - p_one)))
        results[corpus] = {"n_subjects": int(len(np.unique(subs))), "n": int(len(X)),
                           "nboot": nboot, "conditions": rows}
        print(f"\n== {corpus} (subject-clustered, {len(np.unique(subs))} subjects) ==")
        for n in list(CONDITIONS) + ["naive"]:
            r = rows[n]
            print(f"  {n:18s} AUROC={r['auroc']:.3f} 95%CI[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] "
                  f"p(vs0.5)={r['p_vs_0.5']:.3f} sig={r['sig_above_chance']}")

    with open(os.path.join(OUT_DIR, "auroc_significance.json"), "w") as f:
        json.dump(results, f, indent=2)
    _write_md(results)
    return results


def _write_md(results):
    lines = ["### Subject-clustered AUROC significance (10-subject bootstrap, orientation fixed)", ""]
    for corpus, R in results.items():
        lines += [f"**{CORPORA[corpus]}** ({R['n_subjects']} subjects, N={R['n']}, {R['nboot']} boots)", "",
                  "| Condition | AUROC | 95% CI (subject) | p vs 0.5 | > chance? | naive−model | p(naive>model) |",
                  "|---|---|---|---|---|---|---|"]
        for n in list(CONDITIONS) + ["naive"]:
            r = R["conditions"][n]
            nm = CONDITIONS.get(n, "Naive baseline")
            dm = f"{r['naive_minus_model']:+.3f}" if "naive_minus_model" in r else "—"
            pnm = f"{r['p_naive_gt_model']:.3f}" if "p_naive_gt_model" in r else "—"
            lines.append(f"| {nm} | {r['auroc']:.3f} | [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | "
                         f"{r['p_vs_0.5']:.3f} | {'yes' if r['sig_above_chance'] else 'no'} | {dm} | {pnm} |")
        lines.append("")
    with open(os.path.join(OUT_DIR, "auroc_significance.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    run(nboot=a.nboot)
