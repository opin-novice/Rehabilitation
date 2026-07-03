"""Full true leave-one-subject-out (LOSO) on KIMORE, all 5 SSL conditions — for TNSRE.

The plan calls it '78-fold Stratified-LOSO'; KIMORE_pooled has 77 subjects (the plan's
78 counts one dropped in preprocessing), so true LOSO = 77 folds via LeaveOneGroupOut
(one subject held out per fold). Uses the irds_only encoders. Writes to a SEPARATE root
(results/kimore_loso_78fold/) so the 5-fold results (outputs/ssl_results/) are kept.

After training, pools out-of-fold predictions (N=380) per condition and runs the Paper-1
sample-level protocol: per-exercise Spearman rho, 20-seed bootstrap 95% CIs, and pairwise
Wilcoxon signed-rank on absolute error with Holm-Bonferroni FWER correction.

Usage:
  python src/selfsup/run_loso78.py           # resumable (skips completed conditions)
  python src/selfsup/run_loso78.py --force   # recompute every condition from scratch
"""
import os, sys, json, glob, shutil, subprocess, itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

PROJECT_ROOT = os.getcwd()
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from selfsup.registry import CONDITIONS, PRIMARY_CONTRAST  # noqa: E402

TRAIN_LOSO = os.path.join(PROJECT_ROOT, "src", "train_loso.py")
POOL = "irds_only"
POOLED = "KIMORE_pooled"
RESULTS = os.path.join("results", "kimore_loso_78fold")
ENC = {k: os.path.join("outputs", "ssl_pretrain", POOL, f"{k}_encoder.pt")
       for k in ("contrastive", "masked")}
FORCE = "--force" in sys.argv


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def run_condition(name, cond):
    out_dir = os.path.join(RESULTS, os.path.basename(cond.out_subdir))
    done = os.path.join(out_dir, "loso_results.json")
    if os.path.exists(done) and not FORCE:
        print(f"[78fold] SKIP {name} (already complete: {done})", flush=True)
        return out_dir
    if FORCE and os.path.exists(out_dir):
        shutil.rmtree(out_dir)   # only wipe on --force; otherwise resume salvages partial folds
    cmd = [sys.executable, TRAIN_LOSO, "--model_type", "tcn", "--loso", "--resume",
           "--pooled_dir", POOLED, "--out_dir", out_dir,
           "--epochs", "100", "--batch_size", "16", "--patience", "100", "--d_model", "128"]
    if cond.init_ckpt_key:
        enc = ENC[cond.init_ckpt_key]
        assert os.path.exists(enc), f"missing encoder {enc}"
        cmd += ["--init_ckpt", enc]
    if cond.freeze_encoder:
        cmd += ["--freeze_encoder"]
    print(f"[78fold] RUN {name} -> {out_dir}", flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    return out_dir


# --------------------------------------------------------------------------- #
# Sample-level statistics (Paper-1 protocol)                                   #
# --------------------------------------------------------------------------- #
def pool_oof(out_dir):
    files = sorted(glob.glob(os.path.join(out_dir, "fold_*", "oof_predictions.csv")))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["exercise_id"] < 5].reset_index(drop=True)   # KIMORE exercises only
    df.to_csv(os.path.join(out_dir, "oof_predictions_all.csv"), index=False)
    return df


def mean_per_exercise_rho(df):
    rhos = {}
    for e in range(5):
        s = df[df["exercise_id"] == e]
        if len(s) >= 5:
            r, _ = spearmanr(s["y_true"], s["y_pred"])
            if not np.isnan(r):
                rhos[e] = float(r)
    mean = float(np.mean(list(rhos.values()))) if rhos else float("nan")
    return mean, rhos


def bootstrap_ci(df, seeds=20, n_boot=500):
    """20-seed stratified bootstrap of the mean-per-exercise Spearman rho (Paper-1 style)."""
    eids = [e for e in range(5) if (df["exercise_id"] == e).sum() >= 5]
    sub = {e: df[df["exercise_id"] == e] for e in eids}
    means = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            rs = []
            for e in eids:
                s = sub[e]
                idx = rng.integers(0, len(s), len(s))
                r, _ = spearmanr(s["y_true"].values[idx], s["y_pred"].values[idx])
                if not np.isnan(r):
                    rs.append(r)
            if rs:
                means.append(float(np.mean(rs)))
    if not means:
        return float("nan"), float("nan")
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def holm_bonferroni(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, ps[i] * (m - rank))
        adj[i] = min(running, 1.0)
    return adj


def paired_wilcoxon(df_a, df_b):
    on = ["subject_id", "exercise_id"]
    m = df_a[on + ["abs_error"]].merge(df_b[on + ["abs_error"]], on=on, suffixes=("_a", "_b"))
    ae_a, ae_b = m["abs_error_a"].values, m["abs_error_b"].values
    delta = float(ae_a.mean() - ae_b.mean())   # >0 => B lower error => B better
    if len(m) < 10 or np.all(ae_a - ae_b == 0):
        return float("nan"), len(m), delta
    try:
        _, p = wilcoxon(ae_a, ae_b, alternative="two-sided")
    except ValueError:
        return float("nan"), len(m), delta
    return float(p), len(m), delta


def compute_stats(cond_dirs):
    oof = {name: pool_oof(d) for name, d in cond_dirs.items()}
    oof = {k: v for k, v in oof.items() if v is not None}
    names = [n for n in CONDITIONS if n in oof]

    per_cond = {}
    for name in names:
        df = oof[name]
        mean, rhos = mean_per_exercise_rho(df)
        lo, hi = bootstrap_ci(df)
        per_cond[name] = {"mean_rho": mean, "ci_low": lo, "ci_high": hi,
                          "n_samples": int(len(df)),
                          "per_exercise_rho": {str(k): v for k, v in rhos.items()}}

    scratch_rho = per_cond.get("scratch", {}).get("mean_rho", float("nan"))
    scratch_hi = per_cond.get("scratch", {}).get("ci_high", float("nan"))

    # pairwise Wilcoxon over all condition pairs, Holm-corrected
    pairs = list(itertools.combinations(names, 2))
    raw = []
    rows = []
    for a, b in pairs:
        p, n, delta = paired_wilcoxon(oof[a], oof[b])
        raw.append(p if not np.isnan(p) else 1.0)
        winner = b if delta > 0 else a
        rows.append({"a": a, "b": b, "n_matched": n, "delta_abs_err": delta,
                     "winner": winner, "p_raw": p})
    adj = holm_bonferroni(raw)
    for row, ap in zip(rows, adj):
        row["p_adjusted"] = ap
        row["significant"] = bool(ap < 0.05)

    # beats_scratch: mean rho strictly above scratch AND lower error at FWER
    beats = {}
    for name in names:
        if name == "scratch":
            beats[name] = False
            continue
        rho_gt = per_cond[name]["mean_rho"] > scratch_rho
        sig_row = next((r for r in rows
                        if {r["a"], r["b"]} == {name, "scratch"}), None)
        better = sig_row and sig_row["winner"] == name and sig_row["significant"]
        beats[name] = bool(rho_gt and better)

    pc_a, pc_b = PRIMARY_CONTRAST
    primary = None
    if pc_a in oof and pc_b in oof:
        p, n, delta = paired_wilcoxon(oof[pc_a], oof[pc_b])
        primary = {"a": pc_a, "b": pc_b, "p": p, "n_matched": n, "delta_abs_err": delta}

    return {"protocol": "true LOSO (LeaveOneGroupOut), 77 folds; sample-level pooled OOF (N=380); "
                        "20-seed stratified bootstrap 95% CI; pairwise Wilcoxon + Holm-Bonferroni",
            "per_condition": per_cond, "beats_scratch": beats,
            "pairwise_wilcoxon": rows, "primary_contrast": primary,
            "n_folds": None}


def write_markdown(stats, path):
    pc = stats["per_condition"]
    order = [c for c in CONDITIONS if c in pc]
    order = sorted(order, key=lambda c: -pc[c]["mean_rho"])
    lines = ["## N. KIMORE fine-tuning — full 77-fold true LOSO (Table N)", "",
             "| Condition | Mean rho | 95% CI (20-seed bootstrap) | Beats scratch |",
             "|---|---|---|---|"]
    for c in order:
        r = pc[c]
        mark = "**" if c == "scratch" else ""
        lines.append(f"| {c} | {mark}{r['mean_rho']:.3f}{mark} | "
                     f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}] | "
                     f"{'Yes' if stats['beats_scratch'].get(c) else ('—' if c=='scratch' else 'No')} |")
    prim = stats["primary_contrast"]
    if prim:
        lines += ["", f"- Primary contrast {prim['a']} vs {prim['b']}: paired Wilcoxon "
                      f"p={prim['p']:.3f} (N={prim['n_matched']} matched), "
                      f"Δabs-err={prim['delta_abs_err']:+.3f}."]
    n_sig = sum(1 for r in stats["pairwise_wilcoxon"] if r["significant"])
    lines += [f"- Pairwise Wilcoxon (Holm-Bonferroni over {len(stats['pairwise_wilcoxon'])} pairs): "
              f"{n_sig} significant.",
              "- Sample-level pooled OOF, N=380; true leave-one-subject-out (77 folds)."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    print(f"[78fold] START (force={FORCE}) -> {RESULTS}", flush=True)
    cond_dirs = {}
    for name, cond in CONDITIONS.items():
        cond_dirs[name] = run_condition(name, cond)

    print("[78fold] all conditions done; computing sample-level stats", flush=True)
    stats = compute_stats(cond_dirs)
    with open(os.path.join(RESULTS, "stats78.json"), "w") as f:
        json.dump(stats, f, indent=2)
    write_markdown(stats, os.path.join(RESULTS, "table78.md"))

    print("\n=== 77-fold true LOSO — KIMORE mean rho (sample-level) ===", flush=True)
    for name in sorted(stats["per_condition"], key=lambda c: -stats["per_condition"][c]["mean_rho"]):
        r = stats["per_condition"][name]
        print("  %-16s rho=%.4f  CI=[%.3f, %.3f]  beats_scratch=%s"
              % (name, r["mean_rho"], r["ci_low"], r["ci_high"], stats["beats_scratch"].get(name)),
              flush=True)
    print(f"[78fold] DONE -> {RESULTS}/stats78.json, table78.md", flush=True)


if __name__ == "__main__":
    main()
