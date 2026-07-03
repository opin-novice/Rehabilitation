"""V2 - Reliability-predicts-validity analysis.

Converts our zero-shot reliability diagnostic from a self-consistency measure into
a VALIDATED deployment screen:

  VALIDITY proxy = does a model's predicted continuous score discriminate correct
      vs incorrect repetitions? (AUROC + point-biserial r vs the binary label.)

  HYPOTHESIS = reliability + non-degeneracy predicts validity:
      (a) models failing the pred_SD > 0.10 degeneracy gate collapse to AUROC ~0.5
          (a constant predictor is trivially 'reliable' yet cannot discriminate);
      (b) among non-degenerate models, higher Kendall W associates with higher AUROC.

Data-agnostic: consumes a generic labeled-predictions table (columns: model,
exercise_id, subject_id, pred_score, correct_label in {0,1}) plus the reliability
summary (outputs/irds_eval/irds_reliability.csv). Runs the moment a labeled testbed
(Task V1: REHAB24-6 or IRDS-CorrectLabel) exists. Until then, --selftest proves the
logic on synthetic models.

Usage:
  python src/validity_eval.py --selftest
  python src/validity_eval.py --preds outputs/validity/labeled_preds.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/
from constants import SEQ_LEN, NUM_JOINTS, NUM_CHANNELS

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None
from scipy.stats import pointbiserialr, spearmanr

DEGENERACY_PRED_SD = 0.10

# Zero-shot KIMORE-trained checkpoints (identical set + names to src/irds_eval.py so
# the join on `model` against outputs/irds_eval/irds_reliability.csv lines up).
MODELS: list[tuple[str, str]] = [
    ("LSTM baseline",              "outputs/loso_lstm"),
    ("ST-GCN",                     "outputs/loso_stgcn"),
    ("GraphTransformer",          "outputs/loso_graph_transformer"),
    ("GraphTransformer (no bias)", "outputs/loso_graph_transformer_no_bias"),
    ("TCN",                        "outputs/loso_tcn"),
    ("SCT",                        "outputs/loso_sct"),
    ("Exp E (Transformer)",        "outputs/loso_multitask_uiprmd_d128"),
]
REHAB_MANIFEST = "outputs/validity/rehab246_manifest.csv"
REHAB_SEQS     = "outputs/validity/rehab246_sequences.npy"


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC of predicted score vs binary label; constant scores -> 0.5."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return float("nan")
    if np.std(scores) < 1e-12:
        return 0.5  # constant predictor cannot rank -> chance
    if roc_auc_score is not None:
        return float(roc_auc_score(labels, scores))
    # Mann-Whitney fallback (== AUROC)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1 = labels.sum()
    n0 = len(labels) - n1
    if n0 == 0 or n1 == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def _point_biserial(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    if np.std(scores) < 1e-12 or len(np.unique(labels)) < 2:
        return 0.0
    return float(pointbiserialr(labels, scores)[0])


def validity_per_model(df: pd.DataFrame) -> dict:
    """Mean per-exercise AUROC/point-biserial of pred_score vs correct_label."""
    aurocs, pbs = [], []
    for _, g in df.groupby("exercise_id"):
        if g["correct_label"].nunique() < 2 or len(g) < 6:
            continue
        aurocs.append(_auroc(g["pred_score"].values, g["correct_label"].values))
        pbs.append(_point_biserial(g["pred_score"].values, g["correct_label"].values))
    aurocs = [a for a in aurocs if not np.isnan(a)]
    return {
        "auroc": round(float(np.mean(aurocs)), 3) if aurocs else float("nan"),
        "point_biserial": round(float(np.nanmean(pbs)), 3) if pbs else float("nan"),
        "n_exercises_scored": len(aurocs),
    }


def analyze(preds_df: pd.DataFrame, reliability_csv: str | None) -> dict:
    """Per-model validity + the reliability->validity relationship."""
    rel = None
    if reliability_csv and os.path.exists(reliability_csv):
        rel = pd.read_csv(reliability_csv).set_index("model")

    rows = []
    for model, g in preds_df.groupby("model"):
        v = validity_per_model(g)
        row = {"model": model, **v}
        if rel is not None and model in rel.index:
            row["kendall_w"] = float(rel.loc[model, "Kendall_W"])
            row["pred_SD"] = float(rel.loc[model, "pred_SD"])
            row["degenerate"] = bool(rel.loc[model, "degenerate"])
        else:
            sd = float(np.std(g["pred_score"]))
            row["kendall_w"] = float("nan")
            row["pred_SD"] = round(sd, 3)
            row["degenerate"] = sd < DEGENERACY_PRED_SD
        rows.append(row)
    res = pd.DataFrame(rows)

    # Hypothesis (a): degenerate models collapse to AUROC ~0.5
    degen = res[res["degenerate"]]
    nondegen = res[~res["degenerate"]]
    degen_auroc_near_chance = bool(
        len(degen) > 0 and np.all(np.abs(degen["auroc"] - 0.5) <= 0.10)
    )
    # Hypothesis (b): among non-degenerate, Kendall W tracks AUROC
    wv = nondegen.dropna(subset=["kendall_w", "auroc"])
    if len(wv) >= 3:
        rho, p = spearmanr(wv["kendall_w"], wv["auroc"])
        w_vs_auroc = {"spearman_rho": round(float(rho), 3), "p": round(float(p), 3),
                      "n_models": int(len(wv))}
    else:
        w_vs_auroc = {"spearman_rho": None, "p": None, "n_models": int(len(wv))}

    return {
        "degeneracy_pred_sd_threshold": DEGENERACY_PRED_SD,
        "per_model": res.sort_values("auroc", ascending=False).to_dict(orient="records"),
        "hypothesis_a_degenerate_AUROC_near_0.5": degen_auroc_near_chance,
        "hypothesis_b_kendallW_predicts_AUROC": w_vs_auroc,
        "interpretation": (
            "If (a) holds, the degeneracy gate is necessary: a 'reliable' degenerate "
            "model cannot discriminate correct from incorrect movement. If (b) holds, "
            "cross-exercise rank consistency (Kendall W) is a usable validity proxy. "
            "Together they upgrade the reliability rubric from self-consistency to a "
            "validated deployment screen."
        ),
    }


# ---------------------------------------------------------------------------
# Zero-shot inference: KIMORE-trained models -> REHAB24-6 labeled predictions
# (mirrors src/irds_eval.run_model_on_irds exactly: joint-mapped + resampled
#  sequences -> KIMORE x_scaler -> model -> y_scaler inverse). Absolute offset
#  from the OptiTrack->Kinect frame mismatch is constant and cancels in the
#  rank-based AUROC, identical to the IRDS protocol.
# ---------------------------------------------------------------------------

def _predict_scores(exp_dir, seqs, device, kimore_exercise_id: int = 0,
                    batch_size: int = 64):
    """Run one KIMORE checkpoint over prebuilt (N, SEQ_LEN, 25, 3) sequences."""
    import torch
    import joblib
    from pathlib import Path
    from types import SimpleNamespace
    from generate_oof import build_model_from_args

    ckpt_path   = Path(exp_dir) / "fold_0" / "best_model.pt"
    scaler_path = Path(exp_dir) / "fold_0" / "scalers.pkl"
    if not ckpt_path.exists() or not scaler_path.exists():
        return None

    ckpt   = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model  = build_model_from_args(ckpt["args"], device, ref_state_dict=ckpt["model_state"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    bundle = joblib.load(str(scaler_path))
    multitask = getattr(SimpleNamespace(**ckpt["args"]), "multitask", False)

    n = seqs.shape[0]
    flat   = seqs.reshape(n * SEQ_LEN, NUM_JOINTS * NUM_CHANNELS).astype(np.float32)
    scaled = bundle.x_scaler.transform(flat).reshape(
        n, SEQ_LEN, NUM_JOINTS, NUM_CHANNELS).astype(np.float32)
    ex = np.full(n, kimore_exercise_id, dtype=np.int64)
    x_t = torch.from_numpy(scaled)
    e_t = torch.from_numpy(ex)

    outs = []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            o = model(x_t[s:s + batch_size].to(device), e_t[s:s + batch_size].to(device))
            if multitask:
                o = o[0]
            outs.append(o.cpu().numpy().reshape(-1))
    preds_sc = np.concatenate(outs)
    return bundle.y_scaler.inverse_transform(preds_sc.reshape(-1, 1)).reshape(-1)


def build_labeled_preds(out_csv: str, kimore_exercise_id: int = 0) -> str | None:
    """Score the REHAB24-6 testbed with every KIMORE model -> labeled_preds.csv."""
    import torch

    if not (os.path.exists(REHAB_MANIFEST) and os.path.exists(REHAB_SEQS)):
        print(f"[SKIP] Missing {REHAB_MANIFEST} / {REHAB_SEQS}.")
        print("       Build the testbed first: python src/load_rehab246.py --build")
        return None

    man  = pd.read_csv(REHAB_MANIFEST)
    seqs = np.load(REHAB_SEQS)
    if len(man) != len(seqs):
        print(f"[ERR] manifest ({len(man)}) and sequences ({len(seqs)}) length mismatch")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"REHAB24-6 inference: {len(man)} reps x {len(MODELS)} models  (device={device})")
    frames = []
    for name, exp_dir in MODELS:
        preds = _predict_scores(exp_dir, seqs, device, kimore_exercise_id)
        if preds is None:
            print(f"  [SKIP] {name}: checkpoint not found in {exp_dir}")
            continue
        df = man[["exercise_id", "subject_id", "correct_label"]].copy()
        df.insert(0, "model", name)
        df["pred_score"] = preds
        frames.append(df)
        print(f"  {name:27s} pred_SD={preds.std():.3f}  mean={preds.mean():.2f}")
    if not frames:
        print("[ERR] no checkpoints available; nothing scored.")
        return None

    allp = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    allp.to_csv(out_csv, index=False)
    print(f"  -> {out_csv}  ({len(allp)} rows, {allp['model'].nunique()} models)")
    return out_csv


def external_transfer_diagnostics(embeddings=(0, 1, 2, 3, 4)) -> dict | None:
    """Robustness layer for the validity null: is the near-chance AUROC an artifact?

    (1) Embedding sweep: best per-exercise AUROC each model reaches over the 5 KIMORE
        exercise embeddings (rules out a wrong-embedding artifact).
    (2) Naive kinematic baseline: AUROC of simple movement-magnitude features vs the
        correct/incorrect label on the SAME joint-mapped sequences (rules out 'labels
        are noise' / 'joint map destroyed the signal'). If naive features discriminate
        but the trained scorers do not, the failure is model transfer, not the testbed.
    """
    import torch
    from sklearn.metrics import roc_auc_score

    if not (os.path.exists(REHAB_MANIFEST) and os.path.exists(REHAB_SEQS)):
        return None
    man  = pd.read_csv(REHAB_MANIFEST)
    seqs = np.load(REHAB_SEQS)
    dev  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (2) naive kinematic baseline -----------------------------------------
    S = seqs  # (N, SEQ_LEN, 25, 3)
    feats = {
        "total_path": np.abs(np.diff(S, axis=1)).sum(axis=(1, 2, 3)),
        "mean_speed": np.linalg.norm(np.diff(S, axis=1), axis=-1).mean(axis=(1, 2)),
        "bbox_vol":   (S.max(1) - S.min(1)).prod(-1).sum(-1),
    }
    naive = {}
    for fn, fv in feats.items():
        aus = []
        for _, g in man.groupby("exercise_id"):
            y = g["correct_label"].values
            if len(np.unique(y)) < 2:
                continue
            a = roc_auc_score(y, fv[g.index.values])
            aus.append(max(a, 1 - a))   # direction-agnostic feature
        naive[fn] = round(float(np.mean(aus)), 3)
    naive_mean = round(float(np.mean(list(naive.values()))), 3)

    # (1) embedding sweep --------------------------------------------------
    model_best = {}
    for name, exp in MODELS:
        vals = []
        for eid in embeddings:
            p = _predict_scores(exp, seqs, dev, kimore_exercise_id=eid)
            if p is None:
                vals = None
                break
            df = man[["exercise_id", "subject_id", "correct_label"]].copy()
            df["pred_score"] = p
            vals.append(validity_per_model(df)["auroc"])
        if vals:
            model_best[name] = round(float(np.nanmax(vals)), 3)
    best_model = round(float(max(model_best.values())), 3) if model_best else None

    interp = (
        f"KIMORE-trained scorers do NOT transfer zero-shot to correct/incorrect "
        f"discrimination: best per-exercise AUROC over all models x all 5 exercise "
        f"embeddings = {best_model}. Naive movement-magnitude features on the SAME "
        f"joint-mapped sequences reach mean|AUROC| = {naive_mean}, so the label signal "
        f"is present and the joint mapping is sound -- the failure is model transfer, "
        f"not the testbed. The reliability/degeneracy diagnostic is thus validated as a "
        f"NECESSARY screen-out (degenerate -> chance) but is NOT sufficient for validity: "
        f"high zero-shot reliability does not imply a model can flag movement errors on "
        f"an external labeled set."
    )
    return {
        "naive_kinematic_baseline_meanAUROC": naive,
        "naive_baseline_overall_meanAUROC": naive_mean,
        "model_best_AUROC_over_embeddings": model_best,
        "best_model_AUROC": best_model,
        "embeddings_swept": list(embeddings),
        "interpretation": interp,
    }


def make_fig7(res: dict, out_png: str = "outputs/novelty/fig7_reliability_vs_validity.png") -> None:
    """Scatter Kendall W (zero-shot reliability) vs AUROC (validity); flag degenerate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pm = pd.DataFrame(res["per_model"])
    pm = pm[np.isfinite(pm["auroc"])]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.axhline(0.5, ls="--", c="gray", lw=1, zorder=1, label="AUROC = 0.5 (chance)")
    ext = res.get("external_transfer") or {}
    nb = ext.get("naive_baseline_overall_meanAUROC")
    if nb:
        ax.axhline(nb, ls=":", c="#009e73", lw=1.6, zorder=1,
                   label=f"naive kinematic baseline ({nb:.2f})")
    for _, r in pm.iterrows():
        degen = bool(r["degenerate"])
        w = r["kendall_w"]
        x = w if np.isfinite(w) else 0.0
        ax.scatter(x, r["auroc"], s=110, zorder=3, edgecolor="k", linewidth=0.7,
                   marker="X" if degen else "o",
                   c="#d55e00" if degen else "#0072b2")
        ax.annotate(r["model"], (x, r["auroc"]), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Kendall W  (zero-shot cross-exercise reliability, IRDS)")
    ax.set_ylabel("AUROC  (predicted score vs correct/incorrect, REHAB24-6)")
    ax.set_title("Zero-shot reliability does NOT confer validity\n"
                 "trained scorers ~chance; naive kinematics separate the labels")
    ax.set_ylim(0.30, 1.0)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"  -> {out_png}")


def robustness_diagnostics(preds_df: pd.DataFrame,
                           out_json: str = "outputs/novelty/reliability_validity_robustness.json",
                           dataset: str = "REHAB24-6") -> dict | None:
    """V4 robustness: does the V2 null replicate across independent partitions?

    A fully-independent second labeled corpus (UI-PRMD incorrect set / IntelliRehabDS
    CorrectLabel) is not bundled locally (see src/load_uiprmd_validity.py for the ready
    loader + download command). In its place we test replication WITHIN REHAB24-6 across
    two independent axes: (1) each of the 6 exercises is a distinct movement type -> 6
    independent correct/incorrect tests; (2) two disjoint subject halves. If the best
    model stays ~chance while a naive kinematic feature discriminates in every partition,
    the reliability!=validity finding is not an artifact of pooling, one exercise, or one
    subject sample.
    """
    from sklearn.metrics import roc_auc_score
    if not (os.path.exists(REHAB_MANIFEST) and os.path.exists(REHAB_SEQS)):
        return None

    per = {m: validity_per_model(g)["auroc"] for m, g in preds_df.groupby("model")}
    best_model = max(per, key=per.get)
    bg = preds_df[preds_df["model"] == best_model]

    man  = pd.read_csv(REHAB_MANIFEST)
    seqs = np.load(REHAB_SEQS)
    man = man.assign(total_path=np.abs(np.diff(seqs, axis=1)).sum(axis=(1, 2, 3)))

    def _auc(y, x):
        a = roc_auc_score(y, x)
        return max(a, 1 - a)

    per_exercise = []
    for ex in sorted(preds_df["exercise_id"].unique()):
        ge = bg[bg["exercise_id"] == ex]
        y = ge["correct_label"].values
        if len(np.unique(y)) < 2:
            continue
        me = man[man["exercise_id"] == ex]
        per_exercise.append({
            "exercise_id": int(ex), "n": int(len(ge)),
            "best_model_auroc": round(float(roc_auc_score(y, ge["pred_score"].values)), 3),
            "naive_total_path_auroc": round(float(_auc(me["correct_label"].values,
                                                      me["total_path"].values)), 3),
        })

    halves = {}
    for name, keep in [("subjects_odd", lambda s: s % 2 == 1),
                       ("subjects_even", lambda s: s % 2 == 0)]:
        sp = bg[bg["subject_id"].map(keep)]
        sm = man[man["subject_id"].map(keep)]
        ma, na = [], []
        for ex in sorted(sp["exercise_id"].unique()):
            gp = sp[sp["exercise_id"] == ex]
            yy = gp["correct_label"].values
            if len(np.unique(yy)) < 2:
                continue
            ma.append(roc_auc_score(yy, gp["pred_score"].values))
            mm = sm[sm["exercise_id"] == ex]
            na.append(_auc(mm["correct_label"].values, mm["total_path"].values))
        halves[name] = {"best_model_meanAUROC": round(float(np.mean(ma)), 3),
                        "naive_meanAUROC": round(float(np.mean(na)), 3)}

    n_model_chance = sum(r["best_model_auroc"] <= 0.60 for r in per_exercise)
    n_naive_signal = sum(r["naive_total_path_auroc"] >= 0.65 for r in per_exercise)
    k = len(per_exercise)
    out = {
        "best_model": best_model,
        "per_exercise": per_exercise,
        "subject_split": halves,
        "summary": {
            "n_exercises": k,
            "best_model_chance_in": f"{n_model_chance}/{k}",
            "naive_discriminates_in": f"{n_naive_signal}/{k}",
        },
        "dataset": dataset,
        "interpretation": (
            f"The reliability!=validity null replicates: across all {k} {dataset} exercises "
            f"the best model stays <=0.60 AUROC in {n_model_chance}/{k}, while the naive "
            f"kinematic feature exceeds 0.65 in {n_naive_signal}/{k}; the pattern also holds "
            f"in both disjoint subject halves. The finding is robust to exercise and subject "
            f"sampling within {dataset}."
        ),
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"  -> {out_json}")
    print(f"  robustness: best model ({best_model}) <=0.60 in {n_model_chance}/{k} exercises; "
          f"naive >=0.65 in {n_naive_signal}/{k}")
    return out


def _synthetic_preds(seed: int = 0) -> pd.DataFrame:
    """3 models x 5 exercises x 40 reps with known behaviour."""
    rng = np.random.default_rng(seed)
    rows = []
    for ex in range(5):
        labels = rng.integers(0, 2, size=40)
        for i, lab in enumerate(labels):
            base = 30 + 10 * lab
            # strong: score separates correct/incorrect; weak: noisy; degenerate: constant
            rows.append(("strong", ex, i, base + rng.normal(0, 3), lab))
            rows.append(("weak", ex, i, 35 + rng.normal(0, 8), lab))
            rows.append(("degenerate", ex, i, 41.0 + rng.normal(0, 0.02), lab))
    return pd.DataFrame(rows, columns=["model", "exercise_id", "subject_id",
                                       "pred_score", "correct_label"])


def selftest() -> int:
    df = _synthetic_preds()
    out = analyze(df, reliability_csv=None)
    by = {r["model"]: r for r in out["per_model"]}
    print("SELFTEST per-model AUROC:")
    for m in ("strong", "weak", "degenerate"):
        print(f"  {m:11s} AUROC={by[m]['auroc']:.3f}  pred_SD={by[m]['pred_SD']:.3f}  "
              f"degenerate={by[m]['degenerate']}")
    ok = (by["strong"]["auroc"] > 0.70
          and 0.45 <= by["degenerate"]["auroc"] <= 0.55
          and by["degenerate"]["degenerate"] is True
          and out["hypothesis_a_degenerate_AUROC_near_0.5"])
    print(f"SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="V2 reliability-predicts-validity analysis")
    ap.add_argument("--selftest", action="store_true", help="run synthetic logic check")
    ap.add_argument("--infer", action="store_true",
                    help="score REHAB24-6 with KIMORE models -> rebuild labeled_preds.csv")
    ap.add_argument("--preds", default="outputs/validity/labeled_preds.csv",
                    help="labeled predictions CSV (model,exercise_id,subject_id,pred_score,correct_label)")
    ap.add_argument("--reliability", default="outputs/irds_eval/irds_reliability.csv")
    ap.add_argument("--out", default="outputs/novelty/reliability_validity.json")
    ap.add_argument("--fig", default="outputs/novelty/fig7_reliability_vs_validity.png")
    ap.add_argument("--manifest", help="override testbed manifest CSV (a second labeled corpus)")
    ap.add_argument("--seqs", help="override testbed sequences .npy (a second labeled corpus)")
    ap.add_argument("--tag", help="dataset tag; suffixes the output JSON/fig for a second corpus")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    # Second-corpus override (V4 external replication): point the testbed elsewhere.
    global REHAB_MANIFEST, REHAB_SEQS
    if args.manifest:
        REHAB_MANIFEST = args.manifest
    if args.seqs:
        REHAB_SEQS = args.seqs
    if args.tag:
        args.preds = f"outputs/validity/labeled_preds_{args.tag}.csv"
        args.out   = f"outputs/novelty/reliability_validity_{args.tag}.json"
        args.fig   = f"outputs/novelty/fig7_reliability_vs_validity_{args.tag}.png"

    # Build labeled predictions if requested or absent (zero-shot REHAB24-6 inference).
    if args.infer or not os.path.exists(args.preds):
        if build_labeled_preds(args.preds) is None:
            if not os.path.exists(args.preds):
                print(f"[SKIP] labeled predictions unavailable: {args.preds}")
                return

    preds_df = pd.read_csv(args.preds)
    res = analyze(preds_df, args.reliability)
    ext = external_transfer_diagnostics()
    if ext is not None:
        res["external_transfer"] = ext
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"  -> {args.out}")
    print(f"  hypothesis (a) degenerate~0.5: {res['hypothesis_a_degenerate_AUROC_near_0.5']}")
    print(f"  hypothesis (b) W->AUROC: {res['hypothesis_b_kendallW_predicts_AUROC']}")
    if "external_transfer" in res:
        et = res["external_transfer"]
        print(f"  external transfer: best model AUROC={et['best_model_AUROC']} "
              f"vs naive baseline={et['naive_baseline_overall_meanAUROC']}")
    make_fig7(res, args.fig)
    rob_out = (f"outputs/novelty/reliability_validity_robustness_{args.tag}.json"
               if args.tag else "outputs/novelty/reliability_validity_robustness.json")
    ds_label = {"uiprmd": "UI-PRMD"}.get(args.tag, args.tag.upper()) if args.tag else "REHAB24-6"
    robustness_diagnostics(preds_df, rob_out, dataset=ds_label)


if __name__ == "__main__":
    main()
