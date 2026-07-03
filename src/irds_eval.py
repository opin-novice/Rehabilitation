"""Zero-shot external reliability validation on IntelliRehabDS (IRDS).

DATASET STRUCTURE (confirmed by bone-length analysis):
  m{EE}_s{SS}_e{RR}_positions.txt
  - EE: exercise type (01-10, ten different rehabilitation exercises)
  - SS: subject (01-10, the SAME 10 subjects perform all exercises)
  - RR: repetition (01-10, ten trials of each exercise per subject)
  Total: 10 exercises x 10 subjects x 10 repetitions = 1000 sequences
  22 Kinect joints x 3 coordinates = 66 columns per frame

NOTE ON LABELS: This dataset does not carry healthy/patient labels in the filename.
We therefore evaluate zero-shot RELIABILITY properties, which are the standard clinical
criteria for automated assessment instruments:

  1. Test-Retest ICC (ICC(2,1), two-way random, absolute agreement; repetitions treated as
     random occasions): Across 10 repetitions of the same exercise by the same subject,
     predictions should be consistent (ICC > 0.75 = acceptable; > 0.90 = excellent).

  2. Cross-Exercise Rank Consistency (Kendall W): Rank the 10 subjects by predicted quality
     for each exercise. Kendall W (coefficient of concordance) measures agreement across
     the 10 exercise rankings. W > 0.7 = good; W > 0.9 = excellent.
     Clinical interpretation: a model with high W captures a stable subject-level quality
     attribute ("how well this person moves") that generalises across exercise types.

  3. Between-Subject Discriminability (F-ratio): ANOVA F-statistic testing whether
     between-subject variance exceeds within-subject (between-repetition) variance.
     High F = model can distinguish subjects; low F = model collapses to mean.

  4. Score Distribution Spread: Mean and SD of predicted scores across all sequences.
     A model that predicts nearly the same score for every sequence (SD < 0.5) has
     failed to discriminate and cannot be used clinically.

Usage:
  python src/irds_eval.py
  python src/irds_eval.py --irds_dir "Segmented Movements/Kinect/Positions" --out_dir outputs/irds_eval
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from scipy.stats import f_oneway

sys.path.insert(0, str(Path(__file__).parent))
from rehab_dataset import NUM_CHANNELS, NUM_JOINTS, ScalerBundle
from generate_oof import build_model_from_args

SEQ_LEN    = 100
IRDS_JOINTS = 22
IRDS_COLS   = IRDS_JOINTS * NUM_CHANNELS  # 66

MODELS: list[tuple[str, str]] = [
    ("LSTM baseline",             "outputs/loso_lstm"),
    ("ST-GCN",                    "outputs/loso_stgcn"),
    ("GraphTransformer",          "outputs/loso_graph_transformer"),
    ("GraphTransformer (no bias)","outputs/loso_graph_transformer_no_bias"),
    ("TCN",                       "outputs/loso_tcn"),
    ("SCT",                       "outputs/loso_sct"),
    ("Exp E (Transformer)",       "outputs/loso_multitask_uiprmd_d128"),
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _resample(seq: np.ndarray, target: int) -> np.ndarray:
    T = seq.shape[0]
    if T == target:
        return seq
    f = interp1d(np.linspace(0, 1, T), seq, axis=0, kind="linear")
    return f(np.linspace(0, 1, target)).astype(np.float32)


def load_irds_sequence(fpath: str) -> np.ndarray:
    """Load one IRDS file -> [SEQ_LEN, 25, 3]. Pads 22->25 joints with zeros."""
    raw = np.loadtxt(fpath, delimiter=",", dtype=np.float32)
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    T = raw.shape[0]
    xyz22 = raw.reshape(T, IRDS_JOINTS, NUM_CHANNELS)
    xyz25 = np.zeros((T, NUM_JOINTS, NUM_CHANNELS), dtype=np.float32)
    xyz25[:, :IRDS_JOINTS, :] = xyz22
    resampled = _resample(xyz25.reshape(T, -1), SEQ_LEN)
    return resampled.reshape(SEQ_LEN, NUM_JOINTS, NUM_CHANNELS)


def load_all_irds(irds_dir: str) -> pd.DataFrame:
    """Return metadata DataFrame for all IRDS sequences."""
    records = []
    for fname in sorted(os.listdir(irds_dir)):
        if not fname.endswith("_positions.txt"):
            continue
        parts = fname.replace("_positions.txt", "").split("_")
        if len(parts) != 3:
            continue
        try:
            eid = int(parts[0][1:])   # m03 -> exercise 3
            sid = int(parts[1][1:])   # s07 -> subject 7
            rid = int(parts[2][1:])   # e05 -> repetition 5
        except (IndexError, ValueError):
            continue
        records.append({
            "filepath":    os.path.join(irds_dir, fname),
            "exercise_id": eid,
            "subject_id":  sid,
            "rep_id":      rid,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Sequence cache export (for downstream SSL pooling)
# ---------------------------------------------------------------------------

def build_cache(irds_dir: str, out_dir: str) -> dict:
    """Export every IRDS sequence to an (N, SEQ_LEN, 25, 3) .npy + manifest CSV.

    Consumed by src/selfsup/data.load_irds_unlabeled (glob '*seq*.npy' +
    '*manifest*.csv', subject_id column). IRDS carries no correctness labels,
    so no correct_label column is written; the loader treats labels as None.
    """
    meta_df = load_all_irds(irds_dir)
    if meta_df.empty:
        print(f"[SKIP] build_cache: no '*_positions.txt' sequences under {irds_dir}")
        return {}

    n = len(meta_df)
    seqs = np.zeros((n, SEQ_LEN, NUM_JOINTS, NUM_CHANNELS), dtype=np.float32)
    for i, row in enumerate(meta_df.itertuples()):
        seqs[i] = load_irds_sequence(row.filepath)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    seq_path = os.path.join(out_dir, "irds_sequences.npy")
    np.save(seq_path, seqs)

    man = pd.DataFrame({
        "rep_uid": [f"irds_m{e:02d}_s{s:02d}_e{r:02d}"
                    for e, s, r in zip(meta_df.exercise_id,
                                       meta_df.subject_id,
                                       meta_df.rep_id)],
        "exercise_id": meta_df.exercise_id.values,
        "subject_id":  meta_df.subject_id.values,
        "rep_id":      meta_df.rep_id.values,
    })
    man_path = os.path.join(out_dir, "irds_manifest.csv")
    man.to_csv(man_path, index=False)

    print(f"IRDS cache: {n} sequences | {meta_df.exercise_id.nunique()} exercises | "
          f"{meta_df.subject_id.nunique()} subjects")
    print(f"  -> {seq_path}  shape={seqs.shape}")
    print(f"  -> {man_path}")
    return {"n": n, "seq_path": seq_path, "man_path": man_path}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_model_on_irds(
    exp_dir: str,
    meta_df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 64,
    kimore_exercise_id: int = 0,
) -> np.ndarray | None:
    """Zero-shot inference. Returns predicted scores aligned to meta_df rows."""
    ckpt_path   = Path(exp_dir) / "fold_0" / "best_model.pt"
    scaler_path = Path(exp_dir) / "fold_0" / "scalers.pkl"
    if not ckpt_path.exists() or not scaler_path.exists():
        return None

    ckpt      = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model     = build_model_from_args(ckpt["args"], device, ref_state_dict=ckpt["model_state"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    bundle: ScalerBundle = joblib.load(str(scaler_path))
    multitask = getattr(SimpleNamespace(**ckpt["args"]), "multitask", False)

    n = len(meta_df)
    flat = np.zeros((n, SEQ_LEN * NUM_JOINTS * NUM_CHANNELS), dtype=np.float32)
    for i, row in enumerate(meta_df.itertuples()):
        flat[i] = load_irds_sequence(row.filepath).reshape(-1)

    scaled = bundle.x_scaler.transform(
        flat.reshape(n * SEQ_LEN, NUM_JOINTS * NUM_CHANNELS)
    ).reshape(n, SEQ_LEN, NUM_JOINTS, NUM_CHANNELS).astype(np.float32)

    ex_ids = np.full(n, kimore_exercise_id, dtype=np.int64)
    x_t = torch.from_numpy(scaled)
    e_t = torch.from_numpy(ex_ids)

    preds_sc = []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            xb = x_t[s:s+batch_size].to(device)
            eb = e_t[s:s+batch_size].to(device)
            out = model(xb, eb)
            if multitask:
                out = out[0]
            preds_sc.append(out.cpu().numpy().reshape(-1))

    preds_sc = np.concatenate(preds_sc)
    preds = bundle.y_scaler.inverse_transform(preds_sc.reshape(-1, 1)).reshape(-1)
    return preds


# ---------------------------------------------------------------------------
# Reliability metrics
# ---------------------------------------------------------------------------

def icc_two_way(score_matrix: np.ndarray) -> float:
    """ICC(2,1) two-way mixed-effects, absolute agreement.

    score_matrix: shape (n_subjects, k_repetitions)
    This is the standard reliability model:
      rows = subjects (random effect), cols = repetitions (fixed effect / occasions)
    ICC(2,1) = (MSr - MSe) / (MSr + (k-1)*MSe + k*(MSc-MSe)/n)
    Reference: Shrout & Fleiss (1979) Eq. for ICC(2,1); Koo & Li (2016).
    """
    mat = np.asarray(score_matrix, dtype=float)
    n, k = mat.shape
    if n < 2 or k < 2:
        return float("nan")

    grand = mat.mean()
    row_means = mat.mean(axis=1, keepdims=True)
    col_means = mat.mean(axis=0, keepdims=True)

    SSr = k * np.sum((row_means.squeeze() - grand) ** 2)       # between subjects
    SSc = n * np.sum((col_means.squeeze() - grand) ** 2)       # between repetitions
    SSt = np.sum((mat - grand) ** 2)                            # total
    SSe = SSt - SSr - SSc                                       # residual/error

    MSr = SSr / (n - 1)
    MSc = SSc / (k - 1)
    MSe = SSe / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0.0

    denom = MSr + (k - 1) * MSe + k * max(MSc - MSe, 0) / n
    if denom < 1e-12:
        return float("nan")
    return float(np.clip((MSr - MSe) / denom, -1, 1))


def kendall_w(rank_matrix: np.ndarray) -> float:
    """Kendall W (coefficient of concordance) for a (raters x subjects) rank matrix."""
    m, n = rank_matrix.shape   # m=exercises (raters), n=subjects
    if m < 2 or n < 2:
        return float("nan")
    rank_sums = rank_matrix.sum(axis=0)
    mean_rs   = rank_sums.mean()
    ss        = np.sum((rank_sums - mean_rs) ** 2)
    W = 12 * ss / (m ** 2 * (n ** 3 - n))
    return float(np.clip(W, 0, 1))


# ---------------------------------------------------------------------------
# Evaluation per model
# ---------------------------------------------------------------------------

def evaluate_reliability(meta_df: pd.DataFrame, preds: np.ndarray, model_name: str) -> dict:
    df = meta_df.copy()
    df["pred"] = preds

    exercises = sorted(df["exercise_id"].unique())
    subjects  = sorted(df["subject_id"].unique())
    n_ex, n_sub = len(exercises), len(subjects)

    # ── 1. Test-retest ICC(2,1) per exercise ─────────────────────────────────
    # For each exercise: build (n_subjects x n_reps) matrix, compute ICC(2,1).
    # Rows = subjects, Cols = repetitions.  Measures rep-to-rep consistency.
    icc_cells = []
    for eid in exercises:
        ex_df = df[df["exercise_id"] == eid]
        reps = sorted(ex_df["rep_id"].unique())
        mat_rows = []
        for sid in subjects:
            row = []
            for rid in reps:
                v = ex_df[(ex_df["subject_id"] == sid) & (ex_df["rep_id"] == rid)]["pred"].values
                row.append(float(v[0]) if len(v) > 0 else np.nan)
            mat_rows.append(row)
        mat = np.array(mat_rows)  # (n_subjects, n_reps)
        if not np.isnan(mat).any() and mat.shape[0] >= 2 and mat.shape[1] >= 2:
            icc_val = icc_two_way(mat)
            if not np.isnan(icc_val):
                icc_cells.append(icc_val)
    mean_icc = float(np.mean(icc_cells)) if icc_cells else float("nan")

    # ── 2. Cross-exercise rank consistency (Kendall W) ──────────────────────
    # Build rank matrix: rows=exercises, cols=subjects
    # Entry = rank of subject sid when scoring exercise eid (mean over repetitions)
    subj_ex_mean = df.groupby(["exercise_id", "subject_id"])["pred"].mean().unstack("subject_id")
    # Rank subjects within each exercise (ascending: lower rank = lower score)
    rank_mat = subj_ex_mean.rank(axis=1).values   # shape (n_ex, n_sub)
    W = kendall_w(rank_mat)

    # ── 3. Between-subject discriminability (ANOVA F per exercise) ───────────
    f_vals = []
    for eid in exercises:
        ex_df = df[df["exercise_id"] == eid]
        # Each subject's 10 repetitions as a group
        groups = [ex_df[ex_df["subject_id"] == sid]["pred"].values for sid in subjects]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            F, p = f_oneway(*groups)
            if not np.isnan(F):
                f_vals.append(F)
    mean_F = float(np.mean(f_vals)) if f_vals else float("nan")

    # ── 4. Score distribution ────────────────────────────────────────────────
    pred_mean = float(preds.mean())
    pred_std  = float(preds.std())
    # Within-subject SD (mean of per-subject SDs across all exercises)
    within_sd = float(df.groupby(["exercise_id","subject_id"])["pred"].std().mean())
    # Between-subject SD (SD of per-subject means across exercises)
    between_sd = float(df.groupby("subject_id")["pred"].mean().std())

    # ── 5. Subject rank agreement across exercise-id choices ─────────────────
    # Measure whether the ordering of subjects is consistent across the 10 exercises
    # via mean pairwise Spearman rho between exercise-level subject rankings
    ex_scores = {}
    for eid in exercises:
        ex_scores[eid] = df[df["exercise_id"]==eid].groupby("subject_id")["pred"].mean().values
    if len(ex_scores) >= 2:
        from scipy.stats import spearmanr
        rhos = []
        eids = sorted(ex_scores.keys())
        for i in range(len(eids)):
            for j in range(i+1, len(eids)):
                rho, _ = spearmanr(ex_scores[eids[i]], ex_scores[eids[j]])
                rhos.append(rho)
        mean_cross_rho = float(np.mean(rhos))
    else:
        mean_cross_rho = float("nan")

    return {
        "model":              model_name,
        "n_sequences":        len(df),
        "n_exercises":        n_ex,
        "n_subjects":         n_sub,
        "ICC_testretest":     round(mean_icc, 3),
        "Kendall_W":          round(W, 3),
        "mean_F_between_subj":round(mean_F, 2),
        "pred_mean":          round(pred_mean, 2),
        "pred_SD":            round(pred_std, 2),
        "within_subject_SD":  round(within_sd, 2),
        "between_subject_SD": round(between_sd, 2),
        "SB_ratio":           round(between_sd / within_sd, 3) if within_sd > 0.01 else float("nan"),
        "mean_cross_ex_rho":  round(mean_cross_rho, 3),
        "degenerate":         bool(pred_std < 0.10),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--irds_dir", default=r"Segmented Movements\Kinect\Positions",
        help="Path to IRDS Kinect Positions directory."
    )
    parser.add_argument("--out_dir", default="outputs/irds_eval")
    parser.add_argument("--build", action="store_true",
                        help="Export the IRDS sequence cache (.npy + manifest) and exit. "
                             "Required before SSL pooling with --pool irds_only.")
    parser.add_argument("--kimore_exercise_id", type=int, default=0,
                        help="KIMORE exercise embedding index (0=Trunk Lateral Flex.) for all IRDS.")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.build:
        build_cache(args.irds_dir, args.out_dir)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_df = load_all_irds(args.irds_dir)
    n_ex   = meta_df["exercise_id"].nunique()
    n_sub  = meta_df["subject_id"].nunique()
    n_rep  = meta_df["rep_id"].nunique()

    print(f"IRDS: {len(meta_df)} sequences | "
          f"{n_ex} exercises | {n_sub} subjects | {n_rep} repetitions each")
    print(f"Design: {n_sub} subjects x {n_ex} exercises x {n_rep} reps (confirmed same subjects across exercises)")
    print(f"Evaluation: zero-shot reliability (ICC, Kendall W, subject discriminability)")
    print(f"KIMORE exercise embedding: {args.kimore_exercise_id}\n")

    # Sensitivity check: try all 5 KIMORE exercise embeddings
    print("=== Embedding sensitivity (TCN, exercise_id 0-4) ===")
    tcn_dir = "outputs/loso_tcn"
    ckpt_path = Path(tcn_dir) / "fold_0" / "best_model.pt"
    if ckpt_path.exists():
        eid_iccs = []
        for eid_kimore in range(5):
            preds = run_model_on_irds(tcn_dir, meta_df, device, kimore_exercise_id=eid_kimore)
            if preds is not None:
                res = evaluate_reliability(meta_df, preds, "TCN")
                eid_iccs.append(res["ICC_testretest"])
                print(f"  exercise_id={eid_kimore}: ICC={res['ICC_testretest']:.3f}  "
                      f"W={res['Kendall_W']:.3f}  SD={res['pred_SD']:.2f}")
        icc_range = max(eid_iccs) - min(eid_iccs) if eid_iccs else 0
        print(f"  ICC range across embeddings: {icc_range:.3f}  "
              f"({'STABLE' if icc_range < 0.02 else 'SENSITIVE'})\n")
    else:
        print("  [SKIP] TCN checkpoint not found\n")

    results = []
    for model_name, exp_dir in MODELS:
        print(f"[{model_name}]")
        preds = run_model_on_irds(exp_dir, meta_df, device,
                                  kimore_exercise_id=args.kimore_exercise_id)
        if preds is None:
            print(f"  [SKIP] checkpoint not found")
            continue

        res = evaluate_reliability(meta_df, preds, model_name)
        results.append(res)

        icc_interp = ("excellent" if res["ICC_testretest"] >= 0.90
                      else "good" if res["ICC_testretest"] >= 0.75
                      else "moderate" if res["ICC_testretest"] >= 0.50
                      else "poor")
        w_interp = ("excellent" if res["Kendall_W"] >= 0.90
                    else "good" if res["Kendall_W"] >= 0.70
                    else "moderate" if res["Kendall_W"] >= 0.50
                    else "poor")

        print(f"  ICC(2,1) test-retest     = {res['ICC_testretest']:.3f}  [{icc_interp}]")
        print(f"  Kendall W (rank consist.)= {res['Kendall_W']:.3f}  [{w_interp}]")
        print(f"  Mean F (between-subj.)   = {res['mean_F_between_subj']:.2f}")
        print(f"  S/B ratio                = {res['SB_ratio']:.3f}  (between/within SD)")
        print(f"  Cross-exercise rho       = {res['mean_cross_ex_rho']:.3f}")
        print(f"  Score: mean={res['pred_mean']:.1f}  SD={res['pred_SD']:.2f}  "
              f"(within={res['within_subject_SD']:.2f} / between={res['between_subject_SD']:.2f})")
        if res["degenerate"]:
            print("  [WARNING] DEGENERATE: pred_SD<0.10 — model predicts near-constant output; "
                  "ICC/Kendall W are NOT interpretable for this model.")
        print()

    # ── Embedding sensitivity: all models × all 5 KIMORE exercise IDs ────────
    print("=== Embedding sensitivity (all models, exercise_id 0-4) ===")
    sens_rows = []
    for model_name, exp_dir in MODELS:
        for eid_kimore in range(5):
            preds = run_model_on_irds(exp_dir, meta_df, device, kimore_exercise_id=eid_kimore)
            if preds is None:
                continue
            res = evaluate_reliability(meta_df, preds, f"{model_name}_eid{eid_kimore}")
            sens_rows.append({
                "model":       model_name,
                "exercise_id": eid_kimore,
                "ICC":         res["ICC_testretest"],
                "Kendall_W":   res["Kendall_W"],
                "cross_ex_rho":res["mean_cross_ex_rho"],
                "pred_SD":     res["pred_SD"],
            })
    if sens_rows:
        sens_df = pd.DataFrame(sens_rows)
        sens_path = os.path.join(args.out_dir, "embedding_sensitivity_full.csv")
        sens_df.to_csv(sens_path, index=False)
        print(f"  Full sensitivity -> {sens_path}")

        # Per-model mean ± SD across the 5 embeddings
        sens_sum = sens_df.groupby("model").agg(
            ICC_mean=("ICC", "mean"), ICC_sd=("ICC", "std"),
            W_mean=("Kendall_W", "mean"), W_sd=("Kendall_W", "std"),
            rho_mean=("cross_ex_rho", "mean"), rho_sd=("cross_ex_rho", "std"),
        ).reset_index()
        sens_sum_path = os.path.join(args.out_dir, "embedding_sensitivity_summary.csv")
        sens_sum.to_csv(sens_sum_path, index=False)
        print(f"  Summary -> {sens_sum_path}")
        for _, row in sens_sum.iterrows():
            print(f"  {row['model']}: ICC {row['ICC_mean']:.3f}+-{row['ICC_sd']:.3f}  "
                  f"W {row['W_mean']:.3f}+-{row['W_sd']:.3f}")
        print()

    if not results:
        print("No models evaluated.")
        return

    summary_df = pd.DataFrame(results)
    sp = os.path.join(args.out_dir, "irds_reliability.csv")
    summary_df.to_csv(sp, index=False)
    print(f"Summary -> {sp}")

    report_path = os.path.join(args.out_dir, "irds_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("IRDS Zero-Shot External Reliability Validation\n")
        f.write("=" * 72 + "\n\n")
        f.write("Dataset structure: 10 subjects x 10 exercises x 10 repetitions = 1000 sequences\n")
        f.write("Bone-length analysis confirms: m=exercise type, s=subject (same 10 subjects\n")
        f.write("across all exercises). No healthy/patient labels present in file naming.\n\n")
        f.write("Protocol: KIMORE-trained models applied zero-shot (no IRDS training).\n")
        f.write(f"KIMORE exercise embedding {args.kimore_exercise_id} used for all sequences.\n")
        f.write("22-joint IRDS skeleton padded to 25 joints (zeros for joints 22-24).\n")
        f.write("Temporal resampling: IRDS frames -> 100 frames (linear interpolation).\n\n")
        f.write("Clinical reliability thresholds (Koo & Li, 2016; Landis & Koch, 1977):\n")
        f.write("  ICC > 0.90 = excellent;  ICC 0.75-0.90 = good;\n")
        f.write("  ICC 0.50-0.75 = moderate;  ICC < 0.50 = poor\n")
        f.write("  Kendall W > 0.70 = good agreement; W > 0.90 = excellent\n\n")
        f.write("-" * 72 + "\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\nInterpretation:\n")
        f.write("  ICC(2,1): Test-retest reliability across 10 repetitions per (exercise, subject) cell.\n")
        f.write("  Kendall W: Agreement in subject ranking across all 10 exercise types.\n")
        f.write("  S/B ratio: Signal-to-background = between-subject SD / within-subject SD.\n")
        f.write("             >1.0 means model discriminates subjects better than noise.\n")
        f.write("  Cross-exercise rho: Mean Spearman across all exercise-pair subject rankings.\n")
        f.write("             Positive = subjects who score high on one exercise score high on others.\n")
        f.write("  degenerate: TRUE if pred_SD<0.10 (model collapsed to near-constant output);\n")
        f.write("             ICC/Kendall W are NOT interpretable for such models.\n")
        degen_models = summary_df.loc[summary_df["degenerate"], "model"].tolist()
        if degen_models:
            f.write(f"\n  WARNING - degenerate models (exclude from reliability claims): {degen_models}\n")

    print(f"Report -> {report_path}")
    print("\nIRDS reliability evaluation complete.")


if __name__ == "__main__":
    main()
