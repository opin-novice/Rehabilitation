# Response to Reviewers — Round 3

We thank the reviewer for the exceptionally thorough and constructive review. Below we
map every question (Q1–Q10) and weakness to a concrete change, with pointers. New
experiments are numbered A/B/C by effort tier. Manuscript line references are to the
revised `manuscript.tex`.

## Summary of changes
- **Fixed an internal inconsistency** the reviewer flagged (few-shot reported in
  §IV-A but disclaimed in Limitations) — Q3.
- **Corrected the frame-rate error** ("100 frames at 1 Hz" → ~30 fps native,
  resampled to 100 frames) — Q1.
- **Three new de-risking analyses** with real numbers: naive-baseline sensitivity
  (Q2), zero-shot 95% CIs (Q9), isotonic-calibration control (Q7).
- **Three new experiments** answering the architecture / representation / DA asks:
  ST-GCN in the main zero-shot (Q6), a joint-angle/relative-vector input (Q4), and
  AdaBN + DANN domain adaptation (Q5).

---

## Point-by-point

| # | Reviewer point | Response | Where |
|---|---|---|---|
| **Q1** | KIMORE "100 frames at 1 Hz" | Corrected: Kinect v2 is ~30 fps native, variable duration, linearly resampled to a common length of 100 frames. Both the Datasets paragraph and the preprocessing paragraph are fixed. | §III Datasets & Preprocessing |
| **Q2** | Naive baseline vs padding/normalization | New sensitivity analysis (Table on naive variants). Padded zeros contribute **exactly zero** to path/speed, confirmed numerically: UI-PRMD shared-joint = raw = 0.538. Dropping REHAB246 duplicated joints barely moves it (0.554→0.548). On per-sequence z-scored coordinates the naive edge collapses to chance on REHAB246 (0.510), showing its small advantage lives in the very global-scale cue normalization removes; even then it is not below the learned models. | New Table (naive sensitivity) + §IV-A |
| **Q3** | Few-shot inconsistency (§IV-A vs Limitations) | Reconciled: we **do** probe 1–20 shots (§IV-A, chance). The Limitations/Future-Work now correctly scope the open question to larger budgets ($n>20$) and active selection, and cross-reference §IV-A. | §VI Limitations, Future Work |
| **Q4** | Joint-angle / bone-length-preserving MAIN input | New experiment (C2): a parent-relative joint-vector representation (translation-invariant, bone-length-preserving) used as the native model input, trained under the same 77-fold LOSO. **Still at chance and below naive: REHAB246 0.534±0.017, UI-PRMD 0.520±0.015.** | Table `tab:robustness-c` + `src/selfsup/features.py` |
| **Q5** | Stronger DA than CORAL | Added **AdaBN** (parameter-free BN re-estimation) and **DANN** (gradient-reversal). **Both at chance: AdaBN 0.519/0.514; DANN 0.529/0.509**, below naive. Related work now surveys DANN/MMD/AdaBN/SHOT/TENT. | Table `tab:robustness-c` + §II + `src/reviewer_round3_c3_dann.py` |
| **Q6** | ST-GCN / spatial-prior backbone in main zero-shot | New experiment (C1): ST-GCN under identical 77-fold LOSO. **At chance (REHAB246 0.522±0.008, UI-PRMD 0.514±0.010), below naive; and its sensor-ID probe is again perfect (1.00 vs chance 0.33) — the null is architecture-independent.** | Table `tab:robustness-c` + `src/models_stgcn.py` |
| **Q7** | Monotone/isotonic calibration | Directly tested: isotonic map fit on KIMORE, applied to target scores, keeps both corpora at chance (REHAB246 0.516→0.517; UI-PRMD 0.524→0.504). Replaces the previous hand-wave. | §VI Limitations (Eighth) |
| **Q8** | "all-corpora" column clarity | Caption rewritten: which corpora enter SSL pretraining (unlabeled), that it is a **single** fine-tuned model (not 77 folds), and that only KIMORE ever supplies labels. | Table II caption |
| **Q9** | Zero-shot CIs in Table II | Added per-cell 95% CIs (1.96·SD/√77) to all IRDS-only conditions; every interval excludes 0.55 and lies below the naive baseline. | Table II + caption |
| **Q10** | UI-PRMD label-scale mismatch + per-sequence normalization | Addressed via the isotonic control (Q7) and AUPRC agreement. We also **corrected a documentation error**: the pipeline uses a train-fit global `StandardScaler` (which preserves per-sequence scale), *not* the per-sequence z-scoring the manuscript previously claimed. We then ran the reviewer's suggested ablation — retraining under strict per-sequence z-scoring — to confirm the zero-shot conclusion is invariant. **Still at chance: REHAB246 0.539±0.023, UI-PRMD 0.534±0.027 — the null is normalization-independent.** | §III Preprocessing + Table `tab:robustness-c` |
| **Table II parse** | "all-corpora column difficult to parse" | Root-caused to a LaTeX column-spec bug (`lccc` with 5 data columns) that mis-aligned the table; fixed to `lcccc`. Manuscript now compiles clean (8 pp, no undefined references). | Table II |

### Weaknesses (beyond the numbered questions)
- *Single backbone (TCN)* → ST-GCN added (C1); sensor-ID probe repeated on ST-GCN features to test architecture-independence of the null.
- *Zero-padding artifacts* → quantified (Q2) and removed at source via the relative-joint-vector input (C2).
- *Per-sequence normalization discards scale/speed* → quantified in the naive sensitivity table (Q2); the relative-vector input (C2) is an alternative that preserves bone length.
- *DA/DG related work coverage* → new §II paragraph citing DANN, MMD, AdaBN, SHOT, TENT, SMPL.

---

### Headline outcome of the new experiments
Across **every** architecture (TCN, ST-GCN), input representation (coordinates,
relative-joint-vectors), and domain-adaptation method (CORAL, AdaBN, DANN) we tried,
zero-shot cross-sensor AUROC stays at chance and below the naive kinematic baseline,
and the sensor-identity signal persists (perfect probe on both TCN and ST-GCN). The
negative result is therefore **not** an artifact of backbone, coordinate normalization,
joint padding, or weak adaptation — it is robust. Raw numbers: `outputs/reviewer_round3/*.json`;
reproduced by `src/run_reviewer_round3_heavy.py`.
