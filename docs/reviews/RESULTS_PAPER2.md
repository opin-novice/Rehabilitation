# Paper-2 Results — SSL Pretraining vs. Zero-Shot Cross-Sensor Transfer

**Primary run:** `python src/selfsup/run_all.py --pool irds_only --n_folds 5` · **Date:** 2026-07-03 · GPU (CUDA)
**Scope:** REAL KIMORE (380 samples, 77 subjects, 5-fold Stratified-LOSO); SSL pretraining on the
IRDS pool (1000 unlabeled Kinect sequences, 300 epochs); zero-shot on REAL REHAB24-6 (1057,
OptiTrack, **pure cross-sensor** — held out of the pool) and UI-PRMD (2000, Kinect). Fine-tune 100
epochs. This is the plan's intended IRDS→KIMORE clean design.

Three runs corroborate each other and all land in the same cell:
1. **Clean `irds_only`** (primary, below) — IRDS-only pretraining, the pure zero-shot design.
2. **`all_corpora` scale ablation** (§1b) — IRDS+REHAB24-6+UI-PRMD pool (~4× data), pre-empts "pool too small."
3. **UI-PRMD pilot** (§5, earlier, bounded budget) — first real-data run; direction identical.

## 1. KIMORE fine-tuning — clean `irds_only` (Table 1)
| Condition | Mean rho | 95% CI | Beats scratch |
|---|---|---|---|
| scratch | **0.614** | [0.533, 0.694] | — |
| contrastive_ft | 0.581 | [0.457, 0.681] | No |
| masked_ft | 0.568 | [0.438, 0.655] | No |
| contrastive_lp | 0.499 | [0.390, 0.604] | No |
| masked_lp | 0.350 | [0.246, 0.444] | No |

- **No SSL condition beats the scratch baseline** (`beats_scratch` all False). Training from scratch
  is the best condition; fine-tuning roughly recovers it, linear-probing (frozen features) is worse.
- Pairwise Wilcoxon: nothing significant after Holm-Bonferroni (all p_adj ≥ 0.625; n=5 folds →
  underpowered, CIs overlap heavily). Treat condition differences as descriptive.
- Primary contrast contrastive_ft vs masked_ft: p=0.625, median Δrho=0.019 → **no paradigm difference.**
- **Probe sanity PASSES** (frozen-encoder rho 0.499 / 0.350 > 0): the encoders learned real structure,
  so the null is *not* an artifact of a dead encoder.

## 1b. Scale ablation — `all_corpora` pool (Table 1b)
Same 5 conditions, re-run with encoders pretrained on IRDS+REHAB24-6+UI-PRMD (~4× the unlabeled
data; KIMORE excluded as LOSO leakage guard). Results in `outputs/ssl_results_allcorpora/`.

| Condition | rho (`irds_only`) | rho (`all_corpora`, ~4× data) |
|---|---|---|
| scratch | **0.614** | **0.614** |
| contrastive_ft | 0.581 | 0.534 |
| masked_ft | 0.568 | 0.556 |
| masked_lp | 0.350 | 0.515 |
| contrastive_lp | 0.499 | 0.499 |

- **Scratch still wins with ~4× more pretraining data**; no SSL condition beats it under either pool.
  contrastive fine-tuning actually *drops* with more data (0.581 → 0.534). The ceiling is pinned at
  the scratch baseline regardless of pool size → **the null is not a data-scale artifact.**
- Determinism check: scratch reproduces rho to 16 decimal places (0.6135619795904689) across both
  runs → identical folds, so the `irds_only` vs `all_corpora` comparison is exactly apples-to-apples.

## 2. Zero-shot cross-sensor — clean `irds_only` (Table 2 — AUROC)
| Corpus | scratch | contrastive_lp | contrastive_ft | masked_lp | masked_ft | **naive** |
|---|---|---|---|---|---|---|
| REHAB24-6 (OptiTrack, pure) | 0.515 | 0.517 | 0.513 | 0.524 | 0.513 | **0.554** |
| UI-PRMD (Kinect) | 0.517 | 0.519 | 0.517 | 0.513 | 0.527 | **0.538** |

- All conditions sit at **chance (0.51–0.53)**; the **naive path-length+speed baseline beats every
  SSL model** on both corpora — replicating the Paper-1 finding.
- Rank-transfer Spearman (Table 3) ≈ 0 (|rho| ≤ 0.05) everywhere → no ordinal transfer.
- IRDS zero-shot AUROC is **N/A by design**: IRDS carries no correctness labels and is the
  pretraining corpus, so it is not a labeled zero-shot test.
- Degeneracy gate: REHAB24-6 predictions are non-degenerate for scratch/contrastive_ft/masked_ft
  (pred_SD 0.13–0.68), so low AUROC is a genuine transfer failure, not collapse.

## 3. Interpretation grid (RESEARCH_PLAN_2 §8.2)
| KIMORE rho improved? | Zero-shot improved? | → Finding |
|---|---|---|
| **No** | **No** | **The barrier is fundamental domain shift, not representation quality.** |

This is the plan's **definitive-negative** cell, and the *strong* version of Paper-1: contrastive AND
masked-motion SSL, pretrained on real unlabeled skeletons, fail to rescue zero-shot cross-sensor
scoring — the naive kinematic baseline remains unbeaten, while the probe sanity check rules out
under-training as the cause. Both SSL paradigms are statistically indistinguishable.

## 4. Robustness — the null holds across four axes
The negative result is not a single-run artifact. It reproduces across:
- **Pretext task:** contrastive *and* masked-motion (§1, §5).
- **Pool composition:** IRDS-only vs IRDS+REHAB24-6+UI-PRMD (§1 vs §1b), and the earlier UI-PRMD pool (§5).
- **Pool size:** ~1k vs ~5k unlabeled sequences (§1b) — no change.
- **External corpus:** REHAB24-6 (pure cross-sensor) and UI-PRMD, both at chance, both below naive.

## 5. Corroborating UI-PRMD pilot (earlier, superseded)
An earlier bounded-budget pilot (`src/selfsup/run_real_pilot.py`; UI-PRMD pretraining pool, 60/80
epochs) reproduced the same definitive-negative direction — no SSL condition beat scratch, and the
naive baseline was unbeaten zero-shot. Its point estimates are pilot-grade and are fully superseded
by the clean `irds_only` results in §1–§2; the stale pilot numbers are not reported here to avoid
confusion.

## 6. Threats / honesty
- **Statistical power:** the 5-fold pairwise tests (§1) are underpowered (n=5 folds). The full
  77-fold sample-level analysis (§7, N=380) resolves this and confirms the direction — SSL
  fine-tuning is statistically indistinguishable from scratch, and linear-probing is significantly
  worse.
- **scratch > SSL on KIMORE**: does not support "SSL helps"; mildly suggests SSL init is not
  beneficial at this scale, and the scale ablation (§1b) shows more data does not close the gap.
- **`all_corpora` zero-shot is transductive**: for that pool, REHAB24-6 / UI-PRMD are *in* the
  pretraining set, so any zero-shot number from it is an optimistic upper bound. The pure
  cross-sensor zero-shot story stays with the `irds_only` results (§2).

## 7. Full 77-fold true leave-one-subject-out (sample-level, TNSRE-grade)
True leave-one-subject-out (`LeaveOneGroupOut`, one fold per subject → 77 folds; KIMORE_pooled has
77 subjects, the plan's "78" counting one dropped in preprocessing). Out-of-fold predictions are
pooled to N=380 and analysed with the Paper-1 sample-level protocol: per-exercise Spearman rho,
20-seed stratified bootstrap 95% CI, and pairwise Wilcoxon on absolute error with Holm-Bonferroni
FWER correction over 10 pairs. Encoders: `irds_only`. Full artifacts in `results/kimore_loso_78fold/`.

| Condition | Mean rho (per-exercise, pooled) | 95% CI | Beats scratch |
|---|---|---|---|
| scratch | **0.836** | [0.785, 0.867] | — |
| masked_ft | 0.823 | [0.773, 0.854] | No |
| contrastive_ft | 0.816 | [0.762, 0.851] | No |
| contrastive_lp | 0.689 | [0.617, 0.738] | No |
| masked_lp | 0.679 | [0.612, 0.727] | No |

- **Scratch is still best; no SSL condition beats it** — now at full statistical power (N=380).
- **SSL fine-tuning is statistically indistinguishable from scratch:** scratch vs contrastive_ft
  Wilcoxon p=0.079 (Holm-adj 0.318); scratch vs masked_ft p=0.105 (adj 0.318). SSL init neither
  helps nor hurts the fine-tuned model — the powered version of the null the 5-fold run could only hint at.
- **SSL linear-probing is significantly *worse* than scratch:** scratch vs contrastive_lp adj
  p=3.4e-14; scratch vs masked_lp adj p=7.3e-18. Frozen SSL features are inferior; full fine-tuning
  only recovers to parity. (6 of 10 pairwise tests are significant — every one a fine-tune-vs-linear-probe
  or scratch-vs-linear-probe gap, never an SSL gain.)
- **No paradigm difference:** contrastive_ft vs masked_ft Wilcoxon p=0.805.
- **Metric note:** these mean-rho values use the per-exercise pooled Spearman (the KIMORE-literature
  metric, comparable to Karlov 2024 = 0.744), NOT the fold-level mean of §1 — the two estimators are
  not directly comparable, so do not read a "0.61→0.84 gain" into the change of protocol. Under this
  metric the scratch TCN matches/exceeds published SOTA under strict LOSO while SSL adds nothing.

## 8. Remaining work (optional, non-blocking)
- Optional: source a genuinely physician-scored external corpus (vet FineRehab) for a continuous
  external validity test.

## 9. Artifacts
- Clean `irds_only`: `outputs/ssl_results/` — stats.json, tables/table{1..4}, figures/fig{1,2}.
- Scale ablation: `outputs/ssl_results_allcorpora/` — per-condition loso_results.json + summary.json.
- Full 77-fold true LOSO: `results/kimore_loso_78fold/` — per-condition `fold_*/` (oof + metrics.json
  resume markers), loso_results.json, `stats78.json`, `table78.md`.
- Encoders: `outputs/ssl_pretrain/irds_only/` and `outputs/ssl_pretrain/all_corpora/`
  (contrastive/masked_encoder.pt + provenance).
- IRDS cache: `outputs/irds_eval/irds_sequences.npy` (1000,100,25,3) + `irds_manifest.csv`
  (built via `python src/irds_eval.py --build`).
- Folds: `outputs/folds.json`.
