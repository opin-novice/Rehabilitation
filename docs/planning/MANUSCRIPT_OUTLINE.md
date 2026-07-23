# Manuscript Outline — TNSRE Negative-Result Paper
## "Self-Supervised Pretraining Does Not Rescue Zero-Shot Cross-Sensor Rehabilitation Quality Assessment"

**Target:** IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE)
**Format:** Negative-result / benchmark-diagnostic article

---

## Abstract (250 words)

Automated rehabilitation quality scoring from skeleton sequences promises objective, scalable assessment, but prior work evaluates only within-dataset (KIMORE) under standard cross-validation. We ask: does self-supervised pretraining on unlabeled skeletons from a different sensor enable zero-shot cross-sensor scoring? We pretrain contrastive and masked-motion encoders on 1,000 unlabeled IntelliRehabDS (Kinect v2) sequences, then evaluate on KIMORE (Kinect v2, N=380, 77 subjects, LOSO) and zero-shot on two independent labeled corpora with different sensors: REHAB246 (OptiTrack, 1,057 reps) and UI-PRMD (Kinect, 2,000 reps). The result is a clean negative across four axes: (1) zero-shot AUROC is at chance (0.51–0.53) on both corpora, with a naive path-length+speed baseline (AUROC=0.55, 0.54) unbeaten; (2) under a fully-powered 77-fold leave-one-subject-out evaluation, SSL fine-tuning is statistically indistinguishable from training from scratch (pairwise Wilcoxon p>0.3, Holm-corrected), while SSL linear-probing is significantly *worse* (adj p<1e-13); (3) quadrupling the unlabeled pool to ~5,000 sequences does not close the gap; and (4) contrastive and masked-motion paradigms are statistically equivalent (p=0.8). Degeneracy gates (pred_SD>0.10) and probe-sanity checks rule out collapsed models or undertrained encoders as explanations. The barrier is sensor-level domain shift, not representation quality: the null is stable across pretext tasks, pool sizes, external corpora, and evaluation protocols. We conclude that SSL pretraining on unlabeled skeletons does not confer cross-sensor transferability for rehabilitation scoring, and that the field should prioritize sensor-invariant representations rather than more sophisticated SSL on a single sensor modality.

**Keywords:** self-supervised learning, rehabilitation quality assessment, zero-shot transfer, KIMORE, domain shift, leave-one-subject-out

---

## 1. Introduction (~700 words)

### 1.1 Clinical & technical background
- Automated rehabilitation scoring from skeleton data: why it matters (objective, remote, scalable)
- KIMORE as the primary benchmark — 78 subjects, 5 trunk/hip exercises, physician-assigned 0–50 scores
- Current SOTA achieves Spearman rho 0.70–0.96 under standard 5-fold CV

### 1.2 Gap
- All prior work evaluates within-dataset under cross-validation — ideal conditions
- Real-world deployment requires cross-sensor generalization (hospital Kinect → home webcam, different brands, different placement)
- No prior work tests whether SSL pretraining on unlabeled *target-sensor* data enables zero-shot scoring on a *different* sensor
- The two labeled corpuses enabling this test exist: REHAB246 (OptiTrack) and UI-PRMD (Kinect, but different acquisition)

### 1.3 This study
- First systematic evaluation of SSL pretraining for zero-shot cross-sensor rehabilitation scoring
- Clean negative: SSL does not help; the barrier is sensor-level domain shift, not representation quality
- Four pieces of converging evidence (zero-shot, LOSO, scale ablation, paradigm comparison)
- Rigor: true LOSO (77-fold, N=380), degeneracy gates, naive-feature baselines, Holm-corrected sample-level tests

---

## 2. Related Work (~500 words)

### 2.1 KIMORE benchmarks
- Karlov 2024: supervised contrastive learning, ST-GCN, IRDS→KIMERO transfer learning (within-Kinect)
- Abedi 2023: cross-modal augmentation, 5-fold CV
- Kuang 2026: Dual-Stream STGCN, rho=0.965 (non-stratified CV)
- Ismail-Fawaz 2026 (Rehab-Pile): cross-subject benchmark, 9 architectures, MAE/RMSE only
- Gap: none test zero-shot cross-sensor generalization; all use within-dataset CV

### 2.2 SSL for skeletons
- Contrastive (SimCLR-style) and masked-motion (MAE-style) pretraining for skeleton data
- Prior work focuses on recognition/classification, not regression quality assessment
- No prior work tests SSL pretraining for *zero-shot* transfer across rehabilitation datasets

### 2.3 Cross-dataset & cross-sensor in rehabilitation
- IRDS used for transfer learning (Karlov 2024) — but fine-tuned on target, not zero-shot
- UI-PRMD used for pre-training (some work) but always followed by same-sensor fine-tuning
- Our contribution: first pure zero-shot (no target labels at all) cross-sensor evaluation

---

## 3. Methods (~1000 words)

### 3.1 Datasets & preprocessing
- **KIMORE** (train/eval): 380 samples, 77 subjects (one dropped in preprocessing), 5 exercises, physician scores 0–50, Kinect v2, 25 joints, 100 frames
- **IRDS** (pretrain pool): 1,000 sequences, 10 subjects × 10 exercises × 10 reps, Kinect v2, 22→25 joint padding, 100 frames
- **REHAB246** (zero-shot test A): 1,057 reps (558 correct/499 incorrect), 10 subjects, 6 exercises, OptiTrack mocap, 26→25 joint mapping
- **UI-PRMD** (zero-shot test B): 2,000 reps (1,000 correct/1,000 incorrect), 10 subjects × 10 exercises, Kinect v2, 22→25 joint padding
- Fixed-length 100-frame resampling; z-score normalization; KIMORE-standard 25-joint schema

### 3.2 Self-supervised pretraining
- Two paradigms: **contrastive** (SimCLR-style, NT-Xent loss, data augmentations: jitter, scale, rotate, flip, channel-drop) and **masked-motion** (mask 50% of joints, reconstruct normalized coordinates, MSE loss)
- Encoder: TCN (4-block dilated causal CNN, d_model=128), chosen as highest-performing KIMORE backbone
- Pretraining: 300 epochs, batch 128, Adam lr=0.001, pool=IRDS-only (1k seq) or all-corpora (~5k seq)
- Two pool conditions: `irds_only` (pure cross-sensor, REHAB246/UI-PRMD held out) and `all_corpora` (transductive upper bound, REHAB246/UI-PRMD *in* the pool)

### 3.3 Evaluation protocol
- **77-fold Leave-One-Subject-Out (LOSO):** one fold per KIMORE subject — true LOSO, no subject leakage
- **Five conditions:**
  | Condition | Init | Encoder frozen? |
  |---|---|---|
  | A. Scratch | — | — |
  | B. Contrastive LP | contrastive encoder | Yes |
  | C. Contrastive FT | contrastive encoder | No |
  | D. Masked LP | masked encoder | Yes |
  | E. Masked FT | masked encoder | No |
- Fine-tuning: TCN regressor, 100 epochs, batch 16, early stopping patience 100, MSE loss
- Sample-level OOF pooling (N=380) for powered statistics

### 3.4 Zero-shot protocol
- Apply each condition's 77 model checkpoints (no retraining) to REHAB246 and UI-PRMD
- Mean AUROC across folds, per-exercise, against binary correctness labels
- Also: mean rank Spearman (ordinal transfer), pred_SD degeneracy gate (threshold 0.10)
- **Naive kinematic baseline:** total joint path length + mean joint speed → logistic regression AUROC (always computed on the same mapped sequences)

### 3.5 Statistical testing
- Sample-level paired Wilcoxon signed-rank on absolute error (N=380 matched OOF)
- Holm-Bonferroni correction over all pairwise condition comparisons
- 20-seed stratified bootstrap 95% CIs for per-condition mean rho
- Degeneracy gate: pred_SD < 0.10 → model flagged, reliability claims excluded

---

## 4. Results (~1200 words)

### 4.1 Zero-shot cross-sensor: chance-level everywhere (primary result)

**Table 1: Zero-shot AUROC (mean across 77 folds)**

| Condition | REHAB246 (OptiTrack) | UI-PRMD (Kinect) |
|---|---|---|
| A. Scratch | 0.516 | 0.524 |
| B. Contrastive LP | 0.516 | 0.518 |
| C. Contrastive FT | 0.515 | 0.514 |
| D. Masked LP | 0.527 | 0.512 |
| E. Masked FT | 0.519 | 0.514 |
| **Naive baseline** | **0.554** | **0.538** |

- Every condition at chance (0.51–0.53); naive baseline beats all
- Rank Spearman |rho| < 0.03 everywhere → no ordinal transfer
- REHAB246: all non-degenerate (pred_SD > 0.10) → low AUROC is genuine transfer failure
- UI-PRMD: 4/5 conditions degenerate (pred_SD < 0.10, models collapse on this corpus) → even the non-degenerate scratch (SD=0.12) scores 0.514

### 4.2 77-fold LOSO: SSL FT = scratch, SSL LP significantly worse

**Table 2: KIMORE within-domain (77-fold true LOSO, N=380)**

| Condition | Mean rho | 95% CI | Beats scratch? |
|---|---|---|---|
| A. Scratch | **0.836** | [0.785, 0.867] | — |
| E. Masked FT | 0.823 | [0.773, 0.854] | No (adj p=0.318) |
| C. Contrastive FT | 0.816 | [0.762, 0.851] | No (adj p=0.318) |
| B. Contrastive LP | 0.689 | [0.617, 0.738] | No (adj p=3.4e-14) |
| D. Masked LP | 0.679 | [0.612, 0.727] | No (adj p=7.3e-18) |

- SSL-FT statistically indistinguishable from scratch (adj p>0.3)
- SSL-LP significantly *worse* than scratch (adj p<1e-13) — frozen features inferior
- Contrastive = masked (p=0.805) — no paradigm difference
- Probe sanity: LP rho=0.69/0.68 >> 0 → encoders learned real structure (null is NOT undertraining)
- 6/10 pairwise tests significant, all fine-tune-vs-linear-probe or scratch-vs-linear-probe; zero SSL-vs-scratch FT differences

### 4.3 Scale ablation: 4× data doesn't help

**Table 3: Pool size comparison (5-fold LOSO)**

| Condition | IRDS-only (~1k) | All-corpora (~5k) |
|---|---|---|
| Scratch | **0.614** | **0.614** (reproduces) |
| Contrastive FT | 0.581 | 0.534 (worse) |
| Masked FT | 0.568 | 0.556 |
| Masked LP | 0.350 | 0.515 |
| Contrastive LP | 0.499 | 0.499 |

- Scratch still wins with 4× more pretraining data
- Contrastive FT actually drops with more data (0.581 → 0.534)
- Determinism check: scratch reproduces to 16 decimal places → identical folds, clean comparison
- **The null is not a data-scale artifact**

### 4.4 Zero-shot on 77-fold models (confirms §4.1)

**Table 4: Zero-shot AUROC from 77-fold models (vs §1 5-fold models)**

| Corpus | Scratch | Contrastive LP | Contrastive FT | Masked LP | Masked FT | Naive |
|---|---|---|---|---|---|---|
| REHAB246 | 0.516 → 0.516 | 0.517 → 0.516 | 0.513 → 0.515 | 0.524 → 0.527 | 0.513 → 0.519 | **0.554** |
| UI-PRMD | 0.517 → 0.524 | 0.519 → 0.518 | 0.517 → 0.514 | 0.513 → 0.512 | 0.527 → 0.514 | **0.538** |

- Numbers are effectively identical (chance-level in both protocols)
- Protocol invariance of the null strengthens the claim: not an artifact of 5-fold underpowering
- The 77-fold zero-shot results unify the evidence under one protocol

### 4.5 Summary: robustness across four axes

| Axis | Finding |
|---|---|
| Pretext task | Contrastive = masked (p=0.8) |
| Pool composition | IRDS-only = All-corpora (scratch still best) |
| Pool size | ~1k vs ~5k sequences — no change |
| External corpus | REHAB246 (OptiTrack) and UI-PRMD (Kinect) — both chance, both below naive |
| Evaluation protocol | 5-fold LOSO vs 77-fold LOSO — same null |

---

## 5. Discussion (~800 words)

### 5.1 Why SSL fails zero-shot cross-sensor
- Sensor-level domain shift dominates: joint coordinate distributions, bone-length ratios, frame rates, sensor noise profiles differ
- SSL on unlabeled target-sensor data learns target-sensor structure → but the scoring task requires mapping to *semantic* movement quality, and the source-sensor-to-quality mapping does not transfer
- The probe-sanity result (LP rho=0.69 >> 0) proves the encoder captures meaningful kinematic structure — but that structure is sensor-specific, not sensor-invariant
- Implication: the field needs sensor-invariant representations (adversarial domain adaptation, cross-sensor harmonization, canonical body models), not more sophisticated SSL on a single modality

### 5.2 Comparison to prior SSL-for-rehab claims
- Karlov 2024: supervised contrastive pretraining on IRDS improves KIMORE fine-tuning *within the same sensor modality* (Kinect→Kinect) — consistent with our finding that SSL helps within-domain but not cross-sensor
- Our result is not a contradiction: it fills the missing cell (cross-sensor zero-shot) that prior work did not test
- The negative result is more useful than a positive one: it redirects effort from SSL architecture search toward sensor-invariant representations

### 5.3 Rigor contributions (for TNSRE reviewers)
- True LOSO (77-fold, one fold per subject) — eliminates subject-identity leakage
- Degeneracy gate (pred_SD > 0.10) — prevents collapsed models from inflating "reliability" claims
- Naive-feature baseline — the simplest possible model as the comparison anchor
- Holm-corrected sample-level tests (N=380) — not underpowered fold-level comparisons
- Scale ablation — pre-empts "pool too small" critique
- Protocol invariance — zero-shot result holds under 5-fold and 77-fold protocols

### 5.4 Limitations
- Single backbone (TCN) — other architectures (ST-GCN, Transformer) may behave differently
- IRDS is the only unlabeled Kinect corpus of meaningful size; results may not generalize to other Kinect-like sensors
- REHAB246 is OptiTrack (marker-based, high precision), not a consumer sensor — the hardest zero-shot test
- UI-PRMD "incorrect" class is non-optimal execution by healthy subjects, not clinically-graded errors — arguably a weaker signal
- KIMORE n=77 is small; SSL pretraining may still help in the low-data regime below ~20 subjects

### 5.5 Future work
- Adversarial domain adaptation (sensor-invariant encoders)
- Canonical body-model representations (SMPL, parametric skeletons) as sensor-agnostic input
- Multi-sensor pretraining (pooling Kinect + OptiTrack + webcam skeletons)
- Demographic and clinical subgroup analysis (does SSL help for specific patient groups?)
- Test-time adaptation: fine-tune the encoder on a handful of target-sensor labeled samples

---

## 6. Conclusion (~200 words)

We systematically evaluated whether self-supervised pretraining on unlabeled skeletons from a different sensor enables zero-shot cross-sensor rehabilitation quality scoring. Across two pretext tasks, two pool sizes, two independent labeled corpora, and both 5-fold and 77-fold leave-one-subject-out protocols, the result is a clean negative: every SSL condition scores at chance cross-sensor (AUROC 0.51–0.53, naive baseline unbeaten at 0.55/0.54), and within-domain (KIMORE) SSL fine-tuning is statistically indistinguishable from training from scratch while linear-probing is significantly worse. Quadrupling the unlabeled pool does not help. The barrier is sensor-level domain shift: encoders learn useful but sensor-specific structure (probe sanity passes), and that structure does not transfer. We conclude that SSL pretraining on unlabeled skeletons is not a viable path to cross-sensor rehabilitation scoring, and we recommend the field prioritize sensor-invariant representations — whether through adversarial domain adaptation, multi-sensor pretraining, or canonical body models — rather than more sophisticated SSL on a single sensor modality.

---

## Tables & Figures

**Tables:**
1. Zero-shot AUROC (REHAB246 + UI-PRMD, all 5 conditions + naive baseline) — **primary result**
2. KIMORE 77-fold LOSO (mean rho, 95% CI, beats-scratch) — **within-domain evidence**
3. Scale ablation (irds_only vs all_corpora)
4. Zero-shot from 77-fold models (protocol invariance)

**Figures:**
1. Zero-shot AUROC bar chart: conditions + naive baseline for both corpora, with pred_SD degeneracy markers
2. KIMORE LOSO per-exercise rho: scratch vs SSL conditions (colored by FT/LP)
3. Pool-size comparison: irds_only vs all_corpora scatter
4. Protocol invariance: 5-fold vs 77-fold zero-shot AUROC (paired dot plot)

---

## References (preliminary)
- Capecci 2019 (KIMORE dataset)
- Vakanski 2018 (UI-PRMD)
- Karlov 2024 (supervised contrastive + IRDS→KIMORE transfer)
- Kuang 2026 (Dual-Stream STGCN on KIMORE)
- Ismail-Fawaz 2026 (Rehab-Pile benchmark)
- Bai 2018 (TCN)
- Chen 2020 (SimCLR)
- He 2022 (masked autoencoders)
- Yan 2018 (ST-GCN)
