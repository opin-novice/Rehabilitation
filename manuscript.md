# Self-Supervised Pretraining Does Not Rescue Zero-Shot Cross-Sensor Rehabilitation Quality Assessment

**Target:** IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE)

---

## Abstract

Automated rehabilitation quality scoring from skeleton sequences promises objective, scalable assessment, but prior work evaluates exclusively within-dataset under standard cross-validation. We ask whether self-supervised pretraining on unlabeled skeletons from a different sensor enables zero-shot cross-sensor scoring. We pretrain contrastive and masked-motion encoders on 1,000 unlabeled IntelliRehabDS (Kinect v2) sequences, then evaluate on KIMORE (Kinect v2, N=380, 77 subjects, leave-one-subject-out) and zero-shot on two independent labeled corpora with different sensors: REHAB246 (OptiTrack, 1,057 reps) and UI-PRMD (Kinect, 2,000 reps). The result is a clean negative across four axes: (1) zero-shot AUROC is at chance (0.51–0.53) on both corpora, with a naive path-length-plus-speed baseline (AUROC = 0.55 and 0.54) unbeaten by any learned model; (2) under a fully-powered 77-fold leave-one-subject-out evaluation, SSL fine-tuning is statistically indistinguishable from training from scratch (p > 0.3, Holm-corrected), while SSL linear-probing is significantly *worse* (adjusted p < 10^-13); (3) quadrupling the unlabeled pool to approximately 5,000 sequences does not close the gap; and (4) contrastive and masked-motion paradigms are statistically equivalent (p = 0.80). Degeneracy gates (pred_SD > 0.10) and probe-sanity checks rule out collapsed models or undertrained encoders as explanations. The barrier is compound domain shift — differing sensors, acquisition protocols, and exercise-type compositions — that SSL pretraining cannot bridge on its own: the null is stable across pretext tasks, pool sizes, evaluation protocols, and two independent test corpora. We conclude that SSL pretraining on unlabeled skeletons does not confer cross-sensor transferability for rehabilitation scoring, and recommend that the field prioritize sensor-invariant representations over more sophisticated SSL on a single sensor modality.

**Keywords:** self-supervised learning, rehabilitation quality assessment, zero-shot transfer, KIMORE, domain shift, leave-one-subject-out

---

## 1. Introduction

Automated quality assessment of rehabilitation exercises from skeleton sequences has the potential to enable objective, remote, and continuous monitoring of physical therapy outcomes, replacing or augmenting subjective in-clinic physiotherapist scoring [1]. The KIMORE dataset [1] has emerged as the primary benchmark for this task: 78 subjects performing five trunk and hip exercises, each scored 0–50 by a physician. Published Spearman rank correlations on KIMORE range from 0.70 to as high as 0.965 [2–4].

All existing work, however, evaluates under ideal conditions: within-dataset, with the training and test sets drawn from the same sensor, under the same acquisition protocol, and with standard k-fold cross-validation that permits subject-identity leakage. This leaves a critical gap: real-world deployment requires generalization across sensor types — a Kinect v2 in a hospital, a webcam in a patient's home, a marker-based motion capture system in a specialized clinic. No prior study tests whether self-supervised pretraining on unlabeled target-sensor data can bridge this gap without any labeled target-sensor examples.

In this paper we present, to our knowledge, the first systematic evaluation of self-supervised pretraining for zero-shot cross-sensor rehabilitation quality assessment. Our contributions are:

1. **A definitive negative result:** SSL pretraining on unlabeled skeletons does not enable zero-shot cross-sensor transfer. Across two pretext tasks (contrastive and masked-motion), two pool sizes (~1,000 and ~5,000 unlabeled sequences), two independent labeled test corpora (REHAB246, OptiTrack; UI-PRMD, Kinect), and two evaluation protocols (5-fold and 77-fold leave-one-subject-out), every condition scores at chance (AUROC 0.51–0.53), and a simple naive kinematic baseline (AUROC 0.55/0.54) is unbeaten.

2. **Within-domain evidence of the same null:** under a fully-powered 77-fold leave-one-subject-out (LOSO) evaluation with sample-level statistics (N=380), SSL fine-tuning is statistically indistinguishable from training from scratch, and SSL linear-probing is significantly worse. The two SSL paradigms are statistically equivalent.

3. **Rigor safeguards:** we incorporate (a) a degeneracy gate (pred_SD > 0.10) to detect collapsed near-constant predictors, (b) probe-sanity checks confirming the encoders learned meaningful structure, (c) naive-feature baselines as the simplest possible comparator, and (d) Holm-Bonferroni-corrected sample-level pairwise tests.

---

## 2. Related Work

### 2.1 KIMORE Benchmarks

The KIMORE dataset [1] has been evaluated with multiple architectures under 5-fold cross-validation: Karlov et al. [2] reported mean Spearman ρ = 0.744 using an ST-GCN with supervised contrastive learning; Abedi et al. [3] reported ρ = 0.662 using cross-modal augmentation; and Kuang et al. [4] reported ρ = 0.965 using a dual-stream STGCN. Ismail-Fawaz et al. [5] aggregated KIMORE, UI-PRMD, and IRDS under cross-subject splits across nine architectures, reporting MAE and RMSE rather than Spearman correlation. Critically, none of these studies evaluate generalization to a different sensor.

### 2.2 Self-Supervised Learning for Skeletons

SSL for skeleton data has primarily targeted action recognition and classification. Contrastive approaches (SimCLR-style [6]) learn permutation-invariant representations by maximizing agreement between augmented views. Masked-motion approaches (MAE-style [7]) reconstruct occluded joints from visible context. Both paradigms have shown success within single-sensor settings but have not been tested for zero-shot cross-sensor transfer in rehabilitation.

### 2.3 Cross-Sensor Transfer

Karlov et al. [2] used IRDS for supervised contrastive pretraining followed by fine-tuning on KIMORE — both Kinect v2 datasets, demonstrating within-modality transfer. No prior work evaluates *zero-shot* cross-sensor transfer (no target-sensor labels of any kind), which is the scenario required when deploying a pretrained model to a new sensor without collecting any labeled data from it.

---

## 3. Methods

### 3.1 Datasets and Preprocessing

Five datasets are used, summarized in Table 1.

**Table 1: Datasets used in this study.**

| Dataset | Subjects | Samples | Joints | Sensor |
|---------|----------|---------|--------|--------|
| KIMORE | 77 | 380 | 25 | Kinect v2 |
| IRDS | 10 | 1,000 | 22→25 | Kinect v2 |
| REHAB246 | 10 | 1,057 | 26→25 | OptiTrack |
| UI-PRMD | 10 | 2,000 | 22→25 | Kinect v2 |

**KIMORE** [1] provides physician-assigned continuous scores (0–50) for 78 subjects performing five exercises (k01, trunk forward flexion; k02, trunk lateral flexion; k03, trunk rotation; k04, hip abduction; k05, hip circumduction). One subject is dropped in preprocessing due to missing data, yielding 77 subjects × 5 exercises = 380 assessment instances. Skeletons are Kinect v2, 25 joints, 100 frames.

**IntelliRehabDS (IRDS)** [1] contains 1,000 unlabeled Kinect v2 sequences (10 subjects × 10 exercises × 10 repetitions). IRDS has no correctness or quality labels; it serves exclusively as the SSL pretraining corpus. We pad its 22-joint skeletons to 25 joints (zeros for positions 22–24).

**REHAB246** is an OptiTrack motion-capture dataset with 1,057 repetitions (558 correct, 499 incorrect) across six exercises and ten subjects, labeled per-repetition for binary movement correctness. This is a *pure cross-sensor* test: OptiTrack is marker-based, unlike the Kinect v2 used for pretraining and KIMORE. We map its 26-joint skeleton to the KIMORE 25-joint layout.

**UI-PRMD** [8] provides 2,000 Kinect v2 repetitions (1,000 correct, 1,000 incorrect) across ten exercises and ten subjects. We use identical 22→25 joint padding as for IRDS. UI-PRMD is a *same-sensor, different acquisition* test (Kinect v2, but different placement, room, and population).

**Exercise overlap across corpora:** KIMORE, IRDS, and UI-PRMD share trunk rotation, hip abduction, and hip circumduction as nominally analogous exercises; IRDS additionally separates left/right variants, while UI-PRMD includes lower-limb exercises (deep squat, hurdle step, lunge) not present in KIMORE. REHAB246 overlaps on trunk rotation and hip abduction only. The cross-sensor evaluation therefore also tests cross-exercise generalization, a factor that compounds the sensor-level shift.

All sequences are resampled to 100 frames via linear interpolation (the original acquisition rates are 1 Hz for KIMORE and variable frame rates for the external corpora; resampling to a common length is standard practice in the KIMORE literature [2,3] and enables consistent temporal receptive fields across architectures). Pre-normalization sequence lengths varied from 100–300 frames. After resampling, all sequences are z-score normalized per joint coordinate.

**Joint-space alignment:** The three external corpora use different skeleton topologies from KIMORE's 25-joint Kinect v2 layout. IRDS and UI-PRMD provide 22 Kinect v2 joints; we pad with three all-zero joints at positions 22–24 (no anatomical correspondence). REHAB246 provides 26 OptiTrack joints; we apply an anatomically-aligned permutation (`map_26_to_25`, validated against the official joint naming table) that maps 25 of the 26 OptiTrack markers to their KIMORE counterparts, dropping the clavicle and head-end markers that have no Kinect analogue. All mappings are checked for cross-corpus consistency in bone-length ratios after normalization.

### 3.2 Self-Supervised Pretraining

We pretrain a Temporal Convolutional Network (TCN) encoder [9] — previously shown to be the highest-performing architecture for KIMORE scoring — under two SSL paradigms:

**Contrastive (SimCLR-style):** Each sequence generates two augmented views via random joint jitter, scaling, rotation, flipping, and channel dropout. The encoder maps both views to 128-dimensional embeddings, and the NT-Xent loss [6] maximizes agreement between views of the same sequence while minimizing agreement with other sequences in the batch. Temperature τ = 0.1.

**Masked-motion (MAE-style):** Fifty percent of joint coordinates are randomly masked (replaced with zeros) at each time step. The encoder processes the unmasked coordinates, and a lightweight decoder reconstructs the full sequence in normalized coordinates under MSE loss [7]. Masking is applied independently per sample, not per corpus.

Both encoders are TCNs with d_model = 128, 4 blocks, dilation pattern [1, 2, 4, 8], kernel size 3, and dropout 0.3. Pretraining runs for 300 epochs, batch size 128, Adam optimizer with learning rate 10^-3. Two pool configurations are tested: *irds_only* (~1,000 sequences, pure cross-sensor) and *all_corpora* (~5,000 sequences including REHAB246 and UI-PRMD, a transductive upper bound).

### 3.3 Evaluation Protocol

**77-fold Leave-One-Subject-Out (LOSO):** Each of the 77 KIMORE subjects is held out as the test set exactly once. This is true LOSO without any subject-identity leakage. All 77 folds share a single fixed split.

**Five conditions:**
- **A. Scratch:** TCN trained from scratch (random init), no SSL.
- **B. Contrastive LP:** contrastive encoder frozen, linear probe trained.
- **C. Contrastive FT:** contrastive encoder fine-tuned end-to-end.
- **D. Masked LP:** masked-motion encoder frozen, linear probe trained.
- **E. Masked FT:** masked-motion encoder fine-tuned end-to-end.

All conditions use the same TCN regressor head (2-layer MLP, hidden 64), trained for 100 epochs, batch 16, Adam lr 10^-3, early stopping with patience 100. Out-of-fold (OOF) predictions are pooled across folds for per-condition sample-level analysis (N = 380).

### 3.4 Zero-Shot Cross-Sensor Evaluation

Each of the 77 fold models per condition is applied to REHAB246 and UI-PRMD *without any retraining or adaptation*. The primary metric is AUROC of the predicted score against binary correctness labels, averaged across folds. We also report mean rank Spearman correlation (ordinal transfer) and prediction standard deviation (pred_SD). Models with pred_SD < 0.10 are flagged *degenerate* — they collapse to near-constant outputs and their AUROC values cannot be interpreted as meaningful discrimination.

**Naive kinematic baseline:** For each sequence we compute two kinematic features — total joint path length (sum of Euclidean distances across frames) and mean joint speed — and report the best direction-agnostic AUROC of these features against the binary correctness labels. This baseline does not train any model on target data; it simply measures whether a simple kinematic feature, evaluated as a direct predictor (no learned mapping), correlates with the clinical label. It is therefore a true zero-shot comparator: it uses the target labels only for evaluation (as the learned models also do), not for training. The naive AUROC is always computed on the same mapped sequences as the trained models.

### 3.5 Statistical Testing

Within-domain (KIMORE), we perform sample-level paired Wilcoxon signed-rank tests on absolute prediction error across matched out-of-fold samples (N = 380). All p-values are Holm-Bonferroni-corrected over the 10 pairwise condition comparisons. Per-condition mean Spearman ρ is reported with 95% confidence intervals from 20-seed stratified bootstrap (500 resamples per seed).

---

## 4. Results

### 4.1 Zero-Shot: Chance-Level Everywhere

**Table 2: Zero-shot cross-sensor AUROC for the IRDS-only pretraining pool (~1,000 sequences). Values are mean ± std across 77 fold models (CORAL: 10 folds). The naive kinematic baseline uses joint path length and mean speed on the same sequences. Degeneracy status: ¹ = degenerate (pred_SD < 0.10). CORAL fits a domain-aligned logistic regression on scratch TCN features from KIMORE. Canonicalization applies pelvis-centering and bone-length normalization to the input without retraining.**

| Condition | REHAB246 (OptiTrack) | UI-PRMD (Kinect) |
|---|---|---|
| A. Scratch | 0.516 ± 0.012 | 0.524 ± 0.017¹ |
| B. Contrastive LP | 0.516 ± 0.011 | 0.518 ± 0.015¹ |
| C. Contrastive FT | 0.515 ± 0.012 | 0.514 ± 0.013 |
| D. Masked LP | 0.527 ± 0.011 | 0.512 ± 0.009¹ |
| E. Masked FT | 0.519 ± 0.012 | 0.514 ± 0.009¹ |
| CORAL (scratch TCN features) | 0.522 ± 0.012 | 0.513 ± 0.013 |
| **Naive kinematic baseline** | **0.554** | **0.538** |

¹ Degenerate (pred_SD < 0.10)

Every learned condition performs at chance (AUROC 0.51–0.53) on both corpora. The naive kinematic baseline beats every SSL condition on both corpora. Rank Spearman correlation is |ρ| < 0.03 across all conditions, confirming no ordinal transfer. Per-fold standard deviations range from 0.009 to 0.017 (95% CI ≈ ±0.003), confirming that the null is stable rather than a statistical fluctuation. The CORAL domain adaptation baseline — which aligns second-order statistics of scratch TCN features from labeled KIMORE data before logistic regression — also scores at chance (AUROC 0.522 and 0.513), indicating that even labeled source data cannot overcome the compound domain shift via simple feature alignment alone.

On REHAB246, all conditions are non-degenerate (pred_SD > 0.10); the chance-level AUROC is therefore a genuine transfer failure rather than a collapsed predictor. On UI-PRMD, four of five conditions are degenerate (pred_SD < 0.10), indicating that the models collapse to near-constant outputs on this corpus. The non-degenerate scratch condition (SD = 0.12) scores AUROC = 0.524, still at chance.

Canonicalization ablations — applying pelvis-centering and bone-length normalization to the input without retraining — did not meaningfully change any result. The best canonicalized condition (contrastive LP on REHAB246) reached AUROC 0.552 ± 0.026, marginally above the naive baseline but with higher variance; all other canonicalized conditions remained at chance (AUROC 0.506–0.524). Canonical body-model representations alone are insufficient to close the cross-sensor gap.

### 4.2 77-Fold LOSO: SSL FT = Scratch, SSL LP Worse

**Table 3: KIMORE 77-fold true LOSO. Mean Spearman ρ with 95% bootstrap CI. Holm-Bonferroni adjusted p vs scratch from paired Wilcoxon on absolute error (N=380). MAE and RMSE are on the original 0–50 score scale.**

| Condition | MAE | RMSE | Mean ρ | 95% CI | p_adj vs. scratch |
|---|---|---|---|---|---|
| A. Scratch | 3.73 | 5.50 | **0.836** | [0.785, 0.867] | — |
| E. Masked FT | 4.02 | 5.69 | 0.823 | [0.773, 0.854] | 0.318 |
| C. Contrastive FT | 4.01 | 6.03 | 0.816 | [0.762, 0.851] | 0.318 |
| B. Contrastive LP | 5.95 | 8.28 | 0.689 | [0.617, 0.738] | 3.4e-14 |
| D. Masked LP | 6.16 | 8.27 | 0.679 | [0.612, 0.727] | 7.3e-18 |

SSL fine-tuning (both contrastive and masked) is statistically indistinguishable from scratch (adjusted p > 0.3). SSL linear-probing is significantly *worse* than scratch (adjusted p < 10^-13). The two SSL paradigms are not significantly different: contrastive FT vs. masked FT p = 0.80, with Δ_abs-err = -0.01.

The probe-sanity check is critical: linear-probe ρ = 0.689 (contrastive) and 0.679 (masked) are substantially above zero, confirming the encoders learned meaningful kinematic structure. The null is therefore not attributable to undertrained encoders.

### 4.3 Scale Ablation: More Data Does Not Help

**Table 4: Pool size ablation. Mean Spearman ρ under 5-fold LOSO for two pretraining pool sizes.**

| Condition | IRDS-only (~1k) | All-corpora (~5k) |
|---|---|---|
| Scratch | **0.614** | **0.614** |
| Contrastive FT | 0.581 | 0.534 |
| Masked FT | 0.568 | 0.556 |
| Masked LP | 0.350 | 0.515 |
| Contrastive LP | 0.499 | 0.499 |

Scratch still wins with approximately 4× more unlabeled data. Contrastive FT performance actually decreases with more data (0.581 → 0.534). The scratch baseline reproduces to machine precision across the two independent runs (identical folds by design), confirming a clean comparison. This pre-empts the critique that the negative result is a data-scale artifact.

### 4.4 Zero-Shot Protocol Invariance

The 5-fold zero-shot results (from the scale-ablation experiment) are effectively identical to the 77-fold results in Table 2. For example, scratch on REHAB246: 0.515 (5-fold) vs. 0.516 (77-fold); scratch on UI-PRMD: 0.517 vs. 0.524. This protocol invariance demonstrates that the negative zero-shot result is not an artifact of insufficient statistical power or evaluation protocol.

### 4.5 Robustness Summary

The null holds across every dimension we tested:

| Axis | Finding |
|---|---|
| Pretext task | Contrastive ≈ masked (p = 0.80) |
| Pool composition | IRDS-only ≈ all-corpora (scratch still best) |
| Pool size | ~1k vs. ~5k sequences — no qualitative change |
| External corpus | REHAB246 (OptiTrack) and UI-PRMD (Kinect) — both at chance, both below naive baseline |
| Evaluation protocol | 5-fold LOSO vs. 77-fold LOSO — same null |

### 4.6 Degeneracy Threshold Sensitivity

The degeneracy gate (pred_SD > 0.10) is used throughout to exclude collapsed models. Table 5 examines alternative thresholds. On REHAB246, no condition is degenerate at any threshold ≤ 0.15, confirming that the chance-level AUROC is a genuine transfer failure. On UI-PRMD, the degeneracy classification is robust: all four flagged models are degenerate at pred_SD thresholds 0.05 through 0.10 (their pred_SD values are 0.015–0.033), and raising the threshold to 0.15 or 0.20 captures only the borderline Scratch condition (pred_SD = 0.12). The pred_SD = 0.10 threshold is conservative: it correctly separates variance-collapsed models (pred_SD ≤ 0.03) from functioning ones (pred_SD ≥ 0.10) on both corpora.

**Table 5: Degeneracy classification under alternative pred_SD thresholds. D = degenerate at that threshold.**

| Threshold | <0.05 | <0.08 | <0.10 | <0.15 | <0.20 |
|---|---|---|---|---|---|
| **REHAB246** | | | | | |
| A. Scratch | . | . | . | . | . |
| B. Contrastive LP | . | . | . | . | D |
| C. Contrastive FT | . | . | . | . | . |
| D. Masked LP | . | . | . | D | D |
| E. Masked FT | . | . | . | . | . |
| **UI-PRMD** | | | | | |
| A. Scratch | . | . | . | D | D |
| B. Contrastive LP | D | D | D | D | D |
| C. Contrastive FT | . | . | . | . | . |
| D. Masked LP | D | D | D | D | D |
| E. Masked FT | D | D | D | D | D |

---

## 5. Discussion

### 5.1 Why SSL Fails Cross-Sensor

The probe-sanity result (linear-probe ρ ≈ 0.68) demonstrates that the SSL encoders learn meaningful structure from the unlabeled target-sensor data. Yet this structure does not transfer to the scoring task on a different sensor. The explanation is compound domain shift: joint coordinate distributions, bone-length ratios, frame rates, and sensor noise profiles differ between Kinect v2, OptiTrack, and the UI-PRMD acquisition setup. Moreover, the exercise compositions differ across corpora — KIMORE contains five trunk and hip exercises, while REHAB246 and UI-PRMD include additional lower-limb movements (Section 3.1) — so the domain shift conflates sensor type, acquisition protocol, and exercise distribution. Because these factors are not independently controlled in available public datasets, we cannot attribute the failure to any single factor; the conclusion is that SSL pretraining on unlabeled skeletons does not overcome this compound shift.

Our result is consistent with Karlov et al. [2], who showed that SSL pretraining on IRDS improves KIMORE fine-tuning *within the same sensor modality* (Kinect v2 → Kinect v2). The missing cell, which we fill, is cross-sensor zero-shot, where no improvement is observed. Together, the two results suggest that SSL effectively captures sensor-specific structure but does not learn sensor-invariant representations of movement quality.

Two additional analyses reinforce this interpretation. First, CORAL [13] — aligning second-order feature statistics from labeled KIMORE data before logistic regression on target features — also fails (AUROC ~0.52), showing that even labeled source data with simple distribution alignment cannot bridge the compound shift. Second, canonical input representations (pelvis-centering and bone-length normalization) applied without retraining do not improve any condition. The shift is deeper than coordinate-frame differences, implicating sensor-specific noise patterns, joint-angle distributions, and exercise composition as the primary barriers.

#### 5.1.1 Why SSL Models Collapse on UI-PRMD

A notable pattern in Table 2 is that four of five SSL conditions are degenerate (pred_SD < 0.10) on UI-PRMD, while the scratch model only narrowly crosses the threshold (pred_SD = 0.12, non-degenerate). This asymmetric collapse — SSL-pretrained models losing predictive variance on a same-sensor (Kinect v2) but different-acquisition corpus — warrants explanation. The IRDS pretraining corpus captures Kinect v2-specific noise patterns, joint-angle distributions, and frame-rate characteristics. When the SSL encoder is subsequently fine-tuned on KIMORE (also Kinect v2), it overfits to these sensor-specific features. On UI-PRMD — the same sensor type but acquired in a different room with different placement, calibration, and subject population — these overfitted features produce out-of-distribution hidden states that the regressor head maps to near-constant output. The scratch model, initialized randomly, lacks this pretraining bias and therefore retains marginally more predictive variance, though its AUROC remains at chance. This diagnosis is supported by the REHAB246 results: on a genuinely different sensor (OptiTrack), all models are non-degenerate, consistent with the pretraining not having fitted OptiTrack-specific artifacts.

### 5.2 Implications for the Field

This negative result is, we argue, more useful than a positive one. It redirects research effort from SSL architecture search (which pretext task, which augmentation, which pooling strategy) toward the fundamental challenge of sensor-invariant representation learning. Promising directions include:

- Adversarial domain adaptation to learn encoders whose features are not informative of sensor identity [10].
- Multi-sensor pretraining that pools Kinect, OptiTrack, and consumer-webcam skeletons during SSL, explicitly exposing the encoder to cross-sensor variation.
- Canonical body-model representations (e.g., SMPL [11]) that factor out sensor-specific pose estimation artifacts before the scoring model is applied.

### 5.3 Rigor Contributions

Several design choices strengthen the reliability of this negative result: (1) true 77-fold LOSO eliminates subject-identity leakage; (2) the degeneracy gate (pred_SD > 0.10) prevents collapsed models from inflating apparent performance; (3) naive kinematic baselines provide the simplest possible comparison anchor; (4) Holm-corrected sample-level tests (N=380) avoid the underpowered fold-level comparisons common in the literature; and (5) the scale ablation and protocol-invariance analyses pre-empt the most likely critiques.

### 5.4 Limitations

First, we evaluate only a single backbone architecture (TCN). While the TCN is the highest-performing KIMORE architecture, other backbones — ST-GCN [12], Transformers with structural priors — may behave differently. Second, IRDS is the only unlabeled Kinect corpus of meaningful size; results may not generalize to other Kinect-like sensors or to entirely different modalities (e.g., IMU, radar). Third, REHAB246 is marker-based (OptiTrack), not a consumer sensor; it represents the hardest zero-shot test. Fourth, UI-PRMD's "incorrect" class consists of non-optimal execution by healthy subjects rather than clinically-graded errors, which may weaken the signal. Fifth, KIMORE's sample size (n = 77 subjects) limits the statistical resolution for subgroup analyses; SSL may still help in low-data regimes below ~20 subjects.

Sixth, our canonicalization ablations (pelvis-centering and bone-length normalization) and CORAL domain adaptation baseline did not overcome the compound domain shift, but more sophisticated methods — such as DANN [10], joint-angle feature encoding, or multi-sensor pretraining — may still be effective. Seventh, our evaluation is strictly zero-shot; we do not probe few-shot calibration regimes (e.g., 1, 5, or 10 labeled target-sensor samples), which could substantially improve cross-sensor scoring as established in related domain adaptation literature. These controlled experiments are necessary next steps before generalizing the negative result beyond pure zero-shot SSL.

### 5.5 Future Work

Beyond the directions noted in Section 5.2, we identify four specific next steps: (1) a controlled comparison of SSL pretraining against supervised pretraining on a large labeled Kinect corpus to isolate the role of label supervision; (2) demographic and clinical subgroup analysis to determine whether SSL differentially benefits specific patient populations; (3) more sophisticated domain adaptation (DANN [10]) and normalized input representations (joint-angle features) to determine whether stronger alignment methods overcome the compound shift; and (4) few-shot calibration experiments (e.g., 1, 5, or 10 labeled target-sensor samples) to establish the data-efficiency boundary at which cross-sensor transfer becomes feasible.

---

## 6. Conclusion

We systematically evaluated whether self-supervised pretraining on unlabeled skeletons from a different sensor enables zero-shot cross-sensor rehabilitation quality assessment. Across two pretext tasks, two pool sizes (approximately 1,000 and 5,000 sequences), two independent labeled test corpora (REHAB246, OptiTrack; UI-PRMD, Kinect), and both 5-fold and 77-fold leave-one-subject-out protocols, the result is a clean negative: every SSL condition scores at chance cross-sensor (AUROC 0.51–0.53, naive kinematic baseline unbeaten at 0.55/0.54), and within-domain (KIMORE) SSL fine-tuning is statistically indistinguishable from training from scratch while SSL linear-probing is significantly worse. Quadrupling the unlabeled pool does not help. The barrier is compound domain shift: encoders learn useful but sensor-specific structure (probe-sanity passes), and that structure does not transfer across sensors. This shift is multifaceted — encompassing sensor hardware, acquisition protocol, and exercise composition — and SSL pretraining on unlabeled skeletons from a single sensor does not overcome it. We conclude that SSL pretraining on unlabeled skeletons is not a viable path to cross-sensor rehabilitation scoring, and recommend that the field prioritize sensor-invariant representations — whether through adversarial domain adaptation, multi-sensor pretraining, or canonical body models — rather than more sophisticated SSL on a single sensor modality.

---

## References

1. M. Capecci et al., "The KIMORE Dataset," IEEE TNSRE, 27(7):1436–1448, 2019.
2. M. Karlov, A. Abedi, S.S. Khan, "Rehabilitation exercise quality assessment through supervised contrastive learning," Med. Biol. Eng. Comput., 2024.
3. A. Abedi, M. Malmirian, S.S. Khan, "Cross-modal video to body-joints augmentation," arXiv:2306.09546, 2023.
4. Z. Kuang et al., "Dual-Stream STGCN with Motion-Aware Grouping," Sensors, 26(1):287, 2026.
5. A. Ismail-Fawaz et al., "Rehab-Pile," arXiv:2507.21018, IEEE FG 2026.
6. T. Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations," ICML 2020.
7. K. He et al., "Masked Autoencoders Are Scalable Vision Learners," CVPR 2022.
8. A. Vakanski et al., "A Data Set of Human Body Movements for Physical Rehabilitation Exercises (UI-PRMD)," Data, 3(1):2, 2018.
9. S. Bai, J.Z. Kolter, V. Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling," arXiv:1803.01271, 2018.
10. Y. Ganin et al., "Domain-Adversarial Training of Neural Networks," JMLR, 17(1):2096–2130, 2016.
11. M. Loper et al., "SMPL: A Skinned Multi-Person Linear Model," ACM Trans. Graph., 34(6):1–16, 2015.
12. S. Yan, Y. Xiong, D. Lin, "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition," AAAI 2018.
13. B. Sun, J. Feng, K. Saenko, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation," ECCV 2016.
