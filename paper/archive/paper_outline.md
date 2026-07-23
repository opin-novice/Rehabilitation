# Paper Outline — Draft for Q1 Submission
# "Benchmarking Deep Learning Architectures for Automated Rehabilitation Quality
#  Scoring Under Clinically Valid Leave-One-Subject-Out Evaluation"

Target journals (priority order):
1. Computers in Biology and Medicine (IF ~7.7, Q1) — most realistic, 3-4 month review
2. Biomedical Signal Processing and Control (IF ~8.0, Q1) — signal-processing framing
3. IEEE Journal of Biomedical and Health Informatics (IF ~7.7, Q1) — requires clinical co-author

---

## Abstract (target: 250 words)

Automated rehabilitation quality scoring — assigning continuous clinical scores to Kinect
skeleton sequences — has been reported with Spearman rho up to 0.80 on the KIMORE benchmark.
However, all prior work uses standard five-fold cross-validation without documented
stratification, which risks subject-identity leakage. This paper is primarily an
evaluation-methodology and reliability-diagnostic contribution: we quantify how much reported
KIMORE performance is an artifact of evaluation protocol, and we introduce a label-free
zero-shot reliability-plus-degeneracy diagnostic for cross-dataset deployment screening. We
present the first Stratified Leave-One-Subject-Out (LOSO), sample-level benchmark on KIMORE
(78 subjects, 5 exercises) with zero-shot external reliability validation on IntelliRehabDS
(10 subjects, 10 exercises).

We train and evaluate seven architectures: BiLSTM, ST-GCN, GraphTransformer (with and
without bone-distance attention bias), Temporal Convolutional Network (TCN), Spatial-Channel
Transformer (SCT), and a multitask dual-stage transformer (Exp E). Statistical comparisons
use N=380 matched out-of-fold samples rather than five fold-level aggregates.

TCN achieves the numerically highest KIMORE mean rho (0.549; pooled OOF, N~76 per
exercise), though pairwise differences among architectures are not significant after
Holm-Bonferroni correction over 28 tests; only TCN and LSTM significantly beat the Ridge
handcrafted-feature baseline (rho=0.450), and no deep-learning architecture significantly
outperforms another on KIMORE. On IntelliRehabDS, evaluated
zero-shot as a label-free reliability study (the filename `m` field denotes exercise type,
not patient group, verified by bone-length analysis), all models show excellent test-retest
reliability (ICC > 0.90) — though reliability is necessary-not-sufficient for validity, and we
gate all reliability claims behind a pred_SD > 0.10 non-degeneracy screen (one model reaches
ICC=0.95 only by collapsing to a near-constant output). On an external labeled set (REHAB24-6;
1,057 repetitions with per-repetition correct/incorrect ground truth) no model discriminates
movement correctness above chance zero-shot (best AUROC 0.58 vs 0.71 for naive kinematics on the
same sequences), and this null replicates on a second, fully independent labeled corpus (UI-PRMD,
2,000 correct/incorrect repetitions; best AUROC 0.53), empirically confirming that zero-shot
reliability is necessary-but-not-sufficient for validity. Cross-exercise rank consistency diverges
and does not track KIMORE accuracy: the BiLSTM (KIMORE rho=0.521; pairwise differences not significant
after Holm-Bonferroni) has the lowest IRDS Kendall W (0.047), while the bone-distance-free
GraphTransformer, mid-pack on KIMORE accuracy, has the highest (0.608). Across the N=7 models
this KIMORE-rank vs IRDS-consistency relationship is negative but statistically not significant
(Spearman r=-0.39, p=0.38; exploratory). The lead quantitative contribution is an empirical
protocol-inflation analysis: a 20-seed 2x2 (subject-leakage x clinical-stratification)
decomposition shows that subject-identity leakage inflates measured Spearman rho by +0.026
(95% CI [0.0001, 0.053]), whereas clinical-group stratification has a negligible effect
(~0.00). All models significantly beat mean-prediction baselines (p < 1e-8).

Keywords: rehabilitation quality assessment, KIMORE, skeleton-based action recognition,
temporal convolutional network, zero-shot reliability, leave-one-subject-out evaluation

---

## 1. Introduction (target: ~800 words)

1.1 Clinical motivation
- Manual physiotherapist scoring is subjective, resource-intensive, session-limited
- Kinect skeleton-based automated scoring enables objective, remote, continuous monitoring
- KIMORE dataset as the primary benchmark (78 subjects, 5 trunk/hip exercises, 0-50 clinical score)

1.2 Gap in existing work
- All published work uses 5-fold CV without stratification (Karlov 2024, Abedi 2023, Guo 2021)
- No prior work tests out-of-distribution generalization (KIMORE -> IRDS)
- N=5 fold-level comparisons lack statistical power; sample-level tests not performed
- [INSERT 2-paragraph block from clinical_narrative.md: Introduction/Motivation section]

1.3 Contributions
This work is positioned as an evaluation-methodology and reliability-diagnostic study, not a
claim of architectural superiority. Contributions, in order of centrality:
(1) **Protocol-inflation decomposition (lead):** a 20-seed 2x2 (subject-leakage x clinical-stratification) decomposition isolating subject-identity leakage as a +0.026 rho inflation (95% CI [0.0001, 0.053]; clinical-stratification effect ~0), plus a controlled preprocessing-vs-protocol analysis showing the score-range and joint-count differences behind apparent SOTA are NOT the cause (score range rank-irrelevant; 25->18 joints moves rho by +0.005).
(2) **Zero-shot reliability + degeneracy diagnostic:** the first label-free zero-shot KIMORE->IRDS reliability evaluation (ICC, Kendall W, cross-exercise rho), gated by a pred_SD>0.10 non-degeneracy screen, with reliability treated as necessary-but-not-sufficient for validity; includes empirical verification of the IRDS filename schema, and is validated against external movement-quality labels on two independent corpora (REHAB24-6 and UI-PRMD), where zero-shot reliability is shown empirically necessary-but-not-sufficient for validity (best AUROC 0.58 and 0.53 respectively, vs 0.71/0.65 for naive kinematics on the same sequences).
(3) The first Stratified-LOSO, sample-level (N=380) KIMORE benchmark; a Ridge handcrafted-feature baseline under identical LOSO shows the DL advantage is limited to top models (only TCN, LSTM beat Ridge after FWER correction).
(4) The KIMORE-accuracy vs IRDS-consistency dissociation framed as a power-analysed CASE OBSERVATION (Spearman r=-0.39, p=0.38, N=7 models; underpowered) rather than a population claim.
(5) Per-exercise clinical interpretation of sensor-geometry effects (pending clinical co-author review).

---

## 2. Related Work (target: ~600 words)

2.1 KIMORE benchmark results (cite Karlov 2024, Abedi 2023, Guo 2021, Capecci 2019)
- Table: published Spearman rho per exercise (Table 1 seed)
- Note: no prior work reports R² on KIMORE; all use Spearman; all use 5-fold CV

2.2 Skeleton-based action recognition architectures
- ST-GCN (Yan 2018), graph-topology priors, temporal convolutions
- Transformer attention for skeleton sequences (limitations: attention dilutes local temporal patterns)
- TCN for time-series (Bai 2018): advantages for periodic motion

2.3 Cross-dataset evaluation in rehabilitation
- IRDS (Capecci 2019 original; Karlov 2024 uses IRDS for transfer learning pretraining)
- No prior work evaluates zero-shot KIMORE->IRDS without any IRDS training

2.4 Cross-Subject Benchmarking Protocols
- Closest concurrent work is Rehab-Pile (Ismail-Fawaz et al., arXiv:2507.21018, IEEE FG 2026),
  which aggregates KIMORE + UI-PRMD + IRDS across 9 architectures under cross-subject splits.
  It scales benchmark *breadth* (more datasets, more models); our contribution is orthogonal
  *depth*: sample-level statistics (N=380), zero-shot KIMORE->IRDS reliability, and a 2x2
  protocol-inflation decomposition. The two are complementary.

| Study | Eval Protocol | N_models | Sample-level stats | Zero-shot reliability | Protocol-inflation quantification |
|-------|---------------|----------|--------------------|-----------------------|-----------------------------------|
| Rehab-Pile (Ismail-Fawaz 2026) | Cross-subject | 9 | No | No | No |
| Ours | Stratified LOSO | 7 | Yes (N=380) | Yes (KIMORE->IRDS ICC/Kendall W) | Yes (2x2 decomposition) |

- Framing: they scale breadth, we rigorize depth.

---

## 3. Materials and Methods (target: ~1200 words)

3.1 Datasets
- KIMORE: 78 subjects, 5 exercises, 0-50 clinical scores, Kinect v2, 25 joints
  - 5 clinical groups: Expert (n=17), NotExpert (n=27), BackPain (n=8), Parkinson (n=16), Stroke (n=10)
  - Total: 380 assessment instances (5 exercises × ~76 subjects per exercise)
- IntelliRehabDS (IRDS): 10 subjects x 10 exercises x 10 repetitions = 1000 Kinect sequences (22 joints).
  Filename `m{EE}_s{SS}_e{RR}`: EE=exercise type, SS=subject, RR=repetition, verified empirically
  by bone-length analysis (see src/verify_irds_labels.py; ratio 0.43). No healthy/patient labels
  exist; IRDS is used only for zero-shot reliability validation.
  - Used ONLY for zero-shot external validation; no training on IRDS

3.2 Evaluation Protocol
- Stratified 5-fold LOSO: each fold's val set contains subjects from all 5 clinical groups
- Contrast: subject-identity leakage (non-grouped CV) inflates Spearman rho by +0.026 (95% CI [0.0001, 0.053]); clinical-group stratification effect is ~0 (see 4.6 decomposition)
- Out-of-fold (OOF) predictions: N=380 matched predictions collected across all folds
- Per-exercise Spearman rho: pool all OOF predictions per exercise, compute rank correlation
  (matches published KIMORE papers' methodology; N~76-82 subjects per exercise)
- Statistical tests at N=380: paired Wilcoxon signed-rank + t-test on abs_error
  (N=5 fold-level Wilcoxon provides minimum detectable p=0.0625; N=380 is well-powered)
- Mean-prediction baseline: for each fold, predict the training mean as val predictions

3.3 Architectures (Table 2)
(a) BiLSTM baseline: 2-layer bidirectional LSTM, hidden=128 per direction, 25J×3C flattened input
(b) ST-GCN: spatial-temporal graph conv, adjacency = KIMORE kinematic chain
(c) GraphTransformer: dual-stage (spatial Transformer with bone-distance ALiBi attention, then
    temporal Transformer); use_graph_bias=True
(d) GraphTransformer (ablation): same as (c) with use_graph_bias=False
(e) TCN: 4-block dilated causal convolution (dilation 1,2,4,8), input=flattened 25J×3C
(f) SCT: single-stage Transformer over unified T×J=2500 token sequence
(g) Exp E: multitask dual-stage Transformer trained on KIMORE + UI-PRMD (15 exercises)

All models: exercise embedding (KIMORE ex. 0-4), multitask auxiliary heads (PO + CF scores),
data augmentation, 5-fold Stratified LOSO, early stopping (patience=20, monitor val RMSE)

3.4 IRDS Zero-Shot Reliability Validation
- Filename-schema verification: IRDS filenames follow the schema m{EE}_s{SS}_e{RR}, where
  EE = exercise type, SS = subject, RR = repetition. We empirically verified this via bone-length
  analysis (src/verify_irds_labels.py): the bone-length ratio across m-field groups = 0.43
  (consistent with the same anatomical subjects recurring across m values), confirming that the
  m field encodes exercise type, not patient group. This corrects a plausible schema
  misinterpretation and is what enables our cross-exercise reliability analysis. (Reproducibility
  asset, not a standalone contribution.)
- Temporal resampling: IRDS sequences (~70 frames) -> 100 frames (linear interpolation)
- Joint padding: IRDS 22 joints -> KIMORE 25 joints (zeros for positions 22-24)
- Scaler: KIMORE fold-0 x_scaler applied to IRDS without refitting
- Exercise embedding: KIMORE exercise 0 used as neutral default for all IRDS sequences;
  sensitivity across all 5 KIMORE embeddings reported (mean +/- SD)
- Metrics: (1) ICC(2,1) test-retest across 10 repetitions per (exercise, subject) cell;
  (2) Kendall W coefficient of concordance across the 10 exercise-wise subject rankings;
  (3) between-subject ANOVA F-ratio; (4) mean pairwise cross-exercise Spearman rho
- Reliability != validity, and reliability metrics are gameable by degeneracy: IRDS has no
  clinical labels, so a model can be highly self-consistent (high ICC/Kendall W) while being
  clinically wrong. We therefore treat reliability as NECESSARY-BUT-NOT-SUFFICIENT and gate
  every reliability claim behind a non-degeneracy screen (prediction spread pred_SD > 0.10).
  A model failing this gate is excluded from reliability claims regardless of its ICC.

---

## 4. Results (target: ~1000 words)

4.1 KIMORE OOF Performance (Table 3 — main result table)

| Model         | RMSE  | MAE  | R²    | Spearman | Params |
|---------------|-------|------|-------|----------|--------|
| Mean baseline | 10.30 | —    | 0.00  | —        | —      |
| ST-GCN        | 9.18  | 7.24 | 0.19  | 0.447    | —      |
| BiLSTM        | 8.99  | 7.16 | —     | 0.521    | —      |
| GraphTransf.  | 9.20  | 7.34 | —     | 0.464    | 833K   |
| GT (no bias)  | 9.11  | 7.26 | —     | 0.451    | 833K   |
| SCT           | 9.23  | 7.28 | 0.196 | 0.416    | 272K   |
| Exp E         | 9.33  | 7.36 | —     | 0.463    | —      |
| **TCN**       |**8.38**|**6.26**|**0.327**|**0.549**|433K |

4.2 Per-Exercise Spearman (Table 4 — comparable to literature)

| Model / Paper           | k01   | k02   | k03   | k04   | k05   | Mean  |
|-------------------------|-------|-------|-------|-------|-------|-------|
| TCN (ours, LOSO)        | 0.371 | 0.584 | 0.465 | 0.618 | 0.709 | 0.549 |
| BiLSTM (ours, LOSO)     | 0.407 | 0.439 | 0.555 | 0.638 | 0.566 | 0.521 |
| Exp E (ours, LOSO)      | 0.356 | 0.466 | 0.402 | 0.570 | 0.522 | 0.463 |
| GraphTransf. (ours)     | 0.387 | 0.453 | 0.437 | 0.508 | 0.536 | 0.464 |
| GT no-bias (ours)       | 0.329 | 0.414 | 0.448 | 0.548 | 0.515 | 0.451 |
| ST-GCN (ours, LOSO)     | 0.365 | 0.411 | 0.405 | 0.523 | 0.530 | 0.447 |
| SCT (ours, LOSO)        | 0.321 | 0.435 | 0.347 | 0.487 | 0.490 | 0.416 |
| Abedi et al. 2023       | 0.76  | 0.61  | 0.73  | 0.54  | 0.67  | 0.662 |
| Karlov et al. 2024 SOTA | 0.79  | 0.62  | 0.77  | 0.80  | 0.74  | 0.744 |
| Dual-Stream STGCN 2026 † | 0.950 | 0.964 | 0.985 | 0.964 | 0.963 | 0.965 |
| Rehab-Pile (FG 2026) ‡ | — | — | — | — | — | MAE/RMSE only |

Note on protocol difference: Published values use 5-fold CV (non-stratified).
Our values use Stratified LOSO — strictly harder (no subject leakage, balanced clinical groups per fold).

† Dual-Stream STGCN with Motion-Aware Grouping (Kuang et al., *Sensors* 26(1):287, 2026,
DOI 10.3390/s26010287) reports mean Spearman rho ~0.965 on KIMORE under non-stratified CV,
versus our TCN rho=0.549 under Stratified LOSO. We deliberately do NOT attribute this ~0.41
gap solely to protocol. A controlled decomposition (src/preprocessing_control.py ->
outputs/novelty/preprocessing_control.json) rules out the two preprocessing differences:
Spearman is rank-invariant to the [0,50]->[0,100] score-range change (delta = 0.000), and
reducing 25->18 joints moves LOSO rho by only +0.005; the leaky-vs-LOSO inflation is a stable
+0.030 (25 joints) / +0.034 (18 joints), consistent with the +0.026 main effect in 4.6.
Thus preprocessing contributes ~0 and the measurable evaluation-protocol inflation is only
~+0.03; the large residual gap to 0.965 is attributable to architecture/training and to
leakage specific to the original non-stratified splits that we cannot reproduce without their
data. The defensible claim is therefore narrower but solid: leakage-controlled evaluation
yields substantially lower, more realistic rho, and the controlled protocol-inflation effect
is +0.026 (not the headline gap).

‡ Rehab-Pile (Ismail-Fawaz et al., IEEE FG 2026 / arXiv:2507.21018) benchmarks KIMORE with nine
architectures under a cross-subject (non-stratified, fold-averaged, 18-joint, score-[0,100])
protocol and reports MAE/RMSE rather than Spearman rho; it is therefore not directly
rho-comparable to our row, and we do not fabricate rho values for it. For context, ST-GCN
attains the best MAE/RMSE on KIMORE in their setup — independent evidence that structural priors
help on this dataset, complementing our ablation. A direct head-to-head would require re-running
our pipeline on their 18-joint/[0,100] KIMORE variant (see Future Work).

4.3 Statistical Significance (Table 5 — pairwise Wilcoxon at N=380)
Key results (Holm-Bonferroni correction over 28 tests, alpha~~0.0018):
- All models beat mean-prediction baseline: p < 1e-8 (Wilcoxon)
- After correction, only TCN > Ridge (adj p=0.005, r=0.221) and LSTM > Ridge (adj p=0.033, r=0.192) are significant
- No DL-vs-DL pair is significant after correction (largest: TCN vs SCT, raw p=0.008, adj p=0.21, r=0.157)
- Architecture choice within the DL family is not statistically decisive on KIMORE; differences reported as effect sizes
- GraphTransformer vs GT no-bias: raw p=0.199 (not sig.)

4.4 GraphTransformer Ablation (bone-distance bias)
- With bias: rho=0.464, Kendall W=0.533, cross-ex rho=0.481
- Without bias: rho=0.451, Kendall W=0.608, cross-ex rho=0.564
- KIMORE difference not significant (raw p=0.199)
- On IRDS reliability, the no-bias variant shows unexpectedly higher cross-exercise rank consistency
  (Kendall W 0.608 vs 0.533) and cross-exercise rho (0.564 vs 0.481)
- Interpretation: removing the fixed graph prior lets the model learn less KIMORE-specific,
  more transferable spatial representations; reported as an observation, not a proven mechanism

4.5 IRDS Zero-Shot Reliability Validation (Table 6)

| Model                     | ICC(2,1) | Kendall W | cross-ex rho | mean F | pred_SD | Degenerate? |
|---------------------------|----------|-----------|--------------|--------|---------|-------------|
| GraphTransformer (no bias)| 0.985    | 0.608     | 0.564        | 2061   | 0.31    | no          |
| GraphTransformer          | 0.953    | 0.533     | 0.481        | 648    | 0.03    | YES (collapsed) |
| Exp E (Transformer)       | 0.984    | 0.501     | 0.445        | 2844   | 0.20    | no          |
| SCT                       | 0.944    | 0.479     | 0.421        | 437    | 0.43    | no          |
| ST-GCN                    | 0.971    | 0.375     | 0.305        | 1395   | 0.97    | no          |
| TCN                       | 0.965    | 0.211     | 0.124        | 659    | 0.33    | no          |
| BiLSTM                    | 0.904    | 0.047     | -0.059       | 198    | 0.16    | no          |

Table 6 caption: IRDS zero-shot reliability. A model is flagged Degenerate when its IRDS
prediction spread collapses (pred_SD < 0.10): the GraphTransformer (with bone-distance bias)
collapses to near-constant outputs on IRDS (pred_SD=0.03), so its apparently high ICC/Kendall W
reflects a degenerate near-constant predictor rather than genuine discriminative reliability and
should be discounted.

Worked example that reliability is gameable (and why the degeneracy gate is essential): the
bone-distance GraphTransformer scores ICC=0.953 and Kendall W=0.533 on IRDS — numbers that, taken
alone, look strong — yet it achieves them by emitting an almost constant score (pred_SD=0.03). A
constant predictor is trivially "reliable" (perfectly repeatable) while being clinically useless.
This single case demonstrates that ICC/Kendall W are necessary-but-not-sufficient signals and
must be read jointly with the pred_SD > 0.10 non-degeneracy gate; we apply that gate throughout.

All models achieve excellent test-retest ICC. Cross-exercise rank consistency (Kendall W,
cross-ex rho) diverges and does NOT track KIMORE ranking. Embedding sensitivity (TCN across
5 KIMORE exercise embeddings): ICC range = 0.000 (stable).

4.6 Protocol-Inflation Decomposition (lead contribution)

To isolate *why* prior KIMORE results sit above ours, we ran a 2x2 factorial over the two
protocol design choices that distinguish standard CV from our evaluation: subject leakage
(KFold vs GroupKFold) x clinical-group stratification (plain vs stratified). Each cell is the
mean per-exercise Spearman rho over 20 seeds (source: outputs/novelty/protocol_decomposition.json).

| | Unstratified | Stratified |
|---|---|---|
| **Subjects leak** (KFold) | 0.516 | 0.521 |
| **No leak** (GroupKFold / our Stratified LOSO) | 0.496 | 0.489 |

Main effects relative to our Stratified LOSO reference cell (rho=0.489):
- **Subject-leakage main effect = +0.026 rho (95% CI [0.0001, 0.053], excludes 0 -> statistically significant).** Allowing the same subject into train and test inflates measured rho.
- Clinical-stratification main effect = -0.001 rho (95% CI [-0.028, +0.024], includes 0 -> negligible). Balancing clinical groups per fold has essentially no effect on the point estimate.
- Interaction = +0.011; total inflation of leaky-unstratified KFold vs our Stratified LOSO = +0.027 rho.

Interpretation: the inflation in non-stratified five-fold CV is driven almost entirely by
subject-identity leakage, not by clinical-group balancing. This is the paper's lead
contribution: it explains, with a controlled factorial and bootstrap CIs, why ungrouped CV
overstates KIMORE benchmark performance, and why subject-level (LOSO) evaluation is the
correct default for rehabilitation quality scoring.

4.7 Validating the reliability diagnostic against movement-quality labels (REHAB24-6)

A reliability metric is only useful if it tracks *validity* — whether a model's score actually
separates well-performed from poorly-performed movement. IRDS carries no such labels, so we tested
this directly on REHAB24-6 (Zenodo 13305826), an OptiTrack-mocap set of 1,057 repetitions
(558 correct / 499 incorrect) across 6 exercises and 10 subjects with per-repetition binary
correctness labels. We mapped its 26-joint OptiTrack Motive skeleton to the KIMORE 25-joint
Kinect-v2 layout (validated against the official joints_names.txt) and applied each KIMORE-trained
model strictly zero-shot, identical to the IRDS protocol. Validity is quantified as the AUROC of
the predicted score against the correct/incorrect label, averaged per exercise
(source: outputs/novelty/reliability_validity.json; Figure 7).

The result is a clean negative: no model discriminates correct from incorrect movement above
chance. Per-model AUROC ranges 0.512–0.547 (point-biserial |r| < 0.07), and the best value over
*all seven models × all five KIMORE exercise embeddings* is only 0.58. The IRDS-degenerate
GraphTransformer sits at AUROC=0.52, consistent with the degeneracy gate. Crucially, this is not
an artifact of the cross-dataset joint mapping or the labels: three naive kinematic features
(total joint path, mean speed, bounding-box volume) computed on the *same* mapped sequences reach
mean |AUROC| = 0.71 (per-exercise up to 0.87). The correctness signal is therefore present and
linearly accessible; the trained scorers simply fail to transfer it zero-shot. This null
replicates across independent partitions (outputs/novelty/reliability_validity_robustness.json):
the naive feature exceeds AUROC 0.65 in 6/6 exercises while the best model stays at chance in 5/6,
and the pattern holds in both disjoint subject halves — so it is not an artifact of pooling, any
single exercise, or subject sampling.

**Second-corpus replication (UI-PRMD).** We then repeated the identical zero-shot protocol on a
second, fully independent labeled corpus — the UI-PRMD correct/incorrect set (Vakanski et al. 2018;
2,000 segmented Kinect repetitions, 1,000 correct / 1,000 incorrect, 10 exercises × 10 subjects),
processed through the same 22->25 joint padding used for IRDS (source:
outputs/novelty/reliability_validity_uiprmd.json; Figure 7b). The core finding replicates: no
KIMORE-trained scorer exceeds chance (per-model AUROC 0.452–0.533; best over all seven models ×
five exercise embeddings only 0.533), and the IRDS-degenerate GraphTransformer again collapses to
AUROC=0.50 — so the pred_SD non-degeneracy gate holds on a second corpus. One honest caveat
distinguishes the two testbeds: UI-PRMD's "incorrect" repetitions are healthy subjects performing
arbitrary *non-optimal* (not clinically-graded) movements whose strongest correct/incorrect cue is
repetition duration (≈148 vs 68 frames), a cue our fixed-length SEQ_LEN=100 resampling deliberately
removes. Even the naive kinematic baseline therefore reaches only mean |AUROC| 0.65 here (vs 0.71 on
REHAB24-6), making UI-PRMD a harder, subtler test of the correctness signal. The trained scorers
nonetheless remain at chance (0.53) and below the naive features, and the null holds in 9/10
exercises and both disjoint subject halves
(outputs/novelty/reliability_validity_robustness_uiprmd.json). Across both labeled corpora — one
with clinically-defined errors, one with subtle non-optimal execution — zero-shot reliability does
not confer validity: the diagnostic is a necessary screen-out, not a sufficient proof of clinical
validity.

This sharpens the central methodological claim. High zero-shot reliability (ICC > 0.90, Kendall W
up to 0.61) does *not* imply validity: a model can be perfectly self-consistent across repetitions
and exercises yet score movement quality at chance on an external labeled set. The
reliability-plus-degeneracy diagnostic is thus validated as a *necessary screen-out* — the
degeneracy gate correctly flags the collapsed model, and no model that fails it could be valid —
but it is *not a sufficient screen-in*. We accordingly treat the deployment rubric (§5.5.2) as a
set of necessary exclusion criteria, not a certificate of clinical validity, and we recommend that
any cross-dataset deployment be validated against movement-quality labels before clinical use.

---

## 5. Discussion (target: ~800 words)

5.1 Why TCN outperforms attention models on KIMORE
- Dilated causal convolution captures local temporal patterns at multiple scales
- Periodic motion (hip circumduction, k05: TCN rho=0.709 vs. others 0.49-0.57)
- Attention mechanisms dilute local patterns when sequence length T=100
- [Pull from clinical_narrative.md section on k05 circumduction]

5.2 The generalization dissociation — KIMORE accuracy vs. IRDS reliability (exploratory)
- Across the N=7 evaluated models, KIMORE mean rho and IRDS Kendall W show a negative but
  statistically NOT significant rank relationship (Spearman r=-0.39, p=0.38); we present this
  as an exploratory observation, not a confirmed effect (the comparison is underpowered — see 5.x power analysis)
- BiLSTM has high KIMORE rho (0.521; pairwise differences not significant after Holm-Bonferroni)
  yet the lowest IRDS cross-exercise rank consistency (Kendall W=0.047)
- TCN has the numerically highest KIMORE rho (0.549) but only mid-pack IRDS consistency (W=0.211)
- GT-no-bias is mid-pack on KIMORE (rho=0.451) but highest IRDS consistency (W=0.608)
- Hypothesis (not proven): BiLSTM memorizes KIMORE patient-group motor signatures; GT-no-bias
  learns less dataset-specific spatial representations that generalize better
- [Pull from clinical_narrative.md section on generalization paradox]

5.3 Structural priors and transferability
- Bone-distance bias: on KIMORE marginal (rho 0.464 vs 0.451, p=0.199);
  on IRDS the no-bias variant has higher cross-exercise rank consistency (W=0.608 vs 0.533)
- TCN's local temporal receptive field captures periodic motion well (k05 rho=0.709)
  but does not guarantee cross-exercise rank consistency
- SCT (unified T×J attention): global attention without structure learns KIMORE-specific patterns
  that do not transfer (W=0.479, cross-ex rho=0.421)
- Key insight: purely learned attentive representations may memorize dataset-specific patterns;
  representations that generalize better across exercise types may emerge from reduced structural priors
- [Pull from clinical_narrative.md: Sections 3 and 4]

5.4 Per-exercise clinical interpretation
- k05 (Hip Circumduction): TCN's temporal multi-scale advantage (periodic motion)
- k02 (Trunk Forward Flexion): sagittal-plane sensor limitation
- k04 (Hip Abduction): most reliable exercise for all architectures (frontal plane, clear ROM)
- [Pull full per-exercise explanations from clinical_narrative.md]

5.5 Limitations
- KIMORE n=78 is small for deep learning; all results subject to LOSO fold variance
- IRDS has only N=10 subjects, so all IRDS reliability statistics carry wide confidence intervals; the KIMORE-rank vs IRDS-consistency dissociation (r=-0.39, p=0.38) is underpowered and presented as exploratory only
- The GraphTransformer (with bone-distance bias) collapses to near-constant predictions out-of-distribution on IRDS (pred_SD=0.03); its high ICC/Kendall W is a degenerate artifact and is excluded from reliability claims
- Human-rater reliability for the KIMORE clinical scores is not confirmed in the source dataset paper; we therefore anchor interpretation against an external human-rater ceiling — the cross-diagnostic Movement Quality Score reports inter-rater ICC[2,1]=0.93 (J. Rehabil. Med.; 2 physiotherapists, 68 inpatients), within a broader physiotherapy ICC band of ~0.6-0.93 — rather than a dataset-specific KIMORE human baseline (see human_rater_baseline.md). Our models' IRDS test-retest ICC (>0.90) should be read against this ~0.93 human ceiling, not against perfect agreement
- IRDS carries no clinical severity labels; we therefore report reliability (consistency) rather than diagnostic accuracy. Cross-exercise consistency is a proxy for, not a direct measure of, clinical validity
- The reliability≠validity result (§4.7) is established on two independent labeled corpora — REHAB24-6 (1,057 reps, clinically-defined errors; naive AUROC 0.71) and UI-PRMD (2,000 reps, arbitrary non-optimal execution; naive 0.65) — with within-corpus replication across exercises and both subject halves in each. The corpora are complementary: REHAB24-6 carries a strong correctness signal and cleanly shows the trained scorers fail to transfer it, while UI-PRMD's signal is weaker (its principal duration cue removed by fixed-length resampling) and is a harder test, yet the scorers again sit at chance (best AUROC 0.53). A clinically-graded third corpus (e.g. IntelliRehabDS CorrectLabel) would further strengthen external validity
- Exercise embedding uses KIMORE exercise 0 as neutral for IRDS; sensitivity analysis across all 5 embeddings shows ICC stable (range < 0.001), indicating robustness to this choice
- No clinical co-author on current draft — per-exercise interpretation is engineering inference

5.5.1 Statistical power of the dissociation (case observation + design guideline)
The KIMORE-rank vs IRDS-consistency dissociation (Spearman r=-0.393, p=0.38) is underpowered.
A Monte-Carlo power analysis (2000 sims/cell, alpha=0.05; source
outputs/novelty/power_analysis.json) shows that at our actual design (7 models x 10 IRDS
subjects) statistical power to detect this effect is only ~0.11. No realistic budget reaches
80% power: even 30 models x 50 subjects yields power ~0.55, and power scales primarily with the
number of models, not subjects (7 models: ~0.11; 30 models: ~0.54). The field design guideline
that follows is that cross-dataset generalization claims of this effect size require on the
order of >=30-50 candidate models (well beyond a typical single-paper benchmark) before a
medium dissociation can be detected at 80% power. We therefore present the dissociation as a
case observation and a power/design guideline, not a confirmed effect.

5.5.2 Clinical Deployment Readiness Criteria
Beyond KIMORE ranking, we propose a 4-criterion deployment-readiness rubric (source
outputs/novelty/deployment_rubric.json): (1) sample-level significant improvement over the
mean baseline at N>100; (2) cross-exercise consistency Kendall W > 0.5; (3) non-degenerate
predictions (pred_SD > 0.10); (4) priority to frontal-plane exercises. Applying it, only
**GraphTransformer (no bias)** (W=0.608, pred_SD=0.31) and **Exp E** (W=0.501, pred_SD=0.20)
pass all four criteria; the bias GraphTransformer is flagged degenerate (pred_SD=0.03) despite
W=0.533, and the KIMORE-best TCN fails the consistency screen (W=0.211). Deployment selection
should therefore use these criteria, not KIMORE accuracy ranking alone. These four criteria are
*necessary exclusion* filters, not a validity guarantee: §4.7 shows that even rubric-passing
models score movement correctness at chance on the external REHAB24-6 labeled set zero-shot
(best AUROC 0.58, vs 0.71 for naive kinematics), so any cross-dataset deployment additionally
requires validation against movement-quality labels.

5.6 Future work
- IRDS-pretrained TCN fine-tuned on KIMORE (Karlov 2024 approach applied to TCN)
- Larger rehabilitation datasets (UIPRMD-extended, NTU RGB+D with clinical labels)
- Clinical co-author review of per-exercise biomechanical interpretations

---

## 6. Conclusion (target: ~200 words)

We presented a rigorously-evaluated multi-architecture benchmark for rehabilitation quality
scoring under Stratified Leave-One-Subject-Out evaluation, paired with a cautionary
protocol-inflation analysis, with zero-shot external reliability validation on IntelliRehabDS.
We frame this work as a rigorously-evaluated benchmark and protocol-inflation cautionary study
rather than a claim of architectural superiority. Using N=380 out-of-fold predictions rather than fold-level
aggregates, we demonstrated that after FWER correction, only TCN and LSTM significantly
outperform the Ridge handcrafted-feature baseline; no DL architecture dominates another on
KIMORE. All seven architectures significantly beat the mean-prediction baseline (p < 1e-8).

A secondary, exploratory finding is a generalization dissociation: BiLSTM, with high KIMORE rho
(0.521; pairwise differences not significant after Holm-Bonferroni), shows the lowest IRDS
cross-exercise rank consistency (Kendall W=0.047), whereas the bone-distance-free
GraphTransformer, mid-pack on KIMORE (rho=0.451), shows the highest (W=0.608). Across the N=7
models this relationship is negative but not significant (r=-0.39, p=0.38), so we present it as
a cautionary case observation rather than a confirmed effect. It nonetheless challenges the
convention of selecting rehabilitation AI architectures by benchmark ranking alone.

For clinical deployment, we recommend selecting models by a 4-criterion readiness rubric, not
by KIMORE ranking alone: (1) statistically significant improvement over the mean baseline at
sample level (N>100); (2) Kendall W > 0.5 cross-exercise consistency as a minimum generalization
screen; (3) non-degenerate predictions (pred_SD > 0.10) to exclude collapsed near-constant
models; (4) priority to frontal-plane exercises (Hip Abduction, Hip Circumduction) for highest
reliability. Under this rubric only GraphTransformer (no bias) and Exp E qualify, whereas the
KIMORE-best TCN does not (Kendall W=0.211). Finally, we stress-tested the diagnostic against
external correctness labels on two independent corpora (REHAB24-6, 1,057 labeled repetitions;
and UI-PRMD, 2,000 labeled repetitions): no model exceeded chance discrimination zero-shot
(best AUROC 0.58 and 0.53 respectively) while naive kinematics on the same sequences reached
0.71 and 0.65, underscoring that our zero-shot reliability diagnostic is a necessary deployment
screen, not a sufficient proof of clinical validity.

---

## References (with DOIs where confirmed)

- Capecci, M., Ceravolo, M.G., Ferracuti, F., Iarlori, S., Monteriu, A., Romeo, L., Verdini, F.
  "The KIMORE Dataset: KInematic Assessment of MOvement and Clinical Scores for Remote
  Monitoring of Physical REhabilitation." *IEEE Trans. Neural Syst. Rehabil. Eng.* 27(7):
  1436-1448, 2019. DOI: 10.1109/TNSRE.2019.2923060.
- Vakanski, A., Jun, H.-P., Paul, D., Baker, R. "A Data Set of Human Body Movements for Physical
  Rehabilitation Exercises (UI-PRMD)." *Data* 3(1):2, 2018. DOI: 10.3390/data3010002.
- Ismail-Fawaz, A., et al. "Rehab-Pile" (cross-subject rehabilitation benchmark aggregating
  KIMORE + UI-PRMD + IRDS). arXiv:2507.21018, IEEE FG 2026.
- Kuang, Z., Yin, Z., Yang, Y., Zhao, J., Sun, L. "Dual-Stream STGCN with Motion-Aware Grouping
  for Rehabilitation Action Quality Assessment." *Sensors* 26(1):287, 2026. DOI: 10.3390/s26010287.
- Karlov, M., Abedi, A., Khan, S.S. "Rehabilitation Exercise Quality Assessment through
  Supervised Contrastive Learning with Hard and Soft Negatives" (ST-GCN, IRDS->KIMORE transfer
  learning). *Med. Biol. Eng. Comput.*, 2024/2025. DOI: 10.1007/s11517-024-03177-x.
  arXiv:2403.02772.
- Abedi, A., Malmirian, M., Khan, S.S. "Cross-modal Video to Body-joints Augmentation for
  Rehabilitation Exercise Quality Assessment." arXiv:2306.09546, 2023. [journal DOI not yet confirmed]
- Yan, S., Xiong, Y., Lin, D. "Spatial Temporal Graph Convolutional Networks for Skeleton-Based
  Action Recognition." *AAAI* 2018. arXiv:1801.07455.
- Bai, S., Kolter, J.Z., Koltun, V. "An Empirical Evaluation of Generic Convolutional and
  Recurrent Networks for Sequence Modeling." arXiv:1803.01271, 2018.
- Mennella, C., et al. "Evaluating inter- and intra-rater reliability in assessing upper limb
  compensatory movements post-stroke." *J. NeuroEng. Rehabil.* 21:217, 2024.
  DOI: 10.1186/s12984-024-01506-7.
- Cutting Movement Assessment Score (CMAS) inter-/intra-rater reliability. *Int. J. Sports
  Phys. Ther.* (see human_rater_baseline.md for full clinical-ICC context).
- "Inter-rater reliability and construct validity of a cross-diagnostic movement quality score
  for rehabilitation assessment." *J. Rehabil. Med.* (MQS inter-rater ICC[2,1]=0.93; human-rater
  ceiling anchor). https://medicaljournalssweden.se/jrm/article/view/45701

---

## Tables and Figures Needed

1. Table 1: Literature comparison (Spearman rho by exercise + protocol column) — DONE (sample_stats)
2. Table 2: Architecture summary (params, depth, key design choice)
3. Table 3: Full KIMORE OOF results (RMSE, MAE, R², Spearman, beats baseline sig.)
4. Table 4: Per-exercise Spearman vs. literature — DONE (per_exercise_spearman.csv)
5. Table 5: Pairwise sample-level tests (N=380 Wilcoxon) — DONE (pairwise_sample_level.csv)
6. Table 6: IRDS reliability (ICC, Kendall W, cross-ex rho) — DONE (irds_reliability.csv)

Figures:
F1. Dataset overview: KIMORE clinical groups, score distributions, 5 exercise skeletons
F2. Architecture diagrams: TCN vs GraphTransformer vs SCT (one panel each)
F3. Per-exercise violin plots: predicted vs true score for best 3 models
F4. IRDS cross-exercise rank-consistency: Kendall W and cross-ex rho per model (bar chart)
F5. KIMORE rho vs IRDS Kendall W scatter: visualising the dissociation
F6. Pairwise effect-size heatmap (rank-biserial r; FWER-significant pairs marked) — DONE (fig6_pairwise_effect_heatmap.png)
F7. Reliability vs validity (REHAB24-6): Kendall W vs AUROC per model, degenerate flagged, naive-kinematic baseline band — DONE (fig7_reliability_vs_validity.png)
F7b. Reliability vs validity, second corpus (UI-PRMD): same axes, cross-dataset replication of the null — DONE (fig7_reliability_vs_validity_uiprmd.png)

---

## Estimated Timeline

Week 1 (current):    All results complete. Clinical narrative drafted.
Week 2:              Write Sections 3 (Methods) and 4 (Results) using outline above.
Week 3:              Write Sections 1 (Intro), 5 (Discussion), 6 (Conclusion).
                     Generate all tables and figures from existing CSVs.
Week 4:              Internal review. Find clinical co-author (physiotherapist or
                     rehabilitation physician) for Methods/Discussion review.
Week 5-6:            Co-author revision. Polish references. Final submission.

Target: Computers in Biology and Medicine submission by Week 6.
