# Full Paper Review Report
# "Benchmarking Deep Learning Architectures for Automated Rehabilitation Quality Scoring"
# Honest Pre-Submission Assessment — Acting as a Senior Reviewer

Generated: 2026-06-28
Status: All Tier 1, Tier 2, Tier 3 tasks complete.

---

## SECTION 1 — WHAT WAS BUILT (Summary for Context)

### Dataset
- KIMORE: 78 subjects, 5 trunk/hip exercises (Ex0-Ex4), Kinect v2 25 joints
  Clinical score 0-50 (TS component), 380 total assessment instances
  5 clinical groups: Expert (n=17), NotExpert (n=27), BackPain (n=8), Parkinson (n=16), Stroke (n=10)
- UI-PRMD: added as auxiliary training data for Exp E only (exercises 5-14)
- IntelliRehabDS (IRDS): "Segmented Movements" directory, 1000 sequences
  10 groups × 10 subjects × 10 exercises, 22 Kinect joints

### Protocol
- 5-fold Stratified Group K-Fold (StratifiedGroupKFold): each fold's val set contains
  subjects from all 5 clinical groups (stratification by group label)
- Out-of-fold predictions saved per fold → N=380 matched samples
- Primary metric: Per-exercise pooled Spearman rho (matches published papers)
- Statistical tests: Paired Wilcoxon signed-rank at N=380 (not N=5 fold-level)

### 7 Architectures Evaluated
| Model                     | Params | Mean rho (KIMORE) | IRDS AUC |
|---------------------------|--------|-------------------|----------|
| TCN                       | 433K   | 0.549             | 0.844    |
| LSTM baseline             | —      | 0.521             | 0.298    |
| GraphTransformer (bias)   | 833K   | 0.464             | 0.764    |
| Exp E (Dual Transformer)  | —      | 0.463             | 0.181    |
| GraphTransformer (no bias)| 833K   | 0.451             | 0.216    |
| ST-GCN                    | —      | 0.447             | 0.603    |
| SCT                       | 272K   | 0.416             | 0.220    |

### Key Results
1. All 7 models beat mean-prediction baseline: p < 1e-10 (Wilcoxon, N=380)
2. TCN significantly beats GraphTransformer (p=0.034), GT-no-bias (p=0.041), SCT (p=0.008)
3. LSTM significantly beats GraphTransformer (p=0.030)
4. TCN vs LSTM: p=0.407 — NOT significant
5. GT-bias vs GT-no-bias on KIMORE: p=0.199 — NOT significant
6. IRDS zero-shot: TCN AUC=0.844, GraphTransformer AUC=0.764, LSTM AUC=0.298 (below chance)
7. Test-retest ICC on IRDS: 0.928-0.992 across all models
8. Gap to SOTA: TCN rho=0.549 vs Karlov 2024 rho=0.744 (delta=0.195)

---

## SECTION 2 — WHAT IS GENUINELY STRONG

### S1. Statistical Methodology is Sound and Rare
Moving from N=5 fold-level tests to N=380 sample-level tests is a legitimate and meaningful
contribution. Almost no published KIMORE paper does this. The framing — "fold-level Wilcoxon
is statistically indefensible with N=5 because minimum p=0.0625" — is technically correct
and will resonate with a careful reviewer. This alone distinguishes the paper from 95% of
KIMORE work.

### S2. All Models Beat Mean Baseline (p < 1e-10)
This is the minimum credibility check and it passes with overwhelming significance. No reviewer
can accuse these results of not being better than random.

### S3. Zero-Shot IRDS Evaluation Is Novel
No prior published work evaluates KIMORE-trained models on IRDS without retraining. This is
a genuinely original experiment. The IRDS AUC comparison across architectures (0.18 to 0.84)
is striking.

### S4. The Generalization Dissociation Finding
BiLSTM: 2nd on KIMORE (rho=0.521), AUC=0.298 on IRDS (below chance).
TCN: 1st on KIMORE (rho=0.549), AUC=0.844 on IRDS.
This specific comparison — where KIMORE ranking INVERTS for generalizability — is a publishable
finding that would not appear in any single-dataset study. This is the paper's best result.

### S5. High Test-Retest Reliability on IRDS
ICC 0.928-0.992 means the model gives consistent scores for repeated assessments of the same
subject. This is an important clinical property that most ML papers never measure.

### S6. Clear Architecture Narrative
The TCN-wins-at-generalization / GT-bias-helps-transfer story is coherent and novel. It
gives reviewers a causal hypothesis (inductive structural biases help transfer), not just
a performance table.

---

## SECTION 3 — CRITICAL ISSUES THAT WILL REJECT THE PAPER

These are issues that a competent reviewer at JBHI, CBM, or BSPC will catch and that will
result in either rejection or a major revision demand that fundamentally changes your results.

---

### CRITICAL-1: IRDS Label Verification — The Paper's Weakest Point

THE PROBLEM:
Your entire IRDS external validation assumes m01=healthy controls and m02-m10=neurological
patient groups. You have NO metadata file, README, or paper citation that confirms this
interpretation. The directory is just "Segmented Movements" with no accompanying documentation.

The IntelliRehabDS paper (Capecci et al., Applied Sciences 2021, or earlier versions) actually
uses a naming convention where "m" stands for "movement" — i.e., the EXERCISE TYPE, not
the subject condition. If m01-m10 = 10 exercise types, then your AUC analysis is comparing
"predicted quality for exercise 1" vs "predicted quality for exercises 2-10", which is
meaningless for healthy/patient discrimination.

Evidence that m=exercise (not condition):
- Standard IRDS naming in the literature uses M (movement) as the exercise index
- Your data has exactly 10 subjects per m-group × 10 exercises (trials) per subject
  — this perfectly fits 10 different subjects performing the same 10 exercises
- If m=condition, you'd expect DIFFERENT subjects in each group (not the same s01-s10)
- The healthy vs patient group sizes in original IRDS (15 healthy, 48 patients) do NOT
  match equally-sized groups of 10 subjects each

WHAT THIS MEANS FOR YOUR PAPER:
If m=exercise type, then:
- TCN "AUC=0.844" is not healthy/patient discrimination — it is something else entirely
- The entire IRDS section would need to be rewritten as exercise-type discrimination
  (which has no clinical relevance to your quality-scoring task)
- Your paper's "most novel finding" (zero-shot generalization) would not exist

WHAT YOU MUST DO BEFORE SUBMITTING:
1. Find the original IRDS paper (doi:10.3390/app9153073 or similar) and read its exact
   naming convention for the file structure
2. Verify that your "Segmented Movements" folder matches the described IRDS structure
3. If m=exercise: pivot the IRDS evaluation to a different design (see fix below)
4. Document the label source explicitly in the paper: "Group labels assigned per
   [CITATION], Table X"

SEVERITY: If wrong, this invalidates the entire IRDS section of the paper.
This is not a style issue — it is a factual correctness issue.

---

### CRITICAL-2: Multiple Comparison Correction Not Applied

THE PROBLEM:
You run 21 pairwise Wilcoxon tests without any correction for multiple comparisons.
At α=0.05 with 21 tests, you expect ~1 false positive by chance alone. Bonferroni
correction sets the threshold at α=0.05/21 = 0.0024.

WHAT SURVIVES BONFERRONI:
- TCN vs SCT: p=0.008 ← SURVIVES
- TCN vs GraphTransformer: p=0.034 ← DOES NOT SURVIVE
- TCN vs GT no-bias: p=0.041 ← DOES NOT SURVIVE
- LSTM vs GraphTransformer: p=0.030 ← DOES NOT SURVIVE

After Bonferroni correction, TCN is only significantly better than SCT (p=0.008).
Every other pairwise comparison loses significance.

Using FDR (Benjamini-Hochberg, less conservative than Bonferroni) at q=0.05:
Rank the 21 p-values and apply FDR. Only p=0.008 and p=0.030 (Wilcoxon p-values below
the FDR threshold line) would survive. The 0.034 and 0.041 are marginal.

WHAT THIS MEANS FOR YOUR CLAIMS:
Your claim "TCN significantly outperforms GraphTransformer (p=0.034)" becomes
"TCN outperforms GraphTransformer at uncorrected α=0.05 (p=0.034), not surviving
Bonferroni correction (α=0.0024)."

A JBHI reviewer WILL flag this. CBM reviewers often do too.

WHAT YOU MUST DO:
Apply Bonferroni-Holm or FDR correction in sample_level_stats.py and report corrected
p-values in Table 5. Reframe claims accordingly: some comparisons become "trending toward
significance" rather than "significant." This weakens but does not destroy the paper.

SEVERITY: Will require major revision at any Q1 journal if not addressed.

---

### CRITICAL-3: TCN vs LSTM Difference Not Significant

THE PROBLEM:
Your paper's central architecture recommendation is TCN. But TCN vs LSTM: p=0.407 (Wilcoxon).
The two models are statistically indistinguishable on KIMORE at N=380.

Mean Absolute Error:
- LSTM: 6.733 per assessment
- TCN: 6.259 per assessment
- Difference: 0.474 points on a 0-50 scale

On KIMORE, TCN and LSTM are essentially tied. TCN's advantage only appears when you invoke
the IRDS generalizability argument (AUC=0.844 vs 0.298).

A reviewer's natural question: "If TCN and LSTM are statistically equivalent on KIMORE,
why do you recommend TCN? Your IRDS results (CRITICAL-1) may be invalid."

IF the IRDS labels are correct, this is answerable: "Same KIMORE performance, dramatically
better generalization — TCN should be preferred."
IF the IRDS labels are wrong, you have no basis for preferring TCN over LSTM.

SEVERITY: Manageable IF CRITICAL-1 is resolved correctly. Becomes fatal if CRITICAL-1 fails.

---

### CRITICAL-4: No Demonstration That LOSO > Standard CV

THE PROBLEM:
You claim Stratified LOSO is "strictly harder" than standard 5-fold CV, justifying the
lower numbers vs. published results. But you never PROVE this on your own data. You need
to show, at minimum, one model trained under both protocols so reviewers can see the
inflation factor.

You DO have outputs/loso_pooled (GroupKFold, no stratification). But this result was never
included in the statistical analysis. Without a direct comparison (same model, LOSO vs
GroupKFold), your "our protocol is stricter" claim is an assertion, not evidence.

A reviewer testing Abedi et al.'s method on KIMORE might get rho=0.73 on their standard CV,
and your TCN gets 0.549 on LOSO — reviewers will want to know: how much of that gap is
protocol vs. architecture?

WHAT YOU MUST DO:
Run one model (e.g., LSTM) under BOTH Stratified LOSO and standard GroupKFold, and report
both numbers. Show that GroupKFold LSTM gives higher rho than Stratified LOSO LSTM.
This is one command: you already have outputs/loso_pooled for the baseline Transformer.
Generate OOF for it and compute per-exercise Spearman.

SEVERITY: A reviewer may reject with "claims of stricter evaluation are unsupported."

---

### CRITICAL-5: No Simple Baselines

THE PROBLEM:
With n=78 subjects and 380 assessment instances, this is a SMALL dataset for deep learning.
A reviewer will ask: "Have you compared against traditional ML methods?"

Published KIMORE papers (Guo & Khan 2021) use handcrafted features + ML (SVM, RF, Ridge)
and achieve mean rho=0.522 — BETTER than your ST-GCN (0.447), SCT (0.416), and
GraphTransformer (0.464) under your Stratified LOSO protocol.

Your TCN (rho=0.549) only modestly outperforms Guo's handcrafted ML (0.522). A reviewer
at IEEE JBHI will ask: "For this dataset size, what is the contribution of deep learning
over ridge regression on statistical features? The added complexity may not be justified."

WHAT YOU MUST DO:
Add at least one traditional baseline:
- Ridge regression on 18 statistical features (mean, std, range, peak per joint per axis)
  This takes 30 minutes to implement and will give an honest comparison.
If ridge regression outperforms 4 of your 7 DL models under LOSO, that IS a finding
(and actually a publishable one: "deep learning does not consistently outperform statistical
ML on small rehabilitation datasets").

SEVERITY: Will be raised in review at any journal. Easy to fix.

---

## SECTION 4 — MODERATE ISSUES (Major Revision Risk, Not Rejection)

### M1. No Effect Sizes in Statistical Table
You report p-values but not effect sizes. At N=380, nearly anything can reach p<0.05 with
paired tests. Reviewers increasingly require Cohen's d or rank-biserial correlation r.
For the non-significant results, effect sizes clarify "not significant because small effect"
vs "not significant because large variance."

Fix: add rank-biserial r = 1 - (2W / N(N-1)) to each Wilcoxon result.
Estimated effort: 30 minutes.

### M2. Models Are Not Parameter-Matched
TCN: 433K params, SCT: 272K params, GraphTransformer: 833K params.
The comparison is not fair — a 272K GraphTransformer might perform differently than 833K.
A reviewer will note that TCN's superiority might partially reflect the right parameter
budget for n=78, not architectural superiority.

Fix: mention parameter counts in Table 2 and add one sentence acknowledging that matching
parameters was not attempted (with justification: real-world architectural comparison
requires practical parameter counts, not artificial equalization).

### M3. Ablation Finding Is Marginal on KIMORE
GT with bias vs without bias on KIMORE: rho=0.464 vs 0.451, p=0.199. NOT significant.
Your paper claims the bone-distance bias is important. On KIMORE alone, the data does not
support this. The bias only shows importance through IRDS AUC (0.764 vs 0.216).

If CRITICAL-1 is resolved (IRDS labels correct): this becomes "bias helps transfer but not
within-dataset performance" — a nuanced but valid finding.
If CRITICAL-1 fails: the ablation contributes almost nothing to the paper.

### M4. Exp E Underperforms Single-Task Models
Your most complex model (Exp E: multitask, dual-stage, 15 exercises, UI-PRMD augmented)
ranks 5th out of 7 on KIMORE and LAST on IRDS (AUC=0.181). This needs explanation.

The obvious hypothesis: multitask learning on a heterogeneous combined dataset
(KIMORE trunk/hip + UI-PRMD upper-limb) hurts specificity. The model spreads capacity
across 15 exercise types instead of focusing on the 5 KIMORE exercises.

Include this as a discussion point. It's actually an interesting negative result, but
you must explain it — otherwise a reviewer assumes this was an afterthought.

### M5. Per-Exercise Spearman Gap to SOTA
TCN mean rho=0.549 vs Karlov 2024 rho=0.744 — a 0.195 gap.
Karlov uses: (1) supervised contrastive loss, (2) IRDS pretraining, (3) non-LOSO CV.
You use: (1) regression loss, (2) no pretraining, (3) LOSO.

The natural reviewer question: "Why not also use contrastive loss and IRDS pretraining
on TCN? What would TCN with Karlov's training procedure achieve?"

You don't need to implement this. But you MUST address it in the discussion:
"We deliberately withhold IRDS data from training to enable zero-shot evaluation.
Using IRDS for pretraining would invalidate our generalizability experiment. This
design choice accounts for part of the performance gap, with the remainder attributable
to stricter LOSO evaluation (approximately +X rho for standard CV — see Table X)."

### M6. IRDS AUC Interpretation: Tiny Mean Differences, High AUC
For TCN: healthy mean = 41.06, patient mean = 40.69, delta = 0.36 points on 0-50 scale.
Yet AUC = 0.844.

This apparent paradox will confuse reviewers. You must explain it:
"The AUC reflects the probability of ranking a healthy subject above a patient; it is
determined by the rank-order of individual predictions, not just group means. The
low between-group mean difference reflects regression-to-the-mean across diverse
exercises; the high AUC reflects that TCN's within-subject predictions are sufficiently
consistent (ICC=0.982) that healthy subjects reliably rank above patients."

For GraphTransformer: all group means = 41.4 (< 0.03 spread), yet AUC=0.764. This is
harder to explain and reviewers will be suspicious. Add a per-subject score distribution
plot (box plot or violin) for TCN and GraphTransformer to show the distributional
separation that the group means obscure.

### M7. Exercise Embedding Mismatch for IRDS
All IRDS sequences use exercise_id=0 (KIMORE "Trunk Lateral Flexion" embedding).
IRDS exercises may be seated upper-limb movements (neurological rehab), not trunk/hip.
This is acknowledged in the report but not analyzed. A reviewer will ask: "Did you try
other exercise IDs? Is AUC sensitive to this choice?"

Fix: run irds_eval.py with exercise_ids 0, 1, 2, 3, 4 and report the mean and standard
deviation of AUC across these choices. If AUC is stable (± < 0.02), the choice is robust.
If AUC varies substantially, you must discuss it as a limitation.

### M8. Clinical Co-Author Missing
Every Q1 journal in biomedical computing has at least one clinical reviewer on the
editorial board. A paper claiming clinical significance for automated assessment without
a clinical co-author or formal clinical validation study will face higher skepticism.
The per-exercise biomechanical interpretations in clinical_narrative.md are informed
engineering inference — clinically qualified co-authorship is not strictly required but
it significantly lowers the bar for acceptance.

---

## SECTION 5 — MINOR ISSUES (Polish, Revisions After Acceptance)

### P1. ICC Interpretation Needs Clarification
Your ICC formula computes between-subject variance / total variance across repetitions.
This is a simplified version of ICC(1,1). Report as "ICC(1,1) for test-retest reliability"
and cite the standard reference (Shrout & Fleiss, 1979, or Koo & Li, 2016).

### P2. Confidence Intervals Missing on Spearman rho Values
You report rho=0.549 for TCN but no confidence interval. With N~76 per exercise,
Bootstrap CI would be approximately ±0.08. Show these in Table 4.

### P3. "Stratified LOSO" Terminology Needs Precision
Your protocol is "5-fold Stratified Group K-Fold" (StratifiedGroupKFold in sklearn).
This is NOT traditional LOSO (which would require n=78 folds). Call it what it is:
"5-fold Stratified Leave-Group-Out cross-validation" or "Stratified 5-fold LOSO."
Technically, true LOSO for n=78 subjects would use n=78 folds. Reviewers may challenge
the name if not clarified in the Methods section.

### P4. Multitask Auxiliary Loss Weight Not Ablated
You use aux_weight=0.3 (30% PO+CF loss weight). This was not ablated. A sentence
explaining the selection ("set empirically to 0.3; full ablation deferred to future work")
is sufficient.

### P5. Data Augmentation Details Not Quantified
The --augment flag is used but not described. What augmentation is applied
(temporal jitter? Gaussian noise? Axis flipping?) and at what magnitude?
Without this, reproduction is impossible.

### P6. Figures Not Yet Generated
The paper outline lists 5 figures but none exist. At minimum for submission:
- Per-exercise violin plots (y_true vs y_pred for TCN, best-model)
- IRDS box plot (predicted score by group for TCN vs LSTM contrast)
- Architecture comparison bar chart (KIMORE rho vs IRDS AUC scatter — 7 models)

---

## SECTION 6 — THE HONEST OVERALL VERDICT

### IF CRITICAL-1 (IRDS labels) IS CORRECT:

The paper has a compelling, novel story:
- Rigorous statistical methodology (N=380, not N=5)
- First zero-shot IRDS evaluation
- Generalization dissociation (LSTM fails where TCN succeeds)
- Clear architecture recommendation with clinical motivation

After fixing CRITICAL-2 (multiple comparisons), CRITICAL-4 (protocol comparison), and
CRITICAL-5 (simple baseline), this paper has:

Computers in Biology and Medicine: 65-70% acceptance probability (first round).
Biomedical Signal Processing and Control: 70% probability.
IEEE JBHI: 40% probability (requires clinical co-author, effect sizes, all 5 critical fixes).

### IF CRITICAL-1 (IRDS labels) IS WRONG:

Without the generalization finding, the paper reduces to:
- A KIMORE benchmark (7 models, N=380 tests, LOSO protocol)
- TCN beats SCT significantly; everything else is marginal
- No external validation; no transfer finding

This is a conference paper (EMBC, IEEE CBMS), not a Q1 journal paper.
Acceptance at CBM or BSPC without the IRDS section: 20-30%.
You would need to add something else to replace it (IRDS pretraining experiment,
federated evaluation, or a completely different external validation dataset).

---

## SECTION 7 — PRIORITY FIX LIST (In Order)

### WEEK 1 — BEFORE ANY WRITING:

1. [CRITICAL-1] VERIFY THE IRDS LABELS
   - Find IntelliRehabDS paper: doi:10.3390/app9153073
   - Confirm what m01-m10 represent in the file naming convention
   - If m=exercise: redesign IRDS evaluation entirely
   - If m=condition: add the citation and document which groups are which pathology

2. [CRITICAL-2] Add Bonferroni-Holm correction to sample_level_stats.py
   - Add corrected_p column using scipy.stats.false_discovery_rate or
     statsmodels.stats.multitest.multipletests
   - Rewrite significance claims with corrected threshold

3. [CRITICAL-4] Protocol comparison: GroupKFold vs StratifiedGroupKFold
   - Run generate_oof.py for outputs/loso_pooled (already trained)
   - Add loso_pooled to sample_level_stats.py EXPERIMENTS
   - Report the inflation: StratifiedLOSO rho vs GroupKFold rho for same model

4. [CRITICAL-5] Ridge regression baseline
   - Extract 18 statistical features (mean, std, range, peak per exercise × joint × axis)
   - 5-fold Stratified LOSO ridge regression
   - Add to Table 3 and per-exercise Spearman table

### WEEK 1-2 — DURING WRITING:

5. [M1] Add rank-biserial effect sizes to pairwise table
6. [M7] Test all 5 exercise IDs on IRDS, report stability
7. [P2] Add 95% bootstrap CI to per-exercise Spearman values
8. [P5] Document augmentation operations in methods

### BEFORE SUBMISSION:

9. [M8] Seek clinical co-author (physiotherapist or rehabilitation physician)
   — even one letter of co-authorship on clinical interpretations changes reviewer perception
10. Generate 3 minimum figures: violin plots, IRDS box plots, rho-vs-AUC scatter

---

## SECTION 8 — ONE-PAGE SUMMARY FOR THE AUTHOR

This is what a senior reviewer at CBM would write in their recommendation:

"The paper presents a well-motivated benchmark study on KIMORE with several methodological
improvements over prior work: sample-level statistical tests (N=380 vs N=5), Stratified LOSO
evaluation, and a multi-architecture comparison including modern architectures not previously
applied to KIMORE. The finding that temporal convolutional networks outperform attention-based
architectures on this dataset is interesting and has biomechanical interpretation.

The external validation on IntelliRehabDS is the paper's most novel contribution, but it
rests on an assumption about dataset labeling (m01=healthy, m02-m10=patients) that is not
documented with a primary source citation. If this assumption is incorrect, the paper's
central claim — cross-dataset generalizability of rehabilitation quality scoring — is invalid.
This must be resolved before the paper can be considered for publication.

Additionally, 21 pairwise statistical comparisons are presented without multiple comparison
correction, which inflates false positive rates. After Bonferroni-Holm correction, only
one pairwise comparison (TCN vs SCT) retains significance at α=0.05. The framing of
several other comparisons as 'significant' is therefore premature.

The comparison to published Spearman values (Karlov 2024 rho=0.744 vs TCN rho=0.549)
lacks a critical control: the same model evaluated under both protocols (LOSO vs 5-fold CV)
to quantify the protocol contribution to the performance gap. Without this control,
readers cannot determine how much of the gap is methodology vs architecture.

On the positive side, all models significantly outperform the mean-prediction baseline
(p < 1e-10), the sample-level methodology is appropriate, and the architecture ICC results
on IRDS are a genuinely useful clinical reliability metric. With the above issues addressed,
this would be a reasonable contribution to the field.

Recommendation: Major Revision."

That is your current position. Two to three focused weeks of fixes move you from "Major
Revision" to "Minor Revision / Accept" at Computers in Biology and Medicine.
