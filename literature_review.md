# Literature Review
## Automated Skeleton-Based Rehabilitation Quality Scoring: Benchmarks, Evaluation Protocols, and Cross-Dataset Generalization

*Standalone review supporting the manuscript "Benchmarking Deep Learning Architectures for Automated Rehabilitation Quality Scoring Under Clinically Valid Leave-One-Subject-Out Evaluation" (`paper_outline.md`). Compiled via Tavily web search + sequential-thinking, 2026-06-29. Supersedes the Step-0 competitive scan in `KIMORE-Literature.md`.*

---

## 1. Scope and method of this review

This review maps the published landscape around six questions that determine the manuscript's novelty and framing:

1. The current state-of-the-art (SOTA) on the **KIMORE** rehabilitation-scoring benchmark.
2. Recent **surveys** of skeleton/video action-quality assessment (AQA) and rehabilitation assessment (2024–2026).
3. **Leave-one-subject-out (LOSO) / subject-leakage** evaluation methodology.
4. **TCN vs. Transformer** inductive biases for periodic human motion.
5. **Cross-dataset generalization** in skeleton-based modeling.
6. **Graph-transformer attention-bias** designs (relative positional / structural priors).

Search was conducted with the Tavily "advanced" depth across general and academic sources; the most material competitor (Rehab-Pile) was retrieved in full text and read end-to-end. Each section closes with the implication for the manuscript's contributions (a)–(e) as numbered in `paper_outline.md §1.3`.

---

## 2. Executive summary — where the manuscript stands

| Manuscript contribution | Verdict after review | Required action |
|---|---|---|
| (a) *First* multi-architecture KIMORE benchmark without subject leakage | **Partially pre-empted** by Rehab-Pile (IEEE FG 2026) | **Reframe wording** (§3) |
| (b) Zero-shot KIMORE→IRDS *reliability* study (ICC, Kendall W, cross-exercise ρ) | **Novel — no precedent found** | Keep; lead with it |
| (c) Dissociation: KIMORE rank ≠ IRDS cross-exercise consistency | **Holds qualitatively only** (population test underpowered, see §6 + EXECUTION_PLAN) | Demote to *case observation* |
| (d) Ridge LOSO baseline + protocol-inflation quantification | **Novel and strengthened** by new ρ≈0.96 SOTA | **Lead headline** |
| (e) Per-exercise clinical interpretation | Uncontested | Keep; needs clinical co-author |

**Single most important external work:** **Rehab-Pile** (Ismail-Fawaz et al., IEEE FG 2026; extended arXiv:2507.21018) — a 60-dataset, 9-architecture rehabilitation benchmark that *includes KIMORE, UI-PRMD and IRDS*. It must be cited prominently and explicitly differentiated (§3).

---

## 3. The competing benchmark: Rehab-Pile (threat #1)

**Citation:** Ismail-Fawaz, Devanne, Berretti, Weber, Forestier. *"A Standardized Benchmark for Skeleton-Based Rehabilitation Assessment Using Deep Learning,"* IEEE Int. Conf. on Automatic Face & Gesture Recognition (FG) 2026, Kyoto. Extended version: *"Deep Learning for Skeleton Based Human Motion Rehabilitation Assessment: A Benchmark,"* arXiv:2507.21018 (Jul 2025). Project page: msd-irimas.github.io/pages/DeepRehabPile.

**What it is.** An aggregated open archive ("Rehab-Pile") of **60 datasets** — 21 extrinsic-regression, 39 classification — spanning **KIMORE, UI-PRMD, IRDS, KERAAL, KINECAL, EHE**, benchmarked with **nine architectures**: FCN, Hybrid-Inception (H-Inception), LITEMV, DisjointCNN, ConvLSTM, MotionGRU, Vanilla Transformer (VanTran), ConvTran, and **ST-GCN** (the only one consuming a true skeleton graph; the rest treat input as a flattened multivariate time series).

**The four differentiators that preserve the manuscript's contribution:**

| Axis | Rehab-Pile (FG 2026) | This manuscript |
|---|---|---|
| **Protocol on KIMORE** | Cross-subject 5-fold (≥10 unhealthy subjects → 5-fold; LOSO only for tiny datasets). **No clinical-group stratification.** Fold-averaged metrics. | **Stratified** LOSO (5 clinical groups balanced per fold) + **sample-level** inference at N=380 (paired Wilcoxon / t-test) |
| **Cross-dataset transfer** | **None.** Each dataset evaluated in isolation. IRDS used **only as binary classification** (`IRDS_CLF_BN`). No ICC / Kendall W / test-retest anywhere. | **Zero-shot KIMORE→IRDS reliability** (ICC(2,1), Kendall W, cross-exercise ρ) |
| **Architectures** | Generic TSC + ST-GCN | TCN, **bone-distance ALiBi GraphTransformer (+ ablation)**, SCT, multitask KIMORE+UI-PRMD |
| **KIMORE preprocessing/metric** | 18 joints, 3 dims, score [0,100], Fourier-resampled to mean length; **MAE/RMSE** primary | 25 joints, [0,50]; **Spearman ρ** primary |

**Corroboration worth citing:** Rehab-Pile reports that **ST-GCN wins MAE/RMSE on KIMORE and UI-PRMD** under its fair cross-subject setup — independent evidence that structural priors matter on these datasets, useful context for the manuscript's ablation.

**Required manuscript edits:**
- Rewrite contribution (a): from *"first multi-architecture KIMORE benchmark without subject leakage"* → *"first KIMORE benchmark under* ***Stratified*** *LOSO with* ***sample-level*** *statistical inference (N=380) and an explicit protocol-inflation analysis, paired with the first* ***zero-shot cross-dataset reliability*** *study."*
- Add a Related-Work paragraph (§2.4) using the table above.
- Add Rehab-Pile's 9 architectures + cross-subject protocol to the Table-1 protocol column.

---

## 4. KIMORE state-of-the-art and the protocol-inflation argument (→ contribution d)

### 4.1 Published competitive landscape

| Paper | Year | Method | Protocol | Metric | KIMORE result (per-exercise k01–k05 / mean) |
|---|---|---|---|---|---|
| **Dual-Stream ST-GCN (Motion-Aware Grouping)** | **2026** | Two-stream ST-GCN + motion grouping | Non-stratified split | Spearman ρ | **0.950 / 0.964 / 0.985 / 0.964 / 0.963** (≈0.96) |
| Karlov et al. | 2024 | Supervised contrastive + ST-GCN, IRDS→KIMORE transfer | 5-fold CV | Spearman ρ | 0.79 / 0.62 / 0.77 / 0.80 / 0.74 (≈0.74) |
| Abedi et al. | 2023 | Cross-modal video→joints aug. + LSTM | 5-fold CV | Spearman ρ | 0.76 / 0.61 / 0.73 / 0.54 / 0.67 (≈0.66) |
| Guo & Khan | 2021 | Exercise-specific features + ML | 5-fold CV (implied) | Spearman ρ | 0.55 / 0.64 / 0.63 / 0.37 / 0.42 (≈0.52) |
| Karagoz et al. | 2023 | Supervised sequential contrastive + LSTM | 5-fold CV | Spearman ρ | 0.40 / 0.65 / 0.47 / 0.50 / 0.41 (≈0.49) |
| **This work (TCN)** | 2026 | TCN, dilated causal conv | **Stratified LOSO, N=380** | Spearman ρ | 0.371 / 0.584 / 0.465 / 0.618 / 0.709 (**0.549**) |

*Notes:* No prior KIMORE paper reports R²; all use Spearman ρ and (where stated) non-stratified 5-fold CV. Spearman is ordinal; ρ≈0.80 → R²≈0.64 only under a linear assumption.

### 4.2 The protocol-inflation exhibit

The **Dual-Stream ST-GCN (*Sensors* 2026, MDPI 1424-8220/26/1/287)** reports KIMORE ρ ≈ **0.95–0.98** — higher than Karlov — under a non-stratified split. Placed beside this manuscript's Stratified-LOSO ceiling of ρ ≈ 0.55, it is the strongest available demonstration that **the headline gap between "SOTA" and a leakage-controlled protocol is dominated by evaluation protocol, not architecture.** This is the manuscript's most defensible lead contribution.

### 4.3 IRDS–KIMORE coupling to date

**Supervised Contrastive Learning with Hard & Soft Negatives** (arXiv:2403.02772) builds a single model across KIMORE exercises and exploits that IRDS and KIMORE share an **identical joint adjacency**, using IRDS→KIMORE *supervised transfer pretraining*. This establishes that prior IRDS↔KIMORE coupling has only ever been **supervised transfer** — never **label-free zero-shot reliability**, which is this manuscript's contribution (b).

---

## 5. Surveys anchoring the introduction (→ §1 framing)

- **"A Decade of Action Quality Assessment"**, IJCV 2025 (arXiv:2502.02817; Yin et al.; 200+ papers, PRISMA). The largest AQA survey; documents the field-wide absence of standardized protocols — the central motivation for this manuscript's rigor.
- **"A survey of deep learning-based action quality assessment"**, *Journal of Big Data* 2026 (10.1186/s40537-026-01409-5). Covers AQA to 2025; introduces multimodal **FineRehab** (50 subjects, 16 exercises, RGB + IMU + skeleton) — a candidate future-work dataset.
- **Lei et al., "A Survey of Vision-Based Human Action Evaluation Methods"**, *Sensors* 2019 (PMC6806217). Foundational taxonomy distinguishing action *recognition* vs. *prediction* vs. *evaluation*.

---

## 6. LOSO / subject-leakage methodology (→ contribution d; §5.5 limitation)

- **"Distributional bias compromises leave-one-out cross-validation"** (PMC12662204, 2024). Standard LOO/LPO-CV induces a *negative* correlation between train- and test-fold mean labels; because models regress to the training mean, this **deflates** auROC/auPR/R² across folds. **Critical nuance for the manuscript:** this is a *downward* bias of LOO on tiny folds — *opposite in direction* to the *upward* inflation from subject leakage in non-grouped CV. The text must keep the two mechanisms distinct: the "+0.05–0.06 ρ" claim is about leakage, not distributional bias. Acknowledge in §5.5 that LOSO trades leakage-control for higher fold variance (partly mitigated by pooling OOF predictions per exercise at N≈76).
- **Lones, "How to avoid machine learning pitfalls"** (arXiv:2108.02497). Canonical rule that *datasets with multiple samples per subject must keep each subject's data together when splitting* — direct justification for subject-wise folds; also warns of "Frankenstein datasets" assembled from public sources (a caveat relevant to Rehab-Pile-style aggregation).
- **Kapoor & Narayanan, "Leakage and the Reproducibility Crisis in ML-based Science"** (reproducible.cs.princeton.edu). Leakage taxonomy: "no clean train/test separation" and "test set not drawn from the distribution of interest." Frame the zero-shot IRDS evaluation as the corrective for the latter.
- **Leave-one-complex-out (LOCO) CV.** General principle that holding out entire correlated groups yields realistic generalization estimates — the conceptual parent of Stratified LOSO + zero-shot dataset hold-out.

---

## 7. TCN vs. Transformer for periodic motion (→ §5.1)

Cross-source consensus: **dilated causal convolution (TCN)** carries a strong inductive bias for **local, multi-scale temporal patterns** at low data cost; **self-attention** captures **global long-range** dependencies but dilutes local structure on short sequences. For short (T=100), quasi-periodic rehab signals, the local bias wins — explaining TCN's dominance on k05 hip circumduction (ρ=0.709 vs. 0.49–0.57 for attention models; see `clinical_narrative.md`).

- **Bai, Kolter & Koltun (2018)** — canonical TCN; cite as primary.
- **TCAN** (Temporal Convolutional Attention Network) — dilated conv + sparse attention; achieves an extended receptive field with fewer conv layers.
- **Hybrid TCN+Transformer** works (traffic forecasting, PMC12017482; Time-Transformer, arXiv:2312.11714) repeatedly partition roles as **TCN = local features, Transformer = global** — the exact framing for §5.1 and a natural future-work hybrid.

---

## 8. Cross-dataset generalization in skeleton modeling (→ contributions b/c)

The entire cross-dataset skeleton literature targets **classification / action recognition** domain generalization — **none performs cross-dataset rehabilitation-quality *regression* or *reliability***. This confirms contribution (b) has no precedent.

- **"Recovering Complete Actions for Cross-dataset Skeleton Action Recognition"**, NeurIPS 2024 (arXiv:2410.23641). Recover-and-resample augmentation for unseen domains (PKU-MMD / NTU-RGBD / ETRI); +5% average on unseen datasets. The SOTA *classification* analogue; its temporal-mismatch insight echoes this manuscript's 70→100-frame resampling of IRDS.
- **TAHAR**, WACV 2024. Cross-attention transformer autoencoder; cross-dataset *recognition* generalization, small→large.
- **"Towards Universal Skeleton-Based Action Recognition"** (arXiv:2604.17013) and **SCoPLe** (CVPR 2025, zero-shot skeleton recognition). Zero-shot *recognition*, not quality regression.

**Framing line for §2.3 / §5.2:** *Prior cross-dataset skeleton work targets categorical action recognition; this is the first evaluation of cross-dataset* ***test-retest reliability and rank consistency*** *for a continuous rehabilitation-quality regressor (zero-shot KIMORE→IRDS).*

---

## 9. Graph-transformer attention-bias designs (→ §4.4 / §5.3 ablation)

The manuscript's bone-distance ALiBi bias is, in graph-transformer terms, a **fixed relative positional encoding (RPE)** over the joint graph. The literature both contextualizes and *explains* the finding that *removing* it improves cross-dataset consistency (Kendall W 0.533→0.608):

- **GSTN — Graph Skeleton Transformer** (MDPI *Symmetry* 14(8):1547). Injects skeleton adjacency + centrality encoding into the attention map (Graphormer-style) — the canonical "structure-in-attention" prior the bias variant implements.
- **SkelFormer** (PMC12795391, 2025). Finds **dynamic, *learned* joint correlations beat a "rigid anatomical prior."** This **directly supports** the manuscript's observation that the bone-distance-*free* GraphTransformer transfers better — the strongest single citation for §5.3.
- **Black et al., "Comparing Graph Transformers via Positional Encodings"**, ICML 2024 (PMLR v235). Formal APE-vs-RPE analysis (RPE = per-node-pair feature, e.g. shortest-path/distance, added to attention). Cite to classify the bone-distance bias as an RPE and discuss its expressivity trade-off.
- **Press et al., ALiBi (Attention with Linear Biases).** The parameter-free, distance-penalized attention mechanism the spatial bias adapts.

**Reframed §5.3 takeaway:** *a fixed structural RPE (bone-distance bias) marginally helps in-distribution (KIMORE ρ 0.464 vs 0.451, n.s.) but appears to* ***hurt transfer*** *by encoding KIMORE-specific geometry — consistent with SkelFormer's learned-over-rigid finding.*

---

## 10. Synthesised gap analysis — what is genuinely open

1. **No leakage-controlled, *stratified*, sample-level KIMORE benchmark with quantified protocol inflation.** Rehab-Pile is cross-subject but not stratified and reports fold-level metrics only.
2. **No label-free, zero-shot, cross-dataset *reliability* evaluation** of rehabilitation-quality regressors. All cross-dataset skeleton work is classification; all IRDS↔KIMORE coupling is supervised transfer.
3. **No human-rater reliability anchor** for the ICC numbers (flagged internally in `EXECUTION_PLAN.md`, problem #5) — an open need the field has not standardized.
4. **Underpowered generalization claims** are endemic: the dissociation here (r=−0.393, p=0.38, N=7 models / 10 IRDS subjects) cannot be a population claim — mirroring the field-wide small-N problem the surveys flag.
5. **Structural-prior transferability** (fixed RPE vs. learned attention) is unresolved for *rehabilitation* skeletons specifically.

*(These open problems feed directly into the companion `NOVELTY_OPPORTUNITIES.md`.)*

---

## 11. Consolidated reference list (BibTeX keys to add)

1. Ismail-Fawaz et al. 2026 — Rehab-Pile, IEEE FG 2026 / arXiv:2507.21018 **[critical]**
2. Dual-Stream ST-GCN Motion-Aware Grouping, *Sensors* 2026, MDPI 1424-8220/26/1/287 **[protocol-inflation exhibit]**
3. SupCon Hard/Soft Negatives, arXiv:2403.02772
4. Karlov et al. 2024 (SupCR + ST-GCN, KIMORE SOTA — 5-fold)
5. Abedi et al. 2023; Guo & Khan 2021; Karagoz et al. 2023 (KIMORE landscape)
6. Capecci et al. 2019 — KIMORE dataset, IEEE TNSRE 27:1436
7. Distributional bias compromises LOOCV, PMC12662204
8. Lones, ML pitfalls, arXiv:2108.02497
9. Kapoor & Narayanan, Leakage / Reproducibility crisis (Princeton)
10. Bai, Kolter & Koltun 2018 — TCN
11. TCAN — temporal convolutional attention
12. Time-Transformer, arXiv:2312.11714
13. Recovering Complete Actions, NeurIPS 2024, arXiv:2410.23641
14. GSTN, MDPI *Symmetry* 14(8):1547
15. SkelFormer, PMC12795391 **[ablation support]**
16. Black et al., Comparing Graph Transformers via PE, ICML 2024 PMLR v235
17. Press et al., ALiBi
18. Yan, Xiong & Lin 2018 — ST-GCN
19. A Decade of AQA, IJCV 2025, arXiv:2502.02817
20. AQA survey, *J. Big Data* 2026, 10.1186/s40537-026-01409-5
21. Lei et al. 2019 — Vision-based human action evaluation survey, PMC6806217

---

## 12. Open follow-ups

- [ ] Pull Rehab-Pile's exact KIMORE MAE/RMSE table (§6 of arXiv:2507.21018) for a head-to-head row, noting protocol/preprocessing differences.
- [ ] Obtain the Dual-Stream ST-GCN (Sensors 2026) split description verbatim to state its non-stratification explicitly.
- [ ] Optionally re-run the pipeline on Rehab-Pile's 18-joint/[0,100] KIMORE variant for an apples-to-apples ST-GCN comparison.
- [ ] Source an inter-/intra-rater reliability figure for KIMORE clinical scores to anchor the ICC results.
