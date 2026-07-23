# Novelty Opportunities & Publishable Open Problems
## Derived from the `D:\Rehabilation` codebase + the completed literature review

*Produced via Sequential Thinking, 2026-06-29. Companion to `literature_review.md` (gap analysis §10), `paper_outline.md`, and `EXECUTION_PLAN.md`. Purpose: a reusable menu of de-risked, publishable contributions this codebase can support.*

---

## 0. Method & scoring doctrine

An open problem is treated as **publishable** when: (i) it is real and unsolved in the current literature; (ii) the codebase is already ≈60–90% of the way to addressing it (de-risked); and (iii) it yields a **falsifiable** claim. Each candidate is scored 1–5:

- **N** = Novelty (5 = no precedent)
- **F** = Feasibility from *current* code (5 = already implemented)
- **V** = Venue fit (5 = clear top-tier target)
- **R** = Risk (1 = low / safe, 5 = high)

### Asset inventory the candidates build on
- **Protocol engine:** `src/train_loso.py` (Stratified LOSO), `src/protocol_inflation.py` (quantifies non-stratified inflation), `src/ridge_baseline.py`, `src/sample_level_stats.py` (N=380 Wilcoxon + bootstrap CI), `src/generate_oof.py`.
- **Architectures:** `src/models.py` + `src/models_stgcn.py` — 7 models incl. bone-distance **ALiBi GraphTransformer** with a `use_graph_bias` flag (+ ablation), TCN, SCT, BiLSTM, multitask Exp E (KIMORE+UI-PRMD).
- **Zero-shot reliability:** `src/irds_eval.py` — ICC(2,1), Kendall W, cross-exercise ρ, **degeneracy detector** (`pred_SD < 0.10`).
- **Data assets:** KIMORE (78 subj / 5 ex), IRDS (10 subj / 10 ex), UI-PRMD (15 ex); `src/verify_irds_labels.py` (filename-schema verification by bone length).
- **Honest weaknesses (from `EXECUTION_PLAN.md`, all confirmed):** no significant KIMORE headline; dissociation underpowered (r=−0.393, p=0.38, N=7 models / 10 subjects); GraphTransformer OOD prediction-collapse (≈const 41.4); no human-rater anchor; small N. *These are converted into contributions below.*

---

## 1. Candidate open problems (scored)

### N1 — Protocol-inflation **decomposition** (FLAGSHIP) · N4 F5 V5 R1
**Problem.** Reported KIMORE Spearman ranges from ρ≈0.74 (Karlov 2024) to ρ≈0.96 (Dual-Stream ST-GCN, *Sensors* 2026), while a leakage-controlled Stratified-LOSO ceiling is ρ≈0.55. How much of that gap is *protocol* vs *architecture*? Rehab-Pile (FG 2026) enforces cross-subject splits but **never quantifies the inflation**.
**Falsifiable claim.** A fixed, architecture-invariant Δρ is attributable to evaluation protocol, decomposable into (a) subject leakage and (b) clinical-group non-stratification.
**Minimal experiment.** Run each model under a 2×2 grid — {random KFold, GroupKFold} × {unstratified, clinical-group-stratified} → LOSO — and report Δρ per source. `protocol_inflation.py` already measures the headline; needs the 2×2 sweep wrapper.
**Why novel.** First *inflation-decomposition* on a rehab benchmark; isolates two leakage mechanisms separately.

### N2 — Reliability-first cross-dataset evaluation **paradigm** (FLAGSHIP) · N5 F5 V4 R2
**Problem.** Rehab models are selected by in-distribution accuracy; none is screened by label-free *reliability* on an unseen dataset. All cross-dataset skeleton work in the literature is *classification* (lit-review §8).
**Falsifiable claim.** In-distribution KIMORE accuracy does **not** predict zero-shot test-retest reliability / cross-exercise rank consistency on IRDS — so reliability must be measured directly.
**Minimal experiment.** Already implemented in `irds_eval.py` (ICC(2,1), Kendall W, cross-exercise ρ). Contribution is the *protocol*, not the specific correlation (which is underpowered — see N10).
**Why novel.** First zero-shot, label-free cross-dataset *reliability* protocol for rehabilitation-quality regression.

### N3 — OOD prediction-collapse diagnostic · N4 F5 V4 R1
**Problem.** Under domain shift the GraphTransformer collapses to a near-constant output, which **spuriously inflates** ICC and Kendall W (low variance → high apparent agreement).
**Falsifiable claim.** Reliability metrics are untrustworthy when prediction SD falls below a threshold; a variance-collapse guard is necessary before reporting ICC/W.
**Minimal experiment.** Detector exists (`pred_SD < 0.10`). Formalize: show ICC/W vs pred_SD curves; demonstrate the bone-distance bias as a structural prior that *induces* collapse OOD.
**Why novel.** ICC/W-inflation-under-collapse is a real, under-documented failure mode in reliability reporting.

### N4 — Structural-prior **strength vs. transferability** (high-upside) · N5 F3 V5 R3
**Problem.** The fixed bone-distance bias helps in-distribution marginally (ρ 0.464 vs 0.451, n.s.) but the *no-bias* variant transfers better (Kendall W 0.608 vs 0.533). SkelFormer independently finds learned > rigid anatomical prior.
**Falsifiable claim.** There is an inverse (Pareto) relationship between structural-prior strength and OOD transferability for skeleton rehab models.
**Minimal experiment.** Add a scalar λ on the graph bias (fixed RPE → partially learnable → fully learned attention); sweep λ, plot in-distribution ρ vs OOD reliability. Needs a small code knob in `models.py`.
**Why novel.** Most "ML-conference-shaped" result here; turns an ablation footnote into a general principle.

### N5 — Periodicity-matched temporal inductive bias · N4 F4 V4 R2
**Problem.** TCN dominates specifically on the periodic exercise (k05 circumduction, ρ=0.709 vs 0.49–0.57); `clinical_narrative.md` argues this anecdotally.
**Falsifiable claim.** The TCN-vs-attention performance margin is a monotone function of per-exercise signal periodicity.
**Minimal experiment.** Compute a per-exercise periodicity score (autocorrelation / dominant-FFT-peak) and correlate it with the TCN-minus-attention Δρ across the 5 exercises. Pure analysis on existing data/OOF.
**Why novel.** Converts a clinical anecdote into a quantitative law linking architecture choice to signal structure.

### N6 — Heterogeneous-dataset multitask training for OOD reliability · N3 F4 V3 R3
**Problem.** Exp E trains multitask on KIMORE+UI-PRMD. Does heterogeneous multitask supervision improve *OOD reliability* even when it doesn't help in-distribution ρ?
**Minimal experiment.** Single-dataset vs multitask ablation, scored on IRDS reliability. Model exists. *Recommend folding into N2 rather than a standalone paper.*

### N7 — Clinically-actionable deployment-readiness rubric · N3 F5 V4 R2
**Problem.** No standardized screen for "is this model safe to deploy on a new clinic's data?"
**Contribution.** A composite criterion: (1) sample-level significance over the mean baseline (N>100); (2) Kendall W > 0.5 cross-exercise consistency; (3) non-degenerate predictions (pred_SD guard); (4) frontal-plane exercise priority. Already drafted in `paper_outline.md §6`.
**Why novel.** A validated model-selection rubric tied to the dissociation evidence; strong for *Computers in Biology and Medicine*.

### N8 — IRDS label-schema correction (reproducibility note) · N2 F5 V2 R1
**Contribution.** `verify_irds_labels.py` empirically proves the filename `m` field = exercise type, not patient group (bone-length ratio 0.43). Releasing this verification corrects a plausible mis-use of IRDS labels. Supporting data note, not a standalone paper.

### N9 — Human-rater reliability anchor (high value, external-data gated) · N4 F2 V5 R4
**Problem.** ICC≈0.90 is uninterpretable without the inter-/intra-rater ICC of the KIMORE physiotherapist labels (flagged internally as problem #5).
**Contribution.** Estimate the human ceiling and report automated-vs-human reliability ratio.
**Gating.** Needs KIMORE rater-agreement data (Capecci 2019 or re-annotation) + ideally a clinical co-author. Highest clinical impact; longest horizon. Target *IEEE JBHI*.

### N10 — Power analysis / design guideline for generalization claims · N4 F5 V4 R1
**Problem.** The dissociation is underpowered (N=7 models, 10 IRDS subjects, p=0.38). Rather than overclaim, derive *how many* models/subjects are needed to detect a KIMORE-rank vs OOD-reliability dissociation.
**Falsifiable claim.** A specified minimum (models × subjects) is required to detect the observed effect size at 80% power.
**Minimal experiment.** Simulate from the existing OOF/IRDS distributions — pure analysis, no new training.
**Why novel.** Converts a weakness into a reusable design guideline for the field; reinforces the methodology framing.

---

## 2. Publishable bundles (paper-shaped)

### 🟢 PAPER 1 — *recommended now* (methodology, low risk)
**"A Reliability-First, Leakage-Controlled Benchmark for Rehabilitation Quality Scoring."**
Bundle: **N1 + N2 + N3 + N7 + N8 + N10.**
Lead with the protocol-inflation decomposition (N1); propose the zero-shot reliability evaluation paradigm (N2) with the degeneracy guardrail (N3) and the deployment rubric (N7); bound the dissociation honestly with the power analysis (N10); ship the IRDS schema verification (N8) as a reproducibility asset.
*Why safe:* fully supported by existing code + outputs; makes **methodology** (not a winning architecture) the contribution, sidestepping every weakness in `EXECUTION_PLAN.md`. **Venue:** *Computers in Biology and Medicine* / *Biomedical Signal Processing & Control*. This both validates and sharpens the current execution plan.

### 🟡 PAPER 2 — follow-up (ML venue, medium risk, high novelty)
**"Structural and Temporal Inductive Biases vs. Cross-Dataset Transferability in Skeleton Rehabilitation Models."**
Bundle: **N4 + N5.**
The graph-bias-strength sweep producing an accuracy-vs-transfer Pareto frontier (N4) + the periodicity law (N5). Needs two new code knobs (interpolatable graph-bias λ; periodicity metric) and re-runs. **Venue:** pattern-recognition / WACV-style methods venue.

### 🔴 PAPER 3 — optional, clinical (long horizon, high impact)
**Human-rater reliability anchor + clinician-validated deployment criteria.**
Bundle: **N9 (+ clinical extension of N7).** Gated on external rater-agreement data + a clinical co-author. **Venue:** *IEEE JBHI*.

---

## 3. Recommended sequencing

1. **Ship Paper 1** using the current results (no new training) — finish `EXECUTION_PLAN.md` Tasks 4–8, then add N1's 2×2 decomposition and N10's power simulation (both pure analysis).
2. **Add the λ knob** (N4) and periodicity metric (N5) → Paper 2 while Paper 1 is under review.
3. **Pursue clinical co-author + rater data** (N9) in parallel for the long-horizon Q1 clinical paper.

---

## 4. Quick-reference scoring table

| ID | Opportunity | N | F | V | R | Bundle |
|----|-------------|---|---|---|---|--------|
| N1 | Protocol-inflation decomposition | 4 | 5 | 5 | 1 | Paper 1 (lead) |
| N2 | Reliability-first eval paradigm | 5 | 5 | 4 | 2 | Paper 1 |
| N3 | OOD prediction-collapse diagnostic | 4 | 5 | 4 | 1 | Paper 1 |
| N4 | Structural-prior transferability sweep | 5 | 3 | 5 | 3 | Paper 2 (lead) |
| N5 | Periodicity-matched temporal bias | 4 | 4 | 4 | 2 | Paper 2 |
| N6 | Heterogeneous multitask for OOD | 3 | 4 | 3 | 3 | fold into N2 |
| N7 | Deployment-readiness rubric | 3 | 5 | 4 | 2 | Paper 1 |
| N8 | IRDS label-schema note | 2 | 5 | 2 | 1 | Paper 1 (asset) |
| N9 | Human-rater reliability anchor | 4 | 2 | 5 | 4 | Paper 3 (lead) |
| N10 | Power-analysis design guideline | 4 | 5 | 4 | 1 | Paper 1 |
