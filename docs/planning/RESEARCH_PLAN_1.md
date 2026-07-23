# Research Plan: From Benchmark Critique to Generalizable Rehabilitation AI
## A Complete Documentation of Strategy, Architecture, and Execution Path

**Author:** Opin (NSU CSE, Senior Capstone CSE499A)  
**Date Created:** July 2, 2026  
**Status:** Active Plan — Pre-implementation  
**Goal:** Q1 journal publication (primary target: *Computers in Biology and Medicine*)

---

## Table of Contents

1. [Existing Paper — What Was Done](#1-existing-paper--what-was-done)
2. [What the Existing Paper Proved (Feynman Summary)](#2-what-the-existing-paper-proved-feynman-summary)
3. [What the Existing Paper Is Missing](#3-what-the-existing-paper-is-missing)
4. [Strategic Decision — Why This Path](#4-strategic-decision--why-this-path)
5. [The New Research Idea](#5-the-new-research-idea)
6. [Datasets Reference](#6-datasets-reference)
7. [Engineering Architecture — Phase by Phase](#7-engineering-architecture--phase-by-phase)
8. [Expected Results](#8-expected-results)
9. [Compute Budget](#9-compute-budget)
10. [Publication Strategy](#10-publication-strategy)
11. [5-Month Execution Timeline](#11-5-month-execution-timeline)
12. [One-Line Pitch for Professor](#12-one-line-pitch-for-professor)
13. [Open Questions and Decisions Needed](#13-open-questions-and-decisions-needed)

---

## 1. Existing Paper — What Was Done

**Full title:**  
*"Benchmarking Deep Learning Architectures for Automated Rehabilitation Quality Scoring Under Clinically Valid Leave-One-Subject-Out Evaluation"*

**Root directory:** `D:/Rehabilation`  
**Project status:** 26/26 tasks complete, 1 external task deferred (IntelliRehabDS)

### 1.1 The Problem This Paper Solved

Rehabilitation AI papers claimed models scoring patient exercises as well as doctors — Spearman ρ up to 0.95 on the KIMORE dataset. This paper asked: are those numbers real, or is the evaluation protocol inflated?

**Answer: inflated.** The paper proved it, quantified it, and set the correct benchmark.

### 1.2 Three Contributions of the Existing Paper

**Contribution 1 — Protocol Inflation Proof**  
Prior papers used non-stratified cross-validation, allowing the same patient to appear in both training and test folds. This means the model partially learned the identity of the "held-out" patient during training — an artificial advantage that does not exist in real deployment.

- Fix: **Stratified Leave-One-Subject-Out (LOSO)** — entire persons excluded from training.
- Measured inflation: **+0.026 Spearman ρ** from subject-identity leakage (statistically significant across 20 seeds).
- Explains why 0.95 appears in prior papers when the honest number is ~0.55.

**Contribution 2 — Zero-Shot Validity Failure**  
Even the best model (TCN, ρ=0.549) fails to distinguish correct from incorrect movements on external datasets:

| External Dataset | Best AUROC | Interpretation |
|---|---|---|
| REHAB24-6 | 0.58 | Near chance (0.5 = random) |
| UI-PRMD | 0.53 | Essentially random |

Baseline comparison: a naive feature (total joint path length + mean speed) achieves AUROC 0.71 and 0.65 respectively — beating trained deep models. The models learned dataset-specific scoring priors, not general movement quality.

**Contribution 3 — First Rigorous Fair Benchmark**  
Leaderboard under Stratified-LOSO with 20-seed bootstrap CIs and Holm-Bonferroni multiple comparison correction:

| Model | Spearman ρ | 95% CI | Stat. sig. vs Ridge |
|---|---|---|---|
| TCN | 0.549 | [0.518, 0.580] | ✓ significant |
| LSTM | 0.521 | — | ✓ significant |
| GraphTransformer | ~0.47 | — | Not significant |
| ST-GCN | ~0.46 | — | Not significant |
| SCT | ~0.45 | — | Not significant |
| Exp E | ~0.45 | — | Not significant |
| Ridge (baseline) | 0.382 | — | — |

Key finding: only TCN and LSTM significantly outperform the dumb Ridge baseline after correction. The "fancy" deep models don't.

**Dissociation finding:** KIMORE rank ≠ IRDS cross-exercise consistency rank. Spearman correlation between the two rankings = -0.39 (not significant). The model that "scores well" and the model that "ranks consistently across exercises" are different models — these measure genuinely different things.

---

## 2. What the Existing Paper Proved (Feynman Summary)

Simple version of the full story:

> A doctor wants an AI to score rehabilitation exercises. Prior papers said this works beautifully — ρ=0.95. This paper says: stop and look at how you're evaluating. When you test correctly (keep each patient entirely out of training), performance drops to ρ=0.549, and even that is only for TCN and LSTM — the other deep models don't significantly beat a simple Ridge regression. Worse, even the ρ=0.549 model, when deployed on completely new patients doing exercises from different datasets, can't tell the difference between correct and incorrect movements (AUROC ≈ chance). The honest conclusion: the models are learning something about the KIMORE data distribution, but not what matters clinically.

**Key vocabulary used in the paper:**

| Term | Plain meaning |
|---|---|
| Stratified LOSO | Test on one person at a time, never train on that person |
| Protocol inflation | Scores look better than they really are due to sloppy testing |
| Subject-identity leakage | Model secretly sees who the test person is during training |
| AUROC | Area Under ROC Curve — 0.5 = coin flip, 1.0 = perfect |
| Spearman ρ | Rank correlation — 0 = random, 1 = perfect match with doctor |
| Kendall W | Cross-exercise ranking consistency — 0 = random, 1 = perfectly consistent |
| Zero-shot transfer | Test the model on new data without any retraining |
| Degeneracy gate | Filter out models that give everyone the same score |
| pred_SD | Prediction standard deviation — below 0.10 means model is useless |

---

## 3. What the Existing Paper Is Missing

Based on external review (professor feedback), three things prevent this from being a guaranteed Q1:

| Gap | Impact on Acceptance | Difficulty to Fix |
|---|---|---|
| No algorithmic advance — paper says "models fail" but doesn't fix it | High — reviewer will ask "where is the contribution?" | Medium — requires new experiments |
| No third labeled external corpus (IntelliRehabDS) | High — doc 1 says +20% Q1 probability | Low — dataset is public |
| No clinical co-author | Medium — rehabilitation journals want physio expertise | Low effort — one email needed |
| Small dataset (78 subjects, ~380 reps) | Medium — acknowledged limitation | Cannot fix without new data collection |

**The core missing piece:** The paper proves the problem. It does not propose a solution. That is the gap this research plan closes.

---

## 4. Strategic Decision — Why This Path

### 4.1 Options Considered

**Option A (Rejected): CardioTwin — Physics-Informed Cardiac Digital Twin**  
- Requires: hospital MRI data, CFD fluid solver (OpenFOAM), cardiologist co-author, HPC cluster
- Timeline: minimum 3 years
- Feasibility for NSU resource-limited lab: **3/10**
- Verdict: A PhD-level multi-institution project. Starting here means abandoning all existing results and beginning from zero in a completely new domain.

**Option B (Rejected): Pure evaluation framework paper**  
- Contribution would be even more "methodology only" than the existing paper
- No algorithmic advance at all
- Harder to sell, not easier

**Option C (Chosen): Extend existing paper with contrastive pretraining**  
- Uses all existing code, datasets, and pipeline
- Adds one novel algorithm that directly addresses the proven failure mode
- Adds one new dataset (IntelliRehabDS) — publicly available
- Feasibility: **8.5/10**
- Timeline: 5 months
- GPU required: RTX 5070 12GB (already owned) — sufficient

### 4.2 Core Logic

The existing paper proves: models fail zero-shot because they learn **dataset-specific scoring priors** rather than **general movement representations.**

The natural algorithmic fix: force the backbone to learn general movement representations **before** it ever sees a score label — using self-supervised pretraining on unlabeled movement data.

The unlabeled data needed for this already exists: **IRDS** — 2,589 repetitions, 9 exercises, 29 subjects, zero clinical labels. Currently used only as a cross-exercise consistency testbed. This plan repurposes it as a pretraining corpus.

---

## 5. The New Research Idea

### 5.1 Title

**"Contrastive Skeleton Pretraining for Cross-Dataset Generalization in Rehabilitation Movement Quality Assessment"**

### 5.2 Core Hypothesis

If we pretrain the TCN backbone on a large unlabeled pool of diverse skeleton sequences using contrastive self-supervised learning, it will learn movement representations that transfer better to unseen datasets and sensors — improving both fine-tuned KIMORE ρ and zero-shot external AUROC.

### 5.3 Why Contrastive Learning Specifically

Contrastive learning (SimCLR-style) trains an encoder to:
- Produce **similar representations** for two augmented views of the same movement (positive pair)
- Produce **dissimilar representations** for different movements (negatives)

With no labels at all, the model is forced to learn what makes movements similar — which should be structural and kinematic properties (joint angles, temporal patterns, spatial coordination) rather than KIMORE-specific scoring conventions.

### 5.4 Three Contributions of the New Paper

1. **Novel algorithm:** First contrastive pretraining framework for skeleton-based rehabilitation quality assessment. Directly addresses the zero-shot transfer failure identified in the benchmark paper.

2. **Skeleton-specific augmentation taxonomy:** Identifies and ablates 6 augmentation strategies for clinical skeleton sequences. Establishes which augmentations preserve clinical meaning versus destroy it — a methodological contribution specific to this domain that does not exist in the literature.

3. **Third external labeled corpus (IntelliRehabDS):** Physician-scored labels (not just binary correct/incorrect), enabling the strongest possible external validity test: does the model score exercises the way a doctor would, on data it has never seen?

### 5.5 Narrative Arc of the New Paper

> Paper 1 (existing): Models fail zero-shot. Benchmarks are inflated. Here is the correct leaderboard.  
> Paper 2 (this plan): We know why they fail. We try the principled fix. Here is what it achieves — and what it cannot achieve.

Both outcomes are publishable. If pretraining improves AUROC: "contrastive pretraining partially solves the transfer gap." If it doesn't: "the transfer failure is architectural — domain shift between sensor types and scoring conventions is a fundamental barrier beyond representation quality." Either way, this is the next chapter of the research story.

---

## 6. Datasets Reference

| Dataset | Subjects | Exercises | Reps | Labels | Sensor | Role in this plan |
|---|---|---|---|---|---|---|
| KIMORE | 78 (44 healthy, 34 patients) | 5 | ~380 | Physician score 0–50 | Kinect | Primary training/evaluation (Stratified-LOSO) |
| IRDS | 29 | 9 | 2,589 | **None** | Kinect | **Pretraining corpus** (unlabeled, used for contrastive learning) |
| REHAB24-6 | 10 | 6 | 1,057 | Binary correct/incorrect | OptiTrack | External validity test (zero-shot AUROC) |
| UI-PRMD | 10 | 10 | 2,000 | Binary correct/incorrect | Kinect + Vicon | External validity test (zero-shot AUROC) |
| **IntelliRehabDS** | **30** | **9** | **TBD** | **Physician scores** | **Kinect** | **New — strongest external validity test** |

**IntelliRehabDS notes:**
- Publicly available — download location: search "IntelliRehabDS dataset" or check PhysioNet / IEEE DataPort
- Uses Kinect skeleton — same format as KIMORE, minimal preprocessing needed
- Has physician-assigned quality scores (not just binary) — enables Spearman ρ computation against human expert judgment on external data
- This is the dataset Doc 1 (professor review) specifically recommended, with +20% Q1 probability estimate

---

## 7. Engineering Architecture — Phase by Phase

### 7.1 Phase 1 — Skeleton Augmentation Module

**Goal:** Build the augmentation pipeline that generates positive pairs for contrastive training.

**Duration:** ~2 weeks  
**Files to create:** `augmentations.py` in existing project

#### Six Augmentations to Implement

```python
# Each augmentation takes: skeleton tensor of shape (T, J, 3)
# where T=timesteps, J=25 joints, 3=xyz coordinates
# Returns: augmented tensor of same shape

def temporal_crop(x, min_ratio=0.80):
    """Sample a random contiguous subsequence, resample to SEQ_LEN=100."""
    T = x.shape[0]
    crop_len = int(T * random.uniform(min_ratio, 1.0))
    start = random.randint(0, T - crop_len)
    cropped = x[start:start+crop_len]
    return resample_to_fixed_length(cropped, 100)

def joint_masking(x, n_joints=4):
    """Zero out n_joints random joints across all frames."""
    joints_to_mask = random.sample(range(25), n_joints)
    x_aug = x.clone()
    x_aug[:, joints_to_mask, :] = 0.0
    return x_aug

def speed_perturbation(x, min_rate=0.8, max_rate=1.2):
    """Stretch or compress the time axis, resample to SEQ_LEN=100."""
    rate = random.uniform(min_rate, max_rate)
    new_len = int(x.shape[0] * rate)
    return resample_to_fixed_length(x, 100, intermediate_len=new_len)

def gaussian_joint_noise(x, sigma=0.02):
    """Add Gaussian noise to all joint coordinates."""
    noise = torch.randn_like(x) * sigma
    return x + noise

def rotation_3d(x, max_angle_deg=15):
    """Rotate skeleton around vertical (Y) axis."""
    angle = random.uniform(-max_angle_deg, max_angle_deg)
    angle_rad = math.radians(angle)
    R = rotation_matrix_y(angle_rad)  # 3x3 rotation matrix
    return (x @ R.T)  # apply to xyz coordinates

def limb_scaling(x, min_scale=0.9, max_scale=1.1):
    """Scale joint positions from skeleton centroid."""
    centroid = x.mean(dim=(0, 1), keepdim=True)
    scale = random.uniform(min_scale, max_scale)
    return centroid + (x - centroid) * scale

def get_augmentation_pair(x):
    """Return two randomly augmented views of the same sequence."""
    aug_pool = [temporal_crop, joint_masking, speed_perturbation,
                gaussian_joint_noise, rotation_3d, limb_scaling]
    aug1 = random.sample(aug_pool, 2)
    aug2 = random.sample(aug_pool, 2)
    view1 = x
    for aug in aug1:
        view1 = aug(view1)
    view2 = x
    for aug in aug2:
        view2 = aug(view2)
    return view1, view2
```

**Design rationale:**
- `temporal_crop`: valid because exercise quality is in movement pattern, not absolute timing
- `joint_masking`: valid because neighboring joints carry compensatory information
- `speed_perturbation`: valid for pretraining (motor similarity holds), **but note**: for clinical scoring this is NOT valid — speed is clinically relevant. Use only in pretraining, not as a data augmentation during fine-tuning.
- `gaussian_joint_noise`: simulates Kinect tracking noise — explicitly a domain invariance augmentation
- `rotation_3d`: valid because quality is in joint angles relative to each other, not absolute orientation
- `limb_scaling`: valid because body size shouldn't determine exercise quality

**Ablation experiment:** Run contrastive pretraining with each augmentation removed one at a time. Report which subset gives best downstream KIMORE ρ. This ablation is a section in the paper.

---

### 7.2 Phase 2 — Contrastive Pretraining on IRDS

**Goal:** Pretrain the TCN backbone to produce general movement representations.

**Duration:** 3 weeks (includes tuning runs)  
**Compute:** ~4–6 hours per 300-epoch run on RTX 5070 12GB

#### Architecture

```
Input: skeleton sequence (T=100, J=25, C=3)
  ↓
[TCN Encoder — same architecture as existing best model]
  ↓ representation vector r ∈ R^128
  ↓
[Projection Head — 2-layer MLP]
  Layer 1: Linear(128, 64) + BatchNorm + ReLU
  Layer 2: Linear(64, 32)
  ↓ projection z ∈ R^32
  ↓
[NT-Xent Loss]
```

The projection head is **discarded after pretraining**. Only the TCN encoder weights are carried forward to fine-tuning.

#### NT-Xent Loss (SimCLR)

```python
def nt_xent_loss(z1, z2, temperature=0.07):
    """
    z1, z2: shape (batch_size, projection_dim)
    For each sequence i, (z1[i], z2[i]) is the positive pair.
    All other 2*(batch_size-1) pairs are negatives.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    
    sim = torch.mm(z, z.T) / temperature  # (2B, 2B)
    
    # Mask self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, float('-inf'))
    
    # Positive indices: for i in [0,B), positive is i+B; for i in [B,2B), positive is i-B
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels])
    
    loss = F.cross_entropy(sim, labels)
    return loss
```

#### Pretraining Hyperparameters

```yaml
encoder: TCN  # same architecture as fine-tuned KIMORE model
projection_dim: 32
temperature: 0.07
batch_size: 128  # fits in 12GB VRAM
learning_rate: 3e-4
optimizer: Adam (weight_decay=1e-4)
epochs: 300
scheduler: cosine annealing (min_lr=1e-6)
dataset: IRDS (2,589 sequences)
augmentations: 2 random per view
```

#### Monitoring During Pretraining

Track a **linear probe** accuracy on a held-out 20% of IRDS:
- Train a linear layer on top of frozen encoder every 50 epochs
- Proxy: cluster quality of the learned representations
- Stop if linear probe accuracy plateaus or degrades for 50 epochs

Save checkpoint at best linear probe accuracy (not best training loss).

---

### 7.3 Phase 3 — Fine-Tuning on KIMORE (Stratified-LOSO)

**Goal:** Compare pretrained vs. scratch TCN on KIMORE under the same rigorous protocol used in the existing paper.

**Duration:** ~1 week (compute runs overnight)

#### Two Experimental Conditions

**Condition A — Linear probe:**
```
Pretrained TCN encoder (frozen weights)
  ↓
[Single linear regression head: 128 → 1]
  ↓
MSE loss against physician scores
```
Tests: are contrastive features already informative for quality scoring without fine-tuning?

**Condition B — Full fine-tune:**
```
Pretrained TCN encoder (unfrozen, low LR)
  ↓
[Regression head: 128 → 1]
  ↓
MSE loss against physician scores
```
Tests: does pretraining provide a better initialization for quality scoring than random init?

**Baseline (existing):**  
TCN trained from scratch on KIMORE — ρ=0.549 [0.518, 0.580]

#### Evaluation Protocol (identical to existing paper)

- Stratified LOSO: 78-fold (one per subject)
- 20-seed bootstrap for confidence intervals
- Holm-Bonferroni correction for pairwise comparisons
- Report: Spearman ρ at repetition level (N≈380), per-subject ρ, and bootstrap CI

**Success threshold:** If pretrained full fine-tune achieves ρ > 0.565 with non-overlapping CI lower bound vs. scratch baseline, report as significant improvement.

---

### 7.4 Phase 4 — Zero-Shot Validity Re-Evaluation

**Goal:** Test whether pretraining improves the model's ability to separate correct from incorrect movements on datasets it has never seen.

**Duration:** ~1 week

#### Evaluation on Three External Corpora

For each corpus:
1. Take the KIMORE fine-tuned model (pretrained + full fine-tune) — no retraining
2. Run inference on all repetitions
3. Compute AUROC against correct/incorrect labels

| Corpus | Existing AUROC (scratch TCN) | Target (pretrained TCN) | Interpretation if no improvement |
|---|---|---|---|
| REHAB24-6 | 0.58 | > 0.65 | Domain shift (OptiTrack vs Kinect) is fundamental barrier |
| UI-PRMD | 0.53 | > 0.58 | Clinical definition mismatch is fundamental barrier |
| IntelliRehabDS | N/A (new) | > 0.60 | Baseline must be computed first |

**Note:** IntelliRehabDS has physician scores, so report Spearman ρ against physician grades as primary metric here, with AUROC on a binarized version as secondary.

#### The "No Improvement" Finding Is Still Publishable

If pretrained model shows AUROC ≤ 0.60 on all external corpora:

> "Even with contrastive pretraining on 2,589 diverse unlabeled sequences, zero-shot transfer to new sensors and scoring contexts fails. This indicates the transfer barrier is not representational diversity — it is a fundamental mismatch between KIMORE's physician scoring conventions and the binary correct/incorrect labeling in external datasets, compounded by sensor domain shift between Kinect depth data and OptiTrack/Vicon precision capture."

That is a stronger, more conclusive version of the existing paper's finding. It rules out the "easy fix" and points toward what the field actually needs (shared scoring rubrics, cross-sensor adaptation).

---

### 7.5 Phase 5 — IntelliRehabDS Integration

**Goal:** Add the first external dataset with physician-assigned quality scores.

**Duration:** ~2 weeks (preprocessing + experiments)

#### Steps

1. **Download** IntelliRehabDS — check IEEE DataPort or PhysioNet for access
2. **Joint mapping** — map IntelliRehabDS joint indices to the 25-joint Kinect format used by KIMORE. Some joints may need interpolation or dropping if sensor configurations differ.
3. **Normalization** — apply same pelvis-centering and unit normalization as KIMORE preprocessing
4. **Exercise filtering** — select exercises with the closest biomechanical match to KIMORE exercises (trunk rotation, arm raise, squat variants)
5. **Run inference** — apply KIMORE-trained model (both scratch and pretrained) with no retraining
6. **Compute metrics:**
   - Spearman ρ against physician scores (primary — directly comparable to KIMORE metric)
   - AUROC against binary quality threshold (secondary)
   - Kendall W across exercises (consistency metric)

#### Why This Dataset Changes the Narrative

REHAB24-6 and UI-PRMD validity tests are limited by binary labels — a doctor looking at a 30/50 performance would never call it simply "incorrect." IntelliRehabDS physician scores allow a richer question: when our model says "this is a 35/50 performance" and the doctor says "this is a 32/50 performance" — how well do those track? That is clinical validity in the strong sense.

---

## 8. Expected Results

### 8.1 Conservative Estimates

| Metric | Existing paper | Expected (this plan) | Notes |
|---|---|---|---|
| KIMORE ρ — TCN from scratch | 0.549 [0.518, 0.580] | Unchanged (baseline) | Reproduced exactly |
| KIMORE ρ — Linear probe | N/A | 0.48–0.52 | Shows contrastive features partially transfer |
| KIMORE ρ — Full fine-tune | N/A | 0.560–0.580 | Modest improvement, significant trend |
| REHAB24-6 AUROC | 0.58 | 0.62–0.70 | Optimistic: 0.70; conservative: 0.62 |
| UI-PRMD AUROC | 0.53 | 0.55–0.62 | Harder corpus — smaller gain |
| IntelliRehabDS ρ | N/A (new) | 0.35–0.55 | Wide range — depends on dataset structure |
| IRDS Kendall W (pretrained) | 0.047–0.608 | Higher for pretrained TCN | Tests if pretraining helps consistency |

### 8.2 Interpretation Grid

| KIMORE ρ improved? | External AUROC improved? | Interpretation | Publishability |
|---|---|---|---|
| Yes | Yes | Pretraining works — strong positive finding | Q1 |
| Yes | No | Pretraining helps KIMORE but not transfer — representational gap remains | Q1 (stronger claim about the problem) |
| No | Yes | Pretraining helps generalization even without KIMORE score improvement | Q1 (surprising finding) |
| No | No | The problem is fundamental domain shift, not representation quality | Q2–Q1 (definitive negative) |

**All four outcomes are publishable.** This is a well-designed experiment.

---

## 9. Compute Budget

Hardware: **RTX 5070 12GB VRAM**

| Task | Estimated Time | VRAM Usage | Notes |
|---|---|---|---|
| Augmentation ablation runs (6 configs × 100 epochs) | 12 hours | ~4GB | Run sequentially |
| Full contrastive pretraining (300 epochs) | 4–6 hours | ~5GB | Batch=128 |
| KIMORE LOSO — linear probe (78 folds) | ~10 hours | ~3GB | Head only, fast |
| KIMORE LOSO — full fine-tune (78 folds) | ~40 hours | ~6GB | Run overnight |
| Zero-shot inference on 3 corpora | < 1 hour | ~2GB | Trivial |
| Baseline reproduction verification | ~40 hours | ~6GB | Verify existing rho=0.549 |
| **Total** | **~107 hours** | — | Spread over ~2 weeks of overnight runs |

**Critical note:** Run full fine-tune LOSO jobs overnight. 40 hours continuous is fine on the RTX 5070. Use a job queue script to run all 78 folds sequentially without supervision.

---

## 10. Publication Strategy

### 10.1 Primary Target

**Computers in Biology and Medicine**  
- Impact Factor: ~7.0 (Q1 in Biomedical Engineering)
- Acceptance probability (this plan, with all 5 phases complete): **70–80%**
- Why: accepts methodology papers + AI + honest negative results if rigorous. The evaluation framework and contrastive pretraining together check all their boxes.
- Expected review time: 3–4 months

### 10.2 Secondary Target

**Journal of NeuroEngineering and Rehabilitation**  
- Impact Factor: ~5.5 (Q1 in Rehabilitation)
- Acceptance probability: **50–60%** (increases to 65–70% with clinical co-author)
- Requires: at minimum one physiotherapist co-author reviewing the clinical interpretation section

### 10.3 Stretch Target (only after IntelliRehabDS + clinical co-author)

**IEEE Journal of Biomedical and Health Informatics**  
- Impact Factor: ~7.7 (strong Q1)
- Acceptance probability: **30–40%**
- Do not submit here without both additions

### 10.4 Do Not Submit To (currently)

**IEEE Transactions on Medical Imaging** — requires stronger novelty, larger datasets, and clinical deployment evidence. Not feasible for this work at this stage.

### 10.5 How to Get a Clinical Co-Author

**Target:** One physiotherapist or rehabilitation physician at NSU Health Center, Dhaka Medical College, or any Dhaka rehabilitation hospital.

**Email pitch (one paragraph):**
> "I am an undergraduate CS researcher at NSU working on AI for rehabilitation exercise quality scoring. My work has been published/is being submitted to [journal]. I am looking for a clinical collaborator who can review the clinical interpretation section of the paper and co-author it. The time commitment would be a single 1-hour consultation session. I can share the paper draft at your convenience."

**Outcome needed:** Co-author bio, one-pass review of clinical language, sign-off on interpretations. This is enough to satisfy the "no clinical co-author" objection from reviewers.

---

## 11. 5-Month Execution Timeline

### Month 1 — Build and Pretrain

| Week | Task |
|---|---|
| 1 | Implement `augmentations.py` — all 6 augmentation functions with unit tests |
| 2 | Implement SimCLR training loop — NT-Xent loss, linear probe monitoring, checkpoint saving |
| 3 | Download and preprocess IntelliRehabDS — joint mapping, normalization, exercise selection |
| 4 | Run full 300-epoch pretraining on IRDS. Monitor linear probe. Verify loss curves are healthy. |

### Month 2 — Ablate and Tune

| Week | Task |
|---|---|
| 5 | Run augmentation ablation: 6 configs × 100 epochs each. Rank augmentations by downstream linear probe ρ. |
| 6 | Tune temperature (τ ∈ {0.05, 0.07, 0.10}) and batch size (64, 128). Select best config. |
| 7 | Final pretraining run with best config (300 epochs). Save encoder checkpoint. |
| 8 | Run KIMORE LOSO linear probe experiment (78 folds). Compare ρ to scratch baseline. |

### Month 3 — Full Evaluation

| Week | Task |
|---|---|
| 9 | Run KIMORE LOSO full fine-tune experiment (78 folds). Overnight. |
| 10 | Run zero-shot evaluation on REHAB24-6, UI-PRMD, IntelliRehabDS. Compute all metrics. |
| 11 | Statistical analysis: bootstrap CIs, Wilcoxon tests, Holm-Bonferroni correction. Reproduce all existing paper statistics for comparison. |
| 12 | Generate all figures: training curves, ρ comparison bars with CIs, AUROC comparison, augmentation ablation table. |

### Month 4 — Write and Collaborate

| Week | Task |
|---|---|
| 13 | Contact physiotherapist collaborator. Share draft abstract and key findings. |
| 14 | Write manuscript: Introduction, Related Work, Methods |
| 15 | Write manuscript: Experiments, Results, Discussion |
| 16 | First complete draft. Internal review. Send to collaborator for clinical interpretation review. |

### Month 5 — Polish and Submit

| Week | Task |
|---|---|
| 17 | Revise based on co-author feedback. Format to journal template (Computers in Biology and Medicine uses Elsevier template). |
| 18 | Final proofreading. Supplementary materials (code release, dataset preprocessing scripts). |
| 19 | Submit to Computers in Biology and Medicine. |
| 20 | Submit code to GitHub (public repository for reproducibility — major Q1 acceptance factor). |

---

## 12. One-Line Pitch for Professor

> "Our previous paper proved that rehabilitation AI benchmarks are inflated and that current models fail zero-shot transfer — this paper asks why they fail, proposes contrastive skeleton pretraining on unlabeled movement data as a principled fix, and validates against three independent corpora including the first physician-scored external dataset, either demonstrating partial transfer recovery or definitively ruling out the representational explanation for the failure."

---

## 13. Open Questions and Decisions Needed

These questions are unresolved and need decisions before or during Month 1:

| # | Question | Options | Decision needed by |
|---|---|---|---|
| 1 | Which SimCLR variant to use? | Standard SimCLR vs MoCo v2 vs BYOL (no negative pairs) | Week 2 |
| 2 | Should projection head be 2-layer or 3-layer MLP? | Start with 2-layer (128→64→32), ablate if needed | Week 2 |
| 3 | IntelliRehabDS access — is it freely available or requires request? | Check IEEE DataPort / PhysioNet / paper supplementary | Week 3 |
| 4 | Joint mapping between IntelliRehabDS and KIMORE — exact correspondence? | Must read IntelliRehabDS paper for joint index table | Week 3 |
| 5 | Do we retrain all 7 models with pretraining, or just TCN? | Recommend TCN + LSTM only (best performers) | Week 4 |
| 6 | Clinical co-author contact — which institution first? | NSU Health Center vs DMC vs other | Month 3 |
| 7 | Should fine-tuning use same LR as scratch training or lower? | Start with 1/10 of original LR, ablate | Week 9 |
| 8 | Code release policy — full code or just key modules? | Full code strongly recommended for Q1 | Month 5 |

---

## Appendix A — File Structure (Recommended)

```
D:/Rehabilation/
├── existing_work/          # Original benchmark paper code (do not modify)
│   ├── models/
│   ├── evaluation/
│   └── results/
├── contrastive_pretraining/    # NEW — this plan's code
│   ├── augmentations.py        # Phase 1
│   ├── simclr_trainer.py       # Phase 2
│   ├── linear_probe.py         # Phase 2 monitoring
│   ├── finetune_loso.py        # Phase 3
│   └── checkpoints/
├── datasets/
│   ├── kimore/             # Existing
│   ├── irds/               # Existing
│   ├── rehab24_6/          # Existing
│   ├── ui_prmd/            # Existing
│   └── intellirehabds/     # NEW — Phase 5
├── results/
│   ├── pretraining_curves/
│   ├── ablation_tables/
│   ├── kimore_loso_pretrained/
│   └── zero_shot_external/
└── paper/
    ├── manuscript.tex
    ├── figures/
    └── supplementary/
```

---

## Appendix B — Key References to Read Before Writing

1. **SimCLR paper:** Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations," ICML 2020 — for NT-Xent loss implementation and projection head design
2. **MoCo v2:** Chen et al., "Improved Baselines with Momentum Contrastive Learning," arXiv 2020 — alternative to SimCLR if negative pair sampling is a bottleneck
3. **BYOL:** Grill et al., "Bootstrap Your Own Latent," NeurIPS 2020 — no negative pairs, may be better for small batch sizes
4. **Skeleton contrastive learning:** Search "skeleton sequence contrastive learning action recognition" — check if anyone has done this for action recognition (different task, but augmentation design insights transfer)
5. **IntelliRehabDS paper:** Read carefully for joint definition, scoring rubric, and clinical context
6. **Capecci et al. 2019:** Original KIMORE paper — already cited in existing work, but re-read for scoring methodology details before writing IntelliRehabDS comparison section

---

*End of Research Plan Document*  
*Version 1.0 — July 2, 2026*  
*Next review: End of Month 1 (after pretraining feasibility confirmed)*
