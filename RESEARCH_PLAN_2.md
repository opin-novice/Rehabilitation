# Research Plan v2 — Corrected After Literature Review
## Does Self-Supervised Pretraining Rescue Zero-Shot Cross-Sensor Generalization in Rehabilitation Movement Scoring?

**Author:** Opin (NSU CSE, Senior Capstone CSE499A)  
**Version:** 2.0 — Post-Literature-Review Correction  
**Date:** July 2, 2026  
**Supersedes:** RESEARCH_PLAN_1.md (contained two factual errors and one pre-empted claim)  
**Goal:** Q1 journal publication (primary target: *Computers in Biology and Medicine*)

---

## Changelog from v1 → v2

| What changed | Why |
|---|---|
| IRDS and IntelliRehabDS merged into one dataset row | They are the same dataset (Miron et al. 2021, Zenodo 4610859) |
| Contribution 3 (physician-scored external corpus) deleted | IntelliRehabDS has only binary labels — no continuous physician scores exist in any of the four corpora except KIMORE |
| Contribution 1 rewritten | "First contrastive pretraining" pre-empted by Karlov 2025 and SSL-Rehab; reframed to "first honest zero-shot cross-sensor SSL evaluation" |
| Masked-motion pretraining arm added | Converts "scooped by SSL-Rehab" into "first head-to-head comparison of the two SSL paradigms" |
| Phase 5 (IntelliRehabDS integration) deleted | That dataset is already in the pipeline as IRDS |
| Must-read literature section added | 8 papers that must be read and cited before writing |
| Timeline extended by 2 weeks | For masked-motion implementation |

---

## Table of Contents

1. [What the Literature Review Found](#1-what-the-literature-review-found)
2. [What Still Stands — The Genuine Gap](#2-what-still-stands--the-genuine-gap)
3. [The New Research Framing](#3-the-new-research-framing)
4. [Three Corrected Contributions](#4-three-corrected-contributions)
5. [How This Differs from Every Prior Paper](#5-how-this-differs-from-every-prior-paper)
6. [Corrected Datasets Reference](#6-corrected-datasets-reference)
7. [Engineering Architecture — Updated](#7-engineering-architecture--updated)
8. [Expected Results](#8-expected-results)
9. [Compute Budget](#9-compute-budget)
10. [Publication Strategy](#10-publication-strategy)
11. [Updated 6-Month Timeline](#11-updated-6-month-timeline)
12. [Must-Read Literature](#12-must-read-literature)
13. [Open Questions](#13-open-questions)

---

## 1. What the Literature Review Found

### 1.1 Factual Error — IRDS and IntelliRehabDS Are the Same Dataset

Section 6 of v1 listed them as two separate datasets. They are one:

**IntelliRehabDS (= IRDS):**
- Miron, Sadawi, Grosan, Ismail, Hussain — *Data* 6(5):46, 2021
- Zenodo record 4610859
- 29 subjects (15 patients + 14 healthy controls), 9 gestures, ~2,589 repetitions
- Microsoft Kinect One, 25 joints, 30 fps
- Labels: binary only — `CorrectLabel = 1 (correct) or 2 (incorrect)`, annotated by two independent annotators

**Consequence for v1:**
- Phase 5 "add IntelliRehabDS" adds no new data — it's already in the pipeline
- Contribution 3 ("physician-scored external corpus enabling Spearman ρ") is impossible with this dataset
- The claim that IRDS was "unlabeled" was also wrong — it has binary labels

### 1.2 Factual Error — No Physician-Scored External Corpus Exists

Among all four corpora in the pipeline:

| Corpus | Label type |
|---|---|
| KIMORE | Physician score 0–50 (continuous) ✓ |
| IRDS / IntelliRehabDS | Binary: correct (1) or incorrect (2) |
| REHAB24-6 | Binary: correct or incorrect |
| UI-PRMD | Binary: correct or incorrect |

There is no second physician-scored corpus available. KIMORE is unique in this regard. The "strongest external validity test via Spearman ρ on physician grades" does not exist with available data.

### 1.3 Pre-emption — "First Contrastive Pretraining" Is Not First

Three prior papers have already used contrastive or self-supervised pretraining for rehabilitation assessment:

**Karlov, Abedi & Khan (2025)** — *Med Biol Eng Comput* 63(1):15–28 / arXiv 2403.02772  
ST-GCN + supervised contrastive with hard and soft negatives. Trains on IRDS, transfers to KIMORE as regression. Uses UI-PRMD, IRDS, and KIMORE. Nearly identical dataset pipeline to v1. Key difference: *supervised* (uses binary labels during pretraining) and performs *target fine-tuning* (not zero-shot).

**SSL-Rehab — Kourbane, Papadakis, Andries (IMT Atlantique)**  
Foundation model pretrained on 3D skeletons via *masked-motion self-supervision* with progressive masking + LoRA. Fine-tuned on KIMORE and UI-PRMD. Reports improved cross-dataset generalization under limited labels. Most threatening prior work — it is "SSL pretraining → better rehab generalization," differing from v1 mainly in using masked-motion rather than SimCLR/NT-Xent and in fine-tuning on the target dataset.

**Yao, Lei, Zhang, Du, Gao (2023)** — *IEEE TNSRE*  
Multi-task contrastive framework for rehabilitation exercise performance assessment. Claims SOTA.

### 1.4 What Is Confirmed and Still Stands

**Point Cloud Transformer for Rehab Assessment (arXiv 2606.30309, 2026)** — a 2026 SOTA paper using KIMORE + UI-PRMD + IRDS writes:

> *"We did not perform cross-dataset validation because the three datasets use different sensors, joint counts, exercise sets, and scoring conventions… a shared joint mapping and score normalization is left as future work."*

Karlov (2025) and SSL-Rehab both *fine-tune on the target dataset* — they never report honest zero-shot performance. This is the gap. It is confirmed real, confirmed current, and confirmed un-addressed by the field's most recent work.

---

## 2. What Still Stands — The Genuine Gap

The following claim is **fully defensible** and **not pre-empted**:

> *Nobody has evaluated SSL pretraining for rehabilitation scoring in a true zero-shot cross-sensor setting — no target fine-tuning, Stratified-LOSO, degeneracy gates, naive-feature baselines — and compared contrastive vs. masked-motion pretraining under identical conditions.*

Broken down:

**Gap 1:** True zero-shot cross-sensor evaluation  
Prior work hides transfer behind fine-tuning on the target corpus. A model "tested on UI-PRMD" that was *also trained on UI-PRMD* is not a zero-shot test. Every published SSL paper in this space uses target data. Paper 1 (existing benchmark paper) already established the zero-shot evaluation protocol. This paper applies it to SSL-pretrained models.

**Gap 2:** Contrastive vs. masked-motion, head-to-head  
SSL-Rehab uses masked-motion. Karlov uses supervised contrastive. Nobody has run both on the same data with the same downstream evaluation to determine which produces better zero-shot transfer. Running both and comparing is a clean, valuable benchmark contribution.

**Gap 3:** Clinical-validity augmentation taxonomy  
Action-recognition literature (AimCLR, MsMCLR, SkeletonCLR etc.) has ablated augmentations extensively — but for classification accuracy, not for clinical scoring quality. The question "which augmentations preserve clinical meaning for movement scoring?" is different and unanswered. The insight that speed perturbation is valid for pretraining but invalid for fine-tuning (because movement duration is clinically significant for scoring) does not appear anywhere in the literature.

---

## 3. The New Research Framing

### v1 Title (indefensible)
*"Contrastive Skeleton Pretraining for Cross-Dataset Generalization in Rehabilitation Movement Quality Assessment"*

### v2 Title (defensible)
*"Does Self-Supervised Pretraining Rescue Zero-Shot Cross-Sensor Generalization in Rehabilitation Movement Scoring? A Rigorous Benchmark of Contrastive vs. Masked-Motion Pretraining"*

### Why This Framing Is Stronger

v1 positioned the paper as "we did something new (contrastive pretraining)."  
v2 positions it as "we rigorously answered a question the field has been avoiding (does SSL help zero-shot transfer?)."

Benchmark papers asking "does X actually work under rigorous conditions?" are exactly what Paper 1 was. This paper applies the same methodology to the SSL question. It is the natural continuation of the same research program, not a pivot.

The framing also makes the negative result explicitly valuable: whether SSL helps or not, the rigorous answer to this question is publishable because the question has never been answered honestly.

---

## 4. Three Corrected Contributions

### Contribution 1 (rewritten)
**First honest zero-shot cross-sensor evaluation of self-supervised pretraining for rehabilitation movement scoring.**

Prior work (Karlov 2025, SSL-Rehab) either uses supervised signals during pretraining or fine-tunes on the target dataset. No existing paper has tested whether SSL-pretrained representations transfer *without any target data* from Kinect (training domain) to OptiTrack/Vicon (test domain). We evaluate this under the same Stratified-LOSO protocol, degeneracy gates, and naive-feature baselines established in Paper 1. This directly inherits and extends Paper 1's methodological rigor.

### Contribution 2 (new)
**Head-to-head benchmark: contrastive (SimCLR/NT-Xent) vs. masked-motion (MAE-style) pretraining on identical rehabilitation data under identical evaluation.**

SSL-Rehab uses masked-motion. This plan originally proposed contrastive only. Adding both arms turns the comparison itself into the contribution. The field currently has two papers using different SSL paradigms with different datasets and different evaluation protocols — no controlled comparison exists. We run both paradigms on the same pretraining corpus (IRDS), fine-tune on the same target (KIMORE, Stratified-LOSO), and evaluate zero-shot on the same external corpora.

### Contribution 3 (kept, repositioned)
**Clinical-validity augmentation taxonomy for skeleton-based movement scoring.**

Action-recognition SSL literature (AimCLR AAAI 2022, asymmetric augmentation Sensors 2022, MsMCLR, SkeletonGCL ICLR 2023) has extensively ablated augmentations for classification accuracy. We ablate for clinical scoring fidelity — a strictly different objective. We identify which augmentations should be applied during pretraining only vs. during fine-tuning vs. neither, based on whether they preserve the clinical signal (joint angles, spatial coordination, movement range). This is the first augmentation taxonomy grounded in clinical movement semantics rather than classification accuracy.

---

## 5. How This Differs from Every Prior Paper

| Paper | SSL type | Labeled during pretraining? | Zero-shot tested? | Stratified LOSO? | Naive-feature baseline? |
|---|---|---|---|---|---|
| Karlov 2025 | Supervised contrastive | ✓ (binary labels used) | ✗ (target fine-tuning) | ✗ | ✗ |
| SSL-Rehab | Masked-motion | ✗ | ✗ (target fine-tuning) | ✗ | ✗ |
| Yao 2023 | Multi-task contrastive | ✓ | ✗ | ✗ | ✗ |
| Point Cloud Transformer 2026 | None | — | ✗ (explicitly deferred) | ✗ | ✗ |
| **This paper** | **Contrastive + Masked-motion** | **✗ (both self-supervised)** | **✓ (primary evaluation)** | **✓ (inherited from Paper 1)** | **✓ (inherited from Paper 1)** |

The last row is entirely unique. No prior paper fills all five columns.

---

## 6. Corrected Datasets Reference

| Dataset | Subjects | Exercises/Gestures | Reps | Labels | Sensor | Role in this plan |
|---|---|---|---|---|---|---|
| KIMORE | 78 (44 healthy, 34 patients) | 5 | ~380 | Physician score 0–50 (continuous) | Kinect v2 | Training + evaluation under Stratified-LOSO |
| IRDS / IntelliRehabDS | 29 (15 patients, 14 healthy) | 9 | ~2,589 | Binary: correct=1 / incorrect=2 (two annotators) | Kinect One | **Pretraining corpus** (self-supervised, labels not used) + binary zero-shot test |
| REHAB24-6 | 10 | 6 | 1,057 | Binary: correct / incorrect | OptiTrack (precision MoCap) | Zero-shot cross-sensor test (Kinect→OptiTrack domain shift) |
| UI-PRMD | 10 | 10 | 2,000 | Binary: correct / incorrect | Kinect + Vicon | Zero-shot cross-sensor test (Kinect source domain, Vicon reference) |

**Important corrections from v1:**
- IRDS and IntelliRehabDS are the same dataset — do not list separately
- IRDS labels are binary, not absent — but we do NOT use them during pretraining
- No physician-scored external corpus exists — KIMORE is the only one
- The "Phase 5 — IntelliRehabDS integration" from v1 is deleted; IRDS binary labels can be used as-is for zero-shot AUROC, which was already planned

**On finding a genuine fourth corpus:**  
The literature review flags KERAAL and Toronto stroke/TRI sets as candidates that might have continuous clinical scores. This requires a dedicated search. If one is found before Month 4, it can be added and would strengthen Contribution 1 significantly. However, the paper is publishable without it.

---

## 7. Engineering Architecture — Updated

### 7.1 Phase 1 — Skeleton Augmentation Module (unchanged from v1)

Build `augmentations.py` with the six functions: temporal crop, joint masking, speed perturbation, Gaussian joint noise, 3D rotation, limb scaling.

**Clinical-validity classification (updated from v1):**

| Augmentation | Pretraining? | Fine-tuning? | Reason |
|---|---|---|---|
| Temporal crop (80–100%) | ✓ | ✓ | Subsequence selection; doesn't change quality signal |
| Joint masking | ✓ | ✓ | Simulates occlusion; neighboring joints compensate |
| Speed perturbation | ✓ | **✗** | Duration is clinically significant for scoring; destroys quality signal if used during fine-tuning |
| Gaussian joint noise | ✓ | ✓ | Simulates sensor noise; quality signal preserved |
| 3D rotation (±15°) | ✓ | ✓ | Quality is in relative joint angles, not absolute orientation |
| Limb scaling (0.9–1.1×) | ✓ | **✗** | Body size normalization exists in preprocessing; reapplying corrupts calibrated coordinates |

**Ablation experiment:** Run contrastive pretraining with each augmentation removed one at a time. Report which subset gives best downstream KIMORE ρ. Separately, run with speed perturbation enabled during fine-tuning vs. disabled — this directly demonstrates the clinical-validity claim.

### 7.2 Phase 2A — Contrastive Pretraining (SimCLR/NT-Xent)

Identical to v1 Phase 2. Encoder: TCN backbone. Projection head: 2-layer MLP (128→64→32). Loss: NT-Xent at τ=0.07. Batch=128. Epochs=300. Dataset: IRDS (labels ignored).

```python
def nt_xent_loss(z1, z2, temperature=0.07):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, float('-inf'))
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels])
    return F.cross_entropy(sim, labels)
```

### 7.3 Phase 2B — Masked-Motion Pretraining (NEW — added in v2)

**Goal:** Train the same TCN backbone using MAE/BERT-style masked prediction as an alternative SSL paradigm, for direct comparison with the contrastive arm.

**Architecture:**
```
Input: skeleton sequence (T=100, J=25, C=3)
  ↓
[Masking module: randomly zero out mask_ratio=0.30 of joint-frame slots]
  ↓
[TCN Encoder — same architecture as contrastive arm]
  ↓
[Decoder head — 2-layer MLP: 128 → 256 → J*C]
  ↓
[MSE loss on masked positions only]
```

The decoder head is discarded after pretraining, identical to the contrastive projection head.

**Masking strategy — two variants to ablate:**
- **Joint masking:** Randomly select 30% of joints, mask across all timesteps for those joints
- **Temporal masking:** Randomly select 30% of timesteps, mask all joints at those frames

Run both; report which gives better downstream ρ. SSL-Rehab uses progressive masking (ratio increases during training); implement this as a third variant.

```python
class MaskedMotionPretraining(nn.Module):
    def __init__(self, encoder, decoder, mask_ratio=0.30,
                 mask_type='joint'):  # 'joint' or 'temporal'
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type

    def forward(self, x):
        # x: (B, T, J, 3)
        B, T, J, C = x.shape
        x_masked, mask = self.apply_mask(x)
        # Flatten joint+coord: (B, T, J*C) for TCN input
        x_flat = x_masked.view(B, T, J * C)
        repr = self.encoder(x_flat)         # (B, 128)
        pred = self.decoder(repr)           # (B, T*J*C)
        pred = pred.view(B, T, J, C)
        # Loss only on masked positions
        loss = F.mse_loss(pred[mask], x[mask])
        return loss

    def apply_mask(self, x):
        B, T, J, C = x.shape
        x_masked = x.clone()
        mask = torch.zeros(B, T, J, C, dtype=torch.bool)
        if self.mask_type == 'joint':
            n_mask = int(J * self.mask_ratio)
            for b in range(B):
                joint_idx = torch.randperm(J)[:n_mask]
                x_masked[b, :, joint_idx, :] = 0.0
                mask[b, :, joint_idx, :] = True
        elif self.mask_type == 'temporal':
            n_mask = int(T * self.mask_ratio)
            for b in range(B):
                t_idx = torch.randperm(T)[:n_mask]
                x_masked[b, t_idx, :, :] = 0.0
                mask[b, t_idx, :, :] = True
        return x_masked, mask
```

**Hyperparameters (masked-motion):**
```yaml
mask_ratio: 0.30
mask_type: joint  # ablate: temporal, progressive
learning_rate: 1e-3  # higher than contrastive — reconstruction is easier signal
optimizer: AdamW (weight_decay=0.05)
epochs: 300
batch_size: 128
scheduler: cosine annealing
```

### 7.4 Phase 3 — Fine-Tuning on KIMORE (Stratified-LOSO)

Run three experimental conditions, all under identical 78-fold Stratified-LOSO:

| Condition | Initialization | Head trained | Expected ρ |
|---|---|---|---|
| A — Scratch (baseline) | Random | Full fine-tune | 0.549 (existing result) |
| B — Contrastive pretrained | IRDS SimCLR checkpoint | Linear probe only | 0.48–0.52 |
| C — Contrastive pretrained | IRDS SimCLR checkpoint | Full fine-tune | 0.560–0.580 |
| D — Masked-motion pretrained | IRDS MAE checkpoint | Linear probe only | 0.48–0.53 |
| E — Masked-motion pretrained | IRDS MAE checkpoint | Full fine-tune | 0.555–0.575 |

The comparison C vs. E is the head-to-head benchmark (Contribution 2). The comparison B vs. D is the linear probe test of raw representation quality.

Statistical analysis: same 20-seed bootstrap + Holm-Bonferroni as Paper 1. Add a paired Wilcoxon test between C and E specifically (the primary comparison).

### 7.5 Phase 4 — Zero-Shot Cross-Sensor Evaluation

Apply the best checkpoint from each pretraining arm (zero fine-tuning on target) to REHAB24-6, UI-PRMD, and IRDS (using its binary labels now).

For each external corpus, compute:
- AUROC against binary correct/incorrect labels (primary metric for external validity)
- pred_SD (degeneracy check — reject if < 0.10)
- Naive-feature baseline AUROC (total path length + mean speed) for comparison

The key question: does either SSL paradigm produce AUROC significantly above the naive baseline and the scratch TCN zero-shot baseline (0.58 on REHAB24-6, 0.53 on UI-PRMD from Paper 1)?

**IRDS zero-shot evaluation (new — this was not done in Paper 1):**  
IRDS has binary labels. We can now compute AUROC on IRDS zero-shot — the model was pretrained on IRDS without labels, so evaluating it on IRDS labels is a valid test of whether the pretraining learned anything clinically useful. This is a genuine in-domain test.

### 7.6 Augmentation Ablation Experiment

Run contrastive pretraining with each augmentation removed one at a time (6 ablation runs), then:
- Report downstream KIMORE ρ for each
- Report IRDS zero-shot AUROC for each
- Compare speed perturbation ON vs. OFF during fine-tuning specifically

This produces Table 3 in the paper: "Clinical Validity of Augmentations — Effect on Scoring Performance."

---

## 8. Expected Results

### 8.1 Result Grid

| Experiment | Metric | Expected range | Notes |
|---|---|---|---|
| TCN scratch (baseline) | KIMORE ρ | 0.549 [0.518, 0.580] | Reproduced from Paper 1 |
| Contrastive linear probe | KIMORE ρ | 0.48–0.52 | Tests raw representation quality |
| Contrastive full fine-tune | KIMORE ρ | 0.560–0.582 | Key result — vs. scratch |
| Masked-motion linear probe | KIMORE ρ | 0.48–0.53 | |
| Masked-motion full fine-tune | KIMORE ρ | 0.555–0.578 | Key result — vs. scratch and vs. contrastive |
| Scratch TCN (zero-shot) | REHAB24-6 AUROC | 0.58 | From Paper 1 |
| Contrastive (zero-shot) | REHAB24-6 AUROC | 0.60–0.70 | Main question |
| Masked-motion (zero-shot) | REHAB24-6 AUROC | 0.58–0.68 | Main question |
| Scratch TCN (zero-shot) | UI-PRMD AUROC | 0.53 | From Paper 1 |
| Contrastive (zero-shot) | UI-PRMD AUROC | 0.54–0.63 | |
| Masked-motion (zero-shot) | UI-PRMD AUROC | 0.54–0.62 | |
| Contrastive (zero-shot) | IRDS AUROC | unknown | New — first time tested |
| Masked-motion (zero-shot) | IRDS AUROC | unknown | New — first time tested |

### 8.2 Interpretation of All Outcomes

| SSL improves KIMORE ρ? | SSL improves zero-shot AUROC? | Finding | Publishability |
|---|---|---|---|
| Yes | Yes | SSL rescues both fine-tuned scoring and zero-shot transfer | Strong Q1 |
| Yes | No | SSL helps within-domain generalization but not cross-sensor transfer — sensor domain shift is a fundamental barrier | Q1 (important bound) |
| No | Yes | SSL improves transfer without improving fine-tuned accuracy — SSL learns transferable features, not scoring-specific features | Q1 (surprising finding) |
| No | No | SSL does not rescue zero-shot cross-sensor transfer — the barrier is representational mismatch at the sensor level, not representation diversity | Q1–Q2 (definitive negative with strong prior art to compare against) |

All four outcomes are publishable because all four are *answers to a question the field has never asked rigorously*.

### 8.3 The Contrastive vs. Masked-Motion Comparison

This comparison is the contribution regardless of absolute performance. If contrastive > masked-motion: "SimCLR-style augmentation-invariance is more useful than reconstruction objectives for clinical scoring." If masked-motion > contrastive: "SSL-Rehab's choice of objective is validated under honest zero-shot conditions." If equal: "Both paradigms converge to the same representations — the pretraining objective matters less than the domain gap at inference time." All three are publishable conclusions.

---

## 9. Compute Budget

Hardware: RTX 5070 12GB VRAM

| Task | Estimated time | VRAM |
|---|---|---|
| Augmentation ablation (6 configs × 100 epochs) | 12 hours | ~4GB |
| Contrastive pretraining — full run (300 epochs) | 4–6 hours | ~5GB |
| Masked-motion pretraining — full run (300 epochs) | 5–8 hours | ~6GB (decoder adds memory) |
| Masked-motion ablation (joint vs. temporal vs. progressive) | 15 hours | ~6GB |
| KIMORE LOSO — all 5 conditions (78 folds × 5) | ~200 hours | ~6GB | 
| Zero-shot inference (all 3 corpora × 2 SSL arms) | < 2 hours | ~3GB |
| **Total** | **~240 hours** | — |

**Note on KIMORE LOSO compute:** 200 hours = ~8–9 days of overnight running. Stagger across 3 weeks, 8 hours per night. Use a shell loop:
```bash
for fold in $(seq 0 77); do
    python train_loso.py --fold $fold --init contrastive --mode finetune
    python train_loso.py --fold $fold --init masked_motion --mode finetune
    python train_loso.py --fold $fold --init scratch --mode finetune
done
```

---

## 10. Publication Strategy

### 10.1 Primary Target (unchanged from v1)

**Computers in Biology and Medicine**  
Estimated acceptance probability with this v2 framing: **65–75%**  
Why this framing is a better fit: benchmark papers answering "does method X actually work under rigorous conditions?" are this journal's bread and butter. The contrastive vs. masked-motion comparison adds the algorithmic dimension that v1 lacked.

### 10.2 Secondary Target

**Journal of NeuroEngineering and Rehabilitation**  
Estimated acceptance: 50–60% (65% with clinical co-author)  
The zero-shot cross-sensor angle is clinically framed: "can we deploy on a new sensor without collecting new labeled data?"

### 10.3 Positioning Statement for Cover Letter

> *"Existing SSL-based rehabilitation scoring systems report improved generalization, but uniformly achieve this through target-dataset fine-tuning — never under zero-shot cross-sensor conditions. We present the first rigorous zero-shot evaluation of both contrastive and masked-motion self-supervised pretraining for rehabilitation movement scoring, using the Stratified-LOSO protocol and multi-corpus external validation established in our companion paper [Paper 1]. We find that [result], which [supports/refutes] the hypothesis that SSL pretraining rescues the cross-sensor transfer failure documented in Paper 1."*

### 10.4 What Differentiates This from Karlov 2025 (the closest competitor)

Reviewers will ask why this is not redundant with Karlov 2025. The answer is precise:

| | Karlov 2025 | This paper |
|---|---|---|
| SSL type | Supervised contrastive (uses binary labels) | Self-supervised (labels not used during pretraining) |
| Transfer protocol | Fine-tunes on target dataset | Zero-shot — no target data at all |
| Evaluation rigor | Standard KFold | Stratified-LOSO + bootstrap CIs + multiple comparison correction |
| Degeneracy gate | Not reported | Explicit pred_SD threshold |
| Naive-feature baseline | Not reported | Explicit (total path length + mean speed) |
| Sensor domain shift | Evaluated in-domain | Explicitly cross-sensor (Kinect→OptiTrack, Kinect→Vicon) |

---

## 11. Updated 6-Month Timeline

### Month 1 — Implement Both Pretraining Arms

| Week | Task |
|---|---|
| 1 | Implement `augmentations.py` — all 6 functions with unit tests. Classify each by clinical validity. |
| 2 | Implement `simclr_trainer.py` — NT-Xent loss, projection head, training loop, linear probe monitoring |
| 3 | Implement `masked_motion_trainer.py` — masking module (joint and temporal), decoder, MSE loss on masked positions |
| 4 | Run short 100-epoch pilot of both arms. Verify loss curves are healthy. Check linear probe trends. |

### Month 2 — Run Ablations and Full Pretraining

| Week | Task |
|---|---|
| 5 | Augmentation ablation: 6 configs × 100 epochs contrastive. Rank augmentations. |
| 6 | Masking strategy ablation: joint vs. temporal vs. progressive × 100 epochs masked-motion. |
| 7 | Full contrastive pretraining (300 epochs, best augmentation config). Save encoder. |
| 8 | Full masked-motion pretraining (300 epochs, best masking strategy). Save encoder. |

### Month 3 — Full KIMORE Evaluation

| Week | Task |
|---|---|
| 9 | Run KIMORE LOSO for all 5 conditions (scratch, contrastive LP, contrastive FT, masked-motion LP, masked-motion FT). Run overnight in batches. |
| 10 | Continue overnight LOSO runs. Collect all 78-fold results. |
| 11 | Statistical analysis: bootstrap CIs for all conditions, Wilcoxon pairwise tests, Holm-Bonferroni correction. |
| 12 | Zero-shot evaluation on REHAB24-6, UI-PRMD, IRDS (binary labels now used). Generate all result tables. |

### Month 4 — Figures, Co-author, Writing

| Week | Task |
|---|---|
| 13 | Generate all figures: training curves, ρ comparison bar charts with CIs, AUROC comparison, augmentation ablation table, SSL paradigm comparison table. |
| 14 | Read the 8 must-cite papers (see Section 12). Draft Related Work section. Contact physiotherapist collaborator. |
| 15 | Write Methods and Experiments sections. |
| 16 | Write Introduction, Discussion, Conclusion. First complete draft. |

### Month 5 — Revision and Submission

| Week | Task |
|---|---|
| 17 | Internal review pass. Send to co-author (physiotherapist) for clinical language review. |
| 18 | Revise. Format to Computers in Biology and Medicine (Elsevier) template. |
| 19 | Write response-to-reviewers style self-critique. Fix any weak sections. |
| 20 | Submit. Release code on GitHub (public repo, MIT license). |

### Month 6 — Buffer and Parallel Tasks

| Task |
|---|
| Respond to reviewer requests for minor revisions (likely) |
| Begin search for physician-scored external corpus (KERAAL, Toronto TRI sets) for potential follow-up |
| Write CSE499A capstone report incorporating both Paper 1 and Paper 2 results |

---

## 12. Must-Read Literature

Read these before writing a single sentence of the manuscript. Papers are listed in reading priority order.

### Tier 1 — Must read completely

**1. Karlov, Abedi & Khan (2025)**  
*"Rehabilitation exercise quality assessment through supervised contrastive learning with hard and soft negatives"*  
*Med Biol Eng Comput* 63(1):15–28 | arXiv 2403.02772  
**Why:** Closest competitor. Read the IRDS→KIMORE transfer section line by line. Understand exactly what they did and didn't do (target fine-tuning, supervised labels, no LOSO). The differentiation table (Section 5) is built from this paper.

**2. SSL-Rehab — Kourbane, Papadakis, Andries (IMT Atlantique)**  
*"Assessment of Physical Rehabilitation Exercises Through Self-Supervised Learning of 3D Skeleton Representations"*  
gitlab-pages.imt-atlantique.fr/ssl-rehab-78a770  
**Why:** Closest SSL pre-emption. Establishes the masked-motion paradigm for rehab. Read the architecture, masking strategy, LoRA fine-tuning, and cross-dataset generalization claims carefully. Every sentence in your Related Work that distinguishes your paper from theirs comes from this reading.

**3. Miron et al. — IntelliRehabDS**  
*Data* 6(5):46, 2021 | Zenodo 4610859  
**Why:** You must verify the label structure yourself. The literature review says binary only — confirm this by reading the Data Availability and File Description sections. Also verify: are 29 subjects or 30? Are 2,589 reps the exact count?

**4. Point Cloud Transformer for Rehab — arXiv 2606.30309 (2026)**  
**Why:** This is the field's own admission that the gap exists. Quote their exact statement about cross-dataset validation being deferred as future work. This is your primary justification for why the gap is both real and current.

### Tier 2 — Read the abstract and methods, cite

**5. Yao, Lei, Zhang, Du, Gao (2023)** — *IEEE TNSRE*  
Multi-task contrastive for rehab AQA. Citable as additional prior contrastive work.

**6. "Towards Universal Skeleton-Based Action Recognition" — arXiv 2604.17013**  
Cross-sensor skeleton harmonization. Read for joint mapping strategies (Kinect-25 → OptiTrack format). Cite for the cross-format transfer methodology.

**7. AimCLR — AAAI 2022**  
Extreme augmentation strategies for skeleton contrastive learning. Cite in the augmentation taxonomy section as the action-recognition prior art your clinical-validity framing goes beyond.

**8. EMBC 2025 — Frame-Level Stroke Rehab (MOMENT foundation model)**  
Pretraining beats LSTM for rehab generalization (AUC 0.73 vs 0.58). Cite as independent evidence that pretraining helps — but also note they fine-tune, which is different from your zero-shot evaluation.

---

## 13. Open Questions

| # | Question | Options | Needed by |
|---|---|---|---|
| 1 | Masked-motion decoder: MLP vs. Transformer decoder? | Start with 2-layer MLP for speed; ablate Transformer if resources allow | Week 3 |
| 2 | SSL-Rehab uses LoRA for fine-tuning — should we implement LoRA as a 6th condition? | Optional: adds one condition, increases comparison scope, but adds complexity. Decide after seeing Month 2 results. | Month 3 |
| 3 | IRDS binary labels: are they from Zenodo or do we need to contact authors? | Check Zenodo 4610859 directly — should be public | Week 3 |
| 4 | Masking ratio: 0.30 or different? | SSL-Rehab uses progressive masking starting at 0.15; MAE uses 0.75. Try 0.30 (balanced). Ablate in Month 2. | Week 3 |
| 5 | Joint mapping REHAB24-6 (OptiTrack) → KIMORE (Kinect 25-joint): which joints to drop/interpolate? | Read "Towards Universal Skeleton-Based Action Recognition" (arXiv 2604.17013) for mapping strategies | Month 1 |
| 6 | Should LSTM also get the SSL pretraining treatment, or only TCN? | Start with TCN only (best Paper 1 performer). LSTM is optional second arm. | Month 2 |
| 7 | Clinical co-author — which institution to contact first? | NSU Health Center → Dhaka Medical College Rehabilitation → BIRDEM Hospital | Month 4 |
| 8 | Fourth physician-scored corpus search — KERAAL, Toronto TRI — how to access? | Requires dedicated literature + data repository search. Not blocking main experiments. | Month 5 |
| 9 | Code release: full training pipeline public? | Yes, strongly recommended. Creates reproducibility foundation and the journal expects it. | Month 5 |

---

## Appendix A — File Structure (Updated)

```
D:/Rehabilation/
├── paper1_benchmark/          # Original benchmark paper (do not modify)
├── paper2_ssl_pretraining/    # This plan's code
│   ├── augmentations.py       # Phase 1 — all 6 augmentations
│   ├── simclr_trainer.py      # Phase 2A — NT-Xent contrastive pretraining
│   ├── masked_motion_trainer.py   # Phase 2B — MAE-style masked pretraining
│   ├── linear_probe.py        # Monitoring during pretraining
│   ├── finetune_loso.py       # Phase 3 — KIMORE fine-tuning all 5 conditions
│   ├── zeroshot_eval.py       # Phase 4 — external corpus AUROC
│   ├── ablation_runner.py     # Runs all augmentation ablation configs
│   └── checkpoints/
│       ├── simclr_best.pt
│       └── masked_motion_best.pt
├── datasets/
│   ├── kimore/
│   ├── irds/                  # = IntelliRehabDS (binary labels included)
│   ├── rehab24_6/
│   └── ui_prmd/
├── results/
│   ├── pretraining_curves/
│   ├── augmentation_ablation/
│   ├── masking_ablation/
│   ├── kimore_loso_all_conditions/
│   └── zeroshot_external/
└── paper/
    ├── manuscript.tex
    ├── figures/
    └── supplementary/
```

---

## Appendix B — Key Claims and Their Evidence

This table should be kept updated as experiments run, to track every claim in the paper:

| Claim | Evidence source | Status |
|---|---|---|
| Prior papers use non-stratified CV, inflating ρ by +0.026 | Paper 1 experiments | ✓ Established |
| Best honest KIMORE ρ = 0.549 [TCN, Stratified-LOSO] | Paper 1 experiments | ✓ Established |
| Zero-shot AUROC ≈ 0.58 / 0.53 (scratch TCN) | Paper 1 experiments | ✓ Established |
| Karlov 2025 uses supervised contrastive + target fine-tuning | Karlov arXiv 2403.02772 | ✓ Literature |
| SSL-Rehab uses masked-motion + target fine-tuning | SSL-Rehab paper | ✓ Literature |
| Cross-dataset validation explicitly deferred in 2026 SOTA | arXiv 2606.30309 | ✓ Literature |
| IRDS = IntelliRehabDS, binary labels only | Miron et al. 2021 + Zenodo 4610859 | ✓ Verified |
| Contrastive pretraining ρ (fine-tune) | Month 3 experiments | ⬜ Pending |
| Masked-motion pretraining ρ (fine-tune) | Month 3 experiments | ⬜ Pending |
| Zero-shot AUROC (contrastive) on REHAB24-6 | Month 3 experiments | ⬜ Pending |
| Zero-shot AUROC (masked-motion) on REHAB24-6 | Month 3 experiments | ⬜ Pending |
| Speed perturbation harms fine-tuning performance | Month 2 ablation | ⬜ Pending |

---

*End of RESEARCH_PLAN_2.md*  
*Version 2.0 — July 2, 2026*  
*Supersedes: RESEARCH_PLAN_1.md*  
*Next review: End of Month 2 (after both pretraining arms complete)*
