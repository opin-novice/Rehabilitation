# Literature Review & Novelty Audit — RESEARCH_PLAN_1.md

**Topic:** Contrastive skeleton pretraining for cross-dataset generalization in rehabilitation movement quality assessment
**Method:** Tavily (advanced) + Sequential Thinking
**Date:** 2026-07-02
**Verdict:** The plan's engineering is sound, but **two of its three headline contributions rest on factual errors**, and its central "first" claim is pre-empted. A reframe (below) preserves a genuine, publishable contribution.

---

## 1. Executive Summary

| Plan claim | Status | Evidence |
|---|---|---|
| "First contrastive pretraining framework for skeleton-based rehab quality assessment" | ❌ **False as stated** | Karlov et al. 2024/25 (supervised contrastive, IRDS→KIMORE transfer); Yao et al. 2023 (multi-task contrastive, TNSRE); SSL-Rehab (self-supervised foundation model, KIMORE+UI-PRMD) |
| "Third external labeled corpus (IntelliRehabDS) with physician scores enabling Spearman ρ" | ❌ **False** | IntelliRehabDS has only a **binary correctness label (1/2)** from two annotators — no continuous physician score |
| IRDS and IntelliRehabDS are two different datasets | ❌ **False** | **IRDS *is* IntelliRehabDS** (Miron et al. 2021, Zenodo 4610859). Same 29 subjects, 9 gestures, ~2,589 reps |
| Skeleton-specific augmentation taxonomy | ⚠️ **Partially pre-empted** | Extensive augmentation ablations exist in action-recognition SSL (AimCLR, asymmetric aug, MsMCLR). The *clinical-validity* framing is fresh |
| Zero-shot cross-sensor transfer fails / is under-studied | ✅ **Confirmed gap** | Point Cloud Transformer (2026) explicitly defers cross-dataset validation; Karlov & SSL-Rehab both *fine-tune* on target, never report honest zero-shot |

**Bottom line:** Contributions 1 and 3 as written cannot survive review. Contribution 2 (evaluation rigor) is the real strength. Reframe around *honest zero-shot cross-sensor generalization* and add a masked-motion pretraining arm.

---

## 2. Two Factual Errors That Must Be Fixed Before Anything Else

### 2.1 IRDS *is* IntelliRehabDS — they are the same dataset
Section 6's dataset table lists **"IRDS"** (row: pretraining corpus, 29 subj, 2,589 reps, unlabeled) and **"IntelliRehabDS"** (row: 30 subj, 9 ex, *physician scores*, "new — strongest external validity test") as **separate** datasets. They are one dataset:

- **IntelliRehabDS (IRDS)**, Miron, Sadawi, Grosan, Ismail, Hussain — *Data* 6(5):46, 2021; Zenodo record 4610859.
- 29 subjects (15 patients + 14 healthy controls), 9 gestures, ~2,589 repetitions, Microsoft Kinect One, 25 joints, 30 fps.

Phase 5 ("add IntelliRehabDS") and Contribution 3 therefore **do not add a new dataset** — they re-use the pretraining corpus. This double-counting must be removed.

### 2.2 IntelliRehabDS has no physician scores — only a binary correctness label
Zenodo & the *Data* paper are explicit: each file is `..._CorrectLabel_Position.txt`, where **CorrectLabel = 1 (correct) or 2 (incorrect)**, annotated independently by two annotators. There is **no continuous physician quality score**. Consequences:

- Contribution 3 ("physician-scored external corpus enabling Spearman ρ against doctor grades on unseen data") is **impossible** with this dataset.
- The claim that it gives "the strongest possible external validity test" via ρ vs. physician grades is unsupported.
- Only **KIMORE** among the four corpora has continuous physician scores (0–50). UI-PRMD is binary; REHAB24-6 is binary; IRDS is binary. So the "score the way a doctor would, on unseen data" test has **no valid target dataset** unless you find a genuinely new physician-scored corpus.

> **Action:** Either (a) find a real fourth corpus with continuous clinical scores, or (b) drop Contribution 3 and reposition IRDS honestly as an unlabeled pretraining pool + binary external validity test (which is what it can actually support).

---

## 3. State of the Art — What Already Exists

### 3.1 Contrastive learning for rehab AQA (direct competitors)
- **Karlov, Abedi & Khan — "Rehabilitation exercise quality assessment through supervised contrastive learning with hard and soft negatives,"** *Med Biol Eng Comput* 63(1):15–28, 2025 (arXiv 2403.02772). ST-GCN + **supervised contrastive** with hard/soft negatives; single model across all exercise types; evaluated on **UI-PRMD, IRDS, KIMORE**. **Crucially, they train a contrastive model on IRDS and transfer-learn to KIMORE as regression** — nearly the exact pipeline RESEARCH_PLAN_1 proposes. Difference: theirs is *supervised* contrastive and uses target fine-tuning.
- **Yao, Lei, Zhang, Du, Gao — "A Contrastive Learning Network for Performance Metric and Assessment of Physical Rehabilitation Exercises,"** *IEEE TNSRE* 2023. Multi-task contrastive framework, claims SOTA.
- **Karagoz et al.** — supervised contrastive LSTM, exercise-specific, KIMORE.
- **Abedi, Malmirian & Khan** — cross-modal video→body-joints augmentation for rehab AQA (ECML/PKDD workshops 2023).

### 3.2 Self-supervised pretraining for rehab (closest pre-emption of the core idea)
- **SSL-Rehab — Kourbane, Papadakis, Andries (IMT Atlantique), "Assessment of Physical Rehabilitation Exercises Through Self-Supervised Learning of 3D Skeleton Representations."** Foundation model pretrained on 3D skeletons via **masked-motion self-supervision** with progressive masking + **LoRA**; fine-tuned on **KIMORE and UI-PRMD**; reports SOTA and **improved cross-dataset generalization under limited labels**. This is the single most threatening prior work — it is "self-supervised pretraining → better rehab generalization," differing from the plan mainly in using masked-motion rather than contrastive/NT-Xent.
- **Du, Graham, Depp, Nguyen (EMBC 2021)** — GCN with self-supervised regularization for rehab assessment.
- **Frame-Level Real-Time Assessment of Stroke Rehab (EMBC 2025)** — uses the **MOMENT** time-series foundation model, fine-tuned under LOSO; pretrained foundation model beats LSTM (AUC 0.73 vs 0.58) and generalizes better to new patients. Independent evidence that pretraining helps generalization — cite as support, but also as prior art.

### 3.3 Skeleton contrastive SSL (the augmentation-taxonomy field)
Augmentation design for skeleton contrastive learning is a mature sub-field, almost all in **action recognition**: SkeletonCLR, **AimCLR** (extreme augmentations, AAAI 2022), asymmetric augmentation (Sensors 2022), MsMCLR (8-augmentation strong branch), SkeletonBYOL, ActCLR (CVPR 2023), SkeletonGCL (ICLR 2023), PCM3++ (contrastive + masked motion). Your six augmentations (crop, joint mask, speed, noise, rotation, limb scale) overlap heavily with these. **The novel angle is not the augmentations themselves but the clinical-validity question** — which augmentations preserve vs. destroy clinically meaningful signal (your own note that speed perturbation is invalid for scoring is exactly the right instinct).

### 3.4 Cross-sensor / heterogeneous skeleton transfer
- **"Towards Universal Skeleton-Based Action Recognition"** (arXiv 2604.17013) — handles heterogeneous skeletons (Kinect-25 vs MoCap-22 vs 2D-17) and reports strong cross-format transfer. Directly relevant to your Kinect↔OptiTrack/Vicon joint-mapping problem; read for the input-harmonization strategy.
- **CD-SEAFNet** (2025), kernelized 3D skeleton domain adaptation (BMVC 2018) — cross-domain skeleton AR baselines.

### 3.5 Independent confirmation of *your* gap
- **A Point Cloud Transformer for Rehab Assessment** (arXiv 2606.30309, 2026) uses KIMORE + UI-PRMD + IRDS and states plainly: *"we did not perform cross-dataset validation because the three datasets use different sensors, joint counts, exercise sets, and scoring conventions… a shared joint mapping and score normalization is left as future work."* This is a 2026 SOTA paper conceding exactly the problem your plan targets — strong justification that the gap is real and current.

---

## 4. Research Gaps That Survive the Literature

1. **True zero-shot cross-sensor generalization (no target fine-tuning).** Karlov and SSL-Rehab both *fine-tune / transfer-learn on the target dataset*. Nobody has reported an honest **zero-shot** Kinect→OptiTrack (REHAB24-6) / Kinect→Vicon (UI-PRMD) evaluation with Stratified-LOSO, degeneracy gates, and naive-feature baselines. This is the defensible core, and it inherits Paper 1's rigor.
2. **Contrastive vs. masked-motion pretraining, head-to-head, for rehab transfer.** SSL-Rehab establishes masked-motion; the field lacks a controlled comparison against contrastive/NT-Xent on identical rehab data + protocol. Run both arms — this converts "we got scooped by SSL-Rehab" into "we benchmark the two SSL paradigms fairly."
3. **Clinical-validity augmentation taxonomy.** Which augmentations preserve clinical meaning for *scoring* (not classification)? Genuinely under-explored; action-recognition ablations optimize accuracy, not clinical fidelity.
4. **Definitive negative result.** The field keeps deferring cross-dataset validation or hiding transfer behind fine-tuning. A rigorous "SSL pretraining does *not* rescue zero-shot cross-sensor scoring" (if that is what you find) is a real, citable contribution — and every outcome in your 2×2 grid remains publishable.

---

## 5. Recommended Reframe

**Old framing (indefensible):** "First contrastive pretraining framework + first physician-scored external corpus."

**New framing (defensible):**
> *"Does self-supervised pretraining rescue zero-shot cross-sensor generalization in rehabilitation movement scoring? A rigorous benchmark of contrastive vs. masked-motion pretraining."*

Concrete edits to the plan:
- **Delete** the duplicate IntelliRehabDS row in Section 6; merge into the single IRDS entry; relabel IRDS as "unlabeled pretraining pool + binary external validity."
- **Delete Contribution 3** (physician-scored external corpus) or replace it by sourcing a genuinely new corpus with continuous clinical scores (candidates to vet: KERAAL, Toronto stroke/TRI sets, or a new KIMORE-like release — none confirmed here; requires a dedicated search).
- **Rewrite Contribution 1** to "first *honest zero-shot cross-sensor* evaluation of SSL pretraining for rehab scoring, comparing contrastive and masked-motion paradigms," explicitly citing and differentiating from Karlov 2025 and SSL-Rehab.
- **Add a masked-motion pretraining arm** (SSL-Rehab-style) alongside the contrastive arm so the comparison is the contribution.
- **Keep** Contribution 2 (augmentation taxonomy) but reposition it as *clinical-validity* ablation, not a general augmentation study.
- **Cite Point Cloud Transformer (2606.30309)** as the field's own admission that the gap exists.

---

## 6. Must-Read / Must-Cite Before Writing

1. Karlov, Abedi, Khan (2025), *Med Biol Eng Comput* — arXiv 2403.02772 (**closest competitor; read the IRDS→KIMORE transfer section in full**)
2. SSL-Rehab, Kourbane/Papadakis/Andries — ssl-rehab-78a770.gitlab-pages.imt-atlantique.fr (**closest SSL pre-emption**)
3. Yao et al., *IEEE TNSRE* 2023 — contrastive rehab AQA
4. Miron et al., IntelliRehabDS, *Data* 6(5):46, 2021 + Zenodo 4610859 (**verify labels yourself — binary only**)
5. Point Cloud Transformer, arXiv 2606.30309 (2026) — gap confirmation + current SOTA on all 3 datasets
6. "Towards Universal Skeleton-Based Action Recognition," arXiv 2604.17013 — cross-sensor joint harmonization
7. AimCLR (AAAI 2022) + asymmetric augmentation (Sensors 2022) — augmentation-ablation prior art
8. EMBC 2025 stroke-rehab (MOMENT foundation model) — pretraining-helps-generalization evidence
