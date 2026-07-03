# Novelty Assessment — RESEARCH_PLAN_2.md

**Idea:** "Does self-supervised pretraining rescue zero-shot cross-sensor generalization in rehabilitation movement scoring? A benchmark of contrastive vs. masked-motion pretraining."
**Method:** Tavily (advanced) + Sequential Thinking · **Date:** 2026-07-02

## Verdict at a glance
- **Novelty score: 5.5 / 10** (solid incremental benchmark contribution; not a conceptual breakthrough)
- **Publication chance:** CBM primary **35–50%** (the plan's 65–75% is optimistic); realistic Q2 landing (IEEE Access / Sensors / Biomed. Signal Process. Control) **60–70%**
- **Strongest asset:** the *honest zero-shot cross-sensor* framing + inherited Paper-1 rigor
- **Weakest asset:** the contrastive-vs-masked comparison (saturated elsewhere) and a tiny 2,589-sequence pretraining corpus

---

## 1. Prior Work (closest → farthest)

| Work | Venue/year | Overlap with v2 | Distinction v2 can still claim |
|---|---|---|---|
| **Karlov, Abedi & Khan** | Med Biol Eng Comput 2025 (arXiv 2403.02772) | Contrastive on IRDS → transfer to KIMORE; same 3 datasets | v2 is *self-supervised* (no labels in pretrain) + *zero-shot* (no target fine-tune) + Stratified-LOSO |
| **SSL-Rehab** (Kourbane et al., IMT Atlantique) | preprint | Masked-motion SSL foundation model, KIMORE+UI-PRMD, "better generalization" | SSL-Rehab *fine-tunes on target*; v2 tests zero-shot and adds a contrastive arm |
| **Care-PD** | NeurIPS 2025 | Clinical movement (PD gait) score prediction under within/cross-dataset/leave-one-dataset-out; encoder-vs-handcrafted baseline | Different task (PD gait, SMPL mesh); v2 is rehab exercises + cross-*sensor*. But shows the "rigorous cross-dataset clinical-movement benchmark" idea is already in the air |
| **Calibration-free sEMG** | Sci Rep 2025 | SSL pretrain + domain alignment, cross-dataset rehab, direct transfer | Different modality (sEMG, not skeleton) |
| **SSL for Time Series: Contrastive or Generative?** / **SSL: Generative or Contrastive** | TKDE 2023 + others | *Exactly* the contrastive-vs-masked question | v2's is skeleton-rehab-scoring specific; but the comparison methodology is not new |
| **SkeletonMAE (ICCV 2023), CMAE, CAN, iBOT, PCM3++, Contrastive-Mask-Learning (2025)** | 2022–2025 | Masked vs contrastive (and hybrids) on skeletons; known result: contrastive usually ≥ reconstruction | v2 doesn't propose a hybrid — it only compares, which these already do |
| **Point Cloud Transformer for Rehab** | arXiv 2606.30309, 2026 | Same KIMORE+UI-PRMD+IRDS trio; explicitly defers cross-dataset validation | Confirms v2's gap is real & current |
| **FineRehab** | CVPR-W 2024 | Kinect+IMU rehab dataset **with expert 0–4 quality scores** (completeness/correction/smoothness), 16 actions | ⚠️ Possible counterexample to v2's claim that "no physician-scored 4th corpus exists" — vet it |

---

## 2. Novelty Score Breakdown

| Contribution | Score | Reasoning |
|---|---|---|
| **C1 — Honest zero-shot cross-sensor SSL eval for rehab scoring** | **7/10** | Genuinely unaddressed in rehab-exercise scoring; Karlov & SSL-Rehab both fine-tune on target. Inherits Paper 1's Stratified-LOSO + degeneracy gates + naive baselines. But Care-PD/sEMG show the concept is emerging |
| **C2 — Contrastive vs. masked-motion head-to-head** | **3.5/10** | The comparison is a *saturated* question in general SSL, time series, and skeleton AR (TKDE survey, SkeletonMAE, CMAE). Only the rehab-scoring application is new; reviewers will discount this |
| **C3 — Clinical-validity augmentation taxonomy** | **5/10** | Augmentation ablations are common (AimCLR, MsMCLR); the "preserve clinical scoring signal, not classification accuracy" framing is fresh but thin as a standalone contribution |
| **Composite** | **≈5.5/10** | A rigorous, well-scoped *benchmark/negative-result* paper — valuable but incremental |

---

## 3. Weaknesses (ranked by severity)

1. **Tiny pretraining corpus (highest risk).** IRDS = ~2,589 sequences. SimCLR/MAE typically need 10k–1M samples; skeleton SSL works pretrain on NTU (56k–114k). SSL may learn little useful signal at this scale — this threatens the whole premise. *Mitigation:* pool IRDS+KIMORE+UI-PRMD+REHAB24-6 unlabeled sequences into the pretraining pool, or pretrain on NTU and adapt.
2. **C2 is a solved question.** "Contrastive vs. masked" is answered repeatedly; frame it as a *domain-specific replication*, not a novel comparison, or drop it as a headline contribution.
3. **Cross-sensor framing is imprecise.** UI-PRMD *has a Kinect capture* (22 joints) — using it is not cross-sensor. Joint counts differ across corpora: KIMORE/IRDS = 25, UI-PRMD Kinect = 22 / Vicon = 39, REHAB24-6 OptiTrack = 26. The joint-mapping step is under-specified and error-prone (cf. your REHAB24-6 26→25 mapping fix). Define exactly which capture/joint set is source vs. target for each corpus.
4. **The likely result is negative.** Paper 1 already shows naive kinematics beat trained models cross-corpus (AUROC ~0.74 vs ~0.5). SSL is unlikely to push AUROC above the naive baseline. A negative benchmark is publishable but harder to place — which is why 65–75% at CBM is optimistic.
5. **Metric conflation.** Thresholding a KIMORE-trained continuous regressor against binary external labels for AUROC conflates score-scale mismatch with representation transfer — the exact issue Point Cloud Transformer names when deferring cross-dataset validation. Address explicitly (e.g., rank-based transfer, per-corpus calibration).
6. **Possibly-false premise.** "No physician-scored 4th corpus exists" — FineRehab (CVPR-W 2024) has expert 0–4 quality scores. Verify before asserting; if valid, it strengthens C1 substantially and should be added.
7. **Author/scope realism.** Single undergraduate author, ~240 GPU-hours on one RTX 5070, 5 LOSO conditions × 78 folds. Timeline is tight; a clinical co-author is still only "planned."

---

## 4. Publication Chance (honest)

| Target | Plan's estimate | Realistic estimate | Condition |
|---|---|---|---|
| Computers in Biology and Medicine (Q1, IF~7) | 65–75% | **35–50%** | Needs a *clear* result (positive, or a decisive well-argued negative) + clinical co-author |
| J. NeuroEngineering & Rehabilitation (Q1) | 50–60% | **35–45%** | Needs clinical co-author + clinical framing |
| IEEE Access / Sensors / Biomed. Signal Process. & Control (Q2) | — | **60–70%** | Most realistic home for a rigorous benchmark/negative result |
| Workshop (e.g., CVPR/MICCAI-W) | — | **70–80%** | Fast, low-risk fallback |

**Upside path to ~55% at CBM:** (a) fix the pretraining-scale risk with a larger unlabeled pool; (b) add a genuinely physician-scored external corpus (vet FineRehab); (c) demote C2 to a supporting ablation and lead with C1 + the clinical-validity story; (d) secure the clinical co-author before submission.

---

## 5. Bottom Line
v2 is a **legitimate, well-corrected benchmark study** — the zero-shot cross-sensor question is real and current (confirmed by a 2026 SOTA paper deferring exactly this). But its novelty is *incremental* (the two closest works, Karlov 2025 and SSL-Rehab, already occupy most of the conceptual space), and its headline contrast (contrastive vs. masked) is a solved question elsewhere. Treat it as a rigorous "does SSL actually help under honest conditions?" paper — a strong Q2 / borderline-Q1 contribution, not a guaranteed Q1.
