# Literature Review: Cross-Sensor Point Cloud Self-Supervised Learning for Rehabilitation Exercise Assessment

**Research Path:** `research/cross-sensor-pointcloud-ssl`
**Date:** July 2026
**Codebase:** `D:\Rehabilation\src\selfsup\`

---

## 1. Skeleton-Based Rehabilitation Exercise Assessment

### 1.1 Evolution of Assessment Methods

Automated physical rehabilitation exercise assessment (APRE) has evolved from handcrafted feature engineering to deep learning-based approaches. Early methods relied on Euclidean distance, joint angle flexion, and manual thresholding (Da Gama et al.), which lacked generalization across patients and exercises (Sardari et al., *Computers in Biology and Medicine*, 2023).

The introduction of affordable depth sensors—particularly Microsoft Kinect—enabled skeleton-based assessment at scale. The Kinect provides 25-joint 3D positions at 30 Hz, making it suitable for home-based rehabilitation monitoring. Key benchmark datasets emerged:

- **KIMORE** (Kinect v2, 25 joints): 78 subjects (34 patients with motor dysfunction), 5 exercises, physician scores
- **UI-PRMD** (Kinect v2 + Vicon): 10 subjects, 10 exercises, Gaussian mixture model scores
- **IRDS** (Kinect One, 25 joints): Binary correctness labels
- **EHE** (Kinect v2): Elderly home environment, Alzheimer's patients
- **KERAAL** (Kinect V2 + Vicon): Low back pain rehabilitation with error annotations

### 1.2 Graph Convolutional Networks for Skeleton Assessment

Graph Convolutional Networks (GCNs) have become the dominant architecture for skeleton-based assessment due to the natural graph topology of human skeletons (joints as nodes, bones as edges). Yan et al. extended GCNs to Spatio-Temporal Graph Convolutional Networks (ST-GCN), modeling dynamic skeletons across temporal sequences.

**EGCN++** (TPAMI 2024) proposed an ensemble-based fusion strategy (MLE-PO) combining position and orientation features at data and model levels. Validated on UI-PRMD, KIMORE, and EHE, MLE-PO outperformed single-modal methods and achieved higher consistency with clinical evaluations. The framework demonstrated that position-learned joint weights can regularize orientation stream training.

**Deb et al.** proposed a Spatio-Temporal GCN with self-attention for explainable scoring, providing attention maps showing which joints contribute most to assessment decisions—the first attempt at interpretable rehabilitation scoring.

### 1.3 Transformer-Based Approaches

Transformer architectures have been adapted for rehabilitation assessment, leveraging self-attention to capture long-range spatial-temporal dependencies:

- **Point Cloud Transformer for Rehab** (2025): Uses curve-based point cloud aggregation (adapted from CurveNet) with axial self-attention to model joint-level relevance. Achieves SOTA on KIMORE, UI-PRMD, and IRDS with small model size and fast inference.

- **Skeleton-Based Transformer for Error Classification** (Marusic et al., 2025): HyperFormer-inspired model that classifies specific errors in low-back pain exercises, providing per-joint importance scores for patient feedback.

### 1.4 Multi-Modal and Multi-Dataset Approaches

**FineRehab** (CVPR 2024 Workshop) introduced a multi-modality dataset with 16 actions from 50 participants, captured by two Kinect cameras and 17 IMUs. Benchmarking with ST-GCN and UNIK showed that dense network structures (UNIK) learn motion information more effectively (92.63% accuracy).

**Deep Learning Benchmark for Skeleton-Based Rehab** (Ismail-Fawaz et al., 2025) aggregated existing datasets into "Rehab-Pile" and proposed a unified benchmarking framework, addressing the lack of standardized evaluation protocols.

---

## 2. Self-Supervised Learning for Skeleton and Point Clouds

### 2.1 Contrastive Learning

Contrastive learning has been successfully adapted for skeleton-based representation learning. The core idea is to learn invariant representations by pulling positive pairs together and pushing negative pairs apart in embedding space.

**SimCLR-style methods** use NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss with two augmented views of the same skeleton sequence as positive pairs. Key improvements include:

- **HICLR** (Zhang et al., 2022): Hierarchical inconsistent contrastive learning with growing augmentations
- **Actionlet-Dependent Contrastive Learning** (Lin et al., CVPR 2023): Actionlet-level contrast for unsupervised skeleton recognition
- **Part-Aware Contrastive Learning** (Hua et al., 2023): Part-level contrastive objectives
- **Focalized Contrastive View-Invariant Learning** (Men et al., 2023): View-invariant representations

**RCMCL** (Akgül et al., 2025) proposed Robust Cross-Modal Contrastive Learning for RGB-D, skeleton, and point cloud fusion. Key innovations include:
- Cross-Modal Consistency Loss (L_CM) for feature alignment
- Intra-Modal Self-Distillation Loss (L_IM) for representation quality
- Degradation Simulation Loss (L_deg) with Adaptive Modality Gating (AMG)
- Only 11.5% degradation under critical dual modality dropout (vs. significantly higher for supervised baselines)

### 2.2 Masked Modeling

Masked modeling adapts BERT/MAE-style pre-training to point clouds and skeletons.

**Point-BERT** (Yu et al., CVPR 2022) introduced BERT-style pre-training for 3D point clouds:
- Point Tokenization via dVAE (discrete Variational AutoEncoder) to generate discrete point tokens
- Masked Point Modeling (MPM) task: predict original point tokens at masked locations
- Block-wise masking strategy (25-45% ratio)
- Achieves 93.8% on ModelNet40, 83.1% on ScanObjectNN (hardest)

**Point-MAE** (Pang et al., ECCV 2022) proposed a cleaner masked autoencoder approach:
- Irregular point patch division with high masking ratio (60-80%)
- Asymmetric encoder-decoder: encoder processes only unmasked patches
- Shifting mask tokens to decoder prevents location information leakage
- Reconstruction target is raw coordinates (noise-free) vs. Point-BERT's dVAE tokens
- 85.18% on ScanObjectNN, 94.04% on ModelNet40

**Point-M2AE** (Zhang et al., NeurIPS 2022) extended to multi-scale hierarchical architecture:
- Pyramid encoder progressively models spatial geometries
- Multi-scale masking strategy with consistent visible regions across scales
- Local spatial self-attention during fine-tuning
- 86.43% on ScanObjectNN (+3.36% over Point-MAE)

**SkeletonMAE** (Wu et al., 2022; Yan et al., ICCV 2023) adapted masked autoencoders specifically for skeleton sequences:
- Spatial-temporal masked autoencoding with graph-based architecture
- Motion-aware reconstruction targets

### 2.3 Combined Contrastive + Masked Approaches

Recent works recognize that contrastive learning and masked modeling capture complementary information:

**CML (Contrastive Mask Learning)** (Sensors, 2025):
- Integrates mask learning into multi-level contrastive learning
- Instance-level contrast (Siamese BYOL structure) + cluster-level contrast
- Masked skeleton provides novel contrast views; reconstruction serves as extra positives
- Gradient flow from student branch guides mask learning with high-level semantics
- 86.8% (X-Sub), 91.1% (X-View) on NTU-60

**PCM3++ (Prompted Contrast with Masked Motion Modeling)** (Zhang et al., IJCV 2026):
- Comprehensive survey + novel method integrating contrastive and MSM
- Joint, clip, and sequence level representations
- Clip-level contrastive learning for short-term motion modeling
- Post-distillation for compact representation space
- Evaluated on recognition, retrieval, detection, and few-shot learning

**Comprehensive Survey** (Zhang et al., IJCV 2026):
- First systematic survey of self-supervised skeleton-based action representation learning
- Taxonomy: context-based, generative learning, contrastive learning
- Key finding: most works use single paradigm, limiting generalization
- Proposed method integrating different granularity objectives

### 2.4 Self-Supervised Learning for Rehabilitation

**SSL-Rehab** (Kourbane et al., *Computer Vision and Image Understanding*, 2024):
- Decreasing masked motion modeling: progressive reduction of masking ratios (90% → 75% → 50%)
- Three-stage pretraining on NTU-60 with LoRA (Low-Rank Adaptation)
- GCN-based embedding for spatial dependencies
- Outperforms MAMP on KIMORE and UI-PRMD across all metrics
- Key insight: varying masking ratios capture diverse motion complexities

**Hierarchical Contrastive Representation** (Kuang et al., *IEEE TNSRE*, 2024):
- Multi-view skeletal data (positional + angular joint information)
- Novel contrastive loss for regression tasks
- >30% reduction in MAD on KIMORE and UIPRMD
- Captures both global and local movement characteristics

**Supervised Contrastive Learning** (Karlov et al., *MBEC*, 2024):
- Hard and soft negatives for rehabilitation exercise quality assessment
- Transfer from IRDS (Kinect One) to KIMORE (Kinect v2)
- Achieves 0.744 mean Spearman rho across KIMORE exercises

---

## 3. Cross-Sensor Domain Generalization

### 3.1 The Cross-Sensor Challenge

Different motion capture sensors produce fundamentally different skeleton data due to:
- **Joint count mismatch**: Kinect v2 (25), Kinect One (25), OptiTrack (variable), Vicon (39)
- **Coordinate system differences**: Sensor-specific reference frames
- **Noise profiles**: Consumer-grade (Kinect) vs. research-grade (OptiTrack, Vicon)
- **Temporal characteristics**: Different sampling rates and jitter patterns

The **Vogtareuth Rehab datasets** (2025) demonstrated this explicitly: Pose-ResNet trained on generic pose data achieved 93.90% PCK on Kinect data but dropped to 49.60% on OptiTrack data for complex rehabilitation postures, showing severe domain shift.

### 3.2 Domain Generalization Approaches

**Recover-and-Resample** (NeurIPS 2024):
- Complete action prior: human actions start with low feature diversity (rest poses) and increase
- Two-step stochastic action completion: extrapolate from boundary poses + temporal transforms
- Linear transforms learned via k-means clustering
- +5.7% average accuracy on unseen datasets vs. ERM

**Cross-Modal Knowledge Transfer** (SkeFi, 2026):
- Transfer from data-rich RGB modality to wireless sensors (LiDAR, mmWave)
- Enhanced Temporal Correlation Adaptive Graph Convolution (TC-AGC)
- Frame interactive enhancement for missing/inconsecutive frames
- Dual temporal convolution for multi-scale temporal modeling

**DeSPITE** (Kreutz et al., ICCV 2025):
- Deep Skeleton-Pointcloud-IMU-Text Embedding
- Joint embedding space across four modalities
- Cross-modal matching, retrieval, and temporal moment retrieval
- Effective pre-training for point cloud HAR

### 3.3 Sensor Harmonization

**Canonical Schema Approach** (as implemented in this codebase):
- Map all corpora to a canonical joint schema (Kinect-v2 25-joint order)
- Per-sensor index tables for joint mapping
- `cross_sensor` flag: True when sensor ≠ Kinect (e.g., REHAB24-6 with OptiTrack)
- Centralizes joint mapping for unified encoder consumption

---

## 4. Point Cloud Representations for Rehabilitation

### 4.1 Point Cloud as Skeleton Representation

Skeleton joint positions can be naturally represented as point clouds (T × J × 3), where:
- T = temporal frames
- J = number of joints
- 3 = (x, y, z) coordinates

This representation enables:
- **Geometric operations**: Farthest point sampling, KNN queries
- **Augmentation**: Translation, jitter, rotation in 3D space
- **Point cloud architectures**: PointNet++, DGCNN, CurveNet

### 4.2 Point Cloud Augmentation for SSL

The codebase implements a clinically-informed augmentation taxonomy:

| Augmentation | Pretrain | Finetune | Clinical Justification |
|---|---|---|---|
| `temporal_crop` | ✓ | ✓ | Time warping is clinically valid |
| `joint_mask` | ✓ | ✓ | Joint dropout simulates occlusion |
| `gaussian_noise` | ✓ | ✓ | Sensor noise is inherent |
| `rotation_y` | ✓ | ✓ | Viewpoint variation is safe |
| `speed_perturb` | ✓ | ✗ | Duration carries clinical scoring signal |
| `limb_scale` | ✓ | ✗ | Geometry normalized upstream |

This taxonomy ensures that augmentations used during pretraining do not destroy clinically meaningful information needed for downstream assessment.

---

## 5. Gap Analysis and Research Positioning

### 5.1 Identified Gaps

1. **Single Paradigm Limitation**: Most SSL works use either contrastive OR masked modeling, not systematic comparison under cross-sensor constraints (Zhang et al., IJCV 2026 survey confirms this)

2. **Cross-Sensor SSL for Rehab**: No existing work systematically evaluates contrastive vs. masked SSL with strict cross-sensor evaluation (Kinect → OptiTrack) in rehabilitation

3. **Clinical-Validity Augmentations**: Existing augmentation strategies for skeleton SSL do not explicitly encode clinical validity constraints

4. **Statistical Rigor**: Few works report bootstrap confidence intervals with multiple comparison corrections (Holm-Bonferroni) for SSL comparison

5. **Pool-Based Pretraining**: Multi-dataset SSL with corpus harmonization and pool ablation (irds_only vs. all_corpora) is unexplored

### 5.2 This Work's Contributions

This project addresses these gaps through:

1. **Systematic SSL Comparison**: Head-to-head comparison of contrastive (NT-Xent) vs. masked (motion reconstruction) under identical cross-sensor conditions

2. **Clinical-Validity Augmentation Taxonomy**: Explicit distinction between pretrain-safe and finetune-safe augmentations based on clinical scoring requirements

3. **Strict Cross-Sensor Evaluation**: Pretrain on Kinect corpora → zero-shot on OptiTrack (REHAB24-6), with proper corpus harmonization to canonical 25-joint schema

4. **Five-Condition Experimental Design**: scratch, contrastive_lp, contrastive_ft, masked_lp, masked_ft—enabling fine-grained analysis of SSL benefits

5. **Statistical Rigor**: Bootstrap CIs over folds, Holm-Bonferroni corrections across pairwise Wilcoxon tests, linear probe sanity gate

6. **Pool Ablation**: irds_only vs. all_corpora to study data scaling effects on SSL pretraining

### 5.3 Pipeline Architecture

```
Layer 0 (Data):      harmonize.py → pretrain_pool.py
Layer 1 (Encoder):   models_stgcn.build_encoder + heads.py
Layer 2 (Pretext):   augmentations.py → pretext.py → pretrain.py → linear_probe.py
Layer 3 (Downstream): train_loso.py (--init_ckpt) + validity_eval.py
Layer 4 (Orchestration): config.py → registry.py → run_all.py
```

---

## References

### Rehabilitation Assessment
1. EGCN++: A New Fusion Strategy for Ensemble Learning in Skeleton-Based Rehabilitation Exercise Assessment. *IEEE TPAMI*, 2024.
2. Kourbane, I., Papadakis, P., Andries, M. SSL-Rehab: Assessment of physical rehabilitation exercises through self-supervised learning of 3D skeleton representations. *CVIU*, 2024.
3. Kuang, Z. et al. Hierarchical Contrastive Representation for Accurate Evaluation of Rehabilitation Exercises via Multi-View Skeletal Representations. *IEEE TNSRE*, 2024.
4. Karlov, M. et al. Rehabilitation exercise quality assessment through supervised contrastive learning with hard and soft negatives. *MBEC*, 2024.
5. Li, J. et al. FineRehab: A Multi-modality and Multi-task Dataset for Rehabilitation Analysis. *CVPR Workshop*, 2024.
6. Ismail-Fawaz, A. et al. Deep Learning for Skeleton Based Human Motion Rehabilitation Assessment: A Benchmark. *arXiv*, 2025.
7. Marusic, A. et al. Skeleton-Based Transformer for Classification of Errors and Better Feedback in Low Back Pain Physical Rehabilitation Exercises. *arXiv*, 2025.
8. A Point Cloud Transformer for Remote Monitoring and Automated Assessment of Physical Rehabilitation Exercises. *arXiv*, 2025.
9. Sardari, S. et al. Artificial Intelligence for skeleton-based physical rehabilitation action evaluation: A systematic review. *Computers in Biology and Medicine*, 2023.

### Self-Supervised Learning
10. Zhang, J. et al. Self-Supervised Skeleton-Based Action Representation Learning: A Benchmark and Beyond. *IJCV*, 2026.
11. Zhang, J. et al. Prompted Contrast with Masked Motion Modeling: Towards Versatile 3D Action Representation Learning. *ACM MM*, 2023.
12. CML: Contrastive Mask Learning for Self-Supervised 3D Skeleton-Based Action Recognition. *Sensors*, 2025.
13. Lin, L. et al. Actionlet-Dependent Contrastive Learning for Unsupervised Skeleton-Based Action Recognition. *CVPR*, 2023.
14. Wu, W. et al. SkeletonMAE: Spatial-Temporal Masked Autoencoders for Self-Supervised Skeleton Action Recognition. *arXiv*, 2022.
15. Yan, H. et al. SkeletonMAE: Graph-Based Masked Autoencoder for Skeleton Sequence Pre-Training. *ICCV*, 2023.

### Point Cloud SSL
16. Yu, X. et al. Point-BERT: Pre-Training 3D Point Cloud Transformers with Masked Point Modeling. *CVPR*, 2022.
17. Pang, Y. et al. Masked Autoencoders for Point Cloud Self-supervised Learning. *ECCV*, 2022.
18. Zhang, R. et al. Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training. *NeurIPS*, 2022.

### Cross-Sensor and Domain Generalization
19. Recovering Complete Actions for Cross-dataset Skeleton Action Recognition. *NeurIPS*, 2024.
20. Kreutz, T. et al. DeSPITE: Exploring Contrastive Deep Skeleton-Pointcloud-IMU-Text Embeddings for Advanced Point Cloud Human Activity Understanding. *ICCV*, 2025.
21. Akgül, H. et al. RCMCL: A Unified Contrastive Learning Framework for Robust Multi-Modal Action Understanding. *arXiv*, 2025.
22. SkeFi: Cross-Modal Knowledge Transfer for Wireless Skeleton-Based Action Recognition. 2026.
23. Vogtareuth Rehab Depth Datasets: Benchmark for Marker-less Posture Estimation in Rehabilitation. 2025.
24. Enhanced Human Skeleton Tracking for Improved Joint Position and Depth Accuracy in Rehabilitation Exercises. *Applied Sciences*, 2025.

### Point Cloud for Rehabilitation
25. PoinTS: mmWave Radar-based PointNet and Transformer Network for Human Skeleton Estimation in Privacy-Preserving Rehabilitation. *IEEE EUROCON*, 2025.
