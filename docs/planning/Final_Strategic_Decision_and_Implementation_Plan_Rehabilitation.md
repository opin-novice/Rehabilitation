## **Final Strategic Decision and Implementation Plan** 

Repo implementation -> your plan -> MAMBA/JAMBA/TTT/xLSTM comparison -> one core topic -> ablation strategy -> execution roadmap 

**Final decision:** choose **Dynamic Hypergraph Hierarchical Mamba Network for Automated Assessment of Physical Rehabilitation Exercises** as the main research topic. Use the original GitHub repo as Baseline 0, use your H-ST-Mamba/DHGNN/pretraining plan as the target framework, and use xLSTM, Jamba-style, and TTT as comparative temporal backbones. 

Final strategic decision and implementation plan 

Page 1 

## **1. The core topic we should commit to** 

**Core research title:** Dynamic Hypergraph Hierarchical Mamba Network with Physiology-Informed Contrastive Pretraining for Automated Assessment of Physical Rehabilitation Exercises. 

This is the strongest topic because it is specific enough to implement, broad enough to publish, and directly evolves from the repo and your uploaded plan. The original repo gives a CurveNet + Transformer baseline. Your uploaded plan argues that the current Transformer-style approach misses dynamic joint importance, anatomical hierarchy, long-sequence efficiency, and pretraining for small rehabilitation datasets. The best decision is to keep these ideas, but make **Mamba** the main temporal backbone rather than trying to make every new architecture the main method. 

## **Why this should be the main goal** 

- G[It has a clean paper story: baseline Transformer rehab -> dynamic hypergraph -> hierarchy -> Mamba temporal] modeling -> physiology-aware pretraining. 

- G[It fits your resource constraint better than a Jamba-style or TTT-first strategy.] 

- G[It allows strong ablations: remove hypergraph, remove hierarchy, replace Mamba with xLSTM/Jamba/TTT, remove] pretraining, remove anatomical loss. 

- G[It is not just a model swap. It connects temporal modeling with rehabilitation biomechanics and remote assessment.] 

**Do not frame the paper as:** Mamba beats Transformer always. Frame it as: **an anatomical dynamic hypergraph + hierarchical Mamba design improves rehabilitation assessment under fair subject-wise evaluation.** 

## **2. What we are comparing** 

**==> picture [436 x 116] intentionally omitted <==**

**----- Start of picture text -----**<br>
Candidate Role in project Use in final paper<br>Repo implementation: CurveNBas e line 0. Rept + Transfo r oduce and repair as much as possible.mer Main baseline to beat.<br>Your uploaded plan: H-ST-Mamba + DHGNN + pretrainingStrategic blueprint. Contains the publishable direction.Converted into a staged, testable implementation.<br>MAMBA / H-ST-Mamba Main model. Primary contribution.<br>xLSTM Stable low-risk temporal baseline. Strong comparison against Mamba.<br>JAMBA-style Hybrid attention + Mamba + MoE comparison. Optional high-capacity ablation if time/GPU allow.<br>TTT High-novelty adaptive temporal comparison. Exploratory extension, not first target.<br>**----- End of picture text -----**<br>


Final strategic decision and implementation plan 

Page 2 

## **3. Final decision matrix** 

The decision scores below are strategic research scores, not final experimental results. They combine expected accuracy potential, 12GB VRAM feasibility, implementation difficulty, novelty, and publication value. Actual accuracy must be measured after training on UI-PRMD/KIMORE under subject-independent splits. 

## **Interpretation** 

- G **[MAMBA wins]**[ because it is the best balance of novelty, feasibility, and direct alignment with your plan.] 

- G **[xLSTM is second]**[ because it is stable and resource-friendly, making it a strong baseline or backup main method.] 

- G **[JAMBA-style is third]**[ because it can test attention + Mamba + expert routing, but it adds complexity and overfitting] risk. 

- G **[TTT is fourth]**[ because it is very novel but difficult to implement rigorously and slow for a first publishable result.] 

- G **[The repo is not the final goal]**[, but it is essential as the first reproducible baseline.] 

## **4. Strategic evolution path** 

The path should be sequential. Do not implement the full proposed model immediately. First reproduce the repo baseline, then build a clean baseline, then add components one by one. The main model becomes M3: DHGNN + limb hierarchy + Bi-Mamba + anatomical loss + optional contrastive pretraining. 

Final strategic decision and implementation plan 

Page 3 

## **5. Implementation blueprint** 

The implementation should use one shared data pipeline and one shared spatial/anatomical pipeline. The temporal backbone should be the replaceable module. This makes the comparison scientifically fair. 

|**Module**|**Implementation choice**|**Why**|
|---|---|---|
|Data format|NPZ arrays: X=(N,T,J,C), y=(N,), groups=(N,)|Easy to reuse for all methods.|
|Preprocessing|root-center, scale-normalize, temporal resample to T=120|root-center, scale-normalize, temporal resample to T=120<br>Reduces subject/camera variation.|
|Spatial encoder|Dynamic soft hypergraph convolution|Captures multi-joint kinetic chains.|
|Hierarchy|Pool joints into torso, arms, legs|Matches rehabilitation biomechanics.|
|Temporal backbone|Bi-Mamba as main; xLSTM/Jamba/TTT as swappable baselines|Bi-Mamba as main; xLSTM/Jamba/TTT as swappable baselines<br>Fair comparison.|
|Head|Temporal attention pool + MLP score regressor|Outputs movement quality score.|
|Loss|MSE + optional anatomical smoothness/bone-length loss|MSE + optional anatomical smoothness/bone-length loss<br>Supports clinical quality regression.|
|Evaluation|MAE, RMSE, R2, Pearson; optional threshold accuracyAvoids relying on only accuracy.|MAE, RMSE, R2, Pearson; optional threshold accuracyAvoids relying on only accuracy.|



## **Minimal model pseudo-code** 

`class FinalRehabModel(nn.Module): def __init__(self, temporal_backbone="mamba"): self.spatial = DynamicHypergraphConv(in_dim=3, hidden_dim=96) self.hierarchy = LimbPool(in_dim=96, out_dim=128) self.temporal = build_temporal_backbone(temporal_backbone, dim=128) self.head = TemporalAttentionHead(dim=256)` 

`def forward(self, x): # x: (B, T, 25, 3) x = normalize_and_augment(x) x = self.spatial(x)       # (B, T, 25, 96) x = self.hierarchy(x)     # (B, T, 5, 128) x = self.temporal(x)      # (B, T, 5, 256) score = self.head(x)      # (B, 1) return score` 

## **6. Ablation study: the experiment plan that proves the paper** 

## **Ablation table to execute** 

|**ID**|**Model / experiment**|**Question answered**|**Decision rule**|
|---|---|---|---|
|A0|Original repo CurveNet + Transformer|What is the starting baseline?|Must reproduce before claiming improvement.|
|A1|Clean Transformer baseline with same data loader|Clean Transformer baseline with same data loader<br>Is improvement from better code/data pipeline?|Use as controlled baseline.|
|A2|DHGNN + Transformer|Does dynamic hypergraph help without Mamba?|Keep DHGNN if it improves MAE/accuracy and robustness.|



Final strategic decision and implementation plan 

Page 4 

|**ID**|**Model / experiment**|**Question answered**|**Decision rule**|/time.<br>lean accuracy.<br>e.<br>p.<br>me VRAM.|
|---|---|---|---|---|
|A3|DHGNN + Bi-Mamba, no hierarchy|Does Mamba help temporal modeling?|Keep Mamba if it improves or reduces VRAM||
|A4|DHGNN + hierarchy + Bi-Mamba|Does anatomical hierarchy help?|Main model if A4 > A3.||
|A5|A4 + anatomical loss|Does physiology constraint stabilize training?|Keep if robustness improves without hurting c||
|A6|A5 + contrastive pretraining|Does NTU/pretraining solve small-data overfitting?|Keep if cross-dataset and 5-fold mean improv||
|A7|Replace Mamba with xLSTM|Is xLSTM a stronger stable baseline?|Report even if xLSTM wins; it becomes backu||
|A8|Replace Mamba with Jamba-style hybr|idDoes attention + SSM + MoE help?|Only keep if it clearly beats Mamba under sa||
|A9|Replace Mamba with TTT|Does adaptive sequence memory improve patient gene|ralization?<br>Use as exploratory extension.||



Final strategic decision and implementation plan 

Page 5 

## **7. Final paper structure** 

|**Section**|**Content**|
|---|---|
|Introduction|Remote rehabilitation needs automated quality assessment; current Transformer/graph models miss dynamic anatomical hierarchy a|
|Related work|Repo Transformer rehab, ST-GCN/STGA-style skeleton models, Mamba/SSM, xLSTM, Jamba-style hybrid, TTT, contrastive pretrain|
|Method|Dynamic hypergraph spatial encoder, limb hierarchy, Bi-Mamba temporal backbone, physiology-informed pretraining.|
|Experiments|UI-PRMD and KIMORE; subject-wise 5-fold CV; cross-dataset generalization; noise robustness.|
|Ablation|A0-A9 table above. This is the proof section.|
|Discussion|Why Mamba helps or does not help; limitations; deployment feasibility.|
|Conclusion|A reproducible dynamic hypergraph hierarchical Mamba framework for rehabilitation assessment.|



## **8. Exact evaluation metrics and plots** 

- G **[Regression metrics:]**[ MAE, RMSE, R2, Pearson correlation for movement quality score.] 

- G **[Classification metrics:]**[ accuracy, balanced accuracy, macro-F1, confusion matrix if labels are correct/incorrect] classes. 

- G **[Robustness:]**[ performance under Gaussian pose noise sigma 0.01, 0.03, 0.05 and frame drop 10%, 20%.] 

- G **[Efficiency:]**[ parameters, training time per epoch, inference FPS, max VRAM used.] 

- G **[Visualizations:]**[ training curve, prediction-vs-ground-truth scatter, temporal attention heatmap, hyperedge/joint-group] importance, failure cases. 

## **Core result tables you must create** 

|**Table/Figure**|**What it should show**|
|---|---|
|Main benchmark table|A0-A9 models against UI-PRMD and KIMORE mean +/- std.|
|Ablation bar chart|Incremental contribution of DHGNN, hierarchy, Mamba, anatomical loss, pretraining.|
|Robustness curve|Performance drop as pose noise increases.|
|Resource scatter plot|VRAM vs MAE or FPS vs MAE.|
|Prediction scatter|Ground truth score vs predicted score.|
|Interpretability figure|Important frames and important joint groups for each exercise.|



## **9. 14-week execution plan** 

|**Weeks**|**Action**|**Output**|
|---|---|---|
|1-2|Repo reproduction and dataset organization|Baseline repo notes, clean folder, processed NPZ files, smoke tests.|
|3-4|Clean controlled Transformer baseline|A1 result, clean training/eval scripts, first metric table.|
|5-6|Build main DHGNN + Bi-Mamba model|A2-A4 results and first paper-worthy model.|



Final strategic decision and implementation plan 

Page 6 

|**Weeks**|**Action**|**Output**|
|---|---|---|
|7-9|Ablations: anatomy loss, xLSTM, Jamba, TTT|A5, A7, A8, A9 comparison.|
|10-11|Contrastive pretraining and fine-tuning|A6 results, cross-dataset transfer tests.|
|12-13|Robustness and efficiency tests|Noise curve, resource table, inference time.|
|14|Paper/report writing|Final manuscript, code README, reproducibility appendix.|



Final strategic decision and implementation plan 

Page 7 

## **10. What to pick now: final decision plan** 

**Pick MAMBA / H-ST-Mamba as the main model.** Build your research around Dynamic Hypergraph Hierarchical Mamba. Do not pick Jamba, TTT, or xLSTM as the main model. Use them as comparison approaches. Do not pick the GitHub repo as the final project; use it as the baseline. 

|**Choice**|**Decision**|**Reason**|
|---|---|---|
|Main paper model|DHGNN + Hierarchy + Bi-Mamba + physiology pretraining<br>Best balance of novelty, feasibility, and resource fit.||
|Primary baseline|GitHub repo CurveNet + Transformer|Directly related to your starting project.|
|Strong backup baseline|xLSTM|Stable and easy to train; good check against Mamba.|
|Optional advanced baseline|Jamba-style|Tests hybrid attention/SSM/MoE, but resource-heavy.|
|Experimental novelty baseline|TTT|Interesting adaptation story, but high-risk for first paper.|



## **Decision rule during experiments** 

G[If A4 beats A1/A0 and is more efficient, the paper is already viable.] 

- G[If A6 pretraining improves cross-dataset or robustness, it becomes the strongest contribution.] 

- G[If xLSTM beats Mamba, keep the topic but rename it around dynamic hypergraph hierarchical temporal modeling and] report Mamba honestly as a comparison. 

- G[If Jamba/TTT fail, still publish the failure as part of the ablation because it proves your main model is not just a] random architecture choice. 

## **11. Implementation commands** 

Use the code package already generated for the temporal approaches. The main commands are: 

```
# 1. Install
unzip Rehabilitation_Temporal_Approaches_Code.zip
cd temporal_approaches_output/code
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 2. Smoke tests on synthetic data
python train.py --approach mamba --synthetic --synthetic-n 16 --epochs 1 --batch-size 4
    --target-len 20
python train.py --approach xlstm --synthetic --synthetic-n 16 --epochs 1 --batch-size 4
    --target-len 20
python train.py --approach jamba --synthetic --synthetic-n 16 --epochs 1 --batch-size 2
    --target-len 20
python train.py --approach ttt --synthetic --synthetic-n 16 --epochs 1 --batch-size 2 --target-
    len 20
```

```
# 3. Real data layout
data/processed/X.npy       # (N,T,25,3)
data/processed/y.npy       # (N,)
data/processed/groups.npy  # subject IDs for subject-wise splitting
```

```
# 4. Train main model
python train.py --approach mamba --data data/processed --epochs 120 --batch-size 8 --target-len
     120
```

```
# 5. Train comparisons
python train.py --approach xlstm --data data/processed --epochs 120 --batch-size 8 --target-len
     120
python train.py --approach jamba --data data/processed --epochs 120 --batch-size 4 --target-len
     120
python train.py --approach ttt --data data/processed --epochs 100 --batch-size 4 --target-len
    100
```

```
# 6. Visualize results
python visualize_results.py --run runs/mamba
python visualize_results.py --run runs/xlstm
python visualize_results.py --run runs/jamba
python visualize_results.py --run runs/ttt
```

## **12. Risks and how to avoid failure** 

|**Risk**|**Why it can ruin the project**|**Mitigation**|
|---|---|---|
|Subject leakage|Same subject in train/test can inflate results.|Use GroupKFold by subject and publish split IDs.|



Final strategic decision and implementation plan 

Page 8 

|**Risk**|**Why it can ruin the project**|**Mitigation**|set tests.|
|---|---|---|---|
|Too many models at onc|eYou will not know what caused success/failure.|Ablate in A0-A9 order.||
|Overclaiming expected a|ccuracy<br>Reviewers reject unsupported claims.|Call numbers hypotheses until real experiments are done.||
|Wrong joint mapping|Architecture becomes meaningless if limbs are wro|ng.<br>Plot skeletons and verify every joint index.||
|mamba-ssm install probl|ems<br>Can block progress early.|Use PyTorch fallback first; install official kernel later.||
|Small dataset overfitting|Accuracy may collapse on new subjects.|Use augmentation, pretraining, early stopping, and cross-data||



## **13. References and evidence used** 

The uploaded plan states the main motivation: Transformer Rehab uses CurveNet + Transformer, identifies fixed graph/anatomical/temporal limitations, proposes H-ST-Mamba, DHGNN, and pretraining, and lists the expected ablation roadmap and expected final results on pages 1-25. The chart on page 9 gives the expected H-ST-Mamba improvements, page 18 gives pretraining improvement expectations, page 21 gives final proposed results, and page 24 lists the ablation study design. 

G[Uploaded plan: Technical Deep Learning Research: Beating Transformer Rehabilitation SOTA, pages 1-25.] 

G[King-Rafat Transformer_Rehabilitation GitHub repository: baseline repo implementation.] 

G[Mamba: Linear-Time Sequence Modeling with Selective State Spaces, arXiv:2312.00752.] 

G[Jamba: A Hybrid Transformer-Mamba Language Model, arXiv:2403.19887.] 

G[Learning to Learn at Test Time / TTT Layers, arXiv:2407.04620.] 

G[xLSTM: Extended Long Short-Term Memory, arXiv:2405.04517.] 

Final strategic decision and implementation plan 

Page 9 

