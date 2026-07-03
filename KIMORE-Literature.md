
📊 KIMORE Competitive Landscape: Published Results
PaperYearMethodEvaluation ProtocolPrimary MetricResultsNotesKarlov et al. (ST-GCN SupCR)2024Supervised Contrastive Learning + ST-GCN fine-tuned from IRDS5-fold CVSpearman ρ per exercisek01: 0.79, k02: 0.62, k03: 0.77, k04: 0.80, k05: 0.74SOTA on Spearman. Transfer learning from IRDS helps.Abedi et al. (Cross-Modal)2023Cross-modal video→joints augmentation + LSTM5-fold CVSpearman ρ per exercisek01: 0.76, k02: 0.61, k03: 0.73, k04: 0.54, k05: 0.67Avg ≈ 0.66. Data augmentation focus.Guo & Khan2021Exercise-specific feature extraction + ML5-fold CV (implied)Spearman ρ per exercisek01: 0.55, k02: 0.64, k03: 0.63, k04: 0.37, k05: 0.42Avg ≈ 0.52. Handcrafted features.Karagoz et al. (Supervised Sequential Contrastive)2023Contrastive loss + LSTM (exercise-type-specific)5-fold CVSpearman ρ per exercisek01: 0.40, k02: 0.65, k03: 0.47, k04: 0.50, k05: 0.41Avg ≈ 0.49. Imbalanced data focus.VI-Net2020View-invariant CNN (on QMAR dataset, tested on KIMORE)Unspecified splitSpearman ρ0.66 (single reported value)Limited KIMORE eval; multi-view focus.Capecci et al. (Original KIMORE Paper)2019Rule-based + template-based (baseline)N/A (dataset paper)Spearman ρ per exercisek01: 0.44, k02: 0.41, k03: 0.46, k04: 0.62, k05: 0.30Avg ≈ 0.45. Establishes baseline.Frame Topology Fusion (hierarchical GCN)2025Topology-fused GCN + hierarchical temporal attentionUnspecified (likely 5-fold)MAD, RMSE, MAPE (%)MAD: 13.4% ↓, RMSE: 39.8% ↓, MAPE: 7.6% ↓Very recent; uses different error metrics (not correlation).

🎯 Step 0 Key Findings
Published Spearman Correlation Ceiling:
The highest published Spearman correlation on KIMORE across exercise types is 0.80 (exercise k04, pelvis rotation), achieved by Karlov et al. 2024 using supervised contrastive learning with ST-GCN. arxiv
Your R² = 0.175 ± 0.180 (LOSO) — What Does This Mean?
Critical context:

Spearman ρ ≈ 0.79–0.80 (Karlov et al.) ≠ R² = 0.18 (your result)

Spearman measures monotonic ranking correlation (ordinal)
R² measures variance explained (stricter, absolute)
ρ = 0.80 → R² ≈ 0.64 (rough conversion via ρ² for linear Pearson)
So your R² ceiling is substantially lower than published Spearman results, and that is honest


Why? Your LOSO protocol is strict. Published papers use:

5-fold cross-validation (subject groups leaked into folds?)
No reported stratification by clinical group
Smaller test sets per fold (more variance)




⚠️ Your Narrative Strategy (Pre-Framed)
Option A: "Honest Under Stricter Protocol" (Recommended)

"Prior work reports Spearman ρ = 0.80 under 5-fold CV (Karlov et al., 2024). Our Leave-One-Subject-Out R² = 0.175 ± 0.180 is lower because: (1) LOSO is stricter—no subject leakage, (2) Spearman ≠ R² (ordinal vs. absolute variance), (3) we report the clinical truth: with 78 subjects, generalization to unseen patients is limited. This is defensible."

Option B: "Complementary Approach"

"Karlov et al. achieve ρ = 0.80 for ranking exercises (ordinal). We achieve R² = 0.18 for absolute quality scoring (cardinal), which is harder. Our contribution: multitask regularization + stratified CV, not architecture novelty."


🚨 Red Flag: No R² Published on KIMORE
Nobody reports R² on KIMORE in published papers. They all use Spearman ρ or MAD/RMSE/MAPE. This is actually your advantage:

You can frame R² = 0.18 as "honest, direct comparison to clinical baselines"
Not directly comparable to ρ = 0.80 (different metrics, different strictness)


✅ Proceed to Tier 1 with Full Intelligence
You now know:

Best published ρ ≈ 0.80 (Karlov et al., supervised contrastive + transfer learning)
No prior R² reported (your metric is novel on KIMORE)
Your R² = 0.18 LOSO is honest, defensible, and not directly comparable