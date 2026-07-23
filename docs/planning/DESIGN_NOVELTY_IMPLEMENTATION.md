# Design & Implementation - Novelty-Finding Analyses

*Implements the publishable open problems in `NOVELTY_OPPORTUNITIES.md`. Built with Sequential Thinking + Serena over the existing codebase, 2026-06-29.*

---

## 1. Design goals

- **Scalable / extensible.** One registry (`src/novelty/config.py`) is the single source of truth for paths, models, and thresholds. Adding a model or dataset = one edit there; every analysis picks it up automatically.
- **Read-only & no-retraining (Paper-1 bundle).** All analyses consume artifacts the main pipeline already produced under `outputs/` (OOF predictions, `irds_reliability.csv`) and `KIMORE_pooled/`. They never retrain, so they are cheap and reproducible.
- **Graceful degradation.** Every loader returns `None` with a `[SKIP]` note when an artifact is missing; analyses skip rather than crash.
- **Convention-matched.** Reuses existing functions (`protocol_inflation.extract_features` / `eval_protocol`, the OOF schema `subject_id, exercise_id, y_true, y_pred`), and follows the established `argparse main()` + JSON/CSV-to-`outputs/` style.

## 2. Package layout (`src/novelty/`)

| File | Opportunity | What it does |
|---|---|---|
| `config.py` | infra | Path bootstrap (adds `src/` to `sys.path`), model registry (mirrors `sample_level_stats.EXPERIMENTS`), thresholds, effect-size priors. |
| `io_utils.py` | infra | `load_pooled_kimore`, `load_oof_table`/`load_all_oof`, `load_irds_reliability` - robust, graceful. |
| `protocol_decomposition.py` | **N1** | 2x2 factorial {subject-leakage} x {clinical-stratification} inflation decomposition with bootstrap CIs. |
| `periodicity.py` | **N5** | Per-exercise periodicity (spectral concentration) vs TCN-minus-attention rho margin. |
| `power_analysis.py` | **N10** | Monte-Carlo power / design guideline for the dissociation claim. |
| `deployment_rubric.py` | **N7** | Composite accuracy x consistency x integrity deployment screen. |
| `run_all.py` | orchestrator | Subcommands `decomp / periodicity / power / rubric / all`. |

Plus a model edit (**N4**) in `src/models.py`: a continuous `graph_bias_lambda` knob (default `1.0`) on `GraphAwareSpatialEncoder` / `GraphAwareTransformerRegressor` / `build_model`, interpolating the bone-distance structural prior from full (1.0) to off (0.0) for the Paper-2 transferability sweep - backward-compatible with existing checkpoints.

## 3. How to run

```bash
python src/novelty/run_all.py all              # everything
python src/novelty/run_all.py decomp --seeds 20 # N1 only
python src/novelty/run_all.py power  --sims 2000
```
All outputs land in `outputs/novelty/` (`*.json` + `*.csv`).

## 4. Method per analysis

**N1 - 2x2 decomposition.** Identical estimator (RidgeCV on identical features) across four splitters - `KFold` (leak, unstrat), `StratifiedKFold` by clinical group (leak, strat), `GroupKFold` by subject (no-leak, unstrat), `StratifiedGroupKFold` (no-leak, strat = our LOSO). Reports additive **main effects** + interaction, with bootstrap 95% CIs over shuffle seeds.

**N5 - periodicity law.** Per-sample periodicity = max non-DC power-spectral bin / total non-DC power of the per-frame deviation-energy signal (bounded [0,1]); averaged per exercise; correlated across the 5 exercises with rho(TCN)-mean(rho over attention models).

**N10 - power analysis.** Bivariate-normal latent (KIMORE rho, IRDS W) with correlation `rho_target` (estimated from real artifacts when present, else -0.393), measurement noise on W shrinking as `sqrt(n0/n_subjects)`. Grid over n_models x n_subjects -> power = P(significant & correct sign).

**N7 - rubric.** (1) accuracy: mean per-exercise Spearman > 0 AND bootstrap 95% CI lower bound > 0; (2) consistency: Kendall W >= 0.50; (3) integrity: pred_SD >= 0.10 (not variance-collapsed). Deploy-ready = all three.

## 5. Results from the current artifacts (verified runs)

- **N1:** subject-leakage main effect **+0.029 rho** (CI95 [0.007, 0.052] - excludes 0); clinical-stratification main effect **+0.001** (CI95 [-0.022, 0.025] - negligible); interaction +0.025. => **Leakage, not stratification, is the dominant inflation source** for the Ridge estimator. This is the clean decomposition the paper previously lacked. (DL models are expected to show a larger leakage effect, matching the paper's +0.05-0.06.)
- **N7:** only **GraphTransformer (no bias)** (W=0.608) and **Exp E** (W=0.501) pass all three criteria; **GraphTransformer is correctly rejected as degenerate** (pred_SD < 0.10); most models fail Kendall W >= 0.50. Matches the manuscript narrative.
- **N10:** at the observed effect (r=-0.393) **no realistic budget reaches 80% power** (30 models x 50 subjects ~ 0.53). => the dissociation must stay a *case observation*; the guideline quantifies why.
- **N5:** with the current spectral-concentration metric, Spearman(periodicity, margin) = **-0.30, p=0.62 (null)**. Honest negative: the simple k05 anecdote does **not** generalize under this operationalization. Action: refine the periodicity metric (e.g., per-joint trajectory circularity / autocorrelation-peak) before treating N5 as publishable. The pipeline is in place to re-test instantly.

## 6. Mapping to the publishable bundles

- **Paper 1 (ready now):** N1 + N7 + N10 are implemented and produce defensible results directly from existing outputs. N2/N3/N8 were already in `irds_eval.py` / `verify_irds_labels.py`; this package orchestrates them.
- **Paper 2 (follow-up):** N4 knob is wired (`graph_bias_lambda`) for the structural-prior sweep; N5 metric needs refinement (pipeline ready).
- **Paper 3:** N9 (human-rater anchor) remains external-data gated.
