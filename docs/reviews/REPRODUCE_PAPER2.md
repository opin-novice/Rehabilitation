# Reproducing Paper 2 (SSL Pretraining for Zero-Shot Cross-Sensor Rehab Scoring)

Implements ARCHITECTURE_PAPER2.md / RESEARCH_PLAN_2.md. All Paper-2 code lives in
`src/selfsup/` and reuses the Paper-1 pipeline in `src/` via two non-breaking hooks
(`forward_features` on the backbones; `--init_ckpt` / `--freeze_encoder` on `train_loso.py`).

## 0. Environment
```bash
pip install -r requirements.txt          # numpy, pandas, scikit-learn, scipy, matplotlib, torch
```
Hardware for the full run: one RTX 5070 12 GB (AMP, batch 128, encoder-only checkpoints).

## 1. Fast acceptance test (no datasets required, ~40 s CPU)
```bash
python src/selfsup/run_all.py --smoke
```
Runs the entire DAG on dummy data: folds -> pretrain(contrastive+masked, 2 pools) ->
5 LOSO conditions -> zero-shot(REHAB246/UIPRMD/IRDS) -> stats -> tables/figures.
Writes everything under `outputs/_smoke/` and prints `[SMOKE TEST PASSED]`.
Component self-test of the SSL primitives: `python src/selfsup/selftest.py`.

## 2. Full run on real data
Prerequisite — materialize the canonical corpus caches with the existing loaders:
```bash
python src/prepare_kimore.py && python src/pool_exercises.py     # -> KIMORE_pooled/
python src/load_rehab246.py --build                              # -> outputs/validity/
python src/load_uiprmd_validity.py --build                       # -> outputs/validity_uiprmd/
python src/irds_eval.py --build                                  # -> IRDS cached sequences
```
Then run the pipeline (primary = clean zero-shot pool):
```bash
# one command (folds -> pretrain -> 5x LOSO -> zeroshot -> stats -> tables/figures)
python src/selfsup/run_all.py --pooled_dir KIMORE_pooled --pool irds_only --n_folds 78
# scale ablation (transductive upper bound; REHAB246/UIPRMD enter the pool)
python src/selfsup/run_all.py --pool all_corpora
```
Or step-by-step (each step is resumable — existing outputs are skipped):
```bash
python src/selfsup/folds.py       --pooled_dir KIMORE_pooled --n_folds 78 --out outputs/folds.json
python src/selfsup/run_pretrain.py --folds_json outputs/folds.json          # 4 encoders + provenance
python src/selfsup/run_all.py --plan                                        # prints the 5 LOSO commands
python src/selfsup/stats.py && python src/selfsup/make_tables.py && python src/selfsup/make_figures.py
```

## 3. The five conditions (registry.CONDITIONS)
| Condition | init_ckpt | freeze_encoder |
|---|---|---|
| scratch (baseline rho=0.549) | — | no |
| contrastive_lp | contrastive_encoder.pt | yes |
| contrastive_ft | contrastive_encoder.pt | no |
| masked_lp | masked_encoder.pt | yes |
| masked_ft | masked_encoder.pt | no |

## 4. Outputs
```
outputs/folds.json                              # single source of truth (LOSO split)
outputs/ssl_pretrain/{irds_only,all_corpora}/{contrastive,masked}_encoder.pt(.provenance.json)
outputs/ssl_results/{A_scratch..E_masked_ft}/loso_results.json + zeroshot_*.json
outputs/ssl_results/stats.json                  # bootstrap CIs, Holm-Bonferroni, primary Wilcoxon, probe sanity
outputs/ssl_results/tables/table{1..4}_*.{csv,md}
outputs/ssl_results/figures/fig{1,2}_*.png
```

## 5. Design guarantees (why the result is defensible)
- **Fair comparison:** one `build_encoder`, one `folds.json`, matching `--d_model` across all arms.
- **No leakage:** the pretraining pool excludes every KIMORE LOSO test subject (all of KIMORE),
  so REHAB24-6/UI-PRMD stay pure zero-shot under `--pool irds_only`. `--pool all_corpora` is
  explicitly the transductive upper bound.
- **Honest negative:** the naive-feature baseline (path length + mean speed) and the pred_SD
  degeneracy gate appear in every zero-shot table; the linear-probe sanity check confirms the
  encoder beat chance, so a null cannot be dismissed as under-trained.
- **Reproducibility:** every checkpoint carries a provenance sidecar (git SHA, config hash, seed,
  pool manifest); the DAG is resumable by artifact presence.

## 6. Full-run cost (per RESEARCH_PLAN_2 §9)
~240 GPU-hours total (pretraining both paradigms x 2 pool scales + 5 conditions x 78 folds +
zero-shot). Stagger the LOSO jobs overnight; the DAG resumes from completed folds.
