# Self-Supervised Pretraining for Zero-Shot Cross-Sensor Rehabilitation Quality Assessment

**A definitive negative result:** SSL pretraining on unlabeled skeletons does **not** rescue
zero-shot cross-sensor transfer for rehabilitation scoring. The barrier is sensor-level domain
shift, not representation quality.

[![Paper](https://img.shields.io/badge/paper-manuscript.tex-blue)](manuscript.tex)
[![Results](https://img.shields.io/badge/results-77--fold%20LOSO-success)](results/kimore_loso_78fold/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## Key Finding

| Question | Answer |
|----------|--------|
| Does SSL pretraining enable zero-shot cross-sensor scoring? | **No.** AUROC at chance (0.51–0.53) on both test corpora. |
| Does SSL fine-tuning improve within-domain (KIMORE) scoring? | **No.** SSL FT = scratch (p>0.3). SSL LP is significantly **worse** (p<1e-13). |
| Does more unlabeled data help? | **No.** Quadrupling the pool (~1k → ~5k) does not close the gap. |
| Does pretext task matter? | **No.** Contrastive ≈ masked-motion (p=0.80). |

The naive kinematic baseline (joint path length + mean speed) beats every learned model
on both test corpora.

---

## Project Structure

```
├── manuscript.tex              # Full TNSRE manuscript (LaTeX)
├── manuscript.md               # Manuscript in Markdown
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── .gitignore
│
├── src/
│   ├── selfsup/                # SSL pipeline (the main contribution)
│   │   ├── pretrain.py         #   SSL pretraining (contrastive + masked)
│   │   ├── pretext.py          #   Pretext task definitions
│   │   ├── augmentations.py    #   Data augmentations
│   │   ├── data.py             #   Corpus loaders and adapters
│   │   ├── folds.py            #   LOSO split generation
│   │   ├── harmonize.py        #   Cross-sensor joint mapping
│   │   ├── heads.py            #   Encoder heads (projection, reconstruction, regression)
│   │   ├── zeroshot_eval.py    #   Zero-shot cross-sensor evaluation
│   │   ├── linear_probe.py     #   Probe-sanity check
│   │   ├── naive_baseline.py   #   Path-length + speed baseline
│   │   ├── stats.py            #   Statistical tests
│   │   ├── run_all.py          #   Resumable DAG orchestrator
│   │   ├── run_loso78.py       #   77-fold LOSO runner
│   │   ├── selftest.py         #   Self-test / smoke test
│   │   ├── make_tables.py      #   Paper tables
│   │   └── make_figures.py     #   Paper figures
│   │
│   ├── models_stgcn.py         # Backbones (TCN, LSTM, ST-GCN, etc.)
│   ├── train_loso.py           # LOSO training (+ --init_ckpt hook)
│   ├── train.py                # Base training loop
│   ├── evaluate.py             # Evaluation metrics
│   ├── rehab_dataset.py        # Dataset classes and scalers
│   ├── constants.py            # Constants (SEQ_LEN=100, joints, edges)
│   ├── load_rehab246.py        # REHAB246 loader
│   ├── load_uiprmd_validity.py # UI-PRMD loader
│   ├── irds_eval.py            # IRDS loader + reliability eval
│   ├── prepare_kimore.py       # KIMORE preprocessing
│   ├── pool_exercises.py       # KIMORE exercise pooling
│   └── ...                     # Additional utilities
│
├── results/
│   └── kimore_loso_78fold/     # Full 77-fold LOSO results
│       ├── A_scratch/          #   Condition A: from scratch (best: rho=0.836)
│       ├── B_contrastive_lp/   #   Condition B: contrastive linear probe
│       ├── C_contrastive_ft/   #   Condition C: contrastive fine-tune
│       ├── D_masked_lp/        #   Condition D: masked linear probe
│       ├── E_masked_ft/        #   Condition E: masked fine-tune
│       ├── stats78.json        #   All pairwise statistics
│       └── table78.md          #   Summary table
│
├── outputs/                    # Generated outputs (see .gitignore — not in repo)
│   ├── ssl_pretrain/           #   Pretrained encoders
│   ├── ssl_results/            #   5-fold results
│   └── validity/               #   Cached corpus sequences
│
├── MANUSCRIPT_OUTLINE.md       # Detailed paper outline
├── REPRODUCE_PAPER2.md         # Reproduction guide
├── ARCHITECTURE_PAPER2.md      # Architecture design doc
├── RESEARCH_PLAN_2.md          # Original research plan
├── RESULTS_PAPER2.md           # All results summary
│
├── KIMORE/                     # Raw KIMORE data (not tracked in git)
├── KIMORE_pooled/              # Pooled KIMORE (not tracked)
├── KIMORE_processed/           # Processed KIMORE (not tracked)
└── ...                         # Other data directories (gitignored)
```

---

## Datasets

| Dataset | Source | Type | Samples | Sensor | Labels |
|---------|--------|------|---------|--------|--------|
| **KIMORE** | Capecci 2019 | Training/eval | 380 (77 subjects × 5 exercises) | Kinect v2 | Physician score (0–50) |
| **IRDS** | Capecci 2019 | SSL pretraining | 1,000 (10 × 10 × 10) | Kinect v2 | Unlabeled |
| **REHAB246** | Zenodo | Zero-shot test | 1,057 (correct/incorrect) | OptiTrack | Binary correctness |
| **UI-PRMD** | Vakanski 2018 | Zero-shot test | 2,000 (correct/incorrect) | Kinect v2 | Binary correctness |

---

## Quick Start

### Prerequisites
- Python 3.10+
- PyTorch 2.2+ (CUDA recommended for full run)
- Dependencies: `pip install -r requirements.txt`

### Smoke test (no data needed, ~40 s CPU)
```bash
python src/selfsup/run_all.py --smoke
```
Runs the entire pipeline on dummy data: folds → pretrain → LOSO → zero-shot → stats.

### Full reproduction
```bash
# 1. Prepare data caches
python src/prepare_kimore.py && python src/pool_exercises.py
python src/irds_eval.py --build
python src/load_rehab246.py --build
python src/load_uiprmd_validity.py --build

# 2. Run the full pipeline (primary: IRDS-only pool)
python src/selfsup/run_all.py --pooled_dir KIMORE_pooled --pool irds_only --n_folds 78

# 3. Scale ablation (transductive upper bound)
python src/selfsup/run_all.py --pool all_corpora

# 4. Generate tables and figures
python src/selfsup/make_tables.py && python src/selfsup/make_figures.py
```

All steps are **resumable** — completed steps are skipped on re-run.

### Compute budget
Full run: ~240 GPU-hours on a single RTX 5070 12 GB (AMP, batch 128).
The DAG is designed to resume from partial completion (e.g., staggered overnight LOSO jobs).

---

## The Five Conditions

| Label | Condition | Init checkpoint | Encoder frozen? |
|-------|-----------|-----------------|-----------------|
| A | Scratch (from scratch) | None | No |
| B | Contrastive LP | Contrastive encoder | **Yes** |
| C | Contrastive FT | Contrastive encoder | No |
| D | Masked LP | Masked encoder | **Yes** |
| E | Masked FT | Masked encoder | No |

---

## Core Results

### Zero-shot cross-sensor (primary finding)

| Condition | REHAB246 AUROC | UI-PRMD AUROC |
|-----------|:---------------:|:--------------:|
| Scratch | 0.516 | 0.524 |
| Contrastive LP | 0.516 | 0.518 |
| Contrastive FT | 0.515 | 0.514 |
| Masked LP | **0.527** | 0.512 |
| Masked FT | 0.519 | 0.514 |
| **Naive baseline** | **0.554** | **0.538** |

### 77-fold LOSO (within-domain)

| Condition | Mean ρ | 95% CI | vs. scratch (adj. p) |
|-----------|:------:|:------:|:--------------------:|
| **Scratch** | **0.836** | [0.785, 0.867] | — |
| Masked FT | 0.823 | [0.773, 0.854] | 0.318 |
| Contrastive FT | 0.816 | [0.762, 0.851] | 0.318 |
| Contrastive LP | 0.689 | [0.617, 0.738] | 3.4e-14 |
| Masked LP | 0.679 | [0.612, 0.727] | 7.3e-18 |

---

## Design Guarantees

- **Fair comparison:** one encoder factory, one `folds.json`, matching `d_model` across all conditions
- **No leakage:** pretraining pool excludes all KIMORE subjects; asserted and logged
- **Degeneracy gate:** models with `pred_SD < 0.10` are flagged as collapsed (cannot claim discrimination)
- **Probe sanity:** linear-probe ρ ≈ 0.68 confirms encoders learned real structure
- **Reproducibility:** every checkpoint carries provenance (git SHA, config hash, seed, pool manifest)
- **Holm-Bonferroni correction:** all pairwise tests corrected for FWER over 10 comparisons
- **Protocol invariance:** zero-shot result holds under both 5-fold and 77-fold protocols

---

## Manuscript

The full manuscript is written for **IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE)**:

- [`manuscript.tex`](manuscript.tex) — LaTeX source (IEEEtran format)
- [`manuscript.md`](manuscript.md) — Markdown version for easy review

### Outline
1. **Introduction** — motivation, gap, contributions
2. **Related Work** — KIMORE benchmarks, SSL for skeletons, cross-sensor transfer
3. **Methods** — datasets, SSL pretraining, 77-fold LOSO, zero-shot protocol, statistical testing
4. **Results** — zero-shot (chance-level), LOSO (SSL FT = scratch), scale ablation, protocol invariance
5. **Discussion** — why SSL fails, implications, rigor hooks, limitations, future work
6. **Conclusion**

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{rehabilitation2026ssl,
  title={Self-Supervised Pretraining Does Not Rescue Zero-Shot Cross-Sensor Rehabilitation Quality Assessment},
  author={Author, A. and Author, B. and Author, C.},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  year={2026}
}
```

---

## License

MIT
