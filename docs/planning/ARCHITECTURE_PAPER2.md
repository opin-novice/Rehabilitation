# Paper-2 Architecture — Scalable SSL Pretraining for Zero-Shot Cross-Sensor Rehab Scoring

**Target:** Q1 (Computers in Biology and Medicine) · **Basis:** RESEARCH_PLAN_2.md
**Principle:** reuse the Paper-1 pipeline in `src/`; add SSL as a new `src/selfsup/` package with exactly **two non-breaking hooks** into existing code.

> Implementation note: the package is named `selfsup` (not `ssl`) to avoid colliding with Python's stdlib `ssl` module. Verified end-to-end via `python src/selfsup/selftest.py`.
**Design date:** 2026-07-02

---

## 0. What already exists (verified in `src/`) — reuse, do not rewrite

| Concern | Existing module | Reused for |
|---|---|---|
| Canonical arrays `(N,100,25,3)`, scaling, loaders | `rehab_dataset.py` (`SkeletonRegressionDataset`, `ScalerBundle`, `make_dataloaders`), `constants.py` (`SEQ_LEN=100, NUM_JOINTS=25, NUM_CHANNELS=3, SKELETON_EDGES`) | All data I/O |
| Backbones | `models_stgcn.py` (`TCNRegressor`, `LSTMRegressor`, `STGCNRegressor`, `SCTRegressor`), `models.py` (`RehabTransformerRegressor`, graph variant), `build_model` | Shared encoder |
| Stratified-LOSO training | `train_loso.py` (`train_one_fold`, `main`, `seed_everything`; already has `extra_xyz/extra_eids/extra_y` hooks) | Phase 3 fine-tune (all 5 conditions) |
| Zero-shot + degeneracy | `validity_eval.py` (`_auroc`, `DEGENERACY_PRED_SD`, `external_transfer_diagnostics`, `validity_per_model`) | Phase 4 |
| External corpora | `load_rehab246.py`, `load_uiprmd_validity.py`, `irds_eval.py`, `verify_irds_labels.py` | Harmonization + eval |
| Stats / OOF / tables / figs | `statistical_tests.py`, `sample_level_stats.py`, `generate_oof.py`, `generate_tables.py`, `make_figures.py`, `novelty/run_all.py` (DAG pattern) | Results layer |

**Critical refactor identified:** the regressors compute a pooled embedding then apply `head`, but expose **no way to (a) read the embedding or (b) load a pretrained encoder**. That is the one architectural change everything else depends on.

---

## 1. Layered architecture

```
LAYER 4  Orchestration & Reproducibility   src/ssl/{config,registry,run_all}.py
              │  resumable DAG, config-hash provenance, git SHA, seed list
LAYER 3  Downstream (Phase 3/4)   train_loso.py(+--init_ckpt) · validity_eval.py · stats/*
              │  5 conditions, zero-shot AUROC + rank metric, bootstrap CI, Holm-Bonferroni
LAYER 2  SSL pretext (Phase 1/2)   src/ssl/{augmentations,pretext,pretrain,linear_probe}.py
              │  PretextTask ABC → ContrastiveNTXent | MaskedMotion
LAYER 1  Encoder abstraction   forward_features() on every backbone + src/ssl/heads.py
              │  ONE encoder, three heads (projection / decoder / regression)
LAYER 0  Data harmonization   src/ssl/{harmonize,pretrain_pool}.py
              │  canonical 25-joint schema, cross-sensor joint maps, unlabeled pool
```

Each layer depends only on the one below. New backbone / new SSL method / new corpus each touch exactly one seam (see §7).

---

## 2. Layer 0 — Data harmonization  *(fixes novelty-review weaknesses #1 tiny corpus, #3 cross-sensor)*

**`src/ssl/harmonize.py`** — one canonical schema, explicit per-sensor joint maps.

```
CANONICAL = Kinect-v2 25-joint order (constants.SKELETON_EDGES)
JOINT_MAPS = {
  "KIMORE":     identity(25),
  "IRDS":       identity(25),
  "UIPRMD_KINECT":  map 22 -> 25 (interpolate missing spine/hand joints),
  "REHAB24_6":  map OptiTrack 26 -> 25   # reuse memory: rehab24-6-joint-mapping
}
harmonize(corpus) -> HarmonizedArrays(
    x: (N,100,25,3), meta: {corpus, sensor, subject_uid, label, label_type})
```
- Centralizes the mapping the plan leaves under-specified; `subject_uid = f"{corpus}:{sid}"` prevents cross-corpus ID collisions.
- Records `sensor` so cross-sensor claims are precise (note: UI-PRMD has a Kinect capture — flag it as *not* cross-sensor; REHAB24-6 OptiTrack is the true cross-sensor test).

**`src/ssl/pretrain_pool.py`** — the fix for "2,589 sequences is too small for SSL".

```
build_pool(exclude_kimore_test_subjects: set[str]) -> UnlabeledPool
  pools UNLABELED sequences from IRDS + KIMORE + UI-PRMD + REHAB24-6
  ASSERT no subject_uid in exclude set  (leakage guard, logged)
  returns (X_pool, manifest{counts per corpus, hash})
```
- Two pool sizes are first-class experiment variables: `pool=irds_only` (~2.6k) vs `pool=all_corpora` (~6–7k). This lets the paper show a null is **not** an artifact of data scale (§6, requirement 5).

---

## 3. Layer 1 — Encoder abstraction  *(enables fair comparison = the core Q1 lever)*

**Modify each backbone (non-breaking):** add `forward_features(x) -> (B, D)` returning the pooled embedding; keep `forward` = `head(forward_features(x))`. Add a factory:

```python
# models_stgcn.py / models.py
def build_encoder(model_type, **kw) -> Encoder   # returns backbone WITHOUT head
class Encoder(Protocol):
    out_dim: int
    def forward_features(self, x: Tensor) -> Tensor  # (B, D)
```

**`src/ssl/heads.py`** — three interchangeable heads on the same embedding:
- `ProjectionHead(D→64→32)` — NT-Xent (discarded after pretrain)
- `ReconstructionDecoder(D→…→T*J*C)` — masked-motion (discarded after pretrain)
- `RegressionHead(D→1)` — reuse existing regressor head

> Guarantees the **identical encoder** across scratch / contrastive / masked conditions — reviewers reject SSL papers that quietly change capacity between arms.

---

## 4. Layer 2 — SSL pretext tasks (Phases 1–2), pluggable

**`src/ssl/augmentations.py`** (Phase 1) — 6 augmentations + a clinical-validity registry:

```python
AUG_REGISTRY = {                       # (pretrain_ok, finetune_ok)
  "temporal_crop":  (True,  True),
  "joint_mask":     (True,  True),
  "gaussian_noise": (True,  True),
  "rotation_y":     (True,  True),
  "speed_perturb":  (True,  False),    # duration is clinically meaningful for scoring
  "limb_scale":     (True,  False),    # already normalized in preprocessing
}
```

**`src/ssl/pretext.py`** — one interface, many methods (scalability seam):

```python
class PretextTask(ABC):
    def build_head(self, enc: Encoder) -> nn.Module: ...
    def loss(self, enc, head, batch) -> Tensor: ...

class ContrastiveNTXent(PretextTask):   # SimCLR, τ=0.07, two augmented views
class MaskedMotion(PretextTask):        # mask ∈ {joint, temporal, progressive}, MSE on masked slots
# future: CMAEHybrid(PretextTask) — add without touching the trainer
```

**`src/ssl/pretrain.py`** — a single generic trainer consuming any `PretextTask`:
- AMP + cosine schedule (fits RTX 5070 12 GB), batch 128.
- **Checkpoint by linear-probe quality, not loss** (uses Layer-2 probe).
- Saves **encoder-only** `*.pt` + provenance sidecar `{git_sha, config_hash, seed, pool_manifest_hash}`.

**`src/ssl/linear_probe.py`** — held-out pool probe; doubles as the **pretraining-sanity check** (encoder must beat chance, proving SSL learned *something* — makes a downstream null credible).

---

## 5. Layer 3 — Downstream (Phases 3–4), reuse Paper-1

**Modify `train_loso.py`:** add `--init_ckpt PATH` and `--freeze_encoder`. `train_one_fold` loads the pretrained encoder into `build_model`, optionally freezes it. This yields all **5 conditions with zero new training code**:

| Condition | `--model_type` | `--init_ckpt` | `--freeze_encoder` |
|---|---|---|---|
| A scratch (baseline ρ=0.549) | tcn | — | no |
| B contrastive linear-probe | tcn | simclr_best.pt | yes |
| C contrastive full-finetune | tcn | simclr_best.pt | no |
| D masked linear-probe | tcn | mae_best.pt | yes |
| E masked full-finetune | tcn | mae_best.pt | no |

**Fold determinism:** a single `folds.json` (78 Stratified-LOSO folds) is written once and consumed by **every** condition and by pretraining's leakage guard — one source of truth, no split drift.

**Zero-shot (Phase 4):** reuse `validity_eval.py` AUROC + `DEGENERACY_PRED_SD` gate on REHAB24-6 / UI-PRMD / IRDS. **Add a rank-based transfer metric** (Spearman of prediction vs. label order) to defuse the metric-conflation critique (thresholding a continuous regressor against binary labels). **Naive-feature baseline** (path length + mean speed) printed in every zero-shot table.

**Stats:** reuse `generate_oof.py` + `statistical_tests.py` + `sample_level_stats.py` for 20-seed bootstrap CIs + Holm-Bonferroni; **add paired Wilcoxon** for the primary C-vs-E contrast; pre-register success threshold (ρ>0.565, non-overlapping CI).

---

## 6. Layer 4 — Orchestration & reproducibility (the Q1 gate)

**`src/ssl/config.py`** — dataclasses (`PretrainCfg`, `FinetuneCfg`, `ExperimentCfg`) → YAML; deterministic `config_hash()`.
**`src/ssl/registry.py`** — maps `condition → (checkpoint, results_dir, provenance)`; single source for tables/figures.
**`src/ssl/run_all.py`** — resumable DAG (skip nodes whose `config_hash` already completed), mirroring `src/novelty/run_all.py`:

```
harmonize → build_pool(±scale) → pretrain{contrastive,masked} → linear_probe(sanity)
          → LOSO×{A,B,C,D,E} → zeroshot{rehab246,uiprmd,irds} → stats → tables+figures
```

**Reviewer-facing guarantees baked into the design:**
1. **Fair comparison** — one `build_encoder`, one `folds.json`, one seed list across arms.
2. **No leakage** — pool excludes KIMORE LOSO test subjects; asserted + logged.
3. **Reproducibility** — provenance sidecar on every artifact (git SHA, config hash, seed, input hash); `requirements.txt` lock; public GitHub release.
4. **Statistical rigor** — bootstrap CIs, Holm-Bonferroni, paired Wilcoxon, degeneracy gate, pre-registered threshold.
5. **Bulletproof negative** — naive baseline in every table + two pretraining scales + linear-probe sanity, so a null cannot be dismissed as under-powered or under-scaled.

---

## 7. Scalability (why this design holds up)

| Extension | Work required | Untouched |
|---|---|---|
| New backbone (e.g., ConvTran) | implement `forward_features` + register in `build_encoder` | pretext, trainer, downstream, stats |
| New SSL method (e.g., CMAE) | implement one `PretextTask` subclass | encoder, trainer, downstream |
| New corpus | add a joint map in `harmonize.py` | everything else |
| New downstream metric | add to `validity_eval.py` | training, pretraining |

Compute stays within RTX 5070 12 GB: AMP, batch 128, encoder-only checkpoints, and a resumable DAG that never recomputes a completed node.

---

## 8. Proposed layout (Paper-1 `src/` untouched except 2 hooks)

```
src/
├── (existing Paper-1 files — unchanged)
├── models_stgcn.py         # + forward_features() on each backbone   [hook 1]
├── train_loso.py           # + --init_ckpt / --freeze_encoder        [hook 2]
└── selfsup/                 # NEW package (mirrors src/novelty/ convention)
    ├── harmonize.py         # Layer 0
    ├── pretrain_pool.py     # Layer 0
    ├── heads.py             # Layer 1
    ├── augmentations.py     # Layer 2  (Phase 1)
    ├── pretext.py           # Layer 2  (ContrastiveNTXent, MaskedMotion)
    ├── pretrain.py          # Layer 2  (generic trainer)
    ├── linear_probe.py      # Layer 2  (monitor + sanity check)
    ├── config.py            # Layer 4
    ├── registry.py          # Layer 4
    └── run_all.py           # Layer 4  (resumable DAG)
outputs/
    ├── folds.json                      # shared LOSO split (single source of truth)
    ├── pretrain/{contrastive,masked}/  # encoder ckpt + provenance
    └── ssl_results/{A..E, zeroshot, stats, tables, figures}/
```

---

## 9. Build order (maps to the plan's timeline)

1. **Encoder refactor + `folds.json`** (Layer 1) — unblocks everything; verify regressors still reproduce ρ=0.549.
2. **harmonize + pretrain_pool** (Layer 0) — with leakage assertions.
3. **augmentations + pretext + pretrain + linear_probe** (Layer 2) — pilot 100-epoch runs.
4. **`--init_ckpt` hook + 5-condition LOSO** (Layer 3).
5. **zeroshot + rank metric + stats** (Layer 3).
6. **config/registry/run_all + provenance** (Layer 4) — reproducibility appendix + code release.
