# Project facts — SE(3)-Equivariant Rehabilitation Assessment

Regenerated 2026-08-12 on branch `merge/from-capstone` (60 commits, 2026-07-03 .. 2026-08-12).
Every line below is citable against this working tree. A draft may only compress what is here.

> **Provenance.** This file replaces a version carried over from the `capstone` fork. That version
> asserted three claims this line has since retracted: a 3.03 MAD worst-case viewpoint swing (the
> real worst sequence is 20.12), a 5.8e-9 "real 45-degree NTU camera displacement" anchor (the
> paired-camera measurement was defective; see `dcf934d`), and a node-failure win for EGRU (the
> point-cloud baseline wins that axis outright; see `212a2d7`). Do not reintroduce them.

> **Stale-doc warning.** `CLAUDE.md` still describes a *Neural CDE* as the method and "week-one
> de-risking" as the current phase. The code and paper moved past that: the CDE is a certified
> control that **fails the mean-predictor floor** (`paper_wacv.tex:184,240`; `src/cde_model.py`) and
> the deployed method is the **EGRU**. `README.md` is the accurate map.

---

## Purpose

- Skeleton-based automated assessment of physical rehabilitation exercises, in which **viewpoint
  invariance of the predicted clinical score is a theorem rather than a learned tolerance**.
- Paper: *"Viewpoint Invariance Is a Theorem, Not a Fit: An SE(3)-Equivariant Recurrence for
  Rehabilitation Exercise Assessment"*, WACV 2026 Applications track, 8 pp.
  — `paper/wacv_submission/paper_wacv.tex:40`
- Deployed artifact: **RehabSense**, a browser app scoring a live webcam session or an uploaded
  video and returning a report card. — `frontend/README.md`, `frontend/server.py`
- Research monorepo, not one paper's code: 5,171 tracked files; 91 flat modules in `src/` plus
  `src/reviewer/` (13), `src/tests/` (7), `src/selfsup/` (23).

## Stack

- PyTorch >=2.2, NumPy, SciPy, scikit-learn, pandas, einops, h5py — `requirements.txt`
- **e3nn 0.6.0** — steerable / O(3)-irrep tensor-field layers (`o3.Irreps`, `o3.spherical_harmonics`,
  `FullyConnectedTensorProduct`, `Gate`) — `src/equivariant_gru.py:96-150`
- **MediaPipe >=0.10.30** Tasks `PoseLandmarker` — pose extraction — `demo/pose_backend.py`
- OpenCV >=4.8 — camera loop and overlay — `demo/app.py`
- **FastAPI >=0.110 + uvicorn** — web backend — `frontend/requirements.txt`, `frontend/server.py`
- torchdiffeq / torchcde — only for the *failing* Neural-CDE control — `src/cde_model.py`

## Architecture — research pipeline (the model)

```
skeleton x in R^(T x J x 3), root-relative
  -> steerable message-passing encoder       src/equivariant_gru.py:72  (EquivariantSkeletonEncoder)
       irreps 32x0e + 8x1o, lmax=2, 2 layers, 25 joints
  -> invariant cut  Pi                       src/equivariant_gru.py     (InvariantProjection)
       parity-even 160-d (tanh scalars, log vector norms, pairwise cosines)
       parity-odd  123-d (triple products, anatomical signed volumes) = 283-d total
  -> Bi-GRU over [Pi ; dt_k ; speed]         hidden 128, 1 layer
  -> mean/max pool + MLP head (128) -> s
```

- One seam: everything left of the cut is equivariant, everything right of it sees only invariants,
  so the score is invariant for **arbitrary weights**. — Proposition 1, `paper_wacv.tex:227-231`
- Root-relative coordinates make translation invariance automatic.
- The recurrence consumes the sensor's **actual inter-arrival times** dt_k (no fixed frame grid).
- Deployed size: **0.66 M parameters**.

## Architecture — deployed system (RehabSense)

```
browser webcam ~15 fps JPEG over WS /ws/live   frontend/static/app.js ; frontend/server.py
  (or POST /api/upload, first 3 min of a video, true video timestamps)
  -> MediaPipe PoseLandmarker (VIDEO mode), 33 landmarks   demo/pose_backend.py
  -> Kinect-25 remap + training-matched preprocess         demo/mp_to_kinect.py
  -> (a) explainable biomechanics -> Movement Quality /100 demo/feedback.py
     (b) EGRU 5-fold ensemble -> AI score /50 (calibrated) demo/engine.py
  -> report card JSON + rendered PNG                       demo/report_card.py
```

- Endpoints: `WS /ws/live`, `POST /api/upload` + `GET /api/job/{id}`, `GET /api/exercises`,
  `/api/health`, `/api/bones` — `frontend/README.md`
- Headline Movement Quality is **five explainable biomechanics metrics** (range of motion, SPARC
  smoothness, symmetry/balance, tempo & rhythm, consistency) — not the network. The network rides
  along as a smaller calibrated badge because it was trained on Kinect depth and webcam input is a
  domain shift. — `demo/feedback.py`, `demo/SESSION_DEMO.md`
- Ensemble: 5 folds, `outputs/cde_block2/egru_s0_pooled_f{0..4}.pt` (2 MB each); the PCT baseline
  ensemble is 21 MB/fold. Runs on CPU.

## Datasets

- **KIMORE** — 77 subjects, 380 pooled per-sequence rows, 5 exercises, clinical score 0-50.
  Protocol: pooled over exercises, subject-disjoint 5-fold, 3 seeds; bootstrap over n=77 subjects.
- **REHAB24-6** (OptiTrack) — 1,057 repetitions, 10 subjects, binary correct/incorrect —
  `src/load_rehab246.py`
- **NTU RGB+D Cross-View** — 18,674 test clips; three simultaneous cameras per performance —
  `src/ntu_dataset.py`, `src/ntu_paired_camera.py`, `src/run_xview_evaluation.py`
- **UI-PRMD** — third-corpus replication. Built via `src/load_uiprmd_validity.py --build`
  (`--geometry fk`, the corrected coordinates; `--geometry raw` reproduces the original bone-offset
  bug, quarantined in `outputs/validity_uiprmd_raw/`). Trained by `src/train_uiprmd.py`.
- IntelliRehabDS loader also present — `src/irds_eval.py`

## Results — measured

`src/aggregate_final.py` is the source of truth: it regenerates every table from banked JSON.

**Clean accuracy** (pooled KIMORE, subject-disjoint, mean+/-std over 3 seeds; MAD of 50) — `tab:accuracy`

| Model | MAD | Params | rho pooled | rho within-ex | Delta vs floor (95% CI) |
|---|---|---|---|---|---|
| PCT (baseline) | 6.47 +/- 0.20 | 4.91 M | 0.60 +/- 0.02 | 0.55 +/- 0.02 | -1.87 [-2.72, -1.06] |
| PCT + rotation aug. | 7.47 +/- 0.23 | 4.91 M | — | — | — |
| InvariantGRU (hand-crafted) | 6.31 +/- 0.16 | 0.21 M | 0.59 +/- 0.01 | 0.56 +/- 0.01 | -2.03 [-2.89, -1.23] |
| **EGRU (ours)** | **6.73 +/- 0.30** | **0.66 M** | 0.56 +/- 0.04 | 0.52 +/- 0.04 | -1.60 [-2.37, -0.88] |
| Mean-predictor floor | 8.31 +/- 0.05 | — | 0.17 +/- 0.01 | -0.07 +/- 0.02 | — |

- **Spearman rho is reported, not deferred** — banked over 3 seeds at
  `outputs/cde_block2/spearman_pooled3seed.json`, schema `{"rho": {tag: {pooled_mean, within_exercise_mean, ...}}}`,
  read by `src/aggregate_final.py:138`. The tie survives the metric change: rho separates the models
  by at most 0.05. The rho null must be stated — the mean predictor scores pooled rho = 0.17 while
  ignoring its input, and exactly 0 once between-exercise differences are removed.
- **Nondeterminism floor 0.33 MAD** (fixed configuration; the seed-to-seed spread is 0.48) — larger
  than every clean-accuracy gap, so the models **tie** and the benchmark metric is saturated.
- **Epoch-selection inflation +1.22 MAD (18.6%)** on the single-exercise slice, measured on the
  *baseline* — `tab:protocol3seed`, `src/protocol_null.py`. The signal there is seed-fragile
  (clears zero in 1/3 seeds); pooling fixes it for every model.
- **Exercise-conditioning asymmetry closed.** Pooled PCT alone was scored blind to exercise id.
  Re-trained with identical conditioning (3 seeds, both arms): clean 6.59 +/- 0.07 vs 6.47 +/- 0.20,
  augmented 7.42 +/- 0.22 vs 7.47 +/- 0.23 — inside the floor and opposite in sign — and the
  per-sequence swing is *larger* (21.43 vs 20.12). — `src/train_baseline_pct.py --exercise-cond`

**Viewpoint** — `fig:viewpoint`, `tab:pareto`

| Model | Degradation under camera rotation |
|---|---|
| **EGRU (ours)** | mean **9e-6** MAD, worst single sequence 3e-4 — certified, not empirical |
| InvariantGRU | exactly 0 (positive control, invariant by construction) |
| Ridge on hand-crafted invariants | 1e-14, fp64 roundoff (positive control) |
| PCT (baseline) | mean 9.33 at worst angle, worst sequence 33.96; crosses the floor between 30 and 45 deg |
| PCT + rotation aug. | mean **3.01**, 95th pct 8.65, **worst single sequence 20.12** |
| TCN | 13.17 |
| ST-GCN | 9.18 |

- Viewpoint fragility is **generic to non-equivariance**, not PCT-specific: three independently
  designed non-equivariant architectures (temporal-conv, graph-conv, attention) all degrade 9-13 MAD.
- The 20.12 figure is a global max over 380 held-out sequences x 3 seeds; the fold-averaged max would
  read a flattering 12.81. Not an undertraining artefact: at 60/180/540 epochs the mean shift is
  3.01/3.27/2.94 and the worst 20.1/21.4/19.9. — `src/pct_convergence_sweep.py`
- **Eight numerical gates E1-E8**, all pass, all re-runnable, over **Haar-random SO(3)** rather than a
  single azimuth sweep — `src/certify_egru.py`, `src/equivariance_suite.py`

**NTU RGB+D — the guarantee is measured, and the honest half is reported**

- Haar-random SO(3) rotation of all 18,674 Cross-View test clips changes **not one** prediction:
  agreement 100.00%, Top-1 82.95% unchanged, drift 2.3e-13 in fp64. ST-GCN agrees on 91.89% and loses
  3.03 points, with a drift precision cannot move (ratio 1.0008) — structural, not numerical.
- **A rotated skeleton is not a relocated camera.** On NTU's 18,674 *simultaneous three-camera*
  triples — three skeleton estimates of one event — our agreement falls to **73.03% against ST-GCN's
  73.86%: a small but resolvable deficit** (paired exact binomial, 5,394 discordant triples, p=0.037).
  The theorem removes a viewpoint change's geometry exactly and none of the tracker noise a
  relocation adds. — `src/ntu_paired_camera.py`
- Generalisation: trained without rotation augmentation on cameras 2-3, tested on camera 1, the
  encoder reaches 82.98% Top-1 (T=100, single-clip, one seed).

**Sensor-node failure** (MAD as k Kinect nodes freeze; 3 seeds x 5 folds, no retraining) — `tab:nodefail`

| Dead nodes k | EGRU (ours) | InvariantGRU | +oracle mask | PCT |
|---|---|---|---|---|
| 1 | 7.10 | 9.78 | 9.39 | 6.76 |
| 4 | 8.41 | 15.96 | 15.25 | 7.51 |
| 8 | 10.50 | 18.37 | 16.33 | 8.73 |
| **MAD lost** | +3.76 | +12.05 | +10.02 | **+2.27** |

- **PCT wins this axis outright.** Node robustness is not a property of equivariance and the paper
  does not claim it. The table supports something narrower: the guarantee is not paid for with a
  robustness collapse. k=8 crosses the floor for every model, EGRU included.
- A single dead joint drives the hand-crafted model from 6.31 to 9.78 — through the floor — because
  its features *name* the joints they depend on. An **oracle liveness mask** closes only ~25% of the
  gap (+12.05 -> +10.02), so the gap is architectural, not a missing-information artifact.
- Ordering holds under five failure operators (freeze random/left-only, stuck-at-lag, sporadic burst,
  axis-depth noise) — `tab:nodefail_modes`, `src/joint_failure.py`

**The Pareto grid — the headline claim** — `tab:pareto`

| Model | Params | Clean | 90 deg | 2 dead | Mean degr. |
|---|---|---|---|---|---|
| **EGRU (ours)** | 0.66 M | 6.73 | 6.73 ok | 7.55 ok | **9e-6** |
| InvariantGRU | 0.21 M | 6.31 | 6.31 ok | 11.64 fail | **0** |
| Lighter E(n) (EGNN) | ~0.6 M | 6.88 | 6.88 ok | 8.44 fail | 1.4e-5 |
| Canon-PCA + PCT | 4.91 M | 6.85 | 6.85 ok | 7.79 ok | ~0 (empirical) |
| PCT (baseline) | 4.91 M | 6.47 | 10.13 fail | 7.10 ok | 9.33 |
| PCT + rot | 4.91 M | 7.47 | 7.68 ok | 7.76 ok | 3.01 |

- **No model wins all five columns, and that is the claim.** PCT is both the most accurate and the
  most node-robust. Three models clear every stress — ours, Canon-PCA+PCT, and augmented PCT — so we
  are not the only survivor. Among those three, ours alone runs at 0.66 M rather than 4.91 M and
  alone rests on a theorem rather than an estimated frame or an augmentation tolerance.
- The EGNN arm is now **tuned** across depth, width and coordinate-clamp over 3 seeds: it loses
  +6.4 +/- 1.3 MAD over 8 dead joints (vs +3.76) and crosses the floor already at two (8.44 vs 7.55).

**Canonicalization steelman — the strongest objection, closed on conditioning not accuracy**

- Canon-PCA+PCT matches every offline axis: clean 6.85, 7.79 at two dead joints, viewpoint
  empirically exact for arbitrary SO(3) (worst canonical-coordinate shift 3.7e-11 over 32 rotations),
  with axes sign-disambiguated and the frame forced right-handed. — `research_egnn/canonicalize.py`
- The difference is **conditioning**, via Davis-Kahan. On KIMORE the eigen-gap is small far more often
  than not: **18% of frames within 0.05 of degeneracy, 45% within 0.10** (a standing body is nearly
  axially symmetric). Injecting camera-frame noise at 2% of body scale, the residual rotation between
  two camera poses grows from 1.2 deg where gamma>0.25 to **11.1 deg where gamma<0.02 — 9x
  amplification, 14.5% of frames past 45 deg**.
- A better sign rule does not fix it (third-moment rule: 12.1 deg, catastrophic flips only trimmed to
  11.1%) because a sign rule resolves a *discrete* ambiguity and degeneracy is a *continuous* one.
  Temporal smoothing is worse (28.8 deg; 31.6% past 45 deg).

**Clinical interpretation of the MAD scale**

- **No ICC or kappa is reported for KIMORE's clinical Total Score, or for REHAB24-6's.** A targeted
  search could not surface one and none is fabricated — the paper flags this as its clearest open
  measurement gap.
- External context only: movement-quality rating shows CMAS ICC 0.58-0.91, cross-diagnostic MQS 0.93.
  MCIDs on cognate 0-66/0-100 motor-recovery scales: Fugl-Meyer UE 4-12.4, LE 6, STREAM 1.9-4.8.
- **KIMORE has no clinical score bands.** The five groups are recruitment categories: on Es5 an Expert
  scores 25.0 while a Parkinson patient scores 50.0. A hardcoded `CLINICAL_BANDS` dict was removed
  from `src/clinical_analysis.py` for exactly this reason — do not reintroduce it.
- Measured against the score's actual resolution (`src/clinical_resolution.py`): the augmented
  baseline's mean swing is **0.30 between-subject SD** and its worst sequence **2.01 SD**, exceeding
  the gap separating **20.2% of subject pairs, 83.0% at the tail** — pairs a camera can reorder.

**Second corpus, REHAB24-6** (3 seeds, subject-disjoint CV; identical 0.66 M model as a BCE logit)

| Model | AUROC | AUROC @ 90 deg | Node-fail lost |
|---|---|---|---|
| **EGRU (ours)** | **0.738 +/- 0.01** | **0.738** (logit drift <=1.5e-4) | +0.14 |
| PCT (baseline) | 0.707 +/- 0.02 | 0.52 — chance (drift 23.2) | **+0.08** |

- The two models **cross** between k=4 and k=8 frozen joints; PCT finishes 0.021 AUROC ahead at k=8.
- At 10 subjects this corpus replicates *structural* properties, not accuracy.

**Deployment / edge** — `tab:deploy`

| Arithmetic | Invariance floor | Score shift (MAD of 50) |
|---|---|---|
| fp64 | 5.75e-16 | 3.3e-14 |
| fp32 | 2.68e-07 | 8.9e-06 |
| fp16 | 2.62e-03 | 0.024 |
| bf16 | 8.34e-03 | 0.195 |
| int8 weights only | 3.09e-07 | 1.2e-05 (theorem holds exactly) |
| int8 activations only | 3.38e-02 | 0.056 |
| int8 weights + acts | 3.40e-02 | **0.051** |

- fp16 is 3.2x tighter than bf16 — equivariance pays in mantissa bits, which bf16 trades for exponent
  range. The int8 cliff is **solely activation grid-rounding**; weight quantization preserves the
  theorem exactly. — `src/precision_budget.py`, `src/int8_quant_budget.py`
- `src/precision_budget.py` **drops the e3nn `_w3j` buffers before `load_state_dict`**. Without that
  guard an fp32 checkpoint overwrites the natively built fp64 Wigner-3j tables and `inv_floor` reads
  3.788e-10 instead of 2.130e-16 — six orders. The published fp64 row is the clean value.
- **Causal streaming**: 8.9 ms/frame = 112 fps (RTX-class GPU, batch 1), usable score after ~55 frames,
  at a 3.2-point NTU X-View cost vs bidirectional (79.8 vs 83.0). GPU latency is an upper bound;
  **edge-SoC profiling is not done**. — `src/ttfs_benchmark.py`, `src/train_stream_egru.py`
- **A real camera move**: replaying one clip to a webcam from seven physical camera poses, our score
  moves 0.74 of fifty on average (worst 1.41) against PCT's 2.16 (worst 3.25). One subject, one
  exercise; it bounds *shift*, not accuracy, and ours is not zero. — `src/real_viewpoint_probe.py`

**Per-family ablation of the cut** (each family zeroed at inference, no retraining): pairwise cosines
carry the most accuracy (+1.83 MAD to remove) and the most node-failure robustness (+7.29); learned
triple products +2.75; fixed anatomical volumes inert (+0.19). — `src/ablation_invfamily.py`

## Prior work being compared against

- **Target paper**: Rafat et al., "A Point Cloud Transformer for Remote Monitoring and Automated
  Assessment of Physical Rehabilitation Exercises", IEEE JBHI 2026 (accepted), arXiv:2606.30309 —
  bibitem `pct2026`. Reimplemented as `src/models_curvenet.py`, trained by `src/train_baseline_pct.py`.
- Published numbers are **not** entered into `tab:accuracy` because no protocol matches. Kuang et al.
  report MAD 0.10-0.16 in the *same* 0-50 units as our 6.3-6.7 — lower by a factor of **39-67** — under
  an 8:1:1 *sample-level* stratified split. Those units are recovered from the published tables
  (`src/published_units_audit.py`); reading them wrong would move the comparison by 8.5x.
- Controls implemented in-repo: InvariantGRU (`src/invariant_controls.py`), Canon-PCA + PCT
  (`research_egnn/canonicalize.py`), EGNN (`research_egnn/egnn_encoder.py`), ST-GCN
  (`src/models_stgcn.py`), TCN (`src/train_tcn_wacv.py`), ridge-on-invariants (`src/ridge_baseline.py`)

## Honest status

**Working end to end**
- Training, certification, aggregation: `src/train_egru.py`, `src/certify_egru.py`,
  `src/aggregate_final.py` (regenerates every paper table from banked JSON)
- 5-fold EGRU + PCT checkpoints under `outputs/cde_block2/`; ~2,470 banked result files under `outputs/`
- Live webcam demo with a self-checking invariance gate: `demo/smoke_test.py`
- Session demo + report card: `demo/app_session.py`, `demo/smoke_session.py`, `demo/report_card.py`
- Web frontend, live + upload modes: `frontend/server.py` (ported from the capstone fork, 2026-08-12)
- WACV 8pp submission and a 17pp arXiv twin both compile
- `scripts/check_paper_claims.py` re-checks the paper's numbers against banked artifacts

**Partial / caveated**
- The AI badge on webcam reads low (~11-15/50) — the model was trained on Kinect depth; it needs
  per-setup calibration via `demo/calibrate.py`, and the headline score deliberately does not depend
  on it — `demo/SESSION_DEMO.md`
- Edge-SoC profiling is not done; streaming latency is GPU-measured and reported as an upper bound
- REHAB24-6 has only 10 subjects — structural replication, explicitly not an accuracy benchmark
- `src/train_uiprmd.py` (third corpus) is wired and compiles but has **no banked results yet**;
  it needs `outputs/validity_uiprmd/` (present, the fk build) and a run
- The NTU paired-camera deficit (73.03% vs 73.86%) is conceded and still carries a protocol confound

**Negative results, reported with equal prominence**
- **Irregular sampling is a null on KIMORE**: the fixed-grid baseline wins at every drop level. A
  Lomb-Scargle census explains it — 97.7% of *positional* energy sits below the resampling corner,
  though **70% of *velocity* energy sits above**, so resampling preserves pose geometry while
  low-passing the derivative band. On healthy-form exercises the discriminative signal is positional,
  so resampling acts as a denoiser; the boundary would move for tremor-dominated tasks. The mechanism
  is real (a 5 Hz tremor recovers at r=1.000 through 70% drops where a resampled estimator decays to
  0.785). — `src/irregular_data.py`, `src/bandwidth_law.py`
- **Restoring chirality** is principled but unrewarded on healthy-form exercises (+0.11 MAD, inside
  the floor, for 16.7% more parameters). Adopted anyway, on principle not accuracy. — `src/chirality.py`
- The **Neural CDE** continuous-time variant is certified to the same standard but **fails the
  mean-predictor floor**; kept as a control — `paper_wacv.tex:184,240`, `src/cde_model.py`
- Earlier thread (Paper 2): SSL pretraining does **not** rescue zero-shot cross-sensor scoring — a
  definitive negative over 78-fold LOSO — `src/selfsup/`, `paper/archive/manuscript.tex`

**Absent / weak**
- **No CI, no test-runner config, no coverage.** `src/tests/` is 7 ad-hoc dev scripts (now portable —
  the hardcoded `D:/Rehabilation/src` paths were removed 2026-08-12)
- No authentication, no persistence layer, no deployment config (no Dockerfile, no CI workflow)
- No commercialization code anywhere in the repo
- Two `NotImplementedError` stubs, both deliberate refusals to fake a baseline —
  `src/baselines.py`, `src/models_stgcn.py`
- `CLAUDE.md` is stale relative to the code (see the warning at the top of this file)
- `src/INDEX.md` exists in the capstone fork but not here; the `README.md` code map is the substitute

## Gaps (ask the user)

1. **Business model** — the repo contains zero commercialization artifacts: no pricing, no billing,
   no deployment config, no licence beyond MIT. Any BUSINESS section must come from stated intent.
2. **Novelty framing** — the repo supports several honest framings (vs the PCT target paper; vs
   augmentation; vs canonicalization; vs hand-crafted invariants). The current paper headlines the
   **Pareto frontier on which no model dominates**; a report may reasonably headline a narrower one,
   but not one the evidence retracted.
3. **Metrics** — no gap. The strongest single pairing for a 100-word abstract is **6.73 MAD at 0.66 M
   parameters with a 9e-6 worst-case viewpoint shift, against the baseline's 6.47 MAD at 4.91 M and a
   20.12 MAD worst-sequence swing even after rotation augmentation**.
