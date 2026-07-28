# HISTORY.md — The Research Journey

> Reconstructed from the repository on **2026-07-27**: 36 commits, 4 branches, ~98 source
> modules in `src/`, three paper drafts, and the planning/reference document set under `docs/`.
>
> This is written for a researcher joining the project cold. It is **not** a changelog — it is an
> account of what was believed, what was tried, what broke, and what the evidence forced us to
> change. Where the repository does not preserve an answer, the text says **Unknown** rather than
> guessing.
>
> **Reading order for a newcomer:** this file → `docs/reference/PROJECT_BRIEF.md` (theory) →
> `CLAUDE.md` (working rules) → `src/aggregate_final.py` (regenerates every number in the paper
> from banked JSON; it is the source of truth for the tables).

---

## Research Overview

### Objective

Build and defend a skeleton-based rehabilitation-exercise assessment model whose **invariance to
camera viewpoint is a theorem, not a learned tolerance** — and, just as importantly, build the
*evaluation protocol* under which such a claim can be checked rather than asserted.

The current paper is titled **"Viewpoint Invariance Is a Theorem, Not a Fit: An SE(3)-Equivariant
Recurrence for Rehabilitation Exercise Assessment"** (WACV 2026 Applications track,
`paper/wacv_submission/`).

### The problem

Automated rehabilitation scoring promises remote monitoring: a patient exercises at home, a depth
camera watches, a model returns a clinical quality score. The dominant research regime trains and
tests on **clean, frontal, single-sensor** recordings and ranks models by a single aggregate error
against a clinician's score.

That regime rewards capacity and penalises nothing a deployed device actually meets:

- a camera set against a different wall,
- a tracking node that freezes (the characteristic consumer-depth failure),
- dropped frames and irregular inter-arrival times.

A clinical screening instrument has a stricter obligation than low average error: **it must return
the same patient the same score from a different room.** That is a per-prediction guarantee, and
aggregate MAD cannot see it.

### Why it matters

Two findings from this project's own history make the case concrete:

1. **Aggregate accuracy on KIMORE is saturated.** Three architectures of vastly different inductive
   bias — a point-cloud transformer, a steerable equivariant recurrence, and a GRU on six
   hand-crafted invariant features — are mutually indistinguishable, every gap falling inside a
   measured **0.33 MAD nondeterminism floor**. A benchmark that cannot tell those apart is not
   measuring architecture.
2. **A rotation-augmented baseline looks solved and is not.** Its *aggregate* accuracy curve across
   camera azimuth is flat — by the field's metric it has solved viewpoint — yet an individual
   patient's score still moves by up to **20.12 MAD of 50** when the camera moves. The errors
   cancel in the mean; they do not vanish.

The thesis, then, is about the *axis of evaluation* as much as the architecture.

---

## Initial Idea

### Provenance: this is the third research generation in this directory

The repository is explicitly a **monorepo of related research threads** (`README.md`). Understanding
the current work requires knowing what preceded it, because the current protocol discipline is
inherited scar tissue from the earlier threads.

#### Generation 0 — the protocol-inflation benchmark (predates this git history)

Described in `docs/planning/RESEARCH_PLAN_1.md` as "the existing paper", status *26/26 tasks
complete*. Title: *"Benchmarking Deep Learning Architectures for Automated Rehabilitation Quality
Scoring Under Clinically Valid Leave-One-Subject-Out Evaluation."*

Its question: rehabilitation-AI papers claimed Spearman ρ up to 0.95 on KIMORE. **Are those numbers
real, or is the protocol inflated?** Answer: inflated.

Surviving artifacts (`outputs/tables/`, `outputs/novelty/`) preserve the 2×2 decomposition:

| Cell | Protocol | Mean ρ |
|---|---|---|
| leak / unstratified | KFold, subjects leak | 0.5161 |
| leak / stratified | StratifiedKFold, subjects leak | 0.5210 |
| noleak / unstratified | GroupKFold (subject-level) | 0.4960 |
| noleak / stratified | **StratifiedGroupKFold — "Stratified LOSO"** | 0.4894 |

Measured subject-identity leakage: **+0.026 ρ**. Best honest model TCN at ρ≈0.549 against a
literature reporting 0.95. It also found a **zero-shot validity failure**: the best model reached
AUROC 0.58 on REHAB24-6 and 0.53 on UI-PRMD, while a *naive* feature (total joint path length +
mean speed) reached 0.71 and 0.65 — beating trained deep models.

*Publication venue and status: **Unknown** — not recorded in this repository.*

#### Generation 1 — Paper 2: does SSL rescue cross-sensor transfer? (2026-07-03 → 07-05)

Commits `88fe67c` … `cbc020a`. Branch `research/cross-sensor-pointcloud-ssl`.

`RESEARCH_PLAN_1.md` was superseded by `RESEARCH_PLAN_2.md` after a literature review found two
factual errors and one pre-empted claim:

- IRDS and IntelliRehabDS are the **same dataset** (Miron et al. 2021, Zenodo 4610859), listed as two.
- The planned "physician-scored external corpus" contribution was deleted — IntelliRehabDS has only
  binary labels; **no continuous physician scores exist in any of the four corpora except KIMORE**.
- "First contrastive pretraining" was pre-empted (Karlov 2025, SSL-Rehab) → reframed to "first
  honest zero-shot cross-sensor SSL evaluation", with a masked-motion arm added to convert being
  scooped into the first head-to-head comparison of the two SSL paradigms.

**Result: a definitive negative.** SSL pretraining does not rescue zero-shot cross-sensor scoring.

| Condition | Mean ρ (KIMORE, 5-fold) | Beats scratch? |
|---|---|---|
| **scratch** | **0.614** | — |
| contrastive_ft | 0.581 | No |
| masked_ft | 0.568 | No |
| contrastive_lp | 0.499 | No |
| masked_lp | 0.350 | No |

A scale ablation with ~4× the pretraining pool (`all_corpora`) left scratch still winning — **the
null is not a data-scale artifact**. A frozen-encoder probe passed (ρ = 0.499 / 0.350 > 0), so the
null is not an artifact of a dead encoder either. At full 77-fold true LOSO, scratch reached
ρ = 0.836 (`archive/legacy_results/kimore_loso_78fold/table78.md`).

Four rounds of documented review response followed (`669aa5d`, `f9e3602`, `4959e87`, `b79df5f`,
`bc4c503`, `6e6c34e`, `e7a1df6`), adding sensor-ID probes, few-shot and partial fine-tuning, AUPRC,
per-fold AUROC CIs, a CORAL baseline, canonicalization ablation and SWAD. *Whether this manuscript
was submitted externally, and to what outcome, is **Unknown**.*

**Lesson carried forward:** this thread is why the current project treats a mean-predictor floor,
subject-disjoint splits, and a naive-feature control as non-negotiable.

### Generation 2 — the current project's original idea

Beginning `79dee71` (2026-07-10, "Add CurveNet architecture and KIMORE reproduction scripts"). The
design document is `docs/reference/PROJECT_BRIEF.md`.

**Target paper:** *"A Point Cloud Transformer for Remote Monitoring and Automated Assessment of
Physical Rehabilitation Exercises"* (arXiv:2606.30309) — CurveNet-style point-cloud geometry plus
axial self-attention regressing an exercise-quality score. Its reference implementation is vendored
at `Transformer_Rehabilitation/`.

**The diagnosis** (PROJECT_BRIEF §1) was that the target's vulnerability is *not* accuracy but that
every representational commitment discretises or flattens something natively continuous or physical:
time discretised into frame tokens on a fixed axis; geometry flattened into Euclidean ℝ³ when pose is
fundamentally rotational; occlusion fragility.

**Three proposals were generated and two were killed before any code was written** (PROJECT_BRIEF §2):

| Proposal | Paradigm | Fatal attack | Verdict |
|---|---|---|---|
| **A** — Neural Biomechanical Controller | Neural CDE + Lagrangian/Hamiltonian dynamics; impairment as recovered controller parameters | **Non-identifiable**: inferring mass matrix, potential and torque from position-only data is underdetermined; also requires double-differentiating noisy positions | **KILLED** |
| **B** — Hyperbolic + SE(3)-equivariant | Assess on ℍⁿ × SO(3) | Hyperbolic half **marginal at skeleton scale** (~25 nodes, depth ~5 — Euclidean embeds fine in 16–32 dims); distortion argument is asymptotic and irrelevant here | **Equivariance half survives; hyperbolic dropped** |
| **C** — Optimal-Transport bridge | Score = Wasserstein distance to an expert distribution | **Distribution-from-single-trajectory mismatch**; too heavy against a baseline whose brand is CPU-real-time | **KILLED** |

**The original architecture** was the synthesis: an **SE(3)-Equivariant Neural Controlled
Differential Equation**. Natural cubic spline control path → latent state carrying an orthogonal
representation ρ(g) of SE(3) → steerable (e3nn) vector field inside the CDE integrand → invariant
read-out.

**Initial assumptions, stated explicitly and worth flagging because several did not survive:**

1. Continuous time (the CDE) would buy **irregular-sampling robustness** as a structural win.
2. SE(3)-equivariance would buy **viewpoint invariance** as a theorem.
3. Placing steerable layers *inside a CDE integrand* was novel.
4. Both claims were "provable structural properties, not empirical hopes."
5. The model would stay lightweight, conceding nothing on the target's efficiency turf.

Assumption 2 held completely. **Assumption 1 did not survive contact with the corpus, and the CDE
itself was eventually dropped.**

---

## Research Evolution

### Phase A — Week-one de-risking: the equivariance certificate (2026-07-10 → ~07-13)

Before any Method section was drafted, two gates had to pass (`CLAUDE.md`, PROJECT_BRIEF §7):
**(A)** numerical integration must not break architectural equivariance; **(B)** at least one
authentically irregular data sequence must exist, so the stress test is not "engineered".

**The core theoretical lever** (PROJECT_BRIEF §5) — and the single most important insight of this
phase — is that *truncation error is largely a red herring*:

> A fixed-step explicit Runge–Kutta step is a linear combination of vector-field evaluations with
> **scalar** Butcher coefficients. Equivariant operations are closed under composition and
> scalar-weighted combination. So for a fixed-step explicit solver with a genuinely intertwining
> f_θ, equivariance holds **stage-by-stage, exactly, to floating-point roundoff, independent of
> step size.** Transformed and untransformed integrations accumulate identical truncation *in their
> own frames*.

The real danger is **adaptive step-grid divergence**: `dopri5` chooses steps from an error-estimate
norm, and the default per-component scaled RMS is **not group-invariant**. Under a rotation the
solver therefore picks a *different step sequence*, the two integrations de-align, and equivariance
dies for reasons that have nothing to do with the architecture.

The fix was a custom per-irrep-normalised invariant norm `N_eq`, validated against a guaranteed-safe
`rtol=0, scalar atol` fallback. Measured (`docs/reference/outputs_equivariance_certificate.txt`):

```
[Task 1] Orthogonality audit    max‖RᵀR − I‖_F = 1.146e-15   (≤ 1e-12 ✓)
[Gate A] Fixed-step exactness   euler 1.50e-15, rk4 1.87e-15 (≤ 1e-13 ✓), step-size independent
[Gate B] Adaptive divergence    default norm: drift 1.96e-05, D_grid_max = 91  ← diverges
                                N_eq:         drift 1.16e-15, D_grid_max = 0   ← PASS
                                drift-vs-angle flat: 1.22e-15 → 1.42e-15 over θ ∈ [0.1, π]
[Gate C] Precision scaling      fp32 vs fp64 (see certificate)
```

The **flat drift-vs-angle curve** matters as much as the magnitude: a true symmetry gives a flat
curve, a modelling violation grows with rotation angle. Together with the precision-scaling ratio,
it blocks the two standard dismissals ("you just used a tighter tolerance", "that's a bug not
roundoff").

### Phase B — the CDE fails, and the architecture pivots (~2026-07-13 → 07-19)

**This is the most consequential change in the project's history, and the original headline
architecture did not survive it.**

The Neural CDE variant was implemented (`src/cde_model.py`, `src/cde_model_mp.py`,
`src/train_cde.py`) and certified to the same standard as everything else. It then **failed to beat
the per-exercise mean-predictor floor** — a model that ignores its input entirely.

Rather than delete it, the project kept it as a **certified negative control** and says so in the
paper: *"we implement the Neural CDE variant (adaptive Dormand–Prince solver) as a certified
control that fails the floor."*

The architecture pivoted to a **discrete equivariant recurrence — the EGRU**
(`src/equivariant_gru.py`): the same steerable encoder and the same invariant cut, but a plain GRU
over the invariant code rather than an ODE solve. Viewpoint invariance — the claim that actually
held — is a property of the *cut*, not of the time-integration, so it transferred intact.

**Lesson:** the equivariance certificate was built for a model that was subsequently abandoned, and
still paid for itself, because it certified the component that survived.

### Phase C — the irregular-sampling pillar collapses, and is *explained* (~07-13 → 07-22)

Assumption 1 (continuous time buys irregular-sampling robustness) was tested as **Block 2** and came
back **null**. The critique in `docs/planning/wacv_evaluation_and_action_plan.md` is unusually
direct about why, and is worth reading in full as a model of self-audit:

- **The 4 Hz Nyquist wall.** `kimore_cde_data.load_sample` uniform-index-subsamples every recording
  to `max_len=150`. A 37.9 s recording therefore reaches the model at ~4 Hz → Nyquist ~1.9 Hz. The
  project's own spectral census puts 97.8% of exercise energy below 2.19 Hz — **the model's input
  Nyquist sits below the corner of the band claimed to carry the signal.**
- **The oracle problem.** The headline r = 1.000 tremor recovery came from `bandwidth_law.fit_amp`,
  a `lstsq` projection onto sin(2πf₀t) at *known* f₀ — an oracle matched filter, at a sample rate
  the trained network never sees. **The network was not in the experiment.**
- **Two of three arms were already grid-free.** `dt` is fed to the GRU *and* to the InvariantGRU
  baseline; only PCT is forced through resampling.

Instead of burying the null, it was **converted into a derived result — the "bandwidth law"**
(`src/bandwidth_law.py`, reported as T4). The mechanism is real and measurable:

| drop rate | resampled r | grid-free r |
|---|---|---|
| 0% | +0.998 | +1.000 |
| 30% | +0.952 | +1.000 |
| 70% | +0.785 | **+1.000** |

Irregular samples do not alias, so a tone the uniform grid folds away is recovered exactly even at
70% drop. It buys nothing *on KIMORE* because 97.8% of KIMORE's positional energy lies below the
corner that resampling destroys — resampling **denoises** here (it removes 70% of the velocity band,
which is mostly Kinect jitter amplified by differencing). **Block 2's null is predicted by the
corpus, not a refutation of the method.** The paper claims a scope boundary, not a win.

### Phase D — "why e3nn at all?" and the node-failure experiment (~07-19 → 07-22)

A tie is an obligation. If a plain GRU on six hand-crafted invariant features (`InvariantGRU`,
0.21M params) is *as accurate* and *as viewpoint-invariant* as the steerable machinery (0.66M),
the steerable machinery is unjustified. Rhetoric loses that argument; a measurement might win it.

**Block 4** (`src/joint_failure.py`) freezes *k* Kinect tracking nodes at their first-frame position
— the characteristic consumer-depth failure — and gives all arms the **byte-identical** corrupted
sequence (SHA-256 asserted equal across pathways).

| k dead nodes | EGRU | InvariantGRU | PCT | floor |
|---|---|---|---|---|
| 0 | 6.73 | 6.31 | 6.46 | 8.31 |
| 1 | 7.10 | **9.78** | 6.76 | 8.31 |
| 2 | 7.55 | 11.64 | 7.10 | 8.31 |
| 8 | 10.50 | 18.37 | 8.73 | 8.31 |
| **lost 0→8** | **+3.76** | **+12.05** | **+2.27** | — |

A **single** dead joint drives the hand-crafted model from 6.31 to 9.78 — through the floor —
because its features *name* the joints they depend on. The graph encoder pools learned messages over
25 joints, so a dead node dilutes instead of detonating.

**Reported honestly, against interest:** PCT is the *most* node-robust of the three (+2.27) —
attention can simply down-weight an anomalous point. It just has no viewpoint invariance to trade
for it. An earlier version of this experiment decided H1/H0 by comparing EGRU against InvariantGRU
only, with PCT computed, printed, and **excluded from the verdict**; the action plan flagged that as
a vulnerability and it was corrected.

### Phase E — chirality: closing the O(3) parity hole (~07-22)

The original invariant cut was O(3)-invariant, which silently assumes handedness does not matter —
false for rehabilitation, where left/right asymmetry is clinical signal. Admitting **parity-odd
pseudo-scalars** (`src/chirality.py`) narrows the group O(3) → SO(3) so the model can see trajectory
handedness, at a cost of +0.11 MAD (inside the floor) and 16.7% more parameters.

Critically, the viewpoint theorem is unaffected (det = +1 on proper rotations), and gates E6–E8 were
added to certify it. The chiral model was adopted **on that basis, not an accuracy one** — an
important distinction the paper makes explicitly.

### Phase F — deployment as a systems property (2026-07-18, `dd875b3`)

Two questions a clinical reviewer asks that a benchmark never does:

- **F7c — does the theorem survive quantization?** (`src/int8_quant_budget.py`) The invariant floor
  climbs fp64 (5.75e-16) → fp16 (2.6e-3) → bf16 (8.3e-3) → int8 (3.4e-2). **fp16 is ~3.2× tighter
  than bf16** — equivariance pays in mantissa bits, which bf16 trades away for exponent range. The
  int8 cliff is **solely activation grid-rounding**; weight-only quantization preserves the theorem.
- **F8 — can it stream?** (`src/train_stream_egru.py`, `src/ttfs_benchmark.py`) The offline model is
  O(T) (bidirectional + global pool). A causal variant (unidirectional + running mean) gives an
  honest time-to-first-score at **8.9 ms/frame ≈ 112 fps**, costing 3.17 points of NTU accuracy
  (79.81 vs 82.98).

### Phase G — second corpus, second dataset, external validity (~07-17 → 07-21)

- **NTU RGB+D Cross-View** (`src/ntu_dataset.py`): trained *without* rotation augmentation on
  cameras 2–3, tested on the displaced camera 1 → **82.98% Top-1** against a faithful ST-GCN control
  at **80.38%**. What NTU contributes is *generalisation*, not invariance — invariance there is
  **inherited** from the proposition, not separately measured. This distinction was enforced late
  and deliberately (see Debugging Milestones).
- **REHAB24-6** (`src/train_rehab246.py`, `src/load_rehab246.py`): replicates the viewpoint and
  node-failure properties on a second corpus with a different sensor (OptiTrack).

### Phase H — the live demo (2026-07-16 → 07-20)

Branch `capstone-showcase` (`d922082`, `595cc00`, `16e45c7`, `347c264`, `5684e85`, `70e577d`) — a
self-contained webcam demo carrying its own trained weights. A 3-act structure rotates the viewpoint
live: EGRU stays flat (drift 3.8e-5) while PCT collapses (8.0 of 50). Camo wireless-camera support
and a video-recording fallback were added for rehearsal safety.

### Phase I — condensation, purge, and adversarial self-review (2026-07-23 → 07-26)

- `b261332` — repository reorganised into the current directory structure.
- **17pp IEEEtran → 8pp WACV**, planned in `docs/reviews/WACV_CUT_LIST.md`. Negative results
  compressed to one sentence each in the main body with tables moved to supplementary; the viewpoint
  table promoted to a **hero figure**; the EGNN row folded into the Pareto grid.
- `a9413e7` — **"Purge unbacked invariance numbers."** Two figures (5.8e-9 NTU, 1.4e-8 streaming)
  were *inheritance arguments* with **zero backing artifacts**. They were removed everywhere and
  replaced with measured numbers (82.98 vs 80.38). This is the project's clearest instance of its
  own honesty discipline being applied retroactively.
- `6ff3486` — TCN, ST-GCN and Ridge added to the viewpoint sweep, so viewpoint fragility could be
  shown to be **generic to non-equivariance** rather than a PCT-specific defect.

### Phase J — steelmanning the baseline (2026-07-27, current)

The most recent phase inverted the usual direction of effort: instead of strengthening our model,
it systematically hunted for ways the *comparison* was tilted in our favour. See **Important
Experiments** below for outcomes. Commits `fa8ef2a`, `6f9d64e`, `3486f79`, `0326d5b`, `8ae820c`.

---

## Important Experiments

### E1 — Protocol audit: what can this benchmark even measure?

- **Goal.** Establish the resolution limit before comparing anything.
- **What changed.** `src/protocol_null.py`, `src/seed_distribution.py`, `src/determinism.py`.
- **Why.** Inherited directly from Generation 0's protocol-inflation finding.
- **Outcome.** Three numbers that gate every later claim: an **18.6% epoch-selection inflation**
  (+1.2 ± 0.4 MAD on the commonly-reported single-exercise slice); a **0.33 MAD nondeterminism
  floor** from non-reproducible cuDNN/scatter-add kernels; and a **per-exercise mean-predictor
  floor** at 8.31 MAD. On the single-exercise slice the honest number carries no reliable
  subject-level signal (bootstrap Δ = −0.44, clearing zero in only 1 of 3 seeds) — which is why the
  project pooled the five exercises.

### E2 — Clean accuracy: the tie is the result

- **Goal.** Compare architectures on the field's own metric.
- **Outcome.** No separation. Every gap inside the 0.33 floor.

| Model | Group | MAD | Params | ρ pooled | ρ within-ex | Δ vs floor (95% CI) |
|---|---|---|---|---|---|---|
| PCT (baseline) | — | 6.47 ± 0.20 | 4.91M | 0.60 | 0.55 | −1.87 [−2.72, −1.06] |
| PCT + rotation aug. | — | 7.47 ± 0.23 | 4.91M | — | — | — |
| InvariantGRU | SO(3) | 6.31 ± 0.16 | 0.21M | 0.59 | **0.56** | −2.03 [−2.89, −1.23] |
| **EGRU (ours)** | SO(3) | 6.73 ± 0.30 | 0.66M | 0.56 | 0.52 | −1.60 [−2.37, −0.88] |
| Mean-predictor floor | — | 8.31 ± 0.05 | — | **0.17** | **−0.07** | — |

The tie coexists with every model genuinely beating the floor — the intervals are all entirely
below zero. **This is the premise for the whole paper**, not an embarrassment: a benchmark on which
three architectures of vastly different inductive bias cannot be told apart is not measuring
architecture.

### E3 — Viewpoint (Block 3): the structural win

- **Goal.** Test the theorem against a rotation sweep the models never trained on.
- **Outcome.** Exact, at every angle, for weights never seen rotated.

| Model | 0° | 45° | 90° | 180° | mean degr | p95 | worst seq |
|---|---|---|---|---|---|---|---|
| EGRU SO(3) | 6.73 | 6.73 | 6.73 | 6.73 | **9.15e-06** | 2.83e-05 | 2.98e-04 |
| InvariantGRU SO(3) | 6.31 | 6.31 | 6.31 | 6.31 | **0** | 0 | 0 |
| Ridge (hand-crafted) | 7.29 | 7.29 | 7.29 | 7.29 | 9.51e-15 | — | — |
| PCT (baseline) | 6.46 | 8.46 | 10.06 | 10.38 | 9.33 | 20.66 | **33.96** |
| PCT + rot-aug | 7.47 | 7.53 | 7.67 | 7.57 | 3.01 | 8.65 | **20.12** |
| TCN | 6.26 | 8.59 | 10.38 | 12.72 | 13.17 | 23.14 | 33.64 |
| ST-GCN | 6.69 | 8.66 | 10.56 | 10.39 | 9.18 | 20.13 | 41.71 |

PCT crosses the mean-predictor floor between 30° and 45°. The pattern holds across three
independently-designed non-equivariant architectures, so **viewpoint fragility is generic to
non-equivariance**, not a PCT defect. InvariantGRU and Ridge are *positive controls* (invariant by
construction), not independent evidence.

### E4 — Node failure (Block 4): the answer to "why e3nn?"

Covered under Phase D. The discriminating experiment between two otherwise-indistinguishable models.

### E5 — The bandwidth law (Block 5): a null converted into a derived result

Covered under Phase C.

### E6 — EGNN and canonicalization sandbox (`research_egnn/`, isolated)

- **Goal.** Two reviewer questions: does *lighter* E(n)-equivariance suffice, and is cheap
  PCA-canonicalization just as good?
- **Outcome (3 seeds × 5 folds + coordinate-clamp sweep).** EGNN **ties** on clean accuracy
  (6.88 ± 0.16 vs EGRU 6.73) and is viewpoint-exact by construction behind the identical cut — but
  is **~2× more node-fail brittle** (+6.39 ± 1.27 vs EGRU's +3.76, a gap of ~3.6 SEM). A
  coordinate-clamp sweep **did not rescue it**, falsifying the "damping mitigates the feature-loss
  cliff" hypothesis; if anything the trend runs opposite.
- **Canon-PCA + PCT** matches every offline axis and is empirically ≈0 on viewpoint — but its
  guarantee is only as good as its frame estimate, which turns **discontinuous near covariance
  degeneracy**: measured 21% frame flips >45° per transition and 17.9% near-degenerate frames
  (eigen-gap < 0.05). The paper confronts this directly rather than ignoring the cheap alternative.
- **Honest caveat preserved in `FINDINGS.md`:** the EGNN is untuned, so "sabotaged-baseline" risk is
  acknowledged; this is a probe, and the sandbox is explicitly **not** part of the WACV main claims.

### E7 — Fairness sweep: three ways the comparison might be tilted (2026-07-27)

Three audits run *against our own interest*. **All three came back null**, which is the strongest
possible outcome — the holes were real, and closing them changed nothing.

| Audit | Finding | Outcome |
|---|---|---|
| **Exercise conditioning** | PCT accepted `num_exercises` and **silently ignored it**, so the baseline alone was scored blind to the exercise while EGRU one-hots it and the floor is per-exercise | Clean 6.59 ± 0.07 vs 6.47 ± 0.20; augmented 7.42 ± 0.22 vs 7.47 ± 0.23 — inside the floor, **opposite in sign**. Viewpoint swing if anything *larger* when conditioned (21.43 vs 20.12) |
| **EGRU epoch budget** | `train_egru.py` defaulted to 80 epochs vs PCT's 60 — an asymmetry in **our** favour | EGRU SO(3) 6.723 ± 0.262 @60 vs 6.734 ± 0.302 @80 (Δ = −0.010); O(3) Δ = +0.012. The extra epochs bought **nothing** |
| **Convergence** | "The baseline is just undertrained" | At 60/180/540 epochs the augmented arm's swing is **flat**: mean 3.01/3.27/2.94, worst sequence 20.1/21.4/19.9 |

A fourth — the **k=20 steelman** (`--exercise-cond --k 20` together; the reference uses a
neighbourhood of 20 while our CLI default shadowed it at 10) — also came back **null**. The
strongest form of the baseline is marginally *more* accurate (clean 6.408 ± 0.031 vs 6.465 ± 0.202;
augmented 7.315 ± 0.217 vs 7.471 ± 0.234, both inside the floor) and marginally *more*
viewpoint-fragile (clean mean degradation 10.21 vs 9.33, worst sequence 36.31 vs 33.96). The
augmented arm's degradation — the quantity the central claim rests on — does not move: 3.06 vs 3.01
mean, 20.22 vs 20.12 worst sequence. Seed spread collapses at k=20 (±0.031 vs ±0.202), a real
stability gain with no effect on the mean.

**All four fairness audits are null.** Every identified way the comparison favoured us was closed,
and closing them changed no headline number.

### E8 — Reproducing the reference paper's own protocol

- **Goal.** Explain the apparent gap between our honestly-measured PCT number and the reference's
  published one.
- **What was believed going in.** A ~35.8× discrepancy attributable to their test-set epoch selection.
- **Outcome — the premise was wrong.** Reading their vendored source settled it:
  - `engine/trainer.py:59-65` selects the saved checkpoint by **minimum test MAD**; `train.py:16`
    builds only `(train_loader, test_loader)` — **no validation split exists anywhere.**
  - `engine/trainer.py:73 evaluate_mad` (the per-epoch print *and* the selection metric) does **not**
    inverse-transform → **standardised units**. `engine/evaluator.py` (used by `eval.py`) **does** →
    **score units**. Their target scaler is a `StandardScaler` with **σ = 8.466**.
  - Running **their released checkpoint** through **their own `eval.py`**: **MAD 5.3461**.

  | | protocol | MAD (score units) |
  |---|---|---|
  | their released checkpoint | ex1, 80/20, test-selected | 5.35 |
  | **ours, single-exercise, test-selected** | ex1 slice | **5.21 ± 0.19** |
  | ours, single-exercise, honest | ex1 slice | 6.42 ± 0.44 |
  | ours, pooled subject-disjoint, honest | pooled | 6.47 ± 0.20 |

  **We reproduce them.** 5.21 vs 5.35 is inside our own 0.33 noise floor. The "35.8× gap" was a
  units mismatch stacked on a protocol difference. It had never entered the paper, so nothing needed
  retracting.
- Not leakage: KIMORE ex1 has 77 sequences ≈ one per subject, so their random split is effectively
  subject-disjoint. (This concern was raised during the audit and **withdrawn**.)
- **Their full 2000-epoch protocol, re-run from their code** (seed 145, `scripts/run_reference_protocol.sh`,
  artifacts at `Transformer_Rehabilitation/repro_s145/`). Their `eval.py` on the checkpoint their own
  rule selected: **MAD 4.0885**, which my parse of their training curve reproduces to 4 decimals
  (4.088) — the pipeline is validated end to end.

  | statistic (score units) | value |
  |---|---|
  | **min over 2000 test evals** — *their selection rule* | **4.09** |
  | median of last 200 epochs | 6.91 |
  | final epoch | 9.56 |
  | train MAD at final epoch | 1.01 |

  **The selection effect is now measured inside their own pipeline: +2.82 MAD** versus a median late
  epoch (+5.47 versus the final one). Train 1.01 against test 9.56 at the final epoch is a 9.5× gap —
  selecting the minimum of 2000 test evaluations on a 16-sequence test set is the mechanism, and the
  overfitting makes it large. **Strip the selection and their pipeline lands where ours does**: their
  6.91 against our 6.42 (single-exercise) and 6.47 (pooled, subject-disjoint).

  Their code still does not reach the published figure: **0.483 standardised against a claimed
  0.185**, short by 2.6×, at their own budget under their own test-selection. Their released
  checkpoint (5.35) is worse than this reproduction (4.09), so it is not their best run either.

  **Caveat: single seed** (145, theirs). This establishes that the selection effect exists and is
  large in their pipeline; it does not quantify how much of the specific 4.09 is seed luck.

### E9 — Spearman ρ reported against its own null (2026-07-27)

- **Goal.** KIMORE's literature reports ρ (0.74–0.965); reporting MAD alone reads as evasive.
- **What changed.** ρ computed per seed on that seed's *complete* out-of-fold vector (never per
  fold — a 15-sequence fold gives a rank correlation dominated by noise), both pooled and
  within-exercise. Raw OOF predictions banked so future rank statistics need no GPU.
- **Outcome.** The **floor's within-exercise ρ is −0.069 — statistically zero**, while its pooled ρ
  is 0.173. A predictor that ignores its input scores 0.17 purely from between-exercise score
  differences. Between-exercise variance is only **4.8%** of total, so pooling is mildly rather than
  grossly inflationary — but the null is now *measured*. The tie also replicates: ρ separates the
  models by at most 0.05. Our ρ (0.52–0.60) sits **below** the literature band, which is what the
  protocol argument predicts.

---

## Current Architecture

```
raw skeleton  (J = 25 joints, irregular timestamps t_k)
      │
      ├─ root-relative coordinates  ──────────► translation invariance is automatic
      │
      ▼
STEERABLE ENCODER  (e3nn tensor-field / message passing on the skeleton graph)
      │   per-joint latents in O(3) irreps: n₀ type-0 scalars s_j, n₁ type-1 vectors v_j
      │   equivariant:   g · h  =  ρ(g) h    with ρ(g) ORTHOGONAL (Wigner-D)
      ▼
THE INVARIANT CUT  Π   ← the load-bearing component
      │   explicit projection onto a generating set of invariants
      │   (norms, pairwise inner products, + parity-ODD pseudo-scalars for chirality)
      │   invariant:  Π(ρ(g) h) = Π(h)   for ARBITRARY WEIGHTS  → Prop. (invariance)
      ▼
RECURRENCE  (GRU over the invariant code, consuming actual inter-arrival dt)
      │   bidirectional + global mean pool   (offline, O(T))
      │   unidirectional + running mean      (causal streaming, O(1), 8.9 ms/frame)
      ▼
SCORE  (0–50 clinical scale)
```

**How it evolved to this.** The original design put steerable layers *inside a Neural CDE
integrand*. Two things changed:

1. The **CDE was dropped** — it failed the mean-predictor floor (Phase B). It survives in the repo
   as a certified negative control (`src/cde_model.py`, `src/cde_model_mp.py`).
2. The claim moved from "continuous time + equivariance" to **equivariance alone**, because Block 2
   showed the continuous-time pillar was empty *on this corpus* and the bandwidth law explained why.

What survived is the part that was always a theorem: **invariance is a property of the cut Π, not of
the time-integration.** Everything downstream of Π consumes scalars, so the score is invariant for
arbitrary weights — not approximately, not after training, not on average.

**Key components and where they live:**

| Component | File | Note |
|---|---|---|
| EGRU (current model) | `src/equivariant_gru.py` | steerable encoder + invariant cut + GRU |
| Chirality / pseudo-scalars | `src/chirality.py` | O(3) → SO(3); `LR_PAIRS` |
| PCT baseline | `src/models_curvenet.py` | reference-style point-cloud transformer |
| Hand-crafted control | `src/invariant_controls.py` | InvariantGRU, 6 named features |
| TCN / ST-GCN baselines | `src/models_stgcn.py` | non-equivariant comparators |
| Neural CDE (dropped) | `src/cde_model*.py` | certified control; fails the floor |
| Certification | `src/certify_egru.py`, `certify_mp.py`, `certify_phase1.py`, `equivariance_suite.py` | gates E1–E8 |
| **Number-of-record aggregation** | **`src/aggregate_final.py`** | **regenerates every paper table from banked JSON** |

**Working invariants (from `CLAUDE.md`, and they are enforced, not aspirational):**

- Only **orthogonal** reps ρ(g) on solver-visible state. Never expose a non-orthogonal `Linear`
  mixing irreps of unequal scale to a carried latent.
- The adaptive error norm must be **group-invariant** (`N_eq`), validated against the `rtol=0`
  fallback.
- **Relative** joint coordinates (subtract root).
- fp64 for all equivariance certification; fp32 only for the precision-scaling diagnostic.
- A failed gate is a **pivot signal**, not something to paper over.

---

## Debugging Milestones

These cost real time and are the ones most likely to bite a newcomer again.

1. **The e3nn default-dtype trap.** e3nn builds Clebsch–Gordan coefficients in the **default dtype at
   construction time**. Calling `.to(float64)` afterwards yields a *fake* fp64 certificate:
   **2.8e-8 instead of 8.5e-15**. Set the default dtype before constructing anything.
2. **The ±0.33 MAD "hardware floor" was not hardware.** It was non-deterministic atomics
   (cuDNN GRU backward + e3nn `index_add_`). After determinisation the equivariance violation is
   **bitwise zero** (G4 = 0.000e+00), which promoted dead-node invariance from a measurement to a
   **theorem**. The 0.33 figure survives as the *training* nondeterminism floor, which is a
   different quantity — do not conflate them.
3. **A mean reported as a worst case.** "3.03 MAD per-sequence" was the **mean over sequences at the
   worst angle**, and `aggregate_final` was additionally *averaging per-fold maxima* (giving 12.81 —
   "the worst sequence in a typical fold"). The true global max is **20.12**. Fixed with
   `agg_global_max`; a max of maxes needs no averaging and loses nothing. **Never fold-average a max.**
4. **A figure script silently dead.** `make_fig_guarantee.py` pointed `ROOT` at its own directory and
   had not run since the repo reorganisation — which is exactly how its hardcoded 3.03/9.42 drifted
   from the artifacts unnoticed. Figures now read `final_tables.json`; `HERE` (write) and `ROOT`
   (read) are kept distinct.
5. **A hyperparameter that loads silently and is wrong.** `pct_convergence_sweep.py` hardcoded
   `k=10` (the kNN neighbourhood the CurveNet encoder gathers over). The steelman checkpoints were
   trained at `k=20`. Because `k` changes **no weight shape**, `load_state_dict(strict=True)`
   accepted them without complaint and returned wrong predictions — angle-0 clean MAD read **8.63**
   against the training log's **6.408**, with no error raised anywhere. This is a worse failure mode
   than #6 below: that one crashed, this one produced plausible numbers. `k` is now an explicit
   argument with no inferable default, recorded in the output JSON. **Cross-check any evaluation
   script against the training log's own number for the same checkpoint** — that disagreement was
   the only signal.
6. **Making a dead argument live breaks its callers.** Commit `fa8ef2a` made `num_exercises`
   functional; every site that hardcoded `=5` and then loaded a banked 256-wide checkpoint broke.
   The *load* sites crashed loudly; the **train sites did not** — they would have silently trained a
   different model, invalidating the audit numbers the paper cites. Fixed with
   `build_pct_for_checkpoint()`, which reads conditioning off the weights.
7. **A param count that drifted from the paper.** The same change made `aggregate_final` report
   4,915,785 (4.92M) — the *conditioned* count — for models whose accuracy comes from *unconditioned*
   checkpoints (4,914,609 = 4.91M). The param column must describe the **checkpoint**, not the class.
8. **Unit conflation across codebases.** The reference implementation reports standardised units in
   its training log and score units in `eval.py`, differing by σ = 8.466. Comparing across that
   boundary inflated an apparent gap by ~8.5× (see E8).
9. **REHAB24-6 joint mapping.** A placeholder bug in the OptiTrack 26 → Kinect 25 anatomical map
   (note: `joints_names.txt` has a trailing 's' in its name).
10. **A filename-collision bug in `protocol_null`** that silently overwrote audit outputs.

---

## Remaining Challenges

**Scientific**

1. **No clinical reliability anchor.** No inter-/intra-rater ICC or κ is published for KIMORE's
   clinical Total Score or REHAB24-6's, and a targeted search (including attempts to retrieve the
   Capecci et al. full text) could not surface one. The paper flags this as **its clearest open
   measurement gap** and refuses to fabricate one, using MCID bands from cognate instruments
   (Fugl-Meyer 4–12.4) as plausibility bounds only.
2. **Small-N corpus.** 77 subjects. The action plan is explicit that *"there is no path to 80%
   acceptance on a 45-subject dataset carrying a negative Block-2 result."*
3. **Our ρ is below the published band** (0.52–0.60 vs 0.74–0.965). Defensible via the protocol
   argument, but it is a number a reviewer will react to before reading the argument.
4. **An unverified arithmetic claim about a competitor.** `paper_wacv.tex:134` argues Kuang et al.'s
   0.10–0.16 is implausibly low "even after rescaling", assuming a **linear 0–100 → 0–50** rescale.
   The reference paper's numbers turned out to be **z-scored**, not min-max. If Kuang's are too, that
   sentence's arithmetic fails the same way — and it is load-bearing for the "published numbers are
   protocol artifacts" claim. **Cannot be checked without their code. OPEN.**
5. **The steerable encoder's node-failure advantage is not fully explained.** The EGNN comparison
   establishes *that* it is more robust behind an identical cut, not *why*. Also, the EGNN is
   untuned, so a residual "sabotaged baseline" risk remains.
6. **PCT is the most node-robust model of the three** and the paper says so. The claim is a Pareto
   argument, not a dominance argument.

**Engineering**

7. Training is not bit-reproducible (nondeterministic kernels) outside the determinised
   certification path.
8. `src/` is a flat 98-module package where every script does `sys.path.insert(0, …)`. New files
   must stay at the top level.
9. `outputs/` is gitignored — all experimental artifacts are local-only and reproducible only by
   re-running the harnesses.

**Open at time of writing**

10. ~~The k=20 steelman run~~ — **RESOLVED, null.** See E7.
11. ~~The 2000-epoch reference reproduction~~ — **RESOLVED.** See E8. Single seed (145, theirs):
    establishes that the selection effect exists and is large in their pipeline, not how much of
    the specific figure is seed luck. Multi-seed remains open if a reviewer presses.
12. Paper is **9 PDF pages** — body ends on p8, references spill to p9. Compliant under the usual
    "8 pages excluding references" reading; WACV 2026's exact wording **not verified**.

---

## Future Directions

Drawn from the action plan, cut list, reviewer responses and unfinished code. **These are
reconstructions of stated intent, not commitments.**

1. **Ground both headline claims inside the forward pass.** The action plan's central pivot: the
   viewpoint claim is an algebraic identity and the grid-free claim used a NumPy oracle. `[F1c]`
   proposes a two-head model, `L = Huber(ŷ,y) + λ·Huber(â,a)`, with tremor amplitude
   `a ~ U[0.005, 0.05] m` at frequency `f₀ ~ U[4,6] Hz` **unknown to the model** — making tremor
   recovery a property of the network rather than of `numpy.linalg.lstsq`. **Status: proposed.**
2. **`[F1a]` The `dt` ablation, as a gate.** Run {EGRU, InvariantGRU} × {dt, no-dt} *before* building
   anything further on the irregular-sampling pillar. If removing `dt` changes nothing — likely at
   4 Hz — the pillar is already empty. **Status: Unknown whether run.**
3. **`[F1b]` Native-rate invariant band-power.** Per-joint velocity **norm** at the sensor's native
   ~30 Hz *before* subsampling, then sliding-window Lomb–Scargle band-power in [2, 8] Hz carried as a
   type-0 scalar. Two spec bugs are already documented: the window must be ≥1 s (a 0.25 s segment
   gives Δf ≈ 4 Hz and cannot resolve a 2 Hz edge), and it must be the **norm** of velocity, not the
   vector, or the certificate breaks.
4. **`[F2a]` Mask-aware, degree-renormalised message passing.** Gate every edge by both endpoints'
   liveness and renormalise by surviving in-degree. Since `m_i` is a type-0 scalar this is a scalar
   reweighting of an equivariant message, so the Wigner-D law is untouched — but the plan insists on
   **re-certifying with E1/E2 rather than assuming it**. Partially landed (`encoder.dead_scalar`
   exists and is read only under a mask).
5. **Real multi-camera viewpoint evidence**, replacing synthetic rotation of ground-truth skeletons.
   Partially delivered via NTU X-View; a paired-camera capture (`Variant B1`, `4de444f` — screen-to-
   webcam cross-viewpoint consistency) exists as a demo runbook but **the experiment is unrun**.
6. **Reduce the nondeterminism floor** from ±0.33 MAD toward <0.05.
7. **Tune the EGNN properly** to retire the sabotaged-baseline objection on E6.
8. **Obtain or measure a clinical reliability anchor** for KIMORE — the acknowledged top gap.

---

## Timeline

| Date | Milestone |
|---|---|
| *pre-repo* | **Generation 0** — LOSO protocol-inflation benchmark; leakage quantified at +0.026 ρ; zero-shot validity failure found (naive features beat deep models) |
| 2026-07-02 | `RESEARCH_PLAN_1.md` authored (Opin, NSU CSE capstone) |
| 2026-07-02 | `RESEARCH_PLAN_2.md` supersedes it after literature review: IRDS = IntelliRehabDS, physician-score contribution deleted, SSL framing corrected |
| **2026-07-03** | **`88fe67c` — repo begins.** SSL pretraining for zero-shot cross-sensor assessment |
| 2026-07-03 | Zero-shot eval on 77-fold LOSO models; TNSRE negative-result outline |
| 2026-07-04 | Full manuscript draft; four documented rounds of review response |
| 2026-07-05 | SWAD added; Paper 2 (**definitive negative**) closed out |
| **2026-07-10** | **`79dee71` — pivot.** CurveNet + KIMORE reproduction: the PCT baseline arrives |
| ~07-10 → 07-13 | Week-one de-risking; equivariance certificate Gates A/B/C pass; `N_eq` invariant norm solves `dopri5` step-grid divergence |
| ~07-13 | **Neural CDE fails the mean-predictor floor** → architecture pivots to the discrete EGRU; CDE retained as certified control |
| ~07-13 → 07-22 | Block 2 (irregular sampling) returns **null**; converted into the derived **bandwidth law** |
| 2026-07-16 | `d922082` — live webcam demo branch (`capstone-showcase`); Camo support, video fallback |
| 2026-07-17 | NTU RGB+D Cross-View harness |
| **2026-07-18** | `dd875b3` — **int8 precision budget (F7c)** + **causal streaming EGRU / TTFS (F8)** |
| 2026-07-19 | `6a03d3b` — paper source with edge-deployment section |
| 2026-07-20 | `70e577d` — single-model session demo with report card |
| 2026-07-21 | REHAB24-6 second-corpus trainer |
| 2026-07-22 | Chirality (O(3) → SO(3)) certified via E6–E8; determinism fix turns the ±0.33 "hardware floor" into bitwise zero |
| **2026-07-23** | `b261332` — repository reorganised; 17pp → 8pp WACV condensation planned |
| 2026-07-26 | `6ff3486` — TCN/ST-GCN/Ridge baselines: viewpoint fragility shown to be **generic to non-equivariance** |
| 2026-07-26 | `a9413e7` — **unbacked invariance numbers purged**; NTU reframed as generalisation, not invariance |
| 2026-07-26 | `4de444f` — Variant B1 paired-camera capture runbook (experiment unrun) |
| **2026-07-27** | `fa8ef2a` — exercise-conditioning fairness arm: **null** |
| 2026-07-27 | `6f9d64e` — per-sequence degradation reported as a **distribution**; true worst case 20.12, not 3.03 |
| 2026-07-27 | `3486f79` — `num_exercises` regression swept across every load and train site |
| 2026-07-27 | `0326d5b` — **Spearman ρ reported against its own null**; param count corrected to 4.91M |
| 2026-07-27 | `8ae820c` — budget-symmetry (**EGRU@60 passes**), steelman and reference-reproduction harnesses; **35.8× gap shown to be a units artifact — we reproduce the reference at 5.21 vs 5.35** |

---

## A note on the working culture, for whoever inherits this

The repository's most distinctive property is not the architecture — it is that **the project
repeatedly attacked its own results and published the outcome either way.** Concretely:

- Two of three original proposals were killed before implementation, on validity grounds.
- The headline architecture (Neural CDE) was dropped when it failed a floor, and kept as a control.
- A null result (Block 2) was explained rather than buried, and the explanation became a contribution.
- Numbers with no backing artifact were **purged from the paper** (`a9413e7`) rather than defended.
- A statistic that flattered the paper (3.03) was replaced by the honest one (20.12) — which happened
  to be *stronger*, but was changed because it was wrong, not because it was better.
- Three separate audits were run looking for ways the comparison favoured us. All three came back
  null, and all three are reported.
- The competitor's code was run rather than characterised, and the result **contradicted our own
  prior belief** about a 35.8× gap. That belief was abandoned.

`CLAUDE.md` states the rule that generated all of this: *"A failed gate is a pivot signal — stop and
diagnose, don't paper over it."* And: *"'Zero technical risk' is not the target — risk retired to
measured gates is."*

If you change a number in the paper, change it in `src/aggregate_final.py` first and let the paper
inherit it. Every table in the submission is regenerable from banked JSON by that one script, and
the two worst bugs in this project's history (the drifted figure, the drifted param count) were both
cases where that discipline lapsed.
