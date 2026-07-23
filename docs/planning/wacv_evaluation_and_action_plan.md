# WACV Evaluation & Execution Plan: Equivariant GRU (EGRU)

> Status of this document: corrected v2. Differences from v1 are marked **[CORRECTED]** /
> **[RESTORED]** / **[ADDED]** so the reasoning is auditable, not just the conclusion.

## 1. Executive Summary & Acceptance Probability

* **Status:** Reject as submitted. Major revision required.
* **Core issue:** the two headline claims — **viewpoint invariance** and **grid-free
  processing** — are each validated by an experiment that never touches the trained network.
  Viewpoint is an algebraic identity (`f(Rx) = f(x)`, proved analytically, then "measured").
  Grid-free is a NumPy least-squares oracle (`bandwidth_law.fit_amp`) fitting a tone at a
  *known* frequency, at a sample rate (30 Hz) the trained model never sees (it sees ~4 Hz).
* **The pivot:** ground both claims inside the forward pass; make occlusion-robustness a
  *theorem* rather than a slope; determinize CUDA; convert O(T) offline → O(1) streaming;
  and validate viewpoint on **real multi-camera data**, not only on our own simulator.

| Metric | Current | Target |
| :--- | :--- | :--- |
| Acceptance probability | **8%** | **45% accept / 35% borderline / 20% reject** **[CORRECTED]** |
| Run-to-run noise floor | ±0.33 MAD | < 0.05 MAD |
| Real-time engine | O(T) offline (bidirectional + global pool) | O(1) online (unidirectional + running mean) |
| Viewpoint evidence | synthetic rotation of GT skeleton | **NTU RGB+D X-View (real cameras)** + sensor-degradation simulator **[RESTORED]** |
| Node failure | "we degrade less than InvGRU" (PCT beats both) | **exact invariance to a dead node's reported value** |

**[CORRECTED] On the probability.** v1 said "45%–80%". There is no path to 80% on a
45-subject dataset carrying a negative Block-2 result. Planning against 80% is how a rebuttal
gets under-prepared. The honest post-fix number is **45% accept**.

---

## 2. Vulnerability Ledger → Fix, one-to-one

### [V1] The grid-free illusion (the 4 Hz Nyquist wall)
**Critique.** `kimore_cde_data.load_sample:79-81` uniform-index-subsamples every recording to
`max_len=150`. A 37.9 s recording reaches the model at ~4 Hz → Nyquist ~1.9 Hz. Our own
spectral census puts 97.8% of exercise energy below 2.19 Hz, i.e. **the model's input Nyquist
sits below the corner of the band we claim carries the signal.** The r = 1.000 tremor result
comes from `fit_amp` (a `lstsq` projection onto `sin(2πf₀t)` at *known* f₀) run at
`--full-len 100000` — an oracle matched filter at a rate the model never sees. We demonstrated
that Lomb–Scargle beats linear interpolation (Lomb 1976). The network is not in the experiment.

**Also fatal:** the one architectural feature implementing "grid-free" is `dt` as a GRU input
(`equivariant_gru.py:264-281`) — and `invariant_controls.py:90` feeds `dt` to the *baseline*
too. Two of three arms are grid-free. Only PCT is forced through `R`.

**→ [F1a] GATE FIRST — the `dt` ablation. [ADDED]** Run `{EGRU, InvariantGRU} × {dt, no-dt}`
**before** building anything on top of this claim. If removing `dt` changes nothing (likely, at
4 Hz), the irregular-sampling pillar is *already empty* and F1b is the only thing that can
resuscitate it. Find this out first, not third.

**→ [F1b] Native-rate invariant band-power.** Compute per-joint velocity **norm** at the
sensor's native ~30 Hz *before* subsampling, then a sliding-window Lomb–Scargle band-power in
[2, 8] Hz, and carry it as a type-0 scalar channel into the GRU.

> **[CORRECTED] Two bugs in the v1 spec of this feature:**
> 1. **Spectral resolution.** v1 said "band-power over each downsampled segment." A segment is
>    ~1140/150 ≈ 7–8 native frames ≈ 0.25 s, so Δf ≈ 1/0.25 s ≈ **4 Hz** — it cannot resolve a
>    2 Hz band edge, and the feature would be windowing artefact. Use a **sliding window of
>    ≥ 1 s (~30 native frames) centred on each retained frame**, hop = subsample stride. Same
>    output shape (150, 25); real spectral resolution.
> 2. **Velocity is a 1o vector, not a scalar.** The rotation-invariant quantity is its **norm**.
>    Take the band-power of ‖v_j(t)‖, not of v_j(t), or the equivariance certificate breaks.

**→ [F1c] Put the network in the tremor experiment.** Two-head model,
`L = Huber(ŷ, y) + λ·Huber(â, a)`, with `a ~ U[0.005, 0.05] m` at **`f₀ ~ U[4,6] Hz` unknown to
the model** (no oracle frequency). Identical Gilbert-Elliott drop mask on both arms. Report
`corr(a_true, â)` vs drop rate, EGRU-with-band-power vs PCT-after-`R`. *That* is a property of
our model rather than of `numpy.linalg.lstsq`.

### [V2] Message passing is blind to sensor occlusion
**Critique.** PCT beats us at **every** k in the node-failure sweep (k=1: 6.76 vs our 7.10;
k=8: 8.73 vs our 10.50), and at k=8 we are **above the mean-predictor floor (8.25)** — worse
than a constant. `joint_failure.py:254-263` decides H1/H0 by comparing EGRU against
**InvariantGRU only**; PCT is computed, printed, and excluded from the verdict. And
`equivariant_gru.py:154` (`agg.index_add_`) sums messages from dead nodes as if they were live.
We built the one architecture that *could* route around a dead node and never wired in the mask.

**→ [F2a] Mask-aware, degree-renormalised message passing.** Liveness `m_i ∈ {0,1}`; gate every
edge by both endpoints; renormalise by surviving in-degree:

```
h̃_j = Σ_{i∈N(j)} m_i m_j · TP(h_i, Y_ij; w_ij)  /  ( ε + Σ_{i∈N(j)} m_i m_j )
```

`m_i` is a type-0 scalar, so this is a scalar reweighting of an equivariant message: the
Wigner-D law is untouched and the viewpoint theorem survives. **Re-certify with `certify_egru`
E1/E2 — do not assume it.**

> **[CORRECTED] v1's GOAL 3 point 4 preserves garbage.** "Ensure the hidden state of an occluded
> node remains unchanged" — but `h_j⁰` is *seeded from the joint's own position*
> (`equivariant_gru.py:138-140`: `init_scalar(radius…)`, `v = x * init_gain`). For a dead joint
> `x_j` is frozen/zero, so "unchanged" means the node keeps broadcasting a **confidently wrong
> pose** into the graph forever. Needs a **learned dead-node embedding**.
>
> **Non-obvious equivariance constraint:** the dead embedding's **vector (1o) channels must be
> exactly zero**. A learned *constant* non-zero vector does not rotate with the input, so it
> would break equivariance outright. Only the 0e scalars may be learned.

> **[CORRECTED] The mask must reach the READOUT, not just the message passing.** Neither v1 nor
> the original critique said this. `InvariantProjection.forward` does `pj.mean(1)`, `pj.amax(1)`
> over joints and computes bone lengths from raw `x` — a dead joint pollutes **all three**
> (it enters the mean, it can *dominate* the max, and every bone touching it is a garbage
> length). Masked mean (sum/count), masked amax (−inf before reduction), endpoint-gated bones.

**→ [F2b] The real prize: state the invariance as a THEOREM. [ADDED]** If every path from `x_j`
to the output is gated, then

```
f(x, m) = f(x', m)   for all x, x' differing only at joints where m_j = 0
```

**exactly** — the model is *provably independent of what a failed sensor reports*. That is
categorically stronger than "degrades gracefully", it is a property PCT cannot have, and it is
testable to machine precision. Paths that must be gated: (a) `h⁰` init → dead embedding;
(b) messages on incident edges → `e_live`; (c) projection bone lengths → endpoint gate;
(d) projection pooled per-joint features → masked mean/amax; (e) speed/displacement channels →
mask. Test numerically; a failure names the leaked path.

**→ [F2c] Fix the verdict logic. [ADDED]** Add PCT to `joint_failure.py:254`. A three-arm race
scored on two arms is a selection, not a result. Also run the missing **`use_speed=False` ×
node-failure** cell: `mode="hold"` freezes joints at frame 0, which annihilates exactly the
`log1p(speed)` channels — it is entirely possible the equivariant encoder is *fine* and the
bolted-on speed channel is what dies. Report **k\*** (dead nodes at which each model crosses the
floor) for every arm.

**→ [F2d] Train with node dropout,** `k ~ U{0..4}` over the 24 non-root joints; test to k=8 so
the hard levels are extrapolation, as in Block 2.

### [V3] Non-deterministic noise floor
**Critique.** ±0.33 MAD is **20% of the entire effect size** (floor 8.25 − model 6.6 ≈ 1.65) and
is wider than every model-vs-model gap reported. Under it, "PCT 6.465 vs EGRU 6.619" is not a
tie we measured — it is a comparison we could not perform.

> **[CORRECTED] There are TWO nondeterministic kernels, not one.** v1 blames only e3nn's
> `index_add_`. The other is **cuDNN's fused GRU backward**, which also accumulates with atomics.
> Fix only the scatter and the floor will not drop — and you will wrongly conclude the fix failed.

**→ [F3a] Dense incidence, not scatter.** The graph is fixed (25 joints, 24 bones, 48 directed
edges). Precompute `A ∈ R^{J×E}`, `A[dst(e), e] = 1`; then `agg = einsum('je,nec->njc', A, m)`
is the **exact same sum** with a fixed reduction order. Deterministic, faster at this size, and
**numerically equivalent → existing checkpoints stay valid.**

**→ [F3b] Global determinism.** `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`torch.use_deterministic_algorithms(True, warn_only=False)`, `cudnn.deterministic=True`,
`cudnn.benchmark=False`, and **disable the fused cuDNN RNN path** so the GRU backward is
deterministic ATen. (`src/determinism.py`.)

**→ [F3c] What the determinism is FOR. [ADDED]** Dropping the floor is a means. The ends are:
* **Paired subject-clustered bootstrap CI** on `|e_EGRU| − |e_PCT|` (reuse
  `protocol_null.bootstrap_delta`, B = 10 000, resample **subjects**). A *stated equivalence with
  a tight interval* is a strength; an unstated tie hidden behind a noise floor is a weakness.
* **ICC(2,1) across camera azimuths**, treating the k azimuths as raters and n subjects as
  targets. EGRU = 1.000 by construction. Rotation-augmented PCT, with its 3.03 MAD per-patient
  swing, should land ≈ 0.5–0.7 — **below the 0.75 clinical acceptability threshold.** One number
  a clinician instantly understands. **Put it in the abstract.**

### [V4] The tautological viewpoint experiment
**Critique.** `block2_transforms.rotate_sample` rotates the *ground-truth* 3D skeleton. Our model
is invariant by construction, so degradation is 9e-6. This is a **unit test of `f(Rx) = f(x)`**;
it cannot fail and carries zero information. Worse, for a *deployment* venue: moving a Kinect to
180° does not give you a rotated skeleton, **it gives you a wrong skeleton** — self-occlusion,
depth collapse, limb swap, confidence dropout. Rotation is the one component of cross-view error
our theorem removes for free and the one nobody is actually bottlenecked by.

**→ [F4a] PRIMARY: NTU RGB+D 60, Cross-View (X-View). [RESTORED — v1 deleted this]**
Three *physically distinct* Kinect v2 cameras, real per-view skeletons, real per-view tracking
failure. Train on views 2+3, test on view 1. **This is the single highest-leverage item in the
plan.** A simulator — however good — is still *our* simulator, and the reply writes itself:
*"the authors invented the failure model under which they win."* You cannot answer that with a
better simulator; you answer it with real cameras and published ST-GCN/CTR-GCN numbers.
KIMORE stays as the clinical regression task. **NTU = existence proof; simulator = mechanism.**

**→ [F4b] SECONDARY: occlusion-aware sensor-degradation simulator.** Rotate, then degrade the
*tracker*: incidence-angle depth noise `σ = σ₀(1 + κ(1 − cos θ))`; self-occlusion freeze via
raycast against a capsule body model built from `KINECT_BONES`; left/right limb swap with
probability rising past ~60°.

> **[CORRECTED] v1's swap model is unrealistically i.i.d.** Per-frame independent joint swaps
> produce *flicker*, which (a) is not what a Kinect does and (b) is trivially filtered — it
> understates the difficulty. Real tracker swaps **persist for stretches**. Model them with a
> two-state Markov chain — **reuse `corruption_pipeline.gilbert_elliott_mask`** with "swapped" as
> the bad state. And swap **whole limb chains** (shoulder→elbow→wrist→hand→thumb), not individual
> joint IDs, or you produce anatomically impossible skeletons no model would be fooled by.

Expected outcome, and it is *good* for us: degradation stops being 0.000, so the experiment
becomes an experiment, and the claim upgrades to — *"equivariance removes the geometric component
of cross-view error exactly (δ = 9e-6), isolating tracker degradation as the sole residual;
under realistic degradation we lose X, augmented-PCT loses Y > X, clean PCT diverges."*
**Bonus: limb swap is a parity event** — it is precisely the failure the pseudo-scalars exist to
detect, which finally motivates V5.

### [V5] Parity-odd chirality features have no demonstrated function
**Critique.** We report that O(3) *and* SO(3) both separate left from right (4.5e-2 / 6.2e-2) —
refuting the stated clinical motivation — then charge **+16.7% parameters** and report **no
KIMORE MAD delta**. As written it reads as engineering pride.

**→ [F5a] The handedness task.** Build CW/CCW discrimination; train a linear probe on frozen
invariant features. The O(3) model's projection is parity-even, so `proj(x) = proj(Mx)`
**exactly** — the two classes are literally identical inputs to the probe.

> **[CORRECTED] Do not merely *train a probe and report ≈50%*.** For O(3) the chance result is an
> **analytic identity**: assert `proj(x) == proj(Mx)` to machine precision. Reporting "50.1%" from
> a training run invites "you undertrained it." Reporting "the features are bitwise identical, so
> *any* classifier is at chance" is unanswerable. Probe splits must be **subject-disjoint**.

**→ [F5b] Limb-swap detection (ties V5 to V4b). [ADDED]** Using F4b, inject Markov limb swaps and
ask each model to flag them. A parity-even model **cannot** distinguish a swapped skeleton from a
mirrored valid one; the pseudo-scalars can. Report detection AUROC vs swap rate for O(3) vs
SO(3). If SO(3) detects the dominant Kinect profile-view failure and O(3) sits at 0.5, the 16.7%
is bought and paid for and chirality becomes a *deployment* contribution.

**→ [F5c] Report the MAD delta** with/without chiral, post-determinism. If neutral, print it and
say so. Silence is worse than a null.

### [V6] Baselines are unmatched and un-anchored
**Critique.** Every comparison is a model we trained ourselves. No ST-GCN, no CTR-GCN, no
published KIMORE number, no capacity-matched PCT. The efficiency claim rests on a baseline we
chose to make 4.9M params on 45 subjects — which reads as *"they over-parameterised the baseline
8× and it still tied or beat them."*

**→ [F6]** (a) capacity-matched PCT ≈ 0.57M (`dim=96, spatial_depth=3, temporal_depth=2`);
(b) **ST-GCN + CTR-GCN** — they are the reference skeleton family and they come free with F4a's
NTU pipeline; (c) published KIMORE numbers with an **explicit protocol statement**. Our MAD ≈ 6.5
is subject-disjoint/pooled; much of the literature uses within-subject splits and reports far
lower error. Say it loudly, with the `protocol_null.py` leakage table, or a reviewer assumes we
are 6× worse than SOTA. **[ADDED] F6 has a phase now — v1 listed it and never scheduled it.**

### [V7] The efficiency claim is parameters, not latency **[ADDED — v1 had no F7]**
**Critique.** 7.4× smaller checkpoint ≠ faster. `o3.FullyConnectedTensorProduct(...,
shared_weights=False)` runs per-frame over B×T×48 edges, and
`e3nn.set_optimization_defaults(jit_script_fx=False)` (line 61) *disables the optimiser we need*.
At batch = 1 this is dozens of launch-bound kernels. **Our 0.57M model may well be slower in
wall-clock than the 4.9M PCT**, and a reviewer who suspects that and finds no latency table wins.

**→ [F7a] Measure it.** Jetson Orin Nano 8GB + Raspberry Pi 5 (CPU). Batch = 1, 200 warmup +
2000 steady frames, `cuda.synchronize()` per frame. Report **p50/p95/p99** (not the mean),
**deadline-miss rate at 33.3 ms** (30 Hz Kinect), peak RSS/VRAM, **mJ/frame** (tegrastats INA3221
rails), and **time-to-first-score**. Benchmark PCT in its *sliding-window deployment*, stride ∈
{1, 5, 10, 25} — give it its best case, as we did with `R` in Block 2.

**→ [F7b] Optimisation ladder** (do in order): dense incidence (already in F3a) → re-enable
`jit_script_fx=True` + `torch.compile(mode="max-autotune")` → **CUDA Graphs** for the fixed-shape
batch-1 per-frame step (typically 3–10× when launch-bound) → ONNX/TensorRT FP16.
*Do not* distil the encoder into an MLP: that discards the theorem, which is the paper.

**→ [F7c] The novel result hiding here: equivariance has a PRECISION BUDGET. [ADDED]**
Quantisation is **not** an orthogonal transformation, so it does not commute with ρ(g) and the
theorem has a numerical floor nobody has published for e3nn on edge hardware. Point the existing
Task-6 precision-scaling instrument (`certify_egru`) at the deployment stack:

```
δ_eq(π) = max_{g ∈ SO(3)} ‖ s_π(g·x) − s_π(x) ‖_∞ ,   π ∈ {fp64, fp32, fp16, bf16, int8}
```

swept over 64 azimuths × the test set, **reported in MAD units on the 0–50 clinical scale** so it
is directly comparable to the noise floor and to PCT's 3.03 MAD viewpoint swing. Expected:
exact in fp64, ~1e-3 fp32, ~1e-1 fp16, destroyed at int8 → *"deployable equivariance has a
16-bit precision floor."* A theorem, its numerical budget, and the hardware config under which it
holds. This is the one thing in the paper no reviewer can call a tautology.

### [V8] The streaming claim contradicts the code
**Critique.** `equivariant_gru.py:252` — `bidirectional=True`, followed by a masked mean-pool over
the full sequence. **A bidirectional GRU cannot stream**: the backward pass needs the last frame.
We have an O(T) offline model. Claiming real-time while shipping this is the discrepancy that
turns a borderline review hostile.

**→ [F8] Unidirectional + causal running mean.**
`h̄_k = h̄_{k-1} + (1/k)(h_k − h̄_{k-1})`, score emitted at **every** frame k, reproducing the
offline number exactly at k = T. Report the MAD cost; if it is inside the (now-tiny) noise band,
we have a genuinely streaming clinical scorer — **arguably a stronger applications story than the
equivariance itself.** The deployment table then reads:

| | PCT | EGRU (unidirectional) |
| :-- | :-- | :-- |
| State | T=100 frame buffer | `(h ∈ R^128, k, h̄)` |
| Memory | O(T·J·3) | **O(1)** |
| Cost / score | O(T·J²) attention ÷ stride | **O(J + E)** |
| **Time-to-first-score** | **20–38 s (buffer fill)** | **1 frame (~33 ms)** |
| Update rate | once per stride | **every frame** |

The transformer cannot emit *any* score until the exercise is over. TTFS, not FLOPs, is the
number a clinician feels.

---

## 3. Execution Timeline

**Phase 0 — the gate (do before anything else). [ADDED]**
`F1a` the `dt` ablation. If `dt` is inert, the irregular-sampling claim is already dead and F1b
is its only life support. Knowing this changes what Phase 2 is for.

**Phase 1 — stabilisation & determinism.** `F3a` dense incidence (numerically equivalent →
checkpoints survive), `F3b` global determinism (**both** kernels), re-measure the seed spread,
then `F3c` bootstrap CI + ICC(2,1). *Everything downstream is uninterpretable until the floor is
real.*

**Phase 2 — streaming & bandwidth.** `F8` unidirectional + running mean. `F1b` native-rate
band-power (≥1 s sliding window; norm, not vector). `F1c` two-head tremor recovery through the
network.

**Phase 3 — occlusion & stress.** `F2a` mask-aware MP + dead-node embedding + masked readout;
`F2b` prove the exact dead-node invariance; `F2c` fix the verdict logic + `use_speed` ablation;
`F2d` node-dropout training. `F4b` sensor simulator (Markov limb-swap on chains).

**Phase 4 — external validity.** `F4a` **NTU RGB+D X-View** + `F6` ST-GCN/CTR-GCN/capacity-matched
PCT (one pipeline, both deliverables). `F5a/b` chirality tasks.

**Phase 5 — deployment.** `F7a` Jetson/Pi latency + energy table; `F7b` optimisation ladder;
`F7c` the fp16/int8 equivariance precision budget.

**Phase 6 — writing.**

### Dependency notes
* Phase 4 (NTU) is the **highest-leverage** item but the longest lead time — **start the data
  download and loader in parallel with Phase 1.** Do not serialise it behind Phase 3.
* `F5b` (limb-swap detection) *depends on* `F4b` (the simulator that generates swaps). Chirality
  cannot be justified until the simulator exists.
* `F7c` reuses the existing `certify_egru` harness — cheap, and it is the most novel single
  result available. Do not let it slip to the end.
