# Project Brief — SE(3)-Equivariant Neural CDE for Rehabilitation Assessment

> Consolidated design document. Captures the full arc of the research discussion:
> target-paper analysis → paradigm selection → winning approach → mathematics →
> week-one de-risking protocol. This is the context/spec for the codebase.

---

## 0. TL;DR

We are writing a top-tier paper (CVPR/ICCV/NeurIPS/MICCAI) that **outperforms** the target
paper below — not incrementally, but by a paradigm shift.

- **Target paper:** *"A Point Cloud Transformer for Remote Monitoring and Automated
  Assessment of Physical Rehabilitation Exercises"* (arXiv:2606.30309). It uses
  CurveNet-style point-cloud geometry + axial self-attention to regress a 0–1 exercise
  quality score from skeleton joints.
- **Our paradigm:** an **SE(3)-Equivariant Neural Controlled Differential Equation (CDE)**.
  Assessment becomes an **equivariant flow on the pose manifold, in continuous time**.
- **Two structural wins the target cannot contest:**
  1. **Viewpoint invariance** — guaranteed by SE(3)-equivariance (a *theorem*, not learned).
  2. **Irregular-sampling robustness** — native to the CDE formulation (no fixed frame grid).
- **We stay lightweight** (first-order flow; no optimal-transport bridge, no heavy solver),
  so we do NOT surrender the target's home-turf efficiency claim.
- **Strategy:** never fight on their turf (marginal MAD on clean KIMORE). Win on
  capability axes where the gap is *qualitative*.

**Current phase:** Week-one de-risking. Two gates must pass before we draft the Method section:
(A) numerical integration must not break architectural equivariance; (B) we must have at least
one authentically irregular data sequence so the stress test isn't "engineered."

---

## 1. Why the target paper is beatable (the diagnosis)

Its true vulnerability is not accuracy — it is that **every representational commitment is a
discretization or flattening of something natively continuous, hierarchical, or physical:**

- **Time is discretized** into frame tokens on a fixed axis (axial attention). No principled
  handling of dropped frames / variable frame rate / jitter — the real Kinect deployment regime.
- **Geometry is flattened** into Euclidean R³ (dot-product attention, cosine similarity), but
  pose is fundamentally rotational (SO(3) per joint).
- **Health is a black-box scalar** — a regression output with no model of what a healthy body
  *does*; feedback is post-hoc gradient attribution (Integrated Gradients), which the paper
  itself notes suffers from saturation artifacts.
- **Occlusion fragility** — point clouds + attention degrade when joints vanish, which Kinect
  does constantly.

We attack **time + geometry** at the foundation; we deliberately do NOT try to recover physical
forces (see §3, killed proposal A) because that path is non-identifiable.

---

## 2. The three proposals we generated, and why we killed two

| Proposal | Paradigm | Fatal attack (meta-review) | Verdict |
|---|---|---|---|
| **A — Neural Biomechanical Controller** | Neural CDE + Lagrangian/Hamiltonian dynamics; infer impairment as recovered controller parameters | **Non-identifiable:** jointly inferring mass matrix, potential, and torque from *position-only* data is underdetermined (no force/EMG ground truth). Also requires double-differentiation of noisy positions → amplifies sensor noise. | **KILLED** (core validity flaw) |
| **B — Hyperbolic + SE(3)-Equivariant Geometry** | Assess on ℍⁿ × SO(3): hyperbolic for the kinematic tree, equivariant for pose | **Hyperbolic half is marginal at skeleton scale** (~25 nodes, depth ~5 → Euclidean embeds fine by 16–32 dims). Distortion argument is asymptotic/irrelevant here. Stapling critique: if equivariance is the win, why hyperbolic? | **Equivariance half SURVIVES; hyperbolic half dropped** |
| **C — Optimal-Transport Bridge to a "health manifold"** | Score = Wasserstein/Schrödinger-bridge distance to expert distribution; transport map = corrective feedback | **Distribution-from-single-trajectory mismatch** (OT needs distributions; clinic gives one trajectory → high-variance empirical Wasserstein). **Too heavy** (unstable, compute-hungry) vs. a paper whose brand is CPU-real-time. Monotonicity experiment is mildly circular. | **KILLED** (data mismatch + efficiency contradiction) |

**The winner is a synthesis of the surviving organs:** keep SE(3)-equivariance (from B),
keep continuous-time CDE dynamics (from A) **minus** the identifiability-killing force
decomposition, drop hyperbolic (B) and OT (C entirely).

→ **SE(3)-Equivariant Neural CDE.** Both its claims are *provable structural properties*
(equivariance by architecture; irregular-sampling by formulation), not empirical hopes. It is
lean, defensible, and genuinely novel for rehab assessment.

---

## 3. The winning method — mathematics

### 3.1 State and group

A body pose is a point on configuration manifold $\mathcal{M} = SE(3)^J$ (or $SO(3)^J \times \mathbb{R}^3$)
for $J$ joints. A rigid motion $g=(R,t)\in SE(3)$ acts on raw joint observations $x\in\mathbb{R}^{3\times J}$:

$$g \cdot x = R\,x + t\,\mathbf{1}^\top.$$

Latent state $Z(\tau)$ carries a representation $\rho$ of $SE(3)$: $g\cdot Z = \rho(g)Z$.

**Invariance requirement (the assessment score):**

$$s(g \cdot x) = s(x) \quad \forall g \in SE(3).$$

Achieved by making every intermediate layer **equivariant** and the final read-out **invariant**.

### 3.2 Continuous-time model (Neural CDE core)

Given irregular observations $\{(x_{t_k}, t_k)\}$, fit a **natural cubic spline** control path
$X:[t_0,t_N]\to\mathbb{R}^{3\times J}$ (differentiable, defined for arbitrary spacing — this is
what buys irregular-sampling robustness). The latent evolves as a controlled differential equation:

$$Z(\tau) = Z(t_0) + \int_{t_0}^{\tau} f_\theta\big(Z(s)\big)\, dX(s)
= Z(t_0) + \int_{t_0}^{\tau} f_\theta\big(Z(s)\big)\,\frac{dX}{ds}\,ds.$$

Assessment is a functional of the terminal state: $s = h_\psi\big(Z(t_N)\big)$.

There is **no frame axis to align** — the CDE integrates against whatever data arrived, whenever.

### 3.3 The intertwining constraint (equivariance of the flow)

For the pipeline to be SE(3)-invariant we need the flow to be equivariant: if $X\mapsto g\cdot X$,
then $Z\mapsto\rho(g)Z$. The **sufficient condition on the vector field**:

$$f_\theta\big(\rho(g)\,Z\big)\cdot\big(R\,\tfrac{dX}{ds}\big) \;=\; \rho(g)\Big[f_\theta(Z)\cdot\tfrac{dX}{ds}\Big]
\qquad \forall\, g=(R,t)\in SE(3).$$

**How we satisfy it:**
- $f_\theta$ operates only on **relative** joint geometry (differences $x_i-x_j$ and their
  spherical-harmonic embeddings) → **translation invariance** is automatic (constant killed by $dX$).
- Rotational order preserved via **steerable / tensor-field layers** (Clebsch–Gordan-constrained
  weights that commute with the group action, à la e3nn) → **rotation equivariance**.
- Placing steerable layers *inside a CDE integrand* is (to our knowledge) new.

Proving equivariance survives numerical integration is a **core contribution** (see §5).

### 3.4 Loss functions

$$\mathcal{L} = \mathcal{L}_{\text{Huber}}(s,y)
\;+\; \lambda_{\text{eq}}\,\mathcal{L}_{\text{equiv}}
\;+\; \lambda_{\text{smooth}}\,\mathcal{L}_{\text{path}}
\;+\; \lambda_{\text{cons}}\,\mathcal{L}_{\text{consistency}}.$$

**(1) Score (robust regression to clinician labels — matches target metric for fair comparison):**

$$\mathcal{L}_{\text{Huber}}(s,y)=\begin{cases}\tfrac12(s-y)^2 & |s-y|\le\delta\\ \delta|s-y|-\tfrac12\delta^2 & \text{otherwise.}\end{cases}$$

**(2) Equivariance regularizer (soft certificate; ≈0 by construction — reporting it near
machine-epsilon is evidence for reviewers):**

$$\mathcal{L}_{\text{equiv}} = \mathbb{E}_{g\sim SE(3)}\big[\,\|Z_{g\cdot X}(t_N)-\rho(g)Z_X(t_N)\|^2\,\big].$$

**(3) Temporal-path smoothness (soft physical prior; bounded-velocity latent flow — survives
without any force claim):**

$$\mathcal{L}_{\text{path}} = \int_{t_0}^{t_N}\big\|\tfrac{d}{ds}Z(s)\big\|^2\,ds.$$

**(4) Sampling-consistency (operationalizes the irregular-sampling advantage — force terminal
states of the same motion under different sampling to agree):**

$$\mathcal{L}_{\text{consistency}} = \big\|h_\psi(Z_X(t_N)) - h_\psi(Z_{X'}(t_N))\big\|^2,$$

where $X'$ is a randomly sub-sampled/jittered copy of the same motion.

---

## 4. Why it beats the target (structural, not decimal)

- **Kills discrete-time flaw:** Neural CDE is *defined* on continuous time / irregular samples —
  native data model, not a patch. Baselines must re-interpolate onto a fixed grid and degrade.
- **Kills Euclidean-mismatch flaw (viewpoint):** SE(3)-equivariance *guarantees* invariance to
  camera azimuth / body placement. Target learns it from data and decays off-distribution.
- **Occlusion robustness:** missing joints become gaps in the control path the ODE integrates
  through using learned dynamics as prior — more graceful than attention over a corrupted cloud.
- **Stays lightweight:** first-order flow, no OT bridge → we keep the efficiency turf.
- **Interpretability without the trap:** we make *no* force/torque claims (avoiding proposal A's
  non-identifiability); feedback comes from the equivariant attention/attribution over joints.

---

## 5. THE CORE THEORETICAL LEVER — equivariance drift

**Key insight (this shapes the whole verification and the solver config):**
**Truncation error is largely a red herring for equivariance. The real danger is adaptive
step-grid divergence.**

### 5.1 Fixed-step solvers preserve equivariance *exactly*

An explicit Runge–Kutta step is a linear combination of vector-field evaluations with *scalar*
Butcher coefficients. Equivariant ops are closed under composition and scalar-weighted
combination. So for a **fixed-step** explicit solver (Euler/midpoint/RK4) with a genuinely
intertwining $f_\theta$ and equivariant initial state, equivariance holds **stage-by-stage,
exactly, to floating-point roundoff — independent of step size**. Truncation corrupts trajectory
*accuracy*, but transformed/untransformed integrations accumulate identical truncation in their
frames, so the relation $Z_{g\cdot X}=\rho(g)Z_X$ stays exact.

Euler induction:

$$Z_{n+1}=Z_n+h\,f_\theta(Z_n)\Delta X_n
\xrightarrow{X\to g\cdot X}
\rho(g)Z_n+h\,f_\theta(\rho(g)Z_n)(R\Delta X_n)=\rho(g)\big[Z_n+h f_\theta(Z_n)\Delta X_n\big].$$

### 5.2 Where equivariance genuinely CAN break

1. **Adaptive solvers (`dopri5`)** — the step controller picks steps from an error-estimate norm.
   If that norm is not group-invariant, $g\cdot X$ gets a *different step sequence* → integrations
   de-align → equivariance dies. **This is the real risk.**
2. **Non-orthogonal representation $\rho(g)$** — mixing feature types of different scales under a
   non-unitary rep breaks norm invariance and injects real violation.
3. **Non-equivariant spline construction** — must verify spline$(g\cdot X)=g\cdot$spline$(X)$.
   Natural cubic splines are linear in control points and reproduce constants, so rotation commutes
   and translation passes through (then killed by $dX$). ✔

### 5.3 Drift metric

For sampled $g\in SE(3)$ and path $X$, with $\mathrm{Solve}_\Theta(X)=Z_X(t_N)$ under solver
config $\Theta=(\text{method},\text{atol},\text{rtol},\text{dtype})$:

$$\Delta_{\text{eq}}(g,X)=\big\|\mathrm{Solve}_\Theta(g\cdot X)-\rho(g)\mathrm{Solve}_\Theta(X)\big\|_2,
\qquad
\delta_{\text{eq}}(g,X)=\frac{\Delta_{\text{eq}}(g,X)}{\|\mathrm{Solve}_\Theta(X)\|_2+\varepsilon}.$$

Aggregate over Haar-uniform group samples and real motions:

$$\bar\delta_{\text{eq}}(\theta_{\max})=\mathbb{E}_{g\sim\mathrm{Haar}(SE(3)),\,\|\log R\|\le\theta_{\max}}\;\mathbb{E}_{X\sim\mathcal{D}}\big[\delta_{\text{eq}}(g,X)\big].$$

**Critical plot:** $\bar\delta_{\text{eq}}$ vs. $\theta_{\max}$. True symmetry → **flat** curve; a
modeling violation → **grows with rotation magnitude**.

### 5.4 The decisive diagnostic: float32 vs float64

- Drift drops ~7 orders (≈$10^{-6}\to10^{-13}$) fp32→fp64 ⇒ **pure roundoff, architecture exactly
  equivariant** (the result we want).
- Drift barely moves ⇒ **genuine symmetry violation** (bug to fix before any paper).
- Roundoff accumulates as $\sim\sqrt{N_{\text{steps}}}\,\epsilon_{\text{mach}}$ ⇒ also confirm
  sub-linear ($\sqrt N$) growth with sequence length.

### 5.5 The unassailable certificate = three joint conditions

1. **Magnitude:** fp64 relative drift $\bar\delta_{\text{eq}}\le 10^{-11}$ (target ~$10^{-13}$).
2. **Invariance to $\theta_{\max}$:** drift-vs-angle curve flat (slope ≈ 0).
3. **Precision scaling:** ~7-order fp32→fp64 drop (proves roundoff, not violation).

Contrast row (target Point Cloud Transformer): equivariance error $O(10^{-1}\!-\!10^{0})$, **grows
with rotation angle**, **invariant to float precision** (modeling gap, not roundoff). Flat-at-ε vs.
rising-at-$O(1)$ is the kill-shot; conditions (2)–(3) block the "you just used a tighter tolerance"
dismissal.

---

## 6. Orthogonal representation diagnostics (why orthogonality ALONE isn't enough)

### 6.1 The `dopri5` scaled-norm subtlety

`dopri5` accepts a step when a **per-component scaled RMS** of the embedded error $e$ (4th–5th
order difference) is below tolerance:

$$\mathcal{N}(e)=\sqrt{\frac1n\sum_i\Big(\frac{e_i}{\text{atol}+\text{rtol}\cdot\max(|y_i|,|\hat y_i|)}\Big)^2}=\frac{1}{\sqrt n}\|We\|_2,\quad W=\mathrm{diag}(w_i).$$

**Step 1 — error is equivariant:** $e$ is a scalar-weighted combo of equivariant stage evals ⇒
$X\to g\cdot X \Rightarrow e\to\rho(g)e$.

**Step 2 — norm invariance condition:** $\mathcal{N}(\rho(g)e)=\mathcal{N}(e)\;\forall e$ iff

$$\rho(g)^\top W^2\rho(g)=W^2 \iff [\rho(g),W^2]=0\ \ (\rho(g)\text{ orthogonal}).$$

**Step 3 — factorizes into two conditions:**
- **(A) Orthogonality:** if $W\propto I$, reduces to $\rho(g)^\top\rho(g)=I$. Wigner-D matrices
  (type-0 identity, type-1 rotations, higher-ℓ real Wigner-D) are orthogonal ⇒ ✔ automatically in e3nn.
- **(B) Commuting scaling — the subtle killer:** with the **default per-component** $W$,
  $[\rho(g),W^2]=0$ is generically **false** even for orthogonal $\rho(g)$. A type-1 block rotates a
  vector's 3 components, but the default scaling weights them differently ($\max(|y_i|,|\hat y_i|)$
  per component) ⇒ $W^2$ not constant within the rotating triple ⇒ different step grids ⇒ drift.

**Crux:** a rotation preserves only the **L2 norm of each irrep block**, not individual component
magnitudes. So the ONLY exactly-invariant error norms are built from **per-irrep block norms**.

### 6.2 The fix — manifestly invariant, isotropic error norm

$$\mathcal{N}_{\text{eq}}(e)=\sqrt{\frac1F\sum_{f=1}^{F}\frac{1}{d_f}\|e_f\|_2^2},\qquad
e_f=\text{sub-vector for irrep }f,\;d_f=\dim(\text{irrep }f).$$

Invariance is immediate ($D_f(g)$ orthogonal ⇒ $\|D_f(g)e_f\|_2=\|e_f\|_2$; sum of invariants).
The $1/d_f$ factor makes it **isotropic across feature types** (each *feature* contributes equally
regardless of dimension).

**Two safe solver configurations:**
1. **Principled (preferred):** pass $\mathcal{N}_{\text{eq}}$ as the solver error norm (torchdiffeq
   supports a custom norm via options/adjoint hooks — **exact keyword is version-dependent; verify
   against installed version**).
2. **Guaranteed-safe fallback:** **rtol = 0, scalar atol** ⇒ $W=\tfrac{1}{\text{atol}}I$ commutes
   with everything. Costs relative-tolerance adaptivity but is bulletproof; use as the control that
   MUST pass. If the custom-norm path matches it, the custom norm is validated.

### 6.3 Constructing $Z$ for an isotropic space

- **Block-orthogonal $\rho(g)$ only:** assemble solver-visible state purely from irrep features so
  $\rho(g)=\bigoplus_f D_f(g)$ is orthogonal. **Never** expose a non-orthogonal `Linear` (mixing
  irreps with unequal scale) to the *integrator's carried latent* — those belong INSIDE $f_\theta$'s
  tensor-product layers. **Most likely silent-break location; audit explicitly.**
- **Equivariant per-irrep normalization:** normalize each feature by a **scalar gain of its own
  invariant norm** (equivariant LayerNorm): $\tilde z_f=\gamma_f z_f/(\|z_f\|_2+\epsilon)$, $\gamma_f$
  a type-0 scalar ⇒ commutes with $D_f(g)$; equalizes scales so no type dominates the error norm.
- **Match norm block structure to state block structure:** the $d_f$ partition in
  $\mathcal{N}_{\text{eq}}$ must equal the irrep partition of $Z$; assert at construction.

---

## 7. Week-One Operational Runsheet

Dependency-ordered. Tasks 1→7 = equivariance/solver track (each gates the next). Task 8 runs **in
parallel from day 1** (acquisition lead time). Tasks 9–10 depend on 8. "Pivot" = structural
failure, stop before writing Method.

| # | Objective | Verification Metric | Pass/Fail Threshold | Dep |
|---|---|---|---|---|
| 1 | **Rep orthogonality audit.** Every solver-visible block uses orthogonal Wigner-D; no non-orthogonal `Linear` on carried latent. | $\max_{g}\|\rho(g)^\top\rho(g)-I\|_F$ | **≤ 1e-12 (fp64)** / ≤1e-6 (fp32). Fail→refactor into $f_\theta$; **pivot if unfixable.** | — |
| 2 | **Fixed-step algebraic-exactness in code.** Euler/midpoint/RK4 preserve equivariance to roundoff, step-size-independent. Mock $f_\theta$. | $\delta_{\text{eq}}$ at $h\in\{T/50,T/100,T/500\}$ | **fp64 $\delta_{\text{eq}}$ ≤ 1e-13 at every $h$**; $\max_h\le 2\times\min_h$. Fail→intertwining bug. | 1 |
| 3 | **Custom invariant norm $\mathcal{N}_{\text{eq}}$** implemented, unit-tested vs rtol=0 control. | $\|\mathcal{N}_{\text{eq}}(\rho(g)e)-\mathcal{N}_{\text{eq}}(e)\|/\mathcal{N}_{\text{eq}}(e)$ | **≤ 1e-13 (fp64)**; must match rtol=0 drift within 1 order. | 1 |
| 4 | **Adaptive-solver drift sweep.** `dopri5` × atol,rtol∈{1e-3…1e-9} with (a) default norm, (b) $\mathcal{N}_{\text{eq}}$. | $\bar\delta_{\text{eq}}(\theta_{\max})$ vs rotation magnitude; curve slope | With $\mathcal{N}_{\text{eq}}$: **fp64 ≤ 1e-12** AND slope≈0 (flat). Default norm expected to fail (validates 6.1B). | 3 |
| 5 | **Step-grid divergence log.** Matched step sequences for $X$ vs $g\cdot X$. | $D_{\text{grid}}=|\mathcal{T}(X)\triangle\mathcal{T}(g\cdot X)|$ | **$D_{\text{grid}}=0$ exactly** under $\mathcal{N}_{\text{eq}}$; >0 under default (documents failure). | 4 |
| 6 | **Precision-scaling diagnostic.** Prove residual = roundoff. | $r=\bar\delta_{\text{eq}}^{fp32}/\bar\delta_{\text{eq}}^{fp64}$; length-scaling fit | **$r\ge 10^5$** (≈7 orders); length exponent $0.5\pm0.2$. Fail ($r\approx1$)→hidden violation, back to Task 1. | 4 |
| 7 | **Real-layer swap-in.** Mock → real e3nn/steerable layers; re-run 2,4,6. | Same $\delta_{\text{eq}},\bar\delta_{\text{eq}},r$ on real arch | **All Task 2/4/6 thresholds hold with real layers.** Fail→a real layer non-orthogonal; isolate via Task 1 per-layer. | 2,4,6 |
| 8 | **P0 — acquire ≥1 authentically irregular sequence.** Record Kinect/webcam under CPU load, or source streamed clinical capture with true arrival timestamps. | $\text{CV}=\sigma_{\Delta t}/\mu_{\Delta t}$; contiguous gaps | **CV ≥ 0.2** and ≥1 burst gap ≥3 frames. Fail→keep sourcing; do NOT rely on synthetic-only for Exp 2. | — (parallel) |
| 9 | **Corruption pipeline** (Gilbert-Elliott drops + Gaussian/AR(1) jitter + Gamma inter-arrivals) + spectral-loss quantification. | $\mathcal{I}_{\text{loss}}(\mathcal{R})=\|\widehat{\mathcal{R}x}(\omega)-\hat x(\omega)\|^2_{L^2(\omega>\omega_c)}$ in tremor band | **$\mathcal{I}_{\text{loss}}(\text{linear interp})>0$ measurably**; synthetic CV/burst stats match Task-8 real seq within 20%. | 8 |
| 10 | **Fairness smoke test.** Identical $(\tilde t_k,x_k)$ to both; baseline forced through best-case $\mathcal{R}\in\{$linear,cubic,zero-fill$\}$, ours via spline-on-actual-stamps. | Input-identity hash; degradation $|s^{(m)}_{\text{corrupt}}-s^{(m)}_{\text{clean}}|$ | **Hash identical** for both (no input asymmetry) AND full pipeline produces degradation-vs-level curve. | 9 |

**Go/no-go for Method drafting:** Tasks 1–7 green (certificate real + survives real architecture)
AND Task 8 green (real irregular anchor). Tasks 9–10 green → Experiment 2 build-ready. **Pivot
signal:** Task 6 returns $r\approx1$ or Task 7 fails.

---

## 8. Authentically irregular data pipeline (details)

Every corruption is a published, physically-motivated stochastic model of a named failure,
applied to **timestamps**, producing $(\tilde t_k, x_k)$ pairs. Both methods receive **identical**
corrupted data.

### 8.1 Frame drops — Bernoulli thinning + Gilbert–Elliott bursts

Pure Poisson is wrong (real drops cluster). Two-state Markov chain (Good/Bad):

$$P=\begin{pmatrix}1-p_{GB}&p_{GB}\\p_{BG}&1-p_{BG}\end{pmatrix},\;
\Pr(\text{drop}\mid G)=\epsilon_G\approx0,\;\Pr(\text{drop}\mid B)=\epsilon_B\approx1.$$

Stationary bad-state prob $\pi_B=\frac{p_{GB}}{p_{GB}+p_{BG}}$ sets drop rate; $1/p_{BG}$ sets mean
burst length. Sweep effective drop 10%→70%. Produces contiguous, hostile dropouts.
(Independent thinning — geometric gaps — kept as a baseline-realism control.)

### 8.2 Timestamp jitter — Gaussian thermal + AR(1) oscillator drift

$$\tilde t_k = t_k + \eta_k + d_k,\quad \eta_k\sim\mathcal{N}(0,\sigma_j^2),\quad
d_k=\alpha d_{k-1}+\xi_k,\;\xi_k\sim\mathcal{N}(0,\sigma_d^2).$$

**Monotonicity constraint:** enforce $\tilde t_k<\tilde t_{k+1}$ via clamping $\sigma_j\le\tfrac13\Delta t$
and rejection/projection of order violations (never create impossible reordering).

### 8.3 Variable frame rate / streaming latency — Gamma inter-arrivals

$$\Delta_k\sim\mathrm{Gamma}(\kappa,\theta),\quad \mathbb{E}[\Delta_k]=\kappa\theta,\;\mathrm{Var}=\kappa\theta^2,$$

mean modulated by an OU-style throttling process (30→10 fps episodes). Heavy right tail =
occasional latency spikes; small $\kappa$ = burstier arrivals.

### 8.4 Fairness architecture (steelman baseline, then win)

- **Baselines (fixed-grid):** their published architecture requires uniform $T\times J\times 3$;
  they MUST apply a resampling operator $\mathcal{R}$. Give them their **best case** — report
  **linear interp** (standard), **cubic** (stronger), zero/forward-fill (lower bound). Beating their
  best workaround defuses "sabotage."
- **Ours (CDE):** spline on the **actual** $\tilde t_k$, integrate on $[\tilde t_0,\tilde t_N]$. No resampling.
- **Why fair yet devastating (quantified):** resampling onto a grid is a low-pass operator; measure
  the spectral energy it destroys above cutoff $\omega_c$ — exactly the **tremor/jerk band** that is
  clinically decisive. The baseline provably destroys diagnostic signal by a mechanism intrinsic to
  its fixed-grid assumption.
- **Metric = self-referential degradation, per method:**

$$\text{Degradation}(m,\text{level}) = \big|s^{(m)}_{\text{corrupt(level)}} - s^{(m)}_{\text{clean}}\big|.$$

  Each method measured against its OWN pristine score — no cross-method scaling, no baked-in
  advantage. Sweep corruption level on x-axis; our flat curve vs. their rising curves is the figure.
- **Pre-empt "engineered":** (i) cite the target paper's own discussion of Kinect frame-drop/jitter
  to establish irregular sampling as the real deployment regime; (ii) **P0: acquire ≥1 real
  irregular sequence** to anchor the synthetic sweep.

---

## 9. Baseline & experiment strategy (results table)

**Rule:** do not fight on clean-benchmark decimals. Win on capability axes.

**Mandatory baselines (re-run, identical splits/protocol):** target Point Cloud Transformer
(CurveNet + axial attention), Deb et al. GCN, Mourchid et al. D-STGCNT, ST-GCN.
**Datasets:** KIMORE, UI-PRMD (regression), IRDS (classification). **Metrics:** MAD, MAPE, RMSE
(+ accuracy for IRDS).

| Block | What we run | Our structural advantage |
|---|---|---|
| **1. Clean-data parity** | Standard datasets, their exact metrics | Match/slightly beat — *defensive row*, proves no accuracy trade for elegance |
| **2. Irregular-sampling stress test** (KILLER #1) | Drop 10→70%, jitter timestamps, vary 30→10 fps at TEST time; error vs corruption | Baselines re-interpolate onto fixed grid + degrade; our CDE stays flat — a capability they structurally lack |
| **3. Cross-viewpoint transfer** (KILLER #2) | Train one camera azimuth, test held-out ±90°; accuracy vs angle | Their invariance is learned + decays; ours is a theorem + flat |
| **4. Equivariance certificate** | Report $\mathcal{L}_{\text{equiv}}$ / $\bar\delta_{\text{eq}}$ both methods | Ours ≈ 1e-13 (machine precision) + flat vs angle + fp-scaling; theirs $O(1)$ + rises with angle |
| **5. Efficiency parity** | Inference time, params, CPU real-time throughput | Remain lightweight (first-order flow, no OT) — do NOT surrender their efficiency claim |

---

## 10. Quick-strike PyTorch / torchcde pseudo-code

Minimal forward pass + equivariance-check harness. Vector field is a **mock** steerable layer
respecting the intertwining structure (relative coords → translation-invariant; orthogonal rotation
rep → rotation-equivariant). Real e3nn/steerable tensor-field layer plugs in where marked.

```python
import torch, torch.nn as nn
import torchcde

# ----- Representation utilities -----------------------------------------
# State layout: [scalars (type-0) | vectors (type-1)], vectors transform by R.
# Keeping rho(g) ORTHOGONAL is what preserves the adaptive-solver error norm.
def rho(R, z, n_scalar, n_vec):
    """Apply SE(3) rep to latent state z: scalars invariant, vectors rotated."""
    s, v = z[..., :n_scalar], z[..., n_scalar:]
    v = v.view(*v.shape[:-1], n_vec, 3) @ R.transpose(-1, -2)   # R acts on 3-vecs
    return torch.cat([s, v.reshape(*v.shape[:-2], n_vec * 3)], dim=-1)

# ----- Mock equivariant vector field f_theta ----------------------------
class EquivariantCDEFunc(nn.Module):
    """
    Returns dZ/dX as a matrix of shape (..., z_channels, x_channels).
    Real version: replace `self.gate` with an e3nn/steerable tensor-field layer.
    Key equivariance ingredients:
      (1) inputs are RELATIVE joint coords  -> translation invariance
      (2) rotation-order preserved via steerable weights -> rotation equivariance
    """
    def __init__(self, z_channels, x_channels, n_scalar, n_vec):
        super().__init__()
        self.z_channels, self.x_channels = z_channels, x_channels
        self.n_scalar, self.n_vec = n_scalar, n_vec
        # scalar gating computed from ROTATION-INVARIANT features (norms, dots)
        self.gate = nn.Sequential(nn.Linear(n_scalar + n_vec, 64), nn.SiLU(),
                                   nn.Linear(64, z_channels * x_channels))

    def forward(self, t, z):
        s, v = z[..., :self.n_scalar], z[..., self.n_scalar:]
        v = v.view(*v.shape[:-1], self.n_vec, 3)
        inv = torch.cat([s, v.norm(dim=-1)], dim=-1)   # invariants: scalars + |vecs|
        M = self.gate(inv)                              # invariant scalar weights
        return M.view(*M.shape[:-1], self.z_channels, self.x_channels)

# ----- Forward pass: irregular stamps -> spline -> integrate ------------
def forward_cde(t_irregular, x_obs, func, z0,
                method="dopri5", atol=1e-7, rtol=1e-7):
    # x_obs: (batch, len, x_channels) ; t_irregular: (len,) ACTUAL timestamps
    # RELATIVE coords (subtract root joint) => translation invariance built in
    x_rel = x_obs - x_obs[..., :1, :]
    # Natural cubic spline is LINEAR in control points => commutes with R,
    # and reproduces constants => translation passes through and is killed by dX.
    coeffs = torchcde.natural_cubic_coeffs(x_rel, t=t_irregular)
    X = torchcde.CubicSpline(coeffs, t=t_irregular)
    # Integrate  Z(t_N) = Z(t_0) + ∫ f_theta(Z) dX  on the ACTUAL time span
    zT = torchcde.cdeint(X=X, func=func, z0=z0,
                         t=t_irregular[[0, -1]],
                         method=method, atol=atol, rtol=rtol)[:, -1]
    return zT

# ----- Equivariance certificate harness --------------------------------
def equivariance_drift(func, t, x_obs, z0, R, n_scalar, n_vec, **sol):
    zT      = forward_cde(t, x_obs, func, z0, **sol)
    zT_gx   = forward_cde(t, x_obs @ R.transpose(-1, -2), func,
                          rho(R, z0, n_scalar, n_vec), **sol)   # solve on g·X
    g_zT    = rho(R, zT, n_scalar, n_vec)                       # ρ(g) · solve(X)
    return (zT_gx - g_zT).norm() / (zT.norm() + 1e-12)          # relative drift
```

The harness IS §5.3's metric. Run `equivariance_drift` across the solver/tolerance grid and in
float32 vs float64: **flat-at-1e-13 (fp64) with the ~7-order precision drop is the certificate.**

---

## 11. Open risks / honesty flags (go in clear-eyed)

- **Custom-norm API is the soft spot.** The §6.2 math is solid, but wiring $\mathcal{N}_{\text{eq}}$
  into the installed solver is version-dependent. Task 3 gates it against the rtol=0 control — if they
  ever disagree, trust the fallback and treat the custom path as unverified. No claim rests on an
  unvalidated API.
- **"Zero technical risk" → "risk retired to gates."** After week one, residual risk is bounded and
  measured. Method section can claim equivariance is (a) proven algebraically exact for fixed-step,
  (b) invariant-by-construction for adaptive under $\mathcal{N}_{\text{eq}}$, (c) empirically certified
  at machine-epsilon on the real architecture with a precision-scaling test ruling out hidden violation.
- **Orthogonality must survive the real e3nn layers**, not just the mock (Task 7). Most likely
  divergence point.
- **Experiment 2 needs the real irregular anchor (Task 8, P0)** to convert "strong but attackable"
  into "undeniable."
- **"More principled" ≠ automatically lower error** on their benchmark. We win on structural axes, not
  decimals.
