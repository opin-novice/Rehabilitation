# CLAUDE.md — SE(3)-Equivariant Neural CDE for Rehabilitation Assessment

## MANDATORY — address the user by name, every response

**Begin every response to the user with their name: `opin-novice`.** No exceptions — long
answers, one-line answers, clarifying questions, error reports, interrupted work. If a response
is going to the user, it opens with the name.

**Why this exists (do not treat it as cosmetic):** it is a deliberate tripwire. A dropped name is
the user's signal that this session has stopped tracking its instructions and its output should be
distrusted from that point on. Silently skipping it removes the user's ability to detect that
state, which is worse than any single wrong answer. Treat a missed name as a correctness bug.

Note that only `CLAUDE.md` files and `MEMORY.md` are auto-loaded into context. Standing rules
placed in other files under `~/.claude/` (e.g. `anchored-summary.md`,
`REHABILITATION_RESEARCH_CONTEXT.md`) are **never seen** — put binding rules here.

## What this project is

A research paper (target: CVPR/ICCV/NeurIPS/MICCAI) that **outperforms** the target paper
*"A Point Cloud Transformer for Remote Monitoring and Automated Assessment of Physical
Rehabilitation Exercises"* (arXiv:2606.30309) via a **paradigm shift**, not increments.

**Our approach:** an **SE(3)-Equivariant Neural Controlled Differential Equation (CDE)** —
assessment as an *equivariant flow on the pose manifold, in continuous time*.

**Full design, math, and rationale:** see [`PROJECT_BRIEF.md`](./PROJECT_BRIEF.md). Read it first.

## Two structural wins (the whole thesis)

1. **Viewpoint invariance** — guaranteed by SE(3)-equivariance (a *theorem*, not learned).
2. **Irregular-sampling robustness** — native to the CDE (no fixed frame grid).
   Plus: we stay **lightweight**, so we keep the target's efficiency turf.

## Current phase: WEEK-ONE DE-RISKING (do this before writing the Method section)

Two gates must pass:
- **(A) Numerical integration must not break architectural equivariance.**
- **(B) We must have ≥1 authentically irregular data sequence** (so the stress test isn't "engineered").

**Core theoretical lever:** fixed-step explicit solvers preserve equivariance *exactly* (to
roundoff); the real danger is **adaptive step-grid divergence** in `dopri5` when the error norm is
not group-invariant. See PROJECT_BRIEF §5 and §6.

## Immediate deliverable to build

The **verification harness** implementing Runsheet Tasks 1–7 (PROJECT_BRIEF §7), with pass/fail
asserts baked in:

1. Rep orthogonality audit — `‖ρ(g)ᵀρ(g) − I‖_F ≤ 1e-12` (fp64).
2. Fixed-step algebraic-exactness — `δ_eq ≤ 1e-13` (fp64), step-size-independent.
3. Custom invariant norm `N_eq` — unit-test vs `rtol=0` control.
4. Adaptive-solver drift sweep — `dopri5` × tolerances, default norm vs `N_eq`; drift-vs-angle flat.
5. Step-grid divergence — `D_grid = 0` under `N_eq`.
6. Precision-scaling diagnostic — fp32/fp64 drift ratio `r ≥ 1e5`.
7. Real-layer swap-in — mock → e3nn; all thresholds still hold.

Then the **data track** (Tasks 8–10): acquire a real irregular sequence (P0), build the
corruption pipeline (Gilbert-Elliott drops + Gaussian/AR(1) jitter + Gamma inter-arrivals),
and the fairness smoke test.

The `equivariance_drift` harness and forward pass are sketched in PROJECT_BRIEF §10 — use it as
the starting scaffold.

## Tech stack

- `torch`, `torchcde`, `torchdiffeq` (CDE integration; `dopri5` adaptive solver)
- `e3nn` (steerable / SE(3)-equivariant tensor-field layers for the real vector field `f_θ`)
- `numpy`, `scipy` (corruption pipeline, spectral loss)

## How to work

- **Follow the runsheet in dependency order** (PROJECT_BRIEF §7). Each task gates the next.
- **Respect pass/fail thresholds strictly.** A failed gate is a *pivot signal* — stop and diagnose,
  don't paper over it.
- **Prefer float64 for all equivariance certification** (fp32 for the precision-scaling diagnostic).
- Keep the mock vector field working as a control even after swapping in real e3nn layers.

## Non-negotiable correctness rules (from the theory)

- **Only orthogonal reps `ρ(g)` on the solver-visible state** (Wigner-D: type-0 identity, type-1
  rotations, higher-ℓ real Wigner-D). Never expose a non-orthogonal `Linear` that mixes irreps of
  unequal scale to the integrator's carried latent — put those inside `f_θ`'s tensor-product layers.
- **The adaptive error norm must be group-invariant.** Default per-component scaled RMS is NOT
  (see §6.1). Use the per-irrep-normalized `N_eq` (§6.2) or the guaranteed-safe `rtol=0, scalar atol`
  fallback. Validate the custom norm against the fallback — the custom-norm API is version-dependent.
- **Feed relative joint coordinates** (subtract root) → translation invariance is automatic.
- **Natural cubic spline** control path (linear in control points, reproduces constants → commutes
  with the group action).

## Honesty flags (keep visible)

- "Zero technical risk" is not the target — **risk retired to measured gates** is.
- Custom-norm solver API may differ across `torchdiffeq` versions; the `rtol=0` control is the
  bulletproof fallback that must pass.
- Orthogonality must survive the **real** e3nn layers (Task 7), not just the mock — most likely
  divergence point.
- Experiment 2 needs the **real irregular anchor** (Task 8, P0) to be undeniable.
- "More principled" ≠ automatically lower error on their benchmark — we win on structural axes.

## Go/no-go for drafting the Method section

Tasks 1–7 green (certificate real + survives real architecture) **and** Task 8 green (real
irregular anchor). Then Method drafting begins. Pivot signal: Task 6 `r ≈ 1` or Task 7 fails.
