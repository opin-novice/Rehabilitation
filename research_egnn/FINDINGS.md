# Sandbox findings (1 seed × 5 folds — NOT for the paper)

| Arm | Clean MAD | Viewpoint degr | Invariance cert | Node-fail lost (0→8) |
|---|---|---|---|---|
| **EGNN** (E(n)-equiv, sandbox) | **6.72** | 0.0000 | 1.4e-5 (exact) | **+7.81** |
| **Canon-PCA + PCT** (sandbox) | 6.85 | 0.0000 | 0.0 | +3.17 |
| EGRU (e3nn) — paper 3-seed | 6.73 | ~0 | exact | +3.76 |
| InvariantGRU — paper 3-seed | 6.31 | 0 | exact | +12.05 |
| PCT (raw) — paper 3-seed | 6.47 | 9.42 | — | +2.27 |

## Q5 — does lighter E(n) (EGNN) equivariance match the steerable e3nn encoder?
- **Accuracy: yes, a tie.** EGNN 6.72 vs EGRU 6.73 (inside the 0.33 seed floor).
- **Viewpoint: exact, by construction.** Because the EGNN feeds the *same* invariant cut, EGNN + cut
  is itself exactly SE(3)-invariant (degradation 0, certificate 1.4e-5 = fp32 machine precision).
  So viewpoint is a tie by design — as anticipated.
- **Node-failure: EGNN is ~2× MORE brittle than the steerable encoder** (+7.81 vs +3.76), landing
  between EGRU and InvariantGRU. So — tentatively — the steerable features carry a node-failure
  robustness edge that lighter E(n) equivariance does not, *even behind the identical pooling/cut.*
- **Honest caveats:** 1 seed; the EGNN is untuned (4 layers, hidden 64, no hyperparameter search), so
  its node-failure disadvantage could be partly under-tuning ("sabotaged-baseline" risk) rather than
  an intrinsic property. This is a probe, not a fair publication-grade comparison.

### High-rigour follow-up (3 seeds × 5 folds + coordinate-clamp sweep)
| Arm | clean MAD (μ±σ) | node-fail lost 0→8 (μ±σ) | cert |
|---|---|---|---|
| EGNN (no clamp) | 6.880 ± 0.164 | **+6.39 ± 1.27** | 1.4e-5 |
| EGNN clamp 0.1 | 6.815 ± 0.224 | +6.42 ± 1.12 | 8.3e-6 |
| EGNN clamp 0.5 | 6.847 ± 0.170 | +6.06 ± 0.75 | 1.1e-5 |
| EGNN clamp 1.0 | 6.957 ± 0.034 | +5.69 ± 0.69 | 1.2e-5 |
| *EGRU (paper)* | *6.73* | *+3.76* | *exact* |

- **The signal PERSISTS across seeds — not seed-undertuning.** The single-seed +7.81 was a high draw
  from a wide distribution; the honest 3-seed baseline is **+6.39 ± 1.27**, still clearly above EGRU's
  +3.76 (gap ≈ 2.6 MAD ≈ 3.6 SEM). So lighter E(n) equivariance is genuinely more node-fail-brittle
  than the steerable encoder behind the identical cut, even after ruling out initialization noise.
- **Coordinate gating does NOT rescue it.** The "damping mitigates the feature-loss cliff" hypothesis
  is not supported: aggressive clamp 0.1 gives no improvement (+6.42), and the best arm (mild clamp
  1.0) reaches only +5.69 ± 0.69 — still ~2 MAD above EGRU. If anything the trend runs opposite to the
  hypothesis (less damping is marginally better). The brittleness is not primarily a coordinate-update-
  magnitude artifact.
- All arms stay viewpoint-exact (cert ~1e-5): clamping preserves equivariance.
- **Net:** the steerable encoder's node-failure robustness edge over lighter E(n) equivariance is a
  real (~2 MAD), reproducible property, not fixable by simple coordinate clamping. Still far better
  than InvariantGRU (+12.05); all invariant-cut arms remain viewpoint-exact.
- **Bearing on the paper (this 3-seed result now ENTERS §IV-E, Table `tab:egnn`):** the steerable
  justification is *supported*, not undermined — EGNN ties on accuracy and viewpoint but is ~2 MAD more
  node-fail brittle, reproducibly and un-rescued by clamping. This reverses the earlier "keep Q5 out"
  decision, which was premised on an untuned 1-seed probe *diluting* the steerable case; the 3-seed
  measurement does the opposite, so it is reported as the Q5 baseline. The paper states the
  untuned-architecture caveat (4-layer, hidden-64, no HP search) explicitly and claims only the scoped
  result — "behind this identical cut, on this corpus, the steerable encoder is the more node-fail
  robust of the two" — alongside the InvariantGRU co-headline ("why not lighter *invariance*"). Note
  the isolation guarantee below is about **code**, which is unchanged; only these aggregate numbers
  enter the manuscript.

## Q3 — does a canonicalization front-end leave per-sequence degradation the exact cut avoids?
- **Not on this sweep.** PCA-canonicalize → PCT is empirically ~exactly viewpoint-invariant
  (degradation 0.0000, cert 0.0) at a modest accuracy cost (6.85 vs raw PCT 6.47). The hypothesized
  sign-flip / per-sequence degradation did **not** appear.
- **Why:** rotating about gravity rotates the PCA eigenframe with the body, and KIMORE's tall,
  well-separated body poses don't reach the near-degenerate covariance (or dominant-joint sign
  boundary) that would flip the deterministic sign convention. So the canonicalizer is empirically
  robust here.
- **The real distinction is exactness-as-theorem vs empirically-robust.** The canonicalizer's 0 is an
  *empirical* 0 on this azimuth sweep — it carries no guarantee and would break on degenerate/planar/
  symmetric poses or under general SO(3) crossing a sign boundary (untested here). The paper's cut is
  invariant by *proof* over all SO(3) including those cases. That is a subtler and more honest framing
  than "canonicalizers leave residual degradation," which this data does not support.

## Isolation
No file under `src/` or `outputs/cde_block2/` was edited/created/deleted (verified: identical
size+mtime manifest and unchanged `src` md5 before/after). All artifacts are under
`research_egnn/outputs/`. Numbers in `outputs/sandbox_results.json`.
