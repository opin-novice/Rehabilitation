# Rebuttal draft — Reviewer 2.1: "the canonicalization steelman isn't fully closed"

Drafted 2026-07-30. Companion to `REBUTTAL_R1_5_pareto.md`.

---

## The objection, as stated

> Your sharpest technical point is "Canon-PCA+PCT ties us but its invariance is only
> empirical (18% of frames near degeneracy, 21% frame flips)." A reviewer will counter:
> "Use a better canonicalizer — SVD with sign-tie-breaking, a small learned/equivariant
> frame predictor, or temporal smoothing — and the degeneracy argument weakens." Right now
> you only defeat naive per-frame PCA.

Half right, and we had made both halves possible. Acted on; see the changes at the bottom.

## The response paragraph

> We acknowledge that our PCA baseline already includes deterministic sign disambiguation
> and right-handedness enforcement — a sentence we omitted and have now added. Under clean
> rotation, canonicalization is exactly equivariant (residual $3\times10^{-6}$ degrees),
> which is why our prior tests showed no failure. The failure regime is sensor noise near
> covariance degeneracy, where Davis–Kahan predicts $O(\lVert E\rVert/\gamma)$
> amplification. We tested the reviewer's three suggested improvements: a better sign rule
> marginally *worsens* amplification ($11.9\times$ vs $9.2\times$), temporal smoothing more
> than doubles it ($22.9\times$), and learned predictors are blocked by the stabilizer
> obstruction (Kaba et al. 2023). The frame-level instability is model-free and applies to
> any spectral canonicalizer.

## The measurement (research_egnn/canon_noise_probe.py)

Camera-frame sensor noise at 2% of body RMS radius; residual rotation between the canonical
frame recovered at two camera poses; medians, binned by relative eigen-gap $\gamma$.

| canonicalizer | $\gamma<0.02$ | $\gamma>0.25$ | amplification | frames $>45^\circ$ |
|---|---|---|---|---|
| argmax (the shipped baseline) | 11.12° | 1.21° | 9.2× | 14.5% |
| skew — third-moment sign rule | 12.06° | 1.01° | 11.9× | 11.1% |
| temporal — Procrustes smoothing | 28.80° | 1.26° | 22.9× | 31.6% |

**Clean control: max residual $2.96\times10^{-6}$ degrees, all variants.** Quote this first.
It concedes the reviewer's strongest implicit point before they make it.

Artifact ships at `paper/wacv_submission/verification/canon_noise_probe.json`.

## Why each suggestion fails

**Sign-tie-breaking.** Already in our baseline, and a *better* rule doesn't help: a sign rule
resolves a **discrete** ambiguity ($\pm$ per axis), whereas degeneracy is a **continuous**
one — when two eigenvalues coincide the eigenvectors are undetermined within a plane. The
third-moment rule trims catastrophic flips (14.5% → 11.1%) and leaves the amplification
intact. Both facts are in the table; quote them together or the concession looks like a loss.

**Temporal smoothing.** Empirically the worst option (22.9×, flips more than double): a frame
locked onto the wrong branch *persists* through the history chain. It also makes the frame
history-dependent, so per-frame equivariance is traded for lag — the invariance becomes
approximate and delayed rather than exact, which is a strictly worse guarantee than the one
under discussion.

**Learned / equivariant frame predictor.** Not tested — say so plainly — and covered by
theory instead. If $g\in\mathrm{Stab}(x)$ then equivariance forces $c(x)=c(gx)=g\,c(x)$,
hence $g=e$: no exactly-equivariant canonicalizer exists at a configuration with non-trivial
stabilizer, learned or otherwise. Globally, a continuous equivariant section of the frame
bundle need not exist even where pointwise choices do. Cite Kaba et al. 2023.

## Precision that matters

Do **not** write "canonicalization is discontinuous at degeneracy." It is not: rotating the
input sends the covariance to $RCR^{\top}$, whose eigenvectors are exactly $RV$, and we
measure $3\times10^{-6}$ degrees. A reviewer can refute that phrasing with our own probe. The
claim is about **conditioning** — $O(\lVert E\rVert/\gamma)$ sensitivity to input perturbation
— not about exactness.

Do **not** claim the obstruction covers "any frame estimator." Only estimators reading the
*covariance* are bound by covariance degeneracy; one using higher moments or joint identity
could be well-defined there. The two defensible statements are (i) Davis–Kahan for spectral
frames and (ii) the stabilizer obstruction for exact equivariance. Those cover all three
suggestions without overreaching.

## Anticipated follow-ups

**"Your noise model is arbitrary."** σ is expressed as a fraction of body RMS radius, so it is
scale-free, and the amplification is reported as a *ratio across eigen-gap bins* at fixed σ —
the conclusion is a slope, not a magnitude, so it does not depend on picking the right σ.

**"Show the harm at the score level, not the frame level."** Deliberately out of scope and said
so in the paper: the frame measurement is model-free and binds any spectral canonicalizer,
whereas output-level harm depends on the corpus and on how much jitter the downstream network
learned to absorb. Plain-EGRU checkpoints exist (`outputs/cde_block2/egru_s{0,1,2}_pooled_f*.pt`)
but come from the pooled protocol while the canon arm uses the sandbox fold scheme; a
cross-pipeline comparison would risk mismatched held-out sets. If pressed, this is runnable —
match the folds first.

**"18% and 45% sound cherry-picked."** They are thresholds on the same distribution, both
reported: $\gamma<0.05$ on 18% of frames, $\gamma<0.10$ on 45%. The anatomical reason is worth
giving — a standing body is nearly axially symmetric, so $\lambda_2\approx\lambda_3$ in the
transverse plane much of the time. Degeneracy is the typical regime here, not a corner case.

## What changed in the paper

1. Added the sentence stating the baseline already sign-disambiguates by dominant-joint
   projection and forces right-handedness — the omission that invited the objection.
2. Rewrote the canonicalization paragraph around **conditioning**: concedes clean-rotation
   exactness at $3\times10^{-6}$, cites Davis–Kahan for $O(\lVert E\rVert/\gamma)$, reports
   the three-variant measurement, covers learned frames via the stabilizer obstruction.
3. Added citations `daviskahan1970` and `kaba2023canon` (previously only
   `puny2022frameaveraging`).
4. Stated the frame-vs-score scoping explicitly rather than leaving it as a silent omission.
