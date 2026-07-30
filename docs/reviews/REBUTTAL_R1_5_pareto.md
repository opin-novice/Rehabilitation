# Rebuttal draft — Reviewer 1.5: "the positive claim lives inside a constructed niche"

Drafted 2026-07-30, while the reasoning was fresh. Covers the node-failure /
Pareto-framing objection and the one follow-up it is most likely to generate.

---

## The objection, as stated

> The positive claim lives inside a constructed niche, and the baseline wins the axis
> outright. Your honest admission that PCT is the most node-robust model (+2.27 vs your
> +3.76) guts the naive reading of "what the steerable encoder buys." Your defense —
> "among models with an exact viewpoint guarantee, only the graph one survives node
> failure" — is a subset carefully drawn to exclude the strongest model. A reviewer can
> call this gerrymandering.

**This was correct and we acted on it.** The paper no longer defends the node-failure
axis. See the changes listed at the bottom.

## The answer, if it returns as "you still use a conjunction"

> We do not claim EGRU dominates every axis. We publish the full frontier (Table 6) and
> state in the abstract, in contribution (iii), and in the results text that PCT wins
> clean-data accuracy **and** node robustness outright. We locate EGRU as the only model
> that is simultaneously exact-by-theorem, stress-complete, and $7.4\times$ smaller than
> the next alternative. That is a description of an empirical Pareto point, not a subset
> drawn to exclude competitors — the competitor's two wins are printed in the same table
> and named in the same sentences.

The distinguishing test we would offer the reviewer: **a gerrymander hides the excluded
model's wins; a frontier prints them.** Ours are in the abstract, the contribution list,
the results text, the Pareto table, and the supplement.

## Supporting numbers (all from Table 6 unless noted)

| axis | winner | our value |
|---|---|---|
| clean MAD | **PCT 6.47** | 6.73 (inside the 0.33 MAD nondeterminism floor) |
| node failure, KIMORE | **PCT +2.27** | +3.76 |
| node failure, REHAB24-6 | **PCT** (crosses at $k{=}4$, +0.021 AUROC at $k{=}8$) | — (supp §nodefail) |
| viewpoint exactness | **ours, by theorem** | $9\times10^{-6}$ vs 3.01 mean / 20.12 worst (PCT+rot) |
| parameters | **ours 0.66M** | vs 4.91M |

Three models clear every stress: ours, Canon-PCA+PCT, PCT+rot. Of those, ours is the only
one at 0.66M rather than 4.91M and the only one whose invariance is a theorem rather than
an estimated frame (Canon-PCA, empirical, marked †) or an augmentation tolerance (PCT+rot).

## The one column that is not a trade

Every other column is an axis we may lose. The last one is different in kind, and this is
the sentence to hold:

> 3.01 MAD mean per-sequence degradation is a tolerance that happens to be small on this
> data and carries a 20 MAD tail on a 0–50 scale. $9\times10^{-6}$ is roundoff around a
> value the theorem fixes at zero. Aggregate flatness is shared; per-patient exactness is
> not.

Do **not** let this be recast as "a fourth axis." Framing it as one axis among four
demotes a theorem to a tie-break, which is the single largest unforced error available
here.

## Anticipated follow-ups

**"Your conjunction has three terms; with enough terms anyone is unique."**
Agreed in general, which is why the terms are not free parameters: each was fixed by the
deployment setting before the results (a home device is moved → viewpoint; a consumer
depth sensor drops joints → node failure; it runs on a phone-class budget → parameters;
it must be at least as accurate as what it replaces → clean MAD). We did not add a term
after seeing who won. And we report the axes we lose on the same page.

**"REHAB24-6 shows you are *less* node-robust, so 'stress-complete' is doing work you
have not earned."**
Stress-complete refers to clearing the 8.31 MAD floor at 90° and two dead nodes on
KIMORE, which is what Table 6 tabulates — not to winning the node-failure axis. The
second corpus is where PCT overtakes us, and we say so ourselves in supp §nodefail, which
publishes the whole curve including our 0.021 AUROC deficit at $k{=}8$.

**"Then what is the contribution, if the baseline is more accurate and more robust?"**
A per-patient guarantee at 1/7th the parameters. On a 0–50 clinical scale, the augmented
baseline can move an individual patient's score by 20.1 MAD purely by moving the camera;
ours cannot move it at all, for any weights, by Prop. 1. That is the deployable property,
and it is the one the aggregate metric cannot see — which is also the paper's protocol
point.

## Do not do these

- **Do not touch supplement §nodefail.** It already reports PCT's $k{=}4$ crossover and
  $k{=}8$ lead in our own words ("our claim is a trade, not a dominance"). That honesty is
  the shield; editing it would look like concealment and would hand the reviewer a much
  worse story than the one they are complaining about.
- **Do not adopt "node-graceful"** as a claimed property. It is refutable directly from
  our own appendix.
- **Do not re-introduce** "what the steerable encoder buys: sensor-node failure" or "a
  measured justification for the steerable encoder." Both were removed precisely because
  they promise an axis we lose.

## What changed in the paper (for the response letter)

1. §4 heading: "What the steerable encoder buys: sensor-node failure" → "Does the
   guarantee cost robustness?", framed as a tax check. PCT's win stated in bold, on both
   corpora. Added: "we neither claim it nor define a subset of models within which we win
   it."
2. Abstract: the "what the steerable encoder buys" sentence replaced by the frontier and
   an explicit no-dominator statement.
3. Contribution (iii): "a measured justification for the steerable encoder" → a Pareto
   characterisation naming the baseline's two wins.
4. Table 6 paragraph: now leads "no model wins all five columns, and that — not any single
   stress — is the claim", then walks the frontier.
5. Introduction: the node-failure paragraph now scopes itself — "That is one pairwise
   comparison, not an axis we own: the point-cloud baseline is more node-robust than
   either of us."

Residual, accepted knowingly: the locating statement is still a conjunction. The
mitigation is that the concession is now as prominent as the claim, in five places.
