# Rebuttal draft — Reviewer 2.4: "the irregular-sampling story is a null but still dressed as a feature"

Drafted 2026-08-02. Companion to `REBUTTAL_R2_3_protocol_narrative.md`. Shipped in commit `f202186`.

---

## The objection, as stated

> "Consumes the sensor's actual inter-arrival times" is presented as a design virtue, yet §Results
> reports the fixed-grid baseline beats it at every drop level. Reviewers will notice the tension.
> Fix: it's fine as a scoped negative result — just make sure every place you mention dt-consumption
> flags that it's a null on KIMORE (positional-energy argument) and not oversold as an advantage.

Right, and **the cause is a factual inconsistency rather than tone.** Acted on; see the bottom.

## The response paragraph

> The reviewer identified a real inconsistency. The paper enumerated its negative results three times
> and named a different pair each time; the introduction substituted the Neural CDE failure for the
> irregular-sampling null. Those are different claims — the CDE is a model that fails the
> mean-predictor floor, whereas the irregular null is dt-consumption not beating resampling — so a
> reader of the introduction was told dt-consumption is part of the design and handed a list of
> negatives that did not contain it, with the null five hundred lines away. All three enumerations now
> name the same pair, and the architectural mention carries an inline flag: the capability is real but
> KIMORE does not reward it, retained because it is free and demonstrably real off this corpus.

## The inconsistency, precisely

| where | pair named |
|---|---|
| abstract | CDE fails floor + **irregular null** |
| intro | CDE fails floor + **parity channel** |
| results | **irregular null** + parity channel |

Unified on **{irregular-sampling null, chirality null}**. The Neural CDE remains what Related Work,
Method and the supplement already call it — a certified *control that fails the floor* ($8.43$ vs.\
$8.25$ MAD) — rather than being double-counted as a headline negative. **Those three passages were
verified intact after the edit**; demoting the CDE from the headline pair must not delete it, which
would be hiding a negative.

## The claim is measured, not asserted

"The fixed-grid baseline is better at every drop level" was re-checked against
`outputs/cde_block2/block2_aug_s{0,1,2}_results.json` before leaning on it in the intro:
**fixed-grid wins 18/18** (3 seeds × 6 drop levels, $0$–$70\%$), best-case operator granted to the
baseline at every level. The sentence is accurate.

The mechanism is derived, not post-hoc: a model-free Lomb–Scargle census finds $97.7\%$ of KIMORE's
*positional* energy below the resampling corner — though $70\%$ of *velocity* energy lies above it — so
resampling preserves pose geometry while low-passing the derivative band, acting as a **denoiser** on
healthy-form exercises where the discriminative signal is positional. The null was predictable before
the experiment.

## Precision that matters

Do **not** disown the capability. The mechanism is demonstrated: a physiologically calibrated $5$~Hz
tremor is recovered from the true irregular stamps at $r{=}1.000$ under drop rates to $70\%$, where a
resampled estimator decays to $0.785$. The claim is a **scope boundary, not a defeat**, and that
sentence is load-bearing — flagging the null must not slide into calling the design a mistake.

Do **not** add hedges to the five passages that already flag the null correctly (abstract, Related Work,
Method, the Results negative-result paragraph, and the supplement's "Irregular-sampling null (derived)"
subsection). This is not a paper that hides the result; it was a paper whose intro contradicted its own
results section. Blanket hedging costs page budget and makes a well-handled negative read as defensive.

Do **not** conflate the two negatives when answering. "The CDE fails the floor" and "dt-consumption is a
null" are separate findings with separate evidence. Conflating them is precisely the error that created
this objection.

## Anticipated follow-ups

**"Then why keep dt-consumption in the architecture at all?"** Because it is free — it costs no
parameters and no accuracy — and the tremor result shows the mechanism works where the corpus rewards
it. The honest framing is that KIMORE's healthy-form exercises are the wrong corpus for it, not that the
capability is wrong. This is now stated inline at the architectural mention rather than left for the
reader to discover.

**"Isn't the tremor result synthetic?"** Yes, and it is described as physiologically calibrated rather
than observed. It establishes the mechanism, not corpus-level benefit, and is not used to claim the
latter.

**"Would the boundary move on a tremor-dominated corpus?"** That is the paper's own claim, stated in
both the Results passage and the supplement. It is a prediction we have not tested and should not
present as a result.

## What changed in the paper

1. Abstract: negatives realigned to {irregular-sampling null, unrewarded parity restoration}.
2. Intro negatives list: replaced "a continuous-time variant that fails the floor" with native
   consumption of irregular arrivals, "which a fixed grid beats at every drop level here."
3. Intro design pitch: the inter-arrival clause now carries an inline flag — a capability KIMORE does
   not reward, kept because it is free and demonstrably real off this corpus.
4. Nothing else touched. Related Work, Method and the supplement were left exactly as they were and
   re-verified.

No statistic changed. Content still ends on page 8; $0$ overfull boxes.
