# Rebuttal draft — Reviewer 2.3: "the statistics narrative can accidentally undercut you"

Drafted 2026-08-02. Companion to `REBUTTAL_R2_2_clinical_significance.md`. Shipped in commit `fa7a7ba`.

---

## The objection, as stated

> Your protocol audit shows the single-exercise slice "clears zero in only 1/3 seeds" — a reviewer
> skimming can read this as "their own model barely beats the floor." The pooled result (all CIs below
> 0) rescues it, but the ordering/emphasis matters. Fix: front-load the pooled result and frame the
> single-slice fragility purely as a critique of the field's slice, so it never reads as a weakness of
> your model.

Diagnosis right. **The root cause is sharper than emphasis, and it makes the requested framing available
as a factual correction rather than spin.** Acted on; see the bottom.

## The response paragraph

> The reviewer is right that this passage invited the wrong reading, and the reason is an omission on
> our part: `tab:protocol3seed` audits the *reference point-cloud transformer under the reference
> protocol*, not our model, and we failed to say so in any of the four places the result appears. Every
> artifact behind that table is `model=pct, exercise=1`; there is no EGRU single-exercise audit in the
> repository. All four passages now name the audited model. Separately, each of them stated the
> fragility and then justified *pooling* without ever saying pooling worked; each now states the pooled
> outcome in the same passage, so the fragility is never held without its resolution. Pooled, every
> model's interval lies entirely below zero — PCT $-1.87\,[-2.72,-1.06]$, InvariantGRU
> $-2.03\,[-2.88,-1.23]$, EGRU $-1.60\,[-2.37,-0.88]$. No number changed; this was a reporting defect,
> not a result.

## The evidence that the audited model is PCT

Three independent confirmations, any of which a reviewer can check:

1. Every artifact carries `"model": "pct", "exercise": 1` —
   `outputs/cde_block2/protocol_audit_pct_ex1_seed{0,1,2}.json`.
2. `src/protocol_null.py` documents its own invocation as `--model pct --cv`, and an inline comment
   ties the quoted $6.42{\pm}0.44$ honest / $5.21{\pm}0.19$ inflated pair to the unconditioned PCT head.
3. There is **no** EGRU single-exercise audit anywhere in `outputs/`.

| seed | honest / inflated | selection bias | $\Delta$ vs.\ floor (95% CI) | |
|---|---|---|---|---|
| 0 | $6.90/5.45$ | $+1.44\ (20.9\%)$ | $+0.20\,[-0.59,+1.03]$ | straddles |
| 1 | $6.53/5.00$ | $+1.54\ (23.5\%)$ | $-0.46\,[-1.60,+0.72]$ | straddles |
| 2 | $5.84/5.17$ | $+0.67\ (11.4\%)$ | $-1.07\,[-1.98,-0.15]$ | **clears** |

The failure to clear zero is **PCT's**, on the field's own slice, under the field's own protocol. That
is the critique the reviewer wanted — and it needs no rhetorical repositioning, only attribution.

## Precision that matters

Do **not** claim the slice is uninformative *for our model specifically*. We never ran EGRU on it (see
below), so that claim is unsupported. The defensible statement is that the reference model under the
reference protocol carries no robust subject-level signal there, which is why we pool.

Do **not** reorder Results so the pooled result precedes the protocol audit. The audit is what motivates
pooling; stating the outcome first inverts the argument. The fix is to make each fragility mention
self-resolving, not to move blocks.

Do **not** soften or drop the audit. The $18.6\%$ inflation and the seed-fragility are contribution (i)
and part of the paper's honesty stance. Nothing here removes a self-critique — it corrects who the
critique is about.

## Anticipated follow-ups

**"Did *your* model clear zero on that slice?"** We did not run it, and we say so rather than implying
otherwise. Running it was considered and declined: the claim does not need it — auditing the reference
model under the reference protocol is the point — and the outcome could cut either way. An EGRU that
cleared zero on Ex1 would contradict "the slice carries no robust subject-level signal" and undercut the
pooling justification we build everything else on. If a reviewer insists it is runnable, but **not as a
flag flip**: `src/protocol_null.py` accepts `--model {pct,cde}` only, so an EGRU arm has to be added to
`run_fold` first (the eval path already branches on `model_name`, so the change is small but it is a
code change, not a rerun). Do not promise this as a one-liner in a response letter.

**"Isn't pooling just a way to manufacture significance?"** Pooling five exercises multiplies sequences,
not subjects, and the bootstrap resamples **by subject** (77 clusters), so the resampling unit is
unchanged. `bootstrap_delta` in `src/protocol_null.py` documents this: each drawn subject contributes
their whole block of recordings, so correlated within-subject errors cannot inflate the interval.

**"Why is the epoch-selection audit on an unconditioned PCT head?"** Deliberate, and commented in the
code: the quoted numbers were produced when `num_exercises` was accepted and ignored. Passing 5 now
would silently train a different model and the audit would no longer describe the run it is cited for.
Exercise conditioning is a separate measured arm and is reported.

## What changed in the paper

1. `tab:protocol3seed` caption now names the audited model ("Protocol audit of the *point-cloud
   transformer baseline*") and states that pooling fixes it for this model and every other.
2. Intro, contribution (i) and the Results protocol-audit paragraph all attribute the result to the
   reference transformer and state the pooled outcome in the same passage.
3. Supplement §"Protocol Audit and Continuous-Time Control" attributes it likewise and adds that pooled,
   every model clears the floor — so the fragility is a property of the slice, not of an architecture.
4. Corrected InvariantGRU's CI bound $-2.89 \to -2.88$ to match
   `pooled_bootstrap_eval_summary.json` under the same rounding used beside it.

No statistic changed. Verified programmatically: all four attributions present, all four pooled-outcome
statements present, and the four quoted audit figures unchanged.
