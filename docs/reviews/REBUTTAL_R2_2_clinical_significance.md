# Rebuttal draft — Reviewer 2.2: "the clinical significance anchor is borrowed, not KIMORE-specific"

Drafted 2026-08-02. Companion to `REBUTTAL_R2_1_canonicalization.md`. Shipped in commit `fa7a7ba`.

---

## The objection, as stated

> Clinical significance anchor is borrowed, not KIMORE-specific (self-admitted). For a clinical
> Applications paper, "3.03 MAD matters" rests on MCIDs from Fugl-Meyer / other scales. This is
> handled very honestly, but it remains the weakest link in the "why should anyone care" chain.
> Fix: strengthen the practical hook — a clinical co-author, or a short argument tying MAD-of-50 to
> KIMORE's own 5-group clinical stratification (does a 3-point swing move a patient across the
> Expert/Parkinson/Stroke score bands?).

Diagnosis right. **The suggested fix is not executable, and executing it literally would have made the
paper worse.** Acted on with a substitute that answers the same intent; see the bottom.

## The response paragraph

> We agree the borrowed MCID was the weakest link, and have replaced it as the load-bearing claim.
> The reviewer's specific suggestion — does a swing cross KIMORE's clinical bands — has no honest
> answer, because KIMORE has no such bands: the five group labels record how a subject was
> *recruited*, not what score range they occupy, and on every one of the five exercises the
> lowest-scoring control sits below the highest-scoring patient (on Es5 an Expert scores $25.0$ while
> a Parkinson patient scores a perfect $50.0$). Answering it would have required inventing the
> boundaries. We therefore measure the swing against the score's actual *resolution*: the
> between-subject sd is $10.01$, so the augmented baseline's mean per-sequence swing of $3.01$ is
> $0.30$ sd and its worst sequence $2.01$ sd; and the swing exceeds the score gap separating $20.2\%$
> of subject pairs, rising to $83.0\%$ at the tail — pairs whose clinical ordering a camera
> relocation can invert. That is an in-dataset consequence with no invented constant. The
> Fugl-Meyer/STREAM band is retained as corroboration from adjacent instruments.

## The measurement (`src/clinical_resolution.py`)

Reads the five `KIMORE_processed/Exercise{k}/meta.csv` files (380 subject–exercise cells, 77 subjects).
Swing values are **read from the banked Block-3 aggregate**, never recomputed, so this analysis cannot
drift from the numbers in the paper.

| | mean $3.01$ | p95 $8.65$ | max $20.12$ |
|---|---|---|---|
| as between-subject sd ($10.01$) | $0.30$ | $0.86$ | **$2.01$** |
| % of all subject pairs reorderable | **$20.2$** | $47.7$ | **$83.0$** |
| % of patient-vs-control pairs | $12.8$ | $34.3$ | $74.2$ |
| × median within-subject sd ($4.97$) | $0.60$ | $1.74$ | $4.05$ |

Independently recomputed through a separate stdlib-only path: $14251$ pairs, $20.17\%$, sd $10.0146$ —
exact match. Artifacts at `outputs/clinical_resolution/` (gitignored; regenerate by running the script).

## Why the band question cannot be answered

Per exercise, lowest-scoring control vs highest-scoring patient — **every row inverted**:

| | lowest control | highest patient | margin |
|---|---|---|---|
| Es1 | $34.00$ (NE\_ID10) | $50.00$ (P\_ID3) | $+16.00$ |
| Es2 | $23.00$ (NE\_ID10) | $50.00$ (B\_ID7) | $+27.00$ |
| Es3 | $30.00$ (NE\_ID10) | $48.00$ (P\_ID1) | $+18.00$ |
| Es4 | $26.00$ (NE\_ID11) | $42.33$ (B\_ID6) | $+16.33$ |
| Es5 | $19.67$ (NE\_ID2) | $50.00$ (P\_ID6) | $+30.33$ |

**The trap we walked past.** `src/clinical_analysis.py` contained a hardcoded `CLINICAL_BANDS` dict
(`CG/Expert: (38,50)`, `GPP/Parkinson: (5,35)`) commented as "typical score bands." Both are falsified
by the rows above. It was the obvious thing to reach for in answering this reviewer, and using it would
have traded a *disclosed* weakness (borrowed MCID) for an *undisclosed* fabrication. Deleted, with a
comment recording why so it is not reintroduced. If the reviewer presses on bands, this is the answer:
we looked, the bands are not there, and we say so.

## Precision that matters

Do **not** claim this shows the score fails to discriminate pathology. It does not discriminate — the
table above proves that — but that is not our claim and conceding it as one invites "then why model the
score at all?" The claim is about **resolution**: a camera move perturbs the clinical output by an
amount comparable to the differences that output is meant to express.

Do **not** run a group-discrimination AUC as supporting evidence. It answers a question we are not
asking and hands the reviewer the framing above.

Do **not** describe the within-subject spread as a reliability coefficient. Different exercises are
different measurements, so it is an *upper bound* on within-subject noise — which is what makes it a
conservative yardstick, and that conservatism is the point worth stating.

## Anticipated follow-ups

**"Your pair-reordering statistic is inflated by near-ties."** Report the median separation alongside
it: $9.00$ overall, $12.67$ for patient-vs-control. The $20.2\%$ figure is against a distribution whose
centre is well above the swing, so it is not a near-tie artifact. The patient-vs-control stratum is the
conservative read and is reported.

**"Five subjects have odd scores."** B\_ID3, B\_ID4, P\_ID10, S\_ID5, S\_ID6 carry values that are not
multiples of $1/3$, unlike every other subject's rater averages. Detected programmatically, not
hardcoded, and every statistic is reported twice — with and without them. Excluding them moves nothing
material ($0.30$ sd, $21.3\%$).

**"Why 77 subjects, not 78?"** `KIMORE_processed` holds $380$ of a nominal $385$ subject–exercise cells
across $77$ subjects; five are absent upstream in the released data. Stated in the supplement, and
consistent with the $n{=}77$ already quoted elsewhere in the paper.

**"Get a clinical co-author."** Not addressed by this analysis and we should not pretend otherwise. The
honest position: the in-dataset consequence stands on its own, and clinician endorsement would
strengthen the interpretation rather than the measurement.

## What changed in the paper

1. Intro clinical paragraph: added a five-step passage — acknowledge the band premise, show it is false
   with the Es5 inversion, pivot to score resolution, give the numbers, keep the MCID as corroboration.
   The MCID is demoted from load-bearing to corroborating.
2. Results, abstract and Deployment: the swing and the int8 cliff are now also expressed in
   between-subject sd ($2.01$ sd; $0.005$ sd), so the safety claim no longer rests solely on a borrowed
   number.
3. New supplement section "Score Resolution, and Why Not Clinical Bands" with the inversion and
   pair-separation tables and the odd-subject note.
4. Deleted the fabricated `CLINICAL_BANDS` dict from `src/clinical_analysis.py`.
5. New `src/clinical_resolution.py`.
