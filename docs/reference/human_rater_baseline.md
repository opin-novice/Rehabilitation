# Human-Rater Reliability Baseline

Context anchor for interpreting automated movement-quality scores: how reliably do
*human* clinicians agree when rating the same rehabilitation performance? This sets the
practical ceiling any automated system can be expected to match.

## (a) Does Capecci et al. 2019 (KIMORE) report rater reliability?

Capecci, Ceravolo, Ferracuti, Iarlori, Monteriu, Romeo & Verdini, "The KIMORE Dataset:
KInematic Assessment of MOvement and Clinical Scores for Remote Monitoring of Physical
REhabilitation," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*,
vol. 27, no. 7, pp. 1436-1448, 2019. DOI: 10.1109/TNSRE.2019.2923060.

The KIMORE clinical scores were assigned by clinicians via a structured clinical
questionnaire (cTS/pTS components). However, an explicit **inter-/intra-rater
reliability coefficient (e.g., ICC) for those clinician scores could not be confirmed**
from the accessible record during this review.

> **[TODO: confirm Capecci 2019 rater ICC]** - targeted web searches (June 2026) did not
> surface any reported inter-/intra-rater ICC (or Cohen's/Fleiss' kappa) for the KIMORE
> clinical scores; the dataset paper appears to report clinician scores from a structured
> questionnaire without an accompanying rater-agreement coefficient. This remains to be
> confirmed against the full IEEE TNSRE 2019 text. Do **not** fabricate a number; if the
> paper reports none, state that explicitly in the manuscript (as done here).

## (b) Physiotherapy movement-quality rating reliability from the literature

Where a dataset does not report rater reliability, the broader physiotherapy literature
provides a defensible context band. Observational movement-quality ratings by trained
physical therapists typically fall in a **moderate-to-good ICC range (~0.6-0.9)**, with
substantial spread driven by tool, rater expertise, and movement complexity:

| Assessment context | Inter-rater | Intra-rater | Source |
|---|---|---|---|
| Cross-diagnostic Movement Quality Score (MQS), 2 PTs, 68 inpatients | **ICC[2,1] = 0.93** (excellent) | - | J. Rehabil. Med. (MQS inter-rater reliability & construct validity) |
| Cutting Movement Assessment Score (CMAS), trained PTs | ICC 0.58-0.91 | ICC 0.70-0.95 | Int. J. Sports Phys. Ther., CMAS reliability study |
| Upper-limb compensatory movements post-stroke (video) | ICC >= 0.75 (reaching/drinking/returning phases) | - | Mennella et al., J. NeuroEng. Rehabil. 21:217, 2024 (DOI 10.1186/s12984-024-01506-7) |
| Passive physiological accessory movement, lumbar spine | ICC 0.03-0.37 (below acceptable) | - | Man. Ther. (novice manual therapists) |

Takeaway: well-defined, well-trained protocols reach good agreement (ICC ~0.75-0.95),
but harder, less-structured judgements can fall well below ICC 0.4. The strongest directly
comparable anchor is the cross-diagnostic MQS at **inter-rater ICC[2,1] = 0.93**, which we adopt
as the human reliability ceiling; a realistic human benchmark band for rehabilitation
movement-quality rating is **ICC ~0.6-0.93**.

## Methods insert (one paragraph)

> As an interpretive anchor for the automated scores, we note the reliability of human
> raters performing the analogous task. The KIMORE clinical scores were assigned by
> clinicians through a structured questionnaire; the dataset paper does not provide a
> confirmed inter-/intra-rater ICC for these scores [TODO: confirm Capecci 2019 rater
> ICC]. As external context, observational movement-quality ratings by trained physical
> therapists typically achieve moderate-to-good agreement (ICC ~0.6-0.9; e.g., CMAS
> inter-rater ICC 0.58-0.91 and intra-rater 0.70-0.95; post-stroke upper-limb phase
> ratings ICC >= 0.75), while less-structured judgements can fall below ICC 0.4. Model
> performance should be read against this human ceiling rather than against perfect
> agreement.

## Limitations insert (one sentence)

> Because the KIMORE clinical scores lack a confirmed inter-/intra-rater reliability
> coefficient, our Spearman/Kendall agreement metrics are benchmarked against an assumed
> human ceiling (ICC ~0.6-0.9 from the broader physiotherapy literature) rather than a
> dataset-specific human-rater baseline.

---
*Sources:* Int. J. Sports Phys. Ther. (CMAS inter/intra-rater reliability);
J. NeuroEng. Rehabil. 21:217 (2024), DOI 10.1186/s12984-024-01506-7 (post-stroke
upper-limb compensatory-movement rating reliability); Man. Ther. (passive accessory
movement reliability in novice manual therapists). Capecci et al., IEEE TNSRE 27(7):
1436-1448, 2019, DOI 10.1109/TNSRE.2019.2923060.

## (c) 2026-07-23 follow-up: MCID (magnitude-of-change) anchor, not just ICC (agreement)

A second targeted search (WACV rebuttal, W1) confirmed the (a) finding still holds — no
inter-/intra-rater ICC/kappa for KIMORE or REHAB24-6 is surfaced by search engines or the
papers' abstracts/summaries (scispace, ResearchGate, Semantic Scholar, Zenodo record all
checked; the KIMORE questionnaire is the 10-item EAAQ, summed into TS/PO/CF, with no visible
reliability coefficient reported). No further hunting is warranted without direct full-text
access to Capecci et al. 2019's methods section.

However, a *different* and directly useful anchor was found: minimal clinically important
difference (MCID) values on comparable 0-66/0-100-point rehabilitation motor-assessment scales:

| Scale | MCID | Population | Source |
|---|---|---|---|
| Fugl-Meyer upper-extremity | 4-12.4 pts | chronic stroke | multiple (see search) |
| Fugl-Meyer lower-extremity | 6 pts | chronic stroke | PubMed 27086865 |
| Fugl-Meyer (TBI, UE/LE/Motor) | 6.2 / 3.2 / 8.4 pts | traumatic brain injury | tandfonline 10.1080/14737175.2021.1968299 |
| Stroke Rehab. Assessment of Movement (STREAM) | 1.9-4.8 pts/subscale | stroke | Ovid nener 10.1177/1545968308316385 |

This is used in `paper_wacv.tex` (Results, "Clinical interpretation of the MAD scale" paragraph,
after `fig:viewpoint`) as a plausibility bound: our 3.03-MAD (of 50) per-sequence viewpoint swing
sits in the same order of magnitude as these MCIDs, and the int8/streaming score shifts
(<=0.05 MAD) sit one to two orders of magnitude below them. Explicitly flagged in the paper as
a bound from adjacent instruments, not a KIMORE-specific threshold — the missing KIMORE-specific
ICC/MCID remains the paper's clearest open measurement gap.

## (d) 2026-07-23 second follow-up: direct-fetch attempts, and re-anchoring on construct match

Tried to get past the search-snippet wall by fetching Capecci et al. 2019 directly rather than
relying on secondary summaries: `https://doi.org/10.1109/TNSRE.2019.2923060` (redirects to
`ieeexplore.ieee.org/document/8736767/`) returned no extractable text (publisher access wall), and
the ResearchGate copy (`researchgate.net/publication/333791841_...`) returned HTTP 403. Both are a
firm boundary, not a "search harder" gap — no further automated retrieval will get past them without
institutional/publisher access or a co-author who has it.

Given that, reprioritized the anchor in `paper_wacv.tex`: the paragraph now LEADS with the
movement-QUALITY rating ICCs (CMAS 0.58-0.93, MQS 0.93) from finding (b) above rather than the
Fugl-Meyer/STREAM MCIDs from finding (c). Rationale: CMAS/MQS rate the same construct KIMORE's
clinicians rate — how well a movement was executed — whereas Fugl-Meyer/STREAM measure a different
construct (post-stroke motor-function recovery) repurposed as a magnitude analogy. The MCID numbers
are kept as a secondary, explicitly-labeled magnitude-of-change cross-check, not the primary anchor.
The disclosure sentence in the paper now names the direct-fetch attempts specifically, so the
"we tried harder than a search-engine pass" claim is concrete and checkable by a reviewer.
