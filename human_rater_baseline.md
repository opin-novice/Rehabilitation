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
