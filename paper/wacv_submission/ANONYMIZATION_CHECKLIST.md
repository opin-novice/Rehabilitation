# WACV 2026 Double-Blind — Anonymization Checklist

WACV review is double-blind. This tracks what is already safe and what must be scrubbed before the
submission (and any linked code) goes out. Status as of this pass.

## ✅ Already clean (submission PDFs)
- `paper_wacv.tex` / `supp_wacv.tex`: `\author{}` empty, compiled with `\usepackage[review,applications]{wacv}`
  (anonymizes + adds line numbers). No author names, affiliations, emails, URLs, funding, or
  acknowledgments anywhere in either source (grep-verified: north south / nsu / dhaka / bangladesh /
  opin / rafat / shafin / rahman / github / apurba / acknowledge / grant — none present).
- The target-paper citation `\bibitem{pct2026}` is **title + venue only** (no author names), so citing
  it does not reveal the relationship. Keep it that way.

## ✅ Now done (this pass)
- **`src/` code neutralized.** `models_curvenet.py`, `models_stgcn.py`, and `train_reproduce.py` no
  longer name the target's authors, the personal repo, or "the research team" — all replaced with
  neutral third-person "the target point-cloud-transformer baseline / paper." Grep-verified: no
  `opin-novice / King-Rafat / Rafat / Kazi / Tiange / North South / shafin / research team` left in
  `src/`. (Original CurveNet method name retained — that is standard attribution, not identifying.)

## ⚠️ Must still fix BEFORE submitting (external to the PDFs)
1. **Code release link.** Do NOT put a real GitHub URL in the paper. The **demo** still references
   `github.com/opin-novice/Rehabilitation` (in `demo/README.md`, `demo/CAMERA_TEST_GUIDE.md`,
   `demo/CAMO_SETUP.md`). If you release code for review, use an anonymized mirror
   (`anonymous.4open.science`) and do not ship the `demo/` docs, or scrub the URL from them first.
2. **Non-code docs still carry names** (only a risk if bundled with a code release, NOT in the PDFs):
   `docs/planning/*`, `docs/reference/rehabilitation_transformer_report.md`, `history.md`, and the
   `paper/arxiv_extended/` twin. Exclude these from any review-time code drop, or scrub them.
3. **The target paper is your own group's / lab-adjacent work.** `pct2026` (target authors, same
   institution as the archived Paper-2 manuscript) — self-citation-as-target is allowed under
   double-blind ONLY in third person, which the paper already does ("the point-cloud transformer,"
   "the target"). Never add "our prior work" phrasing. Fine as written; just never relax it.

## 🔵 Decision deferred (Tier 0.2 — arXiv-twin timing)
The 17pp IEEEtran twin in `paper/arxiv_extended/` is a near-identical, same-title version. WACV permits
prior arXiv posting, but a searchable twin under real names lets a reviewer de-anonymize in one query.
Options (pick before the review period):
- **(A) Safest:** do not post the arXiv twin until after WACV decisions. Recommended.
- **(B) If you must post:** post it anonymized (no author block) and do not reference "the extended
  version" from the submission (the submission currently does not — keep it that way).
Either way: do not cite, link, or hint at the twin inside `paper_wacv.tex` / `supp_wacv.tex`.

## Pre-submission grep (run once more right before upload)
```
grep -niE "north south|nsu|dhaka|bangladesh|opin|rafat|shafin|rahman|github|apurba|acknowledge|grant|\\\\author\\{[^}]" paper_wacv.tex supp_wacv.tex
```
Expect: only the empty `\author{}` lines. Anything else = leak.
