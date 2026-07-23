# WACV 2026 (Applications) — Finalized Cut List (17pp journal → 8pp main + supplementary)

Source of truth for the condensation. Main body target **≤ 8pp excl. references** under `wacv.sty`
(`\usepackage[review,applications]{wacv}`). The 17pp IEEEtran twins (`paper.tex`/`paper_overleaf.tex`)
are **frozen** as the arXiv extended version. New primary source: `paper_wacv.tex` (+ `supp_wacv.tex`).

## Locked decisions (from user, this session)
- **Edge section:** compress `tab:precision` + `tab:streaming` → **one deployment table + inline numbers** (~1.25pp).
- **Negative results:** protocol-null and irregular-sampling-null → **one sentence each in main**, tables (`tab:protocol`, `tab:resampling`) → SUPP.
- **EGNN:** **fold the EGNN row into `tab:pareto`** (drop standalone `tab:egnn`); "steerable vs lighter E(n)" made inline.

## Section-by-section mapping

| Content | Source | Verdict | Note |
|---|---|---|---|
| §I Introduction | paper.tex:64 | MAIN (trim ~40%) | Abstract+intro **lead with per-sequence trust** (3.03 vs 9×10⁻⁶) |
| §II Related Work (5 subsecs) | paper.tex:102 | MAIN (merge 5→2¶) | Equivariance / learned-invariance-price / missing-joints |
| §III invariant cut definition | methodology.tex:164 | MAIN | Keep Eq.(cut) + Eq.(pi) |
| Prop: SO(3)-invariance of read-out | methodology.tex:278 | MAIN (statement); **proof → SUPP** | THE theorem |
| `tab:invsets` (283-d parity spec) | methodology.tex:227 | SUPP | 1-line dim summary in main |
| Props/Remarks (biases; chirality-free; reflection≠LR; CDE-norm ×2) | meth:148,297,310,471,503 | SUPP | |
| Numerical certificate E1–E8 | methodology.tex:373 | MAIN (1¶ digest); full → SUPP | Keep 2.9×10⁻¹⁶ + 5.8×10⁻⁹ |
| CDE control + spectral calibration | meth:422,525 | SUPP | 1 honest sentence in main ("CDE fails the floor") |
| §IV `tab:protocol` | results.tex:74 | **SUPP** (1 sentence main) | negative result |
| §IV `tab:accuracy` | results.tex:153 | MAIN (compact) | "the tie is the point" |
| §IV `tab:viewpoint` | results.tex:234 | **→ HERO FIGURE (MAIN)** | flat EGRU vs PCT blow-up |
| §IV `tab:resampling` | results.tex:326 | **SUPP** (1 sentence main) | negative result |
| §IV `tab:nodefail` + `tab:nodefail_modes` | results.tex:441,482 | MAIN **3-row digest**; full 5-op → SUPP | the differentiator |
| §IV `tab:egnn` | results.tex:526 | **FOLD into `tab:pareto`** | drop standalone float |
| §IV `tab:pareto` (whole-thesis grid) | results.tex:559 | MAIN | money table; +EGNN row |
| §IV `tab:ablation` (per-family) | results.tex:619 | SUPP | |
| §V `tab:precision` + `tab:streaming` | paper.tex:177,230 | MAIN **merged → 1 table** | int8 cliff + 112fps/TTFS inline |
| §VI Conclusion | paper.tex:253 | MAIN (~4 lines) | |
| **Second corpus (Workstream C, NEW)** | — | MAIN panel **if clean, else SUPP** | reserve ~0.4pp; droppable |

## Main-paper float budget (target ≤ 8pp)
1. **Fig 1** — hero: viewpoint curve (EGRU flat vs PCT ↑ with azimuth)  ← replaces `tab:viewpoint`
2. **Tab 1** — clean-accuracy tie (`tab:accuracy`, compact)
3. **Tab 2** — node-failure 3-row digest
4. **Tab 3** — Pareto grid **+ EGNN row** (`tab:pareto`)
5. **Tab 4** — edge deployment (merged precision+streaming)
6. *(opt)* **Tab 5 / panel** — second-corpus replication

Rough budget: Intro 1.0 · RelWork 0.75 · Method 2.0 · Results 2.75 (incl. Fig 1) · Edge 1.25 · Concl 0.25 ≈ **8.0pp**.

## Depends-on (experiments feeding main)
- Tuned EGNN numbers (Workstream B) → Pareto EGNN row.
- Second-corpus results (Workstream C) → Tab 5 / panel (droppable).
- Hero figure → plot from `tab:viewpoint` data (reuse `src/plot_ablation.py` patterns).

## Integrity carry-over (Workstream D)
- ✅ chirality cost +0.11 (done in journal twins; port corrected value).
- ⬜ verify "16.7%" param claim vs true `sum(p.numel())` counts.
- ⬜ seed-std clause (EGRU SO(3) 0.30 vs 0.08) — parity-odd capacity, not instability.
- ⬜ Q7 seed-distribution → one-line footnote (R1 letter is NOT in the WACV package).
