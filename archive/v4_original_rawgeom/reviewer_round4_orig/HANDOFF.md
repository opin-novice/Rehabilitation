# Round-4 revision — handoff (COMPLETE)

Reviewer verdict: **Major Revision**. All R4-T1..T7 done. manuscript.tex/.pdf canonical
(10 pp, compiles clean: no undefined refs, no overfull boxes). manuscript.md fully synced
(Tables renumbered to 1-9, Figure 1 added). RESPONSE_TO_REVIEWERS.md has Round-4 section.
All work was analysis/text on ALREADY-TRAINED models — no GPU retraining.

## Final state (2026-07-04)
- T1 auroc-sig -> tab:auroc-sig (Table 3 in .md) + §IV-A paragraph. DONE.
- T2 shared-joint probe -> 1.000 both backbones, folded into §IV-A sensor-probe para. DONE.
- T3 pairwise Wilcoxon -> tab:pairwise-full (Table 7 in .md) + Table V note. DONE.
- T4 t-SNE -> figures/embedding_tsne.png = fig:embedding-tsne (Figure 1). DONE.
- T5 softened definitive->robust + reframed 4 claim sites (abstract/contrib/IV-A/concl). DONE.
- T6 DA/DG one-per-family + IRM/SWAD/TENT, +2 bibitems (arjovsky2019irm, cha2021swad). DONE.
- T7 subject-clustering + LOSO-leaky caveats added; compiled; .md synced; response letter. DONE.

---

## Original plan (archived below)

## DONE
- **T1 (concerns #1,#2,#6): subject-clustered AUROC significance** — `src/reviewer_round4_stats.py`
  -> `outputs/reviewer_round4/auroc_significance.{json,md}` (+ cached preds_*.npz).
  **KEY RESULT (drives everything):** with a subject-cluster bootstrap over the 10 target
  subjects, orientation fixed on the full sample:
  - REHAB246: NO learned model > chance (all CIs include 0.5); only naive clears it
    (0.554, CI[0.506,0.605], p=0.035).
  - UI-PRMD: only Scratch (0.527) and naive (0.538) > 0.5, trivial effect.
  - naive−model paired diffs NOT significant for ~all conditions (p 0.24-0.71).
  => "naive unbeaten" does NOT survive. Reframe thesis (user-approved) to:
     "no method — learned or naive — achieves subject-level-meaningful cross-sensor
     discrimination; learned models are indistinguishable from chance at the subject level."
- **T3 (concern #5): full 10-pair Wilcoxon matrix** — extracted from
  `results/kimore_loso_78fold/stats78.json` (pairwise_wilcoxon). Pattern: scratch ≈
  contrastive_ft ≈ masked_ft (all n.s.); every LP-vs-FT/scratch pair sig (p_adj<1e-11).
  Build appendix table `tab:pairwise-full`; note in Table V why only vs-scratch shown.

- **T2 (concern #7): shared-joint sensor probe** — RAN. `outputs/reviewer_round4/probe_sharedjoints.{json,md}`.
  With {7,11,22,23,24} zeroed across ALL corpora, 3-way probe stays **1.000** for BOTH
  TCN and ST-GCN (chance 0.33). => sensor separation is NOT a zero-padding artifact.
- **T4 (concern #4): t-SNE figure** — RAN. `figures/embedding_tsne.png` (2x2 TCN+ST-GCN x
  sensor/label). Left col: three sensors cleanly separated. Right col: correct/incorrect
  fully intermixed within each sensor cluster. Visual proof of the 1.00 probe.

## TEXT EDITS REMAINING (manuscript.tex is canonical; regen manuscript.md after)
- **T5 (concern #3):** soften "definitive"->"robust/strong evidence"; rewrite the
  "naive unbeaten" claims in Abstract, contribution #1, Sec IV-A, Conclusion per T1 result.
  Add new table `tab:auroc-sig` (from auroc_significance.md) in Sec IV-A.
- **T6 (concern #8):** DA/DG depth — justify CORAL/AdaBN/DANN as one-per-family; cite IRM,
  SWAD, TTA as untested alternatives (+2 bibitems: arjovsky2019irm, cha2021swad).
- **T7 minors + close-out:** add \begin{figure} for embedding_tsne.png + tab:pairwise-full;
  tighten Sec IV-F redundancy; restate LOSO-vs-leaky caveat near rho=0.836 vs 0.744;
  add subject-clustering caveat to Limitations; then: pdflatex x2 (verify figure floats,
  no undefined refs), regenerate manuscript.md from .tex, add Round-4 section to
  RESPONSE_TO_REVIEWERS.md.

## GOTCHAS
- manuscript.md is a manual convenience copy (no pandoc on machine) — regenerate from .tex.
- Table numbering in .md: datasets=1, zeroshot=2, naive-sens=3, robustness-c=4, loso=5,
  pool=6, deg=7; new auroc-sig + pairwise-full tables will renumber — keep .md in sync.
- AUROC uses direction-agnostic max(a,1-a) per fold in older scripts; T1 fixes orientation
  on full sample then freezes it (the correct, non-optimistic approach) — keep that.
