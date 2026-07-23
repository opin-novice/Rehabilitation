# Execution Plan — Phase 2: Pre-Submission Scientific Hardening

> **For the coding agent (Deepseek V4 Flash):** Step-by-step task list. Do tasks **in order**.
> Each task has WHY / FILES / EXACT CHANGES / VERIFY. Run VERIFY before moving on.
> Working dir `D:\Rehabilation`. Shell PowerShell. Python `python`.
> **Do NOT delete or retrain anything in `outputs/loso_*/`.**

> ## ⏯️ RESUME STATUS — START AT TASK 4
> **Tasks 1, 2, 3 are ALREADY DONE and verified — DO NOT redo them.**
> - ✅ **Task 1** — `src/irds_eval.py`: `degenerate` flag added (`pred_SD<0.10`); warning printed;
>   `degenerate` column in `irds_reliability.csv`; report caveat added. Verified: GraphTransformer=True, others False.
> - ✅ **Task 2** — `src/irds_eval.py`: all ICC labels now read **ICC(2,1)**; zero "ICC(1,1)" strings remain.
> - ✅ **Task 3** — `src/sample_level_stats.py`: helper replaced with `bootstrap_mean_rho_ci(df)`
>   (stratified per-exercise resample → mean of per-exercise rho); call site updated.
>   ⚠️ NOT yet re-run end-to-end — run `python src/sample_level_stats.py` once and confirm each
>   model's `Mean rho 95% CI` brackets its own `Mean rho` before continuing.
>
> **Begin work at Task 4. Then do 5, 6, 7, 8 and the Final Verification.**

---

## Context

Phase 1 (evidence engineering) is complete and verified. A second expert review then found
**6 valid scientific problems** that would draw reviewer fire or a desk-reject. All 6 were
confirmed against the actual code/outputs:

1. **ICC label mismatch** — `src/irds_eval.py` computes ICC(2,1) (formula in `icc_two_way`) but
   the module docstring (line ~15) and the print (line 386) call it "ICC(1,1)"; the paper Table 6
   says "ICC(2,1)". Desk-reject risk.
2. **GraphTransformer is degenerate on IRDS** — predicts a near-constant ~41.4 for every input
   (pred_SD=0.03, within=0.00, S/B ratio=nan). Its ICC=0.953 / Kendall W=0.533 are therefore
   spuriously inflated (corroborated: W=0.480±0.124, wildly unstable across embeddings). Must be
   exposed and caveated, not reported as a clean result.
3. **The "dissociation" headline is not statistically significant** — F5 correlation between
   KIMORE rho and IRDS Kendall W across the 7 models is r=−0.393, **p=0.383** (N=7 models, IRDS
   N=10 subjects). Cannot be stated as a proven population effect.
4. **"LSTM is 2nd-best on KIMORE"** implies a rank order that is not significant after FWER.
5. **No human-rater reliability baseline** — ICC≈0.90 has no clinical anchor without the
   inter/intra-rater reliability of the original KIMORE physiotherapist scores.
6. **Bootstrap CI is the wrong statistic** — point estimate `Mean rho` = mean of per-exercise
   Spearmans, but `bootstrap_spearman_ci` returns a single *pooled* all-exercise Spearman CI
   (`src/sample_level_stats.py:205-211`). Proof: TCN mean=0.549 but CI=[0.529, 0.661], not centered.

**Honest verdict:** the science is rigorous but the paper currently has **no statistically
supported headline** (null KIMORE comparison + dissociation p=0.38 + one degenerate model).
This plan repositions it as a **rigorous, hypothesis-generating benchmark + methodology paper**
and makes every claim match the statistics. The most solid contribution is the **protocol-inflation
quantification** — lead with it.

**Surviving qualitative finding (defensible):** even excluding the degenerate GraphTransformer,
TCN (best KIMORE rho=0.549) has low IRDS consistency (W=0.211) while GT-no-bias (mid-pack
rho=0.451) has the best (W=0.608). State as a *case observation*, not a population correlation.

---

## STRATEGIC NOTE (read before Task 1)

Two paths exist; this plan executes Path A (achievable now). Path B is flagged as optional.
- **Path A (default): fix + reframe honestly, submit.** All tasks below. Targets *Computers in
  Biology and Medicine* / *Biomedical Signal Processing & Control* as a benchmark+methodology study.
- **Path B (optional, larger): raise the ceiling.** Acquire more IRDS subjects / a 3rd dataset to
  actually power the dissociation test, and fix GraphTransformer's OOD collapse. Out of scope for a
  coding agent; note as future work unless the user explicitly asks.

---

## TASK LIST (severity order)

### Task 1 — Expose & caveat the GraphTransformer degeneracy (most serious)

**WHY:** A model that outputs a near-constant value has trivially inflated ICC/W. Reporting it
as a clean result is misleading.

**FILES:** `src/irds_eval.py`; later `paper_outline.md` (Task 8).

**EXACT CHANGES:**
1. In `evaluate_reliability` return dict, ensure `pred_SD`, `within_subject_SD`,
   `between_subject_SD`, `SB_ratio` are already present (they are). Add a boolean
   `"degenerate": bool(pred_SD < 0.10)` to the dict.
2. In `main()`, when printing each model, if `res["degenerate"]` print a clear warning line:
   `"  [WARNING] DEGENERATE: pred_SD<0.10 — ICC/W are not interpretable for this model."`
3. In the saved `irds_reliability.csv`, the `pred_SD` and `SB_ratio` columns must be retained
   (they are) and a new `degenerate` column added so the flag is machine-readable.
4. Do **not** delete GraphTransformer; transparency > omission. The caveat goes in the paper (Task 8).

**VERIFY:**
```powershell
python src/irds_eval.py
python -c "import pandas as pd; d=pd.read_csv('outputs/irds_eval/irds_reliability.csv'); print(d[['model','pred_SD','SB_ratio','degenerate']])"
```
Expect GraphTransformer row flagged `degenerate=True`; others False.

---

### Task 2 — Make ICC labeling consistent (desk-reject fix)

**WHY:** Code prints "ICC(1,1)" but computes ICC(2,1); paper says ICC(2,1).

**DECISION (recommended): keep the ICC(2,1) computation (it is correct as implemented) and
relabel everything to ICC(2,1)**, plus add a one-line justification. (Rationale: repetitions are
treated as a random sample of occasions → two-way random, absolute-agreement ICC(2,1) is the
standard test-retest choice. Reimplementing a one-way ICC risks a new bug.)

**FILES:** `src/irds_eval.py`.

**EXACT CHANGES:**
1. Module docstring (~line 15): change "Test-Retest ICC (ICC1,1)" → "Test-Retest ICC (ICC(2,1),
   two-way random, absolute agreement; repetitions treated as random occasions)".
2. Print line 386: change "ICC(1,1) test-retest" → "ICC(2,1) test-retest".
3. In the report `.txt` writer, ensure any ICC mention reads "ICC(2,1)".
4. Confirm `paper_outline.md` Table 6 already says ICC(2,1) (it does) — no change needed there.

**VERIFY:**
```powershell
Select-String -Path src/irds_eval.py -Pattern "ICC\(1,1\)|ICC1,1"
```
Expect **zero matches**.

---

### Task 3 — Fix the bootstrap CI to match the point estimate

**WHY:** CI must be for "mean of per-exercise Spearman", not pooled Spearman.

**FILES:** `src/sample_level_stats.py`.

**EXACT CHANGES:** Replace `bootstrap_spearman_ci` with a version that takes the full per-model
`df` and, on each of n_boot resamples, **resamples rows within each exercise (stratified)**,
recomputes per-exercise Spearman, averages them, and collects that mean. Return the 2.5/97.5
percentiles. Update the call site (line ~209) to pass `df` instead of the two arrays.

```python
def bootstrap_mean_rho_ci(df, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    eids = [e for e in range(5) if (df["exercise_id"] == e).sum() >= 5]
    sub = {e: df[df["exercise_id"] == e] for e in eids}
    means = []
    for _ in range(n_boot):
        rhos = []
        for e in eids:
            s = sub[e]
            idx = rng.integers(0, len(s), len(s))
            yt = s["y_true"].values[idx]; yp = s["y_pred"].values[idx]
            r, _ = spearmanr(yt, yp)
            if not np.isnan(r):
                rhos.append(r)
        if rhos:
            means.append(np.mean(rhos))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)
```

**VERIFY:**
```powershell
python src/sample_level_stats.py
python -c "import pandas as pd; d=pd.read_csv('outputs/sample_stats/per_exercise_spearman.csv'); print(d[['Model','Mean rho','Mean rho 95% CI']].head(8))"
```
Expect each CI to now bracket its own `Mean rho` (e.g. TCN CI centered near 0.549, not 0.595).

---

### Task 4 — Reframe the dissociation honestly + report F5 p-value

**WHY:** r=−0.393, p=0.383 cannot support a "proven dissociation" claim.

**FILES:** `src/make_figures.py` (F5 caption text); `paper_outline.md` (Task 8 covers prose).

**EXACT CHANGES:**
1. In `make_figures.py` F5, keep the title but add a subtitle line:
   `"N=7 models; correlation NOT significant (exploratory)"`. Ensure the printed/embedded text
   states r and p explicitly.
2. Anywhere the figure or its caption implies a trend, change wording to
   "no significant rank-order relationship (r=-0.39, p=0.38) — individual models nonetheless
   show divergent KIMORE-vs-IRDS behaviour (e.g. BiLSTM)".

**VERIFY:**
```powershell
python src/make_figures.py
Get-ChildItem outputs/figures/fig5_kimore_vs_irds.png
```
Open F5; confirm it states r, p, and "exploratory / not significant".

---

### Task 5 — Add a pairwise effect-size heatmap (Figure F6)

**WHY:** Rank-biserial r is printed but not tabulated; reviewers want it at a glance.

**FILES:** `src/make_figures.py`.

**EXACT CHANGES:** Add F6 `fig6_pairwise_effect_heatmap.png`: read
`outputs/sample_stats/pairwise_sample_level.csv`, build a symmetric model×model matrix of the
`Effect r` values (diagonal = 0), render as an annotated heatmap (colorblind-friendly, e.g.
`cmap="cividis"`), title "Pairwise effect size (rank-biserial r) — KIMORE abs-error". Mark the two
FWER-significant pairs (TCN>Ridge, LSTM>Ridge) with an asterisk in the annotation.

**VERIFY:**
```powershell
python src/make_figures.py
Get-ChildItem outputs/figures/fig6_pairwise_effect_heatmap.png
```

---

### Task 6 — Journal-quality figure style pass

**WHY:** matplotlib defaults are not submission-grade.

**FILES:** `src/make_figures.py`.

**EXACT CHANGES:** At the top, set a shared style applied to every figure:
`plt.rcParams` with `font.size=12`, `font.family="DejaVu Sans"`, `axes.spines.top/right=False`,
`figure.dpi=300`, `savefig.bbox="tight"`. Define one **colorblind-friendly** model→color dict
(Okabe-Ito palette) and use it consistently across F1, F2, F4, F5. Keep all existing figure logic.

**VERIFY:**
```powershell
python src/make_figures.py; Get-ChildItem outputs/figures/*.png
```
Expect all 6 PNGs regenerated; spot-check one for consistent palette + no top/right spines.

---

### Task 7 — Human-rater reliability baseline (writing + literature)

**WHY:** Needed to anchor "ICC=0.90 is excellent" for a clinical journal.

**FILES:** new `human_rater_baseline.md` (notes); later folded into `paper_outline.md` Section 3/5.

**EXACT CHANGES:** Create `human_rater_baseline.md` recording: (a) whether Capecci et al. 2019
(the KIMORE dataset paper) reports inter-/intra-rater reliability of the clinical scores — quote
the exact value + citation if present; (b) if absent, state that explicitly and cite 2–3
physiotherapy movement-quality rating-reliability ranges from the literature (typical clinical
ICC 0.6–0.9) as context. Add a one-paragraph Methods insert and a one-sentence Limitations insert.
> NOTE: this requires reading the KIMORE paper; if the agent cannot access it, leave a clearly
> marked `[TODO: confirm Capecci 2019 rater ICC]` placeholder rather than inventing a number.

**VERIFY:** `Test-Path human_rater_baseline.md` → True; no fabricated numbers (placeholders allowed).

---

### Task 8 — Propagate honest framing into `paper_outline.md`

**WHY:** Abstract/Results/Discussion must match the corrected statistics.

**FILES:** `paper_outline.md`.

**EXACT CHANGES:**
1. **"2nd-best" / ordering language** (everywhere): replace with "numerically highest / second
   highest mean rho (pairwise differences not significant after Holm-Bonferroni)".
2. **Abstract + Discussion 5.2**: state the dissociation as exploratory — "KIMORE rank and IRDS
   cross-exercise consistency showed no significant correlation (r=-0.39, p=0.38, N=7 models);
   however individual architectures diverged markedly (BiLSTM: strong KIMORE, near-random IRDS
   consistency W=0.047)". Lead the contributions with **protocol inflation**.
3. **Table 6**: add `pred_SD` and `degenerate` columns; add caption sentence: "GraphTransformer
   collapsed to near-constant predictions on IRDS (pred_SD=0.03); its ICC/W are not interpretable
   and it is excluded from the consistency discussion."
4. **Limitations 5.5**: add bullets for (a) IRDS N=10 subjects → wide CIs, dissociation is
   hypothesis-generating; (b) GraphTransformer OOD collapse; (c) human-rater ICC anchor (Task 7).
5. **Conclusion**: soften to "we provide a rigorously-evaluated benchmark and a cautionary
   protocol-inflation analysis" rather than any superiority/generalization claim.

**VERIFY:**
```powershell
Select-String -Path paper_outline.md -Pattern "2nd-best|second-best|best on KIMORE"
Select-String -Path paper_outline.md -Pattern "p=0.38|exploratory|pred_SD|degenerate"
```
Expect the first to return no superiority-implying phrasing; the second to show the new honest text.

---

## FINAL VERIFICATION (run in order)

```powershell
python src/irds_eval.py
python src/sample_level_stats.py
python src/make_figures.py
Select-String -Path src/irds_eval.py -Pattern "ICC\(1,1\)|ICC1,1"          # expect: none
python -c "import pandas as pd; d=pd.read_csv('outputs/irds_eval/irds_reliability.csv'); print(d[['model','pred_SD','degenerate']])"
python -c "import pandas as pd; d=pd.read_csv('outputs/sample_stats/per_exercise_spearman.csv'); print(d[['Model','Mean rho','Mean rho 95% CI']].head(8))"
Get-ChildItem outputs/figures/*.png                                          # expect 6 PNGs
Select-String -Path paper_outline.md -Pattern "exploratory|degenerate"       # expect matches
```

**Done when:**
- No "ICC(1,1)" strings remain in code; all ICC labels read ICC(2,1).
- `irds_reliability.csv` has a `degenerate` flag (GraphTransformer=True).
- Each model's Spearman CI brackets its own mean rho.
- F5 states r/p and "exploratory"; F6 heatmap exists; 6 figures total, consistent palette.
- `human_rater_baseline.md` exists (real citation or marked TODO).
- `paper_outline.md` contains no significance-implying ordering language; dissociation framed as
  hypothesis-generating; GraphTransformer degeneracy disclosed.

## Out of scope (human decisions)
- Acquiring more IRDS subjects / a 3rd dataset (Path B).
- Fixing GraphTransformer's OOD collapse (architecture research, not a fix-up).
- Final journal selection and the clinical co-author.
