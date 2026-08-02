# -*- coding: utf-8 -*-
"""
clinical_resolution.py
======================
Is the camera-induced score swing large relative to the score differences KIMORE is
actually being asked to resolve?

WHY THIS EXISTS, AND WHY IT IS NOT THE OBVIOUS ANALYSIS
-------------------------------------------------------
The natural reviewer question is "does a 3.01-point swing move a patient across KIMORE's
clinical bands (Expert / Parkinson / Stroke)?"  That question has no honest answer,
because KIMORE's groups are not score strata.  On Es5 an Expert (E_ID14) scores 25.0
while a Parkinson patient (P_ID6) scores a perfect 50.0 -- the groups do not merely
overlap, they invert.  The group label records how the subject was RECRUITED, not what
band their score falls in.  Answering the band question would require inventing the
bands, which is exactly the fabrication this paper refuses elsewhere (we decline to
fabricate an ICC we could not find in the literature).

So we change the question from BAND CROSSING to SCORE RESOLUTION, which is answerable
from the dataset with no invented constant:

  (a) effect size    -- the swing as a fraction of the between-subject SD of the score.
  (b) ordinal cost   -- the fraction of subject pairs separated by less than the swing.
                        Those pairs are the ones a camera relocation can REORDER, and
                        ordinal comparison is what these scores are used for.
  (c) own-scale ref  -- each subject's spread across their own five exercises, giving
                        the instrument's natural variation as a comparator.

WHAT THIS DOES *NOT* CLAIM
--------------------------
This measures score RESOLUTION, not group DISCRIMINABILITY.  The claim is that a camera
move perturbs the clinical output by an amount comparable to the differences the output
is meant to express.  It is NOT a claim that the score classifies pathology -- given the
inversion above it plainly does not, and we do not need it to.

The swing values are measurements, not assumptions: they are read from the banked
aggregate (outputs/cde_block2/final_tables.json, arm "PCT + rot-aug"), which
src/aggregate_final.py derives from the Block-3 rotation sweep.  Nothing here recomputes
them.

Run:  python src/clinical_resolution.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Same source-of-truth constants the rest of the pipeline uses.
PROCESSED_ROOT = "KIMORE_processed"
SCORE_MAX = 50.0
N_EXERCISES = 5
EXPECTED_GROUPS = {"CG/Expert", "CG/NotExpert", "GPP/BackPain", "GPP/Parkinson", "GPP/Stroke"}

FINAL_TABLES = os.path.join("outputs", "cde_block2", "final_tables.json")
SWING_ARM = "PCT + rot-aug"     # the augmented baseline: the strongest form of the comparator


# ---------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------
def load_swings(path: str = FINAL_TABLES) -> dict:
    """Read the measured per-sequence viewpoint swing for the augmented baseline.

    We take these rather than recompute so that this script cannot silently disagree with
    the numbers already in the paper.  `max_degr` is the worst ANGLE's mean over sequences;
    `worst_degr_p95` / `worst_degr_max` are the per-sequence tail, which is what a
    per-patient claim needs.  A missing field is an error, never a flattering default.
    """
    with open(path) as f:
        blob = json.load(f)

    def walk(o):
        if isinstance(o, dict):
            if o.get("model") == SWING_ARM and "worst_degr_max" in o:
                yield o
            for v in o.values():
                yield from walk(v)
        elif isinstance(o, list):
            for v in o:
                yield from walk(v)

    cands = list(walk(blob))
    if not cands:
        raise SystemExit(f"No '{SWING_ARM}' entry with tail stats in {path}. "
                         f"Re-run src/aggregate_final.py before this script.")
    rec = cands[-1]
    swings = {"mean": float(rec["max_degr"]),
              "p95": float(rec["worst_degr_p95"]),
              "max": float(rec["worst_degr_max"])}
    for k, v in swings.items():
        if not np.isfinite(v):
            raise SystemExit(f"Swing '{k}' is not finite in {path}.")
    return swings


def load_scores(root: str = PROCESSED_ROOT) -> pd.DataFrame:
    """Concatenate the five per-exercise meta.csv files into one long frame."""
    frames = []
    for k in range(1, N_EXERCISES + 1):
        p = os.path.join(root, f"Exercise{k}", "meta.csv")
        if not os.path.exists(p):
            raise SystemExit(f"Missing {p}. Run src/prepare_kimore.py first.")
        df = pd.read_csv(p)
        df["exercise_k"] = k
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def audit_inputs(df: pd.DataFrame) -> dict:
    """Check what we actually loaded, and report it rather than assuming it.

    A silent path or filter change would otherwise produce entirely plausible wrong
    numbers, so the row/subject/group census is part of the artifact.
    """
    groups = set(df["group"].unique())
    audit = {"n_rows": int(len(df)),
             "n_subjects": int(df["subject_name"].nunique()),
             "n_exercises": int(df["exercise_k"].nunique()),
             "groups": sorted(groups),
             "rows_per_exercise": {int(k): int(v) for k, v in
                                   df.groupby("exercise_k").size().items()},
             "subjects_per_group": {g: int(n) for g, n in
                                    df.groupby("group")["subject_name"].nunique().items()}}
    if groups != EXPECTED_GROUPS:
        raise SystemExit(f"Group labels changed: got {sorted(groups)}, "
                         f"expected {sorted(EXPECTED_GROUPS)}.")
    if audit["n_exercises"] != N_EXERCISES:
        raise SystemExit(f"Expected {N_EXERCISES} exercises, got {audit['n_exercises']}.")
    # 78 subjects x 5 exercises = 390 is the nominal full grid; a handful of
    # subject/exercise cells are absent from the released data. We record the shortfall
    # instead of asserting it away.
    audit["nominal_full_grid"] = audit["n_subjects"] * N_EXERCISES
    audit["missing_cells"] = audit["nominal_full_grid"] - audit["n_rows"]
    return audit


def flag_odd_scores(df: pd.DataFrame, tol: float = 1e-6) -> list[str]:
    """Subjects whose scores are not multiples of 1/3.

    Every ordinarily-rated subject has a score of the form n/3 (three raters averaged).
    A handful of subjects carry dense floats instead (e.g. B_ID3 Es5 po_score 0.00514...),
    marking them as arriving by a different route in the released data. We detect this
    rather than hardcoding a subject list, and re-report every statistic without them so
    no headline number rests on them.
    """
    x = df["score"].to_numpy(dtype=float) * 3.0
    odd = np.abs(x - np.rint(x)) > tol
    return sorted(df.loc[odd, "subject_name"].unique().tolist())


# ---------------------------------------------------------------------------------------
# (a) Effect size against the population the instrument measures
# ---------------------------------------------------------------------------------------
def between_subject_sd(df: pd.DataFrame, swings: dict) -> dict:
    """Between-subject SD of the clinical Total Score, and the swing as a fraction of it.

    Reported per exercise, because the pooled SD across exercises also absorbs the
    difference between exercise means, which is not between-subject variation. The
    per-exercise mean is the primary figure; the pooled value is given alongside so the
    gap between the two is visible rather than hidden by the choice.
    """
    per_ex = {}
    for k, g in df.groupby("exercise_k"):
        sd = float(g["score"].std(ddof=1))
        per_ex[int(k)] = {"n": int(len(g)), "mean": float(g["score"].mean()), "sd": sd,
                          "swing_frac": {n: float(v / sd) for n, v in swings.items()}}
    sd_mean = float(np.mean([v["sd"] for v in per_ex.values()]))
    sd_pooled = float(df["score"].std(ddof=1))
    return {"per_exercise": per_ex,
            "sd_within_exercise_mean": sd_mean,
            "sd_pooled_all_rows": sd_pooled,
            "swing_over_sd_within_exercise": {n: float(v / sd_mean) for n, v in swings.items()},
            "swing_over_sd_pooled": {n: float(v / sd_pooled) for n, v in swings.items()},
            "swing_over_scale": {n: float(v / SCORE_MAX) for n, v in swings.items()}}


# ---------------------------------------------------------------------------------------
# (b) Ordinal consequence -- pairs the swing can reorder
# ---------------------------------------------------------------------------------------
def pair_separations(df: pd.DataFrame) -> pd.DataFrame:
    """All within-exercise subject pairs and their score separation.

    Pairs are formed WITHIN an exercise only: two subjects' scores are comparable when
    they performed the same exercise, and not otherwise. Each pair is labelled by stratum
    so the reordering rate can be read for the comparisons that carry clinical weight
    (patient vs control) as well as overall.
    """
    recs = []
    for k, g in df.groupby("exercise_k"):
        rows = list(g[["subject_name", "group", "score"]].itertuples(index=False))
        for a, b in itertools.combinations(rows, 2):
            recs.append({"exercise_k": int(k),
                         "subject_a": a.subject_name, "subject_b": b.subject_name,
                         "group_a": a.group, "group_b": b.group,
                         "separation": float(abs(a.score - b.score)),
                         "cross_group": a.group != b.group,
                         "patient_vs_control": a.group.split("/")[0] != b.group.split("/")[0]})
    return pd.DataFrame(recs)


def reorderable_fractions(pairs: pd.DataFrame, swings: dict) -> dict:
    """Fraction of pairs whose separation is smaller than the swing.

    A pair separated by less than the camera-induced swing is a pair whose ORDER the
    camera can invert: the instrument cannot be trusted to say which of the two scored
    higher. Strict `<` -- an exactly-equal separation is already tied, not reordered by
    the swing.
    """
    strata = {"all_pairs": pairs,
              "cross_group": pairs[pairs["cross_group"]],
              "patient_vs_control": pairs[pairs["patient_vs_control"]]}
    out = {}
    for name, sub in strata.items():
        sep = sub["separation"].to_numpy(dtype=float)
        out[name] = {"n_pairs": int(len(sep)),
                     "median_separation": float(np.median(sep)) if len(sep) else float("nan"),
                     "frac_below": {n: float((sep < v).mean()) for n, v in swings.items()}}
    return out


# ---------------------------------------------------------------------------------------
# (c) The instrument's own variation
# ---------------------------------------------------------------------------------------
def within_subject_spread(df: pd.DataFrame, swings: dict) -> dict:
    """Spread of each subject's own five exercise scores.

    This is a same-patient, same-session comparator: how much one person's score already
    moves across the tasks they were rated on. It is not a reliability coefficient and we
    do not present it as one -- different exercises are different measurements, so this is
    an upper bound on within-subject noise, which makes it a CONSERVATIVE yardstick for
    the swing.
    """
    g = df.groupby("subject_name")["score"]
    sd = g.std(ddof=1).dropna().to_numpy(dtype=float)
    rng = (g.max() - g.min()).dropna().to_numpy(dtype=float)
    return {"n_subjects": int(len(sd)),
            "sd_median": float(np.median(sd)), "sd_mean": float(sd.mean()),
            "range_median": float(np.median(rng)), "range_mean": float(rng.mean()),
            "frac_subjects_whose_range_is_below_swing":
                {n: float((rng < v).mean()) for n, v in swings.items()},
            "swing_over_median_within_subject_sd":
                {n: float(v / float(np.median(sd))) for n, v in swings.items()}}


def group_ranges(df: pd.DataFrame) -> dict:
    """Per-group score range per exercise -- the evidence that bands do not exist.

    Recorded in the artifact itself so the reason we did not answer the band question is
    checkable, not merely asserted in prose.
    """
    out = {}
    for (g, k), sub in df.groupby(["group", "exercise_k"]):
        lo, hi = sub["score"].min(), sub["score"].max()
        out.setdefault(g, {})[int(k)] = {
            "n": int(len(sub)), "min": float(lo), "max": float(hi),
            "argmin": str(sub.loc[sub["score"].idxmin(), "subject_name"]),
            "argmax": str(sub.loc[sub["score"].idxmax(), "subject_name"])}
    return out


def band_premise_check(gr: dict) -> dict:
    """Does any control subject score below any patient subject on the same exercise?

    One counterexample is enough to retire the band framing, so we surface the most
    extreme one per exercise rather than a summary statistic.
    """
    ctrl = [g for g in gr if g.startswith("CG/")]
    pat = [g for g in gr if g.startswith("GPP/")]
    out = {}
    for k in range(1, N_EXERCISES + 1):
        c_lo = min(((gr[g][k]["min"], gr[g][k]["argmin"], g) for g in ctrl if k in gr[g]),
                   default=None)
        p_hi = max(((gr[g][k]["max"], gr[g][k]["argmax"], g) for g in pat if k in gr[g]),
                   default=None)
        if c_lo is None or p_hi is None:
            continue
        out[k] = {"lowest_control": {"score": c_lo[0], "subject": c_lo[1], "group": c_lo[2]},
                  "highest_patient": {"score": p_hi[0], "subject": p_hi[1], "group": p_hi[2]},
                  "inverted": bool(p_hi[0] > c_lo[0]),
                  "inversion_margin": float(p_hi[0] - c_lo[0])}
    return out


# ---------------------------------------------------------------------------------------
def analyse(df: pd.DataFrame, swings: dict) -> dict:
    pairs = pair_separations(df)
    return {"effect_size": between_subject_sd(df, swings),
            "reordering": reorderable_fractions(pairs, swings),
            "within_subject": within_subject_spread(df, swings),
            "_pairs": pairs}


def write_report(res: dict, res_clean: dict, swings: dict, audit: dict, gr: dict,
                 premise: dict, odd: list[str], path: str) -> None:
    L = []
    add = L.append
    add("KIMORE SCORE RESOLUTION vs. VIEWPOINT-INDUCED SCORE SWING")
    add("=" * 72)
    add("")
    add("SCOPE. This measures score RESOLUTION, not group DISCRIMINABILITY. The claim is")
    add("that a camera relocation perturbs the clinical output by an amount comparable to")
    add("the score differences that output is meant to express. It is NOT a claim that the")
    add("score classifies pathology.")
    add("")
    add(f"Swing source: {FINAL_TABLES}, arm '{SWING_ARM}' (measured, not assumed)")
    add(f"  mean over sequences (worst angle) : {swings['mean']:.4f} of {SCORE_MAX:.0f}")
    add(f"  per-sequence p95                  : {swings['p95']:.4f}")
    add(f"  per-sequence global max           : {swings['max']:.4f}")
    add("")
    add("-" * 72)
    add("INPUT CENSUS")
    add(f"  rows {audit['n_rows']}  subjects {audit['n_subjects']}  "
        f"exercises {audit['n_exercises']}")
    add(f"  nominal full grid {audit['nominal_full_grid']}  "
        f"missing cells {audit['missing_cells']}")
    for g, n in sorted(audit["subjects_per_group"].items()):
        add(f"    {g:<16s} {n:3d} subjects")
    add(f"  non-third-fraction subjects ({len(odd)}): {', '.join(odd) if odd else 'none'}")
    add("")
    add("-" * 72)
    add("WHY NOT CLINICAL BANDS")
    add("  KIMORE's groups are recruitment categories, not score strata. Per exercise,")
    add("  the lowest-scoring control vs the highest-scoring patient:")
    for k, v in sorted(premise.items()):
        c, p = v["lowest_control"], v["highest_patient"]
        flag = "INVERTED" if v["inverted"] else "ordered"
        add(f"    Es{k}: {c['subject']:>8s} ({c['group']:<13s}) {c['score']:6.2f}   vs   "
            f"{p['subject']:>8s} ({p['group']:<13s}) {p['score']:6.2f}   [{flag}, "
            f"margin {v['inversion_margin']:+6.2f}]")
    add("  A swing cannot be said to cross a boundary that the data does not contain.")
    add("")

    def block(r, label):
        e, o, w = r["effect_size"], r["reordering"], r["within_subject"]
        add("-" * 72)
        add(f"RESULTS [{label}]")
        add("")
        add("(a) EFFECT SIZE -- swing as a fraction of between-subject SD")
        add(f"    between-subject SD, within-exercise mean : {e['sd_within_exercise_mean']:.3f}")
        add(f"    between-subject SD, pooled over rows     : {e['sd_pooled_all_rows']:.3f}")
        for n in ("mean", "p95", "max"):
            add(f"      swing {n:>4s} = {swings[n]:6.3f}  ->  "
                f"{e['swing_over_sd_within_exercise'][n]:5.2f} SD   "
                f"({100 * e['swing_over_scale'][n]:5.1f}% of the 0-50 scale)")
        add("")
        add("(b) ORDINAL COST -- fraction of subject pairs the swing can reorder")
        for stratum in ("all_pairs", "cross_group", "patient_vs_control"):
            s = o[stratum]
            add(f"    {stratum:<20s} n={s['n_pairs']:6d}  "
                f"median sep {s['median_separation']:6.2f}")
            for n in ("mean", "p95", "max"):
                add(f"        separated by < {swings[n]:6.3f} : "
                    f"{100 * s['frac_below'][n]:5.1f}% of pairs")
        add("")
        add("(c) THE INSTRUMENT'S OWN VARIATION -- each subject across their 5 exercises")
        add(f"    within-subject SD    median {w['sd_median']:.3f}  mean {w['sd_mean']:.3f}")
        add(f"    within-subject range median {w['range_median']:.3f}  mean {w['range_mean']:.3f}")
        for n in ("mean", "p95", "max"):
            add(f"      swing {n:>4s} = {swings[n]:6.3f}  ->  "
                f"{w['swing_over_median_within_subject_sd'][n]:5.2f}x the median "
                f"within-subject SD")
        add("")

    block(res, "all subjects")
    block(res_clean, f"excluding {len(odd)} non-third-fraction subjects")
    add("-" * 72)
    add("Both blocks are reported so that no headline number depends on the handful of")
    add("subjects whose released scores are not rater averages.")

    Path(path).write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nReport -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=PROCESSED_ROOT)
    ap.add_argument("--final_tables", default=FINAL_TABLES)
    ap.add_argument("--out_dir", default=os.path.join("outputs", "clinical_resolution"))
    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    swings = load_swings(args.final_tables)
    df = load_scores(args.root)
    audit = audit_inputs(df)
    odd = flag_odd_scores(df)
    gr = group_ranges(df)
    premise = band_premise_check(gr)

    res = analyse(df, swings)
    df_clean = df[~df["subject_name"].isin(odd)]
    res_clean = analyse(df_clean, swings)

    pairs = res.pop("_pairs")
    res_clean.pop("_pairs")
    pairs.to_csv(os.path.join(args.out_dir, "pair_separation.csv"), index=False)

    summary = {"swings": swings, "swing_source": args.final_tables, "swing_arm": SWING_ARM,
               "input_audit": audit, "non_third_fraction_subjects": odd,
               "group_ranges": gr, "band_premise_check": premise,
               "all_subjects": res, "excluding_odd_subjects": res_clean,
               "scope_note": ("Measures score resolution, not group discriminability. "
                              "Does not claim the score classifies pathology.")}
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    write_report(res, res_clean, swings, audit, gr, premise, odd,
                 os.path.join(args.out_dir, "resolution_report.txt"))
    print(f"Summary  -> {os.path.join(args.out_dir, 'summary.json')}")
    print(f"Pairs    -> {os.path.join(args.out_dir, 'pair_separation.csv')}")


if __name__ == "__main__":
    main()
