#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
check_paper_claims.py
=====================
Assert that the hand-typed tables in paper_wacv.tex agree with the banked artifacts -- in VALUE and
in MARKUP.

Why this exists. Figures in this repo are generated (`make_fig_*.py` read `final_tables.json`;
nothing is transcribed), but the LaTeX tables are typed by hand, and that is the one place a number
or a mark can drift from the artifact without anything failing. It did: `tab:nodefail` printed
EGRU $8.41$ at $k{=}4$ and PCT $8.73$ at $k{=}8$, both PAST the $8.31$ mean-predictor floor, with
neither underlined -- while its own caption states "Underline: past the floor" and `tab:pareto`
marked EGNN's $8.44$ as a failure at the same floor. Two omissions, both flattering us, in a paper
whose entire argument is that it concedes more than it has to.

A value check alone would NOT have caught that: 8.41 was the correct number, correctly transcribed.
The defect was in the markup, so the markup is what this file makes checkable.

Checks:
  1. VALUES     -- every numeric cell in tab:nodefail matches final_tables.json to printed precision.
  2. MARKUP     -- the floor rule is applied exactly: value > floor <=> \underline{} in tab:nodefail,
                   and <=> $\times$ (not \checkmark) in tab:pareto's stress columns.
  3. CROSSINGS  -- the k at which each model first crosses the floor, re-derived from the artifact,
                   matches what the prose and the tab:pareto caption claim.
  4. ABSTRACT   -- the abstract's node-failure bound is consistent with the re-derived crossing.

Usage:
    python scripts/check_paper_claims.py            # exit 1 on any violation
    python scripts/check_paper_claims.py --verbose
"""

import argparse
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(_ROOT, "paper", "wacv_submission", "paper_wacv.tex")
TABLES = os.path.join(_ROOT, "outputs", "cde_block2", "final_tables.json")

# Model key in final_tables' nodefail rows -> column index in tab:nodefail.
# tab:nodefail columns: k | EGRU | InvariantGRU | +oracle | PCT
NODEFAIL_COLS = {"egru": 0, "invgru": 1, "pct": 3}          # col 2 (+oracle) has no artifact key
NODEFAIL_KS = [1, 4, 8]                                     # the k values the table prints


def load(tex_path=TEX):
    with open(tex_path, encoding="utf-8") as fh:
        tex = fh.read()
    with open(TABLES, encoding="utf-8") as fh:
        art = json.load(fh)
    return tex, art


def table_body(tex, label):
    """The rows of the tabular carrying \\label{<label>}."""
    i = tex.index(f"\\label{{{label}}}")
    seg = tex[i:]
    # Slice AFTER the \midrule token, not from it: leaving the token attached to the first row made
    # that row fail the data-row regex below and be skipped in silence -- which is the same class of
    # defect this script exists to catch.
    start = seg.index("\\midrule") + len("\\midrule")
    body = seg[start:seg.index("\\bottomrule")]
    return [r.strip() for r in body.split("\\\\") if "&" in r]


def parse_cell(cell):
    """-> (value or None, underlined, has_times, has_check). Markup is what we are auditing."""
    underlined = "\\underline" in cell
    times = "$\\times$" in cell or cell.rstrip().endswith("\\times$")
    check = "\\checkmark" in cell
    m = re.search(r"(\d+\.\d+)", cell.replace("\\,", ""))
    return (float(m.group(1)) if m else None), underlined, times, check


def check_nodefail(tex, art, fails, verbose):
    floor = art["floor"]
    grid = {r["k"]: r for r in art["nodefail"]}
    rows = table_body(tex, "tab:nodefail")
    rows = [r for r in rows if re.match(r"^\$\d+\$", r.strip())]     # data rows only, skip 'MAD lost'
    # Fail loudly if the parse silently lost a row -- a skipped row is an unchecked claim.
    found = sorted(int(re.search(r"\d+", r).group()) for r in rows)
    if found != NODEFAIL_KS:
        fails.append(f"tab:nodefail: parsed k rows {found}, expected {NODEFAIL_KS} "
                     f"(a row was dropped or added -- the audit would be incomplete)")

    for row in rows:
        cells = [c.strip() for c in row.split("&")]
        k = int(re.search(r"\d+", cells[0]).group())
        if k not in grid:
            fails.append(f"tab:nodefail prints k={k}, absent from the artifact grid")
            continue
        for model, col in NODEFAIL_COLS.items():
            val, underlined, _, _ = parse_cell(cells[col + 1])
            truth = grid[k][model]
            if val is None:
                fails.append(f"tab:nodefail k={k} {model}: no number parsed")
                continue
            # 1. VALUE
            if abs(val - truth) > 0.005:
                fails.append(f"tab:nodefail k={k} {model}: prints {val}, artifact {truth:.3f}")
            # 2. MARKUP -- the check that would have caught 8.41 and 8.73
            past = truth > floor
            if past and not underlined:
                fails.append(f"tab:nodefail k={k} {model}: {truth:.3f} is PAST the floor "
                             f"({floor:.3f}) but is NOT underlined")
            if underlined and not past:
                fails.append(f"tab:nodefail k={k} {model}: {truth:.3f} clears the floor "
                             f"({floor:.3f}) but IS underlined")
            if verbose:
                print(f"  nodefail k={k:<2} {model:<7} {truth:6.3f}  "
                      f"{'past' if past else 'clears'}  underlined={underlined}")


def check_pareto_markup(tex, art, fails, verbose):
    """tab:pareto's stress columns must mark failure with $\\times$ and success with \\checkmark."""
    floor = art["floor"]
    for row in table_body(tex, "tab:pareto"):
        cells = [c.strip() for c in row.split("&")]
        name = re.sub(r"\\[a-zA-Z]+|[{}$]", "", cells[0]).strip()
        for cell in cells[3:5]:                                   # the 90-degree and k=2 columns
            val, _, times, check = parse_cell(cell)
            if val is None or not (times or check):
                continue
            past = val > floor
            if past and not times:
                fails.append(f"tab:pareto {name}: {val} is past the floor ({floor:.3f}) "
                             f"but is not marked as a failure")
            if (not past) and times:
                fails.append(f"tab:pareto {name}: {val} clears the floor ({floor:.3f}) "
                             f"but is marked as a failure")
            if verbose:
                print(f"  pareto   {name:<22} {val:6.2f}  "
                      f"{'past' if past else 'clears'}  times={times} check={check}")


def first_crossing(art, model):
    """Smallest k at which `model` is past the floor -- re-derived, never transcribed."""
    floor = art["floor"]
    for r in sorted(art["nodefail"], key=lambda r: r["k"]):
        if r[model] > floor:
            return r["k"]
    return None


def check_claims(tex, art, fails, verbose):
    """The prose must not out-run the grid."""
    flat = re.sub(r"\s+", " ", tex)
    eg, pct = first_crossing(art, "egru"), first_crossing(art, "pct")
    if verbose:
        print(f"  first crossing: egru k={eg}  pct k={pct}")

    # tab:pareto's caption names both crossings; they must be the re-derived ones.
    for model, k in (("ours", eg), ("point-cloud baseline", pct)):
        if f"{model} first crosses the floor at $k{{=}}{k}$" not in flat and \
                f"{model} at $k{{=}}{k}$" not in flat:
            fails.append(f"tab:pareto caption does not state the re-derived crossing for "
                         f"{model} (k={k})")

    # The abstract must bound the stress claim at the largest k that still CLEARS.
    if "clearing every stress" in flat:
        bound = {1: "one", 2: "two", 3: "three", 4: "four", 6: "six", 8: "eight"}.get(eg - 1)
        want = f"clearing every stress up to {bound} simultaneous node failures"
        if want not in flat:
            fails.append(f"abstract says 'clearing every stress' but EGRU first fails at k={eg}; "
                         f"expected the bounded form: '{want}'")


PAIRED = os.path.join(_ROOT, "outputs", "ntu_paired", "paired_camera.json")
PAIRED64 = os.path.join(_ROOT, "outputs", "ntu_paired", "paired_camera_fp64.json")


def check_paired_camera(tex, fails, verbose):
    """The NTU rotation (R1) and paired-camera (P1) numbers are PROSE, not a table -- which is
    exactly why they need asserting. Every one of them was typed by hand into the tex.

    Silently skipped if the artifacts are absent (outputs/ is not tracked), because a missing file
    means 'not re-run here', not 'claim disproved'. It is announced either way, so a skip can never
    be mistaken for a pass.
    """
    if not (os.path.isfile(PAIRED) and os.path.isfile(PAIRED64)):
        print("  paired-camera: artifacts absent, SKIPPED (run src/ntu_paired_camera.py)")
        return
    with open(PAIRED, encoding="utf-8") as fh:
        a32 = json.load(fh)
    with open(PAIRED64, encoding="utf-8") as fh:
        a64 = json.load(fh)
    # LaTeX writes thousands as 18{,}674; normalise so the artifact's 18,674 can match.
    flat = re.sub(r"\s+", " ", tex).replace("{,}", ",")

    n = a32["scope"]["n_triples"]
    eg, sg = a32["results"]["egru"], a32["results"]["stgcn_full"]
    eg64 = a64["results"]["egru"]

    # (claimed literal in tex, truth, tolerance, label)
    checks = [
        (f"{n // 1000},{n % 1000:03d}", None, None, "n_triples"),
        (f"{eg['rotation_control']['agreement'] * 100:.2f}", None, None, "R1 agreement (ours)"),
        (f"{sg['rotation_control']['agreement'] * 100:.2f}", None, None, "R1 agreement (ST-GCN)"),
        (f"{eg['agreement_all_three'] * 100:.2f}", None, None, "P1 all-three (ours)"),
        (f"{sg['agreement_all_three'] * 100:.2f}", None, None, "P1 all-three (ST-GCN)"),
        (f"{eg['top1_per_camera']['1']:.2f}" if "1" in eg["top1_per_camera"]
         else f"{eg['top1_per_camera'][1]:.2f}", None, None, "R1 Top-1 (ours)"),
    ]
    for literal, _t, _tol, label in checks:
        if literal not in flat:
            fails.append(f"paired-camera: {label} = {literal} not found in the prose")
        elif verbose:
            print(f"  paired   {label:<24} {literal}")

    # ST-GCN's rotation accuracy LOSS is DERIVED, so re-derive it at full precision -- subtracting
    # the two rounded values printed by the run gives 3.02 where the truth is 3.03, and that is the
    # error this check exists to catch. Scope the search to the sentence that reports it: a bare
    # substring search passes on PCT's unrelated 3.03 MAD elsewhere in the paper, which is a false
    # pass, not a check.
    top1 = sg["top1_per_camera"].get("1", sg["top1_per_camera"].get(1))
    drop = top1 - sg["rotation_control"]["top1_rotated"]
    anchor = flat.find(f"{sg['rotation_control']['agreement'] * 100:.2f}")
    window = flat[anchor:anchor + 320] if anchor >= 0 else ""
    if f"{drop:.2f}" not in window:
        fails.append(f"paired-camera: ST-GCN rotation accuracy loss = {drop:.2f} not found in the "
                     f"sentence reporting its agreement (window: {window[:90]!r})")
    elif verbose:
        print(f"  paired   {'R1 acc loss (ST-GCN)':<24} {drop:.2f}  (exact {drop:.4f})")

    # Ours must be roundoff-limited and theirs must NOT be: the contrast IS the claim, so assert the
    # PROPERTY, not just the printed digits. A regression that poisons fp64 (see the _w3j trap in
    # ntu_paired_camera.load_arm) shows up here as a collapsed ratio.
    if eg64["rotation_control"]["logit_linf_max"] > 1e-10:
        fails.append(f"paired-camera: our fp64 rotation drift "
                     f"{eg64['rotation_control']['logit_linf_max']:.3e} exceeds 1e-10 -- the fp64 "
                     f"certificate is no longer exact (check the _w3j load guard)")
    if a64["results"]["stgcn_full"]["rotation_control"]["logit_linf_max"] < 1.0:
        fails.append("paired-camera: the ST-GCN control's fp64 drift collapsed; it is supposed to "
                     "be structural and precision-independent")
    if verbose:
        print(f"  paired   {'fp64 drift (ours)':<24} "
              f"{eg64['rotation_control']['logit_linf_max']:.3e}")

    # P1 is the axis we LOSE, so the guard runs in that direction: if the paired test resolves
    # against us, the prose must concede it rather than restate parity. This is the one claim a
    # later editing pass would be tempted to soften.
    t = a32.get("p1_paired_test")
    if not t:
        fails.append("paired-camera: artifact has no p1_paired_test (re-run src/ntu_paired_camera.py)")
        return
    nd_lit = f"{t['n_discordant'] // 1000},{t['n_discordant'] % 1000:03d}"
    if nd_lit not in flat:
        fails.append(f"paired-camera: discordant count {nd_lit} not in the prose")
    if f"{t['p_value']:.3f}" not in flat:
        fails.append(f"paired-camera: p-value {t['p_value']:.3f} not in the prose")
    ours, theirs = t["agreement"]["egru"], t["agreement"]["stgcn_full"]
    if t["p_value"] < 0.05 and theirs > ours:
        if not any(w in flat for w in ("deficit", "worse", "behind", "loses to")):
            fails.append(f"paired-camera: P1 resolves AGAINST us (p={t['p_value']:.3f}, "
                         f"{ours * 100:.2f} vs {theirs * 100:.2f}) but the prose concedes nothing")
    if verbose:
        print(f"  paired   {'P1 paired test':<24} p={t['p_value']:.3f}  "
              f"discordant {nd_lit}  ({'against us' if theirs > ours else 'for us'})")


def main():
    ap = argparse.ArgumentParser(description="Audit paper tables against the banked artifacts.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--tex", default=TEX,
                    help="paper to audit (used to self-test this checker against a known-bad copy)")
    args = ap.parse_args()

    tex, art = load(args.tex)
    fails = []
    print(f"floor = {art['floor']:.4f} MAD   (source: outputs/cde_block2/final_tables.json)")
    check_nodefail(tex, art, fails, args.verbose)
    check_pareto_markup(tex, art, fails, args.verbose)
    check_claims(tex, art, fails, args.verbose)
    check_paired_camera(tex, fails, args.verbose)

    if fails:
        print(f"\nFAIL -- {len(fails)} violation(s):")
        for f in fails:
            print(f"  * {f}")
        return 1
    print("\nPASS -- table values, floor markup, and prose claims all agree with the artifact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
