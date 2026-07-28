#!/usr/bin/env python
"""Regenerate the multi-corpus replication tables from banked artifacts.

The paper reports the node-failure result on REHAB24-6 as a single scalar ("AUROC lost over 8 frozen
nodes", +0.14 / +0.08). The full degradation curve was already computed and stored -- `evaluate_fold`
in train_rehab246.py records `nf_auroc` at every level in NF_LEVELS, per fold, per seed -- so the
curve costs no compute to publish, only the reading of it. A reviewer asking "show the curve, not the
endpoint" is asking for something already paid for.

Two spreads exist in these runs and they differ by a factor of ~5. Reporting the wrong one is the easy
mistake here:

  fold-level sd  -- spread of the 5 subject-disjoint folds within one seed. Large (~0.07 AUROC),
                    because a fold is 2 held-out subjects out of 10.
  seed-level sd  -- spread of the 3 per-seed means. Small (~0.01), because averaging 5 folds cancels
                    most of the subject variation.

The paper's "$\\pm$0.01" is the seed-level one. That is a legitimate statistic but it describes
seed reproducibility, NOT confidence in the AUROC given this subject pool, and a table that does not
say which it is invites the reader to assume the stronger reading. This script prints both and labels
them.

Usage:
    python src/second_corpus_tables.py                 # all corpora found
    python src/second_corpus_tables.py --corpus rehab246
    python src/second_corpus_tables.py --latex         # emit the supplement table body
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CORPORA = ["rehab246", "uiprmd"]

# Display names, in the order the paper lists them.
MODEL_LABEL = {
    "egru_chiral": "EGRU (ours)",
    "egru": "EGRU (ours, parity-even)",
    "pct": "PCT (baseline)",
}
MODEL_ORDER = ["egru_chiral", "egru", "pct"]


def _sd(xs):
    return float(statistics.stdev(xs)) if len(xs) > 1 else 0.0


def load_corpus(corpus: str) -> dict[str, list[dict]]:
    """Group every banked seed-summary for a corpus by model tag."""
    pat = str(REPO / "outputs" / corpus / f"{corpus}_*_s*.json")
    runs = defaultdict(list)
    for path in sorted(glob.glob(pat)):
        base = os.path.basename(path)[len(corpus) + 1: -len(".json")]
        model = base.rsplit("_s", 1)[0]
        with open(path) as fh:
            runs[model].append(json.load(fh))
    return runs


def summarise(seed_summaries: list[dict]) -> dict:
    """Collapse a model's seeds into the statistics the paper quotes."""
    per_seed_clean, per_seed_lost, per_seed_view_auroc = [], [], []
    fold_clean, drifts = [], []
    curve = defaultdict(list)          # k -> one entry per seed (each a mean over folds)

    for s in seed_summaries:
        rows = s["rows"]
        fold_clean += [r["clean_auroc"] for r in rows]
        per_seed_clean.append(float(np.mean([r["clean_auroc"] for r in rows])))
        per_seed_lost.append(float(np.mean([r["nf_lost_0to8"] for r in rows])))
        per_seed_view_auroc.append(float(np.mean([r["view_auroc_90deg"] for r in rows])))
        drifts.append(float(s["view_logit_drift_max"]))
        levels = sorted({int(k) for r in rows for k in r["nf_auroc"]})
        for k in levels:
            curve[k].append(float(np.mean([r["nf_auroc"][str(k)] for r in rows])))

    return {
        "n_seeds": len(seed_summaries),
        "n_folds": len(seed_summaries[0]["rows"]),
        "n_params": seed_summaries[0].get("n_params"),
        "clean_auroc": (float(np.mean(per_seed_clean)), _sd(per_seed_clean)),
        "clean_auroc_fold_sd": _sd(fold_clean),
        "view_auroc_90": (float(np.mean(per_seed_view_auroc)), _sd(per_seed_view_auroc)),
        "view_drift_max": max(drifts),
        "nf_lost_0to8": (float(np.mean(per_seed_lost)), _sd(per_seed_lost)),
        "curve": {k: (float(np.mean(v)), _sd(v)) for k, v in sorted(curve.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=None, choices=CORPORA)
    ap.add_argument("--latex", action="store_true", help="emit the supplement table body")
    ap.add_argument("--out", default=str(REPO / "outputs" / "second_corpus_tables.json"))
    args = ap.parse_args()

    wanted = [args.corpus] if args.corpus else CORPORA
    payload = {}

    for corpus in wanted:
        runs = load_corpus(corpus)
        if not runs:
            print(f"[{corpus}] no banked runs under outputs/{corpus}/ -- skipping")
            continue
        summ = {m: summarise(r) for m, r in runs.items()}
        payload[corpus] = summ

        first = next(iter(summ.values()))
        print(f"\n{'='*78}\n{corpus}  ({first['n_seeds']} seeds x {first['n_folds']} "
              f"subject-disjoint folds)\n{'='*78}")
        print("  %-26s %-18s %-9s %-11s %s"
              % ("model", "clean AUROC", "@90deg", "drift", "node-fail lost 0->8"))
        for m in [x for x in MODEL_ORDER if x in summ] + [x for x in summ if x not in MODEL_ORDER]:
            s = summ[m]
            ca, cs = s["clean_auroc"]
            va, _ = s["view_auroc_90"]
            la, ls = s["nf_lost_0to8"]
            print("  %-26s %.3f +/- %.3f     %.3f     %.1e     %+.3f +/- %.3f"
                  % (MODEL_LABEL.get(m, m), ca, cs, va, s["view_drift_max"], la, ls))

        print("\n  spread, read carefully:")
        for m in summ:
            s = summ[m]
            print("    %-26s seed-level sd %.3f   |   fold-level sd %.3f  (%.1fx larger)"
                  % (MODEL_LABEL.get(m, m), s["clean_auroc"][1], s["clean_auroc_fold_sd"],
                     s["clean_auroc_fold_sd"] / max(s["clean_auroc"][1], 1e-9)))

        levels = sorted(next(iter(summ.values()))["curve"].keys())
        print("\n  node-failure curve -- AUROC at k frozen joints (the paper reports only k=0 -> k=%d)"
              % max(levels))
        print("    %-26s" % "model" + "".join("   k=%-6d" % k for k in levels))
        for m in [x for x in MODEL_ORDER if x in summ] + [x for x in summ if x not in MODEL_ORDER]:
            c = summ[m]["curve"]
            print("    %-26s" % MODEL_LABEL.get(m, m)
                  + "".join("  %.3f    " % c[k][0] for k in levels))

    if args.latex and payload:
        print("\n" + "=" * 78 + "\nsupplement table body (node-failure curve)\n" + "=" * 78)
        for corpus, summ in payload.items():
            levels = sorted(next(iter(summ.values()))["curve"].keys())
            print("%% %s" % corpus)
            for m in [x for x in MODEL_ORDER if x in summ]:
                c = summ[m]["curve"]
                cells = " & ".join("$%.3f$" % c[k][0] for k in levels)
                print("%s & %s \\\\" % (MODEL_LABEL.get(m, m), cells))

    if payload:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print("\nwrote %s" % args.out)
        return 0
    print("no corpora found")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
