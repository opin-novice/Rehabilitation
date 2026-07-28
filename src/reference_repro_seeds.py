#!/usr/bin/env python
"""Aggregate the reference paper's own protocol across seeds.

`scripts/run_reference_protocol{,_seeds}.sh` run the reference implementation unmodified, capturing
the full per-epoch curve. This parses those logs and reports, per seed and pooled:

    test-selected   min over all 2000 test evaluations -- the statistic their checkpoint-selection
                    rule bakes in (engine/trainer.py:59-65 selects on minimum test MAD; there is no
                    validation split anywhere in their pipeline)
    late-median     median over the last 200 epochs -- the honest statistic
    final           final-epoch test MAD
    selection       late-median - test-selected, i.e. what the selection rule is worth

Their per-epoch print (`engine/trainer.py:73 evaluate_mad`) does NOT inverse-transform, so the logged
"val MAD" is in standardised units. Their `eval.py` does, via `engine/evaluator.py`. We convert with
the same StandardScaler sigma their Data_Proc fits on the full label vector, recomputed from the CSV
rather than hard-coded, and cross-check the converted test-selected value against their own eval.log.

A single seed shows the selection effect exists. Only the spread across seeds says how much of any
particular number is seed luck.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
REF = REPO / "Transformer_Rehabilitation"

EPOCH_RE = re.compile(r"\[epoch (\d+)\] train MAD=([0-9.]+)\s+val MAD=([0-9.]+)")
EVAL_RE = re.compile(r"MAD\s*:\s*([0-9.]+)")


def scaler_sigma(ex: str = "Kimore_ex1") -> float:
    """Reproduce the sigma of their target StandardScaler (Data_Proc/data_processing.py:47,71).

    sc2 is fit on the whole label vector before any split, so sigma is a dataset constant and does
    not vary with --seed. sklearn uses the population std (ddof=0).
    """
    y = pd.read_csv(REF / "KIMORE" / ex / "Train_Y.csv", header=None).values.ravel().astype(float)
    return float(y.std())


def parse_run(run_dir: Path, sigma: float, tail: int = 200) -> dict | None:
    log = run_dir / "train.log"
    if not log.exists():
        return None
    epochs, train_mad, val_mad = [], [], []
    done = False
    with open(log, "r", errors="ignore") as fh:
        for line in fh:
            m = EPOCH_RE.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_mad.append(float(m.group(2)))
                val_mad.append(float(m.group(3)))
            elif "Training done in" in line:
                done = True
    if not epochs:
        return None

    # A run is complete only when their trainer printed its terminal line. Contiguity of the epoch
    # indices is not enough: a run still in progress is contiguous too, and silently folding one in
    # would report a truncated curve's minimum as if it were the 2000-epoch minimum.
    v = np.array(val_mad)
    rec = {
        "seed": int(run_dir.name.split("_s")[-1]),
        "epochs_parsed": len(epochs),
        "complete": done and len(epochs) == max(epochs) + 1,
        "sigma": sigma,
        # standardised units, as logged
        "selected_std": float(v.min()),
        "selected_epoch": int(epochs[int(v.argmin())]),
        "late_median_std": float(np.median(v[-tail:])),
        "final_std": float(v[-1]),
        "final_train_std": float(train_mad[-1]),
    }
    # score units, the axis their eval.py and the published table both use
    for k in ("selected", "late_median", "final", "final_train"):
        rec[k] = rec[f"{k}_std"] * sigma

    ev = run_dir / "eval.log"
    if ev.exists():
        m = EVAL_RE.search(ev.read_text(errors="ignore"))
        if m:
            rec["eval_py_mad"] = float(m.group(1))
            rec["parse_vs_evalpy_abs_err"] = abs(rec["eval_py_mad"] - rec["selected"])

    rec["selection_effect"] = rec["late_median"] - rec["selected"]
    rec["selection_effect_pct"] = 100.0 * rec["selection_effect"] / rec["late_median"]
    rec["train_test_gap"] = rec["final"] / rec["final_train"]
    return rec


def mean_sd(xs: list[float]) -> tuple[float, float]:
    return (float(np.mean(xs)), float(statistics.stdev(xs)) if len(xs) > 1 else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ex", default="Kimore_ex1")
    ap.add_argument("--tail", type=int, default=200, help="epochs in the late-median window")
    ap.add_argument("--published", type=float, default=0.185,
                    help="their published MAD for this exercise, in score units (see E10)")
    ap.add_argument("--out", default=str(REPO / "outputs" / "reference_repro_seeds.json"))
    args = ap.parse_args()

    sigma = scaler_sigma(args.ex)
    runs = sorted(REF.glob("repro_s*"), key=lambda p: int(p.name.split("_s")[-1]))
    recs = [r for r in (parse_run(d, sigma, args.tail) for d in runs) if r]
    if not recs:
        print("no parsable runs under %s" % REF)
        return 1

    print("Reference protocol, %s, their code unmodified. StandardScaler sigma = %.4f" % (args.ex, sigma))
    print("(score units; their logged per-epoch value is standardised and multiplied by sigma)\n")
    print("  seed  epochs  test-selected  late-median  final  train(final)  selection  eval.py")
    for r in recs:
        flag = "" if r["complete"] else "  [INCOMPLETE]"
        ev = ("%9.4f" % r["eval_py_mad"]) if "eval_py_mad" in r else "        -"
        print("  %4d  %6d  %13.4f  %11.4f  %5.2f  %12.3f  %9.3f%s%s"
              % (r["seed"], r["epochs_parsed"], r["selected"], r["late_median"], r["final"],
                 r["final_train"], r["selection_effect"], ev, flag))

    complete = [r for r in recs if r["complete"]]
    if len(complete) < len(recs):
        print("\n  (only complete runs enter the pooled statistics)")

    summary = {}
    if complete:
        print("\nPooled over %d complete seed(s)%s" % (len(complete),
              " -- SINGLE SEED, spread not yet measurable" if len(complete) == 1 else ""))
        for key, label in [("selected", "test-selected (their rule)"),
                           ("late_median", "late-median (honest)"),
                           ("final", "final epoch"),
                           ("selection_effect", "selection effect"),
                           ("selection_effect_pct", "selection effect (%)")]:
            m, s = mean_sd([r[key] for r in complete])
            summary[key] = {"mean": m, "sd": s, "values": [r[key] for r in complete]}
            print("  %-28s %7.3f +/- %.3f" % (label, m, s))

        sel = [r["selected"] for r in complete]
        print("\n  test-selected range across seeds: %.3f - %.3f (spread %.3f)"
              % (min(sel), max(sel), max(sel) - min(sel)))
        m, s = mean_sd(sel)
        print("  vs their published %.3f: %.0fx above (mean), %.0fx at the luckiest seed"
              % (args.published, m / args.published, min(sel) / args.published))
        summary["published"] = args.published
        summary["factor_above_published_mean"] = m / args.published

        errs = [r["parse_vs_evalpy_abs_err"] for r in complete if "parse_vs_evalpy_abs_err" in r]
        if errs:
            print("  curve-parse vs their eval.py, max abs err: %.4f MAD (pipeline cross-check)"
                  % max(errs))
            summary["max_parse_vs_evalpy_abs_err"] = max(errs)

    payload = {"exercise": args.ex, "sigma": sigma, "tail": args.tail,
               "runs": recs, "summary": summary}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
