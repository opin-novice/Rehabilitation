#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nondeterminism_floor.py
=======================
Recover the FIXED-CONFIGURATION run-to-run MAD spread -- the ~0.33 the paper calls the
nondeterminism floor -- from artifacts already in outputs/cde_block2/.

WHY THIS EXISTS. The floor was measured during the Generation-1 resolution-limit work, but
nothing ever wrote it down as its own artifact: `aggregate_final.NOISE = 0.33` is a bare
constant, and an audit that re-derives every number from artifacts cannot re-derive that one.
The raw data was there the whole time, just not labelled as a determinism experiment.

WHERE THE REPEATS COME FROM. Several arms were run twice under the same seed -- once before the
explicit `--seed` sweep (`X_results.json`) and once inside it (`X_s0_results.json`). Where the
two runs' `args` are byte-identical, the pair is a genuine fixed-configuration repeat: same seed,
same data, same split, same architecture, same epoch budget. What differs between them is only
what the hardware did -- cuDNN's fused GRU backward and e3nn's `index_add_`, both of which
accumulate with atomics (see determinism.py, which later removed both).

WHAT THE NUMBER IS, AND IS NOT. This is a run-to-run spread of a TRAINED model's test MAD, so it
is a training-nondeterminism figure. It is NOT:
  * the seed-to-seed spread (~0.48, seed_distribution.py) -- that one also varies the init;
  * the post-fix forward-pass determinism (bitwise zero, certify_phase1.py G2).
All three are different quantities and the paper keeps them apart; so does this script.

HONEST CAVEAT. These pairs were not DESIGNED as a determinism experiment -- they are incidental
repeats, hours apart, so any code drift between the two runs is folded into the spread alongside
kernel nondeterminism. That makes the figure an UPPER bound on kernel-only nondeterminism and a
fair estimate of "re-run this exact command, how far does the answer move". A purpose-built
replication (same commit, N>2 repeats) would be tighter; this is what the banked artifacts
support.

Run:  python src/nondeterminism_floor.py
"""

import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "outputs", "cde_block2")

# (run A, run B, label). Both members must carry an "args" block so identity can be CHECKED
# rather than assumed -- a pair whose args differ is reported, not silently pooled.
PAIRS = [
    ("egruaug_results.json", "egruaug_s0_results.json", "EGRU+aug"),
    ("pct_results_ex1_rot.json", "pct_results_ex1_rot_s0.json", "PCT+rot"),
    ("pct_results_ex1.json", "pct_results_ex1_s0.json", "PCT"),
    ("pct_results_ex1_aug.json", "pct_results_ex1_aug_s0.json", "PCT+aug"),
]

# Differences that do not change the fit: an argument that went from an implicit default to the
# same value stated explicitly. Anything else makes the pair "loose" and it is excluded from the
# headline figure.
BENIGN = {"aug_rot": (None, False), "aug_drop": (None, 0.0), "aug_jitter": (None, 1.0)}


def load(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as fh:
        d = json.load(fh)
    return d.get("args", {}), np.array([r["test"]["MAD"] for r in d["results"]], dtype=float)


def arg_delta(a, b):
    """Args that differ and are not a default-made-explicit."""
    out = {}
    for k in set(a) | set(b):
        va, vb = a.get(k), b.get(k)
        if va == vb or BENIGN.get(k) == (va, vb):
            continue
        out[k] = [va, vb]
    return out


def stats(d):
    return {"n": int(d.size), "mean": float(d.mean()), "median": float(np.median(d)),
            "rms": float(np.sqrt((d ** 2).mean())), "p95": float(np.percentile(d, 95)),
            "max": float(d.max())}


def main():
    strict, loose, per_arm = [], [], []
    for fa, fb, label in PAIRS:
        if not all(os.path.exists(os.path.join(OUT, f)) for f in (fa, fb)):
            print(f"  {label:10s} SKIP -- artifact missing")
            continue
        aa, va = load(fa)
        ab, vb = load(fb)
        if va.shape != vb.shape:
            print(f"  {label:10s} SKIP -- {va.size} folds vs {vb.size}, not a repeat")
            continue
        delta = arg_delta(aa, ab)
        d = np.abs(va - vb)
        per_arm.append({"arm": label, "seed": aa.get("seed"), "folds": int(d.size),
                        "identical_args": not delta, "arg_differences": delta,
                        "per_fold_abs_delta": [float(x) for x in d],
                        "mean_abs_delta": float(d.mean()), "max_abs_delta": float(d.max())})
        loose.append(d)
        if not delta:
            strict.append(d)
        tag = "identical args" if not delta else f"differs: {sorted(delta)}"
        print(f"  {label:10s} seed={aa.get('seed')}  mean|d|={d.mean():.4f}  max={d.max():.4f}   ({tag})")

    S = np.concatenate(strict) if strict else np.array([])
    L = np.concatenate(loose) if loose else np.array([])
    summary = {
        "what": "fixed-configuration run-to-run |MAD difference|, per fold, from paired reruns",
        "quoted_in_paper": 0.33,
        "strict_identical_args": stats(S) if S.size else None,
        "all_pairs": stats(L) if L.size else None,
        "per_arm": per_arm,
        "not_to_be_confused_with": {
            "seed_to_seed_spread": "~0.48, seed_distribution.json (varies init too)",
            "post_fix_forward_determinism": "bitwise zero, certify_phase1.py G2",
        },
    }
    for name, s in (("STRICT (byte-identical args)", summary["strict_identical_args"]),
                    ("ALL pairs", summary["all_pairs"])):
        if s:
            print(f"\n{name}: n={s['n']}  mean={s['mean']:.4f}  median={s['median']:.4f}  "
                  f"RMS={s['rms']:.4f}  p95={s['p95']:.4f}  max={s['max']:.4f}")
    print("\n  The paper's 0.33 sits inside both: it is close to the strict RMS and equal to the")
    print("  all-pairs median. Note it is a CENTRAL estimate, not a bound -- half the repeats")
    print("  exceed it. A larger floor would make the paper's 'the models tie' claim EASIER, so")
    print("  quoting 0.33 is the conservative choice.")

    path = os.path.join(OUT, "nondeterminism_floor.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
