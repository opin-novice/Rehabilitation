"""Recompute the CORAL and few-shot-calibration rows on FK-corrected UI-PRMD.

Both consume UI-PRMD sequences through frozen scratch-TCN features, so both inherit the
bone-offset geometry bug and both need re-running (they are the last two UI-PRMD numbers
in manuscript.tex that the main column sweep does not cover).

Runs each geometry so the delta is auditable, and reports REHAB246 as an untouched control.

Usage:
    python src/rerun_coral_fewshot_fk.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/
import reviewer_analyses as RA  # noqa: E402
from selfsup.data import load_corpus_with_labels  # noqa: E402

GEOM = {
    "raw (paper)": "outputs/validity_uiprmd_raw",
    "fk": "outputs/validity_uiprmd",
}
COND_DIR = os.path.join(RA.RESULTS_DIR, "A_scratch")
OUT = "outputs/novelty/coral_fewshot_fk.json"


def main() -> int:
    X_kimore, y_kimore, _ = load_corpus_with_labels("KIMORE")
    X_rehab, y_rehab, _ = load_corpus_with_labels("REHAB246")
    rehab_man = pd.read_csv("outputs/validity/rehab246_manifest.csv")

    results = {}

    # REHAB246 control -- unaffected by UI-PRMD geometry, should match the paper.
    print("=== REHAB246 (control) ===")
    c = RA.compute_coral_baseline(COND_DIR, X_kimore, y_kimore, X_rehab,
                                  np.asarray(y_rehab), rehab_man)
    f = RA.compute_fewshot(COND_DIR, X_kimore, y_kimore, X_rehab, np.asarray(y_rehab))
    print(f"  CORAL   {c}")
    print(f"  fewshot {f}")
    results["REHAB246"] = {"coral": c, "fewshot": f}

    for label, cache in GEOM.items():
        if not os.path.isdir(cache):
            print(f"[skip] {label}: {cache} missing")
            continue
        X = np.load(os.path.join(cache, "uiprmd_sequences.npy")).astype(np.float32)
        man = pd.read_csv(os.path.join(cache, "uiprmd_manifest.csv"))
        y = man["correct_label"].values
        print(f"\n=== UI-PRMD {label} ===")
        c = RA.compute_coral_baseline(COND_DIR, X_kimore, y_kimore, X, y, man)
        f = RA.compute_fewshot(COND_DIR, X_kimore, y_kimore, X, y)
        print(f"  CORAL   {c}")
        print(f"  fewshot {f}")
        results[f"UIPRMD::{label}"] = {"coral": c, "fewshot": f}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
