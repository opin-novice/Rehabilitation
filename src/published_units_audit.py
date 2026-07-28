#!/usr/bin/env python
"""Recover the units of the published KIMORE MAD numbers from the published tables themselves.

Why this exists
---------------
The related-work paragraph claims the KIMORE literature's MAD values (0.10-0.56) are not
protocol-comparable to ours (6.3-6.7). That claim is only meaningful if the two are on the same
scale, and the scale is *not* stated in those papers. Guessing wrong is not a small error: the
reference point-cloud transformer standardises its targets with a StandardScaler (sigma = 8.466 on
Kimore_ex1) and its own per-epoch print is in standardised units, so "0.185" could plausibly have
been either.

The published tables settle it without needing anyone's source code. Every one of these papers
reports MAD and MAPE for the same run, and MAPE (as implemented throughout this lineage) is

    MAPE = 100 * mean(|y - yhat| / |y|)

so the ratio recovers the magnitude of whatever array was fed to the metric:

    100 * MAD / MAPE  ~=  a harmonic-weighted mean of |y|.

If the metrics were computed on raw clinical scores that ratio must land near KIMORE's per-exercise
score means (35-41, on the 0-50 clinical Total Score). If they were computed on standardised
targets it must land near mean|z| ~= 0.8, and the printed MAPE would have to be tens of percent
rather than the fractions of a percent actually published.

Gates
-----
G1  recovered magnitude sits in the clinical-score band, not the standardised band.
G2  the standardised reading is excluded by >= 20x on the printed MAPE.
G3  the recovered magnitude tracks the true per-exercise means (rank agreement / relative error).

Sources for the hard-coded tables are cited inline; both are public.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# KIMORE clinical Total Score, 0-50, one value per recording.
KIMORE_Y = REPO / "KIMORE_processed" / "Exercise{ex}" / "Train_Y.csv"

# ---------------------------------------------------------------------------
# Published KIMORE tables, transcribed verbatim. Columns are Ex1..Ex5.
# ---------------------------------------------------------------------------

# Kuang et al., "Dual-Stream STGCN with Motion-Aware Grouping for Rehabilitation Action Quality
# Assessment", Sensors 26(1):287, 2026 -- Table 3 ("Performance comparison on KIMORE dataset").
# The table restates seven prior methods alongside their own, which is what makes this a test of
# the whole lineage's convention rather than of one paper.
KUANG_TABLE3 = {
    #                     MAD                                  MAPE
    "Du et al. 2021":    ([1.271, 2.199, 1.123, 0.880, 1.864], [3.228, 6.001, 3.421, 2.584, 5.620]),
    "Deb et al. 2022":   ([0.799, 0.774, 0.374, 0.347, 0.621], [1.926, 1.272, 0.728, 0.824, 1.591]),
    "Yao et al. 2023":   ([0.444, 0.303, 0.142, 0.121, 0.292], [1.105, 0.864, 0.437, 0.341, 0.808]),
    "Mour. et al. 2023": ([0.641, 0.753, 0.210, 0.206, 0.399], [1.623, 0.974, 0.613, 0.541, 1.217]),
    "Zhang et al. 2025": ([0.622, 0.491, 0.206, 0.204, 0.390], [1.508, 0.952, 0.536, 0.483, 1.113]),
    "Kuang et al. 2025": ([0.186, 0.235, 0.111, 0.053, 0.223], [0.431, 0.749, 0.271, 0.173, 0.692]),
    "Kuang et al. 2026": ([0.102, 0.119, 0.100, 0.163, 0.110], [0.187, 0.297, 0.207, 0.257, 0.585]),
}
# Sardari 2024 and Xiao 2024 rows are omitted: they report no MAPE, so the ratio is undefined.

# Kazi Rafat et al., "A Point Cloud Transformer for Remote Monitoring and Automated Assessment of
# Physical Rehabilitation Exercises" -- KIMORE per-exercise table, as published in the authors'
# released README (Transformer_Rehabilitation/README.md, "KIMORE - our method, per exercise").
PCT_TABLE = {
    "Point-cloud transformer": ([0.185, 0.560, 0.128, 0.256, 0.388],
                                [0.543, 1.891, 0.336, 0.766, 1.199]),
}


def kimore_score_stats() -> dict[int, dict[str, float]]:
    """Per-exercise clinical Total Score statistics, straight off disk."""
    out = {}
    for ex in range(1, 6):
        path = Path(str(KIMORE_Y).format(ex=ex))
        if not path.exists():
            raise FileNotFoundError(f"missing KIMORE labels: {path}")
        y = pd.read_csv(path, header=None).values.ravel().astype(float)
        z = (y - y.mean()) / y.std()
        out[ex] = {
            "n": int(y.size),
            "min": float(y.min()),
            "max": float(y.max()),
            "mean": float(y.mean()),
            "std": float(y.std()),
            "mean_abs_score": float(np.abs(y).mean()),
            "mean_abs_z": float(np.abs(z).mean()),
            # E[1/|z|] is what converts a standardised MAD into a MAPE. It is finite here only
            # because the sample is finite; that is the point -- even this optimistic estimate
            # lands two orders of magnitude above the published MAPE.
            "mean_inv_abs_z": float((1.0 / np.abs(z)).mean()),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "outputs" / "published_units_audit.json"))
    args = ap.parse_args()

    stats = kimore_score_stats()
    true_mean = np.array([stats[ex]["mean"] for ex in range(1, 6)])
    inv_abs_z = np.array([stats[ex]["mean_inv_abs_z"] for ex in range(1, 6)])

    tables = {"Kuang 2026 Table 3": KUANG_TABLE3, "PCT README": PCT_TABLE}

    print("KIMORE clinical Total Score (0-50), per exercise")
    print("  ex   n     min    max    mean     std   mean|z|")
    for ex in range(1, 6):
        s = stats[ex]
        print("  %2d  %3d  %6.2f %6.2f  %6.3f  %6.3f   %5.3f"
              % (ex, s["n"], s["min"], s["max"], s["mean"], s["std"], s["mean_abs_z"]))

    print("\nRecovered target magnitude  100*MAD/MAPE  (score-unit reading predicts the row above,")
    print("standardised reading predicts ~0.8)")
    header = "  %-24s" % "method" + "".join("   Ex%d" % e for e in range(1, 6)) + "    median"
    print(header)

    rows, all_recovered = [], []
    for tname, table in tables.items():
        print("  -- %s" % tname)
        for method, (mad, mape) in table.items():
            mad_a, mape_a = np.array(mad, float), np.array(mape, float)
            recovered = 100.0 * mad_a / mape_a
            all_recovered.append(recovered)
            # What MAPE the standardised reading would have had to print.
            mape_if_standardised = 100.0 * mad_a * inv_abs_z
            rows.append({
                "table": tname,
                "method": method,
                "mad": mad,
                "mape": mape,
                "recovered_target_magnitude": recovered.tolist(),
                "mape_if_targets_were_standardised": mape_if_standardised.tolist(),
                "standardised_reading_overshoot": (mape_if_standardised / mape_a).tolist(),
            })
            print("  %-24s" % method + "".join("%6.1f" % v for v in recovered)
                  + "    %6.1f" % np.median(recovered))
    print("  %-24s" % "TRUE score mean" + "".join("%6.1f" % v for v in true_mean))
    print("  %-24s" % "TRUE mean|z| (standardised)"[:24]
          + "".join("%6.2f" % stats[e]["mean_abs_z"] for e in range(1, 6)))

    recovered_all = np.concatenate(all_recovered)
    median_recovered = float(np.median(recovered_all))
    overshoot = np.concatenate([np.array(r["standardised_reading_overshoot"]) for r in rows])
    min_overshoot = float(overshoot.min())

    # Relative error of the recovered magnitude against the true per-exercise mean.
    rel_err = np.concatenate([
        np.abs(np.array(r["recovered_target_magnitude"]) - true_mean) / true_mean for r in rows])
    median_rel_err = float(np.median(rel_err))

    score_lo, score_hi = float(true_mean.min()) * 0.6, float(true_mean.max()) * 1.4
    g1 = score_lo <= median_recovered <= score_hi
    g2 = min_overshoot >= 20.0
    g3 = median_rel_err <= 0.25

    print("\nGates")
    print("  G1 recovered magnitude in clinical-score band [%.1f, %.1f]: median %.2f -> %s"
          % (score_lo, score_hi, median_recovered, "PASS" if g1 else "FAIL"))
    print("  G2 standardised reading excluded by >=20x on printed MAPE: worst case %.1fx -> %s"
          % (min_overshoot, "PASS" if g2 else "FAIL"))
    print("  G3 recovered magnitude tracks true per-exercise means (median rel. err <=25%%): "
          "%.1f%% -> %s" % (100 * median_rel_err, "PASS" if g3 else "FAIL"))

    verdict = "SCORE_UNITS" if (g1 and g2 and g3) else "INCONCLUSIVE"
    print("\nVERDICT: published KIMORE MAD is in %s" % verdict)
    ours_lo, ours_hi = 6.3, 6.7
    gap = {}
    if verdict == "SCORE_UNITS":
        print("  -> the published numbers are on the same 0-50 clinical scale as our "
              "%.1f-%.1f MAD; the comparison is direct, with no rescaling in between." % (ours_lo, ours_hi))
        for label, mad in [("Kuang et al. 2026", KUANG_TABLE3["Kuang et al. 2026"][0]),
                           ("Point-cloud transformer", PCT_TABLE["Point-cloud transformer"][0])]:
            lo, hi = min(mad), max(mad)
            f_lo, f_hi = ours_lo / hi, ours_hi / lo
            gap[label] = {"published_mad_band": [lo, hi], "factor_below_ours": [f_lo, f_hi]}
            print("     %-24s MAD %.3f-%.3f  ->  %.0fx-%.0fx below ours"
                  % (label, lo, hi, f_lo, f_hi))

    payload = {
        "kimore_score_stats": {str(k): v for k, v in stats.items()},
        "rows": rows,
        "median_recovered_magnitude": median_recovered,
        "min_standardised_overshoot": min_overshoot,
        "median_relative_error": median_rel_err,
        "gates": {"G1_in_score_band": g1, "G2_standardised_excluded": g2,
                  "G3_tracks_true_means": g3},
        "verdict": verdict,
        "gap_vs_ours": gap,
        "our_mad_band": [ours_lo, ours_hi],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\nwrote %s" % args.out)
    return 0 if verdict == "SCORE_UNITS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
