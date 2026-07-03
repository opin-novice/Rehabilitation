"""N5 - Periodicity-matched temporal inductive bias.

Turns the k05-circumduction anecdote (clinical_narrative.md) into a quantitative
law: the TCN-minus-attention performance margin should track per-exercise signal
PERIODICITY.

For each KIMORE sample we compute a periodicity score = spectral concentration of
the per-frame deviation-energy signal (fraction of non-DC spectral power at the
dominant frequency; bounded [0, 1]). We average per exercise, then correlate it
across the 5 exercises with the per-exercise Delta-rho = rho(TCN) - mean(rho over
attention models). Falsifiable claim: corr(periodicity, Delta-rho) > 0.

Usage:
  python src/novelty/periodicity.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/

import numpy as np
from scipy.stats import spearmanr

from novelty import config, io_utils
from constants import SEQ_LEN, NUM_JOINTS, NUM_CHANNELS, JOINT_STARTS


def _periodicity_score(seq: np.ndarray) -> float:
    """Spectral concentration of a [T, J, C] sequence's deviation energy.

    1. Center each coordinate over time (removes static posture / DC offset).
    2. Energy signal s[t] = sum over joints,channels of centered coord^2.
    3. Score = max non-DC power-spectral bin / total non-DC power  in [0, 1].
       Pure single-frequency motion -> ~1; broadband / aperiodic -> ~0.
    """
    seqc = seq - seq.mean(axis=0, keepdims=True)
    s = (seqc ** 2).sum(axis=(1, 2))           # [T]
    s = s - s.mean()
    if np.allclose(s, 0):
        return 0.0
    power = np.abs(np.fft.rfft(s)) ** 2
    power[0] = 0.0                              # drop residual DC
    total = power.sum()
    if total <= 0:
        return 0.0
    return float(power.max() / total)


def _per_exercise_periodicity(data: dict) -> dict[int, float]:
    X_raw, eid = data["X_raw"], data["exercise_ids"]
    n = len(eid)
    xyz_cols: list[int] = []
    for st in JOINT_STARTS:
        xyz_cols.extend([st, st + 1, st + 2])
    X_3d = X_raw.reshape(n, SEQ_LEN, 100)
    per_sample = np.zeros(n)
    for i in range(n):
        seq = X_3d[i][:, xyz_cols].reshape(SEQ_LEN, NUM_JOINTS, NUM_CHANNELS)
        per_sample[i] = _periodicity_score(seq)
    return {e: float(per_sample[eid == e].mean()) for e in range(config.N_EXERCISES)}


def _per_exercise_rho(df) -> dict[int, float]:
    out: dict[int, float] = {}
    for e in range(config.N_EXERCISES):
        sub = df[df["exercise_id"] == e]
        if len(sub) >= 5:
            rho, _ = spearmanr(sub["y_true"].values, sub["y_pred"].values)
            out[e] = float(rho)
    return out


def run() -> dict:
    data = io_utils.load_pooled_kimore()
    oof = io_utils.load_all_oof()
    if data is None or not oof:
        return {"error": "pooled KIMORE or OOF predictions unavailable"}

    periodicity = _per_exercise_periodicity(data)

    if config.TCN_NAME not in oof:
        return {"error": f"{config.TCN_NAME} OOF missing"}
    tcn_rho = _per_exercise_rho(oof[config.TCN_NAME])

    attn_present = [m for m in config.ATTENTION_MODELS if m in oof]
    attn_rho_by_ex: dict[int, list[float]] = {e: [] for e in range(config.N_EXERCISES)}
    for m in attn_present:
        for e, r in _per_exercise_rho(oof[m]).items():
            attn_rho_by_ex[e].append(r)

    rows = []
    xs, ys = [], []
    for e in range(config.N_EXERCISES):
        if e not in tcn_rho or not attn_rho_by_ex[e]:
            continue
        attn_mean = float(np.mean(attn_rho_by_ex[e]))
        delta = tcn_rho[e] - attn_mean
        rows.append({
            "exercise_id": e,
            "exercise": config.EXERCISE_NAMES[e],
            "periodicity": round(periodicity[e], 4),
            "rho_tcn": round(tcn_rho[e], 4),
            "rho_attention_mean": round(attn_mean, 4),
            "delta_tcn_minus_attention": round(delta, 4),
        })
        xs.append(periodicity[e])
        ys.append(delta)

    if len(xs) >= 3:
        rho_corr, p_corr = spearmanr(xs, ys)
    else:
        rho_corr, p_corr = float("nan"), float("nan")

    results = {
        "attention_models_used": attn_present,
        "per_exercise": rows,
        "periodicity_vs_margin_spearman": round(float(rho_corr), 4),
        "periodicity_vs_margin_pvalue": round(float(p_corr), 4),
        "n_exercises": len(xs),
        "claim": "corr(periodicity, rho_TCN - rho_attention) > 0",
    }

    out = config.ensure_out()
    path = os.path.join(out, "periodicity.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("PERIODICITY vs TCN-minus-attention margin")
    for r in rows:
        print(f"  ex{r['exercise_id']} {r['exercise']:22s} "
              f"period={r['periodicity']:.3f}  delta={r['delta_tcn_minus_attention']:+.3f}")
    print(f"  Spearman(periodicity, delta) = {rho_corr:.3f} (p={p_corr:.3f})")
    print(f"  -> {path}")
    return results


def main() -> None:
    argparse.ArgumentParser(description="N5 periodicity law").parse_args()
    run()


if __name__ == "__main__":
    main()
