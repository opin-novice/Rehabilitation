#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
variant_b1_score.py
===================
Variant B1 — "Screen-to-webcam consistency" scoring pass.

Decoupled from capture: loads all take NPZ artifacts from a directory, runs EGRU / PCT /
InvariantGRU on each take's full sequence and on growing time-prefixes (1.0s cadence)
for score-vs-time traces, computes per-clip per-model cross-viewpoint spread.

Writes outputs/variant_b1/consistency_scores.json. Never touches a camera; re-runnable
any time without re-recording.

Usage:
    python src/variant_b1_score.py                              # uses demo/takes/
    python src/variant_b1_score.py --take-dir demo/takes --out outputs/variant_b1
    python src/variant_b1_score.py --synthetic-only              # only score synthetic takes
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo"))

from mp_to_kinect import preprocess_window  # noqa: E402

# ---------------------------------------------------------------------------
# Loading take artifacts
# ---------------------------------------------------------------------------
def load_take(npz_path):
    """Load one take NPZ -> dict with metadata + raw skeleton."""
    data = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(data["_metadata"].item()))
    return {
        "x": data["x"].astype(np.float64),               # (T, 25, 3)
        "t": data["t"].astype(np.float64),               # (T,)
        "clip_id": meta["clip_id"],
        "viewpoint": meta["viewpoint"],
        "exercise": int(meta["exercise"]),
        "synthetic": bool(meta["synthetic"]),
        "n_frames": int(meta["n_frames"]),
        "path": npz_path,
    }


def discover_takes(take_dir, synthetic_only=False):
    """Discover all .npz files in take_dir and group by clip_id."""
    takes = []
    for fname in sorted(os.listdir(take_dir)):
        if not fname.endswith(".npz"):
            continue
        take = load_take(os.path.join(take_dir, fname))
        if synthetic_only and not take["synthetic"]:
            continue
        takes.append(take)
    return takes


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_prefix(engine, x_seq, t_seq, exercise, angle=0.0):
    """Score a prefix of the skeleton stream at a given time cadence.

    Returns dict with:
        "t_grid": list of prefix-end timestamps
        "egru": list of EGRU scores
        "pct": list of PCT scores
        "invgru": list of InvariantGRU scores (if loaded)
    """
    result = {"t_grid": [], "egru": [], "pct": []}
    if engine.invgru:
        result["invgru"] = []

    # Build growing prefixes at 1.0s cadence
    t_start = t_seq[0]
    t_end = t_seq[-1]
    t_cursor = 1.0
    while t_cursor <= t_end:
        mask = t_seq <= t_cursor
        if mask.sum() < 16:          # skip too-short prefixes
            t_cursor += 1.0
            continue
        x_pre = x_seq[mask]
        t_pre = t_seq[mask]
        sample = preprocess_window(x_pre, t_pre)
        pred = engine.predict(sample, exercise=exercise, angle=angle)
        result["t_grid"].append(t_cursor)
        result["egru"].append(pred["egru"])
        result["pct"].append(pred["pct"])
        if "invgru" in pred:
            result["invgru"].append(pred["invgru"])
        t_cursor += 1.0

    if not result["t_grid"]:
        # fallback: score full sequence
        sample = preprocess_window(x_seq, t_seq)
        pred = engine.predict(sample, exercise=exercise, angle=angle)
        result["t_grid"].append(t_end)
        result["egru"].append(pred["egru"])
        result["pct"].append(pred["pct"])
        if "invgru" in pred:
            result["invgru"].append(pred["invgru"])

    return result


# ---------------------------------------------------------------------------
# Spread metrics
# ---------------------------------------------------------------------------
def compute_spread(scores):
    """max - min and MAD across viewpoints."""
    arr = np.array(scores)
    return {
        "max_minus_min": float(arr.max() - arr.min()),
        "mad": float(np.mean(np.abs(arr - arr.mean()))),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Variant B1 scoring: load takes, run models, compute consistency.")
    ap.add_argument("--take-dir", type=str,
                    default=os.path.join(os.path.dirname(__file__), "..", "demo", "takes"),
                    help="Directory with take NPZ artifacts")
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__), "..",
                                         "outputs", "variant_b1"),
                    help="Output directory for consistency_scores.json")
    ap.add_argument("--ckpt-dir", type=str, default="outputs/cde_block2",
                    help="Checkpoint directory")
    ap.add_argument("--synthetic-only", action="store_true",
                    help="Only score takes marked as synthetic")
    ap.add_argument("--no-invgru", action="store_true",
                    help="Skip InvariantGRU loading (faster if checkpoints missing)")
    args = ap.parse_args()

    # --- Discover takes ---
    takes = discover_takes(args.take_dir, synthetic_only=args.synthetic_only)
    if not takes:
        raise SystemExit(f"[score] no take artifacts found in {args.take_dir}")
    print(f"[score] found {len(takes)} takes in {args.take_dir}")

    # --- Load engine ---
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))
    from engine import DemoEngine

    print("[score] loading models (EGRU + PCT" +
          ("" if args.no_invgru else " + InvariantGRU") + ")...")
    eng = DemoEngine(ckpt_dir=args.ckpt_dir, load_invgru=not args.no_invgru)

    # --- Crop to common duration per clip ---
    # Ensures all viewpoints of the same clip cover the identical time window,
    # removing the confound of different-duration takes scoring different motion.
    clip_takes = defaultdict(list)
    for take in takes:
        clip_takes[take["clip_id"]].append(take)

    for cid, group in clip_takes.items():
        min_dur = min(t["t"][-1] for t in group)
        print(f"[score] clip {cid}: cropping all {len(group)} takes to {min_dur:.2f}s "
              f"(min across viewpoints)")
        for take in group:
            mask = take["t"] <= min_dur
            take["x"] = take["x"][mask]
            take["t"] = take["t"][mask]
            take["n_frames"] = len(take["t"])

    # --- Score every take ---
    per_take = []
    clip_groups = defaultdict(list)
    for take in takes:
        cid = take["clip_id"]
        print(f"[score]  {os.path.basename(take['path'])}: "
              f"{take['n_frames']} frames, viewpoint={take['viewpoint']}")
        sample = preprocess_window(take["x"], take["t"])
        pred = eng.predict(sample, exercise=take["exercise"], angle=0.0)
        trace = score_prefix(eng, take["x"], take["t"], exercise=take["exercise"])

        rec = {
            "clip_id": cid,
            "viewpoint": take["viewpoint"],
            "exercise": take["exercise"],
            "n_frames": take["n_frames"],
            "synthetic": take["synthetic"],
            "score_egru": pred["egru"],
            "score_pct": pred["pct"],
            "trace": trace,
        }
        if "invgru" in pred:
            rec["score_invgru"] = pred["invgru"]
        per_take.append(rec)
        clip_groups[cid].append(rec)

    # --- Compute per-clip cross-viewpoint spread ---
    models = ["egru", "pct"]
    if not args.no_invgru:
        models.append("invgru")

    clip_results = {}
    for cid, recs in clip_groups.items():
        ex = recs[0]["exercise"]
        n_vp = len(recs)
        vp_names = [r["viewpoint"] for r in recs]
        spread = {}
        for m in models:
            vals = [r.get(f"score_{m}", 0.0) for r in recs]
            spread[m] = compute_spread(vals)
            spread[m]["values"] = vals
            spread[m]["viewpoints"] = vp_names
        clip_results[cid] = {
            "exercise": ex,
            "n_viewpoints": n_vp,
            "viewpoints": vp_names,
            "spread": spread,
        }
        print(f"[score] clip {cid} (ex {ex}): "
              f"EGRU spread {spread['egru']['max_minus_min']:.4f}  "
              f"PCT spread {spread['pct']['max_minus_min']:.4f}" +
              (f"  InvGRU spread {spread['invgru']['max_minus_min']:.4f}"
               if "invgru" in spread else ""))

    # --- Build output ---
    output = {
        "experiment": "variant_b1",
        "description": "Screen-to-webcam consistency (no ground truth)",
        "per_take": per_take,
        "per_clip": clip_results,
        "models": models,
    }

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "consistency_scores.json")
    json.dump(output, open(out_path, "w"), indent=2)
    print(f"[score] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
