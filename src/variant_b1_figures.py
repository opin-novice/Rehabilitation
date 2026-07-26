#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
variant_b1_figures.py
=====================
Variant B1 — "Screen-to-webcam consistency" figures.

Reads outputs/variant_b1/consistency_scores.json; writes:
  - outputs/variant_b1/results_table.csv
  - outputs/variant_b1/figures/{clip_id}_trace.png + .pdf
    (score-vs-time overlay: EGRU / PCT / InvariantGRU × viewpoints)

Figure styling follows make_fig_guarantee.py: matplotlib Agg backend, serif font size 8.

Usage:
    python src/variant_b1_figures.py
    python src/variant_b1_figures.py --input outputs/variant_b1/consistency_scores.json
"""

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Styling (matches make_fig_guarantee.py)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

EGRU_COLOR = "#1a7f6b"
PCT_COLOR = "#c24a3f"
INVGRU_COLOR = "#6b5b8a"          # neutral accent (not teal, not red)

VIEWPOINT_LS = {
    "straight_on": "-",
    "plus30": "--",
    "minus30": ":",
    "tilt_up": "-.",
    "tilt_down": (0, (3, 1, 1, 1)),
    "closer": (0, (5, 2)),
    "farther": (0, (3, 2, 1, 2)),
}

VIEWPOINT_ALPHA = {
    "straight_on": 1.0,
    "plus30": 0.7,
    "minus30": 0.7,
    "tilt_up": 0.5,
    "tilt_down": 0.5,
    "closer": 0.6,
    "farther": 0.6,
}

MODEL_STYLE = {
    "egru": {"color": EGRU_COLOR, "label": "EGRU (ours)"},
    "pct": {"color": PCT_COLOR, "label": "PCT (baseline)"},
    "invgru": {"color": INVGRU_COLOR, "label": "InvariantGRU"},
}


def make_trace_figure(clip_id, per_take, models, out_dir):
    """Score-vs-time overlay for one clip: EGRU/PCT/InvariantGRU traces across viewpoints."""
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 2.8))

    for rec in per_take:
        vp = rec["viewpoint"]
        trace = rec.get("trace", {})
        t_grid = trace.get("t_grid", [])
        if not t_grid:
            continue
        ls = VIEWPOINT_LS.get(vp, "-")
        alpha = VIEWPOINT_ALPHA.get(vp, 0.5)

        for m in models:
            scores = trace.get(m, [])
            if not scores:
                continue
            color = MODEL_STYLE[m]["color"]
            label = None  # label only once per model via proxy artists
            ax.plot(t_grid, scores, linestyle=ls, color=color, alpha=alpha,
                    linewidth=0.8, marker="", label=label)

    # Proxy artists for legend (one line per model)
    proxies = []
    labels = []
    for m in models:
        style = MODEL_STYLE[m]
        proxies.append(plt.Line2D([], [], color=style["color"], linewidth=1.2))
        labels.append(style["label"])

    # Proxy for line styles = viewpoint
    vp_proxies = []
    vp_labels = []
    for vp in ["straight_on", "plus30", "minus30", "tilt_up", "tilt_down", "closer", "farther"]:
        ls = VIEWPOINT_LS.get(vp, "-")
        vp_proxies.append(plt.Line2D([], [], color="0.4", linestyle=ls, linewidth=0.8))
        vp_labels.append(vp.replace("_", " "))

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Score (0–50)")
    ax.set_title(f"Clip: {clip_id}  —  across viewpoints", fontsize=9, loc="left", pad=4)
    ax.grid(axis="both", which="major", color="0.9", lw=0.4, zorder=0)
    ax.set_ylim(0, 50)

    # Legend: models on top, viewpoints below
    legend1 = ax.legend(proxies, labels, loc="upper left", fontsize=7,
                        framealpha=0.8, edgecolor="0.7")
    ax.add_artist(legend1)
    ax.legend(vp_proxies, vp_labels, loc="upper right", fontsize=6.5,
              framealpha=0.8, edgecolor="0.7", title="viewpoint", title_fontsize=6.5)

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(out_dir, f"{clip_id}_trace.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"[fig] wrote {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Variant B1: generate figures and table from consistency_scores.json")
    ap.add_argument("--input", type=str,
                    default=os.path.join(os.path.dirname(__file__), "..",
                                         "outputs", "variant_b1", "consistency_scores.json"),
                    help="Input JSON from variant_b1_score.py")
    ap.add_argument("--out", type=str,
                    default=os.path.join(os.path.dirname(__file__), "..",
                                         "outputs", "variant_b1"),
                    help="Output directory")
    args = ap.parse_args()

    data = json.load(open(args.input))
    models = data.get("models", ["egru", "pct"])
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # --- Table: per-clip viewpoint spread ---
    table_path = os.path.join(args.out, "results_table.csv")
    with open(table_path, "w", newline="") as fh:
        w = csv.writer(fh)
        header = ["clip_id", "exercise", "n_viewpoints",
                  "EGRU_max-min", "EGRU_MAD", "EGRU_mean",
                  "PCT_max-min", "PCT_MAD", "PCT_mean"]
        if "invgru" in models:
            header += ["InvGRU_max-min", "InvGRU_MAD", "InvGRU_mean"]
        w.writerow(header)

        for cid, clip in sorted(data["per_clip"].items()):
            sp = clip["spread"]
            row = [
                cid, clip["exercise"], clip["n_viewpoints"],
                f"{sp['egru']['max_minus_min']:.4f}",
                f"{sp['egru']['mad']:.4f}",
                f"{sp['egru']['mean']:.2f}",
                f"{sp['pct']['max_minus_min']:.4f}",
                f"{sp['pct']['mad']:.4f}",
                f"{sp['pct']['mean']:.2f}",
            ]
            if "invgru" in sp:
                row += [
                    f"{sp['invgru']['max_minus_min']:.4f}",
                    f"{sp['invgru']['mad']:.4f}",
                    f"{sp['invgru']['mean']:.2f}",
                ]
            w.writerow(row)
    print(f"[fig] wrote {table_path}")

    # --- Figures: trace per clip ---
    per_take_by_clip = {}
    for rec in data["per_take"]:
        cid = rec["clip_id"]
        per_take_by_clip.setdefault(cid, []).append(rec)

    for cid, recs in sorted(per_take_by_clip.items()):
        make_trace_figure(cid, recs, models, fig_dir)

    print("[fig] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
