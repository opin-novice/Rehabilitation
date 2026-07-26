#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_fig_viewpoint.py  --  hero figure for paper_wacv.tex (fig:viewpoint).

MAD vs camera azimuth: certified-invariant models (EGRU SO(3), InvariantGRU SO(3)) are flat by
theorem; the point-cloud transformer degrades through the mean-predictor floor; rotation
augmentation flattens the transformer's aggregate curve but not its per-sequence swing.

Data is loaded from outputs/cde_block2/final_tables.json (key "viewpoint") -- every point traces to
the certified sweep; nothing is hand-entered. Colours are the CVD-safe Okabe-Ito categorical subset
(validated: all CVD checks PASS); series identity is carried by DIRECT LABELS, not colour alone.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ANGLES = [0, 15, 30, 45, 60, 90, 120, 150, 180]           # azimuth sweep (deg)
FLOOR = 8.313360007924256                                  # mean-predictor floor

# Okabe-Ito (CVD-safe). ours = blue (emphasised); non-equivariant = red/orange shades.
STYLE = {
    "EGRU  SO(3) chiral": dict(label="EGRU (ours)",   color="#0072B2", lw=2.4, z=10, ls="-"),
    "InvariantGRU  SO(3)": dict(label="InvariantGRU", color="#009E73", lw=1.8, z=9, ls="-"),
    "Ridge":              dict(label="Ridge (inv.)",  color="#2CA02C", lw=1.6, z=8, ls="--"),
    "PCT (baseline)":     dict(label="PCT",           color="#D55E00", lw=1.8, z=5, ls="-"),
    "ST-GCN":             dict(label="ST-GCN",        color="#FF7F0E", lw=1.6, z=4, ls="-"),
    "TCN":                dict(label="TCN",           color="#D62728", lw=1.6, z=3, ls="-"),
    "PCT + rot-aug":      dict(label="PCT + aug",     color="#E69F00", lw=1.6, z=6, ls="-"),
}


def main():
    # Look for final_tables.json in project root's outputs directory
    # HERE = paper/wacv_submission, go up 3 levels to project root
    import sys
    sys.path.insert(0, HERE)
    root_dir = os.path.abspath(os.path.join(HERE, "../.."))
    tables_path = os.path.join(root_dir, "outputs", "cde_block2", "final_tables.json")
    print(f"Loading from: {tables_path}")
    tables = json.load(open(tables_path))
    vp = {r["model"]: r["mad"] for r in tables["viewpoint"]}

    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.35, 2.5))

    # recessive floor reference
    ax.axhline(FLOOR, color="#9a9a93", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(ANGLES[0] + 2, FLOOR + 0.12, "mean-predictor floor", color="#6f6f68",
            fontsize=6.5, va="bottom")

    for model, st in STYLE.items():
        y = vp[model]
        ax.plot(ANGLES, y, color=st["color"], lw=st["lw"], ls=st["ls"], zorder=st["z"],
                marker="o", ms=3.0, mfc=st["color"], mec="white", mew=0.5,
                solid_capstyle="round")
        # direct label at the right end (identity without a legend box)
        yend = y[-1]
        dy = {"EGRU (ours)": +0.22, "InvariantGRU": -0.22}.get(st["label"], 0.0)
        ax.text(ANGLES[-1] + 3, yend + dy, st["label"], color=st["color"],
                fontsize=7, va="center", ha="left",
                fontweight="bold" if st["label"] == "EGRU (ours)" else "normal")

    ax.set_xlim(0, 232)
    ax.set_ylim(6.0, 13.5)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("Test-time camera azimuth (deg)")
    ax.set_ylabel("MAD (clinical score)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3)
    ax.grid(axis="y", color="#ececec", lw=0.5, zorder=0)

    fig.tight_layout(pad=0.4)
    out = os.path.join(HERE, "fig_viewpoint.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {out}")

    # Also save to outputs for reference
    out2 = os.path.join(root_dir, "outputs", "figures", "fig_viewpoint_with_baselines.pdf")
    os.makedirs(os.path.dirname(out2), exist_ok=True)
    fig.savefig(out2, bbox_inches="tight", pad_inches=0.02)
    print(f"also wrote {out2}")


if __name__ == "__main__":
    main()
