#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
System diagram for the capstone report (Figure 1).

Regenerates ``system_diagram.png`` at the repo root, 300 DPI, greyscale-safe.
Box labels are kept byte-identical to the stage names used in the METHOD section of
``capstone-draft.md`` -- if you rename a stage there, rename it here too.

    python scripts/make_system_diagram.py

Graphviz/mermaid are not installed on the authoring machine; the pipeline is linear
with one branch, which is the case the diagram guide allows matplotlib for.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "system_diagram.png")

FONT = "DejaVu Sans"
INK = "#111111"
FS_BOX, FS_EDGE = 11.5, 9.5
SPINE_X, SPINE_W = 3.05, 4.55
BRANCH_X, BRANCH_W = 8.00, 3.35

# (x_center, y_center, width, height, label, fill, linewidth)
BOXES = {
    "client":  (SPINE_X, 9.35, SPINE_W, 0.98,
                "Browser client\nwebcam  |  video upload", "#ffffff", 1.1),
    "mp":      (SPINE_X, 7.90, SPINE_W, 0.98,
                "MediaPipe\nPoseLandmarker", "#f2f2f2", 1.1),
    "remap":   (SPINE_X, 6.45, SPINE_W, 0.98,
                "Kinect-25 remap\nroot-relative", "#f2f2f2", 1.1),
    "enc":     (SPINE_X, 4.85, SPINE_W, 0.98,
                "Steerable encoder\ne3nn,  2 layers", "#e0e0e0", 1.1),
    "cut":     (SPINE_X, 3.40, SPINE_W, 0.98,
                "Invariant cut\n283 generators", "#c8c8c8", 2.2),
    "gru":     (SPINE_X, 1.95, SPINE_W, 0.98,
                "Bi-GRU + head\n5-fold ensemble", "#e0e0e0", 1.1),
    "bio":     (BRANCH_X, 4.85, BRANCH_W, 0.98,
                "Biomechanics\n5 metrics", "#f2f2f2", 1.1),
    "report":  (5.50, 0.38, 6.70, 0.88,
                "Report card  (PNG + JSON)", "#ffffff", 1.6),
}

# (from, to, label, label_offset_x, label_side)
EDGES = [
    ("client", "mp",     "~15 fps JPEG  /  WebSocket", 0.12, "right"),
    ("mp",     "remap",  "33 landmarks", 0.12, "right"),
    ("remap",  "enc",    "25x3 coords", 0.12, "right"),
    ("enc",    "cut",    "O(3) irreps  32x0e + 8x1o", 0.12, "right"),
    ("cut",    "gru",    "283-d invariants + dt", 0.12, "right"),
]


def _draw_box(ax, spec):
    x, y, w, h, label, fill, lw = spec
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.14",
        linewidth=lw, edgecolor=INK, facecolor=fill, zorder=2))
    ax.text(x, y, label, ha="center", va="center", fontsize=FS_BOX,
            fontname=FONT, color=INK, linespacing=1.40, zorder=3)


def _arrow(ax, p0, p1, rad=0.0, lw=1.3):
    ax.add_patch(FancyArrowPatch(
        p0, p1, connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=13, linewidth=lw,
        color=INK, zorder=1, shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(6.75, 6.05))
    ax.set_xlim(-1.00, 10.5)
    ax.set_ylim(-0.30, 10.05)
    ax.axis("off")

    for spec in BOXES.values():
        _draw_box(ax, spec)

    # --- vertical spine -------------------------------------------------
    for src, dst, label, dx, _side in EDGES:
        xs, ys, _w, hs = BOXES[src][:4]
        _xd, yd, _wd, hd = BOXES[dst][:4]
        y0, y1 = ys - hs / 2 - 0.06, yd + hd / 2 + 0.06
        _arrow(ax, (xs, y0), (xs, y1))
        ax.text(xs + dx + 0.10, (y0 + y1) / 2, label, ha="left", va="center",
                fontsize=FS_EDGE, fontname=FONT, color=INK)

    # --- the seam: this is the whole thesis -----------------------------
    # Left-margin brackets rather than a horizontal rule, so nothing crosses an
    # edge label or the biomechanics branch (which the seam does not govern).
    def _bracket(y_top, y_bot, label):
        x = -0.34
        ax.plot([x, x], [y_bot, y_top], linewidth=1.1, color="#333333", zorder=1)
        for y in (y_top, y_bot):
            ax.plot([x, x + 0.16], [y, y], linewidth=1.1, color="#333333", zorder=1)
        ax.text(x - 0.18, (y_top + y_bot) / 2, label, ha="center", va="center",
                rotation=90, fontsize=FS_EDGE, fontname=FONT, style="italic",
                color="#333333")

    _bracket(BOXES["remap"][1] + BOXES["remap"][3] / 2,
             BOXES["enc"][1] - BOXES["enc"][3] / 2, "equivariant")
    _bracket(BOXES["cut"][1] + BOXES["cut"][3] / 2,
             BOXES["gru"][1] - BOXES["gru"][3] / 2, "invariant by theorem")

    # --- branch: pose buffer also drives the explainable metrics --------
    xr, yr, wr, hr = BOXES["remap"][:4]
    xb, yb, _wb, hb = BOXES["bio"][:4]
    _arrow(ax, (xr + wr / 2 + 0.06, yr), (xb, yb + hb / 2 + 0.06), rad=-0.30)
    ax.text(7.35, 6.68, "same pose buffer", ha="center", va="bottom",
            fontsize=FS_EDGE, fontname=FONT, color=INK)

    # --- both paths join at the report ----------------------------------
    xg, yg, _wg, hg = BOXES["gru"][:4]
    xrep, yrep, _wrep, hrep = BOXES["report"][:4]
    _arrow(ax, (xg, yg - hg / 2 - 0.06), (xg + 0.50, yrep + hrep / 2 + 0.06), rad=0.0)
    ax.text(xg - 0.22, 1.06, "AI score /50", ha="right", va="center",
            fontsize=FS_EDGE, fontname=FONT, color=INK)
    _arrow(ax, (xb, yb - hb / 2 - 0.06), (xb - 0.30, yrep + hrep / 2 + 0.06), rad=0.10)
    ax.text(xb + 0.18, 2.60, "quality /100", ha="left", va="center",
            fontsize=FS_EDGE, fontname=FONT, color=INK)

    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
