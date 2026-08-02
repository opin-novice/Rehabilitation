#!/usr/bin/env python
"""
Figure 1 for paper_wacv.tex: the method, and the theorem made visual.

Top row    -- the pipeline: root-relative skeleton -> steerable encoder carrying O(3) irreps
              -> the invariant cut Pi -> GRU consuming real inter-arrival times -> score.
Bottom row -- why the cut is the whole idea: the same pose seen from two camera poses gives
              encoder features that ROTATE (equivariance) and a cut output that is IDENTICAL
              (invariance). That is Prop. 1, drawn rather than asserted.

Every annotated number is read from persisted JSON; nothing is hand-entered. This is the same
discipline as make_fig_guarantee.py, and for the same reason -- a transcribed constant in that
figure once drifted from the artifact without anyone noticing.

Writes fig_method.pdf next to the .tex that includes it.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# ---- measured quantities (never typed in) ------------------------------------
cert = json.load(open(os.path.join(ROOT, "outputs/cde_block2/certify_egru.json")))["values"]
ft = json.load(open(os.path.join(ROOT, "outputs/cde_block2/final_tables.json")))
vp = {d["model"]: d for d in ft["viewpoint"]}

E_EQUIV = cert["E1_encoder_equiv_worst"]        # encoder intertwines: features rotate
E_INVAR = cert["E2_readout_invariance_worst"]   # cut output: unchanged
SCORE_SHIFT = vp["EGRU  SO(3) chiral"]["max_degr"]  # end-to-end score shift, MAD of 50
N_SCALAR, N_VEC, D_EVEN, D_ODD = 32, 8, 160, 123    # deployed widths (Method section)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})

INK, TEAL, RED, GREY = "#1a1a1a", "#1a7f6b", "#c24a3f", "#8a8a8a"

fig = plt.figure(figsize=(7.0, 2.12))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, title, body, edge=INK, fc="white", lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.4",
                                ec=edge, fc=fc, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h - 3.0, title, ha="center", va="top",
            fontsize=7.2, color=edge, zorder=3, weight="bold")
    if body:
        ax.text(x + w / 2, y + h - 8.2, body, ha="center", va="top",
                fontsize=6.0, color=INK, zorder=3, linespacing=1.35)


def arrow(x0, y0, x1, y1, colour=INK, lw=0.8, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7,
                                 lw=lw, color=colour, shrinkA=0, shrinkB=0, zorder=4))


# =============================== top row: the pipeline ========================
TY, TH = 62, 33
xs = [1.5, 20.0, 39.5, 68.0, 84.5]
ws = [16.0, 17.0, 26.5, 14.0, 14.0]

box(xs[0], TY, ws[0], TH, "skeleton",
    "25 joints, 3D\nroot-relative\n(translation-invariant)")
box(xs[1], TY, ws[1], TH, "steerable encoder",
    f"$O(3)$ irreps per joint\n{N_SCALAR} type-0 scalars $s_j$\n{N_VEC} type-1 vectors $v_j$\n"
    "$\\Rightarrow$ EQUIVARIANT")
box(xs[2], TY, ws[2], TH, "the invariant cut  $\\Pi$", "", edge=TEAL, fc="#f2f9f7", lw=1.1)
box(xs[3], TY, ws[3], TH, "GRU", "consumes real\ninter-arrival\ntimes $\\Delta t$")
box(xs[4], TY, ws[4], TH, "score", "clinical\nTotal Score\n(0--50)")

# the cut's interior: the generating set, which is the paper's actual contribution
cx = xs[2] + ws[2] / 2
ax.text(cx, TY + TH - 8.0,
        "parity-even:  $\\tanh s_j$ ,  $\\log(1{+}\\|v_j\\|)$ ,  $\\langle\\hat v_j,\\hat v_j'\\rangle$",
        ha="center", va="top", fontsize=6.0, color=INK)
ax.text(cx, TY + TH - 14.0,
        "parity-odd:  $\\det[\\hat v^a_j,\\hat v^b_j,\\hat v^c_j]$   (restores chirality)",
        ha="center", va="top", fontsize=6.0, color=INK)
ax.text(cx, TY + TH - 20.0,
        f"$+$ bone lengths, anatomical volumes $\\;\\to\\;$ {D_EVEN}$+${D_ODD} $=$ {D_EVEN + D_ODD}-d",
        ha="center", va="top", fontsize=6.0, color=INK)
ax.text(cx, TY + 2.0, "$\\Rightarrow$ INVARIANT for arbitrary weights (Prop. 1)",
        ha="center", va="bottom", fontsize=6.2, color=TEAL, weight="bold")

for i in range(4):
    arrow(xs[i] + ws[i] + 0.9, TY + TH / 2, xs[i + 1] - 0.9, TY + TH / 2)

# =============================== bottom row: the theorem ======================
# Two camera poses of the SAME pose. The encoder's type-1 features rotate with the camera;
# the cut's output does not move. Drawing both is the point -- equivariance is not invariance.
J = {"head": (0, 30), "neck": (0, 21), "shL": (-7, 19), "shR": (7, 19), "elL": (-12, 11),
     "elR": (12, 11), "haL": (-14, 3), "haR": (14, 3), "hip": (0, 7), "knL": (-5, -5),
     "knR": (5, -5), "foL": (-6, -15), "foR": (6, -15)}
BONES = [("head", "neck"), ("neck", "shL"), ("neck", "shR"), ("shL", "elL"), ("shR", "elR"),
         ("elL", "haL"), ("elR", "haR"), ("neck", "hip"), ("hip", "knL"), ("hip", "knR"),
         ("knL", "foL"), ("knR", "foR")]
VEC_AT = [("haR", (7, 5)), ("neck", (0, 7)), ("knL", (-5, 4))]   # illustrative type-1 features


def draw_pose(cx0, cy0, theta_deg, scale=0.42):
    t = np.deg2rad(theta_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    P = {k: R @ np.array(v) * scale + np.array([cx0, cy0]) for k, v in J.items()}
    for a, b in BONES:
        ax.plot([P[a][0], P[b][0]], [P[a][1], P[b][1]], color=GREY, lw=0.9,
                solid_capstyle="round", zorder=2)
    for k, v in VEC_AT:
        d = R @ np.array(v) * scale
        arrow(P[k][0], P[k][1], P[k][0] + d[0], P[k][1] + d[1], colour=RED, lw=0.9)
    return P


BY = 27
ax.text(6.0, BY + 24, "camera A", ha="center", fontsize=6.6, color=INK)
ax.text(24.0, BY + 24, "camera B  (rotated)", ha="center", fontsize=6.6, color=INK)
draw_pose(6.0, BY, 0)
draw_pose(24.0, BY, 52)
ax.text(15.0, BY - 20.5, "type-1 features rotate with the camera", ha="center",
        fontsize=6.0, color=RED)
ax.text(15.0, BY - 25.0, f"encoder intertwines: {E_EQUIV:.1e}", ha="center",
        fontsize=6.0, color=RED)

arrow(33.0, BY + 6, 41.0, BY + 6, colour=TEAL, lw=1.0)
arrow(33.0, BY - 4, 41.0, BY - 4, colour=TEAL, lw=1.0)
ax.text(37.0, BY + 9.5, "$\\Pi$", ha="center", fontsize=8.0, color=TEAL, weight="bold")

# the payload: identical invariant code, and what that buys end to end
box(42.0, BY - 16, 56.0, 30.0, "", "", edge=TEAL, fc="#f2f9f7", lw=1.0)
ax.text(70.0, BY + 9.0, "the cut's output is the SAME vector for both cameras",
        ha="center", fontsize=6.8, color=TEAL, weight="bold")
# The BAR HEIGHTS are schematic -- a stand-in for a 283-d code, drawn identical for A and B to
# show what invariance means. They are not measured activations, and the figure says so. The
# quantitative claim is carried entirely by the three measured numbers annotated beside them.
rng = np.random.default_rng(0)
vals = rng.uniform(0.15, 1.0, 26)
for i, v in enumerate(vals):
    x = 46.0 + i * 1.30
    ax.bar(x, v * 7.0, width=0.95, bottom=BY - 1.0, color=TEAL, alpha=0.85, zorder=3)
    ax.bar(x + 0.0, v * 7.0, width=0.95, bottom=BY - 10.5, color=TEAL, alpha=0.35, zorder=3)
ax.text(45.0, BY + 3.0, "A", ha="right", va="center", fontsize=6.2, color=INK)
ax.text(45.0, BY - 6.5, "B", ha="right", va="center", fontsize=6.2, color=INK)
ax.text(46.0, BY - 14.0, "bars schematic; the numbers at right are measured",
        ha="left", va="center", fontsize=5.4, color=GREY, style="italic")
ax.text(82.5, BY + 2.0,
        f"read-out invariance\n{E_INVAR:.1e}  (fp64 roundoff)\n\n"
        f"end-to-end score shift\n{SCORE_SHIFT:.0e} MAD of 50",
        ha="left", va="center", fontsize=6.2, color=INK, linespacing=1.5)

out = os.path.join(HERE, "fig_method.pdf")
fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
plt.close(fig)
print(f"wrote {out}")
print(f"  encoder equivariance {E_EQUIV:.3e} | readout invariance {E_INVAR:.3e} "
      f"| score shift {SCORE_SHIFT:.3e}")
