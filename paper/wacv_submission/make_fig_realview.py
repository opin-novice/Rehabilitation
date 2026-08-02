#!/usr/bin/env python
"""
Supplement figure for paper_wacv.tex: "a real camera move, not a synthetic rotation."

Paired score shift across seven PHYSICAL camera poses, relative to each model's own straight-on
score, on the same replayed exercise clip. Source: outputs/real_viewpoint/summary.json, written by
src/real_viewpoint_probe.py. Every number is read from that artifact; nothing is hand-entered.

We plot the ALIGNED arm: the takes film a looping clip, so each recording starts at a different
phase, and the shifts are only interpretable once phase is held fixed and the comparison is paired.
Error bars are the standard error over the paired phases, so a reader can see which shifts are
resolvable at all rather than taking a single number on trust.

Writes fig_realview.pdf next to the .tex that includes it.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

D = json.load(open(os.path.join(ROOT, "outputs/real_viewpoint/summary.json")))
AL = D["aligned"]
assert AL.get("feasible"), "aligned arm not feasible; nothing to plot"
REF = D["scope"]["reference_view"]
VIEWS = D["view_order"]

PRETTY = {"minus30": "$-30^\\circ$", "straight_on": "straight\non", "plus30": "$+30^\\circ$",
          "tilt_up": "tilt\nup", "tilt_down": "tilt\ndown", "closer": "closer", "farther": "farther"}
STYLE = {"egru": ("EGRU (ours)", "#1a7f6b", "o"), "pct": ("PCT", "#c24a3f", "s")}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
})
fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.3))

xs = range(len(VIEWS))
for model, (label, colour, marker) in STYLE.items():
    st = AL["stats"][model]
    mu = [st["per_view"][v]["mean_shift"] for v in VIEWS]
    se = [st["per_view"][v]["sem_shift"] for v in VIEWS]
    ax.errorbar(list(xs), mu, yerr=se, color=colour, marker=marker, ms=3.4, lw=1.1,
                capsize=2, elinewidth=0.8, zorder=3,
                label=f"{label}: mean $|\\Delta|$ {st['mean_abs_shift']:.2f}, "
                      f"worst {st['max_abs_shift']:.2f}")

ax.axhline(0, color="0.35", ls="--", lw=0.8, zorder=2)
ax.set_xticks(list(xs))
ax.set_xticklabels([PRETTY.get(v, v) for v in VIEWS], fontsize=6.6)
ax.set_ylabel("score shift vs.\\ straight-on\n(of $50$, paired by phase)", fontsize=7)
ax.set_xlabel(f"physical camera pose — one subject, one exercise "
              f"({AL['window_s']:.1f}\\,s window, {len(AL['phases_s'])} phases)", fontsize=6.6)
ax.grid(axis="y", which="major", color="0.9", lw=0.4, zorder=0)
ax.legend(fontsize=6.4, loc="lower left", frameon=False, handlelength=1.6)
ax.set_title("Moving the camera, not rotating the skeletons", fontsize=8, loc="left", pad=4)

fig.savefig(os.path.join(HERE, "fig_realview.pdf"), bbox_inches="tight")
print("wrote fig_realview.pdf")
for m in STYLE:
    st = AL["stats"][m]
    print(f"  {m}: mean|shift| {st['mean_abs_shift']:.3f}  max {st['max_abs_shift']:.3f} "
          f"({st['max_abs_shift_view']})  significant {st['n_views_significant']}/{len(VIEWS)-1}")
print("  robustness ratios:",
      {k: round(v["pct_over_egru"], 2) for k, v in D["robustness"].items() if v.get("feasible")})
