"""Generate the four deck figures from banked results.

Run:  python deck/make_charts.py

Writes PNGs into deck/figures/. Reads only JSON and the certificate text log --
see deckdata.py. Styling follows the repo figure contract in
src/variant_b1_figures.py (serif, thin axes, Agg) so the deck matches the paper.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import deckdata as dd

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#444444",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

FIGSIZE = (6.6, 4.6)


def _save(fig, name: str) -> None:
    dd.FIGURES.mkdir(parents=True, exist_ok=True)
    path = dd.FIGURES / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(dd.REPO)}")


# --------------------------------------------------------------------------- #
# 1. hero: viewpoint degradation
# --------------------------------------------------------------------------- #

VIEWPOINT_SERIES = [
    # (key in final_tables.json, legend label, colour, linestyle, linewidth)
    ("PCT (baseline)", "PCT (transformer)", dd.C_PCT, "-", 2.0),
    ("TCN", "TCN", dd.C_TCN, "-", 1.6),
    ("ST-GCN", "ST-GCN", dd.C_STGCN, "-", 1.6),
    ("PCT + rot-aug", "PCT + rotation aug.", dd.C_PCT, "--", 1.6),
    ("Ridge", "Ridge on invariants", dd.C_RIDGE, ":", 1.6),
    ("InvariantGRU  SO(3)", "InvariantGRU (hand-crafted)", dd.C_INVGRU, "-", 2.0),
    ("EGRU  SO(3) chiral", "EGRU (ours, certified)", dd.C_EGRU, "-", 2.6),
]


def fig_viewpoint() -> None:
    series = dd.viewpoint_series()
    floor = dd.rec("mean-predictor floor (MAD)", dd.floor_mad(), dd.FINAL_TABLES)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    if floor is not None:
        ax.axhline(floor, color=dd.C_FLOOR, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(
            183,
            floor + 0.12,
            "mean-predictor floor",
            color="#6f6f6f",
            fontsize=8,
            ha="right",
            va="bottom",
        )

    for key, label, colour, ls, lw in VIEWPOINT_SERIES:
        row = series.get(key)
        if not row or not row.get("mad"):
            dd.MISSING.append(f"viewpoint series missing: {key!r} in {dd.FINAL_TABLES}")
            continue
        mad = dd.rec(f"viewpoint MAD :: {key}", row["mad"], dd.FINAL_TABLES)
        ax.plot(
            dd.ANGLES, mad, color=colour, ls=ls, lw=lw, label=label,
            marker="o", ms=3.0, zorder=3,
        )

    ax.set_xlabel("test-time camera azimuth (degrees)")
    ax.set_ylabel("MAD  (clinical points, lower is better)")
    ax.set_xticks(dd.ANGLES)
    ax.set_xlim(-5, 185)
    ax.set_ylim(5.6, 13.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=2, columnspacing=1.1, handlelength=2.2)

    _save(fig, "fig_viewpoint.png")


# --------------------------------------------------------------------------- #
# 2. node failure
# --------------------------------------------------------------------------- #


def fig_nodefail() -> None:
    rows = dd.nodefail_rows()
    if not rows:
        print("  [n/a] node-failure rows missing; skipping fig_nodefail")
        return

    ks = [r["k"] for r in rows]
    floor = rows[0].get("floor")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    if floor is not None:
        ax.axhline(floor, color=dd.C_FLOOR, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(
            0.05, floor + 0.2, "mean-predictor floor",
            color="#6f6f6f", fontsize=8, va="bottom",
        )

    for key, std_key, label, colour in [
        ("invgru", "invgru_std", "InvariantGRU (hand-crafted)", dd.C_INVGRU),
        ("egru", "egru_std", "EGRU (ours, steerable)", dd.C_EGRU),
        ("pct", "pct_std", "PCT (transformer)", dd.C_PCT),
    ]:
        mad = [r[key] for r in rows]
        std = [r.get(std_key, 0.0) for r in rows]
        dd.rec(f"node-failure MAD :: {key}", mad, dd.FINAL_TABLES)
        ax.plot(ks, mad, color=colour, lw=2.2, marker="o", ms=3.5, label=label, zorder=3)
        ax.fill_between(
            ks,
            [m - s for m, s in zip(mad, std)],
            [m + s for m, s in zip(mad, std)],
            color=colour, alpha=0.14, lw=0, zorder=2,
        )

    ax.set_xlabel("dead sensor nodes  $k$  (joints frozen at their first-frame pose)")
    ax.set_ylabel("MAD  (clinical points, lower is better)")
    ax.set_xticks(ks)
    ax.set_ylim(5.6, 19.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")

    _save(fig, "fig_nodefail.png")


# --------------------------------------------------------------------------- #
# 3. precision budget
# --------------------------------------------------------------------------- #

# (display label, source key, which artifact, verdict band)
PRECISION_ROWS = [
    ("fp64", "fp64", "float", "exact"),
    ("fp32", "fp32", "float", "exact"),
    ("int8 — weights only", "W8", "int8", "exact"),
    ("fp16", "fp16", "float", "degraded"),
    ("bf16", "bf16", "float", "degraded"),
    ("int8 — activations only", "A8", "int8", "broken"),
    ("int8 — weights + acts", "W8A8", "int8", "broken"),
]

BAND_COLOUR = {"exact": dd.C_EGRU, "degraded": "#d1852a", "broken": dd.C_PCT}


def fig_precision() -> None:
    floats = dd.precision_budget()
    ints = dd.int8_budget()
    if not floats and not ints:
        print("  [n/a] precision artifacts missing; skipping fig_precision")
        return

    labels, values, colours = [], [], []
    for label, key, which, band in PRECISION_ROWS:
        src = floats if which == "float" else ints
        source = dd.PRECISION if which == "float" else dd.INT8
        row = (src or {}).get("rows", {}).get(key)
        val = row.get("inv_floor") if row else None
        dd.rec(f"invariance floor :: {label}", val, source)
        if val is None:
            continue
        labels.append(label)
        values.append(val)
        colours.append(BAND_COLOUR[band])

    if not values:
        print("  [n/a] no precision rows resolved; skipping fig_precision")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ypos = list(range(len(labels)))[::-1]
    ax.barh(ypos, values, color=colours, height=0.62, zorder=3)

    for y, v in zip(ypos, values):
        ax.text(v * 1.6, y, f"{v:.2e}", va="center", fontsize=8, color="#333333")

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xscale("log")
    ax.set_xlim(1e-16, 1e3)
    ax.set_xlabel("relative violation of the invariant read-out  (log scale)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    handles = [
        plt.Line2D([], [], color=BAND_COLOUR["exact"], lw=7, label="theorem holds"),
        plt.Line2D([], [], color=BAND_COLOUR["degraded"], lw=7, label="degraded but usable"),
        plt.Line2D([], [], color=BAND_COLOUR["broken"], lw=7, label="the cliff"),
    ]
    ax.legend(handles=handles, loc="upper right")

    _save(fig, "fig_precision.png")


# --------------------------------------------------------------------------- #
# 4. adaptive-solver drift
# --------------------------------------------------------------------------- #


def fig_solver() -> None:
    rows = dd.solver_gate_rows()
    if not rows:
        print("  [n/a] solver gate rows missing; skipping fig_solver")
        return

    for r in rows:
        dd.rec(
            f"solver drift :: {r['label'].replace(chr(10), ' ')}",
            r["drift"],
            dd.CERT_LOG,
        )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ypos = list(range(len(rows)))[::-1]
    colours = [dd.C_PCT if r["norm"] == "default" else dd.C_EGRU for r in rows]
    ax.barh(ypos, [r["drift"] for r in rows], color=colours, height=0.6, zorder=3)

    for y, r in zip(ypos, rows):
        ax.text(
            r["drift"] * 2.2,
            y,
            f"{r['drift']:.2e}    $D_{{grid}}$ = {r['d_grid']}",
            va="center", fontsize=8, color="#333333",
        )

    ax.set_yticks(ypos)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xscale("log")
    ax.set_xlim(1e-16, 1e2)
    ax.set_xlabel("equivariance drift through the adaptive solver  (log scale)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    handles = [
        plt.Line2D([], [], color=dd.C_PCT, lw=7, label="default per-component norm"),
        plt.Line2D([], [], color=dd.C_EGRU, lw=7, label="per-irrep isotropic norm  $N_{eq}$"),
    ]
    ax.legend(handles=handles, loc="lower right")

    _save(fig, "fig_solver.png")


def main() -> None:
    print("Generating deck figures...")
    fig_viewpoint()
    fig_nodefail()
    fig_precision()
    fig_solver()
    if dd.MISSING:
        print("\nMissing / unresolved artifacts:")
        for m in dd.MISSING:
            print(f"  - {m}")
    else:
        print("\nAll figure inputs resolved.")


if __name__ == "__main__":
    main()
