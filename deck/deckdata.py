"""Shared data access for the codebase engineering deck.

Reads only banked run artifacts (result JSON and the certificate text log). It
never imports torch or e3nn, so the deck rebuilds in seconds and cannot perturb
an experiment.

Every value the deck shows is pulled through ``rec()``, which records the value
together with the file it came from. ``build_deck.py`` prints that table at the
end of a build, so any number on a slide can be traced back to an artifact. A
missing artifact yields ``None`` (rendered as ``n/a``) and is reported loudly --
the deck never invents a number to fill a hole.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DECK = REPO / "deck"
FIGURES = DECK / "figures"

# Test-time camera azimuths used by every Block-3 sweep (src/block3_baselines.py).
ANGLES = [0, 15, 30, 45, 60, 90, 120, 150, 180]

# Figure colours, matching the contract in src/variant_b1_figures.py so the deck
# reads as the same family as the paper figures.
C_EGRU = "#1a7f6b"
C_PCT = "#c24a3f"
C_INVGRU = "#6b5b8a"
C_RIDGE = "#7d7d7d"
C_TCN = "#d1852a"
C_STGCN = "#2a6fb5"
C_FLOOR = "#9a9a9a"

# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #

PROVENANCE: list[tuple[str, str, str]] = []
MISSING: list[str] = []


def rec(label: str, value, source: str):
    """Record ``value`` under ``label`` as having come from ``source``."""
    if value is None:
        MISSING.append(f"{label}  <- {source}")
        shown = "n/a"
    elif isinstance(value, float):
        shown = f"{value:.6g}"
    elif isinstance(value, (list, tuple)):
        shown = f"[{len(value)} values]"
    else:
        shown = str(value)
    PROVENANCE.append((label, shown, source))
    return value


def fmt(value, spec: str = "") -> str:
    """Format a possibly-missing value for slide text."""
    if value is None:
        return "n/a"
    return format(value, spec) if spec else str(value)


# --------------------------------------------------------------------------- #
# artifact loading
# --------------------------------------------------------------------------- #


def _load_json(rel: str):
    path = REPO / rel
    if not path.exists():
        MISSING.append(f"file not found: {rel}")
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_text(rel: str):
    path = REPO / rel
    if not path.exists():
        MISSING.append(f"file not found: {rel}")
        return None
    return path.read_text(encoding="utf-8", errors="replace")


FINAL_TABLES = "outputs/cde_block2/final_tables.json"
CERTIFY = "outputs/cde_block2/certify_egru.json"
BANDWIDTH = "outputs/cde_block2/block5_bandwidth_law.json"
PRECISION = "outputs/precision_budget/f7c_precision_budget.json"
INT8 = "outputs/precision_budget/f7c_int8_quant.json"
TTFS = "outputs/ntu_stream/ttfs_benchmark_s42.json"
STREAM = "outputs/ntu_stream/stream_results_s42.json"
REHAB246_EGRU = "outputs/rehab246/rehab246_egru_chiral_s{seed}.json"
REHAB246_PCT = "outputs/rehab246/rehab246_pct_s{seed}.json"
CERT_LOG = "docs/reference/outputs_equivariance_certificate.txt"


def final_tables():
    return _load_json(FINAL_TABLES)


def certify():
    return _load_json(CERTIFY)


def bandwidth():
    return _load_json(BANDWIDTH)


def precision_budget():
    return _load_json(PRECISION)


def int8_budget():
    return _load_json(INT8)


def ttfs():
    return _load_json(TTFS)


def stream():
    return _load_json(STREAM)


def rehab246(pattern: str, seeds=(0, 1, 2)):
    """Mean clean AUROC and worst viewpoint logit drift across seeds.

    Returns ``(auroc_mean, drift_max)``; either element is ``None`` if no seed
    file resolved.
    """
    aurocs, drifts = [], []
    for seed in seeds:
        data = _load_json(pattern.format(seed=seed))
        if not data:
            continue
        pair = data.get("clean_auroc_mean_std")
        if pair:
            aurocs.append(pair[0])
        drift = data.get("view_logit_drift_max")
        if drift is not None:
            drifts.append(drift)
    return (
        sum(aurocs) / len(aurocs) if aurocs else None,
        max(drifts) if drifts else None,
    )


# --------------------------------------------------------------------------- #
# derived views
# --------------------------------------------------------------------------- #


def viewpoint_series() -> dict[str, dict]:
    """model -> {'mad': [9 values], 'max_degr': float|None} from final_tables.json."""
    data = final_tables()
    if not data:
        return {}
    out = {}
    for row in data.get("viewpoint", []):
        out[row["model"]] = {
            "mad": row.get("mad"),
            "max_degr": row.get("max_degr"),
        }
    return out


def accuracy_rows() -> dict[str, dict]:
    """model -> {'mad', 'std', 'params'} from final_tables.json."""
    data = final_tables()
    if not data:
        return {}
    return {r["model"]: r for r in data.get("accuracy", [])}


def nodefail_rows() -> list[dict]:
    data = final_tables()
    return data.get("nodefail", []) if data else []


def floor_mad():
    data = final_tables()
    return data.get("floor") if data else None


def worst_mean_degradation(rel: str):
    """Max over angles of the fold-averaged degradation, from a block3_* dump.

    The baseline sweeps store one row per (fold, angle); the paper quotes the
    fold-averaged curve, so that is what is recomputed here rather than the
    single worst fold.
    """
    data = _load_json(rel)
    if not data:
        return None
    by_angle: dict[int, list[float]] = {}
    for row in data.get("rows", []):
        by_angle.setdefault(row["angle"], []).append(row["degradation"])
    if not by_angle:
        return None
    return max(sum(v) / len(v) for v in by_angle.values())


_SOLVER_RE = re.compile(
    r"^\s*(default|neq)\s+max drift = ([0-9.eE+-]+).*?max D_grid = (\d+)",
    re.MULTILINE,
)


def solver_gate_rows():
    """Adaptive-solver drift and step-grid divergence from the certificate log.

    Returns four rows in file order -- mock field (default, N_eq) then the real
    e3nn field (default, N_eq) -- or ``None`` if the log is absent or its format
    has changed.
    """
    text = _load_text(CERT_LOG)
    if not text:
        return None
    hits = _SOLVER_RE.findall(text)
    if len(hits) < 4:
        MISSING.append(
            f"{CERT_LOG}: expected 4 solver rows, matched {len(hits)}"
        )
        return None
    labels = [
        ("mock field\ndefault norm", "default"),
        ("mock field\n$N_{eq}$", "neq"),
        ("real e3nn\ndefault norm", "default"),
        ("real e3nn\n$N_{eq}$", "neq"),
    ]
    rows = []
    for (label, want), (norm, drift, dgrid) in zip(labels, hits[:4]):
        assert norm == want, f"certificate log row order changed: {norm} != {want}"
        rows.append(
            {
                "label": label,
                "norm": norm,
                "drift": float(drift),
                "d_grid": int(dgrid),
            }
        )
    return rows
