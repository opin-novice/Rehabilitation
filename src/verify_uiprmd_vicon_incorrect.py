"""Verify that Incorrect Segmented Movements/Vicon/Positions is genuine world coordinates.

Background (docs/worklog_2026-07-29.md §2): UI-PRMD's *Kinect* `Positions` files are
parent-relative bone offsets, not world coordinates. Their signature is stark and systematic:
19 of 66 coordinate slots have exactly zero temporal variance *in every file*, and those dead
slots hold **nonzero constants** (12.4 / 25.9 / 26.1 cm) -- fixed bone lengths. Only the root
genuinely translates. Anything trained on them saw a near-static skeleton.

The worklog also recorded that the incorrect half had "no Angles/, no Vicon ... no rotation data
anywhere on disk". That was true of the *extracted directory* but not of the source archive --
`Incorrect Segmented Movements.zip` carries all four subfolders. This script certifies the
Vicon/Positions half that was missing.

The discriminating test is NOT "are there any zero-variance slots" -- the trusted correct side has
those too. It is:

  (1) do markers actually move (median per-slot temporal sd on the order of the control), and
  (2) are the constant slots *zero-valued dropouts* rather than *nonzero-constant bone lengths*.

A dropped/unlabelled Vicon marker is written as (0,0,0) for a whole rep. A bone offset is a fixed
nonzero length. Those are opposite signatures and only the second one invalidates the geometry.

Emits an exclusion manifest of unusable sequences (single-frame segments, dropped markers).

Usage:
    python src/verify_uiprmd_vicon_incorrect.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INC = ROOT / "Incorrect Segmented Movements" / "Vicon" / "Positions"
COR = ROOT / "Segmented Movements" / "Vicon" / "Positions"
OUT = ROOT / "outputs" / "uiprmd_vicon" / "exclusions.json"

N_MARKERS = 39
N_COLS = N_MARKERS * 3
STILL_MM = 1e-9  # Vicon sd is in mm; a live marker never sits at exactly 0.0 sd.
MIN_FRAMES = 2


def audit(directory: Path, label: str) -> dict:
    files = sorted(directory.glob("*.txt"))
    bad_cols, short, dropout, offset_like = [], [], [], []
    median_sds = []

    for f in files:
        arr = np.loadtxt(f, delimiter=",", dtype=np.float64, ndmin=2)
        if arr.shape[1] != N_COLS:
            bad_cols.append(f.name)
            continue
        if arr.shape[0] < MIN_FRAMES:
            short.append({"file": f.name, "frames": int(arr.shape[0])})
            continue

        seq = arr.reshape(arr.shape[0], N_MARKERS, 3)
        sd = seq.std(axis=0)  # (39, 3) temporal sd per coordinate slot
        median_sds.append(float(np.median(sd)))

        dead = sd <= STILL_MM
        if not dead.any():
            continue
        for m in np.flatnonzero(dead.any(axis=1)):
            held = seq[0, m][dead[m]]
            rec = {"file": f.name, "marker": int(m), "n_slots": int(dead[m].sum())}
            if np.allclose(held, 0.0):
                dropout.append(rec)  # marker not labelled this rep -> (0,0,0)
            else:
                rec["held_value_mm"] = [float(v) for v in held]
                offset_like.append(rec)  # fixed nonzero constant -> bone-offset signature

    n_ok = len(files) - len(bad_cols) - len(short)
    hit = {d["file"] for d in dropout} | {d["file"] for d in offset_like}
    return {
        "label": label,
        "dir": str(directory),
        "n_files": len(files),
        "n_usable": n_ok - len({d["file"] for d in dropout} | {d["file"] for d in offset_like}),
        "bad_cols": bad_cols,
        "short": short,
        "dropout": dropout,
        "offset_like": offset_like,
        "files_affected": sorted(hit),
        "median_sd_mm": float(np.median(median_sds)) if median_sds else 0.0,
        "min_median_sd_mm": float(np.min(median_sds)) if median_sds else 0.0,
    }


def report(a: dict) -> None:
    print(f"\n--- {a['label']} ---")
    print(f"  files                 : {a['n_files']}")
    print(f"  wrong column count    : {len(a['bad_cols'])}")
    print(f"  single-frame segments : {len(a['short'])} {[s['file'] for s in a['short']][:3]}")
    print(f"  markers dropped (0,0,0): {len(a['dropout'])} slots-groups "
          f"across {len({d['file'] for d in a['dropout']})} files")
    print(f"  NONZERO-const (offsets): {len(a['offset_like'])}  <-- bone-offset signature")
    print(f"  median per-slot sd    : {a['median_sd_mm']:.3f} mm "
          f"(worst file {a['min_median_sd_mm']:.3f} mm)")
    print(f"  clean sequences       : {a['n_usable']}/{a['n_files']}")


def main() -> int:
    if not INC.is_dir():
        print(f"ABORT: {INC} does not exist.")
        return 2

    inc = audit(INC, "INCORRECT (newly acquired)")
    cor = audit(COR, "CORRECT (known-good control)")
    report(inc)
    report(cor)

    # Bone offsets would show a fixed nonzero constant on a *systematic* fraction of files.
    # Isolated single-axis holds on a handful of files are marker noise, not offset geometry.
    offset_files = len({d["file"] for d in inc["offset_like"]})
    ctrl_offset_files = len({d["file"] for d in cor["offset_like"]})

    print("\n=== GATES ===")
    gates = {
        "all files parse as 117 cols (39 markers x 3)": not inc["bad_cols"],
        "markers genuinely move (median sd >= 1 mm)": inc["median_sd_mm"] >= 1.0,
        "no systematic bone-offset signature (<1% of files)": offset_files <= 0.01 * inc["n_files"],
        "offset-like rate no worse than control": offset_files <= max(ctrl_offset_files * 3, 10),
        "usable sequences >= 900": inc["n_usable"] >= 900,
    }
    for name, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "note": "Sequences to exclude from the UI-PRMD binary task (incorrect side).",
        "single_frame": inc["short"],
        "dropped_markers": inc["dropout"],
        "nonzero_constant": inc["offset_like"],
        "exclude_files": sorted(set(inc["files_affected"]) | {s["file"] for s in inc["short"]}),
    }, indent=2))
    print(f"\n  exclusion manifest -> {OUT.relative_to(ROOT)}")

    if all(gates.values()):
        print("\nVERDICT: PASS -- Vicon world coordinates confirmed on the incorrect half. "
              "The binary correct/incorrect task is unblocked.")
        return 0
    print("\nVERDICT: FAIL -- do not build on this.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
