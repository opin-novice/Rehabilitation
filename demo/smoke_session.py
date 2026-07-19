#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
smoke_session.py
================
Headless gate for the session demo (no camera). Fabricates a moving exercise via the same
synthetic MediaPipe buffer the invariance smoke test uses, runs the full report pipeline, and
asserts everything is well-formed and in-range, then renders a real PNG.

    synthetic MediaPipe frames -> mediapipe_to_kinect25 -> feedback.compute_report
                               -> report_card.render_report -> demo/reports/*.png

Run:  python demo/smoke_session.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mp_to_kinect import mediapipe_to_kinect25                       # noqa: E402
from smoke_test import synthetic_mediapipe_buffer                    # noqa: E402
import feedback as fb                                                # noqa: E402
import report_card as rc                                             # noqa: E402


def build_session(seed=0, T=90, fps=30.0):
    """Raw (T,25,3) Kinect metres + (T,) seconds from the synthetic mover."""
    mp_buf, t = synthetic_mediapipe_buffer(T=T, fps=fps, seed=seed)
    kin = np.stack([mediapipe_to_kinect25(f) for f in mp_buf], axis=0)
    return kin, t


def _move(kin, joints, disp):
    """Return a copy of kin with `disp` (T,) metres added on X to the given joints."""
    k = kin.copy()
    for j in joints:
        k[:, j, 0] += disp
    return k


def check_responsiveness(kin, t):
    """The ROM metric must actually LIGHT UP when the exercise's target body part moves --
    otherwise the biomechanics are decorative. Inject a known motion and require a high ROM%."""
    phase = np.linspace(0, 2 * np.pi, len(t))

    # Ex1 lateral flexion: tilt the whole upper body sideways -> big trunk-angle excursion
    upper = [fb.SPINE_MID, fb.NECK, fb.HEAD, fb.SPINE_SHOULDER, fb.SHO_L, fb.SHO_R,
             fb.ELB_L, fb.ELB_R, fb.WRI_L, fb.WRI_R]
    r1 = fb.compute_report(_move(kin, upper, 0.25 * np.sin(phase)), t, 1)
    assert r1["metrics"][0]["pct"] >= 80, f"lateral-flexion ROM didn't respond: {r1['metrics'][0]}"

    # Ex4 hip abduction: swing ONLY the right leg out -> high ROM + LOW symmetry (one-sided)
    r4 = fb.compute_report(_move(kin, [fb.KNEE_R, fb.ANK_R], 0.30 * np.abs(np.sin(phase))), t, 4)
    assert r4["metrics"][0]["pct"] >= 70, f"abduction ROM didn't respond: {r4['metrics'][0]}"
    assert r4["metrics"][2]["pct"] <= 40, f"one-sided motion should read asymmetric: {r4['metrics'][2]}"
    print(f"[PASS] biomechanics respond: lateral-flexion ROM {r1['metrics'][0]['pct']:.0f}%, "
          f"abduction ROM {r4['metrics'][0]['pct']:.0f}% with symmetry {r4['metrics'][2]['pct']:.0f}% "
          f"(correctly flags one-sided motion)")


def check_report(rep, ex_id):
    assert rep["exercise_id"] == ex_id
    assert 0.0 <= rep["composite"] <= 100.0, rep["composite"]
    assert 1 <= rep["stars"] <= 5, rep["stars"]
    assert len(rep["metrics"]) == 5
    for m in rep["metrics"]:
        assert 0.0 <= m["pct"] <= 100.0, (m["label"], m["pct"])
    assert rep["strengths"] and rep["improvements"], "must always give feedback both ways"
    assert set(rep["strengths"]).isdisjoint(rep["improvements"]), "no line in both columns"
    if rep["ai_badge"] is not None:
        assert 0.0 <= rep["ai_badge"] <= 50.0, rep["ai_badge"]


def main():
    kin, t = build_session()
    print(f"[session] synthetic session: {len(t)} frames, {t[-1]-t[0]:.1f}s")

    print("\n exercise | composite | stars | reps | metrics (rom/smooth/sym/tempo/consist)")
    print("-" * 78)
    last_rep = None
    for ex_id in (1, 2, 3, 4, 5):
        rep = fb.compute_report(kin, t, ex_id, ai_score_raw=11.6)   # 11.6 = the observed webcam raw
        check_report(rep, ex_id)
        pcts = "/".join(f"{m['pct']:.0f}" for m in rep["metrics"])
        print(f" {ex_id} {fb.EXERCISE_NAMES[ex_id][:14]:<14} | "
              f"{rep['composite']:>6.1f}   | {rep['stars']}     | {rep['reps']:>3}  | {pcts}")
        last_rep = rep

    # render one card to prove the renderer + PNG write path
    check_responsiveness(kin, t)

    bgr, png = rc.render_report(last_rep, save=True)
    assert bgr.shape == (rc.H, rc.W, 3), bgr.shape
    assert png and os.path.exists(png), png
    print(f"[PASS] all 5 exercise reports well-formed and in-range.")
    print(f"[PASS] report card rendered: {png}  ({bgr.shape[1]}x{bgr.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
