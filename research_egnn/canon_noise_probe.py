"""Does a *better* canonicalizer escape the degeneracy problem? (Reviewer 2.1)

The objection: "you only defeat naive per-frame PCA; use SVD with sign-tie-breaking, a
learned frame predictor, or temporal smoothing, and the degeneracy argument weakens."

Two things make that objection answerable rather than fatal.

1. Our baseline ALREADY sign-disambiguates (canonicalize.pca_canonicalize orients each
   axis by its dominant joint projection and forces right-handedness). Sign fixing removes
   a DISCRETE ambiguity. Covariance degeneracy is a CONTINUOUS one -- when two eigenvalues
   coincide the eigenvectors are undetermined within a plane -- so no sign rule touches it.

2. Under a *clean* rotation PCA canonicalization is exactly equivariant: rotating x sends
   the covariance to R C R^T, whose eigenvectors are exactly R V. Our own probe measures
   3.7e-11 over 32 random SO(3). So rotation alone can never exhibit the failure, which is
   why every test we have run so far shows nothing. THIS IS THE GAP IN OUR EVIDENCE.

The failure mode is CONDITIONING, not exactness. By Davis-Kahan the estimated eigenvector
subspace moves by O(||E|| / gap) under a perturbation E. A clean rotation is E = 0. Sensor
noise is not. So the discriminating regime is rotation *plus noise*, and the prediction is
a 1/gap amplification curve that no sign rule and no learned point-estimate frame escapes.

This probe is model-free -- it measures the frame, not a score -- so it applies to every
canonicalizer variant with no retraining.

Protocol (physically the right one): sensor noise lives in the CAMERA frame. A camera at
identity observes x + eps; a camera rotated by R observes Rx + eps with the same sensor
noise, NOT R(x + eps). We therefore compare the canonical frame recovered from the two
views and report the residual rotation between them. An exactly invariant read-out has no
frame to recover and is unaffected; a frame estimator's residual scales like 1/gap.

Variants tested:
  argmax    -- the shipped baseline (dominant-joint sign rule)
  sumproj   -- a more stable sign rule (sign of the summed projection)
  temporal  -- Procrustes-align each frame's axes to the previous frame (smoothing)

Run:  python research_egnn/canon_noise_probe.py [--sigmas 0.005 0.01 0.02] [--n-seq 60]
Out:  research_egnn/outputs/canon_noise_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))
sys.path.insert(0, _HERE)

import kimore_cde_data as kd                    # noqa: E402  (read-only)

OUT = os.path.join(_HERE, "outputs")
assert os.path.abspath(OUT).replace("\\", "/").endswith("research_egnn/outputs"), \
    f"refusing to run: output dir {OUT} is not the sandbox"
os.makedirs(OUT, exist_ok=True)


# =============================================================================
# Canonicalizer variants. All share the eigendecomposition; they differ only in
# how the residual sign / in-plane ambiguity is resolved.
# =============================================================================
def _eig(p):
    """Centred coords, principal axes (descending), eigenvalues (descending)."""
    q = p - p.mean(0, keepdims=True)
    w, V = np.linalg.eigh(q.T @ q)
    order = np.argsort(w)[::-1]
    return q, V[:, order].copy(), w[order]


def _right_handed(V):
    if np.linalg.det(V) < 0:
        V[:, 2] = -V[:, 2]
    return V


def frame_argmax(p, prev=None):
    """The shipped baseline: sign by the dominant joint projection."""
    q, V, w = _eig(p)
    for a in range(3):
        proj = q @ V[:, a]
        if proj[np.argmax(np.abs(proj))] < 0:
            V[:, a] = -V[:, a]
    return _right_handed(V), w


def _skew_sign(V, q):
    """Third-moment (skewness) sign rule -- the standard deterministic disambiguation.

    NOTE: the summed *first* moment cannot be used. q is mean-centred, so
    sum_i (q_i . v) = (sum_i q_i) . v = 0 identically, and its sign is pure numerical
    noise. The third moment is generically non-zero and is a smooth function of the
    cloud, so it is far better conditioned than the dominant-joint rule, whose selected
    joint can change between frames and flip an axis on its own.
    """
    for a in range(3):
        proj = q @ V[:, a]
        s = float(np.sum(proj ** 3))
        if s < 0:
            V[:, a] = -V[:, a]
    return V


def frame_skew(p, prev=None):
    """A better-conditioned sign rule than the shipped baseline (third moment)."""
    q, V, w = _eig(p)
    return _right_handed(_skew_sign(V, q)), w


def frame_temporal(p, prev=None):
    """Temporal smoothing: pick the axis signs closest to the previous frame's axes.

    This is the reviewer's third suggestion. Note what it costs: the frame becomes a
    function of history, so the map is no longer equivariant frame-by-frame -- the
    invariance becomes lagged and approximate rather than exact.
    """
    q, V, w = _eig(p)
    if prev is not None:
        for a in range(3):
            if float(V[:, a] @ prev[:, a]) < 0:
                V[:, a] = -V[:, a]
    else:
        V = _skew_sign(V, q)
    return _right_handed(V), w


VARIANTS = {"argmax": frame_argmax, "skew": frame_skew, "temporal": frame_temporal}


def rel_gap(w):
    w = np.maximum(w, 0.0)
    denom = w[0] if w[0] > 0 else 1.0
    return float(min((w[0] - w[1]) / denom, (w[1] - w[2]) / denom))


def frame_angle(Va, Vb):
    """Geodesic angle (deg) between two rotation matrices."""
    c = (np.trace(Va.T @ Vb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rand_so3(rng):
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# =============================================================================
# The probe
# =============================================================================
def run(samples, sigmas, rng, variant_names):
    """Residual frame rotation between two camera poses of the same noisy sensor."""
    res = {}
    for vname in variant_names:
        fn = VARIANTS[vname]
        rows = {s: {"gaps": [], "resid_deg": []} for s in sigmas}
        # control: no noise at all -> must be ~0 (exact equivariance of the estimator)
        ctrl = []
        for s in samples:
            x = np.asarray(s["x"], dtype=np.float64)
            # body scale: RMS radius of the first frame, so sigma is scale-free
            q0 = x[0] - x[0].mean(0, keepdims=True)
            scale = float(np.sqrt((q0 ** 2).sum(1).mean()))
            R = rand_so3(rng)
            # The temporal variant is stateful, so every chain needs its OWN history:
            # the two clean camera poses, and both poses at each noise level.
            prev_a = prev_b = None
            prev_na = {sg: None for sg in sigmas}
            prev_nb = {sg: None for sg in sigmas}
            for f in range(x.shape[0]):
                p = x[f]
                # --- control: clean, two camera poses ---
                Va, wa = fn(p, prev_a)
                Vb, _ = fn(p @ R.T, prev_b)
                prev_a, prev_b = Va, Vb
                # frames should satisfy Vb = R Va exactly; residual measures departure
                ctrl.append(frame_angle(R @ Va, Vb))
                g = rel_gap(wa)
                # --- test: same sensor noise, in the CAMERA frame, at both poses ---
                for sg in sigmas:
                    eps = rng.normal(scale=sg * scale, size=p.shape)
                    Vn_a, _ = fn(p + eps, prev_na[sg])
                    Vn_b, _ = fn(p @ R.T + eps, prev_nb[sg])
                    prev_na[sg], prev_nb[sg] = Vn_a, Vn_b
                    rows[sg]["gaps"].append(g)
                    rows[sg]["resid_deg"].append(frame_angle(R @ Vn_a, Vn_b))
        out = {"control_clean_resid_deg": {
            "median": float(np.median(ctrl)), "p95": float(np.percentile(ctrl, 95)),
            "max": float(np.max(ctrl))}}
        for sg in sigmas:
            g = np.asarray(rows[sg]["gaps"])
            d = np.asarray(rows[sg]["resid_deg"])
            bins = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, 1.01)]
            per_bin = {}
            for lo, hi in bins:
                m = (g >= lo) & (g < hi)
                if m.sum() == 0:
                    continue
                per_bin[f"gap[{lo},{hi})"] = {
                    "n": int(m.sum()),
                    "median_deg": float(np.median(d[m])),
                    "p95_deg": float(np.percentile(d[m], 95)),
                }
            out[f"sigma={sg}"] = {
                "overall_median_deg": float(np.median(d)),
                "overall_p95_deg": float(np.percentile(d, 95)),
                "frac_resid_gt_10deg": float((d > 10).mean()),
                "frac_resid_gt_45deg": float((d > 45).mean()),
                "by_eigengap": per_bin,
            }
        res[vname] = out
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    ap.add_argument("--n-seq", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()

    S = kd.load_all_exercises(max_len=150, verbose=False)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(S))[: args.n_seq]
    samples = [S[i] for i in idx]
    print(f"{len(samples)} sequences, sigmas={args.sigmas} (fraction of body RMS radius)")

    res = run(samples, args.sigmas, rng, args.variants)

    for v, r in res.items():
        c = r["control_clean_resid_deg"]
        print(f"\n=== {v} ===")
        print(f"  clean control (must be ~0): median {c['median']:.2e} deg, max {c['max']:.2e}")
        for sg in args.sigmas:
            k = f"sigma={sg}"
            print(f"  sigma={sg}: median {r[k]['overall_median_deg']:7.3f} deg | "
                  f">10deg {100*r[k]['frac_resid_gt_10deg']:5.1f}% | "
                  f">45deg {100*r[k]['frac_resid_gt_45deg']:5.1f}%")
            for b, s in r[k]["by_eigengap"].items():
                print(f"      {b:18s} n={s['n']:7d}  median {s['median_deg']:7.3f}  p95 {s['p95_deg']:8.3f}")

    path = os.path.join(OUT, "canon_noise_probe.json")
    with open(path, "w") as fh:
        json.dump({"sigmas": args.sigmas, "n_sequences": len(samples),
                   "note": "resid_deg = geodesic angle between the canonical frame recovered "
                           "at two camera poses under identical camera-frame sensor noise; "
                           "the clean control isolates estimator equivariance (must be ~0).",
                   "results": res}, fh, indent=2)
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
