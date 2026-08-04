#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
ntu_paired_camera.py
====================
Two measurements on NTU RGB+D 60, both eval-only (no training, no retraining):

  R1  ROTATION CONTROL -- the same clip, rotated by a Haar-random SO(3) element. This is the
      theorem's own claim: rotate the skeleton, the read-out must not move. Predicted ~0 for the
      EGRU at machine precision; large for a non-equivariant baseline whose preprocessing does not
      canonicalise orientation, and SMALL BUT NON-ZERO for one whose preprocessing does. That last
      residual is the difference between an exact guarantee and a hand-crafted approximation.

  P1  PAIRED CAMERA -- the same physical performance, recorded SIMULTANEOUSLY by three cameras.
      NTU tags are S{setup}C{camera}P{subject}R{rep}A{action}, so every (setup, subject, rep,
      action) present at C001/C002/C003 is one performance seen from three real, physically
      different viewpoints. 18,674 such triples exist. This is "same patient, same score, different
      room" measured on real hardware, at scale, on a public benchmark.

WHY BOTH, IN ONE ARTIFACT. R1 and P1 answer different questions and the contrast IS the result.
R1 isolates the geometric part of a viewpoint change, which the theorem removes exactly. P1 is a
real camera relocation, where the three clips are three *different skeleton estimates* of one event
-- different tracker noise, occlusion and range. No property of the read-out can undo that, so P1's
residual is necessarily larger than R1's. Reporting R1 alone would overstate; reporting P1 alone
would understate. Together they say exactly how much of a real viewpoint change the theorem buys.

WHAT P1 IS NOT. Under the X-View split cameras 2 and 3 are TRAINING cameras and camera 1 is the
test camera, so a C2-C3 pair is train-vs-train and a C1-C2 pair is test-vs-train. Agreement rates
are therefore reported per camera pair and never pooled, and the contamination is stated in the
output. A fully uncontaminated version needs an X-Sub-trained checkpoint (all three cameras appear
on both sides of a subject split); this file takes `--split xsub` for that once one exists.

Each model is evaluated under ITS OWN training preprocessing, because that is what its weights
expect: the EGRU under causal sampling with no view-normalisation, the ST-GCN under uniform
sampling with PreNormalize3D. Mixing them would measure a train/test mismatch, not a viewpoint.

Usage:
    python src/ntu_paired_camera.py --limit 2000
    python src/ntu_paired_camera.py --limit 0            # all 18,674 triples
"""

import argparse
import collections
import hashlib
import json
import os
import re
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import ntu_dataset as nd                       # noqa: E402
from baselines import build_model, count_parameters   # noqa: E402

TAG_RE = re.compile(r"S(\d+)C(\d+)P(\d+)R(\d+)A(\d+)")
CAMERAS = (1, 2, 3)

# LIVENESS. Kinect writes an untracked joint as exactly (0,0,0). After root subtraction every such
# joint lands on the SAME point (-root), so two simultaneously untracked joints produce a
# zero-length relative vector whose direction is undefined -- and an undefined direction is resolved
# arbitrarily rather than equivariantly, which breaks the read-out's invariance. Measured on random
# weights in fp64: k=0 and k=1 untracked joints stay exact (7e-17); k>=2 breaks structurally
# (1e-3, with an fp32/fp64 ratio of 1.0, i.e. not roundoff). Supplying the liveness mask the model
# already accepts restores exactness at every k (~1e-16, ratio ~1e9). The X-View evaluation path
# passes mask=None, so on real data it never supplied it.
XVIEW_ROLE = {1: "test (unseen)", 2: "train (seen)", 3: "train (seen)"}

# Each arm as it was actually trained. Changing any of these silently re-measures something else,
# so they are declared here rather than passed in.
ARMS = {
    "egru": dict(
        ckpt="outputs/ntu_fault/egru_xview_mask.pt", model="egru",
        sample="causal", view_norm=False, use_mask=True,
        note="ours; per-frame root-relative, no view canonicalisation"),
    "stgcn_full": dict(
        ckpt="outputs/ntu_stgcn_std/stgcn_full_xview.pt", model="stgcn_full",
        sample="uniform", view_norm=True, use_mask=False,
        note="published ST-GCN recipe; PreNormalize3D view canonicalisation"),
}


# =============================================================================
# provenance
# =============================================================================
def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


# =============================================================================
# triples
# =============================================================================
def build_triples(items, limit, seed=0):
    """-> list of {key, label, by_cam:{cam: item}} for performances present at all three cameras."""
    groups = collections.defaultdict(dict)
    for it in items:
        m = TAG_RE.search(it["tag"])
        if not m:
            continue
        s, c, p, r, a = (int(x) for x in m.groups())
        groups[(s, p, r, a)][c] = it

    triples = []
    for key, by_cam in groups.items():
        if not all(c in by_cam for c in CAMERAS):
            continue
        labels = {by_cam[c]["label"] for c in CAMERAS}
        if len(labels) != 1:                       # a disagreeing label would corrupt the accuracy
            continue                               # column; drop rather than pick one
        triples.append({"key": key, "label": labels.pop(), "by_cam": by_cam})

    triples.sort(key=lambda t: t["key"])           # deterministic order before any sampling
    if limit and limit < len(triples):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(triples), size=limit, replace=False)
        triples = [triples[i] for i in sorted(idx)]
    return triples


def haar_rotation(rng):
    """A Haar-uniform element of SO(3): QR of a Gaussian, sign-fixed, det forced to +1."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))                    # remove QR's sign ambiguity -> Haar on O(3)
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]                         # ...and restrict to the proper rotations
    return q


# =============================================================================
# scoring
# =============================================================================
def liveness_mask(kp, cfg, dtype=np.float32):
    """(M,T,V) liveness for one clip, mirroring preprocess's body/frame selection exactly.

    A joint is dead in a frame iff its RAW coordinates are exactly zero (Kinect's untracked
    sentinel), read BEFORE root subtraction -- afterwards a dead joint sits at -root and is no
    longer distinguishable from a live joint that happens to be there.
    `dtype` must match the one handed to `nd.preprocess`: the body-selection argsort below ranks
    bodies by a motion norm, and a norm computed at a different precision can order a near-tie
    differently -- which would silently pair a mask with the wrong body.
    """
    t_target = cfg["t_target"]
    kp = np.asarray(kp, dtype=dtype)
    if kp.ndim == 3:
        kp = kp[None]
    M = kp.shape[0]
    if M > nd.MAX_BODIES:
        motion = [np.linalg.norm(np.diff(kp[m], axis=0)).sum() for m in range(M)]
        kp = kp[np.argsort(motion)[::-1][:nd.MAX_BODIES]]
    elif M < nd.MAX_BODIES:
        kp = np.concatenate([kp, np.zeros((nd.MAX_BODIES - M,) + kp.shape[1:], kp.dtype)], axis=0)

    Mb, T, V, _ = kp.shape
    out = np.ones((Mb, t_target, V), dtype=dtype)
    for m in range(Mb):
        body = kp[m]
        if not np.any(body):
            continue
        valid = np.where(np.any(body.reshape(T, -1) != 0, axis=1))[0]
        body = body[: valid[-1] + 1] if len(valid) else body[:0]
        L = len(body)
        if L == 0:
            continue
        if cfg["sample"] == "cyclic":
            continue                                   # not used by either arm here
        if L >= t_target:
            idx = (np.arange(t_target) if cfg["sample"] == "causal"
                   else np.linspace(0, L - 1, t_target).round().astype(int))
            body, n = body[idx], t_target
        else:
            n = L
        out[m, :n] = (np.abs(body).sum(-1) > 0).astype(np.float32)
    return out


@torch.no_grad()
def score(model, kps, cfg, device, batch_size=32, dtype=torch.float32, liveness=True):
    """Raw keypoint arrays -> logits (N, C), using `cfg`'s training-matched preprocessing."""
    out = []
    np_dt = np.float64 if dtype == torch.float64 else np.float32
    use_mask = liveness and bool(getattr(model, "uses_mask", False))
    for i in range(0, len(kps), batch_size):
        chunk = kps[i:i + batch_size]
        # dtype goes INTO preprocess, not onto its result: root subtraction is where an fp32 path
        # would round away the drift this probe exists to measure.
        pre = [nd.preprocess(k, cfg["t_target"], cfg["sample"], False,
                             view_norm=cfg["view_norm"], train=False, dtype=np_dt) for k in chunk]
        x = torch.from_numpy(np.stack([p[0] for p in pre]).astype(np_dt)).to(device)
        length = torch.from_numpy(np.stack([p[1] for p in pre])).to(device)
        present = torch.from_numpy(np.stack([p[2] for p in pre]).astype(np_dt)).to(device)
        mask = None
        if use_mask:
            mask = torch.from_numpy(
                np.stack([liveness_mask(k, cfg, np_dt) for k in chunk]).astype(np_dt)).to(device)
        out.append(model(x, length=length, present=present, mask=mask).double().cpu())
    return torch.cat(out).numpy()


def load_arm(name, device, t_target, dtype=torch.float32):
    """Build and load one arm.

    dtype is set as torch's DEFAULT before construction, not applied afterwards: e3nn materialises
    its Clebsch-Gordan coefficients in the default dtype at CONSTRUCTION time, so a `.double()`
    after the fact yields a model whose weights are fp64 but whose coupling constants are still
    fp32 -- a fake fp64 certificate. `_g4_selftest` in run_xview_evaluation.py takes the same care.

    THAT IS NOT SUFFICIENT ON ITS OWN, and this is the subtle half. e3nn registers the Wigner-3j
    coupling tensors as PERSISTENT buffers inside the compiled tensor product
    (`tps.N._compiled_main_left_right._w3j_*`), so they are written into every checkpoint saved from
    an fp32 training run. Constructing under an fp64 default builds them correctly and then
    `load_state_dict` overwrites them with the fp32-rounded copies -- re-poisoning the model after
    the guard has done its job. Measured on this checkpoint: keeping them costs 6.106e-09 scalar
    equivariance error, dropping them gives 3.331e-16, an improvement of 7 orders of magnitude.
    Loading them is never right at any dtype: they are structural constants fixed by the irreps, so
    the freshly built tensor is exact by construction while the checkpoint's copy carries whatever
    precision the training run happened to use.
    """
    cfg = dict(ARMS[name])
    cfg["t_target"] = t_target
    path = os.path.join(_ROOT, cfg["ckpt"])
    assert os.path.isfile(path), f"missing checkpoint {path}"
    ck = torch.load(path, map_location=device, weights_only=False)
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = build_model(cfg["model"], n_classes=60,
                            use_mask=ck.get("uses_mask", cfg["use_mask"]))
        state = {k: v for k, v in ck["state"].items() if "_w3j" not in k}
        dropped = len(ck["state"]) - len(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Only the structural constants may be absent, and nothing may be unexpected -- otherwise a
        # real weight failed to load and the arm is silently not the trained model.
        assert not unexpected, f"unexpected keys in {name} checkpoint: {list(unexpected)}"
        assert all("_w3j" in k for k in missing), \
            f"{name}: weights missing from checkpoint: {[k for k in missing if '_w3j' not in k]}"
        assert dropped == len(missing), f"{name}: dropped {dropped} w3j keys but {len(missing)} missing"
        model = model.to(device=device, dtype=dtype).eval()
    finally:
        torch.set_default_dtype(old)
    cfg["w3j_rebuilt"] = dropped                            # provenance: fp32 CG constants NOT loaded
    cfg["params"] = count_parameters(model)
    cfg["sha256"] = sha256(path)
    cfg["uses_mask"] = bool(getattr(model, "uses_mask", False))
    cfg["dtype"] = str(dtype).replace("torch.", "")
    return model, cfg


# =============================================================================
# metrics
# =============================================================================
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def drift_stats(a, b):
    """Two logit matrices for the SAME instances -> agreement and drift.

    `agreement` is scale-free and comparable across models. `logit_linf` is not comparable between
    models (their logit scales differ) but is the quantity that must be ~0 for an exactly invariant
    read-out, so it is the diagnostic that certifies R1.
    """
    pa, pb = a.argmax(1), b.argmax(1)
    dl = np.abs(a - b).max(axis=1)
    dp = np.abs(softmax(a) - softmax(b)).max(axis=1)
    return {
        "agreement": float((pa == pb).mean()),
        "logit_linf_mean": float(dl.mean()),
        "logit_linf_p95": float(np.percentile(dl, 95)),
        "logit_linf_max": float(dl.max()),
        "prob_linf_mean": float(dp.mean()),
        "prob_linf_max": float(dp.max()),
    }


def main():
    ap = argparse.ArgumentParser(description="NTU paired-camera + rotation-control probe.")
    ap.add_argument("--data", default=os.path.join(_ROOT, "data", "ntu60_3danno.pkl"))
    ap.add_argument("--limit", type=int, default=2000,
                    help="number of triples to score (0 = all)")
    ap.add_argument("--t-target", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rot-camera", type=int, default=1,
                    help="camera whose clips carry the R1 rotation control")
    ap.add_argument("--liveness", choices=["on", "off"], default="on",
                    help="supply the liveness mask derived from Kinect's untracked-joint sentinel; "
                         "'off' reproduces the X-View eval path, which passes mask=None")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="float64 runs the precision-scaling diagnostic: an exactly invariant "
                         "read-out must drop by orders of magnitude, a broken one must not")
    ap.add_argument("--out", default=os.path.join(_ROOT, "outputs", "ntu_paired", "paired_camera.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    live = args.liveness == "on"
    print(f"[paired] loading {args.data}  (dtype={args.dtype}, liveness={args.liveness})")
    items = nd.load_items(args.data)
    triples = build_triples(items, args.limit, args.seed)
    print(f"[paired] {len(triples)} triples (same performance at C001/C002/C003), device={device}")

    labels = np.array([t["label"] for t in triples])
    rng = np.random.default_rng(args.seed)
    rots = [haar_rotation(rng) for _ in range(len(triples))]

    results = {}
    for name in ARMS:
        model, cfg = load_arm(name, device, args.t_target, dtype)
        print(f"\n[{name}] {cfg['params']:,} params  sample={cfg['sample']}  "
              f"view_norm={cfg['view_norm']}  uses_mask={cfg['uses_mask']}")

        logits, acc = {}, {}
        for cam in CAMERAS:
            kps = [t["by_cam"][cam]["keypoint"] for t in triples]
            logits[cam] = score(model, kps, cfg, device, args.batch_size, dtype, live)
            acc[cam] = float((logits[cam].argmax(1) == labels).mean() * 100)
            print(f"  C{cam:03d}  top-1 {acc[cam]:5.2f}%   [{XVIEW_ROLE[cam]}]")

        # --- P1: paired camera ---
        pairs = {}
        for i, ci in enumerate(CAMERAS):
            for cj in CAMERAS[i + 1:]:
                pairs[f"C{ci}-C{cj}"] = drift_stats(logits[ci], logits[cj])
        agree3 = float(np.mean([len({logits[c].argmax(1)[k] for c in CAMERAS}) == 1
                                for k in range(len(triples))]))

        # --- R1: rotation control, same clip and same camera ---
        cam = args.rot_camera
        kps_rot = [np.einsum("ab,mtvb->mtva", R, t["by_cam"][cam]["keypoint"].astype(np.float64))
                   for t, R in zip(triples, rots)]
        rot_logits = score(model, kps_rot, cfg, device, args.batch_size, dtype, live)
        rot = drift_stats(logits[cam], rot_logits)
        rot["top1_rotated"] = float((rot_logits.argmax(1) == labels).mean() * 100)

        print(f"  R1 rotation control (C{cam:03d}, Haar SO(3)):")
        print(f"     agreement {rot['agreement']*100:6.2f}%   "
              f"top-1 {acc[cam]:.2f} -> {rot['top1_rotated']:.2f}   "
              f"max|dlogit| {rot['logit_linf_max']:.3e}")
        print(f"  P1 paired camera:")
        for k, v in pairs.items():
            print(f"     {k}  agreement {v['agreement']*100:6.2f}%   "
                  f"max|dprob| {v['prob_linf_max']:.3f}")
        print(f"     all three agree: {agree3*100:.2f}%")

        results[name] = {"config": cfg, "top1_per_camera": acc,
                         "rotation_control": rot, "paired_camera": pairs,
                         "agreement_all_three": agree3}
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "what": "NTU RGB+D 60: Haar-SO(3) rotation control (R1) and paired multi-camera "
                "consistency (P1), eval-only on the X-View checkpoints.",
        "scope": {
            "n_triples": len(triples),
            "triple": "one performance (setup, subject, rep, action) recorded simultaneously at "
                      "C001/C002/C003",
            "contamination": "X-View trains on cameras 2-3 and tests on camera 1, so C2-C3 is "
                             "train-vs-train and C1-C2 / C1-C3 are test-vs-train. Pairs are "
                             "reported separately and never pooled.",
            "reports": "prediction agreement (scale-free, comparable across models) and logit/prob "
                       "drift (within-model diagnostic)",
            "camera_roles": {f"C{c}": XVIEW_ROLE[c] for c in CAMERAS},
        },
        "config": {"t_target": args.t_target, "seed": args.seed, "dtype": args.dtype,
                   "liveness": args.liveness,
                   "rot_camera": args.rot_camera, "data": os.path.basename(args.data)},
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[paired] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
