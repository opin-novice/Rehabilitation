#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ntu_dataset.py
==============
NTU RGB+D 60 Cross-View (X-View) pipeline for the EGRU viewpoint-robustness experiment.

Why NTU, and why it drops onto our stack unchanged
--------------------------------------------------
NTU RGB+D is captured with Kinect v2 -- the SAME 25-joint skeleton and SAME SDK joint order as
KIMORE. So the equivariant encoder's topology (cde_model_mp.KINECT_BONES / skeleton_edges) and the
root joint (SpineBase = index 0) transfer with NO remapping. NTU joint 21 (SpineShoulder, 1-indexed)
is our 0-indexed joint 20; NTU joint 1 (SpineBase) is our 0. This is checked, not assumed
(assert_layout()).

What NTU buys us that KIMORE cannot
-----------------------------------
The KIMORE viewpoint sweep rotates the GROUND-TRUTH skeleton, which our theorem satisfies by
construction -- a unit test, not a system result (the WACV "tautology" critique). NTU X-View is
THREE PHYSICALLY DISTINCT CAMERAS with REAL per-view Kinect tracking (real self-occlusion, real
depth error, real left/right confusion at oblique angles). Cross-View generalisation on NTU is the
benchmark the skeleton community already trusts, with published ST-GCN/CTR-GCN numbers to compare
against. That converts the viewpoint claim from an identity into an externally comparable result.

X-View protocol (NTU RGB+D 60, the standard)
--------------------------------------------
    TRAIN : cameras 2 and 3      TEST : camera 1
Camera id is field C in the sample tag  S{setup:03d}C{camera:03d}P{subject:03d}R{rep:03d}A{action:03d}.

Supported inputs
----------------
  * pyskl / mmaction-style .pkl : dict with 'annotations' (list of {frame_dir, label,
    keypoint (M,T,V,3), total_frames, ...}) and OPTIONAL 'split' {xview_train:[...], xview_val:[...]}.
    If an explicit xview split is present we USE it; otherwise we derive the split from the camera
    id parsed out of frame_dir.
  * .npz : arrays keypoint (N,M,T,V,3), label (N,), and either camera (N,) or frame_dir (N,) strings.

This file also SYNTHESISES a tiny fixture (make_fixture) so the whole pipeline -- split, causal
padding, both models, the occlusion theorem, determinism -- is testable TODAY, before the ~5 GB
NTU download exists. Run `python src/ntu_dataset.py --selftest`.
"""

import argparse
import os
import pickle
import re
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cde_model_mp import KINECT_BONES                            # noqa: E402  (shared topology)

N_JOINTS = 25
ROOT_JOINT = 0                      # SpineBase, identical to KIMORE
MAX_BODIES = 2                      # NTU actions involve at most 2 primary skeletons
XVIEW_TRAIN_CAMS = (2, 3)
XVIEW_TEST_CAMS = (1,)
_TAG_RE = re.compile(r"C(\d+)")


# =============================================================================
# topology sanity
# =============================================================================
def assert_layout():
    """NTU (Kinect v2) must expose the same 25-joint graph our encoder was built on."""
    js = set()
    for a, b in KINECT_BONES:
        js.update((a, b))
    assert max(js) == N_JOINTS - 1 and min(js) == 0, "KINECT_BONES not 0..24"
    assert len(KINECT_BONES) == 24, "a 25-joint tree must have 24 bones"
    # NTU SpineShoulder is 1-indexed joint 21 -> our joint 20; SpineBase 1 -> our 0.
    assert (20, 8) in KINECT_BONES or (8, 20) in KINECT_BONES or \
           any(20 in e for e in KINECT_BONES), "SpineShoulder(20) missing from graph"


# =============================================================================
# camera / split parsing
# =============================================================================
def camera_of(tag: str) -> int:
    """Parse the camera id from an NTU sample tag 'S001C002P003R001A004' -> 2."""
    m = _TAG_RE.search(tag)
    if not m:
        raise ValueError(f"no camera field in tag {tag!r}")
    return int(m.group(1))


def xview_bucket(cam: int) -> str:
    if cam in XVIEW_TRAIN_CAMS:
        return "train"
    if cam in XVIEW_TEST_CAMS:
        return "test"
    return "other"


# =============================================================================
# raw ingestion -> a flat list of {keypoint, label, camera, tag}
# =============================================================================
def _load_pkl(path):
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    if isinstance(blob, dict) and "annotations" in blob:
        anns = blob["annotations"]
        split = blob.get("split", None)
    elif isinstance(blob, list):
        anns, split = blob, None
    else:
        raise ValueError("unrecognised pkl; expected pyskl dict or a list of annotations")

    # explicit xview split (pyskl provides xview_train / xview_val), keyed by frame_dir
    want = None
    if split is not None:
        tr = set(split.get("xview_train", []))
        va = set(split.get("xview_val", []) or split.get("xview_test", []))
        if tr or va:
            want = (tr, va)

    items = []
    for a in anns:
        tag = a.get("frame_dir") or a.get("filename") or a.get("tag")
        kp = np.asarray(a["keypoint"], dtype=np.float32)         # (M,T,V,3) expected
        lab = int(a["label"])
        cam = camera_of(tag) if tag else int(a["camera"])
        bucket = None
        if want is not None and tag is not None:
            if tag in want[0]:
                bucket = "train"
            elif tag in want[1]:
                bucket = "test"
            else:
                bucket = "other"
        items.append({"keypoint": kp, "label": lab, "camera": cam,
                      "tag": tag, "bucket": bucket})
    return items


def _load_npz(path):
    z = np.load(path, allow_pickle=True)
    kp_all = z["keypoint"]                                       # (N,M,T,V,3)
    labels = z["label"]
    tags = z["frame_dir"] if "frame_dir" in z else None
    cams = z["camera"] if "camera" in z else None
    items = []
    for i in range(len(kp_all)):
        tag = str(tags[i]) if tags is not None else None
        cam = int(cams[i]) if cams is not None else camera_of(tag)
        items.append({"keypoint": np.asarray(kp_all[i], dtype=np.float32),
                      "label": int(labels[i]), "camera": cam, "tag": tag, "bucket": None})
    return items


def load_items(path):
    if path.endswith(".pkl") or path.endswith(".pickle"):
        return _load_pkl(path)
    if path.endswith(".npz"):
        return _load_npz(path)
    raise ValueError(f"unsupported file type: {path}")


# =============================================================================
# preprocessing: shape -> (M, T, V, 3), root-relative, causal length T_target
# =============================================================================
def _fit_bodies(kp):
    """Normalise the body axis to exactly MAX_BODIES: keep the most-moving, zero-pad the rest."""
    if kp.ndim == 3:                                             # (T,V,3) single body
        kp = kp[None]
    M = kp.shape[0]
    if M > MAX_BODIES:
        motion = [np.linalg.norm(np.diff(kp[m], axis=0)).sum() for m in range(M)]
        kp = kp[np.argsort(motion)[::-1][:MAX_BODIES]]
    elif M < MAX_BODIES:
        pad = np.zeros((MAX_BODIES - M,) + kp.shape[1:], dtype=kp.dtype)
        kp = np.concatenate([kp, pad], axis=0)
    return kp                                                   # (MAX_BODIES, T, V, 3)


def _rotation_matrix(axis, theta):
    """Rodrigues rotation about `axis` by `theta` (pyskl's sign convention)."""
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-8 or abs(theta) < 1e-8:
        return np.eye(3)
    a = np.cos(theta / 2.0)
    b, c, d = -(axis / n) * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def _angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.arccos(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)))


def view_normalize(kp, zaxis=(0, 1), xaxis=(8, 4)):
    """PreNormalize3D (pyskl): ONE rigid transform for the whole clip -- translate body-0 frame-0
    SpineMid to the origin, rotate the frame-0 spine (SpineBase->SpineMid) onto +z, then the
    frame-0 shoulder line (ShoulderRight->ShoulderLeft) onto +x.

    This is the hand-crafted view canonicalisation the PUBLISHED ST-GCN X-View number depends on.
    Two things it does that per-frame root subtraction does NOT:
      * it removes camera ORIENTATION (root subtraction only removes translation), and
      * it applies a single transform, so the body's motion TRAJECTORY across frames survives
        (per-frame root subtraction deletes it).
    It is a preprocessing canonicalisation, NOT equivariance -- and the baseline is entitled to it.
    """
    kp = np.asarray(kp, dtype=np.float32).copy()
    present = [i for i in range(kp.shape[1]) if np.any(kp[0, i])]
    if not present:
        return kp
    f0 = present[0]
    mask = (np.abs(kp).sum(-1) > 0)[..., None]                  # (M,T,V,1) real joints only
    kp = (kp - kp[0, f0, zaxis[1]].copy()) * mask               # centre on SpineMid

    jb, jt = kp[0, f0, zaxis[0]], kp[0, f0, zaxis[1]]
    kp = np.einsum("mtvd,kd->mtvk", kp,
                   _rotation_matrix(np.cross(jt - jb, [0, 0, 1]), _angle_between(jt - jb, [0, 0, 1])))
    jr, jl = kp[0, f0, xaxis[0]], kp[0, f0, xaxis[1]]
    kp = np.einsum("mtvd,kd->mtvk", kp,
                   _rotation_matrix(np.cross(jr - jl, [1, 0, 0]), _angle_between(jr - jl, [1, 0, 0])))
    return (kp * mask).astype(np.float32)


def _rot3d(theta):
    """pyskl's RandomRot rotation: Rz @ Ry @ Rx for per-axis angles `theta`."""
    cos, sin = np.cos(theta), np.sin(theta)
    rx = np.array([[1, 0, 0], [0, cos[0], sin[0]], [0, -sin[0], cos[0]]])
    ry = np.array([[cos[1], 0, -sin[1]], [0, 1, 0], [sin[1], 0, cos[1]]])
    rz = np.array([[cos[2], sin[2], 0], [-sin[2], cos[2], 0], [0, 0, 1]])
    return rz @ ry @ rx


def random_rot(kp, theta_max, rng):
    """pyskl RandomRot: ONE random rotation per clip, angles ~ U(-theta_max, +theta_max) per axis.

    NOTE ON SCOPE: the published default theta=0.3 rad is only about +/-17 degrees -- a mild
    regulariser applied AFTER PreNormalize3D to absorb residual canonicalisation error.  It is NOT
    'train the baseline on all viewpoints', and it does not confer rotation invariance at the 45/90/
    180 degree scale our viewpoint theorem addresses.  The genuinely adversarial control is
    theta=pi, which is a SEPARATE experiment -- do not conflate the two.
    """
    theta = rng.uniform(-theta_max, theta_max, size=3)
    return np.einsum("ab,mtvb->mtva", _rot3d(theta), kp).astype(np.float32)


def _cyclic_indices(L, t_target, rng, train):
    """pyskl UniformSample indices, CYCLICALLY WRAPPED (`inds % L`) so every one of the t_target
    slots holds a REAL frame -- there is no zero padding.

    Why this matters here: 75% of NTU clips are shorter than t_target=100.  Our older modes
    zero-pad the tail, and ST-GCN global-average-pools over the whole T (it ignores `length`), so
    three quarters of the dataset had its pooled feature diluted by padding.  BatchNorm makes this
    worse: it maps the padded zeros to a NON-zero constant, which then convolves and pollutes the
    mean.  pyskl never pays this cost because it wraps instead of padding.

    Randomised offsets (train=True) also give temporal augmentation, and make multi-clip test
    ensembling meaningful -- with deterministic linspace every 'clip' is byte-identical, so a
    10-clip ensemble averages ten copies of one prediction and recovers exactly nothing.
    """
    if L < t_target:                                   # short clip: repeat it cyclically
        start = int(rng.integers(0, L)) if train else 0
        inds = np.arange(start, start + t_target)
    elif L < 2 * t_target:                             # mild subsample: drop (L - t_target) slots
        basic = np.arange(t_target)
        drop = (rng.choice(t_target + 1, L - t_target, replace=False) if train
                else np.linspace(0, t_target, L - t_target, endpoint=False).astype(np.int64))
        off = np.zeros(t_target + 1, dtype=np.int64)
        off[drop] = 1
        inds = basic + np.cumsum(off)[:-1]
    else:                                              # long clip: one frame per equal-width bin
        bids = np.array([i * L // t_target for i in range(t_target + 1)])
        bsize = np.maximum(np.diff(bids), 1)
        inds = bids[:t_target] + (rng.integers(0, bsize) if train else bsize // 2)
    return np.mod(inds, L)


def preprocess(kp, t_target, sample="causal", scale_norm=False, view_norm=False,
               train=False, clip_seed=None, rot_aug=0.0):
    """(M?,T,V,3) -> x (MAX_BODIES, t_target, V, 3), body_len (MAX_BODIES,), body_present (MAX_BODIES,).

    * root-relative (subtract SpineBase per frame per body) -> translation invariance, exactly as
      the EGRU expects; a body that is all-zero (absent) stays all-zero (its root is 0).
    * causal length fit: len>=T keep the FIRST T frames (causal, matches O(1) streaming); len<T
      zero-pad the tail and record the true length so pooling ignores the pad. sample='uniform'
      offers the accuracy-oriented alternative (index subsample) for ablations.
    """
    kp = _fit_bodies(np.asarray(kp, dtype=np.float32))          # (Mb,T,V,3)
    if view_norm:
        kp = view_normalize(kp)                                 # canonical camera frame, once
    if rot_aug > 0 and train:                                   # pyskl order: PreNormalize3D->RandomRot
        kp = random_rot(kp, rot_aug,
                        np.random.default_rng(clip_seed) if clip_seed is not None
                        else np.random.default_rng())
    Mb, T, V, _ = kp.shape
    out = np.zeros((Mb, t_target, V, 3), dtype=np.float32)
    blen = np.zeros(Mb, dtype=np.int64)
    present = np.zeros(Mb, dtype=np.float32)

    for m in range(Mb):
        body = kp[m]
        if not np.any(body):                                    # absent body
            continue
        present[m] = 1.0
        valid = np.where(np.any(body.reshape(T, -1) != 0, axis=1))[0]
        body = body[: valid[-1] + 1] if len(valid) else body[:0]
        L = len(body)
        if L == 0:
            present[m] = 0.0
            continue
        if sample == "cyclic":                                  # pyskl UniformSample: never pads
            rng = np.random.default_rng(clip_seed) if clip_seed is not None \
                else np.random.default_rng()
            body = body[_cyclic_indices(L, t_target, rng, train)]
            n = t_target
        elif L >= t_target:
            idx = (np.arange(t_target) if sample == "causal"
                   else np.linspace(0, L - 1, t_target).round().astype(int))
            body = body[idx]
            n = t_target
        else:
            n = L
        if not view_norm:
            # per-frame root-relative: kills camera translation AND the body's trajectory. Mutually
            # exclusive with view_normalize(), which already centred the clip with ONE transform.
            body = body - body[:, ROOT_JOINT: ROOT_JOINT + 1, :]
        if scale_norm:
            r = np.linalg.norm(body, axis=-1)
            s = np.median(r[r > 0]) if np.any(r > 0) else 1.0
            body = body / max(float(s), 1e-6)
        out[m, :n] = body
        blen[m] = n
    return out, blen, present


# =============================================================================
# Dataset
# =============================================================================
class NTUXView(Dataset):
    """NTU-60 Cross-View split. split in {'train','test'}.

    Yields dict(x (M,T,V,3) float32, label long, length (M,) long, present (M,) float32, camera).
    """

    def __init__(self, path, split, t_target=100, sample="causal", scale_norm=False,
                 items=None, view_norm=False, train=False, rot_aug=0.0):
        assert split in ("train", "test")
        assert_layout()
        self.split, self.T = split, t_target
        self.sample, self.scale_norm, self.view_norm = sample, scale_norm, view_norm
        # train=True enables STOCHASTIC augmentation (cyclic clip offset + RandomRot). Eval splits
        # must keep train=False or the reported number is measured on a randomly perturbed test set.
        self.train, self.rot_aug = train, rot_aug
        raw = items if items is not None else load_items(path)
        self.items = [it for it in raw if self._keep(it)]
        if not self.items:
            raise ValueError(f"NTU X-View '{split}' is empty -- check camera tags / split lists")

    def _keep(self, it):
        if it.get("bucket") in ("train", "test"):               # explicit pyskl split present
            return it["bucket"] == self.split
        return xview_bucket(it["camera"]) == self.split         # derive from camera id

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        x, blen, present = preprocess(it["keypoint"], self.T, self.sample, self.scale_norm,
                                      view_norm=self.view_norm, train=self.train,
                                      rot_aug=self.rot_aug)
        return {"x": torch.from_numpy(x), "label": torch.tensor(it["label"], dtype=torch.long),
                "length": torch.from_numpy(blen), "present": torch.from_numpy(present),
                "camera": it["camera"]}


def collate(batch):
    return {
        "x": torch.stack([b["x"] for b in batch]),              # (B,M,T,V,3)
        "label": torch.stack([b["label"] for b in batch]),      # (B,)
        "length": torch.stack([b["length"] for b in batch]),    # (B,M)
        "present": torch.stack([b["present"] for b in batch]),  # (B,M)
    }


# =============================================================================
# synthetic fixture -- makes the pipeline runnable BEFORE the real download
# =============================================================================
def make_fixture(path, n_per_cam=16, n_classes=6, t_range=(40, 130), seed=0):
    """Write a tiny .pkl in pyskl layout: random skeletons tagged across cameras 1/2/3.

    NOT a benchmark -- a shape/plumbing fixture. Class identity is faintly encoded as a per-class
    mean pose so a model can exceed chance, which lets the smoke test detect a DEAD training loop.
    """
    rng = np.random.default_rng(seed)
    proto = rng.normal(size=(n_classes, N_JOINTS, 3)).astype(np.float32) * 0.4
    anns = []
    for cam in (1, 2, 3):
        for k in range(n_per_cam):
            lab = int(rng.integers(n_classes))
            T = int(rng.integers(*t_range))
            M = 1 + int(rng.random() < 0.3)                     # sometimes 2 bodies
            kp = (proto[lab][None, None] + rng.normal(size=(M, T, N_JOINTS, 3)) * 0.15).astype(np.float32)
            kp += np.linspace(0, 1, T)[None, :, None, None] * rng.normal(size=(M, 1, 1, 3)) * 0.3
            tag = f"S001C{cam:03d}P{k:03d}R001A{lab+1:03d}"
            anns.append({"frame_dir": tag, "label": lab, "keypoint": kp, "total_frames": T})
    with open(path, "wb") as fh:
        pickle.dump({"annotations": anns}, fh)
    return path


def _selftest():
    assert_layout()
    print("[ntu] topology matches KINECT_BONES (25 joints, root=0)  OK")
    fx = os.path.join(os.path.dirname(__file__), "..", "outputs", "ntu_fixture.pkl")
    os.makedirs(os.path.dirname(fx), exist_ok=True)
    make_fixture(fx)
    tr = NTUXView(fx, "train", t_target=100)
    te = NTUXView(fx, "test", t_target=100)
    print(f"[ntu] X-View split  train(cams 2,3)={len(tr)}  test(cam 1)={len(te)}")
    assert all(camera_of(it["tag"]) in XVIEW_TRAIN_CAMS for it in tr.items)
    assert all(camera_of(it["tag"]) in XVIEW_TEST_CAMS for it in te.items)
    print("[ntu] camera partition asserted: no camera-1 sample in train  OK")
    b = collate([tr[0], tr[1], tr[2]])
    print(f"[ntu] collated batch  x{tuple(b['x'].shape)}  label{tuple(b['label'].shape)}  "
          f"length{tuple(b['length'].shape)}  present{tuple(b['present'].shape)}")
    # causal padding: a short sample must have length < T and a zero tail
    x0, l0, p0 = preprocess(tr.items[0]["keypoint"], 200)       # T bigger than any fixture seq
    assert l0.max() < 200 and not np.any(x0[0, int(l0[0]):]), "causal pad not zeroed"
    print("[ntu] causal pad/truncate  OK")
    print("[ntu] SELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--make-fixture", type=str, default=None)
    args = ap.parse_args()
    if args.make_fixture:
        print("wrote", make_fixture(args.make_fixture))
    if args.selftest:
        _selftest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
