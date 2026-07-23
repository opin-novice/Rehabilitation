#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_ntu.py
==============
Validate and reshape whatever NTU-60 skeleton dump you downloaded into the EXACT pyskl layout that
src/ntu_dataset.NTUXView expects, then prove it loads by constructing the X-View split.

The output is one file:
    {"annotations": [ {frame_dir, label, keypoint (M,T,V,3) float32, total_frames}, ... ],
     "split": {"xview_train": [frame_dir...], "xview_val": [frame_dir...]}}
The split is DERIVED from the camera id (train cams 2,3 / test cam 1), so a downstream NTUXView
never has to guess.

Auto-detected inputs (NTU RGB+D 60, 3D skeletons, 25 joints)
------------------------------------------------------------
  1. pyskl / MMAction2 .pkl        already close; validated, keypoint coerced to (M,T,V,3), split
                                   rebuilt from cameras. (dict with 'annotations'.)
  2. ST-GCN / 2s-AGCN .npy+.pkl    data (N,3,T,V,M) + label pickle (names, labels). The most common
                                   preprocessed release. Pass the .npy; the label .pkl is found
                                   next to it or via --labels.
  3. raw .skeleton directory       the canonical NTU dump (one S###C###P###R###A###.skeleton per
                                   sample). Parsed here; up to 2 bodies kept by motion energy.
  4. .npz                          arrays keypoint (N,M,T,V,3), label (N,), frame_dir/camera.

Handles the standard NTU "missing / bad skeleton" ignore list (--ignore-list), and skips samples
whose action label is outside 1..60 (i.e. NTU-120-only classes) with a warning.

Usage
-----
  python scripts/prepare_ntu.py --in <path> --out data/ntu60_xview.pkl
  python scripts/prepare_ntu.py --in ntu/train_data.npy --labels ntu/train_label.pkl --out data/ntu60_xview.pkl
  python scripts/prepare_ntu.py --in ntu/nturgb+d_skeletons/ --out data/ntu60_xview.pkl --ignore-list ntu/missing.txt
  python scripts/prepare_ntu.py --validate data/ntu60_xview.pkl        # just re-check an existing file
"""

import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import ntu_dataset as nd                                             # noqa: E402

V_EXPECT = nd.N_JOINTS          # 25
MAX_BODIES = nd.MAX_BODIES      # 2
_TAG_RE = re.compile(r"S\d{3}C\d{3}P\d{3}R\d{3}A\d{3}")


# =============================================================================
# helpers
# =============================================================================
def clean_tag(name):
    """Pull a canonical S###C###P###R###A### tag out of a filename or path."""
    base = os.path.basename(str(name))
    m = _TAG_RE.search(base)
    return m.group(0) if m else base.split(".")[0]


def label_from_tag(tag):
    """A-field (1..60) -> 0-based class id, or None if unparseable / out of NTU-60 range."""
    m = re.search(r"A(\d{3})", tag)
    if not m:
        return None
    a = int(m.group(1))
    return a - 1 if 1 <= a <= 60 else None


def _coerce_kp(kp):
    """Any of (T,V,3)/(M,T,V,3)/(C,T,V,M)-ish -> (M,T,V,3) float32 with V=25, last dim=3."""
    kp = np.asarray(kp, dtype=np.float32)
    if kp.ndim == 3:                                    # (T,V,3)
        kp = kp[None]
    if kp.ndim != 4:
        raise ValueError(f"cannot interpret keypoint of shape {kp.shape}")
    # find the axis of size 3 (coords) and the axis of size 25 (joints); reorder to (M,T,V,3)
    shp = kp.shape
    v_ax = next((i for i, s in enumerate(shp) if s == V_EXPECT), None)
    c_ax = next((i for i, s in enumerate(shp) if s == 3 and i != v_ax), None)
    if v_ax is None or c_ax is None:
        raise ValueError(f"keypoint {shp} has no clear V=25 / C=3 axes")
    others = [i for i in range(4) if i not in (v_ax, c_ax)]
    # heuristic: the SMALLER remaining axis is bodies (M<=2..4), the larger is time
    m_ax, t_ax = (others if shp[others[0]] <= shp[others[1]] else others[::-1])
    return np.transpose(kp, (m_ax, t_ax, v_ax, c_ax)).astype(np.float32)


# =============================================================================
# input loaders -> list of {tag, label, keypoint (M,T,V,3)}
# =============================================================================
def from_pyskl_pkl(path):
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    anns = blob["annotations"] if isinstance(blob, dict) else blob
    out = []
    for a in anns:
        tag = clean_tag(a.get("frame_dir") or a.get("filename") or a.get("tag"))
        out.append({"tag": tag, "label": int(a["label"]), "keypoint": _coerce_kp(a["keypoint"])})
    return out


def from_stgcn_npy(path, labels_path):
    data = np.load(path, mmap_mode="r")                 # (N,C,T,V,M) typically
    if labels_path is None:
        for cand in ("_label.pkl", "_label.npy"):
            g = path.replace("_data.npy", cand)
            if os.path.exists(g):
                labels_path = g
                break
    if labels_path is None or not os.path.exists(labels_path):
        raise FileNotFoundError("ST-GCN .npy needs its label file; pass --labels <..._label.pkl>")
    if labels_path.endswith(".pkl"):
        with open(labels_path, "rb") as fh:
            names, labels = pickle.load(fh)
    else:
        labels = np.load(labels_path)
        names = [f"sample{i:06d}" for i in range(len(labels))]
    out = []
    for i in range(len(labels)):
        tag = clean_tag(names[i])
        out.append({"tag": tag, "label": int(labels[i]), "keypoint": _coerce_kp(data[i])})
    return out


def read_skeleton_file(path):
    """Parse a raw NTU .skeleton file -> (M<=2, T, 25, 3), keeping the 2 most-moving bodies."""
    with open(path) as fh:
        toks = fh.read().split("\n")
    it = iter(toks)
    n_frames = int(next(it))
    bodies = {}                                         # bodyID -> list of (frame_idx, (25,3))
    for f in range(n_frames):
        n_bodies = int(next(it))
        for _ in range(n_bodies):
            info = next(it).split()
            bid = info[0]
            n_j = int(next(it))
            joints = np.zeros((V_EXPECT, 3), dtype=np.float32)
            for j in range(n_j):
                vals = next(it).split()
                if j < V_EXPECT:
                    joints[j] = [float(vals[0]), float(vals[1]), float(vals[2])]
            bodies.setdefault(bid, []).append((f, joints))
    # densify each body to (T,25,3)
    dense = {}
    for bid, frames in bodies.items():
        arr = np.zeros((n_frames, V_EXPECT, 3), dtype=np.float32)
        for fi, jj in frames:
            arr[fi] = jj
        dense[bid] = arr
    if not dense:
        return np.zeros((1, n_frames, V_EXPECT, 3), dtype=np.float32)
    energy = {bid: float(np.linalg.norm(np.diff(a, axis=0)).sum()) for bid, a in dense.items()}
    keep = sorted(energy, key=energy.get, reverse=True)[:MAX_BODIES]
    return np.stack([dense[b] for b in keep], axis=0)   # (M,T,25,3)


def from_skeleton_dir(path):
    files = sorted(glob.glob(os.path.join(path, "*.skeleton")))
    if not files:
        raise FileNotFoundError(f"no .skeleton files under {path}")
    out = []
    for fp in files:
        tag = clean_tag(fp)
        lab = label_from_tag(tag)
        if lab is None:
            continue
        try:
            kp = read_skeleton_file(fp)
        except Exception as e:                          # a corrupt file must not kill the whole run
            print(f"  WARN skip {tag}: {e!r}")
            continue
        out.append({"tag": tag, "label": lab, "keypoint": kp})
    return out


def _from_npz(path):
    z = np.load(path, allow_pickle=True)
    kp, labels = z["keypoint"], z["label"]
    tags = z["frame_dir"] if "frame_dir" in z else None
    out = []
    for i in range(len(kp)):
        tag = clean_tag(tags[i]) if tags is not None else f"sample{i:06d}"
        out.append({"tag": tag, "label": int(labels[i]), "keypoint": _coerce_kp(kp[i])})
    return out


def detect_and_load(path, labels_path):
    if os.path.isdir(path):
        print("  input: raw .skeleton directory")
        return from_skeleton_dir(path)
    if path.endswith(".npy"):
        print("  input: ST-GCN / 2s-AGCN .npy (+ label pkl)")
        return from_stgcn_npy(path, labels_path)
    if path.endswith(".npz"):
        print("  input: .npz")
        return _from_npz(path)
    if path.endswith(".pkl") or path.endswith(".pickle"):
        print("  input: pyskl / mmaction .pkl")
        return from_pyskl_pkl(path)
    raise ValueError(f"cannot detect NTU format for {path}")


# =============================================================================
# assemble + validate
# =============================================================================
def build(records, ignore=frozenset()):
    anns, cams, labels, skipped = [], {}, {}, {"ignored": 0, "badcam": 0, "range": 0, "shape": 0}
    for r in records:
        tag = r["tag"]
        if tag in ignore:
            skipped["ignored"] += 1
            continue
        try:
            cam = nd.camera_of(tag)
        except ValueError:
            skipped["badcam"] += 1
            continue
        lab = r["label"]
        if not (0 <= lab <= 59):
            skipped["range"] += 1
            continue
        kp = _coerce_kp(r["keypoint"])
        if kp.shape[-2] != V_EXPECT or kp.shape[-1] != 3:
            skipped["shape"] += 1
            continue
        anns.append({"frame_dir": tag, "label": lab, "keypoint": kp,
                     "total_frames": int(kp.shape[1])})
        cams[cam] = cams.get(cam, 0) + 1
        labels[lab] = labels.get(lab, 0) + 1

    split = {"xview_train": [a["frame_dir"] for a in anns
                             if nd.camera_of(a["frame_dir"]) in nd.XVIEW_TRAIN_CAMS],
             "xview_val": [a["frame_dir"] for a in anns
                           if nd.camera_of(a["frame_dir"]) in nd.XVIEW_TEST_CAMS]}
    return {"annotations": anns, "split": split}, cams, labels, skipped


def validate(out_path):
    """Load the produced file through the REAL Dataset and assert the X-View contract."""
    tr = nd.NTUXView(out_path, "train", t_target=100)
    te = nd.NTUXView(out_path, "test", t_target=100)
    assert all(nd.camera_of(it["tag"]) in nd.XVIEW_TRAIN_CAMS for it in tr.items), "cam leak train"
    assert all(nd.camera_of(it["tag"]) in nd.XVIEW_TEST_CAMS for it in te.items), "cam leak test"
    s = tr[0]
    assert s["x"].shape[1:] == (100, V_EXPECT, 3), f"bad sample shape {tuple(s['x'].shape)}"
    ntr = len({it["label"] for it in tr.items})
    print(f"  VALIDATE: train={len(tr)} (cams 2,3)  test={len(te)} (cam 1)  classes_seen={ntr}")
    print(f"  VALIDATE: sample x{tuple(s['x'].shape)}  length{tuple(s['length'].shape)}  OK")
    return len(tr), len(te)


def load_ignore(path):
    if not path:
        return frozenset()
    with open(path) as fh:
        return frozenset(clean_tag(ln) for ln in fh if ln.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, help="NTU dump: .pkl / .npy / .npz / skeleton dir")
    ap.add_argument("--labels", type=str, default=None, help="label pkl for the ST-GCN .npy path")
    ap.add_argument("--out", type=str, default="data/ntu60_xview.pkl")
    ap.add_argument("--ignore-list", type=str, default=None,
                    help="NTU samples_with_missing_skeletons.txt (tags to drop)")
    ap.add_argument("--validate", type=str, default=None,
                    help="skip conversion; just re-validate an existing prepared file")
    args = ap.parse_args()

    if args.validate:
        print(f"re-validating {args.validate}")
        validate(args.validate)
        return 0
    if not args.inp:
        ap.error("--in is required (or use --validate)")

    print(f"preparing NTU-60 X-View from {args.inp}")
    records = detect_and_load(args.inp, args.labels)
    print(f"  loaded {len(records)} raw records")
    blob, cams, labels, skipped = build(records, ignore=load_ignore(args.ignore_list))

    n = len(blob["annotations"])
    if n == 0:
        raise SystemExit("no usable samples after validation -- check the input format/tags")
    print(f"  kept {n} samples   skipped {skipped}")
    print(f"  cameras: " + "  ".join(f"C{c}={cams.get(c,0)}" for c in sorted(cams)))
    print(f"  classes present: {len(labels)}/60   "
          f"(min/med/max count {min(labels.values())}/"
          f"{int(np.median(list(labels.values())))}/{max(labels.values())})")
    print(f"  X-View split: train(cams 2,3)={len(blob['split']['xview_train'])}  "
          f"test(cam 1)={len(blob['split']['xview_val'])}")
    if len(blob["split"]["xview_train"]) == 0 or len(blob["split"]["xview_val"]) == 0:
        print("  WARNING: one X-View bucket is EMPTY -- your tags may lack real C### camera ids.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(blob, fh, protocol=4)
    print(f"  wrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")

    print("re-loading through NTUXView to confirm the contract ...")
    validate(args.out)
    print("PREPARE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
