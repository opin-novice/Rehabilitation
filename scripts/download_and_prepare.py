#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_and_prepare.py
=======================
Pull the OpenMMLab / pyskl preprocessed NTU-60 3D skeletons and confirm they load through our
X-View pipeline. No license form, no 5 GB of raw .skeleton text.

Divergences from the naive "download + re-pickle a flat list" script, and WHY:
  * We DO NOT drop pyskl's own 'split' dict (xview_train / xview_val). For NTU that split is the
    authoritative X-View definition; camera-derived splitting happens to agree, but preserving the
    official one removes any doubt.
  * We DO NOT upcast keypoints to float32 and re-pickle the whole corpus. pyskl stores ~56k samples
    at ~1.2 GB; a float32 rewrite roughly triples that on disk and in RAM for no benefit --
    ntu_dataset.NTUXView reads the pyskl layout NATIVELY and casts per sample at load time.
  * We VALIDATE by actually constructing the X-View split through NTUXView and asserting no
    camera-1 leak into train -- a print of "done" is not verification.

Real size note: the CDN file is ~1.19 GB (HEAD Content-Length), not the 478 MB sometimes quoted.

Run:  python scripts/download_and_prepare.py
      python scripts/download_and_prepare.py --force        # re-download even if cached
"""

import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import ntu_dataset as nd                                             # noqa: E402

URL = "https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_3danno.pkl"
RAW_PATH = os.path.join("data", "ntu60_3danno.pkl")


def _progress(block_num, block_size, total_size):
    done = block_num * block_size
    pct = min(100.0, 100.0 * done / total_size) if total_size > 0 else 0.0
    bar = "#" * int(pct // 2.5)
    print(f"\r  [{bar:<40}] {pct:5.1f}%  {done/1e6:7.1f} / {total_size/1e6:7.1f} MB",
          end="", flush=True)


def download(force=False):
    os.makedirs("data", exist_ok=True)
    if os.path.exists(RAW_PATH) and not force:
        # verify the cached file is complete against the CDN Content-Length
        try:
            r = urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=20)
            remote = int(r.headers.get("Content-Length", 0))
        except Exception:
            remote = 0
        local = os.path.getsize(RAW_PATH)
        if remote == 0 or local == remote:
            print(f"[+] cached {RAW_PATH} ({local/1e6:.1f} MB) looks complete -- skipping download")
            return RAW_PATH
        print(f"[!] cached file is {local/1e6:.1f} MB but CDN reports {remote/1e6:.1f} MB; "
              f"re-downloading")
    print(f"[*] downloading NTU-60 3D annotations from OpenMMLab")
    print(f"    {URL}")
    tmp = RAW_PATH + ".part"
    urllib.request.urlretrieve(URL, tmp, _progress)
    os.replace(tmp, RAW_PATH)
    print(f"\n[+] download complete: {RAW_PATH} ({os.path.getsize(RAW_PATH)/1e6:.1f} MB)")
    return RAW_PATH


def validate(path):
    print("[*] validating through NTUXView (native pyskl read, no rewrite) ...")
    tr = nd.NTUXView(path, "train", t_target=100)
    te = nd.NTUXView(path, "test", t_target=100)
    assert all(nd.camera_of(it["tag"]) in nd.XVIEW_TRAIN_CAMS for it in tr.items), "cam-1 in train!"
    assert all(nd.camera_of(it["tag"]) in nd.XVIEW_TEST_CAMS for it in te.items), "non-cam-1 in test!"
    classes = {it["label"] for it in tr.items}
    s = tr[0]
    print(f"[+] X-View: train(cams 2,3)={len(tr)}  test(cam 1)={len(te)}  classes={len(classes)}/60")
    print(f"[+] sample tensor x{tuple(s['x'].shape)}  length{tuple(s['length'].shape)}  "
          f"present{tuple(s['present'].shape)}")
    assert s["x"].shape[1:] == (100, nd.N_JOINTS, 3), "unexpected sample shape"
    assert len(classes) == 60, f"expected 60 NTU-60 classes, saw {len(classes)}"
    print("[++] READY. Run the X-View comparison with:")
    print(f"     python src/run_xview_evaluation.py --data {path} --models egru stgcn --epochs 60")
    print(f"     python src/run_xview_evaluation.py --data {path} --model egru --use-mask --eval-drop 4")
    return len(tr), len(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    path = RAW_PATH if args.validate_only else download(force=args.force)
    validate(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
