"""Reviewer round-4, T2 (concern #7): is the sensor-ID probe an artifact of zero-padding?

The reviewer notes the padded joints 22-24 (IRDS/UI-PRMD) and REHAB246's duplicated
permutation targets create low-variance channels a probe could trivially use to detect
'which corpus'. We remove that tell: zero the SAME problematic joints {7,11,22,23,24}
(padded slots + REHAB246 duplicates) across ALL three corpora, so no corpus carries a
distinguishing zero-pattern, then re-run the 3-way sensor-identity probe on features from
the scratch TCN and ST-GCN fold models. If accuracy stays ~1.00, the sensor separation
is not a padding artifact.

Run:  python src/reviewer_round4_probe_sharedjoints.py
Out:  outputs/reviewer_round4/probe_sharedjoints.{json,md}
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # src/
from selfsup.zeroshot_eval import _rebuild_model  # noqa: E402
from selfsup.data import load_corpus_with_labels  # noqa: E402

OUT_DIR = "outputs/reviewer_round4"
# padded (22-24) + REHAB246 duplicated permutation targets (7,11) -> zero in ALL corpora
ZERO_JOINTS = [7, 11, 22, 23, 24]
TCN_DIR = "results/kimore_loso_78fold/A_scratch"
STGCN_DIR = "results/kimore_loso_78fold_stgcn"


def _mask(X):
    X = X.copy()
    X[:, :, ZERO_JOINTS, :] = 0.0
    return X


@torch.no_grad()
def _features(model, X, batch=256):
    xt = torch.from_numpy(X.astype(np.float32))
    out = []
    for i in range(0, len(xt), batch):
        out.append(model.forward_features(xt[i:i + batch]).cpu().numpy())
    return np.concatenate(out)


def _probe(ckpt_dir, corpora, n_models=5):
    cps = sorted(glob.glob(os.path.join(ckpt_dir, "fold_*", "best_model.pt")))[:n_models]
    if not cps:
        return None
    accs_masked, accs_full = [], []
    names = list(corpora)
    lm = {n: i for i, n in enumerate(names)}
    for cp in cps:
        m = _rebuild_model(torch.load(cp, map_location="cpu"))
        for masked, store in ((True, accs_masked), (False, accs_full)):
            F, yy = [], []
            for n in names:
                X = _mask(corpora[n]) if masked else corpora[n]
                f = _features(m, X)
                F.append(f); yy.append(np.full(len(f), lm[n]))
            F = np.concatenate(F); yy = np.concatenate(yy)
            store.append(float(np.mean(cross_val_score(
                LogisticRegression(max_iter=1000, C=1), F, yy, cv=3,
                scoring="balanced_accuracy"))))
    return {"masked_shared_joints": float(np.mean(accs_masked)),
            "full_all_joints": float(np.mean(accs_full)),
            "chance": 1.0 / len(names), "n_models": len(cps),
            "zeroed_joints": ZERO_JOINTS}


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    Xk = load_corpus_with_labels("KIMORE")[0]
    Xr = load_corpus_with_labels("REHAB246")[0]
    Xu = load_corpus_with_labels("UIPRMD")[0]
    corpora = {"KIMORE": Xk, "REHAB246": Xr, "UIPRMD": Xu}
    res = {"TCN": _probe(TCN_DIR, corpora), "STGCN": _probe(STGCN_DIR, corpora)}
    with open(os.path.join(OUT_DIR, "probe_sharedjoints.json"), "w") as f:
        json.dump(res, f, indent=2)
    lines = ["### Sensor-ID probe with shared joints only (zero-padding confound removed)", "",
             "| Backbone | 3-way acc (all joints) | 3-way acc (shared joints, {7,11,22,23,24} zeroed) | chance |",
             "|---|---|---|---|"]
    for bb, r in res.items():
        if r:
            lines.append(f"| {bb} | {r['full_all_joints']:.3f} | {r['masked_shared_joints']:.3f} | {r['chance']:.2f} |")
            print(f"{bb}: full={r['full_all_joints']:.3f} shared={r['masked_shared_joints']:.3f}")
    with open(os.path.join(OUT_DIR, "probe_sharedjoints.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return res


if __name__ == "__main__":
    run()
