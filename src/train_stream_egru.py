#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_stream_egru.py  --  F8: the STREAMING (causal) EGRU for an honest Time-to-First-Score.
============================================================================================
The published NTU X-View classifier (baselines.EGRUActionClassifier, 82.98% top-1) is
BIDIRECTIONAL and pools the mean over ALL frames, so it cannot emit a prediction until the whole
clip has arrived -- it has no honest TTFS. This trains the STREAMING variant (streaming=True:
unidirectional GRU + causal running-mean readout) so we can benchmark a true frame-by-frame
latency (ttfs_benchmark.py).

CONTROLLED CONVERSION (the one variable is the temporal readout)
---------------------------------------------------------------
We FREEZE the trained equivariant encoder + invariant projection from the 82.98% checkpoint and
retrain ONLY the unidirectional GRU + head. The encoder is equivariant and the projection invariant
for ANY weights, and both are PER-FRAME (already causal): they produce the identical invariant
feature stream regardless of how the temporal head reads it. So reusing them isolates the single
changed variable -- bidirectional full-pool -> unidirectional causal-pool -- and the viewpoint
theorem is UNTOUCHED (the GRU/head still see only invariants; certify re-checks, not assumes).

WHY THIS IS FAST, AND MEMORY-LEAN (this machine has ~6 GB free of 32 GB)
-----------------------------------------------------------------------
Freezing stops the encoder BACKWARD but not its FORWARD (the e3nn bottleneck), so we PRECOMPUTE the
frozen GRU-input features ONCE per sample and train the GRU+head on the cache -- minutes, not the
overnight full-model run. Two memory disciplines make this survive a loaded desktop:
  * the 1.19 GB pkl is loaded ONCE (nd.load_items) and shared across splits via items=; building a
    NTUXView per split would reload it and OOM.
  * the feature cache is streamed to an ON-DISK memmap (fp16), never held whole in RAM, and the raw
    pkl is freed before training. Peak RSS stays ~ the pkl, not pkl + 4 GB of cache.
Features are cached on CLEAN data (all-live mask): the encoder already owns the occlusion/G4
robustness (a separate axis); the streaming deliverable is the temporal readout + latency, clean.
The architecture is otherwise bit-identical to the 82.98% model, so this is "the same model, read
causally". Training pool == causal running mean at the LAST frame, so training and the streaming
readout share one objective.

Usage:
  python -u src/train_stream_egru.py --selftest
  python -u src/train_stream_egru.py --data data/ntu60_3danno.pkl --epochs 40
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import determinism                                                    # noqa: E402 (CUBLAS env first)
import ntu_dataset as nd                                             # noqa: E402
import run_xview_evaluation as rx                                    # noqa: E402
from baselines import build_model, count_parameters                 # noqa: E402

CKPT_BIDIR = "outputs/ntu_fault/egru_xview_mask.pt"                  # the 82.98% bidirectional model


def transfer_and_freeze(model, ckpt_path, device):
    """Load encoder.* + proj.* from the bidirectional checkpoint into the streaming model and FREEZE
    them. The gru.* / head.* keys of the bidirectional model are the WRONG shape (2x hidden) and are
    deliberately NOT transferred -- they are the new trainable parts."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state"] if "state" in ck else ck
    transfer = {k: v for k, v in sd.items()
                if k.startswith("encoder.") or k.startswith("proj.")}
    info = model.load_state_dict(transfer, strict=False)
    bad = [k for k in info.missing_keys if not (k.startswith("gru.") or k.startswith("head."))]
    if bad or info.unexpected_keys:
        raise RuntimeError(f"transfer mismatch: encoder/proj keys missing={bad} "
                           f"unexpected={info.unexpected_keys}")
    n_frozen = 0
    for name, p in model.named_parameters():
        if name.startswith("encoder.") or name.startswith("proj."):
            p.requires_grad_(False)
            n_frozen += p.numel()
    model.to(device)
    print(f"  transferred + FROZE encoder/proj ({n_frozen:,} params); "
          f"trainable (gru+head) = {count_parameters(model):,}", flush=True)
    return model


class FeatCache:
    """A split's cached GRU-input features on an on-disk fp16 memmap (never whole in RAM), plus the
    small per-sample length/present/label tensors. batch(idx) copies only the requested rows."""
    def __init__(self, path, shape, length, present, label):
        self.path, self.shape = path, shape
        self.length, self.present, self.label = length, present, label
        self.mm = np.memmap(path, dtype=np.float16, mode="r", shape=shape)

    def __len__(self):
        return self.shape[0]

    def batch(self, idx, device):
        idx_np = idx.numpy() if torch.is_tensor(idx) else np.asarray(idx)
        f = torch.from_numpy(np.ascontiguousarray(self.mm[idx_np])).float().to(device)
        return (f, self.length[idx].to(device), self.present[idx].to(device),
                self.label[idx].to(device))


@torch.no_grad()
def cache_features(model, ds, device, batch_size, seed, tag, cache_dir):
    """Run the FROZEN encoder+proj over a split ONCE, streaming the exact GRU-input tensor
    (concat[inv, liveness]) to an on-disk fp16 memmap. Clean data (all-live mask)."""
    os.makedirs(cache_dir, exist_ok=True)
    N = len(ds)
    M = int(ds[0]["x"].shape[0])
    T = ds.T
    Din = model.proj.dim + (nd.N_JOINTS if model.use_mask else 0)
    path = os.path.join(cache_dir, f"_cache_{tag}.dat")
    if os.path.exists(path):                                          # clear any stale/locked file
        try:
            os.remove(path)
        except OSError:
            path = os.path.join(cache_dir, f"_cache_{tag}_{os.getpid()}.dat")

    # WRITE with buffered file I/O, not a w+ memmap. A writable memmap pins its dirty pages in this
    # process's working set (they OOM-killed the run at ~2-3 GB free); plain f.write() dirty pages
    # sit in the OS cache and are RECLAIMABLE. shuffle=False => the loader walks ds.items in index
    # order, so row i of the file is sample i (matching FeatCache). We also free each keypoint the
    # instant it is cached, so RAM DROPS across the pass.
    loader = rx.make_loader(ds, batch_size, False, seed)
    off = 0
    L, P, Y = [], [], []
    t0 = time.time()
    with open(path, "wb", buffering=1 << 20) as fh:
        for bi, batch in enumerate(loader):
            x = batch["x"].to(device)                                # (B,M,T,V,3)
            B, Mb, Tb, V, _ = x.shape
            gin_bodies = []
            for m in range(Mb):
                xb = x[:, m]
                mflat = torch.ones(B * Tb, V, dtype=xb.dtype, device=device) if model.use_mask \
                    else None
                h = model.encoder(xb.reshape(B * Tb, V, 3), mflat)
                inv = model.proj(h, xb.reshape(B * Tb, V, 3), mflat).reshape(B, Tb, -1)
                feats = [inv]
                if model.use_mask:                                  # constant-ones liveness channel
                    feats.append(torch.ones(B, Tb, V, dtype=xb.dtype, device=device))
                gin_bodies.append(torch.cat(feats, dim=-1))
            gin = torch.stack(gin_bodies, dim=1)                    # (B,M,T,Din)
            fh.write(gin.detach().cpu().numpy().astype(np.float16).tobytes())
            for j in range(off, off + B):                          # free keypoints already cached
                ds.items[j]["keypoint"] = None
            off += B
            L.append(batch["length"]); P.append(batch["present"]); Y.append(batch["label"])
            if (bi + 1) % 25 == 0:
                fh.flush()
                if device == "cuda":
                    torch.cuda.empty_cache()
            if (bi + 1) % 50 == 0:
                print(f"    [{tag}] cached {off}/{N}  ({time.time()-t0:.0f}s)", flush=True)
    gb = N * M * T * Din * 2 / 1e9
    print(f"  cached {tag}: ({N},{M},{T},{Din}) fp16 -> {path} ({gb:.2f} GB)  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return FeatCache(path, (N, M, T, Din), torch.cat(L), torch.cat(P), torch.cat(Y))


def head_logits(model, feats, length, present):
    """Feature-space forward of the TRAINABLE part only (gru + head) with the full-sequence
    mean-over-valid-frames pool == causal running mean at the last frame. feats: (B,M,T,Din)."""
    B, M, T, Din = feats.shape
    outs = []
    for m in range(M):
        seq, _ = model.gru(feats[:, m])                             # (B,T,H) unidirectional
        idx = torch.arange(T, device=feats.device).unsqueeze(0)
        vm = (idx < length[:, m].unsqueeze(1)).to(seq.dtype).unsqueeze(-1)
        pooled = (seq * vm).sum(1) / vm.sum(1).clamp(min=1.0)
        outs.append(model.head(pooled))                            # (B,C)
    logits = torch.stack(outs, dim=1)                              # (B,M,C)
    w = present.unsqueeze(-1)
    return (logits * w).sum(1) / w.sum(1).clamp(min=1.0)


@torch.no_grad()
def eval_cached(model, cache, device, batch_size=256):
    model.eval()
    correct = total = 0
    for i in range(0, len(cache), batch_size):
        idx = torch.arange(i, min(i + batch_size, len(cache)))
        f, ln, pr, y = cache.batch(idx, device)
        lo = head_logits(model, f, ln, pr)
        correct += (lo.argmax(1) == y).sum().item()
        total += lo.shape[0]
    return 100.0 * correct / max(total, 1)


def train_head(model, tr, val, te, device, epochs, lr, wd, batch_size, seed, out_dir, tag):
    """Train ONLY gru+head on cached features. Selection on val (clean protocol), report on test."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    N = len(tr)
    g = torch.Generator().manual_seed(seed)
    best, best_ep, best_state = -1.0, -1, None
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(N, generator=g)
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            f, ln, pr, y = tr.batch(idx, device)
            loss = ce(head_logits(model, f, ln, pr), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sched.step()
        sel = eval_cached(model, val if val is not None else te, device)
        if sel > best:
            best, best_ep = sel, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep == 0 or (ep + 1) % max(1, epochs // 10) == 0 or ep == epochs - 1:
            print(f"    ep {ep+1:3d}  sel {sel:5.2f}%  (best {best:5.2f}% @ ep{best_ep+1})  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"{tag}.pt")
            torch.save({"state": best_state, "tag": tag, "best_sel": best,
                        "uses_mask": model.uses_mask, "streaming": True}, path)
            print(f"  saved best (ep {best_ep+1}) -> {path}", flush=True)
    test_acc = eval_cached(model, te, device)
    print(f"  [{tag}] TEST top-1 (streaming, final-frame pool) {test_acc:5.2f}%", flush=True)
    return {"tag": tag, "xview_top1_test": test_acc, "sel_best": best, "sel_epoch": best_ep + 1}


def run_selftest(args, device):
    fx = os.path.join("outputs", "ntu_fixture.pkl")
    os.makedirs("outputs", exist_ok=True)
    nd.make_fixture(fx, n_per_cam=24, n_classes=6)
    tr = nd.NTUXView(fx, "train", t_target=64)
    te = nd.NTUXView(fx, "test", t_target=64)
    print(f"[stream selftest] train={len(tr)} test={len(te)} (6 classes)", flush=True)
    model = build_model("egru", n_classes=6, use_mask=True, streaming=True).to(device)
    cdir = os.path.join("outputs", "ntu_stream", "_selftest_cache")
    trc = cache_features(model, tr, device, 8, 42, "train", cdir)
    tec = cache_features(model, te, device, 8, 42, "test", cdir)
    train_head(model, trc, None, tec, device, epochs=4, lr=1e-3, wd=1e-3,
               batch_size=8, seed=42, out_dir=None, tag="egru_stream_selftest")
    b = nd.collate([te[i] for i in range(min(4, len(te)))])
    model.eval()
    with torch.no_grad():
        st = model.forward_stream(b["x"].to(device), present=b["present"].to(device))
    assert st.dim() == 3 and st.shape[-1] == 6, st.shape
    print(f"[stream selftest] forward_stream -> {tuple(st.shape)} (B,T,C)  PASSED", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--init-from", type=str, default=CKPT_BIDIR)
    ap.add_argument("--t-target", type=int, default=100)
    ap.add_argument("--sample", choices=["causal", "uniform", "cyclic"], default="causal")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--out", type=str, default="outputs/ntu_stream")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    cfg = determinism.enable(seed=args.seed, strict=False)
    print(determinism.report(cfg), flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}\n", flush=True)

    if args.selftest or args.data is None:
        if args.data is None and not args.selftest:
            print("no --data; running fixture selftest.\n", flush=True)
        run_selftest(args, device)
        return 0

    print("=" * 82, flush=True)
    print("F8  STREAMING EGRU  --  unidirectional + causal running mean, frozen equivariant encoder",
          flush=True)
    print("=" * 82, flush=True)
    model = build_model("egru", n_classes=60, use_mask=True, streaming=True)
    transfer_and_freeze(model, args.init_from, device)

    # ---- load the 1.19 GB pkl ONCE, share across splits (a per-split NTUXView would reload+OOM) --
    print("  loading NTU items (once)...", flush=True)
    raw = nd.load_items(args.data)
    # halve the resident footprint: store source keypoints as fp16 (preprocess upcasts to float32 at
    # ntu_dataset.py:293, so the model still sees float32). This machine runs with very little free
    # RAM; fp16 coords are ~mm precision, negligible for NTU.
    for it in raw:
        it["keypoint"] = np.asarray(it["keypoint"], dtype=np.float16)
    gc.collect()
    tr = nd.NTUXView(args.data, "train", t_target=args.t_target, sample=args.sample, train=True,
                     items=raw)
    te = nd.NTUXView(args.data, "test", t_target=args.t_target, sample=args.sample, train=False,
                     items=raw)
    val = None
    if args.val_frac > 0:                                            # carve val from train
        rng = np.random.default_rng(args.seed + 1)
        vn = max(1, int(len(tr.items) * args.val_frac))
        vi = set(rng.choice(len(tr.items), vn, replace=False).tolist())
        val_items = [tr.items[i] for i in vi]
        tr.items = [tr.items[i] for i in range(len(tr.items)) if i not in vi]
        val = nd.NTUXView(args.data, "train", t_target=args.t_target, sample=args.sample,
                          items=val_items, train=False)
        print(f"  val holdout: {len(val_items)} carved -> train={len(tr.items)}", flush=True)
    print(f"NTU-60 X-View  train={len(tr)}  val={0 if val is None else len(val)}  test={len(te)}\n",
          flush=True)

    cdir = os.path.join(args.out, "_cache")
    trc = cache_features(model, tr, device, args.batch_size, args.seed, "train", cdir)
    valc = cache_features(model, val, device, args.batch_size, args.seed, "val", cdir) if val else None
    tec = cache_features(model, te, device, args.batch_size, args.seed, "test", cdir)

    # free the raw pkl + datasets before training (cache lives on disk now)
    del raw, tr, te, val
    gc.collect()
    print("  freed raw pkl; training on the disk-backed feature cache\n", flush=True)

    res = train_head(model, trc, valc, tec, device, args.epochs, args.lr, args.wd,
                     args.batch_size, args.seed, args.out, "egru_stream_xview_mask")
    res["bidir_reference"] = 82.98
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"stream_results_s{args.seed}.json"), "w") as fh:
        json.dump({"args": vars(args), "result": res}, fh, indent=2)
    print(f"\n  streaming {res['xview_top1_test']:.2f}% vs bidirectional 82.98% "
          f"(cost of causal readout = {82.98 - res['xview_top1_test']:+.2f} pts)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
