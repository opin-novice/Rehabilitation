#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ttfs_benchmark.py  --  F8: Time-to-First-Score for the STREAMING EGRU.
======================================================================
The bidirectional 82.98% classifier has NO honest TTFS: bidirectional + mean-over-all-frames means
the first score cannot exist until the last frame arrives. The streaming variant
(train_stream_egru.py: unidirectional GRU + causal running mean) emits a score after frame 1, so it
has a real, measurable live latency. This benchmark reports TWO honest numbers:

  1. LATENCY -- a genuinely INCREMENTAL single-frame step: encoder(one frame) -> proj -> ONE GRU
     step carrying the hidden state -> running-mean update (O(1) sum+count) -> head. This is the
     wall-clock cost the deployment pays PER FRAME, compared to the 33 ms / 30 fps budget. We do NOT
     time the vectorised forward_stream (which sees all T frames at once) -- that would be a lie
     about live latency. Time-to-First-Score = one such step (a score exists after frame 1).

  2. ACCURACY vs OBSERVED FRAMES -- feed prefixes and read the causal running-mean prediction at
     each frame; report test top-1 as a function of how many frames have arrived. Shows the score is
     available immediately and how fast it converges to the full-clip accuracy. "Frames-to-90%%-of-
     final" is the honest "time to a trustworthy score".

The cost of causality (streaming test top-1 < the 82.98%% bidirectional model) is reported by
train_stream_egru.py; this file is about latency and convergence, the things streaming BUYS.

Usage:
  python src/ttfs_benchmark.py --ckpt outputs/ntu_stream/egru_stream_xview_mask.pt \
                               --data data/ntu60_3danno.pkl
  python src/ttfs_benchmark.py --latency-only          # no trained ckpt needed (times the arch)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import determinism                                                    # noqa: E402
import ntu_dataset as nd                                             # noqa: E402
import run_xview_evaluation as rx                                    # noqa: E402
from baselines import build_model                                    # noqa: E402


def load_stream_model(ckpt, device):
    model = build_model("egru", n_classes=60, use_mask=True, streaming=True)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state"] if "state" in ck else ck)
    return model.to(device).eval()


@torch.no_grad()
def _frame_gin(model, xt, device):
    """One frame x_t (V,3) -> GRU input row (1,1,Din): encoder+proj+liveness, exactly as training."""
    V = xt.shape[0]
    xf = xt.reshape(1, V, 3).to(device)
    mflat = torch.ones(1, V, dtype=xf.dtype, device=device) if model.use_mask else None
    h = model.encoder(xf, mflat)
    inv = model.proj(h, xf, mflat)                                   # (1, projdim)
    feats = [inv]
    if model.use_mask:
        feats.append(torch.ones(1, V, dtype=xf.dtype, device=device))
    return torch.cat(feats, dim=-1).reshape(1, 1, -1)                # (1,1,Din)


@torch.no_grad()
def latency(model, device, n_frames=300, warmup=30):
    """Genuinely incremental per-frame latency: carry the GRU hidden state, one frame at a time.
    Decomposed into encoder+proj (the e3nn cost) vs the GRU-step+running-mean+head (the recurrent
    tail), so the report says WHERE the 33 ms budget goes."""
    V = nd.N_JOINTS
    g = torch.Generator().manual_seed(0)
    frames = torch.randn(n_frames, V, 3, generator=g)
    frames = frames - frames[:, :1]                                  # root-relative, as deployed
    cuda = device == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize()

    enc_t, rec_t, tot_t = [], [], []
    h = None
    run_sum = torch.zeros(1, model.gru.hidden_size, device=device)
    for k in range(n_frames):
        sync(); t0 = time.perf_counter()
        gin = _frame_gin(model, frames[k], device)                  # encoder + proj
        sync(); t1 = time.perf_counter()
        out, h = model.gru(gin, h)                                  # ONE causal GRU step
        run_sum = run_sum + out[:, 0]                               # O(1) running-sum update
        pooled = run_sum / (k + 1)                                  # causal running mean
        _ = model.head(pooled)                                      # live logits
        sync(); t2 = time.perf_counter()
        if k >= warmup:                                            # drop warmup (JIT/caches/clocks)
            enc_t.append(t1 - t0); rec_t.append(t2 - t1); tot_t.append(t2 - t0)
    ms = lambda a: 1e3 * float(np.mean(a))
    p95 = lambda a: 1e3 * float(np.percentile(a, 95))
    return {"encoder_proj_ms": ms(enc_t), "recurrent_tail_ms": ms(rec_t),
            "per_frame_ms": ms(tot_t), "per_frame_ms_p95": p95(tot_t),
            "fps": 1e3 / ms(tot_t), "budget_ms": 1000.0 / 30, "n_timed": len(tot_t)}


@torch.no_grad()
def accuracy_vs_frames(model, ds, device, batch_size, seed, frame_grid):
    """Read the causal running-mean prediction at every frame; report test top-1 vs observed frames,
    and frames-to-first-correct (mean over samples of the earliest frame after which the running
    prediction stays on the final, correct class)."""
    loader = rx.make_loader(ds, batch_size, False, seed)
    T = ds.T
    grid = [k for k in frame_grid if k <= T]
    correct = np.zeros(len(grid)); total = 0
    ttfc = []                                                        # frames-to-first-(stably)-correct
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["label"].to(device)
        present, length = batch["present"].to(device), batch["length"].to(device)
        stream = model.forward_stream(x, present=present)           # (B,T,C)
        pred = stream.argmax(-1)                                    # (B,T)
        B = x.shape[0]
        Lmax = length.amax(1).clamp(max=T)                          # last valid frame per sample
        for gi, k in enumerate(grid):
            kk = torch.clamp(Lmax - 1, max=k - 1).clamp(min=0) if k >= T else \
                torch.full((B,), k - 1, device=device)
            pk = pred.gather(1, kk.view(B, 1)).squeeze(1)
            correct[gi] += (pk == y).sum().item()
        total += B
        # frames-to-first-correct: earliest k s.t. pred stays == y through the last valid frame
        yfin = y
        for b in range(B):
            Lb = int(Lmax[b].item())
            pb = pred[b, :Lb]
            stable = (pb == yfin[b])
            # earliest index from which all remaining are correct (streaming "locks on")
            if stable.any():
                # find first index where suffix is all-correct
                rev = torch.flip(stable.to(torch.int8), [0])
                cs = torch.cummin(rev, 0).values                    # 1 while suffix stays correct
                first_from_end = int(cs.sum().item())               # length of all-correct suffix
                ttfc.append(Lb - first_from_end + 1 if first_from_end > 0 else Lb)
            else:
                ttfc.append(Lb)                                    # never locks on
    accs = (100.0 * correct / max(total, 1)).tolist()
    return {"frame_grid": grid, "acc_at_frames": accs, "n": total,
            "mean_frames_to_lock": float(np.mean(ttfc)),
            "median_frames_to_lock": float(np.median(ttfc))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="outputs/ntu_stream/egru_stream_xview_mask.pt")
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--t-target", type=int, default=100)
    ap.add_argument("--sample", choices=["causal", "uniform", "cyclic"], default="causal")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frame-grid", nargs="+", type=int,
                    default=[1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100])
    ap.add_argument("--limit-test", type=int, default=4000,
                    help="accuracy-vs-frames on a random N-sample test subset (0=full). The curve is"
                         " a per-frame accuracy trend; a few thousand samples estimate it tightly and"
                         " keep the memory-starved machine safe. Latency uses synthetic frames (full).")
    ap.add_argument("--latency-only", action="store_true")
    ap.add_argument("--out", type=str, default="outputs/ntu_stream")
    args = ap.parse_args()

    determinism.enable(seed=args.seed, strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 82)
    print("F8  TIME-TO-FIRST-SCORE  --  streaming (causal) EGRU vs the 33 ms / 30 fps budget")
    print("=" * 82)
    print(f"  device: {device}\n")

    if args.latency_only or not os.path.exists(args.ckpt):
        if not os.path.exists(args.ckpt):
            print(f"  no trained ckpt at {args.ckpt} -> timing the untrained streaming architecture "
                  f"(latency is weight-independent).\n")
        model = build_model("egru", n_classes=60, use_mask=True, streaming=True).to(device).eval()
    else:
        model = load_stream_model(args.ckpt, device)

    lat = latency(model, device)
    print("  LATENCY (genuine incremental single-frame step, hidden state carried):")
    print(f"    encoder+proj      : {lat['encoder_proj_ms']:.2f} ms")
    print(f"    GRU step+mean+head: {lat['recurrent_tail_ms']:.3f} ms")
    print(f"    per-frame TOTAL   : {lat['per_frame_ms']:.2f} ms  (p95 {lat['per_frame_ms_p95']:.2f}"
          f" ms)  -> {lat['fps']:.0f} fps")
    print(f"    33 ms/30 fps budget: {'MET' if lat['per_frame_ms'] <= lat['budget_ms'] else 'MISSED'}"
          f"  (TTFS = one step = {lat['per_frame_ms']:.2f} ms)\n")

    report = {"ckpt": args.ckpt if os.path.exists(args.ckpt) else None, "device": device,
              "latency": lat}

    if args.data and os.path.exists(args.ckpt):
        raw = nd.load_items(args.data)
        for it in raw:                                              # fp16 coords: memory-lean load
            it["keypoint"] = np.asarray(it["keypoint"], dtype=np.float16)
        te = nd.NTUXView(args.data, "test", t_target=args.t_target, sample=args.sample,
                         train=False, items=raw)
        if args.limit_test and args.limit_test < len(te.items):     # subset for a fast, lean curve
            rng = np.random.default_rng(args.seed)
            te.items = [te.items[i] for i in
                        rng.choice(len(te.items), args.limit_test, replace=False)]
            print(f"  (accuracy-vs-frames on a {len(te.items)}-sample test subset)\n")
        avf = accuracy_vs_frames(model, te, device, args.batch_size, args.seed, args.frame_grid)
        print("  ACCURACY vs OBSERVED FRAMES (causal running-mean readout):")
        for k, a in zip(avf["frame_grid"], avf["acc_at_frames"]):
            bar = "#" * int(a / 2)
            print(f"    frame {k:>3d} ({k*lat['per_frame_ms']:>6.0f} ms): {a:5.2f}%  {bar}")
        print(f"    mean frames-to-lock-on-correct: {avf['mean_frames_to_lock']:.1f} "
              f"(median {avf['median_frames_to_lock']:.0f})  = "
              f"{avf['mean_frames_to_lock']*lat['per_frame_ms']:.0f} ms to a trustworthy score")
        report["accuracy_vs_frames"] = avf
    else:
        print("  (skip accuracy-vs-frames: need --data and a trained --ckpt)")

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"ttfs_benchmark_s{args.seed}.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
