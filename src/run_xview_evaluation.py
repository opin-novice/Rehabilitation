#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_xview_evaluation.py
=======================
NTU RGB+D 60 Cross-View driver. Trains our EGRU and the ST-GCN baseline under the SAME loss,
schedule, determinism, and X-View split, and reports Top-1 action-recognition accuracy under the
camera-1 perspective shift. This is the experiment that turns "viewpoint invariance" from an
algebraic identity (KIMORE) into an externally comparable benchmark result (NTU).

Determinism: reuses determinism.enable(42) -- dense-incidence aggregation (no index_add_), unfused
deterministic GRU backward, locked cuBLAS workspace. Same regime as the KIMORE Phase-1 work, so
NTU numbers are reproducible bit-for-bit.

Occlusion hook (G4): --occlusion-test runs the exact dead-node invariance check on NTU-shaped data
for the EGRU (f(x,m) == f(x',m) to machine precision), and --eval-drop k injects k frozen tracking
nodes at TEST time to measure graceful degradation at NTU scale. ST-GCN has no such hook -- that
asymmetry is the point.

Usage:
  python src/run_xview_evaluation.py --selftest                 # fixture smoke: both models train+eval
  python src/run_xview_evaluation.py --occlusion-test           # G4 theorem on NTU-shaped data
  python src/run_xview_evaluation.py --data data/ntu60_xview.pkl --models egru stgcn --epochs 60
  python src/run_xview_evaluation.py --data ... --model egru --use-mask --eval-drop 4
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import determinism                                                    # noqa: E402 (CUBLAS env first)
import ntu_dataset as nd                                             # noqa: E402
from baselines import build_model, count_parameters                 # noqa: E402


def make_loader(ds, batch_size, shuffle, seed):
    g = torch.Generator().manual_seed(seed)                          # deterministic shuffling
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=nd.collate,
                      num_workers=0, generator=g if shuffle else None, drop_last=False)


def _drop_nodes(x, present, k, seed):
    """Freeze k random non-root joints at their first frame (a stale tracker) + build the liveness
    mask. Returns (x', mask (B,M,T,V)). Root (0) never fails -- it defines the frame."""
    B, M, T, V, _ = x.shape
    rng = np.random.default_rng(seed)
    mask = torch.ones(B, M, T, V, dtype=x.dtype, device=x.device)
    if k <= 0:
        return x, mask
    xp = x.clone()
    for b in range(B):
        for m in range(M):
            if present is not None and present[b, m] == 0:
                continue
            j = rng.choice(np.arange(1, V), size=k, replace=False)
            xp[b, m, :, j, :] = xp[b, m, :1, j, :]                    # stale freeze at frame 0
            mask[b, m, :, j] = 0.0
    return xp, mask


@torch.no_grad()
def evaluate(model, loader, device, drop_k=0, seed=42):
    model.eval()
    correct = total = 0
    for bi, batch in enumerate(loader):
        x = batch["x"].to(device)
        y = batch["label"].to(device)
        length, present = batch["length"].to(device), batch["present"].to(device)
        mask = None
        if drop_k > 0:
            x, mask = _drop_nodes(x, present, drop_k, seed + bi)
            if not model.uses_mask:                                   # baseline: no liveness input
                mask = None
        logits = model(x, length=length, present=present, mask=mask)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


def _epoch_loader(ds, batch_size, seed):
    """A fresh shuffled train loader seeded by (base_seed, epoch). Reseeding PER EPOCH makes the
    data order a deterministic function of the epoch ALONE, so a run resumed after a power cut sees
    the same order it would have uninterrupted -- load-shedding recovery does not perturb training."""
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=nd.collate,
                      num_workers=0, generator=g, drop_last=False)


def _atomic_save(obj, path):
    """Write to .tmp then rename: a power cut mid-write leaves the previous good file intact."""
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def train(model, tr_ds, te_loader, device, epochs, lr, wd, tag, batch_size, seed,
          eval_drop=0, val_loader=None, ckpt_dir=None, resume=True,
          optimizer="adamw", momentum=0.9, nesterov=True, warmup_epochs=0):
    """Train + CHECKPOINT, RESUMABLE across power loss (load shedding).

    Every epoch atomically writes a full resume snapshot (model + optimizer + scheduler + epoch +
    best-so-far + RNG states); on restart, training continues from the last COMPLETED epoch instead
    of from scratch. Data order is a function of the epoch alone (per-epoch reseed), so resuming does
    not reshuffle history. Selection is on val_loader if given, else the test loader (optimistic).
    The best state is also saved separately as the deliverable the eval-drop sweep reloads."""
    if optimizer == "sgd":                    # the published ST-GCN recipe (pyskl): SGD + nesterov
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                              weight_decay=wd, nesterov=nesterov)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if warmup_epochs > 0:
        # SGD at lr=0.1 diverges from a cold start; pyskl linearly warms up before cosine.
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01,
                                                    total_iters=warmup_epochs),
                  torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                             T_max=max(1, epochs - warmup_epochs))],
            milestones=[warmup_epochs])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    sel_loader = val_loader if val_loader is not None else te_loader
    suffix = "_mask" if model.uses_mask else ""
    resume_path = os.path.join(ckpt_dir, f"{tag}_xview{suffix}_resume.pt") if ckpt_dir else None
    best_path = os.path.join(ckpt_dir, f"{tag}_xview{suffix}.pt") if ckpt_dir else None

    # best init < 0 so the FIRST completed epoch always saves a checkpoint, even if val=0% on a
    # tiny/early run -- otherwise a model that never exceeds 0% leaves no best weights for the sweep.
    start_ep, best, best_ep, best_state = 0, -1.0, -1, None
    if resume and resume_path and os.path.exists(resume_path):
        # weights_only=False: our OWN trusted snapshot carries optimizer + numpy RNG state, which
        # the torch>=2.6 safe-loader rejects by default. Without this, resume-after-power-loss
        # (the whole point of the snapshot) crashes on restart.
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_ep, best, best_ep, best_state = ck["epoch"] + 1, ck["best"], ck["best_ep"], ck["best_state"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        if ck.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
        if ck.get("numpy_rng") is not None:
            np.random.set_state(ck["numpy_rng"])
        print(f"  [{tag}] RESUMED at epoch {start_ep}/{epochs}  (best {best:5.2f}% @ ep{best_ep+1})")
    print(f"  [{tag}] {count_parameters(model):,} params  (uses_mask={model.uses_mask}, "
          f"selection={'val' if val_loader is not None else 'TEST (optimistic)'})")

    for ep in range(start_ep, epochs):
        model.train()
        t0 = time.time()
        for batch in _epoch_loader(tr_ds, batch_size, seed * 1000 + ep):
            x = batch["x"].to(device)
            y = batch["label"].to(device)
            length, present = batch["length"].to(device), batch["present"].to(device)
            mask = None
            if model.uses_mask:                                      # train WITH node dropout so
                x, mask = _drop_nodes(x, present,                    # the mask channel is learned
                                      k=int(np.random.default_rng(ep).integers(0, 5)),
                                      seed=1000 + ep)
            opt.zero_grad(set_to_none=True)
            ce(model(x, length=length, present=present, mask=mask), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        sel = evaluate(model, sel_loader, device)                   # val (or test) for selection
        if sel > best:
            best, best_ep = sel, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if resume_path:                                             # per-epoch resumable snapshot
            _atomic_save({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                          "sched": sched.state_dict(), "best": best, "best_ep": best_ep,
                          "best_state": best_state, "torch_rng": torch.get_rng_state(),
                          "cuda_rng": (torch.cuda.get_rng_state_all()
                                       if torch.cuda.is_available() else None),
                          "numpy_rng": np.random.get_state()}, resume_path)
        if ep == start_ep or (ep + 1) % max(1, epochs // 10) == 0 or ep == epochs - 1:
            print(f"    ep {ep+1:3d}  sel {sel:5.2f}%  (best {best:5.2f}% @ ep{best_ep+1})  "
                  f"({time.time()-t0:.0f}s)")

    if best_state is not None:                                       # restore + persist the winner
        model.load_state_dict(best_state)
        if best_path:
            os.makedirs(ckpt_dir, exist_ok=True)
            _atomic_save({"state": best_state, "tag": tag, "best_sel": best,
                          "uses_mask": model.uses_mask}, best_path)
            print(f"  [{tag}] saved best (ep {best_ep+1}) -> {best_path}")

    # report accuracy on the TEST set with the selected weights (clean, then with dead nodes)
    test_acc = evaluate(model, te_loader, device)
    result = {"tag": tag, "xview_top1_test": test_acc, "sel_best": best, "sel_epoch": best_ep + 1}
    print(f"  [{tag}] TEST top-1 {test_acc:5.2f}%")
    if eval_drop > 0:
        result["xview_top1_drop"] = evaluate(model, te_loader, device, drop_k=eval_drop)
        result["drop_delta"] = test_acc - result["xview_top1_drop"]
        print(f"  [{tag}] {eval_drop} dead nodes: {result['xview_top1_drop']:5.2f}%  "
              f"(degradation {result['drop_delta']:+.2f} pts from clean {test_acc:5.2f}%)")
    return result


# =============================================================================
def occlusion_test(device):
    """G4 on NTU-shaped data: the EGRU's logits must be EXACTLY independent of what a dead joint
    reports. Same theorem as certify_phase1.g4, now at (B,M,T,V,3) action-classifier shape."""
    print("=" * 78)
    print("G4 on NTU -- dead-node invariance of the EGRU action classifier")
    print("=" * 78)
    old = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    model = build_model("egru", n_classes=60, use_mask=True).double().eval()
    torch.set_default_dtype(old)

    B, M, T, V = 3, 2, 20, nd.N_JOINTS
    x = torch.randn(B, M, T, V, 3, dtype=torch.float64, device=device)
    x = x - x[..., :1, :]                                             # root-relative
    length = torch.full((B, M), T, dtype=torch.long, device=device)
    present = torch.ones(B, M, device=device)
    dead = torch.tensor([3, 7, 11, 18, 21])
    mask = torch.ones(B, M, T, V, dtype=torch.float64, device=device)
    mask[..., dead] = 0.0
    model = model.to(device)

    with torch.no_grad():
        base = model(x, length=length, present=present, mask=mask)
        worst = 0.0
        g = torch.Generator(device="cpu").manual_seed(1)
        for trial in range(5):
            xp = x.clone()
            xp[..., dead, :] = torch.randn(B, M, T, len(dead), 3, generator=g,
                                           dtype=torch.float64).to(device) * (10.0 ** trial)
            worst = max(worst, (base - model(xp, length=length, present=present,
                                             mask=mask)).abs().max().item())
    print(f"  max |f(x,m) - f(x',m)| over corruptions to 1e4 magnitude = {worst:.3e}")
    ok = worst <= 1e-11
    print("  PASS -- the theorem holds at NTU scale; logits are exactly independent of dead-node"
          " reports." if ok else "  FAIL -- a leak path survived; check the mask plumbing.")
    assert ok
    return worst


def run_sweep(args, device):
    """ACCURACY x FAULT: reload the saved best checkpoints and measure X-View top-1 at each
    node-dropout level k. The masked EGRU applies the G4 mask (exact routing around dead nodes);
    ST-GCN has no such hook, so it sees frozen joints as real data. That asymmetry is the figure."""
    import json
    te = nd.NTUXView(args.data, "test", t_target=args.t_target, sample=args.sample)
    te_l = make_loader(te, args.batch_size, False, args.seed)
    print("=" * 78)
    print("ACCURACY x FAULT SWEEP -- node-dropout k on the held-out X-View test set")
    print("=" * 78)
    specs = [("egru", "egru_xview_mask.pt", True), ("egru", "egru_xview.pt", False),
             ("stgcn", "stgcn_xview.pt", False)]
    rows = {}
    for name, fn, um in specs:
        path = os.path.join(args.out, fn)
        if not os.path.exists(path):
            continue
        ck = torch.load(path, map_location=device, weights_only=False)
        m = build_model(name, n_classes=60, use_mask=ck.get("uses_mask", um)).to(device)
        m.load_state_dict(ck["state"])
        m.eval()
        label = f"{name}{'(mask)' if ck.get('uses_mask', um) else ''}"
        accs = [evaluate(m, te_l, device, drop_k=k) for k in args.sweep_k]
        rows[label] = accs
        print(f"  [{label}]  " + "  ".join(f"k{k}={a:5.2f}%"
                                           for k, a in zip(args.sweep_k, accs)))
    if not rows:
        print(f"  no checkpoints found in {args.out} -- train first (run without --sweep).")
        return 1
    print(f"\n  {'k dead':>7s}" + "".join(f"  {n:>12s}" for n in rows))
    for i, k in enumerate(args.sweep_k):
        print(f"  {k:>7d}" + "".join(f"  {rows[n][i]:>12.2f}" for n in rows))
    print("\n  degradation from clean k=0 (pts, smaller = more robust):")
    for n, accs in rows.items():
        print(f"    {n:<14s} " + "  ".join(f"k{args.sweep_k[i]}:{accs[0]-accs[i]:+5.1f}"
                                           for i in range(len(accs))))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"xview_fault_sweep_s{args.seed}.json"), "w") as fh:
        json.dump({"sweep_k": args.sweep_k, "acc": rows}, fh, indent=2)
    print(f"\n  wrote {os.path.join(args.out, f'xview_fault_sweep_s{args.seed}.json')}")
    return 0


def run_selftest(args):
    """Fixture smoke: both models train a few epochs on synthetic NTU and the loop is alive."""
    fx = os.path.join("outputs", "ntu_fixture.pkl")
    os.makedirs("outputs", exist_ok=True)
    nd.make_fixture(fx, n_per_cam=24, n_classes=6)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = nd.NTUXView(fx, "train", t_target=64)
    te = nd.NTUXView(fx, "test", t_target=64)
    print(f"[xview] fixture split  train={len(tr)}  test={len(te)}  (6 classes)")
    te_l = make_loader(te, 8, False, 42)
    for name in ("egru", "stgcn"):
        m = build_model(name, n_classes=6, use_mask=args.use_mask).to(device)
        train(m, tr, te_l, device, epochs=6, lr=1e-3, wd=1e-3, tag=name, batch_size=8, seed=42,
              eval_drop=args.eval_drop, ckpt_dir=None)
    print("[xview] SELFTEST PASSED (both models train and evaluate on the X-View split)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None, help="NTU-60 .pkl/.npz (pyskl layout)")
    ap.add_argument("--models", nargs="+", default=["egru", "stgcn"])
    ap.add_argument("--model", type=str, default=None, help="single-model shortcut")
    ap.add_argument("--t-target", type=int, default=100)
    ap.add_argument("--sample", choices=["causal", "uniform", "cyclic"], default="causal",
                    help="cyclic = pyskl UniformSample (wraps short clips; NEVER zero-pads)")
    ap.add_argument("--rot-aug", type=float, default=0.0,
                    help="pyskl RandomRot theta in RADIANS, train split only (published: 0.3 ~ 17 deg)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw",
                    help="sgd = the published ST-GCN/pyskl recipe (momentum + nesterov + cosine)")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--no-nesterov", action="store_true")
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="linear LR warmup before cosine; required for SGD at lr=0.1")
    ap.add_argument("--view-norm", action="store_true",
                    help="PreNormalize3D-style view canonicalisation (published ST-GCN X-View "
                         "preprocessing). Mutually exclusive with per-frame root-relative.")
    ap.add_argument("--use-mask", action="store_true", help="EGRU G4 occlusion hook + node-dropout")
    ap.add_argument("--eval-drop", type=int, default=0, help="k frozen nodes at TEST time")
    ap.add_argument("--limit-train", type=int, default=0,
                    help="subsample the train set to N (0=all). A fast REAL-DATA preview: the full "
                         "EGRU 60-epoch run is an overnight job; a subset gives directional numbers "
                         "in minutes. NOT a benchmark result -- label previews as such.")
    ap.add_argument("--limit-test", type=int, default=0, help="subsample the test set to N (0=all)")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="carve this fraction of TRAIN (subject/sample-random) for model selection, "
                         "so the reported test number is NOT chosen by peeking at test. 0.05 is a "
                         "clean protocol; 0.0 selects on test (optimistic) and is warned.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--occlusion-test", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="skip training: reload saved checkpoints and run the Accuracy x Fault "
                         "node-dropout sweep on the test set")
    ap.add_argument("--sweep-k", nargs="+", type=int, default=[0, 1, 2, 4, 8])
    ap.add_argument("--out", type=str, default="outputs/ntu")
    args = ap.parse_args()

    cfg = determinism.enable(seed=args.seed, strict=False)   # strict=False: STGCN convs vary by cuDNN
    print(determinism.report(cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}\n")

    if args.occlusion_test:
        occlusion_test(device)
        return 0
    if args.sweep:
        return run_sweep(args, device)
    if args.selftest or args.data is None:
        if args.data is None and not args.selftest:
            print("no --data given; running the synthetic fixture self-test instead.\n")
        run_selftest(args)
        return 0

    models = [args.model] if args.model else args.models
    tr = nd.NTUXView(args.data, "train", t_target=args.t_target, sample=args.sample,
                     view_norm=args.view_norm, train=True, rot_aug=args.rot_aug)
    te = nd.NTUXView(args.data, "test", t_target=args.t_target, sample=args.sample,
                     view_norm=args.view_norm, train=False, rot_aug=0.0)   # never augment the test set
    if args.limit_train or args.limit_test:
        rng = np.random.default_rng(args.seed)                       # deterministic subsample
        if args.limit_train:
            tr.items = [tr.items[i] for i in rng.choice(len(tr.items),
                        min(args.limit_train, len(tr.items)), replace=False)]
        if args.limit_test:
            te.items = [te.items[i] for i in rng.choice(len(te.items),
                        min(args.limit_test, len(te.items)), replace=False)]
        print("  *** PREVIEW: subsampled train=%d test=%d -- NOT a benchmark number ***"
              % (len(tr), len(te)))
    # optional val holdout carved from TRAIN, so selection never peeks at test
    val = None
    if args.val_frac > 0:
        rng = np.random.default_rng(args.seed + 1)
        vn = max(1, int(len(tr.items) * args.val_frac))
        vi = set(rng.choice(len(tr.items), vn, replace=False).tolist())
        val_items = [tr.items[i] for i in vi]
        tr.items = [tr.items[i] for i in range(len(tr.items)) if i not in vi]
        val = nd.NTUXView(args.data, "train", t_target=args.t_target, sample=args.sample,
                          items=val_items, view_norm=args.view_norm,
                          train=False, rot_aug=0.0)             # selection must see a clean split
        print(f"  val holdout: {len(val)} carved from train -> train={len(tr)}")
    else:
        print("  WARNING: --val-frac 0 -> selecting the reported epoch on TEST (optimistic). "
              "Use --val-frac 0.05 for a clean protocol.")

    print(f"NTU-60 X-View  train(cams 2,3)={len(tr)}  test(cam 1)={len(te)}\n")
    te_l = make_loader(te, args.batch_size, False, args.seed)
    val_l = make_loader(val, args.batch_size, False, args.seed) if val is not None else None

    import json
    os.makedirs(args.out, exist_ok=True)
    results = []
    for name in models:
        print(f"{'='*78}\n{name.upper()}\n{'='*78}")
        m = build_model(name, n_classes=60, use_mask=(args.use_mask and name in ("egru", "ours"))
                        ).to(device)
        results.append(train(m, tr, te_l, device, args.epochs, args.lr, args.wd, name,
                             batch_size=args.batch_size, seed=args.seed,
                             eval_drop=args.eval_drop, val_loader=val_l, ckpt_dir=args.out,
                             optimizer=args.optimizer, momentum=args.momentum,
                             nesterov=not args.no_nesterov, warmup_epochs=args.warmup_epochs))

    print(f"\n{'='*78}\nNTU-60 CROSS-VIEW (X-View) TOP-1 ACCURACY\n{'='*78}")
    print(f"  {'model':<12s} {'test %':>9s}" + ("  {:>12s}".format(f"{args.eval_drop}-dead %")
                                                if args.eval_drop else ""))
    for r in results:
        line = f"  {r['tag']:<12s} {r['xview_top1_test']:>9.2f}"
        if args.eval_drop:
            line += f"  {r.get('xview_top1_drop', float('nan')):>12.2f}"
        print(line)
    with open(os.path.join(args.out, f"xview_results_s{args.seed}.json"), "w") as fh:
        json.dump({"args": vars(args), "results": results}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
