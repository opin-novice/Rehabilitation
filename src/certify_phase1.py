#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
certify_phase1.py
=================
PHASE 1 GATE. Four properties, each a pass/fail assert. A failed gate is a PIVOT SIGNAL: stop and
diagnose, do not paper over it.

  G1  EQUIVALENCE      dense-incidence aggregation == the old index_add_ scatter, to fp tolerance.
                       This is what licenses "existing checkpoints remain valid". If it fails, the
                       whole published Block 2/3/4 table is invalidated and must be re-run.

  G2  DETERMINISM      repeated forward passes are BITWISE identical, and one training step twice
                       gives bitwise identical gradients. This is the claim "+/-0.33 MAD was never
                       hardware"; it must be MEASURED, not asserted.

  G3  EQUIVARIANCE     s(Rx) == s(x) for random R in SO(3) -- in BOTH mask modes. The mask is a
                       Type-0 scalar so it SHOULD be free, but "should" is what a certificate is
                       for. This is the load-bearing theorem of the paper; the occlusion fix is not
                       allowed to cost it.

  G4  DEAD-NODE        f(x, m) == f(x', m) for any x, x' differing ONLY at joints where m = 0.
      INVARIANCE       i.e. the score is EXACTLY independent of what a failed sensor reports.
                       This is strictly stronger than "degrades gracefully" -- it is a property PCT
                       cannot have at any capacity, and it is the honest headline for Block 4.
                       Every path from x_j to the output must be gated:
                         (a) h_0 init            -> dead embedding (zero 1o channels)
                         (b) incident edges      -> e_live
                         (c) proj bone lengths   -> endpoint gate
                         (d) proj pooled feats   -> masked mean / masked amax
                         (e) speed/displacement  -> liveness gate
                         (f) anatomic pseudo-scalars -> chirality.frame_liveness
                       A G4 failure NAMES the path we missed. That is the point of running it.

Run:  python src/certify_phase1.py            # fp64, cpu
      python src/certify_phase1.py --cuda     # the configuration we actually train in
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import determinism                                            # noqa: E402  (sets CUBLAS env first)
from equivariant_gru import SE3EquivariantGRU                 # noqa: E402
from e3nn import o3                                           # noqa: E402

J = 25


def rand_batch(B, T, device, dtype, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(B, T, J, 3, generator=g, dtype=dtype).to(device)
    x = x - x[:, :, :1, :]                                    # root-relative, as the loader emits
    t = torch.cumsum(torch.rand(B, T, generator=g, dtype=dtype) * 0.05 + 0.01, dim=1).to(device)
    e = torch.randint(0, 5, (B,), generator=g).to(device)
    n = torch.full((B,), T, dtype=torch.long, device=device)
    return t, x, e, n


def build(device, dtype, use_mask, use_chiral=False, seed=0):
    """Construct under the TARGET default dtype -- this is not a stylistic detail.

    e3nn materialises Clebsch-Gordan coefficients and internal weight buffers in
    torch.get_default_dtype() AT CONSTRUCTION TIME. Building under the fp32 default and then
    calling .to(float64) upcasts the STORAGE but cannot recover bits already rounded away: the
    'fp64' run is then silently fp32-limited. Measured, on this model:

        built under fp32 default, cast to fp64 : cut drift 2.8e-08   <- a FAKE fp64 certificate
        built under fp64 default               : cut drift 8.5e-15   <- the real one

    A 3.3e6x difference, and it would have made an EXACT symmetry look like a 1e-8 violation --
    i.e. it would have manufactured a fake architectural break and sent us hunting for it.
    (certify_egru.build already does this correctly, at line 77; this harness did not.)
    """
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        torch.manual_seed(seed)
        m = SE3EquivariantGRU(n_exercises=5, dropout=0.0,
                              use_chiral=use_chiral, use_mask=use_mask)
    finally:
        torch.set_default_dtype(old)
    return m.to(device=device, dtype=dtype).eval()


# =============================================================================
def g1_equivalence(device, dtype):
    """The dense incidence matmul must compute the SAME SUM the scatter did."""
    print("\n[G1] dense incidence  ==  index_add_ scatter")
    m = build(device, dtype, use_mask=False)
    enc = m.encoder
    N, C = 7, 64
    msg = torch.randn(N, enc.n_edges, C, device=device, dtype=dtype)

    ref = torch.zeros(N, J, C, device=device, dtype=dtype)     # the ORIGINAL implementation
    ref.index_add_(1, enc.dst, msg)
    new = torch.einsum("je,nec->njc", enc.incidence, msg)

    err = (ref - new).abs().max().item()
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    print(f"     max |scatter - einsum| = {err:.3e}   (tol {tol:.0e})")
    assert err <= tol, "G1 FAILED: aggregation changed -- checkpoints are INVALID."
    print("     PASS -- aggregation is unchanged, existing checkpoints remain valid.")


def g2_determinism(device, dtype):
    """Repeated forwards bitwise identical; repeated backward gradients bitwise identical."""
    print("\n[G2] determinism (bitwise)")
    m = build(device, dtype, use_mask=False)
    t, x, e, n = rand_batch(4, 12, device, dtype)

    with torch.no_grad():
        outs = [m(t, x, e, n) for _ in range(8)]
    spread = max((o - outs[0]).abs().max().item() for o in outs)
    print(f"     forward:  max spread over 8 repeats = {spread:.3e}")
    assert spread == 0.0, "G2 FAILED (forward): a nondeterministic kernel remains."

    def grad_once():
        m.zero_grad(set_to_none=True)
        m(t, x, e, n).sum().backward()
        return torch.cat([p.grad.reshape(-1) for p in m.parameters() if p.grad is not None])

    ga, gb = grad_once(), grad_once()
    gspread = (ga - gb).abs().max().item()
    print(f"     backward: max |grad_a - grad_b|      = {gspread:.3e}")
    assert gspread == 0.0, "G2 FAILED (backward): cuDNN fused RNN or another atomic op remains."
    print("     PASS -- the +/-0.33 MAD 'hardware floor' was atomics, and it is gone.")


def g3_equivariance(device, dtype, use_mask, use_chiral):
    """s(Rx) == s(x) for random R in SO(3). The mask must not cost the theorem."""
    tag = f"use_mask={use_mask}, use_chiral={use_chiral}"
    print(f"\n[G3] SO(3) invariance of the score  ({tag})")
    m = build(device, dtype, use_mask=use_mask, use_chiral=use_chiral)
    t, x, e, n = rand_batch(4, 12, device, dtype)

    mask = None
    if use_mask:                                              # a NON-trivial mask, not all-ones
        g = torch.Generator(device="cpu").manual_seed(3)
        mask = (torch.rand(4, 12, J, generator=g, dtype=dtype) > 0.25).to(dtype).to(device)
        mask[..., 0] = 1.0                                    # the root defines the frame

    worst = 0.0
    with torch.no_grad():
        base = m(t, x, e, n, mask) if use_mask else m(t, x, e, n)
        for k in range(8):
            R = o3.rand_matrix(dtype=dtype).to(device)
            xr = torch.einsum("btjc,dc->btjd", x, R)
            rot = m(t, xr, e, n, mask) if use_mask else m(t, xr, e, n)
            worst = max(worst, (rot - base).abs().max().item())
    # Thresholds follow PROJECT_BRIEF Task 2 (delta_eq <= 1e-13 in fp64). With the model built
    # under the correct default dtype the score drift lands at ~1e-16, i.e. AT fp64 roundoff, so
    # 1e-13 is a real gate with three orders of headroom rather than a number we tuned to pass.
    tol = 1e-13 if dtype == torch.float64 else 1e-4
    print(f"     max_R |s(Rx) - s(x)| = {worst:.3e}   (tol {tol:.0e})")
    assert worst <= tol, f"G3 FAILED ({tag}): the occlusion fix broke the viewpoint theorem."
    print("     PASS -- viewpoint invariance survives.")


def g4_dead_node(device, dtype, use_chiral):
    """THE THEOREM: the score cannot depend on what a dead sensor reports."""
    print(f"\n[G4] dead-node invariance  f(x,m) == f(x',m)   (use_chiral={use_chiral})")
    m = build(device, dtype, use_mask=True, use_chiral=use_chiral)
    t, x, e, n = rand_batch(4, 12, device, dtype)

    g = torch.Generator(device="cpu").manual_seed(11)
    mask = torch.ones(4, 12, J, dtype=dtype)
    dead = torch.tensor([3, 7, 11, 18, 21])                   # 5 failed tracking nodes (not root)
    mask[..., dead] = 0.0
    mask = mask.to(device)

    with torch.no_grad():
        a = m(t, x, e, n, mask)
        worst = 0.0
        for trial in range(5):
            xp = x.clone()
            # Anything at all: a stale freeze, a collapse to the root, wild garbage, huge values.
            noise = torch.randn(4, 12, len(dead), 3, generator=g, dtype=dtype).to(device)
            xp[:, :, dead, :] = noise * (10.0 ** trial)       # up to 1e4 -- absurd sensor output
            b = m(t, xp, e, n, mask)
            worst = max(worst, (a - b).abs().max().item())

    tol = 1e-13 if dtype == torch.float64 else 1e-5
    print(f"     max |f(x,m) - f(x',m)| over 5 corruptions (to 1e4 magnitude) = {worst:.3e}")
    print(f"     (tol {tol:.0e})")
    if worst > tol:
        print("     G4 FAILED -- a path from x_j to the output is UNGATED. Check, in order:")
        print("       (a) h_0 dead embedding   (b) e_live edge gate   (c) bone endpoints")
        print("       (d) masked mean/amax     (e) speed channels     (f) frame_liveness")
    assert worst <= tol, "G4 FAILED: the dead-node theorem does not hold."
    print("     PASS -- the score is EXACTLY independent of the failed sensor's report.")
    print("     This is a property PCT cannot have at any capacity.")


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true", help="certify on the device we TRAIN on")
    ap.add_argument("--fp32", action="store_true", help="fp32 instead of fp64")
    args = ap.parse_args()

    cfg = determinism.enable(seed=0, strict=True)
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    dtype = torch.float32 if args.fp32 else torch.float64

    print("=" * 78)
    print(f"PHASE 1 CERTIFICATE   device={device}  dtype={str(dtype).split('.')[-1]}")
    print("=" * 78)
    print(determinism.report(cfg))

    g1_equivalence(device, dtype)
    g2_determinism(device, dtype)
    g3_equivariance(device, dtype, use_mask=False, use_chiral=False)
    g3_equivariance(device, dtype, use_mask=True, use_chiral=False)
    g3_equivariance(device, dtype, use_mask=True, use_chiral=True)
    g4_dead_node(device, dtype, use_chiral=False)
    g4_dead_node(device, dtype, use_chiral=True)

    print("\n" + "=" * 78)
    print("ALL PHASE-1 GATES GREEN.")
    print("  G1 checkpoints valid | G2 noise floor -> 0 | G3 theorem intact | G4 theorem NEW")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
