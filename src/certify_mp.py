#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
certify_mp.py
=============
Re-run the equivariance certificate against the MESSAGE-PASSING field (cde_model_mp).

This is the swap-in the brief warns about (PROJECT_BRIEF 11: "Orthogonality must survive the REAL
e3nn layers -- most likely divergence point"). The message-passing field adds three things that
are each a plausible place for the symmetry to break silently:

  * spherical harmonics up to l=2 on the bone directions (a 2e irrep now appears),
  * a per-bone radial weight net (must depend ONLY on the invariant ||r_ij||),
  * position carried INSIDE the solver-visible state (a 1o channel with d pos/ds = dX).

The last one is the subtle one: pos is a Type-1 channel of the integrated latent, so the
representation on Z must rotate it -- if we had carried position as a "scalar-like" passthrough
the state rep would not be orthogonal and the adaptive error norm argument (6.1) would collapse.
The per-joint layout [n_s x 0e | pos 1o | n_v x 1o] keeps rho(g) block-diagonal and orthogonal.

Gates are identical to certify_trainable.py. Same thresholds. No grading on a curve.

Run:  python src/certify_mp.py
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import equivariance_suite as eqs                                # noqa: E402
from cde_model_mp import SE3MessagePassingCDE, count_parameters  # noqa: E402
from certify_trainable import _polar, make_batch                # noqa: E402

from e3nn import o3                                             # noqa: E402

SEED = 0
THRESH_ORTHO = 1e-12
THRESH_DRIFT = 1e-13
THRESH_INVARIANCE = 1e-12
THRESH_PRECISION_RATIO = 1e5


def reps(model, R):
    """D on ONE joint's state ( [n_s x 0e | (1+n_v) x 1o] ) and on a single 1o vector.
    Polar-projected: e3nn's Euler-angle Wigner-D is off-orthogonal by up to ~1e-11, which would
    pollute the drift measurement with the instrument's own error (see certify_trainable)."""
    L = model.layout
    D_1o = _polar(o3.Irreps("1x1o").D_from_matrix(R).to(R.dtype))
    D_J = torch.block_diag(torch.eye(L.n_scalar, dtype=R.dtype),
                           *([D_1o] * L.n_vec_total))
    return D_J, D_1o


def rot(x, D):
    return x @ D.transpose(-1, -2)


def drift(model, t, x, D_J, D_1o, n_steps, n_readout):
    zT = model.terminal_state(t, x, n_steps=n_steps, n_readout=n_readout)
    zT_g = model.terminal_state(t, rot(x, D_1o), n_steps=n_steps, n_readout=n_readout)
    g_zT = rot(zT, D_J)                       # acts on the last (per-joint state) axis
    d_lat = ((zT_g - g_zT).norm() / (zT.norm() + 1e-300)).item()
    s = model.head(zT)
    s_g = model.head(zT_g)
    d_head = ((s_g - s).norm() / (s.norm() + 1e-300)).item()
    return d_lat, d_head


def build(dtype, J=25, seed=SEED):
    torch.manual_seed(seed)
    torch.set_default_dtype(dtype)
    m = SE3MessagePassingCDE(n_joints=J, n_scalar=32, n_vec=8, dropout=0.0,
                             n_readout=4, n_steps=16).to(dtype)
    m.eval()
    torch.set_default_dtype(torch.float32)
    return m


def main():
    print("=" * 78)
    print("EQUIVARIANCE CERTIFICATE -- MESSAGE-PASSING FIELD (cde_model_mp)")
    print("=" * 78)
    J = 25
    m64 = build(torch.float64, J)
    t64, x64 = make_batch(B=3, L=24, J=J, dtype=torch.float64)
    print(f"\nmodel: {count_parameters(m64):,} params | per-joint state "
          f"[{m64.layout.n_scalar}x0e + {m64.layout.n_vec_total}x1o] "
          f"= {m64.layout.dim} dims x {J} joints = {m64.layout.dim*J} solver-visible")
    print(f"spherical harmonics: {m64.field.irreps_sh}  |  bones: {m64.field.n_bones}")

    gen = torch.Generator().manual_seed(SEED + 1)
    res = {}

    # G1 -- orthogonality of the per-joint rep
    Rs = eqs.random_rotations(12, gen, dtype=torch.float64, theta_max=math.pi)
    worst = 0.0
    for R in Rs:
        D_J, _ = reps(m64, R)
        I = torch.eye(D_J.shape[-1], dtype=torch.float64)
        worst = max(worst, (D_J.T @ D_J - I).norm().item())
    res["G1"] = worst <= THRESH_ORTHO
    print(f"\n[G1] rep orthogonality: max ||D^T D - I||_F = {worst:.3e}  "
          f"(<= {THRESH_ORTHO:.0e})  -> {'PASS' if res['G1'] else 'FAIL'}")
    print(f"     ^ position is carried as a 1o channel, so rho(g) stays block-diagonal/orthogonal.")

    # G2 -- fixed-step RK4 exactness, step-size independent
    print(f"\n[G2] RK4 exactness (fp64), 6 rotations x 3 step counts")
    R6 = eqs.random_rotations(6, gen, dtype=torch.float64, theta_max=math.pi)
    per_h, heads = {}, []
    for n_steps in (8, 16, 48):
        ds = []
        for R in R6:
            D_J, D_1o = reps(m64, R)
            dl, dh = drift(m64, t64, x64, D_J, D_1o, n_steps, 4)
            ds.append(dl)
            heads.append(dh)
        per_h[n_steps] = sum(ds) / len(ds)
        print(f"     n_steps={n_steps:3d}   mean delta_eq = {per_h[n_steps]:.3e}")
    w2 = max(per_h.values())
    ratio = w2 / (min(per_h.values()) + 1e-300)
    res["G2"] = w2 <= THRESH_DRIFT and ratio <= 10.0
    print(f"     worst {w2:.3e} (<= {THRESH_DRIFT:.0e}); step-size max/min = {ratio:.2f}"
          f"  -> {'PASS' if res['G2'] else 'FAIL'}")

    # G3 -- read-out invariance
    wh = max(heads)
    res["G3"] = wh <= THRESH_INVARIANCE
    print(f"\n[G3] read-out invariance: worst = {wh:.3e}  (<= {THRESH_INVARIANCE:.0e})"
          f"  -> {'PASS' if res['G3'] else 'FAIL'}")
    print(f"     ^ the score is invariant even though the field now uses l=2 spherical harmonics.")

    # G4 -- drift must not grow with rotation angle
    print(f"\n[G4] drift vs rotation magnitude -- slope test")
    axis = torch.tensor([[0.3, 0.5, 0.81]], dtype=torch.float64)
    lt, ld = [], []
    for th in (0.05, 0.2, 0.5, 1.0, 2.0, 3.0):
        R = eqs.rodrigues_so3(axis, torch.tensor([th], dtype=torch.float64))[0]
        D_J, D_1o = reps(m64, R)
        d = drift(m64, t64, x64, D_J, D_1o, 16, 4)[0]
        lt.append(math.log(th))
        ld.append(math.log(d))
        print(f"     theta={th:4.2f}   delta_eq = {d:.3e}")
    n = len(lt)
    mt, md = sum(lt) / n, sum(ld) / n
    slope = (sum((a - mt) * (b - md) for a, b in zip(lt, ld))
             / sum((a - mt) ** 2 for a in lt))
    worst_a = max(math.exp(v) for v in ld)
    res["G4"] = abs(slope) <= 0.5 and worst_a <= THRESH_DRIFT
    print(f"     log-log slope = {slope:+.3f} (|slope| <= 0.5); worst {worst_a:.3e}"
          f"  -> {'PASS' if res['G4'] else 'FAIL'}")

    # G5 -- precision scaling
    m32 = build(torch.float32, J)
    t32, x32 = make_batch(B=3, L=24, J=J, dtype=torch.float32)
    R3 = eqs.random_rotations(3, gen, dtype=torch.float64, theta_max=math.pi)
    a32, a64 = [], []
    for R in R3:
        D_J, D_1o = reps(m64, R)
        a64.append(drift(m64, t64, x64, D_J, D_1o, 16, 4)[0])
        a32.append(drift(m32, t32, x32, D_J.float(), D_1o.float(), 16, 4)[0])
    f32 = sum(a32) / len(a32)
    f64 = sum(a64) / len(a64)
    r = f32 / (f64 + 1e-300)
    res["G5"] = r >= THRESH_PRECISION_RATIO
    print(f"\n[G5] precision scaling: fp32 {f32:.3e} / fp64 {f64:.3e} = {r:.3e}"
          f"  (>= {THRESH_PRECISION_RATIO:.0e})  -> {'PASS' if res['G5'] else 'FAIL'}")

    print("\n" + "=" * 78)
    for k in ("G1", "G2", "G3", "G4", "G5"):
        print(f"  {k}: {'PASS' if res[k] else 'FAIL'}")
    ok = all(res.values())
    print("=" * 78)
    print("CERTIFICATE HOLDS on the message-passing field. Training may proceed."
          if ok else "CERTIFICATE BROKEN -- pivot signal. Do NOT train through this.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
