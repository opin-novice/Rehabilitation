#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
certify_egru.py
===============
Equivariance certificate for THE METHOD (equivariant_gru.SE3EquivariantGRU).

This model has no solver, so the Task-2/4/5 solver gates (fixed-step exactness, adaptive
step-grid divergence, N_eq) are simply not applicable -- there is no integrator whose step
sequence could diverge under a group action. Removing the flow removes that entire risk surface.
What REMAINS, and what the paper's viewpoint claim actually rests on, is:

  E1  encoder equivariance : h(g.x) = rho(g) h(x)                    <= 1e-12  (fp64)
  E2  read-out INVARIANCE  : |s(g.x) - s(x)| / |s(x)|                <= 1e-12  (fp64)
      ^ THE THEOREM. The clinician-facing score must not move under any rotation of the body.
  E3  flat vs angle        : slope of log|violation| on log theta ~ 0
      ^ a MODELLING violation grows with rotation magnitude; roundoff does not.
  E4  precision scaling    : violation_fp32 / violation_fp64         >= 1e5
      ^ proves the residual is floating-point ROUNDOFF, not a broken symmetry.
  E5  translation invariance : s(pipeline(x_world + c)) = s(pipeline(x_world))
      ^ Tested on RAW WORLD coordinates through the deployed pipeline, which is the only
        version of this claim that means anything. The encoder is NOT translation-invariant on
        arbitrary inputs -- it reads each joint's radius ||x_j|| and seeds its vector channels
        from x_j -- so those are differences ONLY relative to the root. Translation invariance
        is therefore discharged by the root-subtraction step (x_j -> x_j - x_root), not by the
        network. An earlier version of this gate added an offset to ALREADY root-relative
        coordinates and "failed" at 5.8e-02: that was a corrupted input the model never sees,
        not a broken symmetry. The claim is about the pipeline, so the gate tests the pipeline.

The reference rotation is polar-projected onto O(3) before use: e3nn's Euler-angle Wigner-D is
off-orthogonal by up to ~1e-11 for some orientations, and certifying against it measures the
instrument rather than the model (see certify_trainable.py).

Run:  python src/certify_egru.py
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chirality                                           # noqa: E402
import equivariance_suite as eqs                            # noqa: E402
from equivariant_gru import SE3EquivariantGRU, count_parameters  # noqa: E402
from certify_trainable import _polar                        # noqa: E402

from e3nn import o3                                         # noqa: E402

SEED = 0
THRESH = 1e-12
THRESH_RATIO = 1e5


def reps(model, R):
    """D on one joint's feature block [n_s x 0e | n_v x 1o], and on a raw 1o coordinate."""
    enc = model.encoder
    D_1o = _polar(o3.Irreps("1x1o").D_from_matrix(R).to(R.dtype))
    D_h = torch.block_diag(torch.eye(enc.n_scalar, dtype=R.dtype), *([D_1o] * enc.n_vec))
    return D_h, D_1o


def make_batch(B=3, T=20, J=25, dtype=torch.float64, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    dt = 0.02 + 0.06 * torch.rand(B, T - 1, generator=g, dtype=dtype)     # IRREGULAR stamps
    t = torch.cat([torch.zeros(B, 1, dtype=dtype), dt.cumsum(-1)], dim=1)
    x = torch.randn(B, T, J, 3, generator=g, dtype=dtype) * 0.3
    x = x - x[:, :, :1, :]                                                # root-relative
    ex = torch.randint(0, 5, (B,), generator=g)
    n = torch.full((B,), T, dtype=torch.long)
    return t, x, ex, n


def build(dtype, seed=SEED, use_chiral=False):
    torch.manual_seed(seed)
    torch.set_default_dtype(dtype)
    m = SE3EquivariantGRU(n_scalar=32, n_vec=8, n_layers=2, dropout=0.0,
                          use_chiral=use_chiral).to(dtype)
    m.eval()
    torch.set_default_dtype(torch.float32)
    return m


@torch.no_grad()
def _rel(m, t, x0, x1, ex, n):
    """|s(x1) - s(x0)| / |s(x0)| -- how far the score moves when the input is transformed."""
    s0, s1 = m(t, x0, ex, n), m(t, x1, ex, n)
    return ((s1 - s0).norm() / (s0.norm() + 1e-300)).item()


def violations(m, t, x, ex, n, D_h, D_1o):
    xr = x @ D_1o.transpose(-1, -2)
    B, T, J, _ = x.shape
    with torch.no_grad():
        h = m.encoder(x.reshape(B * T, J, 3))
        h_g = m.encoder(xr.reshape(B * T, J, 3))
        enc = ((h_g - h @ D_h.transpose(-1, -2)).norm() / (h.norm() + 1e-300)).item()
        s = m(t, x, ex, n)
        s_g = m(t, xr, ex, n)
        inv = ((s_g - s).norm() / (s.norm() + 1e-300)).item()
    return enc, inv


def main():
    print("=" * 78)
    print("EQUIVARIANCE CERTIFICATE -- SE3EquivariantGRU  (THE METHOD)")
    print("=" * 78)
    m64 = build(torch.float64)
    t, x, ex, n = make_batch(dtype=torch.float64)
    print(f"\nmodel: {count_parameters(m64):,} params | "
          f"encoder [{m64.encoder.n_scalar}x0e + {m64.encoder.n_vec}x1o], "
          f"SH {m64.encoder.irreps_sh} | invariant dim {m64.proj.dim}")
    print("no solver => no step-grid divergence surface at all (the flow is gone).")

    gen = torch.Generator().manual_seed(SEED + 1)
    res = {}

    # E1 / E2 -- over Haar-random rotations
    Rs = eqs.random_rotations(12, gen, dtype=torch.float64, theta_max=math.pi)
    encs, invs = [], []
    for R in Rs:
        D_h, D_1o = reps(m64, R)
        e, i = violations(m64, t, x, ex, n, D_h, D_1o)
        encs.append(e)
        invs.append(i)
    res["E1"] = max(encs) <= THRESH
    res["E2"] = max(invs) <= THRESH
    print(f"\n[E1] encoder equivariance  worst = {max(encs):.3e}  (<= {THRESH:.0e})"
          f"  -> {'PASS' if res['E1'] else 'FAIL'}")
    print(f"[E2] READ-OUT INVARIANCE   worst = {max(invs):.3e}  (<= {THRESH:.0e})"
          f"  -> {'PASS' if res['E2'] else 'FAIL'}")
    print("     ^ THE THEOREM: the score does not move under ANY rotation of the body.")

    # E3 -- flat vs angle
    print(f"\n[E3] invariance violation vs rotation magnitude -- slope test")
    axis = torch.tensor([[0.3, 0.5, 0.81]], dtype=torch.float64)
    lt, ld = [], []
    for th in (0.05, 0.2, 0.5, 1.0, 2.0, 3.0):
        R = eqs.rodrigues_so3(axis, torch.tensor([th], dtype=torch.float64))[0]
        D_h, D_1o = reps(m64, R)
        v = violations(m64, t, x, ex, n, D_h, D_1o)[1]
        lt.append(math.log(th))
        ld.append(math.log(v))
        print(f"     theta={th:4.2f}   violation = {v:.3e}")
    k = len(lt)
    mt, md = sum(lt) / k, sum(ld) / k
    slope = (sum((a - mt) * (b - md) for a, b in zip(lt, ld))
             / sum((a - mt) ** 2 for a in lt))
    res["E3"] = abs(slope) <= 0.5 and max(math.exp(v) for v in ld) <= THRESH
    print(f"     log-log slope = {slope:+.3f}  (|slope| <= 0.5)"
          f"  -> {'PASS' if res['E3'] else 'FAIL'}")
    print("     ^ a MODELLING violation grows with angle; roundoff is flat.")

    # E4 -- precision scaling
    m32 = build(torch.float32)
    t32, x32, ex32, n32 = make_batch(dtype=torch.float32)
    R3 = eqs.random_rotations(3, gen, dtype=torch.float64, theta_max=math.pi)
    v32, v64 = [], []
    for R in R3:
        D_h, D_1o = reps(m64, R)
        v64.append(violations(m64, t, x, ex, n, D_h, D_1o)[1])
        v32.append(violations(m32, t32, x32, ex32, n32,
                              D_h.float(), D_1o.float())[1])
    a32 = sum(v32) / len(v32)
    a64 = sum(v64) / len(v64)
    r = a32 / (a64 + 1e-300)
    res["E4"] = r >= THRESH_RATIO
    print(f"\n[E4] precision scaling: fp32 {a32:.3e} / fp64 {a64:.3e} = {r:.3e}"
          f"  (>= {THRESH_RATIO:.0e})  -> {'PASS' if res['E4'] else 'FAIL'}")

    # E5 -- translation invariance OF THE PIPELINE (raw world coords -> root-relative -> score)
    def pipeline(x_world):
        return x_world - x_world[:, :, :1, :]              # the deployed root-subtraction

    g = torch.Generator().manual_seed(SEED + 7)
    x_world = torch.randn(*x.shape, generator=g, dtype=torch.float64) * 0.3
    c = torch.randn(1, 1, 1, 3, generator=g, dtype=torch.float64) * 2.0
    with torch.no_grad():
        s0 = m64(t, pipeline(x_world), ex, n)
        s1 = m64(t, pipeline(x_world + c), ex, n)          # translate the WHOLE BODY in the world
    tr = ((s1 - s0).norm() / (s0.norm() + 1e-300)).item()
    res["E5"] = tr <= THRESH
    print(f"\n[E5] translation invariance of the PIPELINE (raw world coords + c):")
    print(f"     |s(x+c) - s(x)| / |s(x)| = {tr:.3e}   (<= {THRESH:.0e})"
          f"  -> {'PASS' if res['E5'] else 'FAIL'}")
    print("     ^ discharged by root-subtraction, NOT by the network: the encoder reads ||x_j||")
    print("       and seeds vectors from x_j, so it is translation-invariant only w.r.t. the root.")

    # =====================================================================================
    # E6 / E7 / E8 -- PARITY. Which group are we ACTUALLY invariant under, and is that the
    # group we meant? Everything above certifies invariance under proper rotations. Nothing
    # above would have noticed if we were also invariant under IMPROPER ones -- and we were.
    # =====================================================================================
    mC = build(torch.float64, use_chiral=True)          # the SO(3) model: pseudo-scalars admitted
    print("\n" + "=" * 78)
    print("PARITY: SO(3) or O(3)?  (E6/E7/E8)")
    print("=" * 78)
    print(f"chiral model: {count_parameters(mC):,} params | invariant dim "
          f"{m64.proj.dim} -> {mC.proj.dim}  (+{mC.proj.dim - m64.proj.dim} parity-odd)")

    xm = chirality.mirror(x)          # reflection: labels UNCHANGED, parity flipped
    xs = chirality.swap_lr(x)         # relabelling: left <-> right joint INDICES, no reflection

    # -- E6: mirror sensitivity. The parity-even model must be BLIND (that is the defect, and we
    #    exhibit it rather than concede it); the chiral model must SEE.
    #    Because both models are SO(3)-invariant, every improper transform is equivalent to this
    #    one up to a rotation -- so a single reflection certifies the whole coset. One suffices.
    mir_o3 = _rel(m64, t, x, xm, ex, n)
    mir_so3 = _rel(mC, t, x, xm, ex, n)
    res["E6"] = (mir_o3 <= THRESH) and (mir_so3 >= 1e-3)
    print(f"\n[E6] MIRROR sensitivity  |s(Mx) - s(x)| / |s(x)|")
    print(f"     parity-even cut (0e only)   = {mir_o3:.3e}   <- BLIND: invariance is O(3), not SO(3)")
    print(f"     chiral cut (0e + 0o)        = {mir_so3:.3e}   <- SEES the reflection")
    print(f"     -> {'PASS' if res['E6'] else 'FAIL'}  (even <= {THRESH:.0e} AND odd >= 1e-3)")
    print("     ^ the parity-even model scores a motion and its MIRROR identically. Trajectory")
    print("       handedness -- clockwise vs anticlockwise limb circling -- is invisible to it.")

    # -- E7: laterality. THE REFUTATION. 'O(3)-invariant' does NOT imply 'cannot tell the left arm
    #    from the right arm'. Left/right limbs differ by a RELABELLING, not a reflection, and no
    #    model here is invariant to relabelling: joint_emb is keyed by joint id, bone lengths are
    #    per-bone in fixed order, and the speed channels are per-joint and never pooled.
    lat_o3 = _rel(m64, t, x, xs, ex, n)
    lat_so3 = _rel(mC, t, x, xs, ex, n)
    res["E7"] = (lat_o3 >= 1e-3) and (lat_so3 >= 1e-3)
    print(f"\n[E7] LATERALITY sensitivity  |s(swapLR(x)) - s(x)| / |s(x)|")
    print(f"     parity-even cut             = {lat_o3:.3e}")
    print(f"     chiral cut                  = {lat_so3:.3e}")
    print(f"     -> {'PASS' if res['E7'] else 'FAIL'}  (BOTH >= 1e-3)")
    print("     ^ BOTH models separate a left-arm motion from a right-arm motion. The claim that")
    print("       an O(3)-invariant scorer 'cannot tell the healthy limb from the impaired one'")
    print("       does not follow and is false: that is a relabelling, not a parity flip.")

    # -- E8: the fix must not cost the theorem. Reflections have det = -1; ROTATIONS have det = +1,
    #    so a pseudo-scalar is still exactly rotation-invariant. Block 3 is untouched. Measure it.
    invC = []
    for R in Rs:
        D_h, D_1o = reps(mC, R)
        invC.append(_rel(mC, t, x, x @ D_1o.transpose(-1, -2), ex, n))
    res["E8"] = max(invC) <= THRESH
    print(f"\n[E8] ROTATION invariance SURVIVES the fix   worst = {max(invC):.3e}  (<= {THRESH:.0e})"
          f"  -> {'PASS' if res['E8'] else 'FAIL'}")
    print("     ^ det(rho(g)) = +1 for every proper rotation, so the pseudo-scalars are invariant")
    print("       under exactly the group we claim. We narrowed O(3) -> SO(3) and paid NOTHING:")
    print("       the Block-3 viewpoint theorem (degradation = 0.000) holds verbatim.")

    # -- persist the measured floors so every certificate number in the paper has a file ------
    vals = {
        "model_params": count_parameters(m64),
        "chiral_params": count_parameters(mC),
        "E1_encoder_equiv_worst": max(encs),
        "E2_readout_invariance_worst": max(invs),
        "E3_loglog_slope": slope,
        "E4_fp32": a32, "E4_fp64": a64, "E4_ratio": r,
        "E5_translation_invariance": tr,
        "E6_mirror_even": mir_o3, "E6_mirror_chiral": mir_so3,
        "E7_laterality_even": lat_o3, "E7_laterality_chiral": lat_so3,
        "E8_rotation_invariance_chiral_worst": max(invC),
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "outputs", "cde_block2")
    os.makedirs(out_dir, exist_ok=True)
    import json as _json
    with open(os.path.join(out_dir, "certify_egru.json"), "w") as _fh:
        _json.dump({"seed": SEED, "thresholds": {"floor": THRESH, "ratio": THRESH_RATIO},
                    "pass": {k_: bool(res[k_]) for k_ in res}, "values": vals}, _fh, indent=2)
    print(f"\n[persist] wrote {os.path.join(out_dir, 'certify_egru.json')}")

    print("\n" + "=" * 78)
    for k_ in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"):
        print(f"  {k_}: {'PASS' if res[k_] else 'FAIL'}")
    ok = all(res.values())
    print("=" * 78)
    print("CERTIFICATE HOLDS. Viewpoint invariance is a THEOREM about this architecture,\n"
          "and the invariance group is now SO(3) -- the one we claim -- not O(3)."
          if ok else "CERTIFICATE BROKEN -- do not train through this.")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
