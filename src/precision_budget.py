#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
precision_budget.py  --  F7c: the EQUIVARIANCE PRECISION BUDGET.
================================================================
Quantisation is NOT an orthogonal transformation, so it does not commute with the Wigner-D action
rho(g).  The viewpoint theorem (s(g.x) = s(x), exact in real arithmetic) therefore has a NUMERICAL
floor on edge hardware that nobody has published for e3nn.  This measures it, on the DEPLOYED
clinical model, in the units a clinician reads.

    delta_eq(pi) = max_{g in SO(3)} | s_pi(g.x) - s_pi(x) |    (score-shift under a pure viewpoint
                                                                change, computed entirely at pi)

reported in CLINICAL POINTS on the 0-50 KIMORE scale (the model emits y = score/50 in [0,1], so a
score-shift is multiplied by SCORE_MAX = 50), directly comparable to the ~6.5 MAD clinical error
floor and to PCT's 3.03 MAD rotation swing.

WHY THIS IS A REAL NUMBER AND NOT A TAUTOLOGY:
  * TRAINED weights (egru_pooled_f0.pt) -> the [0,1] output is calibrated, so x50 is genuine
    clinical points.  Random weights would put x50 on an arbitrary scale wearing a 'MAD' label.
  * REAL KIMORE test motion -> not a synthetic stress the failure model was invented around.
  * NATIVE per-precision construction: set_default_dtype(pi) BEFORE building the module, so the
    Clebsch-Gordan coefficients are generated INSIDE the pi grid.  A post-hoc .half()/.to(pi) cast
    leaves the CG tables in fp32 and yields a FAKE certificate (this repo has logged 2.8e-8 fake
    vs 8.5e-15 real for exactly that mistake).  The CG GARBAGE-CHECK below proves the fp16/bf16
    tables did not silently underflow to nonsense.
  * The rotation and the coordinates are BOTH cast to pi (an fp16 deployment rotates fp16 joints
    with an fp16 matrix) -- so this is the true deployment scenario, not a fp64 rotation of fp16
    data.  The fp64 row is the exactness anchor (~1e-11 clinical pts = the theorem).

Run:  python src/precision_budget.py            (default: 40 seqs x 64 azimuths x 4 precisions)
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimore_cde_data as kd                                   # noqa: E402
from equivariant_gru import SE3EquivariantGRU                  # noqa: E402
from certify_trainable import _polar                           # noqa: E402

SCORE_MAX = 50.0                                               # y = score / 50  (kimore_cde_data)
CKPT = "outputs/cde_block2/egru_pooled_f0.pt"
UP_AXIS = 1                                                    # rotate about vertical (azimuth/yaw)

# architecture inferred from the checkpoint (in_dim 216 = proj160 + 1 dt + 5 ex + 50 speed)
ARCH = dict(n_scalar=32, n_vec=8, n_layers=2, lmax=2, gru_hidden=128, head_hidden=128,
            dropout=0.0, n_exercises=5, use_speed=True, use_chiral=False)


def build_at(dtype, ckpt_sd):
    """Construct the model NATIVELY at `dtype` (CG tables built in this precision), THEN load the
    fp32-trained weights (copy_ casts them to `dtype` -- exactly what deploying at `dtype` does)."""
    prev = torch.get_default_dtype()
    torch.manual_seed(0)
    torch.set_default_dtype(dtype)
    m = SE3EquivariantGRU(**ARCH).to(dtype)
    torch.set_default_dtype(prev)
    info = m.load_state_dict(ckpt_sd, strict=False)            # fp32 -> dtype cast on copy_
    # this checkpoint predates encoder.dead_scalar, which is INERT unless mask is not None (see
    # equivariant_gru.py:180-183) -- and we never pass a mask. Anything ELSE missing/unexpected is
    # a real architecture mismatch and must not be silently absorbed.
    allowed_missing = {"encoder.dead_scalar"}
    bad_missing = set(info.missing_keys) - allowed_missing
    if bad_missing or info.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch beyond dead_scalar: missing={bad_missing} "
                           f"unexpected={info.unexpected_keys}")
    m.eval()
    return m


def azimuths(n, dtype):
    """n rotation matrices about the vertical axis, evenly spaced in [0, 2pi), polar-projected onto
    O(3) at fp64 so we measure the MODEL, not our matrix's off-orthogonality, then cast to dtype."""
    R = torch.zeros(n, 3, 3, dtype=torch.float64)
    ang = torch.linspace(0, 2 * math.pi, n + 1, dtype=torch.float64)[:n]
    a = [i for i in range(3) if i != UP_AXIS]
    R[:, UP_AXIS, UP_AXIS] = 1.0
    R[:, a[0], a[0]] = torch.cos(ang); R[:, a[0], a[1]] = -torch.sin(ang)
    R[:, a[1], a[0]] = torch.sin(ang); R[:, a[1], a[1]] = torch.cos(ang)
    R = torch.stack([_polar(r) for r in R])
    return R.to(dtype)


@torch.no_grad()
def cg_garbage_check(m_lp, m64, x64, dtype):
    """Encoder-feature relative error of the low-precision build vs fp64, on real input.
    If the CG tables had underflowed to garbage this is O(1); a HEALTHY build sits at ~eps(dtype)."""
    J = x64.shape[-2]
    h64 = m64.encoder(x64.reshape(-1, J, 3))
    hlp = m_lp.encoder(x64.to(dtype).reshape(-1, J, 3)).to(torch.float64)
    rel = (hlp - h64).norm().item() / (h64.norm().item() + 1e-300)
    eps = torch.finfo(dtype).eps
    return rel, eps, rel < max(1e3 * eps, 1e-2)                # healthy if within ~1000 eps


@torch.no_grad()
def measure(m, seq, R, dtype):
    """Two DISTINCT numbers for one sequence, both computed at `dtype`, both read out in fp64 so the
    MEASUREMENT adds no error beyond the arithmetic-under-test:

      inv_floor  = max_g ||inv_pi(g.x) - inv_pi(x)|| / ||inv_pi(x)||   -- the TRUE arithmetic
                   equivariance floor, on the 160-d invariant projection.  Multi-dimensional, full
                   mantissa resolution, so it does NOT saturate.  This is the precision-budget curve.

      score_dx   = max_g | s_pi(g.x) - s_pi(x) |  x SCORE_MAX          -- the deployment consequence:
                   how far the REPORTED clinical score moves.  Limited BELOW by the fp16/bf16 output
                   quantum, so at low precision it can round to exactly 0 -- which means 'the shift is
                   finer than the displayed score can resolve', NOT 'the model is exactly equivariant'.
    """
    t = seq["t"].to(dtype); x = seq["x"].to(dtype)             # (T,), (T,J,3)
    ex = torch.tensor([seq["exercise"] - 1]); n = torch.tensor([x.shape[0]])
    T, J = x.shape[0], x.shape[1]
    N = R.shape[0]
    xr = torch.einsum("nij,tvj->ntvi", R, x)                   # rotate coords, per rotation

    # -- invariant-feature floor (the real number) --
    inv0 = m.proj(m.encoder(x.reshape(T, J, 3)), x.reshape(T, J, 3)).double()          # (T,D)
    invr = m.proj(m.encoder(xr.reshape(N * T, J, 3)),
                  xr.reshape(N * T, J, 3)).double().reshape(N, T, -1)                    # (N,T,D)
    if not (torch.isfinite(inv0).all() and torch.isfinite(invr).all()):
        return float("nan"), float("nan")
    num = (invr - inv0.unsqueeze(0)).reshape(N, -1).norm(dim=1)
    inv_floor = (num / (inv0.norm() + 1e-300)).max().item()

    # -- reported-score shift (the deployment consequence) --
    s0 = m(t[None], x[None], ex, n).double()
    sr = m(t[None].expand(N, -1), xr, ex.expand(N), n.expand(N)).double()
    score_dx = (sr - s0).abs().max().item() * SCORE_MAX if torch.isfinite(sr).all() else float("nan")
    return inv_floor, score_dx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seqs", type=int, default=40)
    ap.add_argument("--n-azimuth", type=int, default=64)
    ap.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args()

    print("=" * 82)
    print("F7c  EQUIVARIANCE PRECISION BUDGET  --  deployed clinical model, real KIMORE motion")
    print("=" * 82)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "state" in sd:
        sd = sd["state"]

    # ---- real inputs: pooled KIMORE test motion (fp64 base) --------------------------------
    samples = kd.load_all_exercises(max_len=150, verbose=False)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(samples), min(args.n_seqs, len(samples)), replace=False)
    seqs = []
    for i in pick:
        s = samples[i]
        seqs.append({"t": torch.as_tensor(np.asarray(s["t"]), dtype=torch.float64),
                     "x": torch.as_tensor(np.asarray(s["x"]), dtype=torch.float64),
                     "exercise": int(s["exercise"])})
    print(f"inputs: {len(seqs)} real sequences | {args.n_azimuth} azimuths about the vertical axis")
    print(f"weights: {args.ckpt}  (trained; output y in [0,1] -> x{SCORE_MAX:.0f} = clinical points)\n")

    m64 = build_at(torch.float64, sd)
    x_probe = seqs[0]["x"]
    R64 = azimuths(args.n_azimuth, torch.float64)

    precisions = [("fp64", torch.float64), ("fp32", torch.float32),
                  ("fp16", torch.float16), ("bf16", torch.bfloat16)]
    def out_quantum(dt):                                       # displayed-score resolution near 0.85
        # torch.finfo.eps == 2**-mantissa_bits (fp32 1.2e-7, fp16 9.8e-4, bf16 7.8e-3); the ULP of a
        # score ~0.85 is ~0.85*eps. (NB: it is torch.finfo -- no .nmant attribute; that is numpy's.)
        return 0.85 * torch.finfo(dt).eps * SCORE_MAX          # clinical pts per representable step

    print(f"{'prec':>5} | {'CG check vs fp64':>18} | {'INV floor (rel)':>16} | "
          f"{'score dx (clin pts)':>19} | {'out-Q':>7} | verdict")
    print("-" * 92)
    rows = {}
    for name, dt in precisions:
        try:
            m = m64 if dt == torch.float64 else build_at(dt, sd)
            rel, eps, ok = cg_garbage_check(m, m64, x_probe, dt)
            Rp = R64.to(dt)
            invf, sdx = zip(*[measure(m, s, Rp, dt) for s in seqs])
            invf, sdx = np.array(invf), np.array(sdx)
            imax = float(np.nanmax(invf)); smax = float(np.nanmax(sdx))
            q = out_quantum(dt)
            cg = f"{rel:.1e} {'OK' if ok else 'GARBAGE'}"
            # the invariant floor is the arithmetic truth; the score shift is quantum-limited
            sflag = f"{smax:.2e}" + (" (<Q)" if np.isfinite(smax) and smax < q else "")
            verd = ("EXACT" if (name == "fp64" and imax < 1e-9)
                    else "GARBAGE-CG" if not ok
                    else "BROKEN" if not np.isfinite(imax)
                    else "safe<Q" if smax < q
                    else "UNSAFE" if smax > 0.5 else "deployable")
            print(f"{name:>5} | {cg:>18} | {imax:16.3e} | {sflag:>19} | {q:7.1e} | {verd}")
            rows[name] = dict(cg_rel=rel, cg_ok=ok, inv_floor=imax, score_dx=smax, out_quantum=q)
        except Exception as e:
            print(f"{name:>5} | {'forward failed':>18} | {'--':>16} | {'--':>19} | {'--':>7} | "
                  f"{type(e).__name__}: {str(e)[:20]}")
            rows[name] = dict(error=f"{type(e).__name__}: {e}")

    print("-" * 92)
    print("INV floor = max_g ||inv(g.x)-inv(x)||/||inv(x)|| on the 160-d invariant projection: the")
    print("  TRUE arithmetic equivariance floor -- multi-dim, full mantissa, does NOT saturate.")
    print("score dx = shift of the REPORTED 0-50 score. '(<Q)' = below the fp16/bf16 output quantum")
    print("  (out-Q), i.e. the residual violation is finer than the displayed score can resolve --")
    print("  a GOOD deployment property, NOT exact equivariance. 'GARBAGE-CG' = CG table underflowed.")

    import json
    os.makedirs("outputs/precision_budget", exist_ok=True)
    out = "outputs/precision_budget/f7c_precision_budget.json"
    with open(out, "w") as fh:
        json.dump({"ckpt": args.ckpt, "n_seqs": len(seqs), "n_azimuth": args.n_azimuth,
                   "score_max": SCORE_MAX, "rows": rows}, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
