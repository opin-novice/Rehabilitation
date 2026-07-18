#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
int8_quant_budget.py  --  F7c COMPLETION: the INT8 fake-quantisation cliff.
===========================================================================
precision_budget.py measured the equivariance floor across NATIVE float grids (fp64/fp32/fp16/
bf16) and found a clean monotonic degradation matching the mantissa. e3nn has NO native int8
kernel path, so int8 cannot be measured the way the float rows were (there is no set_default_dtype
for int8, no CG table in int8). The faithful -- and only honest -- way to place an int8 point on
the same chart is SIMULATED fake-quantisation: round weights and activations onto an int8 grid
with a learned per-tensor scale, dequantise back to fp32, and run the SAME deployed clinical model
through the SAME viewpoint probe.

WHY THIS IS NOT A TAUTOLOGY, AND WHY THE CLIFF IS ATTRIBUTABLE
-------------------------------------------------------------
The naive experiment -- "quantise everything to int8, report one broken number" -- would be
uninterpretable, because it conflates two effects that are structurally DIFFERENT:

  * WEIGHT quantisation does NOT break the viewpoint theorem. The encoder is equivariant for ANY
    weights (equivariant_gru.py:76): the tensor product is Clebsch-Gordan-constrained, o3.Linear is
    equivariant for any weight matrix, the radial MLPs emit INVARIANT scalar weights from invariant
    inputs, and init_gain scales the type-1 channels by learned SCALARS. Quantised weights are just
    a DIFFERENT equivariant model. So s(g.x) = s(x) still holds to fp32 roundoff.

  * ACTIVATION quantisation DOES break it, structurally. Rounding a type-1 (vector) channel onto a
    fixed int8 grid is basis-dependent: round(rho(g).v) != rho(g).round(v). Likewise rounding the
    input joints AFTER a viewpoint change: round(R x) != R round(x). This non-commutativity with
    rho(g) is the real cliff -- it is a property of the GRID, not of the trained weights.

So we measure FOUR rows and let them decompose the cliff themselves:

    row     weights  activations   what it proves
    ------  -------  -----------   ---------------------------------------------------------
    fp32     fp32       fp32       the reference floor (matches precision_budget.py: 2.68e-7)
    W8       int8       fp32       CONTROL: theorem SURVIVES weight quant (stays ~fp32)
    A8       fp32       int8       the cliff is an ACTIVATION-quant phenomenon
    W8A8     int8       int8       the true edge-deployment point on the F7c chart

The W8 control is what turns "int8 breaks equivariance" (a bare assertion) into "int8 breaks
equivariance SPECIFICALLY through activation grid-rounding, and leaves the SE(3) theorem intact
under weight quantisation" (a measured, attributable statement).

THE "LEARNED SCALE"
-------------------
Per-tensor symmetric int8 (qmax = 127, zero-point 0). The scale is LEARNED BY CALIBRATION: a grid
search over clipping fraction that MINIMISES quantisation MSE on real calibration data (weights:
the tensor itself; activations: a static calibration pass over real KIMORE poses in the CANONICAL
orientation). This is standard post-training-quantisation calibration -- a learned scale, fit to
data, not an SGD-trained parameter. Activation scales are FROZEN after calibration, so the SAME
grid quantises the pose and its rotated view; the cliff is then purely the grid's non-commutativity
with rho(g), not a per-input rescaling artefact.

THE QUANTISATION SURFACE (documented, not hidden)
-------------------------------------------------
  weights    : every nn.Parameter of the deployed model (encoder TP/radial/linear, GRU, head).
  activations: layer-boundary tensors -- (1) input joint coordinates fed to the encoder AND the
               projection, quantised AFTER the viewpoint rotation (the true deployment order);
               (2) the per-frame equivariant feature map h (this is where the type-1 channels get
               grid-rounded -> the structural break); (3) the invariant projection output that
               feeds the recurrence. This is a representative static-PTQ granularity; it is not a
               claim that every intermediate TP accumulator is int8.

Two numbers per row, exactly as precision_budget.measure:
  inv_floor = max_g ||inv(g.x)-inv(x)|| / ||inv(x)||   -- the arithmetic equivariance floor (the
              real cliff; multi-dim, does not saturate).
  score_dx  = max_g |s(g.x)-s(x)| * 50                 -- the reported clinical-score shift in
              CLINICAL POINTS == MAD units on the 0-50 KIMORE scale (compare to the ~6.5 MAD floor
              and PCT's 3.03 MAD rotation swing).

Run:  python src/int8_quant_budget.py        (default: 40 seqs x 64 azimuths, 8 calib seqs)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kimore_cde_data as kd                       # noqa: E402
import precision_budget as pb                      # noqa: E402  (build_at, azimuths, measure)

QMAX = 127                                         # symmetric int8: [-127, 127], zero-point 0
CALIB_CAP = 500_000                                # cap the MSE-search sample per tensor
SCORE_MAX = pb.SCORE_MAX


# ---------------------------------------------------------------------------------------------
# learned (MSE-calibrated) symmetric int8 scale + fake-quant
# ---------------------------------------------------------------------------------------------
def mse_scale(flat, qmax=QMAX, n_grid=64, cap=CALIB_CAP):
    """Per-tensor symmetric scale that MINIMISES round-trip MSE over a clip-fraction grid.

    This is the 'learned scale': a calibration search, not an SGD parameter. clip fraction < 1
    trades clipping error against resolution error -- the MSE-optimal point is the standard PTQ
    choice and is what a deployment toolkit's calibrator lands on."""
    flat = flat.detach().reshape(-1).float()
    amax = flat.abs().max()
    if not torch.isfinite(amax) or amax == 0:
        return torch.tensor(1.0)
    if flat.numel() > cap:
        idx = torch.randperm(flat.numel())[:cap]
        flat = flat[idx]
    best_s, best_e = amax / qmax, float("inf")
    for frac in torch.linspace(0.4, 1.0, n_grid):
        s = (amax * frac) / qmax
        if s <= 0:
            continue
        q = torch.clamp(torch.round(flat / s), -qmax, qmax) * s
        e = (q - flat).pow(2).mean().item()
        if e < best_e:
            best_e, best_s = e, s
    return best_s if torch.is_tensor(best_s) else torch.tensor(best_s)


def fake_quant(t, scale, qmax=QMAX):
    """Quantise onto the int8 grid and dequantise back (simulated int8 in fp32)."""
    return torch.clamp(torch.round(t / scale), -qmax, qmax) * scale


class ActQuant:
    """A calibratable activation quantiser. Three modes:
        'off'   -> pass-through (identity), used for the fp32 / W8 rows.
        'calib' -> record activations to fit the scale later, pass-through.
        'quant' -> quantise with the FROZEN learned scale.
    Shared across hook sites that see the same distribution (the coord quantiser is used by both
    the encoder-input and projection-input hooks)."""
    def __init__(self, name):
        self.name = name
        self.mode = "off"
        self.scale = None
        self._buf = []

    def observe(self, t):
        self._buf.append(t.detach().reshape(-1).float().cpu())

    def learn(self):
        cat = torch.cat(self._buf) if self._buf else torch.tensor([0.0])
        self.scale = mse_scale(cat)
        self._buf = []

    def apply(self, t):
        if self.mode == "off":
            return t
        if self.mode == "calib":
            self.observe(t)
            return t
        return fake_quant(t.float(), self.scale.to(t.device)).to(t.dtype)


# ---------------------------------------------------------------------------------------------
# hook wiring: quantise input coords (post-rotation), the equivariant feature map, and the invariants
# ---------------------------------------------------------------------------------------------
def attach_hooks(model, q_in, q_enc, q_proj):
    handles = []

    def enc_pre(module, args):
        # args = (x,) or (x, mask). Quantise the input coords AFTER the viewpoint rotation.
        xq = q_in.apply(args[0])
        return (xq,) + tuple(args[1:])

    def enc_post(module, args, output):
        return q_enc.apply(output)                 # the per-frame equivariant feature map h

    def proj_pre(module, args):
        # proj.forward(h, x, mask=None): quantise the SAME coords it reads for bone lengths.
        if len(args) >= 2:
            return (args[0], q_in.apply(args[1])) + tuple(args[2:])
        return args

    def proj_post(module, args, output):
        return q_proj.apply(output)                # the invariant projection feeding the recurrence

    handles.append(model.encoder.register_forward_pre_hook(enc_pre))
    handles.append(model.encoder.register_forward_hook(enc_post, with_kwargs=False))
    handles.append(model.proj.register_forward_pre_hook(proj_pre))
    handles.append(model.proj.register_forward_hook(proj_post, with_kwargs=False))
    return handles


def set_mode(quantisers, mode):
    for q in quantisers:
        q.mode = mode


def quantize_weights_(model):
    """Fake-quantise EVERY parameter in place (per-tensor MSE-learned int8 scale)."""
    with torch.no_grad():
        for _, p in model.named_parameters():
            s = mse_scale(p.data)
            p.data.copy_(fake_quant(p.data.float(), s).to(p.data.dtype))


# ---------------------------------------------------------------------------------------------
# one row
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def run_variant(sd, seqs, R, R_id, quant_w, quant_a, calib_n):
    """Build the fp32 model, optionally int8 its weights, optionally int8 its activations
    (calibrated on the canonical-orientation poses), then measure inv_floor + score_dx."""
    model = pb.build_at(torch.float32, sd)         # a FRESH model per variant (no shared state)
    if quant_w:
        quantize_weights_(model)

    q_in = ActQuant("coords")
    q_enc = ActQuant("feat_h")
    q_proj = ActQuant("inv")
    quantisers = [q_in, q_enc, q_proj]
    handles = attach_hooks(model, q_in, q_enc, q_proj)

    if quant_a:
        # STATIC calibration on the CANONICAL orientation (R_id): identity rotation => xr == x, so
        # the scales reflect the real pose distribution, not a rotated one. Then FREEZE.
        set_mode(quantisers, "calib")
        for s in seqs[:calib_n]:
            pb.measure(model, s, R_id, torch.float32)
        for q in quantisers:
            q.learn()
        set_mode(quantisers, "quant")
    else:
        set_mode(quantisers, "off")

    invf, sdx = zip(*[pb.measure(model, s, R, torch.float32) for s in seqs])
    for h in handles:
        h.remove()
    invf, sdx = np.array(invf), np.array(sdx)
    scales = {q.name: (float(q.scale) if q.scale is not None else None) for q in quantisers}
    return float(np.nanmax(invf)), float(np.nanmax(sdx)), scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seqs", type=int, default=40)
    ap.add_argument("--n-azimuth", type=int, default=64)
    ap.add_argument("--calib-seqs", type=int, default=8)
    ap.add_argument("--ckpt", default=pb.CKPT)
    args = ap.parse_args()

    torch.manual_seed(0)
    print("=" * 92)
    print("F7c  INT8 FAKE-QUANTISATION CLIFF  --  deployed clinical model, real KIMORE motion")
    print("=" * 92)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "state" in sd:
        sd = sd["state"]

    samples = kd.load_all_exercises(max_len=150, verbose=False)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(samples), min(args.n_seqs, len(samples)), replace=False)
    seqs = []
    for i in pick:
        s = samples[i]
        seqs.append({"t": torch.as_tensor(np.asarray(s["t"]), dtype=torch.float64),
                     "x": torch.as_tensor(np.asarray(s["x"]), dtype=torch.float64),
                     "exercise": int(s["exercise"])})
    print(f"inputs: {len(seqs)} real sequences | {args.n_azimuth} azimuths | "
          f"{args.calib_seqs} calibration seqs")
    print(f"weights: {args.ckpt}  (trained; y in [0,1] -> x{SCORE_MAX:.0f} = clinical points)")
    print("int8: per-tensor symmetric, qmax=127, MSE-learned scale; activations static-calibrated "
          "in the canonical pose\n")

    R = pb.azimuths(args.n_azimuth, torch.float32)
    R_id = torch.eye(3, dtype=torch.float32).unsqueeze(0)      # identity: calibrate on real poses

    variants = [("fp32", False, False), ("W8", True, False),
                ("A8", False, True), ("W8A8", True, True)]

    print(f"{'variant':>7} | {'weights':>7} | {'acts':>5} | {'INV floor (rel)':>16} | "
          f"{'score dx (clin pts)':>19} | verdict")
    print("-" * 92)
    rows = {}
    fp32_floor = None
    for name, qw, qa in variants:
        imax, smax, scales = run_variant(sd, seqs, R, R_id, qw, qa, args.calib_seqs)
        if name == "fp32":
            fp32_floor = imax
        # a row PRESERVES the theorem if its floor stays within ~10x of the fp32 arithmetic floor;
        # it has fallen off the CLIFF if the floor jumps by orders of magnitude (structural break).
        preserved = fp32_floor is not None and imax <= 10 * fp32_floor
        if name == "fp32":
            verd = "reference"
        elif preserved:
            verd = "THEOREM HOLDS"
        else:
            ratio = imax / (fp32_floor + 1e-300)
            verd = f"CLIFF x{ratio:.0e}" + ("  UNSAFE" if smax > 0.5 else "")
        print(f"{name:>7} | {str(qw):>7} | {str(qa):>5} | {imax:16.3e} | {smax:19.3e} | {verd}")
        rows[name] = dict(quant_weights=qw, quant_acts=qa, inv_floor=imax,
                          score_dx=smax, act_scales=scales)

    print("-" * 92)
    print("INV floor = max_g ||inv(g.x)-inv(x)||/||inv(x)||: the arithmetic equivariance floor.")
    print("W8 stays at the fp32 floor => weight int8 leaves the SE(3) theorem INTACT (equivariance")
    print("  holds for any weights). A8/W8A8 jump => the cliff is ACTIVATION grid-rounding: rounding")
    print("  the type-1 channels / rotated joints onto a fixed grid does not commute with rho(g).")
    print("score dx in CLINICAL POINTS = MAD units on the 0-50 scale (floor ~6.5, PCT swing 3.03).")

    os.makedirs("outputs/precision_budget", exist_ok=True)
    out = "outputs/precision_budget/f7c_int8_quant.json"
    with open(out, "w") as fh:
        json.dump({"ckpt": args.ckpt, "n_seqs": len(seqs), "n_azimuth": args.n_azimuth,
                   "calib_seqs": args.calib_seqs, "score_max": SCORE_MAX, "qmax": QMAX,
                   "fp32_floor": fp32_floor, "rows": rows}, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
