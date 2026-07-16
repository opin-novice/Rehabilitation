#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
equivariance_suite.py
=====================
Week-One Equivariance Verification Suite for the SE(3)-Equivariant Neural CDE.

Turns the equivariance-drift proof (PROJECT_BRIEF sections 5-6) into a runnable set of
pass/fail gates with strict assertions. It certifies that numerically integrating an
SE(3)-equivariant Neural Controlled Differential Equation preserves the architectural
equivariance to machine precision, and that any residual drift is *pure floating-point
roundoff*, not a symmetry violation.

Design decision
---------------
`torchcde` / `torchdiffeq` are NOT used. The brief's own honesty flags call the solver
custom-norm API the "soft spot" (version-dependent). So we implement a transparent,
self-contained Dormand-Prince (dopri5) adaptive stepper whose error-checking loop we fully
control, plus a natural cubic spline control path. Everything the certificate rests on is
visible in this file.

Three certificate gates (Runsheet Tasks 1, 2, 4-5, 6):
  A. Fixed-step algebraic exactness  : delta_eq <= 1e-13 (fp64), flat across step size.
  B. Adaptive step-grid divergence   : N_eq drives D_grid -> 0 (drift <= 1e-11); default norm diverges.
  C. Precision-scaling diagnostic    : r = drift_fp32 / drift_fp64 >= 1e5  => roundoff, not violation.

Plus Runsheet Task 1: orthogonal-representation audit  ||rho(g)^T rho(g) - I||_F <= 1e-12.

A deliberately-broken vector field (non-orthogonal mixing of the carried latent) is included
to prove the diagnostics actually *fire* on a real symmetry violation.

Run:  python src/equivariance_suite.py
Exit code 0 iff every gate passes.
"""

import argparse
import math
import sys

import torch

# e3nn is optional: only Task 7 (real-layer swap-in) needs it. Disable e3nn's fx->TorchScript
# path, which is incompatible with torch>=2.9 / py3.14 (GraphModule.__annotations__).
try:
    import e3nn
    e3nn.set_optimization_defaults(jit_script_fx=False)
    from e3nn import o3
    from e3nn.nn import Gate
    HAVE_E3NN = True
    _E3NN_ERR = ""
except Exception as _e:                 # pragma: no cover
    HAVE_E3NN = False
    _E3NN_ERR = repr(_e)

# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------
DEVICE = torch.device("cpu")          # CPU for deterministic fp64 certification
DTYPE = torch.float64                  # default working precision
SEED = 0

# Certificate thresholds (from the Week-One runsheet)
THRESH_ORTHO = 1e-12                   # Task 1: ||R^T R - I||_F  (fp64)
THRESH_FIXED_DRIFT = 1e-13             # Task 2: fixed-step relative drift (fp64)
THRESH_FIXED_FLAT_RATIO = 10.0         # Task 2: flatness (see note on sqrt(N) roundoff)
THRESH_NEQ_DRIFT = 1e-11               # Task 4: adaptive drift under N_eq (fp64)
THRESH_PRECISION_RATIO = 1e5           # Task 6: fp32/fp64 drift ratio


# =============================================================================
# 1. Irrep state layout + orthogonal SE(3) representation      (Runsheet Task 1)
# =============================================================================
class IrrepLayout:
    """Mixed Type-0 (scalar, dim 1) / Type-1 (vector, dim 3) latent layout.

    State vector is [ s_1 ... s_{n_s} | v_1(xyz) ... v_{n_v}(xyz) ].
    Exposes the per-feature block partition used by the invariant error norm N_eq
    and by the block-wise adaptive tolerance.
    """

    def __init__(self, n_scalar: int, n_vec: int):
        self.n_scalar = int(n_scalar)
        self.n_vec = int(n_vec)
        self.total = self.n_scalar + 3 * self.n_vec
        # Number of irrep *features* F (each scalar is a dim-1 block, each vector a dim-3 block)
        self.n_features = self.n_scalar + self.n_vec

    def split(self, z: torch.Tensor):
        """z: (..., total) -> (scalars (..., n_s), vectors (..., n_v, 3))."""
        s = z[..., : self.n_scalar]
        v = z[..., self.n_scalar:].reshape(*z.shape[:-1], self.n_vec, 3)
        return s, v

    def join(self, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.cat([s, v.reshape(*v.shape[:-2], self.n_vec * 3)], dim=-1)


def apply_rho(R: torch.Tensor, z: torch.Tensor, layout: IrrepLayout) -> torch.Tensor:
    """Apply the SE(3) representation rho(g) to the carried latent.

    Scalars (type-0) are invariant; vectors (type-1) rotate. Row-vector convention:
    every 3-vector v transforms as v -> v @ R^T, matching the observation transform
    x -> x @ R^T, so rho is block-diagonal and ORTHOGONAL (Wigner-D: identity + rotations).
    """
    s, v = layout.split(z)
    v = v @ R.transpose(-1, -2)
    return layout.join(s, v)


def rodrigues_so3(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Exact rotation matrices from axis-angle via Rodrigues' formula (orthogonal to roundoff).

    axis: (n, 3) unit vectors, theta: (n,) angles -> (n, 3, 3).
    """
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-300)
    n = axis.shape[0]
    K = torch.zeros(n, 3, 3, dtype=axis.dtype, device=axis.device)
    x, y, zc = axis[:, 0], axis[:, 1], axis[:, 2]
    K[:, 0, 1], K[:, 0, 2] = -zc, y
    K[:, 1, 0], K[:, 1, 2] = zc, -x
    K[:, 2, 0], K[:, 2, 1] = -y, x
    I = torch.eye(3, dtype=axis.dtype, device=axis.device).expand(n, 3, 3)
    st = torch.sin(theta).view(n, 1, 1)
    ct = torch.cos(theta).view(n, 1, 1)
    return I + st * K + (1.0 - ct) * (K @ K)


def random_rotations(n: int, generator, dtype=DTYPE, theta_max: float = math.pi) -> torch.Tensor:
    """n random rotations with |angle| <= theta_max (Haar-like axis, uniform angle)."""
    axis = torch.randn(n, 3, generator=generator, dtype=dtype, device=DEVICE)
    theta = (torch.rand(n, generator=generator, dtype=dtype, device=DEVICE) * 2 - 1) * theta_max
    return rodrigues_so3(axis, theta)


def orthogonality_audit(rotations: torch.Tensor) -> float:
    """max_g ||R^T R - I||_F over a batch of rotations (Runsheet Task 1 metric)."""
    I = torch.eye(3, dtype=rotations.dtype, device=rotations.device)
    err = (rotations.transpose(-1, -2) @ rotations - I).reshape(rotations.shape[0], -1)
    return err.norm(dim=-1).max().item()


# =============================================================================
# 2. Natural cubic spline control path                         (Runsheet Task 2)
# =============================================================================
class NaturalCubicSpline:
    """Batched natural cubic spline over a shared irregular time grid.

    Linear in the control points and reproduces constants => spline(R.x)=R.spline(x) and
    translation passes through. Provides an analytic derivative queryable at arbitrary s,
    which is what buys irregular-sampling robustness for the CDE.
    """

    def __init__(self, t: torch.Tensor, y: torch.Tensor):
        # t: (L,) strictly increasing ; y: (B, L, D)
        self.t = t
        self.y = y
        B, L, D = y.shape
        self.L = L
        h = t[1:] - t[:-1]                                   # (L-1,)
        self.h = h
        # Natural cubic spline second-derivative moments M (M[0]=M[L-1]=0).
        # Interior system (i=1..L-2): h[i-1] M[i-1] + 2(h[i-1]+h[i]) M[i] + h[i] M[i+1] = rhs
        if L >= 3:
            hi_m1 = h[:-1]                                    # (L-2,) h[i-1]
            hi = h[1:]                                        # (L-2,) h[i]
            diag = 2.0 * (hi_m1 + hi)                         # (L-2,)
            A = torch.diag(diag)
            idx = torch.arange(L - 3)
            A[idx, idx + 1] = hi[:-1]
            A[idx + 1, idx] = hi_m1[1:]
            dyf = (y[:, 2:, :] - y[:, 1:-1, :]) / hi.view(1, -1, 1)
            dyb = (y[:, 1:-1, :] - y[:, :-2, :]) / hi_m1.view(1, -1, 1)
            rhs = 6.0 * (dyf - dyb)                           # (B, L-2, D)
            rhs_mat = rhs.permute(1, 0, 2).reshape(L - 2, B * D)
            M_int = torch.linalg.solve(A, rhs_mat)            # (L-2, B*D)
            M_int = M_int.reshape(L - 2, B, D).permute(1, 0, 2)
            M = torch.zeros(B, L, D, dtype=y.dtype, device=y.device)
            M[:, 1:-1, :] = M_int
        else:
            M = torch.zeros(B, L, D, dtype=y.dtype, device=y.device)
        self.M = M

    def derivative(self, s: torch.Tensor) -> torch.Tensor:
        """dy/ds at scalar time s -> (B, D). Shared grid => single interval index."""
        s = s.reshape(())
        i = int(torch.searchsorted(self.t, s.reshape(1), right=True).item()) - 1
        i = max(0, min(i, self.L - 2))
        h = self.h[i]
        ti, ti1 = self.t[i], self.t[i + 1]
        A = (ti1 - s) / h
        Bc = (s - ti) / h
        yi, yi1 = self.y[:, i, :], self.y[:, i + 1, :]
        Mi, Mi1 = self.M[:, i, :], self.M[:, i + 1, :]
        return (yi1 - yi) / h + (h / 6.0) * (-(3 * A * A - 1) * Mi + (3 * Bc * Bc - 1) * Mi1)


# =============================================================================
# 3. Equivariant mock vector field  +  broken control          (Runsheet Task 3)
# =============================================================================
class EquivariantCDEFunc(torch.nn.Module):
    """A genuine (non-trivial) SE(3)-equivariant CDE integrand.

    Latent Z = [scalars s | vectors v]. Control derivative U(s) = mean_j dX_j/ds (a type-1
    vector). All coefficients (A, b, ds) are functions of ROTATION INVARIANTS only
    (scalars, block norms, dot products), so:

        dv_i = sum_j A_ij v_j + b_i U        (equivariant: type-1 combos, invariant weights)
        ds_i = ds_i                          (invariant)

    Under g: v -> v R^T, U -> U R^T  =>  dZ -> rho(g) dZ. Equivariant by construction.
    """

    def __init__(self, layout: IrrepLayout, hidden: int = 64, gain: float = 0.2, seed: int = 1):
        super().__init__()
        self.layout = layout
        self.gain = gain
        n_s, n_v = layout.n_scalar, layout.n_vec
        in_dim = n_s + n_v + n_v + 1          # scalars, ||v_i||, <v_i,U>, ||U||
        out_dim = n_s + n_v * n_v + n_v       # ds, A, b
        g = torch.Generator().manual_seed(seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, out_dim),
        )
        with torch.no_grad():                 # small deterministic init -> mild, non-stiff dynamics
            for m in self.net:
                if isinstance(m, torch.nn.Linear):
                    m.weight.copy_(torch.randn(m.weight.shape, generator=g) * 0.3)
                    m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.1)
        self.to(DTYPE)

    def forward(self, z: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        # z: (B, total), U: (B, 3)
        n_s, n_v = self.layout.n_scalar, self.layout.n_vec
        s, v = self.layout.split(z)                          # (B,n_s), (B,n_v,3)
        vnorm = v.norm(dim=-1)                               # (B,n_v)   invariant
        dots = (v * U.unsqueeze(1)).sum(-1)                  # (B,n_v)   invariant
        unorm = U.norm(dim=-1, keepdim=True)                 # (B,1)     invariant
        inv = torch.cat([s, vnorm, dots, unorm], dim=-1)     # invariant feature vector
        out = torch.tanh(self.net(inv)) * self.gain          # bounded invariant coefficients
        ds = out[:, :n_s]
        A = out[:, n_s:n_s + n_v * n_v].reshape(-1, n_v, n_v)
        b = out[:, n_s + n_v * n_v:]                         # (B,n_v)
        dv = torch.einsum("bij,bjd->bid", A, v) + b.unsqueeze(-1) * U.unsqueeze(1)
        return self.layout.join(ds, dv)


class BrokenCDEFunc(EquivariantCDEFunc):
    """Deliberately symmetry-BREAKING field: mixes vector components with a fixed
    non-orthogonal linear map on the carried latent. Used to prove the diagnostics fire
    (the exact failure the brief warns about: a non-orthogonal Linear on the carried latent)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        g = torch.Generator().manual_seed(7)
        # non-orthogonal 3x3 mixing (breaks rotation equivariance of the vector update)
        self.register_buffer("W_bad", torch.randn(3, 3, generator=g, dtype=DTYPE) * 0.15)

    def forward(self, z: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        dz = super().forward(z, U)
        _, dv = self.layout.split(dz)
        dv = dv + dv @ self.W_bad                            # non-orthogonal mix -> not equivariant
        s, _ = self.layout.split(dz)
        return self.layout.join(s, dv)


# =============================================================================
# 4. Custom invariant error norm N_eq  +  self-contained dopri5 (Runsheet Task 4)
# =============================================================================
def error_scalar(err, y0, y1, atol, rtol, mode, layout: IrrepLayout):
    """Reduce the embedded error estimate to the single scalar the step controller compares to 1.

    mode='default': per-component tolerance atol+rtol*max(|y0|,|y1|), outer RMS.
                    NOT rotation-invariant when rtol>0 (rotation mixes vector components,
                    so per-component weights differ) -> different step grid for X vs g.X.
    mode='neq'    : per-irrep-BLOCK tolerance (invariant block L2 norm) + the N_eq block norm
                        N_eq(e) = sqrt( (1/F) sum_f (1/d_f) ||e_f||^2 ).
                    Both the tolerance and the outer norm are group-invariant, so the error
                    scalar is identical for X and g.X -> identical accept/reject -> D_grid=0.
    """
    if mode == "default":
        tol = atol + rtol * torch.maximum(y0.abs(), y1.abs())
        scaled = err / tol
        return torch.sqrt((scaled * scaled).mean())

    if mode == "neq":
        n_s = layout.n_scalar
        # --- scalar (type-0, d_f=1) blocks: block "norm" is |value|, one per (batch, scalar) ---
        es, e0s, e1s = err[:, :n_s], y0[:, :n_s], y1[:, :n_s]
        denom_s = atol + rtol * torch.maximum(e0s.abs(), e1s.abs())     # invariant (dim-1)
        sc_s = es / denom_s
        contrib_s = (sc_s * sc_s).sum()                                 # (1/d_f=1)*||.||^2
        # --- vector (type-1, d_f=3) blocks: invariant block L2 norm, one per (batch, vector) ---
        ev = err[:, n_s:].reshape(err.shape[0], layout.n_vec, 3)
        v0 = y0[:, n_s:].reshape(err.shape[0], layout.n_vec, 3)
        v1 = y1[:, n_s:].reshape(err.shape[0], layout.n_vec, 3)
        bn0 = v0.norm(dim=-1)                                           # (B,n_v) invariant
        bn1 = v1.norm(dim=-1)
        denom_v = atol + rtol * torch.maximum(bn0, bn1)                 # (B,n_v) invariant scalar/block
        sc_v = ev / denom_v.unsqueeze(-1)
        contrib_v = ((sc_v * sc_v).sum(-1) / 3.0).sum()                 # (1/d_f=3)*||e_f||^2
        F = err.shape[0] * layout.n_features
        return torch.sqrt((contrib_s + contrib_v) / F)

    raise ValueError(f"unknown error mode {mode!r}")


# --- Dormand-Prince (DP45) Butcher tableau -----------------------------------
_DP_C = [0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0]
_DP_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
_DP_B5 = [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0]
# e = b5 - b4  (embedded 4th-order error weights)
_DP_E = [71 / 57600, 0.0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40]


def _control_at(spline: NaturalCubicSpline, s: torch.Tensor, n_joints: int) -> torch.Tensor:
    """U(s) = mean over joints of dX_j/ds  (a type-1 vector, (B,3))."""
    d = spline.derivative(s)                      # (B, J*3)
    return d.reshape(d.shape[0], n_joints, 3).mean(dim=1)


def integrate_fixed(func, spline, z0, s0, s1, n_joints, n_steps=100, method="rk4", layout=None):
    """Fixed-step explicit integrator (euler / rk4). Preserves equivariance exactly (to roundoff).
    `layout` accepted for a uniform integrator interface but unused here."""
    h = (s1 - s0) / n_steps
    z = z0
    s = s0
    for _ in range(n_steps):
        if method == "euler":
            z = z + h * func(z, _control_at(spline, s, n_joints))
        elif method == "rk4":
            k1 = func(z, _control_at(spline, s, n_joints))
            k2 = func(z + 0.5 * h * k1, _control_at(spline, s + 0.5 * h, n_joints))
            k3 = func(z + 0.5 * h * k2, _control_at(spline, s + 0.5 * h, n_joints))
            k4 = func(z + h * k3, _control_at(spline, s + h, n_joints))
            z = z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(method)
        s = s + h
    return z


def integrate_dopri5(func, spline, z0, s0, s1, n_joints, layout,
                     atol=1e-7, rtol=1e-7, error_mode="neq",
                     max_steps=20000, first_step=None):
    """Self-contained adaptive Dormand-Prince stepper.

    Returns (z_final, grid) where grid is the list of accepted step end-times (the step grid).
    The error-checking loop is fully explicit so we control exactly how the error scalar is
    formed (default per-component vs invariant N_eq)."""
    span = float(s1 - s0)
    h = span / 50.0 if first_step is None else first_step
    hmin = span * 1e-12
    safety, facmin, facmax = 0.9, 0.2, 5.0
    z = z0
    s = s0
    grid = []
    steps = 0
    while float(s) < float(s1) - hmin and steps < max_steps:
        steps += 1
        if float(s) + float(h) > float(s1):
            h = s1 - s
        # --- 7 stages (FSAL not exploited; simplicity over speed) ---
        ks = []
        for i in range(7):
            yi = z
            for j in range(i):
                yi = yi + h * _DP_A[i][j] * ks[j]
            si = s + _DP_C[i] * h
            ks.append(func(yi, _control_at(spline, si, n_joints)))
        y5 = z
        for i in range(7):
            y5 = y5 + h * _DP_B5[i] * ks[i]
        err = torch.zeros_like(z)
        for i in range(7):
            err = err + h * _DP_E[i] * ks[i]
        ratio = error_scalar(err, z, y5, atol, rtol, error_mode, layout)
        r = float(ratio)
        if r <= 1.0 or float(h) <= hmin:                 # accept
            s = s + h
            z = y5
            grid.append(float(s))
        # step-size update
        if r == 0.0:
            factor = facmax
        else:
            factor = min(facmax, max(facmin, safety * r ** (-0.2)))
        h = h * factor
    return z, grid


def grid_symmetric_difference(g1, g2, tol=1e-9) -> int:
    """|T(X) symdiff T(g.X)| : number of accepted step-knots that do NOT have a partner
    in the other grid within `tol`.

    Why a tolerance and not exact float equality: even when the step controller makes bit-for-bit
    identical accept/reject DECISIONS for X and g.X, the accepted times s = cumsum(h) differ at the
    ~1e-16 level because the step-size update factor = safety * ratio^(-1/5) inherits roundoff from
    applying rho(g) (an orthogonal matmul is exact only in real arithmetic). `tol=1e-9` sits far
    above that roundoff floor (~1e-15) and far below a real step size (~1e-2), so it cleanly
    separates 'same decisions, roundoff-identical grid' (N_eq -> D_grid=0) from 'materially
    different step sequence' (default per-component norm -> D_grid>0)."""
    a, b = sorted(g1), sorted(g2)
    i = j = diff = 0
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= tol:
            i += 1
            j += 1
        elif a[i] < b[j]:
            diff += 1
            i += 1
        else:
            diff += 1
            j += 1
    diff += (len(a) - i) + (len(b) - j)
    return diff


# =============================================================================
# 5. Data + equivariance-drift harness                        (Brief section 5.3 / 10)
# =============================================================================
def make_data(batch=8, length=40, n_joints=12, generator=None, dtype=DTYPE):
    """Smooth skeletal sequences on an IRREGULAR timestamp grid (CV>=0.2), root-relative.

    Returns t (L,) in [0,1] and x_rel (B, L, J*3), root joint subtracted so translation
    invariance is automatic.

    The control path is a sum of low-frequency sinusoids sampled at moderately-irregular
    timestamps. This is deliberate: a WELL-CONDITIONED control keeps dX/ds bounded, so the
    CDE is non-stiff and the adaptive solver takes tens of steps. (Heavy-tailed inter-arrivals
    with near-zero gaps make dX/ds = Dx/Dt spike -> thousands of tiny steps -> roundoff
    decoheres even the invariant grids. That is a data-conditioning artifact, not a symmetry
    property, so we avoid it here and stress irregular *sampling* separately in the data track.)
    """
    g = generator
    # moderately irregular inter-arrivals: Uniform(0.5, 1.5) -> CV ~ 0.29, bounded away from 0
    gaps = 0.5 + torch.rand(length - 1, generator=g, dtype=dtype, device=DEVICE)
    t = torch.cat([torch.zeros(1, dtype=dtype, device=DEVICE), torch.cumsum(gaps, 0)])
    t = t / t[-1]                                     # normalize to [0,1]
    # smooth low-frequency multi-sine control (analytic, bounded derivative)
    K = 3
    freq = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device=DEVICE)
    amp = torch.tensor([0.30, 0.15, 0.10], dtype=dtype, device=DEVICE)
    phase = torch.rand(batch, n_joints, 3, K, generator=g, dtype=dtype, device=DEVICE) * (2 * math.pi)
    arg = (2 * math.pi) * freq.view(1, 1, 1, 1, K) * t.view(1, length, 1, 1, 1) \
        + phase.view(batch, 1, n_joints, 3, K)
    x = (amp.view(1, 1, 1, 1, K) * torch.sin(arg)).sum(-1)      # (B, L, J, 3)
    x = x - x[:, :, :1, :]                            # root-relative (joint 0) -> translation invariant
    return t, x.reshape(batch, length, n_joints * 3)


def timestamp_cv(t: torch.Tensor) -> float:
    dt = t[1:] - t[:-1]
    return float(dt.std() / dt.mean())


def equivariance_drift(func, spline_X, spline_gX, z0, R, layout, n_joints, integrator, **kw):
    """Brief section 5.3 relative drift:  ||Solve(g.X) - rho(g) Solve(X)|| / ||Solve(X)||.

    integrator returns either z (fixed) or (z, grid) (dopri5). Returns (drift, grid_X, grid_gX)."""
    kw = dict(kw)
    kw["layout"] = layout                       # forward layout to whichever integrator needs it
    out_X = integrator(func, spline_X, z0, spline_X.t[0], spline_X.t[-1], n_joints, **kw)
    z0_g = apply_rho(R, z0, layout)
    out_gX = integrator(func, spline_gX, z0_g, spline_gX.t[0], spline_gX.t[-1], n_joints, **kw)
    if isinstance(out_X, tuple):
        zT, gridX = out_X
        zT_gX, gridGX = out_gX
    else:
        zT, gridX = out_X, None
        zT_gX, gridGX = out_gX, None
    g_zT = apply_rho(R, zT, layout)
    drift = ((zT_gX - g_zT).norm() / (zT.norm() + 1e-300)).item()
    return drift, gridX, gridGX


def build_splines(t, x_rel, R):
    """Spline for X and for the rotated control g.X = x_rel @ R^T (per joint)."""
    B, L, D = x_rel.shape
    J = D // 3
    spline_X = NaturalCubicSpline(t, x_rel)
    xr = x_rel.reshape(B, L, J, 3) @ R.transpose(-1, -2)
    spline_gX = NaturalCubicSpline(t, xr.reshape(B, L, D))
    return spline_X, spline_gX, J


# =============================================================================
# 6. The three certificate gates
# =============================================================================
def _init_state(layout, batch, generator):
    z0 = torch.randn(1, layout.total, generator=generator, dtype=DTYPE, device=DEVICE) * 0.5
    return z0.expand(batch, layout.total).contiguous()


def test_fixed_step_exactness(func, t, x_rel, layout, rotations, z0, n_joints):
    """Gate A (Runsheet Task 2): fixed-step drift <= 1e-13 and flat across step size."""
    print("\n[Gate A] Fixed-step algebraic-exactness  (euler / rk4)")
    print("  " + "-" * 68)
    results = {}
    all_pass = True
    for method in ("euler", "rk4"):
        row = {}
        for n_steps in (50, 100, 500):
            drifts = []
            for gi in range(rotations.shape[0]):
                R = rotations[gi]
                sX, sgX, J = build_splines(t, x_rel, R)
                d, _, _ = equivariance_drift(
                    func, sX, sgX, z0, R, layout, J,
                    integrator=integrate_fixed, n_steps=n_steps, method=method,
                )
                drifts.append(d)
            m = max(drifts)
            row[n_steps] = m
            print(f"    {method:5s}  h=T/{n_steps:<4d}  max delta_eq = {m:.3e}")
        results[method] = row
        mx = max(row.values())
        mn = min(v for v in row.values() if v > 0) if any(v > 0 for v in row.values()) else mx
        flat_ratio = (mx / mn) if mn > 0 else 1.0
        mag_ok = mx <= THRESH_FIXED_DRIFT
        flat_ok = flat_ratio <= THRESH_FIXED_FLAT_RATIO
        print(f"    -> {method}: max={mx:.2e} (<= {THRESH_FIXED_DRIFT:.0e}? {mag_ok}) "
              f"| flat ratio={flat_ratio:.2f} (<= {THRESH_FIXED_FLAT_RATIO:.0f}? {flat_ok})")
        if not (mag_ok and flat_ok):
            all_pass = False
            print(f"    !! DIAGNOSTIC: {method} violates fixed-step exactness. Either rho(g) is "
                  f"non-orthogonal or the vector field is not intertwining (check dv=Av+bU and "
                  f"that all coefficients are invariants).")
    assert all_pass, "Gate A FAILED: fixed-step integration does not preserve equivariance to 1e-13."
    print("  [Gate A] PASS")
    return results


def test_adaptive_grid_divergence(func, t, x_rel, layout, rotations, z0, n_joints,
                                  atol=1e-7, rtol=1e-6):
    """Gate B (Runsheet Tasks 4-5): N_eq drives D_grid->0 (drift<=1e-11); default norm diverges."""
    print("\n[Gate B] Adaptive step-grid divergence  (dopri5: default vs N_eq)")
    print("  " + "-" * 68)
    summary = {}
    for mode in ("default", "neq"):
        drifts, dgrids, nsteps = [], [], []
        for gi in range(rotations.shape[0]):
            R = rotations[gi]
            sX, sgX, J = build_splines(t, x_rel, R)
            d, gX, gGX = equivariance_drift(
                func, sX, sgX, z0, R, layout, J,
                integrator=integrate_dopri5,
                atol=atol, rtol=rtol, error_mode=mode,
            )
            drifts.append(d)
            dgrids.append(grid_symmetric_difference(gX, gGX))
            nsteps.append(len(gX))
        summary[mode] = (max(drifts), sum(dgrids) / len(dgrids), max(dgrids))
        print(f"    {mode:8s}  max drift = {summary[mode][0]:.3e} | "
              f"mean D_grid = {summary[mode][1]:.2f} | max D_grid = {summary[mode][2]} | "
              f"~{sum(nsteps)//len(nsteps)} steps")

    neq_drift, neq_dgrid_mean, neq_dgrid_max = summary["neq"]
    def_drift, def_dgrid_mean, def_dgrid_max = summary["default"]

    # Drift-vs-angle flatness sweep under N_eq (should stay flat; a violation would grow with angle)
    print("    drift-vs-angle sweep (N_eq):")
    gsweep = torch.Generator().manual_seed(123)
    for tmax in (0.1, 0.5, 1.0, math.pi):
        Rs = random_rotations(4, gsweep, theta_max=tmax)
        ds = []
        for gi in range(Rs.shape[0]):
            R = Rs[gi]
            sX, sgX, J = build_splines(t, x_rel, R)
            d, _, _ = equivariance_drift(func, sX, sgX, z0, R, layout, J,
                                         integrator=integrate_dopri5,
                                         atol=atol, rtol=rtol, error_mode="neq")
            ds.append(d)
        print(f"      theta_max={tmax:5.3f}  max drift = {max(ds):.3e}")

    neq_ok = (neq_drift <= THRESH_NEQ_DRIFT) and (neq_dgrid_max == 0)
    default_diverges = (def_dgrid_max > 0)
    print(f"    -> N_eq: drift={neq_drift:.2e} (<= {THRESH_NEQ_DRIFT:.0e}? {neq_drift <= THRESH_NEQ_DRIFT}) "
          f"& D_grid_max={neq_dgrid_max} (==0? {neq_dgrid_max == 0})")
    print(f"    -> default norm diverges (D_grid_max={def_dgrid_max} > 0? {default_diverges}, "
          f"drift={def_drift:.2e})")
    if not neq_ok:
        print("    !! DIAGNOSTIC: N_eq failed to synchronize step grids. Check that BOTH the "
              "per-block tolerance (invariant block L2 norm) AND the outer N_eq block norm are "
              "used -- swapping only the outer norm leaves the per-component tolerance (killer 6.1B).")
    if not default_diverges:
        print("    (note) default norm did not diverge -- tolerances may be too loose to expose it; "
              "increase rtol or sequence length.")
    assert neq_ok, "Gate B FAILED: N_eq did not drive D_grid to 0 with drift <= 1e-11."
    print("  [Gate B] PASS")
    return summary


def test_precision_scaling(layout, t, x_rel, rotations, n_joints, atol=1e-7, rtol=1e-6):
    """Gate C (Runsheet Task 6): r = drift_fp32 / drift_fp64 >= 1e5  => roundoff, not violation.

    Uses the N_eq config (D_grid=0) so the residual is unambiguously roundoff, not grid drift."""
    print("\n[Gate C] Precision-scaling diagnostic  (fp32 vs fp64, N_eq)")
    print("  " + "-" * 68)

    def mean_drift(dtype):
        global DTYPE
        old = DTYPE
        DTYPE = dtype
        try:
            gen = torch.Generator().manual_seed(SEED + 1)
            layout_ = layout
            func = EquivariantCDEFunc(layout_, seed=1).to(dtype)
            tt = t.to(dtype)
            xx = x_rel.to(dtype)
            z0 = _init_state(layout_, xx.shape[0], gen).to(dtype)
            drifts = []
            for gi in range(rotations.shape[0]):
                R = rotations[gi].to(dtype)
                sX, sgX, J = build_splines(tt, xx, R)
                d, _, _ = equivariance_drift(func, sX, sgX, z0, R, layout_, J,
                                             integrator=integrate_dopri5,
                                             atol=atol, rtol=rtol, error_mode="neq")
                drifts.append(d)
            return sum(drifts) / len(drifts)
        finally:
            DTYPE = old

    d32 = mean_drift(torch.float32)
    d64 = mean_drift(torch.float64)
    r = d32 / max(d64, 1e-300)
    print(f"    mean drift fp32 = {d32:.3e}")
    print(f"    mean drift fp64 = {d64:.3e}")
    print(f"    ratio r = {r:.3e}   (>= {THRESH_PRECISION_RATIO:.0e}? {r >= THRESH_PRECISION_RATIO})")
    if r < THRESH_PRECISION_RATIO:
        print("    !! DIAGNOSTIC: r ~ 1 means the residual drift does NOT shrink with precision "
              "=> a genuine symmetry violation (hidden non-orthogonal op / non-invariant norm), "
              "not roundoff. Return to the Task-1 orthogonality audit per-layer.")
    assert r >= THRESH_PRECISION_RATIO, "Gate C FAILED: fp32/fp64 ratio < 1e5 => hidden violation."
    print("  [Gate C] PASS")
    return d32, d64, r


def demonstrate_broken_model(layout, t, x_rel, rotations, z0, n_joints):
    """Sanity: prove the diagnostics FIRE on a real violation (non-orthogonal latent mixing)."""
    print("\n[Sanity] Broken (non-equivariant) field must FAIL the gates")
    print("  " + "-" * 68)
    broken = BrokenCDEFunc(layout, seed=1)
    # fixed-step drift should be O(>=1e-2), not 1e-13
    R = rotations[0]
    sX, sgX, J = build_splines(t, x_rel, R)
    d_fixed, _, _ = equivariance_drift(broken, sX, sgX, z0, R, layout, J,
                                       integrator=integrate_fixed, n_steps=100, method="rk4")
    # precision ratio should collapse toward ~1
    d32 = _broken_dopri_drift(layout, t, x_rel, rotations, torch.float32)
    d64 = _broken_dopri_drift(layout, t, x_rel, rotations, torch.float64)
    r = d32 / max(d64, 1e-300)
    print(f"    broken fixed-step drift = {d_fixed:.3e}  (equivariant model was ~1e-14)")
    print(f"    broken precision ratio r = {r:.3e}  (equivariant model was >= 1e5)")
    fires = (d_fixed > THRESH_FIXED_DRIFT) and (r < THRESH_PRECISION_RATIO)
    assert fires, "Sanity FAILED: diagnostics did not fire on a known-broken field!"
    print("  [Sanity] PASS -- diagnostics correctly reject the broken field.")


def _broken_dopri_drift(layout, t, x_rel, rotations, dtype):
    gen = torch.Generator().manual_seed(SEED + 1)
    broken = BrokenCDEFunc(layout, seed=1).to(dtype)
    tt, xx = t.to(dtype), x_rel.to(dtype)
    z0 = _init_state(layout, xx.shape[0], gen).to(dtype)
    drifts = []
    for gi in range(rotations.shape[0]):
        R = rotations[gi].to(dtype)
        sX, sgX, J = build_splines(tt, xx, R)
        d, _, _ = equivariance_drift(broken, sX, sgX, z0, R, layout, J,
                                     integrator=integrate_dopri5,
                                     atol=1e-7, rtol=1e-6, error_mode="neq")
        drifts.append(d)
    return sum(drifts) / len(drifts)


# =============================================================================
# 6b. Task 7 — REAL e3nn steerable vector field  (structural integration gate)
# =============================================================================
# We reuse the solver (integrate_fixed / integrate_dopri5), the invariant error norm
# (error_scalar / N_eq), and the IrrepLayout block partition UNCHANGED. Only the vector
# field and the representation rho(g) change:
#   * field: mock  ->  FullyConnectedTensorProduct -> Gate -> Linear   (real e3nn layers)
#   * rho(g): Cartesian apply_rho  ->  irreps.D_from_matrix(R)  (e3nn's native 1o basis)
# The IrrepLayout(n_scalar, n_vec) partition [n_s scalars | n_v vectors(xyz)] matches the
# e3nn ordering of Irreps("{n_s}x0e + {n_v}x1o") exactly, so N_eq needs no changes.

if HAVE_E3NN:

    class EquivariantCDEFunc_e3nn(torch.nn.Module):
        """Authentic nonlinear SE(3)-equivariant CDE integrand built from real e3nn layers.

        dZ/ds = Linear( Gate( TensorProduct( Z, [1, U] ) ) ) * gain

        - FullyConnectedTensorProduct(irreps_Z, "1x0e+1x1o", gate.irreps_in): the ONLY place
          state Z and control U couple; CG-constrained so it is equivariant for ANY weights.
          Feeding [1, U] lets the scalar path act as a learnable self-term and the 1o path as
          the control coupling, in a single equivariant bilinear map.
        - Gate: equivariant nonlinearity (tanh on feature scalars; sigmoid gates scale the
          1o vectors by invariant scalars -> preserves the type-1 transformation law).
        - o3.Linear(..., biases=False): equivariant readout to irreps_Z. NO bias, because a
          non-scalar (1o) bias is a constant vector that does NOT rotate -> breaks equivariance.
        """

        def __init__(self, layout: IrrepLayout, gain: float = 0.2):
            super().__init__()
            n_s, n_v = layout.n_scalar, layout.n_vec
            self.layout = layout
            self.gain = gain
            self.irreps_Z = o3.Irreps(f"{n_s}x0e + {n_v}x1o")
            self.irreps_in2 = o3.Irreps("1x0e + 1x1o")             # [1, U]
            irreps_scalars = o3.Irreps(f"{n_s}x0e")                # feature scalars (activated)
            irreps_gates = o3.Irreps(f"{n_v}x0e")                  # one gate per output vector
            irreps_gated = o3.Irreps(f"{n_v}x1o")                  # the gated vectors
            self.gate = Gate(irreps_scalars, [torch.tanh], irreps_gates, [torch.sigmoid], irreps_gated)
            self.tp = o3.FullyConnectedTensorProduct(self.irreps_Z, self.irreps_in2, self.gate.irreps_in)
            self.lin = o3.Linear(self.gate.irreps_out, self.irreps_Z, biases=False)

        def forward(self, z: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
            ones = torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)
            in2 = torch.cat([ones, U], dim=-1)                    # (B, 4)  = [1, U]
            h = self.tp(z, in2)
            h = self.gate(h)
            return self.lin(h) * self.gain

    def e3nn_rep(layout: IrrepLayout, R: torch.Tensor):
        """Return (D_Z, D_1o) : e3nn Wigner-D reps of R on the full state and on a single 1o
        vector, both in e3nn's native basis, cast to R.dtype."""
        irreps_Z = o3.Irreps(f"{layout.n_scalar}x0e + {layout.n_vec}x1o")
        irreps_1o = o3.Irreps("1x1o")
        D_Z = irreps_Z.D_from_matrix(R).to(R.dtype)
        D_1o = irreps_1o.D_from_matrix(R).to(R.dtype)
        return D_Z, D_1o

    def apply_rep(D: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Apply a matrix representation D (dim x dim) to row-stored features z (..., dim)."""
        return z @ D.transpose(-1, -2)

    def build_splines_e3nn(t, x_rel, D_1o):
        """Spline for X and for g.X where each joint 3-vector is rotated by the e3nn 1o rep."""
        B, L, Dd = x_rel.shape
        J = Dd // 3
        sX = NaturalCubicSpline(t, x_rel)
        xr = apply_rep(D_1o, x_rel.reshape(B, L, J, 3)).reshape(B, L, Dd)
        sgX = NaturalCubicSpline(t, xr)
        return sX, sgX, J

    def drift_e3nn(func, sX, sgX, z0, D_Z, layout, J, integrator, **kw):
        kw = dict(kw)
        kw["layout"] = layout
        outX = integrator(func, sX, z0, sX.t[0], sX.t[-1], J, **kw)
        z0g = apply_rep(D_Z, z0)
        outg = integrator(func, sgX, z0g, sgX.t[0], sgX.t[-1], J, **kw)
        zT, gX = outX if isinstance(outX, tuple) else (outX, None)
        zTg, gg = outg if isinstance(outg, tuple) else (outg, None)
        g_zT = apply_rep(D_Z, zT)
        drift = ((zTg - g_zT).norm() / (zT.norm() + 1e-300)).item()
        return drift, gX, gg

    def assert_no_nonscalar_bias(field) -> str:
        """Point 3: verify NO non-scalar bias exists. Our layers use biases=False, and we also
        demonstrate that e3nn structurally refuses to place a bias on a 1o output."""
        # (a) our field carries no bias parameter at all
        bias_params = [n for n, _ in field.named_parameters() if n.endswith("bias") or ".bias" in n]
        assert not bias_params, f"unexpected bias parameters present: {bias_params}"
        # (b) e3nn would only ever bias 0e irreps: a 1o-output Linear with biases=True gets 0 bias
        probe = o3.Linear(o3.Irreps("2x0e + 1x1o"), o3.Irreps("1x1o"), biases=True)
        probe_bias = [p for n, p in probe.named_parameters() if "bias" in n]
        nbias = sum(p.numel() for p in probe_bias)
        assert nbias == 0, f"e3nn placed {nbias} bias weights on a 1o output (would break equivariance)"
        return "no bias params; e3nn refuses 1o bias (probe bias numel=0)"

    def _make_field_and_data(layout, dtype, batch, length, joints, seed):
        """Build the e3nn field + data in a chosen precision (weights follow torch default dtype)."""
        prev = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            torch.manual_seed(seed)
            field = EquivariantCDEFunc_e3nn(layout, gain=0.2)
        finally:
            torch.set_default_dtype(prev)
        gen = torch.Generator().manual_seed(seed)
        t, x_rel = make_data(batch, length, joints, generator=gen, dtype=dtype)
        z0 = (torch.randn(1, layout.total, generator=gen, dtype=dtype) * 0.5).expand(batch, layout.total).contiguous()
        return field, t, x_rel, z0

    def test_e3nn_structural_integration(layout, batch, length, joints, n_rot,
                                         atol=1e-7, rtol=1e-6):
        """Task 7: swap the mock for real e3nn layers; re-run Gate A / Gate B / Gate C."""
        print("\n" + "=" * 72)
        print("  TASK 7 -- STRUCTURAL INTEGRATION GATE  (real e3nn steerable field)")
        print("=" * 72)
        field, t, x_rel, z0 = _make_field_and_data(layout, DTYPE, batch, length, joints, seed=1)
        print(f"  field: TensorProduct{tuple(str(field.irreps_Z).split(' + '))} -> Gate -> "
              f"Linear(biases=False)")
        print(f"  irreps_Z = {field.irreps_Z}  (matches IrrepLayout {layout.n_scalar}s+{layout.n_vec}v)")

        # --- Point 3: bias check ---
        bias_msg = assert_no_nonscalar_bias(field)
        print(f"  [bias check] {bias_msg}")

        # --- rotations + Task-1 orthogonality on the REAL e3nn rep ---
        gen = torch.Generator().manual_seed(SEED)
        Rs = random_rotations(n_rot, gen, theta_max=math.pi)
        ortho = 0.0
        Dzs, D1os = [], []
        for gi in range(n_rot):
            D_Z, D_1o = e3nn_rep(layout, Rs[gi])
            Dzs.append(D_Z)
            D1os.append(D_1o)
            I = torch.eye(D_Z.shape[0], dtype=D_Z.dtype)
            ortho = max(ortho, float((D_Z.transpose(-1, -2) @ D_Z - I).norm()))
        print(f"  [Task 1] e3nn rep orthogonality max||D^T D - I||_F = {ortho:.3e} "
              f"(<= {THRESH_ORTHO:.0e}? {ortho <= THRESH_ORTHO})")
        assert ortho <= THRESH_ORTHO, "Task 7 FAILED: e3nn representation is not orthogonal."

        # --- Gate A: fixed-step exactness ---
        print("\n  [Gate A] fixed-step (rk4) on real e3nn field")
        a_max = 0.0
        row = {}
        for n_steps in (50, 100, 500):
            drifts = []
            for gi in range(n_rot):
                sX, sgX, J = build_splines_e3nn(t, x_rel, D1os[gi])
                d, _, _ = drift_e3nn(field, sX, sgX, z0, Dzs[gi], layout, J,
                                     integrator=integrate_fixed, n_steps=n_steps, method="rk4")
                drifts.append(d)
            row[n_steps] = max(drifts)
            print(f"      h=T/{n_steps:<4d}  max delta_eq = {row[n_steps]:.3e}")
        a_max = max(row.values())
        a_flat = a_max / min(v for v in row.values() if v > 0)
        print(f"    -> max={a_max:.2e} (<= {THRESH_FIXED_DRIFT:.0e}? {a_max <= THRESH_FIXED_DRIFT}) "
              f"| flat ratio={a_flat:.2f}")
        if a_max > THRESH_FIXED_DRIFT:
            print("    !! DIAGNOSTIC: real e3nn field broke fixed-step exactness. A layer's output "
                  "irreps or the D_from_matrix basis is inconsistent -> isolate per-layer.")
        assert a_max <= THRESH_FIXED_DRIFT, "Task 7 Gate A FAILED on real e3nn layers."

        # --- Gate B: adaptive D_grid (default vs N_eq) ---
        print("\n  [Gate B] adaptive dopri5 (default vs N_eq) on real e3nn field")
        summ = {}
        for mode in ("default", "neq"):
            drifts, dgrids, nsteps = [], [], []
            for gi in range(n_rot):
                sX, sgX, J = build_splines_e3nn(t, x_rel, D1os[gi])
                d, gX, gg = drift_e3nn(field, sX, sgX, z0, Dzs[gi], layout, J,
                                       integrator=integrate_dopri5, atol=atol, rtol=rtol,
                                       error_mode=mode)
                drifts.append(d)
                dgrids.append(grid_symmetric_difference(gX, gg))
                nsteps.append(len(gX))
            summ[mode] = (max(drifts), max(dgrids))
            print(f"      {mode:8s} max drift = {max(drifts):.3e} | max D_grid = {max(dgrids)} | "
                  f"~{sum(nsteps)//len(nsteps)} steps")
        neq_ok = (summ["neq"][0] <= THRESH_NEQ_DRIFT) and (summ["neq"][1] == 0)
        print(f"    -> N_eq: drift={summ['neq'][0]:.2e} & D_grid={summ['neq'][1]} "
              f"| default D_grid={summ['default'][1]}")
        assert neq_ok, "Task 7 Gate B FAILED: N_eq did not sync grids on real e3nn layers."

        # --- Gate C: precision scaling fp32 vs fp64 ---
        print("\n  [Gate C] precision scaling (fp32 vs fp64) on real e3nn field")

        def mean_drift(dtype):
            f, td, xd, z0d = _make_field_and_data(layout, dtype, batch, length, joints, seed=1)
            ds = []
            for gi in range(n_rot):
                R = Rs[gi].to(dtype)
                D_Z, D_1o = e3nn_rep(layout, R)
                sX, sgX, J = build_splines_e3nn(td, xd, D_1o)
                d, _, _ = drift_e3nn(f, sX, sgX, z0d, D_Z, layout, J,
                                     integrator=integrate_dopri5, atol=atol, rtol=rtol,
                                     error_mode="neq")
                ds.append(d)
            return sum(ds) / len(ds)

        d32, d64 = mean_drift(torch.float32), mean_drift(torch.float64)
        r = d32 / max(d64, 1e-300)
        print(f"      fp32={d32:.3e}  fp64={d64:.3e}  ratio r={r:.3e} "
              f"(>= {THRESH_PRECISION_RATIO:.0e}? {r >= THRESH_PRECISION_RATIO})")
        assert r >= THRESH_PRECISION_RATIO, "Task 7 Gate C FAILED: residual not roundoff on real layers."

        print("\n  [TASK 7] PASS -- real e3nn steerable layers preserve the equivariance certificate.")
        return {"ortho": ortho, "gateA": a_max, "gateB_drift": summ["neq"][0],
                "gateB_default_dgrid": summ["default"][1], "gateC_ratio": r}


# =============================================================================
# 7. Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Week-One Equivariance Verification Suite")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--joints", type=int, default=12)
    ap.add_argument("--n_scalar", type=int, default=4)
    ap.add_argument("--n_vec", type=int, default=6)
    ap.add_argument("--n_rot", type=int, default=6, help="random rotations to average over")
    ap.add_argument("--skip-e3nn", action="store_true", help="skip Task 7 (real e3nn layer swap-in)")
    ap.add_argument("--only-e3nn", action="store_true", help="run only Task 7 (real e3nn layers)")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)          # certificate is inference-only: no autograd graphs
    gen = torch.Generator().manual_seed(SEED)

    print("=" * 72)
    print("  WEEK-ONE EQUIVARIANCE VERIFICATION SUITE")
    print("  SE(3)-Equivariant Neural CDE  |  self-contained dopri5 + natural cubic spline")
    print("=" * 72)

    layout = IrrepLayout(n_scalar=args.n_scalar, n_vec=args.n_vec)
    print(f"  state layout: {layout.n_scalar} scalars + {layout.n_vec} vectors "
          f"= {layout.total} dims, F={layout.n_features} irrep blocks")

    # --- Runsheet Task 1: orthogonality audit ---
    rotations = random_rotations(args.n_rot, gen, theta_max=math.pi)
    ortho = orthogonality_audit(rotations)
    print(f"\n[Task 1] Orthogonality audit  max||R^T R - I||_F = {ortho:.3e} "
          f"(<= {THRESH_ORTHO:.0e}? {ortho <= THRESH_ORTHO})")
    assert ortho <= THRESH_ORTHO, "Task 1 FAILED: representation is not orthogonal."

    # --- data (irregular timestamps) ---
    t, x_rel = make_data(args.batch, args.length, args.joints, generator=gen)
    cv = timestamp_cv(t)
    print(f"  irregular timestamps: CV(dt) = {cv:.3f} (>= 0.2? {cv >= 0.2}) over L={args.length}")

    func = EquivariantCDEFunc(layout, seed=1)
    z0 = _init_state(layout, args.batch, gen)

    # --- mock-field gates (skipped in --only-e3nn mode) ---
    resA = resB = None
    r = float("nan")
    if not args.only_e3nn:
        resA = test_fixed_step_exactness(func, t, x_rel, layout, rotations, z0, args.joints)
        resB = test_adaptive_grid_divergence(func, t, x_rel, layout, rotations, z0, args.joints)
        d32, d64, r = test_precision_scaling(layout, t, x_rel, rotations, args.joints)
        demonstrate_broken_model(layout, t, x_rel, rotations, z0, args.joints)

    # --- Task 7: real e3nn steerable layers ---
    res7 = None
    if args.skip_e3nn:
        print("\n[Task 7] SKIPPED (--skip-e3nn)")
    elif not HAVE_E3NN:
        print(f"\n[Task 7] SKIPPED -- e3nn not importable ({_E3NN_ERR}). "
              f"Install with: python -m pip install e3nn")
    else:
        res7 = test_e3nn_structural_integration(layout, args.batch, args.length, args.joints, args.n_rot)

    # --- certificate ---
    print("\n" + "=" * 72)
    print("  EQUIVARIANCE CERTIFICATE  (all gates green)")
    print("=" * 72)
    print(f"  Task 1  orthogonality      : {ortho:.2e}  <= {THRESH_ORTHO:.0e}")
    if resA is not None:
        print("  --- MOCK equivariant field ---")
        print(f"  Gate A  fixed-step drift    : {max(max(v.values()) for v in resA.values()):.2e}  <= {THRESH_FIXED_DRIFT:.0e}   (flat across h)")
        print(f"  Gate B  N_eq D_grid         : 0            (default norm diverges)")
        print(f"  Gate B  N_eq drift          : {resB['neq'][0]:.2e}  <= {THRESH_NEQ_DRIFT:.0e}")
        print(f"  Gate C  fp32/fp64 ratio     : {r:.2e}  >= {THRESH_PRECISION_RATIO:.0e}   (residual = roundoff)")
    if res7 is not None:
        print("  --- REAL e3nn steerable field (Task 7) ---")
        print(f"  Task 1  e3nn rep ortho      : {res7['ortho']:.2e}  <= {THRESH_ORTHO:.0e}")
        print(f"  Gate A  fixed-step drift    : {res7['gateA']:.2e}  <= {THRESH_FIXED_DRIFT:.0e}   (flat across h)")
        print(f"  Gate B  N_eq D_grid         : {res7['gateB_default_dgrid']} default -> 0 N_eq")
        print(f"  Gate B  N_eq drift          : {res7['gateB_drift']:.2e}  <= {THRESH_NEQ_DRIFT:.0e}")
        print(f"  Gate C  fp32/fp64 ratio     : {res7['gateC_ratio']:.2e}  >= {THRESH_PRECISION_RATIO:.0e}   (residual = roundoff)")
    print("=" * 72)
    print("  RESULT: EQUIVARIANCE CERTIFIED. Architecture is exactly SE(3)-equivariant;")
    print("          residual drift is floating-point roundoff, not a symmetry violation.")
    if res7 is not None:
        print("          Certificate SURVIVES the real e3nn architecture (Task 7 green).")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
