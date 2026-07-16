#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cde_model.py
============
The TRAINABLE SE(3)-equivariant Neural CDE (Block-2 phase).

Relationship to the week-one certificate (`equivariance_suite.py`)
-----------------------------------------------------------------
The week-one field certified Tasks 1-7 with a control
    U(s) = mean_j dX_j/ds      (equivariance_suite._control_at)
i.e. the per-joint velocities were AVERAGED into a single 3-vector before reaching f_theta.
That is equivariant -- which is all the certificate needed -- but it collapses the 25 x 3 = 75
control channels onto 3, and the cost is not merely attenuation (measured at ~3x on synthetic
motion) but IDENTIFIABILITY: mean_j dX_j is invariant under any permutation of the joints, so
every joint permutation lies exactly in the null space of the pooling map. Swapping the left and
right arm joints leaves the pooled control BIT-IDENTICAL (measured: 0.0e+00, see
certify_trainable.py) while our per-joint control changes by 58%. A field that cannot tell a
left-arm raise from a right-arm raise cannot score rehabilitation quality, and no downstream
layer can recover what the control already destroyed. A model trained on it would score near
chance -- and that would read as a defeat of the paradigm rather than an artifact of the scaffold.

This module rebuilds the field with PER-JOINT equivariant coupling. Equivariance is retained
by construction (Clebsch-Gordan tensor products are equivariant for ANY weights), so the
Task-2/4/6 gates must be -- and are -- re-run against this field (`certify_trainable.py`).

Why this is a genuine Neural CDE (and the week-one field was not)
----------------------------------------------------------------
A Neural CDE is    dZ/ds = f_theta(Z) dX/ds,   f_theta : R^z -> R^{z x x}
i.e. the vector field may be arbitrarily NONLINEAR in the state Z but must act LINEARLY on
the control derivative. The week-one field fed U through a Gate nonlinearity, making dZ/ds
nonlinear in dX/ds -- fine for an equivariance probe, but it forfeits the Riemann-Stieltjes
reading of the integral and the Kidger et al. (2020) universality/well-posedness results that
the paper will lean on. Here f_theta acts linearly on the control:

    dZ/ds  =  f_time(Psi(Z)) * (dt/ds)  +  sum_j  TP( Psi(Z), dX_j/ds ; W_j )

    * Psi(Z)  : arbitrarily nonlinear, equivariant function of the STATE only.
    * TP(., dX_j ; .) : a Clebsch-Gordan tensor product, hence BILINEAR -- in particular
      linear in dX_j/ds. This is the f_theta(Z) matrix, assembled equivariantly.
    * W_j     : per-joint weights generated from a learned per-joint Type-0 (invariant)
      embedding, so joints are distinguishable without breaking equivariance.

TIME AUGMENTATION (the dt/ds term) is not decoration. A pure CDE integral  int f(Z) dX  is
invariant to reparametrisation of s: it sees the PATH of X, not the speed along it. Movement
tempo is clinically meaningful in rehab, so we integrate against the augmented control
X~(s) = (s, X(s)); the dt/ds = 1 channel restores speed sensitivity. Time is a Type-0 scalar,
so it enters equivariantly.

Which CG paths survive (worth knowing when reading the code):
    1o(Psi) (x) 1o(dX) -> 0e   : scalar channels accumulate  int <v(Z), dX_j>
                                 = motion of joint j projected on a learned direction.
    0e(Psi) (x) 1o(dX) -> 1o   : vector channels accumulate  int s(Z) dX_j.
    (1o (x) 1o has EVEN parity, so it cannot feed a 1o output -- e3nn enforces this for us.)

Solver policy (PROJECT_BRIEF 5.1)
---------------------------------
TRAIN with fixed-step RK4: explicit fixed-step solvers preserve equivariance EXACTLY (to
roundoff), independent of step size, so training never touches the adaptive step-grid
divergence risk surface. Reserve dopri5 + N_eq for certification and final evaluation.

Nothing here mutates `equivariance_suite.py`: that module is the certified week-one artifact
and stays byte-stable. We import its IrrepLayout so the N_eq block partition still applies.
"""

import math

import torch
import torch.nn as nn

import e3nn
e3nn.set_optimization_defaults(jit_script_fx=False)
from e3nn import o3                      # noqa: E402
from e3nn.nn import Gate                 # noqa: E402

from equivariance_suite import IrrepLayout   # noqa: E402  (block partition for N_eq)


# =============================================================================
# 1. Batched natural cubic spline  (per-sample irregular timeline)
# =============================================================================
class BatchedNaturalCubicSpline:
    """Natural cubic spline where EACH sample carries its own irregular time grid.

    `equivariance_suite.NaturalCubicSpline` assumes one grid shared by the batch, which is
    true for the synthetic certificate but false for KIMORE: every recording has its own
    sensor arrival stamps. Same mathematics, batched over the grid.

    Equivariance is inherited unchanged: the spline is LINEAR in the control points and
    reproduces constants, so spline(R.x) = R.spline(x) and a translation passes through the
    spline and is annihilated by dX.

    t : (B, L) strictly increasing per row
    y : (B, L, D)
    """

    def __init__(self, t: torch.Tensor, y: torch.Tensor):
        self.t = t
        self.y = y
        B, L, D = y.shape
        self.B, self.L, self.D = B, L, D
        h = t[:, 1:] - t[:, :-1]                                    # (B, L-1)
        self.h = h

        if L >= 3:
            hi_m1 = h[:, :-1]                                       # (B, L-2)  h[i-1]
            hi = h[:, 1:]                                           # (B, L-2)  h[i]
            # Tridiagonal moment system, batched:
            #   h[i-1] M[i-1] + 2(h[i-1]+h[i]) M[i] + h[i] M[i+1] = 6 (dy_fwd - dy_bwd)
            A = torch.diag_embed(2.0 * (hi_m1 + hi))                # (B, L-2, L-2)
            if L >= 4:
                A = A + torch.diag_embed(hi[:, :-1], offset=1)
                A = A + torch.diag_embed(hi_m1[:, 1:], offset=-1)
            dyf = (y[:, 2:, :] - y[:, 1:-1, :]) / hi.unsqueeze(-1)
            dyb = (y[:, 1:-1, :] - y[:, :-2, :]) / hi_m1.unsqueeze(-1)
            rhs = 6.0 * (dyf - dyb)                                 # (B, L-2, D)
            M_int = torch.linalg.solve(A, rhs)                      # (B, L-2, D)
            M = torch.zeros(B, L, D, dtype=y.dtype, device=y.device)
            M[:, 1:-1, :] = M_int
        else:
            M = torch.zeros(B, L, D, dtype=y.dtype, device=y.device)
        self.M = M

    def _locate(self, s: torch.Tensor) -> torch.Tensor:
        """s: (B,) -> interval index i in [0, L-2] per sample."""
        i = torch.searchsorted(self.t.contiguous(), s.unsqueeze(-1).contiguous(), right=True) - 1
        return i.clamp_(0, self.L - 2)                              # (B, 1)

    def derivative(self, s: torch.Tensor) -> torch.Tensor:
        """dy/ds at a PER-SAMPLE time s: (B,) -> (B, D)."""
        i = self._locate(s)                                         # (B,1)
        gy = i.unsqueeze(-1).expand(-1, -1, self.D)                 # (B,1,D)
        yi = self.y.gather(1, gy).squeeze(1)
        yi1 = self.y.gather(1, gy + 1).squeeze(1)
        Mi = self.M.gather(1, gy).squeeze(1)
        Mi1 = self.M.gather(1, gy + 1).squeeze(1)
        h = self.h.gather(1, i)                                     # (B,1)
        ti = self.t.gather(1, i)
        ti1 = self.t.gather(1, i + 1)
        s = s.unsqueeze(-1)                                         # (B,1)
        A = (ti1 - s) / h
        Bc = (s - ti) / h
        return (yi1 - yi) / h + (h / 6.0) * (-(3 * A * A - 1) * Mi + (3 * Bc * Bc - 1) * Mi1)

    @property
    def t0(self) -> torch.Tensor:
        return self.t[:, 0]

    @property
    def tN(self) -> torch.Tensor:
        return self.t[:, -1]


# =============================================================================
# 2. Equivariant state -> state map  Psi(Z)   (nonlinear in Z, equivariant)
# =============================================================================
class EquivariantNorm(nn.Module):
    """Bounded per-irrep normalisation -- the stabiliser that makes the flow well-posed.

    Each irrep feature is rescaled by a SCALAR function of its own invariant norm:

        z_f  ->  z_f / sqrt(1 + ||z_f||^2 / d_f)

    Equivariance: the multiplier is a function of ||z_f||, which is invariant (D_f(g) is
    orthogonal), and a scalar multiple commutes with D_f(g). So this is exactly the equivariant
    per-irrep normalisation PROJECT_BRIEF 6.3 calls for.

    WHY IT IS NECESSARY, not cosmetic. The state map's self tensor-product is QUADRATIC in Z,
    and the Gate's sigmoid only rescales the vector channels -- it does not bound them. So the
    raw field grows like ||Z||^2, and dZ/ds ~ Z^2 is the textbook finite-time blow-up ODE. It is
    not a hypothetical: the unnormalised field returns NaN at gain >= 1.0 (it survives at
    gain=0.2 only by staying in a small-||Z|| regime, and training moves the weights out of it).
    Bounding the field's input caps ||f|| , so ||Z(s)|| can grow at most LINEARLY in s and the
    flow is globally well-posed.

    The saturating form is chosen over hard normalisation (z_f/||z_f||) deliberately: it is
    strictly MONOTONE in ||z_f||, so feature magnitude -- which carries the state the CDE has
    accumulated -- is compressed rather than destroyed.
    """

    def __init__(self, layout: IrrepLayout):
        super().__init__()
        self.layout = layout

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        s, v = self.layout.split(z)                                   # (B,n_s), (B,n_v,3)
        s = s / torch.sqrt(1.0 + s ** 2)                              # d=1 per scalar feature
        vn = v.pow(2).sum(-1, keepdim=True) / 3.0                     # d=3 per vector feature
        v = v / torch.sqrt(1.0 + vn)
        return self.layout.join(s, v)


class EquivariantStateMap(nn.Module):
    """Psi : Z -> Psi(Z), an equivariant, NONLINEAR function of the state alone.

    Nonlinearity comes from a self tensor-product Z (x) Z (quadratic, equivariant: 0e(x)0e->0e,
    1o(x)1o->0e, 0e(x)1o->1o) followed by an e3nn Gate (tanh on feature scalars; sigmoid gates
    scale the vectors by INVARIANT scalars, preserving the Type-1 transformation law), plus a
    linear residual path so the map does not vanish at small ||Z||.

    The state is passed through EquivariantNorm FIRST, which bounds the quadratic term and makes
    the flow globally well-posed (see EquivariantNorm). Nonlinearity in Z is unrestricted by the
    CDE form -- only the action on dX must stay linear.
    """

    def __init__(self, irreps_Z: o3.Irreps, n_s: int, n_v: int, layout: IrrepLayout):
        super().__init__()
        irreps_scalars = o3.Irreps(f"{n_s}x0e")
        irreps_gates = o3.Irreps(f"{n_v}x0e")
        irreps_gated = o3.Irreps(f"{n_v}x1o")
        self.norm = EquivariantNorm(layout)
        self.gate = Gate(irreps_scalars, [torch.tanh], irreps_gates, [torch.sigmoid], irreps_gated)
        self.tp_self = o3.FullyConnectedTensorProduct(irreps_Z, irreps_Z, self.gate.irreps_in)
        self.res = o3.Linear(irreps_Z, self.gate.irreps_out, biases=False)
        self.irreps_out = self.gate.irreps_out

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        zn = self.norm(z)
        return self.gate(self.tp_self(zn, zn)) + self.res(zn)


# =============================================================================
# 3. The CDE vector field  f_theta   (LINEAR in the control derivative)
# =============================================================================
class EquivariantCDEField(nn.Module):
    """dZ/ds = f_theta(Z) . dX~/ds  with  X~(s) = (s, X_1(s), ..., X_J(s)).

    Assembled as
        dZ/ds = gain * [  f_time(Psi(Z)) * (dt/ds)
                        + sum_j  TP( Psi(Z), dX_j/ds ; W_j )  ]

    `f_time` is an equivariant o3.Linear (no bias: a constant non-scalar bias would be a fixed
    vector that does not rotate, breaking equivariance -- e3nn structurally refuses to place a
    bias on a 1o output, which the certificate already probes).

    `W_j = weight_net(joint_emb[j])` are per-joint tensor-product weights generated from a
    learned Type-0 (invariant) per-joint embedding. Joints are thereby DISTINGUISHABLE -- the
    whole point of the rebuild -- while the weights themselves are rotation-invariant scalars,
    so equivariance is untouched. (Standard e3nn practice; cf. NequIP's radial weight nets.)
    """

    def __init__(self, layout: IrrepLayout, n_joints: int, emb_dim: int = 16,
                 hidden: int = 64, gain: float = 0.2):
        super().__init__()
        n_s, n_v = layout.n_scalar, layout.n_vec
        self.layout = layout
        self.n_joints = n_joints
        self.gain = gain

        self.irreps_Z = o3.Irreps(f"{n_s}x0e + {n_v}x1o")
        self.psi = EquivariantStateMap(self.irreps_Z, n_s, n_v, layout)
        irreps_psi = self.psi.irreps_out

        # (a) control-coupling: bilinear in (Psi(Z), dX_j) => LINEAR in dX_j.  Weights per joint.
        self.tp_ctrl = o3.FullyConnectedTensorProduct(
            irreps_psi, o3.Irreps("1x1o"), self.irreps_Z,
            shared_weights=False, internal_weights=False,
        )
        self.joint_emb = nn.Parameter(torch.randn(n_joints, emb_dim) * 0.5)
        self.weight_net = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, self.tp_ctrl.weight_numel),
        )
        # (b) time channel of the augmented control (dt/ds = 1): equivariant, bias-free.
        self.f_time = o3.Linear(irreps_psi, self.irreps_Z, biases=False)

        self.nfe = 0                      # function evaluations (efficiency reporting)

    def forward(self, z: torch.Tensor, dX: torch.Tensor) -> torch.Tensor:
        """z: (B, Zdim);  dX: (B, J, 3) = per-joint control derivative dX_j/ds."""
        self.nfe += 1
        B, J, _ = dX.shape
        psi = self.psi(z)                                            # (B, Pdim)

        W = self.weight_net(self.joint_emb)                          # (J, weight_numel) invariant
        psi_j = psi.unsqueeze(1).expand(B, J, -1).reshape(B * J, -1)
        dX_j = dX.reshape(B * J, 3)
        W_j = W.unsqueeze(0).expand(B, J, -1).reshape(B * J, -1)
        coupling = self.tp_ctrl(psi_j, dX_j, W_j)                    # (B*J, Zdim)
        coupling = coupling.reshape(B, J, -1).sum(dim=1)             # equivariant sum over joints

        return self.gain * (self.f_time(psi) + coupling)


# =============================================================================
# 4. Equivariant initial state  z0 = zeta(X(t_0))
# =============================================================================
class EquivariantInitial(nn.Module):
    """z0 = zeta(X(t_0)), equivariant by construction.

    Scalars (Type-0) are built from ROTATION INVARIANTS of the initial pose -- the joint radii
    ||x_j|| -- through an unconstrained MLP (any nonlinear map of invariants stays invariant).
    Vectors (Type-1) are an o3.Linear combination of the J initial joint vectors, which is
    exactly the equivariant linear map "Jx1o -> n_v x 1o".

    Note a 0e output cannot be produced from 1o inputs by a LINEAR map, which is why the scalar
    and vector routes are separate: the scalar route goes through invariants, the vector route
    through an equivariant linear layer. Both compose to an equivariant z0.
    """

    def __init__(self, layout: IrrepLayout, n_joints: int, hidden: int = 128,
                 n_exercises: int = 0):
        super().__init__()
        self.layout = layout
        self.n_joints = n_joints
        self.n_exercises = n_exercises
        # The exercise ID is a Type-0 (invariant) label: it enters ONLY the scalar route, where
        # it commutes with every D_f(g). Conditioning on it therefore costs the equivariance
        # certificate nothing -- which is why pooling the five exercises is architecturally free.
        self.scalar_net = nn.Sequential(
            nn.Linear(n_joints + n_exercises, hidden), nn.SiLU(),
            nn.Linear(hidden, layout.n_scalar),
        )
        self.vec_lin = o3.Linear(o3.Irreps(f"{n_joints}x1o"),
                                 o3.Irreps(f"{layout.n_vec}x1o"), biases=False)

    def forward(self, x0: torch.Tensor, ex_id: torch.Tensor = None) -> torch.Tensor:
        """x0: (B, J, 3) root-relative initial pose -> z0: (B, Zdim)."""
        radii = x0.norm(dim=-1)                                       # (B, J) invariant
        if self.n_exercises > 0:
            if ex_id is None:
                raise ValueError("model built with n_exercises > 0 but ex_id was not passed")
            onehot = torch.nn.functional.one_hot(ex_id, self.n_exercises).to(radii.dtype)
            radii = torch.cat([radii, onehot], dim=-1)                # still all invariants
        s = self.scalar_net(radii)                                    # (B, n_s)
        v = self.vec_lin(x0.reshape(x0.shape[0], -1))                 # (B, 3*n_v)
        return torch.cat([s, v], dim=-1)


# =============================================================================
# 5. Invariant read-out  h_psi(Z(t_N))
# =============================================================================
class InvariantHead(nn.Module):
    """s = h_psi(Z), INVARIANT: the whole point of the architecture.

    Invariant features of a [n_s x 0e + n_v x 1o] state under SO(3):
      * the scalar channels                       (n_s)
      * the norms of the vector channels          (n_v)
      * the PAIRWISE DOT PRODUCTS <v_i, v_j>      (n_v(n_v-1)/2)
    A rotation is orthogonal, so every dot product between two Type-1 channels survives it. The
    dots are what encode the RELATIVE GEOMETRY between learned directions -- norms alone say how
    big each channel is but nothing about the angles between them, and angles are most of the
    information. Dropping them was a real bottleneck: with n_v=16 the head saw 48 numbers where
    168 were available, and the model UNDERFIT (training loss stalled at ~6 MAD while the same
    architecture memorised 8 samples to ~0). Adding them costs nothing in equivariance.

    Feeding invariants through an unconstrained MLP keeps the output invariant no matter what the
    MLP learns -- viewpoint invariance of the SCORE is a theorem about the architecture, not a
    property to be trained.
    """

    def __init__(self, layout: IrrepLayout, hidden: int = 128, dropout: float = 0.1,
                 use_dots: bool = True, n_readout: int = 1):
        super().__init__()
        self.layout = layout
        self.use_dots = use_dots
        n_v = layout.n_vec
        n_inv = layout.n_scalar + n_v + (n_v * (n_v - 1) // 2 if use_dots else 0)
        self.n_inv = n_inv
        iu = torch.triu_indices(n_v, n_v, offset=1)
        self.register_buffer("iu_r", iu[0], persistent=False)
        self.register_buffer("iu_c", iu[1], persistent=False)
        self.net = nn.Sequential(
            nn.Linear(n_inv * n_readout, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def invariants(self, z: torch.Tensor) -> torch.Tensor:
        s, v = self.layout.split(z)                                   # (B,n_s), (B,n_v,3)
        feats = [s, v.norm(dim=-1)]
        if self.use_dots:
            # <v_i, v_j> for i<j : invariant under any orthogonal D_1o(g).
            dots = torch.einsum("bic,bjc->bij", v, v)
            feats.append(dots[:, self.iu_r, self.iu_c])
        return torch.cat(feats, dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, Zdim) or (B, K, Zdim) for a K-checkpoint trajectory read-out."""
        if z.dim() == 3:
            B, K, _ = z.shape
            inv = self.invariants(z.reshape(B * K, -1)).reshape(B, -1)
        else:
            inv = self.invariants(z)
        return self.net(inv).squeeze(-1)                               # (B,)


# =============================================================================
# 6. Fixed-step RK4 over PER-SAMPLE integration spans
# =============================================================================
def integrate_rk4(field: EquivariantCDEField, spline: BatchedNaturalCubicSpline,
                  z0: torch.Tensor, n_steps: int = 64,
                  n_readout: int = 1) -> torch.Tensor:
    """Integrate dZ/ds = f(Z) dX~/ds with fixed-step RK4 on each sample's OWN [t_0, t_N].

    Every sample gets the same NUMBER of steps but its own step SIZE h_b = (t_N^b - t_0^b)/n.
    Per-sample step sizes are scalars, so the stage-by-stage equivariance argument of
    PROJECT_BRIEF 5.1 is untouched (equivariant ops are closed under scalar-weighted
    combination). Fully differentiable: we backprop through the solver directly
    (discretise-then-optimise), which at 64 steps is cheaper and better-conditioned than the
    adjoint and needs no adjoint equivariance argument.

    n_readout > 1 returns Z at K EVENLY SPACED CHECKPOINTS along the flow, (B, K, Zdim), with the
    terminal state last. Reading out only Z(t_N) forces the entire sequence through a single
    80-dim bottleneck at the end -- which is what made the model underfit (training loss stalled
    while the very same architecture could memorise 8 samples). Every checkpoint Z(s_k) is an
    equivariant state, so the invariants taken at each are invariant, and concatenating
    invariants stays invariant: the certificate is untouched. This is the temporal pooling the
    baseline has and we had discarded.
    """
    J = field.n_joints
    s = spline.t0                                                     # (B,)
    h = (spline.tN - spline.t0) / n_steps                             # (B,)
    z = z0

    def dz(s_, z_):
        d = spline.derivative(s_)                                     # (B, J*3)
        return field(z_, d.reshape(d.shape[0], J, 3))

    # checkpoint after these step indices (1-based); always includes the final step.
    if n_readout > 1:
        marks = {int(round(n_steps * (k + 1) / n_readout)) for k in range(n_readout)}
    else:
        marks = {n_steps}
    outs = []

    hb = h.unsqueeze(-1)
    for i in range(n_steps):
        k1 = dz(s, z)
        k2 = dz(s + 0.5 * h, z + 0.5 * hb * k1)
        k3 = dz(s + 0.5 * h, z + 0.5 * hb * k2)
        k4 = dz(s + h, z + hb * k3)
        z = z + (hb / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        s = s + h
        if (i + 1) in marks:
            outs.append(z)

    if n_readout > 1:
        return torch.stack(outs, dim=1)                               # (B, K, Zdim)
    return z


# =============================================================================
# 7. The full model
# =============================================================================
class SE3NeuralCDE(nn.Module):
    """Assessment as an SE(3)-equivariant flow on the pose manifold, in continuous time.

        x (irregular stamps)  ->  root-relative  ->  natural cubic spline X~
        z0 = zeta(X(t_0))     ->  dZ/ds = f_theta(Z) dX~/ds  ->  s = h_psi(Z(t_N))

    Translation invariance is automatic (root-relative coordinates; a constant is annihilated
    by dX). Rotation equivariance holds stage-by-stage through the solver. The read-out is
    invariant. Hence s(g.x) = s(x) for all g in SE(3) -- by construction, at every step.

    Input timestamps are the sensor's ACTUAL arrival times. There is no frame grid to resample
    onto, which is what the irregular-sampling experiments exploit.
    """

    def __init__(self, n_joints: int = 25, n_scalar: int = 32, n_vec: int = 16,
                 emb_dim: int = 16, hidden: int = 64, head_hidden: int = 128,
                 gain: float = 0.2, n_steps: int = 64, dropout: float = 0.1,
                 n_exercises: int = 0, n_readout: int = 1, use_dots: bool = True):
        super().__init__()
        self.layout = IrrepLayout(n_scalar, n_vec)
        self.n_joints = n_joints
        self.n_steps = n_steps
        self.n_readout = n_readout
        self.field = EquivariantCDEField(self.layout, n_joints, emb_dim, hidden, gain)
        self.initial = EquivariantInitial(self.layout, n_joints, n_exercises=n_exercises)
        self.head = InvariantHead(self.layout, head_hidden, dropout,
                                  use_dots=use_dots, n_readout=n_readout)

    def terminal_state(self, t: torch.Tensor, x_rel: torch.Tensor,
                       n_steps: int = None, ex_id: torch.Tensor = None,
                       n_readout: int = None) -> torch.Tensor:
        """t: (B, L) actual stamps;  x_rel: (B, L, J, 3) root-relative.

        -> Z(t_N): (B, Zdim), or the K-checkpoint trajectory (B, K, Zdim) if n_readout > 1.
        """
        B, L, J, _ = x_rel.shape
        spline = BatchedNaturalCubicSpline(t, x_rel.reshape(B, L, J * 3))
        z0 = self.initial(x_rel[:, 0], ex_id)
        k = self.n_readout if n_readout is None else n_readout
        return integrate_rk4(self.field, spline, z0, n_steps or self.n_steps, n_readout=k)

    def forward(self, t: torch.Tensor, x_rel: torch.Tensor, n_steps: int = None,
                ex_id: torch.Tensor = None) -> torch.Tensor:
        return self.head(self.terminal_state(t, x_rel, n_steps, ex_id))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
