#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cde_model_mp.py
===============
The SE(3)-equivariant Neural CDE with a MESSAGE-PASSING vector field over the skeleton graph.
This is PROJECT_BRIEF 3.3 as actually specified:

    "f_theta operates only on RELATIVE joint geometry (differences x_i - x_j and their
     SPHERICAL-HARMONIC embeddings) -> translation invariance is automatic"

Why the previous field (cde_model.SE3NeuralCDE) was not enough
--------------------------------------------------------------
It coupled a single GLOBAL latent with each joint's root-relative velocity dX_j independently.
The field therefore never saw a bone: no relative vector between two joints, no spherical-harmonic
direction embedding, no inter-joint distance. It had to invent skeletal geometry from scratch,
through the flow, from 45 training subjects -- and it didn't. Measured, pooled, subject-disjoint:

    PCT (target arch)                         MAD 6.07
    ridge on hand-crafted SO(3)-INVARIANTS    MAD 7.68     <- a LINEAR probe
    per-exercise mean floor                   MAD 8.25
    global-latent CDE (cde_model)             MAD 8.19     <- worse than the linear probe

Losing to a linear model on features it is fully entitled to compute is an expressivity failure,
not a symmetry failure: a companion probe (gravity_probe.py) showed SO(3)-invariant features do
carry signal (7.68 < 8.25 floor), so imposing invariance is NOT what broke it. The field simply
was not given the geometry. This module gives it the geometry.

The design
----------
State is PER JOINT, Z_i = [ n_s scalars (0e) | pos_i (1o) | n_v vectors (1o) ].

POSITION LIVES IN THE STATE, and that is the crux of making this a legal CDE. A strict Neural CDE
requires dZ/ds = f(Z) dX/ds with f a function of the STATE ALONE -- so f may not simply look up
X(s). We therefore carry position as a state channel with the exact identity coupling

    d pos_i / ds  =  dX_i / ds

so pos_i(s) = X_i(s) for free, equivariantly, and f can form r_ij = pos_i - pos_j from Z alone.
(This is the standard control-augmentation trick, and it costs nothing: the coupling is linear.)

The field, per bone (i,j) of the Kinect skeleton:

    r_ij  = pos_i - pos_j                                (1o, from the state)
    d_ij  = ||r_ij||                                     (0e, invariant)
    Y_ij  = SphericalHarmonics_{l<=2}(r_ij / d_ij)       (0e + 1o + 2e)
    w_ij  = RadialMLP(d_ij, bone_embedding)              (INVARIANT weights)
    m_ij  = TensorProduct(h_j, Y_ij ; w_ij)              (CG-constrained => equivariant)
    msg_i = sum over the skeleton neighbours j of m_ij

and the full integrand

    d h_i / ds  =  gain * [  Linear(msg_i) * (dt/ds)            <- drift; the time channel
                           + TP(h_i, dX_i ; W_i)                <- control coupling, per joint
                          ]

Both terms are LINEAR in the augmented control derivative (dt/ds, dX/ds) = (1, dX/ds): msg_i
depends only on Z, and TP(., dX_i ; .) is bilinear hence linear in dX_i. So this remains a genuine
Neural CDE, and the time-augmentation keeps the model sensitive to movement TEMPO (a pure
int f(Z) dX is invariant to reparametrisation of s, i.e. blind to speed).

EQUIVARIANCE IS UNCHANGED AND UNCONDITIONAL. Every operation is a Clebsch-Gordan tensor product,
an o3.Linear without bias, a scalar (invariant) gate, or a sum -- each equivariant for ANY weights.
The representation on the solver-visible state is block-diagonal over joints, identity on the 0e
channels and Wigner-D on each 1o channel, hence ORTHOGONAL. So the N_eq error norm partition and
the whole Task-1/2/4/6 certificate apply unchanged; certify_trainable.py re-runs against this
model and must still pass. It is not assumed -- it is re-measured.
"""

import math

import torch
import torch.nn as nn

import e3nn
e3nn.set_optimization_defaults(jit_script_fx=False)
from e3nn import o3                      # noqa: E402
from e3nn.nn import Gate                 # noqa: E402

from equivariance_suite import IrrepLayout       # noqa: E402
from cde_model import BatchedNaturalCubicSpline  # noqa: E402


# =============================================================================
# Kinect v2 skeleton (25 joints, 24 bones)
# =============================================================================
KINECT_BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3),                       # spine + head
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),      # left arm + hand
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),  # right arm + hand
    (0, 12), (12, 13), (13, 14), (14, 15),                  # left leg
    (0, 16), (16, 17), (17, 18), (18, 19),                  # right leg
]


def skeleton_edges(bones=KINECT_BONES):
    """Bidirectional edge index (2, E) -- messages flow both ways along every bone."""
    src, dst, etype = [], [], []
    for k, (a, b) in enumerate(bones):
        src += [a, b]
        dst += [b, a]
        etype += [k, k]                                      # same bone, both directions
    return (torch.tensor(src, dtype=torch.long),
            torch.tensor(dst, dtype=torch.long),
            torch.tensor(etype, dtype=torch.long))


# =============================================================================
# Per-joint state layout:  [ n_s x 0e | pos (1o) | n_v x 1o ]
# =============================================================================
class JointLayout:
    """Irrep layout of ONE joint's state. The solver-visible latent is J of these, so the
    representation is block-diagonal over joints and remains orthogonal."""

    def __init__(self, n_scalar: int, n_vec: int):
        self.n_scalar = n_scalar
        self.n_vec = n_vec                                   # feature vectors, EXCLUDING pos
        self.n_vec_total = n_vec + 1                         # + the carried position
        self.dim = n_scalar + 3 * self.n_vec_total
        # IrrepLayout view used by N_eq / apply_rho (scalars first, then all 1o blocks)
        self.irrep = IrrepLayout(n_scalar, self.n_vec_total)
        self.irreps_h = o3.Irreps(f"{n_scalar}x0e + {n_vec}x1o")     # state minus pos
        self.h_dim = n_scalar + 3 * n_vec

    def split(self, z):
        """z: (B, J, dim) -> (h (B,J,h_dim), pos (B,J,3))."""
        s = z[..., : self.n_scalar]
        v = z[..., self.n_scalar:].reshape(*z.shape[:-1], self.n_vec_total, 3)
        pos = v[..., 0, :]
        vec = v[..., 1:, :].reshape(*z.shape[:-1], 3 * self.n_vec)
        return torch.cat([s, vec], dim=-1), pos

    def join(self, h, pos):
        s = h[..., : self.n_scalar]
        vec = h[..., self.n_scalar:].reshape(*h.shape[:-1], self.n_vec, 3)
        v = torch.cat([pos.unsqueeze(-2), vec], dim=-2)
        return torch.cat([s, v.reshape(*h.shape[:-1], 3 * self.n_vec_total)], dim=-1)


# =============================================================================
# The message-passing vector field
# =============================================================================
class SkeletonMPField(nn.Module):
    """f_theta: equivariant message passing on the skeleton, LINEAR in the control derivative."""

    def __init__(self, layout: JointLayout, n_joints: int = 25, lmax: int = 2,
                 radial_hidden: int = 64, n_rbf: int = 16, gain: float = 1.0):
        super().__init__()
        self.layout = layout
        self.n_joints = n_joints
        self.gain = gain
        self.nfe = 0

        src, dst, etype = skeleton_edges()
        self.register_buffer("src", src, persistent=False)
        self.register_buffer("dst", dst, persistent=False)
        self.register_buffer("etype", etype, persistent=False)
        self.n_edges = src.numel()
        self.n_bones = len(KINECT_BONES)

        irreps_h = layout.irreps_h
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)          # 0e + 1o + 2e

        # ---- message: TP(h_j, Y_ij) with INVARIANT weights from (d_ij, bone id) -----------
        self.tp_msg = o3.FullyConnectedTensorProduct(
            irreps_h, self.irreps_sh, irreps_h,
            shared_weights=False, internal_weights=False)
        self.n_rbf = n_rbf
        self.bone_emb = nn.Parameter(torch.randn(self.n_bones, 8) * 0.5)
        self.radial = nn.Sequential(
            nn.Linear(n_rbf + 8, radial_hidden), nn.SiLU(),
            nn.Linear(radial_hidden, self.tp_msg.weight_numel),
        )
        self.lin_msg = o3.Linear(irreps_h, irreps_h, biases=False)

        # ---- control coupling: TP(h_i, dX_i) with per-joint invariant weights -------------
        self.tp_ctrl = o3.FullyConnectedTensorProduct(
            irreps_h, o3.Irreps("1x1o"), irreps_h,
            shared_weights=False, internal_weights=False)
        self.joint_emb = nn.Parameter(torch.randn(n_joints, 16) * 0.5)
        self.ctrl_net = nn.Sequential(
            nn.Linear(16, radial_hidden), nn.SiLU(),
            nn.Linear(radial_hidden, self.tp_ctrl.weight_numel),
        )

    def _rbf(self, d):
        """Gaussian radial basis on the bone length. A function of an INVARIANT, so invariant."""
        centres = torch.linspace(0.0, 3.0, self.n_rbf, device=d.device, dtype=d.dtype)
        return torch.exp(-((d.unsqueeze(-1) - centres) ** 2) / 0.15)

    def _bounded(self, h: torch.Tensor) -> torch.Tensor:
        """Equivariant per-irrep bounded normalisation of the FEATURE state h (PROJECT_BRIEF 6.3).

        Each feature is rescaled by a scalar function of its own invariant norm,
            f -> f / sqrt(1 + ||f||^2 / d_f),
        which commutes with D_f(g) (a scalar times an irrep is the same irrep) and is strictly
        monotone in ||f||, so magnitude is compressed rather than destroyed.

        This is NOT cosmetic and NOT optional. Without it the message-passing field is unbounded
        in Z: measured at initialisation, ||dz/ds|| = 40.8 for ||z|| = 18.8, and training drove
        the weights to NaN within one epoch. Bounding the field's input caps ||f||, so ||Z(s)||
        can grow at most LINEARLY in s and the flow is globally well-posed. (The same fix was
        already required for the global-latent field; it simply was not carried over here.)

        Note the POSITION channel is deliberately excluded -- it is not a free feature, it is
        pinned to the control by d pos/ds = dX, and squashing it would corrupt the bone geometry.
        """
        n_s = self.layout.n_scalar
        s = h[..., :n_s]
        v = h[..., n_s:].reshape(*h.shape[:-1], self.layout.n_vec, 3)
        s = s / torch.sqrt(1.0 + s ** 2)
        vn = v.pow(2).sum(-1, keepdim=True) / 3.0
        v = v / torch.sqrt(1.0 + vn)
        return torch.cat([s, v.reshape(*h.shape[:-1], 3 * self.layout.n_vec)], dim=-1)

    def forward(self, z: torch.Tensor, dX: torch.Tensor) -> torch.Tensor:
        """z: (B, J, dim);  dX: (B, J, 3) -> dz/ds: (B, J, dim)."""
        self.nfe += 1
        B, J, _ = dX.shape
        h, pos = self.layout.split(z)                                  # (B,J,hd), (B,J,3)
        h = self._bounded(h)                                           # keeps the flow well-posed

        # ---- bone geometry, computed FROM THE STATE (this is why pos is carried) ---------
        r = pos[:, self.dst] - pos[:, self.src]                        # (B,E,3)  1o
        d = r.norm(dim=-1)                                             # (B,E)    0e invariant
        Y = o3.spherical_harmonics(self.irreps_sh, r, normalize=True,
                                   normalization="component")          # (B,E,9)
        femb = self.bone_emb[self.etype].unsqueeze(0).expand(B, -1, -1)
        w = self.radial(torch.cat([self._rbf(d), femb], dim=-1))       # (B,E,weight_numel)

        msg = self.tp_msg(h[:, self.src].reshape(B * self.n_edges, -1),
                          Y.reshape(B * self.n_edges, -1),
                          w.reshape(B * self.n_edges, -1))
        msg = msg.reshape(B, self.n_edges, -1)

        agg = torch.zeros(B, J, msg.shape[-1], dtype=msg.dtype, device=msg.device)
        agg.index_add_(1, self.dst, msg)                               # sum over neighbours
        drift = self.lin_msg(agg.reshape(B * J, -1)).reshape(B, J, -1)

        # ---- control coupling: bilinear in dX => LINEAR in dX ----------------------------
        Wc = self.ctrl_net(self.joint_emb)                             # (J, weight_numel)
        Wc = Wc.unsqueeze(0).expand(B, -1, -1).reshape(B * J, -1)
        ctrl = self.tp_ctrl(h.reshape(B * J, -1), dX.reshape(B * J, 3), Wc).reshape(B, J, -1)

        dh = self.gain * (drift + ctrl)
        return self.layout.join(dh, dX)                                # d pos/ds = dX exactly


# =============================================================================
# Initial state  z0 = zeta(X(t_0))
# =============================================================================
class MPInitial(nn.Module):
    """Per-joint z0. Scalars from invariants (joint radius + exercise ID); vectors are learned
    scalar gains on the joint's own position vector (a scalar times a 1o is 1o => equivariant);
    the carried position channel is initialised to the actual initial pose."""

    def __init__(self, layout: JointLayout, n_joints: int = 25, n_exercises: int = 0,
                 hidden: int = 64):
        super().__init__()
        self.layout = layout
        self.n_exercises = n_exercises
        self.scalar_net = nn.Sequential(
            nn.Linear(1 + 16 + n_exercises, hidden), nn.SiLU(),
            nn.Linear(hidden, layout.n_scalar),
        )
        self.joint_emb = nn.Parameter(torch.randn(n_joints, 16) * 0.5)
        self.vec_gain = nn.Parameter(torch.randn(layout.n_vec) * 0.3)

    def forward(self, x0: torch.Tensor, ex_id: torch.Tensor = None) -> torch.Tensor:
        B, J, _ = x0.shape
        radius = x0.norm(dim=-1, keepdim=True)                          # (B,J,1) invariant
        emb = self.joint_emb.unsqueeze(0).expand(B, -1, -1)
        feats = [radius, emb]
        if self.n_exercises > 0:
            oh = torch.nn.functional.one_hot(ex_id, self.n_exercises).to(x0.dtype)
            feats.append(oh.unsqueeze(1).expand(B, J, -1))
        s = self.scalar_net(torch.cat(feats, dim=-1))                   # (B,J,n_s)
        vec = x0.unsqueeze(-2) * self.vec_gain.view(1, 1, -1, 1)        # (B,J,n_v,3)
        h = torch.cat([s, vec.reshape(B, J, -1)], dim=-1)
        return self.layout.join(h, x0)                                  # pos channel = X(t_0)


# =============================================================================
# Invariant read-out
# =============================================================================
class MPInvariantHead(nn.Module):
    """Invariants of the per-joint state, pooled over joints, plus BONE LENGTHS.

    Bone lengths ||pos_i - pos_j|| are invariant and are precisely the features the linear ridge
    probe used to beat the previous model. We hand them to the head directly rather than hoping
    the flow rediscovers them.
    """

    def __init__(self, layout: JointLayout, n_joints: int = 25, hidden: int = 256,
                 dropout: float = 0.1, n_readout: int = 1):
        super().__init__()
        self.layout = layout
        n_v = layout.n_vec
        src, dst, _ = skeleton_edges()
        self.register_buffer("bsrc", src[::2].clone(), persistent=False)   # one per bone
        self.register_buffer("bdst", dst[::2].clone(), persistent=False)
        n_bones = self.bsrc.numel()

        iu = torch.triu_indices(n_v + 1, n_v + 1, offset=1)   # dots among [pos, v_1..v_nv]
        self.register_buffer("iu_r", iu[0], persistent=False)
        self.register_buffer("iu_c", iu[1], persistent=False)

        per_joint = layout.n_scalar + (n_v + 1) + iu.shape[1]           # scalars, norms, dots
        n_inv = 2 * per_joint + n_bones                                 # mean+max pooled, +bones
        self.net = nn.Sequential(
            nn.Linear(n_inv * n_readout, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def invariants(self, z: torch.Tensor) -> torch.Tensor:
        """Invariants of the per-joint state. Every quantity here is BOUNDED on purpose.

        The dot products <v_i, v_j> are QUADRATIC in state magnitude, and ||Z(s)|| can still grow
        linearly along the flow even with the field bounded (measured: ||Z(t_N)|| ~ 240). Feeding
        raw dots to an unconstrained MLP is therefore an unbounded input, and it blew up: on
        pooled fold 2 the model emitted predictions in the hundreds (test RMSE 123 on a 0-50
        scale). Bounding the FIELD's input was not sufficient -- the READ-OUT needs it too.

        Fix: take the dots between UNIT-NORMALISED vector channels, i.e. cos angles in [-1, 1],
        and pass the magnitudes through a log1p. The cosine carries the RELATIVE GEOMETRY (which
        is the information we wanted from the dots) with the scale divided out, and log1p keeps
        magnitude information without letting it dominate. Both are scalar functions of invariant
        quantities, so invariance is untouched -- certify_mp G3 re-checks it.
        """
        L = self.layout
        s = z[..., : L.n_scalar]                                        # (B,J,n_s)
        v = z[..., L.n_scalar:].reshape(*z.shape[:-1], L.n_vec_total, 3)
        norms = v.norm(dim=-1)                                          # (B,J,1+n_v)

        vhat = v / (norms.unsqueeze(-1) + 1e-6)                         # unit directions
        cos = torch.einsum("bjic,bjkc->bjik", vhat, vhat)[..., self.iu_r, self.iu_c]
        per_joint = torch.cat([torch.tanh(s), torch.log1p(norms), cos], dim=-1)
        pooled = torch.cat([per_joint.mean(1), per_joint.amax(1)], dim=-1)

        pos = v[..., 0, :]                                              # (B,J,3)
        bone = (pos[:, self.bdst] - pos[:, self.bsrc]).norm(dim=-1)     # (B,n_bones) invariant
        return torch.cat([pooled, torch.log1p(bone)], dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 4:                                                # (B,K,J,dim)
            B, K = z.shape[0], z.shape[1]
            inv = self.invariants(z.reshape(B * K, *z.shape[2:])).reshape(B, -1)
        else:
            inv = self.invariants(z)
        return self.net(inv).squeeze(-1)


# =============================================================================
# Integrator + model
# =============================================================================
def integrate_rk4_mp(field, spline, z0, n_steps=32, n_readout=1):
    """Fixed-step RK4 on each sample's own [t_0, t_N]. Exactly equivariance-preserving
    (PROJECT_BRIEF 5.1: scalar Butcher coefficients commute with the group action)."""
    J = field.n_joints
    s = spline.t0
    h = (spline.tN - spline.t0) / n_steps
    z = z0
    hb = h.view(-1, 1, 1)

    def dz(s_, z_):
        d = spline.derivative(s_)
        return field(z_, d.reshape(d.shape[0], J, 3))

    marks = ({int(round(n_steps * (k + 1) / n_readout)) for k in range(n_readout)}
             if n_readout > 1 else {n_steps})
    outs = []
    for i in range(n_steps):
        k1 = dz(s, z)
        k2 = dz(s + 0.5 * h, z + 0.5 * hb * k1)
        k3 = dz(s + 0.5 * h, z + 0.5 * hb * k2)
        k4 = dz(s + h, z + hb * k3)
        z = z + (hb / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        s = s + h
        if (i + 1) in marks:
            outs.append(z)
    return torch.stack(outs, dim=1) if n_readout > 1 else z


class SE3MessagePassingCDE(nn.Module):
    """SE(3)-equivariant Neural CDE with a skeleton message-passing vector field."""

    def __init__(self, n_joints: int = 25, n_scalar: int = 32, n_vec: int = 8,
                 lmax: int = 2, hidden: int = 64, head_hidden: int = 256,
                 gain: float = 1.0, n_steps: int = 32, dropout: float = 0.1,
                 n_exercises: int = 0, n_readout: int = 4):
        super().__init__()
        self.layout = JointLayout(n_scalar, n_vec)
        self.n_joints = n_joints
        self.n_steps = n_steps
        self.n_readout = n_readout
        self.field = SkeletonMPField(self.layout, n_joints, lmax, hidden, gain=gain)
        self.initial = MPInitial(self.layout, n_joints, n_exercises)
        self.head = MPInvariantHead(self.layout, n_joints, head_hidden, dropout, n_readout)

    def terminal_state(self, t, x_rel, n_steps=None, ex_id=None, n_readout=None):
        B, L, J, _ = x_rel.shape
        spline = BatchedNaturalCubicSpline(t, x_rel.reshape(B, L, J * 3))
        z0 = self.initial(x_rel[:, 0], ex_id)
        k = self.n_readout if n_readout is None else n_readout
        return integrate_rk4_mp(self.field, spline, z0, n_steps or self.n_steps, n_readout=k)

    def forward(self, t, x_rel, n_steps=None, ex_id=None):
        return self.head(self.terminal_state(t, x_rel, n_steps, ex_id))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
