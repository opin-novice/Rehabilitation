#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
equivariant_gru.py
==================
THE METHOD. SE(3)-equivariant, irregular-sampling-native, and NOT a flow.

Why this replaces the Neural CDE
--------------------------------
The CDE was never the claim -- it was a MEANS to two claims: viewpoint invariance and native
irregular sampling. Measured on pooled subject-disjoint KIMORE, it is the one component that
delivers neither and costs everything:

    model                              MAD     deep  invariant  continuous-flow   beats floor?
    -------------------------------- ------   ----  ---------  ---------------   ------------
    PCT (target architecture)          6.07    yes         NO               no    (reference)
    GRU on hand-crafted invariants     6.54    yes        yes               no    YES  5/5
    MLP on hand-crafted invariants     6.66    yes        yes               no    YES
    ridge on hand-crafted invariants   7.68     no        yes               no    YES
    per-exercise mean floor            8.25       -          -                -    -
    SE(3)-equivariant Neural CDE       8.43    yes        yes              YES    NO   (n.s.)

Swap the continuous-time flow for a discrete-time recurrence -- same invariance, same
information, same folds, same 45 subjects -- and MAD goes 8.43 -> 6.54. The flow was the sole
variable. Equivariance is exonerated: an invariant deep model matches the NON-equivariant
transformer to within noise. So we keep the symmetry and drop the flow.

Both structural pillars survive intact:

  1. VIEWPOINT INVARIANCE remains a THEOREM, not a learned property. The spatial encoder is
     equivariant (Clebsch-Gordan tensor products, spherical-harmonic bone embeddings, no
     non-scalar biases); it is followed by an INVARIANT PROJECTION; and every layer after that
     -- the recurrence, the head -- sees only invariants. A function of invariants is invariant
     no matter what it learns. s(g.x) = s(x) for all g in SE(3), by construction, and it is
     re-measured in certify_egru.py rather than asserted.

  2. IRREGULAR SAMPLING remains NATIVE. The model consumes the sensor's ACTUAL inter-arrival
     times dt_k as an input feature at every step. There is no frame grid and no resampling
     operator: irregular stamps are data, not a problem to be corrected. Baselines with a fixed
     frame axis must still apply a resampling operator R and still lose the spectral energy that
     costs them (corruption_pipeline.spectral_loss). The Block-2 argument is unchanged.

What we give up: the Riemann-Stieltjes/CDE formalism. What we keep: both things the paper
actually claims, plus roughly two MAD points of accuracy.

Architecture
------------
    x (B,T,J,3) root-relative, actual stamps t (B,T)
      -> per-frame EQUIVARIANT skeleton message passing (l<=2 spherical harmonics on bones)
      -> per-frame INVARIANT projection   (scalars, norms, cosines, bone lengths)   <-- the cut
      -> [invariants ; dt_k]  ->  bidirectional GRU  ->  masked mean-pool  ->  MLP  ->  score

Everything left of the cut is equivariant; everything right of it is invariant. The cut is where
the theorem is discharged.
"""

import torch
import torch.nn as nn

import e3nn
e3nn.set_optimization_defaults(jit_script_fx=False)
from e3nn import o3                      # noqa: E402
from e3nn.nn import Gate                 # noqa: E402

import chirality as ch                                   # noqa: E402
from cde_model_mp import KINECT_BONES, skeleton_edges     # noqa: E402


# =============================================================================
# 1. Equivariant spatial encoder (one frame at a time, shared weights)
# =============================================================================
class EquivariantSkeletonEncoder(nn.Module):
    """Message passing on the Kinect skeleton with spherical-harmonic bone embeddings.

    ROTATION-equivariant for ANY weights: geometry enters only through
        r_ij = x_i - x_j          (a 1o vector)
        Y_ij = SH_{l<=2}(r_ij)    (steerable)
        w_ij = MLP(||r_ij||, bone_id)   (a function of an INVARIANT => invariant weights)
        m_ij = TP(h_j, Y_ij ; w_ij)     (Clebsch-Gordan constrained)
    and through Gate nonlinearities, whose sigmoid gates are invariant scalars scaling the 1o
    channels -- which preserves the Type-1 transformation law. No bias is ever placed on a
    non-scalar irrep (e3nn structurally refuses to).

    TRANSLATION invariance is NOT supplied by this encoder and we do not pretend otherwise: the
    initialisation reads each joint's radius ||x_j|| and seeds its vector channels from x_j, so
    the encoder is translation-invariant only with respect to the ROOT. Translation invariance of
    the pipeline is discharged upstream, by the root-subtraction x_j -> x_j - x_root in
    kimore_cde_data.load_sample. certify_egru E5 tests it there, on raw world coordinates, which
    is the only place the claim is true and the only place it matters.
    """

    def __init__(self, n_scalar=32, n_vec=8, n_layers=2, lmax=2,
                 n_joints=25, n_rbf=16, radial_hidden=64):
        super().__init__()
        self.n_scalar, self.n_vec, self.n_joints = n_scalar, n_vec, n_joints
        self.irreps_h = o3.Irreps(f"{n_scalar}x0e + {n_vec}x1o")
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.n_rbf = n_rbf

        src, dst, etype = skeleton_edges()
        self.register_buffer("src", src, persistent=False)
        self.register_buffer("dst", dst, persistent=False)
        self.register_buffer("etype", etype, persistent=False)
        self.n_edges = src.numel()

        # DENSE INCIDENCE, NOT SCATTER.  A[j, e] = 1 iff edge e terminates at joint j, so
        #     agg_j = sum_{e: dst(e)=j} m_e   ==   (A @ m)_j
        # is EXACTLY the sum index_add_ computed -- same value, same weights, existing checkpoints
        # unaffected -- but with a FIXED reduction order instead of CUDA atomics. That is what
        # removes the +/-0.33 MAD run-to-run spread (see determinism.py). The graph is the Kinect
        # skeleton and never changes, so (J=25) x (E=48) is a trivial dense matmul; the scatter was
        # buying us nondeterminism in exchange for nothing.
        A = torch.zeros(n_joints, self.n_edges)
        A[dst, torch.arange(self.n_edges)] = 1.0
        self.register_buffer("incidence", A, persistent=False)

        self.joint_emb = nn.Parameter(torch.randn(n_joints, 16) * 0.5)
        self.bone_emb = nn.Parameter(torch.randn(len(KINECT_BONES), 8) * 0.5)
        # h_0: scalars from invariants (joint radius + joint id); vectors from the joint's own
        # position scaled by learned SCALARS (scalar x 1o = 1o, so equivariant).
        self.init_scalar = nn.Sequential(
            nn.Linear(1 + 16, radial_hidden), nn.SiLU(), nn.Linear(radial_hidden, n_scalar))
        self.init_gain = nn.Parameter(torch.randn(n_vec) * 0.3)

        # DEAD-NODE EMBEDDING (used only when a liveness mask is supplied).
        # Gating the EDGES of a failed joint is not enough: h_0 is SEEDED FROM x_j (radius above,
        # position below), so a node frozen at a stale pose would keep broadcasting a confidently
        # wrong pose into the graph forever. A failed sensor must say "I don't know", not "I am at
        # the origin". So a dead node's state is replaced by a learned embedding.
        #
        # ONLY THE SCALARS MAY BE LEARNED. The 1o channels of the dead embedding must be EXACTLY
        # ZERO: a learned CONSTANT vector does not rotate with the input, so h(Rx) != D(R) h(x)
        # and equivariance dies on the spot. Zero is the unique rotation-equivariant constant of
        # type 1. (certify_egru E1/E2 re-measures this rather than trusting the comment.)
        self.dead_scalar = nn.Parameter(torch.randn(n_scalar) * 0.1)

        self.tps, self.radials, self.gates, self.lins = (nn.ModuleList() for _ in range(4))
        for _ in range(n_layers):
            gate = Gate(o3.Irreps(f"{n_scalar}x0e"), [torch.tanh],
                        o3.Irreps(f"{n_vec}x0e"), [torch.sigmoid],
                        o3.Irreps(f"{n_vec}x1o"))
            tp = o3.FullyConnectedTensorProduct(
                self.irreps_h, self.irreps_sh, gate.irreps_in,
                shared_weights=False, internal_weights=False)
            self.tps.append(tp)
            self.radials.append(nn.Sequential(
                nn.Linear(n_rbf + 8, radial_hidden), nn.SiLU(),
                nn.Linear(radial_hidden, tp.weight_numel)))
            self.gates.append(gate)
            self.lins.append(o3.Linear(gate.irreps_out, self.irreps_h, biases=False))

    def _rbf(self, d):
        c = torch.linspace(0.0, 3.0, self.n_rbf, device=d.device, dtype=d.dtype)
        return torch.exp(-((d.unsqueeze(-1) - c) ** 2) / 0.15)

    def forward(self, x, mask=None):
        """x: (N, J, 3) root-relative -> h: (N, J, n_scalar + 3*n_vec) equivariant.

        mask: (N, J) joint liveness in {0,1}, or None.
          None  -> EXACTLY the original behaviour (raw-sum aggregation). Existing checkpoints
                   reproduce their numbers bit-for-bit up to fp reduction order.
          given -> occlusion-aware: dead nodes emit nothing, receive a learned "unknown" state,
                   and the aggregation is renormalised by SURVIVING in-degree so a half-occluded
                   neighbourhood delivers a message of the same SCALE as a full one (otherwise the
                   features of an occluded region silently shrink toward zero, which the head reads
                   as a confident measurement rather than as missing data).

        EQUIVARIANCE IS UNAFFECTED IN BOTH MODES. m_i is a Type-0 scalar; multiplying an
        equivariant message by a scalar and dividing by a scalar (the live degree) commutes with
        every D(g). The viewpoint theorem is untouched -- and re-measured, not assumed, by
        certify_egru E1/E2 with a random mask.
        """
        N, J, _ = x.shape
        radius = x.norm(dim=-1, keepdim=True)                          # (N,J,1) invariant
        emb = self.joint_emb.unsqueeze(0).expand(N, -1, -1)
        s = self.init_scalar(torch.cat([radius, emb], dim=-1))         # (N,J,n_s)
        v = x.unsqueeze(-2) * self.init_gain.view(1, 1, -1, 1)         # (N,J,n_v,3)

        e_live = None
        if mask is not None:
            mj = mask.unsqueeze(-1)                                    # (N,J,1)
            # dead -> learned scalars, ZERO vectors. x_j never enters h_0 for a dead joint.
            s = mj * s + (1.0 - mj) * self.dead_scalar.view(1, 1, -1)
            v = v * mj.unsqueeze(-1)
            e_live = mask[:, self.src] * mask[:, self.dst]             # (N,E) Type-0 edge gate

        h = torch.cat([s, v.reshape(N, J, -1)], dim=-1)

        r = x[:, self.dst] - x[:, self.src]                            # (N,E,3) 1o
        d = r.norm(dim=-1)                                             # (N,E)   invariant

        if e_live is not None:
            # A gated edge's message is multiplied by zero, so its GEOMETRY is irrelevant -- but it
            # is still COMPUTED, and a joint that failed by collapsing onto the root gives r = 0,
            # which makes spherical_harmonics(normalize=True) divide by zero and poison the whole
            # batch with NaN (0 * NaN = NaN). Substitute a safe unit vector on any edge we are
            # about to gate off; the result is discarded either way.
            ok = ((e_live > 0) & (d > 1e-6)).unsqueeze(-1)             # (N,E,1)
            safe = torch.zeros_like(r)
            safe[..., 2] = 1.0
            r = torch.where(ok, r, safe)
            d = torch.where(ok.squeeze(-1), d, torch.ones_like(d))

        Y = o3.spherical_harmonics(self.irreps_sh, r, normalize=True,
                                   normalization="component")
        femb = self.bone_emb[self.etype].unsqueeze(0).expand(N, -1, -1)
        rb = torch.cat([self._rbf(d), femb], dim=-1)

        for tp, radial, gate, lin in zip(self.tps, self.radials, self.gates, self.lins):
            w = radial(rb)
            m = tp(h[:, self.src].reshape(N * self.n_edges, -1),
                   Y.reshape(N * self.n_edges, -1),
                   w.reshape(N * self.n_edges, -1)).reshape(N, self.n_edges, -1)

            if e_live is None:
                # DENSE INCIDENCE, NOT SCATTER. Identical sum, fixed reduction order => the
                # +/-0.33 MAD run-to-run spread was never hardware; it was atomics. See
                # determinism.py.
                agg = torch.einsum("je,nec->njc", self.incidence, m)
            else:
                m = m * e_live.unsqueeze(-1)                           # dead nodes emit NOTHING
                num = torch.einsum("je,nec->njc", self.incidence, m)
                den = torch.einsum("je,ne->nj", self.incidence, e_live)
                agg = num / den.clamp(min=1.0).unsqueeze(-1)           # live-degree renormalise

            upd = lin(gate(agg.reshape(N * J, -1))).reshape(N, J, -1)
            if mask is None:
                h = h + upd                                             # residual; equivariant
            else:
                h = h + upd * mask.unsqueeze(-1)                        # dead state stays inert
        return h


# =============================================================================
# 2. The invariant cut
# =============================================================================
class InvariantProjection(nn.Module):
    """Equivariant features -> INVARIANTS. This is where the theorem is discharged.

    PARITY-EVEN content of [n_s x 0e + n_v x 1o]: the scalars, the norms of the vector channels,
    and the pairwise cosines between them. We use COSINES (unit-normalised dots) rather than raw
    dot products, and log1p on magnitudes -- raw dots are quadratic in feature scale and blew a
    previous model's predictions into the hundreds. Bounded inputs, same information.

    PARITY-ODD content (use_chiral=True): the pseudo-scalars. Every feature listed above is even
    under reflection, so with use_chiral=False this projection is invariant under the full O(3),
    not merely the SO(3) we claim -- and it is therefore blind to TRAJECTORY HANDEDNESS (a
    clockwise and a counter-clockwise limb circle have identical distances at every instant).
    Setting use_chiral=True admits the triple products det[vhat_a, vhat_b, vhat_c], which are the
    complete parity-odd invariant content of n_v type-1 vectors, plus fixed anatomical signed
    volumes. Those pick up det(rho(g)): +1 under rotation (still exactly invariant -- Block 3's
    0.000 degradation is untouched), -1 under reflection (O(3) invariance deliberately broken).
    See chirality.py for the full argument and certify_egru E6/E7/E8 for the measurements.
    """

    def __init__(self, n_scalar, n_vec, n_joints=25, use_chiral=False):
        super().__init__()
        self.n_scalar, self.n_vec = n_scalar, n_vec
        self.use_chiral = use_chiral
        iu = torch.triu_indices(n_vec, n_vec, offset=1)
        self.register_buffer("iu_r", iu[0], persistent=False)
        self.register_buffer("iu_c", iu[1], persistent=False)
        src, dst, _ = skeleton_edges()
        self.register_buffer("bsrc", src[::2].clone(), persistent=False)
        self.register_buffer("bdst", dst[::2].clone(), persistent=False)
        self.n_bones = self.bsrc.numel()
        per_joint = n_scalar + n_vec + iu.shape[1]
        n_odd = 0
        if use_chiral:
            tri = ch.triple_index(n_vec)                 # (3, C(n_vec,3)) LEARNED pseudo-scalars
            self.register_buffer("t_a", tri[0], persistent=False)
            self.register_buffer("t_b", tri[1], persistent=False)
            self.register_buffer("t_c", tri[2], persistent=False)
            per_joint += tri.shape[1]
            n_odd = ch.N_CHIRAL                          # ANATOMIC pseudo-scalars (not pooled)
        self.dim = 2 * per_joint + self.n_bones + n_odd  # mean+max over joints, + bones, + chi

    def forward(self, h, x, mask=None, ablate=None):
        """h: (N,J,hd) equivariant; x: (N,J,3) positions -> (N, dim) invariant.

        THE MASK MUST REACH THE READOUT, not merely the message passing. Every reduction below is
        a leak path for a failed joint, and each leaks differently:
          * mean over joints -- a dead node is averaged in as if it were a measurement;
          * MAX over joints  -- worse: a dead node can DOMINATE the max and hijack the feature;
          * bone lengths     -- any bone touching a dead joint is a garbage length;
          * pseudo-scalars   -- worst: they are DETERMINANTS, so a garbage vertex can flip the
                                SIGN, i.e. corrupt the single bit the parity construction exists
                                to carry (see chirality.frame_liveness).
        Gating all four is what makes the dead-node invariance EXACT rather than approximate.

        ablate: optional set of invariant-FAMILY names to zero out WITHOUT changing self.dim, so the
        marginal contribution of each family can be measured on an EXISTING checkpoint (no retrain).
        Names: {'scalars','norms','cosines','triples','volumes','bones'}. A zeroed family still
        occupies its slice (fed as zeros to the learned input weights); zeroing a per-joint family
        before the mean+max pool yields 0 mean AND 0 max, i.e. information removed, dimension kept.
        Default None == no ablation, bit-identical to the original forward. (Reviewer Q2.)
        """
        ablate = ablate or set()
        N, J, _ = h.shape
        s = h[..., : self.n_scalar]
        v = h[..., self.n_scalar:].reshape(N, J, self.n_vec, 3)
        norms = v.norm(dim=-1)
        vhat = v / (norms.unsqueeze(-1) + 1e-6)
        cos = torch.einsum("njic,njkc->njik", vhat, vhat)[..., self.iu_r, self.iu_c]
        feats = [torch.tanh(s), torch.log1p(norms), cos]
        # Zero the FEATURE, not the underlying norms/vhat, so ablating one family never corrupts the
        # computation of another (vhat feeds cosines and triple products, computed above).
        if "scalars" in ablate:
            feats[0] = torch.zeros_like(feats[0])
        if "norms" in ablate:
            feats[1] = torch.zeros_like(feats[1])
        if "cosines" in ablate:
            feats[2] = torch.zeros_like(feats[2])
        if self.use_chiral:
            # det[vhat_a, vhat_b, vhat_c] over UNIT vectors => bounded in [-1,1], exactly as the
            # cosines are. A raw triple product is CUBIC in feature scale; the same unboundedness
            # that broke an earlier model through quadratic dots would break it worse here.
            chi = (torch.cross(vhat[..., self.t_a, :], vhat[..., self.t_b, :], dim=-1)
                   * vhat[..., self.t_c, :]).sum(-1)                   # (N,J,n_triples)  parity-ODD
            if "triples" in ablate:
                chi = torch.zeros_like(chi)
            feats.append(chi)
        pj = torch.cat(feats, dim=-1)                                  # (N,J,P)

        bone = (x[:, self.bdst] - x[:, self.bsrc]).norm(dim=-1)        # invariant bone lengths
        if "bones" in ablate:
            bone = torch.zeros_like(bone)
        if mask is None:
            pooled = torch.cat([pj.mean(1), pj.amax(1)], dim=-1)
        else:
            mj = mask.unsqueeze(-1)                                    # (N,J,1)
            cnt = mj.sum(1).clamp(min=1.0)                             # (N,1) live joints
            mean = (pj * mj).sum(1) / cnt
            neg = torch.finfo(pj.dtype).min
            amax = torch.where(mj.expand_as(pj) > 0, pj,
                               torch.full_like(pj, neg)).amax(1)
            # A frame with NO live joint leaves every entry at the sentinel, and nan_to_num does not
            # catch that: finfo.min is FINITE (-3.4e38), not -inf, so the line below used to pass it
            # through untouched. It then overflows the head to +-inf and the logits to NaN. Zero it
            # explicitly, which is what the mean branch already does via cnt.clamp(min=1).
            # Measured on NTU: 152 / 18,674 clips (0.81%) hit this -- a second person entering
            # partway through a clip is 'present' but fully untracked in the frames before arrival.
            amax = torch.where(mj.sum(1) > 0, amax, torch.zeros_like(amax))
            amax = torch.nan_to_num(amax, neginf=0.0)                  # all-dead -> 0, not -inf
            pooled = torch.cat([mean, amax], dim=-1)
            bone = bone * (mask[:, self.bsrc] * mask[:, self.bdst])    # both endpoints must live

        out = [pooled, bone]
        if self.use_chiral:
            psi = ch.anatomic_pseudoscalars(x)                         # (N, N_CHIRAL) parity-ODD
            if mask is not None:
                psi = psi * ch.frame_liveness(mask)                    # a dead frame contributes 0
            if "volumes" in ablate:
                psi = torch.zeros_like(psi)
            out.append(psi)
        return torch.cat(out, dim=-1)


# =============================================================================
# 3. The model
# =============================================================================
class SE3EquivariantGRU(nn.Module):
    """SE(3)-invariant score from an irregularly sampled skeleton sequence.

    The recurrence consumes the ACTUAL inter-arrival time dt_k at every step, so irregular
    timestamps are an input, not a defect to be resampled away. No fixed frame grid exists
    anywhere in this model.
    """

    def __init__(self, n_scalar=32, n_vec=8, n_layers=2, lmax=2, n_joints=25,
                 gru_hidden=128, gru_layers=1, head_hidden=128,
                 dropout=0.2, n_exercises=5, use_speed=True, use_chiral=False,
                 use_mask=False, dt_mode="real", use_bandpower=False):
        super().__init__()
        self.encoder = EquivariantSkeletonEncoder(n_scalar, n_vec, n_layers, lmax, n_joints)
        self.proj = InvariantProjection(n_scalar, n_vec, n_joints, use_chiral=use_chiral)
        self.n_ex = n_exercises
        self.n_joints = n_joints
        self.use_speed = use_speed
        self.use_chiral = use_chiral
        # PHASE-0 dt ABLATION. The irregular-sampling claim says the recurrence USES the sensor's
        # real inter-arrival gaps. dt_mode isolates that, applying the SAME transform to the dt
        # INPUT CHANNEL and to the speed = d/dt DIVISION so no leak reconstructs the real grid:
        #   "real"    : true dt                         (the claim as stated -- control)
        #   "ones"    : dt := 1 everywhere              (kills variation AND mean level/tempo)
        #   "seqmean" : dt := this sequence's mean dt   (kills ONLY the variation; tempo survives)
        # A vs seqmean is the CLEAN test of irregular sampling; A vs ones also deletes the duration
        # cue (durations span 2.3-240 s here), so the two must be reported separately.
        assert dt_mode in ("real", "ones", "seqmean")
        self.dt_mode = dt_mode
        # OCCLUSION-AWARE VARIANT. use_mask=False reproduces the published model exactly (raw-sum
        # aggregation, no mask anywhere) so every existing checkpoint stays loadable. use_mask=True
        # is a DIFFERENT ARCHITECTURE -- degree-renormalised aggregation changes the feature scale
        # even on clean data (Kinect in-degree runs 1..4), so it needs its own checkpoints and its
        # own training run. It is not a drop-in, and pretending otherwise would silently invalidate
        # every number in Block 2/3/4.
        self.use_mask = use_mask
        # F1b: native-rate band-limited speed RMS per joint (Type-0 invariant), the high-frequency
        # content the 150-frame subsample destroys at the loader. Off by default; the Phase-0 pivot
        # turns it on together with kimore_cde_data.load_sample(compute_bandpower=True).
        self.use_bandpower = use_bandpower
        # + dt + exercise one-hot; + per-joint displacement & speed; + the liveness vector itself
        # ("this joint is MISSING" is different information from "this joint is at the origin",
        # and the head cannot infer the difference from a zeroed channel); + F1b band-power.
        in_dim = (self.proj.dim + 1 + n_exercises
                  + (2 * n_joints if use_speed else 0)
                  + (n_joints if use_mask else 0)
                  + (n_joints if use_bandpower else 0))
        self.gru = nn.GRU(in_dim, gru_hidden, gru_layers,
                          batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(2 * gru_hidden, head_hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, 1))

    def forward(self, t, x, ex_id=None, n_real=None, mask=None, hf=None, ablate=None):
        """t: (B,T) ACTUAL stamps; x: (B,T,J,3) root-relative; n_real: (B,) true lengths.

        mask: (B,T,J) joint liveness in {0,1}. Only consulted when use_mask=True; defaults to
        all-live. When supplied, the score is EXACTLY INDEPENDENT of what a dead joint reports:
        f(x, m) = f(x', m) for any x, x' differing only where m = 0. That is a theorem about a
        failed sensor, not a robustness slope, and certify_egru measures it to machine precision.

        hf: (B,T,J) F1b native-rate band-power, only consulted when use_bandpower=True.

        ablate: optional set of invariant-family names zeroed in the cut without changing the model
        (reviewer Q2 per-family ablation); passed through to InvariantProjection. Default None ==
        no ablation. It touches ONLY the invariant projection, not the GRU-level speed/dt channels.
        """
        B, T, J, _ = x.shape
        if self.use_mask:
            mask = torch.ones(B, T, J, dtype=x.dtype, device=x.device) if mask is None \
                else mask.to(x.dtype)
            mflat = mask.reshape(B * T, J)
        else:
            mask, mflat = None, None

        h = self.encoder(x.reshape(B * T, J, 3), mflat)
        inv = self.proj(h, x.reshape(B * T, J, 3), mflat, ablate=ablate).reshape(B, T, -1)

        dt = torch.zeros_like(t)
        dt[:, 1:] = t[:, 1:] - t[:, :-1]                   # the sensor's real inter-arrival gaps

        # --- dt ablation. Transform dt ONCE, here, so every downstream use (the input channel AND
        #     the speed division below) sees the same ablated gaps. This is the leakage fix: if the
        #     input channel were ablated but speed kept dividing by the real dt, the model could
        #     reconstruct the true irregular grid from d/speed. There is exactly one real-dt source
        #     (this tensor) and one displacement source (d, which never touches dt), so ablating
        #     here is complete.
        if self.dt_mode == "ones":
            dt = torch.ones_like(dt)
            dt[:, 0] = 0.0                                  # frame 0 has no predecessor, as always
        elif self.dt_mode == "seqmean":
            valid = torch.zeros_like(dt)
            if n_real is not None:
                ar = torch.arange(T, device=x.device).unsqueeze(0)
                valid = ((ar >= 1) & (ar < n_real.unsqueeze(1))).to(dt.dtype)
            else:
                valid[:, 1:] = 1.0
            mean_dt = (dt * valid).sum(1) / valid.sum(1).clamp(min=1.0)   # (B,) real mean gap
            dt = mean_dt.unsqueeze(1).expand(-1, T).clone()
            dt[:, 0] = 0.0

        # Per-joint SPEED. ||x_j(t_k) - x_j(t_k-1)|| is the norm of a difference of positions,
        # hence SO(3)-invariant by construction -- it costs the theorem nothing (certify_egru E2
        # re-checks). It is included because the equivariant encoder is a purely SPATIAL, static
        # map: it describes each frame's pose but never its motion. The hand-crafted control
        # (invariant_controls.InvariantGRU) is handed speeds explicitly and beats this model by
        # ~1.2 MAD on every fold; a recurrence trained on 45 subjects does not reliably learn to
        # difference its own inputs. Divide by dt to get a true velocity magnitude, which also
        # makes the feature correct under IRREGULAR sampling rather than merely per-frame.
        if self.use_speed:
            d = torch.zeros(B, T, J, dtype=x.dtype, device=x.device)
            d[:, 1:] = (x[:, 1:] - x[:, :-1]).norm(dim=-1)
            speed = d / dt.unsqueeze(-1).clamp(min=1e-4)
            ld, ls = torch.log1p(d), torch.log1p(speed)
            if mask is not None:
                # The LAST leak path. A joint frozen by a tracking failure reports d = 0, i.e.
                # speed = 0 -- which the GRU reads as "perfectly still", a CONFIDENT and WRONG
                # measurement, rather than as "unknown". Gate both channels and hand the head the
                # liveness vector so it can tell the two apart. (This channel is also the prime
                # suspect for why Block 4 hurt us at all: 'hold' mode annihilates exactly these
                # features, so the use_speed=False x node-failure cell must be run.)
                live = mask * (mask.roll(1, dims=1))       # need BOTH frames to difference
                live[:, 0] = 0.0
                ld, ls = ld * live, ls * live
            feats = [inv, dt.unsqueeze(-1), ld, ls]
        else:
            feats = [inv, dt.unsqueeze(-1)]
        if mask is not None:
            feats.append(mask)                             # liveness as an explicit input
        if self.use_bandpower:
            # F1b. The grid-free advantage the dt-at-the-recurrence ablation could not deliver: a
            # Type-0 channel carrying the >2 Hz energy the subsample discards. Read at NATIVE rate
            # in the loader; the recurrence just consumes it, no dt-reading required.
            if hf is None:
                hf = torch.zeros(B, T, self.n_joints, dtype=x.dtype, device=x.device)
            hf = torch.log1p(hf)
            if mask is not None:
                hf = hf * mask                             # a dead joint reports no band-power
            feats.append(hf)
        if self.n_ex > 0:
            oh = torch.nn.functional.one_hot(ex_id, self.n_ex).to(x.dtype)
            feats.append(oh.unsqueeze(1).expand(B, T, -1))
        seq, _ = self.gru(torch.cat(feats, dim=-1))

        if n_real is None:
            pooled = seq.mean(1)
        else:                                              # ignore the padded tail
            idx = torch.arange(T, device=x.device).unsqueeze(0)
            m = (idx < n_real.unsqueeze(1)).to(seq.dtype).unsqueeze(-1)
            pooled = (seq * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(pooled).squeeze(-1)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
