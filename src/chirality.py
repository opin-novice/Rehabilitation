#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
chirality.py
============
Parity-odd (PSEUDO-SCALAR) features: the irrep the invariant cut was missing.

The hole
--------
Every quantity our invariant cut emitted was parity-EVEN:

    scalars s            0e     even
    ||v||   (1o norms)          even   (odd x odd)
    cos(v_a, v_b)               even   (odd x odd)
    bone lengths                even

A reflection M (det M = -1) therefore left ALL of them fixed, so the score obeyed
s(Mx) = s(x). The model was invariant under the FULL O(3), not merely SO(3). We claimed
SO(3); we silently bought O(3); and O(3) is strictly stronger than we wanted. Nothing in
Block 3 ever caught it, because Block 3 rotates about the vertical axis -- proper rotations,
det = +1 -- and O(3)-invariance implies SO(3)-invariance. The extra symmetry was free to us
and invisible to the experiment. That is exactly the kind of assumption that should be
measured rather than inherited.

What the hole is NOT
-------------------
It is NOT "the model cannot tell the left arm from the right arm." That claim does not
follow and it is false, which certify_egru E7 demonstrates by measurement. Joint INDICES are
preserved everywhere in this pipeline: `joint_emb` is keyed by joint id, bone lengths are
emitted per bone in a fixed order, and the speed channels are 2*n_joints in index order and
are never pooled. A left-arm raise and a right-arm raise are not related by a reflection --
they are related by a reflection COMPOSED WITH the left/right relabelling LR_SWAP, and the
model is not invariant to the relabelling. It separates them.

What the hole IS
----------------
Two things, one harmless and one real.

  (harmless)  The model scores a body identically to its mirror image WITH LABELS UNCHANGED
              -- a body whose left arm hangs off its right side. Anatomically impossible;
              off-manifold; it costs nothing.

  (REAL)      TRAJECTORY HANDEDNESS. A clockwise and a counter-clockwise limb circle have
              identical pairwise distances at every instant, identical bone lengths, and
              identical speeds. No parity-even feature can separate them. Under O(3)
              invariance the two are provably the same input. Any pathology whose signature
              is the SENSE of a rotation -- compensatory trunk torsion, a circumducting gait,
              the direction of a scapular hitch -- lives entirely in the subspace we deleted.

The fix
-------
Admit pseudo-scalars: parity-ODD scalars, `0o` in e3nn's notation. Under g they pick up
det(rho(g)):

    chi(g.x) = det(g) * chi(x)      =>   rotation (det=+1): INVARIANT
                                         reflection (det=-1): SIGN FLIP

so the score stays exactly SO(3)-invariant -- Block 3's 0.000 degradation is untouched, and
that is a theorem, not a hope -- while O(3)-invariance is deliberately broken. We narrow the
group from O(3) to SO(3), which is the group we meant all along.

The canonical pseudo-scalar built from three type-1 vectors is the triple product

    chi_abc = det[v_a, v_b, v_c] = (v_a x v_b) . v_c

We normalise to UNIT vectors before taking it, so chi in [-1, 1]. This is the same discipline
that made the cut use cosines rather than raw dot products: a raw triple product is CUBIC in
feature scale, and a raw quadratic dot product already blew a previous model's predictions
into the hundreds. Bounded inputs, identical information.

Two families are supplied:

  1. LEARNED   (equivariant_gru.InvariantProjection): triple products over the encoder's own
               1o channels -- C(n_vec, 3) of them per joint.
  2. ANATOMIC  (here): signed volumes over fixed, physically meaningful joint triads and
               kinematic chains. These do not depend on the encoder learning to build them,
               and they give the hand-crafted control (invariant_controls.InvariantGRU) the
               SAME new information, so the co-headline comparison stays matched. Changing
               only one arm's feature set would have made the two incomparable.
"""

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Kinect v2 25-joint layout (confirmed against invariant_controls.KEY_PAIRS, whose
# "wrists/head/elbows/ankles" comment pins 6/10 = wrists, 3 = head, 5/9 = elbows, 14/18 = ankles)
#
#  0 SpineBase      5 ElbowLeft    10 WristRight   15 FootLeft     20 SpineShoulder
#  1 SpineMid       6 WristLeft    11 HandRight    16 HipRight     21 HandTipLeft
#  2 Neck           7 HandLeft     12 HipLeft      17 KneeRight    22 ThumbLeft
#  3 Head           8 ShoulderRight 13 KneeLeft    18 AnkleRight   23 HandTipRight
#  4 ShoulderLeft   9 ElbowRight   14 AnkleLeft    19 FootRight    24 ThumbRight
# -----------------------------------------------------------------------------
N_JOINTS = 25

#: left/right joint index pairs. Applying this permutation is a RELABELLING, not a reflection.
LR_PAIRS = [(4, 8), (5, 9), (6, 10), (7, 11),          # arms
            (12, 16), (13, 17), (14, 18), (15, 19),    # legs
            (21, 23), (22, 24)]                        # hand tips / thumbs

#: index permutation that swaps left and right limbs, leaving the midline (0,1,2,3,20) fixed.
LR_SWAP = np.arange(N_JOINTS)
for _a, _b in LR_PAIRS:
    LR_SWAP[_a], LR_SWAP[_b] = _b, _a

#: sagittal-plane reflection. Kinect: +x is horizontal (left-right), +y up, +z depth.
#  NOTE any improper transform certifies equally well: two reflections differ by a ROTATION,
#  and the model is SO(3)-invariant, so s(M1 x) = s(M2 x) for every improper M1, M2. One suffices.
MIRROR = np.diag([-1.0, 1.0, 1.0])

#: Each entry is a frame of three vectors, given as (head, tail) joint-index pairs meaning
#  x[head] - x[tail]. det of the three is a pseudo-scalar: SO(3)-invariant, reflection-odd.
CHIRAL_FRAMES = [
    # --- kinematic-chain torsions: the signed volume of three consecutive bones. This is the
    #     discrete analogue of a dihedral angle's sign -- the handedness of a limb's posture.
    [(4, 20), (5, 4), (6, 5)],       # L arm : spineShoulder -> shoulder -> elbow -> wrist
    [(8, 20), (9, 8), (10, 9)],      # R arm
    [(5, 4), (6, 5), (7, 6)],        # L forearm -> hand
    [(9, 8), (10, 9), (11, 10)],     # R forearm -> hand
    [(12, 0), (13, 12), (14, 13)],   # L leg : spineBase -> hip -> knee -> ankle
    [(16, 0), (17, 16), (18, 17)],   # R leg
    [(1, 0), (20, 1), (3, 20)],      # spine -> neck -> head
    # --- global body triads: the handedness of the body frame itself.
    [(4, 0), (8, 0), (3, 0)],        # L shoulder, R shoulder, head   (about spineBase)
    [(12, 0), (16, 0), (20, 0)],     # L hip, R hip, spineShoulder
    [(7, 0), (11, 0), (3, 0)],       # L hand, R hand, head  <- where a chiral GESTURE shows up
    [(6, 20), (10, 20), (0, 20)],    # L wrist, R wrist, spineBase (about spineShoulder)
]
N_CHIRAL = len(CHIRAL_FRAMES)

_HEAD = np.array([[p[0] for p in f] for f in CHIRAL_FRAMES])   # (N_CHIRAL, 3)
_TAIL = np.array([[p[1] for p in f] for f in CHIRAL_FRAMES])   # (N_CHIRAL, 3)

EPS = 1e-8


def _det3(u, v, w):
    """Signed volume det[u, v, w] = (u x v) . w, batched over leading dims."""
    if isinstance(u, torch.Tensor):
        return (torch.cross(u, v, dim=-1) * w).sum(-1)
    return np.einsum("...c,...c->...", np.cross(u, v), w)


def anatomic_pseudoscalars(x):
    """x: (..., 25, 3) -> (..., N_CHIRAL) normalised signed volumes in [-1, 1].

    Rotation-INVARIANT, reflection-ODD. Works for numpy or torch, any leading batch shape.

    Normalisation: each of the three vectors is scaled to unit length BEFORE the determinant.
    The raw volume is cubic in body scale, so it would hand the network a size cue dressed up
    as a chirality cue -- and, worse, one whose dynamic range spans orders of magnitude across
    subjects. The unit-normalised determinant is exactly the chirality and nothing else.
    """
    is_t = isinstance(x, torch.Tensor)
    if is_t:
        hd = torch.as_tensor(_HEAD, device=x.device)
        tl = torch.as_tensor(_TAIL, device=x.device)
        v = x[..., hd, :] - x[..., tl, :]                       # (..., N_CHIRAL, 3, 3)
        v = v / (v.norm(dim=-1, keepdim=True) + EPS)
        return _det3(v[..., 0, :], v[..., 1, :], v[..., 2, :])  # (..., N_CHIRAL)
    v = x[..., _HEAD, :] - x[..., _TAIL, :]
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + EPS)
    return _det3(v[..., 0, :], v[..., 1, :], v[..., 2, :])


def frame_liveness(mask):
    """mask: (..., 25) joint liveness in {0,1} -> (..., N_CHIRAL) frame liveness in {0,1}.

    A signed volume is a determinant over three DIFFERENCE vectors spanning six joint indices
    (_HEAD and _TAIL). If ANY of those six joints is a failed tracking node, the volume is not
    'degraded', it is meaningless -- and being a determinant, a garbage vertex can flip its SIGN,
    which is the one bit the whole parity construction exists to carry. So a chiral frame is live
    iff all six of its joints are live, and a dead frame contributes exactly 0.

    This is what keeps the dead-node invariance theorem (see equivariant_gru) TRUE when
    use_chiral and use_mask are on together: without it, x_j of a failed joint would leak into the
    output through the pseudo-scalars, and f(x, m) = f(x', m) would fail.
    """
    if isinstance(mask, torch.Tensor):
        hd = torch.as_tensor(_HEAD, device=mask.device, dtype=torch.long)   # (N_CHIRAL, 3)
        tl = torch.as_tensor(_TAIL, device=mask.device, dtype=torch.long)
        live = mask[..., hd].prod(-1) * mask[..., tl].prod(-1)              # (..., N_CHIRAL)
        return live
    return mask[..., _HEAD].prod(-1) * mask[..., _TAIL].prod(-1)


def mirror(x):
    """Reflect through the sagittal plane. Labels are NOT permuted -- this is a pure parity flip."""
    if isinstance(x, torch.Tensor):
        M = torch.as_tensor(MIRROR, dtype=x.dtype, device=x.device)
        return x @ M.T
    return x @ MIRROR.T


def swap_lr(x):
    """Relabel left <-> right limbs. This is a PERMUTATION of joint indices, not a reflection --
    it is the transformation that actually distinguishes 'raised the left arm' from 'raised the
    right arm', and the model is NOT invariant to it (certify_egru E7 measures exactly this)."""
    return x[..., LR_SWAP, :]


def triple_index(n_vec):
    """All C(n_vec, 3) unordered channel triples, as a (3, n_triples) long tensor.

    These index the LEARNED pseudo-scalars: det over three of the encoder's 1o channels. The
    triple products span the full parity-odd invariant content of n_vec type-1 vectors, so this
    is the complete completion of the cut, not a heuristic sample of it.
    """
    idx = [(a, b, c)
           for a in range(n_vec)
           for b in range(a + 1, n_vec)
           for c in range(b + 1, n_vec)]
    return torch.tensor(idx, dtype=torch.long).T          # (3, n_triples)
