"""Forward kinematics for UI-PRMD Kinect data: bone offsets + Euler angles -> world joint positions.

WHY THIS EXISTS
---------------
UI-PRMD's Kinect `Positions/*.txt` are NOT world coordinates. They are per-frame
parent-relative bone offsets (joint 0 = absolute root translation). 19 of 66 coordinate
slots therefore have exactly zero temporal variance, and any model fed these files sees a
near-static skeleton dragged around by its root. See docs/worklog_2026-07-29.md §2.

The pose actually lives in `Angles/*.txt`. This module runs the forward kinematics that
turns (offsets, angles) into the world-coordinate 22-joint skeleton the rest of the
pipeline assumes it already had.

CONVENTION -- transcribed from the authors' own `Animation.m` (shipped in
`UI-PRMD Visualize.zip`), which is the only normative statement of it:

    eulers_2_rot_matrix([gx, by, az]) = rotz(az) @ roty(by) @ rotx(gx)

so angle column 0 rotates about X, column 1 about Y, column 2 about Z, composed Z@Y@X,
in DEGREES. A joint's angle orients its *outgoing* bones. Axes: +X screen-right, +Y up,
+Z out of screen.

TWO VARIANTS
------------
`reference` -- a literal transcription of Animation.m, including its quirk: the two arm
chains restart their rotation accumulation from identity (`rot_7 = E(ang_chest)`) instead
of inheriting the torso's accumulated `E(waist) @ E(spine)`. Torso and legs accumulate
correctly.

`corrected` -- arms inherit the full torso chain (`rot_7 = rot_4`). This is what standard
BVH forward kinematics does.

The authors flag their own visualisation as imperfect ("correct for the upper body joints,
but not entirely correct for all lower body joints; m03, m04, m05 look unnatural"), so
neither variant should be assumed exact. Use `reference` for anything that must match the
published dataset, and diff against `corrected` as a sensitivity check.
"""

from __future__ import annotations

import numpy as np

N_JOINTS = 22

# Joint order, from Animation.m's comment block (1-indexed there, 0-indexed here).
JOINT_NAMES = [
    "Waist", "Spine", "Chest", "Neck", "Head", "HeadTip",
    "LCollar", "LUpperArm", "LForearm", "LHand",
    "RCollar", "RUpperArm", "RForearm", "RHand",
    "LUpperLeg", "LLowerLeg", "LFoot", "LToes",
    "RUpperLeg", "RLowerLeg", "RFoot", "RToes",
]

# (child, parent, angle_joint, prefix_rot_of) transcribed line-by-line from Animation.m.
# `prefix_rot_of` is the joint whose accumulated rotation left-multiplies this one;
# None means the accumulation restarts at identity (the arm-chain quirk).
_CHAIN = [
    # child, parent, angle,  prefix
    (1,  0,  0,  None),   # rot_2  = E(ang1)
    (2,  1,  1,  1),      # rot_3  = rot_2 * E(ang2)
    (3,  2,  2,  2),      # rot_4  = rot_3 * E(ang3)
    (4,  3,  3,  3),      # rot_5  = rot_4 * E(ang4)
    (5,  4,  4,  4),      # rot_6  = rot_5 * E(ang5)
    (6,  2,  2,  None),   # rot_7  = E(ang3)          <-- restart (left arm)
    (7,  6,  6,  6),      # rot_8  = rot_7 * E(ang7)
    (8,  7,  7,  7),      # rot_9  = rot_8 * E(ang8)
    (9,  8,  8,  8),      # rot_10 = rot_9 * E(ang9)
    (10, 2,  2,  None),   # rot_11 = E(ang3)          <-- restart (right arm)
    (11, 10, 10, 10),     # rot_12 = rot_11 * E(ang11)
    (12, 11, 11, 11),     # rot_13 = rot_12 * E(ang12)
    (13, 12, 12, 12),     # rot_14 = rot_13 * E(ang13)
    (14, 0,  0,  None),   # rot_15 = E(ang1)          (== rot_2; waist is root, so correct)
    (15, 14, 14, 14),     # rot_16 = rot_15 * E(ang15)
    (16, 15, 15, 15),     # rot_17 = rot_16 * E(ang16)
    (17, 16, 16, 16),     # rot_18 = rot_17 * E(ang17)
    (18, 0,  0,  None),   # rot_19 = E(ang1)
    (19, 18, 18, 18),     # rot_20 = rot_19 * E(ang19)
    (20, 19, 19, 19),     # rot_21 = rot_20 * E(ang20)
    (21, 20, 20, 20),     # rot_22 = rot_21 * E(ang21)
]

# In the corrected variant the arm chains inherit the torso's accumulated rotation.
# rot_4 is the chest-accumulated frame E(waist)@E(spine)@E(chest), i.e. entry for child 3.
_CORRECTED_ARM_PREFIX = {6: 3, 10: 3}

# Bone edges for plotting / limb-length checks (from J in Animation.m).
EDGES = [(c, p) for c, p, _, _ in _CHAIN]


def _euler_to_R(ang_deg: np.ndarray) -> np.ndarray:
    """(..., 3) Euler angles in degrees -> (..., 3, 3) rotation, R = Rz @ Ry @ Rx."""
    gx, by, az = np.deg2rad(ang_deg[..., 0]), np.deg2rad(ang_deg[..., 1]), np.deg2rad(ang_deg[..., 2])
    cx, sx = np.cos(gx), np.sin(gx)
    cy, sy = np.cos(by), np.sin(by)
    cz, sz = np.cos(az), np.sin(az)
    z = np.zeros_like(cx)
    o = np.ones_like(cx)
    Rx = np.stack([o, z, z, z, cx, -sx, z, sx, cx], -1).reshape(*cx.shape, 3, 3)
    Ry = np.stack([cy, z, sy, z, o, z, -sy, z, cy], -1).reshape(*cy.shape, 3, 3)
    Rz = np.stack([cz, -sz, z, sz, cz, z, z, z, o], -1).reshape(*cz.shape, 3, 3)
    return Rz @ Ry @ Rx


def forward_kinematics(
    offsets: np.ndarray,
    angles: np.ndarray,
    variant: str = "reference",
    euler_order: str = "zyx",
) -> np.ndarray:
    """(T,22,3) parent-relative offsets + (T,22,3) Euler degrees -> (T,22,3) world positions.

    `variant`: "reference" (literal Animation.m) or "corrected" (arms inherit torso).
    `euler_order` is exposed only so the convention can be *tested* rather than assumed;
    "zyx" is the authors' convention and the default.
    """
    if offsets.shape != angles.shape:
        raise ValueError(f"offsets {offsets.shape} != angles {angles.shape}")
    if offsets.shape[1:] != (N_JOINTS, 3):
        raise ValueError(f"expected (T,{N_JOINTS},3), got {offsets.shape}")
    if variant not in ("reference", "corrected"):
        raise ValueError(f"unknown variant {variant!r}")

    T = offsets.shape[0]
    E = _euler_to_R(angles) if euler_order == "zyx" else _euler_alt(angles, euler_order)

    world = np.zeros((T, N_JOINTS, 3), dtype=np.float64)
    rot = np.zeros((T, N_JOINTS, 3, 3), dtype=np.float64)
    world[:, 0] = offsets[:, 0]  # waist is absolute
    eye = np.broadcast_to(np.eye(3), (T, 3, 3))

    for child, parent, ang_j, prefix in _CHAIN:
        if variant == "corrected" and child in _CORRECTED_ARM_PREFIX:
            prefix = _CORRECTED_ARM_PREFIX[child]
        base = eye if prefix is None else rot[:, prefix]
        R = base @ E[:, ang_j]
        rot[:, child] = R
        world[:, child] = np.einsum("tij,tj->ti", R, offsets[:, child]) + world[:, parent]

    return world


def _euler_alt(ang_deg: np.ndarray, order: str) -> np.ndarray:
    """Alternative Euler compositions, for convention testing only."""
    a = np.deg2rad(ang_deg)
    cs = [(np.cos(a[..., i]), np.sin(a[..., i])) for i in range(3)]
    z = np.zeros_like(cs[0][0])
    o = np.ones_like(cs[0][0])

    def mat(axis: str, i: int) -> np.ndarray:
        c, s = cs[i]
        if axis == "x":
            m = [o, z, z, z, c, -s, z, s, c]
        elif axis == "y":
            m = [c, z, s, z, o, z, -s, z, c]
        else:
            m = [c, -s, z, s, c, z, z, z, o]
        return np.stack(m, -1).reshape(*c.shape, 3, 3)

    # order string is the composition left-to-right; angle column follows the axis letter
    idx = {"x": 0, "y": 1, "z": 2}
    R = mat(order[0], idx[order[0]])
    for ax in order[1:]:
        R = R @ mat(ax, idx[ax])
    return R


def load_pair(pos_path, ang_path) -> tuple[np.ndarray, np.ndarray]:
    """Read a matched positions/angles pair as (T,22,3) each."""
    p = np.loadtxt(pos_path, delimiter=",", dtype=np.float64, ndmin=2)
    a = np.loadtxt(ang_path, delimiter=",", dtype=np.float64, ndmin=2)
    n = min(len(p), len(a))
    return p[:n].reshape(n, N_JOINTS, 3), a[:n].reshape(n, N_JOINTS, 3)
