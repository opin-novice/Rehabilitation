"""Shared geometric + protocol constants. Single source of truth."""
from __future__ import annotations

SEQ_LEN: int = 100
NUM_JOINTS: int = 25
NUM_CHANNELS: int = 3

# Each raw KIMORE joint occupies 4 CSV columns (x, y, z, orientation); we keep x,y,z.
JOINT_STARTS: list[int] = list(range(0, 100, 4))

# Kinect v2 skeleton adjacency (25 joints, 0-indexed).
SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1), (1, 20), (20, 2), (2, 3),
    (20, 4), (4, 5), (5, 6), (6, 7), (6, 22), (7, 21),
    (20, 8), (8, 9), (9, 10), (10, 11), (10, 24), (11, 23),
    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),
]
