# SPDX-License-Identifier: Apache-2.0
"""Marching-cubes lookup tables ported from the reference implementation.

Tables are loaded from ``_mc_tables.npz`` (generated once from
``EfficientPBDCutting/ufrgs/marchingCubes.h``) and exposed as plain ``np.ndarray``\\ s
here. Upload to ``wp.array`` happens in :mod:`.marching_cubes`.

Cube corner ordering (matches ``springGrid.h:806-813``):

    corner 0: (0, 0, 0)        corner 4: (0, 1, 0)
    corner 1: (1, 0, 0)        corner 5: (1, 1, 0)
    corner 2: (1, 0, 1)        corner 6: (1, 1, 1)
    corner 3: (0, 0, 1)        corner 7: (0, 1, 1)

Direction indices (``orientation`` frame axis + sign) used by the paper's
per-particle vertex cache (``Vertice::getPosition`` at springGrid.h:714-727):

    dir 0: -X  dir 1: +X
    dir 2: -Y  dir 3: +Y
    dir 4: -Z  dir 5: +Z
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_TABLES_NPZ = Path(__file__).with_name("_mc_tables.npz")

_DATA = np.load(_TABLES_NPZ)

# [256, 16] int32: per 8-bit case index, a list of triangle edge indices
# terminated by -1. Up to 5 triangles (15 ints) per case.
CASE_TRIANGLES: np.ndarray = _DATA["triangles"].astype(np.int32)

# [256] int32: bitmask of which of the 12 edges are crossed. Unused by our
# kernels (we iterate the triangle list directly) but kept for completeness.
CASE_EDGE_MASK: np.ndarray = _DATA["edges"].astype(np.int32)

# [12, 2] int32: each edge's (corner_a, corner_b) pair.
EDGE_CORNERS: np.ndarray = _DATA["edge_corners"].astype(np.int32)

# [8, 3] int32: per cube-corner (c), the (dx, dy, dz) offset from the cube's
# origin voxel. Matches the switch in ``getVertice`` (springGrid.h:753-762).
CORNER_OFFSETS: np.ndarray = np.asarray(
    [
        (0, 0, 0),  # 0
        (1, 0, 0),  # 1
        (1, 0, 1),  # 2
        (0, 0, 1),  # 3
        (0, 1, 0),  # 4
        (1, 1, 0),  # 5
        (1, 1, 1),  # 6
        (0, 1, 1),  # 7
    ],
    dtype=np.int32,
)

# [12] int32: base direction for each edge when the "a" end is the active
# corner. Subtracting 1 gives the direction when "b" is active. Matches the
# switch in ``getVertice`` at springGrid.h:764-775.
#
#   edges 0, 2, 4, 6   (X edges) -> base dir 1 (+X), fallback 0 (-X)
#   edges 1, 3, 5, 7   (Z edges) -> base dir 5 (+Z), fallback 4 (-Z)
#   edges 8, 9, 10, 11 (Y edges) -> base dir 3 (+Y), fallback 2 (-Y)
EDGE_BASE_DIR: np.ndarray = np.asarray(
    [1, 5, 1, 5, 1, 5, 1, 5, 3, 3, 3, 3],
    dtype=np.int32,
)


def case_triangle_counts() -> np.ndarray:
    """Return ``[256]`` int32 with the triangle count per case."""
    return np.asarray(
        [((row != -1).sum()) // 3 for row in CASE_TRIANGLES],
        dtype=np.int32,
    )
