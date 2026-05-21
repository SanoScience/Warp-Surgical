# SPDX-License-Identifier: Apache-2.0
"""Per-particle oriented frames for deformable marching cubes.

Port of ``SpringGrid::updateOrientation`` (springGrid.h:612-660). Each particle
carries a 3x3 frame whose rows (``orie.x, orie.y, orie.z``) act as local basis
vectors along the marching-cubes cell edges. The frame is updated every step
from the six face-adjacent neighbours: the axis corresponding to each active
neighbour is pushed along the normalised inter-particle direction (weight
``fA``), and the neighbour's previous frame is blended in (weight ``fB``). A
final per-row normalisation keeps rows unit-length; orthogonality is not
enforced - the paper relies on this looseness for smooth surface deformation.

Missing (i.e. cut) neighbours simply drop out of the sum, which matches the
C++ reference. The cross-product fallback from paper Figure 4d is NOT
implemented in that reference and is therefore omitted here for fidelity.

Indexing note: ``particle_neighbors[i, d]`` follows the convention in
``grid.py``: 0=-X, 1=+X, 2=-Y, 3=+Y, 4=-Z, 5=+Z.
"""

from __future__ import annotations

import warp as wp

from newton._src.geometry.flags import ParticleFlags

IDENTITY_MAT33 = wp.constant(wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))


@wp.func
def _safe_normalize(v: wp.vec3) -> wp.vec3:
    n = wp.length(v)
    if n > 1.0e-6:
        return v / n
    return v


@wp.func
def _update_axis(orie_axis: wp.vec3, sign: float, diff: wp.vec3, fA: float) -> wp.vec3:
    return orie_axis + _safe_normalize(diff) * (sign * fA)


@wp.kernel
def initialize_orientations_kernel(orientation_out: wp.array(dtype=wp.mat33)):
    """Set every particle's frame to the identity basis."""
    i = wp.tid()
    orientation_out[i] = IDENTITY_MAT33


@wp.kernel
def update_orientation_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    particle_neighbors: wp.array2d(dtype=wp.int32),
    orientation_in: wp.array(dtype=wp.mat33),
    fA: float,
    fB: float,
    orientation_out: wp.array(dtype=wp.mat33),
):
    """One Jacobi sweep of the paper's oriented-particle update.

    Reads ``orientation_in`` (previous frame) and writes ``orientation_out`` so
    the caller can swap buffers between frames and avoid a kernel-internal race.
    Inactive particles (``ParticleFlags.ACTIVE`` cleared) are skipped and keep
    their previous frame unchanged.
    """
    i = wp.tid()
    if (particle_flags[i] & wp.int32(ParticleFlags.ACTIVE)) == 0:
        orientation_out[i] = orientation_in[i]
        return

    prev = orientation_in[i]
    orie_x = wp.vec3(prev[0, 0], prev[0, 1], prev[0, 2])
    orie_y = wp.vec3(prev[1, 0], prev[1, 1], prev[1, 2])
    orie_z = wp.vec3(prev[2, 0], prev[2, 1], prev[2, 2])
    pos_i = particle_q[i]

    # Unrolled 6-neighbour loop. Each direction contributes its own axis push
    # and a smoothing term from the neighbour's previous orientation.
    n = particle_neighbors[i, 0]  # -X
    if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:
        orie_x = _update_axis(orie_x, -1.0, particle_q[n] - pos_i, fA)
        nb = orientation_in[n]
        orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB
        orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB
        orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB

    n = particle_neighbors[i, 1]  # +X
    if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:
        orie_x = _update_axis(orie_x, 1.0, particle_q[n] - pos_i, fA)
        nb = orientation_in[n]
        orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB
        orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB
        orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB

    n = particle_neighbors[i, 2]  # -Y
    if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:
        orie_y = _update_axis(orie_y, -1.0, particle_q[n] - pos_i, fA)
        nb = orientation_in[n]
        orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB
        orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB
        orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB

    n = particle_neighbors[i, 3]  # +Y
    if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:
        orie_y = _update_axis(orie_y, 1.0, particle_q[n] - pos_i, fA)
        nb = orientation_in[n]
        orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB
        orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB
        orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB

    n = particle_neighbors[i, 4]  # -Z
    if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:
        orie_z = _update_axis(orie_z, -1.0, particle_q[n] - pos_i, fA)
        nb = orientation_in[n]
        orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB
        orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB
        orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB

    n = particle_neighbors[i, 5]  # +Z
    if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:
        orie_z = _update_axis(orie_z, 1.0, particle_q[n] - pos_i, fA)
        nb = orientation_in[n]
        orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB
        orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB
        orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB

    orie_x = _safe_normalize(orie_x)
    orie_y = _safe_normalize(orie_y)
    orie_z = _safe_normalize(orie_z)

    orientation_out[i] = wp.mat33(
        orie_x[0], orie_x[1], orie_x[2],
        orie_y[0], orie_y[1], orie_y[2],
        orie_z[0], orie_z[1], orie_z[2],
    )
