# SPDX-License-Identifier: Apache-2.0
"""Grab/pull-point distance constraints."""

from __future__ import annotations

from collections.abc import Sequence

import warp as wp
from newton._src.geometry.flags import ParticleFlags

_ACTIVE_BIT = wp.constant(wp.int32(int(ParticleFlags.ACTIVE)))


@wp.kernel(enable_backward=False)
def solve_grab_distance_constraints_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    grab_indices: wp.array(dtype=wp.int32),
    grab_offsets: wp.array(dtype=wp.vec3),
    pull_target: wp.vec3,
    stiffness: float,
):
    tid = wp.tid()
    particle_idx = grab_indices[tid]
    if particle_idx < 0:
        return
    if (particle_flags[particle_idx] & _ACTIVE_BIT) == 0:
        return
    if particle_inv_mass[particle_idx] <= 0.0:
        return

    alpha = stiffness
    if alpha < 0.0:
        alpha = 0.0
    if alpha > 1.0:
        alpha = 1.0
    if alpha <= 0.0:
        return

    q = particle_q[particle_idx]
    rest = grab_offsets[tid]
    rest_length = wp.length(rest)
    delta = q - pull_target
    dist = wp.length(delta)

    direction = wp.vec3(1.0, 0.0, 0.0)
    if dist > 1.0e-8:
        direction = delta / dist
    elif rest_length > 1.0e-8:
        direction = rest / rest_length

    goal = pull_target + direction * rest_length
    corrected = q + (goal - q) * alpha
    particle_q[particle_idx] = corrected
    particle_qd[particle_idx] = wp.vec3(0.0, 0.0, 0.0)


def _target_vec3(target: Sequence[float]) -> wp.vec3:
    return wp.vec3(float(target[0]), float(target[1]), float(target[2]))


def project_grab_distance_constraints(
    particle_q: wp.array,
    particle_qd: wp.array,
    particle_inv_mass: wp.array,
    particle_flags: wp.array,
    grab_indices: wp.array,
    grab_offsets: wp.array,
    grab_count: int,
    pull_target: Sequence[float],
    stiffness: float,
    device=None,
) -> None:
    """Project pull-point distance constraints for grabbed particles."""
    count = int(grab_count)
    if count <= 0:
        return
    wp.launch(
        kernel=solve_grab_distance_constraints_kernel,
        dim=count,
        inputs=[
            particle_q,
            particle_qd,
            particle_inv_mass,
            particle_flags,
            grab_indices,
            grab_offsets,
            _target_vec3(pull_target),
            float(stiffness),
        ],
        device=particle_q.device if device is None else device,
    )


__all__ = [
    "project_grab_distance_constraints",
    "solve_grab_distance_constraints_kernel",
]
