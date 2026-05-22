# SPDX-License-Identifier: Apache-2.0
"""Ground-plane contact kernel for the local hex-grid solver."""

from __future__ import annotations

import warp as wp

from newton._src.geometry.flags import ParticleFlags


@wp.kernel(enable_backward=False)
def solve_particle_ground_plane_contacts(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    ground_height: float,
    particle_mu: float,
    dt: float,
    relaxation: float,
    particle_deltas: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:
        return
    if particle_inv_mass[tid] <= 0.0:
        return

    n = wp.vec3(0.0, 0.0, 1.0)
    signed_distance = particle_q[tid][2] - particle_radius[tid] - ground_height
    if signed_distance >= 0.0:
        return

    correction = n * (-signed_distance)

    if particle_mu > 0.0:
        tangential_v = particle_qd[tid] - n * wp.dot(n, particle_qd[tid])
        tangential_speed = wp.length(tangential_v)
        if tangential_speed > 1.0e-8:
            max_friction_step = particle_mu * (-signed_distance)
            velocity_friction_step = tangential_speed * dt
            friction_step = wp.min(max_friction_step, velocity_friction_step)
            correction -= tangential_v * (friction_step / tangential_speed)

    wp.atomic_add(particle_deltas, tid, correction * relaxation)


__all__ = ["solve_particle_ground_plane_contacts"]
