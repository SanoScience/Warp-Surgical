# SPDX-License-Identifier: Apache-2.0
"""Warp kernels for corner-grid shape matching."""

from __future__ import annotations

import warp as wp
from newton._src.geometry.flags import ParticleFlags

UNIFORM_CLUSTER_SIZE = 8
UNIFORM_CLUSTER_SIZE_27 = 27
UNIFORM_CLUSTER_SIZE_125 = 125
MAX_ROTATION_ITERS = 16


@wp.func
def _quat_dot(a: wp.quat, b: wp.quat) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


@wp.func
def _extract_rotation(a: wp.mat33, q_init: wp.quat, max_iters: int) -> wp.quat:
    # Dominant-eigenvector iteration on Horn's 4x4 matrix.
    q4 = wp.vec4(q_init[3], q_init[0], q_init[1], q_init[2])
    if wp.dot(q4, q4) < 1.0e-12:
        q4 = wp.vec4(1.0, 0.0, 0.0, 0.0)
    else:
        q4 = q4 / wp.sqrt(wp.dot(q4, q4))

    iters = max_iters
    if iters < 0:
        iters = 0
    if iters > wp.static(MAX_ROTATION_ITERS):
        iters = wp.static(MAX_ROTATION_ITERS)

    a00 = a[0, 0]
    a01 = a[0, 1]
    a02 = a[0, 2]
    a10 = a[1, 0]
    a11 = a[1, 1]
    a12 = a[1, 2]
    a20 = a[2, 0]
    a21 = a[2, 1]
    a22 = a[2, 2]

    k00 = a00 + a11 + a22
    k01 = a12 - a21
    k02 = a20 - a02
    k03 = a01 - a10
    k11 = a00 - a11 - a22
    k12 = a01 + a10
    k13 = a02 + a20
    k22 = -a00 + a11 - a22
    k23 = a12 + a21
    k33 = -a00 - a11 + a22

    for it in range(wp.static(MAX_ROTATION_ITERS)):
        if it >= iters:
            break

        next_q4 = wp.vec4(
            k00 * q4[0] + k01 * q4[1] + k02 * q4[2] + k03 * q4[3],
            k01 * q4[0] + k11 * q4[1] + k12 * q4[2] + k13 * q4[3],
            k02 * q4[0] + k12 * q4[1] + k22 * q4[2] + k23 * q4[3],
            k03 * q4[0] + k13 * q4[1] + k23 * q4[2] + k33 * q4[3],
        )
        next_norm_sq = wp.dot(next_q4, next_q4)
        if next_norm_sq < 1.0e-20:
            break
        q4 = next_q4 / wp.sqrt(next_norm_sq)

    return wp.quat(q4[1], q4[2], q4[3], q4[0])


@wp.func
def _support_scale_from_alpha(inv_weight: float, support_alpha: float) -> float:
    if inv_weight <= 0.0:
        return 0.0
    if support_alpha <= 0.0:
        return 1.0
    if support_alpha >= 1.0:
        return inv_weight
    if support_alpha == 0.5:
        return wp.sqrt(inv_weight)
    return wp.pow(inv_weight, support_alpha)


@wp.func
def _uniform8_slot_sign_x(slot: int) -> float:
    if slot == 1 or slot == 2 or slot == 5 or slot == 6:
        return 1.0
    return -1.0


@wp.func
def _uniform8_slot_sign_y(slot: int) -> float:
    if slot == 2 or slot == 3 or slot == 6 or slot == 7:
        return 1.0
    return -1.0


@wp.func
def _uniform8_slot_sign_z(slot: int) -> float:
    if slot == 4 or slot == 5 or slot == 6 or slot == 7:
        return 1.0
    return -1.0


@wp.kernel(enable_backward=False)
def finalize_position_update_from_q(
    particle_q_init: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    dt: float,
    v_max: float,
):
    particle_idx = wp.tid()
    if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:
        return

    x0 = particle_q_init[particle_idx]
    x_new = particle_q[particle_idx]
    v_new = (x_new - x0) / dt
    v_new_mag = wp.length(v_new)
    if v_new_mag > v_max:
        v_new *= v_max / v_new_mag
        x_new = x0 + v_new * dt
        particle_q[particle_idx] = x_new

    particle_qd[particle_idx] = v_new


@wp.kernel(enable_backward=False)
def solve_volume_constraints_uniform8(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    indices_by_slot: wp.array(dtype=wp.int32),
    rest_local_positions_by_slot: wp.array(dtype=wp.vec3),
    rest_local_template: wp.array(dtype=wp.vec3),
    coefficients: wp.array(dtype=float),
    cluster_active: wp.array(dtype=wp.int32),
    particle_cluster_inv_weights: wp.array(dtype=float),
    cluster_count: int,
    use_rest_local_template: int,
    stiffness: float,
    particle_deltas: wp.array(dtype=wp.vec3),
):
    cluster_idx = wp.tid()

    if cluster_active[cluster_idx] == 0 or stiffness <= 0.0:
        return

    coeff = coefficients[cluster_idx]
    if coeff <= 0.0:
        return

    i0 = indices_by_slot[0 * cluster_count + cluster_idx]
    i1 = indices_by_slot[1 * cluster_count + cluster_idx]
    i2 = indices_by_slot[2 * cluster_count + cluster_idx]
    i3 = indices_by_slot[3 * cluster_count + cluster_idx]
    i4 = indices_by_slot[4 * cluster_count + cluster_idx]
    i5 = indices_by_slot[5 * cluster_count + cluster_idx]
    i6 = indices_by_slot[6 * cluster_count + cluster_idx]
    i7 = indices_by_slot[7 * cluster_count + cluster_idx]

    if (
        (particle_flags[i0] & ParticleFlags.ACTIVE) == 0
        or (particle_flags[i1] & ParticleFlags.ACTIVE) == 0
        or (particle_flags[i2] & ParticleFlags.ACTIVE) == 0
        or (particle_flags[i3] & ParticleFlags.ACTIVE) == 0
        or (particle_flags[i4] & ParticleFlags.ACTIVE) == 0
        or (particle_flags[i5] & ParticleFlags.ACTIVE) == 0
        or (particle_flags[i6] & ParticleFlags.ACTIVE) == 0
        or (particle_flags[i7] & ParticleFlags.ACTIVE) == 0
    ):
        return

    p0 = particle_q[i0]
    p1 = particle_q[i1]
    p2 = particle_q[i2]
    p3 = particle_q[i3]
    p4 = particle_q[i4]
    p5 = particle_q[i5]
    p6 = particle_q[i6]
    p7 = particle_q[i7]

    ax = ((p1 - p0) + (p2 - p3) + (p5 - p4) + (p6 - p7)) * 0.25
    ay = ((p3 - p0) + (p2 - p1) + (p7 - p4) + (p6 - p5)) * 0.25
    az = ((p4 - p0) + (p5 - p1) + (p6 - p2) + (p7 - p3)) * 0.25
    volume = wp.dot(ax, wp.cross(ay, az))

    q0 = rest_local_template[0]
    q1 = rest_local_template[1]
    q2 = rest_local_template[2]
    q3 = rest_local_template[3]
    q4 = rest_local_template[4]
    q5 = rest_local_template[5]
    q6 = rest_local_template[6]
    q7 = rest_local_template[7]
    if use_rest_local_template == 0:
        q0 = rest_local_positions_by_slot[0 * cluster_count + cluster_idx]
        q1 = rest_local_positions_by_slot[1 * cluster_count + cluster_idx]
        q2 = rest_local_positions_by_slot[2 * cluster_count + cluster_idx]
        q3 = rest_local_positions_by_slot[3 * cluster_count + cluster_idx]
        q4 = rest_local_positions_by_slot[4 * cluster_count + cluster_idx]
        q5 = rest_local_positions_by_slot[5 * cluster_count + cluster_idx]
        q6 = rest_local_positions_by_slot[6 * cluster_count + cluster_idx]
        q7 = rest_local_positions_by_slot[7 * cluster_count + cluster_idx]

    rest_ax = ((q1 - q0) + (q2 - q3) + (q5 - q4) + (q6 - q7)) * 0.25
    rest_ay = ((q3 - q0) + (q2 - q1) + (q7 - q4) + (q6 - q5)) * 0.25
    rest_az = ((q4 - q0) + (q5 - q1) + (q6 - q2) + (q7 - q3)) * 0.25
    rest_volume = wp.dot(rest_ax, wp.cross(rest_ay, rest_az))

    if wp.abs(rest_volume) <= 1.0e-12:
        return

    gx = wp.cross(ay, az)
    gy = wp.cross(az, ax)
    gz = wp.cross(ax, ay)

    denom = float(0.0)
    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        inv_weight = particle_inv_mass[particle_idx] * particle_cluster_inv_weights[particle_idx]
        if inv_weight <= 0.0:
            continue
        grad = (
            gx * _uniform8_slot_sign_x(local_idx)
            + gy * _uniform8_slot_sign_y(local_idx)
            + gz * _uniform8_slot_sign_z(local_idx)
        ) * 0.25
        denom += inv_weight * wp.dot(grad, grad)

    if denom <= 1.0e-20:
        return

    lagrange = -(volume - rest_volume) * coeff * stiffness / denom
    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        inv_weight = particle_inv_mass[particle_idx] * particle_cluster_inv_weights[particle_idx]
        if inv_weight <= 0.0:
            continue
        grad = (
            gx * _uniform8_slot_sign_x(local_idx)
            + gy * _uniform8_slot_sign_y(local_idx)
            + gz * _uniform8_slot_sign_z(local_idx)
        ) * 0.25
        wp.atomic_add(particle_deltas, particle_idx, grad * (lagrange * inv_weight))


@wp.kernel(enable_backward=False)
def compute_shape_matching_cluster_poses_uniform8(
    particle_q: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    indices_by_slot: wp.array(dtype=wp.int32),
    rest_local_positions_by_slot: wp.array(dtype=wp.vec3),
    rest_local_template: wp.array(dtype=wp.vec3),
    coefficients: wp.array(dtype=float),
    cluster_active: wp.array(dtype=wp.int32),
    cluster_count: int,
    use_rest_local_template: int,
    stiffness: float,
    rotation_iterations: int,
    cluster_rotations: wp.array(dtype=wp.quat),
    cluster_translations: wp.array(dtype=wp.vec3),
):
    cluster_idx = wp.tid()

    if cluster_active[cluster_idx] == 0 or stiffness <= 0.0:
        return

    coeff = coefficients[cluster_idx]
    if coeff <= 0.0:
        return

    center = wp.vec3(0.0, 0.0, 0.0)
    rest_sum = wp.vec3(0.0, 0.0, 0.0)
    covariance = wp.mat33(0.0)
    member_count = int(0)

    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:
            continue
        x = particle_q[particle_idx]
        q_rel = rest_local_template[local_idx]
        if use_rest_local_template == 0:
            q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]
        center += x
        rest_sum += q_rel
        covariance += wp.outer(q_rel, x)
        member_count += 1

    if member_count == 0:
        return

    center /= float(member_count)
    # Horn's quaternion extractor here expects the rest-to-current covariance.
    covariance -= wp.outer(rest_sum, center)

    prev_rotation = cluster_rotations[cluster_idx]
    rotation = _extract_rotation(covariance, prev_rotation, rotation_iterations)
    if _quat_dot(rotation, prev_rotation) < 0.0:
        rotation = wp.quat(-rotation[0], -rotation[1], -rotation[2], -rotation[3])

    cluster_rotations[cluster_idx] = rotation
    cluster_translations[cluster_idx] = center


@wp.kernel(enable_backward=False)
def apply_shape_matching_particle_gather_uniform8(
    particle_q_init: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_deltas: wp.array(dtype=wp.vec3),
    particle_cluster_offsets: wp.array(dtype=wp.int32),
    particle_cluster_indices: wp.array(dtype=wp.int32),
    particle_cluster_member_offsets: wp.array(dtype=wp.int32),
    rest_local_positions_by_slot: wp.array(dtype=wp.vec3),
    rest_local_template: wp.array(dtype=wp.vec3),
    coefficients: wp.array(dtype=float),
    cluster_active: wp.array(dtype=wp.int32),
    particle_cluster_inv_weights: wp.array(dtype=float),
    cluster_count: int,
    use_rest_local_template: int,
    stiffness: float,
    cluster_rotations: wp.array(dtype=wp.quat),
    cluster_translations: wp.array(dtype=wp.vec3),
    include_base_delta: int,
    dt: float,
    v_max: float,
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_qd_out: wp.array(dtype=wp.vec3),
):
    particle_idx = wp.tid()

    if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:
        return

    total_delta = wp.vec3(0.0, 0.0, 0.0)
    if include_base_delta != 0:
        total_delta += particle_deltas[particle_idx]

    if particle_inv_mass[particle_idx] > 0.0 and stiffness > 0.0:
        inv_weight = particle_cluster_inv_weights[particle_idx]
        if inv_weight > 0.0:
            cursor = particle_cluster_offsets[particle_idx]
            end = particle_cluster_offsets[particle_idx + 1]
            while cursor < end:
                cluster_idx = particle_cluster_indices[cursor]
                if cluster_active[cluster_idx] != 0:
                    coeff = coefficients[cluster_idx]
                    if coeff > 0.0:
                        member_offset = particle_cluster_member_offsets[cursor]
                        local_idx = member_offset - cluster_idx * wp.static(UNIFORM_CLUSTER_SIZE)
                        q_rel = rest_local_template[local_idx]
                        if use_rest_local_template == 0:
                            q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]
                        goal = cluster_translations[cluster_idx] + wp.quat_rotate(
                            cluster_rotations[cluster_idx],
                            q_rel,
                        )
                        total_delta += (goal - particle_q[particle_idx]) * (coeff * stiffness * inv_weight)
                cursor += 1

    x0 = particle_q_init[particle_idx]
    xp = particle_q[particle_idx]
    x_new = xp + total_delta
    v_new = (x_new - x0) / dt

    v_new_mag = wp.length(v_new)
    if v_new_mag > v_max:
        v_new *= v_max / v_new_mag

    particle_q_out[particle_idx] = x_new
    particle_qd_out[particle_idx] = v_new


@wp.kernel(enable_backward=False)
def solve_shape_matching_clusters_uniform8(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    indices_by_slot: wp.array(dtype=wp.int32),
    rest_local_positions_by_slot: wp.array(dtype=wp.vec3),
    rest_local_template: wp.array(dtype=wp.vec3),
    coefficients: wp.array(dtype=float),
    cluster_active: wp.array(dtype=wp.int32),
    particle_cluster_inv_weights: wp.array(dtype=float),
    cluster_count: int,
    use_rest_local_template: int,
    stiffness: float,
    rotation_iterations: int,
    cluster_rotations: wp.array(dtype=wp.quat),
    cluster_translations: wp.array(dtype=wp.vec3),
    particle_deltas: wp.array(dtype=wp.vec3),
):
    cluster_idx = wp.tid()

    if cluster_active[cluster_idx] == 0 or stiffness <= 0.0:
        return

    coeff = coefficients[cluster_idx]
    if coeff <= 0.0:
        return

    center = wp.vec3(0.0, 0.0, 0.0)
    rest_sum = wp.vec3(0.0, 0.0, 0.0)
    covariance = wp.mat33(0.0)
    member_count = int(0)
    dynamic_count = int(0)

    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:
            continue
        x = particle_q[particle_idx]
        q_rel = rest_local_template[local_idx]
        if use_rest_local_template == 0:
            q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]
        center += x
        rest_sum += q_rel
        covariance += wp.outer(q_rel, x)
        member_count += 1
        if particle_inv_mass[particle_idx] > 0.0:
            dynamic_count += 1

    if member_count == 0:
        return

    center /= float(member_count)
    covariance -= wp.outer(rest_sum, center)

    prev_rotation = cluster_rotations[cluster_idx]
    rotation = _extract_rotation(covariance, prev_rotation, rotation_iterations)
    if _quat_dot(rotation, prev_rotation) < 0.0:
        rotation = wp.quat(-rotation[0], -rotation[1], -rotation[2], -rotation[3])

    cluster_rotations[cluster_idx] = rotation
    cluster_translations[cluster_idx] = center

    if dynamic_count == 0:
        return

    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0 or particle_inv_mass[particle_idx] <= 0.0:
            continue

        q_rel = rest_local_template[local_idx]
        if use_rest_local_template == 0:
            q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]
        goal = center + wp.quat_rotate(rotation, q_rel)
        particle_scale = coeff * stiffness * particle_cluster_inv_weights[particle_idx]
        if particle_scale > 0.0:
            wp.atomic_add(particle_deltas, particle_idx, (goal - particle_q[particle_idx]) * particle_scale)


@wp.kernel(enable_backward=False)
def solve_shape_matching_clusters_uniform8_template_active(
    particle_q: wp.array(dtype=wp.vec3),
    indices_by_slot: wp.array(dtype=wp.int32),
    rest_local_template: wp.array(dtype=wp.vec3),
    coefficients: wp.array(dtype=float),
    cluster_active: wp.array(dtype=wp.int32),
    particle_cluster_inv_weights: wp.array(dtype=float),
    cluster_count: int,
    stiffness: float,
    rotation_iterations: int,
    cluster_rotations: wp.array(dtype=wp.quat),
    cluster_translations: wp.array(dtype=wp.vec3),
    particle_deltas: wp.array(dtype=wp.vec3),
):
    cluster_idx = wp.tid()

    if cluster_active[cluster_idx] == 0 or stiffness <= 0.0:
        return

    coeff = coefficients[cluster_idx]
    if coeff <= 0.0:
        return

    center = wp.vec3(0.0, 0.0, 0.0)
    rest_sum = wp.vec3(0.0, 0.0, 0.0)
    covariance = wp.mat33(0.0)
    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        x = particle_q[particle_idx]
        q_rel = rest_local_template[local_idx]
        center += x
        rest_sum += q_rel
        covariance += wp.outer(q_rel, x)

    center *= 0.125
    covariance -= wp.outer(rest_sum, center)

    prev_rotation = cluster_rotations[cluster_idx]
    rotation = _extract_rotation(covariance, prev_rotation, rotation_iterations)
    if _quat_dot(rotation, prev_rotation) < 0.0:
        rotation = wp.quat(-rotation[0], -rotation[1], -rotation[2], -rotation[3])

    cluster_rotations[cluster_idx] = rotation
    cluster_translations[cluster_idx] = center

    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        goal = center + wp.quat_rotate(rotation, rest_local_template[local_idx])
        particle_scale = coeff * stiffness * particle_cluster_inv_weights[particle_idx]
        if particle_scale > 0.0:
            wp.atomic_add(particle_deltas, particle_idx, (goal - particle_q[particle_idx]) * particle_scale)


@wp.kernel(enable_backward=False)
def solve_shape_matching_clusters_uniform8_colored_gs(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    color_cluster_indices: wp.array(dtype=wp.int32),
    indices_by_slot: wp.array(dtype=wp.int32),
    rest_local_positions_by_slot: wp.array(dtype=wp.vec3),
    rest_local_template: wp.array(dtype=wp.vec3),
    coefficients: wp.array(dtype=float),
    cluster_active: wp.array(dtype=wp.int32),
    particle_cluster_inv_weights: wp.array(dtype=float),
    cluster_count: int,
    use_rest_local_template: int,
    color_start: int,
    stiffness: float,
    support_alpha: float,
    rotation_iterations: int,
    cluster_rotations: wp.array(dtype=wp.quat),
):
    cluster_idx = color_cluster_indices[color_start + wp.tid()]

    if cluster_active[cluster_idx] == 0 or stiffness <= 0.0:
        return

    coeff = coefficients[cluster_idx]
    if coeff <= 0.0:
        return

    center = wp.vec3(0.0, 0.0, 0.0)
    rest_sum = wp.vec3(0.0, 0.0, 0.0)
    covariance = wp.mat33(0.0)
    member_count = int(0)
    dynamic_count = int(0)

    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:
            continue
        x = particle_q[particle_idx]
        q_rel = rest_local_template[local_idx]
        if use_rest_local_template == 0:
            q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]
        center += x
        rest_sum += q_rel
        covariance += wp.outer(q_rel, x)
        member_count += 1
        if particle_inv_mass[particle_idx] > 0.0:
            dynamic_count += 1

    if member_count == 0:
        return

    center /= float(member_count)
    covariance -= wp.outer(rest_sum, center)

    prev_rotation = cluster_rotations[cluster_idx]
    rotation = _extract_rotation(covariance, prev_rotation, rotation_iterations)
    if _quat_dot(rotation, prev_rotation) < 0.0:
        rotation = wp.quat(-rotation[0], -rotation[1], -rotation[2], -rotation[3])

    cluster_rotations[cluster_idx] = rotation

    if dynamic_count == 0:
        return

    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0 or particle_inv_mass[particle_idx] <= 0.0:
            continue

        q_rel = rest_local_template[local_idx]
        if use_rest_local_template == 0:
            q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]
        goal = center + wp.quat_rotate(rotation, q_rel)
        particle_scale = coeff * stiffness * _support_scale_from_alpha(
            particle_cluster_inv_weights[particle_idx],
            support_alpha,
        )
        if particle_scale > 0.0:
            x_new = particle_q[particle_idx] + (goal - particle_q[particle_idx]) * particle_scale
            particle_q[particle_idx] = x_new


@wp.kernel(enable_backward=False)
def solve_shape_matching_clusters_uniform8_colored_gs_template_active(
    particle_q: wp.array(dtype=wp.vec3),
    color_cluster_indices: wp.array(dtype=wp.int32),
    indices_by_slot: wp.array(dtype=wp.int32),
    rest_local_template: wp.array(dtype=wp.vec3),
    coefficients: wp.array(dtype=float),
    cluster_active: wp.array(dtype=wp.int32),
    particle_cluster_inv_weights: wp.array(dtype=float),
    cluster_count: int,
    color_start: int,
    stiffness: float,
    support_alpha: float,
    rotation_iterations: int,
    cluster_rotations: wp.array(dtype=wp.quat),
):
    cluster_idx = color_cluster_indices[color_start + wp.tid()]

    if cluster_active[cluster_idx] == 0 or stiffness <= 0.0:
        return

    coeff = coefficients[cluster_idx]
    if coeff <= 0.0:
        return

    center = wp.vec3(0.0, 0.0, 0.0)
    rest_sum = wp.vec3(0.0, 0.0, 0.0)
    covariance = wp.mat33(0.0)
    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        x = particle_q[particle_idx]
        q_rel = rest_local_template[local_idx]
        center += x
        rest_sum += q_rel
        covariance += wp.outer(q_rel, x)

    center *= 0.125
    covariance -= wp.outer(rest_sum, center)

    prev_rotation = cluster_rotations[cluster_idx]
    rotation = _extract_rotation(covariance, prev_rotation, rotation_iterations)
    if _quat_dot(rotation, prev_rotation) < 0.0:
        rotation = wp.quat(-rotation[0], -rotation[1], -rotation[2], -rotation[3])

    cluster_rotations[cluster_idx] = rotation

    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        goal = center + wp.quat_rotate(rotation, rest_local_template[local_idx])
        particle_scale = coeff * stiffness * _support_scale_from_alpha(
            particle_cluster_inv_weights[particle_idx],
            support_alpha,
        )
        if particle_scale > 0.0:
            x_new = particle_q[particle_idx] + (goal - particle_q[particle_idx]) * particle_scale
            particle_q[particle_idx] = x_new


@wp.kernel(enable_backward=False)
def solve_shape_matching_clusters_uniform27(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    indices_by_slot: wp.array(dtype=wp.int32),
    rest_local_positions_by_slot: wp.array(dtype=wp.vec3),
    rest_local_template: wp.array(dtype=wp.vec3),
    coefficients: wp.array(dtype=float),
    cluster_active: wp.array(dtype=wp.int32),
    particle_cluster_inv_weights: wp.array(dtype=float),
    cluster_count: int,
    use_rest_local_template: int,
    stiffness: float,
    rotation_iterations: int,
    cluster_rotations: wp.array(dtype=wp.quat),
    cluster_translations: wp.array(dtype=wp.vec3),
    particle_deltas: wp.array(dtype=wp.vec3),
):
    cluster_idx = wp.tid()

    if cluster_active[cluster_idx] == 0 or stiffness <= 0.0:
        return

    coeff = coefficients[cluster_idx]
    if coeff <= 0.0:
        return

    center = wp.vec3(0.0, 0.0, 0.0)
    member_count = int(0)
    dynamic_count = int(0)

    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_27)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:
            continue
        center += particle_q[particle_idx]
        member_count += 1
        if particle_inv_mass[particle_idx] > 0.0:
            dynamic_count += 1

    if member_count == 0:
        return

    center /= float(member_count)

    covariance = wp.mat33(0.0)
    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_27)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:
            continue
        x_rel = particle_q[particle_idx] - center
        q_rel = rest_local_template[local_idx]
        if use_rest_local_template == 0:
            q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]
        covariance += wp.outer(q_rel, x_rel)

    prev_rotation = cluster_rotations[cluster_idx]
    rotation = _extract_rotation(covariance, prev_rotation, rotation_iterations)
    if _quat_dot(rotation, prev_rotation) < 0.0:
        rotation = wp.quat(-rotation[0], -rotation[1], -rotation[2], -rotation[3])

    cluster_rotations[cluster_idx] = rotation
    cluster_translations[cluster_idx] = center

    if dynamic_count == 0:
        return

    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_27)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0 or particle_inv_mass[particle_idx] <= 0.0:
            continue

        q_rel = rest_local_template[local_idx]
        if use_rest_local_template == 0:
            q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]
        goal = center + wp.quat_rotate(rotation, q_rel)
        particle_scale = coeff * stiffness * particle_cluster_inv_weights[particle_idx]
        if particle_scale > 0.0:
            wp.atomic_add(particle_deltas, particle_idx, (goal - particle_q[particle_idx]) * particle_scale)


@wp.kernel(enable_backward=False)
def solve_shape_matching_clusters_uniform27_colored_gs(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    color_cluster_indices: wp.array(dtype=wp.int32),
    indices_by_slot: wp.array(dtype=wp.int32),
    rest_local_positions_by_slot: wp.array(dtype=wp.vec3),
    rest_local_template: wp.array(dtype=wp.vec3),
    coefficients: wp.array(dtype=float),
    cluster_active: wp.array(dtype=wp.int32),
    particle_cluster_inv_weights: wp.array(dtype=float),
    cluster_count: int,
    use_rest_local_template: int,
    color_start: int,
    stiffness: float,
    support_alpha: float,
    rotation_iterations: int,
    cluster_rotations: wp.array(dtype=wp.quat),
):
    cluster_idx = color_cluster_indices[color_start + wp.tid()]

    if cluster_active[cluster_idx] == 0 or stiffness <= 0.0:
        return

    coeff = coefficients[cluster_idx]
    if coeff <= 0.0:
        return

    center = wp.vec3(0.0, 0.0, 0.0)
    member_count = int(0)
    dynamic_count = int(0)

    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_27)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:
            continue
        center += particle_q[particle_idx]
        member_count += 1
        if particle_inv_mass[particle_idx] > 0.0:
            dynamic_count += 1

    if member_count == 0:
        return

    center /= float(member_count)

    covariance = wp.mat33(0.0)
    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_27)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:
            continue
        x_rel = particle_q[particle_idx] - center
        q_rel = rest_local_template[local_idx]
        if use_rest_local_template == 0:
            q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]
        covariance += wp.outer(q_rel, x_rel)

    prev_rotation = cluster_rotations[cluster_idx]
    rotation = _extract_rotation(covariance, prev_rotation, rotation_iterations)
    if _quat_dot(rotation, prev_rotation) < 0.0:
        rotation = wp.quat(-rotation[0], -rotation[1], -rotation[2], -rotation[3])

    cluster_rotations[cluster_idx] = rotation

    if dynamic_count == 0:
        return

    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_27)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0 or particle_inv_mass[particle_idx] <= 0.0:
            continue

        q_rel = rest_local_template[local_idx]
        if use_rest_local_template == 0:
            q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]
        goal = center + wp.quat_rotate(rotation, q_rel)
        particle_scale = coeff * stiffness * _support_scale_from_alpha(
            particle_cluster_inv_weights[particle_idx],
            support_alpha,
        )
        if particle_scale > 0.0:
            x_new = particle_q[particle_idx] + (goal - particle_q[particle_idx]) * particle_scale
            particle_q[particle_idx] = x_new


@wp.kernel(enable_backward=False)
def solve_shape_matching_clusters_uniform125(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    indices_by_slot: wp.array(dtype=wp.int32),
    rest_local_positions_by_slot: wp.array(dtype=wp.vec3),
    rest_local_template: wp.array(dtype=wp.vec3),
    coefficients: wp.array(dtype=float),
    cluster_active: wp.array(dtype=wp.int32),
    particle_cluster_inv_weights: wp.array(dtype=float),
    cluster_count: int,
    use_rest_local_template: int,
    stiffness: float,
    rotation_iterations: int,
    cluster_rotations: wp.array(dtype=wp.quat),
    cluster_translations: wp.array(dtype=wp.vec3),
    particle_deltas: wp.array(dtype=wp.vec3),
):
    cluster_idx = wp.tid()

    if cluster_active[cluster_idx] == 0 or stiffness <= 0.0:
        return

    coeff = coefficients[cluster_idx]
    if coeff <= 0.0:
        return

    center = wp.vec3(0.0, 0.0, 0.0)
    member_count = int(0)
    dynamic_count = int(0)

    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_125)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:
            continue
        center += particle_q[particle_idx]
        member_count += 1
        if particle_inv_mass[particle_idx] > 0.0:
            dynamic_count += 1

    if member_count == 0:
        return

    center /= float(member_count)

    covariance = wp.mat33(0.0)
    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_125)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:
            continue
        x_rel = particle_q[particle_idx] - center
        q_rel = rest_local_template[local_idx]
        if use_rest_local_template == 0:
            q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]
        covariance += wp.outer(q_rel, x_rel)

    prev_rotation = cluster_rotations[cluster_idx]
    rotation = _extract_rotation(covariance, prev_rotation, rotation_iterations)
    if _quat_dot(rotation, prev_rotation) < 0.0:
        rotation = wp.quat(-rotation[0], -rotation[1], -rotation[2], -rotation[3])

    cluster_rotations[cluster_idx] = rotation
    cluster_translations[cluster_idx] = center

    if dynamic_count == 0:
        return

    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_125)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0 or particle_inv_mass[particle_idx] <= 0.0:
            continue

        q_rel = rest_local_template[local_idx]
        if use_rest_local_template == 0:
            q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]
        goal = center + wp.quat_rotate(rotation, q_rel)
        particle_scale = coeff * stiffness * particle_cluster_inv_weights[particle_idx]
        if particle_scale > 0.0:
            wp.atomic_add(particle_deltas, particle_idx, (goal - particle_q[particle_idx]) * particle_scale)


@wp.kernel(enable_backward=False)
def solve_shape_matching_clusters_uniform125_colored_gs(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    color_cluster_indices: wp.array(dtype=wp.int32),
    indices_by_slot: wp.array(dtype=wp.int32),
    rest_local_positions_by_slot: wp.array(dtype=wp.vec3),
    rest_local_template: wp.array(dtype=wp.vec3),
    coefficients: wp.array(dtype=float),
    cluster_active: wp.array(dtype=wp.int32),
    particle_cluster_inv_weights: wp.array(dtype=float),
    cluster_count: int,
    use_rest_local_template: int,
    color_start: int,
    stiffness: float,
    support_alpha: float,
    rotation_iterations: int,
    cluster_rotations: wp.array(dtype=wp.quat),
):
    cluster_idx = color_cluster_indices[color_start + wp.tid()]

    if cluster_active[cluster_idx] == 0 or stiffness <= 0.0:
        return

    coeff = coefficients[cluster_idx]
    if coeff <= 0.0:
        return

    center = wp.vec3(0.0, 0.0, 0.0)
    member_count = int(0)
    dynamic_count = int(0)

    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_125)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:
            continue
        center += particle_q[particle_idx]
        member_count += 1
        if particle_inv_mass[particle_idx] > 0.0:
            dynamic_count += 1

    if member_count == 0:
        return

    center /= float(member_count)

    covariance = wp.mat33(0.0)
    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_125)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:
            continue
        x_rel = particle_q[particle_idx] - center
        q_rel = rest_local_template[local_idx]
        if use_rest_local_template == 0:
            q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]
        covariance += wp.outer(q_rel, x_rel)

    prev_rotation = cluster_rotations[cluster_idx]
    rotation = _extract_rotation(covariance, prev_rotation, rotation_iterations)
    if _quat_dot(rotation, prev_rotation) < 0.0:
        rotation = wp.quat(-rotation[0], -rotation[1], -rotation[2], -rotation[3])

    cluster_rotations[cluster_idx] = rotation

    if dynamic_count == 0:
        return

    for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_125)):
        particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]
        if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0 or particle_inv_mass[particle_idx] <= 0.0:
            continue

        q_rel = rest_local_template[local_idx]
        if use_rest_local_template == 0:
            q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]
        goal = center + wp.quat_rotate(rotation, q_rel)
        particle_scale = coeff * stiffness * _support_scale_from_alpha(
            particle_cluster_inv_weights[particle_idx],
            support_alpha,
        )
        if particle_scale > 0.0:
            x_new = particle_q[particle_idx] + (goal - particle_q[particle_idx]) * particle_scale
            particle_q[particle_idx] = x_new


@wp.func
def _slot_uses_upper_x(slot: int) -> int:
    if slot == 1 or slot == 2 or slot == 5 or slot == 6:
        return 1
    return 0


@wp.func
def _slot_uses_upper_y(slot: int) -> int:
    if slot == 2 or slot == 3 or slot == 6 or slot == 7:
        return 1
    return 0


@wp.func
def _slot_uses_upper_z(slot: int) -> int:
    if slot >= 4:
        return 1
    return 0


@wp.kernel(enable_backward=False)
def prolongate_shape_matching_corrections_uniform8_table(
    particle_q_init: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    coarse_q_snapshot: wp.array(dtype=wp.vec3),
    parent_indices: wp.array(dtype=wp.int32),
    parent_weights: wp.array(dtype=float),
    child_cluster: wp.array(dtype=wp.int32),
    cluster_active: wp.array(dtype=wp.int32),
    absolute_projection: int,
    dt: float,
    v_max: float,
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_qd_out: wp.array(dtype=wp.vec3),
):
    particle_idx = wp.tid()
    x_old = particle_q[particle_idx]
    x_new = x_old
    v_old = particle_qd[particle_idx]
    changed = int(0)

    if (
        (particle_flags[particle_idx] & ParticleFlags.ACTIVE) != 0
        and particle_inv_mass[particle_idx] > 0.0
    ):
        cluster_idx = child_cluster[particle_idx]
        if cluster_idx >= 0 and cluster_active[cluster_idx] != 0:
            base = particle_idx * wp.static(UNIFORM_CLUSTER_SIZE)
            interpolated = wp.vec3(0.0, 0.0, 0.0)
            valid = int(1)
            for parent_slot in range(wp.static(UNIFORM_CLUSTER_SIZE)):
                parent_idx = parent_indices[base + parent_slot]
                if parent_idx < 0 or (particle_flags[parent_idx] & ParticleFlags.ACTIVE) == 0:
                    valid = int(0)
                    break
                weight = parent_weights[base + parent_slot]
                if absolute_projection != 0:
                    interpolated += particle_q[parent_idx] * weight
                else:
                    interpolated += (particle_q[parent_idx] - coarse_q_snapshot[parent_idx]) * weight
            if valid != 0:
                if absolute_projection != 0:
                    x_new = interpolated
                else:
                    x_new = x_old + interpolated
                changed = int(1)

    if changed == 0:
        particle_q_out[particle_idx] = x_old
        particle_qd_out[particle_idx] = v_old
        return

    v_new = (x_new - particle_q_init[particle_idx]) / dt
    v_new_mag = wp.length(v_new)
    if v_new_mag > v_max:
        v_new *= v_max / v_new_mag

    particle_q_out[particle_idx] = x_new
    particle_qd_out[particle_idx] = v_new


@wp.kernel(enable_backward=False)
def prolongate_shape_matching_corrections_uniform8(
    particle_q_init: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    coarse_q_snapshot: wp.array(dtype=wp.vec3),
    child_cluster: wp.array(dtype=wp.int32),
    block_keys: wp.array2d(dtype=wp.int32),
    node_grid_xyz: wp.array2d(dtype=wp.int32),
    grid_to_node: wp.array3d(dtype=wp.int32),
    block_size: int,
    max_grid_x: int,
    max_grid_y: int,
    max_grid_z: int,
    cluster_active: wp.array(dtype=wp.int32),
    absolute_projection: int,
    dt: float,
    v_max: float,
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_qd_out: wp.array(dtype=wp.vec3),
):
    particle_idx = wp.tid()
    x_old = particle_q[particle_idx]
    x_new = x_old
    v_old = particle_qd[particle_idx]
    changed = int(0)

    if (
        (particle_flags[particle_idx] & ParticleFlags.ACTIVE) != 0
        and particle_inv_mass[particle_idx] > 0.0
    ):
        cluster_idx = child_cluster[particle_idx]
        if cluster_idx >= 0 and cluster_active[cluster_idx] != 0:
            gx = node_grid_xyz[particle_idx, 0]
            gy = node_grid_xyz[particle_idx, 1]
            gz = node_grid_xyz[particle_idx, 2]
            lx = block_keys[cluster_idx, 0] * block_size
            ly = block_keys[cluster_idx, 1] * block_size
            lz = block_keys[cluster_idx, 2] * block_size
            ux = lx + block_size
            uy = ly + block_size
            uz = lz + block_size
            fx = float(gx - lx) / float(block_size)
            fy = float(gy - ly) / float(block_size)
            fz = float(gz - lz) / float(block_size)
            interpolated = wp.vec3(0.0, 0.0, 0.0)
            valid = int(1)
            if (
                lx < 0
                or ly < 0
                or lz < 0
                or ux > max_grid_x
                or uy > max_grid_y
                or uz > max_grid_z
                or gx < lx
                or gy < ly
                or gz < lz
                or gx > ux
                or gy > uy
                or gz > uz
            ):
                valid = int(0)
            if valid != 0:
                for parent_slot in range(wp.static(UNIFORM_CLUSTER_SIZE)):
                    px = lx
                    wx = 1.0 - fx
                    if _slot_uses_upper_x(parent_slot) != 0:
                        px = ux
                        wx = fx

                    py = ly
                    wy = 1.0 - fy
                    if _slot_uses_upper_y(parent_slot) != 0:
                        py = uy
                        wy = fy

                    pz = lz
                    wz = 1.0 - fz
                    if _slot_uses_upper_z(parent_slot) != 0:
                        pz = uz
                        wz = fz

                    parent_idx = grid_to_node[px, py, pz]
                    if parent_idx < 0 or (particle_flags[parent_idx] & ParticleFlags.ACTIVE) == 0:
                        valid = int(0)
                        break
                    weight = wx * wy * wz
                    if absolute_projection != 0:
                        interpolated += particle_q[parent_idx] * weight
                    else:
                        interpolated += (particle_q[parent_idx] - coarse_q_snapshot[parent_idx]) * weight
            if valid != 0:
                if absolute_projection != 0:
                    x_new = interpolated
                else:
                    x_new = x_old + interpolated
                changed = int(1)

    if changed == 0:
        particle_q_out[particle_idx] = x_old
        particle_qd_out[particle_idx] = v_old
        return

    v_new = (x_new - particle_q_init[particle_idx]) / dt
    v_new_mag = wp.length(v_new)
    if v_new_mag > v_max:
        v_new *= v_max / v_new_mag

    particle_q_out[particle_idx] = x_new
    particle_qd_out[particle_idx] = v_new


@wp.kernel(enable_backward=False)
def project_shape_matching_children_uniform8_table(
    particle_q_init: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    awake_l0_cluster_counts: wp.array(dtype=wp.int32),
    parent_indices: wp.array(dtype=wp.int32),
    parent_weights: wp.array(dtype=float),
    child_cluster: wp.array(dtype=wp.int32),
    projection_active: wp.array(dtype=wp.int32),
    projection_stiffness: float,
    dt: float,
    v_max: float,
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_qd_out: wp.array(dtype=wp.vec3),
):
    particle_idx = wp.tid()
    x_old = particle_q[particle_idx]
    v_old = particle_qd[particle_idx]

    alpha = projection_stiffness
    if alpha <= 0.0:
        particle_q_out[particle_idx] = x_old
        particle_qd_out[particle_idx] = v_old
        return
    if alpha > 1.0:
        alpha = 1.0

    if (
        (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0
        or particle_inv_mass[particle_idx] <= 0.0
        or awake_l0_cluster_counts[particle_idx] > 0
    ):
        particle_q_out[particle_idx] = x_old
        particle_qd_out[particle_idx] = v_old
        return

    cluster_idx = child_cluster[particle_idx]
    if cluster_idx < 0 or projection_active[cluster_idx] == 0:
        particle_q_out[particle_idx] = x_old
        particle_qd_out[particle_idx] = v_old
        return

    base = particle_idx * wp.static(UNIFORM_CLUSTER_SIZE)
    x_target = wp.vec3(0.0, 0.0, 0.0)
    valid = int(1)
    for parent_slot in range(wp.static(UNIFORM_CLUSTER_SIZE)):
        parent_idx = parent_indices[base + parent_slot]
        if parent_idx < 0 or (particle_flags[parent_idx] & ParticleFlags.ACTIVE) == 0:
            valid = int(0)
            break
        x_target += particle_q[parent_idx] * parent_weights[base + parent_slot]

    if valid == 0:
        particle_q_out[particle_idx] = x_old
        particle_qd_out[particle_idx] = v_old
        return

    x_new = x_old + (x_target - x_old) * alpha
    v_new = (x_new - particle_q_init[particle_idx]) / dt
    v_new_mag = wp.length(v_new)
    if v_new_mag > v_max:
        v_new *= v_max / v_new_mag

    particle_q_out[particle_idx] = x_new
    particle_qd_out[particle_idx] = v_new


@wp.kernel(enable_backward=False)
def project_shape_matching_children_uniform8(
    particle_q_init: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    awake_l0_cluster_counts: wp.array(dtype=wp.int32),
    child_cluster: wp.array(dtype=wp.int32),
    block_keys: wp.array2d(dtype=wp.int32),
    node_grid_xyz: wp.array2d(dtype=wp.int32),
    grid_to_node: wp.array3d(dtype=wp.int32),
    block_size: int,
    max_grid_x: int,
    max_grid_y: int,
    max_grid_z: int,
    projection_active: wp.array(dtype=wp.int32),
    projection_stiffness: float,
    dt: float,
    v_max: float,
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_qd_out: wp.array(dtype=wp.vec3),
):
    particle_idx = wp.tid()
    x_old = particle_q[particle_idx]
    v_old = particle_qd[particle_idx]

    alpha = projection_stiffness
    if alpha <= 0.0:
        particle_q_out[particle_idx] = x_old
        particle_qd_out[particle_idx] = v_old
        return
    if alpha > 1.0:
        alpha = 1.0

    if (
        (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0
        or particle_inv_mass[particle_idx] <= 0.0
        or awake_l0_cluster_counts[particle_idx] > 0
    ):
        particle_q_out[particle_idx] = x_old
        particle_qd_out[particle_idx] = v_old
        return

    cluster_idx = child_cluster[particle_idx]
    if cluster_idx < 0 or projection_active[cluster_idx] == 0:
        particle_q_out[particle_idx] = x_old
        particle_qd_out[particle_idx] = v_old
        return

    gx = node_grid_xyz[particle_idx, 0]
    gy = node_grid_xyz[particle_idx, 1]
    gz = node_grid_xyz[particle_idx, 2]
    lx = block_keys[cluster_idx, 0] * block_size
    ly = block_keys[cluster_idx, 1] * block_size
    lz = block_keys[cluster_idx, 2] * block_size
    ux = lx + block_size
    uy = ly + block_size
    uz = lz + block_size
    fx = float(gx - lx) / float(block_size)
    fy = float(gy - ly) / float(block_size)
    fz = float(gz - lz) / float(block_size)
    x_target = wp.vec3(0.0, 0.0, 0.0)
    valid = int(1)
    if (
        lx < 0
        or ly < 0
        or lz < 0
        or ux > max_grid_x
        or uy > max_grid_y
        or uz > max_grid_z
        or gx < lx
        or gy < ly
        or gz < lz
        or gx > ux
        or gy > uy
        or gz > uz
    ):
        valid = int(0)
    if valid != 0:
        for parent_slot in range(wp.static(UNIFORM_CLUSTER_SIZE)):
            px = lx
            wx = 1.0 - fx
            if _slot_uses_upper_x(parent_slot) != 0:
                px = ux
                wx = fx

            py = ly
            wy = 1.0 - fy
            if _slot_uses_upper_y(parent_slot) != 0:
                py = uy
                wy = fy

            pz = lz
            wz = 1.0 - fz
            if _slot_uses_upper_z(parent_slot) != 0:
                pz = uz
                wz = fz

            parent_idx = grid_to_node[px, py, pz]
            if parent_idx < 0 or (particle_flags[parent_idx] & ParticleFlags.ACTIVE) == 0:
                valid = int(0)
                break
            x_target += particle_q[parent_idx] * (wx * wy * wz)

    if valid == 0:
        particle_q_out[particle_idx] = x_old
        particle_qd_out[particle_idx] = v_old
        return

    x_new = x_old + (x_target - x_old) * alpha
    v_new = (x_new - particle_q_init[particle_idx]) / dt
    v_new_mag = wp.length(v_new)
    if v_new_mag > v_max:
        v_new *= v_max / v_new_mag

    particle_q_out[particle_idx] = x_new
    particle_qd_out[particle_idx] = v_new


@wp.kernel(enable_backward=False)
def mark_wake_blocks_from_deleted_cells_kernel(
    deleted_cells: wp.array(dtype=wp.int32),
    deleted_count: wp.array(dtype=wp.int32),
    num_cells: int,
    cell_to_cluster: wp.array(dtype=wp.int32),
    block_keys: wp.array2d(dtype=wp.int32),
    wake_halo_blocks: int,
    wake_mask: wp.array3d(dtype=wp.int32),
):
    i = wp.tid()
    if i >= deleted_count[0]:
        return

    cell_idx = deleted_cells[i]
    if cell_idx < 0 or cell_idx >= num_cells:
        return

    cluster_idx = cell_to_cluster[cell_idx]
    if cluster_idx < 0:
        return

    bx = block_keys[cluster_idx, 0]
    by = block_keys[cluster_idx, 1]
    bz = block_keys[cluster_idx, 2]
    nx = wake_mask.shape[0]
    ny = wake_mask.shape[1]
    nz = wake_mask.shape[2]
    halo = wake_halo_blocks
    if halo < 0:
        halo = 0

    dx = -halo
    while dx <= halo:
        x = bx + dx
        if x >= 0 and x < nx:
            dy = -halo
            while dy <= halo:
                y = by + dy
                if y >= 0 and y < ny:
                    dz = -halo
                    while dz <= halo:
                        z = bz + dz
                        if z >= 0 and z < nz:
                            wake_mask[x, y, z] = 1
                        dz += 1
                dy += 1
        dx += 1


@wp.kernel(enable_backward=False)
def compute_coarse_sleep_projection_kernel(
    cluster_active: wp.array(dtype=wp.int32),
    block_keys: wp.array2d(dtype=wp.int32),
    sleepable: wp.array(dtype=wp.int32),
    wake_mask: wp.array3d(dtype=wp.int32),
    level_enabled: int,
    projection_active: wp.array(dtype=wp.int32),
    projection_count: wp.array(dtype=wp.int32),
):
    cluster_idx = wp.tid()
    value = int(0)
    if level_enabled != 0 and cluster_active[cluster_idx] != 0 and sleepable[cluster_idx] != 0:
        bx = block_keys[cluster_idx, 0]
        by = block_keys[cluster_idx, 1]
        bz = block_keys[cluster_idx, 2]
        if (
            bx >= 0
            and bx < wake_mask.shape[0]
            and by >= 0
            and by < wake_mask.shape[1]
            and bz >= 0
            and bz < wake_mask.shape[2]
            and wake_mask[bx, by, bz] == 0
        ):
            value = int(1)

    projection_active[cluster_idx] = value
    if value != 0:
        wp.atomic_add(projection_count, 0, 1)


@wp.kernel(enable_backward=False)
def mask_l1_projection_covered_by_l2_kernel(
    l1_source_cell: wp.array(dtype=wp.int32),
    l2_cell_to_cluster: wp.array(dtype=wp.int32),
    l2_projection_active: wp.array(dtype=wp.int32),
    l1_projection_active: wp.array(dtype=wp.int32),
    l1_projection_count: wp.array(dtype=wp.int32),
):
    l1_idx = wp.tid()
    if l1_projection_active[l1_idx] == 0:
        return

    source_cell = l1_source_cell[l1_idx]
    parent_l2 = l2_cell_to_cluster[source_cell]
    if parent_l2 >= 0 and l2_projection_active[parent_l2] != 0:
        l1_projection_active[l1_idx] = 0
        wp.atomic_sub(l1_projection_count, 0, 1)


@wp.kernel(enable_backward=False)
def compute_l0_runtime_active_kernel(
    l0_active: wp.array(dtype=wp.int32),
    l0_source_cell: wp.array(dtype=wp.int32),
    l1_cell_to_cluster: wp.array(dtype=wp.int32),
    l1_projection_active: wp.array(dtype=wp.int32),
    has_l1_projection: int,
    l2_cell_to_cluster: wp.array(dtype=wp.int32),
    l2_projection_active: wp.array(dtype=wp.int32),
    has_l2_projection: int,
    runtime_active: wp.array(dtype=wp.int32),
    sleeping_count: wp.array(dtype=wp.int32),
):
    cluster_idx = wp.tid()
    active = l0_active[cluster_idx]
    value = active
    slept = int(0)

    if active != 0:
        source_cell = l0_source_cell[cluster_idx]
        if has_l2_projection != 0:
            parent_l2 = l2_cell_to_cluster[source_cell]
            if parent_l2 >= 0 and l2_projection_active[parent_l2] != 0:
                slept = int(1)
        if slept == 0 and has_l1_projection != 0:
            parent_l1 = l1_cell_to_cluster[source_cell]
            if parent_l1 >= 0 and l1_projection_active[parent_l1] != 0:
                slept = int(1)

    if slept != 0:
        value = int(0)
        wp.atomic_add(sleeping_count, 0, 1)

    runtime_active[cluster_idx] = value


@wp.kernel(enable_backward=False)
def accumulate_runtime_cluster_counts_kernel(
    cluster_active: wp.array(dtype=wp.int32),
    cluster_offsets: wp.array(dtype=wp.int32),
    cluster_indices: wp.array(dtype=wp.int32),
    particle_cluster_counts: wp.array(dtype=wp.int32),
):
    cluster_idx = wp.tid()
    if cluster_active[cluster_idx] == 0:
        return

    cursor = cluster_offsets[cluster_idx]
    end = cluster_offsets[cluster_idx + 1]
    while cursor < end:
        particle_idx = cluster_indices[cursor]
        if particle_idx >= 0:
            wp.atomic_add(particle_cluster_counts, particle_idx, 1)
        cursor += 1


@wp.kernel(enable_backward=False)
def finalize_runtime_cluster_inv_weights_kernel(
    particle_cluster_counts: wp.array(dtype=wp.int32),
    particle_cluster_inv_weights: wp.array(dtype=float),
):
    particle_idx = wp.tid()
    count = particle_cluster_counts[particle_idx]
    if count > 0:
        particle_cluster_inv_weights[particle_idx] = 1.0 / float(count)
    else:
        particle_cluster_counts[particle_idx] = 0
        particle_cluster_inv_weights[particle_idx] = 0.0


__all__ = [
    "accumulate_runtime_cluster_counts_kernel",
    "apply_shape_matching_particle_gather_uniform8",
    "compute_coarse_sleep_projection_kernel",
    "compute_l0_runtime_active_kernel",
    "compute_shape_matching_cluster_poses_uniform8",
    "finalize_position_update_from_q",
    "finalize_runtime_cluster_inv_weights_kernel",
    "mark_wake_blocks_from_deleted_cells_kernel",
    "mask_l1_projection_covered_by_l2_kernel",
    "project_shape_matching_children_uniform8",
    "project_shape_matching_children_uniform8_table",
    "prolongate_shape_matching_corrections_uniform8",
    "prolongate_shape_matching_corrections_uniform8_table",
    "solve_shape_matching_clusters_uniform8",
    "solve_shape_matching_clusters_uniform8_colored_gs",
    "solve_shape_matching_clusters_uniform8_colored_gs_template_active",
    "solve_shape_matching_clusters_uniform8_template_active",
    "solve_shape_matching_clusters_uniform27",
    "solve_shape_matching_clusters_uniform27_colored_gs",
    "solve_shape_matching_clusters_uniform125",
    "solve_shape_matching_clusters_uniform125_colored_gs",
    "solve_volume_constraints_uniform8",
]
