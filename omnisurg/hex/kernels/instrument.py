# SPDX-License-Identifier: Apache-2.0
"""Kinematic instrument contact kernels."""

from __future__ import annotations

import warp as wp
from newton._src.geometry.flags import ParticleFlags
from newton._src.geometry.kernels import triangle_closest_point

_ACTIVE_BIT = wp.constant(wp.int32(int(ParticleFlags.ACTIVE)))


@wp.func
def _active_particle_inv_mass(
    particle_index: int,
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
) -> float:
    if particle_index < 0:
        return 0.0
    if (particle_flags[particle_index] & _ACTIVE_BIT) == 0:
        return 0.0
    w = particle_inv_mass[particle_index]
    if w <= 0.0:
        return 0.0
    return w


@wp.func
def _cell_inv_mass_sum(
    cell_index: int,
    cell_nodes: wp.array2d(dtype=wp.int32),
    cell_active: wp.array(dtype=wp.int32),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
) -> float:
    if cell_index < 0:
        return 0.0
    if cell_active[cell_index] == 0:
        return 0.0

    w = float(0.0)
    for i in range(8):
        w = w + _active_particle_inv_mass(cell_nodes[cell_index, i], particle_inv_mass, particle_flags)
    return w


@wp.func
def _sphere_triangle_fallback_dir(
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
    contact_point: wp.vec3,
) -> wp.vec3:
    normal = wp.cross(v1 - v0, v2 - v0)
    if wp.dot(normal, normal) > 1.0e-12:
        return wp.normalize(normal)

    centroid = (v0 + v1 + v2) / 3.0
    centroid_dir = centroid - contact_point
    if wp.dot(centroid_dir, centroid_dir) > 1.0e-12:
        return wp.normalize(centroid_dir)

    d0 = v0 - contact_point
    d1 = v1 - contact_point
    d2 = v2 - contact_point

    best = d0
    if wp.dot(d1, d1) > wp.dot(best, best):
        best = d1
    if wp.dot(d2, d2) > wp.dot(best, best):
        best = d2
    if wp.dot(best, best) > 1.0e-12:
        return wp.normalize(best)

    return wp.vec3(1.0, 0.0, 0.0)


@wp.func
def _scatter_cell_correction_to_nodes(
    cell_index: int,
    correction: wp.vec3,
    cell_weight: float,
    cell_nodes: wp.array2d(dtype=wp.int32),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_delta_accumulator: wp.array(dtype=wp.vec3),
    particle_delta_counter: wp.array(dtype=wp.int32),
):
    if cell_index < 0:
        return
    if cell_weight <= 0.0:
        return

    for i in range(8):
        p = cell_nodes[cell_index, i]
        w = _active_particle_inv_mass(p, particle_inv_mass, particle_flags)
        if w > 0.0:
            wp.atomic_add(particle_delta_accumulator, p, correction * (w / cell_weight))
            wp.atomic_add(particle_delta_counter, p, 1)


@wp.kernel(enable_backward=False)
def project_kinematic_sphere_particle_positions_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    sphere_q_prev: wp.array(dtype=wp.vec3),
    sphere_q: wp.array(dtype=wp.vec3),
    sphere_count: int,
    sphere_radius: float,
    interpolation_alpha: float,
    relaxation: float,
    max_correction: float,
):
    tid = wp.tid()

    if (particle_flags[tid] & _ACTIVE_BIT) == 0:
        return
    if particle_inv_mass[tid] <= 0.0:
        return

    alpha = interpolation_alpha
    if alpha < 0.0:
        alpha = 0.0
    if alpha > 1.0:
        alpha = 1.0

    contact_relaxation = relaxation
    if contact_relaxation < 0.0:
        contact_relaxation = 0.0
    if contact_relaxation > 1.0:
        contact_relaxation = 1.0

    correction_cap = max_correction
    if correction_cap < 0.0:
        correction_cap = 0.0

    q = particle_q[tid]
    correction_total = wp.vec3(0.0, 0.0, 0.0)
    combined_radius = particle_radius[tid] + sphere_radius

    for sphere_idx in range(sphere_count):
        center = sphere_q_prev[sphere_idx] + (sphere_q[sphere_idx] - sphere_q_prev[sphere_idx]) * alpha
        delta = q - center
        dist = wp.length(delta)
        penetration = combined_radius - dist
        if penetration > 0.0:
            normal = wp.vec3(1.0, 0.0, 0.0)
            if dist > 1.0e-8:
                normal = delta / dist
            correction = normal * (penetration * contact_relaxation)
            correction_len = wp.length(correction)
            if correction_cap > 0.0 and correction_len > correction_cap:
                correction = correction * (correction_cap / correction_len)
            q = q + correction
            correction_total = correction_total + correction

    if wp.dot(correction_total, correction_total) > 0.0:
        particle_q[tid] = q


@wp.kernel(enable_backward=False)
def project_kinematic_sphere_mc_triangle_node_positions_kernel(
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    cell_nodes: wp.array2d(dtype=wp.int32),
    cell_active: wp.array(dtype=wp.int32),
    vertex_pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array2d(dtype=wp.int32),
    triangle_count: int,
    sphere_q_prev: wp.array(dtype=wp.vec3),
    sphere_q: wp.array(dtype=wp.vec3),
    sphere_count: int,
    sphere_radius: float,
    interpolation_alpha: float,
    relaxation: float,
    max_correction: float,
    particle_delta_accumulator: wp.array(dtype=wp.vec3),
    particle_delta_counter: wp.array(dtype=wp.int32),
):
    tri = wp.tid()
    if tri >= triangle_count:
        return

    v0_id = tri_indices[tri, 0]
    v1_id = tri_indices[tri, 1]
    v2_id = tri_indices[tri, 2]
    if v0_id < 0 or v1_id < 0 or v2_id < 0:
        return

    # MC vertex ids are dense: vertex_id(cell, dir) = cell * 6 + dir.
    c0 = v0_id / 6
    c1 = v1_id / 6
    c2 = v2_id / 6

    w0 = _cell_inv_mass_sum(c0, cell_nodes, cell_active, particle_inv_mass, particle_flags)
    w1 = _cell_inv_mass_sum(c1, cell_nodes, cell_active, particle_inv_mass, particle_flags)
    w2 = _cell_inv_mass_sum(c2, cell_nodes, cell_active, particle_inv_mass, particle_flags)
    weight = w0 + w1 + w2
    if weight <= 0.0:
        return

    alpha = interpolation_alpha
    if alpha < 0.0:
        alpha = 0.0
    if alpha > 1.0:
        alpha = 1.0

    contact_relaxation = relaxation
    if contact_relaxation < 0.0:
        contact_relaxation = 0.0
    if contact_relaxation > 1.0:
        contact_relaxation = 1.0

    correction_cap = max_correction
    if correction_cap < 0.0:
        correction_cap = 0.0

    p0 = vertex_pos[v0_id]
    p1 = vertex_pos[v1_id]
    p2 = vertex_pos[v2_id]

    for sphere_idx in range(sphere_count):
        center = sphere_q_prev[sphere_idx] + (sphere_q[sphere_idx] - sphere_q_prev[sphere_idx]) * alpha
        closest_p, _bary, _feature_type = triangle_closest_point(p0, p1, p2, center)

        to_triangle = closest_p - center
        dist = wp.length(to_triangle)
        penetration = sphere_radius - dist
        if penetration <= 0.0:
            continue

        correction_dir = wp.vec3(1.0, 0.0, 0.0)
        if dist > 1.0e-8:
            correction_dir = to_triangle / dist
        else:
            correction_dir = _sphere_triangle_fallback_dir(p0, p1, p2, center)

        correction = correction_dir * (penetration * contact_relaxation)
        correction_len = wp.length(correction)
        if correction_cap > 0.0 and correction_len > correction_cap:
            correction = correction * (correction_cap / correction_len)

        _scatter_cell_correction_to_nodes(
            c0,
            correction * (w0 / weight),
            w0,
            cell_nodes,
            particle_inv_mass,
            particle_flags,
            particle_delta_accumulator,
            particle_delta_counter,
        )
        _scatter_cell_correction_to_nodes(
            c1,
            correction * (w1 / weight),
            w1,
            cell_nodes,
            particle_inv_mass,
            particle_flags,
            particle_delta_accumulator,
            particle_delta_counter,
        )
        _scatter_cell_correction_to_nodes(
            c2,
            correction * (w2 / weight),
            w2,
            cell_nodes,
            particle_inv_mass,
            particle_flags,
            particle_delta_accumulator,
            particle_delta_counter,
        )


@wp.kernel(enable_backward=False)
def apply_kinematic_sphere_mc_triangle_node_deltas_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    particle_delta_accumulator: wp.array(dtype=wp.vec3),
    particle_delta_counter: wp.array(dtype=wp.int32),
):
    p = wp.tid()
    if _active_particle_inv_mass(p, particle_inv_mass, particle_flags) <= 0.0:
        return
    count = particle_delta_counter[p]
    if count <= 0:
        return
    particle_q[p] = particle_q[p] + particle_delta_accumulator[p] / float(count)


def project_kinematic_sphere_particle_positions(
    particle_q: wp.array,
    model,
    sphere_q_prev: wp.array,
    sphere_q: wp.array,
    sphere_count: int,
    sphere_radius: float,
    interpolation_alpha: float,
    relaxation: float,
    iterations: int = 1,
    max_correction: float = 0.0,
    device=None,
) -> None:
    sphere_count = int(sphere_count)
    sphere_radius = float(sphere_radius)
    if sphere_count <= 0 or sphere_radius <= 0.0:
        return
    for _ in range(max(1, int(iterations))):
        wp.launch(
            kernel=project_kinematic_sphere_particle_positions_kernel,
            dim=int(model.particle_count),
            inputs=[
                particle_q,
                model.particle_inv_mass,
                model.particle_radius,
                model.particle_flags,
                sphere_q_prev,
                sphere_q,
                sphere_count,
                sphere_radius,
                float(interpolation_alpha),
                float(relaxation),
                float(max_correction),
            ],
            device=model.device if device is None else device,
        )


def project_kinematic_sphere_mc_triangle_node_positions(
    particle_q: wp.array,
    particle_inv_mass: wp.array,
    particle_flags: wp.array,
    cell_nodes: wp.array,
    cell_active: wp.array,
    vertex_pos: wp.array,
    tri_indices: wp.array,
    triangle_count: int,
    sphere_q_prev: wp.array,
    sphere_q: wp.array,
    sphere_count: int,
    sphere_radius: float,
    interpolation_alpha: float,
    relaxation: float,
    particle_delta_accumulator: wp.array,
    particle_delta_counter: wp.array,
    iterations: int = 1,
    max_correction: float = 0.0,
    device=None,
) -> None:
    triangle_count = int(triangle_count)
    sphere_count = int(sphere_count)
    sphere_radius = float(sphere_radius)
    if triangle_count <= 0 or sphere_count <= 0 or sphere_radius <= 0.0:
        return

    launch_device = particle_q.device if device is None else device
    for _ in range(max(1, int(iterations))):
        particle_delta_accumulator.zero_()
        particle_delta_counter.zero_()
        wp.launch(
            kernel=project_kinematic_sphere_mc_triangle_node_positions_kernel,
            dim=triangle_count,
            inputs=[
                particle_inv_mass,
                particle_flags,
                cell_nodes,
                cell_active,
                vertex_pos,
                tri_indices,
                triangle_count,
                sphere_q_prev,
                sphere_q,
                sphere_count,
                sphere_radius,
                float(interpolation_alpha),
                float(relaxation),
                float(max_correction),
            ],
            outputs=[particle_delta_accumulator, particle_delta_counter],
            device=launch_device,
        )
        wp.launch(
            kernel=apply_kinematic_sphere_mc_triangle_node_deltas_kernel,
            dim=int(particle_q.shape[0]),
            inputs=[
                particle_q,
                particle_inv_mass,
                particle_flags,
                particle_delta_accumulator,
                particle_delta_counter,
            ],
            device=launch_device,
        )


def apply_kinematic_sphere_particle_contacts(
    state,
    model,
    sphere_q_prev: wp.array,
    sphere_q: wp.array,
    sphere_count: int,
    sphere_radius: float,
    interpolation_alpha: float,
    dt: float,
    relaxation: float,
    device,
    iterations: int = 1,
    max_correction: float = 0.0,
    velocity_relaxation: float = 0.0,
) -> None:
    del dt, velocity_relaxation
    project_kinematic_sphere_particle_positions(
        state.particle_q,
        model,
        sphere_q_prev,
        sphere_q,
        sphere_count,
        sphere_radius,
        interpolation_alpha,
        relaxation,
        iterations=iterations,
        max_correction=max_correction,
        device=device,
    )


__all__ = [
    "apply_kinematic_sphere_mc_triangle_node_deltas_kernel",
    "apply_kinematic_sphere_particle_contacts",
    "project_kinematic_sphere_mc_triangle_node_positions",
    "project_kinematic_sphere_mc_triangle_node_positions_kernel",
    "project_kinematic_sphere_particle_positions",
    "project_kinematic_sphere_particle_positions_kernel",
]
