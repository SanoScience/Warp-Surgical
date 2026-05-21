# SPDX-License-Identifier: Apache-2.0
"""Sparse GPU updates for springless hex-grid cell deletion."""

from __future__ import annotations

import warp as wp
from newton._src.geometry.flags import ParticleFlags

_ACTIVE_BIT = wp.constant(wp.int32(int(ParticleFlags.ACTIVE)))


@wp.kernel
def delete_cells_sparse_kernel(
    cell_ids: wp.array(dtype=wp.int32),
    candidate_capacity: int,
    candidate_count: wp.array(dtype=wp.int32),
    num_cells: int,
    cell_nodes: wp.array2d(dtype=wp.int32),
    cell_mass: wp.array(dtype=float),
    cell_active: wp.array(dtype=wp.int32),
    node_support_count: wp.array(dtype=wp.int32),
    node_mass: wp.array(dtype=float),
    deleted_cells: wp.array(dtype=wp.int32),
    deleted_count: wp.array(dtype=wp.int32),
    deleted_total: wp.array(dtype=wp.int32),
):
    """Deactivate each candidate cell once and decrement owned supports.

    ``atomic_cas(cell_active, c, 1, 0)`` makes duplicate ids and already
    deleted cells no-ops. This kernel only mutates primary support state; the
    derived node flags and inverse masses are refreshed by the follow-up kernel
    after all atomics are complete.
    """
    tid = wp.tid()
    if tid >= candidate_capacity or tid >= candidate_count[0]:
        return

    cell_idx = cell_ids[tid]
    if cell_idx < 0 or cell_idx >= num_cells:
        return

    old_active = wp.atomic_cas(cell_active, cell_idx, 1, 0)
    if old_active != 1:
        return

    out_idx = wp.atomic_add(deleted_count, 0, 1)
    if out_idx < candidate_capacity:
        deleted_cells[out_idx] = cell_idx
    wp.atomic_add(deleted_total, 0, 1)

    cell_node_mass = cell_mass[cell_idx] * (1.0 / 8.0)
    for local_node in range(8):
        node_idx = cell_nodes[cell_idx, local_node]
        wp.atomic_sub(node_support_count, node_idx, 1)
        wp.atomic_add(node_mass, node_idx, -cell_node_mass)


@wp.kernel
def delete_single_cell_complete_kernel(
    cell_ids: wp.array(dtype=wp.int32),
    num_cells: int,
    cell_nodes: wp.array2d(dtype=wp.int32),
    cell_mass: wp.array(dtype=float),
    cell_active: wp.array(dtype=wp.int32),
    node_support_count: wp.array(dtype=wp.int32),
    node_mass: wp.array(dtype=float),
    locked_node_mask: wp.array(dtype=wp.int32),
    particle_mass: wp.array(dtype=float),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    deleted_cells: wp.array(dtype=wp.int32),
    deleted_count: wp.array(dtype=wp.int32),
    deleted_total: wp.array(dtype=wp.int32),
):
    """Specialized one-cell delete that updates primary and derived state."""
    cell_idx = cell_ids[0]
    if cell_idx < 0 or cell_idx >= num_cells:
        return

    old_active = wp.atomic_cas(cell_active, cell_idx, 1, 0)
    if old_active != 1:
        return

    deleted_cells[0] = cell_idx
    deleted_count[0] = 1
    wp.atomic_add(deleted_total, 0, 1)

    cell_node_mass = cell_mass[cell_idx] * (1.0 / 8.0)
    for local_node in range(8):
        node_idx = cell_nodes[cell_idx, local_node]
        wp.atomic_sub(node_support_count, node_idx, 1)
        wp.atomic_add(node_mass, node_idx, -cell_node_mass)

        support = node_support_count[node_idx]
        mass = node_mass[node_idx]
        flags = particle_flags[node_idx]
        if support > 0:
            particle_flags[node_idx] = flags | _ACTIVE_BIT
            if locked_node_mask[node_idx] != 0:
                particle_mass[node_idx] = 0.0
                particle_inv_mass[node_idx] = 0.0
            elif mass > 0.0:
                particle_mass[node_idx] = mass
                particle_inv_mass[node_idx] = 1.0 / mass
            else:
                particle_mass[node_idx] = 0.0
                particle_inv_mass[node_idx] = 0.0
        else:
            node_mass[node_idx] = 0.0
            locked_node_mask[node_idx] = 0
            particle_mass[node_idx] = 0.0
            particle_inv_mass[node_idx] = 0.0
            particle_flags[node_idx] = flags & (~_ACTIVE_BIT)


@wp.kernel
def finalize_deleted_cell_nodes_kernel(
    deleted_cells: wp.array(dtype=wp.int32),
    deleted_count: wp.array(dtype=wp.int32),
    num_cells: int,
    cell_nodes: wp.array2d(dtype=wp.int32),
    node_support_count: wp.array(dtype=wp.int32),
    node_mass: wp.array(dtype=float),
    locked_node_mask: wp.array(dtype=wp.int32),
    particle_mass: wp.array(dtype=float),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
):
    """Refresh derived node state for particles touched by deleted cells."""
    cell_slot, local_node = wp.tid()
    if cell_slot >= deleted_count[0]:
        return

    cell_idx = deleted_cells[cell_slot]
    if cell_idx < 0 or cell_idx >= num_cells:
        return

    node_idx = cell_nodes[cell_idx, local_node]
    support = node_support_count[node_idx]
    mass = node_mass[node_idx]
    flags = particle_flags[node_idx]

    if support > 0:
        particle_flags[node_idx] = flags | _ACTIVE_BIT
        if locked_node_mask[node_idx] != 0:
            particle_mass[node_idx] = 0.0
            particle_inv_mass[node_idx] = 0.0
        elif mass > 0.0:
            particle_mass[node_idx] = mass
            particle_inv_mass[node_idx] = 1.0 / mass
        else:
            particle_mass[node_idx] = 0.0
            particle_inv_mass[node_idx] = 0.0
    else:
        node_mass[node_idx] = 0.0
        locked_node_mask[node_idx] = 0
        particle_mass[node_idx] = 0.0
        particle_inv_mass[node_idx] = 0.0
        particle_flags[node_idx] = flags & (~_ACTIVE_BIT)

@wp.kernel
def deactivate_single_deleted_cell_cluster_kernel(
    deleted_cells: wp.array(dtype=wp.int32),
    deleted_count: wp.array(dtype=wp.int32),
    num_cells: int,
    cell_to_cluster: wp.array(dtype=wp.int32),
    cluster_active: wp.array(dtype=wp.int32),
    cluster_offsets: wp.array(dtype=wp.int32),
    cluster_indices: wp.array(dtype=wp.int32),
    particle_cluster_counts: wp.array(dtype=wp.int32),
    particle_cluster_inv_weights: wp.array(dtype=float),
):
    """Deactivate one deleted cell's cluster and refresh touched inverse weights."""
    if deleted_count[0] <= 0:
        return

    cell_idx = deleted_cells[0]
    if cell_idx < 0 or cell_idx >= num_cells:
        return

    cluster_idx = cell_to_cluster[cell_idx]
    if cluster_idx < 0:
        return

    old_active = wp.atomic_cas(cluster_active, cluster_idx, 1, 0)
    if old_active != 1:
        return

    cursor = cluster_offsets[cluster_idx]
    end = cluster_offsets[cluster_idx + 1]
    while cursor < end:
        particle_idx = cluster_indices[cursor]
        if particle_idx >= 0:
            wp.atomic_sub(particle_cluster_counts, particle_idx, 1)
            count = particle_cluster_counts[particle_idx]
            if count > 0:
                particle_cluster_inv_weights[particle_idx] = 1.0 / float(count)
            else:
                particle_cluster_counts[particle_idx] = 0
                particle_cluster_inv_weights[particle_idx] = 0.0
        cursor += 1


@wp.kernel
def deactivate_deleted_cell_clusters_kernel(
    deleted_cells: wp.array(dtype=wp.int32),
    deleted_count: wp.array(dtype=wp.int32),
    num_cells: int,
    cell_to_cluster: wp.array(dtype=wp.int32),
    cluster_active: wp.array(dtype=wp.int32),
    cluster_offsets: wp.array(dtype=wp.int32),
    cluster_indices: wp.array(dtype=wp.int32),
    particle_cluster_counts: wp.array(dtype=wp.int32),
    deactivated_cluster_ids: wp.array(dtype=wp.int32),
    deactivated_count: wp.array(dtype=wp.int32),
):
    """Deactivate parent clusters touched by deleted cells and decrement members once."""
    i = wp.tid()
    if i >= deleted_count[0]:
        return

    cell_idx = deleted_cells[i]
    if cell_idx < 0 or cell_idx >= num_cells:
        return

    cluster_idx = cell_to_cluster[cell_idx]
    if cluster_idx < 0:
        return

    old_active = wp.atomic_cas(cluster_active, cluster_idx, 1, 0)
    if old_active != 1:
        return

    out_idx = wp.atomic_add(deactivated_count, 0, 1)
    if out_idx < deactivated_cluster_ids.shape[0]:
        deactivated_cluster_ids[out_idx] = cluster_idx

    cursor = cluster_offsets[cluster_idx]
    end = cluster_offsets[cluster_idx + 1]
    while cursor < end:
        particle_idx = cluster_indices[cursor]
        if particle_idx >= 0:
            wp.atomic_sub(particle_cluster_counts, particle_idx, 1)
        cursor += 1


@wp.kernel
def finalize_deactivated_cluster_weights_kernel(
    deactivated_cluster_ids: wp.array(dtype=wp.int32),
    deactivated_count: wp.array(dtype=wp.int32),
    cluster_offsets: wp.array(dtype=wp.int32),
    cluster_indices: wp.array(dtype=wp.int32),
    particle_cluster_counts: wp.array(dtype=wp.int32),
    particle_cluster_inv_weights: wp.array(dtype=float),
):
    """Refresh inverse membership weights for particles touched by deactivated clusters."""
    i = wp.tid()
    if i >= deactivated_count[0]:
        return

    cluster_idx = deactivated_cluster_ids[i]
    if cluster_idx < 0:
        return

    cursor = cluster_offsets[cluster_idx]
    end = cluster_offsets[cluster_idx + 1]
    while cursor < end:
        particle_idx = cluster_indices[cursor]
        if particle_idx >= 0:
            count = particle_cluster_counts[particle_idx]
            if count > 0:
                particle_cluster_inv_weights[particle_idx] = 1.0 / float(count)
            else:
                particle_cluster_counts[particle_idx] = 0
                particle_cluster_inv_weights[particle_idx] = 0.0
        cursor += 1


@wp.kernel
def validate_cell_active_kernel(
    cell_active: wp.array(dtype=wp.int32),
    error_count: wp.array(dtype=wp.int32),
    first_error: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    value = cell_active[i]
    if value != 0 and value != 1:
        old = wp.atomic_add(error_count, 0, 1)
        if old == 0:
            first_error[0] = 1000 + i


@wp.kernel
def validate_node_state_kernel(
    node_support_count: wp.array(dtype=wp.int32),
    node_mass: wp.array(dtype=float),
    locked_node_mask: wp.array(dtype=wp.int32),
    particle_mass: wp.array(dtype=float),
    particle_inv_mass: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    error_count: wp.array(dtype=wp.int32),
    first_error: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    support = node_support_count[i]
    mass = node_mass[i]
    effective_mass = particle_mass[i]
    inv_mass = particle_inv_mass[i]
    flags = particle_flags[i]
    locked = locked_node_mask[i]

    bad = int(0)
    code = int(2000 + i)
    if support < 0:
        bad = 1
    if mass < -1.0e-5 or effective_mass < -1.0e-5 or inv_mass < -1.0e-5:
        bad = 1
    if mass != mass or effective_mass != effective_mass or inv_mass != inv_mass:
        bad = 1
    if support <= 0 and (flags & _ACTIVE_BIT) != 0:
        bad = 1
    if support > 0 and (flags & _ACTIVE_BIT) == 0:
        bad = 1
    if locked != 0 and inv_mass != 0.0:
        bad = 1

    if bad != 0:
        old = wp.atomic_add(error_count, 0, 1)
        if old == 0:
            first_error[0] = code

@wp.kernel
def validate_cluster_state_kernel(
    cluster_active: wp.array(dtype=wp.int32),
    particle_cluster_counts: wp.array(dtype=wp.int32),
    particle_cluster_inv_weights: wp.array(dtype=float),
    error_count: wp.array(dtype=wp.int32),
    first_error: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    bad = int(0)

    if i < particle_cluster_counts.shape[0]:
        count = particle_cluster_counts[i]
        inv_weight = particle_cluster_inv_weights[i]
        if count < 0:
            bad = 1
        if inv_weight < -1.0e-6 or inv_weight != inv_weight:
            bad = 1
        if count <= 0 and inv_weight != 0.0:
            bad = 1

    if i < cluster_active.shape[0]:
        active = cluster_active[i]
        if active != 0 and active != 1:
            bad = 1

    if bad != 0:
        old = wp.atomic_add(error_count, 0, 1)
        if old == 0:
            first_error[0] = 4000 + i


@wp.func
def _point_segment_distance_sq(p: wp.vec3, a: wp.vec3, b: wp.vec3):
    ab = b - a
    denom = wp.dot(ab, ab)
    if denom <= 1.0e-20:
        d = p - a
        return wp.dot(d, d)

    t = wp.dot(p - a, ab) / denom
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    q = a + ab * t
    d = p - q
    return wp.dot(d, d)


@wp.func
def _point_triangle_distance_sq(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    ab = b - a
    ac = c - a
    n = wp.cross(ab, ac)
    if wp.dot(n, n) <= 1.0e-20:
        d0 = _point_segment_distance_sq(p, a, b)
        d1 = _point_segment_distance_sq(p, b, c)
        d2 = _point_segment_distance_sq(p, c, a)
        best_dist = d0
        if d1 < best_dist:
            best_dist = d1
        if d2 < best_dist:
            best_dist = d2
        return best_dist

    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        delta_a = p - a
        return wp.dot(delta_a, delta_a)

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        delta_b = p - b
        return wp.dot(delta_b, delta_b)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        q = a + ab * v
        delta_ab = p - q
        return wp.dot(delta_ab, delta_ab)

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        delta_c = p - c
        return wp.dot(delta_c, delta_c)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        q = a + ac * w
        delta_ac = p - q
        return wp.dot(delta_ac, delta_ac)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        q = b + (c - b) * w
        delta_bc = p - q
        return wp.dot(delta_bc, delta_bc)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    q = a + ab * v + ac * w
    delta_face = p - q
    return wp.dot(delta_face, delta_face)


@wp.kernel
def select_ray_surface_cells_kernel(
    num_cells: int,
    cell_nodes: wp.array2d(dtype=wp.int32),
    cell_material: wp.array(dtype=wp.int32),
    cell_active: wp.array(dtype=wp.int32),
    material_cuttable: wp.array(dtype=wp.int32),
    has_material_filter: int,
    particle_q: wp.array(dtype=wp.vec3),
    ray0_origin: wp.vec3,
    ray0_end: wp.vec3,
    ray1_origin: wp.vec3,
    ray1_end: wp.vec3,
    padding: float,
    selected_cells: wp.array(dtype=wp.int32),
    selected_count: wp.array(dtype=wp.int32),
):
    """Append active cells whose deformed centres intersect the two-ray blade surface."""
    c = wp.tid()
    if c >= num_cells:
        return
    if cell_active[c] == 0:
        return
    if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:
        return

    centre = wp.vec3(0.0, 0.0, 0.0)
    for local_node in range(8):
        centre += particle_q[cell_nodes[c, local_node]]
    centre *= 0.125

    d0 = _point_triangle_distance_sq(centre, ray0_origin, ray1_origin, ray1_end)
    d1 = _point_triangle_distance_sq(centre, ray0_origin, ray1_end, ray0_end)
    dist_sq = d0
    if d1 < dist_sq:
        dist_sq = d1
    if dist_sq > padding * padding:
        return

    out_idx = wp.atomic_add(selected_count, 0, 1)
    selected_cells[out_idx] = c


@wp.kernel
def select_ray_segment_cells_kernel(
    num_cells: int,
    cell_nodes: wp.array2d(dtype=wp.int32),
    cell_material: wp.array(dtype=wp.int32),
    cell_active: wp.array(dtype=wp.int32),
    material_cuttable: wp.array(dtype=wp.int32),
    has_material_filter: int,
    particle_q: wp.array(dtype=wp.vec3),
    start_cell_ids: wp.array(dtype=wp.int32),
    ray_dir: wp.vec3,
    depth: float,
    padding: float,
    selected_cells: wp.array(dtype=wp.int32),
    selected_count: wp.array(dtype=wp.int32),
):
    """Append active cells whose deformed centres lie along a picked ray segment."""
    c = wp.tid()
    if c >= num_cells:
        return

    start_cell = start_cell_ids[0]
    if start_cell < 0 or start_cell >= num_cells:
        return
    if cell_active[start_cell] == 0:
        return
    if cell_active[c] == 0:
        return
    if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:
        return

    start = wp.vec3(0.0, 0.0, 0.0)
    centre = wp.vec3(0.0, 0.0, 0.0)
    for local_node in range(8):
        start += particle_q[cell_nodes[start_cell, local_node]]
        centre += particle_q[cell_nodes[c, local_node]]
    start *= 0.125
    centre *= 0.125

    end = start + ray_dir * depth
    dist_sq = _point_segment_distance_sq(centre, start, end)
    if dist_sq > padding * padding:
        return

    out_idx = wp.atomic_add(selected_count, 0, 1)
    selected_cells[out_idx] = c


@wp.kernel
def select_sphere_particle_contact_cells_kernel(
    num_cells: int,
    cell_nodes: wp.array2d(dtype=wp.int32),
    cell_material: wp.array(dtype=wp.int32),
    cell_active: wp.array(dtype=wp.int32),
    material_cuttable: wp.array(dtype=wp.int32),
    has_material_filter: int,
    particle_q: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    sphere_q: wp.array(dtype=wp.vec3),
    sphere_cut_enabled: wp.array(dtype=wp.int32),
    sphere_count: int,
    sphere_radius: float,
    selected_cells: wp.array(dtype=wp.int32),
    selected_count: wp.array(dtype=wp.int32),
):
    """Append active cells whose particles intersect an enabled sphere."""
    c = wp.tid()
    if c >= num_cells:
        return
    if cell_active[c] == 0:
        return
    if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:
        return

    for local_node in range(8):
        node_idx = cell_nodes[c, local_node]
        if (particle_flags[node_idx] & _ACTIVE_BIT) == 0:
            continue
        q = particle_q[node_idx]
        for sphere_idx in range(sphere_count):
            if sphere_cut_enabled[sphere_idx] == 0:
                continue
            radius = sphere_radius + particle_radius[node_idx]
            delta = q - sphere_q[sphere_idx]
            if wp.dot(delta, delta) <= radius * radius:
                out_idx = wp.atomic_add(selected_count, 0, 1)
                selected_cells[out_idx] = c
                return


__all__ = [
    "deactivate_deleted_cell_clusters_kernel",
    "deactivate_single_deleted_cell_cluster_kernel",
    "delete_cells_sparse_kernel",
    "delete_single_cell_complete_kernel",
    "finalize_deactivated_cluster_weights_kernel",
    "finalize_deleted_cell_nodes_kernel",
    "select_ray_segment_cells_kernel",
    "select_ray_surface_cells_kernel",
    "select_sphere_particle_contact_cells_kernel",
    "validate_cell_active_kernel",
    "validate_cluster_state_kernel",
    "validate_node_state_kernel",
]
