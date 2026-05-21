# SPDX-License-Identifier: Apache-2.0
"""Cell-centred heat and instrument selection kernels for corner grids."""

from __future__ import annotations

import warp as wp
from newton._src.geometry.flags import ParticleFlags

_ACTIVE_BIT = wp.constant(wp.int32(int(ParticleFlags.ACTIVE)))
_BURN_HEAT_THRESHOLD = wp.constant(15.0)
_BURN_SLOPE = wp.constant(0.04)
_BURN_BIAS = wp.constant(0.05)


@wp.func
def _point_segment_distance_sq(p: wp.vec3, a: wp.vec3, b: wp.vec3) -> float:
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
def _cell_deformed_center(
    cell_nodes: wp.array2d(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    cell_idx: int,
) -> wp.vec3:
    centre = wp.vec3(0.0, 0.0, 0.0)
    for local_node in range(8):
        centre += particle_q[cell_nodes[cell_idx, local_node]]
    return centre * 0.125


@wp.func
def _cell_intersects_sphere(
    cell_nodes: wp.array2d(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    cell_idx: int,
    sphere_centre: wp.vec3,
    radius: float,
) -> int:
    r2 = radius * radius
    centre = _cell_deformed_center(cell_nodes, particle_q, cell_idx)
    delta = centre - sphere_centre
    if wp.dot(delta, delta) <= r2:
        return 1
    for local_node in range(8):
        node_idx = cell_nodes[cell_idx, local_node]
        corner_delta = particle_q[node_idx] - sphere_centre
        if wp.dot(corner_delta, corner_delta) <= r2:
            return 1
    return 0


@wp.func
def _cell_intersects_capsule(
    cell_nodes: wp.array2d(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    cell_idx: int,
    capsule_p0: wp.vec3,
    capsule_p1: wp.vec3,
    radius: float,
) -> int:
    r2 = radius * radius
    centre = _cell_deformed_center(cell_nodes, particle_q, cell_idx)
    if _point_segment_distance_sq(centre, capsule_p0, capsule_p1) <= r2:
        return 1
    for local_node in range(8):
        node_idx = cell_nodes[cell_idx, local_node]
        if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:
            return 1
    return 0


@wp.kernel
def apply_diathermy_spheres_kernel(
    num_cells: int,
    cell_nodes: wp.array2d(dtype=wp.int32),
    cell_material: wp.array(dtype=wp.int32),
    cell_active: wp.array(dtype=wp.int32),
    material_cuttable: wp.array(dtype=wp.int32),
    has_material_filter: int,
    particle_q: wp.array(dtype=wp.vec3),
    sphere_q: wp.array(dtype=wp.vec3),
    sphere_enabled: wp.array(dtype=wp.int32),
    sphere_power: wp.array(dtype=wp.float32),
    has_sphere_power: int,
    sphere_count: int,
    sphere_radius: float,
    heat_delta: float,
    cell_heat: wp.array(dtype=wp.float32),
):
    """Add heat to active cuttable cells touched by enabled instrument spheres."""
    c = wp.tid()
    if c >= num_cells:
        return
    if cell_active[c] == 0:
        return
    if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:
        return
    if sphere_radius <= 0.0 or heat_delta <= 0.0:
        return

    total_delta = float(0.0)
    for sphere_idx in range(sphere_count):
        if sphere_enabled[sphere_idx] == 0:
            continue
        if _cell_intersects_sphere(cell_nodes, particle_q, c, sphere_q[sphere_idx], sphere_radius) != 0:
            delta = heat_delta
            if has_sphere_power != 0:
                delta = heat_delta * sphere_power[sphere_idx]
            if delta > 0.0:
                total_delta += delta
    if total_delta > 0.0:
        cell_heat[c] = cell_heat[c] + total_delta


@wp.kernel
def diffuse_cell_heat_kernel(
    num_cells: int,
    cell_grid_xyz: wp.array2d(dtype=wp.int32),
    cell_material: wp.array(dtype=wp.int32),
    cell_active: wp.array(dtype=wp.int32),
    grid_to_cell: wp.array3d(dtype=wp.int32),
    material_conductivity: wp.array(dtype=wp.float32),
    heat_in: wp.array(dtype=wp.float32),
    dt: float,
    diffusion: float,
    cooling: float,
    grid_nx: int,
    grid_ny: int,
    grid_nz: int,
    heat_out: wp.array(dtype=wp.float32),
):
    """One 6-neighbour Jacobi heat diffusion sweep over active cells."""
    c = wp.tid()
    if c >= num_cells:
        return
    if cell_active[c] == 0:
        heat_out[c] = 0.0
        return

    h = heat_in[c]
    material_c = cell_material[c]
    cond_c = material_conductivity[material_c]
    gx = cell_grid_xyz[c, 0]
    gy = cell_grid_xyz[c, 1]
    gz = cell_grid_xyz[c, 2]
    accum = float(0.0)

    for direction in range(6):
        nx = gx
        ny = gy
        nz = gz
        if direction == 0:
            nx = gx - 1
        elif direction == 1:
            nx = gx + 1
        elif direction == 2:
            ny = gy - 1
        elif direction == 3:
            ny = gy + 1
        elif direction == 4:
            nz = gz - 1
        else:
            nz = gz + 1

        if nx < 0 or nx >= grid_nx or ny < 0 or ny >= grid_ny or nz < 0 or nz >= grid_nz:
            continue
        nb = grid_to_cell[nx, ny, nz]
        if nb < 0:
            continue
        if cell_active[nb] == 0:
            continue
        cond_nb = material_conductivity[cell_material[nb]]
        accum += (heat_in[nb] - h) * 0.5 * (cond_c + cond_nb)

    out_h = h + float(dt) * float(diffusion) * accum
    out_h = out_h - float(dt) * float(cooling) * h
    if out_h < 0.0:
        out_h = 0.0
    heat_out[c] = out_h


@wp.kernel
def select_overheated_cells_kernel(
    num_cells: int,
    cell_material: wp.array(dtype=wp.int32),
    cell_active: wp.array(dtype=wp.int32),
    material_resistance: wp.array(dtype=wp.float32),
    material_cuttable: wp.array(dtype=wp.int32),
    has_material_filter: int,
    cell_heat: wp.array(dtype=wp.float32),
    fulguration: float,
    cell_burn: wp.array(dtype=wp.float32),
    selected_cells: wp.array(dtype=wp.int32),
    selected_count: wp.array(dtype=wp.int32),
):
    """Append active cuttable cells whose heat exceeds material resistance."""
    c = wp.tid()
    if c >= num_cells:
        return
    if cell_active[c] == 0:
        return

    h = cell_heat[c]
    if h > _BURN_HEAT_THRESHOLD:
        b = cell_burn[c] + (_BURN_SLOPE * (h - _BURN_HEAT_THRESHOLD) + _BURN_BIAS) * fulguration
        if b > 1.0:
            b = 1.0
        cell_burn[c] = b

    material = cell_material[c]
    resistance = material_resistance[material]
    if resistance <= 0.0 or h <= resistance:
        return
    if has_material_filter != 0 and material_cuttable[material] == 0:
        return

    out_idx = wp.atomic_add(selected_count, 0, 1)
    selected_cells[out_idx] = c


@wp.kernel
def reset_deleted_cell_heat_kernel(
    num_cells: int,
    cell_active: wp.array(dtype=wp.int32),
    cell_heat_a: wp.array(dtype=wp.float32),
    cell_heat_b: wp.array(dtype=wp.float32),
    cell_burn: wp.array(dtype=wp.float32),
):
    c = wp.tid()
    if c >= num_cells:
        return
    if cell_active[c] != 0:
        return
    cell_heat_a[c] = 0.0
    cell_heat_b[c] = 0.0
    cell_burn[c] = 0.0


@wp.kernel
def select_capsule_cells_kernel(
    num_cells: int,
    cell_nodes: wp.array2d(dtype=wp.int32),
    cell_material: wp.array(dtype=wp.int32),
    cell_active: wp.array(dtype=wp.int32),
    material_cuttable: wp.array(dtype=wp.int32),
    has_material_filter: int,
    particle_q: wp.array(dtype=wp.vec3),
    capsule_p0: wp.vec3,
    capsule_p1: wp.vec3,
    radius: float,
    selected_cells: wp.array(dtype=wp.int32),
    selected_count: wp.array(dtype=wp.int32),
):
    """Append active cuttable cells whose centre or corners intersect a capsule."""
    c = wp.tid()
    if c >= num_cells:
        return
    if cell_active[c] == 0:
        return
    if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:
        return
    if radius < 0.0:
        return

    if _cell_intersects_capsule(cell_nodes, particle_q, c, capsule_p0, capsule_p1, radius) == 0:
        return
    out_idx = wp.atomic_add(selected_count, 0, 1)
    selected_cells[out_idx] = c


@wp.kernel
def gather_heat_overlay_kernel(
    num_cells: int,
    cell_center_q: wp.array(dtype=wp.vec3),
    cell_active: wp.array(dtype=wp.int32),
    cell_heat: wp.array(dtype=wp.float32),
    max_heat: float,
    min_visible_heat: float,
    out_points: wp.array(dtype=wp.vec3),
    out_colors: wp.array(dtype=wp.vec3),
    out_count: wp.array(dtype=wp.int32),
):
    """Gather active heated cells as coloured debug points."""
    c = wp.tid()
    if c >= num_cells:
        return
    if cell_active[c] == 0:
        return
    h = cell_heat[c]
    if h <= min_visible_heat:
        return

    denom = max_heat
    if denom < 1.0e-6:
        denom = 1.0e-6
    t = h / denom
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    out_idx = wp.atomic_add(out_count, 0, 1)
    out_points[out_idx] = cell_center_q[c]
    out_colors[out_idx] = wp.vec3(0.10 + 0.90 * t, 0.12 + 0.45 * (1.0 - t), 1.0 - t)


__all__ = [
    "apply_diathermy_spheres_kernel",
    "diffuse_cell_heat_kernel",
    "gather_heat_overlay_kernel",
    "reset_deleted_cell_heat_kernel",
    "select_capsule_cells_kernel",
    "select_overheated_cells_kernel",
]
