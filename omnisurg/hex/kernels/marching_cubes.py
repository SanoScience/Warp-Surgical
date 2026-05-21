# SPDX-License-Identifier: Apache-2.0
"""Deformable marching cubes anchored to oriented particles (paper §3.2).

The reference C++ lives in ``ufrgs/marchingCubes.h`` + ``SpringGrid::computeCubes``
/ ``computeMesh`` (springGrid.h:780-873). The key property the paper calls out
(Figure 3b-c): because each vertex is a fixed offset from a specific
(particle, direction) slot rather than an interpolation along a moving cube
edge, the triangulation case for a cell does *not* change as the particles
deform - only the vertex positions translate. This prevents the "case flip"
visual artefact that static-grid MC suffers when deformable fields cross
edges. Cut-induced topology changes are the only thing that re-triangulate.

Pipeline per frame:

1. :func:`compute_cube_cases_kernel` - one thread per cube cell; packs the
   8 corner active flags into an 8-bit case index.
2. :func:`compute_vertex_positions_kernel` - one thread per ``(particle, dir)``
   slot; position = particle_q + orientation[axis] * sign * mc_factor.
3. :func:`emit_triangles_kernel` - one thread per cube cell; reads the case
   table, resolves each edge's vertex to its ``(particle, direction)`` id,
   atomically appends triangle indices.

Vertex ids are dense: ``vertex_id(particle, dir) = particle * 6 + dir``.
Unused (particle, dir) slots still have a position written (cheap) but are
never referenced by any triangle.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp
from newton._src.geometry.flags import ParticleFlags

from . import _mc_tables_gen as _tables

# Keep the module-level constants aligned with the table layout. Precomputing
# them here lets the kernels read fixed strides without extra branches.
_NUM_CASES = 256
_NUM_EDGES = 12
_NUM_DIRS_PER_PARTICLE = 6

# How many triangles can a single cube emit at most. Standard MC tops out at 5.
MC_MAX_TRIS_PER_CASE: int = int(_tables.case_triangle_counts().max())
assert MC_MAX_TRIS_PER_CASE == 5


@dataclass
class MarchingCubesTables:
    """Per-device upload of the paper's MC lookup tables.

    The fields are ``wp.array``\\ s ready to be passed into the MC kernels.
    Build once per device via :func:`upload_mc_tables` and reuse.
    """

    case_triangles: wp.array  # int32[256, 16] - edge indices, -1 terminator
    edge_corners: wp.array  # int32[12, 2] - corner id pair per edge
    edge_base_dir: wp.array  # int32[12] - direction when "a" corner is active
    corner_offsets: wp.array  # int32[8, 3] - grid offsets for the 8 corners


def upload_mc_tables(device: str | wp.context.Device | None = None) -> MarchingCubesTables:
    """Upload the reference MC tables to ``device`` as ``wp.array``\\ s."""
    return MarchingCubesTables(
        case_triangles=wp.array(_tables.CASE_TRIANGLES, dtype=wp.int32, device=device),
        edge_corners=wp.array(_tables.EDGE_CORNERS, dtype=wp.int32, device=device),
        edge_base_dir=wp.array(_tables.EDGE_BASE_DIR, dtype=wp.int32, device=device),
        corner_offsets=wp.array(_tables.CORNER_OFFSETS, dtype=wp.int32, device=device),
    )


@wp.func
def _corner_particle(
    grid_to_particle: wp.array3d(dtype=wp.int32),
    corner_offsets: wp.array2d(dtype=wp.int32),
    cx: int, cy: int, cz: int, c: int,
) -> int:
    ox = corner_offsets[c, 0]
    oy = corner_offsets[c, 1]
    oz = corner_offsets[c, 2]
    return grid_to_particle[cx + ox, cy + oy, cz + oz]


@wp.func
def _corner_active(
    grid_to_particle: wp.array3d(dtype=wp.int32),
    particle_flags: wp.array(dtype=wp.int32),
    corner_offsets: wp.array2d(dtype=wp.int32),
    cx: int, cy: int, cz: int, c: int,
) -> int:
    p = _corner_particle(grid_to_particle, corner_offsets, cx, cy, cz, c)
    if p < 0:
        return 0
    if (particle_flags[p] & wp.int32(ParticleFlags.ACTIVE)) == 0:
        return 0
    return 1


@wp.kernel
def compute_visible_flags_kernel(
    particle_flags: wp.array(dtype=wp.int32),
    particle_material: wp.array(dtype=wp.int32),
    material_visible: wp.array(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    cut_z: float,
    visible_flags: wp.array(dtype=wp.int32),
):
    """Combine ACTIVE + material visibility + world-Z cut into MC flags.

    ``cut_z`` is a world-space altitude above which particles are treated as
    inactive for the MC pipeline (handy for slicing open the anatomy to
    see what's underneath). Pass +inf to disable the cut.
    """
    i = wp.tid()
    f = particle_flags[i]
    if material_visible[particle_material[i]] == 0:
        f = f & (~wp.int32(ParticleFlags.ACTIVE))
    if particle_q[i][2] > cut_z:
        f = f & (~wp.int32(ParticleFlags.ACTIVE))
    visible_flags[i] = f


@wp.kernel
def compute_cube_cases_kernel(
    grid_to_particle: wp.array3d(dtype=wp.int32),
    particle_flags: wp.array(dtype=wp.int32),
    corner_offsets: wp.array2d(dtype=wp.int32),
    cube_cases: wp.array3d(dtype=wp.int32),
):
    """Pack 8-corner active flags into an 8-bit case index per cube cell.

    ``cube_cases.shape == (nx-1, ny-1, nz-1)``. A value of 0 means the cell is
    entirely outside tissue, 255 means entirely inside (no triangles emitted in
    either case).
    """
    cx, cy, cz = wp.tid()
    mask = int(0)
    for c in range(8):
        if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:
            mask = mask | (int(1) << c)
    cube_cases[cx, cy, cz] = mask


@wp.func
def _compute_cube_case(
    grid_to_particle: wp.array3d(dtype=wp.int32),
    particle_flags: wp.array(dtype=wp.int32),
    corner_offsets: wp.array2d(dtype=wp.int32),
    cx: int,
    cy: int,
    cz: int,
) -> int:
    mask = int(0)
    for c in range(8):
        if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:
            mask = mask | (int(1) << c)
    return mask


@wp.func
def _cube_flat_id(cx: int, cy: int, cz: int, ny_cells: int, nz_cells: int) -> int:
    return (cx * ny_cells + cy) * nz_cells + cz


@wp.func
def _clear_fixed_slot(
    slot: int,
    slot_tri_indices: wp.array2d(dtype=wp.int32),
    slot_active: wp.array(dtype=wp.int32),
    slot_to_compact: wp.array(dtype=wp.int32),
):
    slot_active[slot] = 0
    slot_to_compact[slot] = -1
    slot_tri_indices[slot, 0] = 0
    slot_tri_indices[slot, 1] = 0
    slot_tri_indices[slot, 2] = 0


@wp.kernel
def bake_vertex_uv3_kernel(
    particle_grid_xyz: wp.array2d(dtype=wp.int32),
    inv_grid_nx: float,
    inv_grid_ny: float,
    inv_grid_nz: float,
    particle_uv3: wp.array(dtype=wp.vec3),
    vertex_uv3: wp.array(dtype=wp.vec3),
):
    """One-shot UV3 bake: one thread per particle, writes all 6 direction slots.

    UV3 depends only on ``particle_grid_xyz`` (static after lattice build) and
    the grid extents, so this can run once at setup and be reused every frame.
    ``particle_uv3`` stores the cell-centre coordinate for categorical material
    lookups. ``vertex_uv3`` stores the directional surface vertex coordinate,
    matching ``compute_vertex_positions_kernel``'s half-voxel MC offsets.
    """
    p = wp.tid()
    gx = float(particle_grid_xyz[p, 0])
    gy = float(particle_grid_xyz[p, 1])
    gz = float(particle_grid_xyz[p, 2])
    centre = wp.vec3(
        gx * inv_grid_nx,
        gy * inv_grid_ny,
        gz * inv_grid_nz,
    )
    particle_uv3[p] = centre
    base = p * 6
    vertex_uv3[base + 0] = wp.vec3((gx - 0.5) * inv_grid_nx, centre[1], centre[2])
    vertex_uv3[base + 1] = wp.vec3((gx + 0.5) * inv_grid_nx, centre[1], centre[2])
    vertex_uv3[base + 2] = wp.vec3(centre[0], (gy - 0.5) * inv_grid_ny, centre[2])
    vertex_uv3[base + 3] = wp.vec3(centre[0], (gy + 0.5) * inv_grid_ny, centre[2])
    vertex_uv3[base + 4] = wp.vec3(centre[0], centre[1], (gz - 0.5) * inv_grid_nz)
    vertex_uv3[base + 5] = wp.vec3(centre[0], centre[1], (gz + 0.5) * inv_grid_nz)


@wp.kernel
def compute_vertex_positions_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_orientation: wp.array(dtype=wp.mat33),
    mc_factor: float,
    vertex_pos: wp.array(dtype=wp.vec3),
):
    """Write the 6 directional vertex positions per particle.

    Dispatch with ``dim=(num_particles, 6)``. The mapping
    ``vertex_id(p, d) = p * 6 + d`` is the same convention used by the triangle
    emitter below, so unused slots are simply never referenced.

    Directions follow the reference's ``Vertice::getPosition`` switch
    (springGrid.h:714-727): 0=-X, 1=+X, 2=-Y, 3=+Y, 4=-Z, 5=+Z.

    UV3 is baked separately by :func:`bake_vertex_uv3_kernel` since it only
    depends on static per-particle grid coordinates.
    """
    p, d = wp.tid()
    frame = particle_orientation[p]
    # Frame rows are the three local axes.
    ax = wp.vec3(0.0, 0.0, 0.0)
    sign = float(1.0)
    if d == 0:
        ax = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])
        sign = -1.0
    elif d == 1:
        ax = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])
        sign = 1.0
    elif d == 2:
        ax = wp.vec3(frame[1, 0], frame[1, 1], frame[1, 2])
        sign = -1.0
    elif d == 3:
        ax = wp.vec3(frame[1, 0], frame[1, 1], frame[1, 2])
        sign = 1.0
    elif d == 4:
        ax = wp.vec3(frame[2, 0], frame[2, 1], frame[2, 2])
        sign = -1.0
    else:  # d == 5
        ax = wp.vec3(frame[2, 0], frame[2, 1], frame[2, 2])
        sign = 1.0
    vid = p * 6 + d
    vertex_pos[vid] = particle_q[p] + ax * (sign * mc_factor)


@wp.func
def _edge_to_vertex_id(
    grid_to_particle: wp.array3d(dtype=wp.int32),
    particle_flags: wp.array(dtype=wp.int32),
    corner_offsets: wp.array2d(dtype=wp.int32),
    edge_corners: wp.array2d(dtype=wp.int32),
    edge_base_dir: wp.array(dtype=wp.int32),
    cx: int, cy: int, cz: int, edge: int,
) -> int:
    """Resolve an MC edge to the dense vertex id (-1 if the edge is degenerate).

    Picks whichever of the edge's two corners is active and offsets the base
    direction by -1 if that corner is ``b`` rather than ``a``. Matches
    ``getVertice`` in springGrid.h:741-778.
    """
    a = edge_corners[edge, 0]
    b = edge_corners[edge, 1]
    a_active = _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, a)
    b_active = _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, b)
    # A crossed edge has exactly one active endpoint; the other two states are
    # "edge fully inside" (both active) or "edge fully outside" (neither) and
    # are not referenced by any triangle for a well-formed case.
    if a_active == 0 and b_active == 0:
        return -1
    if a_active != 0 and b_active != 0:
        return -1
    active_corner = a
    d = edge_base_dir[edge]
    if a_active == 0:
        active_corner = b
        d = d - 1
    p = _corner_particle(grid_to_particle, corner_offsets, cx, cy, cz, active_corner)
    if p < 0:
        return -1
    return p * 6 + d


@wp.func
def _write_cube_fixed_slots(
    cube_flat: int,
    cx: int,
    cy: int,
    cz: int,
    case: int,
    grid_to_particle: wp.array3d(dtype=wp.int32),
    particle_flags: wp.array(dtype=wp.int32),
    corner_offsets: wp.array2d(dtype=wp.int32),
    edge_corners: wp.array2d(dtype=wp.int32),
    edge_base_dir: wp.array(dtype=wp.int32),
    case_triangles: wp.array2d(dtype=wp.int32),
    cube_tri_counts: wp.array(dtype=wp.int32),
    slot_tri_indices: wp.array2d(dtype=wp.int32),
    slot_active: wp.array(dtype=wp.int32),
    slot_to_compact: wp.array(dtype=wp.int32),
) -> int:
    base = cube_flat * MC_MAX_TRIS_PER_CASE
    for local in range(MC_MAX_TRIS_PER_CASE):
        _clear_fixed_slot(base + local, slot_tri_indices, slot_active, slot_to_compact)

    out_count = int(0)
    if case != 0 and case != 255:
        for t in range(MC_MAX_TRIS_PER_CASE):
            e0 = case_triangles[case, t * 3 + 0]
            if e0 >= 0:
                e1 = case_triangles[case, t * 3 + 1]
                e2 = case_triangles[case, t * 3 + 2]
                v0 = _edge_to_vertex_id(
                    grid_to_particle,
                    particle_flags,
                    corner_offsets,
                    edge_corners,
                    edge_base_dir,
                    cx,
                    cy,
                    cz,
                    e0,
                )
                v1 = _edge_to_vertex_id(
                    grid_to_particle,
                    particle_flags,
                    corner_offsets,
                    edge_corners,
                    edge_base_dir,
                    cx,
                    cy,
                    cz,
                    e1,
                )
                v2 = _edge_to_vertex_id(
                    grid_to_particle,
                    particle_flags,
                    corner_offsets,
                    edge_corners,
                    edge_base_dir,
                    cx,
                    cy,
                    cz,
                    e2,
                )
                if v0 >= 0 and v1 >= 0 and v2 >= 0 and out_count < MC_MAX_TRIS_PER_CASE:
                    slot = base + out_count
                    slot_tri_indices[slot, 0] = v0
                    slot_tri_indices[slot, 1] = v1
                    slot_tri_indices[slot, 2] = v2
                    slot_active[slot] = 1
                    out_count = out_count + 1

    cube_tri_counts[cube_flat] = out_count
    return out_count


@wp.kernel
def emit_triangles_kernel(
    grid_to_particle: wp.array3d(dtype=wp.int32),
    particle_flags: wp.array(dtype=wp.int32),
    cube_cases: wp.array3d(dtype=wp.int32),
    corner_offsets: wp.array2d(dtype=wp.int32),
    edge_corners: wp.array2d(dtype=wp.int32),
    edge_base_dir: wp.array(dtype=wp.int32),
    case_triangles: wp.array2d(dtype=wp.int32),
    max_triangles: int,
    tri_count: wp.array(dtype=wp.int32),
    tri_indices: wp.array2d(dtype=wp.int32),
):
    """One thread per cube cell; atomically append this cell's triangles."""
    cx, cy, cz = wp.tid()
    case = cube_cases[cx, cy, cz]
    if case == 0 or case == 255:
        return

    # Each case emits up to 5 triangles (15 edge references). Unroll the
    # fetch-and-emit loop; -1 terminates early.
    for t in range(5):
        e0 = case_triangles[case, t * 3 + 0]
        if e0 < 0:
            return
        e1 = case_triangles[case, t * 3 + 1]
        e2 = case_triangles[case, t * 3 + 2]
        v0 = _edge_to_vertex_id(grid_to_particle, particle_flags, corner_offsets, edge_corners, edge_base_dir, cx, cy, cz, e0)
        v1 = _edge_to_vertex_id(grid_to_particle, particle_flags, corner_offsets, edge_corners, edge_base_dir, cx, cy, cz, e1)
        v2 = _edge_to_vertex_id(grid_to_particle, particle_flags, corner_offsets, edge_corners, edge_base_dir, cx, cy, cz, e2)
        if v0 < 0 or v1 < 0 or v2 < 0:
            continue
        idx = wp.atomic_add(tri_count, 0, 1)
        if idx < max_triangles:
            tri_indices[idx, 0] = v0
            tri_indices[idx, 1] = v1
            tri_indices[idx, 2] = v2


@wp.kernel
def emit_fixed_slots_kernel(
    grid_to_particle: wp.array3d(dtype=wp.int32),
    particle_flags: wp.array(dtype=wp.int32),
    cube_cases: wp.array3d(dtype=wp.int32),
    corner_offsets: wp.array2d(dtype=wp.int32),
    edge_corners: wp.array2d(dtype=wp.int32),
    edge_base_dir: wp.array(dtype=wp.int32),
    case_triangles: wp.array2d(dtype=wp.int32),
    ny_cells: int,
    nz_cells: int,
    cube_tri_counts: wp.array(dtype=wp.int32),
    slot_tri_indices: wp.array2d(dtype=wp.int32),
    slot_active: wp.array(dtype=wp.int32),
    slot_to_compact: wp.array(dtype=wp.int32),
):
    """Emit each cube into its own fixed 5-triangle slot block."""
    cx, cy, cz = wp.tid()
    cube_flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)
    _write_cube_fixed_slots(
        cube_flat,
        cx,
        cy,
        cz,
        cube_cases[cx, cy, cz],
        grid_to_particle,
        particle_flags,
        corner_offsets,
        edge_corners,
        edge_base_dir,
        case_triangles,
        cube_tri_counts,
        slot_tri_indices,
        slot_active,
        slot_to_compact,
    )


@wp.kernel
def compact_fixed_slots_kernel(
    slot_tri_indices: wp.array2d(dtype=wp.int32),
    slot_active: wp.array(dtype=wp.int32),
    slot_to_compact: wp.array(dtype=wp.int32),
    compact_to_slot: wp.array(dtype=wp.int32),
    max_triangles: int,
    tri_count: wp.array(dtype=wp.int32),
    tri_indices: wp.array2d(dtype=wp.int32),
):
    """Compact active fixed slots into ``tri_indices[:tri_count]``."""
    slot = wp.tid()
    if slot_active[slot] == 0:
        slot_to_compact[slot] = -1
        return

    idx = wp.atomic_add(tri_count, 0, 1)
    if idx >= max_triangles:
        slot_to_compact[slot] = -1
        return
    tri_indices[idx, 0] = slot_tri_indices[slot, 0]
    tri_indices[idx, 1] = slot_tri_indices[slot, 1]
    tri_indices[idx, 2] = slot_tri_indices[slot, 2]
    slot_to_compact[slot] = idx
    compact_to_slot[idx] = slot


@wp.kernel
def mark_dirty_cubes_from_particles_kernel(
    particle_ids: wp.array(dtype=wp.int32),
    particle_count: int,
    particle_grid_xyz: wp.array2d(dtype=wp.int32),
    nx_cells: int,
    ny_cells: int,
    nz_cells: int,
    stamp: int,
    dirty_marks: wp.array(dtype=wp.int32),
    dirty_count: wp.array(dtype=wp.int32),
    dirty_cube_ids: wp.array(dtype=wp.int32),
):
    """Mark the up-to-8 MC cubes incident to each changed particle/cell."""
    i = wp.tid()
    if i >= particle_count:
        return
    p = particle_ids[i]
    if p < 0:
        return
    gx = particle_grid_xyz[p, 0]
    gy = particle_grid_xyz[p, 1]
    gz = particle_grid_xyz[p, 2]
    for ox in range(2):
        cx = gx - ox
        if cx >= 0 and cx < nx_cells:
            for oy in range(2):
                cy = gy - oy
                if cy >= 0 and cy < ny_cells:
                    for oz in range(2):
                        cz = gz - oz
                        if cz >= 0 and cz < nz_cells:
                            flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)
                            old = wp.atomic_exch(dirty_marks, flat, stamp)
                            if old != stamp:
                                dst = wp.atomic_add(dirty_count, 0, 1)
                                if dst < dirty_cube_ids.shape[0]:
                                    dirty_cube_ids[dst] = flat


@wp.kernel
def mark_dirty_cubes_from_particles_device_count_kernel(
    particle_ids: wp.array(dtype=wp.int32),
    particle_count: wp.array(dtype=wp.int32),
    particle_grid_xyz: wp.array2d(dtype=wp.int32),
    nx_cells: int,
    ny_cells: int,
    nz_cells: int,
    stamp: int,
    dirty_marks: wp.array(dtype=wp.int32),
    dirty_count: wp.array(dtype=wp.int32),
    dirty_cube_ids: wp.array(dtype=wp.int32),
):
    """Device-count variant for compact async deletion results."""
    i = wp.tid()
    if i >= particle_count[0]:
        return
    p = particle_ids[i]
    if p < 0:
        return
    gx = particle_grid_xyz[p, 0]
    gy = particle_grid_xyz[p, 1]
    gz = particle_grid_xyz[p, 2]
    for ox in range(2):
        cx = gx - ox
        if cx >= 0 and cx < nx_cells:
            for oy in range(2):
                cy = gy - oy
                if cy >= 0 and cy < ny_cells:
                    for oz in range(2):
                        cz = gz - oz
                        if cz >= 0 and cz < nz_cells:
                            flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)
                            old = wp.atomic_exch(dirty_marks, flat, stamp)
                            if old != stamp:
                                dst = wp.atomic_add(dirty_count, 0, 1)
                                if dst < dirty_cube_ids.shape[0]:
                                    dirty_cube_ids[dst] = flat


@wp.kernel
def remove_dirty_cube_slots_kernel(
    dirty_cube_ids: wp.array(dtype=wp.int32),
    dirty_count: wp.array(dtype=wp.int32),
    cube_tri_counts: wp.array(dtype=wp.int32),
    slot_tri_indices: wp.array2d(dtype=wp.int32),
    slot_active: wp.array(dtype=wp.int32),
    slot_to_compact: wp.array(dtype=wp.int32),
    compact_to_slot: wp.array(dtype=wp.int32),
    tri_count: wp.array(dtype=wp.int32),
    tri_indices: wp.array2d(dtype=wp.int32),
):
    """Sequentially remove old triangles for dirty cubes from the compact list."""
    n = dirty_count[0]
    i = int(0)
    while i < n:
        cube_flat = dirty_cube_ids[i]
        cube_tri_counts[cube_flat] = 0
        base = cube_flat * MC_MAX_TRIS_PER_CASE
        local = int(0)
        while local < MC_MAX_TRIS_PER_CASE:
            slot = base + local
            if slot_active[slot] != 0:
                compact_idx = slot_to_compact[slot]
                if compact_idx >= 0 and compact_idx < tri_count[0]:
                    last_idx = tri_count[0] - 1
                    if compact_idx != last_idx:
                        moved_slot = compact_to_slot[last_idx]
                        tri_indices[compact_idx, 0] = tri_indices[last_idx, 0]
                        tri_indices[compact_idx, 1] = tri_indices[last_idx, 1]
                        tri_indices[compact_idx, 2] = tri_indices[last_idx, 2]
                        compact_to_slot[compact_idx] = moved_slot
                        if moved_slot >= 0:
                            slot_to_compact[moved_slot] = compact_idx
                    compact_to_slot[last_idx] = -1
                    tri_count[0] = last_idx
                _clear_fixed_slot(slot, slot_tri_indices, slot_active, slot_to_compact)
            local = local + 1
        i = i + 1


@wp.kernel
def reemit_dirty_cube_slots_kernel(
    dirty_cube_ids: wp.array(dtype=wp.int32),
    dirty_count: wp.array(dtype=wp.int32),
    nx_cells: int,
    ny_cells: int,
    nz_cells: int,
    grid_to_particle: wp.array3d(dtype=wp.int32),
    particle_flags: wp.array(dtype=wp.int32),
    cube_cases: wp.array3d(dtype=wp.int32),
    corner_offsets: wp.array2d(dtype=wp.int32),
    edge_corners: wp.array2d(dtype=wp.int32),
    edge_base_dir: wp.array(dtype=wp.int32),
    case_triangles: wp.array2d(dtype=wp.int32),
    cube_tri_counts: wp.array(dtype=wp.int32),
    slot_tri_indices: wp.array2d(dtype=wp.int32),
    slot_active: wp.array(dtype=wp.int32),
    slot_to_compact: wp.array(dtype=wp.int32),
    compact_to_slot: wp.array(dtype=wp.int32),
    tri_count: wp.array(dtype=wp.int32),
    tri_indices: wp.array2d(dtype=wp.int32),
):
    """Recompute dirty cubes and append their new active slots."""
    i = wp.tid()
    if i >= dirty_count[0]:
        return

    cube_flat = dirty_cube_ids[i]
    plane = ny_cells * nz_cells
    cx = cube_flat / plane
    rem = cube_flat - cx * plane
    cy = rem / nz_cells
    cz = rem - cy * nz_cells
    if cx < 0 or cx >= nx_cells or cy < 0 or cy >= ny_cells or cz < 0 or cz >= nz_cells:
        return

    case = _compute_cube_case(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz)
    cube_cases[cx, cy, cz] = case
    _write_cube_fixed_slots(
        cube_flat,
        cx,
        cy,
        cz,
        case,
        grid_to_particle,
        particle_flags,
        corner_offsets,
        edge_corners,
        edge_base_dir,
        case_triangles,
        cube_tri_counts,
        slot_tri_indices,
        slot_active,
        slot_to_compact,
    )

    base = cube_flat * MC_MAX_TRIS_PER_CASE
    for local in range(MC_MAX_TRIS_PER_CASE):
        slot = base + local
        if slot_active[slot] != 0:
            dst = wp.atomic_add(tri_count, 0, 1)
            tri_indices[dst, 0] = slot_tri_indices[slot, 0]
            tri_indices[dst, 1] = slot_tri_indices[slot, 1]
            tri_indices[dst, 2] = slot_tri_indices[slot, 2]
            slot_to_compact[slot] = dst
            compact_to_slot[dst] = slot


@dataclass
class MarchingCubesBuffers:
    """Persistent GPU scratch/output buffers for the MC pipeline."""

    cube_cases: wp.array  # int32[nx-1, ny-1, nz-1]
    vertex_pos: wp.array  # vec3[num_particles * 6]
    particle_uv3: wp.array  # vec3[num_particles] - grid-normalised cell centres
    vertex_uv3: wp.array  # vec3[num_particles * 6] - grid-normalised directional tex coords
    tri_indices: wp.array  # int32[max_triangles, 3]
    tri_count: wp.array  # int32[1] - atomic counter
    max_triangles: int
    cube_tri_counts: wp.array  # int32[num_cubes] - active fixed slots per cube
    slot_tri_indices: wp.array  # int32[slot_capacity, 3] - fixed 5 slots per cube
    slot_active: wp.array  # int32[slot_capacity]
    slot_to_compact: wp.array  # int32[slot_capacity] -> compact tri index, or -1
    compact_to_slot: wp.array  # int32[max_triangles] -> fixed slot id, or -1
    dirty_cube_ids: wp.array  # int32[num_cubes] scratch
    dirty_cube_marks: wp.array  # int32[num_cubes] stamp scratch
    dirty_count: wp.array  # int32[1] scratch
    slot_capacity: int
    num_cubes: int
    dirty_stamp: int = 1


def allocate_mc_buffers(
    grid_shape: tuple[int, int, int],
    num_particles: int,
    max_triangles: int | None = None,
    device: str | wp.context.Device | None = None,
) -> MarchingCubesBuffers:
    """Allocate persistent buffers sized for the given grid + particle count.

    ``max_triangles`` defaults to ``5 * (nx-1) * (ny-1) * (nz-1)`` (worst case).
    In practice only boundary cells emit triangles; set a smaller budget if
    you are memory-constrained and willing to clamp.
    """
    nx, ny, nz = grid_shape
    cell_shape = (max(nx - 1, 1), max(ny - 1, 1), max(nz - 1, 1))
    num_cubes = int(cell_shape[0] * cell_shape[1] * cell_shape[2])
    slot_capacity = int(MC_MAX_TRIS_PER_CASE * num_cubes)
    if max_triangles is None:
        max_triangles = slot_capacity
    return MarchingCubesBuffers(
        cube_cases=wp.zeros(cell_shape, dtype=wp.int32, device=device),
        vertex_pos=wp.zeros(num_particles * 6, dtype=wp.vec3, device=device),
        particle_uv3=wp.zeros(num_particles, dtype=wp.vec3, device=device),
        vertex_uv3=wp.zeros(num_particles * 6, dtype=wp.vec3, device=device),
        tri_indices=wp.zeros((max_triangles, 3), dtype=wp.int32, device=device),
        tri_count=wp.zeros(1, dtype=wp.int32, device=device),
        max_triangles=max_triangles,
        cube_tri_counts=wp.zeros(num_cubes, dtype=wp.int32, device=device),
        slot_tri_indices=wp.zeros((slot_capacity, 3), dtype=wp.int32, device=device),
        slot_active=wp.zeros(slot_capacity, dtype=wp.int32, device=device),
        slot_to_compact=wp.full(slot_capacity, -1, dtype=wp.int32, device=device),
        compact_to_slot=wp.full(max_triangles, -1, dtype=wp.int32, device=device),
        dirty_cube_ids=wp.zeros(num_cubes, dtype=wp.int32, device=device),
        dirty_cube_marks=wp.zeros(num_cubes, dtype=wp.int32, device=device),
        dirty_count=wp.zeros(1, dtype=wp.int32, device=device),
        slot_capacity=slot_capacity,
        num_cubes=num_cubes,
    )


def bake_vertex_uv3(
    buffers: MarchingCubesBuffers,
    particle_grid_xyz: wp.array,
    grid_shape: tuple[int, int, int],
    device: str | wp.context.Device | None = None,
) -> None:
    """Populate static texture coordinates from per-particle grid coords.

    UV3 is invariant under deformation and cuts (it keys off
    ``particle_grid_xyz``, which is written once at lattice build and never
    reassigned), so this should be called once after
    :func:`allocate_mc_buffers` and never again for the lifetime of the
    buffers.
    """
    num_particles = int(particle_grid_xyz.shape[0])
    nx, ny, nz = grid_shape
    wp.launch(
        bake_vertex_uv3_kernel,
        dim=num_particles,
        inputs=[
            particle_grid_xyz,
            float(1.0 / max(nx - 1, 1)),
            float(1.0 / max(ny - 1, 1)),
            float(1.0 / max(nz - 1, 1)),
        ],
        outputs=[buffers.particle_uv3, buffers.vertex_uv3],
        device=device,
    )


def compute_mc_vertex_positions(
    buffers: MarchingCubesBuffers,
    particle_q: wp.array,
    particle_orientation: wp.array,
    mc_factor: float,
    device: str | wp.context.Device | None = None,
) -> None:
    """Per-frame: write the 6 directional vertex positions for every particle."""
    num_particles = int(particle_q.shape[0])
    wp.launch(
        compute_vertex_positions_kernel,
        dim=(num_particles, _NUM_DIRS_PER_PARTICLE),
        inputs=[particle_q, particle_orientation, float(mc_factor)],
        outputs=[buffers.vertex_pos],
        device=device,
    )


def compute_mc_topology(
    buffers: MarchingCubesBuffers,
    tables: MarchingCubesTables,
    particle_flags: wp.array,
    grid_to_particle: wp.array,
    device: str | wp.context.Device | None = None,
) -> int:
    """Compute cube cases + emit triangles. Returns the triangle count.

    One host sync is unavoidable here (to get the atomic counter back), so
    callers that cache topology should only invoke this on topology changes.
    """
    buffers.tri_count.zero_()
    buffers.slot_to_compact.fill_(-1)
    buffers.compact_to_slot.fill_(-1)
    nx_cells, ny_cells, nz_cells = buffers.cube_cases.shape
    wp.launch(
        compute_cube_cases_kernel,
        dim=(nx_cells, ny_cells, nz_cells),
        inputs=[grid_to_particle, particle_flags, tables.corner_offsets],
        outputs=[buffers.cube_cases],
        device=device,
    )
    wp.launch(
        emit_fixed_slots_kernel,
        dim=(nx_cells, ny_cells, nz_cells),
        inputs=[
            grid_to_particle,
            particle_flags,
            buffers.cube_cases,
            tables.corner_offsets,
            tables.edge_corners,
            tables.edge_base_dir,
            tables.case_triangles,
            int(ny_cells),
            int(nz_cells),
        ],
        outputs=[
            buffers.cube_tri_counts,
            buffers.slot_tri_indices,
            buffers.slot_active,
            buffers.slot_to_compact,
        ],
        device=device,
    )
    wp.launch(
        compact_fixed_slots_kernel,
        dim=int(buffers.slot_capacity),
        inputs=[
            buffers.slot_tri_indices,
            buffers.slot_active,
            buffers.slot_to_compact,
            buffers.compact_to_slot,
            int(buffers.max_triangles),
        ],
        outputs=[buffers.tri_count, buffers.tri_indices],
        device=device,
    )
    count = int(buffers.tri_count.numpy()[0])
    return min(count, buffers.max_triangles)


def compute_mc_topology_dirty(
    buffers: MarchingCubesBuffers,
    tables: MarchingCubesTables,
    particle_flags: wp.array,
    grid_to_particle: wp.array,
    particle_grid_xyz: wp.array,
    dirty_particle_ids: wp.array | None,
    dirty_particle_count: int,
    dirty_particle_count_device: wp.array | None = None,
    dirty_particle_capacity: int | None = None,
    device: str | wp.context.Device | None = None,
    fallback_fraction: float = 0.02,
    fallback_max_dirty_cubes: int = 4096,
) -> int | None:
    """Locally update MC topology for particles/cells whose ACTIVE bit changed.

    Returns the new compact triangle count, or ``None`` when the caller should
    fall back to :func:`compute_mc_topology`.
    """
    dirty_particle_count = int(dirty_particle_count)
    if dirty_particle_ids is None:
        return None
    use_device_count = dirty_particle_count_device is not None
    if use_device_count:
        dirty_capacity = int(dirty_particle_capacity if dirty_particle_capacity is not None else dirty_particle_ids.shape[0])
    else:
        dirty_capacity = dirty_particle_count
    if dirty_capacity <= 0:
        return None
    # The local slot-map updater assumes the compact draw budget can hold all
    # fixed slots. Custom low-memory callers can still use the full rebuild path.
    if int(buffers.max_triangles) < int(buffers.slot_capacity):
        return None
    max_dirty_candidates = min(int(buffers.num_cubes), max(1, dirty_capacity * 8))
    dirty_rebuild_limit = max(
        64,
        min(
            int(fallback_max_dirty_cubes),
            int(float(buffers.num_cubes) * float(fallback_fraction)),
        ),
    )
    if max_dirty_candidates >= dirty_rebuild_limit:
        return None

    nx_cells, ny_cells, nz_cells = buffers.cube_cases.shape
    if buffers.dirty_stamp >= 2_000_000_000:
        buffers.dirty_cube_marks.zero_()
        buffers.dirty_stamp = 1
    else:
        buffers.dirty_stamp += 1

    buffers.dirty_count.zero_()
    if use_device_count:
        wp.launch(
            mark_dirty_cubes_from_particles_device_count_kernel,
            dim=dirty_capacity,
            inputs=[
                dirty_particle_ids,
                dirty_particle_count_device,
                particle_grid_xyz,
                int(nx_cells),
                int(ny_cells),
                int(nz_cells),
                int(buffers.dirty_stamp),
            ],
            outputs=[buffers.dirty_cube_marks, buffers.dirty_count, buffers.dirty_cube_ids],
            device=device,
        )
    else:
        wp.launch(
            mark_dirty_cubes_from_particles_kernel,
            dim=dirty_particle_count,
            inputs=[
                dirty_particle_ids,
                dirty_particle_count,
                particle_grid_xyz,
                int(nx_cells),
                int(ny_cells),
                int(nz_cells),
                int(buffers.dirty_stamp),
            ],
            outputs=[buffers.dirty_cube_marks, buffers.dirty_count, buffers.dirty_cube_ids],
            device=device,
        )
    wp.launch(
        remove_dirty_cube_slots_kernel,
        dim=1,
        inputs=[
            buffers.dirty_cube_ids,
            buffers.dirty_count,
            buffers.cube_tri_counts,
            buffers.slot_tri_indices,
            buffers.slot_active,
            buffers.slot_to_compact,
            buffers.compact_to_slot,
            buffers.tri_count,
        ],
        outputs=[buffers.tri_indices],
        device=device,
    )
    wp.launch(
        reemit_dirty_cube_slots_kernel,
        dim=max_dirty_candidates,
        inputs=[
            buffers.dirty_cube_ids,
            buffers.dirty_count,
            int(nx_cells),
            int(ny_cells),
            int(nz_cells),
            grid_to_particle,
            particle_flags,
            buffers.cube_cases,
            tables.corner_offsets,
            tables.edge_corners,
            tables.edge_base_dir,
            tables.case_triangles,
            buffers.cube_tri_counts,
            buffers.slot_tri_indices,
            buffers.slot_active,
            buffers.slot_to_compact,
            buffers.compact_to_slot,
            buffers.tri_count,
        ],
        outputs=[buffers.tri_indices],
        device=device,
    )
    count = int(buffers.tri_count.numpy()[0])
    return min(count, buffers.max_triangles)


def run_marching_cubes(
    particle_q: wp.array,
    particle_flags: wp.array,
    particle_orientation: wp.array,
    particle_grid_xyz: wp.array,
    grid_to_particle: wp.array,
    grid_shape: tuple[int, int, int],
    tables: MarchingCubesTables,
    buffers: MarchingCubesBuffers,
    mc_factor: float,
    device: str | wp.context.Device | None = None,
) -> int:
    """Legacy fused MC pass: bakes UV3, writes positions, emits triangles.

    Kept for callers that run MC as a one-shot operation. Per-frame callers
    should split the work via :func:`bake_vertex_uv3` (once),
    :func:`compute_mc_vertex_positions` (every frame), and
    :func:`compute_mc_topology` (only on topology change).
    """
    bake_vertex_uv3(buffers, particle_grid_xyz, grid_shape, device=device)
    compute_mc_vertex_positions(buffers, particle_q, particle_orientation, mc_factor, device=device)
    return compute_mc_topology(buffers, tables, particle_flags, grid_to_particle, device=device)


__all__ = [
    "MC_MAX_TRIS_PER_CASE",
    "MarchingCubesBuffers",
    "MarchingCubesTables",
    "allocate_mc_buffers",
    "bake_vertex_uv3",
    "bake_vertex_uv3_kernel",
    "compute_cube_cases_kernel",
    "compute_mc_topology",
    "compute_mc_topology_dirty",
    "compute_mc_vertex_positions",
    "compute_vertex_positions_kernel",
    "emit_fixed_slots_kernel",
    "emit_triangles_kernel",
    "mark_dirty_cubes_from_particles_device_count_kernel",
    "run_marching_cubes",
    "upload_mc_tables",
]
