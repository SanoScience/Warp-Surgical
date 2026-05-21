# SPDX-License-Identifier: Apache-2.0
"""Springless particle lattice with labelled hexahedral cells.

Topology:

* particles live on shared hexahedral cell vertices
* occupied voxels become active labelled cells that reference 8 particles
* deformation is supplied by shape-matching clusters layered on this topology
"""

from __future__ import annotations

from dataclasses import dataclass

import newton
import numpy as np
import warp as wp

from .io.digimouse import DigimouseAtlas
from .materials import MaterialTable, Phase

# Corner order for a unit voxel cell.
# 0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0)
# 4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
CELL_CORNER_OFFSETS = np.asarray(
    [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ],
    dtype=np.int32,
)

@dataclass
class ShapeMatchingClusters:
    """Cluster sidecar for hex-grid shape matching."""

    offsets: wp.array  # int32[num_clusters + 1]
    indices: wp.array  # int32[num_memberships]
    indices_by_slot: wp.array  # int32[uniform_size * num_clusters], slot-major
    rest_centers: wp.array  # vec3[num_clusters]
    rest_local_positions: wp.array  # vec3[num_memberships]
    rest_local_positions_by_slot: wp.array  # vec3[uniform_size * num_clusters], slot-major
    rest_local_template: wp.array  # vec3[uniform_size], valid when rest_local_template_valid is True
    coefficients: wp.array  # float32[num_clusters]
    active: wp.array  # int32[num_clusters]
    colors: wp.array  # int32[num_clusters], used by colored Gauss-Seidel passes
    color_offsets: wp.array  # int32[9], offsets into color_cluster_indices
    color_cluster_indices: wp.array  # int32[num_clusters], cluster ids grouped by color
    source_cell: wp.array  # int32[num_clusters]
    cell_to_cluster: wp.array  # int32[num_cells], -1 for cells without a cluster
    particle_cluster_counts: wp.array  # int32[num_particles]
    particle_cluster_inv_weights: wp.array  # float32[num_particles]
    particle_cluster_offsets: wp.array  # int32[num_particles + 1]
    particle_cluster_indices: wp.array  # int32[num_memberships]
    particle_cluster_member_offsets: wp.array  # int32[num_memberships]

    num_clusters: int
    num_memberships: int
    uniform_size: int

    offsets_host: np.ndarray
    indices_host: np.ndarray
    indices_by_slot_host: np.ndarray
    rest_centers_host: np.ndarray
    rest_local_positions_host: np.ndarray
    rest_local_positions_by_slot_host: np.ndarray
    rest_local_template_host: np.ndarray
    rest_local_template_valid: bool
    coefficients_host: np.ndarray
    base_coefficients_host: np.ndarray
    active_host: np.ndarray
    colors_host: np.ndarray
    color_offsets_host: np.ndarray
    color_cluster_indices_host: np.ndarray
    source_cell_host: np.ndarray
    cell_to_cluster_host: np.ndarray
    particle_cluster_counts_host: np.ndarray
    particle_cluster_inv_weights_host: np.ndarray
    particle_cluster_offsets_host: np.ndarray
    particle_cluster_indices_host: np.ndarray
    particle_cluster_member_offsets_host: np.ndarray


@dataclass
class ShapeMatchingProlongation:
    """Regular-grid correction prolongation from a coarse shape level."""

    parent_indices: wp.array  # int32[num_particles * 8], -1 for invalid parents
    parent_weights: wp.array  # float32[num_particles * 8]
    child_cluster: wp.array  # int32[num_particles], -1 when no prolongation applies
    node_grid_xyz: wp.array  # int32[num_particles, 3]
    grid_to_node: wp.array  # int32[nx+1, ny+1, nz+1]

    num_particles: int
    fanout: int
    block_size: int
    max_grid_coord: tuple[int, int, int]

    parent_indices_host: np.ndarray
    parent_weights_host: np.ndarray
    child_cluster_host: np.ndarray


@dataclass
class HierarchicalShapeMatchingClusters:
    """Optional coarse shape-matching data for hex-grid experiments."""

    outer8: ShapeMatchingClusters | None
    outer8_prolongation: ShapeMatchingProlongation | None
    full27: ShapeMatchingClusters | None
    l2_outer8: ShapeMatchingClusters | None
    l2_outer8_prolongation: ShapeMatchingProlongation | None
    l2_full125: ShapeMatchingClusters | None
    block_size: int
    l2_block_size: int
    outer8_block_keys_host: np.ndarray | None = None
    l2_outer8_block_keys_host: np.ndarray | None = None
    outer8_sleepable_host: np.ndarray | None = None
    l2_outer8_sleepable_host: np.ndarray | None = None
    outer8_block_keys: wp.array | None = None
    l2_outer8_block_keys: wp.array | None = None
    outer8_sleepable: wp.array | None = None
    l2_outer8_sleepable: wp.array | None = None
    outer8_block_lookup: wp.array | None = None
    l2_outer8_block_lookup: wp.array | None = None
    outer8_block_lookup_host: np.ndarray | None = None
    l2_outer8_block_lookup_host: np.ndarray | None = None
    outer8_block_lookup_shape: tuple[int, int, int] | None = None
    l2_outer8_block_lookup_shape: tuple[int, int, int] | None = None


@dataclass
class HexGridAuxState:
    """Auxiliary arrays and host-side topology for the hex particle lattice."""

    node_grid_xyz: wp.array  # int32[num_nodes, 3]
    node_material: wp.array  # int32[num_nodes]
    node_support_count: wp.array  # int32[num_nodes]
    grid_to_node: wp.array  # int32[nx+1, ny+1, nz+1]

    cell_grid_xyz: wp.array  # int32[num_cells, 3]
    cell_nodes: wp.array  # int32[num_cells, 8]
    cell_material: wp.array  # int32[num_cells]
    cell_active: wp.array  # int32[num_cells]

    # Cell-centered render layer. Written by `update_cell_render_state` each
    # frame; consumed by the marching-cubes pipeline which expects one render
    # particle per voxel center. `grid_to_cell` is static (-1 for empty atlas
    # voxels) and drives MC corner lookups.
    cell_center_q: wp.array  # vec3[num_cells]
    cell_orientation: wp.array  # mat33[num_cells]
    cell_render_flags: wp.array  # int32[num_cells]
    cell_stretch: wp.array  # float32[num_cells], max per-cell absolute constraint strain for display
    grid_to_cell: wp.array  # int32[nx, ny, nz]

    cell_mass: wp.array  # float32[num_cells]

    grid_shape: tuple[int, int, int]
    node_shape: tuple[int, int, int]
    voxel_size: float
    origin: tuple[float, float, float]
    num_nodes: int
    num_cells: int
    materials: MaterialTable

    # Host-side topology for dynamic deletion / mass updates.
    cell_nodes_host: np.ndarray
    cell_material_host: np.ndarray
    cell_mass_host: np.ndarray
    node_support_count_host: np.ndarray
    node_material_host: np.ndarray
    cell_active_host: np.ndarray
    shape_matching_clusters: ShapeMatchingClusters | None = None
    shape_matching_hierarchy: HierarchicalShapeMatchingClusters | None = None


@dataclass
class HexParticleGrid:
    """Result of :func:`build_hex_particle_grid`."""

    model: newton.Model
    state: newton.State
    aux: HexGridAuxState


def _pack_ragged_int32(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(rows) + 1, dtype=np.int32)
    data: list[int] = []
    cursor = 0
    for idx, row in enumerate(rows):
        data.extend(row)
        cursor += len(row)
        offsets[idx + 1] = cursor
    return offsets, np.asarray(data, dtype=np.int32)


def _build_shape_matching_clusters_from_members(
    pg: HexParticleGrid,
    members: np.ndarray,
    *,
    coefficient: float,
    source_cell: np.ndarray,
    cell_to_cluster: np.ndarray,
    colors: np.ndarray | None = None,
    target_device: wp.context.Device,
) -> ShapeMatchingClusters:
    if members.ndim != 2:
        raise ValueError(f"shape-matching members must be a 2D array, got shape {members.shape}")

    num_clusters = int(members.shape[0])
    uniform_size = int(members.shape[1]) if members.shape[0] > 0 else 0
    if num_clusters == 0 or uniform_size == 0:
        raise ValueError("shape-matching cluster set is empty")

    rest_q = pg.model.particle_q.numpy()
    rest_members = rest_q[members]
    rest_centers = rest_members.mean(axis=1).astype(np.float32, copy=False)
    rest_local_rows = (rest_members - rest_centers[:, None, :]).astype(np.float32, copy=False)
    rest_local_positions = rest_local_rows.reshape(-1, 3).astype(np.float32, copy=False)
    rest_local_positions_by_slot = np.ascontiguousarray(
        np.transpose(rest_local_rows, (1, 0, 2)).reshape(-1, 3),
        dtype=np.float32,
    )
    rest_local_template_valid = bool(np.allclose(rest_local_rows, rest_local_rows[:1], rtol=0.0, atol=1.0e-7))
    rest_local_template = np.ascontiguousarray(rest_local_rows[0], dtype=np.float32)

    offsets = (np.arange(num_clusters + 1, dtype=np.int32) * uniform_size).astype(np.int32, copy=False)
    indices = members.reshape(-1).astype(np.int32, copy=False)
    indices_by_slot = np.ascontiguousarray(members.T.reshape(-1), dtype=np.int32)
    coefficients = np.full(num_clusters, float(coefficient), dtype=np.float32)
    active = np.ones(num_clusters, dtype=np.int32)
    if colors is None:
        colors = np.zeros(num_clusters, dtype=np.int32)
    else:
        colors = np.asarray(colors, dtype=np.int32).reshape(-1)
        if colors.shape != (num_clusters,):
            raise ValueError(f"cluster colors must have shape ({num_clusters},), got {colors.shape}")
    if np.any((colors < 0) | (colors >= 8)):
        raise ValueError("cluster colors must be in [0, 7]")
    color_counts = np.bincount(colors, minlength=8).astype(np.int32, copy=False)
    color_offsets = np.zeros(9, dtype=np.int32)
    color_offsets[1:] = np.cumsum(color_counts, dtype=np.int32)
    color_cluster_indices = np.argsort(colors, kind="stable").astype(np.int32, copy=False)

    particle_cluster_counts = np.zeros(pg.model.particle_count, dtype=np.int32)
    np.add.at(particle_cluster_counts, indices, 1)
    particle_cluster_inv_weights = np.zeros(pg.model.particle_count, dtype=np.float32)
    valid_counts = particle_cluster_counts > 0
    particle_cluster_inv_weights[valid_counts] = 1.0 / particle_cluster_counts[valid_counts].astype(np.float32)

    particle_cluster_offsets = np.zeros(pg.model.particle_count + 1, dtype=np.int32)
    particle_cluster_offsets[1:] = np.cumsum(particle_cluster_counts, dtype=np.int32)
    cluster_ids = np.repeat(np.arange(num_clusters, dtype=np.int32), uniform_size)
    member_offsets = np.arange(indices.size, dtype=np.int32)
    # Stable argsort by particle id reproduces the per-particle insertion order
    # of the original write-cursor loop without iterating in Python.
    sort_perm = np.argsort(indices, kind="stable")
    particle_cluster_indices = np.ascontiguousarray(cluster_ids[sort_perm], dtype=np.int32)
    particle_cluster_member_offsets = np.ascontiguousarray(member_offsets[sort_perm], dtype=np.int32)

    return ShapeMatchingClusters(
        offsets=wp.array(offsets, dtype=wp.int32, device=target_device),
        indices=wp.array(indices, dtype=wp.int32, device=target_device),
        indices_by_slot=wp.array(indices_by_slot, dtype=wp.int32, device=target_device),
        rest_centers=wp.array(rest_centers, dtype=wp.vec3, device=target_device),
        rest_local_positions=wp.array(rest_local_positions, dtype=wp.vec3, device=target_device),
        rest_local_positions_by_slot=wp.array(rest_local_positions_by_slot, dtype=wp.vec3, device=target_device),
        rest_local_template=wp.array(rest_local_template, dtype=wp.vec3, device=target_device),
        coefficients=wp.array(coefficients, dtype=wp.float32, device=target_device),
        active=wp.array(active, dtype=wp.int32, device=target_device),
        colors=wp.array(colors, dtype=wp.int32, device=target_device),
        color_offsets=wp.array(color_offsets, dtype=wp.int32, device=target_device),
        color_cluster_indices=wp.array(color_cluster_indices, dtype=wp.int32, device=target_device),
        source_cell=wp.array(source_cell, dtype=wp.int32, device=target_device),
        cell_to_cluster=wp.array(cell_to_cluster, dtype=wp.int32, device=target_device),
        particle_cluster_counts=wp.array(particle_cluster_counts, dtype=wp.int32, device=target_device),
        particle_cluster_inv_weights=wp.array(particle_cluster_inv_weights, dtype=wp.float32, device=target_device),
        particle_cluster_offsets=wp.array(particle_cluster_offsets, dtype=wp.int32, device=target_device),
        particle_cluster_indices=wp.array(particle_cluster_indices, dtype=wp.int32, device=target_device),
        particle_cluster_member_offsets=wp.array(particle_cluster_member_offsets, dtype=wp.int32, device=target_device),
        num_clusters=num_clusters,
        num_memberships=int(indices.size),
        uniform_size=uniform_size,
        offsets_host=offsets.copy(),
        indices_host=indices.copy(),
        indices_by_slot_host=indices_by_slot.copy(),
        rest_centers_host=rest_centers.copy(),
        rest_local_positions_host=rest_local_positions.copy(),
        rest_local_positions_by_slot_host=rest_local_positions_by_slot.copy(),
        rest_local_template_host=rest_local_template.copy(),
        rest_local_template_valid=rest_local_template_valid,
        coefficients_host=coefficients.copy(),
        base_coefficients_host=coefficients.copy(),
        active_host=active.copy(),
        colors_host=colors.copy(),
        color_offsets_host=color_offsets.copy(),
        color_cluster_indices_host=color_cluster_indices.copy(),
        source_cell_host=source_cell.astype(np.int32, copy=True),
        cell_to_cluster_host=cell_to_cluster.astype(np.int32, copy=True),
        particle_cluster_counts_host=particle_cluster_counts.copy(),
        particle_cluster_inv_weights_host=particle_cluster_inv_weights.copy(),
        particle_cluster_offsets_host=particle_cluster_offsets.copy(),
        particle_cluster_indices_host=particle_cluster_indices.copy(),
        particle_cluster_member_offsets_host=particle_cluster_member_offsets.copy(),
    )

def build_hex_particle_grid(
    atlas: DigimouseAtlas,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    particle_radius: float | None = None,
    device: str | wp.context.Device | None = None,
    builder: newton.ModelBuilder | None = None,
    kinematic_bones: bool = True,
) -> HexParticleGrid:
    """Build a springless particle lattice from a labelled voxel atlas."""
    labels = atlas.labels
    nx, ny, nz = labels.shape
    voxel_size = float(atlas.voxel_size)
    ox, oy, oz = origin

    materials = atlas.materials
    phase_lut = materials.phase
    occupied = labels != 0
    num_cells = int(occupied.sum())
    if num_cells == 0:
        raise ValueError("atlas has no occupied voxels; nothing to build")

    cell_coords = np.argwhere(occupied).astype(np.int32)
    cell_labels = labels[cell_coords[:, 0], cell_coords[:, 1], cell_coords[:, 2]].astype(np.int32)
    cell_phase = phase_lut[cell_labels]
    cell_mass = materials.mass[cell_labels].astype(np.float32)
    if kinematic_bones:
        cell_mass[cell_phase == int(Phase.RIGID)] = 0.0

    node_used = np.zeros((nx + 1, ny + 1, nz + 1), dtype=bool)
    for cx, cy, cz in CELL_CORNER_OFFSETS.tolist():
        node_used[cx : cx + nx, cy : cy + ny, cz : cz + nz] |= occupied

    node_coords = np.argwhere(node_used).astype(np.int32)
    num_nodes = int(node_coords.shape[0])
    grid_to_node = np.full((nx + 1, ny + 1, nz + 1), -1, dtype=np.int32)
    grid_to_node[node_coords[:, 0], node_coords[:, 1], node_coords[:, 2]] = np.arange(num_nodes, dtype=np.int32)

    # Per-cell corner -> node lookup (vectorised: one fancy-index over all cells).
    cell_corner_coords = cell_coords[:, None, :] + CELL_CORNER_OFFSETS[None, :, :]  # (C, 8, 3)
    cell_nodes = grid_to_node[
        cell_corner_coords[..., 0],
        cell_corner_coords[..., 1],
        cell_corner_coords[..., 2],
    ].astype(np.int32, copy=False)  # (C, 8)
    flat_corners = cell_nodes.reshape(-1)
    flat_cell_idx = np.repeat(np.arange(num_cells, dtype=np.int32), 8)

    # Per-node aggregation via bincount (vs. np.add.at, which is sequential).
    node_mass = np.bincount(
        flat_corners,
        weights=(cell_mass[flat_cell_idx] * 0.125).astype(np.float64, copy=False),
        minlength=num_nodes,
    ).astype(np.float32, copy=False)
    node_support_count = np.bincount(flat_corners, minlength=num_nodes).astype(np.int32, copy=False)
    n_materials = len(materials)
    flat_material_idx = flat_corners.astype(np.int64) * n_materials + cell_labels[flat_cell_idx].astype(np.int64)
    node_material_votes = np.bincount(
        flat_material_idx, minlength=num_nodes * n_materials
    ).reshape(num_nodes, n_materials)

    node_material = node_material_votes.argmax(axis=1).astype(np.int32)
    node_positions = (
        node_coords.astype(np.float32) * voxel_size
        + np.asarray([ox, oy, oz], dtype=np.float32)
    )
    node_velocities = np.zeros((num_nodes, 3), dtype=np.float32)

    if builder is None:
        builder = newton.ModelBuilder()

    if particle_radius is None:
        particle_radius = voxel_size * 0.2

    builder.add_particles(
        pos=[tuple(p) for p in node_positions.tolist()],
        vel=[tuple(v) for v in node_velocities.tolist()],
        mass=node_mass.tolist(),
        radius=[float(particle_radius)] * num_nodes,
    )

    model = builder.finalize(device=device)
    if model.spring_count != 0:
        raise ValueError(f"HexParticleGrid must be springless; got spring_count={model.spring_count}")
    state = model.state()
    device = wp.get_device(device) if device is not None else model.device

    # Cell-centered render lookup: static (nx, ny, nz) int32 grid, -1 for empty
    # atlas voxels. Indexes into cell arrays so the MC pipeline can do an 8-corner
    # lookup identical to the voxel-center grid in `omnisurg/hex/grid.py`.
    grid_to_cell = np.full((nx, ny, nz), -1, dtype=np.int32)
    grid_to_cell[cell_coords[:, 0], cell_coords[:, 1], cell_coords[:, 2]] = np.arange(num_cells, dtype=np.int32)

    aux = HexGridAuxState(
        node_grid_xyz=wp.array(node_coords, dtype=wp.int32, device=device),
        node_material=wp.array(node_material, dtype=wp.int32, device=device),
        node_support_count=wp.array(node_support_count, dtype=wp.int32, device=device),
        grid_to_node=wp.array(grid_to_node, dtype=wp.int32, device=device),
        cell_grid_xyz=wp.array(cell_coords, dtype=wp.int32, device=device),
        cell_nodes=wp.array(cell_nodes, dtype=wp.int32, device=device),
        cell_material=wp.array(cell_labels, dtype=wp.int32, device=device),
        cell_active=wp.ones(num_cells, dtype=wp.int32, device=device),
        cell_center_q=wp.zeros(num_cells, dtype=wp.vec3, device=device),
        cell_orientation=wp.zeros(num_cells, dtype=wp.mat33, device=device),
        cell_render_flags=wp.zeros(num_cells, dtype=wp.int32, device=device),
        cell_stretch=wp.zeros(num_cells, dtype=wp.float32, device=device),
        grid_to_cell=wp.array(grid_to_cell, dtype=wp.int32, device=device),
        cell_mass=wp.array(cell_mass, dtype=wp.float32, device=device),
        grid_shape=(nx, ny, nz),
        node_shape=(nx + 1, ny + 1, nz + 1),
        voxel_size=voxel_size,
        origin=origin,
        num_nodes=num_nodes,
        num_cells=num_cells,
        materials=materials,
        cell_nodes_host=cell_nodes.copy(),
        cell_material_host=cell_labels.copy(),
        cell_mass_host=cell_mass.copy(),
        node_support_count_host=node_support_count.copy(),
        node_material_host=node_material.copy(),
        cell_active_host=np.ones(num_cells, dtype=np.int32),
        shape_matching_clusters=None,
    )
    return HexParticleGrid(model=model, state=state, aux=aux)


def build_shape_matching_clusters(
    pg: HexParticleGrid,
    coefficient: float = 1.0,
    device: str | wp.context.Device | None = None,
) -> ShapeMatchingClusters:
    """Build one 8-corner shape-matching cluster per active cell."""
    if coefficient < 0.0:
        raise ValueError(f"cluster coefficient must be non-negative, got {coefficient}")

    target_device = wp.get_device(device) if device is not None else pg.model.device
    if target_device != pg.model.device:
        raise ValueError(
            "shape-matching cluster buffers must live on the same device as the model; "
            f"got clusters on {target_device} and model on {pg.model.device}"
        )

    source_cell = np.flatnonzero(pg.aux.cell_active_host != 0).astype(np.int32)
    num_clusters = int(source_cell.size)
    if num_clusters == 0:
        raise ValueError("hex grid has no active cells; cannot build shape-matching clusters")

    cell_nodes = pg.aux.cell_nodes_host[source_cell].astype(np.int32, copy=True)
    if cell_nodes.shape[1] != 8:
        raise ValueError(f"expected 8 corner nodes per cell, got shape {cell_nodes.shape}")

    cell_to_cluster = np.full(pg.aux.num_cells, -1, dtype=np.int32)
    cell_to_cluster[source_cell] = np.arange(num_clusters, dtype=np.int32)
    cell_coords = pg.aux.cell_grid_xyz.numpy().astype(np.int32, copy=False)[source_cell]
    colors = (
        (cell_coords[:, 0] & 1)
        | ((cell_coords[:, 1] & 1) << 1)
        | ((cell_coords[:, 2] & 1) << 2)
    ).astype(np.int32, copy=False)

    clusters = _build_shape_matching_clusters_from_members(
        pg,
        cell_nodes,
        coefficient=float(coefficient),
        source_cell=source_cell,
        cell_to_cluster=cell_to_cluster,
        colors=colors,
        target_device=target_device,
    )
    pg.aux.shape_matching_clusters = clusters
    return clusters


def _lookup_grid_node(grid_to_node: np.ndarray, coord: np.ndarray) -> int:
    if np.any(coord < 0) or np.any(coord >= np.asarray(grid_to_node.shape, dtype=np.int32)):
        return -1
    return int(grid_to_node[int(coord[0]), int(coord[1]), int(coord[2])])


def _make_outer8_prolongation(
    pg: HexParticleGrid,
    *,
    block_size: int,
    block_keys: np.ndarray,
    block_cluster_ids: np.ndarray,
    target_device: wp.context.Device,
) -> ShapeMatchingProlongation:
    """Trilinear prolongation table from L0 nodes to the 8 corners of their L1 block.

    ``block_keys`` (shape ``(M, 3)`` int32) and ``block_cluster_ids`` (shape
    ``(M,)`` int32) describe the active outer8 clusters: a node falling inside
    a block listed in ``block_keys`` will be wired to that block's cluster.
    """
    fanout = 8
    node_coords = pg.aux.node_grid_xyz.numpy().astype(np.int32, copy=False)
    grid_to_node = pg.aux.grid_to_node.numpy()
    grid_shape = np.asarray(grid_to_node.shape, dtype=np.int32)
    max_coord = np.asarray(pg.aux.node_shape, dtype=np.int32) - 1

    num_particles = int(pg.model.particle_count)
    parent_indices = np.full((num_particles, fanout), -1, dtype=np.int32)
    parent_weights = np.zeros((num_particles, fanout), dtype=np.float32)
    child_cluster = np.full(num_particles, -1, dtype=np.int32)

    bs = int(block_size)
    max_grid_coord = tuple(int(v) for v in max_coord.tolist())

    def _make_prolongation() -> ShapeMatchingProlongation:
        return ShapeMatchingProlongation(
            parent_indices=wp.array(parent_indices.reshape(-1), dtype=wp.int32, device=target_device),
            parent_weights=wp.array(parent_weights.reshape(-1), dtype=wp.float32, device=target_device),
            child_cluster=wp.array(child_cluster, dtype=wp.int32, device=target_device),
            node_grid_xyz=pg.aux.node_grid_xyz,
            grid_to_node=pg.aux.grid_to_node,
            num_particles=num_particles,
            fanout=fanout,
            block_size=bs,
            max_grid_coord=max_grid_coord,
            parent_indices_host=parent_indices.reshape(-1).copy(),
            parent_weights_host=parent_weights.reshape(-1).copy(),
            child_cluster_host=child_cluster.copy(),
        )

    if num_particles == 0 or block_keys.shape[0] == 0:
        return _make_prolongation()

    # Pack block_keys -> cluster id into a 3D lookup grid for O(1) indexing.
    cluster_grid_shape = (max_coord // bs) + 2  # +2 over-allocates safely
    cluster_grid = np.full(tuple(int(s) for s in cluster_grid_shape), -1, dtype=np.int32)
    cluster_grid[block_keys[:, 0], block_keys[:, 1], block_keys[:, 2]] = block_cluster_ids

    coords = node_coords  # (N, 3) int32
    mod = coords % bs
    is_zero = mod == 0
    # Nodes with mod all-zero are themselves L+1 particles; keep them as direct
    # Outer8 members only.
    fine_mask = ~np.all(is_zero, axis=1)
    if not np.any(fine_mask):
        return _make_prolongation()

    floor_lower = (coords // bs) * bs
    nonzero_axis_valid = (floor_lower + bs) <= max_coord

    # Preserve the previous one-sided choice as the preferred candidate, then
    # fall back to other neighboring blocks that also contain the node.
    above_valid = (coords + bs) <= max_coord
    below_valid = (coords - bs) >= 0
    use_below_preferred = is_zero & ~above_valid & below_valid
    preferred_lower = np.where(use_below_preferred, coords - bs, floor_lower)
    preferred_axis_valid = np.where(is_zero, above_valid | below_valid, nonzero_axis_valid)
    assigned = np.zeros(num_particles, dtype=bool)
    lookup_shape = np.asarray(cluster_grid.shape, dtype=np.int32)

    def _try_candidate(
        candidate_lower: np.ndarray,
        candidate_axis_valid: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> None:
        valid = fine_mask & candidate_mask & ~assigned & np.all(candidate_axis_valid, axis=1)
        if not np.any(valid):
            return

        keep = np.flatnonzero(valid)
        v_lower = candidate_lower[keep].astype(np.int32, copy=False)
        v_upper = v_lower + bs
        v_frac = (coords[keep] - v_lower).astype(np.float32) / float(bs)
        inside = np.all((v_frac >= 0.0) & (v_frac <= 1.0), axis=1)
        if not np.any(inside):
            return

        keep = keep[inside]
        v_lower = v_lower[inside]
        v_upper = v_upper[inside]
        v_frac = v_frac[inside]
        v_block_keys = (v_lower // bs).astype(np.int32, copy=False)
        key_in_lookup = np.all((v_block_keys >= 0) & (v_block_keys < lookup_shape), axis=1)
        if not np.any(key_in_lookup):
            return

        keep = keep[key_in_lookup]
        v_lower = v_lower[key_in_lookup]
        v_upper = v_upper[key_in_lookup]
        v_frac = v_frac[key_in_lookup]
        v_block_keys = v_block_keys[key_in_lookup]
        cluster_idx = cluster_grid[v_block_keys[:, 0], v_block_keys[:, 1], v_block_keys[:, 2]]
        has_cluster = cluster_idx >= 0
        if not np.any(has_cluster):
            return

        keep = keep[has_cluster]
        v_lower = v_lower[has_cluster]
        v_upper = v_upper[has_cluster]
        v_frac = v_frac[has_cluster]
        cluster_idx = cluster_idx[has_cluster]

        # 8 parent coords per kept node.
        is_lower = (CELL_CORNER_OFFSETS == 0)[None, :, :]  # (1, 8, 3) bool
        parent_coords = np.where(is_lower, v_lower[:, None, :], v_upper[:, None, :])  # (V, 8, 3)
        in_bounds = np.all((parent_coords >= 0) & (parent_coords < grid_shape), axis=2)
        safe = np.where(in_bounds[..., None], parent_coords, 0)
        p_idx = grid_to_node[safe[..., 0], safe[..., 1], safe[..., 2]]
        p_idx = np.where(in_bounds, p_idx, -1)
        parent_ok = np.all(p_idx >= 0, axis=1)
        if not np.any(parent_ok):
            return

        keep = keep[parent_ok]
        p_idx = p_idx[parent_ok]
        v_frac = v_frac[parent_ok]
        cluster_idx = cluster_idx[parent_ok]

        # Trilinear weights: w[c] = prod_axis (1 - frac if offset[c, axis]==0 else frac).
        offs_b = CELL_CORNER_OFFSETS[None, :, :]  # (1, 8, 3)
        w_axis = np.where(offs_b == 0, 1.0 - v_frac[:, None, :], v_frac[:, None, :])  # (V, 8, 3)
        weights = np.prod(w_axis, axis=2).astype(np.float32, copy=False)  # (V, 8)

        parent_indices[keep] = p_idx.astype(np.int32, copy=False)
        parent_weights[keep] = weights
        child_cluster[keep] = cluster_idx.astype(np.int32, copy=False)
        assigned[keep] = True

    _try_candidate(preferred_lower, preferred_axis_valid, np.ones(num_particles, dtype=bool))

    has_plane_axis = np.any(is_zero, axis=1)
    if np.any(fine_mask & has_plane_axis & ~assigned):
        for bits in range(8):
            candidate_lower = floor_lower.copy()
            candidate_axis_valid = nonzero_axis_valid.copy()
            for axis in range(3):
                choose_below = (bits & (1 << axis)) != 0
                if choose_below:
                    axis_lower = coords[:, axis] - bs
                    zero_valid = axis_lower >= 0
                else:
                    axis_lower = coords[:, axis]
                    zero_valid = (coords[:, axis] + bs) <= max_coord[axis]
                candidate_lower[:, axis] = np.where(is_zero[:, axis], axis_lower, floor_lower[:, axis])
                candidate_axis_valid[:, axis] = np.where(
                    is_zero[:, axis],
                    zero_valid,
                    nonzero_axis_valid[:, axis],
                )
            different_from_preferred = np.any(candidate_lower != preferred_lower, axis=1)
            _try_candidate(
                candidate_lower,
                candidate_axis_valid,
                has_plane_axis & different_from_preferred,
            )
            if not np.any(fine_mask & has_plane_axis & ~assigned):
                break

    return _make_prolongation()


def build_hierarchical_shape_matching_clusters(
    pg: HexParticleGrid,
    coefficient: float = 1.0,
    device: str | wp.context.Device | None = None,
) -> HierarchicalShapeMatchingClusters:
    """Build L1 and L2 shape-matching variants for runtime comparison."""
    if coefficient < 0.0:
        raise ValueError(f"cluster coefficient must be non-negative, got {coefficient}")

    target_device = wp.get_device(device) if device is not None else pg.model.device
    if target_device != pg.model.device:
        raise ValueError(
            "hierarchical shape-matching buffers must live on the same device as the model; "
            f"got hierarchy on {target_device} and model on {pg.model.device}"
        )

    l1_block_size = 2
    l2_block_size = 4
    cell_coords = pg.aux.cell_grid_xyz.numpy().astype(np.int32, copy=False)
    cell_active = pg.aux.cell_active_host.astype(np.int32, copy=False)
    grid_to_node = pg.aux.grid_to_node.numpy()

    grid_shape_arr = np.asarray(grid_to_node.shape, dtype=np.int32)
    cell_grid_shape_arr = np.asarray(pg.aux.grid_shape, dtype=np.int32)
    active_cell_grid = np.zeros(tuple(int(s) for s in cell_grid_shape_arr), dtype=np.int32)
    active_cell_coords = cell_coords[cell_active != 0]
    active_cell_grid[active_cell_coords[:, 0], active_cell_coords[:, 1], active_cell_coords[:, 2]] = 1
    active_cell_prefix = np.pad(active_cell_grid, ((1, 0), (1, 0), (1, 0))).cumsum(0).cumsum(1).cumsum(2)

    def _block_has_active_cell_halo(block_origins: np.ndarray, block_size: int) -> np.ndarray:
        """True when the coarse cell block has a one-cell active halo.

        Surface-adjacent full blocks are topologically intact but not safe to
        sleep for rendering: clamping their fine boundary nodes to a coarse
        trilinear cage visibly changes the marching-cubes surface.
        """
        count = int(block_origins.shape[0])
        sleepable = np.zeros(count, dtype=np.int32)
        if count == 0:
            return sleepable

        bs = int(block_size)
        lo = block_origins - 1
        hi = block_origins + bs + 1
        valid = np.all((lo >= 0) & (hi <= cell_grid_shape_arr), axis=1)
        if not np.any(valid):
            return sleepable

        lo_v = lo[valid]
        hi_v = hi[valid]
        p = active_cell_prefix
        box_counts = (
            p[hi_v[:, 0], hi_v[:, 1], hi_v[:, 2]]
            - p[lo_v[:, 0], hi_v[:, 1], hi_v[:, 2]]
            - p[hi_v[:, 0], lo_v[:, 1], hi_v[:, 2]]
            - p[hi_v[:, 0], hi_v[:, 1], lo_v[:, 2]]
            + p[lo_v[:, 0], lo_v[:, 1], hi_v[:, 2]]
            + p[lo_v[:, 0], hi_v[:, 1], lo_v[:, 2]]
            + p[hi_v[:, 0], lo_v[:, 1], lo_v[:, 2]]
            - p[lo_v[:, 0], lo_v[:, 1], lo_v[:, 2]]
        )
        sleepable[np.flatnonzero(valid)] = (box_counts == (bs + 2) ** 3).astype(np.int32, copy=False)
        return sleepable

    def _make_level(block_size: int, include_full: bool):
        bs = int(block_size)
        block_volume = bs ** 3

        active_mask = cell_active != 0
        if not np.any(active_mask):
            return None, None, None, None, None

        active_cell_idx = np.flatnonzero(active_mask).astype(np.int32, copy=False)
        active_coords = cell_coords[active_cell_idx]
        active_block_keys = (active_coords // bs).astype(np.int32, copy=False)

        # Group active cells by block key. unique returns rows in lex order so the
        # cluster numbering matches the original "for block_key in sorted(...)" pass.
        unique_keys, inv, counts = np.unique(
            active_block_keys, axis=0, return_inverse=True, return_counts=True
        )
        full_block_mask = counts == block_volume
        if not np.any(full_block_mask):
            return None, None, None, None, None

        # First active cell index in each unique block (lowest cell_idx wins; matches
        # the original which appended in cell-iteration order).
        sort_inv = np.argsort(inv, kind="stable")
        sorted_inv = inv[sort_inv]
        block_starts = np.searchsorted(sorted_inv, np.arange(unique_keys.shape[0]))
        first_cell_in_block = active_cell_idx[sort_inv[block_starts]]

        # Restrict to the "full" blocks (those with a complete bs^3 cells in them).
        full_block_keys = unique_keys[full_block_mask]
        full_source_cells = first_cell_in_block[full_block_mask]
        K = int(full_block_keys.shape[0])
        block_origins = full_block_keys * bs  # (K, 3)
        full_block_sleepable = _block_has_active_cell_halo(block_origins, bs)

        # Per-active-cell, the index into [0..K) for its full-block, or -1 otherwise.
        full_block_id_for_unique = np.full(unique_keys.shape[0], -1, dtype=np.int32)
        full_block_id_for_unique[full_block_mask] = np.arange(K, dtype=np.int32)
        active_full_block_id = full_block_id_for_unique[inv]  # (Na,)

        def _block_colors(block_keys: np.ndarray) -> np.ndarray:
            return (
                (block_keys[:, 0] & 1)
                | ((block_keys[:, 1] & 1) << 1)
                | ((block_keys[:, 2] & 1) << 2)
            ).astype(np.int32, copy=False)

        # ---- outer8: 8 corners of each block at stride bs ----
        outer_offsets = CELL_CORNER_OFFSETS * bs  # (8, 3)
        outer_coords = block_origins[:, None, :] + outer_offsets[None, :, :]  # (K, 8, 3)
        outer_in_bounds = np.all(
            (outer_coords >= 0) & (outer_coords < grid_shape_arr), axis=2
        )
        safe_outer = np.where(outer_in_bounds[..., None], outer_coords, 0)
        outer_lookup = grid_to_node[safe_outer[..., 0], safe_outer[..., 1], safe_outer[..., 2]]
        outer_lookup = np.where(outer_in_bounds, outer_lookup, -1)
        outer_block_valid = np.all(outer_lookup >= 0, axis=1)

        outer8 = None
        outer8_prolongation = None
        outer8_block_keys_host = None
        outer8_sleepable_host = None
        if np.any(outer_block_valid):
            outer_members = outer_lookup[outer_block_valid].astype(np.int32, copy=False)
            outer_block_keys = full_block_keys[outer_block_valid].astype(np.int32, copy=False)
            outer8_block_keys_host = outer_block_keys.copy()
            outer8_sleepable_host = full_block_sleepable[outer_block_valid].astype(np.int32, copy=True)
            outer_sources = full_source_cells[outer_block_valid].astype(np.int32, copy=False)

            # Map [0..K) -> outer cluster id (or -1 for outer-invalid blocks).
            full_to_outer = np.full(K, -1, dtype=np.int32)
            num_outer = int(outer_block_valid.sum())
            full_to_outer[outer_block_valid] = np.arange(num_outer, dtype=np.int32)

            outer_cell_to_cluster = np.full(pg.aux.num_cells, -1, dtype=np.int32)
            cluster_for_active = np.where(
                active_full_block_id >= 0,
                full_to_outer[np.maximum(active_full_block_id, 0)],
                np.int32(-1),
            )
            outer_cell_to_cluster[active_cell_idx] = cluster_for_active

            outer8 = _build_shape_matching_clusters_from_members(
                pg,
                outer_members,
                coefficient=float(coefficient),
                source_cell=outer_sources,
                cell_to_cluster=outer_cell_to_cluster,
                colors=_block_colors(outer_block_keys),
                target_device=target_device,
            )
            outer8_prolongation = _make_outer8_prolongation(
                pg,
                block_size=bs,
                block_keys=outer_block_keys,
                block_cluster_ids=full_to_outer[outer_block_valid].astype(np.int32, copy=False),
                target_device=target_device,
            )

        full = None
        if include_full and K > 0:
            n = bs + 1
            ax = np.arange(n, dtype=np.int32)
            full_offsets = np.stack(
                np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1
            ).reshape(-1, 3)  # (n^3, 3)
            full_coords = block_origins[:, None, :] + full_offsets[None, :, :]
            full_in_bounds = np.all(
                (full_coords >= 0) & (full_coords < grid_shape_arr), axis=2
            )
            safe_full = np.where(full_in_bounds[..., None], full_coords, 0)
            full_lookup = grid_to_node[safe_full[..., 0], safe_full[..., 1], safe_full[..., 2]]
            full_lookup = np.where(full_in_bounds, full_lookup, -1)
            full_block_valid = np.all(full_lookup >= 0, axis=1)

            if np.any(full_block_valid):
                full_members = full_lookup[full_block_valid].astype(np.int32, copy=False)
                full_sources = full_source_cells[full_block_valid].astype(np.int32, copy=False)

                full_to_full = np.full(K, -1, dtype=np.int32)
                num_full = int(full_block_valid.sum())
                full_to_full[full_block_valid] = np.arange(num_full, dtype=np.int32)

                full_cell_to_cluster = np.full(pg.aux.num_cells, -1, dtype=np.int32)
                cluster_for_active = np.where(
                    active_full_block_id >= 0,
                    full_to_full[np.maximum(active_full_block_id, 0)],
                    np.int32(-1),
                )
                full_cell_to_cluster[active_cell_idx] = cluster_for_active

                full = _build_shape_matching_clusters_from_members(
                    pg,
                    full_members,
                    coefficient=float(coefficient),
                    source_cell=full_sources,
                    cell_to_cluster=full_cell_to_cluster,
                    colors=_block_colors(full_block_keys[full_block_valid].astype(np.int32, copy=False)),
                    target_device=target_device,
                )
        return outer8, outer8_prolongation, full, outer8_block_keys_host, outer8_sleepable_host

    outer8, outer8_prolongation, full27, outer8_block_keys_host, outer8_sleepable_host = _make_level(
        l1_block_size,
        include_full=True,
    )
    l2_outer8, l2_outer8_prolongation, l2_full125, l2_outer8_block_keys_host, l2_outer8_sleepable_host = _make_level(
        l2_block_size,
        include_full=True,
    )

    def _make_block_lookup(block_keys_host: np.ndarray | None):
        if block_keys_host is None or block_keys_host.size == 0:
            return None, None
        keys = np.asarray(block_keys_host, dtype=np.int32)
        shape_arr = keys.max(axis=0).astype(np.int32, copy=False) + 1
        shape = (int(shape_arr[0]), int(shape_arr[1]), int(shape_arr[2]))
        lookup = np.full(shape, -1, dtype=np.int32)
        lookup[keys[:, 0], keys[:, 1], keys[:, 2]] = np.arange(keys.shape[0], dtype=np.int32)
        return lookup, shape

    outer8_block_lookup_host, outer8_block_lookup_shape = _make_block_lookup(outer8_block_keys_host)
    l2_outer8_block_lookup_host, l2_outer8_block_lookup_shape = _make_block_lookup(l2_outer8_block_keys_host)

    hierarchy = HierarchicalShapeMatchingClusters(
        outer8=outer8,
        outer8_prolongation=outer8_prolongation,
        full27=full27,
        l2_outer8=l2_outer8,
        l2_outer8_prolongation=l2_outer8_prolongation,
        l2_full125=l2_full125,
        block_size=l1_block_size,
        l2_block_size=l2_block_size,
        outer8_block_keys_host=outer8_block_keys_host,
        l2_outer8_block_keys_host=l2_outer8_block_keys_host,
        outer8_sleepable_host=outer8_sleepable_host,
        l2_outer8_sleepable_host=l2_outer8_sleepable_host,
        outer8_block_keys=(
            None
            if outer8_block_keys_host is None
            else wp.array(outer8_block_keys_host, dtype=wp.int32, device=target_device)
        ),
        l2_outer8_block_keys=(
            None
            if l2_outer8_block_keys_host is None
            else wp.array(l2_outer8_block_keys_host, dtype=wp.int32, device=target_device)
        ),
        outer8_sleepable=(
            None
            if outer8_sleepable_host is None
            else wp.array(outer8_sleepable_host, dtype=wp.int32, device=target_device)
        ),
        l2_outer8_sleepable=(
            None
            if l2_outer8_sleepable_host is None
            else wp.array(l2_outer8_sleepable_host, dtype=wp.int32, device=target_device)
        ),
        outer8_block_lookup=(
            None
            if outer8_block_lookup_host is None
            else wp.array(outer8_block_lookup_host, dtype=wp.int32, device=target_device)
        ),
        l2_outer8_block_lookup=(
            None
            if l2_outer8_block_lookup_host is None
            else wp.array(l2_outer8_block_lookup_host, dtype=wp.int32, device=target_device)
        ),
        outer8_block_lookup_host=outer8_block_lookup_host,
        l2_outer8_block_lookup_host=l2_outer8_block_lookup_host,
        outer8_block_lookup_shape=outer8_block_lookup_shape,
        l2_outer8_block_lookup_shape=l2_outer8_block_lookup_shape,
    )
    pg.aux.shape_matching_hierarchy = hierarchy
    return hierarchy


__all__ = [
    "CELL_CORNER_OFFSETS",
    "HexGridAuxState",
    "HexParticleGrid",
    "HierarchicalShapeMatchingClusters",
    "ShapeMatchingClusters",
    "ShapeMatchingProlongation",
    "build_hex_particle_grid",
    "build_hierarchical_shape_matching_clusters",
    "build_shape_matching_clusters",
]
