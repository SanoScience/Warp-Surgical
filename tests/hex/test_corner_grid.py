# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0
"""Corner-node lattice builder and deletion tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import warp as wp
from newton._src.geometry.flags import ParticleFlags

from omnisurg.hex.corner_delete import make_corner_deletion_state
from omnisurg.hex.corner_grid import (
    CELL_CORNER_OFFSETS,
    CONSTRAINT_BODY_DIAGONAL,
    CONSTRAINT_EDGE,
    CONSTRAINT_FACE_DIAGONAL,
    build_corner_grid,
    build_corner_l1_shape_matching_clusters,
    build_corner_shape_matching_clusters,
)
from omnisurg.hex.io.digimouse import DigimouseAtlas, load_digimouse
from omnisurg.hex.kernels.corner_render import HoverPicker, update_cell_render_state
from omnisurg.hex.materials import DEFAULT_MATERIALS, MUSCLE, SKIN, MaterialTable
from omnisurg.hex.render import CryoTextureAtlas
from omnisurg.hex import _legacy_corner_app


def _atlas_from_labels(labels: np.ndarray, voxel: float = 0.01) -> DigimouseAtlas:
    return DigimouseAtlas(
        labels=labels.astype(np.uint8),
        voxel_size=voxel,
        materials=MaterialTable(DEFAULT_MATERIALS),
    )


def _sparse_outer8_plane_regression_labels() -> np.ndarray:
    labels = np.zeros((5, 4, 6), dtype=np.uint8)
    labels[0:2, 2:4, 4:6] = 1
    labels[4, 0, 0] = 1
    return labels


def _spring_index_from_node_coords(pg, a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    grid_to_node = pg.aux.grid_to_node.numpy()
    na = int(grid_to_node[a])
    nb = int(grid_to_node[b])
    if na > nb:
        na, nb = nb, na
    pairs = pg.model.spring_indices.numpy().reshape(-1, 2)
    for idx, (i, j) in enumerate(pairs.tolist()):
        if int(i) == na and int(j) == nb:
            return idx
    raise KeyError((a, b))


def _cell_index_from_coord(pg, coord: tuple[int, int, int]) -> int:
    coords = pg.aux.cell_grid_xyz.numpy()
    hits = np.nonzero((coords == np.asarray(coord, dtype=np.int32)).all(axis=1))[0]
    if hits.size != 1:
        raise KeyError(coord)
    return int(hits[0])


def _x_ray_through_bounds(points: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) * 0.5
    span = max(float(hi[0] - lo[0]), 1.0e-3)
    origin = (float(lo[0] - span), float(center[1]), float(center[2]))
    direction = (1.0, 0.0, 0.0)
    return origin, direction


def test_single_voxel_corner_grid_counts():
    labels = np.ones((1, 1, 1), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels), edge_ke=10.0, face_ke=20.0, body_ke=30.0)

    assert pg.aux.num_cells == 1
    assert pg.aux.num_nodes == 8
    assert pg.aux.num_springs == 28

    node_mass = pg.model.particle_mass.numpy()
    assert np.allclose(node_mass, 0.125)

    spring_type = pg.aux.spring_type.numpy()
    assert int((spring_type == CONSTRAINT_EDGE).sum()) == 12
    assert int((spring_type == CONSTRAINT_FACE_DIAGONAL).sum()) == 12
    assert int((spring_type == CONSTRAINT_BODY_DIAGONAL).sum()) == 4
    assert np.all(pg.aux.spring_owner_count.numpy() == 1)


def test_hover_picker_dirty_aabb_refresh_matches_full_refresh_after_motion():
    pg = build_corner_grid(_atlas_from_labels(np.ones((1, 1, 1), dtype=np.uint8)))
    picker_dirty = HoverPicker(pg.aux.num_cells, pg.model.device)
    picker_full = HoverPicker(pg.aux.num_cells, pg.model.device)

    moved = pg.model.particle_q.numpy() + np.asarray([0.05, -0.01, 0.015], dtype=np.float32)
    pg.model.particle_q.assign(moved)

    dirty_ids = wp.array(np.asarray([0], dtype=np.int32), dtype=wp.int32, device=pg.model.device)
    dirty_count = wp.array(np.asarray([1], dtype=np.int32), dtype=wp.int32, device=pg.model.device)
    picker_dirty.refresh_dirty_aabbs(pg.aux, pg.model.particle_q, dirty_ids, dirty_count, 1)
    picker_full.refresh_aabbs(pg.aux, pg.model.particle_q)

    ray_origin, ray_direction = _x_ray_through_bounds(moved)
    assert picker_dirty.pick(pg.aux, ray_origin, ray_direction) == 0
    assert picker_full.pick(pg.aux, ray_origin, ray_direction) == 0


def test_hover_picker_zero_dirty_count_keeps_deleted_cell_unpickable():
    pg = build_corner_grid(_atlas_from_labels(np.ones((1, 1, 1), dtype=np.uint8)))
    picker = HoverPicker(pg.aux.num_cells, pg.model.device)
    picker.refresh_aabbs(pg.aux, pg.model.particle_q)

    cell_active = pg.aux.cell_active.numpy()
    cell_active[0] = 0
    pg.aux.cell_active.assign(cell_active)

    dirty_ids = wp.array(np.asarray([0], dtype=np.int32), dtype=wp.int32, device=pg.model.device)
    dirty_count = wp.array(np.asarray([0], dtype=np.int32), dtype=wp.int32, device=pg.model.device)
    picker.refresh_dirty_aabbs(pg.aux, pg.model.particle_q, dirty_ids, dirty_count, 1)

    ray_origin, ray_direction = _x_ray_through_bounds(pg.model.particle_q.numpy())
    assert picker.pick(pg.aux, ray_origin, ray_direction) == -1


def test_hover_picker_fused_render_refresh_updates_pick_bounds_and_render_state():
    pg = build_corner_grid(_atlas_from_labels(np.ones((1, 1, 1), dtype=np.uint8)))
    picker = HoverPicker(pg.aux.num_cells, pg.model.device)

    moved = pg.model.particle_q.numpy() + np.asarray([-0.02, 0.03, 0.01], dtype=np.float32)
    pg.model.particle_q.assign(moved)
    picker.refresh_render_state_and_aabbs(pg.aux, pg.model.particle_q)

    ray_origin, ray_direction = _x_ray_through_bounds(moved)
    assert picker.pick(pg.aux, ray_origin, ray_direction) == 0
    assert np.allclose(pg.aux.cell_center_q.numpy()[0], moved.mean(axis=0), atol=1.0e-7)
    assert int(pg.aux.cell_render_flags.numpy()[0]) & int(ParticleFlags.ACTIVE)


def test_hover_picker_material_filter_skips_front_non_cuttable_cell():
    labels = np.asarray([[[MUSCLE.id]], [[SKIN.id]]], dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels, voxel=0.01))
    picker = HoverPicker(pg.aux.num_cells, pg.model.device)
    picker.refresh_aabbs(pg.aux, pg.model.particle_q)

    muscle_cell = _cell_index_from_coord(pg, (0, 0, 0))
    skin_cell = _cell_index_from_coord(pg, (1, 0, 0))
    ray_origin, ray_direction = _x_ray_through_bounds(pg.model.particle_q.numpy())

    assert picker.pick(pg.aux, ray_origin, ray_direction) == muscle_cell

    cuttable = np.zeros(len(DEFAULT_MATERIALS), dtype=np.int32)
    cuttable[SKIN.id] = 1
    cuttable_wp = wp.array(cuttable, dtype=wp.int32, device=pg.model.device)

    assert picker.pick(pg.aux, ray_origin, ray_direction, material_cuttable=cuttable_wp) == skin_cell


def test_cell_render_state_reports_max_cell_stretch():
    pg = build_corner_grid(_atlas_from_labels(np.ones((1, 1, 1), dtype=np.uint8), voxel=0.01))

    update_cell_render_state(pg.aux, pg.model.particle_q, device=pg.model.device, compute_stretch=True)
    assert np.allclose(pg.aux.cell_stretch.numpy(), 0.0, atol=1.0e-6)

    moved = pg.model.particle_q.numpy()
    node = int(pg.aux.grid_to_node.numpy()[1, 0, 0])
    moved[node, 0] += 0.002
    pg.model.particle_q.assign(moved)

    update_cell_render_state(pg.aux, pg.model.particle_q, device=pg.model.device, compute_stretch=True)
    assert np.allclose(pg.aux.cell_stretch.numpy()[0], 0.2, atol=1.0e-5)


def test_cell_render_state_zeroes_stretch_for_inactive_cells():
    pg = build_corner_grid(_atlas_from_labels(np.ones((1, 1, 1), dtype=np.uint8), voxel=0.01))
    moved = pg.model.particle_q.numpy()
    moved[int(pg.aux.grid_to_node.numpy()[1, 0, 0]), 0] += 0.002
    pg.model.particle_q.assign(moved)
    pg.aux.cell_active.zero_()

    update_cell_render_state(pg.aux, pg.model.particle_q, device=pg.model.device, compute_stretch=True)
    assert np.allclose(pg.aux.cell_stretch.numpy(), 0.0, atol=1.0e-6)


def test_stress_atlas_maps_cold_and_hot_cell_stretch():
    device = wp.get_device()
    atlas = CryoTextureAtlas(max_triangles=1, device=device, tile_size=1)
    tri_indices = wp.array(np.asarray([[0, 6, 12]], dtype=np.int32), dtype=wp.int32, device=device)
    stretch = wp.array(np.asarray([0.0, 0.0, 0.0], dtype=np.float32), dtype=wp.float32, device=device)

    atlas.rebuild_stress(tri_indices, 1, stretch, color_scale=1.0)
    cold = atlas.texture[0, 0].astype(np.int32)

    stretch.assign(np.asarray([1.0, 1.0, 1.0], dtype=np.float32))
    atlas.rebuild_stress(tri_indices, 1, stretch, color_scale=1.0)
    hot = atlas.texture[0, 0].astype(np.int32)

    assert cold[2] > cold[0]
    assert hot[0] > hot[2]


def test_shared_constraints_scale_with_owner_count():
    labels = np.ones((2, 1, 1), dtype=np.uint8)
    pg = build_corner_grid(
        _atlas_from_labels(labels),
        edge_ke=10.0,
        edge_kd=1.0,
        face_ke=20.0,
        face_kd=2.0,
        body_ke=30.0,
        body_kd=3.0,
    )

    shared_edge = _spring_index_from_node_coords(pg, (1, 0, 0), (1, 1, 0))
    shared_face_diag = _spring_index_from_node_coords(pg, (1, 0, 0), (1, 1, 1))
    unique_body = _spring_index_from_node_coords(pg, (0, 0, 0), (1, 1, 1))

    owner_count = pg.aux.spring_owner_count.numpy()
    stiffness = pg.model.spring_stiffness.numpy()
    damping = pg.model.spring_damping.numpy()
    spring_type = pg.aux.spring_type.numpy()

    assert owner_count[shared_edge] == 2
    assert owner_count[shared_face_diag] == 2
    assert owner_count[unique_body] == 1

    assert spring_type[shared_edge] == CONSTRAINT_EDGE
    assert spring_type[shared_face_diag] == CONSTRAINT_FACE_DIAGONAL
    assert spring_type[unique_body] == CONSTRAINT_BODY_DIAGONAL

    assert stiffness[shared_edge] == 20.0
    assert damping[shared_edge] == 2.0
    assert stiffness[shared_face_diag] == 40.0
    assert damping[shared_face_diag] == 4.0
    assert stiffness[unique_body] == 30.0
    assert damping[unique_body] == 3.0


def test_cell_deletion_updates_mass_flags_and_shared_constraints():
    labels = np.ones((2, 1, 1), dtype=np.uint8)
    pg = build_corner_grid(
        _atlas_from_labels(labels),
        edge_ke=10.0,
        edge_kd=1.0,
        face_ke=20.0,
        face_kd=2.0,
        body_ke=30.0,
        body_kd=3.0,
    )
    state = make_corner_deletion_state(pg.model, pg.aux)

    left_cell = _cell_index_from_coord(pg, (0, 0, 0))
    shared_edge = _spring_index_from_node_coords(pg, (1, 0, 0), (1, 1, 0))
    unique_body = _spring_index_from_node_coords(pg, (0, 0, 0), (1, 1, 1))
    grid_to_node = pg.aux.grid_to_node.numpy()
    shared_node = int(grid_to_node[1, 0, 0])
    isolated_node = int(grid_to_node[0, 0, 0])

    deleted = state.delete_cells([left_cell])
    assert deleted == 1
    assert int((state.cell_active != 0).sum()) == 1

    assert state.spring_support_count[shared_edge] == 1
    assert state.spring_stiffness[shared_edge] == 10.0
    assert state.spring_damping[shared_edge] == 1.0
    assert state.spring_support_count[unique_body] == 0
    assert state.spring_stiffness[unique_body] == 0.0

    assert np.isclose(state.node_mass[shared_node], 0.125)
    assert np.isclose(state.node_inv_mass[shared_node], 8.0)
    assert state.node_support_count[shared_node] == 1

    assert state.node_mass[isolated_node] == 0.0
    assert state.node_inv_mass[isolated_node] == 0.0
    assert state.node_support_count[isolated_node] == 0
    assert (state.particle_flags[isolated_node] & int(ParticleFlags.ACTIVE)) == 0


def test_locked_nodes_stay_kinematic_after_cell_deletion():
    labels = np.ones((2, 1, 1), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels))
    state = make_corner_deletion_state(pg.model, pg.aux)

    left_cell = _cell_index_from_coord(pg, (0, 0, 0))
    grid_to_node = pg.aux.grid_to_node.numpy()
    shared_node = int(grid_to_node[1, 0, 0])

    assert state.lock_nodes([shared_node]) == 1
    assert state.node_mass[shared_node] > 0.0
    assert state.node_inv_mass[shared_node] == 0.0
    assert pg.model.particle_mass.numpy()[shared_node] == 0.0
    assert pg.model.particle_inv_mass.numpy()[shared_node] == 0.0

    assert state.delete_cells([left_cell]) == 1
    wp.synchronize_device(pg.model.device)

    assert state.node_support_count[shared_node] == 1
    assert state.node_mass[shared_node] > 0.0
    assert state.node_inv_mass[shared_node] == 0.0
    assert pg.model.particle_mass.numpy()[shared_node] == 0.0
    assert pg.model.particle_inv_mass.numpy()[shared_node] == 0.0
    assert state.locked_node_mask[shared_node]


def test_locked_nodes_stay_kinematic_after_device_cell_deletion():
    labels = np.ones((2, 1, 1), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels))
    state = make_corner_deletion_state(pg.model, pg.aux)

    left_cell = _cell_index_from_coord(pg, (0, 0, 0))
    grid_to_node = pg.aux.grid_to_node.numpy()
    shared_node = int(grid_to_node[1, 0, 0])
    cell_ids = wp.array(np.asarray([left_cell], dtype=np.int32), dtype=wp.int32, device=pg.model.device)

    assert state.lock_nodes([shared_node]) == 1
    assert state.delete_device_cells(cell_ids, 1) == 1
    wp.synchronize_device(pg.model.device)

    assert state.node_support_count[shared_node] == 1
    assert state.node_mass[shared_node] > 0.0
    assert state.node_inv_mass[shared_node] == 0.0
    assert pg.model.particle_mass.numpy()[shared_node] == 0.0
    assert pg.model.particle_inv_mass.numpy()[shared_node] == 0.0
    assert state.locked_node_mask[shared_node]


def test_unlock_all_nodes_restores_supported_masses_after_deletion():
    labels = np.ones((2, 1, 1), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels))
    state = make_corner_deletion_state(pg.model, pg.aux)

    left_cell = _cell_index_from_coord(pg, (0, 0, 0))
    grid_to_node = pg.aux.grid_to_node.numpy()
    deleted_only_node = int(grid_to_node[0, 0, 0])
    shared_node = int(grid_to_node[1, 0, 0])

    assert state.lock_nodes([deleted_only_node, shared_node]) == 2
    assert state.delete_cells([left_cell]) == 1
    assert not state.locked_node_mask[deleted_only_node]
    assert state.locked_node_mask[shared_node]

    assert state.unlock_all_nodes() == 1
    wp.synchronize_device(pg.model.device)

    expected_inv_mass = 1.0 / state.node_mass[shared_node]
    assert not state.locked_node_mask[shared_node]
    assert state.node_inv_mass[shared_node] == pytest.approx(expected_inv_mass)
    assert pg.model.particle_mass.numpy()[shared_node] == pytest.approx(state.node_mass[shared_node])
    assert pg.model.particle_inv_mass.numpy()[shared_node] == pytest.approx(expected_inv_mass)
    assert state.node_inv_mass[deleted_only_node] == 0.0
    assert pg.model.particle_inv_mass.numpy()[deleted_only_node] == 0.0


def test_set_locked_nodes_replaces_lock_set_and_restores_masses():
    labels = np.ones((2, 1, 1), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels))
    state = make_corner_deletion_state(pg.model, pg.aux)

    grid_to_node = pg.aux.grid_to_node.numpy()
    left_outer_node = int(grid_to_node[0, 0, 0])
    shared_node = int(grid_to_node[1, 0, 0])

    assert state.lock_nodes([left_outer_node, shared_node]) == 2
    assert pg.model.particle_mass.numpy()[left_outer_node] == 0.0
    assert pg.model.particle_mass.numpy()[shared_node] == 0.0

    assert state.set_locked_nodes([shared_node]) == 1
    wp.synchronize_device(pg.model.device)

    assert not state.locked_node_mask[left_outer_node]
    assert state.locked_node_mask[shared_node]
    assert pg.model.particle_mass.numpy()[left_outer_node] == pytest.approx(state.node_mass[left_outer_node])
    assert pg.model.particle_inv_mass.numpy()[left_outer_node] == pytest.approx(1.0 / state.node_mass[left_outer_node])
    assert pg.model.particle_mass.numpy()[shared_node] == 0.0
    assert pg.model.particle_inv_mass.numpy()[shared_node] == 0.0

    assert state.set_locked_nodes([]) == 0
    wp.synchronize_device(pg.model.device)

    assert not state.locked_node_mask[shared_node]
    assert pg.model.particle_mass.numpy()[shared_node] == pytest.approx(state.node_mass[shared_node])
    assert pg.model.particle_inv_mass.numpy()[shared_node] == pytest.approx(1.0 / state.node_mass[shared_node])


def test_deletion_state_reset_restores_topology_and_keeps_current_stiffness():
    labels = np.ones((2, 1, 1), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels), edge_ke=10.0, face_ke=20.0, body_ke=30.0)
    state = make_corner_deletion_state(pg.model, pg.aux)
    clusters = build_corner_shape_matching_clusters(pg)

    left_cell = _cell_index_from_coord(pg, (0, 0, 0))
    shared_edge = _spring_index_from_node_coords(pg, (1, 0, 0), (1, 1, 0))
    grid_to_node = pg.aux.grid_to_node.numpy()
    shared_node = int(grid_to_node[1, 0, 0])

    state.set_family_stiffnesses(edge=50.0)
    assert state.lock_nodes([shared_node]) == 1
    assert state.delete_cells([left_cell]) == 1
    assert state.deleted_total == 1
    assert state.locked_node_mask[shared_node]
    assert np.any(clusters.active_host == 0)

    state.reset()
    wp.synchronize_device(pg.model.device)

    assert state.deleted_total == 0
    assert state.last_deleted_count == 0
    assert state.topology_revision == 2
    assert np.all(state.cell_active == 1)
    assert np.all(pg.aux.cell_active.numpy() == 1)
    assert state.node_support_count[shared_node] == 2
    assert not state.locked_node_mask[shared_node]
    assert pg.model.particle_inv_mass.numpy()[shared_node] == pytest.approx(state.node_inv_mass[shared_node])
    assert state.spring_support_count[shared_edge] == 2
    assert state.spring_stiffness[shared_edge] == pytest.approx(100.0)
    assert pg.model.spring_stiffness.numpy()[shared_edge] == pytest.approx(100.0)
    assert np.all(clusters.active_host == 1)
    assert np.all(clusters.active.numpy() == 1)


def test_cell_deletion_updates_shape_matching_cluster_activity_and_weights():
    labels = np.ones((2, 1, 1), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels))
    clusters = build_corner_shape_matching_clusters(pg)
    state = make_corner_deletion_state(pg.model, pg.aux)

    left_cell = _cell_index_from_coord(pg, (0, 0, 0))
    right_cell = _cell_index_from_coord(pg, (1, 0, 0))
    grid_to_node = pg.aux.grid_to_node.numpy()
    shared_node = int(grid_to_node[1, 0, 0])
    deleted_only_node = int(grid_to_node[0, 0, 0])
    survivor_only_node = int(grid_to_node[2, 0, 0])

    deleted = state.delete_cells([left_cell])

    assert deleted == 1
    assert np.array_equal(clusters.source_cell_host, np.asarray([left_cell, right_cell], dtype=np.int32))
    assert np.array_equal(clusters.active_host, np.asarray([0, 1], dtype=np.int32))
    assert clusters.particle_cluster_counts_host[shared_node] == 1
    assert clusters.particle_cluster_counts_host[deleted_only_node] == 0
    assert clusters.particle_cluster_counts_host[survivor_only_node] == 1
    assert clusters.particle_cluster_inv_weights_host[shared_node] == pytest.approx(1.0)
    assert clusters.particle_cluster_inv_weights_host[deleted_only_node] == pytest.approx(0.0)
    assert clusters.particle_cluster_inv_weights_host[survivor_only_node] == pytest.approx(1.0)


def test_cell_deletion_ignores_duplicates_and_updates_device_state():
    labels = np.ones((2, 1, 1), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels), edge_ke=10.0, face_ke=20.0, body_ke=30.0)
    clusters = build_corner_shape_matching_clusters(pg)
    state = make_corner_deletion_state(pg.model, pg.aux)

    left_cell = _cell_index_from_coord(pg, (0, 0, 0))
    right_cell = _cell_index_from_coord(pg, (1, 0, 0))
    shared_edge = _spring_index_from_node_coords(pg, (1, 0, 0), (1, 1, 0))

    deleted = state.delete_cells([left_cell, left_cell, right_cell])
    wp.synchronize_device(pg.model.device)

    assert deleted == 2
    assert np.all(state.cell_active == 0)
    assert np.all(pg.aux.cell_active.numpy() == 0)
    assert state.spring_support_count[shared_edge] == 0
    assert pg.aux.spring_support_count.numpy()[shared_edge] == 0
    assert pg.model.spring_stiffness.numpy()[shared_edge] == pytest.approx(0.0)
    assert np.all(clusters.active_host == 0)
    assert np.all(clusters.active.numpy() == 0)


def test_device_cell_deletion_handles_duplicates_sentinel_and_shape_matching():
    labels = np.ones((2, 1, 1), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels), edge_ke=10.0, face_ke=20.0, body_ke=30.0)
    clusters = build_corner_shape_matching_clusters(pg)
    state = make_corner_deletion_state(pg.model, pg.aux)

    left_cell = _cell_index_from_coord(pg, (0, 0, 0))
    right_cell = _cell_index_from_coord(pg, (1, 0, 0))
    shared_edge = _spring_index_from_node_coords(pg, (1, 0, 0), (1, 1, 0))
    cell_ids = wp.array(
        np.asarray([left_cell, left_cell, 0x7FFFFFFF, right_cell], dtype=np.int32),
        dtype=wp.int32,
        device=pg.model.device,
    )

    deleted = state.delete_device_cells(cell_ids, 4)
    wp.synchronize_device(pg.model.device)

    assert deleted == 2
    assert state.topology_revision == 1
    assert state.last_deleted_count == 2
    assert np.array_equal(
        state.last_deleted_cells_host,
        np.asarray([left_cell, right_cell], dtype=np.int32),
    )
    assert np.array_equal(state.last_deleted_cells_device.numpy(), state.last_deleted_cells_host)
    assert np.all(state.cell_active == 0)
    assert np.all(pg.aux.cell_active.numpy() == 0)
    assert state.spring_support_count[shared_edge] == 0
    assert pg.model.spring_stiffness.numpy()[shared_edge] == pytest.approx(0.0)
    assert np.all(clusters.active_host == 0)
    assert np.all(clusters.active.numpy() == 0)

    deleted_again = state.delete_device_cells(cell_ids, 4)

    assert deleted_again == 0
    assert state.topology_revision == 1
    assert state.last_deleted_count == 0


def test_async_device_cell_deletion_compacts_actual_deletes_and_lazy_syncs():
    labels = np.ones((2, 1, 1), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels), edge_ke=10.0, face_ke=20.0, body_ke=30.0)
    clusters = build_corner_shape_matching_clusters(pg)
    state = make_corner_deletion_state(pg.model, pg.aux)

    left_cell = _cell_index_from_coord(pg, (0, 0, 0))
    right_cell = _cell_index_from_coord(pg, (1, 0, 0))
    cell_ids = wp.array(
        np.asarray([left_cell, left_cell, 0x7FFFFFFF, right_cell], dtype=np.int32),
        dtype=wp.int32,
        device=pg.model.device,
    )

    result = state.delete_device_cells_async(cell_ids, 4)
    wp.synchronize_device(pg.model.device)

    assert not result.host_synced
    assert int(result.deleted_count_device.numpy()[0]) == 2
    assert np.array_equal(
        result.deleted_cells_device.numpy()[:2],
        np.asarray([left_cell, right_cell], dtype=np.int32),
    )
    assert state.last_deleted_count == 0
    assert np.all(state.cell_active == 1)
    assert np.all(pg.aux.cell_active.numpy() == 0)

    state.sync_host_mirrors("all")

    assert result.host_synced
    assert state.last_deleted_count == 2
    assert np.array_equal(state.last_deleted_cells_host, np.asarray([left_cell, right_cell], dtype=np.int32))
    assert np.all(state.cell_active == 0)
    assert np.all(clusters.active_host == 0)


def test_validate_device_state_reports_negative_support_count():
    labels = np.ones((1, 1, 1), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels))
    state = make_corner_deletion_state(pg.model, pg.aux)

    support = pg.aux.node_support_count.numpy()
    support[0] = -1
    pg.aux.node_support_count.assign(support)

    with pytest.raises(RuntimeError, match="device deletion state validation failed"):
        state.validate_device_state()


def test_ray_surface_deletion_removes_intersected_cells_only():
    labels = np.ones((3, 1, 1), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels, voxel=0.01))
    state = make_corner_deletion_state(pg.model, pg.aux)

    middle_cell = _cell_index_from_coord(pg, (1, 0, 0))
    deleted = state.delete_cells_by_ray_surface(
        pg.state.particle_q,
        ray0_origin=np.asarray([0.015, -0.005, 0.005], dtype=np.float32),
        ray0_direction=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        ray1_origin=np.asarray([0.015, 0.015, 0.005], dtype=np.float32),
        ray1_direction=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        depth=0.01,
        padding=0.001,
    )
    wp.synchronize_device(pg.model.device)

    assert deleted == 1
    assert state.last_deleted_count == 1
    assert np.array_equal(state.last_deleted_cells_host, np.asarray([middle_cell], dtype=np.int32))
    assert state.cell_active[middle_cell] == 0
    assert int((state.cell_active != 0).sum()) == 2


def test_ray_surface_deletion_honors_depth_limit():
    labels = np.ones((1, 1, 3), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels, voxel=0.01))
    state = make_corner_deletion_state(pg.model, pg.aux)

    first = _cell_index_from_coord(pg, (0, 0, 0))
    second = _cell_index_from_coord(pg, (0, 0, 1))
    third = _cell_index_from_coord(pg, (0, 0, 2))
    deleted = state.delete_cells_by_ray_surface(
        pg.state.particle_q,
        ray0_origin=np.asarray([0.005, -0.005, 0.005], dtype=np.float32),
        ray0_direction=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        ray1_origin=np.asarray([0.005, 0.015, 0.005], dtype=np.float32),
        ray1_direction=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        depth=0.011,
        padding=0.001,
    )
    wp.synchronize_device(pg.model.device)

    assert deleted == 2
    assert np.array_equal(state.last_deleted_cells_host, np.asarray([first, second], dtype=np.int32))
    assert state.cell_active[first] == 0
    assert state.cell_active[second] == 0
    assert state.cell_active[third] == 1


def test_ray_segment_deletion_starts_at_picked_cell_and_uses_single_batch():
    labels = np.ones((1, 1, 4), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels, voxel=0.01))
    state = make_corner_deletion_state(pg.model, pg.aux)

    first = _cell_index_from_coord(pg, (0, 0, 0))
    second = _cell_index_from_coord(pg, (0, 0, 1))
    third = _cell_index_from_coord(pg, (0, 0, 2))
    fourth = _cell_index_from_coord(pg, (0, 0, 3))
    start_cell = wp.array(np.asarray([second], dtype=np.int32), dtype=wp.int32, device=pg.model.device)

    deleted = state.delete_cells_by_ray_segment_from_cell(
        pg.state.particle_q,
        start_cell,
        ray_direction=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        depth=0.011,
        padding=0.001,
    )
    wp.synchronize_device(pg.model.device)

    assert deleted == 2
    assert state.topology_revision == 1
    assert state.last_deleted_count == 2
    assert set(state.last_deleted_cells_host.tolist()) == {second, third}
    assert state.cell_active[first] == 1
    assert state.cell_active[second] == 0
    assert state.cell_active[third] == 0
    assert state.cell_active[fourth] == 1


def test_ray_segment_deletion_passes_through_non_cuttable_cells():
    labels = np.asarray([[[MUSCLE.id]], [[SKIN.id]], [[MUSCLE.id]], [[SKIN.id]]], dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels, voxel=0.01))
    state = make_corner_deletion_state(pg.model, pg.aux)
    picker = HoverPicker(pg.aux.num_cells, pg.model.device)
    picker.refresh_aabbs(pg.aux, pg.model.particle_q)

    front_muscle = _cell_index_from_coord(pg, (0, 0, 0))
    first_skin = _cell_index_from_coord(pg, (1, 0, 0))
    middle_muscle = _cell_index_from_coord(pg, (2, 0, 0))
    second_skin = _cell_index_from_coord(pg, (3, 0, 0))
    cuttable = np.zeros(len(DEFAULT_MATERIALS), dtype=np.int32)
    cuttable[SKIN.id] = 1
    cuttable_wp = wp.array(cuttable, dtype=wp.int32, device=pg.model.device)
    ray_origin, ray_direction = _x_ray_through_bounds(pg.model.particle_q.numpy())

    start_cell = picker.pick_device(pg.aux, ray_origin, ray_direction, material_cuttable=cuttable_wp)
    assert int(start_cell.numpy()[0]) == first_skin

    deleted = state.delete_cells_by_ray_segment_from_cell(
        pg.state.particle_q,
        start_cell,
        ray_direction=np.asarray(ray_direction, dtype=np.float32),
        depth=0.031,
        material_cuttable=cuttable_wp,
        padding=0.001,
    )
    wp.synchronize_device(pg.model.device)

    assert deleted == 2
    assert set(state.last_deleted_cells_host.tolist()) == {first_skin, second_skin}
    assert state.cell_active[front_muscle] == 1
    assert state.cell_active[first_skin] == 0
    assert state.cell_active[middle_muscle] == 1
    assert state.cell_active[second_skin] == 0


def test_ray_surface_deletion_respects_material_cuttable_mask():
    labels = np.asarray([[[SKIN.id]], [[MUSCLE.id]]], dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels, voxel=0.01))
    state = make_corner_deletion_state(pg.model, pg.aux)

    skin_cell = _cell_index_from_coord(pg, (0, 0, 0))
    muscle_cell = _cell_index_from_coord(pg, (1, 0, 0))
    cuttable = np.zeros(len(DEFAULT_MATERIALS), dtype=np.int32)
    cuttable[SKIN.id] = 1
    deleted = state.delete_cells_by_ray_surface(
        pg.state.particle_q,
        ray0_origin=np.asarray([-0.005, 0.005, 0.005], dtype=np.float32),
        ray0_direction=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        ray1_origin=np.asarray([0.025, 0.005, 0.005], dtype=np.float32),
        ray1_direction=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        depth=0.01,
        material_cuttable=wp.array(cuttable, dtype=wp.int32, device=pg.model.device),
        padding=0.001,
    )
    wp.synchronize_device(pg.model.device)

    assert deleted == 1
    assert np.array_equal(state.last_deleted_cells_host, np.asarray([skin_cell], dtype=np.int32))
    assert state.cell_active[skin_cell] == 0
    assert state.cell_active[muscle_cell] == 1


def test_family_stiffness_update_rebuilds_live_spring_values():
    labels = np.ones((2, 1, 1), dtype=np.uint8)
    pg = build_corner_grid(
        _atlas_from_labels(labels),
        edge_ke=10.0,
        edge_kd=1.0,
        face_ke=20.0,
        face_kd=2.0,
        body_ke=30.0,
        body_kd=3.0,
    )
    state = make_corner_deletion_state(pg.model, pg.aux)

    shared_edge = _spring_index_from_node_coords(pg, (1, 0, 0), (1, 1, 0))
    shared_face = _spring_index_from_node_coords(pg, (1, 0, 0), (1, 1, 1))
    unique_body = _spring_index_from_node_coords(pg, (0, 0, 0), (1, 1, 1))

    state.set_family_stiffnesses(edge=7.0, face=11.0, body=13.0)

    assert state.spring_stiffness[shared_edge] == 14.0
    assert state.spring_stiffness[shared_face] == 22.0
    assert state.spring_stiffness[unique_body] == 13.0


def test_material_stiffness_scale_rebuilds_live_spring_values():
    labels = np.ones((2, 1, 1), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels), edge_ke=10.0, face_ke=20.0, body_ke=30.0)
    state = make_corner_deletion_state(pg.model, pg.aux)

    shared_edge = _spring_index_from_node_coords(pg, (1, 0, 0), (1, 1, 0))
    unique_body = _spring_index_from_node_coords(pg, (0, 0, 0), (1, 1, 1))
    material_scale = np.ones(len(pg.aux.materials), dtype=np.float32)
    material_scale[1] = 0.25

    state.set_material_stiffness_scale(material_scale)

    assert state.spring_stiffness[shared_edge] == 5.0
    assert state.spring_stiffness[unique_body] == 7.5

    state.set_family_stiffnesses(edge=8.0, body=12.0)

    assert state.spring_stiffness[shared_edge] == 4.0
    assert state.spring_stiffness[unique_body] == 3.0


def test_shape_matching_cluster_builder_single_voxel_rest_data():
    labels = np.ones((1, 1, 1), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels))
    clusters = build_corner_shape_matching_clusters(pg, coefficient=0.75)

    assert clusters.num_clusters == 1
    assert clusters.num_memberships == 8
    assert clusters.uniform_size == 8
    assert np.array_equal(clusters.offsets_host, np.asarray([0, 8], dtype=np.int32))
    assert np.array_equal(clusters.indices_host, pg.aux.cell_nodes_host[0])
    assert np.allclose(clusters.coefficients_host, np.asarray([0.75], dtype=np.float32))
    assert np.all(clusters.particle_cluster_counts_host == 1)
    assert np.allclose(clusters.particle_cluster_inv_weights_host, 1.0)
    assert np.array_equal(clusters.particle_cluster_offsets_host, np.arange(9, dtype=np.int32))
    assert np.array_equal(clusters.particle_cluster_indices_host, np.zeros(8, dtype=np.int32))
    assert np.array_equal(clusters.indices_host[clusters.particle_cluster_member_offsets_host], np.arange(8, dtype=np.int32))

    rest_positions = pg.model.particle_q.numpy()[pg.aux.cell_nodes_host[0]]
    expected_center = rest_positions.mean(axis=0)
    expected_local = rest_positions - expected_center[None, :]

    assert np.allclose(clusters.rest_centers_host[0], expected_center)
    assert np.allclose(clusters.rest_local_positions_host.reshape(1, 8, 3)[0], expected_local)


def test_shape_matching_cluster_builder_inverse_membership_csr():
    labels = np.ones((2, 2, 2), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels))
    clusters = build_corner_shape_matching_clusters(pg)

    assert clusters.particle_cluster_offsets_host.shape == (pg.model.particle_count + 1,)
    assert clusters.particle_cluster_indices_host.shape == (clusters.num_memberships,)
    assert clusters.particle_cluster_member_offsets_host.shape == (clusters.num_memberships,)
    assert np.array_equal(
        np.diff(clusters.particle_cluster_offsets_host),
        clusters.particle_cluster_counts_host,
    )

    for particle_idx in range(pg.model.particle_count):
        start = int(clusters.particle_cluster_offsets_host[particle_idx])
        end = int(clusters.particle_cluster_offsets_host[particle_idx + 1])
        member_offsets = clusters.particle_cluster_member_offsets_host[start:end]
        cluster_indices = clusters.particle_cluster_indices_host[start:end]
        assert np.all(clusters.indices_host[member_offsets] == particle_idx)
        assert np.all(cluster_indices == member_offsets // clusters.uniform_size)


def test_l0_shape_matching_cluster_colors_are_conflict_free():
    labels = np.ones((3, 3, 3), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels))
    clusters = build_corner_shape_matching_clusters(pg)

    coords = pg.aux.cell_grid_xyz.numpy()[clusters.source_cell_host]
    expected = (
        (coords[:, 0] & 1)
        | ((coords[:, 1] & 1) << 1)
        | ((coords[:, 2] & 1) << 2)
    ).astype(np.int32)
    assert np.array_equal(clusters.colors_host, expected)
    assert np.array_equal(clusters.colors.numpy(), expected)
    assert np.array_equal(clusters.color_offsets_host, clusters.color_offsets.numpy())
    assert np.array_equal(clusters.color_cluster_indices_host, clusters.color_cluster_indices.numpy())
    assert clusters.color_offsets_host.shape == (9,)
    assert clusters.color_cluster_indices_host.shape == (clusters.num_clusters,)
    assert clusters.color_offsets_host[0] == 0
    assert clusters.color_offsets_host[-1] == clusters.num_clusters
    assert np.all(np.diff(clusters.color_offsets_host) >= 0)
    assert np.array_equal(
        np.sort(clusters.color_cluster_indices_host),
        np.arange(clusters.num_clusters, dtype=np.int32),
    )

    members = clusters.indices_host.reshape(clusters.num_clusters, clusters.uniform_size)
    for color in range(8):
        start = int(clusters.color_offsets_host[color])
        end = int(clusters.color_offsets_host[color + 1])
        color_cluster_indices = clusters.color_cluster_indices_host[start:end]
        assert np.all(clusters.colors_host[color_cluster_indices] == color)
        color_members = members[color_cluster_indices].reshape(-1)
        assert np.unique(color_members).size == color_members.size


def test_l1_shape_matching_builder_emits_outer8_full27_and_prolongation():
    labels = np.ones((2, 2, 2), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels))
    hierarchy = build_corner_l1_shape_matching_clusters(pg)

    assert hierarchy.outer8 is not None
    assert hierarchy.full27 is not None
    assert hierarchy.outer8_prolongation is not None
    assert hierarchy.outer8.num_clusters == 1
    assert hierarchy.outer8.uniform_size == 8
    assert hierarchy.full27.num_clusters == 1
    assert hierarchy.full27.uniform_size == 27
    assert np.array_equal(hierarchy.outer8_block_keys_host, np.asarray([[0, 0, 0]], dtype=np.int32))
    assert np.array_equal(hierarchy.outer8_sleepable_host, np.asarray([0], dtype=np.int32))

    grid_to_node = pg.aux.grid_to_node.numpy()
    expected_outer = grid_to_node[
        CELL_CORNER_OFFSETS[:, 0] * 2,
        CELL_CORNER_OFFSETS[:, 1] * 2,
        CELL_CORNER_OFFSETS[:, 2] * 2,
    ].astype(np.int32)
    assert np.array_equal(hierarchy.outer8.indices_host, expected_outer)

    center_node = int(grid_to_node[1, 1, 1])
    assert hierarchy.outer8_prolongation.child_cluster_host[center_node] == 0
    start = center_node * hierarchy.outer8_prolongation.fanout
    end = start + hierarchy.outer8_prolongation.fanout
    assert np.array_equal(hierarchy.outer8_prolongation.parent_indices_host[start:end], expected_outer)
    assert np.allclose(hierarchy.outer8_prolongation.parent_weights_host[start:end], 1.0 / 8.0)

    corner_node = int(grid_to_node[0, 0, 0])
    assert hierarchy.outer8_prolongation.child_cluster_host[corner_node] == -1


def test_l1_outer8_prolongation_covers_sparse_full27_surface_plane_nodes():
    pg = build_corner_grid(_atlas_from_labels(_sparse_outer8_plane_regression_labels()))
    hierarchy = build_corner_l1_shape_matching_clusters(pg)

    assert hierarchy.outer8 is not None
    assert hierarchy.full27 is not None
    assert hierarchy.outer8_prolongation is not None
    assert np.array_equal(hierarchy.outer8_block_keys_host, np.asarray([[0, 1, 2]], dtype=np.int32))

    grid_to_node = pg.aux.grid_to_node.numpy()
    repaired_node = int(grid_to_node[2, 3, 6])
    assert repaired_node >= 0
    assert hierarchy.outer8_prolongation.child_cluster_host[repaired_node] == 0

    expected_parent_coords = np.asarray([0, 2, 4], dtype=np.int32) + CELL_CORNER_OFFSETS * 2
    expected_parents = grid_to_node[
        expected_parent_coords[:, 0],
        expected_parent_coords[:, 1],
        expected_parent_coords[:, 2],
    ].astype(np.int32)
    start = repaired_node * hierarchy.outer8_prolongation.fanout
    end = start + hierarchy.outer8_prolongation.fanout
    assert np.array_equal(hierarchy.outer8_prolongation.parent_indices_host[start:end], expected_parents)
    assert np.allclose(
        hierarchy.outer8_prolongation.parent_weights_host[start:end],
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0], dtype=np.float32),
    )

    outer_direct = np.zeros(pg.model.particle_count, dtype=bool)
    outer_direct[np.unique(hierarchy.outer8.indices_host)] = True
    full_nodes = np.unique(hierarchy.full27.indices_host)
    covered = outer_direct[full_nodes] | (hierarchy.outer8_prolongation.child_cluster_host[full_nodes] >= 0)
    assert full_nodes[~covered].tolist() == []

    coarse_corner = int(grid_to_node[2, 2, 6])
    assert outer_direct[coarse_corner]
    assert hierarchy.outer8_prolongation.child_cluster_host[coarse_corner] == -1

    outside = _legacy_corner_app._active_cells_outside_cluster_coverage(
        pg.aux.cell_active_host,
        hierarchy.outer8.cell_to_cluster_host,
    )
    assert outside.size == 1
    assert np.array_equal(pg.aux.cell_grid_xyz.numpy()[outside[0]], np.asarray([4, 0, 0], dtype=np.int32))
    state = make_corner_deletion_state(pg.model, pg.aux)
    support_before = int(state.node_support_count[repaired_node])
    assert support_before > 0
    assert state.delete_cells(outside) == 1
    assert int(state.node_support_count[repaired_node]) == support_before
    assert (int(state.particle_flags[repaired_node]) & int(ParticleFlags.ACTIVE)) != 0


def test_l2_shape_matching_builder_emits_outer8_and_prolongation():
    labels = np.ones((4, 4, 4), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels))
    hierarchy = build_corner_l1_shape_matching_clusters(pg)

    assert hierarchy.outer8 is not None
    assert hierarchy.full27 is not None
    assert hierarchy.l2_outer8 is not None
    assert hierarchy.l2_outer8_prolongation is not None
    assert hierarchy.outer8.num_clusters == 8
    assert hierarchy.full27.num_clusters == 8
    assert hierarchy.l2_outer8.num_clusters == 1
    assert hierarchy.l2_outer8.uniform_size == 8
    expected_l1_block_keys = np.stack(
        np.meshgrid(np.arange(2, dtype=np.int32), np.arange(2, dtype=np.int32), np.arange(2, dtype=np.int32), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    assert np.array_equal(hierarchy.outer8_block_keys_host, expected_l1_block_keys)
    assert np.array_equal(hierarchy.l2_outer8_block_keys_host, np.asarray([[0, 0, 0]], dtype=np.int32))
    assert np.all(hierarchy.outer8_sleepable_host == 0)
    assert np.array_equal(hierarchy.l2_outer8_sleepable_host, np.asarray([0], dtype=np.int32))

    grid_to_node = pg.aux.grid_to_node.numpy()
    expected_outer = grid_to_node[
        CELL_CORNER_OFFSETS[:, 0] * 4,
        CELL_CORNER_OFFSETS[:, 1] * 4,
        CELL_CORNER_OFFSETS[:, 2] * 4,
    ].astype(np.int32)
    assert np.array_equal(hierarchy.l2_outer8.indices_host, expected_outer)

    center_node = int(grid_to_node[2, 2, 2])
    assert hierarchy.l2_outer8_prolongation.child_cluster_host[center_node] == 0
    start = center_node * hierarchy.l2_outer8_prolongation.fanout
    end = start + hierarchy.l2_outer8_prolongation.fanout
    assert np.array_equal(hierarchy.l2_outer8_prolongation.parent_indices_host[start:end], expected_outer)
    assert np.allclose(hierarchy.l2_outer8_prolongation.parent_weights_host[start:end], 1.0 / 8.0)


def test_active_cells_outside_l1_cluster_coverage_keeps_complete_2x2x2_block():
    labels = np.ones((3, 3, 3), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels))
    hierarchy = build_corner_l1_shape_matching_clusters(pg)

    assert hierarchy.outer8 is not None
    coords = pg.aux.cell_grid_xyz.numpy()
    expected_outside = np.nonzero(
        ~((coords[:, 0] < 2) & (coords[:, 1] < 2) & (coords[:, 2] < 2))
    )[0].astype(np.int32)

    outside = _legacy_corner_app._active_cells_outside_cluster_coverage(
        pg.aux.cell_active_host,
        hierarchy.outer8.cell_to_cluster_host,
    )

    assert np.array_equal(outside, expected_outside)
    assert outside.size == 19

    state = make_corner_deletion_state(pg.model, pg.aux)
    assert state.delete_cells(outside) == 19
    assert int(np.count_nonzero(state.cell_active)) == 8


def test_active_cells_outside_l2_cluster_coverage_keeps_complete_4x4x4_block():
    labels = np.ones((5, 4, 4), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels))
    hierarchy = build_corner_l1_shape_matching_clusters(pg)

    assert hierarchy.l2_outer8 is not None
    coords = pg.aux.cell_grid_xyz.numpy()
    expected_outside = np.nonzero(coords[:, 0] >= 4)[0].astype(np.int32)

    outside = _legacy_corner_app._active_cells_outside_cluster_coverage(
        pg.aux.cell_active_host,
        hierarchy.l2_outer8.cell_to_cluster_host,
    )

    assert np.array_equal(outside, expected_outside)
    assert outside.size == 16

    state = make_corner_deletion_state(pg.model, pg.aux)
    assert state.delete_cells(outside) == 16
    assert int(np.count_nonzero(state.cell_active)) == 64


def test_l1_l2_sleepable_metadata_requires_active_cell_halo():
    labels = np.ones((9, 9, 9), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels))
    hierarchy = build_corner_l1_shape_matching_clusters(pg)

    l1_key_to_cluster = {tuple(key.tolist()): idx for idx, key in enumerate(hierarchy.outer8_block_keys_host)}
    l2_key_to_cluster = {tuple(key.tolist()): idx for idx, key in enumerate(hierarchy.l2_outer8_block_keys_host)}

    assert hierarchy.outer8_sleepable_host[l1_key_to_cluster[(0, 0, 0)]] == 0
    assert hierarchy.outer8_sleepable_host[l1_key_to_cluster[(1, 1, 1)]] == 1
    assert hierarchy.l2_outer8_sleepable_host[l2_key_to_cluster[(0, 0, 0)]] == 0
    assert hierarchy.l2_outer8_sleepable_host[l2_key_to_cluster[(1, 1, 1)]] == 1


def test_cell_deletion_invalidates_l1_shape_matching_clusters():
    labels = np.ones((2, 2, 2), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels))
    build_corner_shape_matching_clusters(pg)
    hierarchy = build_corner_l1_shape_matching_clusters(pg)
    state = make_corner_deletion_state(pg.model, pg.aux)

    deleted = state.delete_cells([_cell_index_from_coord(pg, (0, 0, 0))])
    wp.synchronize_device(pg.model.device)

    assert deleted == 1
    assert hierarchy.outer8 is not None
    assert hierarchy.full27 is not None
    assert np.array_equal(hierarchy.outer8.active_host, np.asarray([0], dtype=np.int32))
    assert np.array_equal(hierarchy.full27.active_host, np.asarray([0], dtype=np.int32))
    assert np.array_equal(hierarchy.outer8.active.numpy(), hierarchy.outer8.active_host)
    assert np.array_equal(hierarchy.full27.active.numpy(), hierarchy.full27.active_host)
    assert np.all(hierarchy.outer8.particle_cluster_counts_host == 0)
    assert np.all(hierarchy.full27.particle_cluster_counts_host == 0)


def test_cell_deletion_invalidates_l2_shape_matching_clusters():
    labels = np.ones((4, 4, 4), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels))
    build_corner_shape_matching_clusters(pg)
    hierarchy = build_corner_l1_shape_matching_clusters(pg)
    state = make_corner_deletion_state(pg.model, pg.aux)

    deleted = state.delete_cells([_cell_index_from_coord(pg, (0, 0, 0))])
    wp.synchronize_device(pg.model.device)

    assert deleted == 1
    assert hierarchy.l2_outer8 is not None
    assert np.array_equal(hierarchy.l2_outer8.active_host, np.asarray([0], dtype=np.int32))
    assert np.array_equal(hierarchy.l2_outer8.active.numpy(), hierarchy.l2_outer8.active_host)
    assert np.all(hierarchy.l2_outer8.particle_cluster_counts_host == 0)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a second device for mismatch validation")
def test_shape_matching_cluster_builder_rejects_mismatched_device():
    labels = np.ones((1, 1, 1), dtype=np.uint8)
    pg = build_corner_grid(_atlas_from_labels(labels), device="cpu")

    with pytest.raises(ValueError, match="same device as the model"):
        build_corner_shape_matching_clusters(pg, device="cuda:0")


def test_shape_matching_cluster_builder_matches_digimouse_active_cells():
    atlas_path = Path("Digimouse/atlas/atlas/atlas_380x992x208.hdr")
    if not atlas_path.exists():
        pytest.skip("Digimouse atlas not available in the repo workspace")

    atlas = load_digimouse(downsample=64)
    pg = build_corner_grid(atlas)
    clusters = build_corner_shape_matching_clusters(pg)

    assert clusters.num_clusters == pg.aux.num_cells
    assert np.array_equal(clusters.source_cell_host, np.arange(pg.aux.num_cells, dtype=np.int32))
    assert np.all(clusters.active_host == 1)
