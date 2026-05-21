from __future__ import annotations

import numpy as np
import warp as wp
from newton._src.geometry.flags import ParticleFlags

from omnisurg.hex.corner_delete import make_corner_deletion_state
from omnisurg.hex.corner_grid import build_corner_grid, build_corner_shape_matching_clusters
from omnisurg.hex.corner_solver import SolverCornerShapeMatching
from omnisurg.hex.io.digimouse import DigimouseAtlas
from omnisurg.hex.kernels.grab import project_grab_distance_constraints
from omnisurg.hex.kernels.instrument import (
    project_kinematic_sphere_mc_triangle_node_positions,
    project_kinematic_sphere_particle_positions_kernel,
)
from omnisurg.hex.materials import DEFAULT_MATERIALS, MaterialTable
from omnisurg.hex._legacy_corner_app import _select_sphere_drag_particles


def _atlas_from_labels(labels: np.ndarray, voxel: float = 0.01) -> DigimouseAtlas:
    return DigimouseAtlas(
        labels=labels.astype(np.uint8),
        voxel_size=voxel,
        materials=MaterialTable(DEFAULT_MATERIALS),
    )


def _run_contact(
    particle_q_np: np.ndarray,
    particle_inv_mass_np: np.ndarray,
    particle_flags_np: np.ndarray,
    *,
    sphere_prev=(0.0, 0.0, 0.0),
    sphere_current=(0.0, 0.0, 0.0),
    sphere_radius: float = 1.0,
    alpha: float = 1.0,
    relaxation: float = 1.0,
    max_correction: float = 0.0,
):
    device = "cpu"
    particle_count = int(particle_q_np.shape[0])
    particle_q = wp.array(np.asarray(particle_q_np, dtype=np.float32), dtype=wp.vec3, device=device)
    particle_inv_mass = wp.array(np.asarray(particle_inv_mass_np, dtype=np.float32), dtype=float, device=device)
    particle_radius = wp.array(np.full(particle_count, 0.1, dtype=np.float32), dtype=float, device=device)
    particle_flags = wp.array(np.asarray(particle_flags_np, dtype=np.int32), dtype=wp.int32, device=device)
    sphere_q_prev = wp.array(np.asarray([sphere_prev], dtype=np.float32), dtype=wp.vec3, device=device)
    sphere_q = wp.array(np.asarray([sphere_current], dtype=np.float32), dtype=wp.vec3, device=device)

    wp.launch(
        project_kinematic_sphere_particle_positions_kernel,
        dim=particle_count,
        inputs=[
            particle_q,
            particle_inv_mass,
            particle_radius,
            particle_flags,
            sphere_q_prev,
            sphere_q,
            1,
            float(sphere_radius),
            float(alpha),
            float(relaxation),
            float(max_correction),
        ],
        device=device,
    )
    wp.synchronize_device(device)
    return particle_q.numpy()


def test_kinematic_sphere_pushes_penetrating_particle_to_combined_radius():
    active = int(ParticleFlags.ACTIVE)

    q = _run_contact(
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        np.asarray([active], dtype=np.int32),
    )

    assert np.allclose(q[0], [1.1, 0.0, 0.0])


def test_kinematic_sphere_skips_inactive_locked_and_outside_particles():
    active = int(ParticleFlags.ACTIVE)
    initial = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    q = _run_contact(
        initial,
        np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        np.asarray([0, active, active], dtype=np.int32),
    )

    assert np.allclose(q, initial)


def test_kinematic_sphere_uses_interpolated_substep_position():
    active = int(ParticleFlags.ACTIVE)

    q = _run_contact(
        np.asarray([[1.5, 0.0, 0.0]], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        np.asarray([active], dtype=np.int32),
        sphere_prev=(0.0, 0.0, 0.0),
        sphere_current=(2.0, 0.0, 0.0),
        alpha=0.5,
    )

    assert np.allclose(q[0], [2.1, 0.0, 0.0])


def test_kinematic_sphere_caps_position_correction():
    active = int(ParticleFlags.ACTIVE)

    q = _run_contact(
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        np.asarray([active], dtype=np.int32),
        max_correction=0.25,
    )

    assert np.allclose(q[0], [0.25, 0.0, 0.0])


def test_solver_projects_kinematic_spheres_inside_constraint_loop_without_velocity_write():
    pg = build_corner_grid(_atlas_from_labels(np.ones((1, 1, 1), dtype=np.uint8)), device="cpu")
    clusters = build_corner_shape_matching_clusters(pg, device="cpu")
    solver = SolverCornerShapeMatching(
        pg.model,
        clusters,
        iterations=2,
        enable_springs=False,
        enable_shape_matching=False,
        enable_self_collisions=False,
        enable_ground_plane=False,
    )
    pg.model.gravity.assign(np.zeros((1, 3), dtype=np.float32))

    state_in = pg.state
    state_out = pg.model.state()
    initial_q = state_in.particle_q.numpy()
    sphere_center = initial_q[0]
    sphere_q_prev = wp.array(np.asarray([sphere_center], dtype=np.float32), dtype=wp.vec3, device="cpu")
    sphere_q = wp.array(np.asarray([sphere_center], dtype=np.float32), dtype=wp.vec3, device="cpu")

    solver.set_kinematic_sphere_contacts(
        sphere_q_prev,
        sphere_q,
        sphere_count=1,
        sphere_radius=0.05,
        interpolation_alpha=1.0,
        relaxation=1.0,
        iterations=1,
        max_correction=0.0,
    )
    solver.step(state_in, state_out, None, None, 0.01)
    wp.synchronize_device("cpu")

    q = state_out.particle_q.numpy()
    qd = state_out.particle_qd.numpy()
    combined_radius = float(pg.model.particle_radius.numpy()[0] + 0.05)
    assert np.allclose(q[0], sphere_center + np.asarray([combined_radius, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(qd[0], [0.0, 0.0, 0.0])


def test_grab_distance_constraint_preserves_rest_length_to_pull_point():
    device = "cpu"
    active = int(ParticleFlags.ACTIVE)
    particle_q = wp.array(np.asarray([[3.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device=device)
    particle_qd = wp.array(np.asarray([[4.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device=device)
    particle_inv_mass = wp.array(np.asarray([1.0], dtype=np.float32), dtype=float, device=device)
    particle_flags = wp.array(np.asarray([active], dtype=np.int32), dtype=wp.int32, device=device)
    grab_indices = wp.array(np.asarray([0], dtype=np.int32), dtype=wp.int32, device=device)
    grab_offsets = wp.array(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device=device)

    project_grab_distance_constraints(
        particle_q,
        particle_qd,
        particle_inv_mass,
        particle_flags,
        grab_indices,
        grab_offsets,
        1,
        (1.0, 0.0, 0.0),
        1.0,
        device=device,
    )
    wp.synchronize_device(device)

    assert np.allclose(particle_q.numpy()[0], [2.0, 0.0, 0.0])
    assert np.allclose(particle_qd.numpy()[0], [0.0, 0.0, 0.0])


def test_solver_projects_grab_distance_constraints_each_iteration():
    pg = build_corner_grid(_atlas_from_labels(np.ones((1, 1, 1), dtype=np.uint8)), device="cpu")
    clusters = build_corner_shape_matching_clusters(pg, device="cpu")
    solver = SolverCornerShapeMatching(
        pg.model,
        clusters,
        iterations=2,
        enable_springs=False,
        enable_shape_matching=False,
        enable_self_collisions=False,
        enable_ground_plane=False,
    )
    pg.model.gravity.assign(np.zeros((1, 3), dtype=np.float32))

    state_in = pg.state
    state_out = pg.model.state()
    initial_q = state_in.particle_q.numpy()
    target = initial_q[0] + np.asarray([0.1, 0.0, 0.0], dtype=np.float32)
    grab_indices = wp.array(np.asarray([0], dtype=np.int32), dtype=wp.int32, device="cpu")
    grab_offsets = wp.array(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device="cpu")

    solver.set_grab_distance_constraints([(grab_indices, grab_offsets, 1, target, 0.5)])
    solver.step(state_in, state_out, None, None, 0.01)
    wp.synchronize_device("cpu")

    q = state_out.particle_q.numpy()
    qd = state_out.particle_qd.numpy()
    expected = initial_q[0] + np.asarray([0.075, 0.0, 0.0], dtype=np.float32)
    assert np.allclose(q[0], expected, atol=1.0e-6)
    assert np.allclose(qd[0], [0.0, 0.0, 0.0])


def test_kinematic_sphere_mc_triangle_contact_loops_over_all_spheres():
    device = "cpu"
    active = int(ParticleFlags.ACTIVE)
    particle_count = 24
    particle_q = wp.array(np.zeros((particle_count, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    particle_inv_mass = wp.array(np.ones(particle_count, dtype=np.float32), dtype=float, device=device)
    particle_flags = wp.array(np.full(particle_count, active, dtype=np.int32), dtype=wp.int32, device=device)
    cell_nodes = wp.array(
        np.asarray(
            [
                np.arange(0, 8, dtype=np.int32),
                np.arange(8, 16, dtype=np.int32),
                np.arange(16, 24, dtype=np.int32),
            ],
            dtype=np.int32,
        ),
        dtype=wp.int32,
        device=device,
    )
    cell_active = wp.array(np.ones(3, dtype=np.int32), dtype=wp.int32, device=device)
    vertex_pos = wp.array(
        np.asarray(
            [
                [0.5, -0.5, -0.5],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.5, 0.5, -0.5],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        dtype=wp.vec3,
        device=device,
    )
    tri_indices = wp.array(np.asarray([[0, 6, 12]], dtype=np.int32), dtype=wp.int32, device=device)
    sphere_q_prev = wp.array(
        np.asarray([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    sphere_q = wp.array(
        np.asarray([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )
    delta_accumulator = wp.zeros(particle_count, dtype=wp.vec3, device=device)
    delta_counter = wp.zeros(particle_count, dtype=wp.int32, device=device)

    project_kinematic_sphere_mc_triangle_node_positions(
        particle_q,
        particle_inv_mass,
        particle_flags,
        cell_nodes,
        cell_active,
        vertex_pos,
        tri_indices,
        triangle_count=1,
        sphere_q_prev=sphere_q_prev,
        sphere_q=sphere_q,
        sphere_count=2,
        sphere_radius=1.0,
        interpolation_alpha=1.0,
        relaxation=1.0,
        particle_delta_accumulator=delta_accumulator,
        particle_delta_counter=delta_counter,
        iterations=1,
        max_correction=0.0,
        device=device,
    )
    wp.synchronize_device(device)

    q = particle_q.numpy()
    expected_x = np.float32((1.0 - 0.5) / 3.0 / 8.0)
    assert np.allclose(q[:, 0], expected_x)
    assert np.allclose(q[:, 1:], 0.0)


def test_solver_can_project_kinematic_spheres_against_mc_triangles():
    pg = build_corner_grid(_atlas_from_labels(np.ones((1, 1, 1), dtype=np.uint8)), device="cpu")
    clusters = build_corner_shape_matching_clusters(pg, device="cpu")
    solver = SolverCornerShapeMatching(
        pg.model,
        clusters,
        iterations=0,
        enable_springs=False,
        enable_shape_matching=False,
        enable_self_collisions=False,
        enable_ground_plane=False,
    )
    pg.model.gravity.assign(np.zeros((1, 3), dtype=np.float32))

    vertex_pos = wp.array(
        np.asarray(
            [
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [0.5, 0.0, 0.5],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        dtype=wp.vec3,
        device="cpu",
    )
    tri_indices = wp.array(np.asarray([[0, 1, 2]], dtype=np.int32), dtype=wp.int32, device="cpu")
    sphere_q_prev = wp.array(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device="cpu")
    sphere_q = wp.array(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device="cpu")

    solver.set_kinematic_sphere_mc_triangle_contacts(
        sphere_q_prev,
        sphere_q,
        sphere_count=1,
        sphere_radius=1.0,
        interpolation_alpha=1.0,
        relaxation=1.0,
        cell_nodes=pg.aux.cell_nodes,
        cell_active=pg.aux.cell_active,
        vertex_pos=vertex_pos,
        tri_indices=tri_indices,
        triangle_count=1,
        iterations=1,
        max_correction=0.0,
    )

    state_in = pg.state
    state_out = pg.model.state()
    initial_q = state_in.particle_q.numpy()
    solver.step(state_in, state_out, None, None, 0.01)
    wp.synchronize_device("cpu")

    expected_dx = np.float32((1.0 - 0.5) / 3.0 / 8.0)
    q = state_out.particle_q.numpy()
    qd = state_out.particle_qd.numpy()
    assert np.allclose(q[:, 0], initial_q[:, 0] + expected_dx)
    assert np.allclose(q[:, 1:], initial_q[:, 1:])
    assert np.allclose(qd, 0.0)


def test_select_sphere_drag_particles_selects_dynamic_particle_contacts():
    active = int(ParticleFlags.ACTIVE)
    particle_q = np.asarray(
        [
            [0.00, 0.0, 0.0],
            [0.12, 0.0, 0.0],
            [0.20, 0.0, 0.0],
            [0.08, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    particle_radius = np.asarray([0.0, 0.03, 0.0, 0.03], dtype=np.float32)
    particle_flags = np.asarray([active, active, active, 0], dtype=np.int32)
    particle_inv_mass = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    selected, offsets = _select_sphere_drag_particles(
        particle_q=particle_q,
        particle_flags=particle_flags,
        particle_inv_mass=particle_inv_mass,
        particle_radius=particle_radius,
        sphere_center=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        sphere_radius=0.1,
    )

    assert np.array_equal(selected, np.asarray([0, 1], dtype=np.int32))
    assert np.allclose(offsets, particle_q[selected])


def test_select_sphere_drag_particles_skips_locked_particles():
    active = int(ParticleFlags.ACTIVE)
    particle_q = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)

    selected, offsets = _select_sphere_drag_particles(
        particle_q=particle_q,
        particle_flags=np.asarray([active], dtype=np.int32),
        particle_inv_mass=np.asarray([0.0], dtype=np.float32),
        particle_radius=np.asarray([0.0], dtype=np.float32),
        sphere_center=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        sphere_radius=0.1,
    )

    assert selected.size == 0
    assert offsets.shape == (0, 3)


def test_sphere_particle_contact_cut_deletes_cells_with_enabled_sphere():
    pg = build_corner_grid(_atlas_from_labels(np.ones((1, 1, 1), dtype=np.uint8)), device="cpu")
    delete_state = make_corner_deletion_state(pg.model, pg.aux)

    sphere_center = pg.state.particle_q.numpy()[0]
    sphere_q = wp.array(np.asarray([sphere_center], dtype=np.float32), dtype=wp.vec3, device="cpu")
    sphere_cut_enabled = wp.array(np.asarray([1], dtype=np.int32), dtype=wp.int32, device="cpu")

    delete_state.delete_cells_by_spheres_particle_contacts_async(
        pg.state.particle_q,
        sphere_q,
        sphere_cut_enabled,
        1,
        0.05,
    )
    deleted = delete_state.sync_last_deleted()

    assert deleted == 1
    assert np.array_equal(delete_state.last_deleted_cells_host, np.asarray([0], dtype=np.int32))


def test_sphere_particle_contact_cut_ignores_disabled_sphere():
    pg = build_corner_grid(_atlas_from_labels(np.ones((1, 1, 1), dtype=np.uint8)), device="cpu")
    delete_state = make_corner_deletion_state(pg.model, pg.aux)

    sphere_center = pg.state.particle_q.numpy()[0]
    sphere_q = wp.array(np.asarray([sphere_center], dtype=np.float32), dtype=wp.vec3, device="cpu")
    sphere_cut_enabled = wp.array(np.asarray([0], dtype=np.int32), dtype=wp.int32, device="cpu")

    delete_state.delete_cells_by_spheres_particle_contacts_async(
        pg.state.particle_q,
        sphere_q,
        sphere_cut_enabled,
        1,
        0.05,
    )
    deleted = delete_state.sync_last_deleted()

    assert deleted == 0
