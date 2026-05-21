# SPDX-License-Identifier: Apache-2.0
"""Hex particle-grid shape-matching solver tests."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp
from newton._src.geometry.flags import ParticleFlags

from omnisurg.hex.deletion import make_hex_deletion_state
from omnisurg.hex.hex_grid import (
    CELL_CORNER_OFFSETS,
    build_hex_particle_grid,
    build_hierarchical_shape_matching_clusters,
    build_shape_matching_clusters,
)
from omnisurg.hex.shape_matching_solver import (
    HIERARCHICAL_SHAPE_MATCHING_FULL27,
    HIERARCHICAL_SHAPE_MATCHING_OFF,
    HIERARCHICAL_SHAPE_MATCHING_OUTER8,
    SHAPE_MATCHING_GS_WEIGHT_AVERAGED,
    SHAPE_MATCHING_GS_WEIGHT_FULL,
    SHAPE_MATCHING_GS_WEIGHT_SQRT,
    SHAPE_MATCHING_SOLVE_COLORED_GS,
    SHAPE_MATCHING_SOLVE_GATHER,
    SHAPE_MATCHING_SOLVE_SCATTER,
    HexShapeMatchingSolver,
)
from omnisurg.hex.io.digimouse import DigimouseAtlas
from omnisurg.hex.materials import DEFAULT_MATERIALS, MaterialTable


def _atlas_from_labels(labels: np.ndarray, voxel: float = 0.01) -> DigimouseAtlas:
    return DigimouseAtlas(
        labels=labels.astype(np.uint8),
        voxel_size=voxel,
        materials=MaterialTable(DEFAULT_MATERIALS),
    )


def _build_pg(labels: np.ndarray, device: str = "cpu"):
    return build_hex_particle_grid(_atlas_from_labels(labels), device=device)


def _sparse_outer8_plane_regression_labels() -> np.ndarray:
    labels = np.zeros((5, 4, 6), dtype=np.uint8)
    labels[0:2, 2:4, 4:6] = 1
    labels[4, 0, 0] = 1
    return labels


def _zero_gravity(model) -> None:
    gravity = model.gravity.numpy()
    gravity[...] = 0.0
    model.gravity.assign(gravity)


def _identity_quats(count: int) -> np.ndarray:
    return np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (count, 1))


def _max_rigid_residual(rest: np.ndarray, q: np.ndarray) -> float:
    rest_center = rest.mean(axis=0)
    q_center = q.mean(axis=0)
    covariance = (q - q_center).T @ (rest - rest_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    reconstructed = (rest - rest_center) @ rotation.T + q_center
    return float(np.max(np.linalg.norm(q - reconstructed, axis=1)))


def _approx_uniform8_volume(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=np.float32)
    ax = ((p[1] - p[0]) + (p[2] - p[3]) + (p[5] - p[4]) + (p[6] - p[7])) * 0.25
    ay = ((p[3] - p[0]) + (p[2] - p[1]) + (p[7] - p[4]) + (p[6] - p[5])) * 0.25
    az = ((p[4] - p[0]) + (p[5] - p[1]) + (p[6] - p[2]) + (p[7] - p[3])) * 0.25
    return float(np.dot(ax, np.cross(ay, az)))


def _reset_cluster_warm_start(solver: HexShapeMatchingSolver) -> None:
    solver.cluster_rotations.assign(_identity_quats(solver.clusters.num_clusters))
    solver.cluster_translations.assign(solver.clusters.rest_centers_host)


def _make_solver(pg, **kwargs) -> HexShapeMatchingSolver:
    clusters = build_shape_matching_clusters(pg)
    return HexShapeMatchingSolver(pg.model, clusters, **kwargs)


@pytest.mark.parametrize("shape_matching_use_gather", [False, True])
def test_shape_matching_translation_is_a_fixed_point(shape_matching_use_gather: bool):
    pg = _build_pg(np.ones((1, 1, 1), dtype=np.uint8))
    _zero_gravity(pg.model)
    solver = _make_solver(
        pg,
        iterations=6,
        enable_shape_matching=True,
        shape_matching_stiffness=0.9,
        shape_matching_use_gather=shape_matching_use_gather,
    )

    state_in = pg.model.state()
    state_out = pg.model.state()

    translated = pg.model.particle_q.numpy() + np.asarray([0.07, -0.03, 0.05], dtype=np.float32)
    state_in.particle_q.assign(translated)
    state_in.particle_qd.zero_()

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)

    assert np.allclose(state_out.particle_q.numpy(), translated, atol=1.0e-5)


def test_shape_matching_rotation_is_a_fixed_point():
    pg = _build_pg(np.ones((1, 1, 1), dtype=np.uint8))
    _zero_gravity(pg.model)
    solver = _make_solver(
        pg,
        iterations=6,
        enable_shape_matching=True,
        shape_matching_stiffness=0.9,
    )

    rest = pg.model.particle_q.numpy()
    center = rest.mean(axis=0)
    angle = math.radians(35.0)
    rot = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rotated = ((rest - center[None, :]) @ rot.T) + center[None, :]

    state_in = pg.model.state()
    state_out = pg.model.state()
    state_in.particle_q.assign(rotated)
    state_in.particle_qd.zero_()

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)

    assert np.allclose(state_out.particle_q.numpy(), rotated, atol=1.0e-5)


def test_shape_matching_rotation_restores_rigid_shape():
    pg = _build_pg(np.ones((1, 1, 1), dtype=np.uint8))
    _zero_gravity(pg.model)
    solver = _make_solver(
        pg,
        iterations=8,
        enable_shape_matching=True,
        shape_matching_stiffness=0.8,
    )

    rest = pg.model.particle_q.numpy()
    center = rest.mean(axis=0)
    angle = math.radians(35.0)
    rot = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rotated = ((rest - center[None, :]) @ rot.T) + center[None, :]
    distorted = rotated.copy()
    delta = np.asarray([0.01, -0.005, 0.0], dtype=np.float32)
    distorted[0] += delta
    distorted[6] -= delta

    state_0 = pg.model.state()
    state_1 = pg.model.state()
    state_0.particle_q.assign(distorted)
    state_0.particle_qd.zero_()

    for _ in range(8):
        solver.step(state_0, state_1, None, None, 1.0 / 60.0)
        state_0, state_1 = state_1, state_0

    initial_error = _max_rigid_residual(rest, distorted)
    final_error = _max_rigid_residual(rest, state_0.particle_q.numpy())

    assert final_error < 1.0e-6
    assert final_error < initial_error * 1.0e-3


def test_zero_inv_mass_particles_remain_fixed_under_shape_matching():
    pg = _build_pg(np.ones((1, 1, 1), dtype=np.uint8))
    _zero_gravity(pg.model)

    mass = pg.model.particle_mass.numpy()
    inv_mass = pg.model.particle_inv_mass.numpy()
    mass[0] = 0.0
    inv_mass[0] = 0.0
    pg.model.particle_mass.assign(mass)
    pg.model.particle_inv_mass.assign(inv_mass)

    solver = _make_solver(
        pg,
        iterations=8,
        enable_shape_matching=True,
        shape_matching_stiffness=0.85,
    )

    rest = pg.model.particle_q.numpy()
    distorted = rest.copy()
    distorted[1:] += np.asarray([0.02, -0.01, 0.015], dtype=np.float32)

    state_0 = pg.model.state()
    state_1 = pg.model.state()
    state_0.particle_q.assign(distorted)
    state_0.particle_qd.zero_()

    for _ in range(5):
        solver.step(state_0, state_1, None, None, 1.0 / 60.0)
        state_0, state_1 = state_1, state_0

    assert np.allclose(state_0.particle_q.numpy()[0], rest[0], atol=1.0e-7)


def test_shape_matching_toggle_changes_distorted_state():
    pg = _build_pg(np.ones((2, 1, 1), dtype=np.uint8))
    _zero_gravity(pg.model)
    rest = pg.model.particle_q.numpy()
    distorted = rest.copy()
    distorted[3] += np.asarray([0.015, -0.008, 0.01], dtype=np.float32)

    def _run(enable_shape_matching: bool) -> np.ndarray:
        solver = _make_solver(
            pg,
            iterations=6,
            enable_shape_matching=enable_shape_matching,
            shape_matching_stiffness=0.8,
        )
        state_in = pg.model.state()
        state_out = pg.model.state()
        state_in.particle_q.assign(distorted)
        state_in.particle_qd.zero_()
        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        return state_out.particle_q.numpy()

    unconstrained = _run(False)
    shape_only = _run(True)

    assert np.isfinite(unconstrained).all()
    assert np.isfinite(shape_only).all()
    assert np.allclose(unconstrained, distorted)
    assert not np.allclose(shape_only, distorted)


def test_solver_supports_ping_pong_state_swap_pattern():
    pg = _build_pg(np.ones((2, 1, 1), dtype=np.uint8))
    _zero_gravity(pg.model)
    solver = _make_solver(
        pg,
        iterations=6,
        enable_shape_matching=True,
        shape_matching_stiffness=0.7,
    )

    distorted = pg.model.particle_q.numpy().copy()
    distorted[5] += np.asarray([0.02, 0.01, -0.01], dtype=np.float32)

    state_0 = pg.model.state()
    state_1 = pg.model.state()
    state_0.particle_q.assign(distorted)
    state_0.particle_qd.zero_()

    for _ in range(5):
        solver.step(state_0, state_1, None, None, 1.0 / 60.0)
        state_0, state_1 = state_1, state_0

    final_q = state_0.particle_q.numpy()
    assert np.isfinite(final_q).all()
    assert not np.allclose(final_q, distorted)


def test_l0_shape_matching_passes_refine_within_one_solver_iteration():
    pg = _build_pg(np.ones((1, 1, 1), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)

    rest = pg.model.particle_q.numpy()
    distorted = rest.copy()
    distorted[0] += np.asarray([0.012, -0.006, 0.003], dtype=np.float32)
    distorted[6] -= np.asarray([0.012, -0.006, 0.003], dtype=np.float32)

    def _run(shape_matching_passes: int) -> np.ndarray:
        solver = HexShapeMatchingSolver(
            pg.model,
            clusters,
            iterations=1,
            enable_shape_matching=True,
            shape_matching_stiffness=0.5,
            shape_matching_passes=shape_matching_passes,
        )
        state_in = pg.model.state()
        state_out = pg.model.state()
        state_in.particle_q.assign(distorted)
        state_in.particle_qd.zero_()
        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        return state_out.particle_q.numpy()

    one_pass = _run(1)
    two_passes = _run(2)

    assert _max_rigid_residual(rest, two_passes) < _max_rigid_residual(rest, one_pass)


def test_l0_colored_gs_matches_scatter_for_single_cluster():
    pg = _build_pg(np.ones((1, 1, 1), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)

    rest = pg.model.particle_q.numpy()
    distorted = rest.copy()
    distorted[0] += np.asarray([0.012, -0.006, 0.003], dtype=np.float32)
    distorted[6] -= np.asarray([0.012, -0.006, 0.003], dtype=np.float32)

    def _run(mode: int) -> np.ndarray:
        solver = HexShapeMatchingSolver(
            pg.model,
            clusters,
            iterations=1,
            enable_shape_matching=True,
            shape_matching_stiffness=0.5,
            shape_matching_mode=mode,
        )
        state_in = pg.model.state()
        state_out = pg.model.state()
        state_in.particle_q.assign(distorted)
        state_in.particle_qd.zero_()
        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        wp.synchronize_device(pg.model.device)
        return state_out.particle_q.numpy()

    assert np.allclose(
        _run(SHAPE_MATCHING_SOLVE_COLORED_GS),
        _run(SHAPE_MATCHING_SOLVE_SCATTER),
        atol=1.0e-7,
    )


def test_l0_colored_gs_translation_is_a_fixed_point():
    pg = _build_pg(np.ones((2, 2, 2), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=2,
        enable_shape_matching=True,
        shape_matching_stiffness=0.9,
        shape_matching_mode=SHAPE_MATCHING_SOLVE_COLORED_GS,
    )

    state_in = pg.model.state()
    state_out = pg.model.state()
    translated = pg.model.particle_q.numpy() + np.asarray([0.07, -0.03, 0.05], dtype=np.float32)
    state_in.particle_q.assign(translated)
    state_in.particle_qd.zero_()

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)
    wp.synchronize_device(pg.model.device)

    assert np.allclose(state_out.particle_q.numpy(), translated, atol=1.0e-5)


def test_shape_matching_use_gather_alias_reflects_direct_mode_changes():
    pg = _build_pg(np.ones((1, 1, 1), dtype=np.uint8))
    solver = _make_solver(pg, shape_matching_mode=SHAPE_MATCHING_SOLVE_SCATTER)

    assert not solver.shape_matching_use_gather

    solver.shape_matching_mode = SHAPE_MATCHING_SOLVE_GATHER
    assert solver.shape_matching_use_gather

    solver.shape_matching_mode = SHAPE_MATCHING_SOLVE_COLORED_GS
    assert not solver.shape_matching_use_gather

    solver.shape_matching_use_gather = True
    assert solver.shape_matching_mode == SHAPE_MATCHING_SOLVE_GATHER

    solver.shape_matching_use_gather = False
    assert solver.shape_matching_mode == SHAPE_MATCHING_SOLVE_SCATTER


def test_l0_colored_gs_weighting_controls_support_scaling():
    pg = _build_pg(np.ones((2, 2, 2), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    center_node = int(pg.aux.grid_to_node.numpy()[1, 1, 1])

    distorted = pg.model.particle_q.numpy().copy()
    distorted[center_node] += np.asarray([0.012, -0.006, 0.004], dtype=np.float32)

    def _run(weighting: int) -> float:
        solver = HexShapeMatchingSolver(
            pg.model,
            clusters,
            iterations=1,
            enable_ground_plane=False,
            enable_shape_matching=True,
            shape_matching_stiffness=0.5,
            shape_matching_passes=1,
            shape_matching_mode=SHAPE_MATCHING_SOLVE_COLORED_GS,
            shape_matching_gs_weighting=weighting,
        )
        state_in = pg.model.state()
        state_out = pg.model.state()
        state_in.particle_q.assign(distorted)
        state_in.particle_qd.zero_()
        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        wp.synchronize_device(pg.model.device)
        return float(np.linalg.norm(state_out.particle_q.numpy()[center_node] - distorted[center_node]))

    averaged = _run(SHAPE_MATCHING_GS_WEIGHT_AVERAGED)
    sqrt = _run(SHAPE_MATCHING_GS_WEIGHT_SQRT)
    full = _run(SHAPE_MATCHING_GS_WEIGHT_FULL)

    assert sqrt > averaged
    assert full > sqrt


def test_l0_colored_gs_support_alpha_controls_support_scaling():
    pg = _build_pg(np.ones((2, 2, 2), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    center_node = int(pg.aux.grid_to_node.numpy()[1, 1, 1])

    distorted = pg.model.particle_q.numpy().copy()
    distorted[center_node] += np.asarray([0.012, -0.006, 0.004], dtype=np.float32)

    def _run(alpha: float) -> float:
        solver = HexShapeMatchingSolver(
            pg.model,
            clusters,
            iterations=1,
            enable_ground_plane=False,
            enable_shape_matching=True,
            shape_matching_stiffness=0.5,
            shape_matching_passes=1,
            shape_matching_mode=SHAPE_MATCHING_SOLVE_COLORED_GS,
            shape_matching_gs_support_alpha=alpha,
        )
        state_in = pg.model.state()
        state_out = pg.model.state()
        state_in.particle_q.assign(distorted)
        state_in.particle_qd.zero_()
        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        wp.synchronize_device(pg.model.device)
        return float(np.linalg.norm(state_out.particle_q.numpy()[center_node] - distorted[center_node]))

    averaged = _run(1.0)
    sqrt = _run(0.5)
    quarter = _run(0.25)
    full = _run(0.0)

    assert sqrt > averaged
    assert quarter > sqrt
    assert full > quarter


def test_l0_colored_gs_velocity_clamp_is_position_consistent():
    pg = _build_pg(np.ones((2, 2, 2), dtype=np.uint8))
    _zero_gravity(pg.model)
    pg.model.particle_max_velocity = 0.025
    clusters = build_shape_matching_clusters(pg)
    center_node = int(pg.aux.grid_to_node.numpy()[1, 1, 1])

    distorted = pg.model.particle_q.numpy().copy()
    distorted[center_node] += np.asarray([0.08, -0.04, 0.03], dtype=np.float32)
    dt = 1.0 / 60.0

    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=1,
        enable_ground_plane=False,
        enable_shape_matching=True,
        shape_matching_stiffness=1.0,
        shape_matching_passes=1,
        shape_matching_mode=SHAPE_MATCHING_SOLVE_COLORED_GS,
        shape_matching_gs_support_alpha=0.0,
    )
    state_in = pg.model.state()
    state_out = pg.model.state()
    state_in.particle_q.assign(distorted)
    state_in.particle_qd.zero_()

    solver.step(state_in, state_out, None, None, dt)
    wp.synchronize_device(pg.model.device)

    q = state_out.particle_q.numpy()
    qd = state_out.particle_qd.numpy()
    displacement = q - distorted
    displacement_norm = np.linalg.norm(displacement, axis=1)
    active = (pg.model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) != 0

    assert np.max(displacement_norm[active]) <= pg.model.particle_max_velocity * dt + 1.0e-7
    assert np.any(displacement_norm[active] > 0.5 * pg.model.particle_max_velocity * dt)
    assert np.allclose(qd[active], displacement[active] / dt, atol=1.0e-6)


def test_l0_shape_matching_relaxation_scales_jacobi_stiffness():
    pg = _build_pg(np.ones((1, 1, 1), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)

    distorted = pg.model.particle_q.numpy().copy()
    distorted[0] += np.asarray([0.012, -0.006, 0.003], dtype=np.float32)
    distorted[6] -= np.asarray([0.012, -0.006, 0.003], dtype=np.float32)

    def _run(stiffness: float, relaxation: float) -> np.ndarray:
        solver = HexShapeMatchingSolver(
            pg.model,
            clusters,
            iterations=1,
            enable_shape_matching=True,
            shape_matching_stiffness=stiffness,
            shape_matching_relaxation=relaxation,
            shape_matching_mode=SHAPE_MATCHING_SOLVE_SCATTER,
        )
        state_in = pg.model.state()
        state_out = pg.model.state()
        state_in.particle_q.assign(distorted)
        state_in.particle_qd.zero_()
        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        wp.synchronize_device(pg.model.device)
        return state_out.particle_q.numpy()

    assert np.allclose(_run(0.25, 2.0), _run(0.5, 1.0), atol=1.0e-7)


@pytest.mark.parametrize(
    ("level_name", "shape", "mode_key"),
    [
        ("l1", (2, 2, 2), "hierarchical_shape_matching_mode"),
        ("l2", (4, 4, 4), "l2_hierarchical_shape_matching_mode"),
    ],
)
def test_hierarchical_shape_matching_relaxation_scales_stiffness(
    level_name: str,
    shape: tuple[int, int, int],
    mode_key: str,
):
    pg = _build_pg(np.ones(shape, dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)

    distorted = pg.model.particle_q.numpy().copy()
    distorted[0] += np.asarray([0.012, -0.006, 0.003], dtype=np.float32)
    distorted[-1] -= np.asarray([0.012, -0.006, 0.003], dtype=np.float32)

    def _run(stiffness: float, relaxation: float) -> np.ndarray:
        kwargs = {
            "iterations": 0,
            "enable_shape_matching": True,
            "shape_matching_stiffness": 0.0,
            "hierarchy": hierarchy,
            mode_key: HIERARCHICAL_SHAPE_MATCHING_FULL27,
        }
        if level_name == "l1":
            kwargs.update(
                {
                    "hierarchical_shape_matching_stiffness": stiffness,
                    "hierarchical_shape_matching_relaxation": relaxation,
                    "hierarchical_shape_matching_passes": 1,
                }
            )
        else:
            kwargs.update(
                {
                    "l2_hierarchical_shape_matching_stiffness": stiffness,
                    "l2_hierarchical_shape_matching_relaxation": relaxation,
                    "l2_hierarchical_shape_matching_passes": 1,
                }
            )
        solver = HexShapeMatchingSolver(pg.model, clusters, **kwargs)
        state_in = pg.model.state()
        state_out = pg.model.state()
        state_in.particle_q.assign(distorted)
        state_in.particle_qd.zero_()
        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        wp.synchronize_device(pg.model.device)
        return state_out.particle_q.numpy()

    assert np.allclose(_run(0.25, 2.0), _run(0.5, 1.0), atol=1.0e-7)


def test_l0_volume_preservation_expands_compressed_cell():
    pg = _build_pg(np.ones((1, 1, 1), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=1,
        enable_ground_plane=False,
        enable_shape_matching=False,
        enable_volume_preservation=True,
        volume_preservation_stiffness=0.5,
        volume_preservation_passes=1,
    )

    rest = pg.model.particle_q.numpy()
    center = rest.mean(axis=0)
    compressed = rest.copy()
    compressed[:, 2] = center[2] + (compressed[:, 2] - center[2]) * 0.5

    state_in = pg.model.state()
    state_out = pg.model.state()
    state_in.particle_q.assign(compressed)
    state_in.particle_qd.zero_()

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)
    wp.synchronize_device(pg.model.device)

    final_q = state_out.particle_q.numpy()
    rest_volume = _approx_uniform8_volume(rest[clusters.indices_host[:8]])
    initial_volume = _approx_uniform8_volume(compressed[clusters.indices_host[:8]])
    final_volume = _approx_uniform8_volume(final_q[clusters.indices_host[:8]])

    assert initial_volume < rest_volume
    assert final_volume > initial_volume
    assert abs(rest_volume - final_volume) < abs(rest_volume - initial_volume)
    assert np.allclose(final_q.mean(axis=0), compressed.mean(axis=0), atol=1.0e-7)


def test_l0_volume_preservation_keeps_equal_volume_shear_fixed():
    pg = _build_pg(np.ones((1, 1, 1), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=1,
        enable_ground_plane=False,
        enable_shape_matching=False,
        enable_volume_preservation=True,
        volume_preservation_stiffness=0.8,
        volume_preservation_passes=2,
    )

    rest = pg.model.particle_q.numpy()
    sheared = rest.copy()
    sheared[:, 0] += 0.4 * sheared[:, 2]

    state_in = pg.model.state()
    state_out = pg.model.state()
    state_in.particle_q.assign(sheared)
    state_in.particle_qd.zero_()

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)
    wp.synchronize_device(pg.model.device)

    assert np.allclose(state_out.particle_q.numpy(), sheared, atol=1.0e-7)


@pytest.mark.parametrize("hierarchical_mode", [HIERARCHICAL_SHAPE_MATCHING_OUTER8, HIERARCHICAL_SHAPE_MATCHING_FULL27])
@pytest.mark.parametrize("hierarchical_shape_matching_use_gs", [False, True])
def test_hierarchical_shape_matching_translation_is_a_fixed_point(
    hierarchical_mode: int,
    hierarchical_shape_matching_use_gs: bool,
):
    pg = _build_pg(np.ones((2, 2, 2), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=0,
        enable_shape_matching=True,
        shape_matching_stiffness=0.0,
        hierarchy=hierarchy,
        hierarchical_shape_matching_mode=hierarchical_mode,
        hierarchical_shape_matching_stiffness=0.9,
        hierarchical_shape_matching_passes=1,
        hierarchical_shape_matching_use_gs=hierarchical_shape_matching_use_gs,
    )

    state_in = pg.model.state()
    state_out = pg.model.state()
    translated = pg.model.particle_q.numpy() + np.asarray([0.07, -0.03, 0.05], dtype=np.float32)
    state_in.particle_q.assign(translated)
    state_in.particle_qd.zero_()

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)
    wp.synchronize_device(pg.model.device)

    assert np.allclose(state_out.particle_q.numpy(), translated, atol=1.0e-5)


def test_hierarchical_shape_matching_clusters_are_colored_by_block_parity():
    pg = _build_pg(np.ones((8, 4, 4), dtype=np.uint8))
    hierarchy = build_hierarchical_shape_matching_clusters(pg)

    assert hierarchy.outer8 is not None
    assert hierarchy.l2_outer8 is not None
    assert hierarchy.l2_full125 is not None
    assert set(hierarchy.outer8.colors_host.tolist()) == set(range(8))
    assert set(hierarchy.l2_outer8.colors_host.tolist()) == {0, 1}
    assert set(hierarchy.l2_full125.colors_host.tolist()) == {0, 1}

    for clusters in (hierarchy.outer8, hierarchy.l2_outer8, hierarchy.l2_full125):
        for color in range(8):
            start = int(clusters.color_offsets_host[color])
            end = int(clusters.color_offsets_host[color + 1])
            seen: set[int] = set()
            for cluster_idx in clusters.color_cluster_indices_host[start:end]:
                members = clusters.indices_host[
                    int(clusters.offsets_host[cluster_idx]) : int(clusters.offsets_host[cluster_idx + 1])
                ]
                assert seen.isdisjoint(int(member) for member in members)
                seen.update(int(member) for member in members)


@pytest.mark.parametrize(
    ("shape", "level_name", "shared_x"),
    [
        ((4, 2, 2), "l1", 2),
        ((8, 4, 4), "l2", 4),
    ],
)
def test_hierarchical_colored_gs_toggles_change_shared_boundary_result(
    shape: tuple[int, int, int],
    level_name: str,
    shared_x: int,
):
    def _run(use_gs: bool) -> np.ndarray:
        pg = _build_pg(np.ones(shape, dtype=np.uint8))
        _zero_gravity(pg.model)
        clusters = build_shape_matching_clusters(pg)
        hierarchy = build_hierarchical_shape_matching_clusters(pg)
        kwargs = {
            "iterations": 0,
            "enable_shape_matching": True,
            "shape_matching_stiffness": 0.0,
            "hierarchy": hierarchy,
        }
        if level_name == "l1":
            kwargs.update(
                {
                    "hierarchical_shape_matching_mode": HIERARCHICAL_SHAPE_MATCHING_OUTER8,
                    "hierarchical_shape_matching_stiffness": 0.9,
                    "hierarchical_shape_matching_passes": 1,
                    "hierarchical_shape_matching_use_gs": use_gs,
                }
            )
        else:
            kwargs.update(
                {
                    "l2_hierarchical_shape_matching_mode": HIERARCHICAL_SHAPE_MATCHING_OUTER8,
                    "l2_hierarchical_shape_matching_stiffness": 0.9,
                    "l2_hierarchical_shape_matching_passes": 1,
                    "l2_hierarchical_shape_matching_use_gs": use_gs,
                }
            )
        solver = HexShapeMatchingSolver(pg.model, clusters, **kwargs)

        distorted = pg.model.particle_q.numpy().copy()
        grid_to_node = pg.aux.grid_to_node.numpy()
        distorted[int(grid_to_node[shared_x, 0, 0])] += np.asarray([0.03, -0.015, 0.01], dtype=np.float32)
        distorted[int(grid_to_node[shared_x, 1, 1])] += np.asarray([-0.02, 0.01, -0.015], dtype=np.float32)

        state_in = pg.model.state()
        state_out = pg.model.state()
        state_in.particle_q.assign(distorted)
        state_in.particle_qd.zero_()

        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        wp.synchronize_device(pg.model.device)
        return state_out.particle_q.numpy()

    jacobi_q = _run(False)
    gs_q = _run(True)

    assert float(np.max(np.abs(jacobi_q - gs_q))) > 1.0e-5


@pytest.mark.parametrize(
    ("level_name", "shape", "fine_node"),
    [
        ("l1", (2, 2, 2), (1, 0, 0)),
        ("l2", (4, 4, 4), (1, 0, 0)),
    ],
)
def test_outer8_hierarchy_prolongation_toggle_controls_fine_nodes(
    level_name: str,
    shape: tuple[int, int, int],
    fine_node: tuple[int, int, int],
):
    def _run(use_prolongation: bool) -> np.ndarray:
        pg = _build_pg(np.ones(shape, dtype=np.uint8))
        _zero_gravity(pg.model)
        clusters = build_shape_matching_clusters(pg)
        hierarchy = build_hierarchical_shape_matching_clusters(pg)
        kwargs = {
            "iterations": 0,
            "enable_shape_matching": True,
            "shape_matching_stiffness": 0.0,
            "hierarchy": hierarchy,
        }
        if level_name == "l1":
            kwargs.update(
                {
                    "hierarchical_shape_matching_mode": HIERARCHICAL_SHAPE_MATCHING_OUTER8,
                    "hierarchical_shape_matching_stiffness": 0.9,
                    "hierarchical_shape_matching_passes": 1,
                    "hierarchical_shape_matching_outer8_prolongation": use_prolongation,
                }
            )
        else:
            kwargs.update(
                {
                    "l2_hierarchical_shape_matching_mode": HIERARCHICAL_SHAPE_MATCHING_OUTER8,
                    "l2_hierarchical_shape_matching_stiffness": 0.9,
                    "l2_hierarchical_shape_matching_passes": 1,
                    "l2_hierarchical_shape_matching_outer8_prolongation": use_prolongation,
                }
            )
        solver = HexShapeMatchingSolver(pg.model, clusters, **kwargs)

        grid_to_node = pg.aux.grid_to_node.numpy()
        corner_node = int(grid_to_node[0, 0, 0])
        distorted = pg.model.particle_q.numpy().copy()
        distorted[corner_node] += np.asarray([0.02, -0.01, 0.005], dtype=np.float32)

        state_in = pg.model.state()
        state_out = pg.model.state()
        state_in.particle_q.assign(distorted)
        state_in.particle_qd.zero_()

        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        wp.synchronize_device(pg.model.device)
        return distorted, state_out.particle_q.numpy(), int(grid_to_node[fine_node])

    distorted_off, final_off, fine_idx = _run(False)
    distorted_on, final_on, fine_idx_on = _run(True)

    assert fine_idx == fine_idx_on
    assert np.allclose(final_off[fine_idx], distorted_off[fine_idx], atol=1.0e-7)
    assert np.linalg.norm(final_on[fine_idx] - distorted_on[fine_idx]) > 1.0e-7


def test_hierarchical_shape_matching_runs_inside_solver_iterations():
    pg = _build_pg(np.ones((2, 2, 2), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=3,
        enable_shape_matching=True,
        shape_matching_stiffness=0.0,
        hierarchy=hierarchy,
        hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_OUTER8,
        hierarchical_shape_matching_stiffness=1.0,
    )

    calls = 0

    def _count_hierarchy_call(current_q, current_qd, next_q, next_qd, dt):
        nonlocal calls
        calls += 1
        return current_q, current_qd, next_q, next_qd

    solver._run_hierarchical_shape_matching = _count_hierarchy_call

    state_in = pg.model.state()
    state_out = pg.model.state()
    solver.step(state_in, state_out, None, None, 1.0 / 60.0)

    assert calls == 3


def test_l0_scatter_clears_hierarchy_delta_buffer_before_solving():
    pg = _build_pg(np.ones((2, 2, 2), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=1,
        enable_shape_matching=True,
        shape_matching_stiffness=1.0,
        hierarchy=hierarchy,
        hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_OUTER8,
        hierarchical_shape_matching_stiffness=1.0,
    )

    stale_delta = np.full((pg.model.particle_count, 3), [0.003, -0.002, 0.001], dtype=np.float32)

    def _leave_stale_hierarchy_delta(current_q, current_qd, next_q, next_qd, dt):
        solver.particle_deltas.assign(stale_delta)
        return current_q, current_qd, next_q, next_qd

    solver._run_hierarchical_shape_matching = _leave_stale_hierarchy_delta

    state_in = pg.model.state()
    state_out = pg.model.state()
    rest_q = pg.model.particle_q.numpy().copy()
    state_in.particle_q.assign(rest_q)
    state_in.particle_qd.zero_()

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)
    wp.synchronize_device(pg.model.device)

    assert np.allclose(state_out.particle_q.numpy(), rest_q, atol=1.0e-7)


@pytest.mark.parametrize("shape_matching_use_computed_prolongation", [False, True])
@pytest.mark.parametrize(
    ("level_name", "shape", "fine_node"),
    [
        ("l1", (2, 2, 2), (1, 1, 1)),
        ("l2", (4, 4, 4), (1, 1, 1)),
    ],
)
def test_outer8_absolute_projection_snaps_distorted_fine_node_to_corner_interpolation(
    shape_matching_use_computed_prolongation: bool,
    level_name: str,
    shape: tuple[int, int, int],
    fine_node: tuple[int, int, int],
):
    pg = _build_pg(np.ones(shape, dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    grid_to_node = pg.aux.grid_to_node.numpy()
    fine_idx = int(grid_to_node[fine_node])
    prolongation = hierarchy.outer8_prolongation if level_name == "l1" else hierarchy.l2_outer8_prolongation
    assert prolongation is not None
    assert prolongation.child_cluster_host[fine_idx] >= 0

    start = fine_idx * prolongation.fanout
    end = start + prolongation.fanout
    parent_indices = prolongation.parent_indices_host[start:end]
    parent_weights = prolongation.parent_weights_host[start:end]
    assert np.all(parent_indices >= 0)
    assert fine_idx not in set(parent_indices.tolist())

    distorted = pg.model.particle_q.numpy().copy()
    distorted[fine_idx] += np.asarray([0.017, -0.011, 0.013], dtype=np.float32)
    expected = np.sum(distorted[parent_indices] * parent_weights[:, None], axis=0)
    assert np.linalg.norm(distorted[fine_idx] - expected) > 1.0e-5

    def _run(absolute_projection: bool) -> np.ndarray:
        kwargs = {
            "iterations": 0,
            "enable_shape_matching": True,
            "shape_matching_stiffness": 0.0,
            "shape_matching_use_computed_prolongation": shape_matching_use_computed_prolongation,
            "hierarchy": hierarchy,
        }
        if level_name == "l1":
            kwargs.update(
                {
                    "hierarchical_shape_matching_mode": HIERARCHICAL_SHAPE_MATCHING_OUTER8,
                    "hierarchical_shape_matching_stiffness": 0.9,
                    "hierarchical_shape_matching_passes": 1,
                    "hierarchical_shape_matching_outer8_prolongation": True,
                    "hierarchical_shape_matching_outer8_absolute_projection": absolute_projection,
                }
            )
        else:
            kwargs.update(
                {
                    "l2_hierarchical_shape_matching_mode": HIERARCHICAL_SHAPE_MATCHING_OUTER8,
                    "l2_hierarchical_shape_matching_stiffness": 0.9,
                    "l2_hierarchical_shape_matching_passes": 1,
                    "l2_hierarchical_shape_matching_outer8_prolongation": True,
                    "l2_hierarchical_shape_matching_outer8_absolute_projection": absolute_projection,
                }
            )
        solver = HexShapeMatchingSolver(pg.model, clusters, **kwargs)

        state_in = pg.model.state()
        state_out = pg.model.state()
        state_in.particle_q.assign(distorted)
        state_in.particle_qd.zero_()

        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        wp.synchronize_device(pg.model.device)
        return state_out.particle_q.numpy()

    delta_q = _run(False)
    absolute_q = _run(True)

    assert np.allclose(delta_q[fine_idx], distorted[fine_idx], atol=1.0e-7)
    assert np.allclose(absolute_q[fine_idx], expected, atol=1.0e-6)


def test_outer8_hierarchy_prolongates_corner_correction_to_interior_node():
    pg = _build_pg(np.ones((2, 2, 2), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=0,
        enable_shape_matching=True,
        shape_matching_stiffness=1.0,
        hierarchy=hierarchy,
        hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_OUTER8,
        hierarchical_shape_matching_stiffness=0.9,
        hierarchical_shape_matching_passes=1,
    )

    grid_to_node = pg.aux.grid_to_node.numpy()
    corner_node = int(grid_to_node[0, 0, 0])
    edge_node = int(grid_to_node[1, 0, 0])
    distorted = pg.model.particle_q.numpy().copy()
    distorted[corner_node] += np.asarray([0.02, -0.01, 0.005], dtype=np.float32)

    state_in = pg.model.state()
    state_out = pg.model.state()
    state_in.particle_q.assign(distorted)
    state_in.particle_qd.zero_()

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)
    wp.synchronize_device(pg.model.device)

    final_q = state_out.particle_q.numpy()
    assert np.linalg.norm(final_q[edge_node] - distorted[edge_node]) > 1.0e-7


@pytest.mark.parametrize("absolute_projection", [False, True])
def test_l1_outer8_surface_plane_node_prolongates_from_selected_neighbor_block(absolute_projection: bool):
    pg = _build_pg(_sparse_outer8_plane_regression_labels())
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)

    grid_to_node = pg.aux.grid_to_node.numpy()
    fine_node = int(grid_to_node[2, 3, 6])
    assert hierarchy.outer8_prolongation is not None
    assert hierarchy.outer8_prolongation.child_cluster_host[fine_node] >= 0
    start = fine_node * hierarchy.outer8_prolongation.fanout
    end = start + hierarchy.outer8_prolongation.fanout
    parent_indices = hierarchy.outer8_prolongation.parent_indices_host[start:end]
    parent_weights = hierarchy.outer8_prolongation.parent_weights_host[start:end]
    parent_node = int(parent_indices[int(np.argmax(parent_weights))])

    distorted = pg.model.particle_q.numpy().copy()
    distorted[parent_node] += np.asarray([0.02, -0.01, 0.005], dtype=np.float32)

    def _run(use_computed: bool) -> np.ndarray:
        solver = HexShapeMatchingSolver(
            pg.model,
            clusters,
            iterations=0,
            enable_shape_matching=True,
            shape_matching_stiffness=0.0,
            hierarchy=hierarchy,
            hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_OUTER8,
            hierarchical_shape_matching_stiffness=0.9,
            hierarchical_shape_matching_passes=1,
            hierarchical_shape_matching_outer8_absolute_projection=absolute_projection,
            shape_matching_use_computed_prolongation=use_computed,
        )
        state_in = pg.model.state()
        state_out = pg.model.state()
        state_in.particle_q.assign(distorted)
        state_in.particle_qd.zero_()

        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        wp.synchronize_device(pg.model.device)
        return state_out.particle_q.numpy()

    table_q = _run(False)
    computed_q = _run(True)

    assert np.allclose(computed_q, table_q, atol=1.0e-7)
    assert np.linalg.norm(table_q[fine_node] - distorted[fine_node]) > 1.0e-7


@pytest.mark.parametrize(
    ("shape", "l1_mode", "l2_mode"),
    [
        ((2, 2, 2), HIERARCHICAL_SHAPE_MATCHING_OUTER8, HIERARCHICAL_SHAPE_MATCHING_OFF),
        ((4, 4, 4), HIERARCHICAL_SHAPE_MATCHING_OFF, HIERARCHICAL_SHAPE_MATCHING_OUTER8),
    ],
)
def test_outer8_computed_and_table_prolongation_match(shape: tuple[int, int, int], l1_mode: int, l2_mode: int):
    pg = _build_pg(np.ones(shape, dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    grid_to_node = pg.aux.grid_to_node.numpy()
    corner_node = int(grid_to_node[0, 0, 0])

    distorted = pg.model.particle_q.numpy().copy()
    distorted[corner_node] += np.asarray([0.02, -0.01, 0.005], dtype=np.float32)

    def _run(use_computed: bool) -> np.ndarray:
        solver = HexShapeMatchingSolver(
            pg.model,
            clusters,
            iterations=0,
            enable_shape_matching=True,
            shape_matching_stiffness=0.0,
            hierarchy=hierarchy,
            hierarchical_shape_matching_mode=l1_mode,
            hierarchical_shape_matching_stiffness=0.9,
            hierarchical_shape_matching_passes=1,
            l2_hierarchical_shape_matching_mode=l2_mode,
            l2_hierarchical_shape_matching_stiffness=0.9,
            l2_hierarchical_shape_matching_passes=1,
            shape_matching_use_computed_prolongation=use_computed,
        )
        state_in = pg.model.state()
        state_out = pg.model.state()
        state_in.particle_q.assign(distorted)
        state_in.particle_qd.zero_()

        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        wp.synchronize_device(pg.model.device)
        return state_out.particle_q.numpy()

    assert np.allclose(_run(True), _run(False), atol=1.0e-7)


def test_full27_hierarchy_directly_corrects_interior_node():
    pg = _build_pg(np.ones((2, 2, 2), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=0,
        enable_shape_matching=True,
        shape_matching_stiffness=0.0,
        hierarchy=hierarchy,
        hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_FULL27,
        hierarchical_shape_matching_stiffness=0.9,
        hierarchical_shape_matching_passes=1,
    )

    center_node = int(pg.aux.grid_to_node.numpy()[1, 1, 1])
    distorted = pg.model.particle_q.numpy().copy()
    distorted[center_node] += np.asarray([0.02, -0.01, 0.005], dtype=np.float32)

    state_in = pg.model.state()
    state_out = pg.model.state()
    state_in.particle_q.assign(distorted)
    state_in.particle_qd.zero_()

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)
    wp.synchronize_device(pg.model.device)

    final_q = state_out.particle_q.numpy()
    assert np.linalg.norm(final_q[center_node] - distorted[center_node]) > 1.0e-7


@pytest.mark.parametrize("l2_hierarchical_shape_matching_use_gs", [False, True])
def test_l2_outer8_hierarchy_translation_is_a_fixed_point(l2_hierarchical_shape_matching_use_gs: bool):
    pg = _build_pg(np.ones((4, 4, 4), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=0,
        enable_shape_matching=True,
        shape_matching_stiffness=0.0,
        hierarchy=hierarchy,
        l2_hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_OUTER8,
        l2_hierarchical_shape_matching_stiffness=0.9,
        l2_hierarchical_shape_matching_passes=1,
        l2_hierarchical_shape_matching_use_gs=l2_hierarchical_shape_matching_use_gs,
    )

    state_in = pg.model.state()
    state_out = pg.model.state()
    translated = pg.model.particle_q.numpy() + np.asarray([0.07, -0.03, 0.05], dtype=np.float32)
    state_in.particle_q.assign(translated)
    state_in.particle_qd.zero_()

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)
    wp.synchronize_device(pg.model.device)

    assert np.allclose(state_out.particle_q.numpy(), translated, atol=1.0e-5)


@pytest.mark.parametrize("l2_hierarchical_shape_matching_use_gs", [False, True])
def test_l2_full125_hierarchy_translation_is_a_fixed_point(l2_hierarchical_shape_matching_use_gs: bool):
    pg = _build_pg(np.ones((4, 4, 4), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    assert hierarchy.l2_full125 is not None
    assert hierarchy.l2_full125.uniform_size == 125

    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=0,
        enable_shape_matching=True,
        shape_matching_stiffness=0.0,
        hierarchy=hierarchy,
        l2_hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_FULL27,
        l2_hierarchical_shape_matching_stiffness=0.9,
        l2_hierarchical_shape_matching_passes=1,
        l2_hierarchical_shape_matching_use_gs=l2_hierarchical_shape_matching_use_gs,
    )

    state_in = pg.model.state()
    state_out = pg.model.state()
    translated = pg.model.particle_q.numpy() + np.asarray([0.07, -0.03, 0.05], dtype=np.float32)
    state_in.particle_q.assign(translated)
    state_in.particle_qd.zero_()

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)
    wp.synchronize_device(pg.model.device)

    assert np.allclose(state_out.particle_q.numpy(), translated, atol=1.0e-5)


def test_l2_full125_hierarchy_directly_corrects_interior_node():
    pg = _build_pg(np.ones((4, 4, 4), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=0,
        enable_shape_matching=True,
        shape_matching_stiffness=0.0,
        hierarchy=hierarchy,
        l2_hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_FULL27,
        l2_hierarchical_shape_matching_stiffness=0.9,
        l2_hierarchical_shape_matching_passes=1,
    )

    center_node = int(pg.aux.grid_to_node.numpy()[2, 2, 2])
    distorted = pg.model.particle_q.numpy().copy()
    distorted[center_node] += np.asarray([0.02, -0.01, 0.005], dtype=np.float32)

    state_in = pg.model.state()
    state_out = pg.model.state()
    state_in.particle_q.assign(distorted)
    state_in.particle_qd.zero_()

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)
    wp.synchronize_device(pg.model.device)

    final_q = state_out.particle_q.numpy()
    assert np.linalg.norm(final_q[center_node] - distorted[center_node]) > 1.0e-7


def test_l2_outer8_hierarchy_prolongates_corner_correction_to_interior_node():
    pg = _build_pg(np.ones((4, 4, 4), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=0,
        enable_shape_matching=True,
        shape_matching_stiffness=0.0,
        hierarchy=hierarchy,
        l2_hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_OUTER8,
        l2_hierarchical_shape_matching_stiffness=0.9,
        l2_hierarchical_shape_matching_passes=1,
    )

    grid_to_node = pg.aux.grid_to_node.numpy()
    corner_node = int(grid_to_node[0, 0, 0])
    edge_node = int(grid_to_node[1, 0, 0])
    distorted = pg.model.particle_q.numpy().copy()
    distorted[corner_node] += np.asarray([0.02, -0.01, 0.005], dtype=np.float32)

    state_in = pg.model.state()
    state_out = pg.model.state()
    state_in.particle_q.assign(distorted)
    state_in.particle_qd.zero_()

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)
    wp.synchronize_device(pg.model.device)

    final_q = state_out.particle_q.numpy()
    assert np.linalg.norm(final_q[edge_node] - distorted[edge_node]) > 1.0e-7


def test_l0_sleep_disabled_leaves_projection_path_inactive_by_default():
    pg = _build_pg(np.ones((2, 2, 2), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=0,
        enable_shape_matching=True,
        shape_matching_stiffness=1.0,
        hierarchy=hierarchy,
        hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_OUTER8,
        hierarchical_shape_matching_stiffness=0.0,
        sleep_l0_shape_matching=False,
    )

    center_node = int(pg.aux.grid_to_node.numpy()[1, 1, 1])
    distorted = pg.model.particle_q.numpy().copy()
    distorted[center_node] += np.asarray([0.02, -0.01, 0.005], dtype=np.float32)

    state_in = pg.model.state()
    state_out = pg.model.state()
    state_in.particle_q.assign(distorted)
    state_in.particle_qd.zero_()

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)
    wp.synchronize_device(pg.model.device)

    assert solver.sleeping_l0_cluster_count == 0
    assert np.array_equal(solver.l0_runtime_active_host, clusters.active_host)
    assert np.allclose(state_out.particle_q.numpy()[center_node], distorted[center_node], atol=1.0e-7)


def test_l0_sleep_does_not_sleep_surface_touching_full_blocks():
    pg = _build_pg(np.ones((2, 2, 2), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=0,
        enable_shape_matching=True,
        shape_matching_stiffness=1.0,
        hierarchy=hierarchy,
        hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_OUTER8,
        hierarchical_shape_matching_stiffness=0.0,
        sleep_l0_shape_matching=True,
    )

    center_node = int(pg.aux.grid_to_node.numpy()[1, 1, 1])
    distorted = pg.model.particle_q.numpy().copy()
    distorted[center_node] += np.asarray([0.02, -0.01, 0.005], dtype=np.float32)

    state_in = pg.model.state()
    state_out = pg.model.state()
    state_in.particle_q.assign(distorted)
    state_in.particle_qd.zero_()

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)
    wp.synchronize_device(pg.model.device)

    assert solver.sleeping_l0_cluster_count == 0
    assert solver.l1_sleep_projection_active_count == 0
    assert np.allclose(state_out.particle_q.numpy()[center_node], distorted[center_node], atol=1.0e-7)


def test_l0_sleep_requires_live_outer8_coarse_constraint():
    pg = _build_pg(np.ones((5, 5, 5), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=0,
        enable_shape_matching=True,
        shape_matching_stiffness=1.0,
        hierarchy=hierarchy,
        hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_OUTER8,
        hierarchical_shape_matching_stiffness=0.0,
        hierarchical_shape_matching_passes=1,
        sleep_l0_shape_matching=True,
    )

    assert solver.sleeping_l0_cluster_count == 0
    assert solver.l1_sleep_projection_active_count == 0
    assert np.array_equal(solver.l0_runtime_active_host, clusters.active_host)

    solver.hierarchical_shape_matching_stiffness = 0.9
    solver.hierarchical_shape_matching_passes = 0
    solver.refresh_l0_sleep_state()
    assert solver.sleeping_l0_cluster_count == 0
    assert solver.l1_sleep_projection_active_count == 0

    solver.shape_matching_stiffness = 0.9
    solver.hierarchical_shape_matching_passes = 1
    solver.refresh_l0_sleep_state()
    assert solver.sleeping_l0_cluster_count == 8
    assert solver.l1_sleep_projection_active_count == 1


def test_l1_l0_sleep_projects_interior_node_to_parent_trilinear_position():
    pg = _build_pg(np.ones((5, 5, 5), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=0,
        enable_shape_matching=True,
        shape_matching_stiffness=1.0,
        hierarchy=hierarchy,
        hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_OUTER8,
        hierarchical_shape_matching_stiffness=0.9,
        sleep_l0_shape_matching=True,
    )

    grid_to_node = pg.aux.grid_to_node.numpy()
    center_node = int(grid_to_node[3, 3, 3])
    corner_nodes = grid_to_node[
        2 + CELL_CORNER_OFFSETS[:, 0] * 2,
        2 + CELL_CORNER_OFFSETS[:, 1] * 2,
        2 + CELL_CORNER_OFFSETS[:, 2] * 2,
    ].astype(np.int32)
    distorted = pg.model.particle_q.numpy().copy()
    distorted[center_node] += np.asarray([0.02, -0.01, 0.005], dtype=np.float32)
    expected = distorted[corner_nodes].mean(axis=0)

    assert solver.sleeping_l0_cluster_count == 8
    assert solver.l0_runtime_particle_cluster_counts_host[center_node] == 0
    assert solver.l1_sleep_projection_active_count == 1

    state_in = pg.model.state()
    state_out = pg.model.state()
    state_in.particle_q.assign(distorted)
    state_in.particle_qd.zero_()

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)
    wp.synchronize_device(pg.model.device)

    assert np.allclose(state_out.particle_q.numpy()[center_node], expected, atol=1.0e-7)


def test_l1_l0_sleep_projection_uses_l0_shape_matching_stiffness():
    pg = _build_pg(np.ones((5, 5, 5), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=0,
        enable_shape_matching=True,
        shape_matching_stiffness=0.5,
        hierarchy=hierarchy,
        hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_OUTER8,
        hierarchical_shape_matching_stiffness=0.9,
        sleep_l0_shape_matching=True,
    )

    grid_to_node = pg.aux.grid_to_node.numpy()
    center_node = int(grid_to_node[3, 3, 3])
    corner_nodes = grid_to_node[
        2 + CELL_CORNER_OFFSETS[:, 0] * 2,
        2 + CELL_CORNER_OFFSETS[:, 1] * 2,
        2 + CELL_CORNER_OFFSETS[:, 2] * 2,
    ].astype(np.int32)
    distorted = pg.model.particle_q.numpy().copy()
    distorted[center_node] += np.asarray([0.02, -0.01, 0.005], dtype=np.float32)
    target = distorted[corner_nodes].mean(axis=0)
    expected = distorted[center_node] + (target - distorted[center_node]) * 0.5

    state_in = pg.model.state()
    state_out = pg.model.state()
    state_in.particle_q.assign(distorted)
    state_in.particle_qd.zero_()

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)
    wp.synchronize_device(pg.model.device)

    assert np.allclose(state_out.particle_q.numpy()[center_node], expected, atol=1.0e-7)


def test_l2_l0_sleep_projects_interior_node_to_parent_trilinear_position():
    pg = _build_pg(np.ones((9, 9, 9), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=0,
        enable_shape_matching=True,
        shape_matching_stiffness=1.0,
        hierarchy=hierarchy,
        l2_hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_OUTER8,
        l2_hierarchical_shape_matching_stiffness=0.9,
        sleep_l0_shape_matching=True,
    )

    grid_to_node = pg.aux.grid_to_node.numpy()
    center_node = int(grid_to_node[6, 6, 6])
    corner_nodes = grid_to_node[
        4 + CELL_CORNER_OFFSETS[:, 0] * 4,
        4 + CELL_CORNER_OFFSETS[:, 1] * 4,
        4 + CELL_CORNER_OFFSETS[:, 2] * 4,
    ].astype(np.int32)
    distorted = pg.model.particle_q.numpy().copy()
    distorted[center_node] += np.asarray([0.02, -0.01, 0.005], dtype=np.float32)
    expected = distorted[corner_nodes].mean(axis=0)

    assert solver.sleeping_l0_cluster_count >= 64
    assert solver.l0_runtime_particle_cluster_counts_host[center_node] == 0
    assert solver.l2_sleep_projection_active_count == 1

    state_in = pg.model.state()
    state_out = pg.model.state()
    state_in.particle_q.assign(distorted)
    state_in.particle_qd.zero_()

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)
    wp.synchronize_device(pg.model.device)

    assert np.allclose(state_out.particle_q.numpy()[center_node], expected, atol=1.0e-7)


def test_l0_sleep_runtime_weights_are_recomputed_from_awake_clusters_after_cut():
    pg = _build_pg(np.ones((7, 5, 5), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=1,
        enable_shape_matching=True,
        shape_matching_stiffness=0.5,
        hierarchy=hierarchy,
        hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_OUTER8,
        sleep_l0_shape_matching=True,
        sleep_l0_wake_halo_blocks=0,
    )
    delete_state = make_hex_deletion_state(pg.model, pg.aux)

    deleted_cell = int(pg.aux.grid_to_cell.numpy()[2, 2, 2])
    deleted = delete_state.delete_cells([deleted_cell])
    assert deleted == 1
    solver.update_l0_sleep_after_deletion(delete_state.last_deleted_cells_host)
    wp.synchronize_device(pg.model.device)

    boundary_node = int(pg.aux.grid_to_node.numpy()[4, 3, 3])
    assert clusters.particle_cluster_inv_weights_host[boundary_node] == pytest.approx(1.0 / 8.0)
    assert solver.l0_runtime_particle_cluster_counts_host[boundary_node] == 4
    assert solver.l0_runtime_particle_cluster_inv_weights_host[boundary_node] == pytest.approx(0.25)


def test_l0_sleep_cut_wakes_l1_block_halo_while_far_blocks_remain_asleep():
    pg = _build_pg(np.ones((10, 5, 5), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=0,
        enable_shape_matching=True,
        shape_matching_stiffness=1.0,
        hierarchy=hierarchy,
        hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_OUTER8,
        sleep_l0_shape_matching=True,
        sleep_l0_wake_halo_blocks=1,
    )
    delete_state = make_hex_deletion_state(pg.model, pg.aux)

    deleted_cell = int(pg.aux.grid_to_cell.numpy()[2, 2, 2])
    deleted = delete_state.delete_cells([deleted_cell])
    assert deleted == 1
    solver.update_l0_sleep_after_deletion(delete_state.last_deleted_cells_host)

    key_to_cluster = {tuple(key.tolist()): idx for idx, key in enumerate(hierarchy.outer8_block_keys_host)}
    assert solver.l1_sleep_projection_active_host[key_to_cluster[(2, 1, 1)]] == 0
    assert solver.l1_sleep_projection_active_host[key_to_cluster[(3, 1, 1)]] == 1
    assert solver.sleeping_l0_cluster_count == 8


def test_l0_sleep_projection_does_not_override_particles_owned_by_awake_l0_clusters():
    pg = _build_pg(np.ones((10, 5, 5), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=0,
        enable_shape_matching=True,
        shape_matching_stiffness=0.0,
        hierarchy=hierarchy,
        hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_OUTER8,
        sleep_l0_shape_matching=True,
        sleep_l0_wake_halo_blocks=1,
    )
    delete_state = make_hex_deletion_state(pg.model, pg.aux)
    deleted_cell = int(pg.aux.grid_to_cell.numpy()[2, 2, 2])
    assert delete_state.delete_cells([deleted_cell]) == 1
    solver.update_l0_sleep_after_deletion(delete_state.last_deleted_cells_host)

    boundary_node = int(pg.aux.grid_to_node.numpy()[6, 3, 3])
    assert solver.l0_runtime_particle_cluster_counts_host[boundary_node] > 0
    assert hierarchy.outer8_prolongation.child_cluster_host[boundary_node] >= 0

    distorted = pg.model.particle_q.numpy().copy()
    distorted[boundary_node] += np.asarray([0.02, -0.01, 0.005], dtype=np.float32)

    state_in = pg.model.state()
    state_out = pg.model.state()
    state_in.particle_q.assign(distorted)
    state_in.particle_qd.zero_()

    solver.step(state_in, state_out, None, None, 1.0 / 60.0)
    wp.synchronize_device(pg.model.device)

    assert np.allclose(state_out.particle_q.numpy()[boundary_node], distorted[boundary_node], atol=1.0e-7)


def test_l0_sleep_cut_wakes_l2_block_halo_while_far_blocks_remain_asleep():
    pg = _build_pg(np.ones((20, 9, 9), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=0,
        enable_shape_matching=True,
        shape_matching_stiffness=1.0,
        hierarchy=hierarchy,
        l2_hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_OUTER8,
        sleep_l0_shape_matching=True,
        sleep_l0_wake_halo_blocks=1,
    )
    delete_state = make_hex_deletion_state(pg.model, pg.aux)

    deleted_cell = int(pg.aux.grid_to_cell.numpy()[4, 4, 4])
    deleted = delete_state.delete_cells([deleted_cell])
    assert deleted == 1
    solver.update_l0_sleep_after_deletion(delete_state.last_deleted_cells_host)

    key_to_cluster = {tuple(key.tolist()): idx for idx, key in enumerate(hierarchy.l2_outer8_block_keys_host)}
    assert solver.l2_sleep_projection_active_host[key_to_cluster[(2, 1, 1)]] == 0
    assert solver.l2_sleep_projection_active_host[key_to_cluster[(3, 1, 1)]] == 1
    assert solver.sleeping_l0_cluster_count == 64


def test_solver_runtime_reset_clears_cut_wake_state():
    pg = _build_pg(np.ones((20, 9, 9), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=0,
        enable_shape_matching=True,
        shape_matching_stiffness=1.0,
        hierarchy=hierarchy,
        l2_hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_OUTER8,
        sleep_l0_shape_matching=True,
        sleep_l0_wake_halo_blocks=1,
    )
    delete_state = make_hex_deletion_state(pg.model, pg.aux)

    initial_sleeping = int(solver.sleeping_l0_cluster_count)
    deleted_cell = int(pg.aux.grid_to_cell.numpy()[4, 4, 4])
    assert delete_state.delete_cells([deleted_cell]) == 1
    assert solver.update_l0_sleep_after_deletion(delete_state.last_deleted_cells_host)
    assert solver.sleeping_l0_cluster_count < initial_sleeping
    assert not solver.l0_fast_uniform8_active
    assert solver._l2_sleep_wake_mask is not None
    assert np.count_nonzero(solver._l2_sleep_wake_mask.numpy()) > 0

    delete_state.reset()
    solver.reset_runtime_state()

    assert solver.l0_fast_uniform8_active
    assert solver.sleeping_l0_cluster_count == initial_sleeping
    assert np.count_nonzero(solver._l2_sleep_wake_mask.numpy()) == 0
    assert solver._l1_sleep_wake_block_keys == set()
    assert solver._l2_sleep_wake_block_keys == set()


def test_l0_sleep_mode_changes_refresh_l2_l1_and_off_masks():
    pg = _build_pg(np.ones((9, 9, 9), dtype=np.uint8))
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)
    hierarchy = build_hierarchical_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=0,
        enable_shape_matching=True,
        shape_matching_stiffness=1.0,
        hierarchy=hierarchy,
        hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_OUTER8,
        l2_hierarchical_shape_matching_mode=HIERARCHICAL_SHAPE_MATCHING_OUTER8,
        sleep_l0_shape_matching=True,
    )

    sleepable_l1_count = int(np.count_nonzero(hierarchy.outer8_sleepable_host))
    assert solver.l2_sleep_projection_active_count == 1
    assert 0 < solver.l1_sleep_projection_active_count < sleepable_l1_count
    assert solver.sleeping_l0_cluster_count > 64

    solver.l2_hierarchical_shape_matching_mode = HIERARCHICAL_SHAPE_MATCHING_OFF
    solver.refresh_l0_sleep_state()
    assert solver.l2_sleep_projection_active_count == 0
    assert solver.l1_sleep_projection_active_count == sleepable_l1_count
    assert solver.sleeping_l0_cluster_count > 0

    solver.hierarchical_shape_matching_mode = HIERARCHICAL_SHAPE_MATCHING_OFF
    solver.refresh_l0_sleep_state()
    assert solver.l1_sleep_projection_active_count == 0
    assert solver.sleeping_l0_cluster_count == 0
    assert np.array_equal(solver.l0_runtime_active_host, clusters.active_host)


@pytest.mark.skipif(
    (not wp.is_cuda_available()) or (not wp.is_mempool_enabled(wp.get_device("cuda:0"))),
    reason="CUDA graph capture requires CUDA with Warp mempool enabled",
)
def test_solver_cuda_graph_capture_replays_with_shape_matching():
    device = "cuda:0"
    pg_loop = _build_pg(np.ones((2, 1, 1), dtype=np.uint8), device=device)
    pg_graph = _build_pg(np.ones((2, 1, 1), dtype=np.uint8), device=device)
    _zero_gravity(pg_loop.model)
    _zero_gravity(pg_graph.model)

    loop_solver = _make_solver(
        pg_loop,
        iterations=6,
        enable_shape_matching=True,
        shape_matching_stiffness=0.75,
        shape_matching_passes=2,
        shape_matching_mode=SHAPE_MATCHING_SOLVE_COLORED_GS,
        shape_matching_gs_weighting=SHAPE_MATCHING_GS_WEIGHT_SQRT,
    )
    graph_solver = _make_solver(
        pg_graph,
        iterations=6,
        enable_shape_matching=True,
        shape_matching_stiffness=0.75,
        shape_matching_passes=2,
        shape_matching_mode=SHAPE_MATCHING_SOLVE_COLORED_GS,
        shape_matching_gs_weighting=SHAPE_MATCHING_GS_WEIGHT_SQRT,
    )

    distorted = pg_loop.model.particle_q.numpy().copy()
    distorted[4] += np.asarray([0.012, -0.008, 0.006], dtype=np.float32)

    loop_state_0 = pg_loop.model.state()
    loop_state_1 = pg_loop.model.state()
    loop_state_0.particle_q.assign(distorted)
    loop_state_0.particle_qd.zero_()

    for _ in range(2):
        loop_state_0.clear_forces()
        loop_solver.step(loop_state_0, loop_state_1, None, None, 1.0 / 60.0)
        loop_state_0, loop_state_1 = loop_state_1, loop_state_0

    expected_q = loop_state_0.particle_q.numpy()

    graph_state_0 = pg_graph.model.state()
    graph_state_1 = pg_graph.model.state()
    graph_state_0.particle_q.assign(distorted)
    graph_state_0.particle_qd.zero_()

    init_state_0 = pg_graph.model.state()
    init_state_1 = pg_graph.model.state()
    init_state_0.particle_q.assign(distorted)
    init_state_0.particle_qd.zero_()
    init_state_1.assign(graph_state_1)

    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        s0, s1 = graph_state_0, graph_state_1
        for _ in range(2):
            s0.clear_forces()
            graph_solver.step(s0, s1, None, None, 1.0 / 60.0)
            s0, s1 = s1, s0

    graph = capture.graph

    graph_state_0.assign(init_state_0)
    graph_state_1.assign(init_state_1)
    _reset_cluster_warm_start(graph_solver)

    wp.capture_launch(graph)

    assert np.allclose(graph_state_0.particle_q.numpy(), expected_q, atol=5.0e-4)


@pytest.mark.skipif(
    (not wp.is_cuda_available()) or (not wp.is_mempool_enabled(wp.get_device("cuda:0"))),
    reason="CUDA graph capture requires CUDA with Warp mempool enabled",
)
def test_solver_cuda_graph_replay_after_device_cell_deletion_without_recapture():
    device = "cuda:0"
    dt = 1.0 / 60.0
    labels = np.ones((3, 1, 1), dtype=np.uint8)

    pg_loop = _build_pg(labels, device=device)
    pg_graph = _build_pg(labels, device=device)
    _zero_gravity(pg_loop.model)
    _zero_gravity(pg_graph.model)

    loop_solver = _make_solver(
        pg_loop,
        iterations=4,
        enable_shape_matching=True,
        shape_matching_stiffness=0.75,
        shape_matching_passes=1,
        shape_matching_mode=SHAPE_MATCHING_SOLVE_SCATTER,
    )
    graph_solver = _make_solver(
        pg_graph,
        iterations=4,
        enable_shape_matching=True,
        shape_matching_stiffness=0.75,
        shape_matching_passes=1,
        shape_matching_mode=SHAPE_MATCHING_SOLVE_SCATTER,
    )
    loop_delete = make_hex_deletion_state(pg_loop.model, pg_loop.aux)
    graph_delete = make_hex_deletion_state(pg_graph.model, pg_graph.aux)

    assert loop_solver.l0_fast_uniform8_active
    assert graph_solver.l0_fast_uniform8_active

    distorted_loop = pg_loop.model.particle_q.numpy().copy()
    distorted_graph = pg_graph.model.particle_q.numpy().copy()
    distorted_loop[4] += np.asarray([0.012, -0.008, 0.006], dtype=np.float32)
    distorted_graph[4] += np.asarray([0.012, -0.008, 0.006], dtype=np.float32)

    loop_state_0 = pg_loop.model.state()
    loop_state_1 = pg_loop.model.state()
    loop_state_0.particle_q.assign(distorted_loop)
    loop_state_0.particle_qd.zero_()
    graph_state_0 = pg_graph.model.state()
    graph_state_1 = pg_graph.model.state()
    graph_state_0.particle_q.assign(distorted_graph)
    graph_state_0.particle_qd.zero_()

    def _capture_one_step() -> object:
        wp.synchronize_device(device)
        with wp.ScopedCapture(device=device, force_module_load=False) as capture:
            graph_state_0.clear_forces()
            graph_solver.step(graph_state_0, graph_state_1, None, None, dt)
        return capture.graph

    def _run_loop_step() -> np.ndarray:
        loop_state_0.clear_forces()
        loop_solver.step(loop_state_0, loop_state_1, None, None, dt)
        return loop_state_1.particle_q.numpy()

    def _delete_cell(pg, delete_state, solver, cell_coord: tuple[int, int, int]) -> tuple[bool, int]:
        cell_idx = int(pg.aux.grid_to_cell.numpy()[cell_coord])
        cell_ids = wp.array(np.asarray([cell_idx], dtype=np.int32), dtype=wp.int32, device=device)
        result = delete_state.delete_device_cells_async(cell_ids, 1)
        graph_shape_changed = solver.update_l0_sleep_after_deletion_device(
            result.deleted_cells_device,
            result.deleted_count_device,
            result.candidate_capacity,
            sync_host=False,
        )
        deleted_now = int(result.deleted_count_device.numpy()[0])
        return bool(graph_shape_changed), deleted_now

    graph_state_0_snapshot = pg_graph.model.state()
    graph_state_1_snapshot = pg_graph.model.state()
    graph_state_0_snapshot.assign(graph_state_0)
    graph_state_1_snapshot.assign(graph_state_1)

    expected_pre_delete = _run_loop_step()
    graph = _capture_one_step()
    capture_count = 1
    graph_state_0.assign(graph_state_0_snapshot)
    graph_state_1.assign(graph_state_1_snapshot)
    _reset_cluster_warm_start(graph_solver)
    wp.capture_launch(graph)

    assert np.allclose(graph_state_1.particle_q.numpy(), expected_pre_delete, atol=5.0e-4)

    loop_state_0.assign(loop_state_1)
    graph_state_0.assign(graph_state_1)

    loop_transition, loop_deleted = _delete_cell(pg_loop, loop_delete, loop_solver, (0, 0, 0))
    graph_transition, graph_deleted = _delete_cell(pg_graph, graph_delete, graph_solver, (0, 0, 0))
    assert loop_deleted == 1
    assert graph_deleted == 1
    assert loop_transition
    assert graph_transition
    assert not loop_solver.l0_fast_uniform8_active
    assert not graph_solver.l0_fast_uniform8_active

    _reset_cluster_warm_start(loop_solver)
    _reset_cluster_warm_start(graph_solver)
    graph_state_0_snapshot.assign(graph_state_0)
    graph_state_1_snapshot.assign(graph_state_1)
    graph = _capture_one_step()
    capture_count += 1
    graph_state_0.assign(graph_state_0_snapshot)
    graph_state_1.assign(graph_state_1_snapshot)
    _reset_cluster_warm_start(graph_solver)

    expected_after_first_delete = _run_loop_step()
    wp.capture_launch(graph)

    assert capture_count == 2
    assert np.allclose(graph_state_1.particle_q.numpy(), expected_after_first_delete, atol=5.0e-4)

    loop_state_0.assign(loop_state_1)
    graph_state_0.assign(graph_state_1)

    loop_transition, loop_deleted = _delete_cell(pg_loop, loop_delete, loop_solver, (2, 0, 0))
    graph_transition, graph_deleted = _delete_cell(pg_graph, graph_delete, graph_solver, (2, 0, 0))
    assert loop_deleted == 1
    assert graph_deleted == 1
    assert not loop_transition
    assert not graph_transition

    expected_after_second_delete = _run_loop_step()
    wp.capture_launch(graph)

    assert capture_count == 2
    assert np.allclose(graph_state_1.particle_q.numpy(), expected_after_second_delete, atol=5.0e-4)


def test_self_collision_toggle_with_shape_matching_changes_overlapping_clusters():
    labels = np.zeros((3, 1, 1), dtype=np.uint8)
    labels[0, 0, 0] = 1
    labels[2, 0, 0] = 1
    pg = _build_pg(labels)
    _zero_gravity(pg.model)
    clusters = build_shape_matching_clusters(pg)

    solver_off = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=4,
        enable_shape_matching=True,
        enable_self_collisions=False,
        shape_matching_stiffness=0.8,
    )
    solver_on = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=4,
        enable_shape_matching=True,
        enable_self_collisions=True,
        shape_matching_stiffness=0.8,
    )

    rest = pg.model.particle_q.numpy().copy()
    cluster_0 = clusters.indices_host[clusters.offsets_host[0] : clusters.offsets_host[1]]
    cluster_1 = clusters.indices_host[clusters.offsets_host[1] : clusters.offsets_host[2]]

    overlap = rest.copy()
    overlap_gap = 0.25 * float(pg.model.particle_radius.numpy()[0])
    overlap[cluster_1] -= np.asarray([2.0 * pg.aux.voxel_size - overlap_gap, 0.0, 0.0], dtype=np.float32)
    initial_pair_dist = np.linalg.norm(overlap[cluster_0][:, None, :] - overlap[cluster_1][None, :, :], axis=2).min()

    def _step_once(solver: HexShapeMatchingSolver) -> np.ndarray:
        _reset_cluster_warm_start(solver)
        state_in = pg.model.state()
        state_out = pg.model.state()
        state_in.particle_q.assign(overlap)
        state_in.particle_qd.zero_()
        solver.step(state_in, state_out, None, None, 1.0 / 60.0)
        return state_out.particle_q.numpy()

    q_off = _step_once(solver_off)
    q_on = _step_once(solver_on)

    off_pair_dist = np.linalg.norm(q_off[cluster_0][:, None, :] - q_off[cluster_1][None, :, :], axis=2).min()
    on_pair_dist = np.linalg.norm(q_on[cluster_0][:, None, :] - q_on[cluster_1][None, :, :], axis=2).min()

    assert off_pair_dist <= initial_pair_dist + 1.0e-6
    assert on_pair_dist > off_pair_dist + 1.0e-4


def test_ground_plane_contact_supports_free_fall_without_penetration():
    pg = _build_pg(np.ones((1, 1, 1), dtype=np.uint8))
    clusters = build_shape_matching_clusters(pg)
    solver = HexShapeMatchingSolver(
        pg.model,
        clusters,
        iterations=4,
        enable_shape_matching=False,
        enable_self_collisions=False,
        enable_ground_plane=True,
        ground_height=0.0,
    )

    rest = pg.model.particle_q.numpy()
    lifted = rest + np.asarray([0.0, 0.0, 0.05], dtype=np.float32)
    radius = float(pg.model.particle_radius.numpy()[0])

    state_0 = pg.model.state()
    state_1 = pg.model.state()
    state_0.particle_q.assign(lifted)
    state_0.particle_qd.zero_()

    initial_mean_z = float(lifted[:, 2].mean())
    for _ in range(30):
        state_0.clear_forces()
        solver.step(state_0, state_1, None, None, 1.0 / 60.0)
        state_0, state_1 = state_1, state_0

    final_q = state_0.particle_q.numpy()
    final_mean_z = float(final_q[:, 2].mean())
    min_z = float(final_q[:, 2].min())

    assert final_mean_z < initial_mean_z
    assert min_z >= radius - 1.0e-4
