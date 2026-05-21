# SPDX-License-Identifier: Apache-2.0
"""Cell-centred heat and explicit instrument mode tests."""

from __future__ import annotations

import numpy as np
import warp as wp

from omnisurg.hex.deletion import make_hex_deletion_state
from omnisurg.hex.hex_grid import build_hex_particle_grid
from omnisurg.hex.heat import make_hex_heat_state
from omnisurg.hex.io.digimouse import DigimouseAtlas
from omnisurg.hex.materials import BONE, DEFAULT_MATERIALS, MUSCLE, MaterialTable
from omnisurg.hex.app_runtime import _normalize_instrument_tool_mode


def _atlas_from_labels(labels: np.ndarray, voxel: float = 0.01) -> DigimouseAtlas:
    return DigimouseAtlas(
        labels=labels.astype(np.uint8),
        voxel_size=voxel,
        materials=MaterialTable(DEFAULT_MATERIALS),
    )


def _line_labels(values: list[int], voxel: float = 0.01) -> DigimouseAtlas:
    labels = np.asarray(values, dtype=np.uint8).reshape((len(values), 1, 1))
    return _atlas_from_labels(labels, voxel=voxel)


def _cell_centres(pg) -> np.ndarray:
    q = pg.state.particle_q.numpy()
    return q[pg.aux.cell_nodes_host].mean(axis=1).astype(np.float32)


def _cuttable_array(material_ids: set[int], device: str = "cpu") -> wp.array:
    values = np.zeros(len(DEFAULT_MATERIALS), dtype=np.int32)
    for material_id in material_ids:
        values[int(material_id)] = 1
    return wp.array(values, dtype=wp.int32, device=device)


def test_diathermy_injection_heats_only_nearby_active_cells():
    pg = build_hex_particle_grid(_line_labels([MUSCLE.id, MUSCLE.id, MUSCLE.id]), device="cpu")
    heat = make_hex_heat_state(pg.model, pg.aux)
    centres = _cell_centres(pg)
    sphere_q = wp.array(np.asarray([centres[0]], dtype=np.float32), dtype=wp.vec3, device="cpu")
    sphere_enabled = wp.array(np.asarray([1], dtype=np.int32), dtype=wp.int32, device="cpu")

    heat.apply_diathermy_spheres(
        pg.state.particle_q,
        sphere_q,
        sphere_enabled,
        1,
        0.003,
        power=600.0,
        dt=1.0 / 60.0,
        material_cuttable=_cuttable_array({MUSCLE.id}),
    )
    wp.synchronize_device("cpu")

    cell_heat = heat.cell_heat_a.numpy()
    assert cell_heat[0] > 0.0
    assert np.allclose(cell_heat[1:], 0.0)


def test_diathermy_injection_uses_per_sphere_analog_power():
    pg = build_hex_particle_grid(_line_labels([MUSCLE.id]), device="cpu")
    heat = make_hex_heat_state(pg.model, pg.aux)
    centre = _cell_centres(pg)[0]
    sphere_q = wp.array(np.asarray([centre], dtype=np.float32), dtype=wp.vec3, device="cpu")
    sphere_enabled = wp.array(np.asarray([1], dtype=np.int32), dtype=wp.int32, device="cpu")
    sphere_power = wp.array(np.asarray([25.0], dtype=np.float32), dtype=wp.float32, device="cpu")

    heat.apply_diathermy_spheres(
        pg.state.particle_q,
        sphere_q,
        sphere_enabled,
        1,
        0.01,
        power=999.0,
        dt=0.5,
        sphere_power=sphere_power,
        material_cuttable=_cuttable_array({MUSCLE.id}),
    )
    wp.synchronize_device("cpu")

    assert np.allclose(heat.cell_heat_a.numpy()[0], 12.5)


def test_diathermy_step_cools_heat_when_tool_power_is_zero():
    pg = build_hex_particle_grid(_line_labels([MUSCLE.id]), device="cpu")
    delete_state = make_hex_deletion_state(pg.model, pg.aux)
    heat = make_hex_heat_state(pg.model, pg.aux)
    centre = _cell_centres(pg)[0]
    sphere_q = wp.array(np.asarray([centre], dtype=np.float32), dtype=wp.vec3, device="cpu")
    sphere_enabled = wp.array(np.asarray([0], dtype=np.int32), dtype=wp.int32, device="cpu")
    sphere_power = wp.array(np.asarray([0.0], dtype=np.float32), dtype=wp.float32, device="cpu")
    heat.cell_heat_a.assign(np.asarray([10.0], dtype=np.float32))

    heat.step_diathermy_async(
        delete_state,
        pg.state.particle_q,
        sphere_q,
        sphere_enabled,
        1,
        0.01,
        dt=0.5,
        power=0.0,
        diffusion=0.0,
        cooling=1.0,
        sphere_power=sphere_power,
        material_cuttable=_cuttable_array({MUSCLE.id}),
    )
    delete_state.sync_last_deleted()

    assert np.allclose(heat.cell_heat_a.numpy()[0], 5.0)
    assert pg.aux.cell_active.numpy()[0] == 1


def test_cell_heat_diffusion_moves_to_face_neighbor_and_skips_deleted_cell():
    pg = build_hex_particle_grid(_line_labels([MUSCLE.id, MUSCLE.id, MUSCLE.id]), device="cpu")
    heat = make_hex_heat_state(pg.model, pg.aux)
    heat.cell_heat_a.assign(np.asarray([100.0, 0.0, 0.0], dtype=np.float32))

    heat.diffuse(dt=1.0, diffusion=0.1, cooling=0.0)
    wp.synchronize_device("cpu")
    diffused = heat.cell_heat_a.numpy()

    assert diffused[0] < 100.0
    assert diffused[1] > 0.0
    assert diffused[2] == 0.0

    pg.aux.cell_active.assign(np.asarray([1, 0, 1], dtype=np.int32))
    heat.cell_heat_a.assign(np.asarray([100.0, 0.0, 0.0], dtype=np.float32))
    heat.diffuse(dt=1.0, diffusion=0.1, cooling=0.0)
    wp.synchronize_device("cpu")
    blocked = heat.cell_heat_a.numpy()

    assert blocked[1] == 0.0
    assert blocked[2] == 0.0


def test_overheated_selection_deletes_only_cuttable_materials():
    pg = build_hex_particle_grid(_line_labels([MUSCLE.id, BONE.id]), device="cpu")
    delete_state = make_hex_deletion_state(pg.model, pg.aux)
    heat = make_hex_heat_state(pg.model, pg.aux)
    heat.cell_heat_a.assign(np.asarray([200.0, 200.0], dtype=np.float32))

    result = heat.delete_overheated_cells_async(
        delete_state,
        material_cuttable=_cuttable_array({MUSCLE.id}),
    )
    assert result.deleted_cells_device is not None
    deleted = delete_state.sync_last_deleted()

    assert deleted == 1
    assert np.array_equal(delete_state.last_deleted_cells_host, np.asarray([0], dtype=np.int32))
    assert np.array_equal(pg.aux.cell_active.numpy(), np.asarray([0, 1], dtype=np.int32))


def test_noncuttable_cells_conduct_heat_but_are_not_deleted():
    pg = build_hex_particle_grid(_line_labels([BONE.id, MUSCLE.id]), device="cpu")
    delete_state = make_hex_deletion_state(pg.model, pg.aux)
    heat = make_hex_heat_state(pg.model, pg.aux)
    cuttable = _cuttable_array({MUSCLE.id})

    heat.cell_heat_a.assign(np.asarray([200.0, 0.0], dtype=np.float32))
    heat.diffuse(dt=1.0, diffusion=1.0, cooling=0.0)
    wp.synchronize_device("cpu")
    diffused = heat.cell_heat_a.numpy()
    assert diffused[1] > 0.0

    heat.cell_heat_a.assign(np.asarray([200.0, 0.0], dtype=np.float32))
    heat.delete_overheated_cells_async(delete_state, material_cuttable=cuttable)
    deleted = delete_state.sync_last_deleted()

    assert deleted == 0
    assert np.array_equal(pg.aux.cell_active.numpy(), np.asarray([1, 1], dtype=np.int32))


def test_diathermy_trigger_heats_without_immediate_below_threshold_delete():
    pg = build_hex_particle_grid(_line_labels([MUSCLE.id]), device="cpu")
    delete_state = make_hex_deletion_state(pg.model, pg.aux)
    heat = make_hex_heat_state(pg.model, pg.aux)
    centre = _cell_centres(pg)[0]
    sphere_q = wp.array(np.asarray([centre], dtype=np.float32), dtype=wp.vec3, device="cpu")
    sphere_enabled = wp.array(np.asarray([1], dtype=np.int32), dtype=wp.int32, device="cpu")

    heat.step_diathermy_async(
        delete_state,
        pg.state.particle_q,
        sphere_q,
        sphere_enabled,
        1,
        0.01,
        dt=1.0 / 60.0,
        power=60.0,
        diffusion=0.0,
        cooling=0.0,
        material_cuttable=_cuttable_array({MUSCLE.id}),
    )
    deleted = delete_state.sync_last_deleted()

    assert deleted == 0
    assert heat.cell_heat_a.numpy()[0] > 0.0
    assert pg.aux.cell_active.numpy()[0] == 1


def test_scissors_capsule_deletes_cells_along_oriented_blade_segment():
    pg = build_hex_particle_grid(_line_labels([MUSCLE.id, MUSCLE.id, MUSCLE.id]), device="cpu")
    delete_state = make_hex_deletion_state(pg.model, pg.aux)
    heat = make_hex_heat_state(pg.model, pg.aux)
    centres = _cell_centres(pg)

    heat.delete_cells_by_capsule_async(
        delete_state,
        pg.state.particle_q,
        centres[0],
        centres[2],
        0.003,
        material_cuttable=_cuttable_array({MUSCLE.id}),
    )
    deleted = delete_state.sync_last_deleted()

    assert deleted == 3
    assert np.array_equal(pg.aux.cell_active.numpy(), np.asarray([0, 0, 0], dtype=np.int32))


def test_bipolar_capsule_deletes_once_even_if_applied_twice():
    pg = build_hex_particle_grid(_line_labels([MUSCLE.id, MUSCLE.id, MUSCLE.id]), device="cpu")
    delete_state = make_hex_deletion_state(pg.model, pg.aux)
    heat = make_hex_heat_state(pg.model, pg.aux)
    centres = _cell_centres(pg)
    cuttable = _cuttable_array({MUSCLE.id})

    heat.delete_cells_by_capsule_async(delete_state, pg.state.particle_q, centres[0], centres[2], 0.003, cuttable)
    first_deleted = delete_state.sync_last_deleted()
    heat.delete_cells_by_capsule_async(delete_state, pg.state.particle_q, centres[0], centres[2], 0.003, cuttable)
    second_deleted = delete_state.sync_last_deleted()

    assert first_deleted == 3
    assert second_deleted == 0


def test_cutting_cli_mode_alias_maps_to_diathermy():
    assert _normalize_instrument_tool_mode("cutting") == "diathermy"
    assert _normalize_instrument_tool_mode("diathermy") == "diathermy"
    assert _normalize_instrument_tool_mode("scissors") == "scissors"
