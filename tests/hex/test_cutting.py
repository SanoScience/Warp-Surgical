# SPDX-License-Identifier: Apache-2.0
"""Electrocautery heat + damage pipeline tests."""

from __future__ import annotations

import numpy as np

from newton._src.geometry.flags import ParticleFlags

from omnisurg.hex.cutting import make_cutting_state
from omnisurg.hex.grid import build_grid
from omnisurg.hex.io.digimouse import DigimouseAtlas
from omnisurg.hex.materials import DEFAULT_MATERIALS, MUSCLE, MaterialTable
from omnisurg.hex.tool import Tool


def _muscle_block(size: int = 5, voxel: float = 0.005) -> DigimouseAtlas:
    pad = 2
    n = size + 2 * pad
    labels = np.zeros((n, n, n), dtype=np.uint8)
    labels[pad : pad + size, pad : pad + size, pad : pad + size] = 1  # MUSCLE id
    return DigimouseAtlas(labels=labels, voxel_size=voxel, materials=MaterialTable(DEFAULT_MATERIALS))


def _active_count(model) -> int:
    flags = model.particle_flags.numpy()
    return int(((flags & int(ParticleFlags.ACTIVE)) != 0).sum())


def test_no_cutting_when_tool_inactive():
    pg = build_grid(_muscle_block())
    state = make_cutting_state(pg.model, pg.aux)
    tool = Tool.cautery_probe(length=0.05, radius=0.01, device=pg.model.device)
    tool.set_pose(np.zeros((1, 3), dtype=np.float32), np.asarray([[0.0, 0.0, 0.05]], dtype=np.float32))
    tool.power = 500.0
    tool.active = False  # pedal up
    n0 = _active_count(pg.model)
    for _ in range(50):
        state.step(tool, dt=1 / 60.0)
    assert _active_count(pg.model) == n0
    # Heat should not rise above zero.
    assert float(state.heat_a.numpy().max()) < 1e-6


def test_high_power_cut_removes_particles_and_disables_springs():
    pg = build_grid(_muscle_block())
    state = make_cutting_state(pg.model, pg.aux)
    tool = Tool.cautery_probe(length=0.05, radius=0.003, device=pg.model.device)
    # Route the probe through the block centre.
    cx = cy = pg.aux.voxel_size * (pg.aux.grid_shape[0] / 2)
    tool.set_pose(
        np.asarray([[cx, cy, 0.0]], dtype=np.float32),
        np.asarray([[cx, cy, 0.05]], dtype=np.float32),
    )
    tool.power = 400.0
    tool.active = True
    n0 = _active_count(pg.model)
    for _ in range(150):
        state.step(tool, dt=1 / 60.0)
    n1 = _active_count(pg.model)
    assert n1 < n0, f"expected at least one particle killed; got {n0} -> {n1}"
    ss = pg.model.spring_stiffness.numpy()
    assert int((ss == 0).sum()) > 0, "expected at least one spring disabled"


def test_muscle_cuts_faster_than_bone():
    # Two adjacent voxels of muscle and bone; the probe envelops both. Bone
    # has a higher resistance threshold (100 vs 80) AND lower conductivity
    # (0.2 vs 0.9), so it should take strictly more steps to die.
    def _setup(label: int) -> tuple[int, int, object, object]:
        labels = np.zeros((5, 5, 5), dtype=np.uint8)
        labels[2, 2, 2] = label
        atlas = DigimouseAtlas(labels=labels, voxel_size=0.005, materials=MaterialTable(DEFAULT_MATERIALS))
        pg = build_grid(atlas)
        state = make_cutting_state(pg.model, pg.aux)
        tool = Tool.cautery_probe(length=0.025, radius=0.01, device=pg.model.device)
        # Probe tip centred over the single particle.
        p = pg.model.particle_q.numpy()[0]
        tool.set_pose(
            np.asarray([[p[0], p[1], 0.0]], dtype=np.float32),
            np.asarray([[p[0], p[1], 0.025]], dtype=np.float32),
        )
        tool.power = 250.0
        tool.active = True
        return pg, state, tool

    def _steps_to_die(label: int, max_steps: int = 500) -> int:
        pg, state, tool = _setup(label)
        flags = pg.model.particle_flags
        for step in range(max_steps):
            state.step(tool, dt=1 / 60.0)
            if (flags.numpy()[0] & int(ParticleFlags.ACTIVE)) == 0:
                return step
        return max_steps

    muscle_steps = _steps_to_die(MUSCLE.id)
    bone_steps = _steps_to_die(3)  # BONE id
    assert muscle_steps < bone_steps, (
        f"muscle should die before bone under identical exposure; "
        f"muscle_steps={muscle_steps} bone_steps={bone_steps}"
    )
