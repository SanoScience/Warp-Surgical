# SPDX-License-Identifier: Apache-2.0
"""Grid builder correctness checks on a tiny synthetic atlas."""

from __future__ import annotations

import numpy as np

from omnisurg.hex.grid import NEI_NX, NEI_NY, NEI_NZ, NEI_PX, NEI_PY, NEI_PZ, build_grid
from omnisurg.hex.io.digimouse import DigimouseAtlas
from omnisurg.hex.materials import DEFAULT_MATERIALS, MaterialTable


def _fake_atlas(shape=(3, 3, 3)) -> DigimouseAtlas:
    # Solid 3x3x3 block of "muscle" (material id 1).
    labels = np.ones(shape, dtype=np.uint8)
    return DigimouseAtlas(
        labels=labels,
        voxel_size=0.01,
        materials=MaterialTable(DEFAULT_MATERIALS),
    )


def test_solid_block_counts():
    pg = build_grid(_fake_atlas(), spring_ke=1e3, spring_kd=1.0)
    # 27 particles in a 3^3 block, springs per axis = 2*3*3 = 18 so 54 total.
    assert pg.aux.num_particles == 27
    assert pg.aux.num_springs == 54


def test_neighbors_are_symmetric():
    pg = build_grid(_fake_atlas(shape=(2, 2, 2)))
    neigh = pg.aux.particle_neighbors.numpy()
    pairs_by_axis = [
        (NEI_PX, NEI_NX),
        (NEI_PY, NEI_NY),
        (NEI_PZ, NEI_NZ),
    ]
    for fwd, back in pairs_by_axis:
        for i in range(pg.aux.num_particles):
            j = int(neigh[i, fwd])
            if j >= 0:
                assert int(neigh[j, back]) == i, f"neighbour asymmetry on axis {fwd}"


def test_bone_particles_are_kinematic():
    labels = np.zeros((3, 3, 3), dtype=np.uint8)
    labels[1, 1, 1] = 3  # BONE
    labels[0, 0, 0] = 1  # MUSCLE
    atlas = DigimouseAtlas(labels=labels, voxel_size=0.01, materials=MaterialTable(DEFAULT_MATERIALS))
    pg = build_grid(atlas)
    mats = pg.aux.particle_material.numpy()
    masses = np.asarray([pg.model.particle_mass.numpy()[i] for i in range(pg.aux.num_particles)])
    # Exactly one bone and one muscle particle.
    bone_idx = int(np.where(mats == 3)[0][0])
    muscle_idx = int(np.where(mats == 1)[0][0])
    assert masses[bone_idx] == 0.0  # kinematic
    assert masses[muscle_idx] > 0.0


def test_origin_offset_respected():
    pg = build_grid(_fake_atlas(shape=(1, 1, 1)), origin=(1.0, 2.0, 3.0))
    pos = pg.model.particle_q.numpy()[0]
    expected = np.asarray([1.0 + 0.5 * 0.01, 2.0 + 0.5 * 0.01, 3.0 + 0.5 * 0.01], dtype=np.float32)
    assert np.allclose(pos, expected)
