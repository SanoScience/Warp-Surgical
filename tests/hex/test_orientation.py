# SPDX-License-Identifier: Apache-2.0
"""Oriented-particle update kernel invariants.

Ports the expected behaviour of ``SpringGrid::updateOrientation`` from
springGrid.h:612-660.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from omnisurg.hex.grid import build_grid
from omnisurg.hex.io.digimouse import DigimouseAtlas
from omnisurg.hex.kernels.orientation import (
    initialize_orientations_kernel,
    update_orientation_kernel,
)
from omnisurg.hex.materials import DEFAULT_MATERIALS, MaterialTable


def _solid_block(shape=(3, 3, 3)) -> DigimouseAtlas:
    labels = np.ones(shape, dtype=np.uint8)
    return DigimouseAtlas(labels=labels, voxel_size=0.01, materials=MaterialTable(DEFAULT_MATERIALS))


def _seed_identity(num_particles, device):
    arr = wp.zeros(num_particles, dtype=wp.mat33, device=device)
    wp.launch(initialize_orientations_kernel, dim=num_particles, inputs=[], outputs=[arr], device=device)
    return arr


def test_rest_state_preserves_identity():
    pg = build_grid(_solid_block((3, 3, 3)))
    N = pg.aux.num_particles
    a = _seed_identity(N, pg.model.device)
    b = wp.zeros_like(a)
    wp.launch(
        update_orientation_kernel,
        dim=N,
        inputs=[pg.model.particle_q, pg.model.particle_flags, pg.aux.particle_neighbors, a, 1.0, 0.1],
        outputs=[b],
        device=pg.model.device,
    )
    wp.synchronize()
    # For a solid grid at rest, each row should stay axis-aligned and unit-length.
    o = b.numpy()
    I = np.eye(3, dtype=np.float32)
    for i in range(N):
        assert np.allclose(o[i], I, atol=1e-5), f"particle {i} drifted: {o[i]}"


def test_interior_particle_row_norms_unit():
    pg = build_grid(_solid_block((5, 5, 5)))
    N = pg.aux.num_particles
    a = _seed_identity(N, pg.model.device)
    b = wp.zeros_like(a)
    # Perturb one interior particle to break symmetry.
    q = pg.model.particle_q.numpy().copy()
    q[0] += np.asarray([0.005, 0.0, 0.0], dtype=np.float32)
    pg.model.particle_q.assign(q)
    wp.launch(
        update_orientation_kernel,
        dim=N,
        inputs=[pg.model.particle_q, pg.model.particle_flags, pg.aux.particle_neighbors, a, 1.0, 0.1],
        outputs=[b],
        device=pg.model.device,
    )
    wp.synchronize()
    o = b.numpy()
    # Every row must be ~unit length (normalise step in the kernel).
    for i in range(N):
        for axis in range(3):
            n = float(np.linalg.norm(o[i, axis]))
            assert abs(n - 1.0) < 1e-4 or n < 1e-6, f"row {axis} on particle {i} has norm {n}"


def test_inactive_particle_keeps_previous_frame():
    from newton._src.geometry.flags import ParticleFlags

    pg = build_grid(_solid_block((2, 2, 2)))
    N = pg.aux.num_particles
    a = _seed_identity(N, pg.model.device)
    b = wp.zeros_like(a)
    # Clear ACTIVE bit on particle 0.
    flags = pg.model.particle_flags.numpy().copy()
    flags[0] &= ~int(ParticleFlags.ACTIVE)
    pg.model.particle_flags.assign(flags)
    # Seed particle 0's prev frame to something non-identity so we can detect a clobber.
    seeded = a.numpy().copy()
    seeded[0] = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    a.assign(seeded)
    wp.launch(
        update_orientation_kernel,
        dim=N,
        inputs=[pg.model.particle_q, pg.model.particle_flags, pg.aux.particle_neighbors, a, 1.0, 0.1],
        outputs=[b],
        device=pg.model.device,
    )
    wp.synchronize()
    assert np.allclose(b.numpy()[0], seeded[0])
