# SPDX-License-Identifier: Apache-2.0
"""Deformable marching-cubes smoke tests on small synthetic grids."""

from __future__ import annotations

import numpy as np
import warp as wp
from newton._src.geometry.flags import ParticleFlags

from omnisurg.hex.grid import build_grid
from omnisurg.hex.io.digimouse import DigimouseAtlas
from omnisurg.hex.kernels import _mc_tables_gen as tables
from omnisurg.hex.kernels.marching_cubes import (
    MC_MAX_TRIS_PER_CASE,
    allocate_mc_buffers,
    bake_vertex_uv3,
    compute_mc_topology,
    compute_mc_topology_dirty,
    run_marching_cubes,
    upload_mc_tables,
)
from omnisurg.hex.kernels.orientation import initialize_orientations_kernel
from omnisurg.hex.materials import DEFAULT_MATERIALS, MaterialTable


def _padded_block(core=(3, 3, 3), pad=2) -> DigimouseAtlas:
    """Solid ``core`` of label 1 surrounded by ``pad`` voxels of background on all sides."""
    nx, ny, nz = (2 * pad + c for c in core)
    labels = np.zeros((nx, ny, nz), dtype=np.uint8)
    labels[pad : pad + core[0], pad : pad + core[1], pad : pad + core[2]] = 1
    return DigimouseAtlas(labels=labels, voxel_size=0.01, materials=MaterialTable(DEFAULT_MATERIALS))


def _run_mc(pg):
    dev = pg.model.device
    wp.launch(
        initialize_orientations_kernel,
        dim=pg.aux.num_particles,
        inputs=[],
        outputs=[pg.aux.particle_orientation],
        device=dev,
    )
    wp.synchronize()
    t = upload_mc_tables(device=dev)
    b = allocate_mc_buffers(pg.aux.grid_shape, pg.aux.num_particles, device=dev)
    count = run_marching_cubes(
        pg.model.particle_q,
        pg.model.particle_flags,
        pg.aux.particle_orientation,
        pg.aux.particle_grid_xyz,
        pg.aux.grid_to_particle,
        pg.aux.grid_shape,
        t,
        b,
        mc_factor=pg.aux.voxel_size * 0.5,
        device=dev,
    )
    return count, b


def _sorted_tris(tris: np.ndarray) -> list[tuple[int, int, int]]:
    if tris.size == 0:
        return []
    canonical = np.sort(np.asarray(tris, dtype=np.int32), axis=1)
    return sorted(tuple(int(v) for v in tri) for tri in canonical.tolist())


def test_mc_tables_are_well_formed():
    assert tables.CASE_TRIANGLES.shape == (256, 16)
    assert tables.CORNER_OFFSETS.shape == (8, 3)
    assert tables.EDGE_CORNERS.shape == (12, 2)
    assert tables.EDGE_BASE_DIR.shape == (12,)
    # empty and full cases emit no triangles.
    assert (tables.CASE_TRIANGLES[0] == -1).all()
    assert (tables.CASE_TRIANGLES[255] == -1).all()
    assert MC_MAX_TRIS_PER_CASE == 5


def test_bake_vertex_uv3_writes_directional_surface_offsets_and_cell_centres():
    dev = "cpu"
    buffers = allocate_mc_buffers((5, 7, 9), 1, device=dev)
    particle_grid_xyz = wp.array(np.asarray([[2, 3, 4]], dtype=np.int32), dtype=wp.int32, device=dev)

    bake_vertex_uv3(buffers, particle_grid_xyz, (5, 7, 9), device=dev)

    np.testing.assert_allclose(buffers.particle_uv3.numpy(), np.asarray([[0.5, 0.5, 0.5]], dtype=np.float32))
    np.testing.assert_allclose(
        buffers.vertex_uv3.numpy()[:6],
        np.asarray(
            [
                [(2.0 - 0.5) / 4.0, 0.5, 0.5],
                [(2.0 + 0.5) / 4.0, 0.5, 0.5],
                [0.5, (3.0 - 0.5) / 6.0, 0.5],
                [0.5, (3.0 + 0.5) / 6.0, 0.5],
                [0.5, 0.5, (4.0 - 0.5) / 8.0],
                [0.5, 0.5, (4.0 + 0.5) / 8.0],
            ],
            dtype=np.float32,
        ),
        atol=1.0e-7,
    )


def test_solid_block_produces_closed_surface():
    pg = build_grid(_padded_block())
    count, buffers = _run_mc(pg)
    assert count > 0, "solid block with padding should emit boundary triangles"
    tris = buffers.tri_indices.numpy()[:count]
    # Each edge of the surface must be shared by exactly two triangles (closed
    # manifold). Collect undirected edges and assert the multiset is balanced.
    edges: dict[tuple[int, int], int] = {}
    for a, b, c in tris.tolist():
        for x, y in ((a, b), (b, c), (c, a)):
            key = (min(x, y), max(x, y))
            edges[key] = edges.get(key, 0) + 1
    boundary_edges = [e for e, n in edges.items() if n != 2]
    assert not boundary_edges, f"surface is not closed; {len(boundary_edges)} boundary edges"


def test_triangle_count_grows_when_particles_are_cut():
    pg = build_grid(_padded_block(core=(5, 5, 5)))
    count_before, _ = _run_mc(pg)
    # Clear ACTIVE bit on a single interior particle to carve a hole.
    flags = pg.model.particle_flags.numpy().copy()
    # Pick the centre particle (core was 5x5x5 inside pad=2, so grid coord (4,4,4))
    centre = pg.aux.grid_to_particle.numpy()[4, 4, 4]
    assert centre >= 0
    flags[centre] &= ~int(ParticleFlags.ACTIVE)
    pg.model.particle_flags.assign(flags)
    count_after, _ = _run_mc(pg)
    assert count_after > count_before, (
        f"hole should introduce more triangles; before={count_before} after={count_after}"
    )


def test_case_index_packs_from_corner_flags():
    # Use a 2-voxel block along X and check the single cube cell's case.
    labels = np.zeros((4, 3, 3), dtype=np.uint8)
    labels[1:3, 1:2, 1:2] = 1  # two particles adjacent on X
    atlas = DigimouseAtlas(labels=labels, voxel_size=0.01, materials=MaterialTable(DEFAULT_MATERIALS))
    pg = build_grid(atlas)
    _, buffers = _run_mc(pg)
    cases = buffers.cube_cases.numpy()
    # The cell at grid (1,1,1) holds corners (0,0,0),(1,0,0),(1,0,1),(0,0,1),(0,1,0),(1,1,0),(1,1,1),(0,1,1).
    # Only corners 0 (=(1,1,1)) and 1 (=(2,1,1)) map to particles, so case bits 0 and 1 should be set -> 3.
    assert int(cases[1, 1, 1]) == 3


def test_dirty_topology_matches_full_rebuild_after_cell_cut():
    pg = build_grid(_padded_block(core=(5, 5, 5)))
    dev = pg.model.device
    wp.launch(
        initialize_orientations_kernel,
        dim=pg.aux.num_particles,
        inputs=[],
        outputs=[pg.aux.particle_orientation],
        device=dev,
    )
    tables_dev = upload_mc_tables(device=dev)
    dirty_buffers = allocate_mc_buffers(pg.aux.grid_shape, pg.aux.num_particles, device=dev)
    full_buffers = allocate_mc_buffers(pg.aux.grid_shape, pg.aux.num_particles, device=dev)

    # Prime the dirty buffers with a full topology so their fixed-slot maps are valid.
    initial_dirty_count = compute_mc_topology(
        dirty_buffers,
        tables_dev,
        pg.model.particle_flags,
        pg.aux.grid_to_particle,
        device=dev,
    )
    assert initial_dirty_count > 0

    flags = pg.model.particle_flags.numpy().copy()
    grid = pg.aux.grid_to_particle.numpy()
    cut_particles = np.asarray([grid[4, 4, 4], grid[4, 4, 5]], dtype=np.int32)
    assert np.all(cut_particles >= 0)
    flags[cut_particles] &= ~int(ParticleFlags.ACTIVE)
    pg.model.particle_flags.assign(flags)

    dirty_ids = wp.array(cut_particles, dtype=wp.int32, device=dev)
    dirty_count = compute_mc_topology_dirty(
        dirty_buffers,
        tables_dev,
        pg.model.particle_flags,
        pg.aux.grid_to_particle,
        pg.aux.particle_grid_xyz,
        dirty_ids,
        int(cut_particles.size),
        device=dev,
    )
    assert dirty_count is not None

    full_count = compute_mc_topology(
        full_buffers,
        tables_dev,
        pg.model.particle_flags,
        pg.aux.grid_to_particle,
        device=dev,
    )
    dirty_tris = dirty_buffers.tri_indices.numpy()[:dirty_count]
    full_tris = full_buffers.tri_indices.numpy()[:full_count]
    assert _sorted_tris(dirty_tris) == _sorted_tris(full_tris)
