# SPDX-License-Identifier: Apache-2.0
"""Build a Newton ``Model`` from a labelled voxel grid.

One particle per occupied voxel, 6-neighbour springs between orthogonally
adjacent active particles, and bone-labelled voxels pinned as kinematic
(zero mass) so tissue slides against them without introducing a rigid-body
solve. This mirrors the construction in
``EfficientPBDCutting/ufrgs/springGrid.h`` but defers the paper's full rigid
cluster flood-fill (``createRigids``) to a later phase.

The builder returns both a ``newton.Model`` and a ``GridAuxState`` holding
the auxiliary GPU arrays (material, grid coords, neighbour indices, heat,
orientation frames, the inverse voxel-to-particle lookup, and per-spring
enable flags) that downstream kernels use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton

from .io.digimouse import DigimouseAtlas
from .materials import MaterialTable, Phase

# Indices into the 6-neighbour table: -X, +X, -Y, +Y, -Z, +Z. Keep in sync
# with the kernels in ``kernels/orientation.py`` and ``kernels/heat.py``.
NEI_NX = 0
NEI_PX = 1
NEI_NY = 2
NEI_PY = 3
NEI_NZ = 4
NEI_PZ = 5


@dataclass
class GridAuxState:
    """GPU-resident auxiliary arrays that travel alongside ``newton.State``.

    All arrays are indexed by particle id unless noted otherwise. ``grid_shape``
    is ``(nx, ny, nz)`` in voxel units and ``voxel_size`` is the edge length in
    metres. ``grid_to_particle`` is an ``int32`` volume where -1 means an
    unoccupied voxel and any non-negative value is the particle id. That
    inverse lookup is what lets the marching-cubes kernel iterate cube cells
    and fetch their 8 corner particles in O(1).
    """

    particle_material: wp.array  # int32[N]
    particle_grid_xyz: wp.array  # int32[N, 3]
    particle_neighbors: wp.array  # int32[N, 6]
    particle_heat: wp.array  # float32[N]
    particle_burnt: wp.array  # float32[N]
    particle_orientation: wp.array  # mat33[N]
    grid_to_particle: wp.array  # int32[nx, ny, nz]
    spring_enabled: wp.array  # int32[num_springs]

    grid_shape: tuple[int, int, int]
    voxel_size: float
    origin: tuple[float, float, float]
    num_particles: int
    num_springs: int
    materials: MaterialTable


@dataclass
class ParticleGrid:
    """Result of :func:`build_grid` - a finalised Newton model plus aux state."""

    model: newton.Model
    state: newton.State
    aux: GridAuxState


def build_grid(
    atlas: DigimouseAtlas,
    spring_ke: float = 1.0e3,
    spring_kd: float = 1.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    device: str | wp.context.Device | None = None,
    builder: newton.ModelBuilder | None = None,
    kinematic_bones: bool = True,
) -> ParticleGrid:
    """Build a Newton model from a Digimouse atlas.

    Args:
        atlas: Parsed atlas whose ``labels`` index into ``atlas.materials``.
        spring_ke: Elastic stiffness for soft-body springs (XPBD units).
        spring_kd: Damping for soft-body springs.
        origin: World-space position of the ``[0,0,0]`` voxel centre (m).
        device: Warp device for the auxiliary arrays. Defaults to Warp's
            current preferred device.
        builder: Optional pre-existing builder to append to. When ``None`` a
            fresh ``newton.ModelBuilder`` is created.
        kinematic_bones: when True (default), skeleton particles are pinned
            with mass 0 so tissue slides against them. Set False to give
            bones the material-defined mass, so the whole body falls together
            under gravity - useful for free-fall / cutting-with-gravity demos
            until the paper's rigid-cluster flood-fill is implemented.

    Returns:
        A ``ParticleGrid`` whose ``model`` is finalised and whose ``aux``
        holds the kernel-facing arrays. The returned ``state`` is the initial
        state from ``model.state()``.
    """
    labels = atlas.labels
    nx, ny, nz = labels.shape
    voxel_size = float(atlas.voxel_size)
    ox, oy, oz = origin

    materials = atlas.materials
    phase_lut = materials.phase  # int32[n_materials]

    # --- Pass 1: assign particle ids to occupied voxels. -----------------
    occupied_mask = labels != 0
    num_particles = int(occupied_mask.sum())
    if num_particles == 0:
        raise ValueError("atlas has no occupied voxels; nothing to build")

    grid_to_particle = np.full((nx, ny, nz), -1, dtype=np.int32)
    occ_coords = np.argwhere(occupied_mask).astype(np.int32)  # (N, 3)
    grid_to_particle[occ_coords[:, 0], occ_coords[:, 1], occ_coords[:, 2]] = np.arange(num_particles, dtype=np.int32)

    # --- Pass 2: per-particle scalar fields. -----------------------------
    particle_labels = labels[occ_coords[:, 0], occ_coords[:, 1], occ_coords[:, 2]].astype(np.int32)
    particle_phase = phase_lut[particle_labels]  # int32[N]
    particle_mass = materials.mass[particle_labels].astype(np.float32)
    if kinematic_bones:
        # Mass 0 pins bone particles so XPBD treats them as fixed colliders.
        particle_mass[particle_phase == int(Phase.RIGID)] = 0.0

    positions = (occ_coords.astype(np.float32) + 0.5) * voxel_size + np.asarray([ox, oy, oz], dtype=np.float32)
    velocities = np.zeros((num_particles, 3), dtype=np.float32)

    # --- Pass 3: push everything into the Newton builder. -----------------
    if builder is None:
        builder = newton.ModelBuilder()
    particle_radius = voxel_size * 0.5
    builder.add_particles(
        pos=[tuple(p) for p in positions.tolist()],
        vel=[tuple(v) for v in velocities.tolist()],
        mass=particle_mass.tolist(),
        radius=[particle_radius] * num_particles,
    )

    # --- Pass 4: 6-neighbour index table. ---------------------------------
    # neighbour directions in (dx, dy, dz) aligned with NEI_* constants.
    dirs = np.array(
        [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ],
        dtype=np.int32,
    )
    particle_neighbors = np.full((num_particles, 6), -1, dtype=np.int32)
    for d, (dx, dy, dz) in enumerate(dirs):
        nbr = np.roll(grid_to_particle, shift=(-dx, -dy, -dz), axis=(0, 1, 2))
        # Kill wrapped-around entries on the boundary.
        if dx > 0:
            nbr[-1, :, :] = -1
        elif dx < 0:
            nbr[0, :, :] = -1
        if dy > 0:
            nbr[:, -1, :] = -1
        elif dy < 0:
            nbr[:, 0, :] = -1
        if dz > 0:
            nbr[:, :, -1] = -1
        elif dz < 0:
            nbr[:, :, 0] = -1
        particle_neighbors[:, d] = nbr[occ_coords[:, 0], occ_coords[:, 1], occ_coords[:, 2]]

    # --- Pass 5: create springs along +X, +Y, +Z edges between active pairs.
    # Each orthogonal edge is visited exactly once (the particle with the
    # lower coordinate owns the edge), which mirrors ``addSpringCube`` in
    # springGrid.h:254-284 where only the first three pairs of the edge
    # lookup table are materialised.
    num_springs = 0
    for d, axis_positive in ((NEI_PX, 0), (NEI_PY, 1), (NEI_PZ, 2)):
        pairs = np.nonzero(particle_neighbors[:, d] >= 0)[0]
        for i in pairs.tolist():
            j = int(particle_neighbors[i, d])
            builder.add_spring(i, j, ke=spring_ke, kd=spring_kd, control=0.0)
            num_springs += 1

    # --- Finalise the model and upload aux arrays to the device. ---------
    model = builder.finalize(device=device)
    state = model.state()

    device = wp.get_device(device) if device is not None else model.device
    aux = GridAuxState(
        particle_material=wp.array(particle_labels, dtype=wp.int32, device=device),
        particle_grid_xyz=wp.array(occ_coords, dtype=wp.int32, device=device),
        particle_neighbors=wp.array(particle_neighbors, dtype=wp.int32, device=device),
        particle_heat=wp.zeros(num_particles, dtype=wp.float32, device=device),
        particle_burnt=wp.zeros(num_particles, dtype=wp.float32, device=device),
        particle_orientation=wp.zeros(num_particles, dtype=wp.mat33, device=device),
        grid_to_particle=wp.array(grid_to_particle, dtype=wp.int32, device=device),
        spring_enabled=wp.ones(num_springs, dtype=wp.int32, device=device),
        grid_shape=(nx, ny, nz),
        voxel_size=voxel_size,
        origin=origin,
        num_particles=num_particles,
        num_springs=num_springs,
        materials=materials,
    )
    return ParticleGrid(model=model, state=state, aux=aux)
