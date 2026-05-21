# SPDX-License-Identifier: Apache-2.0
"""GPU-resident render layer for the springless hex particle lattice.

The physics lives on shared cell vertices but the existing
marching-cubes pipeline expects one render particle per voxel centre with a
local frame. These kernels bridge the two by writing, per frame and per cell:

* ``cell_center_q`` - mean of the 8 vertex positions
* ``cell_orientation`` - intrinsic mat33 from averaged cell-edge vectors
* ``cell_render_flags`` - ``ParticleFlags.ACTIVE`` iff ``cell_active != 0``
* ``cell_stretch`` - max absolute cell-edge/diagonal strain for display

Render occupancy keys off ``cell_active``, not the node ACTIVE flag: a deleted
cell can leave all 8 particles still ACTIVE via support from neighbours, so
using node flags would never open cavities in the MC surface.

Also provides a GPU hover picker: a ray-AABB slab test over active cells with
a float-min + argmin two-pass reduction, returning a single int per frame.
When given a material cuttable mask, non-cuttable cell AABBs are transparent
to the pick ray.
"""

from __future__ import annotations

import warp as wp
from newton._src.geometry.flags import ParticleFlags


@wp.func
def _orthonormalize(ax: wp.vec3, ay: wp.vec3, az: wp.vec3) -> wp.mat33:
    # Gram-Schmidt starting from ax. Cheap and deterministic; intrinsic per
    # cell so we never need a neighbour table like the voxel-centre path.
    nx = wp.length(ax)
    if nx < 1.0e-12:
        ax = wp.vec3(1.0, 0.0, 0.0)
    else:
        ax = ax / nx
    ay = ay - ax * wp.dot(ax, ay)
    ny = wp.length(ay)
    if ny < 1.0e-12:
        # Degenerate: fall back to a stable orthogonal by picking the world
        # axis least parallel to ax.
        if wp.abs(ax[0]) < 0.9:
            ay = wp.vec3(1.0, 0.0, 0.0) - ax * ax[0]
        else:
            ay = wp.vec3(0.0, 1.0, 0.0) - ax * ax[1]
        ay = ay / wp.length(ay)
    else:
        ay = ay / ny
    az = wp.cross(ax, ay)
    # Rows are the three local axes, matching `kernels/marching_cubes.py`
    # `compute_vertex_positions_kernel` which reads `frame[axis, 0..2]`.
    return wp.mat33(
        ax[0], ax[1], ax[2],
        ay[0], ay[1], ay[2],
        az[0], az[1], az[2],
    )


@wp.func
def _constraint_abs_strain(a: wp.vec3, b: wp.vec3, rest_length: float) -> float:
    if rest_length <= 1.0e-12:
        return 0.0
    return wp.abs((wp.length(b - a) / rest_length) - 1.0)


@wp.func
def _cell_max_abs_strain(
    p0: wp.vec3,
    p1: wp.vec3,
    p2: wp.vec3,
    p3: wp.vec3,
    p4: wp.vec3,
    p5: wp.vec3,
    p6: wp.vec3,
    p7: wp.vec3,
    voxel_size: float,
) -> float:
    edge = voxel_size
    face = voxel_size * 1.4142135623730951
    body = voxel_size * 1.7320508075688772
    s = float(0.0)

    s = wp.max(s, _constraint_abs_strain(p0, p1, edge))
    s = wp.max(s, _constraint_abs_strain(p1, p2, edge))
    s = wp.max(s, _constraint_abs_strain(p2, p3, edge))
    s = wp.max(s, _constraint_abs_strain(p3, p0, edge))
    s = wp.max(s, _constraint_abs_strain(p4, p5, edge))
    s = wp.max(s, _constraint_abs_strain(p5, p6, edge))
    s = wp.max(s, _constraint_abs_strain(p6, p7, edge))
    s = wp.max(s, _constraint_abs_strain(p7, p4, edge))
    s = wp.max(s, _constraint_abs_strain(p0, p4, edge))
    s = wp.max(s, _constraint_abs_strain(p1, p5, edge))
    s = wp.max(s, _constraint_abs_strain(p2, p6, edge))
    s = wp.max(s, _constraint_abs_strain(p3, p7, edge))

    s = wp.max(s, _constraint_abs_strain(p0, p2, face))
    s = wp.max(s, _constraint_abs_strain(p1, p3, face))
    s = wp.max(s, _constraint_abs_strain(p4, p6, face))
    s = wp.max(s, _constraint_abs_strain(p5, p7, face))
    s = wp.max(s, _constraint_abs_strain(p0, p5, face))
    s = wp.max(s, _constraint_abs_strain(p1, p4, face))
    s = wp.max(s, _constraint_abs_strain(p3, p6, face))
    s = wp.max(s, _constraint_abs_strain(p2, p7, face))
    s = wp.max(s, _constraint_abs_strain(p0, p7, face))
    s = wp.max(s, _constraint_abs_strain(p3, p4, face))
    s = wp.max(s, _constraint_abs_strain(p1, p6, face))
    s = wp.max(s, _constraint_abs_strain(p2, p5, face))

    s = wp.max(s, _constraint_abs_strain(p0, p6, body))
    s = wp.max(s, _constraint_abs_strain(p1, p7, body))
    s = wp.max(s, _constraint_abs_strain(p2, p4, body))
    s = wp.max(s, _constraint_abs_strain(p3, p5, body))
    return s


@wp.kernel
def update_cell_render_state_kernel(
    cell_nodes: wp.array2d(dtype=wp.int32),  # [num_cells, 8]
    cell_active: wp.array(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    voxel_size: float,
    compute_stretch: int,
    cell_center_q: wp.array(dtype=wp.vec3),
    cell_orientation: wp.array(dtype=wp.mat33),
    cell_render_flags: wp.array(dtype=wp.int32),
    cell_stretch: wp.array(dtype=wp.float32),
):
    """Write cell centre, intrinsic frame, and render flag for every cell.

    Vertex ordering matches ``CELL_CORNER_OFFSETS`` in hex_grid.py:
    ``0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0) 4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)``.

    The frame axes come from averaged vertex-pair deltas:
    * ``ax`` = mean of (c1-c0, c2-c3, c5-c4, c6-c7)  (local +X edges)
    * ``ay`` = mean of (c3-c0, c2-c1, c7-c4, c6-c5)  (local +Y edges)
    * ``az`` = mean of (c4-c0, c5-c1, c6-c2, c7-c3)  (local +Z edges)

    Orthonormalised via Gram-Schmidt. Deleted cells still get a valid frame
    written (cheap) but their render flag clears ACTIVE so the MC pipeline
    skips them at ``compute_cube_cases_kernel`` time.
    """
    c = wp.tid()
    n0 = cell_nodes[c, 0]
    n1 = cell_nodes[c, 1]
    n2 = cell_nodes[c, 2]
    n3 = cell_nodes[c, 3]
    n4 = cell_nodes[c, 4]
    n5 = cell_nodes[c, 5]
    n6 = cell_nodes[c, 6]
    n7 = cell_nodes[c, 7]

    p0 = particle_q[n0]
    p1 = particle_q[n1]
    p2 = particle_q[n2]
    p3 = particle_q[n3]
    p4 = particle_q[n4]
    p5 = particle_q[n5]
    p6 = particle_q[n6]
    p7 = particle_q[n7]

    centre = (p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7) * (1.0 / 8.0)
    cell_center_q[c] = centre

    ax = ((p1 - p0) + (p2 - p3) + (p5 - p4) + (p6 - p7)) * 0.25
    ay = ((p3 - p0) + (p2 - p1) + (p7 - p4) + (p6 - p5)) * 0.25
    az = ((p4 - p0) + (p5 - p1) + (p6 - p2) + (p7 - p3)) * 0.25

    cell_orientation[c] = _orthonormalize(ax, ay, az)

    if cell_active[c] != 0:
        cell_render_flags[c] = wp.int32(ParticleFlags.ACTIVE)
        if compute_stretch != 0:
            cell_stretch[c] = _cell_max_abs_strain(p0, p1, p2, p3, p4, p5, p6, p7, voxel_size)
    else:
        cell_render_flags[c] = wp.int32(0)
        if compute_stretch != 0:
            cell_stretch[c] = 0.0


def update_cell_render_state(
    aux,
    particle_q: wp.array,
    device: str | wp.context.Device | None = None,
    compute_stretch: bool = False,
) -> None:
    """Launch the per-frame render-state update. ``aux`` is a HexGridAuxState."""
    wp.launch(
        update_cell_render_state_kernel,
        dim=int(aux.num_cells),
        inputs=[aux.cell_nodes, aux.cell_active, particle_q, float(aux.voxel_size), int(bool(compute_stretch))],
        outputs=[aux.cell_center_q, aux.cell_orientation, aux.cell_render_flags, aux.cell_stretch],
        device=device,
    )


@wp.kernel
def update_cell_render_state_and_aabbs_kernel(
    cell_nodes: wp.array2d(dtype=wp.int32),  # [num_cells, 8]
    cell_active: wp.array(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    voxel_size: float,
    compute_stretch: int,
    cell_center_q: wp.array(dtype=wp.vec3),
    cell_orientation: wp.array(dtype=wp.mat33),
    cell_render_flags: wp.array(dtype=wp.int32),
    cell_stretch: wp.array(dtype=wp.float32),
    aabb_min: wp.array(dtype=wp.vec3),
    aabb_max: wp.array(dtype=wp.vec3),
):
    """Write render state and picker AABBs from the same 8 corner loads."""
    c = wp.tid()
    n0 = cell_nodes[c, 0]
    n1 = cell_nodes[c, 1]
    n2 = cell_nodes[c, 2]
    n3 = cell_nodes[c, 3]
    n4 = cell_nodes[c, 4]
    n5 = cell_nodes[c, 5]
    n6 = cell_nodes[c, 6]
    n7 = cell_nodes[c, 7]

    p0 = particle_q[n0]
    p1 = particle_q[n1]
    p2 = particle_q[n2]
    p3 = particle_q[n3]
    p4 = particle_q[n4]
    p5 = particle_q[n5]
    p6 = particle_q[n6]
    p7 = particle_q[n7]

    lo = wp.vec3(
        wp.min(wp.min(wp.min(p0[0], p1[0]), wp.min(p2[0], p3[0])), wp.min(wp.min(p4[0], p5[0]), wp.min(p6[0], p7[0]))),
        wp.min(wp.min(wp.min(p0[1], p1[1]), wp.min(p2[1], p3[1])), wp.min(wp.min(p4[1], p5[1]), wp.min(p6[1], p7[1]))),
        wp.min(wp.min(wp.min(p0[2], p1[2]), wp.min(p2[2], p3[2])), wp.min(wp.min(p4[2], p5[2]), wp.min(p6[2], p7[2]))),
    )
    hi = wp.vec3(
        wp.max(wp.max(wp.max(p0[0], p1[0]), wp.max(p2[0], p3[0])), wp.max(wp.max(p4[0], p5[0]), wp.max(p6[0], p7[0]))),
        wp.max(wp.max(wp.max(p0[1], p1[1]), wp.max(p2[1], p3[1])), wp.max(wp.max(p4[1], p5[1]), wp.max(p6[1], p7[1]))),
        wp.max(wp.max(wp.max(p0[2], p1[2]), wp.max(p2[2], p3[2])), wp.max(wp.max(p4[2], p5[2]), wp.max(p6[2], p7[2]))),
    )
    aabb_min[c] = lo
    aabb_max[c] = hi

    cell_center_q[c] = (p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7) * (1.0 / 8.0)

    ax = ((p1 - p0) + (p2 - p3) + (p5 - p4) + (p6 - p7)) * 0.25
    ay = ((p3 - p0) + (p2 - p1) + (p7 - p4) + (p6 - p5)) * 0.25
    az = ((p4 - p0) + (p5 - p1) + (p6 - p2) + (p7 - p3)) * 0.25
    cell_orientation[c] = _orthonormalize(ax, ay, az)

    if cell_active[c] != 0:
        cell_render_flags[c] = wp.int32(ParticleFlags.ACTIVE)
        if compute_stretch != 0:
            cell_stretch[c] = _cell_max_abs_strain(p0, p1, p2, p3, p4, p5, p6, p7, voxel_size)
    else:
        cell_render_flags[c] = wp.int32(0)
        if compute_stretch != 0:
            cell_stretch[c] = 0.0


def update_cell_render_state_and_aabbs(
    aux,
    particle_q: wp.array,
    aabb_min: wp.array,
    aabb_max: wp.array,
    device: str | wp.context.Device | None = None,
    compute_stretch: bool = False,
) -> None:
    """Refresh render state and picker AABBs in one pass."""
    wp.launch(
        update_cell_render_state_and_aabbs_kernel,
        dim=int(aux.num_cells),
        inputs=[aux.cell_nodes, aux.cell_active, particle_q, float(aux.voxel_size), int(bool(compute_stretch))],
        outputs=[aux.cell_center_q, aux.cell_orientation, aux.cell_render_flags, aux.cell_stretch, aabb_min, aabb_max],
        device=device,
    )


# ---------------------------------------------------------------------------
# Hover picking: ray-AABB slab test + float-min argmin. Single int sync/frame.
# ---------------------------------------------------------------------------


@wp.kernel
def build_cell_aabbs_kernel(
    cell_nodes: wp.array2d(dtype=wp.int32),
    cell_active: wp.array(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    aabb_min: wp.array(dtype=wp.vec3),
    aabb_max: wp.array(dtype=wp.vec3),
):
    """Write per-cell AABBs over the 8 deformed corner positions.

    Inactive cells still get AABBs written (cheap); the pick kernel filters
    them by ``cell_active``. Writing a valid box also keeps us from sampling
    stale memory if a downstream reader ignores the active flag.
    """
    c = wp.tid()
    p = particle_q[cell_nodes[c, 0]]
    lo = p
    hi = p
    for k in range(1, 8):
        p = particle_q[cell_nodes[c, k]]
        lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))
        hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))
    aabb_min[c] = lo
    aabb_max[c] = hi


@wp.kernel
def build_dirty_cell_aabbs_kernel(
    dirty_cell_ids: wp.array(dtype=wp.int32),
    dirty_count: wp.array(dtype=wp.int32),
    num_cells: int,
    cell_nodes: wp.array2d(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    aabb_min: wp.array(dtype=wp.vec3),
    aabb_max: wp.array(dtype=wp.vec3),
):
    """Refresh picker AABBs for a compact device list of touched cells."""
    i = wp.tid()
    if i >= dirty_count[0]:
        return
    c = dirty_cell_ids[i]
    if c < 0 or c >= num_cells:
        return

    p = particle_q[cell_nodes[c, 0]]
    lo = p
    hi = p
    for k in range(1, 8):
        p = particle_q[cell_nodes[c, k]]
        lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))
        hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))
    aabb_min[c] = lo
    aabb_max[c] = hi


@wp.kernel
def pick_ray_cell_t_kernel(
    cell_active: wp.array(dtype=wp.int32),
    cell_material: wp.array(dtype=wp.int32),
    material_cuttable: wp.array(dtype=wp.int32),
    has_material_filter: int,
    aabb_min: wp.array(dtype=wp.vec3),
    aabb_max: wp.array(dtype=wp.vec3),
    ray_origin: wp.vec3,
    ray_dir: wp.vec3,
    cell_ray_t: wp.array(dtype=wp.float32),
    min_t: wp.array(dtype=wp.float32),
):
    """Per-cell slab test. Writes hit t (or +inf) and atomic-min into ``min_t[0]``."""
    c = wp.tid()
    if cell_active[c] == 0:
        cell_ray_t[c] = float(1.0e30)
        return
    if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:
        cell_ray_t[c] = float(1.0e30)
        return

    lo = aabb_min[c]
    hi = aabb_max[c]
    t_near = float(-1.0e30)
    t_far = float(1.0e30)

    # Slab test, one axis at a time. Mirrors the numpy version in example 07
    # but all per-cell threads run concurrently.
    for axis in range(3):
        d = ray_dir[axis]
        o = ray_origin[axis]
        amin = lo[axis]
        amax = hi[axis]
        if wp.abs(d) < 1.0e-8:
            if o < amin or o > amax:
                cell_ray_t[c] = float(1.0e30)
                return
        else:
            inv_d = 1.0 / d
            t0 = (amin - o) * inv_d
            t1 = (amax - o) * inv_d
            axis_near = wp.min(t0, t1)
            axis_far = wp.max(t0, t1)
            t_near = wp.max(t_near, axis_near)
            t_far = wp.min(t_far, axis_far)

    t_hit = wp.max(t_near, 0.0)
    if t_far < t_hit:
        cell_ray_t[c] = float(1.0e30)
        return

    cell_ray_t[c] = t_hit
    wp.atomic_min(min_t, 0, t_hit)


@wp.kernel
def pick_ray_cell_match_kernel(
    cell_ray_t: wp.array(dtype=wp.float32),
    min_t: wp.array(dtype=wp.float32),
    sentinel: float,
    hit_cell: wp.array(dtype=wp.int32),
):
    """Pick the lowest-index cell whose ``t`` matches ``min_t``."""
    c = wp.tid()
    if min_t[0] >= sentinel:
        return
    if cell_ray_t[c] == min_t[0]:
        wp.atomic_min(hit_cell, 0, c)


class HoverPicker:
    """Persistent GPU buffers + launch helpers for ray-AABB cell picking.

    Allocates once (sized by ``num_cells``) and reuses per frame. Use
    ``pick_device(...)`` on hot paths that consume the picked id in another
    kernel; ``pick(...)`` is the CPU convenience wrapper and synchronizes one
    int back to the host.
    """

    _SENTINEL: float = 1.0e30
    _NO_HIT_CELL: int = 0x7FFFFFFF

    def __init__(self, num_cells: int, device: wp.context.Device):
        self.num_cells = int(num_cells)
        self.device = device
        self._aabb_min = wp.zeros(self.num_cells, dtype=wp.vec3, device=device)
        self._aabb_max = wp.zeros(self.num_cells, dtype=wp.vec3, device=device)
        self._cell_ray_t = wp.zeros(self.num_cells, dtype=wp.float32, device=device)
        self._min_t = wp.zeros(1, dtype=wp.float32, device=device)
        self._hit_cell = wp.zeros(1, dtype=wp.int32, device=device)

    @property
    def hit_cell_device(self) -> wp.array:
        """Device array containing the last picked cell id or ``_NO_HIT_CELL``."""
        return self._hit_cell

    def refresh_aabbs(self, aux, particle_q: wp.array) -> None:
        """Rebuild per-cell AABBs from the current ``particle_q``."""
        wp.launch(
            build_cell_aabbs_kernel,
            dim=self.num_cells,
            inputs=[aux.cell_nodes, aux.cell_active, particle_q],
            outputs=[self._aabb_min, self._aabb_max],
            device=self.device,
        )

    def refresh_dirty_aabbs(
        self,
        aux,
        particle_q: wp.array,
        dirty_cell_ids: wp.array | None,
        dirty_count_device: wp.array | None,
        dirty_capacity: int | None,
    ) -> None:
        """Refresh AABBs for a compact device list without syncing the count."""
        if dirty_cell_ids is None or dirty_count_device is None:
            return
        capacity = int(dirty_capacity if dirty_capacity is not None else dirty_cell_ids.shape[0])
        capacity = min(capacity, int(dirty_cell_ids.shape[0]))
        if capacity <= 0:
            return
        wp.launch(
            build_dirty_cell_aabbs_kernel,
            dim=capacity,
            inputs=[dirty_cell_ids, dirty_count_device, self.num_cells, aux.cell_nodes, particle_q],
            outputs=[self._aabb_min, self._aabb_max],
            device=self.device,
        )

    def refresh_render_state_and_aabbs(self, aux, particle_q: wp.array, compute_stretch: bool = False) -> None:
        """Refresh render state and picker AABBs in one fused pass."""
        update_cell_render_state_and_aabbs(
            aux,
            particle_q,
            self._aabb_min,
            self._aabb_max,
            device=self.device,
            compute_stretch=compute_stretch,
        )

    def pick_device(
        self,
        aux,
        ray_origin: tuple[float, float, float],
        ray_dir: tuple[float, float, float],
        material_cuttable: wp.array | None = None,
    ) -> wp.array:
        """Run the ray-AABB reduction and leave the result on the device."""
        self._min_t.fill_(float(self._SENTINEL))
        # Using a very large int so atomic_min(hit_cell, 0, c) picks the lowest
        # cell id among all t-matches.
        self._hit_cell.fill_(int(self._NO_HIT_CELL))
        cuttable = material_cuttable if material_cuttable is not None else aux.cell_active

        wp.launch(
            pick_ray_cell_t_kernel,
            dim=self.num_cells,
            inputs=[
                aux.cell_active,
                aux.cell_material,
                cuttable,
                int(material_cuttable is not None),
                self._aabb_min,
                self._aabb_max,
                wp.vec3(float(ray_origin[0]), float(ray_origin[1]), float(ray_origin[2])),
                wp.vec3(float(ray_dir[0]), float(ray_dir[1]), float(ray_dir[2])),
            ],
            outputs=[self._cell_ray_t, self._min_t],
            device=self.device,
        )
        wp.launch(
            pick_ray_cell_match_kernel,
            dim=self.num_cells,
            inputs=[self._cell_ray_t, self._min_t, float(self._SENTINEL)],
            outputs=[self._hit_cell],
            device=self.device,
        )
        return self._hit_cell

    def pick(
        self,
        aux,
        ray_origin: tuple[float, float, float],
        ray_dir: tuple[float, float, float],
        material_cuttable: wp.array | None = None,
    ) -> int:
        """Run the two-kernel ray-AABB reduction. Returns the hit cell id or -1."""
        hit_device = self.pick_device(aux, ray_origin, ray_dir, material_cuttable=material_cuttable)
        hit = int(hit_device.numpy()[0])
        if hit == int(self._NO_HIT_CELL):
            return -1
        return hit


# ---------------------------------------------------------------------------
# Hover wireframe overlay: GPU-resident 12-edge cell outline.
# ---------------------------------------------------------------------------


@wp.kernel
def fill_hover_edges_kernel(
    hover_cell: int,
    cell_nodes: wp.array2d(dtype=wp.int32),
    edge_pairs: wp.array2d(dtype=wp.int32),  # [12, 2]
    particle_q: wp.array(dtype=wp.vec3),
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
):
    """Write the 12 edge segments of ``hover_cell`` into the overlay buffers."""
    i = wp.tid()
    a = edge_pairs[i, 0]
    b = edge_pairs[i, 1]
    starts[i] = particle_q[cell_nodes[hover_cell, a]]
    ends[i] = particle_q[cell_nodes[hover_cell, b]]


class HoverWireframe:
    """Persistent wp.arrays for the 12-edge cell wireframe overlay.

    Replaces example 07's ``_log_hover_cell`` numpy+upload path. A single
    12-thread kernel reads the 8 corner positions directly from GPU memory
    and writes into two pre-allocated vec3 buffers that we hand to
    ``viewer.log_lines`` unchanged each frame.
    """

    def __init__(
        self,
        edge_pairs_np,  # np.ndarray[12, 2] int32 - CELL_EDGE_PAIRS
        device: wp.context.Device,
    ):
        self.device = device
        self._edge_pairs = wp.array(edge_pairs_np, dtype=wp.int32, device=device)
        self._starts = wp.zeros(12, dtype=wp.vec3, device=device)
        self._ends = wp.zeros(12, dtype=wp.vec3, device=device)

    def update(self, aux, particle_q: wp.array, hover_cell: int) -> bool:
        """Refresh the 12 edge segments for ``hover_cell``. Returns False if hidden."""
        if hover_cell < 0:
            return False
        wp.launch(
            fill_hover_edges_kernel,
            dim=12,
            inputs=[int(hover_cell), aux.cell_nodes, self._edge_pairs, particle_q],
            outputs=[self._starts, self._ends],
            device=self.device,
        )
        return True

    @property
    def starts(self) -> wp.array:
        return self._starts

    @property
    def ends(self) -> wp.array:
        return self._ends


__all__ = [
    "HoverPicker",
    "HoverWireframe",
    "build_cell_aabbs_kernel",
    "build_dirty_cell_aabbs_kernel",
    "fill_hover_edges_kernel",
    "pick_ray_cell_match_kernel",
    "pick_ray_cell_t_kernel",
    "update_cell_render_state",
    "update_cell_render_state_and_aabbs",
    "update_cell_render_state_and_aabbs_kernel",
    "update_cell_render_state_kernel",
]
